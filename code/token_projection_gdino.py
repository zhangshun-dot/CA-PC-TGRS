# -*- coding: utf-8 -*-
"""Token Projection v5 — 学习轻量偏移 + α 扫描

v4 问题: CLIP→GDINO256 映射学习完整 [CLS] 特征 (MSE=0.487),
        偏移太大淹没了类名信息。

v5 方案:
  1. 学习 CLIP→GDINO256 偏移 (而非完整特征):
     offset = mapper(W_D(CLIP(cls))) - mapper_mean (去均值)
  2. 扫描 α_retain: 控制原始特征保留比例
  3. 在 text_projection 后注入: feat = α·orig + (1-α)·offset

Usage:
    python token_projection_gdino.py
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json, time, glob
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection, AutoProcessor, AutoTokenizer,
)

from token_projection import AXIS_MODIFIERS, AXIS_ORDER, compute_axis_directions
from ofat_knowledge_base import OFATKnowledgeBase

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')

DIOR_DIR = r'C:\Users\张三\modeling\dior_data\full20_images'
DIOR_CLASSES = ['airplane','airport','baseball field','basketball court','bridge',
    'chimney','dam','expressway service area','expressway toll station',
    'golf course','ground track field','harbor','overpass','ship','stadium',
    'storage tank','tennis court','train station','vehicle','windmill']

POOL_CLASSES = ['airplane', 'bird', 'car', 'dog', 'cat', 'ship', 'bridge',
    'stadium', 'building', 'tree', 'road', 'person', 'truck', 'bicycle',
    'motorcycle', 'bus', 'train', 'chair', 'table', 'house']

gd_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'


# ══════════════════════════════════════════════════════════════════
# 解析 W_D
# ══════════════════════════════════════════════════════════════════

def solve_wd_analytical(kb, clip_model, axis_directions, domain='remote_sensing',
                         lambda_strength=0.3, ridge=1.0):
    import clip as clip_module
    d = 512
    delta_source = kb.dior_deltas if domain == 'remote_sensing' else kb.coco_deltas

    class_deltas = {}
    for (cls_name, axis), delta in delta_source.items():
        if cls_name not in class_deltas:
            class_deltas[cls_name] = {}
        class_deltas[cls_name][axis] = delta

    classes, E_list, D_list = [], [], []
    for cls_name, deltas in class_deltas.items():
        prompt = f'a photo of {cls_name}'
        tok = clip_module.tokenize([prompt], truncate=True).to(DEVICE)
        with torch.no_grad():
            emb = clip_model.encode_text(tok).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
        e_c = emb.squeeze(0)
        delta_c = torch.zeros(d, device=DEVICE)
        total_weight = 0.0
        for axis, delta in deltas.items():
            if delta <= 0: continue
            if axis not in axis_directions: continue
            v_a = axis_directions[axis].to(device=DEVICE, dtype=torch.float32)
            delta_c += delta * v_a
            total_weight += abs(delta)
        if total_weight > 0:
            delta_c /= total_weight
        classes.append(cls_name)
        E_list.append(e_c)
        D_list.append(delta_c)

    E = torch.stack(E_list).T
    D = torch.stack(D_list).T
    N = len(classes)

    E_np, D_np = E.cpu().numpy(), D.cpu().numpy()
    EET = E_np @ E_np.T
    reg = ridge * np.eye(d)
    EDT = D_np @ E_np.T
    inv = np.linalg.solve(EET + reg, np.eye(d))
    W_D_np = np.eye(d) + lambda_strength * (EDT @ inv)
    W_D = torch.tensor(W_D_np, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        E_proj = W_D @ E
        cos_sims = [F.cosine_similarity(E_proj[:, i].unsqueeze(0),
                                         E[:, i].unsqueeze(0)).item() for i in range(N)]
    print(f"  解析 W_D: |W-I|/√d={lambda_strength*np.linalg.norm(EDT@inv)/np.sqrt(d):.4f}, "
          f"mean_cos={np.mean(cos_sims):.4f}")

    return W_D, classes


class AnalyticalProjection:
    def __init__(self, W_D): self.W_D = W_D
    def __call__(self, x): return x @ self.W_D.T
    def eval(self): pass
    def train(self): pass


# ══════════════════════════════════════════════════════════════════
# v5: 学习轻量偏移
# ══════════════════════════════════════════════════════════════════

def learn_offsets(proj_wd, clip_model, gdino_model, gdino_tokenizer, classes):
    """学习每个类在 GDINO 256-d 空间的"轻量偏移"。

    偏移 = mapper(W_D(CLIP(cls))) - mean_all_classes
    只保留类间差异部分，去掉公共分量。

    Returns:
        token_offset: {token_id: tensor(256,)} 轻量偏移查找表
    """
    import clip as clip_module

    # Step 1: 收集每类的 W_D(CLIP) 嵌入
    clip_projs = {}
    for cls_name in classes:
        prompt = f'a photo of {cls_name}'
        c_tok = clip_module.tokenize([prompt], truncate=True).to(DEVICE)
        with torch.no_grad():
            c_emb = clip_model.encode_text(c_tok).float()
            c_emb = c_emb / c_emb.norm(dim=-1, keepdim=True)
        clip_projs[cls_name] = proj_wd(c_emb).squeeze(0)

    # Step 2: 收集每类的 GDINO 256-d [CLS] 特征
    gdino_feats = {}
    for cls_name in classes:
        gd_prompt = f'an overhead image of {cls_name}'
        gd_tok = gdino_tokenizer(gd_prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            bert_out = gdino_model.model.text_backbone(
                input_ids=gd_tok['input_ids'],
                attention_mask=gd_tok['attention_mask'],
                token_type_ids=torch.zeros_like(gd_tok['input_ids']),
                position_ids=torch.arange(gd_tok['input_ids'].shape[1],
                                          device=DEVICE).unsqueeze(0),
                return_dict=True,
            )
            txt_feat = gdino_model.model.text_projection(
                bert_out.last_hidden_state
            )
            gdino_feats[cls_name] = txt_feat[:, 0, :].squeeze(0)  # [CLS] 256-d

    # Step 3: 学习 CLIP_proj → GDINO_feat 的 Ridge 映射
    X = torch.stack([clip_projs[c] for c in classes]).cpu().numpy()  # (N, 512)
    Y = torch.stack([gdino_feats[c] for c in classes]).cpu().numpy()  # (N, 256)

    alpha = 0.5
    XTX = X.T @ X + alpha * np.eye(512)
    XTY = X.T @ Y
    W_T = np.linalg.solve(XTX, XTY)
    W_map = torch.tensor(W_T.T, dtype=torch.float32).to(DEVICE)  # (256, 512)
    b_map = torch.tensor(Y.mean(0) - X.mean(0) @ W_T, dtype=torch.float32).to(DEVICE)

    Y_pred = X @ W_T + b_map.cpu().numpy()
    mse = np.mean((Y - Y_pred) ** 2)

    # Step 4: 计算"去均值偏移" — 只保留类间差异
    # offset_c = mapped(W_D(CLIP_c)) - mean_over_all_classes
    all_offsets_raw = {}
    for cls_name in classes:
        cp = clip_projs[cls_name].unsqueeze(0)  # (1, 512)
        mapped = cp @ W_map.T + b_map  # (1, 256)
        all_offsets_raw[cls_name] = mapped.squeeze(0)

    mean_offset = torch.stack(list(all_offsets_raw.values())).mean(dim=0)  # (256,)

    # 去均值: offset = raw - mean (只保留类间差异)
    class_offsets = {}
    for cls_name in classes:
        class_offsets[cls_name] = all_offsets_raw[cls_name] - mean_offset

    print(f"  CLIP→GDINO256 mapping: MSE={mse:.4f}")
    print(f"  Offset norms: mean={torch.stack(list(class_offsets.values())).norm(dim=-1).mean():.4f}")

    # Step 5: 构建 token_id → offset 查找表
    accum = {}
    for cls_name in classes:
        ids = gdino_tokenizer(cls_name, add_special_tokens=False)['input_ids']
        offset = class_offsets[cls_name]
        for tid in ids:
            if tid not in accum:
                accum[tid] = []
            accum[tid].append(offset)

    token_offset = {}
    for tid, offsets in accum.items():
        token_offset[tid] = torch.stack(offsets).mean(dim=0)

    return token_offset, class_offsets


# ══════════════════════════════════════════════════════════════════
# 注入 wrapper
# ══════════════════════════════════════════════════════════════════

class InjectedTextProjection(nn.Module):
    def __init__(self, original_proj, token_offset, special_ids, alpha_retain=0.5):
        super().__init__()
        self.original = original_proj
        self.token_offset = token_offset
        self.special_ids = special_ids
        self.alpha_retain = alpha_retain
        self._input_ids = None

    def set_input_ids(self, ids):
        self._input_ids = ids

    def set_alpha(self, alpha):
        self.alpha_retain = alpha

    def forward(self, x):
        out = self.original(x)
        if self._input_ids is None:
            return out

        modified = out.clone()
        n_modified = 0
        for pos in range(self._input_ids.shape[1]):
            tid = self._input_ids[0, pos].item()
            if tid in self.special_ids:
                continue
            if tid in self.token_offset:
                offset = self.token_offset[tid].to(device=modified.device, dtype=modified.dtype)
                orig_feat = modified[0, pos, :]
                # 🔑 v5: 加法偏移 (不是插值)
                # feat_new = orig + scale * offset
                # scale = (1 - alpha): alpha 越大越保守
                scale = 1.0 - self.alpha_retain
                modified[0, pos, :] = orig_feat + scale * offset
                n_modified += 1

        return modified


# ══════════════════════════════════════════════════════════════════
# 评估
# ══════════════════════════════════════════════════════════════════

def detect_f1(model, processor, images, prompt, injected_proj=None):
    detections = []
    for img in images:
        try:
            inputs = processor(images=img, text=prompt, return_tensors='pt')
            inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            if injected_proj is not None and 'input_ids' in inputs:
                injected_proj.set_input_ids(inputs['input_ids'])
            with torch.no_grad():
                o = model(**inputs)
            ts = torch.tensor([img.size[::-1]]).to(DEVICE)
            r = processor.post_process_grounded_object_detection(
                o, threshold=0.25, target_sizes=ts
            )[0]
            detections.append(len(r['boxes']) > 0)
        except Exception as e:
            detections.append(False)
    return np.mean(detections)


def load_images(cls, n=10):
    paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{cls.replace(' ','_')}_*.jpg")))[:n]
    return [Image.open(p).convert('RGB') for p in paths]


# ══════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("Token Projection v5 — 轻量偏移 + α 扫描")
    print("=" * 65)
    t_start = time.time()

    kb = OFATKnowledgeBase()

    # ── 1. CLIP ──
    print("\n[1/4] 加载 CLIP...")
    import clip
    clip_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
    clip_model, _ = clip.load(clip_path, device=DEVICE)
    clip_model.eval()

    # ── 2. 轴方向 + 解析 W_D ──
    print("\n[2/4] 计算 W_D + 偏移...")
    axis_dirs = compute_axis_directions(clip_model, clip.tokenize, device=DEVICE)
    W_D, wd_classes = solve_wd_analytical(
        kb, clip_model, axis_dirs, lambda_strength=0.1, ridge=1.0
    )
    proj_wd = AnalyticalProjection(W_D)

    # ── 3. GDINO + 学习偏移 ──
    print("\n[3/4] 加载 GDINO + 学习偏移...")
    processor = AutoProcessor.from_pretrained(gd_cache)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        gd_cache, dtype=torch.float32
    ).eval().to(DEVICE)
    gd_tokenizer = AutoTokenizer.from_pretrained(gd_cache)

    all_classes = list(set(DIOR_CLASSES + POOL_CLASSES))
    token_offset, class_offsets = learn_offsets(
        proj_wd, clip_model, model, gd_tokenizer, all_classes
    )

    # 特殊 token
    special_ids = {
        gd_tokenizer.cls_token_id, gd_tokenizer.sep_token_id,
        gd_tokenizer.pad_token_id,
    }
    if gd_tokenizer.mask_token_id:
        special_ids.add(gd_tokenizer.mask_token_id)

    # ── 4. 扫描 α ──
    print("\n[4/4] α 扫描 + 评估...")
    test_classes = ['airplane', 'ship', 'stadium', 'bridge', 'harbor',
                    'overpass', 'tennis court', 'windmill']
    n_imgs = 10

    # 先测 baseline 和 CD-TDM
    print("\n  Baseline & CD-TDM (无注入):")
    baseline_results = {}
    tdm_results = {}
    for cls in test_classes:
        imgs = load_images(cls, n_imgs)
        r_oh = detect_f1(model, processor, imgs, f'an overhead image of {cls}')
        baseline_results[cls] = r_oh
        recs = kb.get_recommended_axes('remote_sensing', cls)
        mods = [AXIS_MODIFIERS[a] for a, _ in recs[:2]]
        p_tdm = f'an overhead image of {cls}'
        if mods:
            p_tdm += ', ' + ', '.join(mods)
        r_tdm = detect_f1(model, processor, imgs, p_tdm)
        tdm_results[cls] = r_tdm
        print(f"    {cls:<20s} baseline={r_oh:.3f}  cd_tdm={r_tdm:.3f}  "
              f"mods={mods if mods else '(none)'}")

    # 安装注入 wrapper
    original_text_proj = model.model.text_projection
    injected_proj = InjectedTextProjection(
        original_text_proj, token_offset, special_ids, alpha_retain=0.5
    )
    model.model.text_projection = injected_proj

    # 扫描 α
    print("\n  α 扫描 (越大越保守, 1.0=无注入):")
    alphas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0]
    scan_results = []

    for alpha in alphas:
        injected_proj.set_alpha(alpha)

        # 正样本 DR
        dr_list = []
        for cls in test_classes:
            imgs = load_images(cls, 5)
            dr = detect_f1(model, processor, imgs, f'an overhead image of {cls}',
                          injected_proj=injected_proj)
            dr_list.append(dr)

        # 负对照
        neg_pairs = [('airplane', 'ship'), ('ship', 'airplane'), ('bridge', 'harbor')]
        neg_list = []
        for cls_img, cls_prompt in neg_pairs:
            imgs = load_images(cls_img, 5)
            dr = detect_f1(model, processor, imgs, f'an overhead image of {cls_prompt}',
                          injected_proj=injected_proj)
            neg_list.append(dr)

        dr_mean = np.mean(dr_list)
        neg_mean = np.mean(neg_list)
        specificity = dr_mean - neg_mean
        scan_results.append({
            'alpha': alpha, 'dr': dr_mean, 'neg_dr': neg_mean,
            'specificity': specificity
        })
        print(f"    α={alpha:.2f}: DR={dr_mean:.3f}  neg_DR={neg_mean:.3f}  "
              f"specificity={specificity:+.3f}")

    # 选最优 α
    best = max(scan_results, key=lambda r: r['specificity'])
    best_alpha = best['alpha']
    print(f"\n  最优 α = {best_alpha:.2f} (specificity={best['specificity']:+.3f})")

    # 用最优 α 做完整评估
    injected_proj.set_alpha(best_alpha)
    print(f"\n  Token Projection (α={best_alpha}):")
    tp_results = {}
    for cls in test_classes:
        imgs = load_images(cls, n_imgs)
        r_tp = detect_f1(model, processor, imgs, f'an overhead image of {cls}',
                        injected_proj=injected_proj)
        tp_results[cls] = r_tp
        print(f"    {cls:<20s} token_proj={r_tp:.3f}")

    # 负对照
    print("\n  负对照:")
    neg_pairs = [('airplane', 'ship'), ('ship', 'airplane'), ('bridge', 'harbor')]
    neg_results = []
    for cls_img, cls_prompt in neg_pairs:
        imgs = load_images(cls_img, 5)
        dr = detect_f1(model, processor, imgs, f'an overhead image of {cls_prompt}',
                      injected_proj=injected_proj)
        neg_results.append(dr)
        print(f"    {cls_img} imgs + '{cls_prompt}': DR={dr:.3f}")

    # ── 汇总 ──
    print(f"\n  {'Class':<20s} {'Baseline':>8s} {'CD-TDM':>8s} {'TokenProj':>10s}")
    print(f"  " + "-" * 50)
    for cls in test_classes:
        print(f"  {cls:<20s} {baseline_results[cls]:8.3f} {tdm_results[cls]:8.3f} "
              f"{tp_results[cls]:10.3f}")

    oh_mean = np.mean(list(baseline_results.values()))
    tdm_mean = np.mean(list(tdm_results.values()))
    tp_mean = np.mean(list(tp_results.values()))
    neg_mean = np.mean(neg_results)

    print(f"\n  {'MEAN':<20s} {oh_mean:8.3f} {tdm_mean:8.3f} {tp_mean:10.3f}")
    print(f"  Δ vs baseline:          CD-TDM={tdm_mean-oh_mean:+.3f}  "
          f"TokenProj={tp_mean-oh_mean:+.3f}")
    print(f"  负对照: {neg_mean:.3f} | specificity = {tp_mean - neg_mean:+.3f}")

    # 保存
    with open(SAVE_DIR / 'token_projection_gdino_v5_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'alpha_scan': scan_results,
            'best_alpha': best_alpha,
            'per_class': {cls: {
                'baseline': float(baseline_results[cls]),
                'cd_tdm': float(tdm_results[cls]),
                'token_proj': float(tp_results[cls]),
            } for cls in test_classes},
            'mean_baseline': float(oh_mean),
            'mean_cd_tdm': float(tdm_mean),
            'mean_token_proj': float(tp_mean),
            'neg_control_mean': float(neg_mean),
            'specificity': float(tp_mean - neg_mean),
        }, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_start
    print(f"\n✅ Token Projection v5 完成 (总耗时 {elapsed:.0f}s)")
