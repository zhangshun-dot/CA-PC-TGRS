# -*- coding: utf-8 -*-
"""single_scenario_factor_expand.py — (A) 扩因子: 把 condition/relation/uncertainty 加进统一空间

在 single_scenario_weight_probe 的 2 因子 (gd, vis) 基础上, 扩到覆盖框架四类的 7 个分子:
  content : gd  (检测器置信), vis (DINOv2 L11 正原型余弦), clip (CLIP 图文余弦)
  relation: rel (1 - 类间负原型余弦, 越高越不被其他类混淆)
  uncertainty: unc (1 - |vis-clip|/2, 双骨干分歧, 越高越一致=越确定)
  condition: cond (sigmoid(z_k), z_k=CondEncoder 对裁剪框估的观测条件强度), cond_inv (1-cond)
所有因子映射到 (0,1], 用加权几何平均 fused = ∏ f_k^{w_k} (w_k>=0, Σw=1) 融合。
逐场景网格搜权重向量(3 档), 看扩因子后最优 mAP 是否比 2 因子额外涨。
对比基准: 同网格下只留 (gd,vis) 的子空间最优(公平口径) + 前期 α 扫描的 2 因子最优(细口径)。
"""
import os, sys, json, itertools
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import torch
import torch.nn as nn
import clip as clip_module
from torchvision import transforms
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from unified_eval import (get_gdino, get_dinov2, get_bank, get_clip, gdino_detect,
                          encode_dinov2, compute_ap50_img, load_coco, load_dior,
                          PRE_FILTER, VEP_LAYER, EVALS, DEVICE, SAVE_DIR)


# ── 本地复刻 CondEncoder (避免 import v4 拉 v2/v3) ──
class CondEncoder(nn.Module):
    def __init__(self, cond_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, cond_dim))
    def forward(self, x):
        return self.net(x)


def load_cond():
    sd = torch.load(SAVE_DIR / 'unified_space_v4.pt', map_location=DEVICE)
    cond_sd = {k.replace('cond.', ''): v for k, v in sd.items() if k.startswith('cond.')}
    net = CondEncoder(1).to(DEVICE)
    net.load_state_dict(cond_sd)
    net.eval()
    return net


LEVELS = [0.0, 0.5, 1.0]
FACTORS = ['gd', 'vis', 'clip', 'rel', 'unc', 'cond', 'cond_inv']
F2 = {'gd', 'vis'}  # 2 因子子空间


def prep_scenario(images, base_dets, dior_cls, clip_text_feat, cond_enc):
    bank = get_bank()
    pos_proto = bank[dior_cls][VEP_LAYER]['prototype'].to(DEVICE)
    pos_proto = pos_proto / pos_proto.norm()
    # 类间负原型: 所有其他类原型, 取与 f 最像者
    others = []
    for c in bank:
        if c == dior_cls:
            continue
        p = bank[c][VEP_LAYER]['prototype'].to(DEVICE)
        others.append(p / p.norm())
    neg_protos = torch.stack(others)  # (19,768)

    clip_m, clip_pre = get_clip()
    out = []
    for (boxes, scores), img in zip(base_dets, images):
        keep = scores > PRE_FILTER
        if not keep.any():
            out.append({'boxes': np.empty((0, 4)), 'facs': {f: np.empty((0,)) for f in FACTORS},
                        'g': np.empty((0, 4))})
            continue
        boxes_f = boxes[keep]
        scores_f = scores[keep]
        gd = np.clip(scores_f, 1e-6, 1.0)
        # DINOv2 特征 + 正负原型余弦
        vis_l, rel_l, clip_l = [], [], []
        zks = []
        for box in boxes_f:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                vis_l.append(0.0); rel_l.append(0.0); clip_l.append(0.0); zks.append(0.0)
                continue
            crop = img.crop((x1, y1, x2, y2))
            f = encode_dinov2(crop, VEP_LAYER)                 # (768,) 已归一
            cos_pos = float((f @ pos_proto).item())
            cos_neg = float((f @ neg_protos.T).max().item())
            vis_l.append(cos_pos)
            rel_l.append(cos_neg)
            # CLIP 图文余弦
            imt = clip_pre(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                imf = clip_m.encode_image(imt).float()
            imf = imf / imf.norm(dim=-1, keepdim=True)
            clip_l.append(float((imf @ clip_text_feat).item()))
            # CondEncoder z_k
            xa = transforms.ToTensor()(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                zk = cond_enc(xa).item()
            zks.append(zk)
        vis = np.array(vis_l); rel = np.array(rel_l); clipv = np.array(clip_l)
        zk_arr = np.array(zks)
        cond = torch.sigmoid(torch.tensor(zk_arr)).numpy()
        facs = {
            'gd': gd,
            'vis': (vis + 1) / 2,
            'clip': (clipv + 1) / 2,
            'rel': (1 - rel) / 2,
            'unc': 1 - np.abs(vis - clipv) / 2,
            'cond': cond,
            'cond_inv': 1 - cond,
        }
        out.append({'boxes': boxes_f, 'facs': facs})
    return out


def ap_for_weights(prep, gts, w):
    pi = []
    for d in prep:
        if len(d['boxes']) == 0:
            pi.append((d['boxes'], np.empty((0,))))
            continue
        fused = np.ones(len(d['boxes']))
        for k in FACTORS:
            fused = fused * (d['facs'][k] ** w.get(k, 0.0))
        pi.append((d['boxes'], fused))
    return compute_ap50_img(pi, gts)


def main():
    cond_enc = load_cond()
    combos = [dict(zip(FACTORS, c)) for c in itertools.product(LEVELS, repeat=len(FACTORS))]
    combos = [c for c in combos if any(c.values())]

    # 2 因子细口径最优(前期 α 扫描)
    prev = json.load(open(SAVE_DIR / 'weight_probe_results.json', encoding='utf-8'))

    results = {}
    for name, dataset, cls_name, domain, prefix, dior_cls in EVALS:
        print(f'\n[{"="*12}] {name}', flush=True)
        if dataset == 'coco':
            images, gts = load_coco(cls_name)
        else:
            images, gts = load_dior(cls_name)
        if len(images) < 5:
            print('  图太少, 跳过', flush=True)
            continue
        base_dets = gdino_detect(images, f'{prefix} {cls_name}', 0.0)
        clip_m, clip_pre = get_clip()
        tok = clip_module.tokenize([f'{prefix} {cls_name}']).to(DEVICE)
        with torch.no_grad():
            txt = clip_m.encode_text(tok).float().squeeze(0)
        txt = txt / txt.norm()
        prep = prep_scenario(images, base_dets, dior_cls, txt, cond_enc)

        best_all, best_w = -1, None
        best_f2 = -1
        for w in combos:
            ap = ap_for_weights(prep, gts, w)
            if ap > best_all:
                best_all, best_w = ap, dict(w)
            if set(k for k in w if w[k] > 0).issubset(F2):
                if ap > best_f2:
                    best_f2 = ap
        delta_extra = round(best_all - best_f2, 4)
        prev_best = prev.get(name, {}).get('best_ap', None)
        print(f'  2因子(同网格)最优={best_f2:.4f}  扩因子最优={best_all:.4f}  '
              f'Δ额外={delta_extra:+.4f}  (前期α扫描2因子最优={prev_best:.4f})', flush=True)
        print(f'  最优权重={ {k: round(v,2) for k,v in best_w.items() if v>0} }', flush=True)
        results[name] = {'twofactor_grid_best': float(best_f2),
                         'expanded_best': float(best_all),
                         'delta_extra': delta_extra,
                         'prev_2factor_best': prev_best,
                         'best_weights': {k: round(v, 3) for k, v in best_w.items()},
                         'n_img': len(images), 'n_gt': int(sum(len(g) for g in gts))}
    with open(SAVE_DIR / 'weight_probe_expand_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n结果 → {SAVE_DIR / "weight_probe_expand_results.json"}', flush=True)


if __name__ == '__main__':
    main()
