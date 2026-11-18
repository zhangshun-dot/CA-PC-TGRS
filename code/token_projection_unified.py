# -*- coding: utf-8 -*-
"""Token Projection — 三条路线结合

最终框架:
  Phase 1 (CLIP Proxy):  在 CLIP 空间训练 W_D, OFAT δ 监督 → 轴分离度 ~0.24
  Phase 2 (Token Decode): 投影嵌入 → 离散解码为 prompt 文本 → GDINO 检测
  Phase 3 (REINFORCE):    GDINO F1 作为奖励信号, 有限差分优化 W_D 参数

核心洞察:
  - CLIP 是唯一能区分散视觉修饰词的编码器 (平均 cos=0.38)
  - GDINO 256-d 空间中轴方向几乎共线 (平均 cos=0.94)
  - 因此: 在 CLIP 空间做投影 → 解码为 prompt → 喂给 GDINO
  - 最后用 GDINO F1 作为端到端反馈微调

Usage:
    python token_projection_unified.py
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json, time, random, glob
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import BertTokenizer, BertModel, AutoModelForZeroShotObjectDetection

from token_projection import (
    TokenProjection, AXIS_MODIFIERS, AXIS_ORDER, UNIVERSALITY_SCORES,
    compute_axis_directions, train_projection,
)
from ofat_knowledge_base import OFATKnowledgeBase

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')

# ── 数据 ─────────────────────────────────────────────────
DIOR_DIR = r'C:\Users\张三\modeling\dior_data\full20_images'
DIOR_CLASSES = ['airplane','airport','baseball field','basketball court','bridge',
    'chimney','dam','expressway service area','expressway toll station',
    'golf course','ground track field','harbor','overpass','ship','stadium',
    'storage tank','tennis court','train station','vehicle','windmill']


# ══════════════════════════════════════════════════════════════════
# Phase 1: CLIP 空间训练 W_D (Route A 已验证)
# ══════════════════════════════════════════════════════════════════

def phase1_train_wd_clip(kb):
    """返回训练好的 W_D (512-d CLIP 空间) 和轴方向向量."""
    import clip
    print("\n" + "=" * 60)
    print("Phase 1: CLIP 空间训练 W_D")
    print("=" * 60)

    clip_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
    clip_model, _ = clip.load(clip_path, device=DEVICE)
    clip_model.eval()

    axis_dirs = compute_axis_directions(clip_model, clip.tokenize, device=DEVICE)
    proj = TokenProjection(d_model=512, rank=4, domain='remote_sensing')

    history = train_projection(
        proj, kb, axis_directions=axis_dirs,
        epochs=200, lr=0.005, device=DEVICE, verbose=False
    )

    print(f"  |W-I|/√d = {proj.get_adjustment_magnitude():.4f}")
    return proj, axis_dirs, clip_model


# ══════════════════════════════════════════════════════════════════
# Phase 2: 投影嵌入 → 离散 Token 解码 → Prompt 构造
# ══════════════════════════════════════════════════════════════════

class TokenDecoder:
    """将 CLIP 投影嵌入解码为 BERT token, 构造优化后的 prompt。

    流程:
      1. CLIP(cls) → W_D → E_clip_proj
      2. 学习 CLIP→BERT token embedding 映射
      3. 找到最接近的 BERT vocab tokens → 解码为文本
      4. 构造: "{prefix} {cls}, {decoded_modifier_tokens}"
    """

    def __init__(self, proj_clip, clip_model):
        self.proj = proj_clip
        self.clip_model = clip_model

        # 加载 BERT
        self.bert_tok = BertTokenizer.from_pretrained(
            'bert-base-uncased', local_files_only=True
        )
        self.bert_model = BertModel.from_pretrained(
            'bert-base-uncased', local_files_only=True
        ).eval().to(DEVICE)

        # BERT vocab embeddings
        with torch.no_grad():
            self.vocab_emb = self.bert_model.embeddings.word_embeddings.weight[:30000].clone()

        # CLIP→BERT 映射 (简单 Ridge, 在 __init__ 时训练)
        self._train_clip2bert_mapping()

    def _train_clip2bert_mapping(self):
        """训练 CLIP→BERT [CLS] 映射 (Ridge 回归)."""
        import clip as clip_module

        # 使用 OFAT 类名作为配对数据
        kb = OFATKnowledgeBase()
        all_classes = list(set(kb.dior_classes + list(kb.coco.get('classes', {}).keys())[:30]))

        X_clip, X_bert = [], []
        for cls_name in all_classes:
            prompt = f'a photo of {cls_name}'
            c_tok = clip_module.tokenize([prompt], truncate=True).to(DEVICE)
            with torch.no_grad():
                c_emb = self.clip_model.encode_text(c_tok)
                c_emb = c_emb / c_emb.norm(dim=-1, keepdim=True)
            b_tok = self.bert_tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                b_out = self.bert_model(**b_tok)
                b_emb = b_out.last_hidden_state[:, 0, :]
            X_clip.append(c_emb.squeeze(0).cpu().float())
            X_bert.append(b_emb.squeeze(0).cpu().float())

        Xc = torch.stack(X_clip).numpy()
        Xb = torch.stack(X_bert).numpy()

        # Ridge: W = Y^T X (X^T X + αI)^{-1}
        alpha = 0.1
        XTX = Xc.T @ Xc + alpha * np.eye(512)
        XTY = Xc.T @ Xb
        W_T = np.linalg.solve(XTX, XTY)
        self.W_map = torch.tensor(W_T.T, dtype=torch.float32).to(DEVICE)  # (768, 512)
        self.b_map = torch.tensor(Xb.mean(0) - Xc.mean(0) @ W_T, dtype=torch.float32).to(DEVICE)

        # 评估映射质量
        with torch.no_grad():
            pred_cpu = Xc @ W_T + (Xb.mean(0) - Xc.mean(0) @ W_T)
            mse = np.mean((Xb - pred_cpu) ** 2)
        print(f"  CLIP→BERT mapping: MSE={mse:.6f}")

    def project_and_decode(self, cls_name, top_k=5):
        """核心: 投影类嵌入, 解码为修饰词 tokens。

        Args:
            cls_name: 类名
            top_k: 取 top-k 个最接近的 token 作为修饰词候选

        Returns:
            (prefix, decoded_modifier_text, token_ids)
        """
        import clip as clip_module

        # Step 1: CLIP 编码 + W_D 投影
        prompt = f'a photo of {cls_name}'
        c_tok = clip_module.tokenize([prompt], truncate=True).to(DEVICE)
        with torch.no_grad():
            c_emb = self.clip_model.encode_text(c_tok)
            c_emb = c_emb / c_emb.norm(dim=-1, keepdim=True)

        c_proj = self.proj(c_emb.float())

        # Step 2: 映射到 BERT [CLS] 空间
        e_bert_proj = c_proj @ self.W_map.T + self.b_map  # (1, 768)

        # 获取原始 BERT 嵌入作为参考
        b_tok = self.bert_tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            b_out = self.bert_model(**b_tok)
            e_bert_raw = b_out.last_hidden_state[:, 0, :]

        # Step 3: 计算 Δ = 投影带来的修正方向
        delta_bert = e_bert_proj - e_bert_raw  # (1, 768)

        # Step 4: 在 BERT vocab 中找到与 Δ 最接近的 token
        delta_norm = delta_bert.squeeze(0) / (delta_bert.norm() + 1e-8)
        cos_sims = F.cosine_similarity(delta_norm.unsqueeze(0), self.vocab_emb.to(DEVICE), dim=-1)

        # 排除特殊 token
        special_ids = {self.bert_tok.cls_token_id, self.bert_tok.sep_token_id,
                       self.bert_tok.pad_token_id, self.bert_tok.unk_token_id,
                       self.bert_tok.mask_token_id}
        for sid in special_ids:
            if sid is not None and sid < len(cos_sims):
                cos_sims[sid] = -float('inf')

        # 取 top-k (去重子词 ##)
        top_ids = cos_sims.argsort(descending=True)[:top_k * 3]
        decoded_words = []
        seen = set()
        for tid in top_ids:
            word = self.bert_tok.decode([tid], skip_special_tokens=True).strip()
            if word and word not in seen and not word.startswith('##'):
                decoded_words.append(word)
                seen.add(word)
            if len(decoded_words) >= top_k:
                break

        modifier_text = ' '.join(decoded_words)

        # Step 5: 构造 prompt
        prefix = 'an overhead image of'
        full_prompt = f'{prefix} {cls_name}, {modifier_text}' if modifier_text else f'{prefix} {cls_name}'

        return full_prompt, modifier_text


# ══════════════════════════════════════════════════════════════════
# Phase 3: REINFORCE 端到端微调
# ══════════════════════════════════════════════════════════════════

def phase3_reinforce_finetune(proj_clip, decoder, kb, n_steps=50, n_images=5):
    """用 GDINO 检测 F1 作为奖励信号微调 W_D。

    有限差分法:
      - 对 W_D 的 U 矩阵每列施加小扰动
      - 测量检测 F1 变化
      - 梯度 = (F1_pos - F1_base) * perturbation

    Args:
        proj_clip: TokenProjection (d_model=512)
        decoder: TokenDecoder
        kb: OFATKnowledgeBase
        n_steps: 优化步数
        n_images: 每类图像数
    """
    print("\n" + "=" * 60)
    print("Phase 3: REINFORCE 端到端微调 (GDINO F1 奖励)")
    print("=" * 60)

    # 加载 GDINO
    gd_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
    from transformers import AutoProcessor
    gd_proc = AutoProcessor.from_pretrained(gd_cache)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        gd_cache, dtype=torch.float32
    ).eval().to(DEVICE)

    # 选择 5 个有代表性的类用于微调
    train_classes = ['airplane', 'ship', 'stadium', 'bridge', 'harbor']
    print(f"  微调类: {train_classes}")

    # 加载图像
    def load_images(cls, n=n_images):
        paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{cls.replace(' ','_')}_*.jpg")))[:n]
        return [Image.open(p).convert('RGB') for p in paths]

    def detect_f1(images, prompt):
        """计算检测命中率 (简化版 F1 proxy: 有检测框的图像比例)."""
        detections = []
        for img in images:
            inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
            inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            with torch.no_grad():
                o = gd_model(**inputs)
            ts = torch.tensor([img.size[::-1]]).to(DEVICE)
            r = gd_proc.post_process_grounded_object_detection(
                o, threshold=0.25, target_sizes=ts
            )[0]
            detections.append(len(r['boxes']) > 0)
        return np.mean(detections)

    # 基线: overhead prompt
    print("\n  基线 (overhead):")
    baseline_rates = {}
    for cls in train_classes:
        imgs = load_images(cls)
        rate = detect_f1(imgs, f'an overhead image of {cls}')
        baseline_rates[cls] = rate
        print(f"    {cls:<20s} DR={rate:.3f}")

    # CD-TDM prompt
    print("\n  CD-TDM (static):")
    tdm_rates = {}
    for cls in train_classes:
        imgs = load_images(cls)
        recs = kb.get_recommended_axes('remote_sensing', cls)
        mods = [AXIS_MODIFIERS[a] for a, _ in recs[:2]]
        prompt = f'an overhead image of {cls}'
        if mods:
            prompt += ', ' + ', '.join(mods)
        rate = detect_f1(imgs, prompt)
        tdm_rates[cls] = rate
        print(f"    {cls:<20s} DR={rate:.3f}  ({'+' if rate > baseline_rates[cls] else ''}{rate - baseline_rates[cls]:+.3f})")

    # Token Projection prompt
    print("\n  Token Projection (before fine-tune):")
    tp_rates_before = {}
    for cls in train_classes:
        imgs = load_images(cls)
        prompt, mod_text = decoder.project_and_decode(cls)
        rate = detect_f1(imgs, prompt)
        tp_rates_before[cls] = rate
        print(f"    {cls:<20s} DR={rate:.3f}  prompt='{prompt[:60]}...'")

    # ── REINFORCE 微调 ──
    print(f"\n  REINFORCE 微调 ({n_steps} steps):")
    proj_clip.train()
    proj_clip.to(DEVICE)
    lr = 0.005
    sigma = 0.02  # 扰动标准差

    best_mean = np.mean(list(tp_rates_before.values()))
    best_state = {k: v.clone() for k, v in proj_clip.state_dict().items()}

    for step in range(n_steps):
        # 有限差分梯度估计
        grad_U = torch.zeros_like(proj_clip.U)

        # Baseline: 当前 W_D 的 F1
        base_rewards = []
        for cls in train_classes:
            imgs = load_images(cls)
            prompt, _ = decoder.project_and_decode(cls)
            base_rewards.append(detect_f1(imgs, prompt))
        base_mean = np.mean(base_rewards)

        # 对一个随机列方向做有限差分
        col = step % proj_clip.rank
        for sign in [-1, 1]:
            noise = torch.randn(proj_clip.d_model, device=DEVICE) * sigma
            proj_clip.U.data[:, col] += sign * noise

            pos_rewards = []
            for cls in train_classes:
                imgs = load_images(cls)
                prompt, _ = decoder.project_and_decode(cls)
                pos_rewards.append(detect_f1(imgs, prompt))
            pos_mean = np.mean(pos_rewards)

            # 恢复
            proj_clip.U.data[:, col] -= sign * noise

            # 梯度估计
            grad_U[:, col] += sign * (pos_mean - base_mean) * noise / (sigma ** 2)

        # 应用梯度
        proj_clip.U.data += lr * grad_U * 0.5  # 除以 2 因为有正负两个方向

        # 评估
        eval_rewards = []
        for cls in train_classes:
            imgs = load_images(cls)
            prompt, _ = decoder.project_and_decode(cls)
            eval_rewards.append(detect_f1(imgs, prompt))
        eval_mean = np.mean(eval_rewards)

        if eval_mean > best_mean:
            best_mean = eval_mean
            best_state = {k: v.clone() for k, v in proj_clip.state_dict().items()}

        if step % 10 == 0:
            print(f"    Step {step:3d}: DR={eval_mean:.4f}  best={best_mean:.4f}")

    # 恢复最优
    proj_clip.load_state_dict(best_state)
    proj_clip.eval()

    # 最终评估
    print(f"\n  REINFORCE 完成: best DR={best_mean:.4f}")
    print(f"\n  最终对比:")
    print(f"  {'Class':<20s} {'Baseline':>8s} {'CD-TDM':>8s} {'TP(before)':>10s} {'TP(after)':>10s}")
    for cls in train_classes:
        imgs = load_images(cls)
        prompt, _ = decoder.project_and_decode(cls)
        tp_after = detect_f1(imgs, prompt)
        print(f"  {cls:<20s} {baseline_rates[cls]:8.3f} {tdm_rates[cls]:8.3f} "
              f"{tp_rates_before[cls]:10.3f} {tp_after:10.3f}")

    return {
        'baseline': baseline_rates,
        'cd_tdm': tdm_rates,
        'tp_before': tp_rates_before,
        'best_mean': best_mean,
    }


# ══════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("Token Projection — 三条路线结合")
    print("=" * 65)
    print(f"设备: {DEVICE}")

    kb = OFATKnowledgeBase()

    # Phase 1: CLIP 空间 W_D
    proj_clip, axis_dirs_clip, clip_model = phase1_train_wd_clip(kb)

    # Phase 2: Token Decoder (投影 → prompt 文本)
    print("\n" + "=" * 60)
    print("Phase 2: Token 解码器")
    print("=" * 60)
    decoder = TokenDecoder(proj_clip, clip_model)

    # 展示几个类的解码结果
    for cls in ['airplane', 'ship', 'stadium', 'bridge']:
        prompt, mod_text = decoder.project_and_decode(cls)
        print(f"  {cls:<20s} → '{prompt}'")

    # Phase 3: REINFORCE 微调 (需要 GDINO, 可选)
    print("\n" + "=" * 60)
    print("Phase 3 准备: GDINO 端到端微调")
    print("=" * 60)

    try:
        results = phase3_reinforce_finetune(proj_clip, decoder, kb, n_steps=30)
        with open(SAVE_DIR / 'token_projection_reinforce.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  Phase 3 跳过: {e}")
        import traceback
        traceback.print_exc()

    # 保存最终 W_D
    torch.save({
        'U': proj_clip.U.data.cpu(),
        'V': proj_clip.V.data.cpu(),
        'd_model': proj_clip.d_model,
        'rank': proj_clip.rank,
        'domain': proj_clip.domain,
    }, SAVE_DIR / 'token_projection_unified.pt')

    print("\n✅ Token Projection 三条路线结合完成")
