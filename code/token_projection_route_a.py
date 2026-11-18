# -*- coding: utf-8 -*-
"""路线 A: CLIP→BERT 桥接 Token Projection

三步走:
  1. CLIP 空间训练 W_D (Token Universality 约束, 已有 0.228 分离度)
  2. 学习 M: CLIP_512 → BERT_768 (用配对类名嵌入, Ridge 回归)
  3. 应用: CLIP(cls) → W_D → M → BERT_768 → GDINO inputs_embeds

关键创新: 绕过 BERT 空间中视觉方向共线的问题，
          用 CLIP 的视觉文本对齐能力指导投影，
          通过线性映射桥接到 GDINO 的 BERT 空间。
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import BertTokenizer, BertModel

from token_projection import (
    TokenProjection, AXIS_MODIFIERS, AXIS_ORDER, UNIVERSALITY_SCORES,
    compute_axis_directions, train_projection, evaluate_projection,
)
from ofat_knowledge_base import OFATKnowledgeBase

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')


# ══════════════════════════════════════════════════════════════════
# Step 1: CLIP 空间训练 W_D（复用 token_projection.py 的结果）
# ══════════════════════════════════════════════════════════════════

def step1_train_wd_clip(force_retrain=False):
    """在 CLIP 空间中训练 W_D。

    Returns:
        proj_clip: 训练好的 TokenProjection (d_model=512)
        axis_dirs_clip: CLIP 轴方向向量
    """
    import clip

    print("=" * 60)
    print("Step 1: CLIP 空间训练 W_D")
    print("=" * 60)

    kb = OFATKnowledgeBase()

    # 加载 CLIP
    clip_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
    clip_model, _ = clip.load(clip_path, device=DEVICE)
    clip_model.eval()

    # 计算轴方向
    print("\n[1.1] 计算 CLIP 轴方向向量...")
    axis_dirs = compute_axis_directions(clip_model, clip.tokenize, device=DEVICE)

    # 训练 W_D
    print("\n[1.2] 训练 W_D (CLIP 512-d, rank=4)...")
    proj = TokenProjection(d_model=512, rank=4, domain='remote_sensing')

    history = train_projection(
        proj, kb, axis_directions=axis_dirs,
        epochs=200, lr=0.005, device=DEVICE, verbose=True
    )

    # 评估
    results = evaluate_projection(proj, kb, axis_dirs, device=DEVICE)
    s = results['summary']
    print(f"\n  CLIP 空间评估:")
    print(f"    分离度: {s['alignment_separation']:+.4f}")
    print(f"    有益轴对齐: {s['mean_beneficial_alignment']:+.4f}")
    print(f"    |W-I|/√d: {proj.get_adjustment_magnitude():.4f}")

    # 保存
    torch.save({
        'U': proj.U.data.cpu(),
        'V': proj.V.data.cpu(),
        'd_model': proj.d_model,
        'rank': proj.rank,
        'domain': proj.domain,
    }, SAVE_DIR / 'token_projection_wd_clip.pt')

    torch.save({a: d.cpu() for a, d in axis_dirs.items()},
               SAVE_DIR / 'axis_directions_clip.pt')

    return proj, axis_dirs


# ══════════════════════════════════════════════════════════════════
# Step 2: 学习 CLIP → BERT 线性映射 M
# ══════════════════════════════════════════════════════════════════

class CLIP2BERTMapper:
    """CLIP 嵌入空间到 BERT 嵌入空间的线性映射。

    M: CLIP_512 → BERT_768
    用 Ridge 回归学习，正则化防止过拟合。
    """

    def __init__(self, clip_dim=512, bert_dim=768):
        self.clip_dim = clip_dim
        self.bert_dim = bert_dim
        # M @ x_clip + b → x_bert
        self.W = None  # (bert_dim, clip_dim)
        self.b = None  # (bert_dim,)

    def fit(self, clip_embeddings, bert_embeddings, alpha=1.0):
        """Ridge 回归拟合。

        Args:
            clip_embeddings: (N, clip_dim) — CLIP 文本嵌入
            bert_embeddings: (N, bert_dim) — BERT [CLS] 嵌入
            alpha: L2 正则化系数
        """
        X = clip_embeddings.numpy()  # (N, clip_dim)
        Y = bert_embeddings.numpy()  # (N, bert_dim)
        N, d_in = X.shape
        d_out = Y.shape[1]

        # Ridge: W = Y^T X (X^T X + αI)^{-1}
        # 更稳定的解法: (X^T X + αI) W^T = X^T Y
        XTX = X.T @ X
        reg = alpha * np.eye(d_in)
        XTY = X.T @ Y  # (clip_dim, bert_dim)

        W_T = np.linalg.solve(XTX + reg, XTY)  # (clip_dim, bert_dim)
        self.W = torch.tensor(W_T.T, dtype=torch.float32)  # (bert_dim, clip_dim)

        # 偏置: b = mean(Y) - W @ mean(X)
        self.b = torch.tensor(
            Y.mean(axis=0) - X.mean(axis=0) @ W_T,
            dtype=torch.float32
        )

        # 计算拟合误差
        Y_pred = X @ W_T + self.b.numpy()
        mse = np.mean((Y - Y_pred) ** 2)
        r2 = 1 - mse / np.var(Y)
        print(f"  Ridge (α={alpha}): MSE={mse:.6f}, R²={r2:.4f}")

        return self

    def to(self, device):
        self.W = self.W.to(device)
        self.b = self.b.to(device)
        return self

    def __call__(self, x_clip):
        """x_clip: (..., clip_dim) → (..., bert_dim)"""
        x = x_clip.float()
        return x @ self.W.T.to(device=x.device) + self.b.to(device=x.device)


def step2_learn_mapping(kb, proj_clip, axis_dirs_clip):
    """学习 CLIP→BERT 映射，用 OFAT 类名作为配对数据。

    Returns:
        mapper: CLIP2BERTMapper 实例
    """
    import clip

    print("\n" + "=" * 60)
    print("Step 2: 学习 CLIP → BERT 线性映射 M")
    print("=" * 60)

    # 加载模型
    clip_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
    clip_model, _ = clip.load(clip_path, device=DEVICE)
    clip_model.eval()

    bert_tok = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    bert_model = BertModel.from_pretrained('bert-base-uncased', local_files_only=True)
    bert_model.eval().to(DEVICE)

    # 收集类名: COCO 80 + DIOR 20 + 通用类名
    dior_classes = kb.dior_classes
    coco_classes = list(kb.coco.get('classes', {}).keys())[:80]
    extra_classes = [
        'person', 'car', 'dog', 'cat', 'bird', 'tree', 'house',
        'road', 'river', 'mountain', 'cloud', 'bicycle', 'motorcycle',
        'bus', 'train', 'truck', 'boat', 'horse', 'sheep', 'cow',
        'elephant', 'zebra', 'giraffe', 'pizza', 'cake',
    ]

    all_classes = list(set(dior_classes + coco_classes + extra_classes))
    print(f"  配对数据: {len(all_classes)} 个类名")

    # 收集 CLIP 和 BERT 嵌入
    clip_embs = []
    bert_embs = []

    for cls_name in all_classes:
        prompt = f'a photo of {cls_name}'

        # CLIP
        clip_tok = clip.tokenize([prompt], truncate=True).to(DEVICE)
        with torch.no_grad():
            c_emb = clip_model.encode_text(clip_tok)
            c_emb = c_emb / c_emb.norm(dim=-1, keepdim=True)
        clip_embs.append(c_emb.squeeze(0).cpu())

        # BERT
        b_tok = bert_tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            b_out = bert_model(**b_tok)
            b_emb = b_out.last_hidden_state[:, 0, :]  # [CLS]
        bert_embs.append(b_emb.squeeze(0).cpu())

    X_clip = torch.stack(clip_embs)  # (N, 512)
    X_bert = torch.stack(bert_embs)  # (N, 768)

    # 训练映射
    print("\n  训练 CLIP → BERT 映射...")
    mapper = CLIP2BERTMapper(clip_dim=512, bert_dim=768)

    # 尝试不同正则化强度
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        m = CLIP2BERTMapper()
        m.fit(X_clip, X_bert, alpha=alpha)
        # 评估: 检查映射后的嵌入与真实 BERT 嵌入的余弦相似度
        with torch.no_grad():
            pred = m(X_clip)
            cos_sims = F.cosine_similarity(pred, X_bert, dim=-1)
        print(f"    α={alpha:6.2f}  mean cos_sim={cos_sims.mean():.4f}")

    # 选 α=0.1
    mapper.fit(X_clip, X_bert, alpha=0.1)

    # 保存
    torch.save({
        'W': mapper.W.cpu(),
        'b': mapper.b.cpu(),
        'clip_dim': mapper.clip_dim,
        'bert_dim': mapper.bert_dim,
    }, SAVE_DIR / 'clip2bert_mapper.pt')

    return mapper, bert_tok, bert_model, clip_model


# ══════════════════════════════════════════════════════════════════
# Step 3: 构建完整推理 Pipeline, 验证
# ══════════════════════════════════════════════════════════════════

class TokenProjectionPipeline:
    """端到端 Token Projection Pipeline。

    CLIP(类名) → W_D → M → BERT_768 → GDINO inputs_embeds
    """

    def __init__(self, proj_clip, mapper, clip_model, bert_model, bert_tokenizer):
        self.proj = proj_clip
        self.mapper = mapper
        self.clip_model = clip_model
        self.bert_model = bert_model
        self.bert_tok = bert_tokenizer

    def project_class_name(self, cls_name, device=DEVICE):
        """为单个类名生成投影后的 BERT 嵌入。

        流程:
          1. CLIP 编码类名 → E_clip
          2. W_D 投影 → E_clip_proj
          3. M 映射到 BERT 空间 → E_bert_proj
          4. 归一化为 BERT 嵌入尺度

        Args:
            cls_name: 类名字符串

        Returns:
            torch.Tensor: (768,) BERT 空间中的投影嵌入
        """
        import clip as clip_module

        prompt = f'a photo of {cls_name}'

        # Step 1: CLIP 编码
        clip_tok = clip_module.tokenize([prompt], truncate=True).to(device)
        with torch.no_grad():
            e_clip = self.clip_model.encode_text(clip_tok)
            e_clip = e_clip / e_clip.norm(dim=-1, keepdim=True)

        # Step 2: W_D 投影
        e_clip_proj = self.proj(e_clip.float()).squeeze(0)

        # Step 3: M 映射
        self.mapper.to(device)
        e_bert_proj = self.mapper(e_clip_proj)

        # Step 4: 缩放到 BERT [CLS] 的自然范数
        # 获取原始 BERT 嵌入的范数作为参考
        b_tok = self.bert_tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            b_out = self.bert_model(**b_tok)
            e_bert_raw = b_out.last_hidden_state[:, 0, :].squeeze(0)

        e_bert_proj = e_bert_proj * (e_bert_raw.norm() / (e_bert_proj.norm() + 1e-8))

        return e_bert_proj

    def build_projected_inputs_embeds(self, full_prompt, device=DEVICE):
        """为完整的 prompt 构建 inputs_embeds (含投影后的类名 token)。

        对完整 prompt 中属于类名的 token 位置，
        用投影后的嵌入替换原始嵌入。

        Args:
            full_prompt: 如 "an overhead image of airplane, large and prominent"

        Returns:
            dict: tokenizer 输出, 但 attention_mask 不变,
                  inputs_embeds 中类名位置被替换
        """
        tok = self.bert_tok(full_prompt, return_tensors='pt')
        input_ids = tok['input_ids'][0]

        # 获取原始嵌入
        with torch.no_grad():
            emb_layer = self.bert_model.embeddings
            init_embeds = emb_layer(
                input_ids=tok['input_ids'].to(device),
                token_type_ids=tok.get('token_type_ids', torch.zeros_like(tok['input_ids'])).to(device),
            )

        # 找到非特殊 token 位置（类名和修饰词 token）
        special_ids = {
            self.bert_tok.cls_token_id,
            self.bert_tok.sep_token_id,
            self.bert_tok.pad_token_id,
        }

        # 对每个非特殊 token 应用投影
        with torch.no_grad():
            for pos in range(1, len(input_ids) - 1):
                tid = input_ids[pos].item()
                if tid in special_ids:
                    continue
                init_embeds[0, pos, :] = self.proj(
                    init_embeds[0, pos, :].unsqueeze(0).float()
                ).squeeze(0)

        return {
            'inputs_embeds': init_embeds,
            'attention_mask': tok['attention_mask'],
        }


def step3_verify(pipeline, kb):
    """验证完整 Pipeline。

    对比:
      - E_raw: 原始 BERT 类嵌入
      - E_proj: 投影后的 BERT 类嵌入
      - Δ = E_proj - E_raw
      - 检查 Δ 是否沿有益轴方向
    """
    import clip as clip_module

    print("\n" + "=" * 60)
    print("Step 3: Pipeline 验证")
    print("=" * 60)

    # 重新加载 BERT（用于验证）
    bert_tok_v = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    bert_model_v = BertModel.from_pretrained('bert-base-uncased', local_files_only=True)
    bert_model_v.eval().to(DEVICE)

    from token_projection import compute_axis_directions_bert
    axis_dirs_bert = compute_axis_directions_bert(bert_model_v, bert_tok_v, device=DEVICE)

    # 对比每个 DIOR 类
    print("\n  DIOR 类投影效果:")
    print(f"  {'Class':<26s} {'|Δ|':>8s} {'有益轴cos':>10s} {'有害轴cos':>10s} {'分离':>8s}")
    print(f"  " + "-" * 65)

    all_beneficial = []
    all_harmful = []

    for cls_name in kb.dior_classes:
        # 投影嵌入
        e_proj = pipeline.project_class_name(cls_name)

        # 原始 BERT 嵌入
        b_tok = bert_tok_v(f'a photo of {cls_name}', return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            b_out = bert_model_v(**b_tok)
            e_raw = b_out.last_hidden_state[:, 0, :].squeeze(0)

        displacement = e_proj - e_raw
        disp_norm = displacement.norm().item()

        # 计算与各轴方向的余弦相似度
        bene_cos = []
        harm_cos = []

        for axis in AXIS_ORDER:
            delta = kb.get_delta('remote_sensing', cls_name, axis) or 0
            if axis not in axis_dirs_bert:
                continue
            v_a = axis_dirs_bert[axis].to(DEVICE).float()
            cos_sim = F.cosine_similarity(displacement.unsqueeze(0), v_a.unsqueeze(0)).item()

            if delta > 0:
                bene_cos.append(cos_sim)
            else:
                harm_cos.append(cos_sim)

        avg_bene = np.mean(bene_cos) if bene_cos else 0
        avg_harm = np.mean(harm_cos) if harm_cos else 0
        sep = avg_bene - avg_harm

        all_beneficial.append(avg_bene)
        all_harmful.append(avg_harm)

        print(f"  {cls_name:<26s} {disp_norm:8.2f} {avg_bene:+10.4f} {avg_harm:+10.4f} {sep:+8.4f}")

    print(f"\n  平均: 有益轴cos={np.mean(all_beneficial):+.4f}  "
          f"有害轴cos={np.mean(all_harmful):+.4f}  "
          f"分离={np.mean(all_beneficial)-np.mean(all_harmful):+.4f}")


# ══════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("路线 A: CLIP → BERT 桥接 Token Projection")
    print("=" * 65)
    print(f"设备: {DEVICE}")

    kb = OFATKnowledgeBase()

    # Step 1: CLIP 空间 W_D
    proj_clip, axis_dirs_clip = step1_train_wd_clip()

    # Step 2: 学习 CLIP → BERT 映射
    mapper, bert_tok, bert_model, clip_model = step2_learn_mapping(
        kb, proj_clip, axis_dirs_clip
    )

    # Step 3: 构建 Pipeline + 验证
    pipeline = TokenProjectionPipeline(
        proj_clip, mapper, clip_model, bert_model, bert_tok
    )
    step3_verify(pipeline, kb)

    print("\n✅ 路线 A 完成")
