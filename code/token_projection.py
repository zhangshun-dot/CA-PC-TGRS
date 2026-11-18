# -*- coding: utf-8 -*-
"""Token Projection: 学习低秩投影矩阵将类名嵌入映射到视觉对齐空间

核心想法（第三篇论文）:
  不再是改 prompt 字符串（CD-TDM 离散路线），也不依赖 CLIP 逐图匹配（DTDM），
  而是直接在 BERT 嵌入空间学习一个低秩投影 W_D：

    E'_c = W_D · E_c = (I + U_D @ V_D^T) · E_c

  其中 U_D, V_D ∈ R^{d×r}, r << d。投影补偿了域偏移 (η_d ~20%)，
  将文本嵌入"校准"到目标域视觉编码器的对齐空间中。

与 OFAT 框架的关系:
  - Token Universality Score U(a) 决定每个轴的调整自由度
  - U(a) 越高 → W_D 沿该轴方向的调整越小 (冻结)
  - U(a) 越低 → W_D 沿该轴方向的调整越大 (可学习)

训练方式（代理损失，无需运行 GDINO）:
  1. 对每个类，从 OFAT 数据获取哪些修饰轴有帮助 (δ > 0)
  2. 计算每个修饰轴在 BERT 嵌入空间的"方向向量"
  3. 训练 W_D 使投影后的类嵌入与"有帮助的修饰方向"对齐
  4. 正则化: ||W_D - I||_F 惩罚（不要偏离太远）

Usage:
    from token_projection import TokenProjection, train_projection

    proj = TokenProjection(domain='remote_sensing', rank=4)
    train_projection(proj, kb, epochs=200)
    E_projected = proj(E_raw)  # 768-d → 768-d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ── 轴修饰词方向向量: 用 CLIP 文本编码器预计算 ──
# 这些方向向量表示"添加修饰词 a 会往嵌入空间的哪个方向移动"
# v_a = CLIP_text(f"{class_name}, {modifier_a}") - CLIP_text(f"{class_name}")
# 对所有类取平均得到轴级方向向量

AXIS_MODIFIERS = {
    'granularity': 'with fine details',
    'spatial': 'in its natural setting',
    'geometric': 'with distinct structural features',
    'scale': 'large and prominent',
    'spectral': 'with natural coloring',
    'texture': 'with clearly visible surface features',
}

AXIS_ORDER = ['granularity', 'spatial', 'geometric', 'scale', 'spectral', 'texture']

# 来自 token_universality_results.json 的 Universality Score
# U(a) 越高 → 该轴跨域稳定 → W_D 沿该轴调整量应越小
UNIVERSALITY_SCORES = {
    'geometric': 0.76,   # UNIVERSAL
    'spectral':  0.94,   # UNIVERSAL
    'scale':     0.62,   # MIXED
    'spatial':   0.58,   # MIXED
    'granularity': 0.50, # MIXED
    'texture':   0.46,   # MIXED
}


class TokenProjection(nn.Module):
    """低秩域投影矩阵 W_D = I + U_D @ V_D^T。

    设计原则:
      - rank r << d, 只学习跨域调整所需的少数方向
      - 初始化为接近恒等映射 (U, V 小随机初始化)
      - 正则化项 ∑_a (1-U(a)) · ||调整向量沿方向a的分量||²
        使通用轴冻结，混合轴可调
    """

    def __init__(self, d_model=768, rank=4, domain='remote_sensing',
                 universality_regularization=True):
        """
        Args:
            d_model: BERT 隐藏维度 (768 for bert-base-uncased)
            rank: W_D 的秩 (推荐 2-8)
            domain: 目标域标签
            universality_regularization: 是否使用 U(a) 加权正则化
        """
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.domain = domain

        # W_D = I + U @ V^T
        # U: (d, r), V: (d, r)
        # 初始化为小值 → W_D ≈ I
        self.U = nn.Parameter(torch.randn(d_model, rank) * 0.01 / np.sqrt(rank))
        self.V = nn.Parameter(torch.randn(d_model, rank) * 0.01 / np.sqrt(rank))

        self.use_ureg = universality_regularization

    def forward(self, x):
        """投影嵌入向量。

        Args:
            x: (..., d_model) — 原始类名嵌入

        Returns:
            (..., d_model) — 投影后的嵌入
        """
        # W_D = I + U @ V^T
        # E' = (I + U @ V^T) @ E = E + U @ (V^T @ E)
        x = x.float()  # ensure fp32 (CLIP outputs fp16)
        delta = x @ self.V  # (..., r)
        delta = delta @ self.U.T  # (..., d_model)
        return x + delta

    def get_projection_matrix(self):
        """返回完整的 W_D 矩阵 (用于分析和可视化)。"""
        with torch.no_grad():
            I = torch.eye(self.d_model, device=self.U.device)
            W = I + self.U @ self.V.T
        return W

    def get_adjustment_magnitude(self):
        """返回 ||W_D - I||_F / sqrt(d) — 归一化调整幅度。"""
        with torch.no_grad():
            adjustment = self.U @ self.V.T
            frob = adjustment.norm('fro').item()
            return frob / np.sqrt(self.d_model)

    def regularization_loss(self, axis_directions=None):
        """基于 Token Universality 的正则化损失。

        对每个轴 a，计算 W_D 沿该轴方向的调整分量：
          adj_a = direction_a^T @ (W_D - I) @ direction_a

        正则化损失 = Σ_a (1 - U(a)) · |adj_a|²
          - 高 U(a): (1-U) 小 → 允许少量调整
          - 低 U(a): (1-U) 大 → 鼓励更多调整

        Args:
            axis_directions: Dict[str, Tensor(d,)] — 每个轴的嵌入方向向量
                             如果为 None，返回 0

        Returns:
            scalar loss
        """
        if axis_directions is None or not self.use_ureg:
            return torch.tensor(0.0, device=self.U.device)

        adjustment = (self.U @ self.V.T).float()  # (d, d), ensure fp32
        loss = 0.0

        for axis in AXIS_ORDER:
            if axis not in axis_directions:
                continue
            d_vec = axis_directions[axis].float()  # (d,), ensure fp32
            d_vec = d_vec / (d_vec.norm() + 1e-8)

            # 投影: 调整矩阵沿方向 d_vec 的分量
            adj_along_axis = (d_vec @ adjustment @ d_vec).abs()

            # 权重: (1 - U(a))^2 — 通用轴权重小，混合轴权重大
            u_score = UNIVERSALITY_SCORES.get(axis, 0.5)
            weight = (1.0 - u_score) ** 2

            loss = loss + weight * adj_along_axis ** 2

        return loss


def compute_axis_directions(clip_model, clip_tokenize, device='cuda',
                            num_sample_classes=20):
    """用 CLIP 文本编码器计算每个轴的嵌入方向向量。

    方向向量 v_a = 平均过多个类的:
      CLIP_text("a photo of {class_name}, {modifier_a}")
      - CLIP_text("a photo of {class_name}")

    Args:
        clip_model: CLIP 模型
        clip_tokenize: CLIP tokenizer 函数
        device: 设备
        num_sample_classes: 用于估计的类别数

    Returns:
        Dict[str, torch.Tensor]: axis → (d_clip,) 方向向量
    """
    # 通用类别名（跨域共有的概念）
    generic_classes = [
        'airplane', 'bird', 'car', 'dog', 'cat', 'ship', 'bridge',
        'stadium', 'building', 'tree', 'road', 'person', 'truck',
        'bicycle', 'motorcycle', 'bus', 'train', 'chair', 'table', 'house',
    ][:num_sample_classes]

    axis_directions = {}

    for axis in AXIS_ORDER:
        modifier = AXIS_MODIFIERS[axis]
        directions = []

        for cls_name in generic_classes:
            base_prompt = f'a photo of {cls_name}'
            mod_prompt = f'a photo of {cls_name}, {modifier}'

            base_tok = clip_tokenize([base_prompt]).to(device)
            mod_tok = clip_tokenize([mod_prompt]).to(device)

            with torch.no_grad():
                base_emb = clip_model.encode_text(base_tok)
                mod_emb = clip_model.encode_text(mod_tok)
                base_emb = base_emb / base_emb.norm(dim=-1, keepdim=True)
                mod_emb = mod_emb / mod_emb.norm(dim=-1, keepdim=True)

            # 修饰词方向 = 加修饰词 - 不加修饰词 (在归一化空间中)
            direction = mod_emb - base_emb
            directions.append(direction.squeeze(0))

        # 平均方向
        mean_dir = torch.stack(directions).mean(dim=0)
        axis_directions[axis] = mean_dir / mean_dir.norm()

    return axis_directions


def train_projection(proj, kb, axis_directions=None, epochs=200, lr=0.01,
                     device='cuda', verbose=True):
    """用代理损失训练 W_D。

    代理损失 = 对齐损失 + 正则化损失

    对齐损失:
      对每个类 c 和每个轴 a:
        如果 δ(c, a) > 0: 鼓励 E'_c 向 v_a 方向移动
        如果 δ(c, a) ≤ 0: 保持原样

      L_align = - Σ_{c, a: δ>0} δ(c,a) · cos(E'_c - E_c, v_a) · |δ(c,a)|

    正则化损失:
      L_reg = proj.regularization_loss(axis_directions)

    Args:
        proj: TokenProjection 实例
        kb: OFATKnowledgeBase 实例
        axis_directions: Dict[str, Tensor] — 轴方向向量
        epochs: 训练轮数
        lr: 学习率
        device: 设备
        verbose: 打印训练日志

    Returns:
        Dict: 训练历史
    """
    from ofat_knowledge_base import OFATKnowledgeBase

    optimizer = torch.optim.Adam(proj.parameters(), lr=lr)
    history = {'loss': [], 'align_loss': [], 'reg_loss': [], 'magnitude': []}

    # ── 准备训练数据 ──
    domain = proj.domain
    delta_source = kb.dior_deltas if domain == 'remote_sensing' else kb.coco_deltas

    # 收集每个类的 OFAT δ 向量
    class_delta_vectors = {}  # class_name → {axis: delta}
    for (cls_name, axis), delta in delta_source.items():
        if cls_name not in class_delta_vectors:
            class_delta_vectors[cls_name] = {}
        class_delta_vectors[cls_name][axis] = delta

    # ── 用 CLIP 文本编码器获取类名嵌入 ──
    # (BERT 需要 HuggingFace 联网，CLIP 本地可用)
    import clip as clip_module
    clip_path_local = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
    clip_model_local, _ = clip_module.load(clip_path_local, device=device)
    clip_model_local.eval()

    # 检查维度一致性
    clip_dim = clip_model_local.text_projection.shape[1] if hasattr(clip_model_local, 'text_projection') else 512
    assert clip_dim == proj.d_model, f"CLIP dim ({clip_dim}) != proj.d_model ({proj.d_model})"

    # 为每个类计算 CLIP 文本嵌入
    class_embeddings = {}  # class_name → Tensor(d_clip,)

    for cls_name in class_delta_vectors:
        prompt = f'a photo of {cls_name}'
        tok = clip_module.tokenize([prompt], truncate=True).to(device)
        with torch.no_grad():
            emb = clip_model_local.encode_text(tok)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        class_embeddings[cls_name] = emb.squeeze(0).detach().float()

    if verbose:
        n_classes = len(class_embeddings)
        print(f"训练数据: {n_classes} 类, rank={proj.rank}")
        print(f"目标域: {domain}")

    # ── 训练循环 ──
    proj.train()
    proj.to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        total_align = 0.0
        total_reg = 0.0
        n_batches = 0

        for cls_name, emb_raw in class_embeddings.items():
            deltas = class_delta_vectors[cls_name]

            # 前向传播: E' = W_D · E
            emb_proj = proj(emb_raw.unsqueeze(0)).squeeze(0)

            # 对齐损失: 鼓励投影向有益轴方向移动
            align_loss = torch.tensor(0.0, device=device)

            for axis, delta in deltas.items():
                if delta <= 0:
                    continue  # 只对有益轴施加对齐力
                if axis not in axis_directions:
                    continue

                v_a = axis_directions[axis].to(device=device, dtype=torch.float32)
                # 投影位移向量
                displacement = emb_proj - emb_raw  # (d,)
                # 与轴方向的余弦相似度
                cos_sim = F.cosine_similarity(
                    displacement.unsqueeze(0), v_a.unsqueeze(0)
                )
                # 损失: 鼓励 displacement 与 v_a 对齐，权重正比于 δ
                align_loss = align_loss - delta * cos_sim

            # 正则化损失
            reg_loss = proj.regularization_loss(axis_directions)

            # 总损失
            loss = align_loss + 0.01 * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_align += align_loss.item()
            total_reg += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_align = total_align / max(n_batches, 1)
        avg_reg = total_reg / max(n_batches, 1)
        magnitude = proj.get_adjustment_magnitude()

        history['loss'].append(avg_loss)
        history['align_loss'].append(avg_align)
        history['reg_loss'].append(avg_reg)
        history['magnitude'].append(magnitude)

        if verbose and epoch % 20 == 0:
            print(f"  E{epoch:3d} | loss={avg_loss:.4f} | align={avg_align:.4f} | "
                  f"reg={avg_reg:.4f} | |Δ|={magnitude:.4f}")

    proj.eval()

    if verbose:
        print(f"\n训练完成: |W - I|_F/√d = {magnitude:.4f}")
        # 分析调整方向
        adjustment = (proj.U @ proj.V.T).detach()
        if axis_directions:
            print("\n各轴调整幅度 (|direction^T @ Δ @ direction|):")
            for axis in AXIS_ORDER:
                if axis in axis_directions:
                    d_vec = axis_directions[axis].to(device=device, dtype=torch.float32)
                    d_vec = d_vec / (d_vec.norm() + 1e-8)
                    adj = (d_vec @ adjustment @ d_vec).abs().item()
                    u = UNIVERSALITY_SCORES.get(axis, 0.5)
                    print(f"  {axis:14s} adj={adj:.6f}  (1-U)={(1-u)**2:.4f}  U={u:.2f}")

    return history


def evaluate_projection(proj, kb, axis_directions, device='cuda'):
    """评估 W_D 的投影效果。

    指标:
      1. 对齐得分: 投影后类嵌入与有益轴方向的平均余弦相似度
      2. 分离得分: 投影后有益/有害轴方向的可分离性
      3. Per-class 投影位移量

    Args:
        proj: TokenProjection 实例
        kb: OFATKnowledgeBase 实例
        axis_directions: 轴方向向量
        device: 设备

    Returns:
        Dict: 评估指标
    """
    import clip as clip_module
    clip_path_local = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
    clip_model_local, _ = clip_module.load(clip_path_local, device=device)
    clip_model_local.eval()

    domain = proj.domain
    delta_source = kb.dior_deltas if domain == 'remote_sensing' else kb.coco_deltas

    # 收集数据
    class_delta_vectors = {}
    for (cls_name, axis), delta in delta_source.items():
        if cls_name not in class_delta_vectors:
            class_delta_vectors[cls_name] = {}
        class_delta_vectors[cls_name][axis] = delta

    results = {'per_class': {}, 'summary': {}}

    all_beneficial_alignments = []  # δ>0 轴的投影对齐度
    all_harmful_alignments = []     # δ≤0 轴的投影对齐度

    for cls_name, deltas in class_delta_vectors.items():
        prompt = f'a photo of {cls_name}'
        tok = clip_module.tokenize([prompt], truncate=True).to(device)
        with torch.no_grad():
            emb_raw = clip_model_local.encode_text(tok)
            emb_raw = emb_raw / emb_raw.norm(dim=-1, keepdim=True)
            emb_raw = emb_raw.squeeze(0).float()
            emb_proj = proj(emb_raw.unsqueeze(0)).squeeze(0)

        displacement = emb_proj - emb_raw
        displacement_norm = displacement.norm().item()

        per_axis = {}
        for axis in AXIS_ORDER:
            if axis not in axis_directions:
                continue
            delta = deltas.get(axis, 0.0)
            v_a = axis_directions[axis].to(device)
            cos_sim = F.cosine_similarity(displacement.unsqueeze(0), v_a.unsqueeze(0)).item()

            per_axis[axis] = {
                'delta': delta,
                'alignment': cos_sim,
            }

            if delta > 0:
                all_beneficial_alignments.append(cos_sim)
            else:
                all_harmful_alignments.append(cos_sim)

        results['per_class'][cls_name] = {
            'displacement_norm': displacement_norm,
            'per_axis': per_axis,
        }

    results['summary'] = {
        'mean_beneficial_alignment': np.mean(all_beneficial_alignments) if all_beneficial_alignments else 0,
        'mean_harmful_alignment': np.mean(all_harmful_alignments) if all_harmful_alignments else 0,
        'alignment_separation': (
            np.mean(all_beneficial_alignments) - np.mean(all_harmful_alignments)
            if all_beneficial_alignments and all_harmful_alignments else 0
        ),
        'n_beneficial': len(all_beneficial_alignments),
        'n_harmful': len(all_harmful_alignments),
    }

    return results


# ── 与 GDINO 集成的接口 ──
def apply_projection_to_prompt(proj, prompt, bert_tokenizer, bert_model, device='cuda'):
    """将 W_D 应用于 prompt 中的类名 token 嵌入。

    这是一个独立的函数，设计为在 GDINO 推理前调用。
    修改 BERT 的 token embeddings 后通过 inputs_embeds 传入。

    Args:
        proj: TokenProjection 实例
        prompt: 完整的 prompt 字符串，如 "an overhead image of airplane, large and prominent"
        bert_tokenizer: BERT tokenizer
        bert_model: BERT model (用于获取初始嵌入)
        device: 设备

    Returns:
        torch.Tensor: 修改后的 inputs_embeds (1, seq_len, 768)
                      可直接传给 BERT 的 inputs_embeds 参数
    """
    tok = bert_tokenizer(prompt, return_tensors='pt').to(device)

    # 获取初始嵌入
    with torch.no_grad():
        emb_layer = bert_model.embeddings
        init_embeds = emb_layer(
            input_ids=tok['input_ids'],
            token_type_ids=tok.get('token_type_ids'),
        )

    # 对每个非特殊 token 应用 W_D
    # 找到 [CLS] 和 [SEP] 之间的 token 位置
    input_ids = tok['input_ids'][0]
    special_ids = {bert_tokenizer.cls_token_id, bert_tokenizer.sep_token_id,
                   bert_tokenizer.pad_token_id}

    with torch.no_grad():
        for pos in range(1, len(input_ids) - 1):  # 跳过 [CLS] 和 [SEP]
            if input_ids[pos].item() in special_ids:
                continue
            # 对该 token 嵌入应用 W_D
            init_embeds[0, pos, :] = proj(init_embeds[0, pos, :].unsqueeze(0)).squeeze(0)

    return init_embeds


# ══════════════════════════════════════════════════════════════════
# BERT 空间版本 —— 直接对接 GDINO 文本编码器
# ══════════════════════════════════════════════════════════════════

def compute_axis_directions_bert(bert_model, bert_tokenizer, device='cuda',
                                  num_sample_classes=20):
    """用 BERT 文本编码器计算每个轴的嵌入方向向量。

    v_a = mean over classes:
      BERT("a photo of {class}, {modifier}")[CLS]
      - BERT("a photo of {class}")[CLS]

    归一化后得到单位方向向量。
    """
    generic_classes = [
        'airplane', 'bird', 'car', 'dog', 'cat', 'ship', 'bridge',
        'stadium', 'building', 'tree', 'road', 'person', 'truck',
        'bicycle', 'motorcycle', 'bus', 'train', 'chair', 'table', 'house',
    ][:num_sample_classes]

    axis_directions = {}

    for axis in AXIS_ORDER:
        modifier = AXIS_MODIFIERS[axis]
        directions = []

        for cls_name in generic_classes:
            base_prompt = f'a photo of {cls_name}'
            mod_prompt = f'a photo of {cls_name}, {modifier}'

            base_tok = bert_tokenizer(base_prompt, return_tensors='pt').to(device)
            mod_tok = bert_tokenizer(mod_prompt, return_tensors='pt').to(device)

            with torch.no_grad():
                base_out = bert_model(**base_tok)
                mod_out = bert_model(**mod_tok)
                # 取 [CLS] token (位置 0) 作为句子表示
                base_emb = base_out.last_hidden_state[:, 0, :]
                mod_emb = mod_out.last_hidden_state[:, 0, :]

            direction = mod_emb - base_emb
            directions.append(direction.squeeze(0))

        mean_dir = torch.stack(directions).mean(dim=0)
        axis_directions[axis] = mean_dir / (mean_dir.norm() + 1e-8)

    return axis_directions


def train_projection_bert(proj, kb, bert_model, bert_tokenizer,
                           axis_directions=None, epochs=200, lr=0.01,
                           device='cuda', verbose=True):
    """在 BERT 嵌入空间中训练 W_D。

    与 train_projection 相同的代理损失，但使用 BERT [CLS] 嵌入
    而非 CLIP 嵌入。
    """
    optimizer = torch.optim.Adam(proj.parameters(), lr=lr)
    history = {'loss': [], 'align_loss': [], 'reg_loss': [], 'magnitude': []}

    domain = proj.domain
    delta_source = kb.dior_deltas if domain == 'remote_sensing' else kb.coco_deltas

    # 收集 OFAT δ
    class_delta_vectors = {}
    for (cls_name, axis), delta in delta_source.items():
        if cls_name not in class_delta_vectors:
            class_delta_vectors[cls_name] = {}
        class_delta_vectors[cls_name][axis] = delta

    # BERT 类名嵌入
    class_embeddings = {}
    for cls_name in class_delta_vectors:
        prompt = f'a photo of {cls_name}'
        tok = bert_tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = bert_model(**tok)
            emb = out.last_hidden_state[:, 0, :]  # [CLS]
        class_embeddings[cls_name] = emb.squeeze(0).detach().float()

    if verbose:
        print(f"训练数据: {len(class_embeddings)} 类, rank={proj.rank}, d_model={proj.d_model}")
        print(f"目标域: {domain}")

    proj.train()
    proj.to(device)

    for epoch in range(epochs):
        total_loss = total_align = total_reg = 0.0
        n_batches = 0

        for cls_name, emb_raw in class_embeddings.items():
            deltas = class_delta_vectors[cls_name]
            emb_proj = proj(emb_raw.unsqueeze(0)).squeeze(0)

            # 对齐损失
            align_loss = torch.tensor(0.0, device=device)
            for axis, delta in deltas.items():
                if delta <= 0 or axis not in axis_directions:
                    continue
                v_a = axis_directions[axis].to(device=device, dtype=torch.float32)
                displacement = emb_proj - emb_raw
                cos_sim = F.cosine_similarity(displacement.unsqueeze(0), v_a.unsqueeze(0))
                align_loss = align_loss - delta * cos_sim

            reg_loss = proj.regularization_loss(axis_directions)
            # Frobenius 约束: 防止 W_D 无限放大
            frob_reg = (proj.U.norm('fro')**2 + proj.V.norm('fro')**2) / proj.d_model
            loss = align_loss + 0.01 * reg_loss + 0.05 * frob_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_align += align_loss.item()
            total_reg += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_align = total_align / max(n_batches, 1)
        avg_reg = total_reg / max(n_batches, 1)
        magnitude = proj.get_adjustment_magnitude()

        history['loss'].append(avg_loss)
        history['align_loss'].append(avg_align)
        history['reg_loss'].append(avg_reg)
        history['magnitude'].append(magnitude)

        if verbose and epoch % 20 == 0:
            print(f"  E{epoch:3d} | loss={avg_loss:.4f} | align={avg_align:.4f} | "
                  f"reg={avg_reg:.4f} | |Δ|={magnitude:.4f}")

    proj.eval()

    if verbose:
        print(f"\n训练完成: |W - I|_F/√d = {magnitude:.4f}")
        adjustment = (proj.U @ proj.V.T).detach()
        if axis_directions:
            print("\n各轴调整幅度:")
            for axis in AXIS_ORDER:
                if axis in axis_directions:
                    d_vec = axis_directions[axis].to(device=device, dtype=torch.float32)
                    d_vec = d_vec / (d_vec.norm() + 1e-8)
                    adj = (d_vec @ adjustment @ d_vec).abs().item()
                    u = UNIVERSALITY_SCORES.get(axis, 0.5)
                    print(f"  {axis:14s} adj={adj:.6f}  (1-U)²={(1-u)**2:.4f}  U={u:.2f}")

    return history


def evaluate_projection_bert(proj, kb, bert_model, bert_tokenizer,
                              axis_directions, device='cuda'):
    """在 BERT 空间中评估 W_D 投影效果。"""
    domain = proj.domain
    delta_source = kb.dior_deltas if domain == 'remote_sensing' else kb.coco_deltas

    class_delta_vectors = {}
    for (cls_name, axis), delta in delta_source.items():
        if cls_name not in class_delta_vectors:
            class_delta_vectors[cls_name] = {}
        class_delta_vectors[cls_name][axis] = delta

    results = {'per_class': {}, 'summary': {}}
    all_beneficial_alignments = []
    all_harmful_alignments = []

    for cls_name, deltas in class_delta_vectors.items():
        prompt = f'a photo of {cls_name}'
        tok = bert_tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = bert_model(**tok)
            emb_raw = out.last_hidden_state[:, 0, :].squeeze(0).float()
            emb_proj = proj(emb_raw.unsqueeze(0)).squeeze(0)

        displacement = emb_proj - emb_raw
        displacement_norm = displacement.norm().item()

        per_axis = {}
        for axis in AXIS_ORDER:
            if axis not in axis_directions:
                continue
            delta = deltas.get(axis, 0.0)
            v_a = axis_directions[axis].to(device=device, dtype=torch.float32)
            cos_sim = F.cosine_similarity(displacement.unsqueeze(0), v_a.unsqueeze(0)).item()
            per_axis[axis] = {'delta': delta, 'alignment': cos_sim}

            if delta > 0:
                all_beneficial_alignments.append(cos_sim)
            else:
                all_harmful_alignments.append(cos_sim)

        results['per_class'][cls_name] = {
            'displacement_norm': displacement_norm,
            'per_axis': per_axis,
        }

    results['summary'] = {
        'mean_beneficial_alignment': np.mean(all_beneficial_alignments) if all_beneficial_alignments else 0,
        'mean_harmful_alignment': np.mean(all_harmful_alignments) if all_harmful_alignments else 0,
        'alignment_separation': (
            np.mean(all_beneficial_alignments) - np.mean(all_harmful_alignments)
            if all_beneficial_alignments and all_harmful_alignments else 0
        ),
        'n_beneficial': len(all_beneficial_alignments),
        'n_harmful': len(all_harmful_alignments),
    }
    return results


# ── 快速验证 ──
if __name__ == '__main__':
    import os, json
    os.environ['HF_HUB_OFFLINE'] = '1'

    from ofat_knowledge_base import OFATKnowledgeBase
    from transformers import BertTokenizer, BertModel

    print("=" * 65)
    print("Token Projection — BERT 空间训练 (对接 GDINO)")
    print("=" * 65)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    # 1. 加载模型
    print("\n[1/5] 加载 BERT + OFAT_KB...")
    bert_tok = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    bert_model = BertModel.from_pretrained('bert-base-uncased', local_files_only=True)
    bert_model.eval().to(device)
    kb = OFATKnowledgeBase()

    # 2. 计算 BERT 空间中的轴方向向量
    print("\n[2/5] 计算轴方向向量 (BERT 文本编码器)...")
    axis_dirs = compute_axis_directions_bert(bert_model, bert_tok, device=device)

    print("  轴方向间余弦相似度:")
    for i, a1 in enumerate(AXIS_ORDER):
        if a1 not in axis_dirs:
            continue
        sims = []
        for a2 in AXIS_ORDER:
            if a2 not in axis_dirs:
                continue
            sim = F.cosine_similarity(axis_dirs[a1].unsqueeze(0),
                                      axis_dirs[a2].unsqueeze(0)).item()
            sims.append(f"{sim:+.3f}")
        print(f"  {a1:14s} {' '.join(sims)}")

    # 3. 训练 W_D (BERT 768-d 空间)
    print("\n[3/5] 训练 DIOR 域投影 (BERT 768-d, rank=4)...")
    proj = TokenProjection(d_model=768, rank=4, domain='remote_sensing')

    history = train_projection_bert(
        proj, kb, bert_model, bert_tok,
        axis_directions=axis_dirs,
        epochs=200, lr=0.01, device=device, verbose=True
    )

    # 4. 评估
    print("\n[4/5] 评估投影效果...")
    results = evaluate_projection_bert(proj, kb, bert_model, bert_tok,
                                        axis_dirs, device=device)

    s = results['summary']
    print(f"\n  评估结果:")
    print(f"    有益轴平均对齐: {s['mean_beneficial_alignment']:+.4f}")
    print(f"    有害轴平均对齐: {s['mean_harmful_alignment']:+.4f}")
    print(f"    分离度:         {s['alignment_separation']:+.4f}")

    print(f"\n  Per-class 投影位移量 (top 8):")
    per_class = sorted(results['per_class'].items(),
                       key=lambda x: -x[1]['displacement_norm'])
    for cls_name, info in per_class[:8]:
        print(f"  {cls_name:<26s} |Δ|={info['displacement_norm']:.4f}")

    # 5. 保存 W_D 权重
    print("\n[5/5] 保存训练权重...")
    save_dir = Path(r'C:\Users\张三\Desktop\实验数据')
    save_dir.mkdir(exist_ok=True)

    # 保存 proj 参数
    torch.save({
        'U': proj.U.data.cpu(),
        'V': proj.V.data.cpu(),
        'd_model': proj.d_model,
        'rank': proj.rank,
        'domain': proj.domain,
    }, save_dir / 'token_projection_dior_bert.pt')

    # 保存轴方向向量
    axis_dirs_cpu = {a: d.cpu() for a, d in axis_dirs.items()}
    torch.save(axis_dirs_cpu, save_dir / 'axis_directions_bert.pt')

    # 保存评估结果
    eval_save = {
        'summary': results['summary'],
        'per_class': {cls: {'displacement_norm': info['displacement_norm']}
                      for cls, info in results['per_class'].items()},
        'history': {k: v for k, v in history.items()},
        'axis_adjustments': {},
    }
    adjustment = (proj.U @ proj.V.T).detach()
    for axis in AXIS_ORDER:
        if axis in axis_dirs:
            d_vec = axis_dirs[axis].to(device=device, dtype=torch.float32)
            d_vec = d_vec / (d_vec.norm() + 1e-8)
            adj = (d_vec @ adjustment @ d_vec).abs().item()
            eval_save['axis_adjustments'][axis] = {
                'adj': adj,
                'U_score': UNIVERSALITY_SCORES.get(axis, 0.5),
                'weight': (1.0 - UNIVERSALITY_SCORES.get(axis, 0.5)) ** 2,
            }

    with open(save_dir / 'token_projection_eval.json', 'w', encoding='utf-8') as f:
        json.dump(eval_save, f, indent=2, ensure_ascii=False)

    print(f"  权重: {save_dir / 'token_projection_dior_bert.pt'}")
    print(f"  轴方向: {save_dir / 'axis_directions_bert.pt'}")
    print(f"  评估: {save_dir / 'token_projection_eval.json'}")

    print("\n✅ Token Projection (BERT 空间) 训练完成")
