# -*- coding: utf-8 -*-
"""DTDM 轴-CLIP 视觉描述映射与嵌入预计算

设计原则：
  AXIS_DESCRIPTIONS ≠ AXIS_MODIFIERS
  - DESCRIPTIONS: 给 CLIP 做 image-text semantic matching 的自然语言视觉描述
  - MODIFIERS:    实际插入 prompt 的固定修饰词 token（来自 domain_registry.py）

每条 DESCRIPTION 描述的是"什么样的图像会从该轴受益"——CLIP 通过
cosine similarity 判断当前图像是否匹配这个视觉特征。

预计算的文本嵌入在 DTDM 推理时直接查表，避免重复编码。
"""

import torch
import numpy as np
from pathlib import Path

# ── 复用的 AXIS_MODIFIERS（来自 domain_registry.py） ──
# 这是实际插入 prompt 的修饰词 token
AXIS_MODIFIERS = {
    'granularity': 'with fine details',
    'spatial':     'in its natural setting',
    'geometric':   'with distinct structural features',
    'scale':       'large and prominent',
    'spectral':    'with natural coloring',
    'texture':     'with clearly visible surface features',
}

# ── AXIS_DESCRIPTIONS: CLIP 视觉描述（用于 image-text matching） ──
# 每条描述回答："具有什么视觉特征的图像会从该轴修饰词中受益？"
# 设计时考虑了 CLIP 的图文对齐能力——用具体、视觉化的语言

AXIS_DESCRIPTIONS = {
    'scale': (
        "large objects filling the frame, big prominent features, "
        "objects taking up most of the image area"
    ),
    'granularity': (
        "fine intricate details visible, close-up view showing small features, "
        "detailed textures and patterns clearly visible"
    ),
    'spatial': (
        "objects shown in their surrounding environment, scene with context, "
        "objects placed within a wider landscape or setting"
    ),
    'geometric': (
        "strong geometric shapes, angular structures, straight lines, "
        "architectural forms with clear edges and corners"
    ),
    'spectral': (
        "vivid colors present, strong color contrasts, bright and colorful scene, "
        "distinctive color variations"
    ),
    'texture': (
        "surface textures clearly visible, material roughness or smoothness, "
        "patterned surfaces, repetitive visual patterns"
    ),
}

# 轴排序（与 domain_registry.py 一致）
AXIS_ORDER = ['granularity', 'spatial', 'geometric', 'scale', 'spectral', 'texture']


def precompute_axis_embeddings(clip_model, device='cuda'):
    """预计算所有轴描述的 CLIP 文本嵌入。

    Args:
        clip_model: OpenAI CLIP model (loaded via clip.load())
        device: 'cuda' or 'cpu'

    Returns:
        Dict[str, torch.Tensor]: axis_name → normalized text embedding (512-d)
    """
    import clip as clip_module

    texts = []
    axis_names = []
    for axis in AXIS_ORDER:
        desc = AXIS_DESCRIPTIONS[axis]
        texts.append(desc)
        axis_names.append(axis)

    # Tokenize all descriptions at once for efficiency
    text_tokens = clip_module.tokenize(texts, truncate=True).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        # Normalize for cosine similarity
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return {axis: text_features[i] for i, axis in enumerate(axis_names)}


def precompute_modifier_embeddings(clip_model, device='cuda'):
    """预计算修饰词本身的 CLIP 文本嵌入（用于可选的 modifier-image 直接匹配）。

    Args:
        clip_model: OpenAI CLIP model
        device: 'cuda' or 'cpu'

    Returns:
        Dict[str, torch.Tensor]: axis_name → normalized modifier embedding (512-d)
    """
    import clip as clip_module

    texts = []
    axis_names = []
    for axis in AXIS_ORDER:
        mod = AXIS_MODIFIERS[axis]
        texts.append(mod)
        axis_names.append(axis)

    text_tokens = clip_module.tokenize(texts, truncate=True).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return {axis: text_features[i] for i, axis in enumerate(axis_names)}


# ── 快速验证 ──
if __name__ == '__main__':
    import clip

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    clip_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
    model, preprocess = clip.load(clip_path, device=device)
    model.eval()

    axis_emb = precompute_axis_embeddings(model, device)
    mod_emb = precompute_modifier_embeddings(model, device)

    print("\nAxis Description Embeddings (pre-computed):")
    for axis, emb in axis_emb.items():
        print(f"  {axis:14s} shape={emb.shape}  norm={emb.norm().item():.4f}")

    # 检查 description 和 modifier 之间的语义对齐
    print("\nDescription ↔ Modifier cosine similarity:")
    for axis in AXIS_ORDER:
        sim = torch.dot(axis_emb[axis], mod_emb[axis]).item()
        print(f"  {axis:14s} cos_sim={sim:.4f}")

    # 检查不同轴 description 之间的区分度
    print("\nInter-axis description similarity matrix:")
    for i, a1 in enumerate(AXIS_ORDER):
        row = []
        for j, a2 in enumerate(AXIS_ORDER):
            sim = torch.dot(axis_emb[a1], axis_emb[a2]).item()
            row.append(f"{sim:.3f}")
        print(f"  {a1:14s} {'  '.join(row)}")

    print("\n✅ AXIS_DESCRIPTIONS 验证完成")
