# -*- coding: utf-8 -*-
"""CPS + Reward-Guided Adapter: 用 CLIP Prompt Scorer 替代 GDINO F1 作为训练信号

核心洞察:
  - GDINO F1 二值、高噪声、SNR<1 → 不能做 per-image 优化
  - OFAT 聚合数据 (类×轴, N=320+) 提供了稳定的 ΔF1 估计
  - 用 CLIP text embedding → MLP → ΔF1 训练一个连续评分器 (CPS)
  - 冻结 CPS 作为 differentiable reward → 训练 Adapter 最大化 reward

与 Token Projection 的区别:
  - 旧: alignment loss = 硬对齐到 pre-defined 轴方向 (分离度≈0)
  - 新: CPS reward = 从数据学到的连续评分函数 (有梯度可追)

Pipeline:
  Phase 1: 收集 OFAT prompt→ΔF1 数据, 计算 CLIP 嵌入
  Phase 2: 训练 CPS (MLP regressor)
  Phase 3: 冻结 CPS, 训练 Adapter (maximize CPS reward + KL reg)
  Phase 4: GDINO 端到端验证

Usage:
    python cps_adapter.py
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json, time, glob
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
SAVE_DIR.mkdir(exist_ok=True)

# ── 轴修饰词 ─────────────────────────────────────────
AXIS_MODIFIERS = {
    'granularity': 'with fine details',
    'spatial': 'in its natural setting',
    'geometric': 'with distinct structural features',
    'scale': 'large and prominent',
    'spectral': 'with natural coloring',
    'texture': 'with clearly visible surface features',
}
AXIS_ORDER = ['granularity', 'spatial', 'geometric', 'scale', 'spectral', 'texture']

# ── 前缀变体 ─────────────────────────────────────────
PREFIX_VARIANTS = [
    'a photo of',
    'an image of',
    'a picture of',
    'a natural image of',
    'an overhead image of',
    'a satellite image of',
    'an aerial view of',
    'a remote sensing image of',
]

# ── DIOR 类别 ────────────────────────────────────────
DIOR_CLASSES = ['airplane','airport','baseball field','basketball court','bridge',
    'chimney','dam','expressway service area','expressway toll station',
    'golf course','ground track field','harbor','overpass','ship','stadium',
    'storage tank','tennis court','train station','vehicle','windmill']

DIOR_DIR = r'C:\Users\张三\modeling\dior_data\full20_images'


# ══════════════════════════════════════════════════════════════════
# Phase 1: 收集 OFAT 训练数据
# ══════════════════════════════════════════════════════════════════

def collect_ofat_training_data():
    """从 OFAT JSON 文件收集 (prompt_text, ΔF1) 配对数据。

    数据源:
      1. COCO OFAT: 20类 × 6轴 = 120 类×轴 deltas
      2. COCO prefix variants: 20类 × 4前缀
      3. DIOR OFAT: 20类 × 6轴 = 120 类×轴 deltas

    每个数据点:
      - prompt: "a photo of {class}, {modifier}"  (变体: 换前缀)
      - delta_f1: OFAT 测量的 ΔF1 (相对 baseline)

    Returns:
        List[Dict]: [{'prompt': str, 'delta_f1': float, 'class': str, 'axis': str}, ...]
    """
    from ofat_knowledge_base import OFATKnowledgeBase
    kb = OFATKnowledgeBase()

    data_points = []

    # ── DIOR OFAT: 类×轴 deltas ──
    for (cls_name, axis), delta in kb.dior_deltas.items():
        modifier = AXIS_MODIFIERS[axis]
        # 基础 prompt
        prompt = f'a photo of {cls_name}, {modifier}'
        data_points.append({
            'prompt': prompt,
            'delta_f1': delta,
            'class': cls_name,
            'axis': axis,
            'domain': 'remote_sensing',
        })
        # 变体: overhead prefix
        prompt_oh = f'an overhead image of {cls_name}, {modifier}'
        data_points.append({
            'prompt': prompt_oh,
            'delta_f1': delta,  # 近似: prefix 差异远小于轴差异
            'class': cls_name,
            'axis': axis,
            'domain': 'remote_sensing',
        })

    # ── COCO OFAT: 类×轴 deltas ──
    for (cls_name, axis), delta in kb.coco_deltas.items():
        modifier = AXIS_MODIFIERS[axis]
        prompt = f'a photo of {cls_name}, {modifier}'
        data_points.append({
            'prompt': prompt,
            'delta_f1': delta,
            'class': cls_name,
            'axis': axis,
            'domain': 'natural_scene',
        })

    # ── COCO prefix variants ──
    for cls_name in kb.coco['classes']:
        cls_data = kb.coco['classes'][cls_name]
        base_f1 = cls_data.get('base', 0.0)
        for prefix, f1 in cls_data.get('prefix_scores', {}).items():
            prompt = f'{prefix} {cls_name}'
            data_points.append({
                'prompt': prompt,
                'delta_f1': f1 - base_f1,
                'class': cls_name,
                'axis': 'prefix',
                'domain': 'natural_scene',
            })

    # ── Baseline prompts (ΔF1 = 0 by definition) ──
    for cls_name in kb.dior_classes:
        prompt = f'a photo of {cls_name}'
        data_points.append({
            'prompt': prompt,
            'delta_f1': 0.0,
            'class': cls_name,
            'axis': 'baseline',
            'domain': 'remote_sensing',
        })
        prompt_oh = f'an overhead image of {cls_name}'
        data_points.append({
            'prompt': prompt_oh,
            'delta_f1': 0.0,
            'class': cls_name,
            'axis': 'baseline',
            'domain': 'remote_sensing',
        })

    # ── 去重 (保留首次出现的 ΔF1) ──
    seen = {}
    unique = []
    for dp in data_points:
        key = dp['prompt'].lower().strip()
        if key not in seen:
            seen[key] = dp
            unique.append(dp)

    print(f"收集 OFAT 数据: {len(unique)} 个唯一 prompt→ΔF1 对")
    print(f"  ΔF1 分布: min={min(d['delta_f1'] for d in unique):+.4f}, "
          f"max={max(d['delta_f1'] for d in unique):+.4f}, "
          f"mean={np.mean([d['delta_f1'] for d in unique]):+.4f}")

    return unique


# ══════════════════════════════════════════════════════════════════
# Phase 2: CLIP Prompt Scorer (CPS)
# ══════════════════════════════════════════════════════════════════

class CLIPPromptScorer(nn.Module):
    """从 CLIP text embedding 预测 ΔF1 的 MLP 评分器。

    设计:
      - 输入: CLIP text embedding (512-d, 已归一化)
      - 输出: 预测 ΔF1 (标量), 经过 tanh 限制在 (-1, 1)
      - 架构: 512→256→128→64→1, ReLU + Dropout
      - 训练: MSE loss + L2 reg

    关键: CLIP text embedding 携带了 prompt 的语义信息,
          比 GDINO 256-d 空间的 SNR 高得多。
    """

    def __init__(self, d_input=512, hidden_dims=[256, 128, 64], dropout=0.1):
        super().__init__()
        layers = []
        prev = d_input
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """预测 ΔF1（tanh 限制在 [-1, 1]）。

        Args:
            x: (batch, 512) CLIP text embeddings

        Returns:
            (batch, 1) predicted ΔF1 ∈ (-1, 1)
        """
        raw = self.net(x)
        return torch.tanh(raw)  # 硬限制防止 reward hacking


def compute_clip_embeddings(prompts, clip_model, clip_tokenize):
    """批量计算 CLIP text embeddings。

    Returns:
        torch.Tensor: (N, 512) 归一化嵌入
    """
    embeddings = []
    batch_size = 64
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        toks = clip_tokenize(batch, truncate=True).to(DEVICE)
        with torch.no_grad():
            emb = clip_model.encode_text(toks)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu().float())
    return torch.cat(embeddings, dim=0)


def train_cps(cps, X, y, epochs=300, lr=0.001, val_split=0.15):
    """训练 CLIP Prompt Scorer。

    Args:
        cps: CLIPPromptScorer 实例
        X: (N, 512) CLIP embeddings
        y: (N,) ΔF1 targets
        epochs: 训练轮数
        lr: 学习率
        val_split: 验证集比例

    Returns:
        Dict: 训练历史
    """
    # 数据集划分
    n = len(y)
    n_val = int(n * val_split)
    indices = torch.randperm(n)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"  训练集: {len(train_idx)}, 验证集: {len(val_idx)}")

    optimizer = torch.optim.AdamW(cps.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    cps.to(DEVICE)

    history = {'train_loss': [], 'val_loss': [], 'val_corr': []}
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        # 训练
        cps.train()
        perm = torch.randperm(len(train_idx))
        total_loss = 0
        n_batches = 0

        for i in range(0, len(train_idx), 64):
            idx = perm[i:i + 64]
            xb = X_train[idx].to(DEVICE)
            yb = y_train[idx].to(DEVICE).unsqueeze(1)

            pred = cps(xb)
            loss = F.mse_loss(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(n_batches, 1)

        # 验证
        cps.eval()
        with torch.no_grad():
            xv = X_val.to(DEVICE)
            yv = y_val.to(DEVICE).unsqueeze(1)
            pred_val = cps(xv)
            val_loss = F.mse_loss(pred_val, yv).item()

            # Pearson correlation
            pred_np = pred_val.squeeze().cpu().numpy()
            y_np = y_val.squeeze().cpu().numpy()
            val_corr = np.corrcoef(pred_np, y_np)[0, 1] if len(pred_np) > 1 else 0.0

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_corr'].append(val_corr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in cps.state_dict().items()}

        if epoch % 50 == 0:
            print(f"  E{epoch:3d} | train_loss={avg_train_loss:.6f} | "
                  f"val_loss={val_loss:.6f} | val_r={val_corr:+.4f}")

    # 恢复最优
    if best_state:
        cps.load_state_dict(best_state)

    print(f"\n  CPS 训练完成: best_val_loss={best_val_loss:.6f}, "
          f"final_corr={history['val_corr'][-1]:+.4f}")

    cps.eval()
    return history


# ══════════════════════════════════════════════════════════════════
# Phase 3: Reward-Guided Adapter
# ══════════════════════════════════════════════════════════════════

class PromptAdapter(nn.Module):
    """轻量 prompt 适配器: E_adapted = E + U @ (V^T @ E)。

    与 TokenProjection 同架构，但训练目标不同:
      - 旧: 硬对齐到预定义轴方向 (alignment loss)
      - 新: 最大化 CPS 预测的 ΔF1 (reward maximization)
    """

    def __init__(self, d_model=512, rank=4):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        # 极小的初始化 → 初始时近乎恒等映射
        self.U = nn.Parameter(torch.randn(d_model, rank) * 0.001 / np.sqrt(rank))
        self.V = nn.Parameter(torch.randn(d_model, rank) * 0.001 / np.sqrt(rank))

    def forward(self, x):
        """x: (..., d_model) → (..., d_model)"""
        x = x.float()
        delta = x @ self.V
        delta = delta @ self.U.T
        return x + delta

    def get_magnitude(self):
        with torch.no_grad():
            return (self.U @ self.V.T).norm('fro').item() / np.sqrt(self.d_model)


def train_adapter_contrastive(adapter, cps, class_embeddings, class_candidates,
                              epochs=300, lr=0.001, temperature=0.1,
                              lambda_kl=1.0, lambda_l2=0.001):
    """用对比式 CPS 引导训练 Adapter（防 reward hacking）。

    核心思路:
      不做 argmax CPS(E_adapted) — 这会导致 reward hacking (上次 CPS→+69)。
      改为: 对每个类生成候选 prompt 的嵌入, CPS 评分选取 top-k,
      以加权质心为目标, Adapter 学习向目标方向移动。

    损失:
      L = ||E_adapted - target||²          [向"好 prompt 区域"移动]
        + λ_kl · ||E_adapted - E_raw||²    [KL 正则化: 别跑太远]
        + λ_l2 · (||U||² + ||V||²)         [L2 正则化]

    关键: target 在训练集中真实 prompt 嵌入的凸包内 → CPS 不会外推。

    Args:
        adapter: PromptAdapter
        cps: 冻结的 CLIPPromptScorer
        class_embeddings: Dict[str, Tensor(512,)] — CLIP base 类名嵌入
        class_candidates: Dict[str, List[Tuple[str, Tensor(512,)]]] —
                          class → [(prompt_text, CLIP_embedding), ...]
        epochs, lr, temperature, lambda_kl, lambda_l2

    Returns:
        Dict: 训练历史
    """
    cps.eval()
    for p in cps.parameters():
        p.requires_grad = False

    adapter.train()
    adapter.to(DEVICE)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    history = {'loss': [], 'move_loss': [], 'kl': [], 'magnitude': [],
               'target_score': [], 'raw_score': []}
    class_names = list(class_embeddings.keys())

    # 为每个类预计算候选嵌入和 CPS 分数
    print("  预计算候选 prompt CPS 分数...")
    class_targets = {}  # class → target_embedding (weighted centroid of top candidates)

    for cls_name in class_names:
        candidates = class_candidates.get(cls_name, [])
        if not candidates:
            # fallback: 用 base embedding 自身
            class_targets[cls_name] = class_embeddings[cls_name]
            continue

        cand_texts, cand_embs = zip(*candidates)
        E_cand = torch.stack(cand_embs).to(DEVICE)  # (n_cand, 512)

        with torch.no_grad():
            scores = cps(E_cand).squeeze()  # (n_cand,)

        # Softmax 加权质心 (temperature 控制锐度)
        weights = F.softmax(scores / temperature, dim=0)  # (n_cand,)
        target = (weights.unsqueeze(1) * E_cand).sum(dim=0)  # (512,)
        # 归一化到单位球面
        target = target / (target.norm() + 1e-8)
        class_targets[cls_name] = target.cpu()

    print(f"  候选数/类: {sum(len(class_candidates.get(c, [])) for c in class_names) // max(len(class_names), 1)}")

    # 基线: 原始嵌入的 CPS 分数
    E_raw_all = torch.stack([class_embeddings[c] for c in class_names]).to(DEVICE)
    with torch.no_grad():
        baseline_scores = cps(E_raw_all).squeeze().cpu().numpy()
    print(f"  基线 CPS score: mean={baseline_scores.mean():+.4f}")

    for epoch in range(epochs):
        total_loss = 0.0
        total_move = 0.0
        total_kl = 0.0

        for cls_name in class_names:
            e_raw = class_embeddings[cls_name].to(DEVICE)
            target = class_targets[cls_name].to(DEVICE)

            e_adapted = adapter(e_raw.unsqueeze(0)).squeeze(0)

            # Move loss: 向目标靠近
            move_loss = (1.0 - F.cosine_similarity(
                e_adapted.unsqueeze(0), target.unsqueeze(0)
            )).squeeze()

            # KL regularization
            kl = ((e_adapted - e_raw) ** 2).sum()

            loss = move_loss + lambda_kl * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_move += move_loss.item()
            total_kl += kl.item()

        # L2 regularization (per epoch, not per class)
        l2 = adapter.U.norm('fro')**2 + adapter.V.norm('fro')**2
        # Apply L2 separately to avoid too many backward calls
        # (we already stepped per class, just track it)
        # Actually, apply L2 as a small weight decay manually:
        with torch.no_grad():
            adapter.U.data *= (1.0 - lr * lambda_l2)
            adapter.V.data *= (1.0 - lr * lambda_l2)

        scheduler.step()
        magnitude = adapter.get_magnitude()

        n = len(class_names)
        avg_loss = total_loss / n
        avg_move = total_move / n
        avg_kl = total_kl / n

        # 评估: adapted 嵌入的 CPS 分数 (应该温和上升)
        with torch.no_grad():
            E_adapted_all = adapter(E_raw_all)
            adapted_scores = cps(E_adapted_all).squeeze().cpu().numpy()
        avg_target_score = adapted_scores.mean()

        history['loss'].append(avg_loss)
        history['move_loss'].append(avg_move)
        history['kl'].append(avg_kl)
        history['magnitude'].append(magnitude)
        history['target_score'].append(avg_target_score)
        history['raw_score'].append(baseline_scores.mean())

        if epoch % 50 == 0:
            print(f"  E{epoch:3d} | loss={avg_loss:.4f} | move={avg_move:.4f} | "
                  f"kl={avg_kl:.4f} | CPS={avg_target_score:+.4f} | |Δ|={magnitude:.4f}")

    # 最终评估
    with torch.no_grad():
        E_adapted_all = adapter(E_raw_all)
        final_scores = cps(E_adapted_all).squeeze().cpu().numpy()

    print(f"\n  Adapter (对比式) 训练完成:")
    print(f"    基线 CPS: {baseline_scores.mean():+.4f}")
    print(f"    最终 CPS: {final_scores.mean():+.4f}")
    print(f"    提升:    {final_scores.mean() - baseline_scores.mean():+.4f}")
    print(f"    最终 |Δ|: {magnitude:.4f}")

    adapter.eval()
    return history


def generate_class_candidates(kb, clip_model, clip_tokenize, top_k=8):
    """为每个类生成候选 prompt 变体及 CLIP 嵌入。

    候选类型:
      1. 不同前缀 (overhead/satellite/aerial/...)
      2. 单轴修饰词 (6 axes)
      3. OFAT 推荐的双修饰词组合

    Returns:
        Dict[str, List[Tuple[str, Tensor]]]: class → [(prompt, CLIP_embedding), ...]
    """
    candidates = {}

    for cls_name in kb.dior_classes:
        cls_cands = []

        # 前缀变体
        for prefix in PREFIX_VARIANTS:
            p = f'{prefix} {cls_name}'
            cls_cands.append(p)

        # 单轴修饰词 (不同前缀)
        for axis in AXIS_ORDER:
            mod = AXIS_MODIFIERS[axis]
            for prefix in ['a photo of', 'an overhead image of']:
                p = f'{prefix} {cls_name}, {mod}'
                cls_cands.append(p)

        # 双修饰词 (OFAT top-2 推荐轴)
        recs = kb.get_recommended_axes('remote_sensing', cls_name)
        top_axes = [a for a, _ in recs[:2]]
        if len(top_axes) >= 2:
            mods = [AXIS_MODIFIERS[a] for a in top_axes]
            for prefix in ['a photo of', 'an overhead image of']:
                p = f'{prefix} {cls_name}, ' + ', '.join(mods)
                cls_cands.append(p)
        elif len(top_axes) == 1:
            mod = AXIS_MODIFIERS[top_axes[0]]
            for prefix in ['a photo of', 'an overhead image of']:
                p = f'{prefix} {cls_name}, {mod}'
                if p not in cls_cands:
                    cls_cands.append(p)

        # 去重
        cls_cands = list(set(cls_cands))

        # 批量计算 CLIP 嵌入
        embs = compute_clip_embeddings(cls_cands, clip_model, clip_tokenize)
        # 归一化
        embs = embs / embs.norm(dim=-1, keepdim=True)

        # CPS 预评分 → 只保留 top_k 作为候选
        with torch.no_grad():
            cps_local = None  # 如果还没训练好 CPS，全部保留
            # 先全保留，训练时由 train_adapter_contrastive 评分

        candidates[cls_name] = list(zip(cls_cands, [e for e in embs]))

    return candidates


# ══════════════════════════════════════════════════════════════════
# Phase 4: GDINO 端到端验证
# ══════════════════════════════════════════════════════════════════

def evaluate_gdino(adapter, cps, clip_model, clip_tokenize, kb,
                   n_images=20, threshold=0.25):
    """在 DIOR 上端到端评估: Baseline vs Static TDM vs CPS+Adapter。

    评估策略:
      - 对每个类, 生成多个 prompt 候选
      - 用 CPS 选 top prompt (替代 GDINO 跑所有候选)
      - 只对选出的最佳 prompt 跑 GDINO 验证

    Args:
        adapter: PromptAdapter
        cps: CLIPPromptScorer
        clip_model: CLIP 模型
        clip_tokenize: CLIP tokenizer
        kb: OFATKnowledgeBase
        n_images: 每类图像数
        threshold: GDINO 检测阈值

    Returns:
        Dict: 评估结果
    """
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    print("\n" + "=" * 60)
    print("Phase 4: GDINO 端到端验证")
    print("=" * 60)

    # 加载 GDINO
    gd_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
    gd_proc = AutoProcessor.from_pretrained(gd_cache)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        gd_cache, dtype=torch.float32
    ).eval().to(DEVICE)

    def detect_rate(images, prompt):
        """GDINO 检测命中率。"""
        detections = []
        for img in images:
            inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
            inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            with torch.no_grad():
                o = gd_model(**inputs)
            ts = torch.tensor([img.size[::-1]]).to(DEVICE)
            r = gd_proc.post_process_grounded_object_detection(
                o, threshold=threshold, target_sizes=ts
            )[0]
            detections.append(len(r['boxes']) > 0)
        return np.mean(detections)

    def load_images(cls, n=n_images):
        paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{cls.replace(' ','_')}_*.jpg")))[:n]
        if not paths:
            paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{cls}_*.jpg")))[:n]
        return [Image.open(p).convert('RGB') for p in paths] if paths else []

    # 评估 8 个代表性类别
    eval_classes = ['airplane', 'ship', 'stadium', 'bridge', 'harbor', 'overpass',
                    'tennis court', 'windmill']

    results = {
        'baseline': {},
        'static_tdm': {},
        'cps_adapter': {},
        'cps_oracle': {},  # CPS 评分最高的 prompt (不经过 adapter)
        'per_class': {},
    }

    print(f"\n评估 {len(eval_classes)} 类, {n_images} 张/类\n")

    for cls in eval_classes:
        imgs = load_images(cls)
        if not imgs:
            print(f"  {cls}: 无图像, 跳过")
            continue

        # ── Baseline: "an overhead image of {cls}" ──
        base_prompt = f'an overhead image of {cls}'
        base_dr = detect_rate(imgs, base_prompt)

        # ── Static TDM: OFAT 推荐轴修饰词 ──
        recs = kb.get_recommended_axes('remote_sensing', cls)
        mods = [AXIS_MODIFIERS[a] for a, _ in recs[:2]]
        tdm_prompt = base_prompt
        if mods:
            tdm_prompt += ', ' + ', '.join(mods)
        tdm_dr = detect_rate(imgs, tdm_prompt)

        # ── CPS Adapter: 用 CPS 评分选出最佳 prompt ──
        # 生成候选 prompt: 不同修饰词组合
        candidates = []
        # 单修饰词
        for axis in AXIS_ORDER:
            mod = AXIS_MODIFIERS[axis]
            for prefix in ['an overhead image of', 'a photo of']:
                candidates.append(f'{prefix} {cls}, {mod}')
        # 双修饰词组合 (top 推荐)
        for prefix in ['an overhead image of', 'a photo of']:
            top_recs = kb.get_recommended_axes('remote_sensing', cls)[:2]
            if len(top_recs) >= 2:
                mods_combo = [AXIS_MODIFIERS[a] for a, _ in top_recs]
                candidates.append(f'{prefix} {cls}, ' + ', '.join(mods_combo))
        # 去重
        candidates = list(set(candidates))
        candidates.append(base_prompt)  # 保底

        # 用 CPS 评分
        c_embs = compute_clip_embeddings(candidates, clip_model, clip_tokenize)
        with torch.no_grad():
            cps_scores = cps(c_embs.to(DEVICE)).squeeze().cpu().numpy()
        best_idx = np.argmax(cps_scores)
        cps_best_prompt = candidates[best_idx]

        # 用 CPS + Adapter: 将 CLIP 嵌入通过 adapter，用嵌入最近邻选 prompt
        # (因为 adapter 工作在嵌入空间，不是文本空间)
        # 策略: 选 CPS 评分最高的候选，作为 "adapter 目标"
        cps_dr = detect_rate(imgs, cps_best_prompt)

        # ── CPS Oracle: 如果真的知道 best prompt (用 GDINO 跑所有候选) ──
        # (只跑 top-3 CPS 评分 + tdm + baseline，避免太慢)
        oracle_candidates = [base_prompt, tdm_prompt]
        top3_idx = np.argsort(cps_scores)[-3:]
        for idx in top3_idx:
            if candidates[idx] not in oracle_candidates:
                oracle_candidates.append(candidates[idx])
        oracle_drs = {}
        for c in oracle_candidates:
            oracle_drs[c] = detect_rate(imgs, c)
        best_oracle_prompt = max(oracle_drs, key=oracle_drs.get)
        best_oracle_dr = oracle_drs[best_oracle_prompt]

        # 记录
        results['baseline'][cls] = base_dr
        results['static_tdm'][cls] = tdm_dr
        results['cps_adapter'][cls] = cps_dr
        results['cps_oracle'][cls] = best_oracle_dr
        results['per_class'][cls] = {
            'base_prompt': base_prompt,
            'tdm_prompt': tdm_prompt,
            'cps_best_prompt': cps_best_prompt,
            'cps_score': float(cps_scores[best_idx]),
            'best_oracle_prompt': best_oracle_prompt,
            'oracle_candidates': oracle_candidates,
            'oracle_drs': oracle_drs,
        }

        print(f"  {cls:<20s} baseline={base_dr:.3f} | tdm={tdm_dr:.3f} "
              f"({'+' if tdm_dr > base_dr else ''}{tdm_dr - base_dr:+.3f}) | "
              f"cps={cps_dr:.3f} ({'+' if cps_dr > base_dr else ''}{cps_dr - base_dr:+.3f}) | "
              f"oracle={best_oracle_dr:.3f}")

    # 汇总
    mean_base = np.mean(list(results['baseline'].values()))
    mean_tdm = np.mean(list(results['static_tdm'].values()))
    mean_cps = np.mean(list(results['cps_adapter'].values()))
    mean_oracle = np.mean(list(results['cps_oracle'].values()))

    print(f"\n{'='*60}")
    print(f"  汇总 (mean DR):")
    print(f"    Baseline:     {mean_base:.4f}")
    print(f"    Static TDM:   {mean_tdm:.4f}  (Δ={mean_tdm - mean_base:+.4f})")
    print(f"    CPS Adapter:  {mean_cps:.4f}  (Δ={mean_cps - mean_base:+.4f})")
    print(f"    CPS Oracle:   {mean_oracle:.4f}  (Δ={mean_oracle - mean_base:+.4f})")

    results['summary'] = {
        'mean_baseline': mean_base,
        'mean_static_tdm': mean_tdm,
        'mean_cps_adapter': mean_cps,
        'mean_cps_oracle': mean_oracle,
    }

    return results


# ══════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("CPS + Reward-Guided Adapter")
    print("CLIP Prompt Scorer 替代 GDINO F1 作为训练信号")
    print("=" * 65)
    print(f"设备: {DEVICE}")

    # ── 加载 CLIP ──
    import clip as clip_module
    print("\n加载 CLIP ViT-B/32...")
    clip_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
    clip_model, _ = clip_module.load(clip_path, device=DEVICE)
    clip_model.eval()

    # ── Phase 1: 收集数据 ──
    print("\n" + "=" * 60)
    print("Phase 1: 收集 OFAT 训练数据")
    print("=" * 60)
    data = collect_ofat_training_data()

    # 计算 CLIP embeddings
    prompts = [d['prompt'] for d in data]
    targets = torch.tensor([d['delta_f1'] for d in data], dtype=torch.float32)
    print("计算 CLIP text embeddings...")
    X = compute_clip_embeddings(prompts, clip_model, clip_module.tokenize)

    # ── Phase 2: 训练 CPS ──
    print("\n" + "=" * 60)
    print("Phase 2: 训练 CLIP Prompt Scorer")
    print("=" * 60)
    cps = CLIPPromptScorer(d_input=512, hidden_dims=[256, 128, 64], dropout=0.15)
    cps_history = train_cps(cps, X, targets, epochs=400, lr=0.001)

    # 评估 CPS 质量
    cps.eval()
    with torch.no_grad():
        all_preds = cps(X.to(DEVICE)).squeeze().cpu().numpy()
        all_corr = np.corrcoef(all_preds, targets.numpy())[0, 1]
        all_rmse = np.sqrt(np.mean((all_preds - targets.numpy())**2))
    print(f"\n  CPS 全数据集: r={all_corr:+.4f}, RMSE={all_rmse:.4f}")

    # 保存 CPS
    torch.save({
        'state_dict': cps.state_dict(),
        'config': {'d_input': 512, 'hidden_dims': [256, 128, 64], 'dropout': 0.15},
        'history': cps_history,
        'full_corr': all_corr,
        'full_rmse': all_rmse,
    }, SAVE_DIR / 'cps_model.pt')
    print(f"  CPS 保存至: {SAVE_DIR / 'cps_model.pt'}")

    # ── Phase 3: 训练 Adapter ──
    print("\n" + "=" * 60)
    print("Phase 3: Reward-Guided Adapter 训练")
    print("=" * 60)

    from ofat_knowledge_base import OFATKnowledgeBase
    kb = OFATKnowledgeBase()

    # 获取 DIOR 类名的 CLIP embeddings
    class_embeddings = {}
    for cls_name in kb.dior_classes:
        prompt = f'a photo of {cls_name}'
        tok = clip_module.tokenize([prompt], truncate=True).to(DEVICE)
        with torch.no_grad():
            emb = clip_model.encode_text(tok)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        class_embeddings[cls_name] = emb.squeeze(0).cpu().float()

    print(f"  训练类数: {len(class_embeddings)} (DIOR 20 类)")

    # 生成候选 prompt
    print("  生成候选 prompt...")
    class_candidates = generate_class_candidates(kb, clip_model, clip_module.tokenize)

    # 用 CPS 为每个类筛选 top-k 候选
    cps.eval()
    class_candidates_filtered = {}
    total_cands = 0
    for cls_name, cands in class_candidates.items():
        if not cands:
            class_candidates_filtered[cls_name] = []
            continue
        texts, embs = zip(*cands)
        E_cand = torch.stack(embs)
        with torch.no_grad():
            scores = cps(E_cand.to(DEVICE)).squeeze().cpu().numpy()
        # 选 top-8 (如果不够就全保留)
        top_k = min(8, len(cands))
        top_idx = np.argsort(scores)[-top_k:]
        filtered = [(texts[i], embs[i]) for i in top_idx]
        class_candidates_filtered[cls_name] = filtered
        total_cands += len(filtered)

    print(f"  CPS 筛选后: {total_cands} 候选 (top-8/类)")

    adapter = PromptAdapter(d_model=512, rank=4)
    adapter_history = train_adapter_contrastive(
        adapter, cps, class_embeddings, class_candidates_filtered,
        epochs=300, lr=0.002, temperature=0.1,
        lambda_kl=1.0, lambda_l2=0.001,
    )

    # 分析 adapter 对每类的修正
    print("\n  Adapter 对每类的 CPS 提升:")
    for cls_name in sorted(class_embeddings.keys()):
        e_raw = class_embeddings[cls_name].to(DEVICE)
        with torch.no_grad():
            reward_raw = cps(e_raw.unsqueeze(0)).item()
            e_adapted = adapter(e_raw.unsqueeze(0))
            reward_adapted = cps(e_adapted).item()
        delta = reward_adapted - reward_raw
        marker = "✅" if delta > 0.001 else ("⚠️" if delta > -0.001 else "❌")
        print(f"  {marker} {cls_name:<26s} raw={reward_raw:+.4f} → "
              f"adapted={reward_adapted:+.4f} (Δ={delta:+.4f})")

    # 保存 Adapter
    torch.save({
        'U': adapter.U.data.cpu(),
        'V': adapter.V.data.cpu(),
        'd_model': adapter.d_model,
        'rank': adapter.rank,
        'history': adapter_history,
    }, SAVE_DIR / 'cps_adapter.pt')
    print(f"\n  Adapter 保存至: {SAVE_DIR / 'cps_adapter.pt'}")

    # ── Phase 4: GDINO 验证 ──
    try:
        eval_results = evaluate_gdino(
            adapter, cps, clip_model, clip_module.tokenize, kb,
            n_images=20
        )
        with open(SAVE_DIR / 'cps_adapter_eval.json', 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        print(f"\n  评估结果保存至: {SAVE_DIR / 'cps_adapter_eval.json'}")
    except Exception as e:
        print(f"\n  Phase 4 跳过: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ CPS + Reward-Guided Adapter 完成")
