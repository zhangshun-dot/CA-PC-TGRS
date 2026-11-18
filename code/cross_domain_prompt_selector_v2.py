# -*- coding: utf-8 -*-
"""CDPS v2 — 优化域条件调节机制

v1 → v2 改进:
  1. Learnable Domain Embeddings: 域原型从固定向量→可学习参数 (训练优化)
  2. FiLM 调制替代简单拼接: 域信息通过 γ,β 调制类编码器各层
  3. 新域 Few-Shot 适应: 训练时域向量加噪→selector 学会从稀疏信号泛化
  4. Per-Image 可选: 推理时可用单张图像 CLIP 特征替代域原型

架构 (v2):
  ┌──────────────────────────────────────────────────┐
  │          FiLMPromptSelector                       │
  │                                                   │
  │  e_cls ──→ [FC₁] ──→ [FC₂] ──→ head ──→ scores  │
  │               │         │                         │
  │          γ₁,β₁     γ₂,β₂   ← FiLM(d_domain)      │
  │               ↑         ↑                         │
  │          ┌────┴─────────┴────┐                    │
  │          │  Domain Encoder    │                    │
  │          │  (v_domain → γ,β) │                    │
  │          └────────────────────┘                    │
  └──────────────────────────────────────────────────┘

Usage:
    python cross_domain_prompt_selector_v2.py
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json, argparse, glob, copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
SAVE_DIR.mkdir(exist_ok=True)

from token_projection import AXIS_MODIFIERS, AXIS_ORDER, UNIVERSALITY_SCORES
from ofat_knowledge_base import OFATKnowledgeBase

N_AXES = len(AXIS_ORDER)


# ══════════════════════════════════════════════════════════════════
# 1. FiLM 调制层
# ══════════════════════════════════════════════════════════════════

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: h_out = γ ⊙ h + β

    域向量 → 两个小 MLP → (γ, β) → 调制特征
    """

    def __init__(self, in_dim, cond_dim, hidden=64):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim)
        )
        # 初始化为接近恒等: γ≈1, β≈0
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.zeros_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, x, cond):
        """x: (B, D), cond: (B, C)"""
        gamma = self.gamma_net(cond) + 1.0  # 初始 γ≈1
        beta = self.beta_net(cond)          # 初始 β≈0
        return gamma * x + beta


class FiLMPromptSelector(nn.Module):
    """FiLM 调制的跨域 Prompt 选择器。

    域条件通过 FiLM 层注入类编码器，而非简单拼接。
    域嵌入可学习 (训练时优化)。

    输入:
      - e_cls: CLIP 类名嵌入 (512-d)
      - domain_id or v_domain: 域索引 (int) 或域向量 (512-d)

    输出:
      - axis_scores: (n_axes,) — 每个轴的 δ 预测
    """

    def __init__(self, d_model=512, n_axes=6, hidden=256, n_domains=2,
                 learnable_domain_emb=True, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_axes = n_axes
        self.learnable_domain_emb = learnable_domain_emb

        # ── 可学习域嵌入 ──
        if learnable_domain_emb:
            self.domain_embeddings = nn.Parameter(
                torch.randn(n_domains, d_model) * 0.1
            )
            self.domain_to_idx = {}  # 训练时填充
        else:
            self.domain_embeddings = None

        # ── 域编码器: v_domain → FiLM 调制参数 ──
        cond_dim = d_model
        self.film1 = FiLMLayer(hidden, cond_dim, hidden=128)
        self.film2 = FiLMLayer(hidden, cond_dim, hidden=128)

        # ── 类编码器 (被 FiLM 调制) ──
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

        # ── 输出头 ──
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_axes),
        )

    def get_domain_vector(self, domain_id_or_vec):
        """获取域条件向量。

        Args:
            domain_id_or_vec: int (域索引), str (域标签), 或 Tensor (512,)

        Returns:
            Tensor (B, 512)
        """
        if isinstance(domain_id_or_vec, torch.Tensor):
            # 直接给向量
            if domain_id_or_vec.dim() == 1:
                domain_id_or_vec = domain_id_or_vec.unsqueeze(0)
            return domain_id_or_vec.float()

        if isinstance(domain_id_or_vec, (int, np.integer)):
            idx = domain_id_or_vec
        elif isinstance(domain_id_or_vec, str):
            idx = self.domain_to_idx.get(domain_id_or_vec, 0)
        else:
            idx = 0

        if self.domain_embeddings is not None:
            return self.domain_embeddings[idx].unsqueeze(0)
        else:
            return torch.zeros(1, self.d_model, device=next(self.parameters()).device)

    def forward(self, e_cls, domain_id_or_vec):
        """前向传播。

        Args:
            e_cls: (B, 512) CLIP 类名嵌入
            domain_id_or_vec: 域标识或向量

        Returns:
            (B, n_axes) per-axis δ 预测
        """
        v_dom = self.get_domain_vector(domain_id_or_vec)

        # 扩展到 batch
        if v_dom.shape[0] == 1 and e_cls.shape[0] > 1:
            v_dom = v_dom.expand(e_cls.shape[0], -1)

        x = e_cls.float()

        # FiLM 调制
        x = F.relu(self.fc1(x))
        x = self.film1(x, v_dom)          # 域调制 1
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.film2(x, v_dom)          # 域调制 2
        x = self.dropout(x)

        return self.head(x)

    def predict_axes(self, cls_name, domain_id_or_vec, clip_model,
                     top_k=2, threshold=0.03):
        """推理: 预测类的最优修饰轴。"""
        import clip as clip_module

        prompt = f'a photo of {cls_name}'
        tok = clip_module.tokenize([prompt], truncate=True).to(DEVICE)
        with torch.no_grad():
            e = clip_model.encode_text(tok).float()
            e = e / e.norm(dim=-1, keepdim=True)

        scores = self.forward(e, domain_id_or_vec).squeeze(0)

        results = []
        for i in range(self.n_axes):
            s = scores[i].item()
            if s > threshold:
                results.append((AXIS_ORDER[i], s))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def build_prompt(self, cls_name, domain_id_or_vec, clip_model,
                     domain_prefixes=None, top_k=2, threshold=0.03):
        """构建完整 TDM prompt。"""
        if domain_prefixes is None:
            domain_prefixes = {
                0: 'an overhead image of',
                1: 'a photo of',
                'remote_sensing': 'an overhead image of',
                'natural_scene': 'a photo of',
            }

        # 找出域标签对应的前缀
        if isinstance(domain_id_or_vec, (int, np.integer)):
            domain = 'remote_sensing' if domain_id_or_vec == 0 else 'natural_scene'
        elif isinstance(domain_id_or_vec, str):
            domain = domain_id_or_vec
        else:
            domain = 'remote_sensing'

        prefix = domain_prefixes.get(domain, 'an image of')
        axes = self.predict_axes(cls_name, domain_id_or_vec, clip_model,
                                 top_k=top_k, threshold=threshold)

        prompt = f'{prefix} {cls_name}'
        if axes:
            modifiers = [AXIS_MODIFIERS[ax] for ax, _ in axes]
            prompt += ', ' + ', '.join(modifiers)
        return prompt


# ══════════════════════════════════════════════════════════════════
# 2. 训练数据准备 (含域向量增广)
# ══════════════════════════════════════════════════════════════════

def prepare_training_data_v2(kb, clip_model, preprocess):
    """构建训练集 + 域向量增广。

    增广策略:
      - 每个域的原型向量加高斯噪声 σ=0.05 → 模拟新域变体
      - 随机 dropout 域向量维度 → 模拟稀疏观测

    Returns:
        X_cls, X_dom, Y, domain_ids, domain_prototypes, metadata
    """
    import clip as clip_module
    from cross_domain_prompt_selector import build_domain_prototypes

    # 域原型
    prototypes = build_domain_prototypes(clip_model, preprocess)
    domain_list = ['remote_sensing', 'natural_scene']
    domain_to_idx = {d: i for i, d in enumerate(domain_list)}

    X_cls_list, X_dom_list, Y_list, dom_id_list = [], [], [], []
    metadata = []

    for domain in domain_list:
        if domain not in prototypes:
            continue
        v_dom_base = prototypes[domain]
        dom_idx = domain_to_idx[domain]

        classes = (kb.dior_classes if domain == 'remote_sensing'
                   else sorted(set(k[0] for k in kb.coco_deltas.keys())))

        for cls_name in classes:
            prompt = f'a photo of {cls_name}'
            tok = clip_module.tokenize([prompt], truncate=True).to(DEVICE)
            with torch.no_grad():
                e = clip_model.encode_text(tok).float()
                e = e / e.norm(dim=-1, keepdim=True)

            # OFAT δ
            deltas = np.zeros(N_AXES)
            valid = False
            for i, axis in enumerate(AXIS_ORDER):
                d = kb.get_delta(domain, cls_name, axis)
                if d is not None:
                    deltas[i] = d
                    valid = True
            if not valid:
                continue

            # ── 基础样本 ──
            X_cls_list.append(e.squeeze(0).cpu().cpu())
            X_dom_list.append(v_dom_base.cpu().clone())
            Y_list.append(torch.tensor(deltas, dtype=torch.float32))
            dom_id_list.append(dom_idx)
            metadata.append({'domain': domain, 'class': cls_name, 'aug': False})

            # ── 增广样本: 域向量加噪 (3 个变体) ──
            for aug_i in range(3):
                noise = torch.randn(512) * 0.05
                v_dom_aug = v_dom_base.cpu() + noise
                v_dom_aug = v_dom_aug / v_dom_aug.norm()

                X_cls_list.append(e.squeeze(0).cpu())
                X_dom_list.append(v_dom_aug)
                Y_list.append(torch.tensor(deltas, dtype=torch.float32))
                dom_id_list.append(dom_idx)
                metadata.append({'domain': domain, 'class': cls_name, 'aug': True})

            # ── 增广样本: 域向量维度 dropout (1 个变体) ──
            mask = (torch.rand(512) > 0.1).float()
            v_dom_drop = v_dom_base.cpu() * mask
            v_dom_drop = v_dom_drop / (v_dom_drop.norm() + 1e-8)

            X_cls_list.append(e.squeeze(0).cpu())
            X_dom_list.append(v_dom_drop)
            Y_list.append(torch.tensor(deltas, dtype=torch.float32))
            dom_id_list.append(dom_idx)
            metadata.append({'domain': domain, 'class': cls_name, 'aug': True})

    X_cls = torch.stack(X_cls_list)
    X_dom = torch.stack(X_dom_list)
    Y = torch.stack(Y_list)
    domain_ids = torch.tensor(dom_id_list, dtype=torch.long)

    print(f"  训练样本: {len(X_cls)} (含增广)")
    for dom in domain_list:
        n_base = sum(1 for m in metadata if m['domain'] == dom and not m['aug'])
        n_aug = sum(1 for m in metadata if m['domain'] == dom and m['aug'])
        print(f"    {dom}: 基础={n_base}, 增广={n_aug}")

    return X_cls, X_dom, Y, domain_ids, prototypes, metadata, domain_to_idx


# ══════════════════════════════════════════════════════════════════
# 3. 训练器 (v2)
# ══════════════════════════════════════════════════════════════════

def train_selector_v2(selector, X_cls, X_dom, Y, domain_ids,
                      domain_prototypes, domain_to_idx,
                      epochs=500, lr=0.001, batch_size=64,
                      aug_noise=0.03, verbose=True):
    """训练 FiLM Prompt Selector。

    改进:
      - 在线域向量加噪 (每个 epoch 不同噪声)
      - 域嵌入从 CLIP 原型初始化
      - Warmup + Cosine LR
    """
    selector.train()
    selector.to(DEVICE)

    # 初始化可学习域嵌入
    if selector.learnable_domain_emb and selector.domain_embeddings is not None:
        with torch.no_grad():
            for dom, idx in domain_to_idx.items():
                if dom in domain_prototypes:
                    selector.domain_embeddings[idx] = domain_prototypes[dom].clone()
        selector.domain_to_idx = domain_to_idx
        print(f"  域嵌入初始化: {domain_to_idx}")

    X_cls = X_cls.to(DEVICE)
    X_dom_base = X_dom.to(DEVICE)  # 基础域向量 (不增广)
    Y = Y.to(DEVICE)

    optimizer = torch.optim.AdamW(selector.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs,
        pct_start=0.1, anneal_strategy='cos'
    )

    n = len(X_cls)
    history = defaultdict(list)

    print(f"  训练: epochs={epochs}, lr={lr}, batch={batch_size}, N={n}")

    for epoch in range(epochs):
        # 在线增广: 域向量加噪
        if aug_noise > 0:
            noise = torch.randn_like(X_dom_base) * aug_noise
            X_dom_aug = X_dom_base + noise
            X_dom_aug = X_dom_aug / X_dom_aug.norm(dim=-1, keepdim=True)
        else:
            X_dom_aug = X_dom_base

        perm = torch.randperm(n)
        total_loss, n_batches = 0.0, 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x_c = X_cls[idx]
            x_d = X_dom_aug[idx]
            y_true = Y[idx]

            # 用域向量 (非 learnable embedding) 做 FiLM
            y_pred = selector(x_c, x_d)

            loss = ((y_pred - y_true) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if verbose and epoch % 50 == 0:
            with torch.no_grad():
                y_all = selector(X_cls, X_dom_base)
                mse_all = ((y_all - Y) ** 2).mean().item()
                sign_match = (torch.sign(y_all) == torch.sign(Y)).float()
                sign_acc = sign_match.mean().item()
                nonzero = Y.abs() > 0.01
                sign_acc_nz = sign_match[nonzero].mean().item() if nonzero.any() else 0

                # 分域评估
                acc_per_dom = {}
                for d_name, d_idx in domain_to_idx.items():
                    d_mask = (domain_ids == d_idx)
                    if d_mask.any():
                        sm = sign_match[d_mask].mean().item()
                        acc_per_dom[d_name] = sm

            print(f"  E{epoch:3d} | L={avg_loss:.6f} MSE={mse_all:.6f} "
                  f"sign={sign_acc:.4f}(nz={sign_acc_nz:.4f}) "
                  f"RS={acc_per_dom.get('remote_sensing',0):.3f} "
                  f"NS={acc_per_dom.get('natural_scene',0):.3f} "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

        history['loss'].append(avg_loss)
        history['mse'].append(mse_all)
        history['sign_acc'].append(sign_acc)

    selector.eval()
    return history


# ══════════════════════════════════════════════════════════════════
# 4. 评估 (v2)
# ══════════════════════════════════════════════════════════════════

def evaluate_selector_v2(selector, X_cls, X_dom, Y, domain_ids,
                          domain_prototypes, domain_to_idx, metadata,
                          kb, clip_model):
    """评估 + 跨域泛化 + Few-Shot 适应性测试。"""
    selector.eval()
    X_cls = X_cls.to(DEVICE)
    X_dom = X_dom.to(DEVICE)
    Y = Y.to(DEVICE)

    print("\n" + "=" * 60)
    print("CDPS v2 — 评估")
    print("=" * 60)

    with torch.no_grad():
        Y_pred = selector(X_cls, X_dom)

    # ── 分域 Prompt 匹配率 ──
    print(f"\n  Prompt 匹配率 (vs CD-TDM OFAT):")
    domain_prefixes = {0: 'an overhead image of', 1: 'a photo of',
                       'remote_sensing': 'an overhead image of',
                       'natural_scene': 'a photo of'}

    test_cases = [
        (0, 'airplane'), (0, 'ship'), (0, 'vehicle'),
        (0, 'tennis court'), (0, 'stadium'), (0, 'baseball field'),
        (1, 'airplane'), (1, 'boat'), (1, 'car'),
        (1, 'dog'), (1, 'person'), (1, 'bird'),
    ]

    n_match, n_total = 0, 0
    for dom_id, cls in test_cases:
        dom_name = 'remote_sensing' if dom_id == 0 else 'natural_scene'
        recs = kb.get_recommended_axes(dom_name, cls)
        prefix = domain_prefixes[dom_id]
        tdm_p = f'{prefix} {cls}'
        if recs:
            tdm_p += ', ' + ', '.join(AXIS_MODIFIERS[a] for a, _ in recs[:2])

        sel_p = selector.build_prompt(cls, dom_id, clip_model,
                                       domain_prefixes=domain_prefixes,
                                       top_k=2, threshold=0.03)
        match = sel_p == tdm_p
        n_match += match
        n_total += 1
        if not match:
            print(f"  ❌ {dom_name}/{cls}: CD-TDM='{tdm_p}' vs CDPS='{sel_p}'")

    print(f"  匹配率: {n_match}/{n_total} = {n_match/n_total*100:.1f}%")

    # ── Few-Shot 新域适应 ──
    print(f"\n  Few-Shot 新域适应测试:")
    print(f"    场景: 用 natural_scene prototype 的 k-shot 扰动模拟新域,"
          f" 测 prompt 选择稳定性")

    for k in [1, 3, 5, 10]:
        n_stable = 0
        for cls in ['airplane', 'boat', 'car']:
            base_prompt = selector.build_prompt(cls, 1, clip_model,
                                                 domain_prefixes=domain_prefixes,
                                                 top_k=2, threshold=0.03)
            stable = True
            for _ in range(k):
                v_noisy = X_dom[domain_ids == 1][0] + 0.08 * torch.randn(512, device=DEVICE)
                v_noisy = v_noisy / v_noisy.norm()
                new_prompt = selector.build_prompt(cls, v_noisy, clip_model,
                                                    domain_prefixes=domain_prefixes,
                                                    top_k=2, threshold=0.03)
                if new_prompt != base_prompt:
                    stable = False
                    break
            if stable:
                n_stable += 1
        print(f"    k={k:2d}: {n_stable}/3 类保持稳定 (噪声 σ=0.08)")

    # ── Hold-one-domain-out ──
    print(f"\n  跨域泛化 (Hold-One-Domain-Out):")
    for held_out_idx, held_out_name in [(0, 'remote_sensing'), (1, 'natural_scene')]:
        train_mask = (domain_ids != held_out_idx)
        test_mask = (domain_ids == held_out_idx)

        # 快训
        sel_held = FiLMPromptSelector(d_model=512, n_axes=6, n_domains=2,
                                       learnable_domain_emb=False)
        sel_held.to(DEVICE)
        opt = torch.optim.AdamW(sel_held.parameters(), lr=0.001)

        train_xc = X_cls[train_mask]
        train_xd = X_dom[train_mask]
        train_y = Y[train_mask]

        for _ in range(150):
            y_p = sel_held(train_xc, train_xd)
            loss = ((y_p - train_y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        sel_held.eval()
        with torch.no_grad():
            y_held = sel_held(X_cls[test_mask], X_dom[test_mask])
            mse = ((y_held - Y[test_mask]) ** 2).mean().item()
            sign_acc = (torch.sign(y_held) == torch.sign(Y[test_mask])).float().mean().item()

        print(f"    训练≠{held_out_name}, 测试={held_out_name}: "
              f"MSE={mse:.4f}, sign_acc={sign_acc:.4f}")

    return Y_pred


# ══════════════════════════════════════════════════════════════════
# 5. 可视化: FiLM 调制强度分析
# ══════════════════════════════════════════════════════════════════

def analyze_film_modulation(selector, X_dom, domain_to_idx):
    """分析 FiLM 层的域调制强度——量化域信息如何改变特征。"""
    print(f"\n  FiLM 调制强度分析:")

    for dom_name, dom_idx in domain_to_idx.items():
        v_dom = X_dom[0:1].to(DEVICE)  # 取域向量

        # 计算 γ 偏移
        g1 = selector.film1.gamma_net(v_dom).squeeze(0)
        b1 = selector.film1.beta_net(v_dom).squeeze(0)
        g2 = selector.film2.gamma_net(v_dom).squeeze(0)
        b2 = selector.film2.beta_net(v_dom).squeeze(0)

        print(f"    {dom_name}:")
        print(f"      FiLM1: γ∈[{g1.min().item():.3f}, {g1.max().item():.3f}] "
              f"(mean={g1.mean().item():.3f}), β∈[{b1.min().item():.3f}, {b1.max().item():.3f}]")
        print(f"      FiLM2: γ∈[{g2.min().item():.3f}, {g2.max().item():.3f}] "
              f"(mean={g2.mean().item():.3f}), β∈[{b2.min().item():.3f}, {b2.max().item():.3f}]")


# ══════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--aug_noise', type=float, default=0.03)
    args = parser.parse_args()

    print("=" * 60)
    print("CDPS v2 — FiLM 调制跨域 Prompt 选择器")
    print("=" * 60)
    print(f"设备: {DEVICE}")

    # ── 加载 ──
    print("\n[1] 加载 CLIP + OFAT KB...")
    import clip as clip_module
    clip_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
    clip_model, preprocess = clip_module.load(clip_path, device=DEVICE)
    clip_model.eval()
    kb = OFATKnowledgeBase()

    # ── 数据 ──
    print("\n[2] 准备训练数据 (含域向量增广)...")
    (X_cls, X_dom, Y, domain_ids,
     domain_prototypes, metadata, domain_to_idx) = prepare_training_data_v2(
        kb, clip_model, preprocess
    )

    n_domains = len(domain_to_idx)

    # ── 模型 ──
    print(f"\n[3] 构建 FiLMPromptSelector (n_domains={n_domains})...")
    selector = FiLMPromptSelector(
        d_model=512, n_axes=6, hidden=args.hidden, n_domains=n_domains,
        learnable_domain_emb=True, dropout=0.1
    )
    n_params = sum(p.numel() for p in selector.parameters())
    print(f"  参数量: {n_params}")

    # ── 训练 ──
    print(f"\n[4] 训练...")
    history = train_selector_v2(
        selector, X_cls, X_dom, Y, domain_ids,
        domain_prototypes, domain_to_idx,
        epochs=args.epochs, lr=args.lr, aug_noise=args.aug_noise, verbose=True
    )

    # ── 保存 ──
    save_dict = {
        'selector_state': selector.state_dict(),
        'domain_to_idx': domain_to_idx,
        'domain_prototypes': {k: v.cpu() for k, v in domain_prototypes.items()},
        'history': {k: [float(v) for v in vs] for k, vs in history.items()},
    }
    torch.save(save_dict, SAVE_DIR / 'cdps_v2_weights.pt')

    # ── 评估 ──
    print(f"\n[5] 评估...")
    Y_pred = evaluate_selector_v2(
        selector, X_cls, X_dom, Y, domain_ids,
        domain_prototypes, domain_to_idx, metadata, kb, clip_model
    )

    # ── FiLM 分析 ──
    analyze_film_modulation(selector, X_dom, domain_to_idx)

    # ── 降噪健壮性 ──
    print(f"\n  降噪健壮性:")
    for sigma in [0.02, 0.05, 0.10, 0.20]:
        noise = torch.randn_like(X_dom[:20]) * sigma
        X_noisy = (X_dom[:20] + noise)
        X_noisy = X_noisy / X_noisy.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            y_clean = selector(X_cls[:20].to(DEVICE), X_dom[:20].to(DEVICE))
            y_noisy = selector(X_cls[:20].to(DEVICE), X_noisy.to(DEVICE))
            cos_sim = F.cosine_similarity(y_clean.flatten().unsqueeze(0),
                                          y_noisy.flatten().unsqueeze(0)).item()
            n_diff = (torch.sign(y_clean) != torch.sign(y_noisy)).sum().item()
        print(f"    σ={sigma:.2f}: cos={cos_sim:.4f}, 符号翻转={n_diff}/{20*6}")

    print(f"\n  权重: {SAVE_DIR / 'cdps_v2_weights.pt'}")
    print("✅ CDPS v2 完成")


if __name__ == '__main__':
    main()
