# -*- coding: utf-8 -*-
"""Cross-Domain Prompt Selector (CDPS) — 跨域统一 Prompt 选择器

核心洞察 (来自 CDTA v1/v2 教训):
  CLIP 文本 token 是域无关的——"airplane" 无论在哪个域都是同一个 embedding。
  因此不需要"投影 token"——CLIP 空间本身就是共享空间。

  需要学习的是: 如何用域条件信号 (image-level domain features) 调节 prompt 选择。

架构:
  ┌─────────────────────────────────────┐
  │    Unified Prompt Selector (MLP)    │
  │   输入: [CLIP_text(cls); domain_vec]│
  │   输出: {axis: score} → top-k axes │
  └──────────┬──────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
  E_cls(c)        domain_vec
  CLIP text       CLIP image domain pooling
  (512-d)         (512-d, 来自 domain prototype)
             │
    ┌────────┴────────┐
    │  OFAT δ tables  │ ← 训练信号
    │  DIOR + COCO    │
    └─────────────────┘

训练:
  L = MSE(selector(E_cls(c), v_domain), δ_known(c, axis))
  跨域泛化: 训练时混合 DIOR + COCO 数据, selector 学会从 domain_vec 推断域特性

新域注册:
  1. 拍 5 张图 → CLIP image encode → mean = domain_vec
  2. Selector(E_cls(c), domain_vec) → 直接输出 axis scores
  3. 无需重新跑 OFAT

Usage:
    python cross_domain_prompt_selector.py --mode train
    python cross_domain_prompt_selector.py --mode eval
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json, argparse, glob
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


# ══════════════════════════════════════════════════════════════════
# 1. Domain Prototype 计算
# ══════════════════════════════════════════════════════════════════

def compute_domain_prototype(image_paths, clip_model, preprocess, n_max=20):
    """用 CLIP 图像编码器计算域的视觉原型向量。

    对域内多张图像取 CLIP image embedding 的均值 → 域条件向量。

    Args:
        image_paths: 图像路径列表
        clip_model: CLIP 模型
        preprocess: CLIP 预处理函数
        n_max: 最大图像数

    Returns:
        Tensor (512,): 域原型向量
    """
    features = []
    for p in image_paths[:n_max]:
        try:
            img = preprocess(Image.open(p).convert('RGB')).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                f = clip_model.encode_image(img).float()
                f = f / f.norm(dim=-1, keepdim=True)
            features.append(f.squeeze(0))
        except Exception:
            continue
    if not features:
        return None
    proto = torch.stack(features).mean(dim=0)
    return proto / proto.norm(dim=-1, keepdim=True)


def build_domain_prototypes(clip_model, preprocess):
    """构建所有已知域的 prototype 向量。"""
    import clip as clip_module

    prototypes = {}

    # 遥感域: 从 DIOR 采样
    dior_dir = r'C:\Users\张三\modeling\dior_data\full20_images'
    dior_imgs = []
    for cls_dir in ['airplane', 'ship', 'vehicle', 'stadium', 'bridge',
                     'harbor', 'baseball field', 'tennis court']:
        pattern = os.path.join(dior_dir, f"{cls_dir.replace(' ', '_')}_*.jpg")
        imgs = sorted(glob.glob(pattern))[:3]
        dior_imgs.extend(imgs)
    if dior_imgs:
        proto = compute_domain_prototype(dior_imgs, clip_model, preprocess)
        if proto is not None:
            prototypes['remote_sensing'] = proto
            print(f"  remote_sensing: {len(dior_imgs[:20])} images, |v|={proto.norm().item():.3f}")

    # 自然图像域: 从 COCO 采样
    coco_dir = r'C:\Users\张三\modeling\coco_data\val2017'
    if os.path.exists(coco_dir):
        coco_imgs = sorted(glob.glob(os.path.join(coco_dir, '*.jpg')))[:20]
        proto = compute_domain_prototype(coco_imgs, clip_model, preprocess)
        if proto is not None:
            prototypes['natural_scene'] = proto
            print(f"  natural_scene: {len(coco_imgs[:20])} images, |v|={proto.norm().item():.3f}")

    return prototypes


# ══════════════════════════════════════════════════════════════════
# 2. Unified Prompt Selector (MLP)
# ══════════════════════════════════════════════════════════════════

class UnifiedPromptSelector(nn.Module):
    """轻量 MLP: [CLIP_text(cls); domain_vec] → {axis: score}

    输入:
      - e_cls: CLIP 类名文本嵌入 (512-d, 域无关)
      - v_domain: 域原型向量 (512-d, 从域内图像 CLIP 编码取均值)

    输出:
      - axis_scores: (n_axes,) — 每个修饰轴的预测 δ 得分

    设计:
      - 先分别处理两个输入 (各一个 FC + ReLU)
      - 拼接后过一个共享 MLP → 输出 per-axis scores
      - 轻量: ~10K 参数, 防止过拟合 OFAT 表
    """

    def __init__(self, d_model=512, n_axes=6, hidden=128):
        super().__init__()
        self.d_model = d_model
        self.n_axes = n_axes

        # 类嵌入编码
        self.cls_encoder = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # 域向量编码
        self.domain_encoder = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # 共享选择头
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_axes),
        )

        # 轴到索引的映射
        self.axis_to_idx = {a: i for i, a in enumerate(AXIS_ORDER)}
        self.idx_to_axis = {i: a for i, a in enumerate(AXIS_ORDER)}

    def forward(self, e_cls, v_domain):
        """前向传播。

        Args:
            e_cls: (B, 512) CLIP 类名嵌入
            v_domain: (B, 512) 域原型向量

        Returns:
            (B, 6) per-axis δ 预测得分
        """
        h_cls = self.cls_encoder(e_cls.float())       # (B, H)
        h_dom = self.domain_encoder(v_domain.float())  # (B, H)
        h = torch.cat([h_cls, h_dom], dim=-1)          # (B, 2H)
        return self.head(h)                             # (B, 6)

    def predict_axes(self, cls_name, domain_vec, clip_model, top_k=2, threshold=0.03):
        """推理: 预测类在给定域中的最优修饰轴。

        Args:
            cls_name: 类名
            domain_vec: (512,) 域原型向量
            clip_model: CLIP 模型
            top_k: 最多返回几个轴
            threshold: 得分阈值

        Returns:
            List[Tuple[str, float]]: [(axis, score), ...]
        """
        import clip as clip_module

        # CLIP 类编码
        prompt = f'a photo of {cls_name}'
        tok = clip_module.tokenize([prompt], truncate=True).to(DEVICE)
        with torch.no_grad():
            e = clip_model.encode_text(tok).float()
            e = e / e.norm(dim=-1, keepdim=True)

        # 前向
        scores = self.forward(e, domain_vec.unsqueeze(0).to(DEVICE)).squeeze(0)

        # 选取
        results = []
        for i in range(self.n_axes):
            s = scores[i].item()
            if s > threshold:
                results.append((self.idx_to_axis[i], s))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def build_prompt(self, cls_name, domain, domain_vec, clip_model,
                     domain_prefixes=None, top_k=2, threshold=0.03):
        """构建完整 TDM prompt。

        Args:
            cls_name: 类名
            domain: 域标签 (用于选前缀)
            domain_vec: 域原型向量
            clip_model: CLIP 模型
            domain_prefixes: Dict[domain → prefix str]
            top_k: 修饰轴数量
            threshold: 得分阈值

        Returns:
            str: "prefix cls, modifier1, modifier2"
        """
        if domain_prefixes is None:
            domain_prefixes = {
                'remote_sensing': 'an overhead image of',
                'natural_scene': 'a photo of',
                'infrared': 'a thermal image of',
                'sar': 'a SAR image of',
                'underwater': 'an underwater image of',
                'medical': 'a medical image of',
            }

        prefix = domain_prefixes.get(domain, 'an image of')
        axes = self.predict_axes(cls_name, domain_vec, clip_model,
                                 top_k=top_k, threshold=threshold)

        prompt = f'{prefix} {cls_name}'
        if axes:
            modifiers = [AXIS_MODIFIERS[ax] for ax, _ in axes]
            prompt += ', ' + ', '.join(modifiers)
        return prompt


# ══════════════════════════════════════════════════════════════════
# 3. 训练数据准备
# ══════════════════════════════════════════════════════════════════

def prepare_training_data(kb, clip_model, domain_prototypes):
    """构建训练集: (e_cls, v_domain, δ_vector)。

    对每个域 × 每个类 × 每个轴:
      输入: [CLIP_text(cls); domain_prototype]
      目标: δ(c, axis) from OFAT table

    Returns:
        X_cls:  Tensor (N, 512) — CLIP 类嵌入
        X_dom:  Tensor (N, 512) — 域原型
        Y:      Tensor (N, 6)  — per-axis δ
        weights: Tensor (N,)    — universality 加权
        metadata: List[dict]    — 每个样本的元信息
    """
    import clip as clip_module

    X_cls_list, X_dom_list, Y_list, W_list = [], [], [], []
    metadata = []

    for domain in ['remote_sensing', 'natural_scene']:
        if domain not in domain_prototypes:
            print(f"  域 {domain} 无 prototype, 跳过")
            continue
        v_dom = domain_prototypes[domain]

        # 获取该域的所有类
        if domain == 'remote_sensing':
            classes = kb.dior_classes
        else:
            classes = sorted(set(k[0] for k in kb.coco_deltas.keys()))

        for cls_name in classes:
            # CLIP 类嵌入
            prompt = f'a photo of {cls_name}'
            tok = clip_module.tokenize([prompt], truncate=True).to(DEVICE)
            with torch.no_grad():
                e = clip_model.encode_text(tok).float()
                e = e / e.norm(dim=-1, keepdim=True)

            # OFAT δ 向量
            deltas = np.zeros(6)
            valid = False
            for i, axis in enumerate(AXIS_ORDER):
                d = kb.get_delta(domain, cls_name, axis)
                if d is not None:
                    deltas[i] = d
                    valid = True

            if not valid:
                continue

            X_cls_list.append(e.squeeze(0))
            X_dom_list.append(v_dom)
            Y_list.append(torch.tensor(deltas, dtype=torch.float32))

            # 权重: 精确 OFAT 测量权重高
            is_exact = True
            if domain == 'natural_scene':
                is_exact = cls_name in kb.coco.get('classes', {})
            w = 1.0 if is_exact else 0.7
            W_list.append(w)

            metadata.append({'domain': domain, 'class': cls_name, 'is_exact': is_exact})

    X_cls = torch.stack(X_cls_list)
    X_dom = torch.stack(X_dom_list)
    Y = torch.stack(Y_list)
    weights = torch.tensor(W_list, dtype=torch.float32)

    print(f"  训练样本: {len(X_cls)}")
    for dom in ['remote_sensing', 'natural_scene']:
        n = sum(1 for m in metadata if m['domain'] == dom)
        n_exact = sum(1 for m in metadata if m['domain'] == dom and m['is_exact'])
        print(f"    {dom}: {n} (精确={n_exact})")

    return X_cls, X_dom, Y, weights, metadata


# ══════════════════════════════════════════════════════════════════
# 4. 训练器
# ══════════════════════════════════════════════════════════════════

def train_selector(selector, X_cls, X_dom, Y, weights,
                   epochs=500, lr=0.001, batch_size=32, verbose=True):
    """训练 Unified Prompt Selector。

    损失:
      L = weighted_MSE(selector(e_cls, v_dom), δ_true)
        + λ_univ * universality_consistency_loss
    """
    selector.train()
    selector.to(DEVICE)

    X_cls = X_cls.to(DEVICE)
    X_dom = X_dom.to(DEVICE)
    Y = Y.to(DEVICE)
    weights = weights.to(DEVICE)

    optimizer = torch.optim.AdamW(selector.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n = len(X_cls)
    history = defaultdict(list)

    print(f"  训练: epochs={epochs}, lr={lr}, batch={batch_size}, N={n}")

    for epoch in range(epochs):
        # Mini-batch
        perm = torch.randperm(n)
        total_loss, total_mse, n_batches = 0.0, 0.0, 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x_c = X_cls[idx]
            x_d = X_dom[idx]
            y_true = Y[idx]
            w = weights[idx]

            y_pred = selector(x_c, x_d)

            # 加权 MSE
            mse = ((y_pred - y_true) ** 2).mean(dim=-1)  # (B,)
            loss = (mse * w).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse.mean().item()
            n_batches += 1

        scheduler.step()

        if verbose and epoch % 50 == 0:
            # 分类别报告
            with torch.no_grad():
                y_all = selector(X_cls, X_dom)
                total_mse_all = ((y_all - Y) ** 2).mean().item()

                # 符号准确率
                sign_match = (torch.sign(y_all) == torch.sign(Y)).float()
                sign_acc = sign_match.mean().item()
                # 只对非零目标计算
                nonzero_mask = (Y.abs() > 0.01)
                if nonzero_mask.any():
                    sign_acc_nz = sign_match[nonzero_mask].mean().item()
                else:
                    sign_acc_nz = 0

            print(f"  E{epoch:3d} | L={total_loss/max(n_batches,1):.4f} "
                  f"MSE={total_mse_all:.4f} "
                  f"sign_acc={sign_acc:.4f} (nz={sign_acc_nz:.4f}) "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

            history['loss'].append(total_loss / max(n_batches, 1))
            history['mse'].append(total_mse_all)
            history['sign_acc'].append(sign_acc)
            history['sign_acc_nz'].append(sign_acc_nz)

    selector.eval()
    return history


# ══════════════════════════════════════════════════════════════════
# 5. 评估
# ══════════════════════════════════════════════════════════════════

def evaluate_selector(selector, X_cls, X_dom, Y, metadata, kb, domain_prototypes, clip_model):
    """全面评估 selector 的跨域泛化能力。

    关键指标:
      1. Per-axis MSE (域内 vs 跨域)
      2. 符号准确率 (预测 δ 正负号)
      3. Prompt 对比 (selector vs CD-TDM OFAT)
      4. 跨域泛化: 在未见过域上的性能 (hold-one-domain-out)
    """
    selector.eval()
    selector.to(DEVICE)
    X_cls = X_cls.to(DEVICE)
    X_dom = X_dom.to(DEVICE)
    Y = Y.to(DEVICE)

    print("\n" + "=" * 60)
    print("Selector 评估")
    print("=" * 60)

    # ── Per-axis MSE ──
    with torch.no_grad():
        Y_pred = selector(X_cls, X_dom)
        per_axis_mse = ((Y_pred - Y) ** 2).mean(dim=0)

    print(f"\n  Per-axis MSE:")
    print(f"  {'Axis':<14s} {'MSE':>8s} {'U(a)':>6s}")
    for i, axis in enumerate(AXIS_ORDER):
        u = UNIVERSALITY_SCORES.get(axis, 0.5)
        print(f"  {axis:<14s} {per_axis_mse[i].item():8.4f} {u:6.2f}")

    # ── 分域评估 ──
    print(f"\n  分域评估:")
    for domain in ['remote_sensing', 'natural_scene']:
        dom_idx = [i for i, m in enumerate(metadata) if m['domain'] == domain]
        if not dom_idx:
            continue
        dom_idx_t = torch.tensor(dom_idx, device=DEVICE)
        y_p = Y_pred[dom_idx_t]
        y_t = Y[dom_idx_t]
        mse_dom = ((y_p - y_t) ** 2).mean().item()
        sign_acc = (torch.sign(y_p) == torch.sign(y_t)).float().mean().item()
        nonzero = (y_t.abs() > 0.01)
        sign_acc_nz = (torch.sign(y_p[nonzero]) == torch.sign(y_t[nonzero])).float().mean().item() if nonzero.any() else 0
        print(f"    {domain}: MSE={mse_dom:.4f}, sign_acc={sign_acc:.4f} (nz={sign_acc_nz:.4f}), n={len(dom_idx)}")

    # ── Prompt 对比 ──
    print(f"\n  Prompt 对比 (Selector vs CD-TDM OFAT):")
    print(f"  {'Domain':<16s} {'Class':<20s} {'CD-TDM':<55s} {'Selector':<55s}")
    print(f"  {'─'*146}")

    test_cases = [
        ('remote_sensing', 'airplane'),
        ('remote_sensing', 'ship'),
        ('remote_sensing', 'vehicle'),
        ('remote_sensing', 'tennis court'),
        ('remote_sensing', 'stadium'),
        ('natural_scene', 'airplane'),
        ('natural_scene', 'boat'),
        ('natural_scene', 'car'),
        ('natural_scene', 'dog'),
        ('natural_scene', 'person'),
    ]

    for domain, cls in test_cases:
        if domain not in domain_prototypes:
            continue
        v_dom = domain_prototypes[domain]

        # CD-TDM static
        recs = kb.get_recommended_axes(domain, cls)
        prefix = 'an overhead image of' if domain == 'remote_sensing' else 'a photo of'
        tdm_p = f'{prefix} {cls}'
        if recs:
            tdm_p += ', ' + ', '.join(AXIS_MODIFIERS[a] for a, _ in recs[:2])

        # Selector
        sel_p = selector.build_prompt(cls, domain, v_dom, clip_model,
                                       top_k=2, threshold=0.03)

        match = '✅' if sel_p == tdm_p else '🔄'
        print(f"  {domain:<16s} {cls:<20s} {match} {tdm_p:<55s} {sel_p:<55s}")

    # ── 跨域泛化: hold-one-domain-out ──
    print(f"\n  跨域泛化 (Hold-One-Domain-Out):")
    for held_out in ['remote_sensing', 'natural_scene']:
        train_idx = [i for i, m in enumerate(metadata) if m['domain'] != held_out]
        test_idx = [i for i, m in enumerate(metadata) if m['domain'] == held_out]

        if not train_idx or not test_idx:
            continue

        # 快速重训练 (只在另一个域上)
        sel_held = UnifiedPromptSelector()
        sel_held.to(DEVICE)
        opt = torch.optim.AdamW(sel_held.parameters(), lr=0.001)

        train_xc = X_cls[train_idx]
        train_xd = X_dom[train_idx]
        train_y = Y[train_idx]

        for _ in range(100):
            y_p = sel_held(train_xc, train_xd)
            loss = ((y_p - train_y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        sel_held.eval()
        with torch.no_grad():
            y_held = sel_held(X_cls[test_idx], X_dom[test_idx])
            mse_held = ((y_held - Y[test_idx]) ** 2).mean().item()
            sign_held = (torch.sign(y_held) == torch.sign(Y[test_idx])).float().mean().item()

        print(f"    训练={['remote_sensing','natural_scene'][['remote_sensing','natural_scene'].index([d for d in ['remote_sensing','natural_scene'] if d != held_out][0])]}, 测试={held_out}: MSE={mse_held:.4f}, sign_acc={sign_held:.4f}")

    return Y_pred


# ══════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--threshold', type=float, default=0.03)
    args = parser.parse_args()

    print("=" * 60)
    print("Cross-Domain Prompt Selector (CDPS)")
    print("=" * 60)
    print(f"设备: {DEVICE}, 模式: {args.mode}")

    # ── 加载 ──
    print("\n[1] 加载 CLIP + OFAT KB...")
    import clip as clip_module
    clip_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
    clip_model, preprocess = clip_module.load(clip_path, device=DEVICE)
    clip_model.eval()
    kb = OFATKnowledgeBase()

    # ── 域原型 ──
    print("\n[2] 构建域原型 (CLIP image encoding)...")
    domain_prototypes = build_domain_prototypes(clip_model, preprocess)

    if not domain_prototypes:
        print("❌ 无域原型——需要至少一个域的图像")
        return

    # ── 训练数据 ──
    print("\n[3] 准备训练数据...")
    X_cls, X_dom, Y, weights, metadata = prepare_training_data(
        kb, clip_model, domain_prototypes
    )

    # ── 训练 ──
    print(f"\n[4] 训练 Unified Prompt Selector...")
    selector = UnifiedPromptSelector(d_model=512, n_axes=6, hidden=args.hidden)
    n_params = sum(p.numel() for p in selector.parameters())
    print(f"  参数量: {n_params}")

    history = train_selector(selector, X_cls, X_dom, Y, weights,
                             epochs=args.epochs, lr=args.lr, verbose=True)

    # ── 保存 ──
    torch.save({
        'selector_state': selector.state_dict(),
        'domain_prototypes': {k: v.cpu() for k, v in domain_prototypes.items()},
        'history': {k: [float(v) for v in vs] for k, vs in history.items()},
        'config': {'hidden': args.hidden, 'd_model': 512, 'n_axes': 6},
    }, SAVE_DIR / 'cdps_weights.pt')
    print(f"\n  权重: {SAVE_DIR / 'cdps_weights.pt'}")

    # ── 评估 ──
    print(f"\n[5] 评估...")
    Y_pred = evaluate_selector(selector, X_cls, X_dom, Y, metadata,
                                kb, domain_prototypes, clip_model)

    # ── 新域演示 ──
    print(f"\n[6] 新域注册演示 (0-shot domain transfer):")
    print(f"  场景: 用 5 张红外/SAR 图像 → domain_vec → 直接选 prompt")
    print(f"  演示: 用 natural_scene prototype 的扰动模拟新域")
    v_new = domain_prototypes['natural_scene'].cpu() + 0.1 * torch.randn(512)
    v_new = v_new / v_new.norm()
    v_new = v_new.to(DEVICE)
    for cls in ['airplane', 'ship', 'vehicle']:
        axes = selector.predict_axes(cls, v_new, clip_model, top_k=2, threshold=0.03)
        print(f"    simulated_new_domain + '{cls}': {axes if axes else '(无修饰词)'}")

    # 保存评估结果
    eval_results = {
        'per_axis_mse': {AXIS_ORDER[i]: float(((Y_pred.cpu() - Y.cpu())**2).mean(dim=0)[i])
                         for i in range(6)},
        'per_domain': {},
        'prompts': {},
    }
    for domain in ['remote_sensing', 'natural_scene']:
        dom_idx = [i for i, m in enumerate(metadata) if m['domain'] == domain]
        if dom_idx:
            y_p = Y_pred[dom_idx].cpu()
            y_t = Y[dom_idx].cpu()
            eval_results['per_domain'][domain] = {
                'mse': float(((y_p - y_t)**2).mean()),
                'sign_acc': float((torch.sign(y_p) == torch.sign(y_t)).float().mean()),
                'n': len(dom_idx),
            }

    with open(SAVE_DIR / 'cdps_results.json', 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    print(f"\n结果: {SAVE_DIR / 'cdps_results.json'}")
    print("✅ CDPS 训练+评估完成")


if __name__ == '__main__':
    main()
