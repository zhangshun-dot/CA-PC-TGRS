# -*- coding: utf-8 -*-
"""Cross-Domain Token Alignment (CDTA) — 跨域 Token 投影对齐

核心架构:
  Domain A (遥感)    Domain B (自然图像)
       │                    │
  E_A(cls)              E_B(cls)         ← CLIP 文本编码 (512-d)
       │                    │
  P_A (W_A)             P_B (W_B)        ← 域投影器 W_D = I + U_D@V_D^T
       │                    │
       └────────┬───────────┘
                ▼
      Universal Token Space (UTS)        ← 共享 512-d 流形
                │
      Unified TDM Prompt Selector        ← 轴选择在 UTS 中跨域统一

训练信号 (修正版):
  L = L_contrastive + λ1·L_axis + λ2·L_delta + λ3·L_identity

  1. 对比损失: 多层配对 (精确 + CLIP 软配对 + δ符号对齐)
  2. 轴一致性: 同一类在 UTS 中的投影沿 v_a 位移方向应跨域一致
  3. δ 预测损失: 从 UTS 投影重建 OFAT δ → 保证 TDM 可用
  4. 恒等正则: |W_D - I| 不过大

Usage:
    python cross_domain_token_align.py --mode train
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
SAVE_DIR.mkdir(exist_ok=True)

from token_projection import (
    TokenProjection, AXIS_MODIFIERS, AXIS_ORDER, UNIVERSALITY_SCORES,
    compute_axis_directions,
)
from ofat_knowledge_base import OFATKnowledgeBase


# ══════════════════════════════════════════════════════════════════
# 1. Universal Token Space
# ══════════════════════════════════════════════════════════════════

class UniversalTokenSpace(nn.Module):
    """跨域共享的 Token 流形 (512-d CLIP 空间)。

    维护跨域统一的轴方向向量。训练后从两个域投影的平均方向重新估计。
    """

    def __init__(self, axis_dirs_clip=None):
        super().__init__()
        self.dim = 512
        self.axis_dirs = axis_dirs_clip or {}

    def update_axis_directions(self, proj_rs, proj_ns, clip_model):
        """训练后更新 UTS 轴方向: 对 15 个通用类, 两域投影后取平均位移方向。"""
        import clip as clip_module
        generic_classes = [
            'airplane', 'bird', 'car', 'dog', 'cat', 'ship', 'bridge',
            'stadium', 'building', 'tree', 'road', 'person', 'truck',
            'bicycle', 'motorcycle',
        ][:15]

        new_dirs = {}
        for axis in AXIS_ORDER:
            all_directions = []
            for cls_name in generic_classes:
                base = f'a photo of {cls_name}'
                mod = f'a photo of {cls_name}, {AXIS_MODIFIERS[axis]}'
                b_tok = clip_module.tokenize([base], truncate=True).to(DEVICE)
                m_tok = clip_module.tokenize([mod], truncate=True).to(DEVICE)
                with torch.no_grad():
                    b_emb = clip_model.encode_text(b_tok).float()
                    m_emb = clip_model.encode_text(m_tok).float()
                    b_emb = b_emb / b_emb.norm(dim=-1, keepdim=True)
                    m_emb = m_emb / m_emb.norm(dim=-1, keepdim=True)

                dirs = []
                for proj in [proj_rs, proj_ns]:
                    b_proj = proj(b_emb)
                    m_proj = proj(m_emb)
                    d = m_proj - b_proj
                    d = d / (d.norm() + 1e-8)
                    dirs.append(d.squeeze(0))
                avg_dir = torch.stack(dirs).mean(dim=0)
                all_directions.append(avg_dir)

            mean_dir = torch.stack(all_directions).mean(dim=0)
            new_dirs[axis] = mean_dir / (mean_dir.norm() + 1e-8)

        self.axis_dirs = new_dirs
        return new_dirs

    def get_axis_similarity_matrix(self):
        axes = [a for a in AXIS_ORDER if a in self.axis_dirs]
        n = len(axes)
        matrix = np.zeros((n, n))
        for i, a1 in enumerate(axes):
            for j, a2 in enumerate(axes):
                d1, d2 = self.axis_dirs[a1], self.axis_dirs[a2]
                matrix[i, j] = float(F.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0)))
        return axes, matrix


# ══════════════════════════════════════════════════════════════════
# 2. 跨域训练数据准备
# ══════════════════════════════════════════════════════════════════

def prepare_cross_domain_data(kb, clip_model):
    """三层配对策略 + OFAT δ 向量。

    Returns:
        all_pairs: [(rs_cls, ns_cls, e_rs, e_ns, weight), ...]
        rs_embs, ns_embs: Dict[str, Tensor(512,)]
        rs_deltas, ns_deltas: Dict[str, Dict[str, float]]
        negative_rs, negative_ns: 负类池
    """
    import clip as clip_module

    # L1: 精确匹配
    exact_map = [
        ('airplane', 'airplane'), ('ship', 'boat'), ('vehicle', 'car'),
        ('stadium', 'stadium'), ('bridge', 'bridge'), ('harbor', 'boat'),
        ('train station', 'train station'), ('tennis court', 'tennis court'),
    ]

    dior_classes = kb.dior_classes
    coco_class_names = sorted(set(k[0] for k in kb.coco_deltas.keys()))

    def encode_cls(cls_name):
        prompt = f'a photo of {cls_name}'
        tok = clip_module.tokenize([prompt], truncate=True).to(DEVICE)
        with torch.no_grad():
            e = clip_model.encode_text(tok).float()
        return (e / e.norm(dim=-1, keepdim=True)).squeeze(0).detach()

    print("  编码 DIOR 类...")
    rs_embs = {cls: encode_cls(cls) for cls in dior_classes}

    print("  编码 COCO 类...")
    ns_embs = {}
    for cls in coco_class_names:
        if len(cls) < 40:
            ns_embs[cls] = encode_cls(cls)

    # L1: 精确匹配
    print("  构建跨域配对...")
    all_pairs = []
    for rs_cls, ns_cls in exact_map:
        if rs_cls in rs_embs and ns_cls in ns_embs:
            all_pairs.append((rs_cls, ns_cls, rs_embs[rs_cls], ns_embs[ns_cls], 1.0))
            print(f"    [exact] {rs_cls} ↔ {ns_cls}")

    # L2: 软配对 (CLIP cos > 0.85)
    used_rs, used_ns = {p[0] for p in all_pairs}, {p[1] for p in all_pairs}
    rc = [(c, e) for c, e in rs_embs.items() if c not in used_rs]
    nc = [(c, e) for c, e in ns_embs.items() if c not in used_ns]
    soft_count = 0
    for rs_c, rs_e in rc:
        best_ns, best_sim = None, 0
        for ns_c, ns_e in nc:
            sim = float(F.cosine_similarity(rs_e.unsqueeze(0), ns_e.unsqueeze(0)))
            if sim > best_sim:
                best_sim, best_ns = sim, ns_c
        if best_sim > 0.85 and best_ns:
            all_pairs.append((rs_c, best_ns, rs_e, ns_embs[best_ns], 0.3))
            soft_count += 1
            if soft_count <= 8:
                print(f"    [soft]  {rs_c} ↔ {best_ns} (cos={best_sim:.3f})")

    print(f"  总计: {len(all_pairs)} 对 (exact={len(all_pairs)-soft_count}, soft={soft_count})")

    # OFAT δ 向量
    rs_deltas, ns_deltas = {}, {}
    for cls in dior_classes:
        d = {a: kb.get_delta('remote_sensing', cls, a) for a in AXIS_ORDER
             if kb.get_delta('remote_sensing', cls, a) is not None}
        if d: rs_deltas[cls] = d
    for cls in coco_class_names:
        d = {a: kb.get_delta('natural_scene', cls, a) for a in AXIS_ORDER
             if kb.get_delta('natural_scene', cls, a) is not None}
        if d: ns_deltas[cls] = d

    used_rs_all, used_ns_all = {p[0] for p in all_pairs}, {p[1] for p in all_pairs}
    negative_rs = [(c, e) for c, e in rs_embs.items() if c not in used_rs_all]
    negative_ns = [(c, e) for c, e in ns_embs.items() if c not in used_ns_all]

    return all_pairs, rs_embs, ns_embs, rs_deltas, ns_deltas, negative_rs, negative_ns


# ══════════════════════════════════════════════════════════════════
# 3. CrossDomainAligner — 训练跨域投影对齐 (修正版)
# ══════════════════════════════════════════════════════════════════

class CrossDomainAligner:
    """四重损失训练: contrastive + axis_consistency + delta_prediction + identity"""

    def __init__(self, all_pairs, rs_embs, ns_embs,
                 rs_deltas, ns_deltas, negative_rs, negative_ns,
                 axis_dirs_clip, clip_model,
                 rank=4, tau=0.05, lambda_axis=1.0,
                 lambda_delta=0.5, lambda_id=0.01):
        self.all_pairs = all_pairs
        self.rs_embs = rs_embs
        self.ns_embs = ns_embs
        self.rs_deltas = rs_deltas
        self.ns_deltas = ns_deltas
        self.negative_rs = negative_rs
        self.negative_ns = negative_ns
        self.axis_dirs_clip = axis_dirs_clip
        self.clip_model = clip_model
        self.tau = tau
        self.lambda_axis = lambda_axis
        self.lambda_delta = lambda_delta
        self.lambda_id = lambda_id

        self.proj_rs = TokenProjection(d_model=512, rank=rank, domain='remote_sensing')
        self.proj_ns = TokenProjection(d_model=512, rank=rank, domain='natural_scene')
        self.uts = UniversalTokenSpace(axis_dirs_clip)

    def to_device(self):
        self.proj_rs.to(DEVICE)
        self.proj_ns.to(DEVICE)
        return self

    # ── 损失 1: 对比损失 (加权 InfoNCE) ──
    def contrastive_loss(self, pos_pairs, neg_rs_sample, neg_ns_sample):
        if len(pos_pairs) == 0:
            return torch.tensor(0.0, device=DEVICE)

        z_rs_list, z_ns_list, weights = [], [], []
        for _, _, e_rs, e_ns, w in pos_pairs:
            z_rs_list.append(self.proj_rs(e_rs.unsqueeze(0).to(DEVICE)))
            z_ns_list.append(self.proj_ns(e_ns.unsqueeze(0).to(DEVICE)))
            weights.append(w)

        z_rs = torch.cat(z_rs_list, dim=0)
        z_ns = torch.cat(z_ns_list, dim=0)
        N = z_rs.shape[0]

        # 添加负对
        n_neg = min(N, len(neg_rs_sample), len(neg_ns_sample))
        neg_embs_rs = torch.stack([e.to(DEVICE) for _, e in neg_rs_sample[:n_neg]])
        neg_embs_ns = torch.stack([e.to(DEVICE) for _, e in neg_ns_sample[:n_neg]])
        neg_z_rs = self.proj_rs(neg_embs_rs)
        neg_z_ns = self.proj_ns(neg_embs_ns)

        all_z_rs = torch.cat([z_rs, neg_z_rs], dim=0)
        all_z_ns = torch.cat([z_ns, neg_z_ns], dim=0)
        all_z_rs = all_z_rs / all_z_rs.norm(dim=-1, keepdim=True)
        all_z_ns = all_z_ns / all_z_ns.norm(dim=-1, keepdim=True)

        # 双向 InfoNCE
        logits = all_z_rs @ all_z_ns.T / self.tau
        labels = torch.arange(N, device=DEVICE)

        # 加权: 精确配对权重更高
        w_tensor = torch.tensor(weights, device=DEVICE)
        loss_rs2ns = (F.cross_entropy(logits[:N], labels, reduction='none') * w_tensor).mean()
        loss_ns2rs = (F.cross_entropy(logits[:N].T[:N], labels, reduction='none') * w_tensor).mean()

        return (loss_rs2ns + loss_ns2rs) / 2.0

    # ── 损失 2: 轴一致性 (修正版) ──
    def axis_consistency_loss(self):
        """测量：同一类被两域投影后，沿 UTS 各轴方向的位移是否一致。

        对每个共享类和每个轴：
          1. 计算 CLIP 基类嵌入 E 和 E+v_a (添加修饰词)
          2. 分别通过 P_rs 和 P_ns 投影
          3. 比较投影后的位移方向: cos(P_rs(E+v_a)-P_rs(E), P_ns(E+v_a)-P_ns(E))
          4. L = 1 - cos
        """
        import clip as clip_module
        if not self.axis_dirs_clip:
            return torch.tensor(0.0, device=DEVICE)

        total_loss, n_terms = torch.tensor(0.0, device=DEVICE), 0
        eps = 0.1  # 扰动量级

        # 只对精确配对做轴一致性 (软配对不可靠)
        exact_pairs = [(rs, ns, e_rs, e_ns) for rs, ns, e_rs, e_ns, w in self.all_pairs if w >= 1.0]

        for _, _, e_rs, e_ns in exact_pairs:
            for axis in AXIS_ORDER:
                if axis not in self.axis_dirs_clip:
                    continue

                v_a = self.axis_dirs_clip[axis].to(DEVICE).float()
                v_a = v_a / (v_a.norm() + 1e-8)

                # RS: 基类 + 基类+修饰词方向
                e_rs_d = e_rs.to(DEVICE)
                z_rs_base = self.proj_rs(e_rs_d.unsqueeze(0))
                z_rs_mod = self.proj_rs((e_rs_d + eps * v_a).unsqueeze(0))
                dir_rs = z_rs_mod - z_rs_base
                dir_rs = dir_rs / (dir_rs.norm() + 1e-8)

                # NS: 同上
                e_ns_d = e_ns.to(DEVICE)
                z_ns_base = self.proj_ns(e_ns_d.unsqueeze(0))
                z_ns_mod = self.proj_ns((e_ns_d + eps * v_a).unsqueeze(0))
                dir_ns = z_ns_mod - z_ns_base
                dir_ns = dir_ns / (dir_ns.norm() + 1e-8)

                cos_sim = F.cosine_similarity(dir_rs, dir_ns)
                total_loss = total_loss + (1.0 - cos_sim)
                n_terms += 1

        return total_loss / max(n_terms, 1)

    # ── 损失 3: OFAT δ 预测损失 ──
    def delta_prediction_loss(self):
        """从 UTS 投影嵌入预测 OFAT δ(c, a)。

        对每个域内的类:
          δ_pred(c, a) = cos(P_D(E_c), v_a_uts) * sign(δ_known)
          L = MSE(δ_pred, δ_known)
        """
        total_loss, n_terms = torch.tensor(0.0, device=DEVICE), 0

        # 遥感域
        for cls_name, deltas in self.rs_deltas.items():
            if cls_name not in self.rs_embs:
                continue
            e = self.rs_embs[cls_name].unsqueeze(0).to(DEVICE)
            z = self.proj_rs(e)
            z = z / z.norm(dim=-1, keepdim=True)

            for axis, delta_true in deltas.items():
                if axis not in self.axis_dirs_clip:
                    continue
                v_a = self.axis_dirs_clip[axis].to(DEVICE).float()
                delta_pred = float(F.cosine_similarity(z, v_a.unsqueeze(0)))
                # 用已知符号放大信号
                delta_pred = delta_pred * np.sign(delta_true) if delta_true != 0 else 0.0

                total_loss = total_loss + (delta_pred - delta_true) ** 2
                n_terms += 1

        # 自然图像域
        for cls_name, deltas in self.ns_deltas.items():
            if cls_name not in self.ns_embs:
                continue
            e = self.ns_embs[cls_name].unsqueeze(0).to(DEVICE)
            z = self.proj_ns(e)
            z = z / z.norm(dim=-1, keepdim=True)

            for axis, delta_true in deltas.items():
                if axis not in self.axis_dirs_clip:
                    continue
                v_a = self.axis_dirs_clip[axis].to(DEVICE).float()
                delta_pred = float(F.cosine_similarity(z, v_a.unsqueeze(0)))
                delta_pred = delta_pred * np.sign(delta_true) if delta_true != 0 else 0.0

                total_loss = total_loss + (delta_pred - delta_true) ** 2
                n_terms += 1

        return total_loss / max(n_terms, 1)

    # ── 损失 4: 恒等正则 ──
    def identity_loss(self):
        l_rs = (self.proj_rs.U.norm('fro')**2 + self.proj_rs.V.norm('fro')**2) / self.proj_rs.d_model
        l_ns = (self.proj_ns.U.norm('fro')**2 + self.proj_ns.V.norm('fro')**2) / self.proj_ns.d_model
        return (l_rs + l_ns) / 2.0

    # ── 训练循环 ──
    def train(self, epochs=300, lr=0.005, verbose=True):
        self.to_device()
        params = list(self.proj_rs.parameters()) + list(self.proj_ns.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        history = defaultdict(list)

        print(f"\n  训练配置: epochs={epochs}, lr={lr}, τ={self.tau}")
        print(f"  配对: {len(self.all_pairs)} (精确={sum(1 for p in self.all_pairs if p[4]>=1.0)})")
        print(f"  λ_axis={self.lambda_axis}, λ_delta={self.lambda_delta}, λ_id={self.lambda_id}")
        print(f"  δ 监督: RS={len(self.rs_deltas)}类, NS={len(self.ns_deltas)}类")

        for epoch in range(epochs):
            n_neg = min(len(self.all_pairs), len(self.negative_rs), len(self.negative_ns))
            if n_neg == 0:
                break
            idx_rs = np.random.choice(len(self.negative_rs), n_neg, replace=False)
            idx_ns = np.random.choice(len(self.negative_ns), n_neg, replace=False)
            neg_rs_sample = [self.negative_rs[i] for i in idx_rs]
            neg_ns_sample = [self.negative_ns[i] for i in idx_ns]

            l_c = self.contrastive_loss(self.all_pairs, neg_rs_sample, neg_ns_sample)
            l_a = self.axis_consistency_loss()
            l_d = self.delta_prediction_loss()
            l_i = self.identity_loss()

            loss = l_c + self.lambda_axis * l_a + self.lambda_delta * l_d + self.lambda_id * l_i

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k, v in [('loss', loss), ('contrast', l_c), ('axis', l_a),
                         ('delta', l_d), ('ident', l_i)]:
                history[k].append(v.item())
            history['adj_rs'].append(self.proj_rs.get_adjustment_magnitude())
            history['adj_ns'].append(self.proj_ns.get_adjustment_magnitude())

            if verbose and epoch % 30 == 0:
                print(f"  E{epoch:3d} | L={loss.item():.4f} c={l_c.item():.4f} "
                      f"a={l_a.item():.4f} d={l_d.item():.4f} id={l_i.item():.4f} "
                      f"|Δrs|={self.proj_rs.get_adjustment_magnitude():.3f} "
                      f"|Δns|={self.proj_ns.get_adjustment_magnitude():.3f}")

        # 更新 UTS
        print("\n  更新 UTS 轴方向...")
        self.uts.update_axis_directions(self.proj_rs, self.proj_ns, self.clip_model)

        # 报告
        print("\n  跨域对齐度 (投影后):")
        for rs_cls, ns_cls, e_rs, e_ns, w in self.all_pairs:
            with torch.no_grad():
                z_rs = self.proj_rs(e_rs.unsqueeze(0).to(DEVICE))
                z_ns = self.proj_ns(e_ns.unsqueeze(0).to(DEVICE))
                proj_cos = float(F.cosine_similarity(z_rs, z_ns))
                raw_cos = float(F.cosine_similarity(
                    e_rs.unsqueeze(0).to(DEVICE), e_ns.unsqueeze(0).to(DEVICE)))
            tag = '✅' if proj_cos > raw_cos else '⚠️'
            print(f"    {rs_cls}↔{ns_cls}: raw={raw_cos:.4f} → proj={proj_cos:.4f} {tag}")

        return history


# ══════════════════════════════════════════════════════════════════
# 4. UnifiedTDM — UTS 中的跨域 Prompt 选择
# ══════════════════════════════════════════════════════════════════

class UnifiedTDM:
    """跨域统一 TDM 推理器。

    Image → DomainClassifier → P_D → UTS → 轴选择 → prompt 构造
    """

    def __init__(self, proj_rs, proj_ns, uts, clip_model, kb,
                 threshold=0.15, max_modifiers=2):
        self.proj_rs = proj_rs
        self.proj_ns = proj_ns
        self.uts = uts
        self.clip_model = clip_model
        self.kb = kb
        self.threshold = threshold
        self.max_modifiers = max_modifiers

        try:
            from domain_classifier import DomainClassifier
            self.domain_classifier = DomainClassifier()
        except Exception:
            self.domain_classifier = None

    def get_projector(self, domain):
        m = {'remote_sensing': self.proj_rs, 'natural_scene': self.proj_ns}
        return m.get(domain, nn.Identity())

    def _encode_cls_uts(self, cls_name, proj):
        import clip as clip_module
        prompt = f'a photo of {cls_name}'
        tok = clip_module.tokenize([prompt], truncate=True).to(DEVICE)
        with torch.no_grad():
            e = self.clip_model.encode_text(tok).float()
            e = e / e.norm(dim=-1, keepdim=True)
        z = proj(e)
        return z / z.norm(dim=-1, keepdim=True)

    def select_axes_in_uts(self, cls_name, proj):
        """在 UTS 中为类选择最优修饰轴。

        得分 = cos(P_D(E_cls), v_a_uts), 用 universality 做温度。
        """
        z_uts = self._encode_cls_uts(cls_name, proj).detach()

        axis_scores = []
        for axis in AXIS_ORDER:
            if axis not in self.uts.axis_dirs:
                continue
            v_a = self.uts.axis_dirs[axis].to(DEVICE)
            raw_score = float(F.cosine_similarity(z_uts, v_a.unsqueeze(0)))
            u = UNIVERSALITY_SCORES.get(axis, 0.5)
            adjusted = raw_score * u

            if adjusted > self.threshold:
                axis_scores.append((axis, adjusted))

        axis_scores.sort(key=lambda x: -x[1])
        return axis_scores[:self.max_modifiers]

    def build_prompt(self, image_or_domain, cls_name):
        """构建跨域统一 prompt。"""
        if isinstance(image_or_domain, str):
            domain = image_or_domain
        elif self.domain_classifier:
            domain = self.domain_classifier.classify(image_or_domain)
        else:
            domain = 'remote_sensing'

        prefixes = {
            'remote_sensing': 'an overhead image of',
            'natural_scene': 'a photo of',
        }
        prefix = prefixes.get(domain, 'an image of')
        proj = self.get_projector(domain)
        axes = self.select_axes_in_uts(cls_name, proj)

        prompt = f'{prefix} {cls_name}'
        if axes:
            prompt += ', ' + ', '.join(AXIS_MODIFIERS[ax] for ax, _ in axes)
        return prompt


# ══════════════════════════════════════════════════════════════════
# 5. 验证实验
# ══════════════════════════════════════════════════════════════════

def run_validation(aligner, clip_model, kb):
    """跨域验证: CDTA vs Baseline vs CD-TDM static"""

    print("\n" + "=" * 60)
    print("跨域验证实验")
    print("=" * 60)

    utdm = UnifiedTDM(aligner.proj_rs, aligner.proj_ns, aligner.uts,
                       clip_model, kb, threshold=0.10, max_modifiers=2)

    # 跨域对齐度量
    print(f"\n  {'Class Pair':<30s} {'Raw cos':>8s} {'UTS cos':>8s} {'Gain':>8s}")
    print(f"  {'─'*55}")
    gains = []
    for rs_cls, ns_cls, e_rs, e_ns, _ in aligner.all_pairs:
        with torch.no_grad():
            raw = float(F.cosine_similarity(
                e_rs.unsqueeze(0).to(DEVICE), e_ns.unsqueeze(0).to(DEVICE)))
            z_rs = aligner.proj_rs(e_rs.unsqueeze(0).to(DEVICE))
            z_ns = aligner.proj_ns(e_ns.unsqueeze(0).to(DEVICE))
            proj = float(F.cosine_similarity(z_rs, z_ns))
        gains.append(proj - raw)
        print(f"  {rs_cls+'↔'+ns_cls:<30s} {raw:8.4f} {proj:8.4f} {proj-raw:+8.4f}")

    print(f"\n  平均 Gain: {np.mean(gains):+.4f}")
    pos_gain = sum(1 for g in gains if g > 0)
    print(f"  正增益: {pos_gain}/{len(gains)}")

    # Prompt 对比
    print(f"\n  Prompt 对比:")
    test_classes = [
        ('airplane', 'natural_scene'),
        ('boat', 'natural_scene'),
        ('car', 'natural_scene'),
        ('airplane', 'remote_sensing'),
        ('ship', 'remote_sensing'),
        ('vehicle', 'remote_sensing'),
    ]
    for cls, dom in test_classes:
        proj = utdm.get_projector(dom)
        axes = utdm.select_axes_in_uts(cls, proj)
        cdta_p = utdm.build_prompt(dom, cls)
        recs = kb.get_recommended_axes(dom, cls)
        tdm_p = f"{'a photo of' if dom == 'natural_scene' else 'an overhead image of'} {cls}"
        if recs:
            tdm_p += ', ' + ', '.join(AXIS_MODIFIERS[a] for a, _ in recs[:2])
        print(f"  [{dom[:4]:>4s}] {cls:<20s}")
        print(f"    CD-TDM:  '{tdm_p}'")
        print(f"    CDTA:    '{cdta_p}'")
        if axes:
            print(f"    UTS轴:   {[(a, f'{s:.3f}') for a, s in axes]}")

    # 轴方向诊断
    print(f"\n  UTS 轴方向余弦相似度:")
    axes, matrix = utdm.uts.get_axis_similarity_matrix()
    header = ' ' * 14 + ' '.join(f'{a[:6]:>7s}' for a in axes)
    print(f"  {header}")
    for i, a in enumerate(axes):
        row = ' '.join(f'{matrix[i,j]:+.3f}' for j in range(len(axes)))
        print(f"  {a:>14s} {row}")

    return utdm


# ══════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--lambda_axis', type=float, default=1.0)
    parser.add_argument('--lambda_delta', type=float, default=0.5)
    parser.add_argument('--lambda_id', type=float, default=0.01)
    parser.add_argument('--tau', type=float, default=0.05)
    args = parser.parse_args()

    print("=" * 60)
    print("Cross-Domain Token Alignment (CDTA)")
    print("=" * 60)
    print(f"设备: {DEVICE}")

    # CLIP
    print("\n[1] 加载 CLIP...")
    import clip as clip_module
    clip_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
    clip_model, _ = clip_module.load(clip_path, device=DEVICE)
    clip_model.eval()

    # OFAT KB
    print("[2] 加载 OFAT Knowledge Base...")
    kb = OFATKnowledgeBase()

    # 轴方向
    print("[3] 计算 CLIP 轴方向向量...")
    axis_dirs_clip = compute_axis_directions(clip_model, clip_module.tokenize, device=DEVICE)

    # 数据
    print("[4] 准备跨域训练数据...")
    (all_pairs, rs_embs, ns_embs, rs_deltas, ns_deltas,
     negative_rs, negative_ns) = prepare_cross_domain_data(kb, clip_model)

    # 训练
    print("\n[5] 训练 CrossDomainAligner...")
    aligner = CrossDomainAligner(
        all_pairs, rs_embs, ns_embs,
        rs_deltas, ns_deltas, negative_rs, negative_ns,
        axis_dirs_clip, clip_model,
        rank=args.rank, tau=args.tau,
        lambda_axis=args.lambda_axis,
        lambda_delta=args.lambda_delta,
        lambda_id=args.lambda_id,
    )

    history = aligner.train(epochs=args.epochs, lr=args.lr, verbose=True)

    # 保存
    print("\n[6] 保存权重 + 验证...")
    torch.save({
        'proj_rs': {k: v.data.cpu() for k, v in aligner.proj_rs.state_dict().items()},
        'proj_ns': {k: v.data.cpu() for k, v in aligner.proj_ns.state_dict().items()},
        'uts_axis_dirs': {a: d.cpu() for a, d in aligner.uts.axis_dirs.items()},
        'history': {k: [float(v) for v in vs] for k, vs in history.items()},
    }, SAVE_DIR / 'cdta_weights.pt')

    utdm = run_validation(aligner, clip_model, kb)

    # 验证结果
    val_results = {
        'uts_alignment': {},
        'axis_similarity': {},
        'prompts': {},
    }
    for rs_cls, ns_cls, e_rs, e_ns, _ in aligner.all_pairs:
        with torch.no_grad():
            raw = float(F.cosine_similarity(e_rs.unsqueeze(0).to(DEVICE), e_ns.unsqueeze(0).to(DEVICE)))
            z_rs = aligner.proj_rs(e_rs.unsqueeze(0).to(DEVICE))
            z_ns = aligner.proj_ns(e_ns.unsqueeze(0).to(DEVICE))
            proj = float(F.cosine_similarity(z_rs, z_ns))
        val_results['uts_alignment'][f'{rs_cls}↔{ns_cls}'] = {
            'raw_cos': raw, 'uts_cos': proj, 'gain': proj - raw
        }

    for dom in ['remote_sensing', 'natural_scene']:
        for cls in (['airplane', 'ship', 'vehicle'] if dom == 'remote_sensing'
                    else ['airplane', 'boat', 'car']):
            cdta_p = utdm.build_prompt(dom, cls)
            recs = kb.get_recommended_axes(dom, cls)
            prefix = 'an overhead image of' if dom == 'remote_sensing' else 'a photo of'
            tdm_p = f'{prefix} {cls}'
            if recs:
                tdm_p += ', ' + ', '.join(AXIS_MODIFIERS[a] for a, _ in recs[:2])
            val_results['prompts'][f'{dom}/{cls}'] = {
                'cd_tdm': tdm_p, 'cdta': cdta_p
            }

    with open(SAVE_DIR / 'cdta_results.json', 'w', encoding='utf-8') as f:
        json.dump(val_results, f, indent=2, ensure_ascii=False)

    print(f"\n权重: {SAVE_DIR / 'cdta_weights.pt'}")
    print(f"结果: {SAVE_DIR / 'cdta_results.json'}")
    print("✅ CDTA 训练+验证完成")


if __name__ == '__main__':
    main()
