# -*- coding: utf-8 -*-
"""tge_probe.py — 阶段1: CLIP 空间语言外推原型分离度探针 (ExDet TGE 迁移)

设计文档: Desktop\TGE_probe_design.md
协议与 route_proxy_extend.py 完全对齐 (GDINO th=0 -> pre-filter 0.03 -> rank-AUC),
特征编码换成 CLIP ViT-B/32 (本地权重 Desktop\weights\clip\ViT-B-32.pt)。

回答三问:
  H1  外推原型是否比源原型更接近真实目标原型?            → 余弦表
  H2  外推原型分离度 AUC 是否压过同空间真实原型 (B2)?    → B3 vs B2
  H2b boat 外推 AUC 是否跨过失效区 (>0.60)?

基线:
  B1  dinov2 真实原型 AUC — 直接读 route_proxy_probe_12.json (不重算)
  B2  CLIP 真实原型 AUC   — 本脚本重算 (控制"换编码器增益", 关键对照!)
  B3  外推原型 AUC        — 3 变体(joint/cat/dom) x beta 扫描

外推公式 (CLIP 空间, DeltaSpace):
  P_hat = norm( P_src + beta * Δ )
  joint: Δ = d(y) − s(x_s)      # 类别+域联合
  cat:   Δ = d(y) − d(x_s)      # 纯类别
  dom:   Δ = d(x_s) − s(x_s)    # 纯域
  s(x) = text("an overhead image of {x}")   d(x) = text("a photo of {x}")

输出: 实验数据/tge_probe_12.json + 控制台表
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
from pathlib import Path
import numpy as np
import torch
import clip
import unified_eval as U

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_probe_12_mt.json'     # 2026-08-22: 多模板版, 旧单模板版保留对比
PROBE_12 = DATA / 'route_proxy_probe_12.json'   # B1 值来源 (dinov2 口径)

N_COCO = 500          # 与 route_proxy_extend 统一
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
# 2026-08-22 多模板集成 (ExDet 式): 每域 4 模板文本嵌入平均
SRC_TMPLS = [
    'an overhead image of {}',
    'a satellite image of {}',
    'an aerial photograph of {}',
    'a remote sensing image of {}',
]
TGT_TMPLS = [
    'a photo of {}',
    'a photograph of {}',
    'a picture of {}',
    'a natural scene image of {}',
]

# (name, coco_cls, 源 DIOR 类)
TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]
BETAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]   # 2026-08-22: β=2.0 未饱和, 扩上限找峰
# 2026-08-22: 'proj' = 外推前约束, Δ_joint 投影到 [全局域方向, 类别方向] 子空间
VARIANTS = ['joint', 'cat', 'dom', 'proj']


# ══════════════════════════ 基础工具 ══════════════════════════

_clip_m = None
_clip_pre = None

def get_clip():
    global _clip_m, _clip_pre
    if _clip_m is None:
        _clip_m, _clip_pre = U.get_clip()   # (model, preprocess), ViT-B/32
    return _clip_m, _clip_pre


def encode_clip(crop):
    m, pre = get_clip()
    t = pre(crop).unsqueeze(0).to(U.DEVICE)
    with torch.no_grad():
        f = m.encode_image(t).float()
    f = f[0]
    return f / f.norm()


def encode_clip_batch(crops, bs=256):
    m, pre = get_clip()
    feats = []
    for i in range(0, len(crops), bs):
        t = torch.stack([pre(c) for c in crops[i:i + bs]]).to(U.DEVICE)
        with torch.no_grad():
            f = m.encode_image(t).float()
        feats.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(feats)


def text_emb(texts):
    m, _ = get_clip()
    tok = clip.tokenize(texts).to(U.DEVICE)
    with torch.no_grad():
        f = m.encode_text(tok).float()
    return f / f.norm(dim=-1, keepdim=True)


def text_avg(templates, x):
    """多模板文本嵌入平均 (ExDet 式集成): mean over 模板, 归一化 (512,)。"""
    embs = text_emb([t.format(x) for t in templates])   # (n, 512)
    p = embs.mean(dim=0)
    return p / p.norm()


def get_neg_proto(protos, pos_cls):
    """CLIP 空间负原型: 与正原型最接近的 top-3 其他类原型均值 (同 get_neg_proto 协议)。"""
    pos = protos[pos_cls]
    others = torch.stack([protos[c] for c in protos if c != pos_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return neg / neg.norm()


def rank_auc(vis_all, lab_all):
    vis_all = np.asarray(vis_all); lab_all = np.asarray(lab_all)
    pos = vis_all[lab_all == 1]; neg = vis_all[lab_all == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    n_pos, n_neg = len(pos), len(neg)
    order = np.argsort(vis_all)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(vis_all))
    return (rank[lab_all == 1].mean() - (n_pos - 1) / 2) / n_neg


def roundtrip_residual(P_src, delta, beta):
    """外推后矫正: 往返一致性 1−cos(P̂−β·Δ, P_src), 0=完美往返 (纯源域自洽, 零目标域数据)。

    P̂ = norm(P_src + β·Δ), 反向外推 P̃ = norm(P̂ − β·Δ) 应回到 P_src。
    归一化的非线性使 β 越大残差越大 → 残差是"外推强度合理性"的源域可得指标。
    """
    P_src = P_src.cpu()
    delta = delta.cpu()
    P_hat = P_src + beta * delta
    P_hat = P_hat / P_hat.norm()
    P_tilde = P_hat - beta * delta
    P_tilde = P_tilde / (P_tilde.norm() + 1e-8)
    return float(1.0 - (P_tilde @ P_src).item())


def cached_dets(images, prompt, name):
    """GDINO 检测结果 pickle 缓存 (检测是大头, 修 bug 重跑时避免 1 小时重来)。"""
    p = DATA / f'tge_probe_dets_{name}.pkl'
    if p.exists():
        print(f'    [缓存] 检测结果 {p.name}')
        return pickle.load(open(p, 'rb'))
    dets = U.gdino_detect(images, prompt, 0.0)
    pickle.dump(dets, open(p, 'wb'))
    return dets


def box_feature_matrix(images, dets, gts):
    """收集候选框 CLIP 特征 + GT 标签 + 正例特征 (同 route_proxy_extend 循环)。

    注意: labels 按全部候选框收集 (含被过滤的小框), 必须用 idx 的 keep 掩码
    对齐到 crops 编码后的 F 行数 —— batch 化编码的经典索引陷阱。
    """
    crops, idx, labels = [], [], []
    for (boxes, scores), img, g in zip(dets, images, gts):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            continue
        for box in boxes[keep]:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            small = x2 <= x1 + 10 or y2 <= y1 + 10
            lab = 1 if any(U.iou([x1, y1, x2, y2], gt) >= 0.5 for gt in g) else 0
            if small:
                labels.append(lab); idx.append(None); continue
            crops.append(img.crop((x1, y1, x2, y2)))
            idx.append(len(crops) - 1)
            labels.append(lab)
    if not crops:
        return np.zeros((0, 512)), np.zeros(0), torch.zeros(512)
    F = encode_clip_batch(crops).cpu()
    keep = np.array([i is not None for i in idx])          # 对齐 F 行
    labels = np.array(labels)[keep]
    pos_real = (F[labels == 1].mean(dim=0) if (labels == 1).any()
                else torch.zeros(512))
    pos_real = pos_real / (pos_real.norm() + 1e-8)
    return F.numpy(), labels, pos_real


# ══════════════════════════ 主流程 ══════════════════════════

def main():
    # ── 0) B1 (dinov2 口径, 不重算) ──
    b1 = json.load(open(PROBE_12, encoding='utf-8'))

    # ── 1) DIOR 全部 20 类 CLIP 源原型 (负原型需要完整类集) ──
    print('[1/4] 构建 DIOR 20 类 CLIP 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        crops = []
        for img, g in zip(images, gts):
            for box in g:
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 > x1 + 10 and y2 > y1 + 10:
                    crops.append(img.crop((x1, y1, x2, y2)))
        if not crops:
            print(f'  !! {cls}: 无 GT 框, 跳过')
            continue
        F = encode_clip_batch(crops)
        p = F.mean(dim=0)
        protos[cls] = p / p.norm()
        print(f'  {cls:<20s} {len(crops):>4d} 框')

    # ── 2) 文本嵌入 + H1 域方向一致性预检查 ──
    print('\n[2/4] 文本嵌入 + 域方向一致性检查 ...')
    anchor_cls = ['airplane', 'ship', 'vehicle']   # 共享类, 检查 DeltaSpace 方向一致性
    t_src = torch.stack([text_avg(SRC_TMPLS, c) for c in anchor_cls]).cpu()
    t_tgt = torch.stack([text_avg(TGT_TMPLS, c) for c in anchor_cls]).cpu()
    dom_dir = t_tgt - t_src
    cos_mat = dom_dir @ dom_dir.T
    print('  域方向 d(x)-s(x) 余弦矩阵 (一致性 → 对角为正且非对角不高负):')
    for i, c in enumerate(anchor_cls):
        print(f'    {c:<10s} ' + ' '.join(f'{cos_mat[i,j]:+.3f}' for j in range(3)))
    ok = float(cos_mat[0, 1]) > -0.3 and float(cos_mat[0, 2]) > -0.3 and float(cos_mat[1, 2]) > -0.3
    print(f'  → DeltaSpace 方向一致性: {"✅ 通过" if ok else "❌ 冲突 (H1 直接判负)"}')

    # 全局域方向先验 D_dom: 20 类 (d(x)−s(x)) 单位化后平均 (外推前约束 'proj' 用)
    dom_src = torch.stack([text_avg(SRC_TMPLS, c) for c in U.DIOR_ID2NAME.values()])
    dom_tgt = torch.stack([text_avg(TGT_TMPLS, c) for c in U.DIOR_ID2NAME.values()])
    dom_dirs = dom_tgt - dom_src
    dom_dirs = dom_dirs / dom_dirs.norm(dim=-1, keepdim=True)
    D_dom = dom_dirs.mean(dim=0).cpu()
    D_dom = D_dom / D_dom.norm()

    # ── 3) COCO 类: 框池 + 特征 + 基线/外推 AUC ──
    print('\n[3/4] COCO 6 类探针 (N=%d) ...' % N_COCO)
    rows = {}
    for name, coco_cls, src_cls in TARGETS:
        print(f'\n  [{name}] ← DIOR {src_cls} 源原型')
        images, gts = U.load_coco(coco_cls, n=N_COCO)
        dets = cached_dets(images, f'{TGT_TMPL.format(coco_cls)}', name)
        F, labels, pos_real = box_feature_matrix(images, dets, gts)
        if len(F) == 0:
            print('  !! 无候选框, 跳过')
            continue
        n_pos = int(labels.sum())
        print(f'    N={len(images)}图 候选框={len(F)} 正例={n_pos}')

        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = get_neg_proto(protos, src_cls)

        # 文本偏移 (2026-08-22: 多模板平均)
        s_xs = text_avg(SRC_TMPLS, src_cls).cpu()
        d_y = text_avg(TGT_TMPLS, coco_cls).cpu()
        d_xs = text_avg(TGT_TMPLS, src_cls).cpu()
        Delta = d_y - s_xs
        # proj: Δ_joint 投影到 [D_dom, D_cat] 子空间 (外推前约束, 衰减残差 r)
        D_cat = d_y - d_xs
        D_cat_o = D_cat - float(D_cat @ D_dom) * D_dom
        D_cat_o = D_cat_o / (D_cat_o.norm() + 1e-8)
        deltas = {
            'joint': Delta,
            'cat':   D_cat,
            'dom':   d_xs - s_xs,
            'proj':  float(Delta @ D_dom) * D_dom + float(Delta @ D_cat_o) * D_cat_o,
        }

        def auc_of(P):
            vis = np.maximum(0.0, F @ P.cpu().numpy()
                             - U.NEG_LAM * (F @ P_neg.cpu().numpy()))
            return rank_auc(vis, labels)

        # B2: CLIP 真实原型 (关键对照: 换编码器增益)
        b2 = auc_of(P_src)
        # B3: 外推原型
        b3 = {v: {str(b): None for b in BETAS} for v in VARIANTS}
        for v in VARIANTS:
            for b in BETAS:
                p_hat = P_src + b * deltas[v].to(U.DEVICE)
                p_hat = p_hat / p_hat.norm()
                b3[v][str(b)] = round(auc_of(p_hat), 4)
        b3_best = {v: max(b3[v].values()) for v in VARIANTS}

        # H1: 外推原型 vs 源原型 vs 真实目标原型 (余弦, CLIP 空间)
        cos_src = float((P_src.cpu() @ pos_real).item())
        cos_hat = {}
        for v in VARIANTS:
            p_hat = P_src + 1.0 * deltas[v].to(U.DEVICE)
            p_hat = p_hat / p_hat.norm()
            cos_hat[v] = float((p_hat.cpu() @ pos_real).item())

        # 外推后矫正: 往返残差 (每变体 × 每 β, 纯源域自洽指标)
        roundtrip = {v: {str(b): round(roundtrip_residual(P_src, deltas[v], b), 4)
                         for b in BETAS} for v in VARIANTS}
        # 往返一致的最大 β (残差 ≤ 0.15 视为一致, 源域可得准则, 替代目标域 AUC 选择器)
        beta_max_ok = {v: max((b for b in BETAS
                               if roundtrip[v][str(b)] <= 0.15), default=None)
                       for v in VARIANTS}

        rows[name] = {
            'n_images': len(images), 'n_boxes': len(F), 'n_pos': n_pos,
            'B1_dinov2_auc': b1.get(name, {}).get('auc'),
            'B2_clip_auc': round(b2, 4),
            'B3_extrap': b3, 'B3_best': b3_best,
            'roundtrip': roundtrip, 'beta_max_ok': beta_max_ok,
            'cos_src_to_real': round(cos_src, 4),
            'cos_extrap_to_real': {v: round(cos_hat[v], 4) for v in VARIANTS},
        }
        print(f'    B1(dinov2)={b1.get(name,{}).get("auc"):.4f}  '
              f'B2(CLIP)={b2:.4f}  '
              f'B3_best joint={b3_best["joint"]:.4f} cat={b3_best["cat"]:.4f} '
              f'dom={b3_best["dom"]:.4f} proj={b3_best["proj"]:.4f}')
        print(f'    cos 源→真实={cos_src:+.3f}  '
              f'joint→真实={cos_hat["joint"]:+.3f} cat→真实={cos_hat["cat"]:+.3f} '
              f'dom→真实={cos_hat["dom"]:+.3f} proj→真实={cos_hat["proj"]:+.3f}')
        print(f'    往返一致最大β: joint={beta_max_ok["joint"]} cat={beta_max_ok["cat"]} '
              f'dom={beta_max_ok["dom"]} proj={beta_max_ok["proj"]} (残差≤0.15)')

    # ── 4) 汇总表 + 写盘 ──
    print('\n' + '=' * 100)
    hdr = (f'{"类":<16s} {"B1 dino":>8s} {"B2 clip":>8s} {"B3 joint":>8s} '
           f'{"B3 cat":>8s} {"B3 dom":>8s} {"B3 proj":>8s} {"提升vB2":>8s}')
    print(hdr)
    print('-' * len(hdr))
    for name in [t[0] for t in TARGETS]:
        r = rows.get(name)
        if not r:
            continue
        best = max(r['B3_best'].values())
        print(f'{name:<16s} {r["B1_dinov2_auc"]:>8.4f} {r["B2_clip_auc"]:>8.4f} '
              f'{r["B3_best"]["joint"]:>8.4f} {r["B3_best"]["cat"]:>8.4f} '
              f'{r["B3_best"]["dom"]:>8.4f} {r["B3_best"]["proj"]:>8.4f} '
              f'{best - r["B2_clip_auc"]:>+8.4f}')

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'n_coco': N_COCO, 'neg_lam': U.NEG_LAM,
                              'protocol': 'gdino th=0 -> pre-filter 0.03 -> rank-AUC (CLIP space)',
                              'beta': BETAS, 'variants': VARIANTS,
                              'src_tmpl': SRC_TMPL, 'tgt_tmpl': TGT_TMPL},
                   'H1_domain_dir_consistency': cos_mat.numpy().tolist(),
                   'classes': rows}, f, indent=2, ensure_ascii=False)
    print(f'\n✅ -> {OUT}')


if __name__ == '__main__':
    main()
