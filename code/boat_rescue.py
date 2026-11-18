# -*- coding: utf-8 -*-
"""boat_rescue.py — 抢救难类（v2）：把 v3 验证过的机制搬进统一 few-shot 协议

背景: cdvm_v3_results.json 里 boat 用 mah_ctx (上下文1.5x + 马氏) 到 0.3266,
car 用 mah_pos_neg (hard-neg 马氏) 到 0.3574 —— 但 v3 协议有数据泄露
(前20图既建原型又评估, 无留出测试, 无统一 pre-filter)。这里在严格协议下复测:
  前 K=10 张校准 (GT 框建正分布, GDINO 检出 IoU∈[0.1,0.5) 建负分布)
  后 90 张测试, 统一 pre-filter 0.03, 数据集级 AP50, 同一候选框集合

打分变体 (都是 vis = ..., fused = sqrt(gd × vis)):
  cosine      — 现状主表: f·P_cal, 负原型=DIOR类间 top-3 (对账锚点)
  mah_tight   — 校准集 GT 马氏相似度 (exp(-0.5·D²)), 无负项
  mah_ctx     — 上下文(bbox 1.5x)特征的马氏相似度
  mah_pn_tight— mah_tight − λ·mah_hardneg_tight
  mah_pn_ctx  — mah_ctx    − λ·mah_hardneg_ctx

对账: cosine 应复现 final_results.json 的 few_shot 值
  (airplane≈0.8015, boat≈0.2680, car≈0.3198)。
看 mah_ctx / mah_pn_ctx 在严格协议下 boat 能否回到 0.30+。
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
import unified_eval as U

LAYER = 11            # L12
K = 10
ALPHA = 0.5
NEG_LAM = 0.5
BG_IOU_LOW, BG_IOU_HIGH = 0.1, 0.5   # hard negative IoU 区间
SHRINKAGE = 0.3
CTX_EXPAND = 1.5
N_COCO = 100
OUT = r'C:\Users\张三\Desktop\实验数据\boat_rescue_v2.json'


# ── 原型 / 马氏 ──

def get_class_neg(bank, dior_cls, layer):
    pos = bank[dior_cls][layer]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][layer]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return neg / neg.norm()


def encode_l12(crop):
    dino, pre = U.get_dinov2()
    t = pre(crop).unsqueeze(0).to(U.DEVICE)
    with torch.no_grad():
        o = dino(t)
    f = o.hidden_states[LAYER + 1][0, 0, :]
    return f / f.norm()


def crop_feat(img, box):
    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2 <= x1 + 10 or y2 <= y1 + 10:
        return None
    return encode_l12(img.crop((x1, y1, x2, y2)))


def ctx_feat(img, box, expand=CTX_EXPAND):
    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
    x2, y2 = min(img.width, x2), min(img.height, y2)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = (x2 - x1) * expand, (y2 - y1) * expand
    nx1, ny1 = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
    nx2, ny2 = min(img.width, int(cx + nw / 2)), min(img.height, int(cy + nh / 2))
    if nx2 <= nx1 + 5 or ny2 <= ny1 + 5:
        return None
    return encode_l12(img.crop((nx1, ny1, nx2, ny2)))


def mah_params(feats, shrinkage=SHRINKAGE):
    """feats: (N,D) 归一化特征 → (mu, inv_cov)。样本少用收缩 + Cholesky。"""
    K_, D = feats.shape
    mu = feats.mean(dim=0)
    if K_ > 1:
        cov = (feats - mu).T @ (feats - mu) / (K_ - 1)
    else:
        cov = torch.eye(D, device=feats.device) * 1e-3
    cov_s = shrinkage * cov + (1 - shrinkage) * torch.eye(D, device=cov.device)
    cov_s = cov_s + 1e-4 * torch.eye(D, device=cov_s.device)
    try:
        L = torch.linalg.cholesky(cov_s)
        inv = torch.cholesky_inverse(L)
    except RuntimeError:
        inv = torch.eye(D, device=cov_s.device)
    return mu, inv


def mah_sim(f, mu, inv):
    d = (f - mu.to(f.device)).unsqueeze(0)
    m = d @ inv.to(f.device)
    sq = (m * d).sum(dim=1).clamp(min=0)
    return torch.exp(-0.5 * sq).item()


# ── 校准集: 正/负分布 + 目标域原型 ──

def build_pos_neg(fs_imgs, fs_gts, cls_prompt):
    """校准集: GT 框→正特征; GDINO 检出 IoU∈[0.1,0.5)→负特征。"""
    pos_feats, neg_feats = [], []
    dets = U.gdino_detect(fs_imgs, cls_prompt, 0.0)
    for (boxes, scores), img, gs in zip(dets, fs_imgs, fs_gts):
        for gt in gs:
            f = crop_feat(img, gt)
            if f is not None:
                pos_feats.append(f)
        keep = scores > U.PRE_FILTER
        for box in boxes[keep]:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if not gs:
                continue
            mi = max(U.iou([x1, y1, x2, y2], gt) for gt in gs)
            if BG_IOU_LOW <= mi < BG_IOU_HIGH:
                f = crop_feat(img, [x1, y1, x2, y2])
                if f is not None:
                    neg_feats.append(f)
    return pos_feats, neg_feats


def collect_feats(images, gts, dets):
    """测试集: 候选框 + GDINO 分数 + tight & ctx 特征 (一次编码, 复用)。"""
    boxes_all, gd_all, f_tight, f_ctx = [], [], [], []
    for (boxes, scores), img in zip(dets, images):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            continue
        for box, sc in zip(boxes[keep], scores[keep]):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            ft = crop_feat(img, [x1, y1, x2, y2])
            fc = ctx_feat(img, [x1, y1, x2, y2])
            if ft is None:
                continue
            boxes_all.append([x1, y1, x2, y2])
            gd_all.append(float(sc))
            f_tight.append(ft)
            f_ctx.append(fc if fc is not None else ft)
    gt_flat = [g for gs in gts for g in gs]
    return boxes_all, gd_all, f_tight, f_ctx, gt_flat


def ap_for(gd_all, vis_list, boxes_all, gt_flat):
    gd = np.clip(np.array(gd_all), 1e-6, 1.0)
    vs = np.clip(np.array(vis_list), 1e-6, 1.0)
    return U.compute_ap50(boxes_all, np.sqrt(gd * vs), gt_flat)


def main():
    bank = U.get_bank()
    summary = {}

    for name, dataset, cls, domain, prefix, dior_cls in U.EVALS:
        if dataset != 'coco':
            continue
        print('\n' + '=' * 84)
        print(f'[{name}]  严格 few-shot (K={K} 校准 / {N_COCO - K} 测试)')

        images, gts = U.load_coco(cls, n=N_COCO)
        dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)
        fs_imgs, fs_gts = images[:K], gts[:K]
        test_imgs, test_gts, test_dets = images[K:], gts[K:], dets[K:]

        P_src = bank[dior_cls][LAYER]['prototype'].to(U.DEVICE)
        pos_feats, neg_feats = build_pos_neg(fs_imgs, fs_gts, f'{prefix} {cls}')
        print(f'  校准: 正特征={len(pos_feats)}  hard-neg特征={len(neg_feats)}')

        # 目标域原型 + 马氏正分布
        P_tgt = torch.stack(pos_feats).mean(dim=0)
        P_tgt = P_tgt / P_tgt.norm()
        P_cal = (ALPHA * P_src + (1 - ALPHA) * P_tgt)
        P_cal = P_cal / P_cal.norm()
        pos_mu, pos_inv = mah_params(torch.stack(pos_feats))
        neg_mu, neg_inv = None, None
        if len(neg_feats) >= 3:
            neg_mu, neg_inv = mah_params(torch.stack(neg_feats))
        class_neg = get_class_neg(bank, dior_cls, LAYER)

        bx, gd_all, ft, fc, gt = collect_feats(test_imgs, test_gts, test_dets)

        gd_np = np.clip(np.array(gd_all), 1e-6, 1.0)
        base = U.compute_ap50(bx, gd_np, gt)
        print(f'  baseline = {base:.4f}  ({len(test_imgs)} 图, {len(bx)} 候选框)')

        # ── 打分变体 ──
        vis = {}
        # cosine (现状主表, 对账锚点)
        vis['cosine'] = [max(0.0, float((f @ P_cal).item())
                             - NEG_LAM * float((f @ class_neg).item())) for f in ft]
        # 纯余弦(无负原型)
        vis['cos_only'] = [max(0.0, float((f @ P_cal).item())) for f in ft]
        # 马氏 tight
        vis['mah_tight'] = [max(0.0, mah_sim(f, pos_mu, pos_inv)) for f in ft]
        # 马氏 ctx
        vis['mah_ctx'] = [max(0.0, mah_sim(f, pos_mu, pos_inv)) for f in fc]
        # 马氏 pos-neg tight / ctx
        if neg_mu is not None:
            vis['mah_pn_tight'] = [max(0.0, mah_sim(f, pos_mu, pos_inv)
                                       - NEG_LAM * mah_sim(f, neg_mu, neg_inv)) for f in ft]
            vis['mah_pn_ctx'] = [max(0.0, mah_sim(f, pos_mu, pos_inv)
                                     - NEG_LAM * mah_sim(f, neg_mu, neg_inv)) for f in fc]
        else:
            vis['mah_pn_tight'] = vis['mah_pn_ctx'] = None

        row = {'baseline': base, 'n_cal_pos': len(pos_feats), 'n_cal_neg': len(neg_feats),
               'src_tgt_cosine': float(P_src @ P_tgt)}
        order = ['cos_only', 'cosine', 'mah_tight', 'mah_ctx', 'mah_pn_tight', 'mah_pn_ctx']
        print(f'  {"变体":<14s} {"AP50":>8s} {"Δ":>8s}')
        print('  ' + '-' * 34)
        for k in order:
            v = vis[k]
            if v is None:
                continue
            ap = ap_for(gd_all, v, bx, gt)
            row[k] = ap
            mark = '  ← 对账锚点' if k == 'cosine' else ''
            print(f'  {k:<14s} {ap:8.4f} {ap - base:+8.4f}{mark}')

        best_k = max((k for k in order if row.get(k) is not None), key=lambda k: row[k])
        print(f'  最佳 = {best_k} → {row[best_k]:.4f} (Δ={row[best_k] - base:+.4f})')
        summary[name] = row

    # ── 汇总 ──
    print('\n\n' + '=' * 84)
    print(f'{"类":<16s} {"base":>7s} {"cosine":>8s} {"mah_ctx":>8s} {"mah_pn_ctx":>10s} 最佳')
    print('-' * 84)
    for name, r in summary.items():
        keys = [k for k in ['cosine', 'mah_ctx', 'mah_pn_ctx'] if k in r]
        best_k = max(keys, key=lambda k: r[k])
        line = f'{name:<16s} {r["baseline"]:7.4f}'
        for k in ['cosine', 'mah_ctx', 'mah_pn_ctx']:
            line += f' {r.get(k, float("nan")):8.4f}'
        line += f'  {best_k}={r[best_k]:.4f}'
        print(line)

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'layer': LAYER, 'k': K, 'alpha': ALPHA, 'neg_lam': NEG_LAM,
                              'shrinkage': SHRINKAGE, 'ctx_expand': CTX_EXPAND,
                              'neg_iou_range': [BG_IOU_LOW, BG_IOU_HIGH], 'n_coco': N_COCO,
                              'protocol': 'strict: K calib / rest test, pre-filter 0.03, dataset AP50'},
                   'per_class': summary}, f, indent=2, ensure_ascii=False)
    print(f'\n✅ -> {OUT}')


if __name__ == '__main__':
    main()
