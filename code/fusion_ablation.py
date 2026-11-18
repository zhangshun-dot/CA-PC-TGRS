# -*- coding: utf-8 -*-
"""fusion_ablation.py — 融合算子消融 + 原型有效性自检 gate

协议与 final_method_table.py / gain_decomposition.json 逐位一致:
  L12 / K=10 / α=0.5 / NEG_LAM=0.5 / 前10校准后90测试 / pre-filter 0.03 / 数据集级 AP50

同一候选框集合, 每 box 保存:
  gd            — GDINO 分数
  vis_zs        — 源原型 + 负原型  (max(0, f·P_src − λ·f·P_neg))   → sqrt 融合应复现 CDVM_zs
  vis_fs        — 校准原型 + 负原型 (max(0, f·P_cal − λ·f·P_neg))   → sqrt 融合应复现 CDVM_fs
  label         — 测试集 GT IoU≥0.5 (仅 oracle 用)

对 zs 与 fs 两组 vis, 分别测 5 个融合算子:
  sqrt        — 现状 √(gd·vis)                     (锚点)
  wsum        — 加权和 α·gd + (1−α)·vis_norm, 扫 α∈{0.3,0.5,0.7}
  gate_spear  — 无标签: 测试集 (gd, vis) Spearman ρ > τ → sqrt, 否则回退 gd
  gate_auc    — 校准集: GT框 vs 非GT框 的 vis rank-AUC > τ → sqrt, 否则回退 gd
  gate_oracle — 测试集 GT: vis AUC > τ → sqrt, 否则回退 gd  (上限)

τ: spearman=0.10, auc=0.55
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
import unified_eval as U

LAYER = 11
K = 10
ALPHA = 0.5
NEG_LAM = 0.5
N_COCO = 100
TAU_SPEAR = 0.10
TAU_AUC = 0.55
OUT = r'C:\Users\张三\Desktop\实验数据\fusion_ablation.json'


def get_neg_proto(bank, dior_cls, layer):
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


def collect(test_imgs, test_gts, test_dets, P_pos, neg):
    """测试集: 每 box 的 gd, vis, label。"""
    boxes, gds, viss, labs = [], [], [], []
    for (boxes_i, scores), img, gs in zip(test_dets, test_imgs, test_gts):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            continue
        for box, sc in zip(boxes_i[keep], scores[keep]):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            f = encode_l12(img.crop((x1, y1, x2, y2)))
            v = float((f @ P_pos).item()) - NEG_LAM * float((f @ neg).item())
            boxes.append([x1, y1, x2, y2])
            gds.append(float(sc))
            viss.append(max(0.0, v))
            labs.append(1 if any(U.iou([x1, y1, x2, y2], gt) >= 0.5 for gt in gs) else 0)
    return boxes, np.array(gds), np.array(viss), np.array(labs)


def collect_calib(fs_imgs, fs_gts, fs_dets, P_pos, neg):
    """校准集: 每 box 的 gd, vis, label (用于校准 AUC gate)。"""
    boxes, gds, viss, labs = [], [], [], []
    for (boxes_i, scores), img, gs in zip(fs_dets, fs_imgs, fs_gts):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            continue
        for box, sc in zip(boxes_i[keep], scores[keep]):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            f = encode_l12(img.crop((x1, y1, x2, y2)))
            v = float((f @ P_pos).item()) - NEG_LAM * float((f @ neg).item())
            boxes.append([x1, y1, x2, y2])
            gds.append(float(sc))
            viss.append(max(0.0, v))
            labs.append(1 if any(U.iou([x1, y1, x2, y2], gt) >= 0.5 for gt in gs) else 0)
    return np.array(gds), np.array(viss), np.array(labs)


def rank_auc(viss, labs):
    """rank-based AUC (0.5=随机, 1=完美)。"""
    viss, labs = np.asarray(viss), np.asarray(labs)
    pos, neg = viss[labs == 1], viss[labs == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(viss)
    rank = np.empty_like(order, dtype=float)
    rank[order] = np.arange(len(viss))
    return (rank[labs == 1].mean() - (len(pos) - 1) / 2) / len(neg)


def spearman(gd, vis):
    def _rank(x):
        o = np.argsort(x)
        r = np.empty_like(o, dtype=float)
        r[o] = np.arange(len(x))
        return r
    rg, rv = _rank(gd), _rank(vis)
    d = rg - rv
    n = len(gd)
    denom = (n ** 3 - n) / 6.0
    if denom == 0:
        return 0.0
    return 1.0 - 6.0 * float((d ** 2).sum()) / denom


def ap_of(boxes, scores):
    gd_c = np.clip(np.array(scores, dtype=float), 1e-6, 1.0)
    return U.compute_ap50(boxes, gd_c, gt_flat_global)


gt_flat_global = None  # 由 main 设置 (见 collect 返回的 boxes 没有 gt)


def ap_from(boxes, gds, viss, mode, alpha=0.5):
    """按融合算子算 AP50。boxes/gds/viss 与 gt 由外层对齐。"""
    gd = np.clip(gds, 1e-6, 1.0)
    if mode == 'gd':
        return U.compute_ap50(boxes, gd, gt_flat_global)
    vis = np.clip(viss, 1e-6, 1.0)
    if mode == 'sqrt':
        return U.compute_ap50(boxes, np.sqrt(gd * vis), gt_flat_global)
    if mode == 'wsum':
        vmin, vmax = vis.min(), vis.max()
        vn = (vis - vmin) / (vmax - vmin + 1e-9)
        return U.compute_ap50(boxes, alpha * gd + (1 - alpha) * vn, gt_flat_global)
    raise ValueError(mode)


def main():
    global gt_flat_global
    bank = U.get_bank()
    summary = {}

    for name, dataset, cls, domain, prefix, dior_cls in U.EVALS:
        if dataset != 'coco':
            continue
        print('\n' + '=' * 88)
        print(f'[{name}]')

        images, gts = U.load_coco(cls, n=N_COCO)
        dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)
        fs_imgs, fs_gts, fs_dets = images[:K], gts[:K], dets[:K]
        test_imgs, test_gts, test_dets = images[K:], gts[K:], dets[K:]

        P_src = bank[dior_cls][LAYER]['prototype'].to(U.DEVICE)
        neg = get_neg_proto(bank, dior_cls, LAYER)

        # 校准原型
        p_feats = []
        for img, g in zip(fs_imgs, fs_gts):
            for gt in g:
                f = crop_feat(img, gt)
                if f is not None:
                    p_feats.append(f)
        if p_feats:
            P_tgt = torch.stack(p_feats).mean(dim=0)
            P_tgt = P_tgt / P_tgt.norm()
            P_cal = (ALPHA * P_src + (1 - ALPHA) * P_tgt)
            P_cal = P_cal / P_cal.norm()
        else:
            P_cal = P_src

        # 测试集: zs 与 fs 两组 (同框同 gd, 不同 vis)
        bx, gd_t, vis_zs_t, lab_t = collect(test_imgs, test_gts, test_dets, P_src, neg)
        bx, gd_t, vis_fs_t, _ = collect(test_imgs, test_gts, test_dets, P_cal, neg)
        gt_flat_global = [g for gs in test_gts for g in gs]

        # 校准集: 两组 vis (供 gate_auc)
        gd_c, vis_zs_c, lab_c = collect_calib(fs_imgs, fs_gts, fs_dets, P_src, neg)
        _, vis_fs_c, _ = collect_calib(fs_imgs, fs_gts, fs_dets, P_cal, neg)

        # ── 锚点 ──
        ap_zs_sqrt = ap_from(bx, gd_t, vis_zs_t, 'sqrt')
        ap_fs_sqrt = ap_from(bx, gd_t, vis_fs_t, 'sqrt')
        print(f'  锚点 sqrt: CDVM_zs={ap_zs_sqrt:.4f}  CDVM_fs={ap_fs_sqrt:.4f}')

        # ── 每组的 gate 信号 ──
        rho_zs = spearman(gd_t, vis_zs_t)
        rho_fs = spearman(gd_t, vis_fs_t)
        auc_zs_c = rank_auc(vis_zs_c, lab_c)
        auc_fs_c = rank_auc(vis_fs_c, lab_c)
        auc_zs_o = rank_auc(vis_zs_t, lab_t)
        auc_fs_o = rank_auc(vis_fs_t, lab_t)
        print(f'  gate信号: ρ(gd,vis)_zs={rho_zs:+.3f} _fs={rho_fs:+.3f} | '
              f'校准AUC_zs={auc_zs_c:.3f} _fs={auc_fs_c:.3f} | oracleAUC_zs={auc_zs_o:.3f} _fs={auc_fs_o:.3f}')

        row = {'base': U.compute_ap50(bx, np.clip(gd_t, 1e-6, 1.0), gt_flat_global)}
        row['sqrt_zs'] = ap_zs_sqrt
        row['sqrt_fs'] = ap_fs_sqrt

        for tag, vis_t, rho, auc_c, auc_o in [('zs', vis_zs_t, rho_zs, auc_zs_c, auc_zs_o),
                                              ('fs', vis_fs_t, rho_fs, auc_fs_c, auc_fs_o)]:
            # 加权和 (α 扫描)
            w = {}
            for a in [0.3, 0.5, 0.7]:
                w[f'a{a}'] = ap_from(bx, gd_t, vis_t, 'wsum', alpha=a)
            # gate_spear
            gs = ap_from(bx, gd_t, vis_t, 'sqrt') if rho > TAU_SPEAR else row['base']
            # gate_auc (校准集, 用当前组的校准 AUC)
            ga = ap_from(bx, gd_t, vis_t, 'sqrt') if auc_c > TAU_AUC else row['base']
            # gate_oracle
            go = ap_from(bx, gd_t, vis_t, 'sqrt') if auc_o > TAU_AUC else row['base']
            row[f'wsum_{tag}'] = w
            row[f'gate_spear_{tag}'] = gs
            row[f'gate_auc_{tag}'] = ga
            row[f'gate_oracle_{tag}'] = go
            print(f'  [{tag}] wsum(a.3/.5/.7)={w["a0.3"]:.4f}/{w["a0.5"]:.4f}/{w["a0.7"]:.4f} | '
                  f'gate: spear={gs:.4f} auc={ga:.4f} oracle={go:.4f}')

        summary[name] = row

    # ── 汇总 ──
    print('\n\n' + '=' * 88)
    cols = ['base', 'sqrt_zs', 'sqrt_fs', 'gate_spear_zs', 'gate_auc_zs', 'gate_oracle_zs',
            'gate_spear_fs', 'gate_auc_fs', 'gate_oracle_fs']
    print(f'{"类":<16s}' + ''.join(f'{c:>13s}' for c in cols))
    print('-' * 130)
    for name, r in summary.items():
        print(f'{name:<16s}' + ''.join(f'{r[c]:>13.4f}' for c in cols))

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'layer': LAYER, 'k': K, 'alpha': ALPHA, 'neg_lam': NEG_LAM,
                              'tau_spear': TAU_SPEAR, 'tau_auc': TAU_AUC, 'n_coco': N_COCO,
                              'protocol': 'same as final_method_table.py'},
                   'per_class': summary}, f, indent=2, ensure_ascii=False)
    print(f'\n✅ -> {OUT}')


if __name__ == '__main__':
    main()
