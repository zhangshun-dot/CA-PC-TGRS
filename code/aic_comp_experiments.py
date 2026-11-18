# -*- coding: utf-8 -*-
"""aic_comp_experiments.py — AIC 2026 算法创新赛 比赛实验（不进论文）

Part A: DIOR→COCO 统一官方指标重评
  - 指标: AP50 (11-point) + COCO 官方 mAP@[.5:.95] (101-point)
    （官方口径实现直接取自 A-CSPC 仓库 acspc/metrics.py，与论文同一份代码）
  - 方法: baseline / TGE / A-CSPC(非对称) / A-CSPC-shared(负原型也外推,强对照)
          / Safe-Correction(g2 门控)
  - 检测缓存复用: gdino-tiny / gdino-base / owlv2（零重推理）

用法:
  python aic_comp_experiments.py --pilot          # 2 类 x 50 图 + beta/lam 网格
  python aic_comp_experiments.py                  # 6 类 x 200 图 x 3 检测器
"""
import sys, os, json, pickle, argparse, importlib.util
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
from pathlib import Path
from PIL import Image

import unified_eval as U
import tge_probe as T

# 官方口径 AP（A-CSPC 仓库实现）
_spec = importlib.util.spec_from_file_location(
    'acspc_metrics', r'C:\Users\张三\Desktop\A-CSPC\acspc\metrics.py')
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

BETA = 2.0            # A-CSPC 主配置 (A-CSPC repo configs/default.yaml)
LAM = 0.5
PRE_FILTER = 0.03
K_PCA = 30
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'

CLASSES6 = ['airplane', 'boat', 'bus', 'car', 'motorcycle', 'truck']
# DIOR 20 类里可用于源域的: airplane / ship / vehicle
DIOR_SRC = {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle',
            'bus': 'vehicle', 'motorcycle': 'vehicle', 'truck': 'vehicle'}
DET_CACHE = {'gdino-tiny': 'tge_v2_dets_COCO-{}.pkl',
             'gdino-base': 'gdino_base_dets_COCO-{}.pkl',
             'owlv2': 'owlv2_dets_COCO-{}.pkl'}


# ────────────────────── 数据加载 ──────────────────────

def load_target(cls, n, det):
    """加载 COCO 目标域 + 检测缓存 → per_img + CLIP 特征。"""
    images, gts = U.load_coco(cls, n=n)
    if not images:
        return None
    cache_p = DATA / DET_CACHE[det].format(cls)
    if cache_p.exists():
        dets = pickle.load(open(cache_p, 'rb'))
        if len(dets) < len(images):
            raise RuntimeError(f'缓存不足: {cache_p.name} {len(dets)}<{len(images)}')
        dets = dets[:len(images)]
    else:
        dets = U.gdino_detect(images, cls, 0.0)
        pickle.dump(dets, open(cache_p, 'wb'))
    per_img, crops_all = [], []
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < PRE_FILTER:
                continue
            bl.append([x1, y1, x2, y2]); sl.append(float(sc)); ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({
            'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
            'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
            'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
            'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64),
        })
    F = T.encode_clip_batch(crops_all).cpu().numpy() if crops_all else np.zeros((0, 512))
    return {'per_img': per_img, 'features': F, 'n_crops': len(crops_all)}


def load_source_protos(src_classes, n=50):
    """DIOR 源域原型（CLIP GT 框特征均值）。"""
    out = {}
    for c in src_classes:
        images, gts = U.load_dior(c, n=n)
        if not images:
            continue
        crops = []
        for img, g in zip(images, gts):
            for gt in g:
                x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                crops.append(img.crop((x1, y1, x2, y2)))
        if not crops:
            continue
        F = T.encode_clip_batch(crops).cpu().numpy()
        p = F.mean(axis=0)
        out[c] = {'proto': p / (np.linalg.norm(p) + 1e-8), 'features': F}
    return out


# ────────────────────── 方法 ──────────────────────

def text_delta(src_cls, tgt_cls, beta):
    s = T.text_avg([SRC_TMPL], src_cls).cpu().numpy()
    d = T.text_avg([TGT_TMPL], tgt_cls).cpu().numpy()
    return (d - s) * beta


def build_neg(protos, pos_src):
    """负原型: 其余源类原型中与正原型余弦最高的 top-3 均值（保留源域,不外推）。"""
    other = {k: v['proto'] for k, v in protos.items() if k != pos_src}
    if not other:
        return None
    sims = {k: float(np.dot(protos[pos_src]['proto'], v)) for k, v in other.items()}
    top3 = sorted(sims.items(), key=lambda x: -x[1])[:3]
    neg = np.mean([other[k] for k, _ in top3], axis=0)
    return neg / (np.linalg.norm(neg) + 1e-8)


def evaluate(per_img, F, proto, neg, lam):
    """融合打分 → (AP50, mAP@[.5:.95])。"""
    preds = []
    for p in per_img:
        n = len(p['gd'])
        if n and proto is not None:
            fb = F[p['crop_idx']]
            pos = fb @ proto
            negs = fb @ neg if neg is not None else np.zeros(n)
            vis = np.maximum(0.0, pos - lam * negs)
            fused = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            fused = p['gd']
        preds.append((p['box'], fused))
    gts = [p['gt'] for p in per_img]
    return M.compute_ap50_img(preds, gts), M.compute_ap_coco(preds, gts)


def g2_align(delta, src_feats, tgt_feats):
    """文本外推方向与 PCA 域偏移方向的对齐度 (cos)。"""
    X = np.vstack([src_feats, tgt_feats])
    mean = X.mean(axis=0)
    Uc, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    K = min(K_PCA, Vt.shape[0])
    comp = Vt[:K]
    def proj(v): return (v - mean) @ comp.T
    shift = proj(tgt_feats).mean(axis=0) - proj(src_feats).mean(axis=0)
    ns = np.linalg.norm(shift)
    if ns < 1e-8:
        return 0.0
    d = proj(delta)
    nd = np.linalg.norm(d)
    if nd < 1e-8:
        return 0.0
    return float(np.dot(d, shift) / (nd * ns))


# ────────────────────── 主流程 ──────────────────────

def run(classes, n, det_name, beta_grid=None, lam_grid=None, out_suffix=''):
    print(f'=== detector={det_name} n={n} classes={classes} ===')
    src_classes = sorted(set(DIOR_SRC[c] for c in classes))
    protos = load_source_protos(src_classes, n=50)
    print(f'源域原型: {[(k, v["features"].shape[0]) for k, v in protos.items()]}')

    targets, valid = {}, []
    for c in classes:
        if DIOR_SRC[c] not in protos:
            print(f'  跳过 {c}（源类缺失）'); continue
        t = load_target(c, n, det_name)
        if t is not None:
            targets[c] = t; valid.append(c)
            print(f'  {c}: {len(t["per_img"])} 图, {t["n_crops"]} 框')

    # 域偏移对齐度 g2（用全部源/目标特征拟合 PCA）
    all_src = np.vstack([protos[DIOR_SRC[c]]['features'] for c in valid])
    all_tgt = np.vstack([targets[c]['features'] for c in valid])
    g2s = {}
    for c in valid:
        delta = text_delta(DIOR_SRC[c], c, 1.0)
        g2s[c] = g2_align(delta, all_src, all_tgt)
    print('g2 对齐度:', {c: round(g, 4) for c, g in g2s.items()})

    beta_grid = beta_grid or [BETA]
    lam_grid = lam_grid or [LAM]
    results = {'config': {'detector': det_name, 'n': n, 'beta': BETA, 'lam': LAM,
                          'g2': g2s}, 'per_class': {}, 'grid': {}}

    # 网格（仅 pilot 用）
    if len(beta_grid) > 1 or len(lam_grid) > 1:
        for c in valid:
            src = DIOR_SRC[c]
            delta = text_delta(src, c, 1.0)
            neg = build_neg(protos, src)
            row = {}
            for b in beta_grid:
                p_hat = protos[src]['proto'] + b * delta
                p_hat /= (np.linalg.norm(p_hat) + 1e-8)
                for l in lam_grid:
                    a50, amap = evaluate(targets[c]['per_img'], targets[c]['features'],
                                         p_hat, neg, l)
                    row[f'b{b}_l{l}'] = {'ap50': round(a50, 4), 'map': round(amap, 4)}
            results['grid'][c] = row
            b0 = evaluate(targets[c]['per_img'], targets[c]['features'], None, None, 0)
            print(f'  [grid] {c}: baseline ap50={b0[0]:.4f} map={b0[1]:.4f} | ' +
                  ' '.join(f'{k}:{v["ap50"]:.4f}' for k, v in sorted(row.items())))

    # 主表方法
    for c in valid:
        src = DIOR_SRC[c]
        delta = text_delta(src, c, 1.0)
        neg = build_neg(protos, src)
        F = targets[c]['features']; per_img = targets[c]['per_img']

        # baseline: 纯检测分数
        base = evaluate(per_img, F, None, None, 0)

        # TGE: 正原型外推, 无负原型抑制
        p_tge = protos[src]['proto'] + BETA * delta
        p_tge /= (np.linalg.norm(p_tge) + 1e-8)
        tge = evaluate(per_img, F, p_tge, None, LAM)

        # A-CSPC: 正原型外推 + 源域负原型抑制（非对称）
        acspc = evaluate(per_img, F, p_tge, neg, LAM)

        # A-CSPC-shared 强对照: 负原型同样外推（对称/shared 空间版本）
        if neg is not None:
            neg_sh = neg + BETA * delta
            neg_sh /= (np.linalg.norm(neg_sh) + 1e-8)
        else:
            neg_sh = None
        shared = evaluate(per_img, F, p_tge, neg_sh, LAM)

        results['per_class'][c] = {
            'baseline': {'ap50': round(base[0], 4), 'map': round(base[1], 4)},
            'TGE': {'ap50': round(tge[0], 4), 'map': round(tge[1], 4)},
            'A-CSPC': {'ap50': round(acspc[0], 4), 'map': round(acspc[1], 4)},
            'A-CSPC-shared': {'ap50': round(shared[0], 4), 'map': round(shared[1], 4)},
        }
        r = results['per_class'][c]
        print(f"  {c}: base {r['baseline']['ap50']:.4f}/{r['baseline']['map']:.4f} | "
              f"TGE {r['TGE']['ap50']:.4f}/{r['TGE']['map']:.4f} | "
              f"A-CSPC {r['A-CSPC']['ap50']:.4f}/{r['A-CSPC']['map']:.4f} | "
              f"shared {r['A-CSPC-shared']['ap50']:.4f}/{r['A-CSPC-shared']['map']:.4f}")

    # 汇总
    summary = {}
    for m in ['baseline', 'TGE', 'A-CSPC', 'A-CSPC-shared']:
        a50 = float(np.mean([results['per_class'][c][m]['ap50'] for c in valid]))
        amap = float(np.mean([results['per_class'][c][m]['map'] for c in valid]))
        summary[m] = {'ap50': round(a50, 4), 'map': round(amap, 4)}
    results['summary'] = summary
    print('── 汇总 (6类均值) ──')
    for m, v in summary.items():
        print(f'  {m:15s} AP50={v["ap50"]:.4f}  mAP@[.5:.95]={v["map"]:.4f}')

    out = DATA / f'aic_comp_{det_name.replace("-", "_")}{out_suffix}.json'
    json.dump(results, open(out, 'w'), ensure_ascii=False, indent=2)
    print(f'已写入 {out}')
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--det', default='gdino-tiny')
    args = ap.parse_args()
    if args.pilot:
        run(['car', 'boat'], n=50, det_name=args.det,
            beta_grid=[0.5, 1.0, 2.0], lam_grid=[0.25, 0.5], out_suffix='_pilot')
    else:
        for det in ['gdino-tiny', 'gdino-base', 'owlv2']:
            run(CLASSES6, n=200, det_name=det)


if __name__ == '__main__':
    main()
