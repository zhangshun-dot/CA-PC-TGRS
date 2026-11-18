# -*- coding: utf-8 -*-
"""aic_visdrone_experiment.py — AIC 比赛实验 Part B/C
低空无人机独立目标域验证: DIOR(卫星) → VisDrone(无人机低空)
方法: baseline / TGE / A-CSPC(非对称) / A-CSPC-shared(强对照) / Safe-Correction(g2 门控)
指标: AP50 (11-point) + COCO 官方 mAP@[.5:.95] (101-point)

用法:
  python aic_visdrone_experiment.py            # 检测(带缓存) + 评测
  python aic_visdrone_experiment.py --viz      # 额外输出 demo 可视化
"""
import sys, os, json, pickle, argparse, importlib.util
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

import unified_eval as U
import tge_probe as T

_spec = importlib.util.spec_from_file_location(
    'acspc_metrics', r'C:\Users\张三\Desktop\A-CSPC\acspc\metrics.py')
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
VISDRONE = Path(r'C:\Users\张三\Downloads\VisDrone2019-DET-val\VisDrone2019-DET-val')

BETA = 2.0
LAM = 0.5
PRE_FILTER = 0.03
K_PCA = 30
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'

# VisDrone cat_id → COCO 类名（DIOR 源域一律用 vehicle 作地面载具源类）
VIS_CLS = {'car': 4, 'truck': 6, 'bus': 9, 'motorcycle': 10}
SRC_BANK = ['airplane', 'ship', 'vehicle']   # DIOR 源类（vehicle 为所有地面载具的源类）
SRC_OF = {c: 'vehicle' for c in VIS_CLS}


# ─────────────────── 数据 ───────────────────

def load_visdrone_gt():
    """每类: [(fname, gt_boxes_xyxy)]，仅含含该类 GT 的图。"""
    out = {c: [] for c in VIS_CLS}
    for ann in sorted((VISDRONE / 'annotations').glob('*.txt')):
        img_p = VISDRONE / 'images' / (ann.stem + '.jpg')
        if not img_p.exists():
            continue
        per_cls = {c: [] for c in VIS_CLS}
        for line in ann.read_text(encoding='utf-8', errors='ignore').splitlines():
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            try:
                x, y, w, h = map(float, parts[:4]); cat = int(parts[5])
            except ValueError:
                continue
            if w <= 2 or h <= 2:
                continue
            for c, cid in VIS_CLS.items():
                if cat == cid:
                    per_cls[c].append([x, y, x + w, y + h])
        for c, boxes in per_cls.items():
            if boxes:
                out[c].append((img_p, boxes))
    return out


def load_gt_crops(entries, n_src_img=50):
    """源域 GT 框裁剪 → CLIP 特征。"""
    crops = []
    for img_p, boxes in entries[:n_src_img]:
        img = Image.open(img_p).convert('RGB')
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(img.width, int(x2)), min(img.height, int(y2))
            if x2 <= x1 + 4 or y2 <= y1 + 4:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
    if not crops:
        return None
    F = T.encode_clip_batch(crops).cpu().numpy()
    return F


def detect_visdrone(entries, prompt):
    """对一批 VisDrone 图检测，带缓存。"""
    key = prompt.replace(' ', '_').replace('{}', 'CLS')
    cache_p = DATA / f'aic_visdrone_dets_{key}.pkl'
    if cache_p.exists():
        return pickle.load(open(cache_p, 'rb'))
    dets = []
    for i, (img_p, _) in enumerate(entries):
        img = Image.open(img_p).convert('RGB')
        d = U.gdino_detect([img], prompt, 0.0)[0]
        dets.append((d[0], d[1]))
        if (i + 1) % 20 == 0:
            print(f'    检测进度 {i+1}/{len(entries)}', flush=True)
    pickle.dump(dets, open(cache_p, 'wb'))
    return dets


def build_target(entries, dets):
    per_img, crops_all = [], []
    for (boxes, scores), (img_p, gt) in zip(dets, entries):
        img = Image.open(img_p).convert('RGB')
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 4 or y2 <= y1 + 4:
                continue
            if float(sc) < PRE_FILTER:
                continue
            bl.append([x1, y1, x2, y2]); sl.append(float(sc)); ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({
            'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
            'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
            'gt': np.array(gt, dtype=np.float64),
            'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64),
        })
    F = T.encode_clip_batch(crops_all).cpu().numpy() if crops_all else np.zeros((0, 512))
    # 预计算每图 IoU 矩阵（框与 GT 不随方法变化，全部方法/阈值共用）
    for p in per_img:
        p['iou'] = iou_mat(p['box'], p['gt']) if len(p['box']) and len(p['gt']) else None
    return {'per_img': per_img, 'features': F, 'n_crops': len(crops_all)}


def iou_mat(b, g):
    """向量化 IoU: b (N,4) xyxy, g (G,4) xyxy → (N,G)。"""
    N, G = len(b), len(g)
    bx1, by1, bx2, by2 = b[:, 0:1], b[:, 1:2], b[:, 2:3], b[:, 3:4]
    gx1, gy1, gx2, gy2 = g[:, 0], g[:, 1], g[:, 2], g[:, 3]
    iw = np.maximum(0, np.minimum(bx2, gx2) - np.maximum(bx1, gx1))
    ih = np.maximum(0, np.minimum(by2, gy2) - np.maximum(by1, gy1))
    inter = iw * ih
    ab = (bx2 - bx1) * (by2 - by1)
    ag = (gx2 - gx1) * (gy2 - gy1)
    return inter / np.maximum(ab + ag - inter, 1e-9)


# ─────────────────── 评测（快速版，与 A-CSPC metrics 同算法） ───────────────────

def _ap_at_iou_fast(target, fused_list, iou_th):
    n_gt = sum(len(p['gt']) for p in target['per_img'])
    evals = []
    for p, fused in zip(target['per_img'], fused_list):
        if p['iou'] is None or len(fused) == 0:
            if len(fused):
                evals.extend([(float(s), 0) for s in fused])
            continue
        order = np.argsort(-fused)
        ious = p['iou'][order]
        matched = np.zeros(ious.shape[1], dtype=bool)
        for i in range(ious.shape[0]):
            row = ious[i]
            cand = np.where(~matched, row, -1.0)
            j = int(cand.argmax())
            if cand[j] >= iou_th:
                evals.append((float(fused[order[i]]), 1)); matched[j] = True
            else:
                evals.append((float(fused[order[i]]), 0))
    if not evals:
        return 0.0
    evals.sort(key=lambda x: -x[0])
    tp = np.array([e[1] for e in evals])
    tc = np.cumsum(tp); fc = np.cumsum(1 - tp)
    rec = tc / max(n_gt, 1)
    prec = tc / (tc + fc + 1e-8)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        m = rec >= t
        if m.any():
            ap += float(prec[m].max())
    return ap / 101.0


def ap50_fast(target, fused_list):
    return _ap_at_iou_fast(target, fused_list, 0.5)


def map_coco_fast(target, fused_list):
    ths = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    return float(np.mean([_ap_at_iou_fast(target, fused_list, th) for th in ths]))


def eval_both(target, fused_list):
    """一次算两个口径（101-point 循环复用 10 个 IoU 阈值，AP50 即其中之一）。"""
    ths = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    aps = {th: _ap_at_iou_fast(target, fused_list, th) for th in ths}
    return aps[0.5], float(np.mean(list(aps.values())))


# ─────────────────── 方法 ───────────────────

def text_delta(src_cls, tgt_cls):
    s = T.text_avg([SRC_TMPL], src_cls).cpu().numpy()
    d = T.text_avg([TGT_TMPL], tgt_cls).cpu().numpy()
    return d - s


def img_ap(p, fused, iou_th):
    """单图 AP（复用快速版，n_gt=该图 GT 数）。"""
    return _ap_at_iou_fast({'per_img': [p]}, [fused], iou_th)


def bootstrap(target, base_f, acspc_f, R=2000):
    """图像级 bootstrap: A-CSPC vs baseline 增益的 95% CI（AP50 与 mAP）。"""
    n = len(target['per_img'])
    per_img = target['per_img']
    base_im = np.array([img_ap(p, f, 0.5) for p, f in zip(per_img, base_f)])
    acspc_im = np.array([img_ap(p, f, 0.5) for p, f in zip(per_img, acspc_f)])
    gain = acspc_im - base_im

    # 逐图 mAP: 形状 (10 阈值, n 图)
    ths = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    base_map_im = np.stack([np.array([img_ap(p, f, th) for p, f in zip(per_img, base_f)])
                            for th in ths], axis=0)
    acspc_map_im = np.stack([np.array([img_ap(p, f, th) for p, f in zip(per_img, acspc_f)])
                             for th in ths], axis=0)
    gain_map = acspc_map_im - base_map_im  # (10, n)

    rng = np.random.default_rng(42)
    out = {}
    # AP50
    mg50 = np.array([gain[rng.integers(0, n, n)].mean() for _ in range(R)])
    lo, hi = float(np.percentile(mg50, 2.5)), float(np.percentile(mg50, 97.5))
    out['ap50'] = {'mean_gain': round(float(gain.mean()), 4),
                   'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4),
                   'significant': bool((lo > 0) or (hi < 0))}
    # mAP：每图对所有阈值求平均后再跨图 bootstrap
    gain_map_mean = gain_map.mean(axis=0)  # (n,)
    mgm = np.array([gain_map_mean[rng.integers(0, n, n)].mean() for _ in range(R)])
    lo, hi = float(np.percentile(mgm, 2.5)), float(np.percentile(mgm, 97.5))
    out['map'] = {'mean_gain': round(float(gain_map_mean.mean()), 4),
                  'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4),
                  'significant': bool((lo > 0) or (hi < 0))}
    out['n_img'] = n
    return out


def fuse_scores(target, proto, neg, lam):
    fused_list = []
    F = target['features']
    for p in target['per_img']:
        n = len(p['gd'])
        if n and proto is not None:
            fb = F[p['crop_idx']]
            pos = fb @ proto
            negs = fb @ neg if neg is not None else np.zeros(n)
            vis = np.maximum(0.0, pos - lam * negs)
            fused_list.append(np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0)))
        else:
            fused_list.append(p['gd'])
    return fused_list


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--viz', action='store_true')
    ap_.add_argument('--bootstrap', action='store_true')
    args = ap_.parse_args()

    print('=== 源域 DIOR 原型 ===')
    src = {}
    for c in SRC_BANK:
        images, gts = U.load_dior(c, n=50)
        crops = []
        for img, g in zip(images, gts):
            for gt in g:
                x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 4 or y2 <= y1 + 4:
                    continue
                crops.append(img.crop((x1, y1, x2, y2)))
        F = T.encode_clip_batch(crops).cpu().numpy()
        p = F.mean(axis=0)
        src[c] = {'proto': p / (np.linalg.norm(p) + 1e-8), 'features': F}
        print(f'  {c}: {len(crops)} crops')

    print('=== VisDrone 目标域 ===')
    gt_all = load_visdrone_gt()
    for c in VIS_CLS:
        print(f'  {c}: {len(gt_all[c])} 图')

    results = {'config': {'beta': BETA, 'lam': LAM, 'pre_filter': PRE_FILTER,
                          'src_tmpl': SRC_TMPL, 'tgt_tmpl': TGT_TMPL},
               'per_class': {}}

    all_src_feats = np.vstack([src[c]['features'] for c in SRC_BANK])
    demo_dir = DATA / 'aic_demo'
    if args.viz:
        demo_dir.mkdir(exist_ok=True)

    for coco_cls, cat_id in VIS_CLS.items():
        entries = gt_all[coco_cls]
        if not entries:
            continue
        print(f'--- {coco_cls} ({len(entries)} 图) ---', flush=True)
        dets = detect_visdrone(entries, TGT_TMPL.format(coco_cls))
        tgt = build_target(entries, dets)
        print(f'  {tgt["n_crops"]} 检测框进 CLIP', flush=True)

        src_cls = SRC_OF[coco_cls]
        delta = text_delta(src_cls, coco_cls)
        neg = build_neg_from(src, src_cls)
        g2 = g2_align(delta, all_src_feats, tgt['features'])

        base_f = fuse_scores(tgt, None, None, 0)
        p_tge = src[src_cls]['proto'] + BETA * delta
        p_tge /= (np.linalg.norm(p_tge) + 1e-8)
        tge_f = fuse_scores(tgt, p_tge, None, LAM)
        acspc_f = fuse_scores(tgt, p_tge, neg, LAM)
        if neg is not None:
            neg_sh = neg + BETA * delta
            neg_sh /= (np.linalg.norm(neg_sh) + 1e-8)
        else:
            neg_sh = None
        shared_f = fuse_scores(tgt, p_tge, neg_sh, LAM)

        r = {'g2': round(g2, 4), 'n_img': len(entries), 'n_crops': tgt['n_crops']}
        for name, fl in [('baseline', base_f), ('TGE', tge_f),
                         ('A-CSPC', acspc_f), ('A-CSPC-shared', shared_f)]:
            a50, amap = eval_both(tgt, fl)
            r[name] = {'ap50': round(a50, 4), 'map': round(amap, 4)}
            print(f'  {name:15s} AP50={a50:.4f}  mAP@[.5:.95]={amap:.4f}', flush=True)
        results['per_class'][coco_cls] = r

        if args.bootstrap:
            bs = bootstrap(tgt, base_f, acspc_f)
            r['bootstrap'] = bs
            sg = bs['ap50']
            print(f'  [bootstrap] AP50 增益 {sg["mean_gain"]:+.4f} '
                  f'95%CI[{sg["ci_lo"]:+.4f},{sg["ci_hi"]:+.4f}] '
                  f'{"显著" if sg["significant"] else "不显著"} '
                  f'(n_img={bs["n_img"]})', flush=True)

        if args.viz:
            save_demo(demo_dir, coco_cls, entries, tgt, base_f, acspc_f)

    # Safe-Correction: g2 门控（τ 扫描）
    print('=== Safe-Correction 门控分析 (τ 扫描) ===')
    sc = {}
    for tau in [round(0.0 + 0.05 * i, 2) for i in range(13)]:
        picks, a50s, maps_ = [], [], []
        for c, r in results['per_class'].items():
            use_acspc = r['g2'] >= tau
            picks.append(use_acspc)
            m = 'A-CSPC' if use_acspc else 'baseline'
            a50s.append(r[m]['ap50']); maps_.append(r[m]['map'])
        sc[str(tau)] = {'mean_ap50': round(float(np.mean(a50s)), 4),
                        'mean_map': round(float(np.mean(maps_)), 4),
                        'n_use_acspc': int(sum(picks))}
    results['safe_correction_tau_sweep'] = sc
    for t, v in sc.items():
        print(f'  τ={t}: mean AP50={v["mean_ap50"]:.4f} (用 A-CSPC 的类数 {v["n_use_acspc"]})')

    out = DATA / 'aic_visdrone_results.json'
    json.dump(results, open(out, 'w'), ensure_ascii=False, indent=2)
    print(f'已写入 {out}')


def build_neg_from(src, pos_cls):
    other = {k: v['proto'] for k, v in src.items() if k != pos_cls}
    sims = {k: float(np.dot(src[pos_cls]['proto'], v)) for k, v in other.items()}
    top3 = sorted(sims.items(), key=lambda x: -x[1])[:3]
    neg = np.mean([other[k] for k, _ in top3], axis=0)
    return neg / (np.linalg.norm(neg) + 1e-8)


def g2_align(delta, src_feats, tgt_feats):
    X = np.vstack([src_feats, tgt_feats])
    mean = X.mean(axis=0)
    Uc, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    comp = Vt[:min(K_PCA, Vt.shape[0])]
    def proj(v): return (v - mean) @ comp.T
    shift = proj(tgt_feats).mean(axis=0) - proj(src_feats).mean(axis=0)
    d = proj(delta)
    ns, nd = np.linalg.norm(shift), np.linalg.norm(d)
    if ns < 1e-8 or nd < 1e-8:
        return 0.0
    return float(np.dot(d, shift) / (nd * ns))


def save_demo(demo_dir, cls, entries, tgt, base_f, acspc_f):
    """选 GT 最多的 2 张图: GT绿 / baseline红 / A-CSPC蓝。"""
    order = np.argsort([-len(p['gt']) for p in tgt['per_img']])[:2]
    for rank, idx in enumerate(order):
        img_p, gt = entries[idx]
        img = Image.open(img_p).convert('RGB')
        dr = ImageDraw.Draw(img)
        for g in gt:
            dr.rectangle(g, outline=(0, 200, 0), width=3)
        for fused, color in [(base_f[idx], (255, 60, 60)), (acspc_f[idx], (60, 90, 255))]:
            top = np.argsort(-fused)[:20]
            p = tgt['per_img'][idx]
            for i in top:
                if fused[i] < 0.10:
                    continue
                dr.rectangle(list(p['box'][i]), outline=color, width=2)
        out = demo_dir / f'visdrone_{cls}_{img_p.stem}_{rank}.png'
        img.save(out)
        print(f'  demo → {out.name}')


if __name__ == '__main__':
    main()
