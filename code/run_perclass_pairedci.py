# -*- coding: utf-8 -*-
"""run_perclass_pairedci.py — 逐类（逐单元）配对 bootstrap 置信区间

目的：补外部评审第 8 条唯一真缺口——Table 4 只有逐类点估计增益，没有区间。
做法：与 run_closed_loop_balanced.py 完全同一口径（同一份检测缓存、同一套原型、
同一个全局 gate 尺度 c0 与 PRIOR_SCALE、同一个 dc×conf 的 γ），区别只在两点：
  1) 不跨单元平均，而是**逐单元**（detector × class）单独出区间；
  2) 配对：每一次重采样用**同一组图片下标**同时给 base 与 CA-PC 打分，
     取差值 → 区间描述的是"同一批图上的增益"，不是两个 AP 各自的区间。
  3) 额外给出 P(Δ<=0)：bootstrap 中增益 ≤0 的比例（Table 3 caption 曾悬空定义过的量）。

只重算 AP（纯 CPU），不重跑检测、不重跑 CLIP 推理（除了一次性建源原型）。
输出写新文件，不覆盖任何既有产物。

用法:
  HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu ^
  "C:/Users/张三/vg/Scripts/python.exe" -u run_perclass_pairedci.py --K 1000
"""
import os, sys, json, pickle, glob, time, argparse
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cpu')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch
from pathlib import Path

import run_bidirectional as RB

RB.DEVICE = 'cpu'
DEVICE = 'cpu'
BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
LAM = RB.LAM
NEG_TOPK = RB.NEG_TOPK
N_PROTO = RB.N_PROTO

DETECTORS = {
    'gdino': Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache'),
    'owlv2': Path(r'C:\Users\张三\Desktop\实验数据\probe_owl_cache'),
    'yolo':  Path(r'C:\Users\张三\Desktop\实验数据\probe_yolo_cache'),
    'rtdetr': Path(r'C:\Users\张三\Desktop\实验数据\probe_rtdetr_cache'),
}
GATE_FROM = ['gdino', 'owlv2']
OUT = Path(r'C:\Users\张三\Desktop\实验数据\run_perclass_pairedci.json')
FWD = 'fwd_dior2coco'


def unit(x):
    x = np.asarray(x, float)
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def load_blob(cache_dir, tname, eval_cls):
    hits = glob.glob(str(cache_dir / f'{tname}_{eval_cls}_n*.pkl')) or \
           glob.glob(str(cache_dir / f'{tname}_{eval_cls}_off{N_PROTO}_n*.pkl'))
    return pickle.load(open(hits[0], 'rb')) if hits else None


def ap_of(per_img, F, W):
    rows = []
    for p in per_img:
        if len(p['gd']):
            w = F[p['crop_idx']] @ W
            s = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
        else:
            s = np.zeros(0)
        rows.append((p['box'], s))
    return RB.Met.compute_ap50_img(rows, [p['gt'] for p in per_img])


def precompute_rows(per_img, F, W):
    """每个 cell 只算一次分数；bootstrap 只重算 AP。"""
    out = []
    for p in per_img:
        if len(p['gd']):
            w = F[p['crop_idx']] @ W
            s = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
        else:
            s = np.zeros(0)
        out.append((p['box'], s))
    return out


def paired_bootstrap(rows_a, rows_b, gts, K, seed=0):
    """rows_a/base, rows_b/corrected，逐图配对重采样，返回 Δ 的 mean/CI95/P(Δ<=0)。"""
    rng = np.random.default_rng(seed)
    n = len(gts)
    diffs = np.empty(K, float)
    for k in range(K):
        idx = rng.integers(0, n, n)
        ra = [rows_a[i] for i in idx]
        rb_ = [rows_b[i] for i in idx]
        gt = [gts[i] for i in idx]
        diffs[k] = RB.Met.compute_ap50_img(rb_, gt) - RB.Met.compute_ap50_img(ra, gt)
    return (float(diffs.mean()),
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)),
            float(np.mean(diffs <= 0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    K = args.K
    t0 = time.time()

    cfg = RB.Config()
    cfg.model.device = DEVICE
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    bank = {}
    for domain, pool in (('coco', RB.COCO_POOL), ('dior', RB.DIOR_POOL)):
        b = {}
        for cls in pool:
            ims, gts = RB.load_split(cfg, domain, cls, 0, N_PROTO)
            crops = RB.D.gt_crops(ims, gts)
            if not crops:
                continue
            Ft = RB.clip_encode_imgs(crops)
            pt = torch.from_numpy(Ft.mean(0))
            b[cls] = pt / pt.norm()
        bank[domain] = b

    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    E_photo_d = {c: E_photo[i] for i, c in enumerate(all_names)}
    E_over_d = {c: E_over[i] for i, c in enumerate(all_names)}

    data = {}
    pooled_feat = []
    for det, cache_dir in DETECTORS.items():
        cells = []
        for tname in RB.TASKS:
            src_dom, tgt_dom, pairs = RB.TASKS[tname]
            for src_cls, eval_cls in pairs:
                blob = load_blob(cache_dir, tname, eval_cls)
                if blob is None or src_cls not in bank[src_dom]:
                    continue
                per_img = blob['per_img']; F = blob['F']
                P_src = unit(bank[src_dom][src_cls].numpy())
                P_neg = unit(RB.Proto.get_negative_prototype(bank[src_dom], src_cls, NEG_TOPK).numpy())
                dc = E_photo_d[eval_cls] - E_over_d[src_cls]
                dc_mag = float(np.linalg.norm(dc))
                max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
                cells.append({'tname': tname, 'src': src_cls, 'eval': eval_cls, 'det': det,
                              'dc_mag': dc_mag, 'max_conf': max_conf,
                              '_per_img': per_img, '_F': F, '_P_src': P_src,
                              '_P_neg': P_neg, '_dc': dc, '_gts': [p['gt'] for p in per_img]})
                if det in GATE_FROM:
                    pooled_feat.append({'det': det, 'dc_mag': dc_mag, 'max_conf': max_conf})
        data[det] = cells

    # —— 全局 gate 尺度：与 run_closed_loop_balanced.py 逐字一致 ——
    conf_all = np.array([f['max_conf'] for f in pooled_feat])
    c0 = float(np.median(conf_all))
    PRIOR_SCALE = 2.0 / float(np.mean([c['dc_mag'] for cells in data.values() for c in cells]))

    def gate_conf(conf, c=c0, scale=3.0):
        return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))

    def gamma(cell):
        return float(np.clip(PRIOR_SCALE * cell['dc_mag'] * gate_conf(cell['max_conf'], c0, 3.0), 0.0, 4.0))

    print(f'[gate] c0={c0:.6f}  PRIOR_SCALE={PRIOR_SCALE:.6f}   K={K}')

    results = {}
    for det, cells in data.items():
        fwd = [c for c in cells if c['tname'] == FWD]
        print(f'\n[{det}] forward cells={len(fwd)}')
        rows_out = []
        for c in fwd:
            W_base = unit(c['_P_src']) - LAM * c['_P_neg']
            W_capc = unit(c['_P_src'] + gamma(c) * c['_dc']) - LAM * c['_P_neg']
            rb_ = precompute_rows(c['_per_img'], c['_F'], W_base)
            rc = precompute_rows(c['_per_img'], c['_F'], W_capc)
            point = ap_of(c['_per_img'], c['_F'], W_capc) - ap_of(c['_per_img'], c['_F'], W_base)
            t1 = time.time()
            m, lo, hi, pneg = paired_bootstrap(rb_, rc, c['_gts'], K, seed=args.seed)
            rows_out.append({'det': det, 'src': c['src'], 'eval': c['eval'],
                             'gamma': gamma(c), 'dc_mag': c['dc_mag'],
                             'point_gain': float(point), 'boot_mean': m,
                             'ci95': [lo, hi], 'P_gain_le_0': pneg})
            print(f'  {c["eval"]:<11} point={point:+.4f}  boot={m:+.4f}  '
                  f'CI95=[{lo:+.4f},{hi:+.4f}]  P(Δ≤0)={pneg:.3f}   ({time.time()-t1:.1f}s)')
        results[det] = rows_out

    json.dump({'config': {'K': K, 'seed': args.seed, 'lam': LAM, 'neg_topk': NEG_TOPK,
                          'n_proto': N_PROTO, 'c0': c0, 'prior_scale': PRIOR_SCALE,
                          'gate': 'clip(1-tanh(3*(conf-c0)),0,1)', 'metric': 'AP50 11-point',
                          'protocol': 'paired bootstrap over evaluation images (same indices for base and CA-PC)'},
               'per_class': results, 'elapsed_s': time.time() - t0},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
