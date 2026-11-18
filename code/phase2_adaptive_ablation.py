# -*- coding: utf-8 -*-
"""phase2_adaptive_ablation.py — Phase2: 类级自适应 α + shuffle/zero-z_k 消融

为什么需要本脚本:
  Phase1 的 weight_probe_combined_cv.json 只存了"逐场景聚合结果"(Δ无偏/avg_weights),
  没有逐框 facs。要验证"表示质量因子(z_k/z_c)到底有没有因果作用", 必须拿到逐框
  cond/cond_inv(=sigmoid(z_k)), 做 zero / shuffle 消融后重调融合权重, 看增益是否消失。

两段:
  [cache]  GPU: 重跑 prep_scenario, 把逐图逐框 7 维 facs + boxes + gts 缓存到 pkl。
  [ablate] CPU: 在缓存 facs 上离线城市做消融(无需 GPU)。

消融口径(公平: 模块看不到真 z_k, 必须重调权重):
  - zero-z_k : 把所有 crop 的 cond=cond_inv=1.0(几何平均里 1^w=1, 即丢弃 z_k 分支),
               5 因子网格重调, 测测试集 Δ。与原 7 因子 Δ 比 → z_k 是不是必需的。
  - shuffle-z_k: 跨场景所有 crop 打乱 cond/cond_inv(破坏 crop↔自身条件对齐), 7 因子重调, 测 Δ。
  - 类级自适应 α: 按 mean_baseline 分难度桶, 留一法(本类学自同桶其他类)套权重, 测 Δ。

指标: 论文口径 11-point AP50(逐图)。Δ = tuned - raw(gd)。
与 weight_probe_combined_cv.json 的 delta_unbiased 直接可比(同 NFOLD=4 / SEED=42 / BOOT=200)。
"""
import os, sys, json, random, argparse, pickle, itertools
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import torch
import clip as clip_module
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from unified_eval import (get_clip, gdino_detect, load_coco, load_dior,
                          compute_ap50_img, EVALS, DEVICE, SAVE_DIR)
from single_scenario_factor_expand import (prep_scenario, FACTORS, load_cond, ap_for_weights)

NFOLD = 4
SEED = 42
BOOT = 200
LEVELS = [0.0, 0.5, 1.0]
GRID7 = [dict(zip(FACTORS, c)) for c in itertools.product(LEVELS, repeat=len(FACTORS))]
GRID7 = [c for c in GRID7 if any(c.values())]
F5 = ['gd', 'vis', 'clip', 'rel', 'unc']
GRID5 = [dict(zip(F5, c)) for c in itertools.product(LEVELS, repeat=len(F5))]
GRID5 = [c for c in GRID5 if any(c.values())]

EXPANDED = [
    ('COCO-airplane',  'coco', 'airplane', 'natural_scene', 'a photo of',          'airplane',  80),
    ('COCO-boat',      'coco', 'boat',     'natural_scene', 'a photo of',          'ship',      80),
    ('COCO-car',       'coco', 'car',      'natural_scene', 'a photo of',          'vehicle',   80),
    ('DIOR-airplane',  'dior', 'airplane', 'remote_sensing', 'an overhead image of', 'airplane', 50),
    ('DIOR-ship',      'dior', 'ship',     'remote_sensing', 'an overhead image of', 'ship',    50),
    ('DIOR-vehicle',   'dior', 'vehicle',  'remote_sensing', 'an overhead image of', 'vehicle',  50),
    ('DIOR-harbor',    'dior', 'harbor',   'remote_sensing', 'an overhead image of', 'harbor',   50),
    ('DIOR-storage',   'dior', 'storage tank','remote_sensing','an overhead image of','storage tank', 50),
    ('DIOR-tennis',    'dior', 'tennis court','remote_sensing','an overhead image of','tennis court', 50),
    ('DIOR-baseball',  'dior', 'baseball field','remote_sensing','an overhead image of','baseball field', 50),
    ('DIOR-bridge',    'dior', 'bridge',   'remote_sensing', 'an overhead image of', 'bridge',    50),
]


def per_img_fused(prep, w):
    out = []
    for d in prep:
        if len(d['boxes']) == 0:
            out.append((d['boxes'], np.empty((0,))))
            continue
        fused = np.ones(len(d['boxes']))
        for k in FACTORS:
            fused = fused * (d['facs'][k] ** w.get(k, 0.0))
        out.append((d['boxes'], fused))
    return out


# ═══════════ [cache] 逐框 facs 缓存 (GPU) ═══════════
def cache_facs(scenario_names, max_n, out_pkl):
    cond_enc = load_cond()
    clip_m, clip_pre = get_clip()
    cache = {}
    for name, dataset, cls_name, domain, prefix, dior_cls, n_default in EXPANDED:
        if name not in scenario_names:
            continue
        n = min(max_n, n_default) if max_n else n_default
        print(f'[cache] {name} (n={n})', flush=True)
        if dataset == 'coco':
            images, gts = load_coco(cls_name, n=n)
        else:
            images, gts = load_dior(cls_name, n=n)
        if len(images) < 8:
            print('  图太少, 跳过', flush=True); continue
        base_dets = gdino_detect(images, f'{prefix} {cls_name}', 0.0)
        tok = clip_module.tokenize([f'{prefix} {cls_name}']).to(DEVICE)
        with torch.no_grad():
            txt = clip_m.encode_text(tok).float().squeeze(0)
        txt = txt / txt.norm()
        prep = prep_scenario(images, base_dets, dior_cls, txt, cond_enc)
        # 转 numpy 便于存盘
        prep_n = []
        for d in prep:
            facs_n = {k: np.asarray(v, dtype=np.float64) for k, v in d['facs'].items()}
            prep_n.append({'boxes': np.asarray(d['boxes'], dtype=np.float64), 'facs': facs_n})
        cache[name] = {'prep': prep_n, 'gts': [np.asarray(g, dtype=np.float64) for g in gts],
                       'domain': domain, 'dior_cls': dior_cls, 'n_img': len(images)}
        print(f'  缓存 {len(prep_n)} 图, 含框图 {sum(1 for d in prep_n if len(d["boxes"])>0)}', flush=True)
    pickle.dump(cache, open(out_pkl, 'wb'))
    print(f'[cache] 完成 → {out_pkl} ({len(cache)} 场景)', flush=True)


# ═══════════ 变换: zero / shuffle z_k ═══════════
def transform_zero(prep):
    """cond=cond_inv=1.0 → 几何平均里丢弃 z_k 分支。"""
    out = []
    for d in prep:
        facs = {k: v.copy() for k, v in d['facs'].items()}
        if len(d['boxes']) > 0:
            facs['cond'] = np.ones(len(d['boxes']))
            facs['cond_inv'] = np.ones(len(d['boxes']))
        out.append({'boxes': d['boxes'], 'facs': facs})
    return out


def transform_shuffle(prep, rng):
    """跨场景所有 crop 打乱 cond/cond_inv, 破坏 crop↔自身条件对齐。"""
    # 收集
    all_cond, all_cinv, lens = [], [], []
    for d in prep:
        if len(d['boxes']) > 0:
            all_cond.append(d['facs']['cond'])
            all_cinv.append(d['facs']['cond_inv'])
            lens.append(len(d['facs']['cond']))
        else:
            lens.append(0)
    if all_cond:
        cond_cat = np.concatenate(all_cond); cinv_cat = np.concatenate(all_cinv)
        p1 = rng.permutation(len(cond_cat)); p2 = rng.permutation(len(cond_cat))
        cond_cat = cond_cat[p1]; cinv_cat = cinv_cat[p2]
        it = iter(zip(cond_cat, cinv_cat))
    out = []
    for d, L in zip(prep, lens):
        facs = {k: v.copy() for k, v in d['facs'].items()}
        if L > 0:
            cseq = np.array([next(it) for _ in range(L)])
            facs['cond'] = cseq[:, 0]; facs['cond_inv'] = cseq[:, 1]
        out.append({'boxes': d['boxes'], 'facs': facs})
    return out


# ═══════════ 4-fold CV 无偏 Δ (复刻 phase1) ═══════════
def cv_delta(prep_list, gts_list, grid, seed=SEED, boot=BOOT):
    nn = len(prep_list)
    rng = random.Random(seed)
    idx = list(range(nn)); rng.shuffle(idx)
    folds = [idx[i::NFOLD] for i in range(NFOLD)]
    test_all, fold_deltas = [], []
    for f in range(NFOLD):
        test_set = set(folds[f])
        train_prep = [prep_list[i] for i in range(nn) if i not in test_set]
        train_gts = [gts_list[i] for i in range(nn) if i not in test_set]
        test_prep = [prep_list[i] for i in folds[f]]
        test_gts = [gts_list[i] for i in folds[f]]
        best_w, best_ap = None, -1.0
        for w in grid:
            ap = ap_for_weights(train_prep, train_gts, w)
            if ap > best_ap:
                best_ap, best_w = ap, dict(w)
        base_lists = per_img_fused(test_prep, {'gd': 1.0})
        tuned_lists = per_img_fused(test_prep, best_w)
        tuned = compute_ap50_img(tuned_lists, test_gts)
        base = compute_ap50_img(base_lists, test_gts)
        fold_deltas.append(tuned - base)
        for i in range(len(test_prep)):
            test_all.append((base_lists[i], tuned_lists[i], test_gts[i]))
    base_full = compute_ap50_img([b for b, _, _ in test_all], [g for _, _, g in test_all])
    tuned_full = compute_ap50_img([t for _, t, _ in test_all], [g for _, _, g in test_all])
    delta_unbiased = tuned_full - base_full
    rng_b = random.Random(seed + 1); nb = len(test_all); boots = []
    for _ in range(boot):
        sel = rng_b.choices(range(nb), k=nb)
        bs = [test_all[s][0] for s in sel]; ts = [test_all[s][1] for s in sel]; gs = [test_all[s][2] for s in sel]
        boots.append(compute_ap50_img(ts, gs) - compute_ap50_img(bs, gs))
    ci_low = float(np.percentile(boots, 2.5)); ci_high = float(np.percentile(boots, 97.5))
    return float(delta_unbiased), ci_low, ci_high, float(np.std(fold_deltas))


def eval_fixed_weights(prep_list, gts_list, w):
    """本类全集(全为测试)上套固定权重 w 的 Δ(点估计, 用于类级自适应 LOO)。"""
    tuned = ap_for_weights(prep_list, gts_list, w)
    base = ap_for_weights(prep_list, gts_list, {'gd': 1.0})
    return float(tuned - base)


def best_w_on_pool(pool_prep, pool_gts, grid):
    """在合并池上 4-fold CV 选最优权重。"""
    d, _, _, _ = cv_delta(pool_prep, pool_gts, grid)
    # 重新选(返回权重向量): 直接在全池上 grid 选最优
    best_w, best_ap = None, -1.0
    for w in grid:
        ap = ap_for_weights(pool_prep, pool_gts, w)
        if ap > best_ap:
            best_ap, best_w = ap, dict(w)
    return best_w


# ═══════════ [ablate] 主消融 ═══════════
def ablate(cache, base_json):
    base_sc = base_json['scenarios']
    names = list(cache.keys())
    # baseline(原 7 因子, 来自 phase1 聚合)
    orig = {n: base_sc[n]['delta_unbiased'] for n in names if n in base_sc}

    res = {'names': names, 'orig_7factor': orig, 'zero_zk': {}, 'shuffle_zk': {},
           'zero_zk_cachedw': {}, 'shuffle_zk_cachedw': {}, 'class_adaptive_loo': {}}

    # ---- (1) zero-z_k 消融(5 因子重调) ----
    print('\n=== (1) zero-z_k 消融 (丢弃 cond/cond_inv, 5因子重调) ===', flush=True)
    for n in names:
        prep0 = transform_zero(cache[n]['prep'])
        d, lo, hi, fs = cv_delta(prep0, cache[n]['gts'], GRID5)
        res['zero_zk'][n] = {'delta': round(d, 4), 'ci_low': round(lo, 4),
                             'ci_high': round(hi, 4), 'fold_std': round(fs, 4)}
        # 用原 7 因子缓存权重套在 zero 后 facs 上(看原 operating point 能否复现)
        aw = base_sc[n]['avg_weights']
        dc = eval_fixed_weights(prep0, cache[n]['gts'], aw)
        res['zero_zk_cachedw'][n] = round(dc, 4)
        print(f'  {n:16s} zero_Δ={d:+.4f} [CI {lo:+.4f},{hi:+.4f}]  '
              f'原权重套zero_Δ={dc:+.4f}  原7因子Δ={orig.get(n,float("nan")):+.4f}', flush=True)

    # ---- (2) shuffle-z_k 消融(7 因子重调, 多 seed) ----
    print('\n=== (2) shuffle-z_k 消融 (打乱 cond/cond_inv, 7因子重调, 3 seed 均值) ===', flush=True)
    for n in names:
        ds = []
        for seed in [0, 1, 2]:
            rng = np.random.default_rng(seed)
            prep_s = transform_shuffle(cache[n]['prep'], rng)
            d, lo, hi, fs = cv_delta(prep_s, cache[n]['gts'], GRID7)
            ds.append(d)
            if seed == 0:
                res['shuffle_zk'][n] = {'delta': round(d, 4), 'ci_low': round(lo, 4),
                                         'ci_high': round(hi, 4), 'fold_std': round(fs, 4)}
            aw = base_sc[n]['avg_weights']
            dc = eval_fixed_weights(prep_s, cache[n]['gts'], aw)
            if seed == 0:
                res['shuffle_zk_cachedw'][n] = round(dc, 4)
        res['shuffle_zk'][n]['delta_mean3'] = round(float(np.mean(ds)), 4)
        print(f'  {n:16s} shuffle_Δ(3seed均)={np.mean(ds):+.4f}  '
              f'原权重套shuffle_Δ={res["shuffle_zk_cachedw"][n]:+.4f}  '
              f'原7因子Δ={orig.get(n,float("nan")):+.4f}', flush=True)

    # ---- (3) 类级自适应 α: 按 baseline 分 3 难度桶, LOO ----
    print('\n=== (3) 类级自适应 α (按 baseline_AP50 分 3 桶, 留一法学权重) ===', flush=True)
    baselines = {n: base_sc[n]['mean_baseline'] for n in names if n in base_sc}
    bvals = np.array(list(baselines.values()))
    q1, q2 = np.quantile(bvals, [1/3, 2/3])
    def bucket_of(b):
        return 0 if b <= q1 else (1 if b <= q2 else 2)
    buckets = {0: [], 1: [], 2: []}
    for n, b in baselines.items():
        buckets[bucket_of(b)].append(n)
    print(f'  桶阈值 q1={q1:.3f} q2={q2:.3f}; 桶大小={ {k:len(v) for k,v in buckets.items()} }', flush=True)
    for s in names:
        if s not in baselines:
            continue
        bk = bucket_of(baselines[s])
        others = [o for o in buckets[bk] if o != s]
        if not others:
            others = [o for o in names if o != s and o in baselines]  # 孤类退化: 用全体
        pool_prep = [cache[o]['prep'] for o in others]
        pool_gts = [cache[o]['gts'] for o in others]
        # 合并
        pool_prep_f = [d for pl in pool_prep for d in pl]
        pool_gts_f = [g for gl in pool_gts for g in gl]
        bw = best_w_on_pool(pool_prep_f, pool_gts_f, GRID7)
        d = eval_fixed_weights(cache[s]['prep'], cache[s]['gts'], bw)
        res['class_adaptive_loo'][s] = {'delta': round(d, 4), 'bucket': bk,
                                        'learned_from': others,
                                        'w': {k: round(v, 2) for k, v in bw.items() if v > 0}}
        print(f'  {s:16s} 桶={bk} 自适应Δ={d:+.4f} 学自{others} w={ {k:round(v,2) for k,v in bw.items() if v>0} }', flush=True)

    # ---- 汇总 ----
    def m(dct, key):
        vals = [v[key] if isinstance(v, dict) else v for v in dct.values()]
        return float(np.mean(vals)) if vals else float('nan')
    summary = {
        'orig_7factor_mean': m(orig, None) if False else float(np.mean(list(orig.values()))),
        'zero_zk_mean': m(res['zero_zk'], 'delta'),
        'shuffle_zk_mean': m(res['shuffle_zk'], 'delta_mean3'),
        'zero_zk_cachedw_mean': m(res['zero_zk_cachedw'], None),
        'shuffle_zk_cachedw_mean': m(res['shuffle_zk_cachedw'], None),
        'class_adaptive_loo_mean': m(res['class_adaptive_loo'], 'delta'),
    }
    res['summary'] = summary
    print('\n=== 汇总(跨场景均值 Δ=tuned-raw) ===', flush=True)
    print(f'  原7因子(phase1)      : {summary["orig_7factor_mean"]:+.4f}', flush=True)
    print(f'  zero-z_k(重调)        : {summary["zero_zk_mean"]:+.4f}', flush=True)
    print(f'  shuffle-z_k(重调,均)  : {summary["shuffle_zk_mean"]:+.4f}', flush=True)
    print(f'  zero-z_k(原权重套)    : {summary["zero_zk_cachedw_mean"]:+.4f}', flush=True)
    print(f'  shuffle-z_k(原权重套) : {summary["shuffle_zk_cachedw_mean"]:+.4f}', flush=True)
    print(f'  类级自适应α(LOO)      : {summary["class_adaptive_loo_mean"]:+.4f}', flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', action='store_true', help='GPU: 缓存逐框 facs')
    ap.add_argument('--ablate', action='store_true', help='CPU: 跑消融')
    ap.add_argument('--small', action='store_true', help='小样本验证: 2场景×20图')
    ap.add_argument('--scenarios', default='', help='逗号分隔场景名(默认全11)')
    ap.add_argument('--max_n', type=int, default=0, help='每场景图数上限(0=用默认)')
    ap.add_argument('--cache_pkl', default=str(SAVE_DIR / 'phase2_facs_cache.pkl'))
    args = ap.parse_args()

    if args.small:
        args.scenarios = 'COCO-airplane,DIOR-vehicle'
        args.max_n = 20
    scen = [s.strip() for s in args.scenarios.split(',') if s.strip()] if args.scenarios else \
        [e[0] for e in EXPANDED]

    if args.cache:
        cache_facs(scen, args.max_n, args.cache_pkl)
    if args.ablate:
        cache = pickle.load(open(args.cache_pkl, 'rb'))
        base_json = json.load(open(SAVE_DIR / 'weight_probe_combined_cv.json', encoding='utf-8'))
        res = ablate(cache, base_json)
        out = SAVE_DIR / 'phase2_ablation.json'
        json.dump(res, open(out, 'w'), ensure_ascii=False, indent=2)
        print(f'\n结果 → {out}', flush=True)


if __name__ == '__main__':
    main()
