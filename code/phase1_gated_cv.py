# -*- coding: utf-8 -*-
"""phase1_gated_cv.py — 难度门控重加权(假设A的严格验证)

场景级分析已显示: 平铺融合在易类(高 baseline)显著拖垮 AP, 只在难类(低 baseline)有帮助。
本脚本验证"逐图难度门控": 用每图检测器自身置信度作可靠性信号 r,
  若 r < τ(检测器不可靠) → 该图用 tuned 语义融合; 否则用 gd-only。
τ 在每折校准集上选, 测试折上报; k-fold 无偏 + bootstrap(B=200) 出 95% CI。
同时报 ungated(总用 tuned) 作为对照。

指标: compute_ap50_img (逐图 11-point AP50)。
输出: 实验数据/weight_probe_gated_cv.json。
"""
import os, sys, json, random, time
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import torch
import clip as clip_module
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from unified_eval import (get_gdino, get_clip, gdino_detect, load_coco, load_dior,
                          compute_ap50_img, DEVICE, SAVE_DIR)
from single_scenario_factor_expand import (prep_scenario, FACTORS, load_cond, ap_for_weights)
import itertools

NFOLD = 4
SEED = 42
BOOT = 200
LEVELS = [0.0, 0.5, 1.0]
COMBOS = [dict(zip(FACTORS, c)) for c in itertools.product(LEVELS, repeat=len(FACTORS))]
COMBOS = [c for c in COMBOS if any(c.values())]
TAUS = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40]

EXPANDED = [
    ('COCO-airplane',  'coco', 'airplane', 'natural_scene', 'a photo of',        'airplane',      80),
    ('COCO-boat',      'coco', 'boat',     'natural_scene', 'a photo of',        'ship',          80),
    ('COCO-car',       'coco', 'car',      'natural_scene', 'a photo of',        'vehicle',       80),
    ('DIOR-airplane',  'dior', 'airplane', 'remote_sensing', 'an overhead image of', 'airplane',  50),
    ('DIOR-ship',      'dior', 'ship',     'remote_sensing', 'an overhead image of', 'ship',      50),
    ('DIOR-vehicle',   'dior', 'vehicle',  'remote_sensing', 'an overhead image of', 'vehicle',   50),
    ('DIOR-harbor',    'dior', 'harbor',   'remote_sensing', 'an overhead image of', 'harbor',    50),
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


def img_reliability(prep_i):
    """每图检测器可靠性 = 前5高 gd 置信的均值(裁剪到[0,1])。高=可靠。"""
    g = np.asarray(prep_i['facs']['gd'])
    if len(g) == 0:
        return 0.0
    top = np.sort(g)[::-1][:5]
    return float(np.clip(top.mean(), 0.0, 1.0))


def gate_switch(gd_list, tuned_list, use_tuned):
    return tuned_list if use_tuned else gd_list


def main():
    t0 = time.time()
    cond_enc = load_cond()
    results = {}
    for name, dataset, cls_name, domain, prefix, dior_cls, n in EXPANDED:
        ts = time.time()
        print(f'\n[{"="*14}] {name}  (n={n})', flush=True)
        if dataset == 'coco':
            images, gts = load_coco(cls_name, n=n)
        else:
            images, gts = load_dior(cls_name, n=n)
        if len(images) < 8:
            print('  图太少, 跳过', flush=True)
            continue
        base_dets = gdino_detect(images, f'{prefix} {cls_name}', 0.0)
        clip_m, clip_pre = get_clip()
        tok = clip_module.tokenize([f'{prefix} {cls_name}']).to(DEVICE)
        with torch.no_grad():
            txt = clip_m.encode_text(tok).float().squeeze(0)
        txt = txt / txt.norm()
        prep = prep_scenario(images, base_dets, dior_cls, txt, cond_enc)
        nn = len(prep)
        r_all = [img_reliability(prep[i]) for i in range(nn)]

        rng = random.Random(SEED)
        idx = list(range(nn))
        rng.shuffle(idx)
        folds = [idx[i::NFOLD] for i in range(NFOLD)]

        test_all, fold_dg, fold_dt, fold_taus = [], [], [], []
        for f in range(NFOLD):
            test_set = set(folds[f])
            tr = [i for i in range(nn) if i not in test_set]
            te = folds[f]
            train_prep = [prep[i] for i in tr]
            train_gts = [gts[i] for i in tr]
            test_prep = [prep[i] for i in te]
            test_gts = [gts[i] for i in te]
            # 选 best_w (7因子, 同 combined)
            best_w, best_ap = None, -1.0
            for w in COMBOS:
                ap = ap_for_weights(train_prep, train_gts, w)
                if ap > best_ap:
                    best_ap, best_w = ap, dict(w)
            gd_train = per_img_fused(train_prep, {'gd': 1.0})
            tu_train = per_img_fused(train_prep, best_w)
            r_train = [r_all[i] for i in tr]
            # 在 train 上选 τ
            best_tau, best_gap = None, -1.0
            for tau in TAUS:
                gated = [gate_switch(gd_train[j], tu_train[j], r_train[j] < tau)
                         for j in range(len(train_prep))]
                ap = compute_ap50_img(gated, train_gts)
                if ap > best_gap:
                    best_gap, best_tau = ap, tau
            # test 上报
            gd_test = per_img_fused(test_prep, {'gd': 1.0})
            tu_test = per_img_fused(test_prep, best_w)
            r_test = [r_all[i] for i in te]
            gated_test = [gate_switch(gd_test[j], tu_test[j], r_test[j] < best_tau)
                          for j in range(len(test_prep))]
            ap_b = compute_ap50_img(gd_test, test_gts)
            ap_g = compute_ap50_img(gated_test, test_gts)
            ap_t = compute_ap50_img(tu_test, test_gts)
            fold_dg.append(ap_g - ap_b)
            fold_dt.append(ap_t - ap_b)
            fold_taus.append(best_tau)
            for j in range(len(test_prep)):
                test_all.append((gd_test[j], tu_test[j], r_test[j], test_gts[j], f))

        base_full = compute_ap50_img([x[0] for x in test_all], [x[3] for x in test_all])
        gated_full = compute_ap50_img([gate_switch(x[0], x[1], x[2] < fold_taus[x[4]])
                                       for x in test_all], [x[3] for x in test_all])
        tuned_full = compute_ap50_img([x[1] for x in test_all], [x[3] for x in test_all])
        delta_gate = round(gated_full - base_full, 4)
        delta_tuned = round(tuned_full - base_full, 4)
        # bootstrap CI (gated - base), 用每图自身 fold τ
        rng_b = random.Random(SEED + 7)
        nb = len(test_all)
        boots = []
        for _ in range(BOOT):
            sel = rng_b.choices(range(nb), k=nb)
            bs = [gate_switch(test_all[s][0], test_all[s][1],
                              test_all[s][2] < fold_taus[test_all[s][4]]) for s in sel]
            gs = [test_all[s][3] for s in sel]
            bs_gd = [test_all[s][0] for s in sel]
            boots.append(compute_ap50_img(bs, gs) - compute_ap50_img(bs_gd, gs))
        ci_low = round(float(np.percentile(boots, 2.5)), 4)
        ci_high = round(float(np.percentile(boots, 97.5)), 4)
        sig = 'YES' if ci_low > 0 else ('marginal' if delta_gate > -0.01 else 'no')
        print(f'  ungated(tuned) Δ={delta_tuned:+.4f}  gated Δ={delta_gate:+.4f}  '
              f'CI=[{ci_low:+.4f},{ci_high:+.4f}]  τ*={fold_taus}  sign={sig}', flush=True)
        results[name] = {'domain': domain, 'n_img': nn,
                         'mean_baseline': round(base_full, 4),
                         'delta_tuned_ungated': delta_tuned,
                         'delta_gated': delta_gate, 'ci_low': ci_low, 'ci_high': ci_high,
                         'bootstrap_sig': sig, 'tau_star': fold_taus,
                         'mean_reliability': round(float(np.mean(r_all)), 4),
                         'n_gt': int(sum(len(g) for g in gts))}
        print(f'  用时 {time.time()-ts:.1f}s', flush=True)

    dg = [r['delta_gated'] for r in results.values()]
    dt = [r['delta_tuned_ungated'] for r in results.values()]
    sg = sum(1 for r in results.values() if r['bootstrap_sig'] == 'YES')
    out = {'summary': {'n_scenarios': len(results),
                       'mean_delta_gated': round(float(np.mean(dg)), 4),
                       'mean_delta_tuned_ungated': round(float(np.mean(dt)), 4),
                       'n_gated_sig': sg},
           'scenarios': results}
    with open(SAVE_DIR / 'weight_probe_gated_cv.json', 'w', encoding='utf-8') as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print(f'\n完成 {len(results)} 场景: 无门控均Δ={out["summary"]["mean_delta_tuned_ungated"]:+.4f} '
          f'→ 门控均Δ={out["summary"]["mean_delta_gated"]:+.4f}, 显著 {sg}; 总用时 {time.time()-t0:.1f}s', flush=True)
    print(f'结果 → {SAVE_DIR / "weight_probe_gated_cv.json"}', flush=True)


if __name__ == '__main__':
    main()
