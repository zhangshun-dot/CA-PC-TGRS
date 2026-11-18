# -*- coding: utf-8 -*-
"""phase1_gated_unified_cv.py — 用"统一空间自身的可靠性估计"作门控(假设A的再验证)

上一版 phase1_gated_cv.py 用 GDINO 前5置信当可靠性代理, 失败: 净Δ≈无门控(-0.034),
未复现场景级门控的 +0.014。本版换信号源: 用统一空间自己的估计作门控, 两种候选:
  unc      = 1-|vis-clip|/2 (双骨干一致性, 高=语义内容确定→语义重排可信)
  cond_inv = 1-sigmoid(z_k) (CondEncoder 退化估计, 高=观测退化→检测器不可靠→需语义)
门控规则: 每图信号 s>=τ → 用 tuned 语义融合, 否则 gd。τ 在每折校准集选, 测试折上报。
4-fold 无偏 + bootstrap(B=200) 出 95% CI; 同时报 ungated(tuned) 对照。
输出: 实验数据/weight_probe_gated_unified_cv.json。
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
# 干净版: 去掉总开门控的退化阈值(unc≥0.5 恒成立 → tau=0.5 恒为真; cond_inv≥0 → tau=0.0 恒为真)
TAUS = {'unc': [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999],
        'cond_inv': [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]}
FIXED_TAU_UNC = 0.7  # 固定阈值跨折交叉验证, 防每折 tau 过拟合

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


def img_signal(prep_i, key):
    a = np.asarray(prep_i['facs'][key])
    if len(a) == 0:
        return 0.0
    return float(np.nanmean(a))


def build_gated(gd_list, tu_list, sigs, tau, use_mask):
    # use_mask True 的位置用 tu, 否则 gd
    return [tu_list[j] if use_mask[j] else gd_list[j] for j in range(len(gd_list))]


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
        r_unc = [img_signal(prep[i], 'unc') for i in range(nn)]
        r_cond = [img_signal(prep[i], 'cond_inv') for i in range(nn)]

        rng = random.Random(SEED)
        idx = list(range(nn))
        rng.shuffle(idx)
        folds = [idx[i::NFOLD] for i in range(NFOLD)]

        test_all, fold_dg, fold_dt, fold_spec, fold_df, test_all_fixed = [], [], [], [], [], []
        for f in range(NFOLD):
            test_set = set(folds[f])
            tr = [i for i in range(nn) if i not in test_set]
            te = folds[f]
            train_prep = [prep[i] for i in tr]; train_gts = [gts[i] for i in tr]
            test_prep = [prep[i] for i in te]; test_gts = [gts[i] for i in te]
            best_w, best_ap = None, -1.0
            for w in COMBOS:
                ap = ap_for_weights(train_prep, train_gts, w)
                if ap > best_ap:
                    best_ap, best_w = ap, dict(w)
            gd_train = per_img_fused(train_prep, {'gd': 1.0})
            tu_train = per_img_fused(train_prep, best_w)
            ru_tr = [r_unc[i] for i in tr]; rc_tr = [r_cond[i] for i in tr]
            # 选 (gate_type, tau)
            best_spec, best_gap = None, -1.0
            for gtype, taus in TAUS.items():
                sigs = ru_tr if gtype == 'unc' else rc_tr
                for tau in taus:
                    mask = [s >= tau for s in sigs]
                    gated = build_gated(gd_train, tu_train, sigs, tau, mask)
                    ap = compute_ap50_img(gated, train_gts)
                    if ap > best_gap:
                        best_gap, best_spec = ap, (gtype, tau)
            gd_test = per_img_fused(test_prep, {'gd': 1.0})
            tu_test = per_img_fused(test_prep, best_w)
            ru_te = [r_unc[i] for i in te]; rc_te = [r_cond[i] for i in te]
            gtype, tau = best_spec
            sigs = ru_te if gtype == 'unc' else rc_te
            mask = [s >= tau for s in sigs]
            gated_test = build_gated(gd_test, tu_test, sigs, tau, mask)
            ap_b = compute_ap50_img(gd_test, test_gts)
            ap_g = compute_ap50_img(gated_test, test_gts)
            ap_t = compute_ap50_img(tu_test, test_gts)
            fold_dg.append(ap_g - ap_b)
            fold_dt.append(ap_t - ap_b)
            fold_spec.append(best_spec)
            # 固定阈值 unc 门控(防每折 tau 过拟合的干净对照)
            fixed_mask = [s >= FIXED_TAU_UNC for s in ru_te]
            fixed_test = build_gated(gd_test, tu_test, ru_te, FIXED_TAU_UNC, fixed_mask)
            ap_f = compute_ap50_img(fixed_test, test_gts)
            fold_df.append(ap_f - ap_b)
            for j in range(len(test_prep)):
                test_all.append((gd_test[j], tu_test[j], ru_te[j], rc_te[j], test_gts[j], f))
                test_all_fixed.append((gd_test[j], fixed_test[j], ru_te[j], rc_te[j], test_gts[j], f))

        base_full = compute_ap50_img([x[0] for x in test_all], [x[4] for x in test_all])
        tu_full = compute_ap50_img([x[1] for x in test_all], [x[4] for x in test_all])
        # 用逐折 spec: 每图用其 fold 的 spec
        gated_full = []
        for x in test_all:
            f = x[5]; gtype, tau = fold_spec[f]
            sigs = x[2] if gtype == 'unc' else x[3]
            gated_full.append(x[1] if sigs >= tau else x[0])
        delta_gate = round(compute_ap50_img(gated_full, [x[4] for x in test_all]) - base_full, 4)
        delta_tuned = round(tu_full - base_full, 4)
        # 固定阈值 unc 门控聚合
        fixed_full = compute_ap50_img([x[1] for x in test_all_fixed], [x[4] for x in test_all_fixed])
        delta_fixed = round(fixed_full - base_full, 4)
        # bootstrap
        rng_b = random.Random(SEED + 11)
        nb = len(test_all)
        boots, boots_fixed = [], []
        for _ in range(BOOT):
            sel = rng_b.choices(range(nb), k=nb)
            gs = [test_all[s][4] for s in sel]
            bs_gd = [test_all[s][0] for s in sel]
            gl = []
            for s in sel:
                f = test_all[s][5]; gtype, tau = fold_spec[f]
                sigs = test_all[s][2] if gtype == 'unc' else test_all[s][3]
                gl.append(test_all[s][1] if sigs >= tau else test_all[s][0])
            boots.append(compute_ap50_img(gl, gs) - compute_ap50_img(bs_gd, gs))
            fl = [test_all_fixed[s][1] if test_all_fixed[s][2] >= FIXED_TAU_UNC else test_all_fixed[s][0] for s in sel]
            boots_fixed.append(compute_ap50_img(fl, gs) - compute_ap50_img(bs_gd, gs))
        ci_low = round(float(np.percentile(boots, 2.5)), 4)
        ci_high = round(float(np.percentile(boots, 97.5)), 4)
        sig = 'YES' if ci_low > 0 else ('marginal' if delta_gate > -0.01 else 'no')
        ci_low_f = round(float(np.percentile(boots_fixed, 2.5)), 4)
        ci_high_f = round(float(np.percentile(boots_fixed, 97.5)), 4)
        sig_f = 'YES' if ci_low_f > 0 else ('marginal' if delta_fixed > -0.01 else 'no')
        # 统计两种 gate 各胜几折
        from collections import Counter
        gt_count = Counter(s for (s, _) in fold_spec)
        print(f'  ungated Δ={delta_tuned:+.4f}  unified-gated Δ={delta_gate:+.4f}  '
              f'CI=[{ci_low:+.4f},{ci_high:+.4f}]  gate胜出={dict(gt_count)}  spec={fold_spec}  sign={sig}', flush=True)
        results[name] = {'domain': domain, 'n_img': nn, 'mean_baseline': round(base_full, 4),
                         'delta_tuned_ungated': delta_tuned, 'delta_gated': delta_gate,
                         'ci_low': ci_low, 'ci_high': ci_high, 'bootstrap_sig': sig,
                         'gate_spec_per_fold': [list(s) for s in fold_spec],
                         'gate_win_counts': dict(gt_count), 'n_gt': int(sum(len(g) for g in gts))}
        print(f'  用时 {time.time()-ts:.1f}s', flush=True)

    dg = [r['delta_gated'] for r in results.values()]
    dt = [r['delta_tuned_ungated'] for r in results.values()]
    sg = sum(1 for r in results.values() if r['bootstrap_sig'] == 'YES')
    out = {'summary': {'n_scenarios': len(results),
                       'mean_delta_gated': round(float(np.mean(dg)), 4),
                       'mean_delta_tuned_ungated': round(float(np.mean(dt)), 4),
                       'n_gated_sig': sg},
           'scenarios': results}
    with open(SAVE_DIR / 'weight_probe_gated_unified_cv.json', 'w', encoding='utf-8') as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print(f'\n完成 {len(results)} 场景: 无门控均Δ={out["summary"]["mean_delta_tuned_ungated"]:+.4f} '
          f'→ 统一空间门控均Δ={out["summary"]["mean_delta_gated"]:+.4f}, 显著 {sg}; 总用时 {time.time()-t0:.1f}s', flush=True)
    print(f'结果 → {SAVE_DIR / "weight_probe_gated_unified_cv.json"}', flush=True)


if __name__ == '__main__':
    main()
