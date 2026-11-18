# -*- coding: utf-8 -*-
"""phase1_combined_coverage_cv.py — Phase1 合并版: (A)扩场景 + (B)大测试集锁效应

设计(对齐用户 2026-09-24 "B和A一起"):
  (A) 覆盖更多场景: COCO 3 类(n=80) + DIOR 扩到 8 类(n=50), 共 11 个 (domain,class) 场景,
      每个场景产出一张"稳健最优权重向量"(7维), 作为 Phase2 自适应模块的训练种子。
  (B) 大测试集锁效应: 每场景图量从 20-30 提到 50-80, 4-fold 校准/测试分离取无偏 Δ,
      再对测试集做 bootstrap(B=200) 出 95% 置信区间, 把"方向可行"从点估计升级为带区间的结论。

指标: 论文口径 compute_ap50_img (逐图 11-point AP50)。
因子(7): gd / vis(DINOv2正原型) / clip(CLIP图文) / rel(类间负原型) / unc(双骨干分歧) / cond(sigmoid z_k) / cond_inv(1-cond)。
融合: 加权几何平均 fused = ∏ f_k^{w_k} (w_k>=0)。
输出: 实验数据/weight_probe_combined_cv.json (含每场景 Δ无偏 / CI / 稳健权重 / fold权重)。
"""
import os, sys, json, random, time
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import torch
import clip as clip_module
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from unified_eval import (get_gdino, get_clip, gdino_detect, load_coco, load_dior,
                          compute_ap50_img, EVALS, DEVICE, SAVE_DIR)
from single_scenario_factor_expand import (prep_scenario, FACTORS, load_cond, ap_for_weights)
import itertools

NFOLD = 4
SEED = 42
BOOT = 200
LEVELS = [0.0, 0.5, 1.0]
COMBOS = [dict(zip(FACTORS, c)) for c in itertools.product(LEVELS, repeat=len(FACTORS))]
COMBOS = [c for c in COMBOS if any(c.values())]

# (name, dataset, cls_name, domain, prefix, dior_cls, n)
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
        rng = random.Random(SEED)
        idx = list(range(nn))
        rng.shuffle(idx)
        folds = [idx[i::NFOLD] for i in range(NFOLD)]

        test_all, fold_deltas, fold_w = [], [], []
        for f in range(NFOLD):
            test_set = set(folds[f])
            train_prep = [prep[i] for i in range(nn) if i not in test_set]
            train_gts = [gts[i] for i in range(nn) if i not in test_set]
            test_prep = [prep[i] for i in folds[f]]
            test_gts = [gts[i] for i in folds[f]]
            best_w, best_ap = None, -1.0
            for w in COMBOS:
                ap = ap_for_weights(train_prep, train_gts, w)
                if ap > best_ap:
                    best_ap, best_w = ap, dict(w)
            base_lists = per_img_fused(test_prep, {'gd': 1.0})
            tuned_lists = per_img_fused(test_prep, best_w)
            tuned = compute_ap50_img(tuned_lists, test_gts)
            base = compute_ap50_img(base_lists, test_gts)
            fold_deltas.append(tuned - base)
            fold_w.append(best_w)
            for i in range(len(test_prep)):
                test_all.append((base_lists[i], tuned_lists[i], test_gts[i]))

        # 全集一次性 AP (所有测试折图像并集, 每图恰好出现一次)
        base_full = compute_ap50_img([b for (b, _, _) in test_all], [g for (_, _, g) in test_all])
        tuned_full = compute_ap50_img([t for (_, t, _) in test_all], [g for (_, _, g) in test_all])
        delta_unbiased = round(tuned_full - base_full, 4)
        delta_std = round(float(np.std(fold_deltas)), 4)

        # bootstrap 95% CI (配对重采样测试集图像)
        rng_b = random.Random(SEED + 1)
        nb = len(test_all)
        boots = []
        for _ in range(BOOT):
            sel = rng_b.choices(range(nb), k=nb)
            bs = [test_all[s][0] for s in sel]
            ts_ = [test_all[s][1] for s in sel]
            gs = [test_all[s][2] for s in sel]
            ap_b = compute_ap50_img(ts_, gs) - compute_ap50_img(bs, gs)
            boots.append(ap_b)
        ci_low = round(float(np.percentile(boots, 2.5)), 4)
        ci_high = round(float(np.percentile(boots, 97.5)), 4)

        avg_w = {k: round(float(np.mean([fw[k] for fw in fold_w])), 3) for k in FACTORS}
        sig = 'YES' if ci_low > 0 else ('marginal' if delta_unbiased > -0.01 else 'no')
        print(f'  baseline(无偏)={base_full:.4f}  tuned(无偏)={tuned_full:.4f}  '
              f'Δ无偏={delta_unbiased:+.4f}  CI=[{ci_low:+.4f},{ci_high:+.4f}]  fold_std={delta_std:.4f}  sign={sig}',
              flush=True)
        print(f'  稳健权重={ {k: v for k, v in avg_w.items() if v > 0} }', flush=True)
        results[name] = {'domain': domain, 'dior_cls': dior_cls, 'n_img': nn,
                         'n_gt': int(sum(len(g) for g in gts)),
                         'mean_baseline': round(base_full, 4), 'mean_tuned': round(tuned_full, 4),
                         'delta_unbiased': delta_unbiased, 'ci_low': ci_low, 'ci_high': ci_high,
                         'delta_std_folds': delta_std, 'bootstrap_sig': sig,
                         'avg_weights': avg_w,
                         'fold_weights': [{k: round(v, 2) for k, v in fw.items()} for fw in fold_w],
                         'fold_deltas': [round(x, 4) for x in fold_deltas]}
        print(f'  用时 {time.time()-ts:.1f}s', flush=True)

    pos = sum(1 for r in results.values() if r['delta_unbiased'] > 0)
    sig = sum(1 for r in results.values() if r['bootstrap_sig'] == 'YES')
    summary = {'n_scenarios': len(results), 'n_positive': pos,
               'n_bootstrap_sig': sig,
               'mean_delta': round(float(np.mean([r['delta_unbiased'] for r in results.values()])), 4)}
    out = {'summary': summary, 'scenarios': results}
    with open(SAVE_DIR / 'weight_probe_combined_cv.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n完成 {len(results)} 场景, 正增益 {pos}, bootstrap显著 {sig}; 总用时 {time.time()-t0:.1f}s', flush=True)
    print(f'结果 → {SAVE_DIR / "weight_probe_combined_cv.json"}', flush=True)


if __name__ == '__main__':
    main()
