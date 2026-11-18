# -*- coding: utf-8 -*-
"""single_scenario_factor_expand_cv.py — (A) 扩因子 无偏版: 校准/测试分离(k-fold)

修正前期"权重在测试集上选"的选择偏差:
  每场景把图做 4-fold; 对每个 fold: 在 train 折上调最优权重向量, 在 test 折上报 mAP。
  取 fold 均值 = 无偏估计; 各 fold 最优权重取均值 = 该场景稳健最优权重(供 Phase2 自适应用)。
指标: 论文正确口径 compute_ap50_img (逐图 11-point AP50)。
因子(7): gd / vis(DINOv2正原型) / clip(CLIP图文) / rel(类间负原型) / unc(双骨干分歧) / cond(sigmoid z_k) / cond_inv(1-cond)。
融合: 加权几何平均 fused = ∏ f_k^{w_k} (w_k>=0, Σw=1)。
输出: 每场景 mean_baseline / mean_tuned / delta_unbiased / avg_weights + 各 fold 权重。
"""
import os, sys, json, random
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import torch
import clip as clip_module
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from unified_eval import (get_gdino, get_clip, gdino_detect, load_coco, load_dior,
                          EVALS, DEVICE, SAVE_DIR)
from single_scenario_factor_expand import (prep_scenario, FACTORS, load_cond, ap_for_weights)
import itertools

NFOLD = 4
SEED = 42
LEVELS = [0.0, 0.5, 1.0]
COMBOS = [dict(zip(FACTORS, c)) for c in itertools.product(LEVELS, repeat=len(FACTORS))]
COMBOS = [c for c in COMBOS if any(c.values())]


def main():
    cond_enc = load_cond()
    results = {}
    for name, dataset, cls_name, domain, prefix, dior_cls in EVALS:
        print(f'\n[{"="*12}] {name}', flush=True)
        if dataset == 'coco':
            images, gts = load_coco(cls_name)
        else:
            images, gts = load_dior(cls_name)
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

        n = len(prep)
        rng = random.Random(SEED)
        idx = list(range(n))
        rng.shuffle(idx)
        folds = [idx[i::NFOLD] for i in range(NFOLD)]

        test_aps, base_aps, fold_w = [], [], []
        for f in range(NFOLD):
            test_set = set(folds[f])
            train_prep = [prep[i] for i in range(n) if i not in test_set]
            train_gts = [gts[i] for i in range(n) if i not in test_set]
            test_prep = [prep[i] for i in folds[f]]
            test_gts = [gts[i] for i in folds[f]]
            # 在 train 折上调最优权重
            best_w, best_ap = None, -1.0
            for w in COMBOS:
                ap = ap_for_weights(train_prep, train_gts, w)
                if ap > best_ap:
                    best_ap, best_w = ap, dict(w)
            tuned = ap_for_weights(test_prep, test_gts, best_w)
            base = ap_for_weights(test_prep, test_gts, {'gd': 1.0})
            test_aps.append(tuned)
            base_aps.append(base)
            fold_w.append(best_w)
        mean_tuned = float(np.mean(test_aps))
        mean_base = float(np.mean(base_aps))
        delta = round(mean_tuned - mean_base, 4)
        avg_w = {k: round(float(np.mean([fw[k] for fw in fold_w])), 3) for k in FACTORS}
        print(f'  baseline(无偏)={mean_base:.4f}  tuned(无偏)={mean_tuned:.4f}  '
              f'Δ无偏={delta:+.4f}', flush=True)
        print(f'  稳健最优权重={ {k: v for k, v in avg_w.items() if v > 0} }', flush=True)
        results[name] = {'n_img': n, 'mean_baseline': mean_base, 'mean_tuned': mean_tuned,
                         'delta_unbiased': delta, 'avg_weights': avg_w,
                         'fold_weights': [{k: round(v, 2) for k, v in fw.items()} for fw in fold_w]}
    with open(SAVE_DIR / 'weight_probe_cv_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n结果 → {SAVE_DIR / "weight_probe_cv_results.json"}', flush=True)


if __name__ == '__main__':
    main()
