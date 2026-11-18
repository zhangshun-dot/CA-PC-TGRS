# -*- coding: utf-8 -*-
r"""tge_randk_sensitivity.py — 随机 K=10 切分敏感性 (评审意见 §5: 排除按标注顺序取 K 的顺序偏差)

主表前 K=10 校准图按标注顺序取 (airplane 97/boat 121/car 535/truck 250/bus 189/moto 159 张
的前 10 张), 测试集 = images[K:]。本脚本对 N_RAND 个随机种子各做一次随机 K=10 切分
(从全量含 GT 图中均匀采样 10 张作校准, 其余作测试), 在主配置 (β=2.0, λ=0.5) 下重算
每类 base/ext AP 与增益, 报告 5 次切分的增益分布, 检验主结论 (+0.0646 6/6 正) 是否
对 K 切分顺序稳定。

协议: 复用 tge_v2_dets_COCO-*.pkl 全量检测缓存, 逐图 AP50, 与主表同口径。
运行: python tge_randk_sensitivity.py   (~15-25 分钟, 一次 CLIP 编码)
输出: 实验数据/tge_randk_sensitivity.json + 控制台表
"""
import sys, json, pickle
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T
from tge_v2_main import (load_coco_all, clip_proto, gt_crops, collect_features,
                         TARGETS, SRC_TMPL, TGT_TMPL, DATA)

BETA, LAM = 2.0, 0.5
K = 10
N_RAND = 5
SEEDS = [1, 7, 42, 2026, 2027]


def ap50_on(per_img, idx, key):
    sel = [per_img[i] for i in idx]
    return float(U.compute_ap50_img([(p['box'], p[key]) for p in sel],
                                    [p['gt'] for p in sel]))


def main():
    import time
    t0 = time.time()
    print('[1/3] DIOR 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p

    print('[2/3] 加载全量缓存 + CLIP 特征 (一次) ...')
    data = {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = load_coco_all(coco_cls)
        if len(images) < K + 10:
            continue
        dets = pickle.load(open(DATA / f'tge_v2_dets_COCO-{name}.pkl', 'rb'))
        per_img, F = collect_features(images, gts, dets)
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        P_hat = P_src + BETA * (d_y - s_xs).to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        W = (P_hat.cpu().numpy() - LAM * P_neg.cpu().numpy())
        # 预计算 fused (与 K 无关, 测试图集由 idx 决定)
        for p in per_img:
            n = len(p['gd'])
            if n == 0:
                p['fused'] = np.zeros(0)
                continue
            vis = np.clip(np.maximum(0.0, F[p['crop_idx']] @ W), 1e-6, 1.0)
            p['fused'] = np.sqrt(np.clip(p['gd'], 1e-6, 1.0) * vis)
        data[name] = per_img
        print('  %-16s 图=%d 框=%d' % (name, len(per_img), sum(len(p['gd']) for p in per_img)))

    print('[3/3] 随机 K=10 切分 × %d seeds ...' % N_RAND)
    print('=' * 100)
    print('%-12s | %s' % ('seed', ' '.join('%-14s' % n for n, _, _ in TARGETS) + '   mean'))
    print('-' * 100)
    out = {}
    all_means = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        row = {}
        for name, coco_cls, _ in TARGETS:
            per_img = data[name]
            n = len(per_img)
            calib = np.sort(rng.choice(n, K, replace=False))
            test = np.array([i for i in range(n) if i not in calib])
            base = ap50_on(per_img, test, 'gd')
            ext = ap50_on(per_img, test, 'fused')
            row[name] = round(ext - base, 4)
        mg = round(float(np.mean(list(row.values()))), 4)
        all_means.append(mg)
        out[str(seed)] = {'per_class': row, 'mean_gain': mg}
        print('%-12s | %s  %+6.4f' %
              (seed, ' '.join('%+14.4f' % row[n] for n, _, _ in TARGETS), mg))
    print('-' * 100)
    print('均值增益分布: min=%.4f max=%.4f mean=%.4f std=%.4f' %
          (min(all_means), max(all_means), np.mean(all_means), np.std(all_means)))
    print('全部为正: %s | 6/6 类全正次数: %s' %
          (all(m > 0 for m in all_means),
           sum(1 for s in out if all(v > 0 for v in out[s]['per_class'].values()))))
    print('对照 (标注顺序 K): +0.0646, 6/6 正')

    with open(DATA / 'tge_randk_sensitivity.json', 'w', encoding='utf-8') as f:
        json.dump({'seeds': SEEDS, 'per_seed': out,
                   'mean_dist': {'min': min(all_means), 'max': max(all_means),
                                 'mean': float(np.mean(all_means)), 'std': float(np.std(all_means))},
                   'main_table_ref': 0.0646}, f, indent=2, ensure_ascii=False)
    print('\n→ 实验数据/tge_randk_sensitivity.json   (耗时 %.1f 分钟)' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
