# -*- coding: utf-8 -*-
r"""tge_full_config_scan.py — COCO 全量配置扫描 (公平性决定性实验)

背景: VOC 开发域排名 (tge_dev_scan_voc.json) 与 COCO 全量表现严重不一致:
  (2.0,0.5) VOC #9 (弱) 但 COCO 最强 +0.0646;  (1.0,0.25) VOC #7 全正 但 COCO 仅 +0.0166。
  审稿人若要求 source-only 配置, (1.0,0.25)=0.3874 落后 CoOp_fs 0.4329。

本脚本: 一次加载 COCO 全量 CLIP 特征, 扫 9 个 VOC 开发域候选配置 (+ 2 个参照),
  报告每个配置的 COCO 全量 mean gain50。判定是否存在
  "VOC 开发域可选 (排名靠前) 且 COCO 上强" 的 source-only 配置。

协议: 与主表一致 (COCO val2017 全量, K_CALIB=10, 逐图 AP50, 复用 tge_v2_dets 缓存)。
运行: python tge_full_config_scan.py   (~40 分钟, 一次 CLIP 编码)
输出: 实验数据/tge_full_config_scan.json + 控制台对照表
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

K_CALIB = 10
# VOC 开发域候选 (tge_dev_scan_voc.json ranking 前 6 + 全正候选) + 参照
CONFIGS = [(4.0, 0.0), (2.0, 0.25), (0.5, 0.75), (1.0, 0.5), (1.0, 0.75), (2.0, 0.0),
           (0.5, 0.5), (0.5, 0.25), (0.5, 0.0), (1.0, 0.25), (2.0, 0.5)]
# VOC 开发域 mean gain (对照)
VOC_GAIN = {(4.0, 0.0): 0.0208, (2.0, 0.25): 0.0196, (0.5, 0.75): 0.0193,
            (1.0, 0.5): 0.0184, (1.0, 0.75): 0.0168, (2.0, 0.0): 0.0156,
            (1.0, 0.25): 0.0155, (0.5, 0.5): 0.0149, (0.5, 0.25): 0.0134,
            (0.5, 0.0): 0.0116, (2.0, 0.5): 0.0141}


def ap50(per_img, key):
    sel = [per_img[i] for i in range(K_CALIB, len(per_img))]
    return float(U.compute_ap50_img([(p['box'], p[key]) for p in sel],
                                    [p['gt'] for p in sel]))


def fused_w(per_img, F, W):
    for p in per_img:
        n = len(p['gd'])
        if n == 0:
            p['fused'] = np.zeros(0)
            continue
        vis = np.clip(np.maximum(0.0, F[p['crop_idx']] @ W), 1e-6, 1.0)
        p['fused'] = np.sqrt(np.clip(p['gd'], 1e-6, 1.0) * vis)
    return per_img


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
    print('  原型池 %d 类' % len(protos))

    print('[2/3] 加载全量缓存 + CLIP 特征 (一次) ...')
    data = {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = load_coco_all(coco_cls)
        if len(images) < K_CALIB + 10:
            continue
        dets = pickle.load(open(DATA / f'tge_v2_dets_COCO-{name}.pkl', 'rb'))
        per_img, F = collect_features(images, gts, dets)
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        Delta = d_y - s_xs
        data[name] = {'per_img': per_img, 'F': F, 'P_src': P_src, 'P_neg': P_neg, 'Delta': Delta}
        print('  %-16s 框=%d' % (name, sum(len(p['gd']) for p in per_img)))

    print('[3/3] 配置扫描 (%d 配置) ...' % len(CONFIGS))
    print('=' * 100)
    print('%-12s %8s | %8s | %s' % ('config', 'VOC', 'COCO', 'per-class gain50'))
    print('-' * 100)
    out = {}
    for beta, lam in CONFIGS:
        gains = {}
        for name, coco_cls, src_cls in TARGETS:
            d = data[name]
            per_img, F = d['per_img'], d['F']
            base = ap50(per_img, 'gd')
            P_hat = d['P_src'] + beta * d['Delta'].to(U.DEVICE)
            P_hat = P_hat / P_hat.norm()
            W = (P_hat.cpu().numpy() - lam * d['P_neg'].cpu().numpy())
            fused_w(per_img, F, W)
            a = ap50(per_img, 'fused')
            gains[name] = round(a - base, 4)
        mg = round(float(np.mean(list(gains.values()))), 4)
        out[f'b{beta}_l{lam}'] = {'coco_mean_gain': mg, 'per_class': gains}
        vg = VOC_GAIN.get((beta, lam))
        vg_s = ('%+.4f' % vg) if vg is not None else '   -  '
        print('%-12s %8s | %+8.4f | %s' %
              (f'({beta}, {lam})', vg_s, mg,
               ' '.join('%+5.3f' % gains[t[0]] for t in TARGETS)))
    print('=' * 100)

    # 判定: source-only 可选 (VOC 排名前 6 或全正) 且 COCO 强
    print('\n判定 (source-only 候选在 COCO 上的表现):')
    ranked = sorted(out.items(), key=lambda kv: -kv[1]['coco_mean_gain'])
    for k, v in ranked:
        tag = '  <- 参照' if k in ('b2.0_l0.5', 'b1.0_l0.25') else ''
        print('  %-10s COCO=%+.4f%s' % (k, v['coco_mean_gain'], tag))
    best_src = [k for k, v in ranked if k not in ('b2.0_l0.5', 'b1.0_l0.25')][0]
    print('\n  最强 source-only 候选: %s  COCO gain=%+.4f' %
          (best_src, out[best_src]['coco_mean_gain']))
    print('  与 (2.0,0.5) %+.4f 对比' % out['b2.0_l0.5']['coco_mean_gain'])

    with open(DATA / 'tge_full_config_scan.json', 'w', encoding='utf-8') as f:
        json.dump({'configs': out,
                   'voc_gain_ref': {f'b{k[0]}_l{k[1]}': v for k, v in VOC_GAIN.items()}},
                  f, indent=2, ensure_ascii=False)
    print('\n→ 实验数据/tge_full_config_scan.json   (耗时 %.1f 分钟)' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
