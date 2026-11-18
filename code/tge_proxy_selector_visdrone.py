# -*- coding: utf-8 -*-
r"""tge_proxy_selector_visdrone.py — 补齐代理选择器的 VisDrone 格 (13/24 类缺的最后一个域)

目标: 回答"决策选择器能不能设计"——VisDrone 是唯一"外推有害"的目标域,
现有 tge_proxy_selector.json (13类) / tge_proxy_selector_24.json (24类) 都没有它。
本脚本为 DIOR→VisDrone 4 类 (car/truck/bus/motor, 粗到细 vehicle 映射) 补算:
  1) 目标无关代理信号 (agree/sep_src/gd_hi_frac, 与 13/24 类同协议)
  2) 逐类增益 gain = ext − B2 (与 tge_probe_gate 同定义, 可直接并入 gain_map)

协议 (对齐 tge_dccs_grid.py 的 VisDrone 段):
  load_visdrone(cls, n=20) → gdino_detect(TGT_TMPL, th=0) → collect_features
  B2  = sqrt(gd * max(0, f·P_src − λ·f·P_neg))       λ = U.NEG_LAM = 0.5
  ext = sqrt(gd * max(0, f·norm(P_src+2Δ) − λ·f·P_neg))  Δ = d(y)−s(x_s), 源 DIOR-vehicle

运行: python tge_proxy_selector_visdrone.py   (~5-10 分钟, 检测 4×20 图)
输出: 实验数据/tge_proxy_selector_visdrone.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau, entropy
import unified_eval as U
import tge_probe as T
import tge_v2_visdrone as VD
from tge_v2_main import (clip_proto, gt_crops, collect_features, fused_arrays,
                         SRC_TMPL, TGT_TMPL, DATA, PRE_FILTER)

BETA = 2.0
LAM = U.NEG_LAM  # 0.5
VIS_TARGETS = [('VD-car', 'car', 'vehicle'), ('VD-truck', 'truck', 'vehicle'),
               ('VD-bus', 'bus', 'vehicle'), ('VD-motor', 'motor', 'vehicle')]


def proxy_from_perimg(per_img, F, P_src):
    """与 tge_proxy_selector.py 完全相同的代理定义。"""
    gds, srcs, n_boxes, n_small = [], [], 0, 0
    for p in per_img:
        n = len(p['gd'])
        if n:
            gds.append(p['gd'])
            srcs.append(F[p['crop_idx']] @ P_src.cpu().numpy())
            if p['box'].size:
                areas = (p['box'][:, 2] - p['box'][:, 0]) * (p['box'][:, 3] - p['box'][:, 1])
                n_small += int(np.sum(areas < 32 * 32))
        n_boxes += n
    if not gds or sum(len(g) for g in gds) < 10:
        return None
    gd = np.concatenate(gds)
    src = np.concatenate(srcs)
    rho, _ = spearmanr(gd, src)
    tau, _ = kendalltau(gd, src)
    return {'agree_spearman': round(float(rho), 4), 'agree_kendall': round(float(tau), 4),
            'sep_src': round(float(src.std()), 4),
            'gd_hi_frac': round(float(np.mean(gd > 0.3)), 4),
            'small_frac': round(n_small / max(1, n_boxes), 4),
            'n_boxes': int(n_boxes)}


def main():
    import time
    t0 = time.time()
    print('[1/3] DIOR 20 类 CLIP 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p

    print('[2/3] VisDrone 4 类: 检测 + 代理 + gain ...')
    proxy, gains = {}, {}
    for name, vis_cls, src_cls in VIS_TARGETS:
        images, gts = VD.load_visdrone(vis_cls)
        if len(images) < 10:
            print('  !! %s 图太少 (%d), 跳过' % (vis_cls, len(images)))
            continue
        dets = U.gdino_detect(images, TGT_TMPL.format(vis_cls), 0.0)
        per_img, F = collect_features(images, gts, dets)
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(vis_cls)]).cpu()[0]
        Delta = d_y - s_xs

        pr = proxy_from_perimg(per_img, F, P_src)
        if pr is None:
            print('  !! %s 代理框不足, 跳过' % name)
            continue
        proxy[name] = pr

        # AP50 (与 gate 同定义: gain = ext − B2, 相对 gd 基线)
        def ap50_of(W):
            fused_arrays(per_img, F, W)
            return U.compute_ap50_img([(p['box'], p['fused']) for p in per_img],
                                      [p['gt'] for p in per_img])

        a_base = U.compute_ap50_img([(p['box'], p['gd']) for p in per_img],
                                    [p['gt'] for p in per_img])
        W_b2 = (P_src.cpu().numpy() - LAM * P_neg.cpu().numpy())
        a_b2 = ap50_of(W_b2)
        P_hat = P_src + BETA * Delta.to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        W_ext = (P_hat.cpu().numpy() - LAM * P_neg.cpu().numpy())
        a_ext = ap50_of(W_ext)
        gains[name] = {'base': round(float(a_base), 4), 'b2': round(float(a_b2), 4),
                       'ext': round(float(a_ext), 4), 'gain': round(float(a_ext - a_b2), 4),
                       'n_img': len(images)}
        print('  %-10s base=%.4f B2=%.4f ext=%.4f  gain(ext−B2)=%+.4f  '
              'agree=%.3f sep=%.3f small=%.3f  n=%d' %
              (name, a_base, a_b2, a_ext, a_ext - a_b2,
               pr['agree_spearman'], pr['sep_src'], pr['small_frac'], pr['n_boxes']))

    out = {'config': {'protocol': 'v3 proxy 补齐: DIOR→VisDrone 4 类 (N=20, vehicle 粗到细), '
                                  '代理与 gain 均与 13/24 类同协议', 'beta': BETA, 'lam': LAM},
           'proxy': proxy, 'gain_map': {n: gains[n]['gain'] for n in gains},
           'gains': gains}
    with open(DATA / 'tge_proxy_selector_visdrone.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n→ tge_proxy_selector_visdrone.json   (耗时 %.1f 分钟)' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
