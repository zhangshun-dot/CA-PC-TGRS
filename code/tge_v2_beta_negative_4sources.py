# -*- coding: utf-8 -*-
r"""tge_v2_beta_negative_4sources.py — β<0 扫描（闭合 §7 的 (ii) 双向自适应校正边界）

## 为什么跑这个

原 §7 第 (ii) 条「双向自适应校正（弹性锚点框架）」记为**未解决**，关键缺口写在
`A-CSPC_四条局限盘点与两张清单_2026-09-18.md` §2：

    「`coco_to_nwpu_beta_sweep.json`（λ=0）上 β∈[0,2] 三类增益一致单调下降…
      顶点落在 β=0 边界 ⇒ **假设**最优点在负半轴。⚠️ 单调 ≠ 真峰在负半轴，只说明"可测"」

本脚本把 β 网格**向负半轴扩展**，在**冻结协议**（gdino-tiny / COCO val2017 / K_CALIB=10 /
per-image 11-point AP50 / 源模板 overhead / 目标模板 photo）与 λ 四源域脚本**同一套缓存与代码**下，
逐 (源域, 类) 单元格测出完整 β 曲线，回答两个问题：

  Q1  β 曲线在扩展网格 [−4, 4] 上有没有**内点最优**？（真峰，还是单调→边界）
  Q2  若有，最优 β 是不是**跨源域常数**？

## ★ 塌陷守卫（上一轮踩两次的坑，这次内建）

算子出口是 `sqrt(gd · clip(vis, 1e-6, 1.0))`。当 `vis ≤ 0` 对全部框成立时，clip 把它压成常数 ε，
`score` 退化成 `gd` 的单调变换 ⇒ 排序与 baseline 完全相同 ⇒ **AP 结构性地等于 baseline**。
本轮每个 (β, λ) 格都记录 `frac(vis>0)`；`frac≈0` 的格标为 **COLLAPSE**，不得当作
"β 更大所以更差"的证据（λ=1.0 列在上一份报告里正是这个问题）。

输出: 实验数据/tge_v2_beta_negative_4sources.json
运行: C:\python-study\python39\python.exe tge_v2_beta_negative_4sources.py
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json, pickle, time
import numpy as np
import torch

import unified_eval as U
import tge_probe as T
from tge_v2_lambda_4sources import (build_iou_mats, ap50_fast, build_dior_protos,
                                    build_nwpu_protos, build_dota_protos,
                                    build_visdrone_protos, build_coco_cache, prep_cell,
                                    TARGETS, SRC_MAP, COMMON4, FEAT_CACHE, LAMBDAS)
from tge_v2_main import K_CALIB, SRC_TMPL, TGT_TMPL, DATA

BETAS = [-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
LAM_SHOW = 0.5          # 曲线展示用的 λ（冻结值）
OUT = DATA / 'tge_v2_beta_negative_4sources.json'


def main():
    t0 = time.time()
    print('=' * 104)
    print('β<0 扫描：§7 (ii) 双向自适应校正边界   β∈%s   λ∈%s' % (BETAS, LAMBDAS))
    print('=' * 104)

    print('[1/3] COCO 一侧特征 (与源域无关, 一次编码) ...')
    cache = build_coco_cache()

    print('[2/3] 构建四个源域原型池 ...')
    builders = [('DIOR', build_dior_protos), ('NWPU', build_nwpu_protos),
                ('DOTA', build_dota_protos), ('VisDrone', build_visdrone_protos)]
    protos_all = {}
    for name, fn in builders:
        ts = time.time()
        protos_all[name] = fn()
        print('  %-8s 池 %2d 类   (%.1fs)' % (name, len(protos_all[name]), time.time() - ts))

    print('[3/3] β 扩展扫描 (%d 格/单元格) ...' % (len(BETAS) * len(LAMBDAS)))
    out = {'config': {'betas': BETAS, 'lambdas': LAMBDAS, 'lam_show': LAM_SHOW,
                      'detector': 'gdino-tiny', 'k_calib': K_CALIB, 'pre_filter': 0.03,
                      'note': '与 tge_v2_lambda_4sources.py 同协议同缓存；β 向负半轴扩展；'
                              '每格记录 frac(vis>0) 作为塌陷守卫'},
           'sources': {}, 'beta_curve': {}, 'verdict': {}}

    all_cells = []
    for src, _ in builders:
        protos = protos_all[src]
        rows = {}
        print('\n── 源域 %-8s (池 %d 类) ──' % (src, len(protos)))
        for eval_name, coco_cls in TARGETS:
            src_cls = SRC_MAP[src][eval_name]
            if src_cls is None or src_cls not in protos:
                continue
            boxes, gds, gts, mats, n_gt = prep_cell(cache, eval_name)
            P_neg = T.get_neg_proto(protos, src_cls).cpu().numpy()
            s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
            d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
            Delta = (d_y - s_xs).numpy()
            p_src_np = protos[src_cls].cpu().numpy()
            base50 = ap50_fast(mats, gds, n_gt)

            F = cache[eval_name]['F']
            Fc_list = [F[p['crop_idx']] if len(p['gd']) else np.zeros((0, F.shape[1]), np.float32)
                       for p in cache[eval_name]['per_img'][K_CALIB:]]
            n_box = int(sum(len(x) for x in Fc_list))

            def APf(W):
                """返回 (ΔAP, frac(vis>0)) —— 第二项是塌陷守卫。"""
                sl, npos = [], 0
                for Fc_i, gd in zip(Fc_list, gds):
                    if len(gd):
                        vis = np.maximum(0.0, Fc_i @ W)
                        npos += int((vis > 0).sum())
                        sl.append(np.sqrt(gd * np.clip(vis, 1e-6, 1.0)))
                    else:
                        sl.append(np.zeros(0))
                return ap50_fast(mats, sl, n_gt) - base50, npos / max(n_box, 1)

            grid = {}
            for beta in BETAS:
                Ph = p_src_np + beta * Delta
                nrm = np.linalg.norm(Ph)
                Ph = Ph / nrm if nrm > 1e-12 else Ph
                grid['%g' % beta] = {}
                for lam in LAMBDAS:
                    g, fr = APf(Ph - lam * P_neg)
                    grid['%g' % beta]['%g' % lam] = {'gain': round(g, 4),
                                                     'frac_vis_pos': round(fr, 4)}
            curve = {b: grid['%g' % b]['%g' % LAM_SHOW]['gain'] for b in BETAS}
            fr_curve = {b: grid['%g' % b]['%g' % LAM_SHOW]['frac_vis_pos'] for b in BETAS}
            best_beta = max(BETAS, key=lambda b: curve[b])
            interior = best_beta not in (BETAS[0], BETAS[-1])
            rows[eval_name] = {'src_cls': src_cls, 'base50': round(base50, 4),
                               'beta_curve_lam%g' % LAM_SHOW: curve,
                               'frac_vis_pos_lam%g' % LAM_SHOW: fr_curve,
                               'best_beta': best_beta, 'best_gain': curve[best_beta],
                               'gain_at_beta2': curve[2.0],
                               'gain_at_beta0': curve[0.0],
                               'interior_optimum': interior,
                               'grid': grid, 'n_test': cache[eval_name]['n_test']}
            all_cells.append((src, eval_name, best_beta, curve[best_beta], interior, base50))
            cs = ' '.join('%+g:%+.4f' % (b, curve[b]) for b in BETAS)
            print('  %-16s base=%.4f  最优β=%-5g (%+.4f)%s' %
                  (eval_name, base50, best_beta, curve[best_beta],
                   '' if interior else '   <<< 落在网格边界'))
            print('      β曲线(λ=%g) %s' % (LAM_SHOW, cs))
            print('      frac(vis>0)  %s' % ' '.join('%+g:%.2f' % (b, fr_curve[b]) for b in BETAS))
        out['sources'][src] = rows
        mean_curve = {b: round(float(np.mean([rows[c]['beta_curve_lam%g' % LAM_SHOW][b]
                                              for c in rows])), 4) for b in BETAS}
        out['beta_curve'][src] = mean_curve
        mb = max(BETAS, key=lambda b: mean_curve[b])
        print('  ★ %-8s 源域均值最优 β=%-5g (%+.4f)  β=2.0 时 %+.4f  内点=%s'
              % (src, mb, mean_curve[mb], mean_curve[2.0],
                 '是' if mb not in (BETAS[0], BETAS[-1]) else '否'))
        print('     均值曲线 %s' % ' '.join('%+g:%+.4f' % (b, mean_curve[b]) for b in BETAS))

    # ── 判决 ──
    print('\n' + '=' * 104)
    print('★ 判决')
    print('=' * 104)
    n_int = sum(1 for c in all_cells if c[4])
    print('  Q1 内点最优: %d/%d 个单元格在扩展网格 [%g, %g] 上取到内点最优'
          % (n_int, len(all_cells), BETAS[0], BETAS[-1]))
    print('     边界最优的格: %s' % (', '.join('%s/%s(β=%g)' % (c[0], c[1].replace('COCO-', ''), c[2])
                                             for c in all_cells if not c[4]) or '无'))
    src_best = {s: max(BETAS, key=lambda b: out['beta_curve'][s][b]) for s in out['beta_curve']}
    print('  Q2 逐源域均值最优 β: %s' % '  '.join('%s→%g' % (k, v) for k, v in src_best.items()))
    allpos2 = all(out['beta_curve'][s][2.0] == max(out['beta_curve'][s].values())
                  for s in out['beta_curve'])
    print('     β=2.0 是否为四源域共同峰: %s' % ('是' if allpos2 else '否'))
    neg_wins = [(s, e, b) for (s, e, b, g, i, bl) in all_cells if b < 0]
    print('     β<0 取到最优的格: %d 个 %s' % (len(neg_wins), neg_wins[:8] if neg_wins else ''))
    out['verdict'] = {'n_interior': n_int, 'n_cells': len(all_cells),
                      'src_best_beta': src_best, 'beta2_is_common_peak': bool(allpos2),
                      'n_neg_best': len(neg_wins),
                      'neg_best_cells': [list(x) for x in neg_wins]}
    json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n→ %s  (%.0fs)' % (OUT, time.time() - t0))


SRC_TMPL_STR = 'an overhead image of {}'
TGT_TMPL_STR = 'a photo of {}'

if __name__ == '__main__':
    main()