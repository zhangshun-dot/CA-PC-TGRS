# -*- coding: utf-8 -*-
r"""tge_v2_lambda_4sources.py — λ 的四源域交叉验证 (闭合 B3「信息代价」边界)

目的：主表的 (β=2.0, λ=0.5) 是否只是 DIOR 源域上的一个好运气？
在四个**独立遥感源域**上逐类扫描 λ，检验 λ=0.5 是否为跨源域一致峰值。
  源域: DIOR(20 类池) / NWPU VHR-10(10 类池) / DOTA v1.5(16 类池+vehicle 合并) / VisDrone(10 类池)
  目标: COCO val2017 全量 6 类 (VisDrone 源域仅覆盖 4 类: car/truck/bus/motorcycle)
  协议: 完全沿用 tge_v2 冻结协议 (gdino-tiny, pre_filter=0.03, K_CALIB=10, per-image AP50,
        源模板 'an overhead image of {}', 目标模板 'a photo of {}')

效率要点：**检测框与 CLIP 特征与源域无关**（检测 prompt 只用目标类名），
因此 COCO 一侧 per_img + F 只编码一次并缓存；源域之间只差原型池、负原型与文本方向。
全流程待编码框 ≈ 3.06 万，固件 + 550 次 AP 评估，总耗时约 5 分钟。

同时给出 β×λ 二维扫描 (β∈{0,1,2,3,4})，检查 λ 峰是否随 β 漂移。

自检：DIOR 源域那一列必须复现 tge_v2_perclass_lambda_scan.json（同协议同缓存）。

输出: 实验数据/tge_v2_lambda_4sources.json
运行: C:\python-study\python39\python.exe tge_v2_lambda_4sources.py
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json, pickle, time
import numpy as np
import torch

import unified_eval as U
import tge_probe as T
import tge_probe_nwpu as NW
import tge_probe_dota_src as DS
from tge_v2_main import (load_coco_all, clip_proto, gt_crops, collect_features,
                         v2_cached_dets, K_CALIB, SRC_TMPL, TGT_TMPL, DATA)

BETA_FIXED = 2.0
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
BETAS = [0.0, 1.0, 2.0, 3.0, 4.0]
FEAT_CACHE = DATA / 'tge_v2_coco_feats_6cls.pkl'

TARGETS = [('COCO-airplane', 'airplane'), ('COCO-boat', 'boat'),
           ('COCO-car', 'car'), ('COCO-truck', 'truck'),
           ('COCO-bus', 'bus'), ('COCO-motorcycle', 'motorcycle')]

SRC_MAP = {
    'DIOR':     {'COCO-airplane': 'airplane', 'COCO-boat': 'ship',
                 'COCO-car': 'vehicle', 'COCO-truck': 'vehicle',
                 'COCO-bus': 'vehicle', 'COCO-motorcycle': 'vehicle'},
    'NWPU':     {'COCO-airplane': 'airplane', 'COCO-boat': 'ship',
                 'COCO-car': 'vehicle', 'COCO-truck': 'vehicle',
                 'COCO-bus': 'vehicle', 'COCO-motorcycle': 'vehicle'},
    'DOTA':     {'COCO-airplane': 'plane', 'COCO-boat': 'ship',
                 'COCO-car': 'vehicle', 'COCO-truck': 'vehicle',
                 'COCO-bus': 'vehicle', 'COCO-motorcycle': 'vehicle'},
    'VisDrone': {'COCO-airplane': None, 'COCO-boat': None,
                 'COCO-car': 'car', 'COCO-truck': 'truck',
                 'COCO-bus': 'bus', 'COCO-motorcycle': 'motor'},
}
COMMON4 = ['COCO-car', 'COCO-truck', 'COCO-bus', 'COCO-motorcycle']


# ───────────────────── 与 unified_eval.compute_ap50_img 逐行同构的快速版 ─────────────────────

def build_iou_mats(boxes_list, gts_list):
    mats = []
    for b, g in zip(boxes_list, gts_list):
        b = np.asarray(b, dtype=np.float64); g = np.asarray(g, dtype=np.float64)
        if len(b) == 0 or len(g) == 0:
            mats.append(np.zeros((len(b), len(g)))); continue
        x1 = np.maximum(b[:, None, 0], g[None, :, 0]); y1 = np.maximum(b[:, None, 1], g[None, :, 1])
        x2 = np.minimum(b[:, None, 2], g[None, :, 2]); y2 = np.minimum(b[:, None, 3], g[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        a1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        a2 = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
        mats.append(inter / (a1[:, None] + a2[None, :] - inter + 1e-8))
    return mats


def ap50_fast(mats, scores_list, n_gt):
    """与 compute_ap50_img 同算法的加速版（IoU 矩阵预算，其余逻辑逐行照搬）。"""
    if n_gt == 0:
        return 0.0 if any(len(s) for s in scores_list) else 1.0
    evals = []
    for M, s in zip(mats, scores_list):
        if len(s) == 0:
            continue
        s = np.asarray(s)
        matched = [False] * M.shape[1]
        for i in np.argsort(-s):
            best_iou, best_j = 0.0, -1
            for j in range(M.shape[1]):
                if not matched[j] and M[i, j] > best_iou:
                    best_iou, best_j = M[i, j], j
            if best_iou >= 0.5 and best_j >= 0:
                evals.append((float(s[i]), 1)); matched[best_j] = True
            else:
                evals.append((float(s[i]), 0))
    if not evals:
        return 0.0
    evals.sort(key=lambda x: -x[0])
    tp = np.array([e[1] for e in evals])
    tc, fc = np.cumsum(tp), np.cumsum(1 - tp)
    rec = tc / n_gt
    prec = tc / (tc + fc + 1e-8)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        m = rec >= t
        if m.any():
            ap += float(prec[m].max())
    return ap / 11.0


# ───────────────────────── 源域原型池 ─────────────────────────

def build_dior_protos():
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p
    return protos


def build_nwpu_protos():
    protos = {}
    for cls in NW.NWPU_CLASSES:
        images, gts = NW.load_nwpu(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p
    return protos


def build_dota_protos():
    protos = {}
    for cls in DS.DOTA_CLASSES:
        images, gts = DS.load_dota(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p
    imgs_v, gts_v = [], []
    for vc in ('large-vehicle', 'small-vehicle'):
        im2, gt2 = DS.load_dota(vc)
        imgs_v += im2; gts_v += gt2
    protos['vehicle'] = clip_proto(gt_crops(imgs_v, gts_v))
    return protos


def build_visdrone_protos():
    from tge_v2_visdrone import load_visdrone, VISDRONE_CLASSES
    protos = {}
    for cls in VISDRONE_CLASSES:
        images, gts = load_visdrone(cls)
        if len(images) == 0:
            print('    VisDrone 类 %s 无数据, 跳过' % cls); continue
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p
    return protos


# ───────────────────────── COCO 一侧：一次编码并缓存 ─────────────────────────

def build_coco_cache():
    if FEAT_CACHE.exists():
        print('  [缓存] %s' % FEAT_CACHE.name)
        return pickle.load(open(FEAT_CACHE, 'rb'))
    cache = {}
    for eval_name, coco_cls in TARGETS:
        images, gts = load_coco_all(coco_cls)
        dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), eval_name)
        per_img, F = collect_features(images, gts, dets)
        cache[eval_name] = {'per_img': per_img, 'F': F.astype(np.float32),
                            'n_test': len(images) - K_CALIB,
                            'n_box': int(sum(len(p['gd']) for p in per_img))}
        print('  %-16s N_test=%4d 框=%6d' % (eval_name, cache[eval_name]['n_test'],
                                             cache[eval_name]['n_box']))
    pickle.dump(cache, open(FEAT_CACHE, 'wb'))
    print('  → 已缓存 %s (%.0f MB)' % (FEAT_CACHE.name, FEAT_CACHE.stat().st_size / 1e6))
    return cache


def prep_cell(cache, eval_name):
    d = cache[eval_name]
    sel = d['per_img'][K_CALIB:]
    boxes = [p['box'] for p in sel]
    gds = [p['gd'] for p in sel]
    gts = [p['gt'] for p in sel]
    mats = build_iou_mats(boxes, gts)
    n_gt = int(sum(len(g) for g in gts))
    return boxes, gds, gts, mats, n_gt


# ───────────────────────── 主流程 ─────────────────────────

def main():
    t0 = time.time()
    print('=' * 100)
    print('λ 的四源域交叉验证 (B3 边界闭合)   β=%.1f 固定   λ∈%s   β网格=%s'
          % (BETA_FIXED, LAMBDAS, BETAS))
    print('=' * 100)

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

    print('[3/3] λ 扫描 + β×λ 网格 ...')
    out = {'config': {'beta_fixed': BETA_FIXED, 'lambdas': LAMBDAS, 'betas': BETAS,
                      'detector': 'gdino-tiny', 'k_calib': K_CALIB, 'pre_filter': 0.03,
                      'protocol': 'v2 冻结协议: COCO val2017 全量, per-image AP50, '
                                  'per-image AP50 (11-point), 源模板 overhead / 目标模板 photo'},
           'sources': {}, 'beta_lambda_gain': {}, 'headline': {}}

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
            P_src = protos[src_cls].to(U.DEVICE)
            P_neg = T.get_neg_proto(protos, src_cls).cpu().numpy()
            s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
            d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
            Delta = (d_y - s_xs).numpy()
            p_src_np = P_src.cpu().numpy()

            base50 = ap50_fast(mats, gds, n_gt)

            # F 预索引一次, 避免组合扫描里重复 fancy-index
            F = cache[eval_name]['F']
            Fc_list = [F[p['crop_idx']] if len(p['gd']) else np.zeros((0, F.shape[1]), np.float32)
                       for p in cache[eval_name]['per_img'][K_CALIB:]]

            def APf(W):
                sl = []
                for Fc_i, gd in zip(Fc_list, gds):
                    n = len(gd)
                    if n:
                        vis = np.maximum(0.0, Fc_i @ W)
                        sl.append(np.sqrt(gd * np.clip(vis, 1e-6, 1.0)))
                    else:
                        sl.append(np.zeros(0))
                return ap50_fast(mats, sl, n_gt)

            bl_grid = {}
            for beta in BETAS:
                Ph = p_src_np + beta * Delta
                Ph = Ph / np.linalg.norm(Ph)
                bl_grid['%g' % beta] = {}
                for lam in LAMBDAS:
                    bl_grid['%g' % beta]['%g' % lam] = round(APf(Ph - lam * P_neg) - base50, 4)

            lam_res = {('%g' % l): {'gain': bl_grid['%g' % BETA_FIXED]['%g' % l]}
                       for l in LAMBDAS}
            best_lam = max(LAMBDAS, key=lambda l: lam_res['%g' % l]['gain'])
            rows[eval_name] = {'src_cls': src_cls, 'base50': round(base50, 4),
                               'lambda_scan': {k: {'gain': v['gain']} for k, v in lam_res.items()},
                               'best_lambda': best_lam,
                               'best_gain': lam_res['%g' % best_lam]['gain'],
                               'fixed_lam05_gain': lam_res['0.5']['gain'],
                               'n_test': cache[eval_name]['n_test'],
                               'beta_lambda_gain': bl_grid}
            all_cells.append((src, eval_name, best_lam,
                              rows[eval_name]['best_gain'], rows[eval_name]['fixed_lam05_gain'],
                              base50))
            gs = '  '.join('λ=%g:%+.4f' % (l, lam_res['%g' % l]['gain']) for l in LAMBDAS)
            print('  %-16s base=%.4f  %s | 最优λ=%-5g (%+.4f)%s' %
                  (eval_name, base50, gs, best_lam, lam_res['%g' % best_lam]['gain'],
                   '' if best_lam == 0.5 else '   <<< 偏离 0.5'))
        out['sources'][src] = rows
        # 源域均值网格
        mean_grid = {}
        for b in BETAS:
            mean_grid['%g' % b] = {}
            for l in LAMBDAS:
                vals = [rows[c]['beta_lambda_gain']['%g' % b]['%g' % l]
                        for c in rows if c in rows]
                mean_grid['%g' % b]['%g' % l] = round(float(np.mean(vals)), 4)
        out['beta_lambda_gain'][src] = mean_grid
        best_bl = max(((b, l) for b in BETAS for l in LAMBDAS),
                      key=lambda bl: mean_grid['%g' % bl[0]]['%g' % bl[1]])
        print('  %-8s β×λ 最优 = (β=%g, λ=%g)  均值增益 %+.4f' %
              (src, best_bl[0], best_bl[1], mean_grid['%g' % best_bl[0]]['%g' % best_bl[1]]))
        for b in BETAS:
            print('     β=%-3g ' % b + '  '.join('λ=%g:%+.4f' % (l, mean_grid['%g' % b]['%g' % l])
                                                 for l in LAMBDAS))

    n_cell = len(all_cells)
    n_05 = sum(1 for c in all_cells if c[2] == 0.5)
    out['headline'] = {
        'n_cells': n_cell, 'n_best_is_05': n_05, 'all_best_is_05': n_05 == n_cell,
        'mean_gain_at_05': round(float(np.mean([c[4] for c in all_cells])), 4),
        'mean_gain_at_best': round(float(np.mean([c[3] for c in all_cells])), 4),
        'improvement_if_perclass_oracle': round(
            float(np.mean([c[3] for c in all_cells]) - np.mean([c[4] for c in all_cells])), 4),
    }
    common = {s: round(float(np.mean([out['sources'][s][c]['fixed_lam05_gain']
                                      for c in COMMON4 if c in out['sources'][s]])), 4)
              for s in out['sources']
              if all(c in out['sources'][s] for c in COMMON4)}
    out['common4_mean_gain_at_lam05'] = common

    print('\n' + '=' * 100)
    print('%-9s %-16s %8s %11s %8s %10s' % ('源域', '目标类', 'base', 'λ=0.5增益', '最优λ', '最优增益'))
    print('-' * 100)
    for src in out['sources']:
        for k, r in out['sources'][src].items():
            print('%-9s %-16s %8.4f %+11.4f %8g %+10.4f' %
                  (src, k, r['base50'], r['fixed_lam05_gain'], r['best_lambda'], r['best_gain']))
    print('-' * 100)
    print('单元格 %d 个, 最优 λ 恰为 0.5 的有 %d 个  →  %s' %
          (n_cell, n_05, '全部一致 OK' if n_05 == n_cell else '存在偏离 NG'))
    print('λ=0.5 平均增益 %+.4f ; 逐类最优 λ(若可自由选) 平均增益 %+.4f ; 差 %+.4f' %
          (out['headline']['mean_gain_at_05'], out['headline']['mean_gain_at_best'],
           out['headline']['improvement_if_perclass_oracle']))
    print('共同 4 类 (car/truck/bus/motorcycle) 在 λ=0.5 下的逐源域均值增益:')
    for src, v in common.items():
        print('   %-9s %+.4f' % (src, v))
    print('=' * 100)

    # ── 自检: DIOR 列必须复现 tge_v2_perclass_lambda_scan.json ──
    ref_p = DATA / 'tge_v2_perclass_lambda_scan.json'
    if ref_p.exists():
        ref = json.load(open(ref_p, encoding='utf-8'))['results']
        print('\n[自检] DIOR 列 vs tge_v2_perclass_lambda_scan.json')
        bad = 0
        for k, v in ref.items():
            mine = out['sources']['DIOR'].get(k)
            if mine is None:
                print('   ✗ 缺失 %s' % k); bad += 1; continue
            d05 = abs(mine['fixed_lam05_gain'] - v['fixed_lam05_gain'])
            flag = 'OK' if d05 <= 0.002 else 'NG'
            if d05 > 0.002:
                bad += 1
            print('   %-16s 参考λ=0.5 %+.4f | 本次 %+.4f  Δ=%.4f  %s' %
                  (k, v['fixed_lam05_gain'], mine['fixed_lam05_gain'], d05, flag))
        print('   自检结论: %s' % ('全部通过' if bad == 0 else '%d 项超差' % bad))

    out_path = DATA / 'tge_v2_lambda_4sources.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n→ %s   (耗时 %.1f 分钟)' % (out_path.name, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
