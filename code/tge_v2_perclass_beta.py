# -*- coding: utf-8 -*-
r"""tge_v2_perclass_beta.py — 逐类 β 实验

基于 tge_v2_main.py, 将 A-CSPC 的全局 β 替换为逐类 β(c)。
支持预设方案 (scheme_A / scheme_B / oracle) 或自定义 JSON 映射。

基线对比: 固定 β=2.0, λ=0.5 → tge_v2_main_b2_l0.5.json (总增益 +0.0646)

运行:
  python tge_v2_perclass_beta.py --scheme scheme_a --lam 0.5
  python tge_v2_perclass_beta.py --scheme oracle --lam 0.5
  python tge_v2_perclass_beta.py --per-class-beta '{"COCO-airplane":1.0,...}' --lam 0.5

输出: 实验数据/tge_v2_perclass_beta_<scheme>.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
import argparse
from pathlib import Path
import numpy as np
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# ── 主实验配置 (β=2.0, λ=0.5 对应 tge_v2_main_b2_l0.5) ──
DEFAULT_BETA = 2.0
DEFAULT_LAM = 0.5
K_CALIB = 10
R_BOOT = 1000
SEEDS = [42, 20260823, 7]
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
DETECTOR = 'gdino-tiny'
PRE_FILTER = 0.03

TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]

# ── 预设逐类 β 方案 ──
SCHEMES = {
    'fixed': {  # 固定 β=2.0 (复现基线)
        'COCO-airplane': 2.0, 'COCO-boat': 2.0, 'COCO-car': 2.0,
        'COCO-truck': 2.0, 'COCO-bus': 2.0, 'COCO-motorcycle': 2.0,
    },
    'scheme_a': {  # 离散三级: 高风险β=3, 中风险β=2, 低风险β=1
        'COCO-airplane': 1.0, 'COCO-boat': 1.0, 'COCO-car': 2.0,
        'COCO-truck': 2.0, 'COCO-bus': 3.0, 'COCO-motorcycle': 3.0,
    },
    'scheme_b': {  # 连续映射 β = 1.0 + 3.0*R_norm*sat_factor
        'COCO-airplane': 1.00, 'COCO-boat': 2.26, 'COCO-car': 2.79,
        'COCO-truck': 2.83, 'COCO-bus': 2.89, 'COCO-motorcycle': 3.99,
    },
    'oracle': {  # Oracle: 基于实际增益排序选β (上界, 推理时不可用)
        'COCO-airplane': 0.0, 'COCO-boat': 1.5, 'COCO-car': 3.5,
        'COCO-truck': 2.5, 'COCO-bus': 2.5, 'COCO-motorcycle': 3.5,
    },
}


# ── 数据 (与 tge_v2_main.py 相同) ──

def load_coco_all(cls_name):
    return U.load_coco(cls_name, n=100000)


def clip_proto(crops):
    if not crops:
        return None
    F = T.encode_clip_batch(crops)
    p = F.mean(dim=0)
    return p / p.norm()


def gt_crops(images, gts):
    crops = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
    return crops


def v2_cached_dets(images, prompt, name):
    p = DATA / f'tge_v2_dets_COCO-{name}.pkl'
    if p.exists():
        print(f'    [缓存] 检测结果 {p.name}')
        return pickle.load(open(p, 'rb'))
    dets = U.gdino_detect(images, prompt, 0.0)
    pickle.dump(dets, open(p, 'wb'))
    return dets


def collect_features(images, gts, dets):
    per_img = []
    crops_all = []
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < PRE_FILTER:
                continue
            bl.append([x1, y1, x2, y2]); sl.append(float(sc)); ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
                        'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
                        'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
                        'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64)})
    if crops_all:
        F = T.encode_clip_batch(crops_all).cpu().numpy()
    else:
        F = np.zeros((0, 512))
    return per_img, F


def fused_arrays(per_img, F, W):
    for p in per_img:
        n = len(p['gd'])
        if n:
            vis = np.maximum(0.0, F[p['crop_idx']] @ W)
            p['fused'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p['fused'] = np.zeros(0)
    return per_img


def dataset_ap(per_img, key, coco_style=False):
    sel = [per_img[i] for i in range(K_CALIB, len(per_img))]
    preds = [(p['box'], p[key]) for p in sel]
    gts = [p['gt'] for p in sel]
    if coco_style:
        return U.compute_ap_coco(preds, gts)
    return U.compute_ap50_img(preds, gts)


def recall_proposals(per_img):
    n_gt, hit = 0, 0
    for p in per_img:
        gts, boxes = p['gt'], p['box']
        n_gt += len(gts)
        if len(boxes) == 0:
            continue
        matched = np.zeros(len(gts), dtype=bool)
        for b in boxes:
            if matched.all():
                break
            for j, gb in enumerate(gts):
                if matched[j]:
                    continue
                if U.iou(b.tolist(), gb.tolist()) >= 0.5:
                    matched[j] = True
        hit += int(matched.sum())
    return hit / n_gt if n_gt else float('nan')


# ── 主流程 ──

def main():
    import time
    t0 = time.time()
    ap = argparse.ArgumentParser(description='A-CSPC 逐类 β 实验')
    ap.add_argument('--scheme', type=str, default='scheme_a',
                    choices=list(SCHEMES.keys()),
                    help='预设逐类β方案')
    ap.add_argument('--per-class-beta', type=str, default=None,
                    help='自定义JSON映射, 如 \'{"COCO-airplane":1.0,...}\' (覆盖--scheme)')
    ap.add_argument('--lam', type=float, default=DEFAULT_LAM)
    args = ap.parse_args()

    # 确定逐类 β 映射
    if args.per_class_beta:
        per_class_beta = json.loads(args.per_class_beta)
        scheme_name = 'custom'
    else:
        per_class_beta = SCHEMES[args.scheme]
        scheme_name = args.scheme

    lam = args.lam
    tag = '%s_l%g' % (scheme_name, lam)
    out_path = DATA / ('tge_v2_perclass_beta_%s.json' % tag)

    print('=' * 80)
    print('逐类 β 实验: scheme=%s, λ=%.2f' % (scheme_name, lam))
    print('逐类 β 映射:')
    for name, _, _ in TARGETS:
        b = per_class_beta.get(name, DEFAULT_BETA)
        print('  %-16s β=%.2f' % (name, b))
    print('→ 输出 %s' % out_path.name)
    print('=' * 80)

    print('[1/3] DIOR 20 类 CLIP 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p

    print('[2/3] COCO 6 类全量检测 + 特征 ...')
    class_data = []
    for name, coco_cls, src_cls in TARGETS:
        images, gts = load_coco_all(coco_cls)
        if len(images) < K_CALIB + 10:
            print('  !! %s 图太少 (%d), 跳过' % (name, len(images)))
            continue
        dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), name)
        per_img, F = collect_features(images, gts, dets)
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        Delta = d_y - s_xs
        class_data.append({'name': name, 'per_img': per_img, 'F': F, 'gts': gts,
                           'P_src': P_src, 'P_neg': P_neg, 'Delta': Delta,
                           'd_y': d_y, 'coco_cls': coco_cls,
                           'n_pool': len(images), 'n_test': len(images) - K_CALIB})
        n_box = sum(len(p['gd']) for p in per_img)
        print('  %-16s N_pool=%3d N_test=%3d 框=%6d' % (name, len(images),
                                                        len(images) - K_CALIB, n_box))

    print('[3/3] 打分 + AP + bootstrap ...')
    rows = {}
    for d in class_data:
        P_src, P_neg = d['P_src'], d['P_neg']
        beta_c = per_class_beta.get(d['name'], DEFAULT_BETA)

        # baseline (gd)
        base50 = dataset_ap(d['per_img'], 'gd')
        base_coco = dataset_ap(d['per_img'], 'gd', coco_style=True)

        # A-CSPC with per-class β: P̂ = norm(P_src + β(c)·Δ), λ=lam
        if beta_c <= 0:
            # β=0: 跳过校正, 直接用 P_src (等价于不做方向外推)
            P_hat = P_src / P_src.norm()
        else:
            P_hat = P_src + beta_c * d['Delta'].to(U.DEVICE)
            P_hat = P_hat / P_hat.norm()
        W_c = (P_hat.cpu().numpy() - lam * P_neg.cpu().numpy())
        fused_arrays(d['per_img'], d['F'], W_c)
        ours_50 = dataset_ap(d['per_img'], 'fused')
        ours_coco = dataset_ap(d['per_img'], 'fused', coco_style=True)
        rec = recall_proposals(d['per_img'])

        rows[d['name']] = {'n_pool': d['n_pool'], 'n_test': d['n_test'],
                           'beta': beta_c,
                           'recall': round(rec, 4),
                           'base50': round(base50, 4),
                           'ours50': round(ours_50, 4),
                           'gain50': round(ours_50 - base50, 4),
                           'base_coco': round(base_coco, 4),
                           'ours_coco': round(ours_coco, 4),
                           'gain_coco': round(ours_coco - base_coco, 4)}
        print('  %-16s β=%.2f  base50=%.4f ours50=%.4f (%+.4f)' %
              (d['name'], beta_c, base50, ours_50, ours_50 - base50))

    # bootstrap (按图配对, 3 种子)
    boot = {}
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        gains_mean = []
        for _ in range(R_BOOT):
            dsum = 0.0
            for d in class_data:
                n_t = len(d['per_img']) - K_CALIB
                idx = rng.randint(K_CALIB, len(d['per_img']), size=n_t)
                sel = [d['per_img'][i] for i in idx]
                a_base = U.compute_ap50_img([(p['box'], p['gd']) for p in sel],
                                            [p['gt'] for p in sel])
                a_ours = U.compute_ap50_img([(p['box'], p['fused']) for p in sel],
                                            [p['gt'] for p in sel])
                dsum += (a_ours - a_base)
            gains_mean.append(dsum / len(class_data))
        gains_mean = np.array(gains_mean)
        boot[str(seed)] = {'mean': float(np.mean(gains_mean)),
                           'ci95': [float(np.percentile(gains_mean, 2.5)),
                                    float(np.percentile(gains_mean, 97.5))],
                           'p_neg_frac': float(np.mean(gains_mean <= 0))}

    # 汇总
    mean_row = {k: round(float(np.mean([rows[c][k] for c in rows])), 4)
                for k in ['base50', 'ours50', 'gain50', 'base_coco', 'ours_coco', 'gain_coco']}
    out = {'config': {'protocol': 'per-class beta experiment',
                      'scheme': scheme_name,
                      'detector': DETECTOR, 'pre_filter': PRE_FILTER,
                      'neg_lam': lam,
                      'per_class_beta': per_class_beta,
                      'seeds': SEEDS, 'r_boot': R_BOOT},
           'classes': rows, 'mean': mean_row, 'bootstrap': boot}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 80)
    print('%-16s %6s %9s %9s %9s' % ('class', 'β', 'base50', 'ours50', 'Δ50'))
    print('-' * 80)
    for c in rows:
        r = rows[c]
        print('%-16s %6.2f %9.4f %9.4f %+9.4f' %
              (c, r['beta'], r['base50'], r['ours50'], r['gain50']))
    print('-' * 80)
    print('%-16s %6s %9.4f %9.4f %+9.4f' %
          ('mean', '', mean_row['base50'], mean_row['ours50'], mean_row['gain50']))
    for seed in SEEDS:
        b = boot[str(seed)]
        print('boot seed=%d: mean_gain=%+.4f CI=[%+.4f, %+.4f] 负比例=%.4f' %
              (seed, b['mean'], b['ci95'][0], b['ci95'][1], b['p_neg_frac']))
    print('=' * 80)
    print('→ %s   (耗时 %.1f 分钟)' % (out_path.name, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
