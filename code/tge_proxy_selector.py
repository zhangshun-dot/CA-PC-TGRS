# -*- coding: utf-8 -*-
r"""tge_proxy_selector.py — v3 前置验证: 目标域无关失效度代理 × 逐类外推增益相关性

设计目标 (v3 可操作类级选择器):
  g2>0 (方向)  AND  NOT(代理显示无需修复)  →  外推; 否则 B2
本脚本验证核心假设: 存在**目标域无关**的代理信号, 能预测逐类外推增益 (ext−B2)。

代理候选 (全部只用源原型 + 文本 + 候选框分数, 零目标域 GT):
  agree_spearman / agree_kendall : GDINO 分数 ↔ CLIP 源原型分数 在候选框上的排序一致性
                                  (低 = 源原型在框池内失校 → 外推空间大)
  sep_src          : f·P_src 分布标准差 (分离度; 低 = 原型无法区分)
  gd_shape         : gd 分数>0.3 的框占比 (检测器置信度形态)

分析范围: 正向方向 13 类 (COCO 6 + DOTA 7), 目标变量 = tge_probe_gate.json 的 gain (ext−B2)。
COCO 复用 tge_v2 全量检测缓存; DOTA 重新检测 (N=10/类)。

运行: python tge_proxy_selector.py   (~25 分钟)
输出: 实验数据/tge_proxy_selector.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json, pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau
import unified_eval as U
import tge_probe as T
import tge_v2_main as V2
from tge_v2_main import (load_coco_all, clip_proto, gt_crops, collect_features,
                         v2_cached_dets, TARGETS, SRC_TMPL, TGT_TMPL, DATA, PRE_FILTER)

# DOTA (tge_probe_dota 同协议)
DOTA_IMG = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\images'
DOTA_LBL = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\labelTxt-v1.5\DOTA-v1.5_train'
DOTA_CLASSES = ['plane', 'ship', 'storage-tank', 'harbor', 'bridge',
                'large-vehicle', 'small-vehicle']
DOTA_TO_DIOR = {'plane': 'airplane', 'ship': 'ship', 'storage-tank': 'storage tank',
                'harbor': 'harbor', 'bridge': 'bridge',
                'large-vehicle': 'vehicle', 'small-vehicle': 'vehicle'}


def load_dota(dota_cls, n=10):
    quads = defaultdict(list)
    for fn in Path(DOTA_LBL).glob('*.txt'):
        img_name = fn.stem + '.png'
        if not (Path(DOTA_IMG) / img_name).exists():
            continue
        for line in open(fn):
            parts = line.strip().split()
            if len(parts) < 9 or parts[8] != dota_cls:
                continue
            try:
                q = list(map(float, parts[:8]))
                xs, ys = q[0::2], q[1::2]
                quads[img_name].append([min(xs), min(ys), max(xs), max(ys)])
            except Exception:
                continue
    images, gts = [], []
    for img_name, boxes in quads.items():
        from PIL import Image
        images.append(Image.open(Path(DOTA_IMG) / img_name).convert('RGB'))
        gts.append(boxes)
        if len(images) >= n:
            break
    return images, gts


def proxy_from_perimg(per_img, F, P_src):
    """候选框上的代理: 拼接所有框的 gd 与 f·P_src, 算一致性/分离度/置信形态。"""
    gds, srcs = [], []
    for p in per_img:
        n = len(p['gd'])
        if n:
            gds.append(p['gd'])
            srcs.append(F[p['crop_idx']] @ P_src.cpu().numpy())
    if not gds or sum(len(g) for g in gds) < 10:
        return None
    gd = np.concatenate(gds)
    src = np.concatenate(srcs)
    rho, _ = spearmanr(gd, src)
    tau, _ = kendalltau(gd, src)
    return {'agree_spearman': round(float(rho), 4), 'agree_kendall': round(float(tau), 4),
            'sep_src': round(float(src.std()), 4),
            'gd_hi_frac': round(float(np.mean(gd > 0.3)), 4),
            'n_boxes': int(len(gd))}


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

    gate = json.load(open(DATA / 'tge_probe_gate.json', encoding='utf-8'))['rows']
    gain_map = {r['name']: r['gain'] for r in gate}

    print('[2/3] COCO 6 类: 复用全量缓存, 代理计算 ...')
    proxy = {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = load_coco_all(coco_cls)
        dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), name)
        per_img, F = collect_features(images, gts, dets)
        P_src = protos[src_cls]
        pr = proxy_from_perimg(per_img, F, P_src)
        if pr is not None:
            proxy[name] = pr
            print('  %-16s agree=%.3f sep=%.3f  n=%d' %
                  (name, pr['agree_spearman'], pr['sep_src'], pr['n_boxes']))

    print('[3/3] DOTA 7 类: 重新检测 + 代理计算 ...')
    for dota_cls in DOTA_CLASSES:
        dior_cls = DOTA_TO_DIOR[dota_cls]
        name = 'DOTA-' + dota_cls
        images, gts = load_dota(dota_cls)
        if not images:
            continue
        dets = U.gdino_detect(images, SRC_TMPL.format(dior_cls), 0.0)
        per_img, F = collect_features(images, gts, dets)
        pr = proxy_from_perimg(per_img, F, protos[dior_cls])
        if pr is not None:
            proxy[name] = pr
            print('  %-16s agree=%.3f sep=%.3f  n=%d' %
                  (name, pr['agree_spearman'], pr['sep_src'], pr['n_boxes']))

    # 相关分析 (13 正向类)
    names = [n for n in proxy if n in gain_map]
    rows = [(n, proxy[n], gain_map[n]) for n in names]
    print('\n' + '=' * 92)
    print('%-18s %9s %9s %9s %9s %9s' % ('class', 'gain(ext−B2)', 'agree_sp', 'agree_kd',
                                          'sep_src', 'gd_hi'))
    print('-' * 92)
    for n, pr, g in sorted(rows, key=lambda x: -x[2]):
        print('%-18s %+9.4f %9.3f %9.3f %9.3f %9.3f' %
              (n, g, pr['agree_spearman'], pr['agree_kendall'], pr['sep_src'], pr['gd_hi_frac']))
    print('=' * 92)

    import scipy.stats as st
    result = {}
    for key in ['agree_spearman', 'agree_kendall', 'sep_src', 'gd_hi_frac']:
        xs = np.array([pr[key] for _, pr, _ in rows])
        ys = np.array([g for _, _, g in rows])
        rho, p = st.spearmanr(xs, ys)
        result[key] = {'spearman_rho': round(float(rho), 4), 'p': round(float(p), 4)}
        print('proxy[%s] vs gain: ρ=%+.3f  p=%.4f' % (key, rho, p))

    out = {'config': {'protocol': 'v3 proxy: GDINO↔CLIP源原型一致性 × ext−B2 gain, 13 正向类 '
                                  '(COCO6 全量 + DOTA7 N=10), 全部目标域无关信号'},
           'proxy': proxy, 'gain_map': {n: gain_map[n] for n in names},
           'correlation': result}
    with open(DATA / 'tge_proxy_selector.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n→ tge_proxy_selector.json   (耗时 %.1f 分钟)' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
