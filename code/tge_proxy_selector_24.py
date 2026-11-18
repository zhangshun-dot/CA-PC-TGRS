# -*- coding: utf-8 -*-
r"""tge_proxy_selector_24.py — sep 信号在全部 24 类上的复核 (gate 全集)

在 tge_proxy_selector.py (13 正向类) 基础上扩展到 24 类:
  COCO 6 + DOTA 7 + VOC 5 + 反向 DIOR/DOTA 6
目标变量 = tge_probe_gate.json 的 gain (ext−B2)。
代理 = sep_src (CLIP 源原型分数在候选框上的 std), 附 agree/sep/gd_hi 全套。
反向类: 源原型为 COCO (自然), 目标是 DIOR/DOTA (航空)。

运行: python tge_proxy_selector_24.py   (~25 分钟)
输出: 实验数据/tge_proxy_selector_24.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr, kendalltau
import unified_eval as U
import tge_probe as T
from tge_v2_main import (load_coco_all, clip_proto, gt_crops, collect_features,
                         v2_cached_dets, TARGETS, SRC_TMPL, TGT_TMPL, DATA, PRE_FILTER)
import tge_probe_voc as VOC

DOTA_IMG = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\images'
DOTA_LBL = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\labelTxt-v1.5\DOTA-v1.5_train'
DOTA7 = ['plane', 'ship', 'storage-tank', 'harbor', 'bridge', 'large-vehicle', 'small-vehicle']
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
    from PIL import Image
    images, gts = [], []
    for img_name, boxes in quads.items():
        images.append(Image.open(Path(DOTA_IMG) / img_name).convert('RGB'))
        gts.append(boxes)
        if len(images) >= n:
            break
    return images, gts


def build_coco_protos():
    protos = {}
    im, gt = U.load_coco('airplane', n=500)
    protos['airplane'] = clip_proto(gt_crops(im, gt))
    im, gt = U.load_coco('boat', n=500)
    protos['boat'] = clip_proto(gt_crops(im, gt))
    imgs, gts = [], []
    for c in ['car', 'truck', 'bus', 'motorcycle']:
        im2, gt2 = U.load_coco(c, n=500)
        imgs += im2
        gts += gt2
    protos['vehicle'] = clip_proto(gt_crops(imgs, gts))
    return protos


def proxy_from_perimg(per_img, F, P_src):
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
            'gd_hi_frac': round(float(np.mean(gd > 0.3)), 4), 'n_boxes': int(len(gd))}


def main():
    import time
    t0 = time.time()
    print('[1/5] DIOR 20 类 + COCO 3 类源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p
    coco_protos = build_coco_protos()
    print('  DIOR %d 类 + COCO %d 类就绪' % (len(protos), len(coco_protos)))

    gate = json.load(open(DATA / 'tge_probe_gate.json', encoding='utf-8'))['rows']
    gain_map = {r['name']: r['gain'] for r in gate}
    proxy = {}

    print('[2/5] COCO 6 类 (全量缓存) ...')
    for name, coco_cls, src_cls in TARGETS:
        images, gts = load_coco_all(coco_cls)
        dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), name)
        per_img, F = collect_features(images, gts, dets)
        pr = proxy_from_perimg(per_img, F, protos[src_cls])
        if pr: proxy[name] = pr

    print('[3/5] DOTA 7 类 (重新检测) ...')
    for dc in DOTA7:
        name = 'DOTA-' + dc
        images, gts = load_dota(dc)
        if not images:
            continue
        dets = U.gdino_detect(images, SRC_TMPL.format(DOTA_TO_DIOR[dc]), 0.0)
        per_img, F = collect_features(images, gts, dets)
        pr = proxy_from_perimg(per_img, F, protos[DOTA_TO_DIOR[dc]])
        if pr: proxy[name] = pr

    print('[4/5] VOC 5 类 (DIOR 源, 复用检测缓存) ...')
    for name, voc_cls, src_cls in [('VOC-aeroplane', 'aeroplane', 'airplane'),
                                   ('VOC-boat', 'boat', 'ship'),
                                   ('VOC-car', 'car', 'vehicle'),
                                   ('VOC-bus', 'bus', 'vehicle'),
                                   ('VOC-motorbike', 'motorbike', 'vehicle')]:
        images, gts = VOC.load_voc(voc_cls)
        if len(images) < 10:
            continue
        dets = T.cached_dets(images, TGT_TMPL.format(voc_cls), name)
        per_img, F = collect_features(images, gts, dets)
        pr = proxy_from_perimg(per_img, F, protos[src_cls])
        if pr: proxy[name] = pr

    print('[5/5] 反向 6 类 (COCO 源 → DIOR/DOTA 目标) ...')
    for dc, coco_key in [('plane', 'airplane'), ('ship', 'boat'), ('large-vehicle', 'vehicle')]:
        images, gts = load_dota(dc)
        if images:
            dets = U.gdino_detect(images, SRC_TMPL.format(dc), 0.0)
            per_img, F = collect_features(images, gts, dets)
            pr = proxy_from_perimg(per_img, F, coco_protos[coco_key])
            if pr: proxy['REV-DOTA-' + dc] = pr
    for dior_cls, coco_key in [('airplane', 'airplane'), ('ship', 'boat'), ('vehicle', 'vehicle')]:
        images, gts = U.load_dior(dior_cls)
        if images:
            dets = U.gdino_detect(images, TGT_TMPL.format(dior_cls), 0.0)
            per_img, F = collect_features(images, gts, dets)
            pr = proxy_from_perimg(per_img, F, coco_protos[coco_key])
            if pr: proxy['REV-DIOR-' + dior_cls] = pr

    # 相关分析
    import scipy.stats as st
    names = [n for n in proxy if n in gain_map]
    groups = {'正向13': [n for n in names if n.startswith(('COCO-', 'DOTA-'))],
              'VOC5': [n for n in names if n.startswith('VOC-')],
              '反向6': [n for n in names if n.startswith('REV-')],
              '全部24': names}
    print('\n' + '=' * 96)
    print('%-20s %9s %9s %9s %9s' % ('class', 'gain', 'sep_src', 'agree_sp', 'gd_hi'))
    print('-' * 96)
    for n in sorted(names, key=lambda x: gain_map[x]):
        pr = proxy[n]
        print('%-20s %+9.4f %9.3f %9.3f %9.3f' %
              (n, gain_map[n], pr['sep_src'], pr['agree_spearman'], pr['gd_hi_frac']))
    print('=' * 96)

    corr = {}
    for gname, gnames in groups.items():
        xs = np.array([proxy[n]['sep_src'] for n in gnames])
        ys = np.array([gain_map[n] for n in gnames])
        rho, p = st.spearmanr(xs, ys)
        corr[gname] = {'n': len(gnames), 'rho': round(float(rho), 4), 'p': round(float(p), 4)}
        print('sep_src vs gain [%s] (n=%d): ρ=%+.3f  p=%.4f' % (gname, len(gnames), rho, p))

    out = {'config': {'protocol': 'v3 sep 复核: 24 类 (COCO6 全量 + DOTA7 N10 + VOC5 N100 + 反向6), '
                                  '全部目标无关信号'},
           'proxy': proxy, 'gain_map': {n: gain_map[n] for n in names},
           'correlation': corr}
    with open(DATA / 'tge_proxy_selector_24.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n→ tge_proxy_selector_24.json   (耗时 %.1f 分钟)' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
