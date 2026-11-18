# -*- coding: utf-8 -*-
r"""tge_v2_detectors.py — v2 冻结协议 §3: 检测器矩阵 (GDINO-base + OWLv2)

在冻结 (β=1.0, λ=0.25) 下, 对 COCO 全量 6 类评估 GDINO-base 与 OWLv2 两个
检测器的 baseline / A-CSPC 增益 / recall@proposal / AP50 / AP@[0.5:0.95],
与 GDINO-tiny (tge_v2_main.json) 合并为弱→强检测器谱系, 支撑"校正器边界"。

检测器:
  gdino-base : 概率尺度, pre-filter 0.03 (与主表同)
  owlv2      : raw-logit 尺度, pre-filter 0.03 (logit=0 噪声框全部滤除, 见分数尺度陷阱)

检测缓存: tge_v2_dets_GDINOB-*.pkl / tge_v2_dets_OWLV2-*.pkl (各自独立)。
运行: python tge_v2_detectors.py   (~30-40 分钟)
输出: 实验数据/tge_v2_detectors.json
"""
import sys, os, glob, json, pickle, argparse
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T
import tge_v2_main as V2
from tge_v2_main import (load_coco_all, clip_proto, gt_crops, collect_features,
                         fused_arrays, dataset_ap, recall_proposals,
                         TARGETS, FROZEN_BETA, FROZEN_LAM, K_CALIB,
                         SRC_TMPL, TGT_TMPL, DATA, PRE_FILTER)

GDINOB_ROOT = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-base\snapshots'
OWLV2_ROOT = r'C:\Users\张三\.cache\huggingface\hub\models--google--owlv2-base-patch16-ensemble\snapshots'


def _snapshot(root):
    snaps = sorted(glob.glob(os.path.join(root, '*')))
    if not snaps:
        raise FileNotFoundError('找不到权重 snapshot: %s' % root)
    return snaps[0]


_models = {}


def gdinob_detect(images, prompt, thresh=0.0):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    if 'gdinob' not in _models:
        path = _snapshot(GDINOB_ROOT)
        _models['gdinob'] = (AutoProcessor.from_pretrained(path),
                             AutoModelForZeroShotObjectDetection.from_pretrained(
                                 path, dtype=torch.float32).eval().to(U.DEVICE))
        print('    [加载] GDINO-base 模型', flush=True)
    p, m = _models['gdinob']
    out = []
    for img in images:
        ins = p(images=img, text=prompt, return_tensors='pt')
        ins = {k: v.to(U.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in ins.items()}
        with torch.no_grad():
            o = m(**ins)
        ts = torch.tensor([img.size[::-1]]).to(U.DEVICE)
        r = p.post_process_grounded_object_detection(o, threshold=thresh, target_sizes=ts)[0]
        out.append((r['boxes'].cpu().numpy(), r['scores'].cpu().numpy()))
    return out


def owlv2_detect(images, prompt):
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    if 'owlv2' not in _models:
        path = _snapshot(OWLV2_ROOT)
        _models['owlv2'] = (Owlv2Processor.from_pretrained(path),
                            Owlv2ForObjectDetection.from_pretrained(
                                path, dtype=torch.float32).eval().to(U.DEVICE))
        print('    [加载] OWLv2 模型', flush=True)
    op, om = _models['owlv2']
    out = []
    for img in images:
        ins = op(text=prompt, images=img, return_tensors='pt')
        ins = {k: v.to(U.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in ins.items()}
        with torch.no_grad():
            o = om(**ins)
        ts = torch.tensor([img.size[::-1]]).to(U.DEVICE)
        r = op.post_process_object_detection(o, threshold=0.0, target_sizes=ts)[0]
        out.append((r['boxes'].cpu().numpy(), r['scores'].cpu().numpy()))
    return out


def cached_dets(images, prompt, tag, detect_fn):
    p = DATA / f'tge_v2_dets_{tag}.pkl'
    if p.exists():
        print(f'    [缓存] {p.name}')
        return pickle.load(open(p, 'rb'))
    dets = detect_fn(images, prompt)
    pickle.dump(dets, open(p, 'wb'))
    return dets


def run_detector(tag, detect_fn, protos, coco_cls_map):
    """对单个检测器跑全量 6 类, 返回逐类行。protos 由 main 构建一次复用。"""
    rows = {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = load_coco_all(coco_cls)
        if len(images) < K_CALIB + 10:
            print('  !! %s 图太少 (%d), 跳过' % (name, len(images)))
            continue
        prompt = TGT_TMPL.format(coco_cls_map.get(coco_cls, coco_cls))
        dets = cached_dets(images, prompt, tag + '-' + name, detect_fn)
        per_img, F = collect_features(images, gts, dets)
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        Delta = d_y - s_xs

        base50 = dataset_ap(per_img, 'gd')
        base_coco = dataset_ap(per_img, 'gd', coco_style=True)
        P_hat = P_src + FROZEN_BETA * Delta.to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        W = (P_hat.cpu().numpy() - FROZEN_LAM * P_neg.cpu().numpy())
        fused_arrays(per_img, F, W)
        ours50 = dataset_ap(per_img, 'fused')
        ours_coco = dataset_ap(per_img, 'fused', coco_style=True)
        rec = recall_proposals(per_img)
        rows[name] = {'n_pool': len(images), 'n_test': len(images) - K_CALIB,
                      'recall': round(rec, 4),
                      'base50': round(base50, 4), 'ours50': round(ours50, 4),
                      'gain50': round(ours50 - base50, 4),
                      'base_coco': round(base_coco, 4), 'ours_coco': round(ours_coco, 4),
                      'gain_coco': round(ours_coco - base_coco, 4)}
        print('  %-16s base50=%.4f ours50=%.4f (%+.4f) rec=%.4f' %
              (name, base50, ours50, ours50 - base50, rec))
    return rows


def main():
    global FROZEN_BETA, FROZEN_LAM
    import time
    t0 = time.time()
    ap = argparse.ArgumentParser(description='A-CSPC v2 检测器矩阵 (可指定 β/λ)')
    ap.add_argument('--beta', type=float, default=FROZEN_BETA)
    ap.add_argument('--lam', type=float, default=FROZEN_LAM)
    args = ap.parse_args()
    FROZEN_BETA, FROZEN_LAM = args.beta, args.lam
    tag = 'b%g_l%g' % (FROZEN_BETA, FROZEN_LAM)
    out_path = DATA / ('tge_v2_detectors_%s.json' % tag)
    print('配置: β=%.2f λ=%.2f  →  输出 %s' % (FROZEN_BETA, FROZEN_LAM, out_path.name))
    print('[DIOR 20 类 CLIP 源原型 (两检测器共用, 构建一次)]')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p
    print('  原型池 %d 类' % len(protos))
    print('=' * 90)
    print('[检测器矩阵] GDINO-base (tag=GDINOB)')
    rows_b = run_detector('GDINOB', gdinob_detect, protos, {})
    print('=' * 90)
    print('[检测器矩阵] OWLv2 (tag=OWLV2, raw-logit 尺度)')
    rows_o = run_detector('OWLV2', owlv2_detect, protos, {})

    # 汇总
    def mean_rows(rows):
        keys = ['base50', 'ours50', 'gain50', 'base_coco', 'ours_coco', 'gain_coco', 'recall']
        return {k: round(float(np.mean([rows[c][k] for c in rows])), 4) for k in keys}

    out = {'config': {'protocol': 'v2 frozen: β=1.0 λ=0.25, full val2017, per-image AP50 + COCO '
                                  'AP@0.5:0.95; OWLv2 raw-logit pre-filter 0.03',
                      'beta': FROZEN_BETA, 'neg_lam': FROZEN_LAM},
           'gdino_tiny_mean': None,  # 由 tge_v2_main.json 补
           'gdino_base': {'classes': rows_b, 'mean': mean_rows(rows_b)},
           'owlv2': {'classes': rows_o, 'mean': mean_rows(rows_o)}}
    # 并入 GDINO-tiny 均值 (匹配当前配置 tag, 回退 tge_v2_main.json)
    main_json = DATA / ('tge_v2_main_%s.json' % tag)
    if not main_json.exists():
        main_json = DATA / 'tge_v2_main.json'
    if main_json.exists():
        mj = json.load(open(main_json, encoding='utf-8'))
        m = dict(mj.get('mean') or {})
        cls = mj.get('classes') or {}
        if cls:
            m['recall'] = round(float(np.mean([cls[c]['recall'] for c in cls])), 4)
        out['gdino_tiny_mean'] = m
        out['gdino_tiny_source'] = main_json.name
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 100)
    print('%-14s %10s %10s %10s %10s %10s' % ('检测器', 'base50', 'A-CSPC50', 'Δ50', 'Δcoco', 'recall'))
    print('-' * 100)
    for label, r in [('GDINO-tiny', out['gdino_tiny_mean']), ('GDINO-base', out['gdino_base']['mean']),
                     ('OWLv2', out['owlv2']['mean'])]:
        if r is None:
            continue
        print('%-14s %10.4f %10.4f %+10.4f %+10.4f %10.4f' %
              (label, r['base50'], r['ours50'], r['gain50'], r['gain_coco'], r['recall']))
    print('=' * 100)
    print('→ %s   (耗时 %.1f 分钟)' % (out_path.name, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
