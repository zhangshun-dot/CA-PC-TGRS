# -*- coding: utf-8 -*-
"""gen_dior_gsd_det.py — 生成 GSD 退化的 DIOR(vehicle, RS 目标域) gdino 检测缓存

用途: 决定性 A/B 收敛实验的底层数据。对 rev_coco2dior 的目标域(DIOR vehicle)
       做 GSD 降采样(1.0/0.5/0.25/0.125), 用 grounding-dino-tiny 重新检测并 CLIP 编码 crop,
       存为与 run_bidirectional.detect_encode 完全同口径的 blob({per_img, F})。

口径严格对齐 run_bidirectional.detect_encode:
  - gdino detection_threshold=0.0, PRE_FILTER=0.03
  - per_img: box/gd(clip[1e-6,1.0])/gt/crop_idx
  - F = RB.clip_encode_imgs(crops)  (CLIP base patch32, 512d)
  - GSD 退化: degrade_res(im,k)=im.resize((w//k,h//k)).resize((w,h))  (k=1/gsd)

GPU: AC_SPC_DET_DEVICE=cuda  (C:/Users/张三/vg/Scripts/python.exe)
用法:
  HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cuda "C:/Users/张三/vg/Scripts/python.exe" -u gen_dior_gsd_det.py
"""
import os, sys, json, pickle, time, glob
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cuda')
os.environ.setdefault('AC_SPC_COCO_ANN', r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json')
os.environ.setdefault('AC_SPC_COCO_IMG', r'C:\Users\张三\modeling\coco_data\val2017')
os.environ.setdefault('AC_SPC_DIOR_IMG', r'C:\Users\张三\modeling\dior_data\eval_v2_images')
os.environ.setdefault('AC_SPC_DIOR_PARQUET', r'C:\Users\张三\modeling\dior_data\dior_combined.parquet')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import numpy as np
import run_bidirectional as RB

RB.DEVICE = 'cuda'
cfg = RB.Config()
cfg.model.device = 'cuda'
cfg.model.detection_threshold = 0.0
cfg.model.pre_filter_threshold = RB.PRE_FILTER
cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

CACHE_DIR = __import__('pathlib').Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_gsd_cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TASK = 'rev_coco2dior'
EVAL_CLS = 'vehicle'          # DIOR 类 (RS 目标域)
OFF, N = 20, 100
GSD = [(1.0, 1), (0.5, 2), (0.25, 4), (0.125, 8)]


def degrade_res(pil, k=3):
    w, h = pil.size
    return pil.resize((max(1, w // k), max(1, h // k))).resize((w, h))


def detect_encode(images, gts, prompt, key):
    cp = CACHE_DIR / f'{key}.pkl'
    if cp.exists():
        print(f'  [skip] {key}', flush=True)
        return
    t0 = time.time()
    dets = RB.M.gdino_detect(cfg, images, prompt, cfg.model.detection_threshold)
    per_img, crops_all = [], []
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < RB.PRE_FILTER:
                continue
            bl.append([x1, y1, x2, y2]); sl.append(float(sc)); ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({
            'box': np.array(bl, float) if bl else np.zeros((0, 4)),
            'gd': np.clip(np.array(sl, float), 1e-6, 1.0) if sl else np.zeros(0),
            'gt': np.array(g, float) if g else np.zeros((0, 4)),
            'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64),
        })
    F = RB.clip_encode_imgs(crops_all)
    pickle.dump({'per_img': per_img, 'F': F}, open(cp, 'wb'))
    print(f'  [done] {key}: n_img={len(per_img)} n_box={sum(len(p["gd"]) for p in per_img)} '
          f'F={F.shape} ({time.time()-t0:.0f}s)', flush=True)


def main():
    t0 = time.time()
    ims, gts = RB.load_split(cfg, 'dior', EVAL_CLS, OFF, N)
    print(f'[load] DIOR {EVAL_CLS}: {len(ims)} imgs (offset={OFF}, n={N})', flush=True)
    if len(ims) < 8:
        print('!! 图像不足', flush=True); return
    prompt = f'an overhead image of {EVAL_CLS}'
    for gsd, k in GSD:
        dimgs = ims if gsd == 1.0 else [degrade_res(im, k) for im in ims]
        key = f'{TASK}_{EVAL_CLS}_off{OFF}_n{N}_gsd{gsd}'
        detect_encode(dimgs, gts, prompt, key)
    print(f'\n→ 全部 GSD 检测缓存已生成于 {CACHE_DIR}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
