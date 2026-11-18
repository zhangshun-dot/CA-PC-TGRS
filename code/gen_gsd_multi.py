# -*- coding: utf-8 -*-
"""gen_gsd_multi.py — 多类别 × 多检测器 的真实 RS 目标域 GSD 检测缓存生成

背景 (元 2026-10-13 评审): 原 Table(单一 vehicle + 单 gdino) 需扩到【多 RS 类别】+【多检测器】。
  本脚本对反向 COCO→DIOR 的 RS 目标域做 GSD 降采样检测缓存:
    gdino  : vehicle(已存在) + ship + airplane
    owlv2  : vehicle + ship + airplane
  口径对齐 gen_dior_gsd_det.py / run_bidirectional_owlv2.detect_encode。
  GSD=[1.0,0.5,0.25,0.125] (k=1/2/4/8); DIOR off=20, n=100。

输出:
  gdino → 实验数据/probe_bidir_gsd_cache/rev_coco2dior_{cls}_off20_n100_gsd{f}.pkl
  owlv2 → 实验数据/probe_bidir_gsd_owl_cache/rev_coco2dior_{cls}_off20_n100_gsd{f}.pkl

GPU: AC_SPC_DET_DEVICE=cuda
用法:
  HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cuda "C:/Users/张三/vg/Scripts/python.exe" -u gen_gsd_multi.py --dets gdino owlv2
"""
import os, sys, pickle, time, argparse
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cuda')
os.environ.setdefault('AC_SPC_COCO_ANN', r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json')
os.environ.setdefault('AC_SPC_COCO_IMG', r'C:\Users\张三\modeling\coco_data\val2017')
os.environ.setdefault('AC_SPC_DIOR_IMG', r'C:\Users\张三\modeling\dior_data\eval_v2_images')
os.environ.setdefault('AC_SPC_DIOR_PARQUET', r'C:\Users\张三\modeling\dior_data\dior_combined.parquet')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import numpy as np
import torch
from pathlib import Path
import run_bidirectional as RB
import run_bidirectional_owlv2 as RBOWL

GSD = [(1.0, 1), (0.5, 2), (0.25, 4), (0.125, 8)]
CLASSES = ['vehicle', 'ship', 'airplane']
OFF, N = 20, 100
GSD_GDINO_DIR = Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_gsd_cache')
GSD_OWL_DIR = Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_gsd_owl_cache')


def degrade_res(pil, k=3):
    w, h = pil.size
    return pil.resize((max(1, w // k), max(1, h // k))).resize((w, h))


def build_blob(dets, images, gts, clip_enc):
    """与 detect_encode 同口径: per_img + F。clip_enc(list_of_pil)->np.ndarray"""
    per_img, crops_all = [], []
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < 0.03:
                continue
            bl.append([x1, y1, x2, y2]); sl.append(float(sc)); ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({
            'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
            'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
            'gt': np.array(g, dtype=np.float64).reshape(-1, 4) if g else np.zeros((0, 4)),
            'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64),
        })
    F = clip_enc(crops_all)
    return {'per_img': per_img, 'F': F}


def gen_gdino():
    RB.DEVICE = 'cuda'
    cfg = RB.Config(); cfg.model.device = 'cuda'
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'
    GSD_GDINO_DIR.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        ims, gts = RB.load_split(cfg, 'dior', cls, OFF, N)
        print(f'[gdino] DIOR {cls}: {len(ims)} imgs', flush=True)
        if len(ims) < 8:
            print('  !! 图像不足', flush=True); continue
        prompt = f'an overhead image of {cls}'
        for gsd, k in GSD:
            key = f'rev_coco2dior_{cls}_off{OFF}_n{N}_gsd{gsd}'
            cp = GSD_GDINO_DIR / f'{key}.pkl'
            if cp.exists():
                print(f'  [skip] {key}', flush=True); continue
            t0 = time.time()
            dimgs = ims if gsd == 1.0 else [degrade_res(im, k) for im in ims]
            dets = RB.M.gdino_detect(cfg, dimgs, prompt, 0.0)
            blob = build_blob(dets, dimgs, gts, RB.clip_encode_imgs)
            pickle.dump(blob, open(cp, 'wb'))
            print(f'  [done] {key}: n_img={len(blob["per_img"])} '
                  f'n_box={sum(len(p["gd"]) for p in blob["per_img"])} F={blob["F"].shape} '
                  f'({time.time()-t0:.0f}s)', flush=True)


def gen_owl():
    RBOWL.DEVICE = 'cuda'
    cfg = RBOWL.Config(); cfg.model.device = 'cuda'
    GSD_OWL_DIR.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        ims, gts = RBOWL.load_split(cfg, 'dior', cls, OFF, N)
        print(f'[owlv2] DIOR {cls}: {len(ims)} imgs', flush=True)
        if len(ims) < 8:
            print('  !! 图像不足', flush=True); continue
        prompt = f'an overhead image of {cls}'
        for gsd, k in GSD:
            key = f'rev_coco2dior_{cls}_off{OFF}_n{N}_gsd{gsd}'
            cp = GSD_OWL_DIR / f'{key}.pkl'
            if cp.exists():
                print(f'  [skip] {key}', flush=True); continue
            t0 = time.time()
            dimgs = ims if gsd == 1.0 else [degrade_res(im, k) for im in ims]
            dets = RBOWL.owl_detect(dimgs, prompt)
            blob = build_blob(dets, dimgs, gts, RBOWL.clip_encode)
            pickle.dump(blob, open(cp, 'wb'))
            print(f'  [done] {key}: n_img={len(blob["per_img"])} '
                  f'n_box={sum(len(p["gd"]) for p in blob["per_img"])} F={blob["F"].shape} '
                  f'({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dets', nargs='+', default=['gdino', 'owlv2'])
    args = ap.parse_args()
    t0 = time.time()
    if 'gdino' in args.dets:
        print('==== gdino GSD ====', flush=True); gen_gdino()
    if 'owlv2' in args.dets:
        print('==== owlv2 GSD ====', flush=True); gen_owl()
    print(f'\n→ 全部完成 ({time.time()-t0:.0f}s)', flush=True)
