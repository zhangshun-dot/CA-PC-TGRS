# -*- coding: utf-8 -*-
"""gen_qual_ship.py — 补 DIOR 'ship' → COCO 'boat' 的定性 blob（CPU, GDINO-tiny, n=15）

与 qual_{car,truck,bus}_off20_n15.pkl 同规格、同数据根（eval_v2_images），仅多一个类对，
用于 4 面板定性图的 ship 行。缓存写入 probe_bidir_cache/qual_boat_off20_n15.pkl。
"""
import os, sys, pickle
# 必须在 import acspc.config 之前设定数据根（Config 默认值在 import 时绑定）
os.environ['AC_SPC_DIOR_IMG'] = r'C:\Users\张三\modeling\dior_data\eval_v2_images'
os.environ['AC_SPC_DIOR_PARQUET'] = r'C:\Users\张三\modeling\dior_data\dior_combined.parquet'
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cpu')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import numpy as np

sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import run_bidirectional as RB

N_EVAL = 15


def main():
    cfg = RB.Config()
    cfg.model.device = 'cpu'
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'
    print('dior_img_dir =', cfg.paths.dior_img_dir, flush=True)
    images, gts = RB.load_split(cfg, 'dior', 'ship', RB.N_PROTO, N_EVAL)
    print(f'ship images = {len(images)}  gt_sum = {sum(len(g) for g in gts)}', flush=True)
    key = f'qual_boat_off{RB.N_PROTO}_n{len(images)}'
    blob = RB.detect_encode(cfg, images, gts, 'a photo of boat', key)
    pi = blob['per_img']
    print(f'blob n={len(pi)} boxes={sum(len(p["gd"]) for p in pi)} F={blob["F"].shape}', flush=True)
    print('done ->', key + '.pkl', flush=True)


if __name__ == '__main__':
    main()
