# -*- coding: utf-8 -*-
"""run_bidirectional_yoloworld.py — 第三方检测器 YOLO-World v2 双向缓存生成

复用 run_bidirectional_owlv2.py 的缓存格式（per_img: box/gd/gt/crop_idx + F），
仅把检测器换成 ultralytics YOLO-World v2（架构异于 GDINO/OWLv2，用于钉死"检测器条件性"）。
闭环数学检测器无关，本脚本只产出候选框 + CLIP 编码 crop，供 run_closed_loop_balanced.py 复用。

用法 (GPU, vg env):
  HF_HUB_OFFLINE=1 "C:/Users/张三/vg/Scripts/python.exe" -u run_bidirectional_yoloworld.py --n-eval 100
"""
import os, sys, json, pickle, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cuda')

sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPModel, CLIPProcessor

import run_bidirectional as RB

CACHE = RB.CACHE_DIR.parent / 'probe_yolo_cache'
CACHE.mkdir(parents=True, exist_ok=True)
OUT = RB.CACHE_DIR.parent / 'run_bidirectional_yoloworld.json'
WEIGHTS = r'C:\Users\张三\yolov8s-worldv2.pt'
CLIP_ID = 'openai/clip-vit-base-patch32'
DEVICE = os.environ.get('AC_SPC_DET_DEVICE', 'cuda')
N_PROTO = 20
PRE_FILTER = 0.03
BETA, LAM = 2.0, 0.5

CORE = [('airplane', 'airplane'), ('ship', 'boat'), ('vehicle', 'car')]
EXTRA = [('vehicle', 'truck'), ('vehicle', 'bus'), ('vehicle', 'motorcycle')]
JOBS = [('fwd_dior2coco', CORE + EXTRA, 'dior'), ('rev_coco2dior', [(c, d) for d, c in CORE], 'coco')]


def log(*a):
    print(*a, flush=True)


_CLIP = {}


def get_clip():
    if 'model' not in _CLIP:
        _CLIP['model'] = CLIPModel.from_pretrained(CLIP_ID).to(DEVICE).eval()
        _CLIP['proc'] = CLIPProcessor.from_pretrained(CLIP_ID)
    return _CLIP['model'], _CLIP['proc']


@torch.no_grad()
def clip_encode(crops, bs=128):
    # 委派 RB.clip_encode_imgs（内部用 _as_vec 解包 transformers 5.x 的 BaseModelOutputWithPooling）
    return RB.clip_encode_imgs(crops, bs)


_yolo = {}


def get_yolo():
    if 'model' not in _yolo:
        _yolo['model'] = YOLO(WEIGHTS)
    return _yolo['model']


def detect(images, prompt):
    m = get_yolo()
    m.set_classes([prompt])
    out = []
    for img in images:
        res = m.predict(img, conf=PRE_FILTER, verbose=False)
        r = res[0]
        b = r.boxes
        if len(b):
            out.append((b.xyxy.cpu().numpy(), b.conf.cpu().numpy()))
        else:
            out.append((np.zeros((0, 4)), np.zeros(0)))
    return out


def load_split(cfg, domain, cls, offset, n):
    # RB.load_split 内部已按 offset/n 切片并过滤空GT，直接委托
    ims, gts = RB.load_split(cfg, domain, cls, offset, n)
    keep = [(i, g) for i, g in zip(ims, gts) if len(g) > 0]
    return [k[0] for k in keep], [k[1] for k in keep]


def detect_encode(images, gts, prompt, key):
    cp = CACHE / f'{key}.pkl'
    if cp.exists():
        log(f'    [cache] {cp.name}')
        return pickle.load(open(cp, 'rb'))
    t0 = time.time()
    dets = detect(images, prompt)
    per_img, pending, pend_idx, Fs, CH = [], [], [], [], 1500
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < PRE_FILTER:
                continue
            bl.append([x1, y1, x2, y2]); sl.append(float(sc))
            ci.append(len(pend_idx)); pend_idx.append(len(pending))
            pending.append(img.crop((x1, y1, x2, y2)))
            if len(pending) >= CH:
                Fs.append(clip_encode(pending)); pending.clear()
        per_img.append({
            'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
            'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
            'gt': np.array(g, dtype=np.float64).reshape(-1, 4),
            'local_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64),
        })
    if pending:
        Fs.append(clip_encode(pending)); pending.clear()
    F = np.concatenate(Fs) if Fs else np.zeros((0, 512))
    pend_arr = np.asarray(pend_idx, dtype=np.int64)
    for p in per_img:
        p['crop_idx'] = pend_arr[p['local_idx']] if len(p['local_idx']) else np.zeros(0, dtype=np.int64)
        p.pop('local_idx', None)
    log(f'    N={len(images)} boxes={sum(len(p["gd"]) for p in per_img)} ({time.time()-t0:.0f}s)')
    blob = {'per_img': per_img, 'F': F}
    pickle.dump(blob, open(cp, 'wb'))
    return blob


def main():
    ap = __import__('argparse').ArgumentParser()
    ap.add_argument('--n-eval', type=int, default=100)
    args = ap.parse_args()
    cfg = RB.Config(); cfg.model.device = DEVICE
    log('=' * 80)
    log(f'YOLO-World v2 双向缓存生成  device={DEVICE}  N_EVAL={args.n_eval}')
    log('=' * 80)
    res = {}
    for tname, pairs, dom in JOBS:
        rows = {}
        for src_cls, eval_cls in pairs:
            imgs, gts = load_split(cfg, 'coco' if tname == 'fwd_dior2coco' else 'dior',
                                   eval_cls, N_PROTO, args.n_eval)
            if len(imgs) < 8:
                log(f'  !! {tname}/{eval_cls} 图像不足 {len(imgs)}'); continue
            prompt = eval_cls  # YOLO-World 用类名作零样本 prompt
            log(f'\n[{tname}] {eval_cls} ← {dom}:{src_cls}  N={len(imgs)}')
            blob = detect_encode(imgs, gts, prompt, f'{tname}_{eval_cls}_n{len(imgs)}')
            rows[f'{src_cls}->{eval_cls}'] = {
                'n_img': len(imgs), 'n_box': int(sum(len(p['gd']) for p in blob['per_img']))}
            log(f'    boxes={rows[f"{src_cls}->{eval_cls}"]["n_box"]}')
        res[tname] = rows
    json.dump({'config': {'detector': 'yoloworld-v2-small', 'weights': WEIGHTS,
                          'n_eval': args.n_eval, 'pre_filter': PRE_FILTER, 'clip': CLIP_ID},
               'results': res}, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    log(f'\n→ {OUT}')


if __name__ == '__main__':
    main()
