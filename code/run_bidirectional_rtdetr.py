# -*- coding: utf-8 -*-
"""run_bidirectional_rtdetr.py — 第三方检测器 RT-DETR (DETR family) 双向缓存生成

复用 run_bidirectional_yoloworld.py 的缓存格式（per_img: box/gd/gt/crop_idx + F），
把检测器换成 transformers RTDetrForObjectDetection（DETR family，异于 grounding/OWL/one-stage，
用于冲顶会第三异构 family 证据）。RT-DETR 为 closed-vocab COCO 80 类，无原生文本接口：
检测后按目标类筛选（COCO 类名与 6 类 eval 类名完全对齐），CLIP crop 特征与文本锚点
在闭环阶段由 run_closed_loop_balanced.py 统一处理（与 YOLO-World 接入方式一致）。

用法 (GPU, vg env):
  "C:/Users/张三/vg/Scripts/python.exe" -u run_bidirectional_rtdetr.py --n-eval 20
"""
import os, sys, json, pickle, time
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_OFFLINE', '1')  # 权重已落本地目录，运行时不再联网
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cuda')

sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

import run_bidirectional as RB

CACHE = RB.CACHE_DIR.parent / 'probe_rtdetr_cache'
CACHE.mkdir(parents=True, exist_ok=True)
OUT = RB.CACHE_DIR.parent / 'run_bidirectional_rtdetr.json'
# 权重已从 hf-mirror 稳健下载到本地平铺目录（见 rtdetr_download.py），运行时离线加载
RTDETR_DIR = r'C:\Users\张三\Desktop\实验数据\rtdetr_r50vd_coco_o365'
DEVICE = os.environ.get('AC_SPC_DET_DEVICE', 'cuda')
N_PROTO = 20
PRE_FILTER = 0.03
BETA, LAM = 2.0, 0.5

CORE = [('airplane', 'airplane'), ('ship', 'boat'), ('vehicle', 'car')]
EXTRA = [('vehicle', 'truck'), ('vehicle', 'bus'), ('vehicle', 'motorcycle')]
JOBS = [('fwd_dior2coco', CORE + EXTRA, 'dior'), ('rev_coco2dior', [(c, d) for d, c in CORE], 'coco')]


def log(*a):
    print(*a, flush=True)


_m = {}


def get_rtdetr():
    if 'model' not in _m:
        _m['proc'] = RTDetrImageProcessor.from_pretrained(RTDETR_DIR)
        _m['model'] = RTDetrForObjectDetection.from_pretrained(RTDETR_DIR).to(DEVICE).eval()
        # 小写类名 -> id 映射（COCO 类名与 eval 类名对齐）
        _m['label2id'] = {v.lower(): k for k, v in _m['model'].config.id2label.items()}
    return _m['model'], _m['proc'], _m['label2id']


@torch.no_grad()
def clip_encode(crops, bs=128):
    return RB.clip_encode_imgs(crops, bs)


@torch.no_grad()
def detect(images, prompt):
    """RT-DETR 闭集检测，按目标类筛选（prompt=项目 eval 类名，映射到 COCO 闭集原名）"""
    m, proc, label2id = get_rtdetr()
    # RT-DETR 是闭集 COCO 80 类，类名与项目 eval 命名不同：必须映射
    COCO_NAME = {'airplane': 'aeroplane', 'motorcycle': 'motorbike'}
    coco_name = COCO_NAME.get(prompt.lower(), prompt.lower())
    cls_id = label2id.get(coco_name)
    out = []
    for img in images:
        inputs = proc(images=img, return_tensors='pt').to(DEVICE)
        res = m(**inputs)
        probs = res.logits[0].sigmoid()      # [n_query, n_cls]
        boxes = res.pred_boxes[0]            # [n_query, 4] cxcywh normalized
        confs, cids = probs.max(-1)
        keep = confs > PRE_FILTER
        if cls_id is not None:
            keep = keep & (cids == cls_id)
        b, s = boxes[keep], confs[keep]
        if len(b) == 0:
            out.append((np.zeros((0, 4)), np.zeros(0)))
            continue
        W, H = img.size
        cx, cy, w, h = b[:, 0] * W, b[:, 1] * H, b[:, 2] * W, b[:, 3] * H
        xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1).cpu().numpy()
        out.append((xyxy, s.cpu().numpy()))
    return out


def load_split(cfg, domain, cls, offset, n):
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
    ap.add_argument('--n-eval', type=int, default=20)
    args = ap.parse_args()
    cfg = RB.Config(); cfg.model.device = DEVICE
    log('=' * 80)
    log(f'RT-DETR 双向缓存生成  device={DEVICE}  N_EVAL={args.n_eval}')
    log('=' * 80)
    res = {}
    for tname, pairs, dom in JOBS:
        rows = {}
        for src_cls, eval_cls in pairs:
            imgs, gts = load_split(cfg, 'coco' if tname == 'fwd_dior2coco' else 'dior',
                                   eval_cls, N_PROTO, args.n_eval)
            if len(imgs) < 8:
                log(f'  !! {tname}/{eval_cls} 图像不足 {len(imgs)}'); continue
            prompt = eval_cls
            log(f'\n[{tname}] {eval_cls} <- {dom}:{src_cls}  N={len(imgs)}')
            blob = detect_encode(imgs, gts, prompt, f'{tname}_{eval_cls}_n{len(imgs)}')
            rows[f'{src_cls}->{eval_cls}'] = {
                'n_img': len(imgs), 'n_box': int(sum(len(p['gd']) for p in blob['per_img']))}
            log(f'    boxes={rows[f"{src_cls}->{eval_cls}"]["n_box"]}')
        res[tname] = rows
    json.dump({'config': {'detector': 'rtdetr-r50vd-coco-o365', 'weights': RTDETR_DIR,
                          'n_eval': args.n_eval, 'pre_filter': PRE_FILTER},
               'results': res}, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    log(f'\n-> {OUT}')


if __name__ == '__main__':
    main()
