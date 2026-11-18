# -*- coding: utf-8 -*-
"""gen_gsd_multidataset.py — 多数据集(COCO→DOTA / COCO→NWPU) 反向 GSD 检测缓存生成

元 2026-10-13 要求: 把真实 RS 条件轴(GSD on RS target)扩到【多个数据集】。
  DIOR 已做(probe_bidir_gsd_cache)。本脚本补 DOTA + NWPU 目标域(源=COCO 自然域):
    DOTA 目标类: plane/ship/small-vehicle/large-vehicle  (源 COCO: airplane/boat/car/truck)
    NWPU 目标类: 1/2/10 (airplane/ship/vehicle)           (源 COCO: airplane/boat/car)
  复用 run_closed_loop_multidomain 的标注解析口径; GSD 降采样 + gdino 重检测 + CLIP 编码。

★ 内存修复(2026-10-13 v2):
  DOTA 瓦片原生尺寸极不均匀(最小 ~800px, 最大 6573x3727), 一次性 decode 100 张会 OOM。
  现在改为【有界惰性加载】: 每张读入后立即把最大边缩到 MAXSIDE(1024) 并同步缩放框, 再入列表;
  这样 f=1.0 是一个规范的 1024px 工作参考, 后续 GSD 档从该参考继续降采样。
  NWPU 原生 ~950px, 该上限对其为空操作 → 两数据集口径一致。
  另: 绝不与其它重型作业(CLIP 打分)并发运行 —— 二者抢内存会把 CUDA/CPU 作业一起打崩。

输出:
  DOTA → 实验数据/probe_bidir_gsd_dota_cache/rev_coco2dota_{cls}_off20_n{K}_gsd{f}.pkl
  NWPU → 实验数据/probe_bidir_gsd_nwpu_cache/rev_coco2nwpu_{cls}_off20_n{K}_gsd{f}.pkl

用法(GPU): HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cuda "C:/Users/张三/vg/Scripts/python.exe" -u gen_gsd_multidataset.py
"""
import os, sys, pickle, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cuda')
os.environ.setdefault('AC_SPC_COCO_ANN', r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json')
os.environ.setdefault('AC_SPC_COCO_IMG', r'C:\Users\张三\modeling\coco_data\val2017')
os.environ.setdefault('AC_SPC_DIOR_IMG', r'C:\Users\张三\modeling\dior_data\eval_v2_images')
os.environ.setdefault('AC_SPC_DIOR_PARQUET', r'C:\Users\张三\modeling\dior_data\dior_combined.parquet')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import re
import numpy as np
from pathlib import Path
from PIL import Image
import run_bidirectional as RB
import run_closed_loop_multidomain as MD

RB.DEVICE = 'cuda'
OFF, N = 20, 100
MAXSIDE = 1024
GSD = [(1.0, 1), (0.5, 2), (0.25, 4), (0.125, 8)]
DOTA_DIR = Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_gsd_dota_cache')
NWPU_DIR = Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_gsd_nwpu_cache')


def _bounded_open(ip, boxes, maxside=MAXSIDE):
    """打开图像; 若最大边 > maxside 则等比缩小到 maxside, 并把框同步缩放。"""
    im = Image.open(ip).convert('RGB')
    w, h = im.size
    s = maxside / max(w, h)
    if s < 1.0:
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        im = im.resize((nw, nh))
        boxes = [[x1 * s, y1 * s, x2 * s, y2 * s] for x1, y1, x2, y2 in boxes]
        # 缩放后按新尺寸裁剪(消除 round 误差, 并规整 DOTA 少量越界标注)
        boxes = [[max(0.0, min(nw, x1)), max(0.0, min(nh, y1)),
                  max(0.0, min(nw, x2)), max(0.0, min(nh, y2))] for x1, y1, x2, y2 in boxes]
    return im, boxes


def load_dota_bounded(cls, offset, n, maxside=MAXSIDE):
    """复刻 run_closed_loop_multidomain.load_dota 的解析口径, 但惰性+有界加载。"""
    imgdir = MD.DOTA_ROOT / 'images'
    lbdir = MD.DOTA_ROOT / 'labelTxt-v1.5' / 'DOTA-v1.5_val'
    if not lbdir.exists():
        cands = list((MD.DOTA_ROOT / 'labelTxt-v1.5').iterdir())
        lbdir = cands[0] if cands else lbdir
    sel = []
    for f in sorted(lbdir.glob('*.txt')):
        ip = imgdir / f'{f.stem}.png'
        if not ip.exists():
            continue
        boxes = []
        for l in open(f, encoding='utf-8', errors='ignore'):
            l = l.strip()
            if not l or l.startswith('imagesource') or l.startswith('gsd'):
                continue
            parts = l.split()
            if len(parts) < 9 or parts[8] != cls:
                continue
            try:
                coords = [float(v) for v in parts[:8]]
            except ValueError:
                continue
            xs, ys = coords[0::2], coords[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            if x2 - x1 > 4 and y2 - y1 > 4:
                boxes.append([x1, y1, x2, y2])
        if boxes:
            sel.append((ip, boxes))
    sel = sel[offset:offset + n]
    imgs, gts = [], []
    for p, b in sel:
        im, b2 = _bounded_open(p, b, maxside)
        imgs.append(im); gts.append(b2)
    return imgs, gts


def load_nwpu_bounded(cls_idx, offset, n, maxside=MAXSIDE):
    """复刻 run_closed_loop_multidomain.load_nwpu 的解析口径, 惰性+有界加载。"""
    gtdir = MD.NWPU_ROOT / 'ground_truth'
    imgdir = MD.NWPU_ROOT / 'positive_image_set'
    sel = []
    for f in sorted(gtdir.glob('*.txt')):
        ip = imgdir / f'{f.stem}.jpg'
        if not ip.exists():
            continue
        boxes = []
        for l in open(f, encoding='utf-8', errors='ignore'):
            l = l.strip()
            if not l:
                continue
            m = re.match(r'\(([\d.]+),\s*([\d.]+)\),\s*\(([\d.]+),\s*([\d.]+)\),\s*(\d+)', l)
            if not m or int(m.group(5)) != int(cls_idx):
                continue
            x1, y1, x2, y2 = [float(m.group(i)) for i in range(1, 5)]
            if x2 - x1 > 4 and y2 - y1 > 4:
                boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        if boxes:
            sel.append((ip, boxes))
    sel = sel[offset:offset + n]
    imgs, gts = [], []
    for p, b in sel:
        im, b2 = _bounded_open(p, b, maxside)
        imgs.append(im); gts.append(b2)
    return imgs, gts


def degrade_res(pil, k=3):
    w, h = pil.size
    return pil.resize((max(1, w // k), max(1, h // k))).resize((w, h))


def detect_encode(images, gts, prompt, key, outdir):
    cp = outdir / f'{key}.pkl'
    if cp.exists():
        print(f'  [skip] {key}', flush=True); return
    t0 = time.time()
    dets = RB.M.gdino_detect(_cfg(), images, prompt, 0.0)
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
            'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
            'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
            'gt': np.array(g, dtype=np.float64).reshape(-1, 4) if g else np.zeros((0, 4)),
            'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64),
        })
    F = RB.clip_encode_imgs(crops_all)
    pickle.dump({'per_img': per_img, 'F': F}, open(cp, 'wb'))
    del crops_all, dets
    print(f'  [done] {key}: n_img={len(per_img)} n_box={sum(len(p["gd"]) for p in per_img)} '
          f'F={F.shape} ({time.time()-t0:.0f}s)', flush=True)


_CFG = {}

def _cfg():
    if 'c' not in _CFG:
        c = RB.Config(); c.model.device = 'cuda'
        c.model.detection_threshold = 0.0
        c.model.pre_filter_threshold = RB.PRE_FILTER
        c.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'
        _CFG['c'] = c
    return _CFG['c']


def gen_dota():
    DOTA_DIR.mkdir(parents=True, exist_ok=True)
    for cls in ['plane', 'ship', 'small-vehicle', 'large-vehicle']:
        try:
            ims, gts = load_dota_bounded(cls, OFF, N)
        except Exception as e:
            print(f'  !! DOTA {cls} 读取失败: {e}', flush=True); continue
        print(f'[DOTA] {cls}: {len(ims)} imgs (maxside={MAXSIDE})', flush=True)
        if len(ims) < 8:
            print('  !! 图像不足', flush=True); continue
        prompt = f'an overhead image of {cls}'
        for gsd, k in GSD:
            dimgs = ims if gsd == 1.0 else [degrade_res(im, k) for im in ims]
            detect_encode(dimgs, gts, prompt, f'rev_coco2dota_{cls}_off{OFF}_n{len(ims)}_gsd{gsd}', DOTA_DIR)
            del dimgs


def gen_nwpu():
    NWPU_DIR.mkdir(parents=True, exist_ok=True)
    for cls_idx, name in [('1', 'airplane'), ('2', 'ship'), ('10', 'vehicle')]:
        try:
            ims, gts = load_nwpu_bounded(cls_idx, OFF, N)
        except Exception as e:
            print(f'  !! NWPU {cls_idx} 读取失败: {e}', flush=True); continue
        print(f'[NWPU] {cls_idx}({name}): {len(ims)} imgs (maxside={MAXSIDE})', flush=True)
        if len(ims) < 8:
            print('  !! 图像不足', flush=True); continue
        prompt = f'an overhead image of {name}'
        for gsd, k in GSD:
            dimgs = ims if gsd == 1.0 else [degrade_res(im, k) for im in ims]
            detect_encode(dimgs, gts, prompt, f'rev_coco2nwpu_{cls_idx}_off{OFF}_n{len(ims)}_gsd{gsd}', NWPU_DIR)
            del dimgs


if __name__ == '__main__':
    t0 = time.time()
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which in ('both', 'dota'):
        print('==== DOTA GSD ====', flush=True); gen_dota()
    if which in ('both', 'nwpu'):
        print('==== NWPU GSD ====', flush=True); gen_nwpu()
    print(f'\n→ 全部完成 ({time.time()-t0:.0f}s)', flush=True)
