# -*- coding: utf-8 -*-
"""OWLv2 N=500 检测 + 逐图 per-GT recall。

背景: 论文 P1 的 GT 覆盖率是"逐图 per-GT"口径 (每图 GT 仅被本图框池覆盖) 且 N=500,
经验证与论文原数字逐位吻合。现有 tge_recall_probe.py 用全局合并池 (跨图误匹配, 偏高)。
本脚本:
  1) 对每类加载 N=500 图像, 复用已有 N=100 OWLv2 缓存的前 100 图, 仅新检测 101~N 图;
     保存到 tge_probe_dets_OWLV2-COCO-{cls}_n500.pkl (不覆盖 N=100 缓存)。
  2) 用逐图 per-GT recall (logit 尺度 pre-filter 0.03, 含小框) 计算 N=500 覆盖率。
输出: tge_recall_owlv2_n500.json
"""
import sys, os, json, time, pickle
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from pathlib import Path
import numpy as np
import torch
import unified_eval as U

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_recall_owlv2_n500.json'
N_COCO = 500
PRE_FILTER = 0.03
TGT_TMPL = 'a photo of {}'
OWLV2_CACHE = (r'C:\Users\张三\.cache\huggingface\hub'
               r'\models--google--owlv2-base-patch16-ensemble\snapshots')
TARGETS = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']

_owl = {}


def get_owlv2():
    if 'm' not in _owl:
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        osnap = os.listdir(OWLV2_CACHE)[0]
        opath = os.path.join(OWLV2_CACHE, osnap)
        _owl['p'] = Owlv2Processor.from_pretrained(opath)
        _owl['m'] = Owlv2ForObjectDetection.from_pretrained(
            opath, dtype=torch.float32).eval().to(U.DEVICE)
    return _owl['p'], _owl['m']


def owlv2_detect(images, prompt):
    op, om = get_owlv2()
    out = []
    for i, img in enumerate(images):
        ins = op(text=prompt, images=img, return_tensors='pt')
        ins = {k: v.to(U.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in ins.items()}
        with torch.no_grad():
            o = om(**ins)
        ts = torch.tensor([img.size[::-1]]).to(U.DEVICE)
        r = op.post_process_object_detection(o, threshold=0.0, target_sizes=ts)[0]
        out.append((r['boxes'].cpu().numpy(), r['scores'].cpu().numpy()))
        if (i + 1) % 25 == 0:
            print('      ... %d/%d 图' % (i + 1, len(images)), flush=True)
    return out


def iou_pairwise(boxes, gts):
    b = np.asarray(boxes, dtype=np.float64)
    g = np.asarray(gts, dtype=np.float64)
    xx1 = np.maximum(b[:, None, 0], g[None, :, 0])
    yy1 = np.maximum(b[:, None, 1], g[None, :, 1])
    xx2 = np.minimum(b[:, None, 2], g[None, :, 2])
    yy2 = np.minimum(b[:, None, 3], g[None, :, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / (ab[:, None] + ag[None, :] - inter + 1e-8)


def per_image_pergt_recall(dets, images, gts):
    """逐图 per-GT: 每图 GT 被本图任一 pf 框 (logit 尺度 0.03, 含小框) IoU>=0.5 覆盖。"""
    n_gt, hit = 0, 0
    for (boxes, scores), img, gs in zip(dets, images, gts):
        bl = []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if sc > PRE_FILTER:
                bl.append([x1, y1, x2, y2])
        n_gt += len(gs)
        if bl and gs:
            m = iou_pairwise(np.asarray(bl, dtype=np.float64), np.asarray(gs, dtype=np.float64))
            hit += int((m.max(axis=0) >= 0.5).sum())
    return hit / n_gt if n_gt else float('nan'), n_gt


def main():
    t0 = time.time()
    rows = {}
    for cls in TARGETS:
        print('[%s] load N=500 ...' % cls, flush=True)
        images, gts = U.load_coco(cls, n=N_COCO)
        prompt = TGT_TMPL.format(cls)
        old_p = DATA / f'tge_probe_dets_OWLV2-COCO-{cls}.pkl'
        new_p = DATA / f'tge_probe_dets_OWLV2-COCO-{cls}_n500.pkl'
        old = pickle.load(open(old_p, 'rb')) if old_p.exists() else []
        n_img = len(images)
        k = min(len(old), n_img)
        dets = old[:k]
        if k < n_img:
            print('    新检测 %d 图 (复用缓存前 %d 图)...' % (n_img - k, k), flush=True)
            newd = owlv2_detect(images[k:], prompt)
            dets = dets + newd
        pickle.dump(dets, open(new_p, 'wb'))
        r, n_gt = per_image_pergt_recall(dets, images, gts)
        rows[cls] = {'n_images': n_img, 'n_gt': n_gt,
                     'recall_pf_perimg': round(r, 4),
                     'reused_from_n100': k, 'newly_detected': n_img - k}
        print('  %-10s N=%3d GT=%4d  逐图 per-GT recall=%.4f  (%.1fs)'
              % (cls, n_img, n_gt, r, time.time() - t0), flush=True)

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'detector': 'OWLv2-base-patch16-ensemble', 'n_coco': N_COCO,
                              'definition': 'per-image per-GT recall, IoU>=0.5, '
                                            'logit 尺度 pre-filter 0.03, 含小框'},
                   'classes': rows}, f, indent=2, ensure_ascii=False)
    print('DONE in %.1fs -> %s' % (time.time() - t0, OUT), flush=True)


if __name__ == '__main__':
    main()
