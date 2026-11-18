# -*- coding: utf-8 -*-
"""run_closed_loop_calibration.py — #9 置信度校正/Calibration（CA-PC 前后 ECE）

度量: 对每个检测框算"是否正确"(IoU≥0.5 命中 GT)，用同一正确性标签比较
  前置信度 gd (检测器原生)  vs  后置信度 s=√(gd·clip(F@W,1e-6,1)) (CA-PC 调制) 的 ECE。
ECE = Σ_b  (n_b/N)·|acc_b − conf_b|，10 等频 bin。越低越校准。

用法:
  HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu "C:/Users/张三/vg/Scripts/python.exe" -u run_closed_loop_calibration.py
"""
import os, sys, json, pickle, glob, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cpu')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
import numpy as np, torch
import run_bidirectional as RB
from pathlib import Path

RB.DEVICE = 'cpu'; DEVICE = 'cpu'
LAM = RB.LAM; NEG_TOPK = RB.NEG_TOPK; N_PROTO = RB.N_PROTO
DETECTORS = {'gdino': Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache'),
             'owlv2': Path(r'C:\Users\张三\Desktop\实验数据\probe_owl_cache'),
             'yolo':  Path(r'C:\Users\张三\Desktop\实验数据\probe_yolo_cache')}
bal = json.load(open(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced.json'))
c0 = bal['config']['gate']['c0']; PRIOR_SCALE = bal['config']['gate']['prior_scale']
OUT = Path(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_calibration.json')


def unit(x):
    x = np.asarray(x, float); n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def load_blob(cache_dir, tname, eval_cls):
    hits = glob.glob(str(cache_dir / f'{tname}_{eval_cls}_n*.pkl')) or \
           glob.glob(str(cache_dir / f'{tname}_{eval_cls}_off{N_PROTO}_n*.pkl'))
    return pickle.load(open(hits[0], 'rb')) if hits else None


def gate_conf(conf, c=c0, scale=3.0):
    return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area = max(1e-9, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / area


def ece(conf, corr, nb=10):
    conf = np.asarray(conf, float); corr = np.asarray(corr, float)
    if len(conf) < nb:
        return float(np.mean(np.abs(corr - conf)))
    edges = np.quantile(conf, np.linspace(0, 1, nb + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    idx = np.digitize(conf, edges) - 1; idx = np.clip(idx, 0, nb - 1)
    e = 0.0
    for b in range(nb):
        m = idx == b
        if m.sum() == 0:
            continue
        e += m.sum() / len(conf) * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def main():
    t0 = time.time()
    cfg = RB.Config(); cfg.model.device = DEVICE
    cfg.model.detection_threshold = 0.0; cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'
    bank = {}
    for domain, pool in (('coco', RB.COCO_POOL), ('dior', RB.DIOR_POOL)):
        b = {}
        for cls in pool:
            ims, gts = RB.load_split(cfg, domain, cls, 0, N_PROTO)
            crops = RB.D.gt_crops(ims, gts)
            if not crops:
                continue
            Ft = RB.clip_encode_imgs(crops)
            pt = torch.from_numpy(Ft.mean(0)); b[cls] = pt / pt.norm()
        bank[domain] = b
    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    E_photo_d = {c: E_photo[i] for i, c in enumerate(all_names)}
    E_over_d = {c: E_over[i] for i, c in enumerate(all_names)}
    res = {}
    for det, cache_dir in DETECTORS.items():
        pre_c, pre_k, post_c, post_k = [], [], [], []
        for tname in RB.TASKS:
            src_dom, tgt_dom, pairs = RB.TASKS[tname]
            for src_cls, eval_cls in pairs:
                blob = load_blob(cache_dir, tname, eval_cls)
                if blob is None or src_cls not in bank[src_dom]:
                    continue
                per_img = blob['per_img']; F = blob['F']
                P_src = unit(bank[src_dom][src_cls].numpy())
                P_neg = unit(RB.Proto.get_negative_prototype(bank[src_dom], src_cls, NEG_TOPK).numpy())
                dc = E_photo_d[eval_cls] - E_over_d[src_cls]
                dc_mag = float(np.linalg.norm(dc))
                max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
                gamma = float(np.clip(PRIOR_SCALE * dc_mag * gate_conf(max_conf), 0.0, 4.0))
                W = unit(P_src + gamma * dc) - LAM * P_neg
                for p in per_img:
                    gts = p['gt']; boxes = p['box']; gd = p['gd']
                    if len(boxes) == 0:
                        continue
                    w = F[p['crop_idx']] @ W
                    s = np.sqrt(np.clip(np.maximum(0.0, w), 1e-6, 1.0))
                    for bi in range(len(boxes)):
                        cbox = boxes[bi]
                        if len(gts) == 0:
                            corr = 0
                        else:
                            ious = [iou(cbox, g) for g in gts]
                            corr = 1 if max(ious) >= 0.5 else 0
                        pre_c.append(float(gd[bi])); pre_k.append(corr)
                        post_c.append(float(s[bi]) if hasattr(s, '__len__') else float(s)); post_k.append(corr)
        ece_pre = ece(pre_c, pre_k); ece_post = ece(post_c, post_k)
        res[det] = {'ECE_pre': ece_pre, 'ECE_post': ece_post, 'n_boxes': len(pre_c)}
        print(f'  [{det}]  n_boxes={len(pre_c)}  ECE_pre(gd)={ece_pre:.4f}  ECE_post(CA-PC)={ece_post:.4f}'
              f'  (Δ={ece_post-ece_pre:+.4f})')
    json.dump(res, open(OUT, 'w'), indent=2)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
