# -*- coding: utf-8 -*-
"""run_closed_loop_injection.py — #7 注入策略比较（推理期原型空间调制，零训练）

三种把闭环 γ_c 注入检测评分的策略（均复用已有检测缓存 F/per_img，零GT、零微调）:
  1. weight_mod (当前 CA-PC):   P = unit(P_src + γ·dc) − λ·P_neg;  s = √(gd·clip(F@P,1e-6,1))
  2. logit_bias:                w = F@(P_src−λ·P_neg) + γ·(F@dc);  s = √(gd·clip(w,1e-6,1))
  3. affine:                    P = unit((1+γ)·P_src + γ·dc) − λ·P_neg;  s = √(gd·clip(F@P,1e-6,1))
比较: 三检测器上 AP50 与 oracle 覆盖；验证"权重调制是否最优/可有等价替代"。

用法:
  HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu "C:/Users/张三/vg/Scripts/python.exe" -u run_closed_loop_injection.py
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
GATE_FROM = ['gdino', 'owlv2']
bal = json.load(open(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced.json'))
c0 = bal['config']['gate']['c0']; PRIOR_SCALE = bal['config']['gate']['prior_scale']
OUT = Path(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_injection.json')


def unit(x):
    x = np.asarray(x, float); n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def load_blob(cache_dir, tname, eval_cls):
    hits = glob.glob(str(cache_dir / f'{tname}_{eval_cls}_n*.pkl')) or \
           glob.glob(str(cache_dir / f'{tname}_{eval_cls}_off{N_PROTO}_n*.pkl'))
    return pickle.load(open(hits[0], 'rb')) if hits else None


def gate_conf(conf, c=c0, scale=3.0):
    return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))


def ap_weightmod(per_img, F, P):
    rows = []
    for p in per_img:
        if len(p['gd']):
            w = F[p['crop_idx']] @ P
            s = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
        else:
            s = np.zeros(0)
        rows.append((p['box'], s))
    return RB.Met.compute_ap50_img(rows, [p['gt'] for p in per_img])


def ap_logitbias(per_img, F, W0, dc, gamma):
    rows = []
    for p in per_img:
        if len(p['gd']):
            w = F[p['crop_idx']] @ W0 + gamma * (F[p['crop_idx']] @ dc)
            s = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
        else:
            s = np.zeros(0)
        rows.append((p['box'], s))
    return RB.Met.compute_ap50_img(rows, [p['gt'] for p in per_img])


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
    data = {}
    for det, cache_dir in DETECTORS.items():
        cells = []
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
                base = ap_weightmod(per_img, F, P_src - LAM * P_neg)
                # oracle β*
                best_b, best_ap = 0.0, -1
                for bb in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
                    ap = ap_weightmod(per_img, F, unit(P_src + bb * dc) - LAM * P_neg)
                    if ap > best_ap:
                        best_ap, best_b = ap, bb
                dc_mag = float(np.linalg.norm(dc))
                max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
                gamma = float(np.clip(PRIOR_SCALE * dc_mag * gate_conf(max_conf), 0.0, 4.0))
                W0 = P_src - LAM * P_neg
                ap_wm = ap_weightmod(per_img, F, unit(P_src + gamma * dc) - LAM * P_neg)
                ap_lb = ap_logitbias(per_img, F, W0, dc, gamma)
                ap_aff = ap_weightmod(per_img, F, unit((1.0 + gamma) * P_src + gamma * dc) - LAM * P_neg)
                cells.append({'det': det, 'tname': tname, 'src': src_cls, 'eval': eval_cls,
                              'base': base, 'ap_star': best_ap, 'gamma': gamma,
                              'ap_wm': ap_wm, 'ap_lb': ap_lb, 'ap_aff': ap_aff})
        data[det] = cells
    print('\n[#7 注入策略比较]  AP50 (oracle%)\n')
    summary = {}
    for det, cells in data.items():
        ap_star = float(np.mean([c['ap_star'] for c in cells]))
        ap_base = float(np.mean([c['base'] for c in cells]))
        print(f'  [{det}]  n={len(cells)}')
        row = {}
        for key, label in (('ap_wm', 'weight_mod(当前)'), ('ap_lb', 'logit_bias'), ('ap_aff', 'affine')):
            m = float(np.mean([c[key] for c in cells]))
            row[label] = {'ap': m, 'oracle%': m / ap_star * 100, 'vs_base': m - ap_base}
            print(f'    {label:<16}{m:.4f} ({m/ap_star*100:5.1f}%)  vs基线 {m-ap_base:+.4f}')
        print(f'    {"oracle":<16}{ap_star:.4f} (100.0%)')
        print(f'    {"基线β=0":<16}{ap_base:.4f}')
        summary[det] = row
    json.dump({'summary': summary, 'cells': [c for cells in data.values() for c in cells],
               'elapsed_s': time.time() - t0}, open(OUT, 'w'), indent=2, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
