# -*- coding: utf-8 -*-
"""run_closed_loop_fewshot.py — #8 少标注/few-shot 敏感性（标注省多少才能赶上零GT γ）

协议: 每单元把 per_img(含 GT) 拆成 fit(K 张)/test(余下)。
  K=0  → 我们的零GT γ_c=‖dc‖×gate(conf)（闭环方法，零目标域标注）
  K>0  → 在 fit 集上网格搜最优 β，test 集评估（监督少标注基线）
比较: 监督基线需多少目标域 GT 才能追上零GT γ 的 AP50 —— 量化"免标注"省下的标注量。

用法:
  HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu "C:/Users/张三/vg/Scripts/python.exe" -u run_closed_loop_fewshot.py
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
BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
K_VALS = [0, 10, 50, 100, 200]
DETECTORS = {'gdino': Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache'),
             'owlv2': Path(r'C:\Users\张三\Desktop\实验数据\probe_owl_cache'),
             'yolo':  Path(r'C:\Users\张三\Desktop\实验数据\probe_yolo_cache')}
bal = json.load(open(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced.json'))
c0 = bal['config']['gate']['c0']; PRIOR_SCALE = bal['config']['gate']['prior_scale']
OUT = Path(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_fewshot.json')


def unit(x):
    x = np.asarray(x, float); n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def load_blob(cache_dir, tname, eval_cls):
    hits = glob.glob(str(cache_dir / f'{tname}_{eval_cls}_n*.pkl')) or \
           glob.glob(str(cache_dir / f'{tname}_{eval_cls}_off{N_PROTO}_n*.pkl'))
    return pickle.load(open(hits[0], 'rb')) if hits else None


def gate_conf(conf, c=c0, scale=3.0):
    return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))


def ap_of(per_img, F, W):
    rows = []
    for p in per_img:
        if len(p['gd']):
            w = F[p['crop_idx']] @ W
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
                dc_mag = float(np.linalg.norm(dc))
                max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
                gamma = float(np.clip(PRIOR_SCALE * dc_mag * gate_conf(max_conf), 0.0, 4.0))
                # K=0: 零GT γ
                row = {'det': det, 'tname': tname, 'src': src_cls, 'eval': eval_cls}
                # fit/test split by GT-positive indices (deterministic: first K)
                pos_idx = [i for i, p in enumerate(per_img) if len(p['gt']) > 0]
                for K in K_VALS:
                    if K == 0:
                        ap = ap_of(per_img, F, unit(P_src + gamma * dc) - LAM * P_neg)
                    else:
                        if len(pos_idx) <= K:
                            # 不足 K 张 GT 图：用全部做 fit，test 退化（记 -1 表示不可用）
                            ap = -1.0
                        else:
                            fit = [per_img[i] for i in pos_idx[:K]]
                            test = [per_img[i] for i in pos_idx[K:]]
                            best_b, best_ap = 0.0, -1
                            for bb in BETA_GRID:
                                a = ap_of(fit, F, unit(P_src + bb * dc) - LAM * P_neg)
                                if a > best_ap:
                                    best_ap, best_b = a, bb
                            ap = ap_of(test, F, unit(P_src + best_b * dc) - LAM * P_neg)
                    row[f'K{K}'] = ap
                cells.append(row)
        data[det] = cells
    print('\n[#8 few-shot]  每检测器各 K 平均 AP50 (K=0 = 零GT γ)\n')
    summary = {}
    for det, cells in data.items():
        print(f'  [{det}]  n={len(cells)}')
        means = {}
        for K in K_VALS:
            vals = [c[f'K{K}'] for c in cells if c[f'K{K}'] >= 0]
            m = float(np.mean(vals)) if vals else float('nan')
            means[K] = m
            print(f'    K={K:<4} AP50={m:.4f}  (n_valid={len(vals)})')
        summary[det] = means
    json.dump({'K_vals': K_VALS, 'summary': summary,
               'cells': [c for cells in data.values() for c in cells],
               'elapsed_s': time.time() - t0}, open(OUT, 'w'), indent=2, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
