# -*- coding: utf-8 -*-
"""probe_direction_efficiency.py — 把"方向效率"量出来（不靠嘴说）

问题: CA-PC 的注入方向 dc = photo(eval) - over(src) 与 TGE 的 (t_eval - t_bar),
      哪个"更值钱"?
口径上的坑: 直接比 gain/步长 会被"收益递减"污染 —— 走得多的方法边际效率天然低。
           所以必须在【同一条横轴】上比: 把两个方向都归一化, 推【相同的距离 s】,
           比同一步长下谁换回的 AP 多。同一 CLIP 空间、同一 P_src、同一打分/评估 → 唯一变量=方向。

输出: 每个检测器上 gain(s) 曲线 (两条方向), 以及各自峰值与峰值步长。
红线: 纯 B-pipeline 零目标 GT; 评测与 run_sota_tge.py 逐字同源(compute_ap50_img/compute_ap_coco)。
用法: HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu "C:/Users/张三/vg/Scripts/python.exe" -u probe_direction_efficiency.py --dets gdino owlv2
"""
import os, sys, json, pickle, glob, time, argparse
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cpu')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch
from pathlib import Path
import run_bidirectional as RB

RB.DEVICE = 'cpu'
BETA = RB.BETA; LAM = RB.LAM; NEG_TOPK = RB.NEG_TOPK; N_PROTO = RB.N_PROTO
S_GRID = [round(0.2 * i, 1) for i in range(0, 13)]      # 0.0 .. 2.4 的共同步长

DETECTORS = {
    'gdino':  Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache'),
    'owlv2':  Path(r'C:\Users\张三\Desktop\实验数据\probe_owl_cache'),
    'yolo':   Path(r'C:\Users\张三\Desktop\实验数据\probe_yolo_cache'),
    'rtdetr': Path(r'C:\Users\张三\Desktop\实验数据\probe_rtdetr_cache'),
}
DET_LABEL = {'gdino': 'grounding-tiny', 'owlv2': 'OWL-ViT', 'yolo': 'YOLO-style', 'rtdetr': 'RT-DETR'}
OUT = Path(r'C:\Users\张三\Desktop\实验数据\probe_direction_efficiency.json')


def unit(x):
    x = np.asarray(x, float); n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def load_blob(cache_dir, tname, eval_cls):
    hits = glob.glob(str(cache_dir / f'{tname}_{eval_cls}_n*.pkl')) or \
           glob.glob(str(cache_dir / f'{tname}_{eval_cls}_off{N_PROTO}_n*.pkl'))
    return pickle.load(open(hits[0], 'rb')) if hits else None


def score_rows(per_img, F, W):
    rows = []
    for p in per_img:
        if len(p['gd']):
            w = F[p['crop_idx']] @ W
            s = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
        else:
            s = np.zeros(0)
        rows.append((p['box'], s))
    return rows


def eval_pair(per_img, F, W):
    rows = score_rows(per_img, F, W)
    return (float(RB.Met.compute_ap50_img(rows, [p['gt'] for p in per_img])),
            float(RB.Met.compute_ap_coco(rows, [p['gt'] for p in per_img])))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--dets', nargs='+', default=['gdino', 'owlv2'])
    args = ap_.parse_args()
    t0 = time.time()

    cfg = RB.Config(); cfg.model.device = 'cpu'
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = RB.PRE_FILTER
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
        print(f'[protos] {domain}: {len(b)} 类', flush=True)

    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    E_photo_d = {c: E_photo[i] for i, c in enumerate(all_names)}
    t_bar = unit(E_photo.mean(0))

    curves = {}
    for det in args.dets:
        if det not in DETECTORS:
            continue
        cache_dir = DETECTORS[det]
        # 逐 cell 存: 每个 s 下两个方向的 AP50
        per_cell = []
        for tname in RB.TASKS:
            src_dom, tgt_dom, pairs = RB.TASKS[tname]
            for src_cls, eval_cls in pairs:
                blob = load_blob(cache_dir, tname, eval_cls)
                if blob is None or src_cls not in bank[src_dom]:
                    continue
                per_img = blob['per_img']; F = blob['F']
                P_src = unit(bank[src_dom][src_cls].numpy())
                P_neg = unit(RB.Proto.get_negative_prototype(bank[src_dom], src_cls, NEG_TOPK).numpy())
                dc = unit(E_photo_d[eval_cls] - E_over_d[src_cls])
                du = unit(E_photo_d[eval_cls] - t_bar)
                row = {'src': src_cls, 'eval': eval_cls, 'tname': tname,
                       'capc': [], 'tge': []}
                for s in S_GRID:
                    a1, _ = eval_pair(per_img, F, unit(P_src + s * dc) - LAM * P_neg)
                    a2, _ = eval_pair(per_img, F, unit(P_src + s * du) - LAM * P_neg)
                    row['capc'].append(round(a1, 4)); row['tge'].append(round(a2, 4))
                per_cell.append(row)
                print(f'  [{det}] {src_cls}->{eval_cls} done', flush=True)
        # 宏平均曲线
        capc_c = np.mean([r['capc'] for r in per_cell], axis=0)
        tge_c = np.mean([r['tge'] for r in per_cell], axis=0)
        curves[det] = {'s_grid': S_GRID,
                       'capc': [round(float(v), 4) for v in capc_c],
                       'tge': [round(float(v), 4) for v in tge_c],
                       'cells': per_cell}

    # ── 报告 ──
    for det, cv in curves.items():
        base0 = cv['capc'][0]
        print('\n' + '=' * 86, flush=True)
        print(f'{DET_LABEL[det]}  —  同一步长下两个方向换回的 AP50 (base 无修正 = {base0:.4f})', flush=True)
        print('=' * 86, flush=True)
        print(f'{"step":>6}{"CA-PC dc":>12}{"gain":>9}{"TGE dir":>12}{"gain":>9}{"TGE-CAPC":>10}', flush=True)
        for i, s in enumerate(S_GRID):
            g1 = cv['capc'][i] - base0; g2 = cv['tge'][i] - base0
            print(f'{s:>6.1f}{cv["capc"][i]:>12.4f}{g1:>+9.4f}{cv["tge"][i]:>12.4f}{g2:>+9.4f}{g2-g1:>+10.4f}', flush=True)
        i1 = int(np.argmax(cv['capc'])); i2 = int(np.argmax(cv['tge']))
        print(f'  峰值: CA-PC {cv["capc"][i1]:.4f} @s={S_GRID[i1]} | TGE {cv["tge"][i2]:.4f} @s={S_GRID[i2]}'
              f'  → TGE 峰值高 {cv["tge"][i2]-cv["capc"][i1]:+.4f}', flush=True)

    json.dump({'config': {'s_grid': S_GRID, 'beta': BETA, 'lam': LAM, 'neg_topk': NEG_TOPK,
                          'n_proto': N_PROTO, 'zero_target_gt': True,
                          'note': 'both directions L2-normalized then scaled to the SAME step norm s; '
                                  'identical P_src, P_neg, scoring and eval → only the direction differs'},
               'curves': curves, 'elapsed_s': round(time.time() - t0, 1)},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
