# -*- coding: utf-8 -*-
"""run_sota_tge.py — ISPRS 缺口#95: 匹配协议 SOTA 基线对比 (TGE vs CA-PC)

目的: 在 CA-PC 完全相同的检测缓存 + 完全相同评估口径上, 把 CA-PC 与
      ExDet 的 Text-Guided Extrapolation (TGE, arXiv:2606.09360, 训练-free/零真实数据)
      做一对一对比, 验证口径一致、结论可比。

为什么能完美对口径: CA-PC 与 TGE 都是「检测器输出之后、用 CLIP 原型重打分」(post-hoc
      prototype rescoring)。所以直接复用 4 检测器缓存里的 per_img(box/gd/gt) 与 F(CLIP crop 特征),
      只换打分用的正原型 W, 其余 (框、NMS、AP50/mAP 计算) 全部一致 → 唯一变量是原型。

方法 (打分 s = sqrt(gd * clip(max(0, F·W))), 与 CA-PC 完全相同):
  base       : W = unit(P_src)                    - LAM·P_neg        # 仅源类 CLIP 图像原型
  naive_clip : W = unit(t_eval)                   - LAM·P_neg        # 仅目标类文本原型 (最朴素)
  tge        : W = unit(P_src + (t_eval - t_bar)) - LAM·P_neg        # TGE Eq.1 类别外推 (无域项)
  ca_pc      : W = unit(P_src + BETA*(t_eval_photo - t_src_over)) - LAM·P_neg   # CA-PC (含跨域 Δc)
  另报各自 grid 最优 (*) 作为上界, 不 handicap 任一方。

TGE 忠实度: 严格按 ExDet Eq.1  v̂_n = v̄_b + (t_n - t̄_b), DeltaSpace 性质 (文本偏移≈视觉偏移)。
  v̄_b = P_src (源类 CLIP 图像原型, 与 CA-PC 同, 控制变量); t_n = text(目标类); t̄_b = 基类文本均值。
  与 CA-PC 的唯一区别: TGE 的 Δ 在 natural 文本空间内 (无 overhead/域项), CA-PC 显式建模跨域
  (t_eval_photo - t_src_overhead)。这正是 CA-PC 卖点「canonical-anchor 跨域」的对照点。

红线: 纯 B-pipeline (零目标 GT); 绝不混入 A-CSPC rerank 缓存。

用法 (CPU, 不重跑 GPU 检测):
  HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu ^
  "C:/Users/张三/vg/Scripts/python.exe" -u run_sota_tge.py
  # 小样本验证: --dets gdino
  # 全量:        (默认全部 4 检测器)
"""
import os, sys, json, pickle, glob, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cpu')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch
from pathlib import Path
import run_bidirectional as RB

RB.DEVICE = 'cpu'
DEVICE = 'cpu'
BETA = RB.BETA                       # 2.0 (与论文 tab:det 一致)
LAM = RB.LAM                         # 0.5
NEG_TOPK = RB.NEG_TOPK               # 3
N_PROTO = RB.N_PROTO                 # 20
BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

DETECTORS = {
    'gdino':  Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache'),
    'owlv2':  Path(r'C:\Users\张三\Desktop\实验数据\probe_owl_cache'),
    'yolo':   Path(r'C:\Users\张三\Desktop\实验数据\probe_yolo_cache'),
    'rtdetr': Path(r'C:\Users\张三\Desktop\实验数据\probe_rtdetr_cache'),
}
# 论文 tab:det 的检测器显示名
DET_LABEL = {'gdino': 'grounding-tiny', 'owlv2': 'OWL-ViT', 'yolo': 'YOLO-style', 'rtdetr': 'RT-DETR'}
OUT = Path(r'C:\Users\张三\Desktop\实验数据\run_sota_tge.json')


def unit(x):
    x = np.asarray(x, float)
    n = np.linalg.norm(x)
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
    ap50 = RB.Met.compute_ap50_img(rows, [p['gt'] for p in per_img])
    mAP = RB.Met.compute_ap_coco(rows, [p['gt'] for p in per_img])
    return float(ap50), float(mAP)


def main():
    import argparse
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--dets', nargs='+', default=list(DETECTORS.keys()))
    ap_.add_argument('--tasks', nargs='+', default=list(RB.TASKS))
    args = ap_.parse_args()
    t0 = time.time()

    cfg = RB.Config()
    cfg.model.device = DEVICE
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    # ── 源原型池 (CLIP GT crop, 检测器无关) ──
    bank = {}
    for domain, pool in (('coco', RB.COCO_POOL), ('dior', RB.DIOR_POOL)):
        b = {}
        for cls in pool:
            ims, gts = RB.load_split(cfg, domain, cls, 0, N_PROTO)
            crops = RB.D.gt_crops(ims, gts)
            if not crops:
                continue
            Ft = RB.clip_encode_imgs(crops)
            pt = torch.from_numpy(Ft.mean(0))
            b[cls] = pt / pt.norm()
        bank[domain] = b
        print(f'[protos] {domain}: {len(b)} 类')

    # ── 文本原型 ──
    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    E_photo_d = {c: E_photo[i] for i, c in enumerate(all_names)}
    E_over_d = {c: E_over[i] for i, c in enumerate(all_names)}
    t_bar = unit(E_photo.mean(0))                      # TGE 基类文本均值 (DeltaSpace 锚)

    # ── 载入 tab:det 同源 dc×conf gate 尺度 (零GT, 来自 gdino+owlv2 pooled) ──
    gate_cfg = json.load(open(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced.json'))['config']['gate']
    PRIOR_SCALE = gate_cfg['prior_scale']
    c0 = gate_cfg['c0']
    def gate_conf(conf, c=c0, scale=3.0):
        return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))

    data = {}
    for det in args.dets:
        if det not in DETECTORS:
            continue
        cache_dir = DETECTORS[det]
        cells = []
        for tname in args.tasks:
            src_dom, tgt_dom, pairs = RB.TASKS[tname]
            for src_cls, eval_cls in pairs:
                blob = load_blob(cache_dir, tname, eval_cls)
                if blob is None or src_cls not in bank[src_dom]:
                    print(f'  !! {det}/{tname}/{eval_cls} 缺缓存')
                    continue
                per_img = blob['per_img']; F = blob['F']
                P_src = unit(bank[src_dom][src_cls].numpy())
                P_neg = unit(RB.Proto.get_negative_prototype(bank[src_dom], src_cls, NEG_TOPK).numpy())
                t_eval = E_photo_d[eval_cls]                       # text('a photo of '+eval_cls)
                dc = E_photo_d[eval_cls] - E_over_d[src_cls]        # CA-PC 跨域 Δc

                Ws = {
                    'base':       unit(P_src) - LAM * P_neg,
                    'naive_clip': unit(t_eval) - LAM * P_neg,
                    'tge':        unit(P_src + (t_eval - t_bar)) - LAM * P_neg,
                    'ca_pc':      unit(P_src + BETA * dc) - LAM * P_neg,
                }
                # dc×conf gate-balanced (与 tab:det 同源, 保证两表 CA-PC 口径一致)
                max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
                dc_mag = float(np.linalg.norm(dc))
                gamma = float(np.clip(PRIOR_SCALE * dc_mag * gate_conf(max_conf), 0.0, 4.0))
                Ws['ca_pc_g'] = unit(P_src + gamma * dc) - LAM * P_neg
                cell = {'det': det, 'tname': tname, 'src': src_cls, 'eval': eval_cls,
                        'n_img': len(per_img), 'n_box': int(sum(len(p['gd']) for p in per_img)),
                        'dc_mag': round(dc_mag, 4), 'max_conf': round(max_conf, 4), 'gamma': round(gamma, 4)}
                for k, W in Ws.items():
                    a50, m = eval_pair(per_img, F, W)
                    cell[k] = round(a50, 4); cell[k + '_m'] = round(m, 4)
                # grid 最优 (不 handicap 任一方)
                best_tge, bap_tge, bm_tge = -1, -1, 0
                for bb in BETA_GRID:
                    a50, m = eval_pair(per_img, F, unit(P_src + bb * (t_eval - t_bar)) - LAM * P_neg)
                    if a50 > bap_tge:
                        bap_tge, bm_tge, best_tge = a50, m, bb
                cell['tge_star'] = round(bap_tge, 4); cell['tge_star_m'] = round(bm_tge, 4)
                cell['tge_beta'] = best_tge
                best_c, bap_c, bm_c = -1, -1, 0
                for bb in BETA_GRID:
                    a50, m = eval_pair(per_img, F, unit(P_src + bb * dc) - LAM * P_neg)
                    if a50 > bap_c:
                        bap_c, bm_c, best_c = a50, m, bb
                cell['ca_pc_star'] = round(bap_c, 4); cell['ca_pc_star_m'] = round(bm_c, 4)
                cell['ca_pc_beta'] = best_c
                cells.append(cell)
                print(f'  [{det}] {src_cls}->{eval_cls}: base={cell["base"]:.4f} '
                      f'naive={cell["naive_clip"]:.4f} tge={cell["tge"]:.4f}(*{cell["tge_star"]:.4f}) '
                      f'ca_pc={cell["ca_pc"]:.4f}(*{cell["ca_pc_star"]:.4f}) '
                      f'| mAP: base={cell["base_m"]:.4f} ca_pc={cell["ca_pc_m"]:.4f}')
        data[det] = cells
        # ★ 每检测器单独落盘 (防 --dets 子集重跑覆盖 4 检测器全量产物)
        try:
            json.dump({'config': {'beta': BETA, 'lam': LAM, 'neg_topk': NEG_TOPK, 'n_proto': N_PROTO,
                                  'zero_target_gt': True, 'clip': RB.CLIP_ID, 'detector': det,
                                  'tge_ref': 'ExDet arXiv:2606.09360 Eq.1 DeltaSpace (faithful re-impl, training-free)'},
                       'cells': {det: cells}},
                      open(str(OUT).replace('.json', f'_{det}.json'), 'w', encoding='utf-8'),
                      indent=2, ensure_ascii=False)
            print(f'[save] {det} -> run_sota_tge_{det}.json', flush=True)
        except Exception as _e:
            print(f'[warn] per-det save failed: {_e}', flush=True)

    # ── 每检测器宏平均 (AP50 / mAP) ──
    print('\n' + '=' * 92)
    print('匹配协议 SOTA 对比 (DIOR→COCO, 零目标 GT; 数字=宏平均 AP50 / mAP@[.5:.95])')
    print('=' * 92)
    summary = {}
    hdr = f'{"detector":<14}{"base":>14}{"naive":>14}{"TGE":>14}{"TGE*":>14}{"CA-PC":>14}{"CA-PC(g)":>14}'
    print(hdr)
    for det, cells in data.items():
        if not cells:
            continue
        def mean(key):
            return float(np.mean([c[key] for c in cells]))
        row = {k: mean(k) for k in ('base', 'naive_clip', 'tge', 'tge_star', 'ca_pc', 'ca_pc_g',
                                    'base_m', 'naive_clip_m', 'tge_m', 'tge_star_m', 'ca_pc_m', 'ca_pc_g_m')}
        summary[det] = row
        print(f'{DET_LABEL[det]:<14}'
              f'{row["base"]:>7.4f}/{row["base_m"]:<7.4f}'
              f'{row["naive_clip"]:>7.4f}/{row["naive_clip_m"]:<7.4f}'
              f'{row["tge"]:>7.4f}/{row["tge_m"]:<7.4f}'
              f'{row["tge_star"]:>7.4f}/{row["tge_star_m"]:<7.4f}'
              f'{row["ca_pc"]:>7.4f}/{row["ca_pc_m"]:<7.4f}'
              f'{row["ca_pc_g"]:>7.4f}/{row["ca_pc_g_m"]:<7.4f}')
        g_ca = row['ca_pc_g'] - row['base']; g_tge = row['tge'] - row['base']
        print(f'   增益(vs base): CA-PC(gate)=+{g_ca:.4f}  TGE=+{g_tge:.4f}  '
              f'CA-PC(gate) 优于 TGE={g_ca - g_tge:+.4f} (AP50)')

    # ── 汇总落盘: 与已有 aggregate **合并** (子集重跑绝不截断其它检测器) ──
    merged_cells, merged_summary = {}, {}
    if OUT.exists():
        try:
            _prev = json.load(open(OUT, encoding='utf-8'))
            merged_cells = _prev.get('cells', {}) or {}
            merged_summary = _prev.get('summary', {}) or {}
            print(f'[merge] 载入已有 aggregate: {sorted(merged_cells.keys())}', flush=True)
        except Exception as e:
            print(f'[merge] 读取失败, 从空开始: {e}', flush=True)
    merged_cells.update(data)
    merged_summary.update(summary)
    json.dump({'config': {'beta': BETA, 'lam': LAM, 'neg_topk': NEG_TOPK, 'n_proto': N_PROTO,
                          'zero_target_gt': True, 'clip': RB.CLIP_ID,
                          'tge_ref': 'ExDet arXiv:2606.09360 Eq.1 DeltaSpace (faithful re-impl, training-free)',
                          'note': 'identical cached detections + identical AP50/mAP eval as CA-PC; only scoring prototype W differs; '
                                  'aggregate is MERGED so a --dets subset re-run never truncates other detectors'},
               'summary': merged_summary, 'cells': merged_cells,
               'elapsed_s': time.time() - t0},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'[save] aggregate cells={sorted(merged_cells.keys())}', flush=True)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
