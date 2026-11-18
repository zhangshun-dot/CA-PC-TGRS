# -*- coding: utf-8 -*-
r"""tge_v2_detector_grid.py — 检测器谱系的 (β,λ) 全响应面

目的
   论文 §4.8 / Table 11 的检测器谱系里，OWLv2 的增益是 −0.0316（唯一负值）。
   §7(1) 把它解释为"受益对象边界"。审稿人最省力的反驳是：
   "你只是没给 OWLv2 调够参数"。本实验给出决定性回答——
   在 OWLv2 上把 (β,λ) 整个网格扫完（含负半轴），取网格最大值作为
   **允许用目标域 GT 事后挑配置的上界**；若上界仍 ≤0，则负号是检测器属性，不是调参问题。

做法
   每个 (检测器 × 类) 只用缓存的检测框，CLIP 框特征 F 只编码一次；
   W = normalize(P_src + β·Δ) − λ·P_neg 是 512 维向量，改 (β,λ) 后重算 fused 分数几乎免费。
   因此"全网格"的代价 ≈ "单配置"的代价。

信息使用协议
   网格最大值 = oracle 上界（用目标域 GT 挑配置），与论文既有 oracle 行同性质，仅作上界。
   网格内同时记录 (2.0,0.5) 主配置与 (1.0,0.25) 目标域无关配置的取值。

运行
   python tge_v2_detector_grid.py            # 四个检测器全跑
   python tge_v2_detector_grid.py --det owlv2
输出
   实验数据/tge_v2_detector_grid.json
"""
import sys, os, json, pickle, time, argparse
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T
from tge_v2_main import (load_coco_all, collect_features, fused_arrays, dataset_ap,
                         recall_proposals, TARGETS, K_CALIB, SRC_TMPL, TGT_TMPL, DATA)

PROTO_PKL = DATA / 'tge_v2_src_protos.pkl'
OUT = DATA / 'tge_v2_detector_grid.json'
SRC_KEY = 'DIOR'                       # 与 Table 11 谱系一致：DIOR 源原型

BETAS = [-4.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]
MAIN = (2.0, 0.5)                      # 论文主配置
FREE = (1.0, 0.25)                     # 目标域无关配置

DETS = [
    ('gdino-tiny', lambda name: DATA / ('tge_v2_dets_%s.pkl' % name)),
    ('gdino-base', lambda name: DATA / ('tge_v2_dets_GDINOB-%s.pkl' % name)),
    ('owlv2',      lambda name: DATA / ('tge_v2_dets_OWLV2-%s.pkl' % name)),
    ('yolo-world', lambda name: DATA / 'supp_dets_yoloworld' / ('%s.pkl' % name)),
]


def load_dets(path, images):
    """返回 [(boxes, scores), ...]，长度与 images 对齐；对不齐或缺失返回 None。"""
    if not path.exists():
        return None
    d = pickle.load(open(path, 'rb'))
    if isinstance(d, dict):
        # 有些缓存是 {img_id: (boxes, scores)} 形式
        d = list(d.values())
    if len(d) != len(images):
        return None
    for item in d[:3]:
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            return None
    return d


def grid_for_cell(per_img, F, P_src, P_neg, Delta, base50):
    """返回 (grid, best, frac_pos)

    grid[beta][lam] = {'gain': ΔAP50, 'frac_pos': vis>0 的框占比}
    frac_pos 是**塌陷守卫**：frac_pos→0 时 fused=sqrt(gd·clip(vis,1e-6,1)) 退化为 gd 的
    单调变换，排序与基线完全一致 ⇒ gain≡0 是"方法自关闭"，不是真增益。
    """
    Fn = F
    nbox = int(sum(len(p['gd']) for p in per_img))
    grid = {}
    best = None
    best_live = None          # 仅在非塌陷区里取最优（frac_pos ≥ 0.5）
    frac_pos = {}
    for b in BETAS:
        P_hat = P_src + b * Delta.to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        col = {}
        colf = {}
        for lam in LAMS:
            W = (P_hat.cpu().numpy() - lam * P_neg.cpu().numpy())
            fused_arrays(per_img, Fn, W)
            g = dataset_ap(per_img, 'fused') - base50
            npos = 0
            for p in per_img:
                if len(p['gd']):
                    npos += int((np.maximum(0.0, Fn[p['crop_idx']] @ W) > 0).sum())
            fr = npos / max(nbox, 1)
            col['%g' % lam] = round(float(g), 4)
            colf['%g' % lam] = round(fr, 4)
            if best is None or g > best['gain']:
                best = {'beta': b, 'lam': lam, 'gain': round(float(g), 4), 'frac_pos': round(fr, 4)}
            if fr >= 0.5 and (best_live is None or g > best_live['gain']):
                best_live = {'beta': b, 'lam': lam, 'gain': round(float(g), 4), 'frac_pos': round(fr, 4)}
        grid['%g' % b] = col
        frac_pos['%g' % b] = colf
    return grid, best, best_live, frac_pos


def run_detector(det, path_fn, protos, results, t_start):
    print('\n' + '=' * 92, flush=True)
    print('[检测器] %s' % det, flush=True)
    print('=' * 92, flush=True)
    rows = {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = load_coco_all(coco_cls)
        if len(images) < K_CALIB + 10:
            print('  !! %s 图太少(%d)，跳过' % (name, len(images)), flush=True)
            continue
        dets = load_dets(path_fn(name), images)
        if dets is None:
            print('  !! %s 缓存缺失或长度不匹配，跳过' % name, flush=True)
            continue
        t0 = time.time()
        per_img, F = collect_features(images, gts, dets)
        t_feat = time.time() - t0
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        Delta = d_y - s_xs

        base50 = dataset_ap(per_img, 'gd')
        grid, best, best_live, frac_pos = grid_for_cell(per_img, F, P_src, P_neg, Delta, base50)
        rec = recall_proposals(per_img)
        # 全网格统计：所有配置 / 仅非塌陷配置（frac_pos≥0.5）
        all_g, live_g = [], []
        for b in grid:
            for lam in grid[b]:
                all_g.append(grid[b][lam])
                if frac_pos[b][lam] >= 0.5:
                    live_g.append(grid[b][lam])
        rows[name] = {
            'n_pool': len(images), 'recall': round(rec, 4),
            'base50': round(float(base50), 4),
            'n_box_postfilter': int(sum(len(p['gd']) for p in per_img)),
            'gain_main_2.0_0.5': grid['2']['0.5'],
            'gain_free_1.0_0.25': grid['1']['0.25'],
            'gain_zero_extrap_b0_l0.5': grid['0']['0.5'],
            'best': best,
            'best_live': best_live,
            'grid_all_max': round(float(max(all_g)), 4),
            'grid_all_min': round(float(min(all_g)), 4),
            'grid_all_n_pos': int(sum(1 for g in all_g if g > 0)),
            'grid_all_n': len(all_g),
            'grid_live_max': round(float(max(live_g)), 4) if live_g else None,
            'grid_live_min': round(float(min(live_g)), 4) if live_g else None,
            'grid_live_n_pos': int(sum(1 for g in live_g if g > 0)),
            'grid_live_n': len(live_g),
            'frac_vis_pos': frac_pos,
            'grid': grid,
        }
        print('  %-16s base=%.4f 主配置%+.4f | 全网格最优%+.4f (β=%.1f λ=%.2f frac=%.2f) | '
              '非塌陷区最优%+.4f | 非塌陷%d格中正增益%d格 (%.0fs)'
              % (name, base50, grid['2']['0.5'], best['gain'], best['beta'], best['lam'],
                 best['frac_pos'],
                 (best_live['gain'] if best_live else float('nan')),
                 len(live_g), rows[name]['grid_live_n_pos'], t_feat), flush=True)
        # 增量落盘
        results['detectors'][det] = {'classes': rows}
        dump(results, t_start)
    if rows:
        keys = ['base50', 'gain_main_2.0_0.5', 'gain_free_1.0_0.25', 'gain_zero_extrap_b0_l0.5']
        m = {k: round(float(np.mean([rows[c][k] for c in rows])), 4) for k in keys}
        m['grid_all_max_mean'] = round(float(np.mean([rows[c]['grid_all_max'] for c in rows])), 4)
        m['grid_live_max_mean'] = round(float(np.mean([rows[c]['grid_live_max'] for c in rows])), 4)
        m['n_cells_grid_all_max_positive'] = int(sum(1 for c in rows if rows[c]['grid_all_max'] > 0))
        m['n_cells_grid_live_max_positive'] = int(sum(1 for c in rows if rows[c]['grid_live_max'] > 0))
        m['n_cells_main_positive'] = int(sum(1 for c in rows if rows[c]['gain_main_2.0_0.5'] > 0))
        m['recall'] = round(float(np.mean([rows[c]['recall'] for c in rows])), 4)
        results['detectors'][det]['mean'] = m
        dump(results, t_start)
        print('  --> 均值: base=%.4f 主配置%+.4f 目标域无关%+.4f | 全网格最优(含塌陷)%+.4f '
              '| 非塌陷区最优%+.4f | 非塌陷有正增益的类 %d/6'
              % (m['base50'], m['gain_main_2.0_0.5'], m['gain_free_1.0_0.25'],
                 m['grid_all_max_mean'], m['grid_live_max_mean'],
                 m['n_cells_grid_live_max_positive']), flush=True)
    return rows


def dump(results, t_start):
    results['elapsed_min'] = round((time.time() - t_start) / 60, 2)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', default=None, help='只跑某个检测器 (gdino-tiny/gdino-base/owlv2/yolo-world)')
    args = ap.parse_args()
    t_start = time.time()

    protos_all = pickle.load(open(PROTO_PKL, 'rb'))
    protos = protos_all[SRC_KEY]
    # 缓存里存的是 numpy ndarray，统一转到 GPU 张量
    # （tge_probe.get_neg_proto 内部把 others 放到 DEVICE 后再与 pos 相乘，
    #   所以 pos 也必须在 DEVICE 上，否则会 device mismatch）
    protos = {k: (v if torch.is_tensor(v) else torch.from_numpy(np.asarray(v, dtype=np.float32))).to(U.DEVICE)
              for k, v in protos.items()}
    print('源原型: %s, %d 类 (已转 torch)' % (SRC_KEY, len(protos)), flush=True)

    results = {
        'config': {
            'protocol': 'v2 frozen 同协议; DIOR 源原型; pre-filter 0.03; 数据集级 11点 AP50',
            'betas': BETAS, 'lambdas': LAMS,
            'main_config': '%g,%g' % MAIN, 'target_free_config': '%g,%g' % FREE,
            'oracle_note': 'best = 网格内最大值，允许用目标域 GT 事后挑配置的上界',
            'protos': str(PROTO_PKL.name), 'src_key': SRC_KEY,
        },
        'detectors': results_prev(),
    }
    for det, path_fn in DETS:
        if args.det and det != args.det:
            continue
        run_detector(det, path_fn, protos, results, t_start)

    print('\n' + '=' * 104, flush=True)
    print('%-12s %8s %10s %10s %12s %12s %8s'
          % ('检测器', 'base50', '主配置', '目标域无关', '全网格最优', '非塌陷区最优', 'recall'), flush=True)
    print('-' * 104, flush=True)
    for det in results['detectors']:
        m = results['detectors'][det].get('mean') or {}
        if 'base50' not in m:
            continue
        print('%-12s %8.4f %+10.4f %+10.4f %+12.4f %+12.4f %8.4f'
              % (det, m['base50'], m['gain_main_2.0_0.5'], m['gain_free_1.0_0.25'],
                 m.get('grid_all_max_mean', float('nan')), m.get('grid_live_max_mean', float('nan')),
                 m.get('recall', float('nan'))), flush=True)
    print('=' * 104, flush=True)
    print('→ %s  (%.1f 分钟)' % (OUT.name, (time.time() - t_start) / 60), flush=True)


def results_prev():
    if OUT.exists():
        try:
            return json.load(open(OUT, encoding='utf-8')).get('detectors', {})
        except Exception:
            pass
    return {}


if __name__ == '__main__':
    main()
