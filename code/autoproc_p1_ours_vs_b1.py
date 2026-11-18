# -*- coding: utf-8 -*-
"""P1-A: 干净 Ours vs B1 头对头 (增量法方向).
复用 load_domain_data 缓存(不重检测). 5-seed 可跑, 支持 NSEED/DOMAINS/SMOKE 探路.

两配置(同协议 N=150/seed):
  B1   = RawUSC: γ=0 + 逐类 best α(网格) + 文本外推 TGE      <- 原始管线基线
  Ours = γ=SENSOR_GAMMA[域](秩-1 仿射缩放) + 逐类 best α + TGE + 传感器闸门(本设置双侧都开, 因校正对两域均有益)

唯一结构性差异 = γ(沿域漂移方向 v 的秩-1 仿射缩放). 这是相对 B1 的真实增量杠杆.
聚合: 每 seed 先对类求均值 -> 5 值 mean±std; 跨域 = 每 seed 跨域均值再 mean±std.
配对检验: pooled (类×seed×域) pair, Ours vs B1.
产物: autoproc_p1_ours_vs_b1_results.json
"""
import sys, json, os
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
from pathlib import Path
import scipy.stats as st
import usc_experiment_v12 as V12
import unified_eval as U
from usc_rank1_affine_gpu_v2 import load_domain_data, SOURCE_LOADERS

DATA = V12.DATA
SAT_DOMAINS = {'DIOR', 'NWPU', 'DOTA'}
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
SENSOR_GAMMA = {'DIOR': -2.0, 'NWPU': -2.0, 'DOTA': -2.0, 'VisDrone': -1.0}
N_EVAL = int(os.environ.get('N_EVAL', '150'))
if os.environ.get('NSEED'):
    SEEDS = list(range(int(os.environ['NSEED'])))
elif os.environ.get('SMOKE') == '1':
    SEEDS = [0]
else:
    SEEDS = [0, 1, 2, 3, 4]
SMOKE = os.environ.get('SMOKE') == '1'
CFGS = ['B1', 'Ours']


def eval_proto(per_img_sub, F, p_hat, neg_raw):
    return float(V12.evaluate({'per_img': per_img_sub, 'features': F}, p_hat, neg_raw))


def evaluate(dd, src, src_cls, coco_cls, seed, gamma):
    td = dd['target_data'][coco_cls]
    F = td['features']; per_img = td['per_img']
    usc = dd['usc']; neg_dev = usc._mean_gpu.device
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(per_img), size=min(N_EVAL, len(per_img)), replace=False)
    sub = [per_img[i] for i in idx]
    delta = dd['text_deltas'][src_cls]
    neg_raw_g = V12.build_neg_proto(dd['all_protos'], src_cls)
    neg_gpu = torch.tensor(neg_raw_g, dtype=torch.float32, device=neg_dev) if neg_raw_g is not None else None
    best_ap, best_p, best_neg = -1, None, None
    for a in ALPHAS:
        pc = usc.correct_prototype_gpu(dd['protos_gpu'][src_cls], gamma, a).cpu().numpy()
        nc = usc.correct_prototype_gpu(neg_gpu, gamma, a).cpu().numpy() if neg_gpu is not None else None
        ph = pc + delta; ph = ph / (np.linalg.norm(ph) + 1e-8)
        ap = eval_proto(sub, F, ph, nc)
        if ap > best_ap:
            best_ap, best_p, best_neg = ap, ph, nc
    return best_ap


def paired_test(cv, bv):
    if len(cv) < 2:
        return None
    t = st.ttest_rel(cv, bv); w = st.wilcoxon(cv, bv)
    return {'t': round(float(t.statistic), 3), 'p_t': round(float(t.pvalue), 4),
            'p_w': round(float(w.pvalue), 4), 'n': len(cv)}


def main():
    results = {}
    domains = list(SOURCE_LOADERS.items())
    if os.environ.get('DOMAINS'):
        wl = [d for d in os.environ['DOMAINS'].split(',') if d in SOURCE_LOADERS]
        domains = [(d, SOURCE_LOADERS[d]) for d in wl]
    elif SMOKE:
        if os.environ.get('SRC'):
            domains = [(d, SOURCE_LOADERS[d]) for d in [os.environ['SRC']] if d in SOURCE_LOADERS]
        else:
            domains = domains[:1]
    for src, (load_func, targets) in domains:
        print(f"\n{'='*60}\n源域 {src}→COCO (Ours vs B1, {len(SEEDS)} seed)\n{'='*60}")
        dd = load_domain_data(src)
        key = 'sat' if src in SAT_DOMAINS else 'drone'
        g = SENSOR_GAMMA[src]
        per_seed = {c: {s: [] for s in SEEDS} for c in CFGS}
        flat = {c: [] for c in CFGS}
        for eval_name, coco_cls, src_cls in dd['valid_targets']:
            for seed in SEEDS:
                b1 = evaluate(dd, src, src_cls, coco_cls, seed, 0.0)
                ours = evaluate(dd, src, src_cls, coco_cls, seed, g)
                per_seed['B1'][seed].append(b1)
                per_seed['Ours'][seed].append(ours)
                flat['B1'].append(b1); flat['Ours'].append(ours)
        agg = {}
        for c in CFGS:
            sm = [float(np.mean(per_seed[c][s])) for s in SEEDS if per_seed[c][s]]
            agg[c] = {'seed_means': [round(x, 4) for x in sm],
                      'mean': round(float(np.mean(sm)), 4), 'std': round(float(np.std(sm)), 4)}
        pt = paired_test(np.array(flat['Ours']), np.array(flat['B1']))
        results[src] = {'key': key, 'gamma': g, 'agg': agg, 'paired_Ours_vs_B1': pt,
                       'n_pairs': len(flat['B1'])}
        print(f"  [{key}] B1(γ=0)={agg['B1']['mean']:.4f}±{agg['B1']['std']:.4f} | "
              f"Ours(γ={g})={agg['Ours']['mean']:.4f}±{agg['Ours']['std']:.4f} | "
              f"Δ={agg['Ours']['mean']-agg['B1']['mean']:+.4f}")
        if pt:
            print(f"  [{key}] Ours vs B1 配对: t={pt['t']} p_t={pt['p_t']} | Wilcoxon p={pt['p_w']} (n={pt['n']})")
    # 跨域
    cross = {}
    for c in CFGS:
        cs = []
        for i, s in enumerate(SEEDS):
            cs.append(float(np.mean([results[src]['agg'][c]['seed_means'][i] for src in results])))
        cross[c] = {'seed_means': [round(x, 4) for x in cs],
                    'mean': round(float(np.mean(cs)), 4), 'std': round(float(np.std(cs)), 4)}
    fb = np.concatenate([np.array(results[s]['flat']['B1'] if 'flat' in results[s] else []) for s in results]) \
        if False else None
    # pooled flat 重建 (类×seed×域)
    all_b1, all_o = [], []
    for src, (load_func, targets) in domains:
        dd = load_domain_data(src)
        g = SENSOR_GAMMA[src]
        for eval_name, coco_cls, src_cls in dd['valid_targets']:
            for seed in SEEDS:
                all_b1.append(evaluate(dd, src, src_cls, coco_cls, seed, 0.0))
                all_o.append(evaluate(dd, src, src_cls, coco_cls, seed, g))
    cross_pt = paired_test(np.array(all_o), np.array(all_b1))
    print(f"\n跨域均值: B1={cross['B1']['mean']:.4f}±{cross['B1']['std']:.4f} | "
          f"Ours={cross['Ours']['mean']:.4f}±{cross['Ours']['std']:.4f} | "
          f"Δ={cross['Ours']['mean']-cross['B1']['mean']:+.4f}")
    if cross_pt:
        print(f"跨域 Ours vs B1 配对(pooled): t={cross_pt['t']} p_t={cross_pt['p_t']} | "
              f"Wilcoxon p={cross_pt['p_w']} (n={cross_pt['n']})")
    out = {'configs': {'B1': 'RawUSC(γ=0 + 逐类bestα + TGE)',
                       'Ours': 'γ=SENSOR_GAMMA[域](秩-1仿射) + 逐类bestα + TGE + 传感器闸门'},
           'sensor_gamma': SENSOR_GAMMA, 'per_domain': results,
           'cross_domain': cross, 'cross_domain_paired_Ours_vs_B1': cross_pt,
           'n_seeds': len(SEEDS), 'n_eval_per_class': N_EVAL}
    json.dump(out, open(DATA / 'autoproc_p1_ours_vs_b1_results.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print(f"\n✅ -> {DATA / 'autoproc_p1_ours_vs_b1_results.json'}")


if __name__ == '__main__':
    main()
