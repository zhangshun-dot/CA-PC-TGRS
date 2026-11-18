# -*- coding: utf-8 -*-
"""P1.1 强 baseline 矩阵 (4域 x 5 seed, 复用 load_domain_data 缓存, 不重检测).
新增 baseline (与 P0 的 C 上界版同协议 N_EVAL=150, 5 seed):
  B0 = RawCLIP: 源原型直接匹配, 无 USC / 无 TGE (最朴素跨域原型)
  B1 = RawUSC: 原始管线 (γ=0 最优 α + TGE, 即原论文方法), 所有域统一套用
  B2 = FeatCalib: 一阶均值校准 (原型按 μ_t-μ_s 平移) + TGE, 对比我们的二阶 whitening-coloring
上界版 Ours(C) 沿用 autoproc_three_stage_full_seeds_results.json, 不在本脚本重算。
产物: autoproc_p1_baselines_results.json
聚合: 每 seed 先对类求均值 -> 5 值再 mean±std; 跨域 = 每 seed 跨域均值再 mean±std。
"""
import sys, json, os
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
from pathlib import Path
import usc_experiment_v12 as V12
import unified_eval as U
from usc_rank1_affine_gpu_v2 import load_domain_data, SOURCE_LOADERS

DATA = V12.DATA
SAT_DOMAINS = {'DIOR', 'NWPU', 'DOTA'}
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
N_EVAL = int(os.environ.get('N_EVAL', '150'))
if os.environ.get('NSEED'):
    SEEDS = list(range(int(os.environ['NSEED'])))
elif os.environ.get('SMOKE') == '1':
    SEEDS = [0]
else:
    SEEDS = [0, 1, 2, 3, 4]
SMOKE = os.environ.get('SMOKE') == '1'
CFGS = ['B0', 'B1', 'B2', 'B3', 'B4']


def eval_proto(per_img_sub, F, p_hat, neg_raw):
    return float(V12.evaluate({'per_img': per_img_sub, 'features': F}, p_hat, neg_raw))


def evaluate_baselines(dd, src, src_cls, coco_cls, seed):
    td = dd['target_data'][coco_cls]
    F = td['features']; per_img = td['per_img']
    usc = dd['usc']; neg_dev = usc._mean_gpu.device
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(per_img), size=min(N_EVAL, len(per_img)), replace=False)
    sub = [per_img[i] for i in idx]
    proto = dd['all_protos'][src_cls]
    delta = dd['text_deltas'][src_cls]
    neg_raw_g = V12.build_neg_proto(dd['all_protos'], src_cls)
    # B0 RawCLIP: 源原型直接, 无 USC / 无 TGE
    p0 = proto / (np.linalg.norm(proto) + 1e-8)
    neg0 = neg_raw_g
    B0 = eval_proto(sub, F, p0, neg0)
    # B1 RawUSC: 原始管线 γ=0 最优 α + TGE
    best_ap, best_p, best_neg = -1, None, None
    neg_gpu = torch.tensor(neg_raw_g, dtype=torch.float32, device=neg_dev) if neg_raw_g is not None else None
    for a in ALPHAS:
        pc = usc.correct_prototype_gpu(dd['protos_gpu'][src_cls], 0.0, a).cpu().numpy()
        nc = usc.correct_prototype_gpu(neg_gpu, 0.0, a).cpu().numpy() if neg_gpu is not None else None
        ph = pc + delta; ph = ph / (np.linalg.norm(ph) + 1e-8)
        ap = eval_proto(sub, F, ph, nc)
        if ap > best_ap:
            best_ap = ap; best_p = ph; best_neg = nc
    B1 = best_ap
    # B2 FeatCalib: 一阶均值校准 (原型按 μ_t-μ_s 平移) + TGE
    mu_t = F.mean(0); mu_s = dd['source_data'][src_cls]['features'].mean(0)
    p_cal = proto + (mu_t - mu_s)
    p_cal = (p_cal + delta); p_cal = p_cal / (np.linalg.norm(p_cal) + 1e-8)
    if neg_raw_g is not None:
        n_cal = neg_raw_g + (mu_t - mu_s); n_cal = n_cal / (np.linalg.norm(n_cal) + 1e-8)
    else:
        n_cal = None
    B2 = eval_proto(sub, F, p_cal, n_cal)
    # B3 伪-TTA 方案A(标准化): Z'=(Z-μ_batch)/(σ_batch+ε) 直接作用于目标 CLIP 特征
    mu_b = F.mean(0); sd_b = F.std(0) + 1e-8
    F_tta_A = (F - mu_b) / sd_b
    B3 = eval_proto(sub, F_tta_A, p0, neg0)
    # B4 伪-TTA 方案B(match_source): 标准化后匹配源统计 μ_src/σ_src
    src_feats = dd['source_data'][src_cls]['features']
    mu_s = src_feats.mean(0); sd_s = src_feats.std(0) + 1e-8
    F_tta_B = F_tta_A * sd_s + mu_s
    B4 = eval_proto(sub, F_tta_B, p0, neg0)
    return {'B0': B0, 'B1': B1, 'B2': B2, 'B3': B3, 'B4': B4}


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
        print(f"\n{'='*64}\n源域 {src}→COCO (P1 baselines, 5-seed)\n{'='*64}")
        dd = load_domain_data(src)
        key = 'sat' if src in SAT_DOMAINS else 'drone'
        per_seed = {cfg: {s: [] for s in SEEDS} for cfg in CFGS}
        for eval_name, coco_cls, src_cls in dd['valid_targets']:
            for seed in SEEDS:
                r = evaluate_baselines(dd, src, src_cls, coco_cls, seed)
                for cfg in CFGS:
                    per_seed[cfg][seed].append(r[cfg])
        agg = {}
        for cfg in CFGS:
            seed_means = [float(np.mean(per_seed[cfg][s])) for s in SEEDS if per_seed[cfg][s]]
            agg[cfg] = {'seed_means': [round(x, 4) for x in seed_means],
                        'mean': round(float(np.mean(seed_means)), 4),
                        'std': round(float(np.std(seed_means)), 4)}
        results[src] = {'key': key, 'agg': agg,
                        'n_pairs': len(dd['valid_targets']) * len(SEEDS)}
        a = agg
        print(f"  [{key}] B0(RawCLIP)={a['B0']['mean']:.4f}±{a['B0']['std']:.4f} | "
              f"B1(RawUSC)={a['B1']['mean']:.4f}±{a['B1']['std']:.4f} | "
              f"B2(FeatCalib)={a['B2']['mean']:.4f}±{a['B2']['std']:.4f} | "
              f"B3(pTTA-A)={a['B3']['mean']:.4f}±{a['B3']['std']:.4f} | "
              f"B4(pTTA-B)={a['B4']['mean']:.4f}±{a['B4']['std']:.4f}")
    # 跨域
    cross = {}
    for cfg in CFGS:
        cs = []
        for i, s in enumerate(SEEDS):
            cs.append(float(np.mean([results[src]['agg'][cfg]['seed_means'][i] for src in results])))
        cross[cfg] = {'seed_means': [round(x, 4) for x in cs],
                      'mean': round(float(np.mean(cs)), 4),
                      'std': round(float(np.std(cs)), 4)}
    print(f"\n跨域: B0={cross['B0']['mean']:.4f}±{cross['B0']['std']:.4f} | "
          f"B1={cross['B1']['mean']:.4f}±{cross['B1']['std']:.4f} | "
          f"B2={cross['B2']['mean']:.4f}±{cross['B2']['std']:.4f} | "
          f"B3(pTTA-A)={cross['B3']['mean']:.4f}±{cross['B3']['std']:.4f} | "
          f"B4(pTTA-B)={cross['B4']['mean']:.4f}±{cross['B4']['std']:.4f}")
    out = {'configs': {'B0': 'RawCLIP(no USC/no TGE)', 'B1': 'RawUSC(γ=0 best α + TGE)',
                       'B2': 'FeatCalib(1st-order μ_t-μ_s + TGE)'},
           'per_domain': results, 'cross_domain': cross,
           'n_seeds': len(SEEDS), 'n_eval_per_class': N_EVAL}
    json.dump(out, open(DATA / 'autoproc_p1_baselines_results.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print(f"\n✅ -> {DATA / 'autoproc_p1_baselines_results.json'}")


if __name__ == '__main__':
    main()
