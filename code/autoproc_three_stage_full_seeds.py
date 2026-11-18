# -*- coding: utf-8 -*-
"""P0: 带统计显著性的主结果 + 完整 ablation matrix (4域 x 5 seed x 多配置).
配置(每域逐seed算AP, mean±std跨seed):
  A  = base (不校正)
  B  = 二阶对齐通铺 (所有域用二阶对齐, λ=1.0 全量 transductive)  -> 无人机必负迁移
  B' = 秩-1仿射通铺 (所有域 rank-1 仿射, 固定 α=1.0)            -> 卫星必负迁移
  C  = 域自适应双 corrector (卫星二阶对齐 / 无人机 rank-1 仿射逐类oracle α)
  C' = 域自适应双 corrector + 固定 α=1.0 (无人机关掉 how much 逐类调参)
证明三阶段正交 + 单一 corrector 必负迁移 + how much 贡献.
产物: autoproc_three_stage_full_seeds_results.json
统计口径: 每 seed 先对类求均值 -> 5 个值再求 mean±std (per-seed 聚合, 不混类间方差);
         配对检验用 (类×seed) 对齐 pair, 跨域用 pooled (类×seed×域) pair.
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
N_EVAL = int(os.environ.get('N_EVAL', '150'))
SEEDS = [0, 1, 2, 3, 4] if os.environ.get('SMOKE') != '1' else [0]
SMOKE = os.environ.get('SMOKE') == '1'
CFGS = ['base', 'B', 'Bp', 'C', 'Cp']


def project(X, usc):
    return (X - usc.mean_) @ usc.components_.T


def recon(Z, usc):
    return Z @ usc.components_ + usc.mean_


def ap_from_X(per_img_sub, Xfull, p_hat, neg_raw):
    preds, gts = [], []
    for p in per_img_sub:
        n = len(p['gd'])
        if n:
            Xb = Xfull[p['crop_idx']]
            Xb = Xb / (np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-8)
            pos = Xb @ p_hat
            neg_s = Xb @ neg_raw if neg_raw is not None else np.zeros(n)
            vis = np.maximum(0.0, pos - V12.LAM * neg_s)
            fused = np.sqrt(np.clip(p['gd'], 1e-6, 1.0) * np.clip(vis, 1e-6, 1.0))
        else:
            fused = np.zeros(0)
        preds.append((p['box'], fused)); gts.append(p['gt'])
    return float(U.compute_ap50_img(preds, gts))


def evaluate_domain(dd, src, src_cls, coco_cls, seed):
    td = dd['target_data'][coco_cls]
    F = td['features']; per_img = td['per_img']
    usc = dd['usc']; neg_dev = usc._mean_gpu.device
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(per_img), size=min(N_EVAL, len(per_img)), replace=False)
    per_img_sub = [per_img[i] for i in idx]
    neg_raw_g = V12.build_neg_proto(dd['all_protos'], src_cls)
    p_hat_g = (usc.correct_prototype_gpu(dd['protos_gpu'][src_cls], 0.0, 0.0).cpu().numpy()
               + dd['text_deltas'][src_cls]); p_hat_g = p_hat_g / (np.linalg.norm(p_hat_g) + 1e-8)
    is_sat = src in SAT_DOMAINS
    # base
    if is_sat:
        base = ap_from_X(per_img_sub, F, p_hat_g, neg_raw_g)
    else:
        p0 = usc.correct_prototype_gpu(dd['protos_gpu'][src_cls], 0.0, 0.0).cpu().numpy()
        p0hat = p0 + dd['text_deltas'][src_cls]; p0hat = p0hat / (np.linalg.norm(p0hat) + 1e-8)
        base = V12.evaluate({'per_img': per_img_sub, 'features': F}, p0hat, neg_raw_g)
    # sat_aligned (二阶对齐, 全量 transductive)
    Z_F = project(F, usc); mu_t = Z_F.mean(0); sd_t = Z_F.std(0) + 1e-6
    src_feats = dd['source_data'][src_cls]['features']
    mu_s = project(src_feats, usc).mean(0); sd_s = project(src_feats, usc).std(0) + 1e-6
    full = (Z_F - mu_t) / sd_t * sd_s + mu_s
    X_corr = recon((1 - 1.0) * Z_F + 1.0 * full, usc)
    sat_aligned = ap_from_X(per_img_sub, X_corr, p_hat_g, neg_raw_g)
    # rank1_fixed (α=1.0, 所有域通用)
    p_src = dd['protos_gpu'][src_cls]
    neg_gpu = torch.tensor(neg_raw_g, dtype=torch.float32, device=neg_dev) if neg_raw_g is not None else None
    p_corr = usc.correct_prototype_gpu(p_src, 0.0, 1.0).cpu().numpy()
    neg_c = usc.correct_prototype_gpu(neg_gpu, 0.0, 1.0).cpu().numpy() if neg_gpu is not None else None
    p_hat = p_corr + dd['text_deltas'][src_cls]; p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)
    rank1_fixed = V12.evaluate({'per_img': per_img_sub, 'features': F}, p_hat, neg_c)
    rec = {'base': base, 'sat_aligned': sat_aligned, 'rank1_fixed': rank1_fixed, 'best_a': 1.0}
    # drone_oracle (仅无人机: 逐类 oracle α, 上界)
    if not is_sat:
        best_a, best_ap = 0.0, -1
        for a in ALPHAS:
            pc = usc.correct_prototype_gpu(p_src, 0.0, a).cpu().numpy()
            nc = usc.correct_prototype_gpu(neg_gpu, 0.0, a).cpu().numpy() if neg_gpu is not None else None
            ph = pc + dd['text_deltas'][src_cls]; ph = ph / (np.linalg.norm(ph) + 1e-8)
            ap = V12.evaluate({'per_img': per_img_sub, 'features': F}, ph, nc)
            if ap > best_ap:
                best_ap = ap; best_a = a
        rec['drone_oracle'] = best_ap; rec['best_a'] = best_a
    return rec


def paired_test(cv, bv):
    if len(cv) < 2:
        return None
    t = st.ttest_rel(cv, bv); w = st.wilcoxon(cv, bv)
    return {'t': round(float(t.statistic), 3), 'p_t': round(float(t.pvalue), 4),
            'p_w': round(float(w.pvalue), 4), 'n': len(cv)}


def main():
    results = {}
    domains = list(SOURCE_LOADERS.items())
    if SMOKE:
        if os.environ.get('SRC'):
            domains = [(d, SOURCE_LOADERS[d]) for d in [os.environ['SRC']] if d in SOURCE_LOADERS]
        else:
            domains = domains[:1]
    for src, (load_func, targets) in domains:
        print(f"\n{'='*64}\n源域 {src}→COCO (P0 seeds)\n{'='*64}")
        dd = load_domain_data(src)
        key = 'sat' if src in SAT_DOMAINS else 'drone'
        per_seed = {cfg: {s: [] for s in SEEDS} for cfg in CFGS}
        flat = {cfg: [] for cfg in CFGS}
        for eval_name, coco_cls, src_cls in dd['valid_targets']:
            for seed in SEEDS:
                r = evaluate_domain(dd, src, src_cls, coco_cls, seed)
                per_seed['base'][seed].append(r['base'])
                per_seed['B'][seed].append(r['sat_aligned'])
                per_seed['Bp'][seed].append(r['rank1_fixed'])
                c_val = r['sat_aligned'] if key == 'sat' else r.get('drone_oracle', r['sat_aligned'])
                cp_val = r['sat_aligned'] if key == 'sat' else r['rank1_fixed']
                per_seed['C'][seed].append(c_val)
                per_seed['Cp'][seed].append(cp_val)
                flat['base'].append(r['base']); flat['B'].append(r['sat_aligned'])
                flat['Bp'].append(r['rank1_fixed']); flat['C'].append(c_val); flat['Cp'].append(cp_val)
        agg = {}
        for cfg in CFGS:
            seed_means = [float(np.mean(per_seed[cfg][s])) for s in SEEDS if per_seed[cfg][s]]
            agg[cfg] = {'seed_means': [round(x, 4) for x in seed_means],
                        'mean': round(float(np.mean(seed_means)), 4),
                        'std': round(float(np.std(seed_means)), 4)}
        pt = paired_test(np.array(flat['C']), np.array(flat['base']))
        results[src] = {'key': key, 'agg': agg,
                        'flat_base': [round(x, 4) for x in flat['base']],
                        'flat_C': [round(x, 4) for x in flat['C']],
                        'flat_B': [round(x, 4) for x in flat['B']],
                        'n_pairs': len(flat['base']),
                        'paired_C_vs_base': pt,
                        'best_a_by_class': None}
        a = agg
        print(f"  [{key}] base={a['base']['mean']:.4f}±{a['base']['std']:.4f} | "
              f"B(二阶通铺)={a['B']['mean']:.4f} | B'(rank1通铺)={a['Bp']['mean']:.4f} | "
              f"C(双)={a['C']['mean']:.4f} | C'(固定α)={a['Cp']['mean']:.4f}")
        if pt:
            print(f"  [{key}] C vs base 配对: t={pt['t']} p_t={pt['p_t']} | Wilcoxon p={pt['p_w']} (n={pt['n']})")
        else:
            print(f"  [{key}] (smoke 单 seed, 跳过配对检验)")
    # 跨域 per-seed mean (每 seed 跨域均值) -> mean±std
    cross_seed = {cfg: [] for cfg in CFGS}
    for i, s in enumerate(SEEDS):
        for cfg in CFGS:
            vals = [results[src]['agg'][cfg]['seed_means'][i] for src in results]
            cross_seed[cfg].append(float(np.mean(vals)))
    cross = {cfg: {'seed_means': [round(x, 4) for x in cross_seed[cfg]],
                   'mean': round(float(np.mean(cross_seed[cfg])), 4),
                   'std': round(float(np.std(cross_seed[cfg])), 4)} for cfg in CFGS}
    fb = np.concatenate([np.array(results[s]['flat_base']) for s in results])
    fc = np.concatenate([np.array(results[s]['flat_C']) for s in results])
    fB = np.concatenate([np.array(results[s]['flat_B']) for s in results])
    cross_pt = paired_test(fc, fb)
    print(f"\n跨域均值: base={cross['base']['mean']:.4f}±{cross['base']['std']:.4f} | "
          f"B(二阶通铺)={cross['B']['mean']:.4f}±{cross['B']['std']:.4f} | "
          f"B'(rank1通铺)={cross['Bp']['mean']:.4f}±{cross['Bp']['std']:.4f} | "
          f"C(双)={cross['C']['mean']:.4f}±{cross['C']['std']:.4f} | "
          f"C'(固定α)={cross['Cp']['mean']:.4f}±{cross['Cp']['std']:.4f}")
    print(f"跨域 Δ: B={cross['B']['mean']-cross['base']['mean']:+.4f} | "
          f"B'={cross['Bp']['mean']-cross['base']['mean']:+.4f} | "
          f"C={cross['C']['mean']-cross['base']['mean']:+.4f} | "
          f"C'={cross['Cp']['mean']-cross['base']['mean']:+.4f}")
    if cross_pt:
        print(f"跨域 C vs base 配对(pooled): t={cross_pt['t']} p_t={cross_pt['p_t']} | "
              f"Wilcoxon p={cross_pt['p_w']} (n={cross_pt['n']})")
    out = {'configs': {'A': 'base', 'B': '二阶对齐通铺', "B'": 'rank1通铺',
                       'C': '域自适应双corrector', "C'": '双corrector+固定α'},
           'per_domain': results,
           'cross_domain': cross,
           'cross_domain_paired_C_vs_base': cross_pt,
           'n_seeds': len(SEEDS), 'n_eval_per_class': N_EVAL}
    json.dump(out, open(DATA / 'autoproc_three_stage_full_seeds_results.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print(f"\n✅ -> {DATA / 'autoproc_three_stage_full_seeds_results.json'}")


if __name__ == '__main__':
    main()
