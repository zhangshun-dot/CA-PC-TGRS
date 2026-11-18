# -*- coding: utf-8 -*-
"""二阶对齐全量验证 - 泄漏受控版(4域22类 × 5 seed)。
修复两处受质疑点后再下结论:
  (1) 不逐类调 λ: 报告 FIXED λ (0.5, 1.0) 与 best-λ(仅作上界参考),
      fixed λ 对所有类/域一致, 不在测试 AP 上挑参。
  (2) 目标统计留出: mu_t/sd_t 只用 COCO 一半图(crop)算, AP 在另一半上测,
      避免"用测试分布塑造原型又在同批图上评估"的闭环泄露。
  (3) delta_std 修正为 每对 (diag_seed - base_seed) 的真实标准差(非负)。
对齐: Z_corr=(1-λ)Z_F + λ*[(Z_F-μ_t)/σ_t*σ_s+μ_s]; λ=0=恒等=base。
相似度在 512 维还原后算(与方法真实空间一致)。
"""
import sys, json
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
from pathlib import Path
import usc_experiment_v12 as V12
import unified_eval as U
from usc_rank1_affine_gpu_v2 import load_domain_data, SOURCE_LOADERS

DATA = V12.DATA
EVAL_DOMAINS = list(SOURCE_LOADERS.keys())
N_SUB = 50
SEEDS = [0, 1, 2, 3, 4]
LAMBDA = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]      # 含 0=base
FIXED_LAMS = [0.5, 1.0]
SPLIT_SEED = 999                                 # 留出切分固定


def project(X, usc):
    return (X - usc.mean_) @ usc.components_.T


def recon(Z, usc):
    return Z @ usc.components_ + usc.mean_


def ap_from_X(per_img_sub, Xfull):
    preds, gts = [], []
    for p in per_img_sub:
        n = len(p['gd'])
        if n:
            Xb = Xfull[p['crop_idx']]
            Xb = Xb / (np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-8)
            pos = Xb @ p_hat_g
            neg_s = Xb @ neg_raw_g if neg_raw_g is not None else np.zeros(n)
            vis = np.maximum(0.0, pos - V12.LAM * neg_s)
            fused = np.sqrt(np.clip(p['gd'], 1e-6, 1.0) * np.clip(vis, 1e-6, 1.0))
        else:
            fused = np.zeros(0)
        preds.append((p['box'], fused)); gts.append(p['gt'])
    return float(U.compute_ap50_img(preds, gts))


def main():
    global p_hat_g, neg_raw_g
    results = {}
    for src in EVAL_DOMAINS:
        print(f"\n{'='*60}\n源域 {src} → COCO (泄漏受控)\n{'='*60}")
        dd = load_domain_data(src)
        usc = dd['usc']
        cls_rec = []
        for eval_name, coco_cls, src_cls in dd['valid_targets']:
            td = dd['target_data'][coco_cls]
            F = td['features']                              # (N_full,512)
            p_hat_g = (usc.correct_prototype_gpu(dd['protos_gpu'][src_cls], 0.0, 0.0).cpu().numpy()
                       + dd['text_deltas'][src_cls]); p_hat_g = p_hat_g / (np.linalg.norm(p_hat_g) + 1e-8)
            neg_raw_g = V12.build_neg_proto(dd['all_protos'], src_cls)
            Z_F = project(F, usc)                           # (N_full,30)

            # 留出切分: 一半图算目标统计, 另一半测 AP
            n_imgs = len(td['per_img'])
            rng0 = np.random.default_rng(SPLIT_SEED)
            perm = rng0.permutation(n_imgs)
            half = n_imgs // 2
            stats_idx = perm[:half].tolist(); eval_idx = perm[half:].tolist()

            stats_rows = np.concatenate([np.asarray(td['per_img'][i]['crop_idx']).reshape(-1) for i in stats_idx])
            mu_t = Z_F[stats_rows].mean(0); sd_t = Z_F[stats_rows].std(0) + 1e-6
            src_feats = dd['source_data'][src_cls]['features']
            mu_s = project(src_feats, usc).mean(0); sd_s = project(src_feats, usc).std(0) + 1e-6
            eval_perimg = [td['per_img'][i] for i in eval_idx]

            def align(lam):
                full = (Z_F - mu_t) / sd_t * sd_s + mu_s
                return (1 - lam) * Z_F + lam * full
            X_corrs = {lam: recon(align(lam), usc) for lam in LAMBDA}

            # eval-half 全量
            base_eh = ap_from_X(eval_perimg, F)
            best_eh, best_lam = base_eh, 0.0
            for lam in LAMBDA:
                a = ap_from_X(eval_perimg, X_corrs[lam])
                if a > best_eh:
                    best_eh, best_lam = a, lam
            fixed_eh = {lam: ap_from_X(eval_perimg, X_corrs[lam]) for lam in FIXED_LAMS}

            # seed 稳定性(在 eval-half 内再抽 N_SUB)
            k = min(N_SUB, len(eval_idx))
            bs, bests, fixeds = [], [], {lam: [] for lam in FIXED_LAMS}
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                sub = [eval_perimg[i] for i in rng.choice(len(eval_perimg), size=k, replace=False)]
                b = ap_from_X(sub, F)
                d, dl = b, 0.0
                for lam in LAMBDA:
                    a = ap_from_X(sub, X_corrs[lam])
                    if a > d:
                        d, dl = a, lam
                bs.append(b); bests.append(d)
                for lam in FIXED_LAMS:
                    fixeds[lam].append(ap_from_X(sub, X_corrs[lam]))
            bs = np.array(bs); bests = np.array(bests)
            best_delta = bests - bs
            fixed_block = {}
            for lam in FIXED_LAMS:
                fs = np.array(fixeds[lam]); fd = fs - bs
                fixed_block[lam] = {
                    'full': round(fixed_eh[lam], 4),
                    'seed_mean': round(fs.mean(), 4), 'seed_std': round(fs.std(), 4),
                    'delta_mean': round(fd.mean(), 4), 'delta_std': round(fd.std(), 4),
                }
            cls_rec.append({
                'class': coco_cls, 'src': src_cls,
                'base_evalhalf_full': round(base_eh, 4),
                'best_evalhalf_full': round(best_eh, 4), 'best_lam': best_lam,
                'base_seed_mean': round(bs.mean(), 4), 'base_seed_std': round(bs.std(), 4),
                'best_seed_mean': round(bests.mean(), 4), 'best_seed_std': round(bests.std(), 4),
                'best_delta_mean': round(best_delta.mean(), 4), 'best_delta_std': round(best_delta.std(), 4),
                'fixed': fixed_block,
            })
            print(f"  {coco_cls:<11s}({src_cls:<9s}) baseEH={base_eh:.4f} bestEH={best_eh:.4f}(λ={best_lam}) "
                  f"| fixed1.0 EH={fixed_eh[1.0]:.4f} Δseed={fixed_block[1.0]['delta_mean']:+.4f}±{fixed_block[1.0]['delta_std']:.4f}")

        b_m = np.mean([r['base_seed_mean'] for r in cls_rec])
        d_m = np.mean([r['fixed'][1.0]['seed_mean'] for r in cls_rec])
        results[src] = {'base_mean_allcls': round(b_m, 4),
                        'fixed1.0_mean_allcls': round(d_m, 4),
                        'diag_mean_allcls': round(np.mean([r['best_seed_mean'] for r in cls_rec]), 4),
                        'per_class': cls_rec}
        print(f"  → {src} 域均值(留出+固定λ=1.0): base={b_m:.4f} fixed1.0={d_m:.4f} Δ={d_m-b_m:+.4f}")

    json.dump(results, open(DATA / 'autoproc_featalign_controlled_results.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print(f"\n✅ -> {DATA / 'autoproc_featalign_controlled_results.json'}")


if __name__ == '__main__':
    main()
