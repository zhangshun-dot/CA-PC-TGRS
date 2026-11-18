# -*- coding: utf-8 -*-
"""二阶对齐全量验证(4域22类 × 多seed, 修正 λ=0 退化假象):
修正: 旧公式 λ=0 把特征压成源均值常量→相似度退化→AP退回 raw-gd(假象)。
新公式用插值: Z_corr=(1-λ)Z_F + λ*[(Z_F-μ_t)/σ_t*σ_s+μ_s]; λ=0=恒等(=base), λ=1=全对齐。
对齐仅在 PCA(K=30) 算, 相似度在 512 维还原后算(与方法真实空间一致)。
多 seed: 每类随机子采样 N_SUB 张图, 跑 SEEDS 次, 报均值±std(看增益是否稳健)。
"""
import sys, json
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
from pathlib import Path
import torch
import usc_experiment_v12 as V12
import unified_eval as U
from usc_rank1_affine_gpu_v2 import load_domain_data, SOURCE_LOADERS
from sklearn.linear_model import LogisticRegression  # noqa (unused, 保留接口)

DATA = V12.DATA
EVAL_DOMAINS = list(SOURCE_LOADERS.keys())   # 4 域
N_SUB = 50
SEEDS = [0, 1, 2, 3, 4]
LAMBDA = [0.25, 0.5, 0.75, 1.0, 1.25]        # 仅 λ>0 (λ=0=base, 单独记)


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
        print(f"\n{'='*60}\n源域 {src} → COCO\n{'='*60}")
        dd = load_domain_data(src)              # 全量 cache, 不截断
        usc = dd['usc']; protos_gpu = dd['protos_gpu']; text_deltas = dd['text_deltas']
        cls_rec = []
        for eval_name, coco_cls, src_cls in dd['valid_targets']:
            td = dd['target_data'][coco_cls]
            F = td['features']                  # (N_full,512) 真实 CLIP
            p_hat_g = (usc.correct_prototype_gpu(protos_gpu[src_cls], 0.0, 0.0).cpu().numpy()
                       + text_deltas[src_cls]); p_hat_g = p_hat_g / (np.linalg.norm(p_hat_g) + 1e-8)
            neg_raw_g = V12.build_neg_proto(dd['all_protos'], src_cls)
            Z_F = project(F, usc)
            src_feats = dd['source_data'][src_cls]['features']
            mu_s = project(src_feats, usc).mean(0); sd_s = project(src_feats, usc).std(0) + 1e-6
            mu_t = Z_F.mean(0); sd_t = Z_F.std(0) + 1e-6

            def align(lam):
                full = (Z_F - mu_t) / sd_t * sd_s + mu_s
                return (1 - lam) * Z_F + lam * full

            X_corrs = {lam: recon(align(lam), usc) for lam in LAMBDA}

            # 全量(不子采样)
            base_full = ap_from_X(td['per_img'], F)
            d_full, dl_full = base_full, 0.0
            for lam in LAMBDA:
                a = ap_from_X(td['per_img'], X_corrs[lam])
                if a > d_full:
                    d_full, dl_full = a, lam

            # 多 seed 子采样
            n_imgs = len(td['per_img']); k = min(N_SUB, n_imgs)
            bs, ds, dls = [], [], []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                idx = rng.choice(n_imgs, size=k, replace=False)
                sub = [td['per_img'][i] for i in idx]
                b = ap_from_X(sub, F)
                d, dl = b, 0.0
                for lam in LAMBDA:
                    a = ap_from_X(sub, X_corrs[lam])
                    if a > d:
                        d, dl = a, lam
                bs.append(b); ds.append(d); dls.append(dl)
            bs = np.array(bs); ds = np.array(ds)
            cls_rec.append({
                'class': coco_cls, 'src': src_cls,
                'base_full': round(base_full, 4), 'diag_full': round(d_full, 4), 'diag_lam_full': dl_full,
                'base_mean': round(bs.mean(), 4), 'base_std': round(bs.std(), 4),
                'diag_mean': round(ds.mean(), 4), 'diag_std': round(ds.std(), 4),
                'delta_mean': round(ds.mean() - bs.mean(), 4), 'delta_std': round(ds.std() - bs.std(), 4),
                'lam_mode': int(np.bincount(dls).argmax()),
            })
            print(f"  {coco_cls:<11s}({src_cls:<9s}) base={base_full:.4f} diagFull={d_full:.4f}(λ={dl_full}) "
                  f"| seed base={bs.mean():.4f}±{bs.std():.4f} diag={ds.mean():.4f}±{ds.std():.4f} Δ={ds.mean()-bs.mean():+.4f}")

        b_m = np.mean([r['base_mean'] for r in cls_rec]); d_m = np.mean([r['diag_mean'] for r in cls_rec])
        results[src] = {'base_mean_allcls': round(b_m, 4), 'diag_mean_allcls': round(d_m, 4),
                        'per_class': cls_rec}
        print(f"  → {src} 域均值: base={b_m:.4f} diag={d_m:.4f} Δ={d_m-b_m:+.4f}")

    json.dump(results, open(DATA / 'autoproc_featalign_full_results.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print(f"\n✅ -> {DATA / 'autoproc_featalign_full_results.json'}")


if __name__ == '__main__':
    main()
