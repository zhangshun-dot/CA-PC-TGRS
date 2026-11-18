# -*- coding: utf-8 -*-
"""特征空间二阶对齐探针(零重检测, 复用 GDINO 缓存):
问: 识别除框外, 特征对齐环节能否破 +1.25% 天花板?
当前方法只用秩1仿射(v 一维)。本探针在 PCA(K=30) 空间对目标框特征做
各主成分独立的二阶对齐(均值+标准差匹配, whitening-coloring 式, 零训练),
扫描对齐强度 λ, 重算 AP50, 对比 (a)无校正 (b)当前秩1方法(e2e-A) (c)二阶对齐最优。
"""
import sys, json
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
from pathlib import Path
import torch
import usc_experiment_v12 as V12
import unified_eval as U
from usc_rank1_affine_gpu_v2 import load_domain_data

DATA = V12.DATA
EVAL_DOMAINS = ['DIOR', 'VisDrone']
N_TARGET = 50
LAMBDA = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]  # 二阶对齐强度(0=仅平移到源均值, 1=全对齐)


def project(X, usc):
    return (X - usc.mean_) @ usc.components_.T


def main():
    results = {}
    for src in EVAL_DOMAINS:
        print(f"\n{'='*60}\n源域 {src} → COCO (N={N_TARGET})\n{'='*60}")
        dd = load_domain_data(src)
        for coco_cls in list(dd['target_data'].keys()):
            td = dd['target_data'][coco_cls]
            td['per_img'] = td['per_img'][:N_TARGET]
        usc = dd['usc']
        protos_gpu = dd['protos_gpu']
        all_protos = dd['all_protos']
        text_deltas = dd['text_deltas']

        cls_rec = []
        for eval_name, coco_cls, src_cls in dd['valid_targets']:
            td = dd['target_data'][coco_cls]
            F = td['features']                       # (N,512) 目标框特征
            per_img = td['per_img']
            gd = np.concatenate([p['gd'] for p in per_img]) if any(len(p['gd']) for p in per_img) else np.zeros(0)
            # 原型 + 文本外推(不做秩1)
            p_corr = usc.correct_prototype_gpu(protos_gpu[src_cls], 0.0, 0.0).cpu().numpy()  # γ=0,α=0 → 纯源原型
            delta = text_deltas[src_cls]
            p_hat = p_corr + delta
            p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)
            neg_raw = V12.build_neg_proto(all_protos, src_cls)

            # 二阶对齐在 PCA 空间算, 但相似度在 512 维还原后算(与方法的真实空间一致, 公平对比天花板)
            Z_F = project(F, usc)                    # (N_full,K)
            src_feats = dd['source_data'][src_cls]['features']
            Zs = project(src_feats, usc); Zt = Z_F
            mu_s, sd_s = Zs.mean(0), Zs.std(0) + 1e-6
            mu_t, sd_t = Zt.mean(0), Zt.std(0) + 1e-6

            def recon(Xz):
                return Xz @ usc.components_ + usc.mean_

            def ap_from_X(Xfull):
                preds, gts = [], []
                for p in per_img:
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

            base = ap_from_X(F)                     # 真实 512 维相似度(当前方法重排基线)
            best, best_l = base, 0.0
            for lam in LAMBDA:
                Z_corr = (Z_F - mu_t) / sd_t * (lam * sd_s) + mu_s
                ap = ap_from_X(recon(Z_corr))
                if ap > best:
                    best, best_l = ap, lam
            cls_rec.append({'class': coco_cls, 'base': round(base, 4),
                            'diag_best': round(best, 4), 'best_lambda': best_l,
                            'delta': round(best - base, 4)})
            print(f"  {coco_cls:<11s} base={base:.4f} diagBest={best:.4f}(λ={best_l}, Δ{best-base:+.4f})")

        results[src] = {
            'base_mean': float(np.mean([r['base'] for r in cls_rec])),
            'diag_mean': float(np.mean([r['diag_best'] for r in cls_rec])),
            'per_class': cls_rec,
        }
        r = results[src]
        print(f"  → 域均值: base={r['base_mean']:.4f} diag={r['diag_mean']:.4f} (Δ{r['diag_mean']-r['base_mean']:+.4f})")

    with open(DATA / 'autoproc_featalign_probe_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ -> {DATA / 'autoproc_featalign_probe_results.json'}")


if __name__ == '__main__':
    main()
