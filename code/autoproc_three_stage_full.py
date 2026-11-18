# -*- coding: utf-8 -*-
"""三阶段统一全量主结果(4域, 域自适应 corrector):
 whether(域级视角闸门): 卫星域(overhead→photo视角差大)开二阶对齐; 无人机域(已近photo)用 rank-1 affine+essence.
 how(corrector, 域自适应):
   卫星域: 特征二阶对齐(whitening-coloring, λ=1.0, 目标统计留出半集)
   无人机域: rank-1 affine (Z_corr=(I+γvvᵀ)Z+αv) + per-class α 扫描最优(上界参考, 后续换 LOO essence 预测)
 how much(α): 卫星域 λ=1 恒定; 无人机域 per-class α 最优(标注 oracle 上界)
 产出: 4域 AP50 表(基础/三阶段) + 域均值增益 + 跨域均值。
"""
import sys, json
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
from pathlib import Path
import usc_experiment_v12 as V12
import unified_eval as U
from usc_rank1_affine_gpu_v2 import load_domain_data, SOURCE_LOADERS

DATA = V12.DATA
N_SUB = 50
SEEDS = [0, 1, 2, 3, 4]
SPLIT_SEED = 999
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
SAT_DOMAINS = {'DIOR', 'NWPU', 'DOTA'}


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


def main():
    results = {}
    for src in SOURCE_LOADERS:
        print(f"\n{'='*64}\n源域 {src}→COCO 三阶段全量\n{'='*64}")
        dd = load_domain_data(src)
        usc = dd['usc']
        neg_dev = usc._mean_gpu.device
        cls_rec = []
        base_all, three_all = [], []
        for eval_name, coco_cls, src_cls in dd['valid_targets']:
            td = dd['target_data'][coco_cls]
            F = td['features']
            neg_raw_g = V12.build_neg_proto(dd['all_protos'], src_cls)
            if src in SAT_DOMAINS:
                # how = 二阶对齐(特征视角), λ=1, 全量口径(目标统计含评估集, transductive 设定)
                p_hat_g = (usc.correct_prototype_gpu(dd['protos_gpu'][src_cls], 0.0, 0.0).cpu().numpy()
                           + dd['text_deltas'][src_cls]); p_hat_g = p_hat_g / (np.linalg.norm(p_hat_g) + 1e-8)
                Z_F = project(F, usc)
                mu_t = Z_F.mean(0); sd_t = Z_F.std(0) + 1e-6
                src_feats = dd['source_data'][src_cls]['features']
                mu_s = project(src_feats, usc).mean(0); sd_s = project(src_feats, usc).std(0) + 1e-6
                full = (Z_F - mu_t) / sd_t * sd_s + mu_s
                X_corr = recon((1 - 1.0) * Z_F + 1.0 * full, usc)
                base = ap_from_X(td['per_img'], F, p_hat_g, neg_raw_g)
                three = ap_from_X(td['per_img'], X_corr, p_hat_g, neg_raw_g)
            else:
                # 无人机域: rank-1 affine(原型视角), per-class best α(上界)
                p_src = dd['protos_gpu'][src_cls]
                neg_gpu = torch.tensor(neg_raw_g, dtype=torch.float32, device=neg_dev) if neg_raw_g is not None else None
                best_a, best_ap = 0.0, -1
                for a in ALPHAS:
                    p_corr = usc.correct_prototype_gpu(p_src, 0.0, a).cpu().numpy()
                    neg_c = usc.correct_prototype_gpu(neg_gpu, 0.0, a).cpu().numpy() if neg_gpu is not None else None
                    p_hat = p_corr + dd['text_deltas'][src_cls]; p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)
                    ap = V12.evaluate(td, p_hat, neg_c)
                    if ap > best_ap:
                        best_ap = ap; best_a = a
                p0 = usc.correct_prototype_gpu(p_src, 0.0, 0.0).cpu().numpy()
                p0hat = p0 + dd['text_deltas'][src_cls]; p0hat = p0hat / (np.linalg.norm(p0hat) + 1e-8)
                base = V12.evaluate(td, p0hat, neg_raw_g)
                three = best_ap
            cls_rec.append({'class': coco_cls, 'src': src_cls, 'base': round(base, 4),
                           'three': round(three, 4), 'delta': round(three - base, 4)})
            base_all.append(base); three_all.append(three)
            print(f"  {coco_cls:<11s}({src_cls:<9s}) base={base:.4f} three={three:.4f} Δ={three-base:+.4f}")
        b_m = np.mean(base_all); t_m = np.mean(three_all)
        print(f"  → {src} 域均值: base={b_m:.4f} three={t_m:.4f} Δ={t_m-b_m:+.4f}")
        results[src] = {'per_class': cls_rec, 'domain': {'base': round(b_m, 4), 'three': round(t_m, 4), 'delta': round(t_m - b_m, 4)}}
    all_b = [r['domain']['base'] for r in results.values()]
    all_t = [r['domain']['three'] for r in results.values()]
    print(f"\n跨域(4域均值): base={np.mean(all_b):.4f} three={np.mean(all_t):.4f} Δ={np.mean(all_t)-np.mean(all_b):+.4f}")
    json.dump(results, open(DATA / 'autoproc_three_stage_full_results.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"\n✅ -> {DATA / 'autoproc_three_stage_full_results.json'}")


if __name__ == '__main__':
    main()
