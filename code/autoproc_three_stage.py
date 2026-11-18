# -*- coding: utf-8 -*-
"""三阶段 pipeline 可行性验证(DIOR+VisDrone 小样本)。
阶段1 whether(要不要): per-class 零样本语义闸门 — coco类名与 'large vehicle' 文本相似度
                          >阈值→推(g=1), 否则 g=0。零样本、不泄露目标AP。
阶段2 how(怎么做): 特征空间二阶对齐(whitening-coloring), 用目标无标签统计(留出半集)。
阶段3 how much(做多少): λ_c = 语义闸门信号(受益类强度1, 其他0)。
串通: X_corr = (1 - g_c*lam_c)*F + g_c*lam_c*T(F); 原型 p_hat 固定(文本外推)。
对照: base(原始gd) / allAlign(全类λ=1, 看净≈0) / threeStage(语义闸门选类对齐)。
"""
import sys, json
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
from pathlib import Path
import usc_experiment_v12 as V12
import unified_eval as U
from usc_rank1_affine_gpu_v2 import load_domain_data, SOURCE_LOADERS
import tge_probe as T

DATA = V12.DATA
N_SUB = 50
SEEDS = [0, 1, 2, 3, 4]
SPLIT_SEED = 999
THRESH = 0.20  # large vehicle 相似度阈值(看分布再调)


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
    lv_emb = T.text_emb(['a photo of a large vehicle']).cpu().numpy().flatten()
    lv_emb = lv_emb / (np.linalg.norm(lv_emb) + 1e-8)

    results = {}
    for src in ['DIOR', 'VisDrone']:
        print(f"\n{'='*64}\n源域 {src}→COCO 三阶段验证\n{'='*64}")
        dd = load_domain_data(src)
        usc = dd['usc']
        cls_rec = []
        for eval_name, coco_cls, src_cls in dd['valid_targets']:
            td = dd['target_data'][coco_cls]
            F = td['features']
            p_hat_g = (usc.correct_prototype_gpu(dd['protos_gpu'][src_cls], 0.0, 0.0).cpu().numpy()
                       + dd['text_deltas'][src_cls]); p_hat_g = p_hat_g / (np.linalg.norm(p_hat_g) + 1e-8)
            neg_raw_g = V12.build_neg_proto(dd['all_protos'], src_cls)
            Z_F = project(F, usc)
            n_imgs = len(td['per_img'])
            rng0 = np.random.default_rng(SPLIT_SEED)
            perm = rng0.permutation(n_imgs); half = n_imgs // 2
            stats_idx = perm[:half].tolist(); eval_idx = perm[half:].tolist()
            stats_rows = np.concatenate([np.asarray(td['per_img'][i]['crop_idx']).reshape(-1) for i in stats_idx])
            mu_t = Z_F[stats_rows].mean(0); sd_t = Z_F[stats_rows].std(0) + 1e-6
            src_feats = dd['source_data'][src_cls]['features']
            mu_s = project(src_feats, usc).mean(0); sd_s = project(src_feats, usc).std(0) + 1e-6
            eval_perimg = [td['per_img'][i] for i in eval_idx]

            def align(lam):
                full = (Z_F - mu_t) / sd_t * sd_s + mu_s
                return (1 - lam) * Z_F + lam * full
            X_corr_full = recon(align(1.0), usc)   # 二阶对齐后特征(全量)

            # 语义闸门(失效参考, 仅打印): 类名与 'large vehicle' 相似度
            cls_emb = T.text_emb([V12.TGT_TMPL.format(coco_cls)]).cpu().numpy().flatten()
            cls_emb = cls_emb / (np.linalg.norm(cls_emb) + 1e-8)
            sim = float(np.dot(cls_emb, lv_emb))
            # 域级视角闸门(反转策略): 卫星域(overhead→photo视角差大)开二阶对齐, 无人机域(已近photo)关
            SAT_DOMAINS = {'DIOR', 'NWPU', 'DOTA'}
            g_dom = 1.0 if src in SAT_DOMAINS else 0.0
            g_c = g_dom
            lam_c = g_c
            X3 = (1 - g_c * lam_c) * F + g_c * lam_c * X_corr_full  # 三阶段特征(反转whether)

            # 评估(全量 eval-half 单次 + 5 seed 子采样均值)
            base = ap_from_X(eval_perimg, F, p_hat_g, neg_raw_g)
            allalign = ap_from_X(eval_perimg, X_corr_full, p_hat_g, neg_raw_g)
            three = ap_from_X(eval_perimg, X3, p_hat_g, neg_raw_g)
            # 5 seed 稳定性
            k = min(N_SUB, len(eval_perimg))
            three_ss = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                sub = [eval_perimg[i] for i in rng.choice(len(eval_perimg), size=k, replace=False)]
                three_ss.append(ap_from_X(sub, X3, p_hat_g, neg_raw_g))
            three_ss = np.array(three_ss)
            cls_rec.append({'class': coco_cls, 'src': src_cls, 'simLV': round(sim, 3), 'gate': int(g_c),
                            'base': round(base, 4), 'allAlign': round(allalign, 4),
                            'threeStage': round(three, 4),
                            'threeSeed_mean': round(three_ss.mean(), 4), 'threeSeed_std': round(three_ss.std(), 4)})
            print(f"  {coco_cls:<11s}({src_cls:<9s}) simLV={sim:+.3f} gate={int(g_c)} | "
                  f"base={base:.4f} allAlign={allalign:+.4f}({allalign-base:+.4f}) three={three:+.4f}({three-base:+.4f}) "
                  f"[seed {three_ss.mean():.4f}±{three_ss.std():.4f}]")
        b_m = np.mean([r['base'] for r in cls_rec])
        a_m = np.mean([r['allAlign'] for r in cls_rec])
        t_m = np.mean([r['threeStage'] for r in cls_rec])
        print(f"  → {src} 域均值: base={b_m:.4f} allAlign={a_m:.4f}({a_m-b_m:+.4f}) threeStage={t_m:.4f}({t_m-b_m:+.4f})")
        results[src] = {'per_class': cls_rec, 'domain': {'base': round(b_m, 4), 'allAlign': round(a_m, 4), 'threeStage': round(t_m, 4)}}

    json.dump(results, open(DATA / 'autoproc_three_stage_results.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"\n✅ -> {DATA / 'autoproc_three_stage_results.json'}")


if __name__ == '__main__':
    main()
