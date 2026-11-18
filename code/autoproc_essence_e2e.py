# -*- coding: utf-8 -*-
"""AutoProCorrector 端到端 AP 小样本验证(本质先验 vs det_stats):
源域∈{DIOR, VisDrone}, 目标域=COCO。复用 load_domain_data + AffineUSC。
三组对比(每类, 每源域):
  A) 原始USC(γ=0, α=0.5)
  B) det_stats自适应(γ=sensor, α=det_stats LOO预测)   <- 等同 usc_final C4
  C) 本质先验(γ=sensor, α=essence LOO预测)            <- 新方案
重点: VisDrone 源域 C 是否比 B(已知负向 -0.49%) 与 A 翻正。
α 模型标签 oracle_alpha 取自已有 LOO JSON(γ=-2 网格扫描), 不重扫。
"""
import sys, json
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
from pathlib import Path
import torch
import usc_experiment_v12 as V12
import unified_eval as U
import tge_probe as T
from usc_rank1_affine_gpu_v2 import load_domain_data
from sklearn.linear_model import LinearRegression

DEVICE = U.DEVICE
DATA = V12.DATA
SRC_TMPL = V12.SRC_TMPL_OVERHEAD
TGT_TMPL = V12.TGT_TMPL
ALPHAS = np.array(V12.ALPHAS)
GSD = {'DIOR': 1.0, 'NWPU': 1.0, 'DOTA': 0.5, 'VisDrone': 0.3}
VIEW = {'DIOR': 0.0, 'NWPU': 0.0, 'DOTA': 0.0, 'VisDrone': 1.0}
SENSOR_GAMMA = {'DIOR': -2.0, 'NWPU': -2.0, 'DOTA': -2.0, 'VisDrone': -1.0}
EVAL_DOMAINS = ['DIOR', 'VisDrone']
N_TARGET = 50  # 每类 COCO 目标域图数(小样本)


def coco_src_features(coco_cls):
    """纯 COCO 源域 GT 框 CLIP 特征 → 可漂移性先验(零样本, 不碰目标域)。"""
    images, gts = U.load_coco(coco_cls, n=50)
    if not images:
        return None
    crops = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 > x1 + 10 and y2 > y1 + 10:
                crops.append(img.crop((x1, y1, x2, y2)))
    if not crops:
        return None
    F = T.encode_clip_batch(crops).cpu().numpy()
    txt = T.text_avg([SRC_TMPL], coco_cls).cpu().numpy()
    cos = F @ txt
    return {'div': float(np.mean(np.diag(np.cov(F.T)))), 'av': float(np.var(cos))}


def sem_mig(src_cls, coco_cls):
    s = T.text_avg([SRC_TMPL], src_cls).cpu().numpy()
    d = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
    return float(1.0 - float(np.dot(s, d)))


def loo_pred(feat_dict, y_dict):
    """LOO 线性回归预测每类 α。"""
    keys = list(feat_dict.keys())
    X = np.array([feat_dict[k] for k in keys])
    Y = np.array([y_dict[k] for k in keys])
    pred = {}
    for i, k in enumerate(keys):
        m = LinearRegression().fit(np.delete(X, i, 0), np.delete(Y, i))
        raw = float(m.predict(X[i:i + 1])[0])
        pred[k] = float(ALPHAS[np.argmin(np.abs(ALPHAS - raw))])
    return pred


def evaluate_config(dd, src_cls, coco_cls, gamma, alpha):
    usc = dd['usc']
    protos_gpu = dd['protos_gpu']
    all_protos = dd['all_protos']
    target_data = dd['target_data']
    p_corr = usc.correct_prototype_gpu(protos_gpu[src_cls], gamma, alpha).cpu().numpy()
    neg_raw = V12.build_neg_proto(all_protos, src_cls)
    if neg_raw is not None:
        neg_gpu = torch.tensor(neg_raw, dtype=torch.float32, device=DEVICE)
        neg = usc.correct_prototype_gpu(neg_gpu, gamma, alpha).cpu().numpy()
    else:
        neg = None
    s = T.text_avg([SRC_TMPL], src_cls).cpu().numpy()
    d = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
    p_hat = p_corr + (d - s)
    p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)
    return float(V12.evaluate(target_data[coco_cls], p_hat, neg))


def main():
    # 标签 + 特征 (复用已有 LOO JSON 的 oracle_alpha)
    loo = json.load(open(DATA / 'usc_detstats_gamma_alpha_loo_results.json', encoding='utf-8'))
    rows = loo['class_results']
    feat_ess, feat_det, y = {}, {}, {}
    for r in rows:
        dom, coco, src = r['domain'], r['class'], r['src_class']
        cf = coco_src_features(coco)
        if cf is None:
            continue
        feat_ess[(dom, coco)] = [cf['div'], cf['av'], GSD[dom], VIEW[dom], sem_mig(src, coco)]
        feat_det[(dom, coco)] = [r['det_stats']['det_mean'], r['det_stats']['det_var'], r['det_stats']['n_boxes']]
        y[(dom, coco)] = r['oracle_alpha']
    pred_ess = loo_pred(feat_ess, y)
    pred_det = loo_pred(feat_det, y)
    print(f"α 模型: 训练类 {len(y)} 个 (essence + det_stats 两版)")

    results = {}
    for src in EVAL_DOMAINS:
        print(f"\n{'='*60}\n源域 {src} → COCO (N_target={N_TARGET})\n{'='*60}")
        dd = load_domain_data(src)
        # 临时限制目标域图数(小样本)
        for coco_cls in dd['target_data']:
            td = dd['target_data'][coco_cls]
            if 'per_img' in td:
                td['per_img'] = td['per_img'][:N_TARGET]
        ap_a, ap_b, ap_c = [], [], []
        cls_rec = []
        for eval_name, coco_cls, src_cls in dd['valid_targets']:
            a = evaluate_config(dd, src_cls, coco_cls, 0.0, 0.5)
            a_det = pred_det.get((src, coco_cls), 0.5)
            b = evaluate_config(dd, src_cls, coco_cls, SENSOR_GAMMA[src], a_det)
            a_ess = pred_ess.get((src, coco_cls), 0.5)
            c = evaluate_config(dd, src_cls, coco_cls, SENSOR_GAMMA[src], a_ess)
            ap_a.append(a); ap_b.append(b); ap_c.append(c)
            cls_rec.append({'class': coco_cls, 'A': round(a, 4), 'B_det': round(b, 4), 'C_ess': round(c, 4)})
            print(f"  {coco_cls:<12s} A={a:.4f}  B(det)={b:.4f}  C(ess)={c:.4f}")
        results[src] = {
            'A_orig': float(np.mean(ap_a)), 'B_det': float(np.mean(ap_b)), 'C_ess': float(np.mean(ap_c)),
            'per_class': cls_rec,
        }
        print(f"  → 域均值: A={results[src]['A_orig']:.4f}  B(det)={results[src]['B_det']:.4f}  C(ess)={results[src]['C_ess']:.4f}")

    with open(DATA / 'autoproc_essence_e2e_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ -> {DATA / 'autoproc_essence_e2e_results.json'}")


if __name__ == '__main__':
    main()
