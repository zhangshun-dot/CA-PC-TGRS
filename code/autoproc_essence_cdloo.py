# -*- coding: utf-8 -*-
"""AutoProCorrector 本质先验验证(跨域留一, cross-domain LOO):
修正 v1 的缺陷: 源域先验只来自 COCO 源域(通用知识) + 物理GSD/视角(领域元知识) + 文本语义迁移度,
绝不碰目标域图像/统计 (严格零样本)。

公平对比 det_stats:
  - essence: COCO源域特征 + held域物理先验(零样本可得) → 预测 held 域逐类最优α
  - det_stats: 零样本跨域下 held 域目标域统计不可得 → 退化为 训练域 det_stats 均值 proxy

判定: 跨域泛化下, essence 是否 ≥ det_stats proxy (即本质先验真正零样本跨域可用)。
"""
import sys, json
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
from pathlib import Path
import usc_experiment_v12 as V12
import unified_eval as U
import tge_probe as T
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

DEVICE = U.DEVICE
DATA = V12.DATA
SRC_TMPL = V12.SRC_TMPL_OVERHEAD
TGT_TMPL = V12.TGT_TMPL
GSD = {'DIOR': 1.0, 'NWPU': 1.0, 'DOTA': 0.5, 'VisDrone': 0.3}
VIEW = {'DIOR': 0.0, 'NWPU': 0.0, 'DOTA': 0.0, 'VisDrone': 1.0}


def coco_src_features(coco_cls):
    """纯 COCO 源域 GT 框 CLIP 特征 → 可漂移性先验。零样本可得, 不碰目标域。"""
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
    F = T.encode_clip_batch(crops).cpu().numpy()          # (N,512)
    txt = T.text_avg([SRC_TMPL], coco_cls).cpu().numpy()   # (512,)
    cos = F @ txt
    return {'diversity': float(np.mean(np.diag(np.cov(F.T)))),
            'align_var': float(np.var(cos))}


def sem_mig(src_cls, coco_cls):
    s = T.text_avg([SRC_TMPL], src_cls).cpu().numpy()
    d = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
    return float(1.0 - float(np.dot(s, d)))


def main():
    loo = json.load(open(DATA / 'usc_detstats_gamma_alpha_loo_results.json', encoding='utf-8'))
    rows = loo['class_results']
    domains = ['DIOR', 'NWPU', 'DOTA', 'VisDrone']

    # 预计算每个类的 essence 特征(COCO源域+物理+文本) 与 det_stats
    cache = {}
    for r in rows:
        dom, coco, src = r['domain'], r['class'], r['src_class']
        cf = coco_src_features(coco)
        if cf is None:
            cache[(dom, coco)] = None
            continue
        ess = [cf['diversity'], cf['align_var'], GSD[dom], VIEW[dom], sem_mig(src, coco)]
        det = [r['det_stats']['det_mean'], r['det_stats']['det_var'], r['det_stats']['n_boxes']]
        cache[(dom, coco)] = {'ess': ess, 'det': det, 'alpha': r['oracle_alpha']}

    res = {}
    for held in domains:
        tr = [cache[(r['domain'], r['class'])] for r in rows
              if r['domain'] in domains and r['domain'] != held and cache.get((r['domain'], r['class']))]
        te = [cache[(r['domain'], r['class'])] for r in rows
              if r['domain'] == held and cache.get((r['domain'], r['class']))]
        if not tr or not te:
            continue
        Xtr_e = np.array([t['ess'] for t in tr])
        Xtr_d = np.array([t['det'] for t in tr])
        ytr = np.array([t['alpha'] for t in tr])
        m_e = LinearRegression().fit(Xtr_e, ytr)
        m_d = LinearRegression().fit(Xtr_d, ytr)

        Xte_e = np.array([t['ess'] for t in te])
        det_proxy = np.mean(Xtr_d, axis=0)                 # 零样本跨域下 held 目标域统计不可得
        Xte_d = np.tile(det_proxy, (len(te), 1))
        y = np.array([t['alpha'] for t in te])

        pe = m_e.predict(Xte_e)
        pd = m_d.predict(Xte_d)
        re, _ = spearmanr(pe, y)
        rd, _ = spearmanr(pd, y)
        res[held] = {'ess_rho': float(re), 'det_proxy_rho': float(rd), 'n': len(te)}
        print(f"  held={held:<10s} ess ρ={re:+.3f}  det_proxy ρ={rd:+.3f}  (n={len(te)})")

    mean_e = float(np.mean([v['ess_rho'] for v in res.values()]))
    mean_d = float(np.mean([v['det_proxy_rho'] for v in res.values()]))
    print(f"\n跨域LOO均值: essence ρ={mean_e:+.3f}  det_stats(proxy) ρ={mean_d:+.3f}")
    winner = 'essence' if mean_e > mean_d else 'det_stats(proxy)'
    print(f"  → 跨域泛化下更可用的驱动: {winner}")

    out = {'per_domain': res, 'mean_essence_rho': mean_e, 'mean_det_proxy_rho': mean_d, 'winner': winner}
    with open(DATA / 'autoproc_essence_cdloo_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✅ -> {DATA / 'autoproc_essence_cdloo_results.json'}")


if __name__ == '__main__':
    main()
