# -*- coding: utf-8 -*-
"""AutoProCorrector 本质先验验证(最小化):
用"源域可漂移性先验 + 物理GSD/视角 + 语义迁移度"预测逐类最优α, LOO 对比 det_stats(n_boxes)。

不重新检测目标域(最重GPU部分), 标签 oracle_alpha 取自已有
  usc_detstats_gamma_alpha_loo_results.json
只编码源域 GT 框(轻 GPU)。

验证假设(方案 §1.3 how much): 本质先验能否替代目标域统计预测逐类α。
判定: essence 版 LOO 的 spearman(预测α, oracleα) 与 MAE, 对比 det_stats 版。
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

# 物理先验 (米/像素, 遥感近似; VisDrone 倾斜→尺度多变取小值代表尺度不确定性高)
GSD = {'DIOR': 1.0, 'NWPU': 1.0, 'DOTA': 0.5, 'VisDrone': 0.3}
VIEW = {'DIOR': 0.0, 'NWPU': 0.0, 'DOTA': 0.0, 'VisDrone': 1.0}  # 0=正射, 1=倾斜透视

LOADERS = {
    'DIOR': V12.load_dior_src,
    'NWPU': V12.load_nwpu_src,
    'DOTA': V12.load_dota_src,
    'VisDrone': V12.load_visdrone_src,
}


def src_features(domain, src_cls):
    """源域 GT 框 CLIP 特征 → 可漂移性先验。轻 GPU。"""
    loader = LOADERS[domain]
    d = loader(src_cls, n=50)
    if d is None or d['features'] is None or len(d['features']) < 2:
        return None
    F = np.asarray(d['features'], dtype=np.float64)        # (N,512)
    txt = T.text_avg([SRC_TMPL], src_cls).cpu().numpy()    # (512,)
    cos = F @ txt                                          # (N,)
    diversity = float(np.mean(np.diag(np.cov(F.T))))       # 源域外观多样性
    align_mean = float(np.mean(cos))
    align_var = float(np.var(cos))                         # 源域对齐方差
    return {'diversity': diversity, 'align_mean': align_mean, 'align_var': align_var}


def sem_mig(src_cls, coco_cls):
    """语义迁移度: 1 - cos(源文本原型, 目标文本原型)。零样本。"""
    s = T.text_avg([SRC_TMPL], src_cls).cpu().numpy()
    d = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
    return float(1.0 - float(np.dot(s, d)))


def loo_predict(X, y):
    """留一法线性回归预测, 返回预测向量。"""
    n = len(y)
    pred = np.zeros(n)
    for i in range(n):
        Xtr = np.delete(X, i, axis=0)
        ytr = np.delete(y, i)
        m = LinearRegression().fit(Xtr, ytr)
        pred[i] = m.predict(X[i:i + 1])[0]
    return pred


def main():
    loo = json.load(open(DATA / 'usc_detstats_gamma_alpha_loo_results.json', encoding='utf-8'))
    rows = loo['class_results']
    print(f"标签数(类): {len(rows)}")

    ess_feats, det_feats, y_oracle = [], [], []
    ok = 0
    for r in rows:
        dom = r['domain']; coco = r['class']; src = r['src_class']
        sf = src_features(dom, src)
        if sf is None:
            continue
        ds = r['det_stats']
        ess = [sf['diversity'], sf['align_var'], GSD[dom], VIEW[dom], sem_mig(src, coco)]
        det = [ds['det_mean'], ds['det_var'], ds['n_boxes']]
        ess_feats.append(ess)
        det_feats.append(det)
        y_oracle.append(r['oracle_alpha'])
        ok += 1
    print(f"有效类(源域特征可用): {ok}/{len(rows)}")

    X_ess = np.array(ess_feats)
    X_det = np.array(det_feats)
    y = np.array(y_oracle)

    p_ess = loo_predict(X_ess, y)
    p_det = loo_predict(X_det, y)

    rho_ess, _ = spearmanr(p_ess, y)
    rho_det, _ = spearmanr(p_det, y)
    mae_ess = float(np.mean(np.abs(p_ess - y)))
    mae_det = float(np.mean(np.abs(p_det - y)))

    out = {
        'n_valid': ok,
        'essence': {'features': ['diversity', 'align_var', 'GSD', 'view', 'sem_mig'],
                    'spearman_rho_pred_vs_oracle': float(rho_ess), 'mae': mae_ess},
        'det_stats': {'features': ['det_mean', 'det_var', 'n_boxes'],
                      'spearman_rho_pred_vs_oracle': float(rho_det), 'mae': mae_det},
    }
    print("\n=== LOO 预测逐类最优α: 本质先验 vs det_stats ===")
    print(f"  essence : ρ={rho_ess:+.3f}  MAE={mae_ess:.4f}")
    print(f"  det_stat: ρ={rho_det:+.3f}  MAE={mae_det:.4f}")
    better = "essence" if (abs(rho_ess) > abs(rho_det) and mae_ess < mae_det) else "det_stats"
    print(f"  → 更稳的 how-much 驱动: {better}")

    with open(DATA / 'autoproc_essence_loov_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✅ -> {DATA / 'autoproc_essence_loov_results.json'}")


if __name__ == '__main__':
    main()
