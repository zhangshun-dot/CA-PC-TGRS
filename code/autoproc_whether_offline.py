# -*- coding: utf-8 -*-
"""离线验证 whether(语义迁移度) 能否当"推不推"闸门信号。
零 GPU: 复用 e2e JSON 的真实 C-A(推 vs 不推)符号 + LOO JSON 的 src_class,
算 sem_mig(src_concept, coco_concept), 检验 sem_mig 高 ↔ C-A>0。
并离线预估闸门后域均值(D = w*C + (1-w)*A, A 已 γ=0 故 w→0 退 A)。
"""
import json, sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
from pathlib import Path
import torch
import tge_probe as T
import usc_experiment_v12 as V12
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

DATA = V12.DATA
SRC_TMPL = V12.SRC_TMPL_OVERHEAD
TGT_TMPL = V12.TGT_TMPL
GSD = {'DIOR': 1.0, 'NWPU': 1.0, 'DOTA': 0.5, 'VisDrone': 0.3}
VIEW = {'DIOR': 0.0, 'NWPU': 0.0, 'DOTA': 0.0, 'VisDrone': 1.0}

e2e = json.load(open(DATA / 'autoproc_essence_e2e_results.json', encoding='utf-8'))
loo = json.load(open(DATA / 'usc_detstats_gamma_alpha_loo_results.json', encoding='utf-8'))
src_of = {(r['domain'], r['class']): r['src_class'] for r in loo['class_results']}

def sem_mig(src_cls, coco_cls):
    s = T.text_avg([SRC_TMPL], src_cls).cpu().numpy()
    d = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
    return float(1.0 - float(np.dot(s, d)))

rows = []
for dom, res in e2e.items():
    for r in res['per_class']:
        coco = r['class']
        src = src_of.get((dom, coco), coco)
        sm = sem_mig(src, coco)
        ca = r['C_ess'] - r['A']
        rows.append({'dom': dom, 'coco': coco, 'src': src, 'sem_mig': sm,
                     'A': r['A'], 'C': r['C_ess'], 'CA': ca,
                     'GSD': GSD[dom], 'VIEW': VIEW[dom]})

sm = np.array([x['sem_mig'] for x in rows])
ca = np.array([x['CA'] for x in rows])
print(f"样本 n={len(rows)} (DIOR {sum(x['dom']=='DIOR' for x in rows)} + VisDrone {sum(x['dom']=='VisDrone' for x in rows)})")
print(f"sem_mig 范围 [{sm.min():.3f}, {sm.max():.3f}], 均值 {sm.mean():.3f}")
print(f"spearman(sem_mig, C-A) = {spearmanr(sm, ca).statistic:+.3f} (p={spearmanr(sm, ca).pvalue:.3f})")

# 符号一致率: sem_mig 高于中位 → 预测推(C-A>0)
med = np.median(sm)
pred_push = sm > med
acc = np.mean(pred_push == (ca > 0))
print(f"\n硬门(sem_mig>中位 预测推): 符号一致率 = {acc:.2f} ({int(acc*len(rows))}/{len(rows)})")
print(f"  中位 sem_mig = {med:.3f}")
for x in rows:
    mark = '✓' if (x['sem_mig'] > med) == (x['CA'] > 0) else '✗'
    print(f"  {mark} {x['dom']:<9s} {x['coco']:<11s} src={x['src']:<9s} sem_mig={x['sem_mig']:.3f}  C-A={x['CA']:+.4f}")

# 离线预估 D 域均值
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

print("\n=== 离线 D 预估 (D = w*C + (1-w)*A) ===")
# (1) 硬门: 域级 VIEW (DIOR→0 不推, VisDrone→1 推)
for dom in ['DIOR', 'VisDrone']:
    sub = [x for x in rows if x['dom'] == dom]
    w = VIEW[dom]
    dmean = np.mean([w * x['C'] + (1 - w) * x['A'] for x in sub])
    amean = np.mean([x['A'] for x in sub]); cmean = np.mean([x['C'] for x in sub])
    print(f"  [硬门VIEW] {dom}: A={amean:.4f} C={cmean:.4f} D={dmean:.4f} (ΔD-A={dmean-amean:+.4f})")

# (2) 软门: per-class, w=sigmoid(8*(sem_mig-med))
for dom in ['DIOR', 'VisDrone']:
    sub = [x for x in rows if x['dom'] == dom]
    ws = sigmoid(8 * (np.array([x['sem_mig'] for x in sub]) - med))
    ds = [w * x['C'] + (1 - w) * x['A'] for w, x in zip(ws, sub)]
    dmean = np.mean(ds); amean = np.mean([x['A'] for x in sub]); cmean = np.mean([x['C'] for x in sub])
    print(f"  [软门sem_mig] {dom}: A={amean:.4f} C={cmean:.4f} D={dmean:.4f} (ΔD-A={dmean-amean:+.4f}, w均值={ws.mean():.2f})")

# (3) 软门 + GSD 偏置: w=sigmoid(8*(sem_mig-med)) * (1 - 0.5*GSD归一) — 卫星(GSD大)再压
print("\n(注: 离线 D 为 AP 线性混合近似; 真实闸门作用于 γ_eff=w*γ, α_eff=w*α_ess, 需 GPU 实跑坐实)")
json.dump({'rows': rows, 'spearman': float(spearmanr(sm, ca).statistic),
           'sign_acc': float(acc), 'sem_mig_med': float(med)},
          open(DATA / 'autoproc_whether_offline.json', 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
print("\n✅ -> autoproc_whether_offline.json")
