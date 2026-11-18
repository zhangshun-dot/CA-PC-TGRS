# -*- coding: utf-8 -*-
"""多信号集成 whether 闸门探针(零样本代理, 离线):
元提议: 单角度(sem_mig)失效, 多个角度联合判别是否能提升 whether 决策?
设计: 构 n 个零样本代理信号(均推理时可得, 无标签), 标准化后 LOO 逻辑回归
预测"校正是否有益"(标签=C-A符号, 来自 e2e JSON), 对比单信号 sem_mig 一致率。
嵌入: 多信号线性组合=能量, Δ>0 接受/<0 退回(Metropolis), 此处只验方向。
注意: LOO 内学 w 不用目标类自身标签(其他类观测行为预测该类), 非测试集泄露。
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA = V12.DATA
EVAL_DOMAINS = ['DIOR', 'VisDrone']
N_TARGET = 50
GSD = {'DIOR': 1.0, 'NWPU': 1.0, 'DOTA': 0.5, 'VisDrone': 0.3}
VIEW = {'DIOR': 0.0, 'NWPU': 0.0, 'DOTA': 0.0, 'VisDrone': 1.0}
SIGNAL_NAMES = ['baselign', 'sem_mig', 'div', 'av', 'n_boxes', 'det_mean', 'det_var', 'gsd', 'view']


def main():
    # 标签: e2e C-A 符号 (domain, coco_cls) -> y (+1 校正有益 / -1 有害)
    e2e = json.load(open(DATA / 'autoproc_essence_e2e_results.json', encoding='utf-8'))
    y_of, sig_of = {}, {}
    for dom, res in e2e.items():
        for r in res['per_class']:
            y_of[(dom, r['class'])] = 1 if (r['C_ess'] - r['A']) > 0 else -1

    rows = []
    for src in EVAL_DOMAINS:
        dd = load_domain_data(src)
        for coco_cls in list(dd['target_data'].keys()):
            td = dd['target_data'][coco_cls]
            td['per_img'] = td['per_img'][:N_TARGET]
        usc = dd['usc']
        protos_gpu = dd['protos_gpu']
        text_deltas = dd['text_deltas']

        for eval_name, coco_cls, src_cls in dd['valid_targets']:
            # 源原型(γ=0,α=0) + 文本外推 -> 512
            p0 = usc.correct_prototype_gpu(protos_gpu[src_cls], 0.0, 0.0).cpu().numpy()
            p_hat = p0 + text_deltas[src_cls]
            p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)
            F = dd['target_data'][coco_cls]['features']            # (N,512)
            F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
            td = dd['target_data'][coco_cls]
            # baselign: 目标框与源原型余弦均值
            baselign = float(np.mean(F @ p_hat))
            # sem_mig
            s_emb = T.text_avg([V12.SRC_TMPL_OVERHEAD], src_cls).cpu().numpy()
            d_emb = T.text_avg([V12.TGT_TMPL], coco_cls).cpu().numpy()
            sem_mig = float(1.0 - float(np.dot(s_emb, d_emb)))
            # div / av: 源域 COCO 框特征
            Fs = dd['source_data'][src_cls]['features']
            Fs = Fs / (np.linalg.norm(Fs, axis=1, keepdims=True) + 1e-8)
            div = float(np.mean(np.diag(np.cov(Fs.T))))
            cav = Fs @ d_emb
            av = float(np.var(cav))
            # det_stats: 目标域检测统计(推理时合法, 无标签)
            gds = np.concatenate([p['gd'] for p in td['per_img']]) if any(len(p['gd']) for p in td['per_img']) else np.zeros(0)
            n_boxes = float(np.mean([len(p['gd']) for p in td['per_img']])) if td['per_img'] else 0.0
            det_mean = float(np.mean(gds)) if len(gds) else 0.0
            det_var = float(np.var(gds)) if len(gds) else 0.0
            sig = [baselign, sem_mig, div, av, n_boxes, det_mean, det_var, GSD[src], VIEW[src]]
            key = (src, coco_cls)
            if key in y_of:
                sig_of[key] = sig
                rows.append(key)

    keys = list(sig_of.keys())
    X = np.array([sig_of[k] for k in keys], dtype=float)
    Y = np.array([y_of[k] for k in keys])
    print(f"样本 n={len(keys)}: {[k[1] for k in keys]}")
    print(f"标签分布: +1(有益)={int((Y>0).sum())}  -1(有害)={int((Y<0).sum())}")

    # 标准化(按全体, 仅用于展示; LOO 内用训练折标准化避免泄露)
    def loo_eval(Xall, Yall):
        pred = np.zeros(len(Yall)); prob = np.zeros(len(Yall))
        for i in range(len(Yall)):
            tr = [j for j in range(len(Yall)) if j != i]
            mu, sd = Xall[tr].mean(0), Xall[tr].std(0) + 1e-6
            Xtr = (Xall[tr] - mu) / sd
            Xte = (Xall[i] - mu) / sd
            clf = LogisticRegression(C=0.1, max_iter=2000)
            clf.fit(Xtr, Yall[tr])
            pp = clf.predict_proba(Xte.reshape(1, -1))[0, 1]
            prob[i] = pp
            pred[i] = 1 if pp > 0.5 else -1
        return pred, prob

    pred, prob = loo_eval(X, Y)
    acc = float(np.mean(pred == Y))
    auc = float(roc_auc_score((Y > 0).astype(int), prob)) if len(set((Y > 0).astype(int))) > 1 else float('nan')
    print(f"\n[多信号 LOO] 符号一致率={acc:.2f}  AUC={auc:.2f}")
    for k, p, pr, yy in zip(keys, pred, prob, Y):
        print(f"  {k[0]:<9s} {k[1]:<11s} y={yy:+.0f} pred={p:+.0f} p={pr:.2f} {'✓' if p==yy else '✗'}")

    # 基线: 单 sem_mig (索引1) LOO 一致率
    Xs = X[:, [1]]
    pred_s, _ = loo_eval(Xs, Y)
    acc_s = float(np.mean(pred_s == Y))
    print(f"\n[单 sem_mig LOO] 符号一致率={acc_s:.2f}  (之前离线 0.50)")
    print(f"[多数类基线] 全预测+1 一致率={float(np.mean(Y>0)):.2f}")

    out = {'n': len(keys), 'multi_acc': acc, 'multi_auc': auc,
           'single_sem_mig_acc': acc_s, 'majority_acc': float(np.mean(Y > 0)),
           'signal_names': SIGNAL_NAMES}
    json.dump(out, open(DATA / 'autoproc_multisignal_gate_results.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"\n✅ -> autoproc_multisignal_gate_results.json")


if __name__ == '__main__':
    main()
