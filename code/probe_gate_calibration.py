# -*- coding: utf-8 -*-
"""probe_gate_calibration.py — #2 Gate 信号鲁棒性 / conf 校准敏感性

审稿人必问: "gate(conf) 依赖检测器置信度, 若 conf 未校准怎么办?"
本脚本检验: 对目标域 conf 做温度缩放(在 GT 上拟合 T, 作为 dev 校准上界参考)后,
gate=1-tanh(scale*(conf-c0)) 的行为是否改变, 进而 CA-PC oracle% 是否变。

核心假设: gate 用相对置信度 (vs 数据驱动中位数 c0), 对 conf 的单调重标定不敏感
→ 即使原生 conf 未校准, 方法仍工作, 部署时无需 dev 校准。

做法: 三检测器(fwd+rev 9 单元) 取每框 conf + GT; 每检测器在框级拟合温度 T;
重算校准后每单元 max_conf → c0' → gate' → γ'=dc×gate' → 重跑 CA-PC AP;
比较 原生 gate oracle% vs 校准 gate oracle%。并出 γ(conf) before/after 曲线。
"""
import os, sys, json, pickle, csv
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验数据')
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import torch
from transformers import CLIPModel, CLIPProcessor
from acspc import metrics as Met

BIDIR = Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache')
OWL = Path(r'C:\Users\张三\Desktop\实验数据\probe_owl_cache')
YOLO = Path(r'C:\Users\张三\Desktop\实验数据\probe_yolo_cache')
PROTOS = Path(r'C:\Users\张三\Desktop\实验数据\probe_equiv_protos.pkl')
CELLS = Path(r'C:\Users\张三\Desktop\实验数据\closed_loop_cells.csv')
BAL = json.load(open(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced.json'))
OUT = Path(r'C:\Users\张三\Desktop\实验数据\probe_gate_calibration.json')
FIG = Path(r'C:\Users\张三\Desktop\实验数据\figs\fig8_gate_calibration.png')
N_PROTO = 20
CLIP_ID = 'openai/clip-vit-base-patch32'
C0 = BAL['config']['gate']['c0']
SCALE = BAL['config']['gate']['prior_scale']
LAM = BAL['config']['lam']
GATE_FROM = ['gdino', 'owlv2']
CACHE = {'gdino': BIDIR, 'owlv2': OWL, 'yolo': YOLO}
TS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]


def _as_vec(x):
    for a in ('pooler_output', 'image_embeds', 'text_embeds', 'last_hidden_state'):
        if hasattr(x, a) and getattr(x, a) is not None:
            return getattr(x, a)
    return x


def unit(x):
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def neg_of(bank, cls):
    sub = {k: v for k, v in bank.items() if k != cls}
    if not sub:
        return np.zeros(512)
    ks = list(sub)
    P = np.stack([sub[k] for k in ks])
    o = np.argsort(-(P @ bank[cls]))[:3]
    return unit(P[o].mean(0))


def iou(b, gts):
    if len(gts) == 0:
        return 0.0
    x1, y1, x2, y2 = b
    xx1 = np.maximum(x1, gts[:, 0]); yy1 = np.maximum(y1, gts[:, 1])
    xx2 = np.minimum(x2, gts[:, 2]); yy2 = np.minimum(y2, gts[:, 3])
    w = np.maximum(0, xx2 - xx1); h = np.maximum(0, yy2 - yy1)
    inter = w * h
    area_b = max(1e-9, (x2 - x1) * (y2 - y1)); area_g = np.maximum(1e-9, (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1]))
    return float(np.max(inter / (area_b + area_g - inter)))


def fit_T(confs, labels):
    best = None
    for T in TS:
        z = np.log(np.clip(confs, 1e-6, 1 - 1e-6) / (1 - np.clip(confs, 1e-6, 1 - 1e-6))) / T
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        bce = -(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean()
        if best is None or bce < best[1]:
            best = (T, float(bce))
    return best[0]


def main():
    m = CLIPModel.from_pretrained(CLIP_ID).eval()
    pr = CLIPProcessor.from_pretrained(CLIP_ID)

    @torch.no_grad()
    def Ttxt(ts):
        out = []
        for i in range(0, len(ts), 64):
            f = _as_vec(m.get_text_features(
                **pr(text=ts[i:i + 64], return_tensors='pt', padding=True, truncation=True))).float()
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
        return torch.cat(out).numpy()

    bank_all = pickle.load(open(PROTOS, 'rb'))
    cells = list(csv.DictReader(open(CELLS)))

    def ap_of(per_img, F, W):
        for pi in per_img:
            if len(pi['gd']):
                pi['_s'] = np.sqrt(pi['gd'] * np.clip(np.maximum(0.0, F[pi['crop_idx']] @ W), 1e-6, 1.0))
            else:
                pi['_s'] = np.zeros(0)
        return Met.compute_ap50_img([(x['box'], x['_s']) for x in per_img], [x['gt'] for x in per_img])

    def open_cache(det, tname, eval_cls):
        base = CACHE[det]
        for n in (100, 77):
            for pat in (f'{tname}_{eval_cls}_off{N_PROTO}_n{n}.pkl',
                        f'{tname}_{eval_cls}_n{n}.pkl'):
                f = base / pat
                if f.exists():
                    return pickle.load(open(f, 'rb'))
        return None

    # 1) 收集每检测器 框级 conf/label, 拟合 T; 同时记录每单元 raw max_conf
    det_boxes = {d: {'conf': [], 'lab': []} for d in CACHE}
    cell_raw = {}   # (det,direction,src,eval) -> max_conf_list(per image)
    cell_cache = {}
    for r in cells:
        det, direction, src, ev = r['det'], r['direction'], r['src'], r['eval']
        tname = 'fwd_dior2coco' if direction == 'fwd' else 'rev_coco2dior'
        b = open_cache(det, tname, ev)
        if b is None:
            continue
        cell_cache[(det, direction, src, ev)] = (b['per_img'], b['F'])
        maxc = []
        for pi in b['per_img']:
            gd = np.clip(pi['gd'], 1e-6, 1 - 1e-6)
            gts = np.array(pi['gt']) if len(pi['gt']) else np.zeros((0, 4))
            for ci in range(len(gd)):
                lab = 1.0 if iou(pi['box'][ci], gts) >= 0.5 else 0.0
                det_boxes[det]['conf'].append(float(gd[ci]))
                det_boxes[det]['lab'].append(lab)
            if len(gd):
                maxc.append(float(gd.max()))
        cell_raw[(det, direction, src, ev)] = maxc
    Tfit = {}
    for d in CACHE:
        cf = np.array(det_boxes[d]['conf']); lb = np.array(det_boxes[d]['lab'])
        Tfit[d] = fit_T(cf, lb)
        print(f'[{d}] T_fit={Tfit[d]:.2f}  (n_boxes={len(cf)}, pos_frac={lb.mean():.3f})')

    # 2) 计算 c0 (raw) 与 c0' (cal) = GATE_FROM 单元 max_conf 中位数
    def med(xs):
        return float(np.median(np.concatenate(xs))) if xs else 0.0
    raw_cells_gf = [cell_raw[k] for k in cell_raw if k[0] in GATE_FROM]
    cal_max = {}
    for k in cell_raw:
        det, direction, src, ev = k
        b = cell_cache[k][0]
        maxc = []
        for pi in b:
            gd = np.clip(pi['gd'], 1e-6, 1 - 1e-6)
            z = np.log(gd / (1 - gd)) / Tfit[det]
            gc = 1 / (1 + np.exp(-z))
            if len(gc):
                maxc.append(float(gc.max()))
        cal_max[k] = maxc
    cal_cells_gf = [cal_max[k] for k in cal_max if k[0] in GATE_FROM]
    c0_raw = med(raw_cells_gf)
    c0_cal = med(cal_cells_gf)
    print(f'c0_raw={c0_raw:.4f}  c0_cal={c0_cal:.4f}')

    # 3) 逐单元: gate raw/cal, γ, CA-PC AP, oracle%
    res = {'config': {'c0_raw': round(c0_raw, 4), 'c0_cal': round(c0_cal, 4),
                      'scale': SCALE, 'lam': LAM, 'T_fit': Tfit}, 'per_det': {}, 'cells': []}
    per_det = {d: {'gate_raw': [], 'gate_cal': [], 'oraw': [], 'ocal': []} for d in CACHE}
    for r in cells:
        det, direction, src, ev = r['det'], r['direction'], r['src'], r['eval']
        k = (det, direction, src, ev)
        if k not in cell_cache:
            continue
        per_img, F = cell_cache[k]
        base = float(r['base'])
        dc = float(r['dc_mag'])
        mc_raw = float(np.mean(cell_raw[k]))
        mc_cal = float(np.mean(cal_max[k]))
        g_raw = 1 - np.tanh(SCALE * (mc_raw - c0_raw))
        g_cal = 1 - np.tanh(SCALE * (mc_cal - c0_cal))
        bank = bank_all.get('dior' if direction == 'fwd' else 'coco', {})
        P_src = bank[src]; P_neg = neg_of(bank, src)
        dvec = Ttxt([f'a photo of {ev}'])[0] - Ttxt([f'an overhead image of {src}'])[0]
        W_raw = unit(P_src + (dc * g_raw) * dvec) - LAM * P_neg
        W_cal = unit(P_src + (dc * g_cal) * dvec) - LAM * P_neg
        ap_raw = ap_of(per_img, F, W_raw)
        ap_cal = ap_of(per_img, F, W_cal)
        oraw = ap_raw / base; ocal = ap_cal / base
        per_det[det]['gate_raw'].append(g_raw); per_det[det]['gate_cal'].append(g_cal)
        per_det[det]['oraw'].append(oraw); per_det[det]['ocal'].append(ocal)
        res['cells'].append({'det': det, 'dir': direction, 'src': src, 'eval': ev,
                             'mc_raw': round(mc_raw, 4), 'mc_cal': round(mc_cal, 4),
                             'gate_raw': round(g_raw, 4), 'gate_cal': round(g_cal, 4),
                             'oracle_raw': round(oraw, 4), 'oracle_cal': round(ocal, 4),
                             'delta': round(ocal - oraw, 4)})
    # 4) 汇总
    for d in CACHE:
        gr = np.array(per_det[d]['gate_raw']); gc = np.array(per_det[d]['gate_cal'])
        rho, _ = spearmanr(gr, gc)
        oraw = np.array(per_det[d]['oraw']); ocal = np.array(per_det[d]['ocal'])
        res['per_det'][d] = {'n': len(gr), 'spearman_gate': round(float(rho), 4),
                             'mean_abs_gate_diff': round(float(np.mean(np.abs(gr - gc))), 4),
                             'oracle_raw_mean': round(float(oraw.mean()), 4),
                             'oracle_cal_mean': round(float(ocal.mean()), 4),
                             'delta_oracle_mean': round(float((ocal - oraw).mean()), 4)}
        print(f'[{d}] spearman(gate)={rho:+.4f} |Δgate|={np.mean(np.abs(gr-gc)):.4f} | '
              f'oracle raw={oraw.mean():.4f} cal={ocal.mean():.4f} Δ={ocal.mean()-oraw.mean():+.4f}')
    all_gr = np.concatenate([per_det[d]['gate_raw'] for d in CACHE])
    all_gc = np.concatenate([per_det[d]['gate_cal'] for d in CACHE])
    all_or = np.concatenate([per_det[d]['oraw'] for d in CACHE])
    all_oc = np.concatenate([per_det[d]['ocal'] for d in CACHE])
    res['pooled'] = {'n': len(all_gr), 'spearman_gate': round(float(spearmanr(all_gr, all_gc)[0]), 4),
                     'mean_abs_gate_diff': round(float(np.mean(np.abs(all_gr - all_gc))), 4),
                     'oracle_raw_mean': round(float(all_or.mean()), 4),
                     'oracle_cal_mean': round(float(all_oc.mean()), 4),
                     'delta_oracle_mean': round(float((all_oc - all_or).mean()), 4)}
    print(f'[POOLED] spearman(gate)={res["pooled"]["spearman_gate"]:+.4f} '
          f'Δoracle={res["pooled"]["delta_oracle_mean"]:+.4f}')
    json.dump(res, open(OUT, 'w'), indent=2, ensure_ascii=False)

    # 5) 图: γ(conf) before/after 曲线 (per detector) + gate_raw vs gate_cal 散点
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    conf_grid = np.linspace(0.05, 0.95, 50)
    for ax, d in zip(axes, CACHE):
        gr = 1 - np.tanh(SCALE * (conf_grid - c0_raw))
        gc = 1 - np.tanh(SCALE * (conf_grid - c0_cal))
        ax.plot(conf_grid, gr, '-', label=f'raw (c0={c0_raw:.3f})')
        ax.plot(conf_grid, gc, '--', label=f'calibrated (c0={c0_cal:.3f})')
        ax.set_title(f'{d}  gate(conf)  [T={Tfit[d]:.2f}]')
        ax.set_xlabel('target-domain conf'); ax.set_ylabel('gate(conf)'); ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG, dpi=130); plt.close(fig)
    print(f'saved {OUT}  and  {FIG}')


if __name__ == '__main__':
    main()
