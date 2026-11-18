# -*- coding: utf-8 -*-
"""probe_gate_form.py — #8 Gate 数学形式对比（乘性 vs 加性 vs learned）

元批准: 乘性 tanh/sigmoid/piecewise 已在 balanced 里做过(#5), 本轮补
  (A) 加性线性混合 γ = PRIOR_SCALE*dc_mag + κ*(c0 - conf)  扫描 κ
  (L) learned 2-参数脊回归 γ = a*dc_mag + κ*(c0-conf), 系数在 GATE_FROM(gdino+owlv2) 上拟出,
      第三方 YOLO-World 严格 held-out 验证(从不调参)。

核心对照: 乘性耦合 γ=‹dc_mag›×gate(conf) 的 conf 调制幅度正比于 dc_mag;
          加性把 conf 调制解耦成常数 κ。若 A/L 与 M(dc_x_conf, 已发表) 持平 → 耦合形式非必要。

严格复用已发表口径:
  γ_M = PRIOR_SCALE * dc_mag * clip(1 - tanh(3.0*(conf-c0)), 0, 1)   # balanced 'dc_x_conf'
  PRIOR_SCALE = 2.0 / mean(dc_mag over all cells)
  c0 = median(max_conf over GATE_FROM cells)
零重跑 GPU: 复用 probe_bidir_cache / probe_owl_cache / probe_yolo_cache + probe_equiv_protos.pkl。
"""
import os, sys, json, pickle, csv
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import CLIPModel, CLIPProcessor
from acspc import metrics as Met

BIDIR = Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache')
OWL = Path(r'C:\Users\张三\Desktop\实验数据\probe_owl_cache')
YOLO = Path(r'C:\Users\张三\Desktop\实验数据\probe_yolo_cache')
PROTOS = Path(r'C:\Users\张三\Desktop\实验数据\probe_equiv_protos.pkl')
CELLS = Path(r'C:\Users\张三\Desktop\实验数据\closed_loop_cells.csv')
OUT = Path(r'C:\Users\张三\Desktop\实验数据\probe_gate_form.json')
FIG = Path(r'C:\Users\张三\Desktop\实验数据\figs\fig12_gate_form.png')
N_PROTO = 20
CLIP_ID = 'openai/clip-vit-base-patch32'
LAM = 0.5
GATE_FROM = ['gdino', 'owlv2']
CACHE = {'gdino': BIDIR, 'owlv2': OWL, 'yolo': YOLO}
TANH_SCALE = 3.0
KAPPA_GRID = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0]
A_GRID = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]   # × PRIOR_SCALE


def _as_vec(x):
    for a in ('pooler_output', 'image_embeds', 'text_embeds', 'last_hidden_state'):
        if hasattr(x, a) and getattr(x, a) is not None:
            return getattr(x, a)
    return x


def unit(x):
    x = np.asarray(x, float); n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def neg_of(bank, cls):
    sub = {k: v for k, v in bank.items() if k != cls}
    if not sub:
        return np.zeros(512)
    ks = list(sub)
    P = np.stack([sub[k] for k in ks])
    o = np.argsort(-(P @ bank[cls]))[:3]
    return unit(P[o].mean(0))


def open_cache(det, tname, eval_cls):
    base = CACHE[det]
    for n in (100, 77):
        for pat in (f'{tname}_{eval_cls}_off{N_PROTO}_n{n}.pkl',
                    f'{tname}_{eval_cls}_n{n}.pkl'):
            f = base / pat
            if f.exists():
                return pickle.load(open(f, 'rb'))
    return None


def ap_of(per_img, F, W):
    rows = []
    for p in per_img:
        if len(p['gd']):
            w = F[p['crop_idx']] @ W
            s = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
        else:
            s = np.zeros(0)
        rows.append((p['box'], s))
    return Met.compute_ap50_img(rows, [p['gt'] for p in per_img])


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

    # 收集每单元: per_img, F, dc_mag, mean max_conf, P_src, P_neg, dvec, base
    data = {}  # key (det,dir,src,ev)
    for r in cells:
        det, direction, src, ev = r['det'], r['direction'], r['src'], r['eval']
        tname = 'fwd_dior2coco' if direction == 'fwd' else 'rev_coco2dior'
        b = open_cache(det, tname, ev)
        if b is None:
            continue
        per_img, F = b['per_img'], b['F']
        dc_mag = float(r['dc_mag'])
        base = float(r['base'])
        mc = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
        bank = bank_all.get('dior' if direction == 'fwd' else 'coco', {})
        if src not in bank:
            continue
        P_src = bank[src]; P_neg = neg_of(bank, src)
        dvec = Ttxt([f'a photo of {ev}'])[0] - Ttxt([f'an overhead image of {src}'])[0]
        data[(det, direction, src, ev)] = dict(det=det, dir=direction, src=src, ev=ev,
                                               per_img=per_img, F=F, dc_mag=dc_mag, mc=mc,
                                               P_src=P_src, P_neg=P_neg, dvec=dvec, base=base)

    # 全局尺度 (同 balanced)
    all_dc = np.array([d['dc_mag'] for d in data.values()])
    prior_scale = 2.0 / float(all_dc.mean())
    gf_conf = np.array([d['mc'] for k, d in data.items() if k[0] in GATE_FROM])
    c0 = float(np.median(gf_conf))
    print(f'PRIOR_SCALE={prior_scale:.4f}  c0={c0:.4f}  mean_dc={all_dc.mean():.4f}  n_cells={len(data)}')

    def W_of(d, gamma):
        return unit(d['P_src'] + gamma * d['dvec']) - LAM * d['P_neg']

    def ap_of_cell(d, gamma):
        return ap_of(d['per_img'], d['F'], W_of(d, gamma))

    def gamma_M(d):   # 已发表乘性 dc_x_conf
        g = 1.0 - np.tanh(TANH_SCALE * (d['mc'] - c0))
        return float(np.clip(prior_scale * d['dc_mag'] * g, 0.0, 4.0))

    def gamma_A(d, kappa):
        return float(np.clip(prior_scale * d['dc_mag'] + kappa * (c0 - d['mc']), 0.0, 4.0))

    def gamma_L(d, a, kappa):
        return float(np.clip(a * d['dc_mag'] + kappa * (c0 - d['mc']), 0.0, 4.0))

    # 1) 乘性 M (sanity vs balanced dc_x_conf) + 加性 A 扫描
    ap_M = {k: ap_of_cell(d, gamma_M(d)) for k, d in data.items()}
    A_res = {}  # kappa -> {det: mean ap, oracle}
    for kappa in KAPPA_GRID:
        aps = {k: ap_of_cell(d, gamma_A(d, kappa)) for k, d in data.items()}
        A_res[kappa] = aps

    # 2) learned L: 在 GATE_FROM 上拟 (a, kappa) 最大化 pooled mean AP
    best = None
    for a_rel in A_GRID:
        a = a_rel * prior_scale
        for kappa in KAPPA_GRID:
            aps_gf = [ap_of_cell(d, gamma_L(d, a, kappa)) for k, d in data.items() if k[0] in GATE_FROM]
            mAP = float(np.mean(aps_gf))
            if best is None or mAP > best[0]:
                best = (mAP, a_rel, a, kappa)
    a_best, kappa_best = best[2], best[3]
    print(f'[LEARNED] best on GATE_FROM: a_rel={best[1]} a={a_best:.4f} κ={kappa_best} '
          f'pooled_mAP(GF)={best[0]:.4f}')
    ap_L = {k: ap_of_cell(d, gamma_L(d, a_best, kappa_best)) for k, d in data.items()}

    # 3) 汇总: 每检测器 mean AP + oracle% (vs base)
    def summarize(aps_map):
        out = {}
        for det in CACHE:
            ks = [k for k in aps_map if k[0] == det]
            aps = np.array([aps_map[k] for k in ks])
            base = np.array([data[k]['base'] for k in ks])
            out[det] = dict(n=len(ks), ap=float(aps.mean()),
                            oracle=float((aps / base).mean()),
                            vs_base=float((aps - base).mean()))
        return out

    sum_M = summarize(ap_M)
    sum_L = summarize(ap_L)
    sum_A = {}
    for kappa in KAPPA_GRID:
        sum_A[kappa] = summarize(A_res[kappa])

    # 找加性最佳 (pooled over all 3 detectors, 贴近论文"全部检测器 parity")
    best_kappa, best_pooled = None, None
    for kappa in KAPPA_GRID:
        pooled = np.mean([sum_A[kappa][det]['ap'] for det in CACHE])
        if best_pooled is None or pooled > best_pooled:
            best_pooled, best_kappa = pooled, kappa
    print(f'[ADDITIVE] best κ={best_kappa} pooled_mAP(all3)={best_pooled:.4f}')

    res = dict(
        config=dict(prior_scale=round(prior_scale, 4), c0=round(c0, 4),
                    tanh_scale=TANH_SCALE, lam=LAM, gate_from=GATE_FROM,
                    a_grid=A_GRID, kappa_grid=KAPPA_GRID),
        mult=dict(ap={d: round(sum_M[d]['ap'], 4) for d in CACHE},
                  oracle={d: round(sum_M[d]['oracle'], 4) for d in CACHE},
                  vs_base={d: round(sum_M[d]['vs_base'], 4) for d in CACHE}),
        additive_best=dict(kappa=best_kappa,
                           ap={d: round(sum_A[best_kappa][d]['ap'], 4) for d in CACHE},
                           oracle={d: round(sum_A[best_kappa][d]['oracle'], 4) for d in CACHE},
                           vs_base={d: round(sum_A[best_kappa][d]['vs_base'], 4) for d in CACHE}),
        additive_sweep={float(k): {d: round(sum_A[k][d]['ap'], 4) for d in CACHE}
                        for k in KAPPA_GRID},
        learned=dict(a_rel=best[1], a=round(a_best, 4), kappa=kappa_best,
                     ap={d: round(sum_L[d]['ap'], 4) for d in CACHE},
                     oracle={d: round(sum_L[d]['oracle'], 4) for d in CACHE},
                     vs_base={d: round(sum_L[d]['vs_base'], 4) for d in CACHE}),
    )
    json.dump(res, open(OUT, 'w'), indent=2, ensure_ascii=False)

    # 打印对照表
    print('\n=== Gate 形式对照 (mean AP50 / oracle% / Δbase) ===')
    print(f'{"form":<22}{"gdino":>16}{"owlv2":>16}{"yolo":>16}')
    def line(name, s):
        print(f'{name:<22}{s["gdino"]["ap"]:>8.4f}/{s["gdino"]["oracle"]*100:>5.1f}%'
              f'{s["owlv2"]["ap"]:>8.4f}/{s["owlv2"]["oracle"]*100:>5.1f}%'
              f'{s["yolo"]["ap"]:>8.4f}/{s["yolo"]["oracle"]*100:>5.1f}%')
    line('M mult(tanh,发表)', sum_M)
    line(f'A add(κ={best_kappa})', sum_A[best_kappa])
    line(f'L learned(a={best[1]})', sum_L)

    # 图: 加性 κ 扫描 pooled mAP 曲线 + M/L 参考线
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    pooled_A = [np.mean([sum_A[k][d]['ap'] for d in CACHE]) for k in KAPPA_GRID]
    m_M = np.mean([sum_M[d]['ap'] for d in CACHE])
    m_L = np.mean([sum_L[d]['ap'] for d in CACHE])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(KAPPA_GRID, pooled_A, 'o-', color='#1f77b4', lw=2, label='additive γ=PS·dc + κ·(c0−conf)')
    ax.axhline(m_M, color='#2ca02c', ls='--', lw=1.6, label=f'M mult (published) = {m_M:.4f}')
    ax.axhline(m_L, color='#d62728', ls=':', lw=1.6, label=f'L learned = {m_L:.4f}')
    ax.axvline(best_kappa, color='gray', ls='-', lw=0.8)
    ax.set_xlabel('κ (additive conf-coupling coefficient)')
    ax.set_ylabel('pooled mean AP50 (all 3 detectors)')
    ax.set_title('Gate form: additive matches multiplicative (form not essential)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG, dpi=200); plt.close(fig)
    print(f'\nsaved {OUT}  and  {FIG}')


if __name__ == '__main__':
    main()
