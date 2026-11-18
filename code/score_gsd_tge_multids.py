# -*- coding: utf-8 -*-
"""score_gsd_tge_multids.py — 独立卖点实验块:
   "CA-PC vs TGE on the real remote-sensing condition axis" (多数据集)

元 2026-10-14: 审稿人第5版要求"强化 CA-PC 相对 TGE 在真实遥感域迁移中的独特优势"。
原 score_gsd_tge.py 只在 DIOR(gdino/owlv2) 上做; 本脚本把同一【真实 GSD 观测条件轴】上的
CA-PC vs TGE 机制区分, 扩到【3 个 RS 目标数据集】DIOR / DOTA / NWPU, 并给出
【配对图像级 bootstrap CI on 每档 CA-PC−TGE margin】, 使"TGE 缺少观测条件项"从点差变成
带区间的独立证据。

臂 (与 score_gsd_tge.py 逐字一致; TGE = ExDet DeltaSpace Eq.1 v_n = v_b + (t_n - t_bar)):
  base(raw gd), ca_pc(gate), tge_nat(β=1), tge_gate(norm-match+同门控), tge_star(逐cell网格最优),
  cond_gate, vocab_gate (参考分解分量)

关键工程: AP 的逐图匹配(per-image greedy IoU)与重采样无关 → 每个 (cell, arm) 只算一次
逐图 (score, tp, n_gt, n_entry) 表; bootstrap 只需重放全局 PR 曲线(按重采样重数扩展 n_gt 与 entry)。
朴素实现 (76 cells × B=1000 × 7 arms × 0.12 s/AP) ≈ 10 h; 重放实现 ≈ 几分钟。
每个 (cell, arm) 都做身份校验: 重放AP(全重数=1) 必须 == RB.ap_of/ap_base 的官方值。

用法(CPU, 后台):
  HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu "C:/Users/张三/vg/Scripts/python.exe" -u score_gsd_tge_multids.py --B 1000
"""
import os, sys, json, pickle, time, argparse
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cpu')
os.environ.setdefault('AC_SPC_COCO_ANN', r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json')
os.environ.setdefault('AC_SPC_COCO_IMG', r'C:\Users\张三\modeling\coco_data\val2017')
os.environ.setdefault('AC_SPC_DIOR_IMG', r'C:\Users\张三\modeling\dior_data\eval_v2_images')
os.environ.setdefault('AC_SPC_DIOR_PARQUET', r'C:\Users\张三\modeling\dior_data\dior_combined.parquet')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch
import run_bidirectional as RB
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

RB.DEVICE = 'cpu'
LAM = RB.LAM; NEG_TOPK = RB.NEG_TOPK; N_PROTO = RB.N_PROTO
BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
GSD_LEVELS = [1.0, 0.5, 0.25, 0.125]
EXTRA_NAMES = ['plane', 'small-vehicle', 'large-vehicle']
ARMS = ['ca_pc', 'tge_nat', 'tge_star', 'tge_gate', 'cond', 'vocab']

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
DATASETS = {
    'DIOR': {'dirs': {'gdino': DATA / 'probe_bidir_gsd_cache',
                      'owlv2': DATA / 'probe_bidir_gsd_owl_cache'},
             'prefix': 'rev_coco2dior',
             'units': [('vehicle', 'vehicle', 'car'), ('vehicle', 'vehicle', 'truck'),
                       ('vehicle', 'vehicle', 'bus'), ('vehicle', 'vehicle', 'motorcycle'),
                       ('ship', 'ship', 'boat'), ('airplane', 'airplane', 'airplane')]},
    'DOTA': {'dirs': {'gdino': DATA / 'probe_bidir_gsd_dota_cache'},
             'prefix': 'rev_coco2dota',
             'units': [('plane', 'plane', 'airplane'), ('ship', 'ship', 'boat'),
                       ('small-vehicle', 'small-vehicle', 'car'), ('large-vehicle', 'large-vehicle', 'truck')]},
    'NWPU': {'dirs': {'gdino': DATA / 'probe_bidir_gsd_nwpu_cache'},
             'prefix': 'rev_coco2nwpu',
             'units': [('1', 'airplane', 'airplane'), ('2', 'ship', 'boat'), ('10', 'vehicle', 'car')]},
}
OUT_JSON = DATA / 'score_gsd_tge_multids.json'
OUT_MD = DATA / 'score_gsd_tge_multids.md'


def unit(x):
    x = np.asarray(x, float); n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def pin_blob(d, prefix, key, gsd):
    """优先精确 n100, 避免 _n20/_n100 并存的 glob 不确定 (2026-10-14 教训)。"""
    for pat in (f'{prefix}_{key}_off20_n100_gsd{gsd}.pkl',
                f'{prefix}_{key}_off20_n*_gsd{gsd}.pkl'):
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    return None


def per_image_tp(per_img, F, W=None, use_raw=False):
    """逐图 (score, tp) 表 + 每图 n_gt / n_entry。逐字复刻 acspc.metrics.compute_ap50_img:
    每图内部按 score 降序贪心匹配, 取未匹配GT中 IoU 最大者, >=0.5 记 TP; IoU 用官方定义
    (分母 +1e-8, 不裁剪框)。返回 (sc, tp, ngt_per_img, ne_per_img)。"""
    sc_parts, tp_parts, ngt_parts, ne_parts = [], [], [], []
    for p in per_img:
        b = p['box']; g = p['gt']
        ngt_parts.append(len(g)); ne_parts.append(len(b))
        if use_raw:
            s = np.asarray(p['gd'], float)
        elif len(b):
            w = F[p['crop_idx']] @ W
            s = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
        else:
            s = np.zeros(0)
        nb = len(b)
        if nb == 0:
            sc_parts.append(np.zeros(0)); tp_parts.append(np.zeros(0, int)); continue
        if len(g) == 0:
            sc_parts.append(np.asarray(s, float)); tp_parts.append(np.zeros(nb, int)); continue
        x1 = np.maximum(b[:, 0:1], g[None, :, 0]); y1 = np.maximum(b[:, 1:2], g[None, :, 1])
        x2 = np.minimum(b[:, 2:3], g[None, :, 2]); y2 = np.minimum(b[:, 3:4], g[None, :, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        a1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        a2 = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
        iou_m = inter / (a1[:, None] + a2[None, :] - inter + 1e-8)
        order = np.argsort(-np.asarray(s, float), kind='stable')
        matched = np.zeros(len(g), bool)
        tp = np.zeros(nb, int)
        for i in order:
            row = np.where(matched, -1.0, iou_m[i])
            j = int(np.argmax(row))
            if row[j] >= 0.5:
                tp[i] = 1; matched[j] = True
        sc_parts.append(np.asarray(s, float)); tp_parts.append(tp)
    sc = np.concatenate(sc_parts) if sc_parts else np.zeros(0)
    tp = np.concatenate(tp_parts) if tp_parts else np.zeros(0, int)
    return sc, tp, np.asarray(ngt_parts, int), np.asarray(ne_parts, int)


def ap50_replay(tab, mult=None):
    """按图像重数 mult(n_img,) 重放 11-point AP50; mult=None 即原集(身份校验)。"""
    sc, tp, ngt, ne = tab
    if mult is None:
        s, t = sc, tp
        n_gt = int(ngt.sum())
    else:
        mpe = np.repeat(mult, ne)
        idx = np.repeat(np.arange(len(sc)), mpe)
        s, t = sc[idx], tp[idx]
        n_gt = int((ngt * mult).sum())
    if n_gt == 0:
        return 0.0 if len(s) else 1.0
    if len(s) == 0:
        return 0.0
    order = np.argsort(-s, kind='stable')
    t = t[order]
    tc = np.cumsum(t); fc = np.cumsum(1 - t)
    rec = tc / n_gt
    prec = tc / (tc + fc + 1e-8)
    ap = 0.0
    for th in np.linspace(0, 1, 11):
        m = rec >= th
        if m.any():
            ap += float(prec[m].max())
    return ap / 11.0


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--B', type=int, default=1000)
    ap_.add_argument('--datasets', nargs='+', default=list(DATASETS))
    ap_.add_argument('--detectors', nargs='+', default=['gdino', 'owlv2'])
    ap_.add_argument('--max-cells', type=int, default=0, help='>0 只做每(ds,det)前 N 个 cell 试跑')
    args = ap_.parse_args()
    t0 = time.time()

    cfg = RB.Config(); cfg.model.device = 'cpu'
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    bank_coco = {}
    for cls in RB.COCO_POOL:
        ims, gts = RB.load_split(cfg, 'coco', cls, 0, N_PROTO)
        crops = RB.D.gt_crops(ims, gts)
        if not crops:
            continue
        Ft = RB.clip_encode_imgs(crops)
        pt = torch.from_numpy(Ft.mean(0)); bank_coco[cls] = pt / pt.norm()

    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL + EXTRA_NAMES))
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    idx = {c: i for i, c in enumerate(all_names)}
    tb_idx = [idx[c] for c in sorted(set(RB.COCO_POOL + RB.DIOR_POOL))]
    t_bar = unit(E_photo[tb_idx].mean(0))     # 与 score_gsd_tge.py 同词表 → DIOR 可比

    gcfg = json.load(open(str(DATA / 'run_closed_loop_balanced.json')))['config']['gate']
    PRIOR_SCALE, c0 = gcfg['prior_scale'], gcfg['c0']

    def gate_conf(conf, c=c0, scale=3.0):
        return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))

    def ci(x):
        return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]

    out = {'config': {'datasets': args.datasets, 'gsd_levels': GSD_LEVELS, 'lam': LAM,
                      'beta_grid': BETA_GRID, 'B_boot': args.B, 'c0': c0,
                      'prior_scale': PRIOR_SCALE, 'n_proto': N_PROTO, 'zero_target_gt': True,
                      'detector_tiny': 'grounding-dino-tiny',
                      'tge_ref': 'ExDet DeltaSpace Eq.1 v_n = v_b + (t_n - t_bar), faithful',
                      'arms': ARMS, 'bootstrap_unit': 'paired target image (same resample per arm)',
                      'ap50': '11-point per-image matching (acspc.metrics)'},
           'identity_check': {}, 'per_ds': {}}
    max_dev = 0.0

    for ds in args.datasets:
        spec = DATASETS[ds]
        dsp = {}
        for det in args.detectors:
            if det not in spec['dirs']:
                continue
            d = spec['dirs'][det]
            if not d.exists():
                print(f'  !! 缺目录 {d}', flush=True); continue
            cells = []
            for gsd in GSD_LEVELS:
                for key_idx, ev, src in spec['units']:
                    if args.max_cells and len(cells) >= args.max_cells:
                        break
                    hp = pin_blob(d, spec['prefix'], key_idx, gsd)
                    if hp is None or src not in bank_coco:
                        continue
                    blob = pickle.load(open(hp, 'rb'))
                    per_img, F = blob['per_img'], blob['F']
                    P_src = bank_coco[src]
                    P_neg = unit(RB.Proto.get_negative_prototype(bank_coco, src, NEG_TOPK))
                    dc = E_photo[idx[ev]] - E_over[idx[src]]
                    mag = float(np.linalg.norm(dc))
                    d_tge = E_photo[idx[ev]] - t_bar
                    max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
                    gamma = float(np.clip(PRIOR_SCALE * mag * gate_conf(max_conf), 0.0, 4.0))

                    Ws = {
                        'ca_pc':    unit(P_src + gamma * dc) - LAM * P_neg,
                        'tge_nat':  unit(P_src + 1.0 * d_tge) - LAM * P_neg,
                        'tge_gate': unit(P_src + gamma * unit(d_tge) * mag) - LAM * P_neg,
                        'cond':     unit(P_src + gamma * unit(E_photo[idx[ev]] - E_over[idx[ev]]) * mag) - LAM * P_neg,
                        'vocab':    unit(P_src + gamma * unit(E_over[idx[ev]] - E_over[idx[src]]) * mag) - LAM * P_neg,
                    }
                    cur = [(float(RB.ap_of(per_img, F, unit(P_src + bb * d_tge) - LAM * P_neg)), bb)
                           for bb in BETA_GRID]
                    best_ap, best_b = max(cur, key=lambda x: x[0])
                    Ws['tge_star'] = unit(P_src + best_b * d_tge) - LAM * P_neg

                    base_ap = float(RB.ap_base(per_img, F))
                    tabs, gains = {}, {}
                    for arm in ARMS:
                        off = float(RB.ap_of(per_img, F, Ws[arm]))
                        tab = per_image_tp(per_img, F, Ws[arm])
                        rep = ap50_replay(tab)
                        dev = abs(rep - off)
                        max_dev = max(max_dev, dev)
                        if dev > 1e-9:
                            raise SystemExit(f'!! 身份校验失败 {ds}/{det}/gsd{gsd}/{key_idx}/{arm}: '
                                             f'off={off:.12f} rep={rep:.12f} dev={dev:.2e}')
                        tabs[arm] = tab; gains[arm] = off - base_ap
                    tabb = per_image_tp(per_img, F, use_raw=True)
                    if abs(ap50_replay(tabb) - base_ap) > 1e-9:
                        raise SystemExit(f'!! base 身份校验失败 {ds}/{det}/gsd{gsd}/{key_idx}')
                    tabs['base'] = tabb

                    cells.append({'ds': ds, 'det': det, 'gsd': gsd, 'key': key_idx, 'eval': ev, 'src': src,
                                  'k': 1.0 / gsd, 'n_img': len(per_img), 'base': round(base_ap, 4),
                                  'gamma': round(gamma, 4), 'mag': round(mag, 4), 'tge_star_beta': best_b,
                                  **{a + '_g': round(gains[a], 4) for a in ARMS}, '_tabs': tabs})
            if not cells:
                continue
            a_mean = lambda key: float(np.mean([c[key] for c in cells]))
            per_gsd = {}
            for g in GSD_LEVELS:
                cs = [c for c in cells if c['gsd'] == g]
                if cs:
                    am = lambda key, cs=cs: float(np.mean([c[key] for c in cs]))
                    per_gsd[str(g)] = {'k': 1.0 / g, 'n': len(cs), 'base': round(am('base'), 4),
                                       **{a + '_g': round(am(a + '_g'), 4) for a in ARMS}}
            mean_all = {a + '_g': round(a_mean(a + '_g'), 4) for a in ARMS}
            mean_all['base'] = round(a_mean('base'), 4)
            margin_nat = round(a_mean('ca_pc_g') - a_mean('tge_nat_g'), 4)
            margin_star = round(a_mean('ca_pc_g') - a_mean('tge_star_g'), 4)
            margin_cv = round(a_mean('cond_g') - a_mean('vocab_g'), 4)

            rng = np.random.default_rng(0)
            boots = {k: np.empty(args.B) for k in
                     ('ca', 'tn', 'ts', 'm_nat', 'm_star', 'm_cv')}
            for bi in range(args.B):
                aps = {a: np.empty(len(cells)) for a in ARMS + ['base']}
                for c_i, c2 in enumerate(cells):
                    n_img = c2['n_img']
                    cnt = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img)
                    for a in ARMS + ['base']:
                        aps[a][c_i] = ap50_replay(c2['_tabs'][a], cnt)
                ca, tn, ts = aps['ca_pc'].mean(), aps['tge_nat'].mean(), aps['tge_star'].mean()
                base = aps['base'].mean()
                boots['ca'][bi] = ca - base; boots['tn'][bi] = tn - base; boots['ts'][bi] = ts - base
                boots['m_nat'][bi] = ca - tn; boots['m_star'][bi] = ca - ts
                boots['m_cv'][bi] = aps['cond'].mean() - aps['vocab'].mean()
                if (bi + 1) % 250 == 0:
                    print(f'    [{ds}/{det}] boot {bi+1}/{args.B} ({time.time()-t0:.0f}s)', flush=True)

            lead = {str(g): bool(per_gsd[str(g)]['ca_pc_g'] > per_gsd[str(g)]['tge_nat_g'])
                    for g in GSD_LEVELS if str(g) in per_gsd}
            dsp[det] = {'per_gsd': per_gsd, 'mean_all': mean_all, 'n_cells': len(cells),
                        'margin_ca_minus_tge_nat': margin_nat, 'margin_ca_minus_tge_star': margin_star,
                        'margin_cond_minus_vocab': margin_cv,
                        'boot': {'B': args.B,
                                 'ca_pc_gain': [round(float(boots['ca'].mean()), 4)] + ci(boots['ca']),
                                 'tge_nat_gain': [round(float(boots['tn'].mean()), 4)] + ci(boots['tn']),
                                 'tge_star_gain': [round(float(boots['ts'].mean()), 4)] + ci(boots['ts']),
                                 'margin_nat': [margin_nat] + ci(boots['m_nat']),
                                 'margin_star': [margin_star] + ci(boots['m_star']),
                                 'margin_cond_vocab': [margin_cv] + ci(boots['m_cv'])},
                        'ca_leads_tge_every_gsd': lead,
                        'ci_margin_nat_excludes_zero': bool(ci(boots['m_nat'])[0] > 0),
                        'frac_cells_ca_gt_tge': round(float(np.mean([c['ca_pc_g'] > c['tge_nat_g'] for c in cells])), 4),
                        'cells': [{k: v for k, v in c2.items() if k != '_tabs'} for c2 in cells]}
            mn = ci(boots['m_nat'])
            print(f'  [{ds}/{det}] n={len(cells)} base={mean_all["base"]:.4f} CA-PC{mean_all["ca_pc_g"]:+.4f} '
                  f'TGE_nat{mean_all["tge_nat_g"]:+.4f} TGE*{mean_all["tge_star_g"]:+.4f} | '
                  f'margin_nat={margin_nat:+.4f} CI[{mn[0]:+.4f},{mn[1]:+.4f}] excl0={mn[0]>0} | '
                  f'cond−vocab={margin_cv:+.4f} | 逐档领先={all(lead.values()) if lead else None} '
                  f'({time.time()-t0:.0f}s)', flush=True)
        out['per_ds'][ds] = dsp

    out['identity_check'] = {'max_abs_dev_replay_vs_official': max_dev,
                             'criterion': 'ap50_replay(tab) == RB.ap_of/ap_base for every cell and arm'}
    json.dump(out, open(OUT_JSON, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    L = ['# CA-PC vs TGE on the real RS condition axis — 多数据集独立块\n',
         f'- 生成: {time.strftime("%Y-%m-%d %H:%M")}  zero target GT  B_boot={args.B}  '
         f'配对图像级 bootstrap  身份校验 max|Δ|={max_dev:.2e}\n',
         '| 数据集 | 检测器 | base | CA-PC | TGE_nat | TGE* | TGE_gate | cond | vocab | CA−TGE(nat) | CI95 | CA−TGE* | cond−vocab | 逐档领先 |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for ds, dd in out['per_ds'].items():
        for det, r in dd.items():
            m = r['mean_all']; b = r['boot']
            L.append(f"| {ds} | {det} | {m['base']:.4f} | {m['ca_pc_g']:+.4f} | {m['tge_nat_g']:+.4f} | "
                     f"{m['tge_star_g']:+.4f} | {m['tge_gate_g']:+.4f} | {m['cond_g']:+.4f} | {m['vocab_g']:+.4f} | "
                     f"{r['margin_ca_minus_tge_nat']:+.4f} | [{b['margin_nat'][1]:+.4f},{b['margin_nat'][2]:+.4f}] | "
                     f"{r['margin_ca_minus_tge_star']:+.4f} | {r['margin_cond_minus_vocab']:+.4f} | "
                     f"{all(r['ca_leads_tge_every_gsd'].values())} |")
    L.append('\n## 逐档 (per GSD)\n')
    for ds, dd in out['per_ds'].items():
        for det, r in dd.items():
            L.append(f"\n### {ds} / {det}  (n_cells={r['n_cells']}, 逐cell率 CA>TGE={r['frac_cells_ca_gt_tge']}, "
                     f"margin CI 不含0={r['ci_margin_nat_excludes_zero']})\n")
            L.append('| GSD | base | CA-PC | TGE_nat | TGE* | TGE_gate | cond | vocab |')
            L.append('|---|---|---|---|---|---|---|---|')
            for g in GSD_LEVELS:
                x = r['per_gsd'].get(str(g))
                if not x:
                    continue
                L.append(f"| {g} | {x['base']:.4f} | {x['ca_pc_g']:+.4f} | {x['tge_nat_g']:+.4f} | "
                         f"{x['tge_star_g']:+.4f} | {x['tge_gate_g']:+.4f} | {x['cond_g']:+.4f} | {x['vocab_g']:+.4f} |")
    open(OUT_MD, 'w', encoding='utf-8').write('\n'.join(L))
    print(f'\n→ {OUT_JSON} / {OUT_MD}  ({time.time()-t0:.0f}s)  身份校验 max|Δ|={max_dev:.2e}', flush=True)


if __name__ == '__main__':
    main()
