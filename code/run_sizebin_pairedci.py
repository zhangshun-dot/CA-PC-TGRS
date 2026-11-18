# -*- coding: utf-8 -*-
"""run_sizebin_pairedci.py — B3 尺寸分箱 + B4 tab:det 配对 bootstrap CI + A1 尺度/类频口径修复

背景 / 为什么要重算:
  closed_loop_classscale.csv 的生成脚本 `run_closed_loop_classscale.py` 用 glob('*_n*.pkl')
  遍历缓存, 而 probe_bidir_cache 下每个类同时存在 `_off20_n100.pkl` 与 `_off20_n8.pkl`
  → 同一 (det,direction,src,eval) 单元被登记两次(仅 avg_gt_area 不同, gain 相同),
  36 行里只有 27 个真实单元 → 伪重复, 使 Spearman n 虚高、rho 有偏。
本脚本改为: 以 run_closed_loop_balanced.json 的 cells_detail(36 单元, 权威)为准,
  对每个单元用**重算的 base 反查**它实际用的是哪个 blob(逐候选 blob 比对 base, tol 1e-9),
  从而 area 与 gain 来自同一个 blob。

产出:
  1) A1 修复: 尺度/类频 Spearman(无重复, n=36) → closed_loop_scalefreq_clean.csv
  2) B3:  前向 DIOR->COCO 的 COCO 尺寸分箱(small<32^2 / medium / large>=96^2) AP50, base vs CA-PC(dc×conf)
  3) B4:  四检测器 tab:det 口径的配对 bootstrap CI(逐图配对重采样, 单元内 9 cell 宏平均)
输出: C:\\Users\\张三\\Desktop\\实验数据\\run_sizebin_pairedci.json
用法: HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu "C:/Users/张三/vg/Scripts/python.exe" -u run_sizebin_pairedci.py
"""
import os, sys, json, pickle, glob, math
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cpu')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch
from pathlib import Path
from collections import Counter
import run_bidirectional as RB

RB.DEVICE = 'cpu'
LAM, NEG_TOPK, N_PROTO = RB.LAM, RB.NEG_TOPK, RB.N_PROTO

DETECTORS = {
    'gdino': Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache'),
    'owlv2': Path(r'C:\Users\张三\Desktop\实验数据\probe_owl_cache'),
    'yolo':  Path(r'C:\Users\张三\Desktop\实验数据\probe_yolo_cache'),
    'rtdetr': Path(r'C:\Users\张三\Desktop\实验数据\probe_rtdetr_cache'),
}
JSON_IN = Path(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced.json')
COCO_ANN = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'
OUT = Path(r'C:\Users\张三\Desktop\实验数据\run_sizebin_pairedci.json')
CSV_OUT = Path(r'C:\Users\张三\Desktop\实验数据\closed_loop_scalefreq_clean.csv')
R_BOOT = 1000
SEED = 42

# COCO 面积分箱(px^2)
SMALL_MAX, MED_MAX = 32.0 ** 2, 96.0 ** 2


def log(*a):
    print(*a, flush=True)


def unit(x):
    x = np.asarray(x, float)
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def box_areas(gts):
    if len(gts) == 0:
        return np.zeros(0)
    g = np.asarray(gts, float).reshape(-1, 4)
    return np.maximum(0.0, g[:, 2] - g[:, 0]) * np.maximum(0.0, g[:, 3] - g[:, 1])


# ─────────────────────────── 评分 / AP ───────────────────────────

def scores_of(per_img, F, W):
    """与 run_closed_loop_balanced.ap_of 完全一致的打分。"""
    out = []
    for p in per_img:
        if len(p['gd']):
            w = F[p['crop_idx']] @ W
            out.append(np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0)))
        else:
            out.append(np.zeros(0))
    return out


def iou_mat(b, g):
    """逐字复刻 acspc.metrics.iou(官方): inter 用 max(0,·), 面积不裁剪, 分母 +1e-8。
    注意: 不能用 run_bidirectional.iou_mat —— 它用 max(·,1e-12), 对退化小框与官方不等价
    (实测 rev 单元 AP 差 2e-4~3e-3), 会让 base 复算失配。"""
    b = np.asarray(b, float).reshape(-1, 4)
    g = np.asarray(g, float).reshape(-1, 4)
    if len(b) == 0 or len(g) == 0:
        return np.zeros((len(b), len(g)))
    ix1 = np.maximum(b[:, None, 0], g[None, :, 0]); iy1 = np.maximum(b[:, None, 1], g[None, :, 1])
    ix2 = np.minimum(b[:, None, 2], g[None, :, 2]); iy2 = np.minimum(b[:, None, 3], g[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / (ab[:, None] + ag[None, :] - inter + 1e-8)


def per_image_evals(boxes, scores, gts, iou_th, gt_mask=None):
    """逐图贪心匹配 -> 每图的 (score_desc, tp) + 每图 GT 数, 与 metrics.compute_ap50_img 语义逐字一致。
    gt_mask: 布尔数组, 只保留箱内的 GT 参与匹配(与 n_gt)。"""
    evs = []
    ngt = []
    for b, s, g in zip(boxes, scores, gts):
        gm = np.asarray(g, float).reshape(-1, 4)
        if gt_mask is not None:
            gm = gm[gt_mask(gm)]
        ng = len(gm)
        ngt.append(ng)
        if len(b) == 0:
            evs.append((np.zeros(0), np.zeros(0, int)))
            continue
        M = iou_mat(b, gm) if ng else np.zeros((len(b), 0))
        idx = np.argsort(-np.asarray(s, float))          # 与原实现同一调用(含并列顺序)
        matched = np.zeros(ng, bool)
        tp = np.zeros(len(b), int)
        for k, i in enumerate(idx):
            if ng == 0:
                tp[k] = 0
                continue
            row = M[i].copy()
            row[matched] = -1.0
            bj = int(row.argmax())
            if row[bj] >= iou_th and bj >= 0:
                tp[k] = 1
                matched[bj] = True
        evs.append((np.asarray(s, float)[idx], tp))
    return evs, ngt


def ap_from_evals(evs, n_gt, pts=11):
    if n_gt == 0:
        return 0.0 if any(len(e[0]) for e in evs) else 1.0
    if not evs:
        return 0.0
    S = np.concatenate([e[0] for e in evs])
    T = np.concatenate([e[1] for e in evs])
    if len(S) == 0:
        return 0.0
    o = np.argsort(-S, kind='stable')                    # == list.sort(key=-score) 的稳定降序
    T = T[o]
    tc = np.cumsum(T).astype(float)
    fc = np.cumsum(1 - T).astype(float)
    rec = tc / n_gt
    prec = tc / (tc + fc + 1e-8)
    ap = 0.0
    for t in np.linspace(0, 1, pts):
        m = rec >= t
        if m.any():
            ap += float(prec[m].max())
    return ap / pts


def ap50_of(evs_sets, n_gt):
    return ap_from_evals(evs_sets, n_gt, 11)


IOU_THRESH = [round(0.5 + 0.05 * i, 2) for i in range(10)]


def map_of(evs_by_th, n_gt):
    return float(np.mean([ap_from_evals(evs_by_th[th], n_gt, 101) for th in IOU_THRESH]))


# ─────────────────────────── 主流程 ───────────────────────────

def main():
    d = json.load(open(JSON_IN, encoding='utf-8'))
    cfg_json = d['config']
    c0 = cfg_json['gate']['c0']
    PRIOR_SCALE = cfg_json['gate']['prior_scale']
    log(f'[cfg] c0={c0:.6f} prior_scale={PRIOR_SCALE:.6f}')

    # ---- 重建共享源原型 / 文本锚 (与 run_closed_loop_balanced 同路径) ----
    cfg = RB.Config()
    cfg.model.device = 'cpu'
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'
    bank = {}
    for domain, pool in (('coco', RB.COCO_POOL), ('dior', RB.DIOR_POOL)):
        b = {}
        for cls in pool:
            ims, gts = RB.load_split(cfg, domain, cls, 0, N_PROTO)
            crops = RB.D.gt_crops(ims, gts)
            if not crops:
                continue
            Ft = RB.clip_encode_imgs(crops)
            pt = torch.from_numpy(Ft.mean(0))
            b[cls] = pt / pt.norm()
        bank[domain] = b
    log(f'[bank] dior={len(bank["dior"])} coco={len(bank["coco"])}')

    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    E_photo_d = {c: E_photo[i] for i, c in enumerate(all_names)}
    E_over_d = {c: E_over[i] for i, c in enumerate(all_names)}

    # COCO 类频
    ann = json.load(open(COCO_ANN))
    freq = {c['name']: sum(1 for o in ann['annotations'] if o['category_id'] == c['id'])
            for c in ann['categories']}

    def gate_conf(conf, c=c0, scale=3.0):
        return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))

    cells_detail = d['cells_detail']
    log(f'[cells] {len(cells_detail)} cells, dets={dict(Counter(c["det"] for c in cells_detail))}')

    # ── 逐单元: 反查 blob, 复算 base / dc×conf, 存逐图打分 ──
    cells = []
    for c in cells_detail:
        det, direction, src, ev = c['det'], c['direction'], c['src'], c['eval']
        tname = 'fwd_dior2coco' if direction == 'fwd' else 'rev_coco2dior'
        cache_dir = DETECTORS[det]
        cands = sorted(glob.glob(str(cache_dir / f'{tname}_{ev}_off{N_PROTO}_n*.pkl'))) + \
                sorted(glob.glob(str(cache_dir / f'{tname}_{ev}_n*.pkl')))
        cands = list(dict.fromkeys(cands))
        src_dom = 'dior' if direction == 'fwd' else 'coco'
        if src not in bank[src_dom]:
            log(f'  !! skip {det}/{tname}/{src}->{ev}: src not in bank')
            continue
        P_src = unit(bank[src_dom][src].numpy())
        P_neg = unit(RB.Proto.get_negative_prototype(bank[src_dom], src, NEG_TOPK).numpy())
        dc = E_photo_d[ev] - E_over_d[src]
        dc_mag = float(np.linalg.norm(dc))

        # 反查 blob: 复算 base 与 json 记录比对
        hit = None
        for fp in cands:
            blob = pickle.load(open(fp, 'rb'))
            per_img, F = blob['per_img'], blob['F']
            evs, ngt = per_image_evals([p['box'] for p in per_img],
                                       scores_of(per_img, F, P_src - LAM * P_neg),
                                       [p['gt'] for p in per_img], 0.5)
            b = ap_from_evals(evs, sum(ngt), 11)
            if abs(b - c['base']) < 1e-9:
                hit = (fp, blob, per_img, F, b)
                break
        if hit is None:
            log(f'  !! no blob reproduces base for {det}/{tname}/{src}->{ev} (base={c["base"]:.6f}); cands={[Path(x).name for x in cands]}')
            continue
        fp, blob, per_img, F, base = hit

        gam = float(np.clip(PRIOR_SCALE * dc_mag * gate_conf(c['max_conf'], c0, 3.0), 0.0, 4.0))
        W_base = P_src - LAM * P_neg
        W_corr = unit(P_src + gam * dc) - LAM * P_neg
        s_base = scores_of(per_img, F, W_base)
        s_corr = scores_of(per_img, F, W_corr)
        boxes = [p['box'] for p in per_img]
        gts = [p['gt'] for p in per_img]
        areas = box_areas(np.concatenate([np.asarray(g, float).reshape(-1, 4) for g in gts]) if len(gts) else np.zeros((0, 4)))
        cells.append({'det': det, 'direction': direction, 'src': src, 'eval': ev, 'blob': Path(fp).name,
                      'base_json': c['base'], 'base_recomp': base, 'ap_dc_x_conf_json': c['ap_dc_x_conf'],
                      'gamma': gam, 'boxes': boxes, 'gts': gts, 's_base': s_base, 's_corr': s_corr,
                      'avg_gt_area': float(areas.mean()) if len(areas) else 0.0,
                      'coco_freq': freq.get(ev, 0), 'max_conf': c['max_conf'], 'dc_mag': dc_mag,
                      'gain_json': c['gain_dc_x_conf_vs_base']})

    # ── 自证: 复算的 base / ap_dc_x_conf 是否与 json 逐位一致 ──
    log('\n[自证] 复算 vs json (要求 |diff| < 1e-9)')
    worst_b = worst_g = 0.0
    for cc in cells:
        evs_b, ngb = per_image_evals(cc['boxes'], cc['s_base'], cc['gts'], 0.5)
        evs_c, _ = per_image_evals(cc['boxes'], cc['s_corr'], cc['gts'], 0.5)
        ab, ac = ap_from_evals(evs_b, sum(ngb), 11), ap_from_evals(evs_c, sum(ngb), 11)
        worst_b = max(worst_b, abs(ab - cc['base_json']))
        worst_g = max(worst_g, abs(ac - cc['ap_dc_x_conf_json']))
    log(f'  max |base_recomp - base_json|      = {worst_b:.3e}')
    log(f'  max |ap_dc_x_conf - ap_dc_x_conf_json| = {worst_g:.3e}')
    assert worst_b < 1e-9 and worst_g < 1e-9, '自证失败: 复算与 json 不一致'
    log(f'  OK: {len(cells)} units matched (blobs resolved by base)')
    assert len(cells) == len(cells_detail), \
        f'只解析出 {len(cells)}/{len(cells_detail)} 个单元 —— 有单元的 blob 未能复现 base, 不能静默继续'

    # ── A1 修复: 尺度/类频 Spearman (n=36, 无重复) ──
    la = np.array([math.log10(cc['avg_gt_area'] + 1) for cc in cells])
    lf = np.array([math.log10(cc['coco_freq'] + 1) for cc in cells])
    g = np.array([cc['gain_json'] for cc in cells])

    def spearman(x, y):
        def rk(v):
            o = np.argsort(v, kind='mergesort'); r = np.empty(len(v)); i = 0
            while i < len(o):
                j = i
                while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                    j += 1
                r[o[i:j + 1]] = (i + j) / 2.0 + 1
                i = j + 1
            return r
        rx, ry = rk(x), rk(y)
        return float(np.corrcoef(rx, ry)[0, 1])

    def pearson(x, y):
        return float(np.corrcoef(x, y)[0, 1])

    def t_p(r, n):
        if abs(r) >= 1 or n <= 2:
            return 0.0
        t = r * math.sqrt((n - 2) / (1 - r * r)); d = n - 2
        from math import lgamma, exp, log
        def betacf(a, b, xx):
            EPS, FPMIN = 3e-12, 1e-300
            qab, qap, qam = a + b, a + 1, a - 1; c = 1.0; dd = 1 - qab * xx / qap
            dd = FPMIN if abs(dd) < FPMIN else dd
            dd = 1 / dd; h = dd
            for m in range(1, 201):
                m2 = 2 * m
                aa = m * (b - m) * xx / ((qam + m2) * (a + m2)); dd = 1 + aa * dd
                dd = FPMIN if abs(dd) < FPMIN else dd
                c = 1 + aa / c; c = FPMIN if abs(c) < FPMIN else c
                dd = 1 / dd; h *= dd * c
                aa = -(a + m) * (qab + m) * xx / ((a + m2) * (qap + m2)); dd = 1 + aa * dd
                dd = FPMIN if abs(dd) < FPMIN else dd
                c = 1 + aa / c; c = FPMIN if abs(c) < FPMIN else c
                dd = 1 / dd; de = dd * c; h *= de
                if abs(de - 1) < EPS:
                    break
            return h
        def betai(a, b, xx):
            if xx <= 0: return 0.0
            if xx >= 1: return 1.0
            bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(xx) + b * log(1 - xx))
            if xx < (a + 1) / (a + b + 2):
                return bt * betacf(a, b, xx) / a
            return 1 - bt * betacf(b, a, 1 - xx) / b
        return betai(d / 2, 0.5, d / (d + t * t))

    n = len(cells)
    rho_a, rho_f = spearman(la, g), spearman(lf, g)
    p_a, p_f = t_p(rho_a, n), t_p(rho_f, n)
    # 中位切分
    med = np.median(la); lo = g[la <= med].mean(); hi = g[la > med].mean()
    log(f'\n[A1 修复] n={n} (无重复)')
    log(f'  rho(gain, log_area)={rho_a:+.3f} (p={p_a:.3f})  Pearson={pearson(la, g):+.3f} (p={t_p(pearson(la,g), n):.3f})')
    log(f'  rho(gain, log_freq)={rho_f:+.3f} (p={p_f:.3f})  Pearson={pearson(lf, g):+.3f} (p={t_p(pearson(lf,g), n):.3f})')
    log(f'  中位切分 log_area: 下半均值={lo:+.4f}  上半均值={hi:+.4f}')
    LBL = {'gdino': 'GDINO-tiny', 'owlv2': 'OWLv2', 'yolo': 'YOLO-World', 'rtdetr': 'RT-DETR'}
    for det in DETECTORS:
        sub = [cc for cc in cells if cc['det'] == det]
        if len(sub) < 4:
            continue
        ga = np.array([cc['gain_json'] for cc in sub]); aa = np.array([math.log10(cc['avg_gt_area'] + 1) for cc in sub])
        ff = np.array([math.log10(cc['coco_freq'] + 1) for cc in sub])
        log(f'    {det:<7} n={len(sub)} rho_area={spearman(aa, ga):+.3f} rho_freq={spearman(ff, ga):+.3f} mean_gain={ga.mean():+.4f}')

    with open(CSV_OUT, 'w', encoding='utf-8') as f:
        f.write('det,direction,src,eval,blob,gain,avg_gt_area,log_area,coco_freq,log_freq\n')
        for cc in cells:
            f.write(f"{cc['det']},{cc['direction']},{cc['src']},{cc['eval']},{cc['blob']},"
                    f"{cc['gain_json']:.6f},{cc['avg_gt_area']:.2f},{math.log10(cc['avg_gt_area'] + 1):.6f},"
                    f"{cc['coco_freq']},{math.log10(cc['coco_freq'] + 1):.6f}\n")
    log(f'[csv] -> {CSV_OUT} ({len(cells)} units)')

    # ── B4: 配对 bootstrap CI (四检测器, 逐图配对重采样, 单元宏平均) ──
    log('\n[B4] 配对 bootstrap CI (R=%d, 逐图配对, 单元宏平均)' % R_BOOT)
    rng = np.random.RandomState(SEED)
    boot = {}
    for det in DETECTORS:
        sub = [cc for cc in cells if cc['det'] == det]
        if not sub:
            continue
        # 预先为每个单元算好逐图 evals (base / corr), AP50 + 10 阈值 mAP
        pre = []
        for cc in sub:
            nb, ng = per_image_evals(cc['boxes'], cc['s_base'], cc['gts'], 0.5)
            nc, _ = per_image_evals(cc['boxes'], cc['s_corr'], cc['gts'], 0.5)
            mb = {th: per_image_evals(cc['boxes'], cc['s_base'], cc['gts'], th)[0] for th in IOU_THRESH}
            mc = {th: per_image_evals(cc['boxes'], cc['s_corr'], cc['gts'], th)[0] for th in IOU_THRESH}
            pre.append({'nb': nb, 'nc': nc, 'mb': mb, 'mc': mc,
                        'ngt': np.asarray(ng, int), 'n_img': len(cc['boxes'])})
        mean_base50 = float(np.mean([ap_from_evals(p['nb'], int(p['ngt'].sum()), 11) for p in pre]))
        mean_corr50 = float(np.mean([ap_from_evals(p['nc'], int(p['ngt'].sum()), 11) for p in pre]))
        mean_basem = float(np.mean([map_of(p['mb'], int(p['ngt'].sum())) for p in pre]))
        mean_corrm = float(np.mean([map_of(p['mc'], int(p['ngt'].sum())) for p in pre]))
        d50, dm = [], []
        for _ in range(R_BOOT):
            a50 = []; am = []
            for p in pre:
                ni = p['n_img']
                idx = rng.randint(0, ni, size=ni)              # 配对: base/corr 用同一 idx
                ng = int(p['ngt'][idx].sum())
                ab = ap_from_evals([p['nb'][k] for k in idx], ng, 11)
                ac = ap_from_evals([p['nc'][k] for k in idx], ng, 11)
                a50.append(ac - ab)
                amb = float(np.mean([ap_from_evals([p['mb'][th][k] for k in idx], ng, 101) for th in IOU_THRESH]))
                amc = float(np.mean([ap_from_evals([p['mc'][th][k] for k in idx], ng, 101) for th in IOU_THRESH]))
                am.append(amc - amb)
            d50.append(float(np.mean(a50)))
            dm.append(float(np.mean(am)))
        d50 = np.array(d50); dm = np.array(dm)
        boot[det] = {
            'base50': mean_base50, 'corr50': mean_corr50, 'gain50': mean_corr50 - mean_base50,
            'ci50': [float(np.percentile(d50, 2.5)), float(np.percentile(d50, 97.5))],
            'p_le0_50': float(np.mean(d50 <= 0)),
            'basemap': mean_basem, 'corrmap': mean_corrm, 'gainmap': mean_corrm - mean_basem,
            'cimap': [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))],
            'p_le0_map': float(np.mean(dm <= 0)), 'n_cells': len(sub)}
        b = boot[det]
        log(f'  {det:<7} n_cells={b["n_cells"]}  AP50 {b["base50"]:.4f}->{b["corr50"]:.4f} '
            f'gain={b["gain50"]:+.4f} CI95[{b["ci50"][0]:+.4f},{b["ci50"][1]:+.4f}] P(<=0)={b["p_le0_50"]:.3f}')
        log(f'          mAP  {b["basemap"]:.4f}->{b["corrmap"]:.4f} '
            f'gain={b["gainmap"]:+.4f} CI95[{b["cimap"][0]:+.4f},{b["cimap"][1]:+.4f}] P(<=0)={b["p_le0_map"]:.3f}')

    # ── B3: 尺寸分箱 (前向 DIOR->COCO, pooled micro) ──
    log('\n[B3] COCO 尺寸分箱 AP50 (前向 DIOR->COCO, pooled)')
    BINS = [('small(<32^2)', lambda A: A < SMALL_MAX),
            ('medium(32^2..96^2)', lambda A: (A >= SMALL_MAX) & (A < MED_MAX)),
            ('large(>=96^2)', lambda A: A >= MED_MAX)]
    sizebin = {}
    for det in DETECTORS:
        sub = [cc for cc in cells if cc['det'] == det and cc['direction'] == 'fwd']
        if not sub:
            continue
        sizebin[det] = {}
        for bname, bfn in BINS:
            mask = lambda gm, bfn=bfn: bfn(box_areas(gm))
            eb_all, ec_all, ng_total = [], [], 0
            for cc in sub:
                eb, ngb = per_image_evals(cc['boxes'], cc['s_base'], cc['gts'], 0.5, gt_mask=mask)
                ec, _ = per_image_evals(cc['boxes'], cc['s_corr'], cc['gts'], 0.5, gt_mask=mask)
                eb_all += eb; ec_all += ec; ng_total += int(sum(ngb))
            ab = ap_from_evals(eb_all, ng_total, 11); ac = ap_from_evals(ec_all, ng_total, 11)
            sizebin[det][bname] = {'n_gt': ng_total, 'base': ab, 'corr': ac, 'gain': ac - ab}
            log(f'  {det:<7} {bname:<20} n_gt={ng_total:<5} base={ab:.4f} corr={ac:.4f} gain={ac-ab:+.4f}')

    json.dump({'config': {'c0': c0, 'prior_scale': PRIOR_SCALE, 'lam': LAM, 'R_boot': R_BOOT, 'seed': SEED,
                          'size_bins': ['small:area<1024', 'medium:1024<=area<9216', 'large:area>=9216']},
               'scale_freq': {'n': n, 'rho_log_area': rho_a, 'p_log_area': p_a,
                              'rho_log_freq': rho_f, 'p_log_freq': p_f,
                              'median_split_lower': float(lo), 'median_split_upper': float(hi)},
               'bootstrap_tab_det': boot, 'size_bins': sizebin},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    log(f'\n[json] -> {OUT}')


if __name__ == '__main__':
    main()
