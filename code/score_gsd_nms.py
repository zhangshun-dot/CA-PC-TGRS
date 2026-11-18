# -*- coding: utf-8 -*-
"""score_gsd_nms.py — 独立卖点实验块的【标准后处理协议】补充臂

元 2026-10-14: 第5版评审风险③ —— 现在的 §4.22 用"固定候选框、不做 NMS"来隔离
原型重评分的因果效应, 这对消融是对的, 但审稿人还想看"真实部署协议下"结论是否仍成立
(NMS 是标准后处理)。本脚本在同一批 GSD 缓存上, 按【部署顺序】重算:

    s_arm = sqrt( q * clip(max(0, F·W_arm), 1e-6, 1.0) )      # 各臂最终分
      -> 逐图标准贪心 NMS (IoU 阈值 thr, 按 s_arm 降序)
      -> 逐图贪心 GT 匹配 + 全局 11-point AP50 (acspc.metrics 口径)

关键: NMS 与 GT 匹配都是【逐图】操作, 与图像重采样无关 → 沿用"逐图标签预计算 + 重放"
加速 (见 skill detection-dump-offline-reeval §12), 配对图像级 bootstrap 仍然可用。

臂与 score_gsd_tge_multids.py 逐字一致: base / ca_pc / tge_nat / tge_star / tge_gate /
cond / vocab。thr=1.0 等价于不做 NMS, 用于身份校验(须复现 multids 的 macro)。

用法(CPU, 后台):
  HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu "C:/Users/张三/vg/Scripts/python.exe" -u \
      score_gsd_nms.py --B 1000 --datasets DIOR --thrs 1.0 0.65 0.5
"""
import os, sys, json, time, argparse
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cpu')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import score_gsd_tge_multids as S   # 复用 pin_blob / DATASETS / unit / ap50_replay / 词表构建
import run_bidirectional as RB

RB.DEVICE = 'cpu'
LAM = RB.LAM; NEG_TOPK = RB.NEG_TOPK; N_PROTO = RB.N_PROTO
BETA_GRID = S.BETA_GRID; GSD_LEVELS = S.GSD_LEVELS; DATA = S.DATA
import torch


def nms_keep(b, s, thr):
    """逐图标准贪心 NMS: 按 s 降序, 与已保留框 IoU>thr 的后续框压掉。返回保留索引。"""
    n = len(b)
    if n == 0:
        return np.zeros(0, int)
    order = np.argsort(-np.asarray(s, float), kind='stable')
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    alive = np.ones(n, bool)
    keep = []
    for i in order:
        if not alive[i]:
            continue
        keep.append(i)
        bx = b[i]
        xx1 = np.maximum(bx[0], b[:, 0]); yy1 = np.maximum(bx[1], b[:, 1])
        xx2 = np.minimum(bx[2], b[:, 2]); yy2 = np.minimum(bx[3], b[:, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (area[i] + area - inter + 1e-8)
        alive = alive & ~(iou > thr)
        alive[i] = False
    return np.array(keep, int)


def per_image_tp_nms(per_img, F, W=None, thr=0.5, use_raw=False):
    """逐图 (score, tp) 表 + n_gt / n_entry。NMS 先按该臂最终分过滤, 再逐图贪心匹配
    (IoU 分母 +1e-8、不裁剪, 与 acspc.metrics.compute_ap50_img 同口径)。"""
    sc_parts, tp_parts, ngt_parts, ne_parts = [], [], [], []
    for p in per_img:
        b = p['box']; g = p['gt']
        ngt_parts.append(len(g))
        nb = len(b)
        if nb == 0:
            ne_parts.append(0); sc_parts.append(np.zeros(0)); tp_parts.append(np.zeros(0, int)); continue
        if use_raw:
            s = np.asarray(p['gd'], float)
        else:
            w = F[p['crop_idx']] @ W
            s = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
        if np.isfinite(thr) and thr < 1.0:
            k = nms_keep(b, s, thr)
            b = b[k]; s = np.asarray(s, float)[k]
        ne_parts.append(len(b))
        if len(b) == 0:
            sc_parts.append(np.zeros(0)); tp_parts.append(np.zeros(0, int)); continue
        if len(g) == 0:
            sc_parts.append(np.asarray(s, float)); tp_parts.append(np.zeros(len(b), int)); continue
        x1 = np.maximum(b[:, 0:1], g[None, :, 0]); y1 = np.maximum(b[:, 1:2], g[None, :, 1])
        x2 = np.minimum(b[:, 2:3], g[None, :, 2]); y2 = np.minimum(b[:, 3:4], g[None, :, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        a1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        a2 = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
        iou_m = inter / (a1[:, None] + a2[None, :] - inter + 1e-8)
        order = np.argsort(-np.asarray(s, float), kind='stable')
        matched = np.zeros(len(g), bool)
        tp = np.zeros(len(b), int)
        for i in order:
            row = np.where(matched, -1.0, iou_m[i])
            j = int(np.argmax(row))
            if row[j] >= 0.5:
                tp[i] = 1; matched[j] = True
        sc_parts.append(np.asarray(s, float)); tp_parts.append(tp)
    sc = np.concatenate(sc_parts) if sc_parts else np.zeros(0)
    tp = np.concatenate(tp_parts) if tp_parts else np.zeros(0, int)
    return sc, tp, np.asarray(ngt_parts, int), np.asarray(ne_parts, int)


def paired_boot(tab_ca, tab_tge, B, rng):
    """配对图像级 bootstrap: 每 replicate 用同一组重采样像索引评两条臂, 返回 margin 样本。"""
    n = len(tab_ca[2])
    out = np.empty(B)
    for k in range(B):
        m = rng.poisson(1.0, n)
        out[k] = S.ap50_replay(tab_ca, m) - S.ap50_replay(tab_tge, m)
    return out


def aps(tab):
    return S.ap50_replay(tab, None)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--B', type=int, default=1000)
    ap_.add_argument('--datasets', nargs='+', default=['DIOR'])
    ap_.add_argument('--detectors', nargs='+', default=['gdino'])
    ap_.add_argument('--thrs', nargs='+', type=float, default=[1.0, 0.65, 0.5])
    ap_.add_argument('--seed', type=int, default=1234)
    args = ap_.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

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

    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL + S.EXTRA_NAMES))
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    idx = {c: i for i, c in enumerate(all_names)}
    tb_idx = [idx[c] for c in sorted(set(RB.COCO_POOL + RB.DIOR_POOL))]
    t_bar = S.unit(E_photo[tb_idx].mean(0))

    gcfg = json.load(open(str(DATA / 'run_closed_loop_balanced.json')))['config']['gate']
    PRIOR_SCALE, c0 = gcfg['prior_scale'], gcfg['c0']

    def gate_conf(conf, c=c0, scale=3.0):
        return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))

    def ci(x):
        return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]

    out = {'config': {'datasets': args.datasets, 'detectors': args.detectors, 'thrs': args.thrs,
                      'B_boot': args.B, 'lam': LAM, 'beta_grid': BETA_GRID, 'c0': c0,
                      'prior_scale': PRIOR_SCALE, 'protocol': 'deployment order: score -> per-image NMS -> 11-pt AP50',
                      'note': 'thr=1.0 == no NMS (identity check vs score_gsd_tge_multids.json)'},
           'identity': {}, 'rows': []}

    for ds in args.datasets:
        spec = S.DATASETS[ds]
        for det in args.detectors:
            if det not in spec['dirs']:
                continue
            d = spec['dirs'][det]
            percell = {thr: {} for thr in args.thrs}   # thr -> unit -> arms aps + tabs
            for gsd in GSD_LEVELS:
                for key_idx, ev, src in spec['units']:
                    hp = S.pin_blob(d, spec['prefix'], key_idx, gsd)
                    if hp is None or src not in bank_coco:
                        continue
                    import pickle
                    blob = pickle.load(open(hp, 'rb'))
                    per_img, F = blob['per_img'], blob['F']
                    P_src = bank_coco[src]
                    P_neg = S.unit(RB.Proto.get_negative_prototype(bank_coco, src, NEG_TOPK))
                    dc = E_photo[idx[ev]] - E_over[idx[src]]
                    mag = float(np.linalg.norm(dc))
                    d_tge = E_photo[idx[ev]] - t_bar
                    max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
                    gamma = float(np.clip(PRIOR_SCALE * mag * gate_conf(max_conf), 0.0, 4.0))
                    Ws = {
                        'ca_pc':   S.unit(P_src + gamma * dc) - LAM * P_neg,
                        'tge_nat': S.unit(P_src + 1.0 * d_tge) - LAM * P_neg,
                        'tge_gate': S.unit(P_src + gamma * S.unit(d_tge) * mag) - LAM * P_neg,
                        'cond':    S.unit(P_src + gamma * S.unit(E_photo[idx[ev]] - E_over[idx[ev]]) * mag) - LAM * P_neg,
                        'vocab':   S.unit(P_src + gamma * S.unit(E_over[idx[ev]] - E_over[idx[src]]) * mag) - LAM * P_neg,
                    }
                    cur = [(float(RB.ap_of(per_img, F, S.unit(P_src + bb * d_tge) - LAM * P_neg)), bb)
                           for bb in BETA_GRID]
                    best_ap, best_b = max(cur, key=lambda x: x[0])
                    Ws['tge_star'] = S.unit(P_src + best_b * d_tge) - LAM * P_neg

                    for thr in args.thrs:
                        cell = {'unit': key_idx, 'eval': ev, 'src': src, 'gsd': gsd, 'n_img': len(per_img)}
                        tb = per_image_tp_nms(per_img, F, None, thr, use_raw=True)
                        cell['base'] = float(S.ap50_replay(tb, None))
                        tabs = {'base': tb}
                        for arm, W in Ws.items():
                            tabs[arm] = per_image_tp_nms(per_img, F, W, thr)
                            cell[arm] = aps(tabs[arm])
                        cell['tabs'] = tabs
                        percell[thr][(key_idx, ev, src, gsd)] = cell
                    print(f'  [{ds}/{det}] {key_idx}->{src} f={gsd} base={percell[args.thrs[-1]][(key_idx,ev,src,gsd)]["base"]:.3f} '
                          f'CA={percell[args.thrs[-1]][(key_idx,ev,src,gsd)]["ca_pc"]:.3f} '
                          f'TGE={percell[args.thrs[-1]][(key_idx,ev,src,gsd)]["tge_nat"]:.3f}', flush=True)

            for thr in args.thrs:
                cells = list(percell[thr].values())
                if not cells:
                    continue
                arms = ['base', 'ca_pc', 'tge_nat', 'tge_star', 'tge_gate', 'cond', 'vocab']
                macro_raw = {a: float(np.mean([c[a] for c in cells])) for a in arms}
                macro = {a: round(macro_raw[a], 4) for a in arms}
                gains = {a: round(macro_raw[a] - macro_raw['base'], 4) for a in arms if a != 'base'}
                margin_nat = round(macro_raw['ca_pc'] - macro_raw['tge_nat'], 4)
                margin_star = round(macro_raw['ca_pc'] - macro_raw['tge_star'], 4)
                margin_cv = round(macro_raw['cond'] - macro_raw['vocab'], 4)
                # macro 层配对 bootstrap: 每个 cell 内重采样图像, 再对 cell 取均值 (与
                # score_gsd_tge_multids.py 逐字同口径), 故点估计 == macro(CA) − macro(TGE)
                boots = {k: np.empty(args.B) for k in ('ca', 'tn', 'ts', 'm_nat', 'm_star', 'm_cv')}
                for bi in range(args.B):
                    ap = {a: np.empty(len(cells)) for a in arms}
                    for i2, c2 in enumerate(cells):
                        n = c2['n_img']
                        cnt = np.bincount(rng.integers(0, n, n), minlength=n)
                        for a in arms:
                            ap[a][i2] = S.ap50_replay(c2['tabs'][a], cnt)
                    ca = ap['ca_pc'].mean(); tn = ap['tge_nat'].mean(); ts = ap['tge_star'].mean()
                    bs = ap['base'].mean()
                    boots['ca'][bi] = ca - bs; boots['tn'][bi] = tn - bs; boots['ts'][bi] = ts - bs
                    boots['m_nat'][bi] = ca - tn; boots['m_star'][bi] = ca - ts
                    boots['m_cv'][bi] = ap['cond'].mean() - ap['vocab'].mean()
                lead = {str(g): bool(float(np.mean([c['ca_pc'] for c in cells if c['gsd'] == g])) >
                                     float(np.mean([c['tge_nat'] for c in cells if c['gsd'] == g])))
                        for g in GSD_LEVELS if any(c['gsd'] == g for c in cells)}
                row = {
                    'dataset': ds, 'detector': det, 'nms_iou': thr, 'n_cells': len(cells),
                    'base_abs_macro': macro['base'], 'gain_macro': gains,
                    'margin_ca_minus_tge_nat': margin_nat, 'ci_margin_nat': ci(boots['m_nat']),
                    'ci_margin_nat_excludes_zero': bool(ci(boots['m_nat'])[0] > 0),
                    'margin_ca_minus_tge_star': margin_star, 'ci_margin_star': ci(boots['m_star']),
                    'margin_cond_minus_vocab': margin_cv, 'ci_margin_cv': ci(boots['m_cv']),
                    'boot_mean_ca_gain': round(float(boots['ca'].mean()), 4),
                    'boot_mean_tge_gain': round(float(boots['tn'].mean()), 4),
                    'lead_every_gsd_vs_tge_nat': lead, 'lead_all': bool(all(lead.values())) if lead else None,
                    'n_cells_ca_gt_tge_nat': int(sum(1 for c in cells if c['ca_pc'] > c['tge_nat'])),
                    'n_cells_ca_gt_tge_star': int(sum(1 for c in cells if c['ca_pc'] > c['tge_star']))}
                out['rows'].append(row)
                print(f'>> {ds}/{det} thr={thr}: base={macro["base"]:.4f} '
                      f'CAgain={gains["ca_pc"]:+.4f} TGEgain={gains["tge_nat"]:+.4f} TGE*={gains["tge_star"]:+.4f} '
                      f'marginCA-TGE={margin_nat:+.4f} CI[{ci(boots["m_nat"])[0]:+.4f},{ci(boots["m_nat"])[1]:+.4f}] '
                      f'excl0={row["ci_margin_nat_excludes_zero"]} cond-vocab={margin_cv:+.4f} '
                      f'逐档领先={row["lead_all"]} CA>TGE {row["n_cells_ca_gt_tge_nat"]}/{len(cells)} '
                      f'CA>TGE* {row["n_cells_ca_gt_tge_star"]}/{len(cells)}', flush=True)

    outp = DATA / 'score_gsd_nms.json'
    json.dump(out, open(outp, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n[done {time.time()-t0:.0f}s] -> {outp}', flush=True)


if __name__ == '__main__':
    main()
