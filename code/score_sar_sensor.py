# -*- coding: utf-8 -*-
"""score_sar_sensor.py — "真实换传感器" 条件轴（光学 RS ↔ SAR）

元 2026-10-14: 第 4 版评审的第 1 条推荐 —— "在真实跨传感器或跨成像条件的遥感数据上，
验证 CA-PC 相对 TGE 的优势是否依然成立"。§4.23 的 VEDAI 轴是【同一传感器内的成像光谱变化】
(color ↔ NIR)；本脚本再往前一步，做【真实换传感器】：光学遥感 ↔ 合成孔径雷达(SAR)。

设计：目标域 = 船只（ship），源域 = COCO boat（与其它表同口径）。
同一目标类、同一源类，两条臂**只差成像传感器**：

  目标成像  m ∈ {opt, sar}
    opt : DIOR 光学 ship 影像（RGB, 800×800）
    sar : SSDD SAR ship 影像（RadarSat-2 / TerraSAR-X / Sentinel-1 合成, 单类 ship）
  锚词条件 e ∈ {over, sar}
    over: "an overhead image of {c}"
    sar : "a synthetic aperture radar image of {c}"

2×2：(opt,over) 与 (sar,sar) 是【锚词点中真实传感器】的匹配对；
(opt,sar) 与 (sar,over) 是【锚词没点中】的对照。若增益来自观测条件项，
匹配对的 CA−TGE 应明显大于不匹配对。

与 VEDAI 轴不同：这里目标类名 ship ≠ 源类名 boat，**cond / vocab 分量可以真正分离**
（VEDAI 轴上二者同名、joint≡cond，故那里禁止声称分解）。

臂（与 score_gsd_tge_multids.py / score_vedai_spectral.py 逐字一致）:
  base / ca_pc / cond / vocab / tge_nat(β=1) / tge_star(网格最优)
配对图像级 bootstrap 在 macro 层。

用法:
  小样本验证: HF_HUB_OFFLINE=1 "C:/Users/张三/vg/Scripts/python.exe" -u score_sar_sensor.py \
                --n 20 --B 200 --tag probe
  全量:       ... --n 150 --B 1000
"""
import os, sys, json, pickle, time, argparse
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cuda')
os.environ.setdefault('AC_SPC_COCO_ANN', r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json')
os.environ.setdefault('AC_SPC_COCO_IMG', r'C:\Users\张三\modeling\coco_data\val2017')
os.environ.setdefault('AC_SPC_DIOR_IMG', r'C:\Users\张三\modeling\dior_data\eval_v2_images')
os.environ.setdefault('AC_SPC_DIOR_PARQUET', r'C:\Users\张三\modeling\dior_data\dior_combined.parquet')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch
from pathlib import Path
from PIL import Image

import run_bidirectional as RB

LAM, NEG_TOPK, N_PROTO = RB.LAM, RB.NEG_TOPK, RB.N_PROTO
BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

SSDD_ROOT = Path(r'C:\Users\张三\modeling\sar_data\SSDD\data')

# (目标类名, COCO 源类名) —— ship↔boat 与论文 CORE 对一致
UNITS = [('ship', 'boat')]
PROMPT = {'ship': 'ship'}

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
MODALITIES = ['opt', 'sar']
ANCHORS = ['over', 'sar']


def log(*a):
    print(*a, flush=True)


def unit(x):
    x = np.asarray(x, float)          # bank 里存的是 torch tensor, 必须先转 numpy
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def load_ssdd(offset, n):
    """SSDD: YOLO 归一化标签 -> 像素 AABB。只保有 >=1 个目标的图。"""
    imgdir, labdir = SSDD_ROOT / 'images', SSDD_ROOT / 'labels'
    out = []
    for ip in sorted(imgdir.rglob('*.jpg')):
        lp = labdir / ip.relative_to(imgdir).with_suffix('.txt')
        if not lp.exists():
            continue
        try:
            W, H = Image.open(ip).size
        except Exception:
            continue
        boxes = []
        for l in open(lp, encoding='utf-8', errors='ignore'):
            p = l.split()
            if len(p) < 5:
                continue
            try:
                _, xc, yc, bw, bh = [float(v) for v in p[:5]]
            except ValueError:
                continue
            x1, y1 = (xc - bw / 2) * W, (yc - bh / 2) * H
            x2, y2 = (xc + bw / 2) * W, (yc + bh / 2) * H
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(W), x2), min(float(H), y2)
            if x2 - x1 > 4 and y2 - y1 > 4:
                boxes.append([x1, y1, x2, y2])
        if boxes:
            out.append((ip, boxes))
    sel = out[offset:offset + n]
    imgs = [Image.open(p).convert('RGB') for p, _ in sel]
    # 返回 list-of-lists: detect_encode 里 `if g` 对 ndarray 会有歧义
    return imgs, [b for _, b in sel]


def per_image_tp(per_img, F, W=None, use_raw=False):
    """逐字复刻 acspc.metrics.compute_ap50_img 的逐图贪心匹配 (IoU 分母 +1e-8、不裁剪)。"""
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
    ap_.add_argument('--n', type=int, default=150)
    ap_.add_argument('--modalities', nargs='+', default=['opt', 'sar'])
    ap_.add_argument('--tag', default='')
    ap_.add_argument('--seed', type=int, default=1234)
    args = ap_.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    cfg = RB.Config()
    cfg.model.device = RB.DEVICE
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'
    RB.CACHE_DIR = DATA / 'probe_sar_sensor_cache'
    RB.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- COCO 源原型 bank（与其它表同口径） ----
    bank_coco = {}
    for cls in RB.COCO_POOL:
        ims, gts = RB.load_split(cfg, 'coco', cls, 0, N_PROTO)
        crops = RB.D.gt_crops(ims, gts)
        if not crops:
            continue
        Ft = RB.clip_encode_imgs(crops)
        pt = torch.from_numpy(Ft.mean(0)); bank_coco[cls] = pt / pt.norm()

    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    E_sar = RB.text_embs([f'a synthetic aperture radar image of {c}' for c in all_names])
    E_ENV = {'over': E_over, 'sar': E_sar}
    idx = {c: i for i, c in enumerate(all_names)}
    t_bar = unit(E_photo[[idx[c] for c in sorted(set(RB.COCO_POOL + RB.DIOR_POOL))]].mean(0))

    gcfg = json.load(open(str(DATA / 'run_closed_loop_balanced.json')))['config']['gate']
    PRIOR_SCALE, c0 = gcfg['prior_scale'], gcfg['c0']

    def gate_conf(conf, c=c0, scale=3.0):
        return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))

    def ci(x):
        return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]

    out = {'config': {'ssdd_root': str(SSDD_ROOT), 'units': UNITS, 'modalities': args.modalities,
                      'anchors': ANCHORS, 'n_per_arm': args.n, 'B_boot': args.B, 'lam': LAM,
                      'c0': c0, 'prior_scale': PRIOR_SCALE, 'zero_target_gt': True,
                      'axis': 'REAL sensor change (optical RS vs SAR), no resampling',
                      'opt_target': 'DIOR optical ship', 'sar_target': 'SSDD SAR ship',
                      'source': 'COCO boat prototypes',
                      'matched_pairs': ['(opt,over)', '(sar,sar)'],
                      'mismatched_pairs': ['(opt,sar)', '(sar,over)'],
                      'note': 'target name ship != source name boat -> cond/vocab separable here'},
           'identity': {}, 'rows': []}
    max_dev = 0.0

    # ---- 逐 (modality, class) 建/读检测缓存（GPU），随后 CPU 打分 ----
    cache = {}
    for m in args.modalities:
        for name, src in UNITS:
            if m == 'sar':
                imgs, gts = load_ssdd(0, args.n)
            else:
                imgs, gts = RB.load_split(cfg, 'dior', name, 0, args.n)
            if not imgs:
                log(f'  !! {m}/{name} 无图'); continue
            key = f'sensor_{m}_{name}_n{args.n}'
            blob = RB.detect_encode(cfg, imgs, gts, PROMPT[name], key)
            cache[(m, name)] = blob
            log(f'  [cache] {key}: N={len(blob["per_img"])} boxes='
                f'{sum(len(p["gd"]) for p in blob["per_img"])}')

    RB.DEVICE = 'cpu'
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    for m in args.modalities:
        for e in ANCHORS:
            cells = []
            for name, src in UNITS:
                if (m, name) not in cache or src not in bank_coco:
                    continue
                blob = cache[(m, name)]
                per_img, F = blob['per_img'], blob['F']
                P_src = bank_coco[src]
                P_neg = unit(RB.Proto.get_negative_prototype(bank_coco, src, NEG_TOPK))
                ev = name
                dc_joint = E_photo[idx[ev]] - E_ENV[e][idx[src]]
                dc_cond = E_photo[idx[ev]] - E_ENV[e][idx[ev]]
                dc_voc = E_ENV[e][idx[ev]] - E_ENV[e][idx[src]]
                mag = float(np.linalg.norm(dc_joint))
                d_tge = E_photo[idx[ev]] - t_bar
                max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
                gamma = float(np.clip(PRIOR_SCALE * mag * gate_conf(max_conf), 0.0, 4.0))
                Ws = {
                    'ca_pc':  unit(P_src + gamma * dc_joint) - LAM * P_neg,
                    'cond':   unit(P_src + gamma * unit(dc_cond) * mag) - LAM * P_neg,
                    'vocab':  unit(P_src + gamma * unit(dc_voc) * mag) - LAM * P_neg,
                    'tge_nat': unit(P_src + 1.0 * d_tge) - LAM * P_neg,
                }
                cur = [(float(RB.ap_of(per_img, F, unit(P_src + bb * d_tge) - LAM * P_neg)), bb)
                       for bb in BETA_GRID]
                _, best_b = max(cur, key=lambda x: x[0])
                Ws['tge_star'] = unit(P_src + best_b * d_tge) - LAM * P_neg

                base_ap = float(RB.ap_base(per_img, F))
                tabs, gains = {}, {}
                tabs['base'] = per_image_tp(per_img, F, use_raw=True)
                gains['base'] = base_ap
                for arm in ('ca_pc', 'cond', 'vocab', 'tge_nat', 'tge_star'):
                    off = float(RB.ap_of(per_img, F, Ws[arm]))
                    tab = per_image_tp(per_img, F, Ws[arm])
                    dev = abs(ap50_replay(tab) - off)
                    max_dev = max(max_dev, dev)
                    if dev > 1e-9:
                        raise SystemExit(f'!! 身份校验失败 {m}/{e}/{name}/{arm}: {off:.12f} vs {ap50_replay(tab):.12f}')
                    tabs[arm] = tab; gains[arm] = off - base_ap
                if abs(ap50_replay(tabs['base']) - base_ap) > 1e-9:
                    raise SystemExit(f'!! base 身份校验失败 {m}/{e}/{name}')
                cells.append({'cls': name, 'src': src, 'n_img': len(per_img), 'base_ap': round(base_ap, 4),
                              'gamma': round(gamma, 4), 'mag': round(mag, 4), 'tge_star_beta': best_b,
                              'gains': {k: round(v, 4) for k, v in gains.items()}, 'tabs': tabs})
            if not cells:
                continue
            mr = lambda a: float(np.mean([c['gains'][a] for c in cells]))
            macro_base = float(np.mean([c['base_ap'] for c in cells]))
            gains = {a: round(mr(a), 4) for a in ('ca_pc', 'cond', 'vocab', 'tge_nat', 'tge_star')}
            margin_nat = round(mr('ca_pc') - mr('tge_nat'), 4)
            margin_star = round(mr('ca_pc') - mr('tge_star'), 4)
            margin_cv = round(mr('cond') - mr('vocab'), 4)
            boots = {k: np.empty(args.B) for k in ('m_nat', 'm_star', 'm_cv')}
            for bi in range(args.B):
                ap = {a: np.empty(len(cells)) for a in ('base', 'ca_pc', 'cond', 'vocab', 'tge_nat', 'tge_star')}
                for i2, c2 in enumerate(cells):
                    n = c2['n_img']
                    cnt = np.bincount(rng.integers(0, n, n), minlength=n)
                    for a in ap:
                        ap[a][i2] = ap50_replay(c2['tabs'][a], cnt)
                boots['m_nat'][bi] = ap['ca_pc'].mean() - ap['tge_nat'].mean()
                boots['m_star'][bi] = ap['ca_pc'].mean() - ap['tge_star'].mean()
                boots['m_cv'][bi] = ap['cond'].mean() - ap['vocab'].mean()
            matched = (m == 'opt' and e == 'over') or (m == 'sar' and e == 'sar')
            row = {'target_sensor': m, 'anchor_env': e, 'matched': bool(matched),
                   'n_cells': len(cells), 'base_abs_macro': round(macro_base, 4), 'gain_macro': gains,
                   'margin_ca_minus_tge_nat': margin_nat, 'ci_margin_nat': ci(boots['m_nat']),
                   'excl0_nat': bool(ci(boots['m_nat'])[0] > 0),
                   'margin_ca_minus_tge_star': margin_star, 'ci_margin_star': ci(boots['m_star']),
                   'margin_cond_minus_vocab': margin_cv, 'ci_margin_cv': ci(boots['m_cv']),
                   'cells': [{k: v for k, v in c.items() if k != 'tabs'} for c in cells]}
            out['rows'].append(row)
            log(f">> target={m:<3} anchor={e:<4} {'MATCHED ' if matched else 'mismatch'} "
                f"base={macro_base:.4f} CA={gains['ca_pc']:+.4f} cond={gains['cond']:+.4f} "
                f"vocab={gains['vocab']:+.4f} TGE={gains['tge_nat']:+.4f} TGE*={gains['tge_star']:+.4f} | "
                f"CA-TGE={margin_nat:+.4f} CI[{ci(boots['m_nat'])[0]:+.4f},{ci(boots['m_nat'])[1]:+.4f}] "
                f"excl0={row['excl0_nat']} | cond-vocab={margin_cv:+.4f}")

    out['identity'] = {'max_abs_dev_replay_vs_official': max_dev,
                       'criterion': 'ap50_replay(tab) == RB.ap_of/ap_base for every cell and arm'}
    tag = f'_{args.tag}' if args.tag else ''
    outp = DATA / f'score_sar_sensor{tag}.json'
    json.dump(out, open(outp, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    log(f'\n[done {time.time()-t0:.0f}s] identity max|Δ|={max_dev:.2e} -> {outp}')


if __name__ == '__main__':
    main()
