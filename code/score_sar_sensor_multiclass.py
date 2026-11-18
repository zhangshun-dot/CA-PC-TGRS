# -*- coding: utf-8 -*-
"""score_sar_sensor_multiclass.py — Risk① 扩展版："真实换传感器" 条件轴（光学 RS ↔ SAR）多类别 / 独立数据集

元 2026-10-15: 在 score_sar_sensor.py（仅 ship / 仅 SSDD）基础上扩展，回答第 8 版评审 Risk①：
  (a) 不只 ship —— 加入 airplane 类（SAR-AIRcraft-1.0, SAR）；
  (b) 不止一个 SAR 数据集 —— 加入第二个独立 SAR↔光学 ship 对（HRSID 或 SRSDD）。

设计：每个 (目标类, 源类, SAR源) 是一条「单元」，光学臂统一取 DIOR 同名目标，源原型统一取 COCO。
  - OPTICAL 臂（DIOR, RGB）：ship↔boat, airplane↔airplane
  - SAR 臂（各 SAR 数据集）：SSDD ship, HRSID ship, SRSDD ship, SAR-AIRcraft airplane
每单元对锚词 e∈{over, sar} 各出一行，matched = (opt & over) 或 (sar & sar)。
每个 (modality, anchor) 行内只含该单元单个类的 cell（bootstrap 在该类图像上做）。

数据缺失容忍：某 SAR 数据集目录不存在时自动跳过该单元并 log，不影响其余单元出数。
这样把数据集放进 C:\\Users\\张三\\modeling\\sar_data\\ 对应子目录后即可直接全量跑，无需再改脚本。

用法（与旧脚本同口径）:
  小样本验证: HF_HUB_OFFLINE=1 "C:/Users/张三/vg/Scripts/python.exe" -u score_sar_sensor_multiclass.py --n 20 --B 200 --tag smoke
  单数据集全量: ... --units SSDD --n 150 --B 1000
  全量:        ... --n 150 --B 1000
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
import xml.etree.ElementTree as ET

import run_bidirectional as RB

LAM, NEG_TOPK, N_PROTO = RB.LAM, RB.NEG_TOPK, RB.N_PROTO
BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

SAR_DATA = Path(r'C:\Users\张三\modeling\sar_data')
SSDD_ROOT = SAR_DATA / 'SSDD' / 'data'

# (目标类名, COCO 源类名, DIOR 光学类名) —— 光学臂
OPTICAL = [
    ('ship', 'boat', 'ship'),
    ('airplane', 'airplane', 'airplane'),
]
# (目标类名, COCO 源类名, SAR 数据集 key) —— SAR 臂
SAR = [
    ('ship', 'boat', 'SSDD'),
    ('ship', 'boat', 'HRSID'),
    ('ship', 'boat', 'SRSDD'),
    ('airplane', 'airplane', 'SAR-AIRcraft'),
]
SAR_ROOTS = {
    'SSDD': SSDD_ROOT,
    'HRSID': SAR_DATA / 'HRSID',
    'SRSDD': SAR_DATA / 'SRSDD',
    'SAR-AIRcraft': SAR_DATA / 'SAR-AIRcraft-1.0',
}
# SAR-AIRcraft-1.0 (HF 镜像) 的 VOC xml <name> 是具体机型，不是 'airplane'；
# 该数据集整体即飞机，故把所有机型名映射到 airplane（AP 匹配只用框，不看标签类别）。
# 原始 7 类: A220 / A320-321 / A330 / ARJ21 / Boeing737 / Boeing787 / other
SAR_WANT = {
    'SSDD': {'ship'},
    'HRSID': {'ship'},
    'SRSDD': {'ship'},
    'SAR-AIRcraft': {'a220', 'a320/321', 'a330', 'arj21', 'boeing737', 'boeing787', 'other'},
}

PROMPT = {'ship': 'ship', 'airplane': 'airplane'}

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
MODALITIES = ['opt', 'sar']
ANCHORS = ['over', 'sar']


def log(*a):
    print(*a, flush=True)


def unit(x):
    x = np.asarray(x, float)
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


# ---------------- 通用 SAR 数据集加载（自动探测子目录与标注格式） ----------------
def _find_sub(parent, candidates):
    for c in candidates:
        p = parent / c
        if p.exists() and p.is_dir():
            return p
    return None


def _voc_boxes(xmlp, want):
    boxes = []
    try:
        t = ET.parse(str(xmlp)).getroot()
    except Exception:
        return boxes
    for obj in t.iter('object'):
        nm = (obj.findtext('name') or '').strip().lower()
        if nm not in want:
            continue
        b = obj.find('bndbox')
        if b is None:
            continue
        try:
            x1 = float(b.findtext('xmin')); y1 = float(b.findtext('ymin'))
            x2 = float(b.findtext('xmax')); y2 = float(b.findtext('ymax'))
        except Exception:
            continue
        if x2 - x1 > 4 and y2 - y1 > 4:
            boxes.append([x1, y1, x2, y2])
    return boxes


def _yolo_boxes(txtp, imgp, want):
    try:
        W, H = Image.open(imgp).size
    except Exception:
        return []
    boxes = []
    for l in open(txtp, encoding='utf-8', errors='ignore'):
        p = l.split()
        if len(p) < 5:
            continue
        try:
            xc, yc, bw, bh = [float(v) for v in p[1:5]]
        except ValueError:
            continue
        x1, y1 = (xc - bw / 2) * W, (yc - bh / 2) * H
        x2, y2 = (xc + bw / 2) * W, (yc + bh / 2) * H
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(W), x2), min(float(H), y2)
        if x2 - x1 > 4 and y2 - y1 > 4:
            boxes.append([x1, y1, x2, y2])
    return boxes


def load_sar_dataset(root, offset, n, want):
    """通用 SAR 加载：探测 images/JPEGImages 与 Annotations/annotations/labels；
    优先 COCO json（如 HRSID），回退 VOC xml，再回退 YOLO txt。"""
    root = Path(root)
    imgdir = _find_sub(root, ['images', 'JPEGImages', 'JPEG', 'imgs']) or root
    anndir = _find_sub(root, ['Annotations', 'annotations', 'labels', 'xml'])
    coco_map = {}
    js = sorted(anndir.glob('*.json')) if anndir is not None else sorted(root.glob('*.json'))
    for jp in js:
        try:
            coco_map.update(_coco_boxes(jp, want))
        except Exception:
            pass
    out = []
    imgs_all = sorted(imgdir.rglob('*.jpg')) + sorted(imgdir.rglob('*.png'))
    for ip in imgs_all:
        boxes = []
        if ip.stem in coco_map:
            boxes = coco_map[ip.stem]
        elif anndir is not None:
            xlp = anndir / (ip.stem + '.xml')
            if xlp.exists():
                boxes = _voc_boxes(xlp, want)
            else:
                tlp = anndir / (ip.stem + '.txt')
                if tlp.exists():
                    boxes = _yolo_boxes(tlp, ip, want)
        if boxes:
            out.append((ip, boxes))
    sel = out[offset:offset + n]
    imgs = [Image.open(p).convert('RGB') for p, _ in sel]
    return imgs, [b for _, b in sel]


def _coco_boxes(jsonp, want):
    """解析 COCO 格式标注（如 HRSID train2017.json），返回 {图像stem: [[x1,y1,x2,y2], ...]}。"""
    d = json.load(open(jsonp, encoding='utf-8'))
    cat = {c['id']: (c.get('name') or '').strip().lower() for c in d.get('categories', [])}
    id2name = {im['id']: im['file_name'] for im in d.get('images', [])}
    fmap = {}
    for a in d.get('annotations', []):
        nm = cat.get(a.get('category_id'), '').lower()
        if nm not in want:
            continue
        fn = id2name.get(a.get('image_id'))
        if not fn:
            continue
        x, y, w, h = a['bbox']
        x1, y1, x2, y2 = float(x), float(y), float(x) + float(w), float(y) + float(h)
        if x2 - x1 > 4 and y2 - y1 > 4:
            fmap.setdefault(Path(fn).stem, []).append([x1, y1, x2, y2])
    return fmap


def load_ssdd(offset, n):
    """SSDD: YOLO 归一化标签 -> 像素 AABB。"""
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
    return imgs, [b for _, b in sel]


def per_image_tp(per_img, F, W=None, use_raw=False):
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
    ap_.add_argument('--units', nargs='+', default=None,
                     help='仅跑指定 SAR 数据集 key（SSDD/HRSID/SRSDD/SAR-AIRcraft），默认全部存在者')
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
    RB.CACHE_DIR = DATA / 'probe_sar_sensor_mc_cache'
    RB.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- COCO 源原型 bank ----
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

    present_sar = [k for k in [u[2] for u in SAR] if SAR_ROOTS[k].exists()]
    if args.units:
        present_sar = [k for k in present_sar if k in args.units]

    out = {'config': {'ssdd_root': str(SSDD_ROOT), 'present_sar_datasets': present_sar,
                      'optical_units': OPTICAL, 'sar_units_all': SAR,
                      'anchors': ANCHORS, 'n_per_arm': args.n, 'B_boot': args.B, 'lam': LAM,
                      'c0': c0, 'prior_scale': PRIOR_SCALE, 'zero_target_gt': True,
                      'axis': 'REAL sensor change (optical RS vs SAR), multi-class + independent datasets',
                      'source': 'COCO prototypes',
                      'matched_pairs': ['(opt,over)', '(sar,sar)'],
                      'mismatched_pairs': ['(opt,sar)', '(sar,over)'],
                      'note': 'airplane tgt==src -> cond/vocab split collapses (same as VEDAI); report cond only'},
           'identity': {}, 'rows': []}
    max_dev = 0.0

    # ---- 建检测缓存（GPU） ----
    cache = {}   # key -> blob

    def build_optical(tgt, src, diorcls):
        key = f'sensor_opt_dior_{tgt}_n{args.n}'
        if key in cache:
            return
        imgs, gts = RB.load_split(cfg, 'dior', diorcls, 0, args.n)
        if not imgs:
            log(f'  !! optical/{tgt} 无图'); return
        blob = RB.detect_encode(cfg, imgs, gts, PROMPT[tgt], key)
        cache[key] = blob
        log(f'  [cache] {key}: N={len(blob["per_img"])} boxes='
            f'{sum(len(p["gd"]) for p in blob["per_img"])}')

    def build_sar(tgt, src, sar_key):
        key = f'sensor_sar_{sar_key}_{tgt}_n{args.n}'
        if key in cache:
            return
        root = SAR_ROOTS[sar_key]
        if not root.exists():
            log(f'  !! SAR 数据集缺失: {root}（跳过）'); return
        if sar_key == 'SSDD':
            imgs, gts = load_ssdd(0, args.n)
        else:
            imgs, gts = load_sar_dataset(root, 0, args.n, SAR_WANT[sar_key])
        if not imgs:
            log(f'  !! {sar_key} 无可用图（标注/图像缺失？）'); return
        blob = RB.detect_encode(cfg, imgs, gts, PROMPT[tgt], key)
        cache[key] = blob
        log(f'  [cache] {key}: N={len(blob["per_img"])} boxes='
            f'{sum(len(p["gd"]) for p in blob["per_img"])}')

    for tgt, src, diorcls in OPTICAL:
        build_optical(tgt, src, diorcls)
    for tgt, src, sar_key in SAR:
        if sar_key in present_sar:
            build_sar(tgt, src, sar_key)
        else:
            log(f'  [skip] SAR 单元 {sar_key}/{tgt}（数据集不在 present 列表）')

    RB.DEVICE = 'cpu'
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- 逐单元逐 (modality, anchor) 打分 ----
    def compute_cell(blob, tgt, src, e):
        nonlocal max_dev
        per_img, F = blob['per_img'], blob['F']
        if src not in bank_coco:
            return None
        P_src = bank_coco[src]
        P_neg = unit(RB.Proto.get_negative_prototype(bank_coco, src, NEG_TOPK))
        ev = tgt
        dc_joint = E_photo[idx[ev]] - E_ENV[e][idx[src]]
        dc_cond = E_photo[idx[ev]] - E_ENV[e][idx[ev]]
        dc_voc = E_ENV[e][idx[ev]] - E_ENV[e][idx[src]]
        mag = float(np.linalg.norm(dc_joint))
        d_tge = E_photo[idx[ev]] - t_bar
        max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
        gamma = float(np.clip(PRIOR_SCALE * mag * gate_conf(max_conf), 0.0, 4.0))
        Ws = {
            'ca_pc': unit(P_src + gamma * dc_joint) - LAM * P_neg,
            'cond': unit(P_src + gamma * unit(dc_cond) * mag) - LAM * P_neg,
            'vocab': unit(P_src + gamma * unit(dc_voc) * mag) - LAM * P_neg,
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
                raise SystemExit(f'!! 身份校验失败 {tgt}/{e}/{arm}: {off:.12f} vs {ap50_replay(tab):.12f}')
            tabs[arm] = tab; gains[arm] = off - base_ap
        if abs(ap50_replay(tabs['base']) - base_ap) > 1e-9:
            raise SystemExit(f'!! base 身份校验失败 {tgt}/{e}')
        return {'cls': tgt, 'src': src, 'n_img': len(per_img), 'base_ap': round(base_ap, 4),
                'gamma': round(gamma, 4), 'mag': round(mag, 4), 'tge_star_beta': best_b,
                'gains': {k: round(v, 4) for k, v in gains.items()}, 'tabs': tabs}

    jobs = []  # (m, tgt, src, sar_key_or_None, cache_key, dataset_label)
    for tgt, src, diorcls in OPTICAL:
        jobs.append(('opt', tgt, src, None, f'sensor_opt_dior_{tgt}_n{args.n}', 'DIOR(optical)'))
    for tgt, src, sar_key in SAR:
        if sar_key in present_sar:
            jobs.append(('sar', tgt, src, sar_key, f'sensor_sar_{sar_key}_{tgt}_n{args.n}', sar_key))

    for m, tgt, src, sar_key, ckey, dlabel in jobs:
        blob = cache.get(ckey)
        if blob is None:
            continue
        for e in ANCHORS:
            cell = compute_cell(blob, tgt, src, e)
            if cell is None:
                continue
            cells = [cell]
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
            row = {'arm': m, 'target_class': tgt, 'source_class': src, 'sar_dataset': sar_key,
                   'dataset_label': dlabel, 'anchor_env': e, 'matched': bool(matched),
                   'n_cells': len(cells), 'base_abs_macro': round(macro_base, 4), 'gain_macro': gains,
                   'margin_ca_minus_tge_nat': margin_nat, 'ci_margin_nat': ci(boots['m_nat']),
                   'excl0_nat': bool(ci(boots['m_nat'])[0] > 0),
                   'margin_ca_minus_tge_star': margin_star, 'ci_margin_star': ci(boots['m_star']),
                   'margin_cond_minus_vocab': margin_cv, 'ci_margin_cv': ci(boots['m_cv']),
                   'cells': [{k: v for k, v in c.items() if k != 'tabs'} for c in cells]}
            out['rows'].append(row)
            log(f">> {dlabel:<16} {tgt:<9} {m:<3} anchor={e:<4} {'MATCHED ' if matched else 'mismatch'} "
                f"base={macro_base:.4f} CA={gains['ca_pc']:+.4f} cond={gains['cond']:+.4f} "
                f"vocab={gains['vocab']:+.4f} TGE={gains['tge_nat']:+.4f} TGE*={gains['tge_star']:+.4f} | "
                f"CA-TGE={margin_nat:+.4f} CI[{ci(boots['m_nat'])[0]:+.4f},{ci(boots['m_nat'])[1]:+.4f}] "
                f"excl0={row['excl0_nat']} | cond-vocab={margin_cv:+.4f}")

    out['identity'] = {'max_abs_dev_replay_vs_official': max_dev,
                       'criterion': 'ap50_replay(tab) == RB.ap_of/ap_base for every cell and arm'}
    tag = f'_{args.tag}' if args.tag else ''
    outp = DATA / f'score_sar_sensor_multiclass{tag}.json'
    json.dump(out, open(outp, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    log(f'\n[done {time.time()-t0:.0f}s] present_sar={present_sar} identity max|Δ|={max_dev:.2e} -> {outp}')


if __name__ == '__main__':
    main()
