# -*- coding: utf-8 -*-
r"""confirm_devtest_global_split.py — 主配置的「独立确认」实验

回答评审第 3 条：主配置 (beta=2.0, lam=0.5) 是在**同一批** 1291 张评测图上
搜索并报告的，不构成独立确认。本脚本把评测图按 **COCO image ID 全局切分**
成 dev / test（同一张图不会同时进两侧，跨类也不重复），

  1. 在 dev 上按论文同一条选择规则（dev 平均增益最大）与**同一网格**
     beta in {0.5,1,2,4} x lam in {0,0.25,0.5,0.75,1}
     选出 (beta, lam)；
  2. 在**没参与选择**的 test 上报告增益 + 逐类增益 + 配对 bootstrap 区间。

另跑 5 个随机全局划分给分布，并报 (2.0,0.5) 与 (1.0,0.25) 在同 test 上的值作对照。

口径（与 tge_v2_paper_main_recompute.py 完全一致）:
  * 数据   : COCO val2017 六类全量, 每类前 K_CALIB=10 张作校准并排除
  * 检测器 : GDINO-tiny, prompt "a photo of {cls}", th=0.0, 复用已有缓存
  * 特征   : CLIP ViT-B/32 (U.get_clip), 候选框 crop
  * 融合   : s = sqrt(gd * clip(max(0, F @ W))),  W = unit(P_src + beta*Delta) - lam*P_neg
  * 指标   : 数据集级逐图匹配 11-point AP50 (U.compute_ap50_img 的逐位等价向量化版)

用法:
  HF_HUB_OFFLINE=1 "C:/Users/张三/vg/Scripts/python.exe" -u confirm_devtest_global_split.py --mode all
"""
import os, sys, json, pickle, time, argparse
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch
from pathlib import Path
from PIL import Image

import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
CACHE = DATA / 'confirm_split_cache'
CACHE.mkdir(exist_ok=True)
OUT = DATA / 'confirm_devtest_global_split.json'

K_CALIB = 10
PRE_FILTER = 0.03
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'

BETA_GRID = [0.5, 1.0, 2.0, 4.0]
LAM_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

DEV_CLASSES = [
    ('airplane',   'airplane'),
    ('boat',       'boat'),
    ('car',        'car'),
    ('truck',      'truck'),
    ('bus',        'bus'),
    ('motorcycle', 'motorcycle'),
]
# (类 -> 源类) 与论文 DEV_DIOR 一致
COCO2SRC = {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle',
            'truck': 'vehicle', 'bus': 'vehicle', 'motorcycle': 'vehicle'}


def log(*a):
    print(*a, flush=True)


# ══════════════════════ 数据加载（镜像 U.load_coco，多返回 image id） ══════════════════════

def load_coco_with_ids(cls_name):
    from pycocotools.coco import COCO
    coco = COCO(U.COCO_ANN_FILE)
    cids = coco.getCatIds(catNms=[cls_name])
    if not cids:
        return [], [], []
    cat_id = cids[0]
    iids = coco.getImgIds(catIds=[cat_id])
    images, gts, ids = [], [], []
    for iid in iids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[iid], catIds=[cat_id], iscrowd=False))
        if not anns:
            continue
        info = coco.loadImgs([iid])[0]
        p = os.path.join(U.COCO_IMG_DIR, info['file_name'])
        if not os.path.exists(p):
            continue
        try:
            im = Image.open(p).convert('RGB')
        except Exception:
            continue
        g = [[a['bbox'][0], a['bbox'][1],
              a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in anns]
        images.append(im)
        gts.append(g)
        ids.append(int(iid))
    return images, gts, ids


def v2_cached_dets(images, prompt, name):
    p = DATA / f'tge_v2_dets_COCO-COCO-{name}.pkl'
    if p.exists():
        return pickle.load(open(p, 'rb'))
    dets = U.gdino_detect(images, prompt, 0.0)
    pickle.dump(dets, open(p, 'wb'))
    return dets


def collect_features(images, gts, dets):
    """与 tge_v2_paper_main_recompute.collect_features 逐字一致。"""
    per_img, crops_all = [], []
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < PRE_FILTER:
                continue
            bl.append([x1, y1, x2, y2]); sl.append(float(sc)); ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
                        'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
                        'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
                        'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64)})
    F = T.encode_clip_batch(crops_all).cpu().numpy() if crops_all else np.zeros((0, 512))
    return per_img, F


# ══════════════════════ 逐位等价的向量化 AP50 ══════════════════════

def iou_mat(b, g):
    """逐字复制 U.iou 的数值公式（面积不裁剪, epsilon 加 1e-8）。"""
    if len(b) == 0 or len(g) == 0:
        return np.zeros((len(b), len(g)))
    x1 = np.maximum(b[:, None, 0], g[None, :, 0]); y1 = np.maximum(b[:, None, 1], g[None, :, 1])
    x2 = np.minimum(b[:, None, 2], g[None, :, 2]); y2 = np.minimum(b[:, None, 3], g[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    a1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    a2 = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-8)


def image_contrib(box, gd, gt, crop_idx, F, W):
    """返回该图的 (score 降序, tp) 两项；tp 与 U.compute_ap50_img 的逐图贪心一致。"""
    n = len(gd)
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int8)
    w = F[crop_idx] @ W
    s = np.sqrt(gd * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
    order = np.argsort(-np.asarray(s))          # 与官方逐字一致（含并列时的顺序）
    n_gt = len(gt)
    tp = np.zeros(len(order), dtype=np.int8)
    if n_gt:
        IOU = iou_mat(box, gt)                   # (n, n_gt)
        matched = np.zeros(n_gt, dtype=bool)
        for k, i in enumerate(order):
            row = IOU[i].copy()
            row[matched] = -1.0
            j = int(np.argmax(row))
            if row[j] >= 0.5:
                tp[k] = 1
                matched[j] = True
    return np.asarray(s, dtype=np.float64)[order], tp


def ap_from_pool(scores, tps, n_gt):
    """与 U.compute_ap50_img 的池化/PR/11-point 部分逐位一致。"""
    if n_gt == 0:
        return 0.0 if len(scores) else 1.0
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind='stable')   # 官方是 python 稳定排序
    tp = np.asarray(tps, dtype=np.float64)[order]
    tc = np.cumsum(tp)
    fc = np.cumsum(1.0 - tp)
    rec = tc / n_gt
    prec = tc / (tc + fc + 1e-8)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        m = rec >= t
        if m.any():
            ap += float(prec[m].max())
    return ap / 11.0


def ap50_fast(contribs, n_gt):
    if not contribs:
        return 0.0 if n_gt == 0 else 0.0
    sc = np.concatenate([c[0] for c in contribs]) if contribs else np.zeros(0)
    tp = np.concatenate([c[1] for c in contribs]) if contribs else np.zeros(0, dtype=np.int8)
    return ap_from_pool(sc, tp, n_gt)


def ap50_ref(im_preds, im_gts):
    return U.compute_ap50_img(im_preds, im_gts)


# ══════════════════════ 配对 bootstrap（在图级重采样） ══════════════════════

def _per_img_arrays(items):
    """把 [(scores, tp), ...] 预转成 numpy，避免 bootstrap 里反复转换。"""
    sc = [np.asarray(x[0], dtype=np.float64) for x in items]
    tp = [np.asarray(x[1], dtype=np.float64) for x in items]
    return sc, tp


def bootstrap_gain_ci(base_items, w_items, gt_sizes, B=1000, seed=0):
    """单类配对 bootstrap：对 test 的**图**有放回重采样，返回 B 个增益样本。

    同一张图的 base 与 W 分数**成对**重采样（配对设计），故消除了图难度差异。
    """
    n = len(base_items)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    gts = np.asarray(gt_sizes, dtype=np.int64)
    sc_b, tp_b = _per_img_arrays(base_items)
    sc_w, tp_w = _per_img_arrays(w_items)
    out = np.empty(B, dtype=np.float64)
    for b in range(B):
        pick = rng.integers(0, n, n)
        n_gt = int(gts[pick].sum())
        ab = ap_from_pool(np.concatenate([sc_b[i] for i in pick]),
                          np.concatenate([tp_b[i] for i in pick]), n_gt)
        aw = ap_from_pool(np.concatenate([sc_w[i] for i in pick]),
                          np.concatenate([tp_w[i] for i in pick]), n_gt)
        out[b] = aw - ab
    return out


def bootstrap_mean_gain(per_cls_items, B=1000, seed=0):
    """返回 B 个「六类均值增益」样本：逐类独立重采样后按类平均。

    per_cls_items: dict cls -> (base_items, w_items, gt_sizes)
    """
    rng = np.random.default_rng(seed)
    per_cls = {}
    for cls, (bi, wi, gs) in per_cls_items.items():
        arr = bootstrap_gain_ci(bi, wi, gs, B=B, seed=int(rng.integers(1 << 31)))
        if arr is not None:
            per_cls[cls] = arr
    if not per_cls:
        return None, {}
    return np.mean(np.stack(list(per_cls.values()), 0), axis=0), per_cls


def ci_stats(arr):
    if arr is None or len(arr) == 0:
        return None
    lo, hi = np.percentile(arr, [2.5, 97.5])
    n_le0 = int((arr <= 0).sum())
    return {'mean': float(arr.mean()), 'std': float(arr.std(ddof=1)),
            'lo95': float(lo), 'hi95': float(hi),
            'n_boot': int(len(arr)),
            'n_le_0': n_le0, 'n_le_0_frac': float(n_le0) / len(arr),
            'p_le_0': float((arr <= 0).mean())}


def test_items_all(blobs, per_cls, beta, lam):
    """为给定 (beta,lam) 收集 test 每类的 (base_items, w_items, gt_sizes)。"""
    items = {}
    for cls, blob in blobs.items():
        t_idx = per_cls[cls]['test_idx']
        W = make_W(blob, beta, lam)
        bi = base_contribs(blob, t_idx)
        wi = [image_contrib(blob['per_img'][i]['box'], blob['per_img'][i]['gd'],
                            blob['per_img'][i]['gt'], blob['per_img'][i]['crop_idx'],
                            blob['F'], W) for i in t_idx]
        gs = [len(blob['per_img'][i]['gt']) for i in t_idx]
        items[cls] = (bi, wi, gs)
    return items


# ══════════════════════ 构建 / 载入 per-class 缓存 ══════════════════════

def build_protos():
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        crops = []
        for img, g in zip(images, gts):
            for gt in g:
                x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                crops.append(img.crop((x1, y1, x2, y2)))
        if not crops:
            continue
        p = T.encode_clip_batch(crops).mean(dim=0)
        protos[cls] = p / p.norm()
    return protos


def build_or_load(cls, protos):
    cp = CACHE / f'{cls}.pkl'
    if cp.exists():
        log(f'  [cache] {cp.name}')
        return pickle.load(open(cp, 'rb'))
    t0 = time.time()
    images, gts, ids = load_coco_with_ids(cls)
    dets = v2_cached_dets(images, TGT_TMPL.format(cls), cls)
    assert len(dets) == len(images), f'{cls}: dets {len(dets)} != imgs {len(images)}'
    per_img, F = collect_features(images, gts, dets)
    src_cls = COCO2SRC[cls]
    P_src = protos[src_cls].to(U.DEVICE)
    P_neg = T.get_neg_proto(protos, src_cls)
    Delta = (T.text_emb([TGT_TMPL.format(cls)]).cpu()[0]
             - T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0])
    blob = {
        'cls': cls, 'src_cls': src_cls,
        'per_img': per_img, 'F': F, 'img_ids': np.array(ids, dtype=np.int64),
        'P_src': P_src.cpu().numpy(), 'P_neg': P_neg.cpu().numpy(),
        'Delta': Delta.numpy(),
        'n_box': sum(len(p['gd']) for p in per_img),
    }
    pickle.dump(blob, open(cp, 'wb'))
    log(f'  {cls}: {len(per_img)} imgs, {blob["n_box"]} boxes kept ({time.time()-t0:.0f}s)')
    return blob


# ══════════════════════ W / 评估 ══════════════════════

def make_W(blob, beta, lam):
    P_src = blob['P_src']; Delta = blob['Delta']
    P_hat = P_src + beta * Delta
    P_hat = P_hat / np.linalg.norm(P_hat)
    return P_hat - lam * blob['P_neg']


def unit(x):
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def split_indices(blob, keep_mask):
    """返回 (eval 图下标, gt 总数)。keep_mask 长度 = len(per_img)。

    eval = 前 K_CALIB 张之后的图且 keep_mask 为真。
    """
    n = len(blob['per_img'])
    idx = [i for i in range(K_CALIB, n) if keep_mask[i]]
    n_gt = sum(len(blob['per_img'][i]['gt']) for i in idx)
    return idx, n_gt


def base_contribs(blob, idx):
    """baseline 分数就是 gd（排序后），tp 全 0（不做匹配——baseline 也用同一套贪心）。

    注意：baseline 的 AP 同样要经过逐图贪心匹配，所以不能把 tp 全置 0。
    """
    out = []
    for i in idx:
        p = blob['per_img'][i]
        if len(p['gd']) == 0:
            out.append((np.zeros(0), np.zeros(0, dtype=np.int8)))
            continue
        order = np.argsort(-np.asarray(p['gd']))
        n_gt = len(p['gt'])
        tp = np.zeros(len(order), dtype=np.int8)
        if n_gt:
            IOU = iou_mat(p['box'], p['gt'])
            matched = np.zeros(n_gt, dtype=bool)
            for k, ii in enumerate(order):
                row = IOU[ii].copy(); row[matched] = -1.0
                j = int(np.argmax(row))
                if row[j] >= 0.5:
                    tp[k] = 1; matched[j] = True
        out.append((np.asarray(p['gd'], dtype=np.float64)[order], tp))
    return out


# ══════════════════════ 自检：向量化 AP 必须与官方逐位一致 ══════════════════════

def self_check(blobs, n_checks=3, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for cls, blob in blobs.items():
        n = len(blob['per_img'])
        idx = list(range(K_CALIB, n))
        n_gt = sum(len(blob['per_img'][i]['gt']) for i in idx)
        for _ in range(n_checks):
            W = unit(rng.normal(size=512))
            contribs = [image_contrib(blob['per_img'][i]['box'], blob['per_img'][i]['gd'],
                                      blob['per_img'][i]['gt'], blob['per_img'][i]['crop_idx'],
                                      blob['F'], W) for i in idx]
            fast = ap50_fast(contribs, n_gt)
            # 官方实现：用同一批图与分数
            im_preds, im_gts = [], []
            for i in idx:
                p = blob['per_img'][i]
                if len(p['gd']):
                    w = blob['F'][p['crop_idx']] @ W
                    s = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
                else:
                    s = np.zeros(0)
                im_preds.append((p['box'], s)); im_gts.append(p['gt'])
            ref = ap50_ref(im_preds, im_gts)
            worst = max(worst, abs(fast - ref))
            log(f'    [selfcheck] {cls:<11} fast={fast:.10f} ref={ref:.10f} diff={abs(fast-ref):.2e}')
        # baseline 也查一次
        bc = base_contribs(blob, idx)
        fast_b = ap50_fast(bc, n_gt)
        ref_b = ap50_ref([(blob['per_img'][i]['box'], blob['per_img'][i]['gd']) for i in idx],
                         [blob['per_img'][i]['gt'] for i in idx])
        worst = max(worst, abs(fast_b - ref_b))
        log(f'    [selfcheck] {cls:<11} BASE fast={fast_b:.10f} ref={ref_b:.10f} diff={abs(fast_b-ref_b):.2e}')
    return worst


# ══════════════════════ 主流程 ══════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='all', choices=['build', 'check', 'full', 'split', 'all'])
    ap.add_argument('--splits', type=int, default=5)
    ap.add_argument('--boot', type=int, default=1000)
    args = ap.parse_args()
    t0 = time.time()

    log('=' * 92)
    log('主配置「独立确认」实验：按 image ID 全局切分 dev/test')
    log('=' * 92)

    protos = build_protos()
    log(f'[protos] DIOR {len(protos)} 类')

    blobs = {}
    for coco_cls, _ in DEV_CLASSES:
        blobs[coco_cls] = build_or_load(coco_cls, protos)
    log(f'[data] 六类共 {sum(len(b["per_img"]) for b in blobs.values())} 张图, '
        f'{sum(b["n_box"] for b in blobs.values())} 个候选框（pre-filter>= {PRE_FILTER}）')

    if args.mode in ('check', 'all'):
        log('\n[自检] 向量化 AP50 vs 官方 U.compute_ap50_img ...')
        worst = self_check(blobs)
        log(f'[自检] 最大绝对差 = {worst:.3e}  (要求 < 1e-12)')
        assert worst < 1e-12, '向量化 AP 与官方不等价，停手'
        log('[自检] 通过')

    # ── 全量复现（验证管线能对上论文/log） ──
    full = {}
    if args.mode in ('full', 'all'):
        log('\n[全量] 1291 图 (前 K_CALIB=10 张排除) —— 复现论文主数字')
        for cls, blob in blobs.items():
            idx = list(range(K_CALIB, len(blob['per_img'])))
            n_gt = sum(len(blob['per_img'][i]['gt']) for i in idx)
            bc = base_contribs(blob, idx)
            b = ap50_fast(bc, n_gt)
            W = make_W(blob, 2.0, 0.5)
            a = ap50_fast([image_contrib(blob['per_img'][i]['box'], blob['per_img'][i]['gd'],
                                         blob['per_img'][i]['gt'], blob['per_img'][i]['crop_idx'],
                                         blob['F'], W) for i in idx], n_gt)
            full[cls] = {'n_img': len(idx), 'n_gt': n_gt, 'base_ap': float(b),
                         'ap_2.0_0.5': float(a), 'gain': float(a - b)}
            log(f'  {cls:<11} n={len(idx):4d} base={b:.4f}  +delta={a-b:+.4f}   '
                f'(log: base 见下)')
        gm = float(np.mean([v['gain'] for v in full.values()]))
        gb = float(np.mean([v['base_ap'] for v in full.values()]))
        log(f'  六类均值: base={gb:.4f}  gain={gm:+.4f}')
        log('  论文/log 对照: base=[0.7484,0.1758,0.1245,0.2345,0.5317,0.4092] '
            'gain=[+0.0117,+0.0253,+0.0809,+0.0774,+0.0735,+0.1189] mean=+0.0646')

    res = {'config': {'K_CALIB': K_CALIB, 'pre_filter': PRE_FILTER,
                      'beta_grid': BETA_GRID, 'lam_grid': LAM_GRID,
                      'detector': 'gdino-tiny', 'clip': 'ViT-B/32 (openai)',
                      'metric': 'dataset-level per-image-matched 11-point AP50',
                      'elapsed_s': None},
           'full_1291': full}

    # ── 全局切分实验 ──
    if args.mode in ('split', 'all'):
        # 只在「实际参与评测」的唯一图上切（= 论文所述 930 张），
        # 使 dev+test 的 class-image pairs 恰好加总为 1291。
        all_ids = np.unique(np.concatenate([b['img_ids'][K_CALIB:] for b in blobs.values()]))
        log(f'\n[切分] 评测集唯一 COCO image id = {len(all_ids)} 个 '
            f'(论文口径 930；class-image pairs 合计 1291)')

        splits_out = []
        for seed in range(args.splits):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(all_ids))
            dev_ids = set(all_ids[perm[:len(perm) // 2]].tolist())
            test_ids = set(all_ids[perm[len(perm) // 2:]].tolist())
            assert not (dev_ids & test_ids)

            per_cls = {}
            for cls, blob in blobs.items():
                dev_mask = np.array([int(i) in dev_ids for i in blob['img_ids']])
                test_mask = ~dev_mask
                d_idx, d_ngt = split_indices(blob, dev_mask)
                t_idx, t_ngt = split_indices(blob, test_mask)
                # dev 上按论文同规则选 (beta, lam)
                dev_gains = {}
                for be in BETA_GRID:
                    for la in LAM_GRID:
                        W = make_W(blob, be, la)
                        g = ap50_fast([image_contrib(blob['per_img'][i]['box'],
                                                     blob['per_img'][i]['gd'],
                                                     blob['per_img'][i]['gt'],
                                                     blob['per_img'][i]['crop_idx'],
                                                     blob['F'], W) for i in d_idx], d_ngt)
                        dev_gains[(be, la)] = g
                dbc = base_contribs(blob, d_idx)
                d_base = ap50_fast(dbc, d_ngt)
                best = max(dev_gains, key=lambda k: dev_gains[k] - d_base)
                per_cls[cls] = {'dev_idx': d_idx, 'dev_ngt': d_ngt,
                                'test_idx': t_idx, 'test_ngt': t_ngt,
                                'dev_base_ap': float(d_base),
                                'dev_gain_selected': float(dev_gains[best] - d_base),
                                'selected': [float(best[0]), float(best[1])]}
            # 逐类选出的 pair 汇总 -> 用"dev 平均增益最大的全局 pair"更贴近论文（一个 pair 服务全部类）
            pair_scores = {}
            for be in BETA_GRID:
                for la in LAM_GRID:
                    gs = []
                    for cls, blob in blobs.items():
                        pc = per_cls[cls]
                        W = make_W(blob, be, la)
                        a = ap50_fast([image_contrib(blob['per_img'][i]['box'],
                                                     blob['per_img'][i]['gd'],
                                                     blob['per_img'][i]['gt'],
                                                     blob['per_img'][i]['crop_idx'],
                                                     blob['F'], W) for i in pc['dev_idx']],
                                      pc['dev_ngt'])
                        gs.append(a - pc['dev_base_ap'])
                    pair_scores[(be, la)] = float(np.mean(gs))
            sel_pair = max(pair_scores, key=lambda k: pair_scores[k])
            # 并列最优检测（max() 取首次出现 → 网格顺序：β 升序、λ 升序）
            _best = pair_scores[sel_pair]
            dev_ties = [list(k) for k, v in pair_scores.items() if abs(v - _best) < 1e-12]

            # test 上确认
            def test_gain(beta, lam):
                gs, per = [], {}
                for cls, blob in blobs.items():
                    pc = per_cls[cls]
                    t_idx = pc['test_idx']; t_ngt = pc['test_ngt']
                    bc = base_contribs(blob, t_idx)
                    b = ap50_fast(bc, t_ngt)
                    W = make_W(blob, beta, lam)
                    a = ap50_fast([image_contrib(blob['per_img'][i]['box'],
                                                 blob['per_img'][i]['gd'],
                                                 blob['per_img'][i]['gt'],
                                                 blob['per_img'][i]['crop_idx'],
                                                 blob['F'], W) for i in t_idx], t_ngt)
                    gs.append(a - b)
                    per[cls] = {'base_ap': float(b), 'gain': float(a - b)}
                return float(np.mean(gs)), per

            tg_sel, per_sel = test_gain(*sel_pair)
            tg_main, per_main = test_gain(2.0, 0.5)
            tg_tind, per_tind = test_gain(1.0, 0.25)

            # 配对 bootstrap（图级重采样）——评审要的 CI 与 P(Delta<=0)
            boot_sel_arr, boot_sel_cls = bootstrap_mean_gain(
                test_items_all(blobs, per_cls, *sel_pair), B=args.boot, seed=1000 + seed)
            boot_main_arr, _ = bootstrap_mean_gain(
                test_items_all(blobs, per_cls, 2.0, 0.5), B=args.boot, seed=2000 + seed)
            boot_sel = ci_stats(boot_sel_arr)
            boot_main = ci_stats(boot_main_arr)
            boot_sel_per_cls = {c: ci_stats(v) for c, v in boot_sel_cls.items()}

            # test 上的完整网格面：看 dev 选出的点是否同时是 test 最优（排除"碰巧"）
            test_grid = {}
            for be in BETA_GRID:
                for la in LAM_GRID:
                    W = None
                    gs = []
                    for cls, blob in blobs.items():
                        Wc = make_W(blob, be, la)
                        t_idx = per_cls[cls]['test_idx']
                        t_ngt = per_cls[cls]['test_ngt']
                        b = ap50_fast(base_contribs(blob, t_idx), t_ngt)
                        a = ap50_fast([image_contrib(blob['per_img'][i]['box'],
                                                     blob['per_img'][i]['gd'],
                                                     blob['per_img'][i]['gt'],
                                                     blob['per_img'][i]['crop_idx'],
                                                     blob['F'], Wc) for i in t_idx], t_ngt)
                        gs.append(a - b)
                    test_grid[(be, la)] = float(np.mean(gs))
            test_argmax = max(test_grid, key=lambda k: test_grid[k])
            _tbest = test_grid[test_argmax]
            test_ties = [list(k) for k, v in test_grid.items() if abs(v - _tbest) < 1e-12]
            # 逐类 dev/test 样本数（复现用）
            per_cls_counts = {c: {'dev_pairs': len(per_cls[c]['dev_idx']),
                                  'test_pairs': len(per_cls[c]['test_idx']),
                                  'dev_ngt': per_cls[c]['dev_ngt'],
                                  'test_ngt': per_cls[c]['test_ngt']}
                              for c in blobs}
            n_dev = sum(len(per_cls[c]['dev_idx']) for c in blobs)
            n_test = sum(len(per_cls[c]['test_idx']) for c in blobs)
            assert n_dev + n_test == 1291, f'dev+test={n_dev+n_test} != 1291'
            log(f'\n  [split {seed}] dev={n_dev} 图 / test={n_test} 图')
            log(f'    dev 选出 (beta,lam) = {sel_pair}   dev gain={pair_scores[sel_pair]:+.4f}')
            log(f'    TEST: selected={tg_sel:+.4f}   (2.0,0.5)={tg_main:+.4f}   (1.0,0.25)={tg_tind:+.4f}')
            if boot_sel:
                log(f'    bootstrap(sel): mean={boot_sel["mean"]:+.4f} '
                    f'95%CI=[{boot_sel["lo95"]:+.4f},{boot_sel["hi95"]:+.4f}] '
                    f'非正={boot_sel["n_le_0"]}/{boot_sel["n_boot"]}')
            if boot_main:
                log(f'    bootstrap(2.0,0.5): mean={boot_main["mean"]:+.4f} '
                    f'95%CI=[{boot_main["lo95"]:+.4f},{boot_main["hi95"]:+.4f}] '
                    f'非正={boot_main["n_le_0"]}/{boot_main["n_boot"]}')
            log(f'    TEST 网格最优 pair = {test_argmax} (gain={test_grid[test_argmax]:+.4f})'
                f'   dev 选出={sel_pair}')
            log(f'    dev 并列最优数={len(dev_ties)}  test 并列最优数={len(test_ties)}')
            splits_out.append({
                'seed': seed, 'n_dev_img': n_dev, 'n_test_img': n_test,
                'dev_selected': [float(sel_pair[0]), float(sel_pair[1])],
                'dev_gain_selected': float(pair_scores[sel_pair]),
                'test_gain_selected': tg_sel, 'test_per_class_selected': per_sel,
                'test_gain_2.0_0.5': tg_main, 'test_per_class_2.0_0.5': per_main,
                'test_gain_1.0_0.25': tg_tind, 'test_per_class_1.0_0.25': per_tind,
                'boot_selected': boot_sel, 'boot_2.0_0.5': boot_main,
                'boot_selected_per_class': boot_sel_per_cls,
                'per_class_selected_on_dev': {c: per_cls[c]['selected'] for c in per_cls},
                'dev_gain_grid': {f'{b}_{l}': v for (b, l), v in pair_scores.items()},
                'test_gain_grid': {f'{b}_{l}': v for (b, l), v in test_grid.items()},
                'test_argmax_pair': [float(test_argmax[0]), float(test_argmax[1])],
                'test_gain_at_argmax': float(test_grid[test_argmax]),
                'dev_selection_ties': dev_ties,
                'test_argmax_ties': test_ties,
                'per_class_counts': per_cls_counts,
                'dev_image_ids': [int(v) for v in all_ids[perm[:len(perm) // 2]]],
                'test_image_ids': [int(v) for v in all_ids[perm[len(perm) // 2:]]],
            })

        sel_gains = np.array([s['test_gain_selected'] for s in splits_out])
        main_gains = np.array([s['test_gain_2.0_0.5'] for s in splits_out])
        test_argmaxes = [tuple(s['test_argmax_pair']) for s in splits_out]
        sel_ple0 = np.array([s['boot_selected']['p_le_0'] for s in splits_out
                             if s.get('boot_selected')])
        sel_lo = np.array([s['boot_selected']['lo95'] for s in splits_out
                           if s.get('boot_selected')])
        sel_hi = np.array([s['boot_selected']['hi95'] for s in splits_out
                           if s.get('boot_selected')])
        res['splits'] = splits_out
        res['summary'] = {
            'n_splits': len(splits_out),
            'test_gain_selected_mean': float(sel_gains.mean()),
            'test_gain_selected_std': float(sel_gains.std(ddof=1)) if len(sel_gains) > 1 else 0.0,
            'test_gain_selected_min': float(sel_gains.min()),
            'test_gain_selected_max': float(sel_gains.max()),
            'test_gain_selected_pos': int((sel_gains > 0).sum()),
            'test_gain_2.0_0.5_mean': float(main_gains.mean()),
            'test_gain_2.0_0.5_min': float(main_gains.min()),
            'test_gain_2.0_0.5_pos': int((main_gains > 0).sum()),
            'boot_selected_p_le_0_max': float(sel_ple0.max()) if len(sel_ple0) else None,
            'boot_selected_lo95_min': float(sel_lo.min()) if len(sel_lo) else None,
            'boot_selected_hi95_max': float(sel_hi.max()) if len(sel_hi) else None,
            'test_argmax_pairs': [[float(t[0]), float(t[1])] for t in test_argmaxes],
            'dev_selected_pairs': [[float(s['dev_selected'][0]), float(s['dev_selected'][1])]
                                   for s in splits_out],
        }
        log('\n' + '=' * 92)
        log('[汇总] 全局切分独立确认')
        log('=' * 92)
        log(f'  test 增益（dev 选参）: mean={sel_gains.mean():+.4f} '
            f'std={sel_gains.std(ddof=1) if len(sel_gains)>1 else 0:.4f} '
            f'range=[{sel_gains.min():+.4f},{sel_gains.max():+.4f}] '
            f'正={int((sel_gains>0).sum())}/{len(sel_gains)}')
        log(f'  test 增益（固定 2.0,0.5）: mean={main_gains.mean():+.4f} '
            f'range=[{main_gains.min():+.4f},{main_gains.max():+.4f}] '
            f'正={int((main_gains>0).sum())}/{len(main_gains)}')
        if len(sel_ple0):
            log(f'  bootstrap: max P(d<=0)={sel_ple0.max():.3f}  '
                f'CI_lo 最小={sel_lo.min():+.4f}  CI_hi 最大={sel_hi.max():+.4f}')
        log(f'  各 split 选出的 pair: {[s["dev_selected"] for s in splits_out]}')
        log(f'  各 split 的 test 网格最优: '
            f'{[[float(t[0]), float(t[1])] for t in test_argmaxes]}')

    res['config']['elapsed_s'] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    log(f'\n→ {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
