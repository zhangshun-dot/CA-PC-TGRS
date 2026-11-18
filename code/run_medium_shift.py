# -*- coding: utf-8 -*-
"""run_medium_shift.py — 非遥感独立域验证 v2：媒介迁移轴（photo → 非-photo 媒介）

为什么重做（元 2026-10-09 20:47「要」）：
  退化轴(fog/blur/brightness)失败的原因是**轴选错**——CA-PC 的锚对语义是
  「photo ↔ overhead」的**概念/媒介差**，而退化轴只是同一 photo 域的画质差，
  锚对在该轴上语义上不存在。本脚本换成**与锚对语义同构**的非遥感轴：
      source = COCO 照片       锚: "a photo of a {c}"
      target = 同图媒介化(素描)  锚: "a pencil sketch of a {c}"
  这与「a photo of X ↔ an overhead image of X」是同一构造（条件短语对），
  而且 6 个类在 COCO 上有 GT 框，可直接算检测 AP，无需新数据集。

三个必须项（对准上一次的设计错误）：
  1. 真实概念差：先测 base 在媒介图上是否**大幅掉**（→ 有 headroom），
     并测「媒介文本原型 vs 照片文本原型」谁更贴近媒介 crop 特征（→ 锚对确有意义）。
  2. norm-matched 随机方向对照（N=49）：real 方向 gain 在 {real}∪{random} 中的 rank
     → p_c；这是**合法零假设**（上次用错锚轴，不是合法 null）。
  3. 错媒介对照：用**另一种媒介**的方向（如 cartoon 方向打素描目标）→ 方向一致性。

用法：
  HF_HUB_OFFLINE=1 "C:/Users/张三/vg/Scripts/python.exe" -u run_medium_shift.py \
      --k 20 --media sketch cartoon --dets gdino
"""
import os, sys, json, time, pickle, argparse
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cuda')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
import run_bidirectional as RB
import run_cross_vlm as CV
os.environ['HF_HUB_OFFLINE'] = '1'

OUT = r'C:\Users\张三\Desktop\实验数据\run_medium_shift.json'
MCACHE = r'C:\Users\张三\Desktop\实验数据\medium_det_cache'
os.makedirs(MCACHE, exist_ok=True)

N_PROTO, BIG_N = CV.N_PROTO, CV.BIG_N
BETA, LAM, NEG_TOPK = CV.BETA, CV.LAM, CV.NEG_TOPK
BETA_GRID = CV.BETA_GRID
N_RANDOM = 49
RNG_SEED = 20261009

# 媒介锚（与 "an overhead image of {c}" 同构的条件短语）
MEDIUM_ANCHOR = {
    'sketch':   ('a photo of a {c}',    'a pencil sketch of a {c}'),
    'cartoon':  ('a photo of a {c}',    'a cartoon drawing of a {c}'),
    'painting': ('a photo of a {c}',    'an oil painting of a {c}'),
}


# ── 媒介算子（确定性；纯 PIL/numpy，不需要 GPU） ──
def _dodge(a, b):
    return np.clip(a * 255.0 / np.maximum(255.0 - b, 1e-3), 0, 255)


def medium_transform(pil, name):
    """把照片变成非-photo 媒介；同一输入必得同一输出。"""
    im = pil.convert('RGB')
    if name == 'sketch':
        g = np.asarray(ImageOps.grayscale(im)).astype(np.float32)
        inv = 255.0 - g
        blur = np.asarray(Image.fromarray(inv.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=max(2.0, min(im.size) / 120.0)))).astype(np.float32)
        sk = _dodge(g, blur)
        return Image.fromarray(sk.astype(np.uint8)).convert('RGB')
    if name == 'cartoon':
        # 强中值 + 色阶压缩 + 边缘加深，近似卡通化
        sm = im.filter(ImageFilter.MedianFilter(size=5))
        arr = np.asarray(sm).astype(np.float32)
        q = np.round(arr / 48.0) * 48.0                     # 色阶量化
        edge = np.asarray(ImageOps.grayscale(im).filter(ImageFilter.FIND_EDGES)).astype(np.float32)
        m = np.clip(1.0 - edge / 90.0, 0.0, 1.0)[..., None]  # 边缘处变暗
        return Image.fromarray(np.clip(q * m, 0, 255).astype(np.uint8)).convert('RGB')
    if name == 'painting':
        sm = im.filter(ImageFilter.GaussianBlur(radius=1.2))
        arr = np.asarray(sm).astype(np.float32)
        q = np.round(arr / 32.0) * 32.0                      # 油画色块
        return Image.fromarray(np.clip(q, 0, 255).astype(np.uint8)).convert('RGB')
    raise ValueError(name)


def unit(x):
    return CV.unit(x)


def score_blob(per_img, F, W):
    return CV.eval_pair(per_img, F, W)


def rand_dirs(dim, n, seed):
    rs = np.random.RandomState(seed)
    v = rs.randn(n, dim).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dets', nargs='+', default=['gdino'])
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--media', nargs='+', default=['sketch'])
    ap.add_argument('--classes', nargs='+', default=list(RB.COCO_POOL))
    ap.add_argument('--no-clean', action='store_true', help='跳过 clean 参考(已缓存时)')
    args = ap.parse_args()
    t0 = time.time()

    cfg = RB.Config()
    cfg.model.device = CV.DEVICE
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'
    cfg.model.detection_threshold = 0.0

    vlm = CV.VLM('clip_b32')                       # fp32，与全文表格同口径

    # ── 图片集 + 源原型池 ──
    data, proto = {}, {}
    for cls in args.classes:
        ims, gts = RB.load_split(cfg, 'coco', cls, N_PROTO, BIG_N)
        data[cls] = (ims[:args.k], gts[:args.k])
        gc_ = RB.D.gt_crops(data[cls][0], data[cls][1])
        Ft = vlm.enc_img(gc_)
        pt = torch.from_numpy(Ft.mean(0))
        proto[cls] = pt / pt.norm()
        print(f'[proto] {cls}: {len(data[cls][0])} imgs, {len(gc_)} crops', flush=True)

    def pneg_for(cls):
        return unit(RB.Proto.get_negative_prototype(proto, cls, NEG_TOPK).numpy())

    def run_detect(cls, imgs):
        gts = data[cls][1]
        dets = RB.M.gdino_detect(cfg, imgs, f'a photo of {cls}', cfg.model.detection_threshold)
        per_img = []
        for (boxes, scores), img, g in zip(dets, imgs, gts):
            bl, sl = [], []
            for box, sc in zip(boxes, scores):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                if float(sc) < RB.PRE_FILTER:
                    continue
                bl.append([x1, y1, x2, y2]); sl.append(float(sc))
            per_img.append({'box': np.asarray(bl, float), 'gd': np.asarray(sl, float),
                            'gt': np.asarray(g, float), 'crop_idx': np.arange(len(bl))})
        return per_img

    rows = []
    for det in args.dets:
        for cls in args.classes:
            ims, gts = data[cls]
            conds = ([('clean', None)] if not args.no_clean else []) + \
                    [(m, m) for m in args.media]
            for (tag, med) in conds:
                imgs = list(ims) if med is None else [medium_transform(im, med) for im in ims]
                ck = f'{det}__{cls}__{tag}__k{len(ims)}.pkl'
                cp = os.path.join(MCACHE, ck)
                if os.path.exists(cp):
                    blob = pickle.load(open(cp, 'rb'))
                    per_img, F = blob['per_img'], blob['F']
                else:
                    per_img = run_detect(cls, imgs)
                    crops = []
                    for img, p in zip(imgs, per_img):
                        crops.extend(CV.crops_from_cached(img, p))
                    F = vlm.enc_img(crops)
                    pickle.dump({'per_img': per_img, 'F': F}, open(cp, 'wb'))
                    print(f'  [det] {det} {cls:<10} {tag}: {len(imgs)} imgs, '
                          f'{sum(len(p["gd"]) for p in per_img)} boxes', flush=True)

                P_src = unit(proto[cls].numpy())
                P_neg = pneg_for(cls)
                b50, bm = score_blob(per_img, F, P_src - LAM * P_neg)
                rec = {'det': det, 'cls': cls, 'medium': med or 'clean',
                       'n_img': len(per_img), 'n_box': int(sum(len(p['gd']) for p in per_img)),
                       'base': round(b50, 4), 'base_m': round(bm, 4)}

                if med is not None:
                    ph_t = vlm.enc_txt([MEDIUM_ANCHOR[med][0].format(c=cls)])[0]
                    md_t = vlm.enc_txt([MEDIUM_ANCHOR[med][1].format(c=cls)])[0]
                    dc = md_t - ph_t
                    # 诊断：媒介文本原型是否比照片文本原型更贴近媒介 crop 特征
                    align_md = float(np.mean(F @ md_t))
                    align_ph = float(np.mean(F @ ph_t))
                    W_naive = unit(md_t) - LAM * P_neg
                    W_tge = unit(P_src + dc) - LAM * P_neg
                    W_ca = unit(P_src + BETA * dc) - LAM * P_neg
                    n50, nm = score_blob(per_img, F, W_naive)
                    t50, tm = score_blob(per_img, F, W_tge)
                    c50, cm = score_blob(per_img, F, W_ca)
                    # 错媒介方向（另一种媒介的方向打本媒介目标）
                    cw50, cwm = float('nan'), float('nan')
                    for other in args.media:
                        if other == med:
                            continue
                        o_md = vlm.enc_txt([MEDIUM_ANCHOR[other][1].format(c=cls)])[0]
                        o_ph = vlm.enc_txt([MEDIUM_ANCHOR[other][0].format(c=cls)])[0]
                        cw50, cwm = score_blob(per_img, F, unit(P_src + BETA * (o_md - o_ph)) - LAM * P_neg)
                        break
                    # norm-matched 随机方向对照
                    s_norm = BETA * float(np.linalg.norm(dc))
                    RD = rand_dirs(dc.shape[0], N_RANDOM, RNG_SEED + hash((cls, med)) % 1000)
                    rnd = [score_blob(per_img, F, unit(P_src + s_norm * r) - LAM * P_neg)[0] for r in RD]
                    rnd = np.asarray(rnd, float)
                    real_gain = c50 - b50
                    better = int(np.sum(rnd >= c50))
                    rank = better + 1
                    best, bestm, bestb = -1.0, 0.0, 0
                    for bb in BETA_GRID:
                        a50, m = score_blob(per_img, F, unit(P_src + bb * dc) - LAM * P_neg)
                        if a50 > best:
                            best, bestm, bestb = a50, m, bb
                    rec.update({'dc_mag': round(float(np.linalg.norm(dc)), 4),
                                'align_medium': round(align_md, 4), 'align_photo': round(align_ph, 4),
                                'naive': round(n50, 4), 'tge': round(t50, 4),
                                'ca_pc': round(c50, 4), 'ca_pc_m': round(cm, 4),
                                'ca_wrong_medium': (None if cw50 != cw50 else round(cw50, 4)),
                                'rand_mean': round(float(rnd.mean()), 4),
                                'rand_std': round(float(rnd.std()), 4),
                                'rand_best': round(float(rnd.max()), 4),
                                'rank': rank, 'n_rand': N_RANDOM, 'p_c': round(rank / (N_RANDOM + 1), 4),
                                'ca_star': round(best, 4), 'ca_star_m': round(bestm, 4),
                                'star_beta': bestb,
                                # 与全局 BETA 网格匹配的 11 点网格
                                'grid': {str(b): round(score_blob(
                                    per_img, F, unit(P_src + b * dc) - LAM * P_neg)[0], 4) for b in BETA_GRID}})
                rows.append(rec)
                if med is None:
                    print(f'  {det} {cls:<10} clean  base={rec["base"]:.4f} (n_box={rec["n_box"]})', flush=True)
                else:
                    print(f'  {det} {cls:<10} {tag:<9} base={rec["base"]:.4f} '
                          f'ca_pc={rec["ca_pc"]:.4f}(Δ{rec["ca_pc"]-rec["base"]:+.4f}) '
                          f'TGE={rec["tge"]:.4f} wrongMed={rec["ca_wrong_medium"]} '
                          f'rand={rec["rand_mean"]:.4f} rank={rec["rank"]}/{N_RANDOM+1}(p={rec["p_c"]}) '
                          f'| align md={rec["align_medium"]:.4f} ph={rec["align_photo"]:.4f} '
                          f'||dc||={rec["dc_mag"]:.3f}', flush=True)
                json.dump({'config': {'beta': BETA, 'lam': LAM, 'k': args.k,
                                      'media': args.media, 'n_random': N_RANDOM, 'partial': True},
                           'rows': rows}, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    json.dump({'config': {'beta': BETA, 'lam': LAM, 'k': args.k, 'media': args.media,
                          'dets': args.dets, 'n_random': N_RANDOM,
                          'note': 'non-RS medium-transfer axis (COCO photo -> non-photo medium); '
                                  'anchor pair is a condition-phrase pair, structurally identical to '
                                  'photo<->overhead; norm-matched random directions give the null'},
               'rows': rows, 'elapsed_s': round(time.time() - t0, 1)},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
