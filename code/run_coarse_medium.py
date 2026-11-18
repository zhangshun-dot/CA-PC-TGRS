# -*- coding: utf-8 -*-
"""run_coarse_medium.py — 非遥感独立域验证 v3（修正设计错误版）

★ 为什么 v2 是错的（元 2026-10-09 21:11 质疑"会不会实验设计错了" → 查实确有其事）：
  论文主管线里 dc 的两个词是**不同类**的：
      dc = E('a photo of {eval_cls}') − E('an overhead image of {src_cls}')
  且源域词表是**粗粒度**的（DIOR 只有 vehicle / ship / airplane），目标词表是**细粒度**
  （COCO 有 car/truck/bus/motorcycle）。所以 dc 同时携带
      [类迁移项] 类名从粗到细（vehicle→truck）  +  [模板项] 观测条件（overhead→photo）
  我 v1(退化)/v2(媒介) 的探针都用了**同类** dc（class 项恒为 0）→ 只测了其中一个分量。
  已有实测(probe_decomposition.json, 9 格, GDINO n=100)：
      full +0.0623(9/9正) / class_only +0.0377(8/9正) / tmpl_src +0.0293(6/9正)
      rand_mean −0.0333(仅3/9正) / neg_full −0.1158(仅2/9正)
  → 类迁移项约占全量 60%；方向特异性在**正确轴**上其实是成立的（真方向 vs 随机 vs 反向）。
  v1/v2 的零结果因此是**设计错误的必然产物**，不构成对方法的反证。

本脚本（v3）：把非遥感媒介轴做成**与主管线同构**——源域给**粗词表**，目标域给**细词表**：
  source 原型 P_src = COCO 粗类(如 vehicle = car+truck+bus+motorcycle 的 GT crop 合并) 的 CLIP 均值
  target 图      = 同一批细类图做媒介变换(素描/卡通)  ← 非遥感、真媒介差
  方向四项对照：
     full       = E('a sketch of {fine}') − E('a photo of {coarse}')   ← 主管线同构
     class_only = E('a photo of {fine}')  − E('a photo of {coarse}')   ← 只留类迁移
     tmpl_only  = E('a sketch of {coarse}') − E('a photo of {coarse}') ← 只留媒介(类别内)
     neg_full   = −full
  + 49 个 norm-matched 随机方向 → rank/p_c（合法零假设）
预测（若 v1/v2 的失败确由"class 项被清零"导致）：
     class_only 与 full 应为正；tmpl_only 应接近 0；同类无词表差的粗类(airplane/boat)应接近 0。

用法：
  HF_HUB_OFFLINE=1 "C:/Users/张三/vg/Scripts/python.exe" -u run_coarse_medium.py \
      --k 60 --media sketch cartoon --dets gdino
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

OUT = r'C:\Users\张三\Desktop\实验数据\run_coarse_medium.json'
MCACHE = r'C:\Users\张三\Desktop\实验数据\medium_det_cache'   # 与 v2 共用缓存(键同格式)
os.makedirs(MCACHE, exist_ok=True)

N_PROTO, BIG_N = CV.N_PROTO, CV.BIG_N
BETA, LAM, NEG_TOPK = CV.BETA, CV.LAM, CV.NEG_TOPK
N_RANDOM = 49
RNG_SEED = 20261009

# ── 粗→细词表（镜像 DIOR 的粗词表） ──
COARSE = {
    'vehicle':  ['car', 'truck', 'bus', 'motorcycle'],   # DIOR 的 vehicle 覆盖这 4 类
    'ship':     ['boat'],                                 # DIOR ship ≈ COCO boat（同类物体，词不同）
    'airplane': ['airplane'],                             # 词表无差 → 对照组（预期≈0）
}
MEDIUM_ANCHOR = {
    'sketch':  'a pencil sketch of a {c}',
    'cartoon': 'a cartoon drawing of a {c}',
    'painting': 'an oil painting of a {c}',
}
PHOTO = 'a photo of a {c}'


def _dodge(a, b):
    return np.clip(a * 255.0 / np.maximum(255.0 - b, 1e-3), 0, 255)


def medium_transform(pil, name):
    im = pil.convert('RGB')
    if name == 'sketch':
        g = np.asarray(ImageOps.grayscale(im)).astype(np.float32)
        inv = 255.0 - g
        blur = np.asarray(Image.fromarray(inv.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=max(2.0, min(im.size) / 120.0)))).astype(np.float32)
        return Image.fromarray(_dodge(g, blur).astype(np.uint8)).convert('RGB')
    if name == 'cartoon':
        sm = im.filter(ImageFilter.MedianFilter(size=5))
        arr = np.asarray(sm).astype(np.float32)
        q = np.round(arr / 48.0) * 48.0
        edge = np.asarray(ImageOps.grayscale(im).filter(ImageFilter.FIND_EDGES)).astype(np.float32)
        m = np.clip(1.0 - edge / 90.0, 0.0, 1.0)[..., None]
        return Image.fromarray(np.clip(q * m, 0, 255).astype(np.uint8)).convert('RGB')
    if name == 'painting':
        arr = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=1.2))).astype(np.float32)
        return Image.fromarray(np.clip(np.round(arr / 32.0) * 32.0, 0, 255).astype(np.uint8)).convert('RGB')
    raise ValueError(name)


def unit(x):
    return CV.unit(x)


def rand_dirs(dim, n, seed):
    rs = np.random.RandomState(seed)
    v = rs.randn(n, dim).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dets', nargs='+', default=['gdino'])
    ap.add_argument('--k', type=int, default=60)
    ap.add_argument('--media', nargs='+', default=['sketch', 'cartoon'])
    ap.add_argument('--coarse', nargs='+', default=list(COARSE.keys()))
    args = ap.parse_args()
    t0 = time.time()

    cfg = RB.Config()
    cfg.model.device = CV.DEVICE
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'
    cfg.model.detection_threshold = 0.0
    vlm = CV.VLM('clip_b32')

    # ── 目标细类图集（源原型用粗类合并，而目标是细类图） ──
    fine_imgs = {}
    for coarse, fine_list in COARSE.items():
        if coarse not in args.coarse:
            continue
        for f in fine_list:
            if f in fine_imgs:
                continue
            ims, gts = RB.load_split(cfg, 'coco', f, N_PROTO, BIG_N)
            fine_imgs[f] = (ims[:args.k], gts[:args.k])
    # ── 粗类原型：把该粗类下所有细类的 COCO GT crop 合并后取 CLIP 均值 ──
    coarse_proto, coarse_bank = {}, {}
    for coarse, fine_list in COARSE.items():
        if coarse not in args.coarse:
            continue
        cr = []
        for f in fine_list:
            ims, gts = RB.load_split(cfg, 'coco', f, N_PROTO, BIG_N)
            cr.extend(RB.D.gt_crops(ims, gts))
        Ft = vlm.enc_img(cr)
        pt = torch.from_numpy(Ft.mean(0))
        coarse_proto[coarse] = pt / pt.norm()
        print(f'[proto] coarse "{coarse}" <- {fine_list}: {len(cr)} crops', flush=True)
    for coarse in coarse_proto:
        coarse_bank[coarse] = {'__self__': coarse_proto[coarse]}
    # 负原型：用**其它粗类**做负样本
    other = list(coarse_proto.keys())

    def pneg_for(coarse):
        negs = [coarse_proto[o] for o in other if o != coarse]
        if not negs:
            return torch.zeros_like(coarse_proto[coarse])
        m = torch.stack(negs).mean(0)
        return unit(m.numpy())

    def run_detect(fine, imgs):
        gts = fine_imgs[fine][1]
        dets = RB.M.gdino_detect(cfg, imgs, f'a photo of {fine}', cfg.model.detection_threshold)
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
    # ── resume: 载入已完成的格子, 避免重跑被杀后丢失进度 (元 2026-10-09 加固) ──
    done_keys = set()
    if os.path.exists(OUT):
        try:
            with open(OUT, encoding='utf-8') as _f:
                _ex = json.load(_f)
            rows = _ex.get('rows', [])
            done_keys = {(r['det'], r['coarse'], r['fine'], r['medium']) for r in rows}
            print(f'[resume] 载入 {len(rows)} 个已完成格子, 将跳过它们', flush=True)
        except Exception as _e:
            print(f'[resume] 载入失败({_e}), 从头跑', flush=True)
    for det in args.dets:
        for coarse, fine_list in COARSE.items():
            if coarse not in args.coarse:
                continue
            for fine in fine_list:
                ims, gts = fine_imgs[fine]
                conds = [('clean', None)] + [(m, m) for m in args.media]
                for (tag, med) in conds:
                    key = (det, coarse, fine, med or 'clean')
                    if key in done_keys:
                        print(f'  [skip] {det} {coarse}->{fine} {med or "clean"} (已完成)', flush=True)
                        continue
                    imgs = list(ims) if med is None else [medium_transform(im, med) for im in ims]
                    ck = f'{det}__{fine}__{tag}__k{len(ims)}.pkl'
                    cp = os.path.join(MCACHE, ck)
                    if os.path.exists(cp):
                        blob = pickle.load(open(cp, 'rb'))
                        per_img, F = blob['per_img'], blob['F']
                    else:
                        per_img = run_detect(fine, imgs)
                        crops = []
                        for img, p in zip(imgs, per_img):
                            crops.extend(CV.crops_from_cached(img, p))
                        F = vlm.enc_img(crops)
                        pickle.dump({'per_img': per_img, 'F': F}, open(cp, 'wb'))
                        print(f'  [det] {det} {coarse}->{fine:<10} {tag}: {len(imgs)} imgs, '
                              f'{sum(len(p["gd"]) for p in per_img)} boxes', flush=True)

                    P_src = unit(coarse_proto[coarse].numpy())
                    P_neg = pneg_for(coarse)
                    b50, bm = CV.eval_pair(per_img, F, P_src - LAM * P_neg)
                    rec = {'det': det, 'coarse': coarse, 'fine': fine, 'medium': med or 'clean',
                           'n_img': len(per_img), 'n_box': int(sum(len(p['gd']) for p in per_img)),
                           'base': round(b50, 4), 'base_m': round(bm, 4)}
                    if med is not None:
                        ph_fine = vlm.enc_txt([PHOTO.format(c=fine)])[0]
                        ph_coarse = vlm.enc_txt([PHOTO.format(c=coarse)])[0]
                        md_fine = vlm.enc_txt([MEDIUM_ANCHOR[med].format(c=fine)])[0]
                        md_coarse = vlm.enc_txt([MEDIUM_ANCHOR[med].format(c=coarse)])[0]
                        dc_full = md_fine - ph_coarse                 # 主管线同构
                        dc_class = ph_fine - ph_coarse                # 只留类迁移
                        dc_tmpl = md_coarse - ph_coarse               # 只留媒介(类别内)
                        S = {}
                        for nm, dvv in (('full', dc_full), ('class_only', dc_class),
                                        ('tmpl_only', dc_tmpl), ('neg_full', -dc_full)):
                            g50, gm = CV.eval_pair(per_img, F, unit(P_src + BETA * dvv) - LAM * P_neg)
                            S[nm] = round(g50, 4)
                        s_norm = BETA * float(np.linalg.norm(dc_full))
                        RD = rand_dirs(dc_full.shape[0], N_RANDOM, RNG_SEED + hash((coarse, fine, med)) % 997)
                        rnd = np.asarray([CV.eval_pair(per_img, F, unit(P_src + s_norm * r) - LAM * P_neg)[0]
                                          for r in RD], float)
                        best, bestb = -1.0, 0
                        for bb in CV.BETA_GRID:
                            a50, _ = CV.eval_pair(per_img, F, unit(P_src + bb * dc_full) - LAM * P_neg)
                            if a50 > best:
                                best, bestb = a50, bb
                        rank = int(np.sum(rnd >= S['full'])) + 1
                        rec.update({
                            'full': S['full'], 'class_only': S['class_only'],
                            'tmpl_only': S['tmpl_only'], 'neg_full': S['neg_full'],
                            'gain_full': round(S['full'] - b50, 4),
                            'gain_class': round(S['class_only'] - b50, 4),
                            'gain_tmpl': round(S['tmpl_only'] - b50, 4),
                            'gain_neg': round(S['neg_full'] - b50, 4),
                            'rand_mean': round(float(rnd.mean()), 4),
                            'rand_gain_mean': round(float(rnd.mean()) - b50, 4),
                            'rank': rank, 'n_rand': N_RANDOM, 'p_c': round(rank / (N_RANDOM + 1), 4),
                            'oracle': round(best, 4), 'star_beta': bestb,
                            'dc_full_mag': round(float(np.linalg.norm(dc_full)), 4),
                            'dc_class_mag': round(float(np.linalg.norm(dc_class)), 4),
                            'dc_tmpl_mag': round(float(np.linalg.norm(dc_tmpl)), 4)})
                    rows.append(rec)
                    done_keys.add(key)
                    if med is None:
                        print(f'  {det} {coarse}->{fine:<10} clean  base={rec["base"]:.4f} '
                              f'(n_box={rec["n_box"]})', flush=True)
                    else:
                        print(f'  {det} {coarse}->{fine:<10} {tag:<8} base={rec["base"]:.4f} | '
                              f'full={rec["gain_full"]:+.4f} class={rec["gain_class"]:+.4f} '
                              f'tmpl={rec["gain_tmpl"]:+.4f} neg={rec["gain_neg"]:+.4f} '
                              f'rand={rec["rand_gain_mean"]:+.4f} rank={rec["rank"]}/{N_RANDOM+1}'
                              f'(p={rec["p_c"]})', flush=True)
                    json.dump({'config': {'beta': BETA, 'lam': LAM, 'k': args.k, 'media': args.media,
                                          'n_random': N_RANDOM, 'partial': True}, 'rows': rows},
                              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    json.dump({'config': {'beta': BETA, 'lam': LAM, 'k': args.k, 'media': args.media,
                          'dets': args.dets, 'n_random': N_RANDOM, 'coarse_map': COARSE,
                          'note': 'non-RS medium axis with coarse source vocabulary -> fine target '
                                  'vocabulary, mirroring the paper operator; full/class_only/tmpl_only '
                                  'decomposition + norm-matched random directions'},
               'rows': rows, 'elapsed_s': round(time.time() - t0, 1)},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
