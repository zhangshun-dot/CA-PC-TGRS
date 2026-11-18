# -*- coding: utf-8 -*-
"""run_degradation.py — 非遥感独立域验证：COCO 观测退化轴

元 2026-10-09 批准。目的：把 headroom 律放到**非遥感**的观测退化轴验证，
并给出一个**独立** headroom： H_indep = AP_clean_base − AP_degraded_base
（不含 CA-PC 自身 oracle ceiling → 破 ρ(gain,H)=0.90 的构造性自相关）

设计（与全文同口径）：
  - source = clean COCO（P_src = clean COCO GT crop 原型；P_neg = 其它 COCO 类原型）
  - target = **同一批 COCO 图**的退化版（同类名 → 无类映射混淆）
  - dc_deg = E[退化锚文本] − E[清晰锚文本]   ← 锚描述真实位移（应有效）
  - dc_rs  = E[overhead]  − E[photo]         ← 错锚对照（应无效 → 证明锚须忠实）
  - 打分 s = sqrt(gd · clip(max(0, F·W)))，W = unit(P_src + β·dc) − λ·P_neg
  - 检测：GDINO-tiny 在退化图上**重新检测**（无法复用 clean 框），
          同一 detector/prompt 也跑 clean → AP_clean 与 AP_deg 同协议可比。

关键工程点：每张退化图**只生成一次**（确定性 seed），同时用于检测与 crop 特征，
避免"框来自 A 实现、特征来自 B 实现"的口径错配；检测结果与 F 均落盘缓存。

用法：
  HF_HUB_OFFLINE=1 "C:/Users/张三/vg/Scripts/python.exe" -u run_degradation.py \
      --k 40 --corruptions fog gaussian_blur brightness --severities 1 3 5
"""
import os, sys, json, time, pickle, hashlib, argparse
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cuda')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch

# ── numpy 2.x 兼容 shim：imagecorruptions 1.1.2 仍用 np.float_ 等已移除别名 ──
for _old, _new in (('float_', 'float64'), ('int_', 'int64'), ('complex_', 'complex128'),
                   ('unicode_', 'str_'), ('object_', 'object')):
    if not hasattr(np, _old):
        try:
            setattr(np, _old, getattr(np, _new))
        except AttributeError:
            pass

import run_bidirectional as RB
import run_cross_vlm as CV
os.environ['HF_HUB_OFFLINE'] = '1'   # CV import 会置 '0', 改回离线

OUT = r'C:\Users\张三\Desktop\实验数据\run_degradation.json'
DCACHE = r'C:\Users\张三\Desktop\实验数据\degrade_det_cache'
os.makedirs(DCACHE, exist_ok=True)

N_PROTO, BIG_N = CV.N_PROTO, CV.BIG_N
BETA, LAM, NEG_TOPK = CV.BETA, CV.LAM, CV.NEG_TOPK
BETA_GRID = CV.BETA_GRID

CORR_ANCHOR = {
    'fog':            ('a clear photo of a {c}',           'a foggy hazy photo of a {c}'),
    'gaussian_blur':  ('a sharp photo of a {c}',           'a blurred out-of-focus photo of a {c}'),
    'brightness':     ('a well-lit photo of a {c}',        'an overexposed bright photo of a {c}'),
    'gaussian_noise': ('a clean photo of a {c}',           'a noisy grainy photo of a {c}'),
    'pixelate':       ('a high-resolution photo of a {c}', 'a low-resolution pixelated photo of a {c}'),
    'contrast':       ('a normal-contrast photo of a {c}', 'a low-contrast washed-out photo of a {c}'),
}
ANCHOR_WRONG = ('a photo of a {c}', 'an overhead image of a {c}')


def _patch_imagecorruptions():
    """scikit-image ≥0.23 用 channel_axis 取代 multichannel；imagecorruptions 1.1.2 未跟进。
    修补其模块级 gaussian，使 gaussian_blur/defocus_blur/motion_blur 可用。"""
    try:
        import imagecorruptions.corruptions as _ic
        if getattr(_ic, '_compat_patched', False):
            return
        from skimage.filters import gaussian as _sk_gauss

        def _gauss(x, sigma=1.0, multichannel=False, **kw):
            if multichannel and 'channel_axis' not in kw:
                kw['channel_axis'] = -1
            return _sk_gauss(x, sigma=sigma, **kw)

        _ic.gaussian = _gauss
        _ic._compat_patched = True
    except Exception:
        pass


_patch_imagecorruptions()


def _seed_for(*parts):
    h = hashlib.md5('|'.join(map(str, parts)).encode()).hexdigest()[:8]
    return int(h, 16) % (2 ** 31)


def _to_pil(a):
    from PIL import Image
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _fallback(arr, name, s):
    """imagecorruptions 缺失时的兜底自实现（参数按 Hendrycks 表近似）。"""
    a = arr.astype(np.float32)
    if name == 'gaussian_noise':
        return a + np.random.randn(*a.shape) * ([0.08, 0.12, 0.18][min(s, 3) - 1] * 255)
    if name == 'brightness':
        return a + [30, 60, 100][min(s, 3) - 1]
    if name == 'contrast':
        f = [0.75, 0.5, 0.3][min(s, 3) - 1]
        m = a.mean(axis=(0, 1), keepdims=True)
        return (a - m) * f + m
    if name in ('gaussian_blur', 'defocus_blur', 'motion_blur'):
        from PIL import ImageFilter
        return np.asarray(_to_pil(a).filter(ImageFilter.GaussianBlur([1.5, 3.0, 5.0][min(s, 3) - 1]))).astype(np.float32)
    if name == 'pixelate':
        from PIL import Image
        im = _to_pil(a); w, h = im.size
        f = [0.7, 0.45, 0.25][min(s, 3) - 1]
        return np.asarray(im.resize((max(2, int(w * f)), max(2, int(h * f))), Image.BILINEAR)
                          .resize((w, h), Image.BILINEAR)).astype(np.float32)
    if name == 'fog':
        from scipy.ndimage import gaussian_filter
        h, w, _ = a.shape
        d = gaussian_filter(np.random.rand(h, w), sigma=max(h, w) / 12.0)
        d = (d - d.min()) / (d.ptp() + 1e-8)
        t = np.clip(1.0 - [0.55, 0.75, 0.92][min(s, 3) - 1] * d, 0.05, 1.0)[..., None]
        A = 255.0 * np.array([0.95, 0.95, 0.98])
        return a * t + A * (1 - t)
    return a


def corrupt(pil, name, sev, seed_parts):
    """确定性退化：同一 (name, sev, 条件) 必得同一图。"""
    np.random.seed(_seed_for(name, sev, *seed_parts))
    arr = np.asarray(pil.convert('RGB'))
    try:
        from imagecorruptions import corrupt as _c
        # 注意签名: corrupt(image, severity, corruption_name, corruption_number)
        return _to_pil(_c(arr, int(sev), name))
    except Exception as _e:
        if not getattr(corrupt, '_warned', False):
            print(f'  [warn] imagecorruptions 不可用({type(_e).__name__}: {_e}), 退回自实现', flush=True)
            corrupt._warned = True
        return _to_pil(_fallback(arr, name, int(sev)))


def unit(x):
    return CV.unit(x)


def score_blob(per_img, F, W):
    return CV.eval_pair(per_img, F, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dets', nargs='+', default=['gdino'])
    ap.add_argument('--k', type=int, default=40)
    ap.add_argument('--corruptions', nargs='+', default=['fog', 'gaussian_blur', 'brightness'])
    ap.add_argument('--severities', nargs='+', type=int, default=[1, 3, 5])
    ap.add_argument('--classes', nargs='+', default=list(RB.COCO_POOL))
    args = ap.parse_args()
    t0 = time.time()

    cfg = RB.Config()
    cfg.model.device = CV.DEVICE
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'
    cfg.model.detection_threshold = 0.0

    vlm = CV.VLM('clip_b32')                      # fp32, 与全文表格同口径

    # ── 图片集 + 原型池 ──
    data = {}
    proto = {}
    for cls in args.classes:
        ims, gts = RB.load_split(cfg, 'coco', cls, N_PROTO, BIG_N)
        data[cls] = (ims[:args.k], gts[:args.k])
        gc_ = RB.D.gt_crops(data[cls][0], data[cls][1])
        Ft = vlm.enc_img(gc_)
        pt = torch.from_numpy(Ft.mean(0))
        proto[cls] = pt / pt.norm()
        print(f'[proto] {cls}: {len(data[cls][0])} imgs, {len(gc_)} crops', flush=True)

    # 负原型：其它 COCO 类（与全文同机制）
    def pneg_for(cls):
        return unit(RB.Proto.get_negative_prototype(proto, cls, NEG_TOPK).numpy())

    # 错锚 dc（overhead − photo）
    Ec = vlm.enc_txt([ANCHOR_WRONG[0].format(c=c) for c in args.classes])
    Eo = vlm.enc_txt([ANCHOR_WRONG[1].format(c=c) for c in args.classes])
    dc_wrong = {c: Eo[i] - Ec[i] for i, c in enumerate(args.classes)}

    def run_detect(cls, imgs):
        """GDINO-tiny；返回与 CV.score_rows 兼容的 per_img。"""
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
            conds = [('clean', None, None)] + [(c, c, s) for c in args.corruptions for s in args.severities]
            for (tag, corr, sev) in conds:
                # ── 1) 生成/复用退化图（只生成一次） ──
                imgs = list(ims) if corr is None else [corrupt(im, corr, sev, (cls,)) for im in ims]
                ck = f'{det}__{cls}__{tag}_{sev}.pkl'
                cp = os.path.join(DCACHE, ck)
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
                    print(f'  [det] {det} {cls:<10} {tag}S{sev}: {len(imgs)} imgs, '
                          f'{sum(len(p["gd"]) for p in per_img)} boxes', flush=True)

                P_src = unit(proto[cls].numpy())
                P_neg = pneg_for(cls)
                W_base = P_src - LAM * P_neg
                b50, bm = score_blob(per_img, F, W_base)
                rec = {'det': det, 'cls': cls, 'corr': corr or 'clean', 'sev': sev,
                       'n_img': len(per_img), 'n_box': int(sum(len(p['gd']) for p in per_img)),
                       'base': round(b50, 4), 'base_m': round(bm, 4)}

                if corr is not None:
                    cl_t = vlm.enc_txt([CORR_ANCHOR[corr][0].format(c=cls)])[0]
                    dg_t = vlm.enc_txt([CORR_ANCHOR[corr][1].format(c=cls)])[0]
                    dc = dg_t - cl_t
                    dcw = dc_wrong[cls]
                    W_naive = unit(dg_t) - LAM * P_neg                    # 裸退化文本原型
                    W_tge = unit(P_src + dc) - LAM * P_neg                # TGE(无 β/无门控)
                    W_ca = unit(P_src + BETA * dc) - LAM * P_neg          # CA-PC(β=2)
                    W_cw = unit(P_src + BETA * dcw) - LAM * P_neg         # CA-PC 错锚对照
                    n50, nm = score_blob(per_img, F, W_naive)
                    t50, tm = score_blob(per_img, F, W_tge)
                    c50, cm = score_blob(per_img, F, W_ca)
                    w50, wm = score_blob(per_img, F, W_cw)
                    best, bestm, bestb = -1.0, 0.0, 0
                    for bb in BETA_GRID:
                        a50, m = score_blob(per_img, F, unit(P_src + bb * dc) - LAM * P_neg)
                        if a50 > best:
                            best, bestm, bestb = a50, m, bb
                    rec.update({'dc_mag': round(float(np.linalg.norm(dc)), 4),
                                'naive': round(n50, 4), 'naive_m': round(nm, 4),
                                'tge': round(t50, 4), 'tge_m': round(tm, 4),
                                'ca_pc': round(c50, 4), 'ca_pc_m': round(cm, 4),
                                'ca_wrong': round(w50, 4), 'ca_wrong_m': round(wm, 4),
                                'ca_star': round(best, 4), 'ca_star_m': round(bestm, 4),
                                'star_beta': bestb})
                rows.append(rec)
                if corr is None:
                    print(f'  {det} {cls:<10} clean          base={rec["base"]:.4f} '
                          f'(n_box={rec["n_box"]})', flush=True)
                else:
                    print(f'  {det} {cls:<10} {corr}S{sev:<3} base={rec["base"]:.4f}  '
                          f'ca_pc={rec["ca_pc"]:.4f}(Δ{rec["ca_pc"]-rec["base"]:+.4f})  '
                          f'ca_wrong={rec["ca_wrong"]:.4f}(Δ{rec["ca_wrong"]-rec["base"]:+.4f})  '
                          f'tge={rec["tge"]:.4f}  ||dc||={rec["dc_mag"]:.3f}', flush=True)
                json.dump({'config': {'beta': BETA, 'lam': LAM, 'k': args.k,
                                      'corruptions': args.corruptions,
                                      'severities': args.severities, 'partial': True},
                           'rows': rows}, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    json.dump({'config': {'beta': BETA, 'lam': LAM, 'k': args.k, 'corruptions': args.corruptions,
                          'severities': args.severities, 'dets': args.dets,
                          'note': 'non-RS observational-degradation axis; '
                                  'H_indep = AP_clean_base - AP_deg_base (independent of CA-PC oracle)'},
               'rows': rows, 'elapsed_s': round(time.time() - t0, 1)},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
