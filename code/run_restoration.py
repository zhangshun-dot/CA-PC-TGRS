# -*- coding: utf-8 -*-
"""run_restoration.py — 两层理论终判实验：图像侧复原 vs 原型侧校正

问题（元 2026-10-11 批准）：观测级损失若真实且主导（两层理论 reformulated 版），
  P1 图像侧复原应恢复大部分独立 headroom  H = AP_clean_base − AP_deg_base
  P2 若 CA-PC 的增益真是"观测补偿"，复原后 g_CA 应显著缩小（观测已修复→无可补偿）
判读：
  R_rest = (AP_res_base − AP_deg_base) / H        高 → P1 成立（level-1 主导）
  g_CA(res) ≈ g_CA(deg)                            → 增益与观测条件无关（CA-PC 非观测补偿）

设计（与 run_degradation.py 同口径，逐格可比）：
  - source = clean COCO（P_src = clean GT crop 原型；P_neg = 其它类原型）
  - target = 同一批 COCO 图的退化版（确定性 seed 复现，与 run_degradation 同图）
  - 复原 = 盲算子（只知退化族，不知退化参数）：fog→DCP去雾, blur→UnsharpMask, brightness→autocontrast
  - 检测：GDINO-tiny 在**复原图**上重新检测；CLIP-b32 裁剪特征；同一打分器
  - W 变体：base / naive / tge / ca_pc(β=2) / ca_star(β 网格) —— 全部同 run_degradation

用法：
  HF_HUB_OFFLINE=1 "C:/Users/张三/vg/Scripts/python.exe" -u run_restoration.py \
      --k 40 --corruptions fog gaussian_blur brightness --severities 1 3 5
"""
import os, sys, json, time, pickle, argparse
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cuda')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch
import cv2

import run_degradation as RD          # 复用 corrupt/CORR_ANCHOR/unit/_to_pil（main 有守卫）
import run_bidirectional as RB
import run_cross_vlm as CV
os.environ['HF_HUB_OFFLINE'] = '1'

OUT = r'C:\Users\张三\Desktop\实验数据\run_restoration.json'
DEG_JSON = r'C:\Users\张三\Desktop\实验数据\run_degradation.json'
DCACHE = r'C:\Users\张三\Desktop\实验数据\restored_det_cache'
os.makedirs(DCACHE, exist_ok=True)

BETA, LAM, NEG_TOPK = CV.BETA, CV.LAM, CV.NEG_TOPK
BETA_GRID = CV.BETA_GRID


# ── 盲复原算子：只按退化族选择，不用退化的精确参数 ──
def restore_fog(arr, sev):
    """暗通道先验去雾 + 辐射守恒校准：在 tmin 网格上自动选使输出均值≈输入均值的
    透射率下限（修正 DCP 对明亮场景的系统性压黑），全程不用退化参数。"""
    a = arr.astype(np.float32) / 255.0
    dark = cv2.erode(a.min(axis=2), np.ones((15, 15), np.uint8))
    bright = a.max(axis=2).reshape(-1)
    idx = np.argsort(bright)[-max(1, len(bright) // 1000):]
    A = a.reshape(-1, 3)[idx].mean(0)
    t0 = 1.0 - 0.9 * dark / max(float(A.max()), 1e-3)
    t0 = np.clip(cv2.blur(t0, (15, 15)), 0.05, 1.0)
    best = None
    for tmin in (0.3, 0.45, 0.6, 0.75):
        t = np.clip(t0, tmin, 1.0)[..., None]
        J = np.clip((a - A[None, None, :]) / t + A[None, None, :], 0.0, 1.0) * 255.0
        d = abs(float(J.mean()) - float(arr.mean()))
        if best is None or d < best[0]:
            best = (d, J)
    return best[1]


def restore_blur(arr, sev):
    """严重度感知的反卷积近似：UnsharpMask，半径随 severity 增大。"""
    from PIL import ImageFilter
    r = {1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0}.get(int(sev), 2.0)
    im = RD._to_pil(arr).filter(ImageFilter.UnsharpMask(radius=r, percent=200, threshold=2))
    return np.asarray(im).astype(np.float32)


def restore_brightness(arr, sev):
    """盲曝光修复：逐通道直方图拉伸（1% 截断）。"""
    from PIL import ImageOps
    return np.asarray(ImageOps.autocontrast(RD._to_pil(arr), cutoff=1)).astype(np.float32)


RESTORE = {'fog': restore_fog, 'gaussian_blur': restore_blur, 'brightness': restore_brightness}


def clean_ref():
    """从 run_degradation.json 取 clean base AP（同 k/同图/同打分器）。"""
    d = json.load(open(DEG_JSON, encoding='utf-8'))
    return {(r['det'], r['cls']): r['base'] for r in d['rows'] if r['corr'] == 'clean'}


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

    vlm = CV.VLM('clip_b32')

    data, proto = {}, {}
    for cls in args.classes:
        ims, gts = RB.load_split(cfg, 'coco', cls, CV.N_PROTO, CV.BIG_N)
        data[cls] = (ims[:args.k], gts[:args.k])
        gc_ = RB.D.gt_crops(data[cls][0], data[cls][1])
        Ft = vlm.enc_img(gc_)
        pt = torch.from_numpy(Ft.mean(0))
        proto[cls] = pt / pt.norm()
        print(f'[proto] {cls}: {len(data[cls][0])} imgs, {len(gc_)} crops', flush=True)

    def pneg_for(cls):
        return RD.unit(RB.Proto.get_negative_prototype(proto, cls, NEG_TOPK).numpy())

    # 条件匹配锚（与 run_degradation 完全一致）：键 (cls, corr)
    dc_map = {}
    for corr_ in args.corruptions:
        Ec = vlm.enc_txt([RD.CORR_ANCHOR[corr_][0].format(c=c) for c in args.classes])
        Ed = vlm.enc_txt([RD.CORR_ANCHOR[corr_][1].format(c=c) for c in args.classes])
        for i, c in enumerate(args.classes):
            dc_map[(c, corr_)] = Ed[i] - Ec[i]

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

    ref = clean_ref()
    rows, done = [], set()
    if os.path.exists(OUT):
        try:
            rows = json.load(open(OUT, encoding='utf-8')).get('rows', [])
            done = {(r['det'], r['cls'], r['corr'], r['sev']) for r in rows}
        except Exception:
            rows, done = [], set()

    for det in args.dets:
        for cls in args.classes:
            ims, gts = data[cls]
            for corr in args.corruptions:
                for sev in args.severities:
                    if (det, cls, corr, sev) in done:
                        continue
                    try:
                        # 1) 复现退化图（与 run_degradation 逐位一致）→ 复原
                        imgs_res = [RD._to_pil(RESTORE[corr](np.asarray(
                            RD.corrupt(im, corr, sev, (cls,)).convert('RGB')), sev)) for im in ims]
                        # 2) 复原图上重新检测 + 特征（缓存）
                        ck = f'{det}__{cls}__{corr}_{sev}__restored.pkl'
                        cp = os.path.join(DCACHE, ck)
                        if os.path.exists(cp):
                            blob = pickle.load(open(cp, 'rb'))
                            per_img, F = blob['per_img'], blob['F']
                        else:
                            per_img = run_detect(cls, imgs_res)
                            crops = []
                            for img, p in zip(imgs_res, per_img):
                                crops.extend(CV.crops_from_cached(img, p))
                            F = vlm.enc_img(crops)
                            pickle.dump({'per_img': per_img, 'F': F}, open(cp, 'wb'))
                        # 3) 同口径打分
                        P_src = RD.unit(proto[cls].numpy())
                        P_neg = pneg_for(cls)
                        dc = dc_map[(cls, corr)]
                        W_base = P_src - LAM * P_neg
                        W_tge = RD.unit(P_src + dc) - LAM * P_neg
                        W_ca = RD.unit(P_src + BETA * dc) - LAM * P_neg
                        b50, bm = CV.eval_pair(per_img, F, W_base)
                        t50, tm = CV.eval_pair(per_img, F, W_tge)
                        c50, cm = CV.eval_pair(per_img, F, W_ca)
                        best, bestm, bestb = -1.0, 0.0, 0
                        for bb in BETA_GRID:
                            a50, m = CV.eval_pair(per_img, F, RD.unit(P_src + bb * dc) - LAM * P_neg)
                            if a50 > best:
                                best, bestm, bestb = a50, m, bb
                        rec = {'det': det, 'cls': cls, 'corr': corr, 'sev': sev,
                               'n_img': len(imgs_res),
                               'n_box': int(sum(len(p['gd']) for p in per_img)),
                               'ap_clean_base': ref.get((det, cls)),
                               'base': round(b50, 4), 'base_m': round(bm, 4),
                               'tge': round(t50, 4), 'ca_pc': round(c50, 4),
                               'ca_star': round(best, 4), 'star_beta': bestb}
                        rows.append(rec)
                        print(f'  {cls:<10} {corr:<14} S{sev}: base={b50:.4f} '
                              f'tge={t50:.4f} ca={c50:.4f} star={best:.4f}(β={bestb})', flush=True)
                    except Exception as e:
                        import traceback
                        print(f'  [err] {cls} {corr} S{sev}: {type(e).__name__}: {e}', flush=True)
                        traceback.print_exc()
                    json.dump({'config': {'beta': BETA, 'lam': LAM, 'k': args.k,
                                          'corruptions': args.corruptions,
                                          'severities': args.severities, 'partial': True},
                               'rows': rows}, open(OUT, 'w', encoding='utf-8'),
                              indent=1, ensure_ascii=False)

    json.dump({'config': {'beta': BETA, 'lam': LAM, 'k': args.k,
                          'corruptions': args.corruptions, 'severities': args.severities,
                          'dets': args.dets,
                          'note': 'image-side restoration vs prototype-side correction; '
                                  'R_rest=(res_base-deg_base)/(clean_base-deg_base)'},
               'rows': rows, 'elapsed_s': round(time.time() - t0, 1)},
              open(OUT, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
