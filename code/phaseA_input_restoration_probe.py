# -*- coding: utf-8 -*-
"""phaseA_input_restoration_probe.py — Path A 小样本验证: 输入级信号净化是否帮冻结检测器

核心假设(用户选定 A): 净化落在检测器 INPUT(复原 x' 再检测), 应即插即用帮任何方法。
本脚本用 COCO 干净图 z 作真值, 施加两种可控"污染"得到 x, 用经典复原子 R 得 x',
在冻结 GDINO-tiny 上测 11-point AP50(逐图, 论文口径):
  AP(z)   干净上界
  AP(x)   污染后
  AP(x')  净化后再检测
指标: Δ = AP(x') − AP(x)  (净化增益); recovery = AP(x')/AP(z) (恢复比例)

边界(诚实): 污染是合成的(分辨率模糊 / 合成雾霾), 非真实 DIOR→COCO 域偏移;
本实验只验证"输入级净化对冻结检测器是否起效"的机制, 真实域迁移是后续。

复用 unified_eval: load_coco / gdino_detect / compute_ap50_img / PRE_FILTER
复原子均为经典方法(PIL/numpy, 无下载), 体现"方法无关"——任何检测器吃 x' 都应受益。
"""
import os, sys, json
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
from PIL import Image, ImageFilter
import unified_eval as U

PRE_FILTER = U.PRE_FILTER
SAVE_DIR = U.SAVE_DIR
CLASSES = [('airplane', 'a photo of'), ('boat', 'a photo of'), ('car', 'a photo of')]
N = 30


# ── 污染 (degradation) ──
def degrade_res(pil, k=3):
    """分辨率降级(镜像 unified_space_v2.degrade_res): 下采样 k 倍再上采样回原尺寸 → 模糊块。"""
    w, h = pil.size
    return pil.resize((max(1, w // k), max(1, h // k))).resize((w, h))


def degrade_haze(pil, t=0.45, rng=None):
    """合成雾霾(大气散射模型): x = z*t + A*(1-t) + 噪声。A 为近似大气光。"""
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    A = np.array([0.85, 0.88, 0.92])
    x = arr * t + A * (1.0 - t)
    if rng is not None:
        x = x + rng.normal(0.0, 0.02, x.shape).astype(np.float32)
    return Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))


# ── 经典复原 (restoration, 方法无关) ──
def restore_unsharp(pil):
    """非锐化掩码: 恢复模糊丢失的边缘。"""
    return pil.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))


def restore_dehaze(pil, omega=0.95, t0=0.1):
    """暗通道先验单图去雾(He et al. 简化版): 经典方法无关复原子。"""
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    dark = arr.min(axis=2)
    A = float(np.percentile(arr.max(axis=2), 99))
    A = max(A, 1e-3)
    t = 1.0 - omega * (dark / A)
    t = np.clip(t, t0, 1.0)
    out = (arr - A) / t[..., None] + A
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


# ── 检测 + AP50(逐图, 论文口径) ──
def detect_ap50(images, gts, prompt):
    dets = U.gdino_detect(images, prompt, 0.0)
    preds = []
    for (boxes, scores), g in zip(dets, gts):
        keep = scores > PRE_FILTER
        preds.append((boxes[keep], scores[keep]))
    return U.compute_ap50_img(preds, gts)


def main():
    rng = np.random.default_rng(42)
    results = {}
    for cls, prefix in CLASSES:
        print(f'\n=== {cls} ===', flush=True)
        images, gts = U.load_coco(cls, n=N)
        if len(images) < 5:
            print('  图太少跳过', flush=True)
            continue
        prompt = f'{prefix} {cls}'

        ap_z = detect_ap50(images, gts, prompt)                      # 干净上界

        x_res = [degrade_res(im, 3) for im in images]
        ap_x_res = detect_ap50(x_res, gts, prompt)
        xr_res = [restore_unsharp(im) for im in x_res]
        ap_xr_res = detect_ap50(xr_res, gts, prompt)

        x_haze = [degrade_haze(im, t=float(rng.uniform(0.35, 0.6)), rng=rng) for im in images]
        ap_x_haze = detect_ap50(x_haze, gts, prompt)
        xr_haze = [restore_dehaze(im) for im in x_haze]
        ap_xr_haze = detect_ap50(xr_haze, gts, prompt)

        results[cls] = {
            'ap_clean': round(float(ap_z), 4),
            'res_blur': {
                'ap_x': round(float(ap_x_res), 4),
                'ap_xp': round(float(ap_xr_res), 4),
                'delta': round(float(ap_xr_res - ap_x_res), 4),
                'recovery': round(float(ap_xr_res / ap_z), 4) if ap_z > 0 else None,
            },
            'haze': {
                'ap_x': round(float(ap_x_haze), 4),
                'ap_xp': round(float(ap_xr_haze), 4),
                'delta': round(float(ap_xr_haze - ap_x_haze), 4),
                'recovery': round(float(ap_xr_haze / ap_z), 4) if ap_z > 0 else None,
            },
        }
        r = results[cls]
        print(f'  clean={ap_z:.4f}', flush=True)
        print(f'  res : x={r["res_blur"]["ap_x"]:.4f} xp={r["res_blur"]["ap_xp"]:.4f} '
              f'Δ={r["res_blur"]["delta"]:+.4f} recovery={r["res_blur"]["recovery"]}', flush=True)
        print(f'  haze: x={r["haze"]["ap_x"]:.4f} xp={r["haze"]["ap_xp"]:.4f} '
              f'Δ={r["haze"]["delta"]:+.4f} recovery={r["haze"]["recovery"]}', flush=True)

    def mean_delta(key):
        vals = [results[c][key]['delta'] for c in results if key in results[c]]
        return float(np.mean(vals)) if vals else float('nan')

    results['summary'] = {
        'res_blur_delta_mean': mean_delta('res_blur'),
        'haze_delta_mean': mean_delta('haze'),
        'note': '合成污染机制验证; 非真实 DIOR->COCO 域偏移',
    }
    out = SAVE_DIR / 'phaseA_restoration_probe.json'
    json.dump(results, open(out, 'w'), indent=2, ensure_ascii=False)
    print(f'\n结果 → {out}', flush=True)
    print(f'res_blur Δ均值={results["summary"]["res_blur_delta_mean"]:+.4f}  '
          f'haze Δ均值={results["summary"]["haze_delta_mean"]:+.4f}', flush=True)


if __name__ == '__main__':
    main()
