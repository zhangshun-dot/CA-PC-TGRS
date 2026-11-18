# -*- coding: utf-8 -*-
"""phaseA_severe.py — Path A 重度退化重验: 看检测器真掉一大截时, 经典净化能否救回

动机: phaseA 小样本用轻度退化(blur k=3 / haze t=0.35-0.6), 检测器本就鲁棒→headroom 极小,
Δ 全负/≈0 不能区分"净化从不帮" vs "只在重度退化帮"。本脚本加大退化强度, 留出真实 headroom,
看 AP(x') 能否向 AP(z) 恢复。同时保留轻度档作对照。

复用 phaseA 的退化/复原/检测函数(直接复制, 避免 import 路径问题)。
"""
import os, json
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
from PIL import Image, ImageFilter
import unified_eval as U

PRE_FILTER = U.PRE_FILTER
SAVE_DIR = U.SAVE_DIR
CLASSES = [('airplane', 'a photo of'), ('boat', 'a photo of'), ('car', 'a photo of')]
N = 30


def degrade_res(pil, k=3):
    w, h = pil.size
    return pil.resize((max(1, w // k), max(1, h // k))).resize((w, h))


def degrade_haze(pil, t=0.45, rng=None):
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    A = np.array([0.85, 0.88, 0.92])
    x = arr * t + A * (1.0 - t)
    if rng is not None:
        x = x + rng.normal(0.0, 0.02, x.shape).astype(np.float32)
    return Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))


def restore_unsharp(pil):
    return pil.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))


def restore_dehaze(pil, omega=0.95, t0=0.1):
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    dark = arr.min(axis=2)
    A = float(np.percentile(arr.max(axis=2), 99))
    A = max(A, 1e-3)
    t = 1.0 - omega * (dark / A)
    t = np.clip(t, t0, 1.0)
    out = (arr - A) / t[..., None] + A
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


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
    RES_LEVELS = [3, 8]          # 轻度 / 重度模糊
    HAZE_LEVELS = [0.5, 0.25]    # 中度 / 重度雾霾
    for cls, prefix in CLASSES:
        print(f'\n=== {cls} ===', flush=True)
        images, gts = U.load_coco(cls, n=N)
        if len(images) < 5:
            print('  图太少跳过', flush=True)
            continue
        prompt = f'{prefix} {cls}'
        ap_z = detect_ap50(images, gts, prompt)
        row = {'ap_clean': round(float(ap_z), 4), 'res': {}, 'haze': {}}
        for k in RES_LEVELS:
            x = [degrade_res(im, k) for im in images]
            ap_x = detect_ap50(x, gts, prompt)
            xp = [restore_unsharp(im) for im in x]
            ap_xp = detect_ap50(xp, gts, prompt)
            row['res'][f'k{k}'] = {
                'ap_x': round(float(ap_x), 4), 'ap_xp': round(float(ap_xp), 4),
                'delta': round(float(ap_xp - ap_x), 4),
                'recovery': round(float(ap_xp / ap_z), 4) if ap_z > 0 else None}
            print(f'  res k={k}: x={ap_x:.4f} xp={ap_xp:.4f} '
                  f'Δ={ap_xp - ap_x:+.4f} recovery={ap_xp / ap_z:.3f}', flush=True)
        for t in HAZE_LEVELS:
            x = [degrade_haze(im, t=float(t), rng=rng) for im in images]
            ap_x = detect_ap50(x, gts, prompt)
            xp = [restore_dehaze(im) for im in x]
            ap_xp = detect_ap50(xp, gts, prompt)
            row['haze'][f't{t}'] = {
                'ap_x': round(float(ap_x), 4), 'ap_xp': round(float(ap_xp), 4),
                'delta': round(float(ap_xp - ap_x), 4),
                'recovery': round(float(ap_xp / ap_z), 4) if ap_z > 0 else None}
            print(f'  haze t={t}: x={ap_x:.4f} xp={ap_xp:.4f} '
                  f'Δ={ap_xp - ap_x:+.4f} recovery={ap_xp / ap_z:.3f}', flush=True)
        results[cls] = row

    def mean_delta(section, key):
        vals = [results[c][section][key]['delta']
                for c in results if key in results[c].get(section, {})]
        return float(np.mean(vals)) if vals else float('nan')

    results['summary'] = {
        'res_k3_delta_mean': mean_delta('res', 'k3'),
        'res_k8_delta_mean': mean_delta('res', 'k8'),
        'haze_t05_delta_mean': mean_delta('haze', 't0.5'),
        'haze_t025_delta_mean': mean_delta('haze', 't0.25'),
        'note': '重度退化重验: 看检测器真掉时经典净化能否救回',
    }
    out = SAVE_DIR / 'phaseA_restoration_probe_severe.json'
    json.dump(results, open(out, 'w'), indent=2, ensure_ascii=False)
    print(f'\n结果 → {out}', flush=True)
    s = results['summary']
    print(f'res k3 Δ={s["res_k3_delta_mean"]:+.4f}  k8 Δ={s["res_k8_delta_mean"]:+.4f}  '
          f'haze t0.5 Δ={s["haze_t05_delta_mean"]:+.4f}  t0.25 Δ={s["haze_t025_delta_mean"]:+.4f}',
          flush=True)


if __name__ == '__main__':
    main()
