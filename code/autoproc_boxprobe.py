# -*- coding: utf-8 -*-
"""动框天花板探针(零重检测, 复用 GDINO 缓存):
问: GDINO 框几何是否还有提升空间? 即在已有框上做缩放/平移(以 GSD 为动机),
AP 能否显著冲过当前 +1.25% 天花板?
方法: 复用 load_domain_data 缓存(target_data 已含 box/gd/features),
按 config-A(γ=0,α=0.5) 算 fused 分数, 对 box 做几何变换后重算 AP50,
扫描 scale×shift 找每类最优扰动 AP。同时测 raw-gd(无方法) 下的扰动上限。
"""
import sys, json
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
from pathlib import Path
import torch
import usc_experiment_v12 as V12
import unified_eval as U
from usc_rank1_affine_gpu_v2 import load_domain_data

DATA = V12.DATA
EVAL_DOMAINS = ['DIOR', 'VisDrone']
N_TARGET = 50
SENSOR_GAMMA = {'DIOR': -2.0, 'NWPU': -2.0, 'DOTA': -2.0, 'VisDrone': -1.0}
GSD = {'DIOR': 1.0, 'NWPU': 1.0, 'DOTA': 0.5, 'VisDrone': 0.3}
SCALES = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
SHIFTS = [-0.08, 0.0, 0.08]  # 相对框宽/高的中心平移比例


def make_transform(s, dx_f, dy_f):
    def t(box):
        if len(box) == 0:
            return box
        b = box.astype(np.float64).copy()
        cx = (b[:, 0] + b[:, 2]) / 2.0
        cy = (b[:, 1] + b[:, 3]) / 2.0
        w = b[:, 2] - b[:, 0]
        h = b[:, 3] - b[:, 1]
        nw = w * s
        nh = h * s
        b[:, 0] = cx - nw / 2.0
        b[:, 2] = cx + nw / 2.0
        b[:, 1] = cy - nh / 2.0
        b[:, 3] = cy + nh / 2.0
        b[:, 0] += dx_f * w
        b[:, 2] += dx_f * w
        b[:, 1] += dy_f * h
        b[:, 3] += dy_f * h
        return np.clip(b, 0.0, None)
    return t


def fused_scores(td, proto, neg, lam=V12.LAM):
    per_img = td['per_img']
    F = td['features']
    out = []
    for p in per_img:
        n = len(p['gd'])
        if n:
            f = F[p['crop_idx']]
            pos = f @ proto
            neg_s = f @ neg if neg is not None else np.zeros(n)
            vis = np.maximum(0.0, pos - lam * neg_s)
            fused = np.sqrt(np.clip(p['gd'], 1e-6, 1.0) * np.clip(vis, 1e-6, 1.0))
        else:
            fused = np.zeros(0)
        out.append(fused)
    return out


def ap_with_transform(td, proto, neg, transform):
    per_img = td['per_img']
    fused = fused_scores(td, proto, neg)
    preds, gts = [], []
    for p, fu in zip(per_img, fused):
        box = transform(p['box']) if len(p['box']) else p['box']
        preds.append((box, fu))
        gts.append(p['gt'])
    return float(U.compute_ap50_img(preds, gts))


def main():
    results = {}
    for src in EVAL_DOMAINS:
        print(f"\n{'='*60}\n源域 {src} → COCO (N={N_TARGET})\n{'='*60}")
        dd = load_domain_data(src)
        for coco_cls in list(dd['target_data'].keys()):
            td = dd['target_data'][coco_cls]
            td['per_img'] = td['per_img'][:N_TARGET]
        usc = dd['usc']
        protos_gpu = dd['protos_gpu']
        all_protos = dd['all_protos']
        text_deltas = dd['text_deltas']

        # config-A 原型(γ=0,α=0.5) + 文本外推
        cls_rec = []
        for eval_name, coco_cls, src_cls in dd['valid_targets']:
            td = dd['target_data'][coco_cls]
            p_corr = usc.correct_prototype_gpu(protos_gpu[src_cls], 0.0, 0.5).cpu().numpy()
            neg_raw = V12.build_neg_proto(all_protos, src_cls)
            delta = text_deltas[src_cls]
            p_hat = p_corr + delta
            p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)
            neg = neg_raw  # γ=0 时负原型不随校正变

            # 基线: config-A 不改框
            base_ap = ap_with_transform(td, p_hat, neg, make_transform(1.0, 0, 0))
            # raw-gd 基线(无方法)
            raw_preds = [(p['box'], p['gd']) for p in td['per_img']]
            raw_ap = float(U.compute_ap50_img(raw_preds, [p['gt'] for p in td['per_img']]))

            # 探针: 扫描 scale×shift, 用 config-A fused
            best, best_t = base_ap, (1.0, 0, 0)
            for s in SCALES:
                for dx in SHIFTS:
                    for dy in SHIFTS:
                        ap = ap_with_transform(td, p_hat, neg, make_transform(s, dx, dy))
                        if ap > best:
                            best, best_t = ap, (s, dx, dy)
            # 探针 raw-gd 上限
            best_raw, best_raw_t = raw_ap, (1.0, 0, 0)
            for s in SCALES:
                for dx in SHIFTS:
                    for dy in SHIFTS:
                        preds = [(make_transform(s, dx, dy)(p['box']), p['gd']) for p in td['per_img']]
                        ap = float(U.compute_ap50_img(preds, [p['gt'] for p in td['per_img']]))
                        if ap > best_raw:
                            best_raw, best_raw_t = ap, (s, dx, dy)

            cls_rec.append({
                'class': coco_cls, 'base_ap': round(base_ap, 4),
                'probe_ap': round(best, 4), 'probe_t': best_t,
                'raw_ap': round(raw_ap, 4), 'raw_probe_ap': round(best_raw, 4),
                'raw_probe_t': best_raw_t,
                'delta_method': round(best - base_ap, 4),
            })
            print(f"  {coco_cls:<11s} base={base_ap:.4f} probe={best:.4f}(Δ{best-base_ap:+.4f}) "
                  f"raw={raw_ap:.4f} rawProbe={best_raw:.4f}  win_t={best_t}")

        results[src] = {
            'GSD': GSD[src],
            'base_mean': float(np.mean([r['base_ap'] for r in cls_rec])),
            'probe_mean': float(np.mean([r['probe_ap'] for r in cls_rec])),
            'raw_mean': float(np.mean([r['raw_ap'] for r in cls_rec])),
            'raw_probe_mean': float(np.mean([r['raw_probe_ap'] for r in cls_rec])),
            'per_class': cls_rec,
        }
        r = results[src]
        print(f"  → 域均值: base={r['base_mean']:.4f} probe={r['probe_mean']:.4f} "
              f"(Δ{r['probe_mean']-r['base_mean']:+.4f}) | raw={r['raw_mean']:.4f} rawProbe={r['raw_probe_mean']:.4f}")

    with open(DATA / 'autoproc_boxprobe_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ -> {DATA / 'autoproc_boxprobe_results.json'}")


if __name__ == '__main__':
    main()
