# -*- coding: utf-8 -*-
"""few_shot_calib.py — few-shot 目标域原型校准（验证少样本能否救难类）

核心假设: boat 零样本失败是因为 DIOR 俯视船原型覆盖不了 COCO 侧视船。
若允许用 COCO 每类 K 张标注样本校准原型, 能否把判别力拉起来?

协议:
  1. 跨域(COCO)每类留前 K 张图作 few-shot 校准集, 其余测试
  2. 从 few-shot 集 GT 框提取 DINOv2 L12 特征 → 目标域原型 P_tgt
  3. 校准原型 P_cal = normalize(α·P_src + (1-α)·P_tgt), 扫 α
     α=1 → 纯源域(零样本 CDVM); α=0 → 纯目标域 few-shot
  4. 负原型仍用源域 DIOR 其他类(top-3), 在测试集评 CDVM AP50
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import unified_eval as U

LAYER = 11          # L12
K = 10              # few-shot 图数
NEG_LAM = 0.5
ALPHAS = [1.0, 0.7, 0.5, 0.3, 0.0]   # α: 1=纯源域, 0=纯目标域
N_COCO = 100


def get_neg_proto(bank, dior_cls, layer):
    pos = bank[dior_cls][layer]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][layer]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return neg / neg.norm()


def encode_l12(crop):
    dino, pre = U.get_dinov2()
    t = pre(crop).unsqueeze(0).to(U.DEVICE)
    with torch.no_grad():
        o = dino(t)
    f = o.hidden_states[LAYER + 1][0, 0, :]
    return f / f.norm()


def extract_fewshot_proto(images, gts):
    """从 few-shot 图集的所有 GT 框提取 L12 特征, 平均成目标域原型。"""
    feats = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            feats.append(encode_l12(img.crop((x1, y1, x2, y2))))
    if not feats:
        return None, 0
    p = torch.stack(feats).mean(dim=0)
    return p / p.norm(), len(feats)


def main():
    bank = U.get_bank()
    summary = {}
    for name, dataset, cls, domain, prefix, dior_cls in U.EVALS:
        if dataset != 'coco':
            continue
        print('\n' + '=' * 70)
        print(f'[{name}]  few-shot K={K}')
        images, gts = U.load_coco(cls, n=N_COCO)
        fs_imgs, fs_gts = images[:K], gts[:K]
        test_imgs, test_gts = images[K:], gts[K:]

        P_src = bank[dior_cls][LAYER]['prototype'].to(U.DEVICE)
        neg = get_neg_proto(bank, dior_cls, LAYER)
        P_tgt, n_fs = extract_fewshot_proto(fs_imgs, fs_gts)
        print(f'  few-shot 原型: {n_fs} 个 GT 框 | 源域-目标域原型 cosine = {float(P_src @ P_tgt):.4f}')

        # 测试集检测 + 编码
        dets = U.gdino_detect(test_imgs, f'{prefix} {cls}', 0.0)
        boxes_all, gd_all, feats = [], [], []
        for (boxes, scores), img in zip(dets, test_imgs):
            keep = scores > U.PRE_FILTER
            if not keep.any():
                continue
            for box, sc in zip(boxes[keep], scores[keep]):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                feats.append(encode_l12(img.crop((x1, y1, x2, y2))))
                boxes_all.append([x1, y1, x2, y2])
                gd_all.append(float(sc))
        gt_flat = [g for gs in test_gts for g in gs]
        gd = np.clip(np.array(gd_all), 1e-6, 1.0)

        base = U.compute_ap50(boxes_all, gd, gt_flat)
        print(f'  baseline(test)     = {base:.4f}  ({len(test_imgs)} 图)')
        row = {'baseline': base}

        for alpha in ALPHAS:
            if P_tgt is None:
                P_cal = P_src
            else:
                P_cal = alpha * P_src + (1 - alpha) * P_tgt
                P_cal = P_cal / P_cal.norm()
            vis = [max(0.0, float((f @ P_cal).item()) - NEG_LAM * float((f @ neg).item()))
                   for f in feats]
            ap = U.compute_ap50(boxes_all, np.sqrt(gd * np.clip(vis, 1e-6, 1.0)), gt_flat)
            row[f'a{alpha}'] = ap
            print(f'  α={alpha:.1f} ({"源域" if alpha==1 else "few-shot" if alpha==0 else "校准"})  AP={ap:.4f}')

        summary[name] = row

    # 汇总
    print('\n\n' + '=' * 90)
    cols = ['baseline'] + [f'a{a}' for a in ALPHAS]
    print(f'{"Test":<16s}' + ''.join(f'{c:>12s}' for c in cols))
    print('-' * 90)
    for name, r in summary.items():
        print(f'{name:<16s}' + ''.join(f'{r[c]:>12.4f}' for c in cols))
    print('-' * 90)
    print(f'{"总均值":<16s}' + ''.join(
        f'{np.mean([summary[n][c] for n in summary]):>12.4f}' for c in cols))

    print('\nΔ vs baseline:')
    print(f'{"Test":<16s}' + ''.join(f'{c:>12s}' for c in cols[1:]))
    for name, r in summary.items():
        print(f'{name:<16s}' + ''.join(f'{r[c]-r["baseline"]:>+12.4f}' for c in cols[1:]))

    print('\n✅ few-shot 校准完成')


if __name__ == '__main__':
    main()
