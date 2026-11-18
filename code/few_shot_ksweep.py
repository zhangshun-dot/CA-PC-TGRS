# -*- coding: utf-8 -*-
"""few_shot_ksweep.py — few-shot 校准的 K 灵敏度扫描

验证 few-shot 校准救 boat 是否稳健: 扫 K ∈ {5, 10, 20}, 每个 K 下扫 α。
关键看 boat 的 +0.048 是真实规律还是「前 K 张图恰好好」的巧合。
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import unified_eval as U

LAYER = 11
NEG_LAM = 0.5
ALPHAS = [1.0, 0.5, 0.3, 0.0]
KS = [5, 10, 20]
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
    # 缓存每类的图 + 候选框编码, K 扫描复用
    for name, dataset, cls, domain, prefix, dior_cls in U.EVALS:
        if dataset != 'coco':
            continue
        print('\n' + '=' * 70)
        print(f'[{name}]  K 扫描')
        images, gts = U.load_coco(cls, n=N_COCO)
        dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)

        # 编码所有图的候选框（一次, 供所有 K 复用）
        boxes_by_img, gd_by_img, feats_by_img = [], [], []
        for (boxes, scores), img in zip(dets, images):
            keep = scores > U.PRE_FILTER
            bb, gd_l, ff = [], [], []
            if keep.any():
                for box, sc in zip(boxes[keep], scores[keep]):
                    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                    x2, y2 = min(img.width, x2), min(img.height, y2)
                    if x2 <= x1 + 10 or y2 <= y1 + 10:
                        continue
                    ff.append(encode_l12(img.crop((x1, y1, x2, y2))))
                    bb.append([x1, y1, x2, y2])
                    gd_l.append(float(sc))
            boxes_by_img.append(bb)
            gd_by_img.append(gd_l)
            feats_by_img.append(ff)

        P_src = bank[dior_cls][LAYER]['prototype'].to(U.DEVICE)
        neg = get_neg_proto(bank, dior_cls, LAYER)

        print(f'  {"K":>3s} {"n_fs":>5s} ' + ''.join(f'{"α="+str(a):>10s}' for a in ALPHAS))
        for K in KS:
            P_tgt, n_fs = extract_fewshot_proto(images[:K], gts[:K])
            # 测试集 = 后 (N-K) 张
            boxes_all = [b for i in range(K, len(images)) for b in boxes_by_img[i]]
            gd_all = [g for i in range(K, len(images)) for g in gd_by_img[i]]
            feats = [f for i in range(K, len(images)) for f in feats_by_img[i]]
            gt_flat = [g for i in range(K, len(images)) for g in gts[i]]
            gd = np.clip(np.array(gd_all), 1e-6, 1.0)

            row = []
            for alpha in ALPHAS:
                if P_tgt is None:
                    P_cal = P_src
                else:
                    P_cal = alpha * P_src + (1 - alpha) * P_tgt
                    P_cal = P_cal / P_cal.norm()
                vis = [max(0.0, float((f @ P_cal).item()) - NEG_LAM * float((f @ neg).item()))
                       for f in feats]
                ap = U.compute_ap50(boxes_all, np.sqrt(gd * np.clip(vis, 1e-6, 1.0)), gt_flat)
                row.append(ap)
            print(f'  {K:>3d} {n_fs:>5d} ' + ''.join(f'{v:>10.4f}' for v in row))

        # baseline (全部 100 张)
        bb_all = [b for bl in boxes_by_img for b in bl]
        gd_all = [g for gl in gd_by_img for g in gl]
        gt_all = [g for gs in gts for g in gs]
        base = U.compute_ap50(bb_all, np.clip(np.array(gd_all), 1e-6, 1.0), gt_all)
        print(f'  baseline(全{len(images)}张) = {base:.4f}')

    print('\n✅ K 扫描完成')


if __name__ == '__main__':
    main()
