# -*- coding: utf-8 -*-
"""few_shot_robust.py — few-shot 校准随机划分稳健性

对跨域(COCO)3 类, 每类 N_SPLIT 次随机抽取 K=10 张 few-shot 图, 剩余测试。
报告 few-shot CDVM AP 的 mean ± std, 证明结论不依赖某一批标注图。

关键: 用统一随机种子保证可复现, 报告 vs baseline / 零样本。
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import unified_eval as U

LAYER = 11
K = 10
NEG_LAM = 0.5
ALPHA = 0.5
N_COCO = 100
N_SPLIT = 5
SEED = 42


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


def extract_proto(images, gts):
    feats = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            feats.append(encode_l12(img.crop((x1, y1, x2, y2))))
    p = torch.stack(feats).mean(dim=0)
    return p / p.norm()


def eval_on(images, gts, dets, P_cal, neg):
    boxes_all, gd_all, feats = [], [], []
    for (boxes, scores), img in zip(dets, images):
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
    gt_flat = [g for gs in gts for g in gs]
    gd = np.clip(np.array(gd_all), 1e-6, 1.0)
    vis = [max(0.0, float((f @ P_cal).item()) - NEG_LAM * float((f @ neg).item())) for f in feats]
    return U.compute_ap50(boxes_all, np.sqrt(gd * np.clip(vis, 1e-6, 1.0)), gt_flat)


def main():
    bank = U.get_bank()
    rng = np.random.default_rng(SEED)
    summary = {}
    for name, dataset, cls, domain, prefix, dior_cls in U.EVALS:
        if dataset != 'coco':
            continue
        print('\n' + '=' * 70)
        print(f'[{name}]  随机划分稳健性 (N={N_SPLIT}, K={K})')
        images, gts = U.load_coco(cls, n=N_COCO)
        dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)
        P_src = bank[dior_cls][LAYER]['prototype'].to(U.DEVICE)
        neg = get_neg_proto(bank, dior_cls, LAYER)
        n = len(images)
        idx = np.arange(n)

        aps = []
        for s in range(N_SPLIT):
            perm = rng.permutation(idx)
            fs_i, test_i = perm[:K], perm[K:]
            fs_imgs = [images[i] for i in fs_i]
            fs_gts = [gts[i] for i in fs_i]
            test_imgs = [images[i] for i in test_i]
            test_gts = [gts[i] for i in test_i]
            test_dets = [dets[i] for i in test_i]
            P_tgt = extract_proto(fs_imgs, fs_gts)
            P_cal = (ALPHA * P_src + (1 - ALPHA) * P_tgt)
            P_cal = P_cal / P_cal.norm()
            ap = eval_on(test_imgs, test_gts, test_dets, P_cal, neg)
            aps.append(ap)
            print(f'  split {s+1}: AP={ap:.4f}  (源-目标 cosine={float(P_src @ P_tgt):.4f})')

        # 零样本(α=1)在固定测试集上的参照 = 全图零样本, 用每 split 的 test 集也算
        aps_z = []
        for s in range(N_SPLIT):
            perm = rng.permutation(idx)
            test_i = perm[K:]
            test_imgs = [images[i] for i in test_i]
            test_gts = [gts[i] for i in test_i]
            test_dets = [dets[i] for i in test_i]
            aps_z.append(eval_on(test_imgs, test_gts, test_dets, P_src, neg))

        mean, std = float(np.mean(aps)), float(np.std(aps))
        mean_z, std_z = float(np.mean(aps_z)), float(np.std(aps_z))
        summary[name] = {'zero': (mean_z, std_z), 'fewshot': (mean, std)}
        print(f'  → few-shot: {mean:.4f} ± {std:.4f} | 零样本: {mean_z:.4f} ± {std_z:.4f}')

    print('\n\n' + '=' * 80)
    print(f'{"Test":<16s}{"零样本 mean±std":>22s}{"few-shot mean±std":>22s}')
    print('-' * 80)
    for name, r in summary.items():
        print(f'{name:<16s}{r["zero"][0]:>12.4f}±{r["zero"][1]:.4f}'
              f'{r["fewshot"][0]:>12.4f}±{r["fewshot"][1]:.4f}')
    print('-' * 80)
    zm = np.mean([r['zero'][0] for r in summary.values()])
    fm = np.mean([r['fewshot'][0] for r in summary.values()])
    print(f'{"跨域均值":<16s}{zm:>12.4f}          {fm:>12.4f}')
    print(f'{"Δ":<16s}{"":>22s}{fm - zm:>+12.4f}')
    print('\n✅ 稳健性验证完成')


if __name__ == '__main__':
    main()
