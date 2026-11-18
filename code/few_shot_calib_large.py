# -*- coding: utf-8 -*-
"""few_shot_calib_large.py — few-shot 校准 × dinov2-large (1024 维)

与 few_shot_calib.py 逻辑一致, 仅换 dinov2-large 模型 + large bank。
验证: 更强 backbone 叠加 few-shot 校准, 能否进一步救 boat。
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
from torchvision import transforms as T
import unified_eval as U

LAYER = 11
K = 10
NEG_LAM = 0.5
ALPHAS = [1.0, 0.5, 0.3, 0.0]
N_COCO = 100
LARGE_PATH = r'C:\Users\张三\.cache\modelscope\hub\models\facebook\dinov2-large'
LARGE_BANK = r'C:\Users\张三\Desktop\实验数据\exemplar_bank_k5_dinov2_large_layers.pt'

_ld = {}


def get_large():
    if 'model' not in _ld:
        from transformers import AutoModel, AutoConfig
        cfg = AutoConfig.from_pretrained(LARGE_PATH)
        cfg.output_hidden_states = True
        _ld['model'] = AutoModel.from_pretrained(LARGE_PATH, config=cfg).eval().to(U.DEVICE)
        _ld['pre'] = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return _ld['model'], _ld['pre']


def get_bank_large():
    if 'bank' not in _ld:
        _ld['bank'] = torch.load(LARGE_BANK, map_location='cpu')
    return _ld['bank']


def encode_l12(crop):
    model, pre = get_large()
    t = pre(crop).unsqueeze(0).to(U.DEVICE)
    with torch.no_grad():
        o = model(t)
    f = o.hidden_states[LAYER + 1][0, 0, :]
    return f / f.norm()


def get_neg_proto(bank, dior_cls, layer):
    pos = bank[dior_cls][layer]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][layer]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return neg / neg.norm()


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
    bank = get_bank_large()
    summary = {}
    for name, dataset, cls, domain, prefix, dior_cls in U.EVALS:
        if dataset != 'coco':
            continue
        print('\n' + '=' * 70)
        print(f'[{name}]  few-shot K={K} (dinov2-large)')
        images, gts = U.load_coco(cls, n=N_COCO)
        fs_imgs, fs_gts = images[:K], gts[:K]
        test_imgs, test_gts = images[K:], gts[K:]

        P_src = bank[dior_cls][LAYER]['prototype'].to(U.DEVICE)
        neg = get_neg_proto(bank, dior_cls, LAYER)
        P_tgt, n_fs = extract_fewshot_proto(fs_imgs, fs_gts)
        print(f'  few-shot 原型: {n_fs} 框 | 源域-目标域 cosine = {float(P_src @ P_tgt):.4f}')

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
            P_cal = P_src if P_tgt is None else (alpha * P_src + (1 - alpha) * P_tgt)
            P_cal = P_cal / P_cal.norm()
            vis = [max(0.0, float((f @ P_cal).item()) - NEG_LAM * float((f @ neg).item()))
                   for f in feats]
            ap = U.compute_ap50(boxes_all, np.sqrt(gd * np.clip(vis, 1e-6, 1.0)), gt_flat)
            row[f'a{alpha}'] = ap
            print(f'  α={alpha:.1f}  AP={ap:.4f}')

        summary[name] = row

    cols = ['baseline'] + [f'a{a}' for a in ALPHAS]
    print('\n\n' + '=' * 90)
    print(f'{"Test":<16s}' + ''.join(f'{c:>12s}' for c in cols))
    print('-' * 90)
    for name, r in summary.items():
        print(f'{name:<16s}' + ''.join(f'{r[c]:>12.4f}' for c in cols))
    print('-' * 90)
    print(f'{"总均值":<16s}' + ''.join(f'{np.mean([summary[n][c] for n in summary]):>12.4f}' for c in cols))
    print('\nΔ vs baseline:')
    print(f'{"Test":<16s}' + ''.join(f'{c:>12s}' for c in cols[1:]))
    for name, r in summary.items():
        print(f'{name:<16s}' + ''.join(f'{r[c]-r["baseline"]:>+12.4f}' for c in cols[1:]))
    print('\n✅ few-shot large 校准完成')


if __name__ == '__main__':
    main()
