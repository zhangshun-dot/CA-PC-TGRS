# -*- coding: utf-8 -*-
"""large_layer_scan.py — dinov2-large 逐层 CDVM 判别力扫描（car + boat）

验证 large 的 L12 失效是「层选错」还是「本质退化」:
  若浅层(L8-L11)有判别力 → large 需更浅层, 结论修正
  若全层退化 → large 是负面结果, base 是甜点
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
from torchvision import transforms as T
import unified_eval as U

NEG_LAM = 0.5
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


def encode_all_layers(crop):
    model, pre = get_large()
    t = pre(crop).unsqueeze(0).to(U.DEVICE)
    with torch.no_grad():
        o = model(t)
    return [o.hidden_states[l + 1][0, 0, :] for l in range(12)]


def get_neg_proto(bank, dior_cls, layer):
    pos = bank[dior_cls][layer]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][layer]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return neg / neg.norm()


def main():
    bank = get_bank_large()
    for name, dataset, cls, domain, prefix, dior_cls in U.EVALS:
        if name not in ('COCO-boat', 'COCO-car'):
            continue
        print('\n' + '=' * 70)
        print(f'[{name}]  dinov2-large 逐层 CDVM')
        images, gts = U.load_coco(cls, n=N_COCO)
        dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)

        boxes_all, gd_all, feats_all = [], [], []
        for (boxes, scores), img in zip(dets, images):
            keep = scores > U.PRE_FILTER
            if not keep.any():
                continue
            for box, sc in zip(boxes[keep], scores[keep]):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                feats_all.append([f / f.norm() for f in encode_all_layers(img.crop((x1, y1, x2, y2)))])
                boxes_all.append([x1, y1, x2, y2])
                gd_all.append(float(sc))

        gt_flat = [g for gs in gts for g in gs]
        gd = np.clip(np.array(gd_all), 1e-6, 1.0)
        base = U.compute_ap50(boxes_all, gd, gt_flat)
        print(f'  baseline = {base:.4f}')

        aps = []
        for l in range(12):
            pos = bank[dior_cls][l]['prototype'].to(U.DEVICE)
            neg = get_neg_proto(bank, dior_cls, l)
            vis = [max(0.0, float((f[l] @ pos).item()) - NEG_LAM * float((f[l] @ neg).item()))
                   for f in feats_all]
            ap = U.compute_ap50(boxes_all, np.sqrt(gd * np.clip(vis, 1e-6, 1.0)), gt_flat)
            aps.append(ap)
            print(f'  L{l+1:>2d}  AP={ap:.4f}  (Δ={ap-base:+.4f})')

        print(f'  → 最佳层 L{int(np.argmax(aps))+1} (AP={max(aps):.4f})')

    print('\n✅ large 层扫描完成')


if __name__ == '__main__':
    main()
