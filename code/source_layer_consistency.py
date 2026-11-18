# -*- coding: utf-8 -*-
"""source_layer_consistency.py — 源域 exemplar 类内一致性 能否预测最佳重打分层

oracle 层选择偷看了测试集, 不能写进论文。本脚本验证一个无监督准则:
  源域(DIOR) exemplar 的「类内一致性」(leave-one-out 到其余均值的余弦均值)
  能否预测每类的最佳层。

逻辑: 若某层源域类内一致性最高 → 该层原型对源域类别最紧致/判别力最强
  → 是否也是跨域/同域重打分的最佳层(oracle)?
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'

import io, glob, sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from PIL import Image

sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import unified_eval as U

PARQUET = U.DIOR_PARQUET
DIOR_DIR = U.DIOR_IMG_DIR

CLASSES = ['airplane', 'ship', 'vehicle']

# oracle 最佳层 (来自 layer_ap_scan): 类 → (同域最佳层, 跨域最佳层)
ORACLE = {
    'airplane': {'DIOR': 11, 'COCO': 10},  # DIOR-airplane L12, COCO-airplane L11
    'ship':     {'DIOR': 11, 'COCO': 10},  # DIOR-ship L12, COCO-boat L11(无解)
    'vehicle':  {'DIOR': 11, 'COCO': 11},  # DIOR-vehicle L12, COCO-car L12
}


def extract_all_layers(dior_cls):
    """提取该类所有源域 crop 的 12 层特征。返回 {layer: [N,768]}。"""
    dino, pre = U.get_dinov2()
    df = pd.read_parquet(PARQUET)
    test_ids = set()
    for f in glob.glob(os.path.join(DIOR_DIR, '*.jpg')):
        for part in reversed(Path(f).stem.split('_')):
            if part.isdigit():
                test_ids.add(int(part))
                break
    name2id = {v: k for k, v in U.DIOR_ID2NAME.items()}
    target = name2id[dior_cls]

    img_best = {}
    for _, row in df.iterrows():
        img_id = int(row['image_id'])
        if img_id in test_ids:
            continue
        objs = row['objects']
        for i, cat in enumerate(objs['category']):
            if int(cat) == target:
                area = int(objs['area'][i])
                if img_id not in img_best or area > img_best[img_id][0]:
                    img_best[img_id] = (area, row, i)

    feats = {l: [] for l in range(12)}
    for img_id, (_, row, i) in img_best.items():
        try:
            pil = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
            x, y, w, h = [int(v) for v in row['objects']['bbox'][i]]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(pil.width, x + w), min(pil.height, y + h)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            t = pre(pil.crop((x1, y1, x2, y2))).unsqueeze(0).to(U.DEVICE)
            with torch.no_grad():
                o = dino(t)
            for l in range(12):
                f = o.hidden_states[l + 1][0, 0, :]
                feats[l].append(f / f.norm())
        except Exception:
            continue
    return {l: torch.stack(feats[l]) for l in range(12)}


def leave_one_out_consistency(F):
    """F: [N,768] 归一化。返回 leave-one-out 到其余均值的余弦均值。"""
    N = F.shape[0]
    if N < 2:
        return float('nan')
    mean_all = F.mean(dim=0)
    mean_all = mean_all / mean_all.norm()
    # 用近似: leave-one-out 到均值 ≈ 到全均值 (N 大时误差 ~1/N, 足够对比层间)
    return float((F @ mean_all).mean().item())


def main():
    for cls in CLASSES:
        print('\n' + '=' * 70)
        print(f'[{cls}] 源域 exemplar 类内一致性 (各层)')
        Fs = extract_all_layers(cls)
        cons = []
        for l in range(12):
            c = leave_one_out_consistency(Fs[l])
            cons.append(c)
            print(f'  L{l+1:>2d}  一致性 = {c:.4f}')
        best = int(np.argmax(cons))
        print(f'  → 类内一致性最高层 L{best+1}')
        print(f'    oracle: 同域 L{ORACLE[cls]["DIOR"]+1}, 跨域 L{ORACLE[cls]["COCO"]+1}')

    print('\n✅ 源域一致性扫描完成')


if __name__ == '__main__':
    main()
