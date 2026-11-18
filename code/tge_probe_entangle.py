# -*- coding: utf-8 -*-
"""tge_probe_entangle.py — 类间纠缠测量: "放大-抑制"假设的直接检验

假设: 语言外推会把原型推向目标域, 同时放大与近邻类的纠缠 (类间余弦↑),
负原型相对化恰好切除这个纠缠维度 (与 P_neg 方向的相关↑)。

检验 (CLIP 空间):
  A. 纠缠放大: cos(P̂_y, P̂_y') vs cos(P_src_s, P_src_s') —— 外推后类间余弦是否系统性更高
     (重点: vehicle 源 → {bus,car,truck,moto} 外推组, 应显著高于源 vehicle 与自身的结构)
  B. 负原型相关性: cos(P̂_y, P_neg) vs cos(P_src, P_neg) —— 外推原型是否更贴近负原型方向

输出: 控制台表 + 实验数据/tge_probe_entangle.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_probe_entangle.json'

BETA = 2.0
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'

# (coco 类, 源 DIOR 类)
TARGETS = [
    ('airplane',   'airplane'),
    ('boat',       'ship'),
    ('car',        'vehicle'),
    ('truck',      'vehicle'),
    ('bus',        'vehicle'),
    ('motorcycle', 'vehicle'),
]


def build_proto(cls):
    images, gts = U.load_dior(cls)
    crops = []
    for img, g in zip(images, gts):
        for box in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 > x1 + 10 and y2 > y1 + 10:
                crops.append(img.crop((x1, y1, x2, y2)))
    F = T.encode_clip_batch(crops)
    p = F.mean(dim=0)
    return p / p.norm()


def main():
    # 源原型 (3 个 DIOR 类)
    print('[1/2] DIOR 源原型 (CLIP) ...')
    srcs = {s: build_proto(s) for s in ['airplane', 'ship', 'vehicle']}

    # 外推原型 (6 个 COCO 类, joint@2)
    print('[2/2] 外推原型 + 纠缠测量 ...')
    hat = {}
    deltas = {}
    for coco, s in TARGETS:
        s_xs = T.text_emb([SRC_TMPL.format(s)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco)]).cpu()[0]
        delta = d_y - s_xs
        deltas[coco] = delta
        P_hat = srcs[s] + BETA * delta.to(U.DEVICE)
        hat[coco] = (P_hat / P_hat.norm()).cpu()

    rows = {}
    # A) 类间余弦: 源侧 (3×3) vs 外推侧 (6×6)
    src_cos = {}
    for a in ['airplane', 'ship', 'vehicle']:
        for b in ['airplane', 'ship', 'vehicle']:
            src_cos[(a, b)] = float((srcs[a].cpu() @ srcs[b].cpu()).item())
    hat_cos = {}
    for a, _ in TARGETS:
        for b, _ in TARGETS:
            hat_cos[(a, b)] = float((hat[a] @ hat[b]).item())

    print('\n[A] 类间余弦 (外推后, 6×6 上三角):')
    names = [c for c, _ in TARGETS]
    print(f'{"":>12s}' + ''.join(f'{n[:4]:>8s}' for n in names))
    for a in names:
        print(f'{a:>12s}' + ''.join(f'{hat_cos[(a, b)]:>8.3f}' for b in names))

    # 重点: vehicle 源组内 (bus/car/truck/moto) 外推后的纠缠 vs 源 vehicle 与 ship/airplane
    veh_group = ['bus', 'car', 'truck', 'motorcycle']
    hat_intra = np.mean([hat_cos[(a, b)] for i, a in enumerate(veh_group)
                         for b in veh_group[i + 1:]])
    src_intra = np.mean([src_cos[('vehicle', 'ship')], src_cos[('vehicle', 'airplane')],
                         src_cos[('ship', 'airplane')]])
    print(f'\n  外推后 vehicle 组内均值余弦: {hat_intra:.3f}   (bus/car/truck/moto 两两)')
    print(f'  源侧 3 类均值余弦:          {src_intra:.3f}   (airplane/ship/vehicle 两两)')
    print(f'  → 纠缠放大: {"✅" if hat_intra > src_intra + 0.05 else ("≈" if abs(hat_intra - src_intra) < 0.05 else "❌")}')

    # B) 负原型相关性: cos(P̂, P_neg) vs cos(P_src, P_neg)
    print('\n[B] 与负原型 (CLIP top-3 近邻均值) 的余弦:')
    print(f'{"类":<12s} {"cos(源,P_neg)":>14s} {"cos(外推,P_neg)":>16s} {"变化":>8s}')
    b_rows = {}
    for coco, s in TARGETS:
        # 负原型: 其余源类均值 (仅 3 源类, 不能用 topk(3))
        others = torch.stack([v for k, v in srcs.items() if k != s])
        P_neg = (others.mean(dim=0) / others.mean(dim=0).norm()).cpu()
        c_src = float((srcs[s].cpu() @ P_neg).item())
        c_hat = float((hat[coco] @ P_neg).item())
        b_rows[coco] = {'cos_src_neg': round(c_src, 4), 'cos_hat_neg': round(c_hat, 4)}
        print(f'{coco:<12s} {c_src:>+14.3f} {c_hat:>+16.3f} {c_hat - c_src:>+8.3f}')

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'beta': BETA, 'hat_cos': {f'{a}|{b}': round(v, 4) for (a, b), v in hat_cos.items()},
                   'src_cos': {f'{a}|{b}': round(v, 4) for (a, b), v in src_cos.items()},
                   'veh_group_mean': round(hat_intra, 4), 'src_mean': round(src_intra, 4),
                   'neg_cos': b_rows}, f, indent=2, ensure_ascii=False)
    print(f'\n✅ -> {OUT}')


if __name__ == '__main__':
    main()
