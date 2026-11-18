# -*- coding: utf-8 -*-
"""probe_cond_identifiable.py — 条件可识别性基线 (元反馈第 8 节: 根本问题)

为什么需要: v0/v1 的 zk 都学不到 brightness/contrast (R≈0.07~0.13)。
可能的根因不是网络, 而是条件定义本身不可识别:
  合成 c 是"施加到原图的变换", 但网络只见到变换后的图, 不知原图,
  -> 很多 (原图, c) 组合产生相同观测, c 在信息论上不可识别。
  brightness/contrast 尤其会与内容/曝光混淆; contrast 经标准化可能被抹。

本脚本量化"从单张增强图直接回归 c"的天花板:
  P_raw : 输入 raw 图 (ToTensor 0-1, 即我们网络实际输入)
  P_norm: 输入 ImageNet 标准化图 (模拟 CLIP/DINOv2 预处理)
报告每维 Pearson R / R^2 / MSE。

判读:
  - 若某维 R 也极低 -> 该条件从单图不可识别, 不能用来监督 zk,
    必须改成"可观测属性"(估计的模糊量/分辨率) 或"原图-增强图配对差"。
  - 若 P_norm 比 P_raw 明显低(尤其 contrast) -> 预处理抹掉条件, 网络输入需避免该标准化。
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import json
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import unified_eval as U
import unified_space as US

DEVICE = U.DEVICE
SAVE = U.SAVE_DIR / 'probe_cond_identifiable.json'
N = 240
IN = 128
COND_NAMES = ['brightness', 'contrast', 'blur']
SEED = 7

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class Pred(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.net(x)


def r2_p(r):
    r = float(r)
    return r * r


def main():
    print('=' * 74)
    print('条件可识别性基线 — 从单张增强图回归合成 c')
    print('=' * 74)
    random.seed(SEED)
    paths = glob.glob(os.path.join(U.DIOR_IMG_DIR, '*.jpg'))
    random.shuffle(paths)

    X_raw_l, X_norm_l, C_l = [], [], []
    norm_t = transforms.Normalize(mean=MEAN, std=STD)
    for p in paths[:N]:
        im = Image.open(p).convert('RGB').resize((IN, IN))
        c = US.sample_cond()
        aug = US.apply_cond(im, c)
        X_raw_l.append(transforms.ToTensor()(aug))
        X_norm_l.append(norm_t(transforms.ToTensor()(aug)))
        C_l.append(torch.tensor(c))
    X_raw = torch.stack(X_raw_l).to(DEVICE)
    X_norm = torch.stack(X_norm_l).to(DEVICE)
    C = torch.stack(C_l).to(DEVICE)
    print(f'  样本 {N} 图, 条件 c∈[0,1]^3 ({", ".join(COND_NAMES)})')

    def train_eval(X, tag):
        net = Pred().to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for ep in range(60):
            opt.zero_grad()
            loss = F.mse_loss(net(X), C)
            loss.backward(); opt.step()
        with torch.no_grad():
            pred = net(X).cpu().numpy()
        cgt = C.cpu().numpy()
        res = {}
        for d in range(3):
            r = np.corrcoef(pred[:, d], cgt[:, d])[0, 1]
            ss = ((pred[:, d] - cgt[:, d]) ** 2).sum()
            st = ((cgt[:, d] - cgt[:, d].mean()) ** 2).sum()
            r2 = 1 - ss / (st + 1e-12)
            mse = float(((pred[:, d] - cgt[:, d]) ** 2).mean())
            res[COND_NAMES[d]] = {'pearson_r': round(float(r), 3),
                                   'r2': round(float(r2), 3),
                                   'mse': round(mse, 4)}
        print(f'  [{tag}] 每维 (Pearson R / R^2 / MSE):')
        for d in range(3):
            print(f'    {COND_NAMES[d]:<10s} R={res[COND_NAMES[d]]["pearson_r"]:+.3f} '
                  f'R2={res[COND_NAMES[d]]["r2"]:+.3f} MSE={res[COND_NAMES[d]]["mse"]:.4f}')
        return res

    print('\n[从 raw 图回归 c] (我们网络的真实输入口径)')
    raw_res = train_eval(X_raw, 'raw')
    print('\n[从 ImageNet 标准化图回归 c] (CLIP/DINOv2 预处理口径)')
    norm_res = train_eval(X_norm, 'norm')

    # 结论
    raw_max_r = max(abs(raw_res[d]['pearson_r']) for d in COND_NAMES)
    norm_max_r = max(abs(norm_res[d]['pearson_r']) for d in COND_NAMES)
    print('\n[判读]')
    print(f'  raw 最大 |R|={raw_max_r:.3f}, norm 最大 |R|={norm_max_r:.3f}')
    for d in COND_NAMES:
        if abs(raw_res[d]['pearson_r']) < 0.25:
            print(f'  ⚠ {d}: 单图不可识别 (R={raw_res[d]["pearson_r"]:+.3f}) '
                  f'-> 不能作 zk 监督目标, 须改可观测属性/配对差')
        else:
            print(f'  ✓ {d}: 单图可识别 (R={raw_res[d]["pearson_r"]:+.3f})')
    out = {'n': N, 'raw': raw_res, 'norm': norm_res,
           'note': '条件可识别性基线: 若 raw 某维 R<0.25, 该合成条件从单图不可识别, '
                   'v2 应改用可观测属性(模糊量/分辨率)或原图-增强图配对差作 zk 监督.'}
    with open(SAVE, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n结果 -> {SAVE}')


if __name__ == '__main__':
    main()
