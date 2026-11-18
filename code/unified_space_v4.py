# -*- coding: utf-8 -*-
"""unified_space_v4.py — 评审③主线: 条件估计量, 非生成变量

与 v2/v3 最大区别: 去掉解码器/重建。表示不再用于 (z_c,z_k)->x̂,
而是用于 (z_c,z_k)->ŷ。z_k 定义为 observation-condition representation (条件估计量)。

关键修复(评审②步): z_c 内容主干从"3层浅卷积"换成"冻结 DINOv2-base 特征投影",
以证伪 'v3 A 弱=统一空间不可行' 的假设 —— 若 z_c 用 DINOv2 主干后 probe 大幅提升,
说明 v3 的 A 弱是浅编码器语义能力不足, 非结构化表示本身不可行。

结构:
  x_c(清晰) -> 冻结DINOv2 -> patch tokens(14x14x768) -> 1x1 proj -> z_c(14x14x64)
  x_a(降级) -> 浅层CondEncoder -> z_k(标量, 条件强度估计)
  z_c 池化 -> g_c -> y (内容分类, L_content)
  z_k -> MSE(z_k, c) (L_cond, 条件估计)

训练只更新 proj + CondEncoder + g_c; DINOv2 冻结。
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torchvision import transforms

import unified_eval as U
import unified_space_v2 as US2
import unified_space_v3 as V3

DEVICE = U.DEVICE
CONTENT_CH = 64
N_CLASS = 20
K_CHOICES = [1, 2, 3, 4]


def build_labeled():
    return V3.build_labeled()


class ContentEncoder(nn.Module):
    """冻结 DINOv2-base -> 1x1 投影到空间 z_c (B, content_ch, 14, 14)。"""
    def __init__(self, content_ch=CONTENT_CH, freeze=True):
        super().__init__()
        self.dino, self.pre = U.get_dinov2()
        if freeze:
            for p in self.dino.parameters():
                p.requires_grad = False
        self.dino.eval()
        self.proj = nn.Conv2d(768, content_ch, 1)

    def forward(self, x_pre):
        # x_pre: (B,3,224,224) 已 dino_pre 处理 (Resize224 + Norm)
        with torch.no_grad():
            h = self.dino(x_pre).hidden_states[11][:, 1:, :]   # (B,256,768) patch tokens
        B = h.shape[0]
        h = h.reshape(B, 16, 16, 768).permute(0, 3, 1, 2)      # (B,768,16,16)
        return self.proj(h)                                    # (B,content_ch,16,16)


class CondEncoder(nn.Module):
    """浅层卷积从原始图估 z_k (条件强度标量)。独立分支, 内容低泄漏。"""
    def __init__(self, cond_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, cond_dim))

    def forward(self, x):
        return self.net(x)


class V4Net(nn.Module):
    def __init__(self, content_ch=CONTENT_CH, n_class=N_CLASS):
        super().__init__()
        self.content = ContentEncoder(content_ch)
        self.cond = CondEncoder(1)
        self.gc = nn.Linear(content_ch, n_class)

    def zc_pool(self, zc):
        return F.adaptive_avg_pool2d(zc, 1).flatten(1)

    def content_logits(self, zc):
        return self.gc(self.zc_pool(zc))

    def forward(self, xc_pre, xa):
        zc = self.content(xc_pre)          # (B,content_ch,14,14)
        zk = self.cond(xa)                 # (B,1)
        return zc, zk


def train_net(net, epochs=40, bs=16, save_path=None, lam_cond=5.0):
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    pairs = build_labeled()
    random.seed(1); random.shuffle(pairs)
    nb = max(1, len(pairs) // bs)
    print(f'[train v4] {len(pairs)} 图, {epochs} ep, bs {bs}, 冻结DINOv2主干', flush=True)
    for ep in range(epochs):
        random.shuffle(pairs)
        tot = 0.0
        for i in range(0, len(pairs), bs):
            batch = pairs[i:i + bs]
            xc_pre = torch.stack([net.content.pre(Image.open(p).convert('RGB'))
                                  for p, _ in batch]).to(DEVICE)
            ks, cs = zip(*[US2.sample_res() for _ in batch])
            xa = torch.stack([transforms.ToTensor()(
                US2.degrade_res(Image.open(p).convert('RGB').resize((384, 384)), k)).to(DEVICE)
                for (p, _), k in zip(batch, ks)])
            ys = torch.tensor([c for _, c in batch], dtype=torch.long).to(DEVICE)
            cv = torch.tensor(np.stack(cs), dtype=torch.float32).to(DEVICE)
            zc = net.content(xc_pre)
            zk = net.cond(xa)
            L_content = F.cross_entropy(net.content_logits(zc), ys)
            L_cond = F.mse_loss(zk, cv)
            loss = L_content + lam_cond * L_cond
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f'  ep {ep+1}/{epochs} loss={tot/nb:.4f} '
              f'(content={L_content.item():.3f} cond={L_cond.item():.3f} '
              f'vk={zk.var(dim=0).mean().item():.3f})', flush=True)
    if save_path:
        torch.save(net.state_dict(), save_path)
        print(f'[train v4] 已保存 -> {save_path}', flush=True)


if __name__ == '__main__':
    net = V4Net().to(DEVICE)
    train_net(net, epochs=40, bs=16, save_path=U.SAVE_DIR / 'unified_space_v4.pt')
