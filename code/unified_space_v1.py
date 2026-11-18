# -*- coding: utf-8 -*-
"""unified_space_v1.py — 信息统一框架 v1 (修 v0 的两处塌缩)

v0 诊断暴露:
  - zc 部分塌缩 (有效秩 1.83/64, 线性探针 acc 0.22 vs CLIP 0.92): 不变性项 λ_inv=1.0 过强
  - zk 没学到条件 (跨条件一致性 0.998 > zc 0.985; 仅 blur 弱学到): zk 从已被压的 bottleneck 出 + recon 不真需 zk

v1 改动 (对齐 "条件解耦" 初衷, 仍自监督):
  1) λ_inv: 1.0 -> 0.2 (不再把 zc 压塌)
  2) λ_cond: 1.0 -> 5.0 (条件监督大幅加强)
  3) zk 改从【浅层】条件支路出 (raw 增强图 -> 2层卷积 -> 全局池 -> 3), 保留亮度/对比等全局强度信息
  4) 加 zc 反塌缩: 每通道方差推到 1 (L_var_zc), 防所有通道->常数
  5) 加 zk 反塌缩: zk 跨 batch 方差不低于阈值 (L_var_zk), 防 zk 塌成均值 0.5
  6) decode 仍用 zk 做 FiLM (条件调制)

接口与 v0 一致: encode(x)->(zc,zk), decode(zc,zk), pooled_zc, harmonize_d。
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
from PIL import Image, ImageFilter
from torchvision import transforms
import unified_eval as U
import unified_space as US

DEVICE = U.DEVICE
COND_DIM = US.COND_DIM


class CondDisentanglerV1(nn.Module):
    def __init__(self, cond_dim=COND_DIM, content_ch=64):
        super().__init__()
        # ── 内容支路 (深, 求对条件不变) ──
        self.enc1 = nn.Conv2d(3, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.enc3 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.bottleneck = nn.Conv2d(64, content_ch, 3, padding=1)
        self.film = nn.Linear(cond_dim, content_ch * 2)
        self.dec3 = nn.ConvTranspose2d(content_ch, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.head = nn.Conv2d(32, 3, 3, padding=1)
        # ── 条件支路 (浅, 从 raw 增强图出, 保留全局强度) ──
        self.ck1 = nn.Conv2d(3, 16, 3, padding=1, stride=2)
        self.ck2 = nn.Conv2d(16, 16, 3, padding=1, stride=2)
        self.cond_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                       nn.Linear(16, cond_dim))

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        zc = F.relu(self.bottleneck(h))
        zk = self.cond_head(F.relu(self.ck2(F.relu(self.ck1(x)))))
        return zc, zk

    def decode(self, zc, zk):
        gamma, beta = self.film(zk).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        zf = zc * (1.0 + gamma) + beta
        d = F.relu(self.dec3(zf))
        d = F.relu(self.dec2(d))
        return torch.clamp(self.head(d), 0.0, 1.0)

    def forward(self, x):
        zc, zk = self.encode(x)
        return self.decode(zc, zk), zc, zk


def train_net(net, epochs=30, bs=8, save_path=None,
              lam_inv=0.2, lam_cond=5.0, lam_var_zc=0.05, lam_var_zk=0.05,
              zk_var_target=0.03):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    paths = glob.glob(os.path.join(U.DIOR_IMG_DIR, '*.jpg'))
    random.seed(1); random.shuffle(paths)
    imgs = [Image.open(p).convert('RGB').resize((384, 384)) for p in paths]
    print(f'[train v1] {len(imgs)} DIOR 图, {epochs} epoch, batch {bs}', flush=True)
    nb = max(1, len(imgs) // bs)
    for ep in range(epochs):
        random.shuffle(imgs)
        tot = 0.0
        for i in range(0, len(imgs), bs):
            batch = imgs[i:i + bs]
            xc = torch.stack([transforms.ToTensor()(b) for b in batch]).to(DEVICE)
            c_vecs = [US.sample_cond() for _ in batch]
            xa = torch.stack([transforms.ToTensor()(US.apply_cond(b, cv)).to(DEVICE)
                              for b, cv in zip(batch, c_vecs)])
            cv_t = torch.tensor(np.stack(c_vecs), dtype=torch.float32).to(DEVICE)
            recon, zc_a, zk_a = net(xa)
            with torch.no_grad():
                zc_c, _ = net.encode(xc)
            L_recon = F.l1_loss(recon, xa)
            L_inv = F.l1_loss(zc_a.flatten(1), zc_c.flatten(1).detach())
            L_cond = F.mse_loss(zk_a, cv_t)
            # 反塌缩
            var_zc = zc_a.flatten(1).var(dim=0).mean()
            L_var_zc = (var_zc - 1.0).abs()
            var_zk = zk_a.var(dim=0).mean()
            L_var_zk = torch.relu(torch.tensor(zk_var_target, device=DEVICE) - var_zk)
            loss = (L_recon + lam_inv * L_inv + lam_cond * L_cond
                    + lam_var_zc * L_var_zc + lam_var_zk * L_var_zk)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f'  epoch {ep+1}/{epochs} loss={tot/nb:.4f} '
              f'(recon={L_recon.item():.3f} inv={L_inv.item():.3f} cond={L_cond.item():.3f} '
              f'vzc={var_zc.item():.3f} vzk={var_zk.item():.3f})', flush=True)
    if save_path:
        torch.save(net.state_dict(), save_path)
        print(f'[train v1] 已保存 -> {save_path}', flush=True)


def pooled_zc(net, images):
    return US.pooled_zc(net, images)


def harmonize_d(images, net, zk_mean=None):
    return US.harmonize_d(images, net, zk_mean)


if __name__ == '__main__':
    net = CondDisentanglerV1().to(DEVICE)
    train_net(net, epochs=30, bs=8, save_path=U.SAVE_DIR / 'unified_space_v1.pt')
