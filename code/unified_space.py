# -*- coding: utf-8 -*-
"""unified_space.py — 信息统一框架 v0: 条件解耦 (content / condition)

核心(对应长论述修正定义): 统一信息空间 ≠ 单一 embedding, 而是把观测 x 分解成
  (z_content, z_cond) — 内容(对观测条件不变) + 条件(显式变量, 不消除)
第一层职责 = 观测条件分解(条件解耦), 不是"统一/修复成一张干净图"。

自监督训练(无需框标注): 对 DIOR 图施加已知合成条件扰动 c=(亮度,对比,模糊),
  L = L_recon + λ_inv*内容不变性(zc(x_aug)≈zc(x_clean)) + λ_cond*条件回归(zk≈c)
验证:
  ① 跨条件一致性: 同图不同 c 下 zc 余弦 vs CLIP 整图余弦(baseline) → 高=分解成功
  ② 条件预测 MSE: zk 回归 c 的误差 → 低=条件显式化
  ③ 内容保真: 用 zc 重建图(条件取均值, 近似标准条件)喂 GDINO, 与 raw 比 AP50
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

DEVICE = U.DEVICE
COND_DIM = 3  # (亮度, 对比, 模糊) 归一化到 [0,1]


def sample_cond():
    """随机合成观测条件 c (归一化 [0,1]): 亮度/对比/模糊。"""
    b = random.uniform(-0.06, 0.06)
    g = random.uniform(0.85, 1.15)
    sigma = random.uniform(0.0, 1.5)
    return np.array([(b + 0.06) / 0.12, (g - 0.85) / 0.30, sigma / 1.5], dtype=np.float32)


def apply_cond(img, c_vec):
    """按 c_vec 对 PIL 图施加合成观测条件, 返回 PIL。"""
    b = float(c_vec[0] * 0.12 - 0.06)
    g = float(c_vec[1] * 0.30 + 0.85)
    sigma = float(c_vec[2] * 1.5)
    im = np.asarray(img).astype(np.float32) / 255.0
    im = np.clip(im ** g, 0, 1)
    im = np.clip(im + b, 0, 1)
    p = Image.fromarray((im * 255).astype(np.uint8))
    if sigma > 0.01:
        p = p.filter(ImageFilter.GaussianBlur(sigma))
    return p


class CondDisentangler(nn.Module):
    def __init__(self, cond_dim=COND_DIM, content_ch=64):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.enc3 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.bottleneck = nn.Conv2d(64, content_ch, 3, padding=1)
        # 条件头: 从 bottleneck 全局池化回归 c
        self.cond_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(content_ch, cond_dim))
        # 条件 FiLM 调制 bottleneck
        self.film = nn.Linear(cond_dim, content_ch * 2)
        # 解码器
        self.dec3 = nn.ConvTranspose2d(content_ch, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.head = nn.Conv2d(32, 3, 3, padding=1)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        zc = F.relu(self.bottleneck(h))               # (B, content_ch, H/4, W/4)
        zk = self.cond_head(zc)                        # (B, cond_dim)
        return zc, zk

    def decode(self, zc, zk):
        gamma, beta = self.film(zk).chunk(2, dim=1)    # (B, content_ch)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        zf = zc * (1.0 + gamma) + beta
        d = F.relu(self.dec3(zf))
        d = F.relu(self.dec2(d))
        return torch.clamp(self.head(d), 0.0, 1.0)

    def forward(self, x):
        zc, zk = self.encode(x)
        return self.decode(zc, zk), zc, zk


def train_net(net, epochs=30, bs=4, save_path=None, lam_inv=1.0, lam_cond=1.0):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    paths = glob.glob(os.path.join(U.DIOR_IMG_DIR, '*.jpg'))
    random.seed(1); random.shuffle(paths)
    imgs = [Image.open(p).convert('RGB').resize((384, 384)) for p in paths]
    print(f'[train] {len(imgs)} DIOR 图, {epochs} epoch, batch {bs}', flush=True)
    nb = max(1, len(imgs) // bs)
    for ep in range(epochs):
        random.shuffle(imgs)
        tot = 0.0
        for i in range(0, len(imgs), bs):
            batch = imgs[i:i + bs]
            xc = torch.stack([transforms.ToTensor()(b) for b in batch]).to(DEVICE)
            # 同批施加随机条件
            c_vecs = [sample_cond() for _ in batch]
            xa = torch.stack([transforms.ToTensor()(apply_cond(b, cv)).to(DEVICE)
                              for b, cv in zip(batch, c_vecs)])
            cv_t = torch.tensor(np.stack(c_vecs), dtype=torch.float32).to(DEVICE)
            recon, zc_a, zk_a = net(xa)
            with torch.no_grad():
                zc_c, _ = net.encode(xc)
            L_recon = F.l1_loss(recon, xa)
            L_inv = F.l1_loss(zc_a.flatten(1), zc_c.flatten(1).detach())
            L_cond = F.mse_loss(zk_a, cv_t)
            loss = L_recon + lam_inv * L_inv + lam_cond * L_cond
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f'  epoch {ep+1}/{epochs} loss={tot/nb:.4f} '
              f'(recon={L_recon.item():.3f} inv={L_inv.item():.3f} cond={L_cond.item():.3f})', flush=True)
    if save_path:
        torch.save(net.state_dict(), save_path)
        print(f'[train] 已保存 -> {save_path}', flush=True)


def pooled_zc(net, images):
    """对一批 PIL 图返回 L2 归一化的内容表征 (B, content_ch)。"""
    net.eval()
    out = []
    for im in images:
        x = transforms.ToTensor()(im).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            zc, _ = net.encode(x)
            v = F.adaptive_avg_pool2d(zc, 1).flatten(1)
            v = v / v.norm(dim=-1, keepdim=True)
        out.append(v.squeeze(0).cpu())
    return torch.stack(out)  # (B, content_ch)


def harmonize_d(images, net, zk_mean=None):
    """用 z_content 重建(条件取均值=近似标准条件, 非完全消除), 返回 PIL 列表。"""
    if zk_mean is None:
        zk_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).to(DEVICE)
    net.eval()
    out = []
    for im in images:
        x = transforms.ToTensor()(im).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            zc, _ = net.encode(x)
            p = net.decode(zc, zk_mean.unsqueeze(0)).cpu()
        out.append(transforms.ToPILImage()(p.squeeze(0)))
    return out


if __name__ == '__main__':
    net = CondDisentangler().to(DEVICE)
    train_net(net, epochs=30, bs=4, save_path=U.SAVE_DIR / 'unified_space.pt')
