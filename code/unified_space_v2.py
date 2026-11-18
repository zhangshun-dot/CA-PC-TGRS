# -*- coding: utf-8 -*-
"""unified_space_v2.py — 信息统一框架 v2 (按元反馈第 3-8 节重做)

v0/v1 失败根因(已诊断):
  - zc 强不变性压塌 (rank 1.83/64, probe 0.22)
  - zk 从压缩瓶颈出 + 重建忽略 zk -> zk 塌常数
  - 条件目标 (brightness/contrast/blur) 单图不可识别 (基线最大 |R|=0.267)

v2 改动:
  1) 条件 c 换成【分辨率降级倍数 k∈{1,2,3,4}】: 单图可识别(清晰vs模糊块), 且重建可去模糊
  2) 结构: 共享浅层 stem -> 并行 zc(内容)/zk(条件)  [非 x->zc->zk]
  3) zc 监督: VICReg(var+cov 防塌缩) + 跨条件弱不变 warm-up
  4) zk 监督: 直接回归 c (MSE, 强 λ)
  5) 重建: decode(zc,zk) -> 清晰图, 强制 zk 参与(条件依赖)
  6) λ_inv warm-up: 前 T_warm epoch 线性增到 λ_max, 监控 zc probe 不掉

接口: encode(x)->(zc,zk), decode(zc,zk), pooled_zc。
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
from torchvision import transforms
import unified_eval as U

DEVICE = U.DEVICE
CONTENT_CH = 64
K_CHOICES = [1, 2, 3, 4]


def degrade_res(pil, k):
    """分辨率降级: k=1 原图; k>1 下采样 k 倍再上采样回原尺寸(产生模糊块)。"""
    if k <= 1:
        return pil
    w, h = pil.size
    return pil.resize((max(1, w // k), max(1, h // k))).resize((w, h))


def sample_res():
    k = random.choice(K_CHOICES)
    c = (k - 1) / (len(K_CHOICES) - 1)   # 0=清晰, 1=最模糊
    return k, np.array([c], dtype=np.float32)


class CondDisentanglerV2(nn.Module):
    def __init__(self, content_ch=CONTENT_CH):
        super().__init__()
        # 共享浅层 stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU())
        # zc 内容支路 (并行)
        self.bneck = nn.Conv2d(64, content_ch, 3, padding=1)
        self.film = nn.Linear(1, content_ch * 2)
        # zk 条件支路 (并行, 从 stem 全局池)
        self.cond_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                       nn.Linear(64, 1))
        # 解码器
        self.dec3 = nn.ConvTranspose2d(content_ch, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.head = nn.Conv2d(32, 3, 3, padding=1)

    def encode(self, x):
        h = self.stem(x)
        zc = F.relu(self.bneck(h))
        zk = self.cond_head(h)
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


def vicreg(zc, gamma=1.0, lambda_var=1.0, lambda_cov=1.0):
    """zc: (B, D) 或 (B,D,H,W). 返回 (L_var, L_cov)。"""
    if zc.dim() == 4:
        zc = zc.flatten(2).mean(dim=2)          # (B, D) 空间平均, 稳
    B, D = zc.shape
    zc = zc - zc.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0) + 1e-4)
    L_var = torch.mean(F.relu(gamma - std))
    cov = (zc.T @ zc) / (B - 1)
    cov = cov - torch.diag(torch.diag(cov))
    L_cov = (cov ** 2).sum() / D
    return lambda_var * L_var, lambda_cov * L_cov


def train_net(net, epochs=40, bs=16, save_path=None,
              lam_cond=5.0, lam_inv_max=0.3, lam_rec=1.0, lam_cross=1.0,
              zk_var_target=0.02, T_warm=10):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    paths = glob.glob(os.path.join(U.DIOR_IMG_DIR, '*.jpg'))
    random.seed(1); random.shuffle(paths)
    imgs = [Image.open(p).convert('RGB').resize((384, 384)) for p in paths]
    print(f'[train v2] {len(imgs)} DIOR 图, {epochs} epoch, batch {bs}', flush=True)
    nb = max(1, len(imgs) // bs)
    for ep in range(epochs):
        random.shuffle(imgs)
        lam_inv = lam_inv_max * min(1.0, (ep + 1) / T_warm)
        tot = 0.0
        for i in range(0, len(imgs), bs):
            batch = imgs[i:i + bs]
            xc = torch.stack([transforms.ToTensor()(b) for b in batch]).to(DEVICE)
            ks, cs = zip(*[sample_res() for _ in batch])
            xa = torch.stack([transforms.ToTensor()(degrade_res(b, k)).to(DEVICE)
                              for b, k in zip(batch, ks)])
            cv = torch.tensor(np.stack(cs), dtype=torch.float32).to(DEVICE)
            recon, zc_a, zk_a = net(xa)
            with torch.no_grad():
                zc_c, _ = net.encode(xc)
            # 重建目标 = 降级图 xa (含条件), 而非清晰 xc:
            # 否则 zc 已含全部内容, 重建不需 zk -> 解码器忽略 zk (元第5节).
            # 目标 xa 含降级, decode(zc,zk) 必须靠 zk 还原"降级程度" -> zk 被迫参与.
            L_rec = F.l1_loss(recon, xa)
            L_inv = F.l1_loss(zc_a.flatten(1), zc_c.flatten(1).detach())
            # 交叉重建(元第5节): 内容用清晰 xc 的 zc, 条件用降级 xa 的 zk -> 重建 xa
            # 迫使 zk 独立携带"降级程度"(zc 来自 xc 无降级, 重建 xa 模糊只能靠 zk)
            rec_cross = net.decode(zc_c, zk_a)
            L_cross = F.l1_loss(rec_cross, xa)
            L_cond = F.mse_loss(zk_a, cv)
            Lv_zc, Lc_zc = vicreg(zc_a)
            Lv_zc_c, _ = vicreg(zc_c)
            L_var_zk = torch.relu(torch.tensor(zk_var_target, device=DEVICE) - zk_a.var(dim=0).mean())
            loss = (lam_rec * L_rec + lam_inv * L_inv + lam_cross * L_cross
                    + lam_cond * L_cond + Lv_zc + Lc_zc + Lv_zc_c + 0.05 * L_var_zk)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f'  ep {ep+1}/{epochs} λ_inv={lam_inv:.2f} loss={tot/nb:.4f} '
              f'(rec={L_rec.item():.3f} cross={L_cross.item():.3f} inv={L_inv.item():.3f} '
              f'cond={L_cond.item():.3f} vk={zk_a.var(dim=0).mean().item():.3f})', flush=True)
    if save_path:
        torch.save(net.state_dict(), save_path)
        print(f'[train v2] 已保存 -> {save_path}', flush=True)


def pooled_zc(net, images):
    net.eval()
    out = []
    for im in images:
        x = transforms.ToTensor()(im).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            zc, _ = net.encode(x)
            v = F.adaptive_avg_pool2d(zc, 1).flatten(1)
            v = v / v.norm(dim=-1, keepdim=True)
        out.append(v.squeeze(0).cpu())
    return torch.stack(out)


if __name__ == '__main__':
    net = CondDisentanglerV2().to(DEVICE)
    train_net(net, epochs=40, bs=16, save_path=U.SAVE_DIR / 'unified_space_v2c.pt')
