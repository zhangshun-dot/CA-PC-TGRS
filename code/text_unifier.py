# -*- coding: utf-8 -*-
"""text_unifier.py — (c) 文本锚定语义统一 的「学习型」实现

TextConditionedEnhancer: 轻量 UNet 残差增强网络, 以类文本嵌入为 FiLM 条件,
输出小残差加到原图 (img' = clamp(img + r, 0, 1))。残差结构天然保结构。

训练(自监督, 无框标注):
  L = L_align + 0.5*L_preserve_img + 0.5*L_preserve_feat
  L_align          = 1 - cos(CLIP_img(img'), text_anchor)        # 拉向统一信息空间(文本锚)
  L_preserve_img   = L1(img', img)                                # 像素级保结构
  L_preserve_feat  = 1 - cos(CLIP_img(img'), CLIP_img(img))      # 保检测器依赖的语义/几何特征
  text_anchor = 6类COCO文本嵌入均值 (固定, 全局统一空间, 无 per-image oracle)

设计约束: RTX 5060 / 8GB。CLIP 冻结, 仅小网络反向; 384 输入, batch 4。
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
import clip as clip_module
import unified_eval as U

DEVICE = U.DEVICE
COCO6 = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def clip_pre_t(x):
    """把 0-1 图像张量 (B,3,H,W) 直接转成 CLIP ViT-B-32 输入 (B,3,224,224) 归一化。

    跳过 PIL 往返: BILINEAR 缩到 224 (= CLIP Resize+CenterCrop 在 224 输入下等价),
    再按 CLIP 均值/标准差归一。
    """
    x = F.interpolate(x, (224, 224), mode='bilinear', align_corners=False)
    if x.device != torch.device('cpu'):
        mean = _CLIP_MEAN.to(x.device)
        std = _CLIP_STD.to(x.device)
    else:
        mean, std = _CLIP_MEAN, _CLIP_STD
    return (x - mean) / std


class TextConditionedEnhancer(nn.Module):
    def __init__(self, text_dim=512):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.enc3 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.bottleneck = nn.Conv2d(64, 64, 3, padding=1)
        self.film = nn.Linear(text_dim, 64 * 2)          # gamma, beta
        self.dec3 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.head = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x, text_emb):
        # x: (B,3,H,W) 0-1 ; text_emb: (B,512)
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        hb = F.relu(self.bottleneck(h))
        gamma, beta = self.film(text_emb).chunk(2, dim=1)        # (B,64)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)                # (B,64,1,1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        hb = hb * (1.0 + gamma) + beta
        d = F.relu(self.dec3(hb))
        d = F.relu(self.dec2(d))
        r = torch.tanh(self.head(d)) * 0.3                       # 小残差, 保结构
        return torch.clamp(x + r, 0.0, 1.0)


def build_anchor(clip_m):
    """统一信息空间锚: 6类COCO文本嵌入均值 (固定, 全局, 无 oracle)。"""
    toks = clip_module.tokenize([f'a photo of {c}' for c in COCO6]).to(DEVICE)
    with torch.no_grad():
        t = clip_m.encode_text(toks).float()
        t = t / t.norm(dim=-1, keepdim=True)
    return t.mean(0, keepdim=True)                               # (1,512)


def train_net(net, anchor, epochs=40, bs=4, save_path=None):
    clip_m, _ = U.get_clip()
    clip_m.eval()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    paths = glob.glob(os.path.join(U.DIOR_IMG_DIR, '*.jpg'))
    random.seed(0)
    random.shuffle(paths)
    imgs = [Image.open(p).convert('RGB').resize((384, 384)) for p in paths]
    print(f'[train] {len(imgs)} DIOR 图, {epochs} epoch, batch {bs}', flush=True)
    n_batches = max(1, len(imgs) // bs)
    for ep in range(epochs):
        random.shuffle(imgs)
        tot = 0.0
        for i in range(0, len(imgs), bs):
            batch = imgs[i:i + bs]
            x = torch.stack([transforms.ToTensor()(b) for b in batch]).to(DEVICE)
            pred = net(x, anchor.repeat(len(batch), 1))
            fp = clip_m.encode_image(clip_pre_t(pred)).float()
            fp = fp / fp.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                fi = clip_m.encode_image(clip_pre_t(x)).float()
                fi = fi / fi.norm(dim=-1, keepdim=True)
            l_align = (1.0 - (fp @ anchor.T).mean())
            l_pi = F.l1_loss(pred, x)
            l_pf = (1.0 - (fp @ fi.T).mean())
            loss = l_align + 0.5 * l_pi + 0.5 * l_pf
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f'  epoch {ep+1}/{epochs} loss={tot/n_batches:.4f} '
              f'(align={l_align.item():.3f} pi={l_pi.item():.3f} pf={l_pf.item():.3f})', flush=True)
    if save_path:
        torch.save(net.state_dict(), save_path)
        print(f'[train] 已保存 -> {save_path}', flush=True)


def harmonize(images, net, anchor):
    """对一批 PIL 图做文本锚定统一, 返回统一后的 PIL 图列表。"""
    net.eval()
    out = []
    for im in images:
        x = transforms.ToTensor()(im).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            p = net(x, anchor).cpu()
        out.append(transforms.ToPILImage()(p.squeeze(0)))
    return out


if __name__ == '__main__':
    net = TextConditionedEnhancer().to(DEVICE)
    clip_m, _ = U.get_clip()
    anchor = build_anchor(clip_m)
    train_net(net, anchor, epochs=40, bs=4, save_path=U.SAVE_DIR / 'text_unifier.pt')
