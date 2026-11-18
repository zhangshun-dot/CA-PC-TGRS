# -*- coding: utf-8 -*-
"""unified_space_v3.py — 按评审第5节改造: zc 内容任务监督 + 弱不变 + 交叉重建 + zk 泄漏可控

评审核心要求:
  - zc 加直接内容监督 CE(zc->y), 先确保 zc->y(非只靠重建间接得内容)  [治有效秩2.34/塌]
  - 继续弱不变(λ_inv warm-up), 配合 VICReg + 有效秩监控
  - zk 预测条件, 但测内容泄漏 zk->y(应弱)
  - 严格条件依赖: 用独立分辨率预测器 q 测 D(zc,zk) 输出条件, 非整体L1
结构继承 v2(共享浅层并行 + 交叉重建), 仅加内容头 gc 与类标签训练。
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

DEVICE = U.DEVICE
N_CLASS = 20  # DIOR 类数


def build_labeled():
    """从 DIOR 文件名解析 (path, cls_id)。文件名形如 <类名>_*.jpg。"""
    name2id = {v: k for k, v in U.DIOR_ID2NAME.items()}
    pairs = []
    for p in glob.glob(os.path.join(U.DIOR_IMG_DIR, '*.jpg')):
        stem = Path(p).stem
        for name, cid in name2id.items():
            if stem.startswith(name.replace(' ', '_')):
                pairs.append((p, cid)); break
    return pairs


class CondDisentanglerV3(US2.CondDisentanglerV2):
    def __init__(self, content_ch=US2.CONTENT_CH, n_class=N_CLASS):
        super().__init__(content_ch=content_ch)
        self.gc = nn.Linear(content_ch, n_class)   # 内容分类头(仅训练用, 逼 zc 保内容)

    def content_logits(self, zc):
        """zc: (B, content_ch, H, W) -> (B, n_class)。"""
        v = F.adaptive_avg_pool2d(zc, 1).flatten(1)
        return self.gc(v)


def vicreg(zc, gamma=1.0, lambda_var=1.0, lambda_cov=1.0):
    if zc.dim() == 4:
        zc = zc.flatten(2).mean(dim=2)
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
              lam_content=1.0, zk_var_target=0.02, T_warm=10):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    pairs = build_labeled()
    random.seed(1); random.shuffle(pairs)
    print(f'[train v3] {len(pairs)} 带标签 DIOR 图, {epochs} epoch, batch {bs}', flush=True)
    nb = max(1, len(pairs) // bs)
    for ep in range(epochs):
        random.shuffle(pairs)
        lam_inv = lam_inv_max * min(1.0, (ep + 1) / T_warm)
        tot = 0.0
        for i in range(0, len(pairs), bs):
            batch = pairs[i:i + bs]
            xc = torch.stack([transforms.ToTensor()(Image.open(p).convert('RGB').resize((384, 384)))
                              for p, _ in batch]).to(DEVICE)
            ys = torch.tensor([c for _, c in batch], dtype=torch.long).to(DEVICE)
            ks, cs = zip(*[US2.sample_res() for _ in batch])
            xa = torch.stack([transforms.ToTensor()(US2.degrade_res(
                Image.open(p).convert('RGB').resize((384, 384)), k)).to(DEVICE)
                for (p, _), k in zip(batch, ks)])
            cv = torch.tensor(np.stack(cs), dtype=torch.float32).to(DEVICE)
            recon, zc_a, zk_a = net(xa)
            with torch.no_grad():
                zc_c, _ = net.encode(xc)
            L_rec = F.l1_loss(recon, xa)
            L_inv = F.l1_loss(zc_a.flatten(1), zc_c.flatten(1).detach())
            rec_cross = net.decode(zc_c, zk_a)
            L_cross = F.l1_loss(rec_cross, xa)
            L_cond = F.mse_loss(zk_a, cv)
            L_content = F.cross_entropy(net.content_logits(zc_a), ys) \
                + F.cross_entropy(net.content_logits(zc_c), ys)
            Lv_zc, Lc_zc = vicreg(zc_a)
            Lv_zc_c, _ = vicreg(zc_c)
            L_var_zk = torch.relu(torch.tensor(zk_var_target, device=DEVICE) - zk_a.var(dim=0).mean())
            loss = (lam_rec * L_rec + lam_inv * L_inv + lam_cross * L_cross
                    + lam_cond * L_cond + lam_content * L_content
                    + Lv_zc + Lc_zc + Lv_zc_c + 0.05 * L_var_zk)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f'  ep {ep+1}/{epochs} λ_inv={lam_inv:.2f} loss={tot/nb:.4f} '
              f'(rec={L_rec.item():.3f} cross={L_cross.item():.3f} inv={L_inv.item():.3f} '
              f'cond={L_cond.item():.3f} content={L_content.item():.3f} '
              f'vk={zk_a.var(dim=0).mean().item():.3f})', flush=True)
    if save_path:
        torch.save(net.state_dict(), save_path)
        print(f'[train v3] 已保存 -> {save_path}', flush=True)


def pooled_zc(net, images):
    return US2.pooled_zc(net, images)


if __name__ == '__main__':
    net = CondDisentanglerV3().to(DEVICE)
    train_net(net, epochs=40, bs=16, save_path=U.SAVE_DIR / 'unified_space_v3.pt')
