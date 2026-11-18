# -*- coding: utf-8 -*-
"""probe_v4_matrix.py — v4 主线验收 (评审②+③)

目标: 验证"把 z_c 内容主干从浅卷积换成冻结 DINOv2"是否让内容表示大幅变强,
从而证伪 'v3 A 弱 = 统一空间不可行' 的误判 (它更可能是浅编码器语义不足)。

对比基线 (冻结特征直接线性 probe):
  - DINOv2-base raw (patch 平均, 768d) -> y
  - CLIP ViT-B raw (image embed, 512d) -> y
  - v4 z_c (DINOv2 投影后, 64d) -> y        <- 应接近 DINOv2 基线
  - 联合 (z_c, z_k) 拼接 (65d) -> y          <- 应 ≥ z_c 单独 (条件侧信息增益)
  - v3 z_c (64d 浅卷积) -> y = 0.238 (历史参照, 不重跑, 直接对比)

指标:
  A. z_c 内容: probe acc / gc头acc / eff_rank / pairwise_cos / 跨条件一致(cos(zc(xc),zc(xa)))
  B. z_k 泄漏: zk->y acc (应低, 接近随机)
  联合: (z_c,z_k)->y vs z_c->y

输出: 实验数据/probe_v4_matrix.json + 验收表。
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import json
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
import unified_space_v4 as V4
from probe_v3_matrix import linear_probe, eff_rank   # 复用自包含探针/有效秩

DEVICE = U.DEVICE
NET_PATH = U.SAVE_DIR / 'unified_space_v4.pt'
SAVE_JSON = U.SAVE_DIR / 'probe_v4_matrix.json'
N_CLASS = 20
V3_REF_ZC_PROBE = 0.238   # v3 浅卷积主干历史值 (参照, 不重跑)
SEED = 1


def zc_pool_of(net, paths, bs=32):
    net.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch = paths[i:i + bs]
            xc_pre = torch.stack([net.content.pre(Image.open(p).convert('RGB'))
                                  for p in batch]).to(DEVICE)
            zc = net.content(xc_pre)
            feats.append(net.zc_pool(zc).cpu())
    return torch.cat(feats, 0)


def zk_of_v4(net, paths, k, bs=32):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch = paths[i:i + bs]
            xa = torch.stack([transforms.ToTensor()(
                US2.degrade_res(Image.open(p).convert('RGB').resize((384, 384)), k)).to(DEVICE)
                for p in batch])
            out.append(net.cond(xa).cpu())
    return torch.cat(out, 0)


def dino_raw_of(paths, bs=32):
    dino, pre = U.get_dinov2()
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch = paths[i:i + bs]
            x = torch.stack([pre(Image.open(p).convert('RGB')) for p in batch]).to(DEVICE)
            h = dino(x).hidden_states[11][:, 1:, :].mean(1).cpu()   # (B,768)
            feats.append(h)
    return torch.cat(feats, 0)


def clip_raw_of(paths, bs=32):
    clip_m, clip_pre = U.get_clip()
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch = paths[i:i + bs]
            x = torch.stack([clip_pre(Image.open(p).convert('RGB')) for p in batch]).to(DEVICE)
            f = clip_m.encode_image(x).float().cpu()
            feats.append(f)
    return torch.cat(feats, 0)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f'加载 v4 -> {NET_PATH}', flush=True)
    net = V4.V4Net().to(DEVICE)
    net.load_state_dict(torch.load(NET_PATH, map_location=DEVICE))
    net.eval()

    pairs = V3.build_labeled()
    paths = [p for p, _ in pairs]
    y_all = torch.tensor([c for _, c in pairs], dtype=torch.long)
    print(f'带标签 {len(pairs)} 张, {N_CLASS} 类', flush=True)

    res = {}

    # ── 基线: 冻结 DINOv2 / CLIP 原始特征直接 probe ──
    print('\n[baseline] DINOv2 / CLIP raw -> y ...', flush=True)
    dino_raw = dino_raw_of(paths)
    clip_raw = clip_raw_of(paths)
    dino_acc, _, _ = linear_probe(dino_raw, y_all, N_CLASS)
    clip_acc, _, _ = linear_probe(clip_raw, y_all, N_CLASS)
    print(f'  DINOv2 raw probe={dino_acc:.4f}  CLIP raw probe={clip_acc:.4f}', flush=True)
    res['baseline'] = {'dino_raw_probe': round(dino_acc, 4),
                       'clip_raw_probe': round(clip_acc, 4)}

    # ── A. v4 z_c (DINOv2 主干投影) 内容 ──
    print('\n[A] v4 z_c 内容 (DINOv2 主干)...', flush=True)
    zc = zc_pool_of(net, paths)
    zc_n = zc / zc.norm(dim=1, keepdim=True)
    A_probe, _, _ = linear_probe(zc, y_all, N_CLASS)
    with torch.no_grad():
        gc_log = net.gc(zc.to(DEVICE)).cpu()
    gc_acc = (gc_log.argmax(1) == y_all).float().mean().item()
    A_eff = eff_rank(zc.numpy())
    pc = (zc_n @ zc_n.T).clamp(-1, 1).numpy()
    A_pair = float(np.mean(pc[np.triu_indices(len(pc), 1)]))
    res['A_zc_content'] = {
        'zc_probe_acc': round(A_probe, 4),
        'gc_head_acc': round(gc_acc, 4),
        'eff_rank_zc': round(A_eff, 3),
        'pairwise_cos_zc': round(A_pair, 4),
        'v3_ref_zc_probe': V3_REF_ZC_PROBE,
    }
    A_pass = A_probe > 0.45   # 冻结主干应远高于 v3 浅卷积 0.238
    res['A_pass'] = bool(A_pass)
    print(f'  zc探针={A_probe:.4f} (v3浅卷积={V3_REF_ZC_PROBE}) gc头={gc_acc:.4f} '
          f'有效秩={A_eff:.3f} 成对cos={A_pair:.4f} => A_PASS={A_pass}', flush=True)
    del zc, zc_n, gc_log, dino_raw, clip_raw
    # 跨条件一致: 同图 xc(清晰) vs xa(降级k=3) 的 zc cos (冻结 DINOv2 应有高不变性)
    with torch.no_grad():
        xc_pre = torch.stack([net.content.pre(Image.open(p).convert('RGB'))
                              for p in paths]).to(DEVICE)
        zc_c = net.zc_pool(net.content(xc_pre))
        xa_pre = torch.stack([net.content.pre(
            US2.degrade_res(Image.open(p).convert('RGB').resize((384, 384)), 3))
            for p in paths]).to(DEVICE)
        zc_a = net.zc_pool(net.content(xa_pre))
        zc_c_n = zc_c / zc_c.norm(dim=1, keepdim=True)
        zc_a_n = zc_a / zc_a.norm(dim=1, keepdim=True)
        cross_consist = float((zc_c_n * zc_a_n).sum(1).mean().item())
    res['A_zc_content']['zc_cross_cond_cos'] = round(cross_consist, 4)
    print(f'  zc跨条件一致cos={cross_consist:.4f}', flush=True)
    del xc_pre, xa_pre, zc_c, zc_a

    # ── B. z_k 内容泄漏 ──
    print('\n[B] z_k 内容泄漏...', flush=True)
    zk_a = zk_of_v4(net, paths, k=3)
    B_probe, _, _ = linear_probe(zk_a, y_all, N_CLASS)
    res['B_zk_leak'] = {'zk_probe_acc': round(B_probe, 4)}
    B_pass = B_probe < 0.15
    res['B_pass'] = bool(B_pass)
    print(f'  zk探针acc={B_probe:.4f} => B_PASS={B_pass}', flush=True)
    del zk_a

    # ── 联合 (z_c, z_k) -> y ──
    print('\n[joint] (z_c, z_k) -> y ...', flush=True)
    zk3 = zk_of_v4(net, paths, k=3)
    zc_again = zc_pool_of(net, paths)
    joint = torch.cat([zc_again, zk3], dim=1)   # (N,65)
    J_probe, _, _ = linear_probe(joint, y_all, N_CLASS)
    res['joint'] = {'zc_probe': round(A_probe, 4), 'joint65_probe': round(J_probe, 4),
                    'delta_joint_minus_zc': round(J_probe - A_probe, 4)}
    Joint_pass = J_probe >= A_probe - 0.02     # 条件侧信息不应拖垮内容
    res['Joint_pass'] = bool(Joint_pass)
    print(f'  zc单独={A_probe:.4f} 联合={J_probe:.4f} Δ={J_probe-A_probe:+.4f} => Joint_pass={Joint_pass}', flush=True)
    del zk3, zc_again, joint

    # ── 总验收 ──
    overall = A_pass and B_pass and Joint_pass
    res['overall_accept'] = bool(overall)
    res['gate'] = {'A_zc_content': A_pass, 'B_zk_leak': B_pass, 'joint_zc_zk': Joint_pass}
    with open(SAVE_JSON, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f'\n=== v4 主线验收 overall_accept = {overall} ===', flush=True)
    print(f'  A zc内容={A_pass}  B zk泄漏弱={B_pass}  联合≥zc={Joint_pass}', flush=True)
    print(f'→ {SAVE_JSON}')


if __name__ == '__main__':
    main()
