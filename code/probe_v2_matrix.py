# -*- coding: utf-8 -*-
"""probe_v2_matrix.py — v2 诊断矩阵 (元反馈第 10 节: A/B/C/D + 帕累托 + 条件依赖)

A 仅内容: zc -> 类 probe (对照 CLIP/DINOv2), 看内容是否保留
B 仅条件: zk -> c (分辨率 k) 回归 R^2, 看条件是否可识别(应 > 基线天花板 0.27)
C 双分支: 同模型两分支并存
D 弱不变: λ_inv warm-up 已内建, 监控 zc probe 不掉
帕累托: zc probe acc (content retention) vs zc 跨条件一致 (invariance); zk R^2; eff rank
条件依赖(元第5节): 固定 zc 改 zk -> 重建清晰度变; 屏蔽 zk -> 重建质量降

结论保守, 不称"验证成功"; 仅当 A 成立(zc 不塌) 且 B 成立(zk 可识别) 才进反事实交换。
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
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import unified_eval as U
import unified_space_v2 as US2

DEVICE = U.DEVICE
SAVE = U.SAVE_DIR / 'probe_v2_matrix.json'
NET_PATH = U.SAVE_DIR / 'unified_space_v2c.pt'
N_CLASS_PROBE = 10
N_PER_CLS = 15
SEED = 5


def linear_probe(X, y, n_class, epochs=400, lr=1e-2):
    X = X.float()
    W = torch.zeros(n_class, X.shape[1], requires_grad=True)
    b = torch.zeros(n_class, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=1e-3)
    yt = y.long()
    for _ in range(epochs):
        logits = X @ W.T + b
        loss = F.cross_entropy(logits, yt)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (logits.argmax(1) == yt).float().mean().item()
    return acc


def eff_rank(mat):
    s = np.linalg.svd(mat, compute_uv=False)
    s = s / (s.sum() + 1e-12)
    s = s[s > 1e-12]
    return float(np.exp(-(s * np.log(s)).sum()))


def dinov2_full(img, dino, pre):
    t = pre(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = dino(t).hidden_states[U.VEP_LAYER + 1][0, 0, :]
    return (f / f.norm(dim=-1, keepdim=True)).squeeze(0).cpu()


def clip_full(img, m, pre):
    with torch.no_grad():
        f = m.encode_image(pre(img).unsqueeze(0).to(DEVICE)).float()
    return (f / f.norm(dim=-1, keepdim=True)).squeeze(0).cpu()


def main():
    print('=' * 74)
    print('v2 诊断矩阵 — A(zc内容)/B(zk条件)/C(双分支)/D(弱不变)+帕累托')
    print('=' * 74)
    random.seed(SEED)
    clip_m, clip_pre = U.get_clip(); clip_m.eval()
    dino, dino_pre = U.get_dinov2(); dino.eval()
    net = US2.CondDisentanglerV2().to(DEVICE)
    if os.path.exists(NET_PATH):
        net.load_state_dict(torch.load(NET_PATH, map_location=DEVICE))
        print(f'[net] 加载 -> {NET_PATH}')
    else:
        print('[net] 未找到, 自监督训练 v2...')
        US2.train_net(net, epochs=40, bs=16, save_path=NET_PATH)
    net.eval()

    # ══ A: zc 内容 probe ══
    print('\n[ A ] zc 内容保留 — 线性探针 zc->类')
    probe_classes = list(U.DIOR_ID2NAME.values())[:N_CLASS_PROBE]
    zcs, clips, dinos, yl = [], [], [], []
    for ci, cls in enumerate(probe_classes):
        imgs, _ = U.load_dior(cls, n=N_PER_CLS)
        if len(imgs) < 5:
            continue
        for im in imgs:
            im = im.convert('RGB').resize((384, 384))
            zcs.append(US2.pooled_zc(net, [im]).squeeze(0))
            clips.append(clip_full(im, clip_m, clip_pre))
            dinos.append(dinov2_full(im, dino, dino_pre))
            yl.append(ci)
    zcs = torch.stack(zcs); clips = torch.stack(clips); dinos = torch.stack(dinos)
    yl = torch.tensor(yl); n_cls = int(yl.max()) + 1
    acc_zc = linear_probe(zcs, yl, n_cls)
    acc_clip = linear_probe(clips, yl, n_cls)
    acc_dino = linear_probe(dinos, yl, n_cls)
    Z = zcs.numpy()
    iu = np.triu_indices(len(Z), 1)
    pc = (Z @ Z.T)[iu]
    rZ = eff_rank(Z)
    print(f'  样本 {len(zcs)} / {n_cls} 类, 随机基线 {1/n_cls:.3f}')
    print(f'  zc probe={acc_zc:.4f}  CLIP={acc_clip:.4f}  DINOv2={acc_dino:.4f}')
    print(f'  zc/CLIP={acc_zc/max(acc_clip,1e-9):.3f}  '
          f'{"内容保留 ✓" if acc_zc>1.5/n_cls else "塌缩 ✗"}')
    print(f'  zc 两两余弦 mean={pc.mean():.4f} std={pc.std():.4f}  有效秩={rZ:.2f}/{Z.shape[1]}')

    # ══ B: zk -> c 回归 (分辨率) ══
    print('\n[ B ] zk 条件可识别 — zk->分辨率c 回归 R^2 (基线天花板 |R|≈0.27)')
    paths = glob.glob(os.path.join(U.DIOR_IMG_DIR, '*.jpg'))
    random.shuffle(paths)
    zk_l, c_l = [], []
    for p in paths[:120]:
        im = Image.open(p).convert('RGB').resize((384, 384))
        k, c = US2.sample_res()
        xa = transforms.ToTensor()(US2.degrade_res(im, k)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, zk = net.encode(xa)
        zk_l.append(zk.squeeze(0).cpu()); c_l.append(torch.tensor(c))
    Zk = torch.stack(zk_l).numpy(); Cc = torch.stack(c_l).numpy()
    r = np.corrcoef(Zk[:, 0], Cc[:, 0])[0, 1]
    ss = ((Zk[:, 0] - Cc[:, 0]) ** 2).sum(); st = ((Cc[:, 0] - Cc[:, 0].mean()) ** 2).sum()
    r2 = 1 - ss / (st + 1e-12)
    print(f'  zk->c  Pearson R={r:+.3f}  R2={r2:+.3f}  '
          f'{"条件可识别 ✓" if abs(r)>0.5 else "仍不可识别 ✗"}')

    # ══ C/D: 跨条件一致 + 条件依赖测试(元第5节, 正确隔离) ══
    print('\n[ C/D ] 跨条件一致 + 条件依赖(正确隔离: 内容用xc, 条件用xa)')
    zc_cos_l, zk_diff_l, dep_diff_l = [], [], []
    for p in paths[120:200]:
        im = Image.open(p).convert('RGB').resize((384, 384))
        xc_t = transforms.ToTensor()(im).unsqueeze(0).to(DEVICE)
        k1, _ = US2.sample_res(); k4, _ = US2.sample_res()
        if k1 == 1: k1 = 2
        if k4 == 1: k4 = 4
        xa1 = transforms.ToTensor()(US2.degrade_res(im, k1)).unsqueeze(0).to(DEVICE)
        xa4 = transforms.ToTensor()(US2.degrade_res(im, k4)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            zc_c, _ = net.encode(xc_t)             # 清晰内容(无降级)
            zc_a1, zk_k1 = net.encode(xa1)         # 降级k1
            _, zk_k4 = net.encode(xa4)             # 降级k4
            # 条件依赖: 固定清晰内容 zc_c, 换 zk -> 重建模糊度应随条件变
            rec_k1 = net.decode(zc_c, zk_k1)
            rec_k4 = net.decode(zc_c, zk_k4)
        # zc 跨条件一致: zc(xa1降级) vs zc(xc清晰) 应高
        zc_cos_l.append(float(F.cosine_similarity(zc_a1.flatten(1), zc_c.flatten(1)).item()))
        zk_diff_l.append(float((zk_k1 - zk_k4).abs().item()))   # zk 随条件变
        dep_diff_l.append(float(F.l1_loss(rec_k1, rec_k4).item()))  # 重建随条件变
    zc_cons = float(np.mean(zc_cos_l))
    zk_diff = float(np.mean(zk_diff_l))
    dep = float(np.mean(dep_diff_l))
    print(f'  跨条件一致: zc(xa) vs zc(xc)={zc_cons:.4f}(应高)')
    print(f'  zk 随条件变: 同图不同k的 |zk|差={zk_diff:.4f}  '
          f'{"zk真随条件变 ✓" if zk_diff>0.01 else "zk不随条件变 ✗"}')
    print(f'  条件依赖(正确隔离): 固定zc(xc)改zk 重建L1差={dep:.4f}  '
          f'{"重建真依赖zk ✓" if dep>0.02 else "重建忽略zk ✗"}')

    # ══ 帕累托汇总 ══
    print('\n[ 帕累托 ]')
    print(f'  content retention(zc probe)={acc_zc:.4f}  vs  invariance(zc跨条件一致)={zc_cons:.4f}')
    print(f'  condition predictability(zk R2)={r2:+.4f}  eff_rank(zc)={rZ:.2f}')
    A_ok = acc_zc > 1.5 / n_cls
    B_ok = abs(r) > 0.5
    print(f'  A成立(zc不塌)={A_ok}  B成立(zk可识别)={B_ok}  '
          f'{"=> 可进反事实交换" if A_ok and B_ok else "=> 暂不做反事实/下游, 继续修"}')

    out = {
        'A_zc_content': {'acc_zc': round(acc_zc, 4), 'acc_clip': round(acc_clip, 4),
                         'acc_dino': round(acc_dino, 4), 'random': round(1 / n_cls, 4),
                         'eff_rank': round(rZ, 2), 'pairwise_cos_mean': round(float(pc.mean()), 4),
                         'pairwise_cos_std': round(float(pc.std()), 4), 'A_ok': bool(A_ok)},
        'B_zk_cond': {'pearson_r': round(float(r), 3), 'r2': round(float(r2), 3),
                      'baseline_ceiling_r': 0.27, 'B_ok': bool(B_ok)},
        'C_D': {'zc_consistency': round(zc_cons, 4), 'zk_diff_cond': round(zk_diff, 4),
                'zk_varies_with_cond': bool(zk_diff > 0.01),
                'cond_depend_L1': round(dep, 4), 'recon_depends_on_zk': bool(dep > 0.02)},
        'verdict': '可进反事实/下游' if (A_ok and B_ok) else '继续修表示',
        'note': 'v2 矩阵: 条件改分辨率(可识别); 共享浅层并行 + VICReg + 弱不变 warm-up + 重建依赖zk.'}
    with open(SAVE, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n结果 -> {SAVE}')


if __name__ == '__main__':
    main()
