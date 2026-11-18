# -*- coding: utf-8 -*-
"""probe_zc_diag.py — 信息统一框架 v0 的"诚实诊断"(对应元/导反馈: 不得称"验证成功")

三个必查项:
  ① zc 防塌缩 (Content Accuracy):
       - 线性探针 zc -> 类, 准确率对照 CLIP 整图 / DINOv2 L11 [CLS] 基线
       - 若 zc 探针 ≈ 基线, 则内容保留(未塌缩); 若 ≈ 随机, 则塌缩(致命)
       - 塌缩几何检查: zc 矩阵 SVD 有效秩 + 两两余弦均值/标准差
  ② zk 条件显式性:
       - 跨条件一致性: 同图不同 c 下 zc 余弦(应高) vs zk 余弦(应低) = 真解耦
       - zk -> c 线性回归: 每维 Pearson R + 整体 MSE(对照 probe_unified_space 的 0.0875)
  ③ 反事实条件交换 (counterfactual condition swap):
       - 取 (zc(x1), zk(x2)) 解码 -> 重编码 (zc_s, zk_s)
       - 校验 cos(zc_s, zc1)≈1(内容保真) 且 cos(zk_s, zk2)≈1(条件保真)
       - 证明解耦是"真"的, 而非编进同一向量
       - 保存若干交换样例 PNG 供人眼复核

结论一律保守, 不写"验证成功/框架通过"。
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
import unified_space as US

DEVICE = U.DEVICE
VER = sys.argv[1] if len(sys.argv) > 1 else 'v0'
if VER == 'v1':
    import unified_space_v1 as USM
    NET_CLS = USM.CondDisentanglerV1
    NET_PATH = U.SAVE_DIR / 'unified_space_v1.pt'
    SAVE = U.SAVE_DIR / 'probe_zc_diag_v1.json'
    SWAP_DIR = U.SAVE_DIR / 'zc_swap_samples_v1'
else:
    NET_CLS = US.CondDisentangler
    NET_PATH = U.SAVE_DIR / 'unified_space.pt'
    SAVE = U.SAVE_DIR / 'probe_zc_diag.json'
    SWAP_DIR = U.SAVE_DIR / 'zc_swap_samples'
N_CLASS_PROBE = 10      # 线性探针用多少 DIOR 类
N_PER_CLS = 15          # 每类取多少图
SEED = 3


# ───────── 线性探针 (自包含, 不依赖 sklearn) ─────────
def linear_probe(X, y, n_class, epochs=400, lr=1e-2, wd=1e-3):
    """X:(N,D) 已 L2 归一, y:(N,) long。返回 (acc, W,b)。"""
    X = X.float()
    W = torch.zeros(n_class, X.shape[1], requires_grad=True)
    b = torch.zeros(n_class, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    yt = y.long()
    for _ in range(epochs):
        logits = X @ W.T + b
        loss = F.cross_entropy(logits, yt)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (logits.argmax(1) == yt).float().mean().item()
    return acc, (W.detach(), b.detach())


def eff_rank(mat):
    """矩阵有效秩 = exp(熵 of 归一奇异值)。"""
    s = np.linalg.svd(mat, compute_uv=False)
    s = s / (s.sum() + 1e-12)
    s = s[s > 1e-12]
    ent = -(s * np.log(s)).sum()
    return float(np.exp(ent))


# ───────── 特征抽取 ─────────
def dinov2_full_img(img, dino, pre):
    t = pre(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        o = dino(t)
    f = o.hidden_states[U.VEP_LAYER + 1][0, 0, :]
    return f / f.norm(dim=-1, keepdim=True)


def clip_img_feat(img, clip_m, clip_pre):
    with torch.no_grad():
        f = clip_m.encode_image(clip_pre(img).unsqueeze(0).to(DEVICE)).float()
    return f / f.norm(dim=-1, keepdim=True)


def main():
    print('=' * 74)
    print('信息统一框架 v0 — 诚实诊断 (zc 防塌缩 / zk 显式 / 反事实交换)')
    print('=' * 74)
    random.seed(SEED)

    clip_m, clip_pre = U.get_clip()
    clip_m.eval()
    dino, dino_pre = U.get_dinov2()
    dino.eval()
    net = NET_CLS().to(DEVICE)
    if os.path.exists(NET_PATH):
        net.load_state_dict(torch.load(NET_PATH, map_location=DEVICE))
        print(f'[net] 加载 -> {NET_PATH}')
    else:
        print(f'[net] 未找到 {NET_PATH}, 自监督训练 v1...')
        USM.train_net(net, epochs=30, bs=8, save_path=NET_PATH) if VER == 'v1' else \
            US.train_net(net, epochs=30, bs=4, save_path=NET_PATH)
    net.eval()

    # ═══ ① zc 防塌缩: 跨类线性探针 ═══
    print('\n' + '─' * 74)
    print('① zc 防塌缩 — 线性探针 zc->类 (对照 CLIP / DINOv2)')
    print('─' * 74)
    probe_classes = list(U.DIOR_ID2NAME.values())[:N_CLASS_PROBE]
    zcs, clips, dinos, ylab = [], [], [], []
    per_cls_count = {}
    for ci, cls in enumerate(probe_classes):
        imgs, _ = U.load_dior(cls, n=N_PER_CLS)
        if len(imgs) < 5:
            continue
        for im in imgs:
            im = im.convert('RGB').resize((384, 384))
            zc = US.pooled_zc(net, [im]).squeeze(0)        # (64,) 已归一
            zcs.append(zc)
            clips.append(clip_img_feat(im, clip_m, clip_pre).squeeze(0).cpu())
            dinos.append(dinov2_full_img(im, dino, dino_pre).squeeze(0).cpu())
            ylab.append(ci)
            per_cls_count[cls] = per_cls_count.get(cls, 0) + 1
    zcs = torch.stack(zcs); clips = torch.stack(clips); dinos = torch.stack(dinos)
    ylab = torch.tensor(ylab)
    n_cls = int(ylab.max().item()) + 1
    print(f'  样本: {len(zcs)} 图 / {n_cls} 类 (每类 {per_cls_count})')
    print(f'  基线(随机) ≈ {1.0/n_cls:.3f}')

    acc_zc = linear_probe(zcs, ylab, n_cls)[0]
    acc_clip = linear_probe(clips, ylab, n_cls)[0]
    acc_dino = linear_probe(dinos, ylab, n_cls)[0]
    print(f'  zc  探针 acc = {acc_zc:.4f}')
    print(f'  CLIP 探针 acc = {acc_clip:.4f}')
    print(f'  DINOv2探针 acc = {acc_dino:.4f}')
    print(f'  zc/CLIP 比 = {acc_zc/max(acc_clip,1e-9):.3f}  '
          f'{"内容保留(≠塌缩) ✓" if acc_zc > 1.5/n_cls else "zc 丢内容/塌缩 ✗"}')

    # 塌缩几何检查
    Z = zcs.cpu().numpy()
    cos = (Z @ Z.T)
    iu = np.triu_indices(len(Z), 1)
    pair_cos = cos[iu]
    rZ = eff_rank(Z)
    print(f'  zc 两两余弦: mean={pair_cos.mean():.4f} std={pair_cos.std():.4f} '
          f'(≈1&std≈0 即塌缩为点)')
    print(f'  zc 有效秩 = {rZ:.2f} / {Z.shape[1]} (秩越高=结构越丰富)')

    # ═══ ② zk 条件显式性 ═══
    print('\n' + '─' * 74)
    print('② zk 条件显式性 — 跨条件一致性 + zk->c 回归')
    print('─' * 74)
    paths = glob.glob(os.path.join(U.DIOR_IMG_DIR, '*.jpg'))
    random.shuffle(paths)
    zc_cos_l, zk_cos_l = [], []
    zk_feats, c_true = [], []
    for p in paths[:80]:
        im = Image.open(p).convert('RGB').resize((384, 384))
        c1, c2 = US.sample_cond(), US.sample_cond()
        a1, a2 = US.apply_cond(im, c1), US.apply_cond(im, c2)
        zc1, zk1 = net.encode(transforms.ToTensor()(a1).unsqueeze(0).to(DEVICE))
        zc2, zk2 = net.encode(transforms.ToTensor()(a2).unsqueeze(0).to(DEVICE))
        zc1n = F.normalize(zc1.flatten(1), dim=-1)
        zc2n = F.normalize(zc2.flatten(1), dim=-1)
        zk1n = F.normalize(zk1, dim=-1); zk2n = F.normalize(zk2, dim=-1)
        zc_cos_l.append(float((zc1n @ zc2n.T).item()))
        zk_cos_l.append(float((zk1n @ zk2n.T).item()))
        # 用 a1 的 zk 与其真实 c1 做回归
        zk_feats.append(zk1.squeeze(0).detach().cpu())
        c_true.append(torch.tensor(c1))
    zc_cons = float(np.mean(zc_cos_l))
    zk_cons = float(np.mean(zk_cos_l))
    print(f'  跨条件一致性  zc 余弦(应高)={zc_cons:.4f}   zk 余弦(应低)={zk_cons:.4f}')
    print(f'  Δ(zc-zk)={zc_cons-zk_cons:+.4f}  '
          f'{"真解耦(内容不变/条件变) ✓" if zc_cons > zk_cons + 0.05 else "解耦不足 ✗"}')

    # zk -> c 回归 (每维 Pearson R)
    Zk = torch.stack(zk_feats).numpy()        # (N,3)
    C = torch.stack(c_true).numpy()           # (N,3)
    cond_names = ['brightness', 'contrast', 'blur']
    c_mse = float(np.mean((Zk - C) ** 2))
    per_dim_r = []
    for d in range(3):
        if Zk[:, d].std() < 1e-6 or C[:, d].std() < 1e-6:
            per_dim_r.append(0.0)
        else:
            r = float(np.corrcoef(Zk[:, d], C[:, d])[0, 1])
            per_dim_r.append(r)
    print(f'  zk->c 回归 MSE={c_mse:.4f} (对照 v0 0.0875)')
    print(f'  每维 Pearson R: ' + ', '.join(
        f'{n}={per_dim_r[i]:+.3f}' for i, n in enumerate(cond_names)))

    # ═══ ③ 反事实条件交换 ═══
    print('\n' + '─' * 74)
    print('③ 反事实条件交换 (zc1, zk2) -> 重解码 -> 重编码校验')
    print('─' * 74)
    os.makedirs(SWAP_DIR, exist_ok=True)
    swap_consistency = []
    n_sample = min(8, len(paths) // 2)
    saved = 0
    for k in range(n_sample):
        im1 = Image.open(paths[2 * k]).convert('RGB').resize((384, 384))
        im2 = Image.open(paths[2 * k + 1]).convert('RGB').resize((384, 384))
        x1 = transforms.ToTensor()(im1).unsqueeze(0).to(DEVICE)
        x2 = transforms.ToTensor()(im2).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            zc1, zk1 = net.encode(x1)
            zc2, zk2 = net.encode(x2)
            x_swap = net.decode(zc1, zk2)          # 内容1 + 条件2
            x_rec = net.decode(zc1, zk1)           # 重建(同源) 控制
            zc_s, zk_s = net.encode(x_swap)
        # 校验
        zc1n = F.normalize(zc1.flatten(1), dim=-1)
        zk2n = F.normalize(zk2, dim=-1)
        zcs_n = F.normalize(zc_s.flatten(1), dim=-1)
        zks_n = F.normalize(zk_s, dim=-1)
        cs_zc = float((zc1n @ zcs_n.T).item())
        cs_zk = float((zk2n @ zks_n.T).item())
        swap_consistency.append((cs_zc, cs_zk))
        if saved < 6:
            x_swap_p = transforms.ToPILImage()(x_swap.squeeze(0).cpu().clamp(0, 1))
            x_rec_p = transforms.ToPILImage()(x_rec.squeeze(0).cpu().clamp(0, 1))
            # 拼 (x1 | x2 | swap | rec) 便于人眼看
            w, h = im1.size
            canvas = Image.new('RGB', (w * 4, h))
            canvas.paste(im1, (0, 0)); canvas.paste(im2, (w, 0))
            canvas.paste(x_swap_p.resize((w, h)), (2 * w, 0))
            canvas.paste(x_rec_p.resize((w, h)), (3 * w, 0))
            canvas.save(os.path.join(SWAP_DIR, f'swap_{k:02d}.png'))
            saved += 1
    cs_zc_m = float(np.mean([a for a, _ in swap_consistency]))
    cs_zk_m = float(np.mean([b for _, b in swap_consistency]))
    print(f'  交换后重编码: cos(zc_s, zc1)={cs_zc_m:.4f} (内容保真, 应≈1)')
    print(f'               cos(zk_s, zk2)={cs_zk_m:.4f} (条件保真, 应≈1)')
    print(f'  {"解耦真实(交换可重解) ✓" if cs_zc_m>0.9 and cs_zk_m>0.9 else "解耦不纯 ✗"}')
    print(f'  交换样例 -> {SWAP_DIR} (x1 | x2 | swap(zc1,zk2) | rec(zc1,zk1))')

    # ── 汇总 ──
    out = {
        'content_accuracy': {
            'n_samples': len(zcs), 'n_classes': n_cls,
            'acc_zc': round(acc_zc, 4), 'acc_clip': round(acc_clip, 4),
            'acc_dino': round(acc_dino, 4),
            'random_baseline': round(1.0 / n_cls, 4),
            'zc_retains_content': bool(acc_zc > 1.5 / n_cls),
            'zc_pairwise_cos_mean': round(float(pair_cos.mean()), 4),
            'zc_pairwise_cos_std': round(float(pair_cos.std()), 4),
            'zc_eff_rank': round(rZ, 2), 'zc_dim': Z.shape[1]},
        'cond_explicit': {
            'zc_consistency': round(zc_cons, 4), 'zk_consistency': round(zk_cons, 4),
            'delta_zc_minus_zk': round(zc_cons - zk_cons, 4),
            'true_disentangle': bool(zc_cons > zk_cons + 0.05),
            'zk_to_c_mse': round(c_mse, 4), 'v0_mse': 0.0875,
            'zk_to_c_per_dim_r': {cond_names[i]: round(per_dim_r[i], 3) for i in range(3)}},
        'counterfactual_swap': {
            'cos_zc_s_zc1': round(cs_zc_m, 4), 'cos_zk_s_zk2': round(cs_zk_m, 4),
            'true_disentangle': bool(cs_zc_m > 0.9 and cs_zk_m > 0.9),
            'n_samples': n_sample},
        'note': '诚实诊断 v0: 不称"验证成功"。zc 防塌缩 / zk 显式 / 反事实交换三关。'
    }
    with open(SAVE, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n{"="*74}')
    print(f'结果 -> {SAVE}')
    print('注: 本脚本只诊断"解耦是否真实/是否塌缩", 不证明下游增益。')


if __name__ == '__main__':
    main()
