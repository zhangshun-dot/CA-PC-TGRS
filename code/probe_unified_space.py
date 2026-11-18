# -*- coding: utf-8 -*-
"""probe_unified_space.py — 信息统一框架 v0 验证

验证 (对应长论述"条件分解而非消除"):
  ① 跨条件一致性: 同图不同合成条件 c 下, 内容表征 zc 余弦 vs CLIP 整图余弦(baseline)
  ② 条件预测 MSE: zk 回归真实 c 的误差
  ③ 内容保真检测: A(raw) vs D(zc 重建图) AP50

若 ① zc 一致性 > CLIP baseline, 则证明框架把观测条件从内容里分解出来了
(这正是"统一信息空间"相对"强行统一图像"的核心能力)。
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
import clip as clip_module
import unified_eval as U
import unified_space as US

DEVICE = U.DEVICE
PRE_FILTER = U.PRE_FILTER
N_PER_CLASS = 20
SAVE = U.SAVE_DIR / 'probe_unified_space.json'
NET_PATH = U.SAVE_DIR / 'unified_space.pt'
CLASSES = [('airplane', 'airplane'), ('ship', 'boat'), ('vehicle', 'car')]


def filter_one(dets, g):
    boxes, scores = dets
    keep = np.asarray(scores) > PRE_FILTER
    if keep.any():
        return (np.asarray(boxes)[keep], np.asarray(scores)[keep])
    return (np.zeros((0, 4)), np.zeros(0))


def per_img_ap(dets, gts):
    ip, ig = [], []
    for d, g in zip(dets, gts):
        b, s = filter_one(d, g)
        ip.append((b, s)); ig.append(g)
    return U.compute_ap50_img(ip, ig)


def clip_cos(im1, im2, clip_m, clip_pre):
    with torch.no_grad():
        f1 = clip_m.encode_image(clip_pre(im1).unsqueeze(0).to(DEVICE)).float()
        f2 = clip_m.encode_image(clip_pre(im2).unsqueeze(0).to(DEVICE)).float()
    f1 = f1 / f1.norm(dim=-1, keepdim=True)
    f2 = f2 / f2.norm(dim=-1, keepdim=True)
    return float((f1 @ f2.T).item())


def main():
    print('=' * 72)
    print('信息统一框架 v0 — 条件分解验证 (content/condition 解耦)')
    print('=' * 72)

    clip_m, clip_pre = U.get_clip()
    clip_m.eval()
    net = US.CondDisentangler().to(DEVICE)
    if os.path.exists(NET_PATH):
        net.load_state_dict(torch.load(NET_PATH, map_location=DEVICE))
        print(f'[net] 加载 -> {NET_PATH}')
    else:
        print('[net] 未找到权重, 自监督训练...')
        US.train_net(net, epochs=30, bs=4, save_path=NET_PATH)

    # ── ① 跨条件一致性 ──
    paths = glob.glob(os.path.join(U.DIOR_IMG_DIR, '*.jpg'))
    random.seed(2); random.shuffle(paths)
    test_paths = paths[:60]
    ours_cos, clip_cos_l = [], []
    for p in test_paths:
        im = Image.open(p).convert('RGB').resize((384, 384))
        c1, c2 = US.sample_cond(), US.sample_cond()
        a1, a2 = US.apply_cond(im, c1), US.apply_cond(im, c2)
        zc = US.pooled_zc(net, [a1, a2])            # (2, content_ch) 已归一化
        ours_cos.append(float((zc[0] @ zc[1]).item()))
        clip_cos_l.append(clip_cos(a1, a2, clip_m, clip_pre))
    ours_mean = float(np.mean(ours_cos))
    clip_mean = float(np.mean(clip_cos_l))
    print(f'\n[① 跨条件一致性] 同图不同 c 下的表征余弦')
    print(f'    我们的 zc : {ours_mean:.4f}')
    print(f'    CLIP baseline: {clip_mean:.4f}')
    print(f'    Δ(我们-CLIP)= {ours_mean-clip_mean:+.4f}  '
          f'{"zc 更条件不变 ✓" if ours_mean>clip_mean else "未占优 ✗"}')

    # ── ② 条件预测 MSE ──
    mse_list = []
    for _ in range(200):
        im = Image.open(random.choice(paths)).convert('RGB').resize((384, 384))
        cv = US.sample_cond()
        xa = US.apply_cond(im, cv)
        x = transforms.ToTensor()(xa).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, zk = net.encode(x)
        mse_list.append(float(F.mse_loss(zk.cpu(), torch.tensor(cv).unsqueeze(0)).item()))
    cond_mse = float(np.mean(mse_list))
    print(f'\n[② 条件预测] zk 回归 c 的 MSE = {cond_mse:.4f} (c∈[0,1]^3, 越低=条件越显式)')

    # ── ③ 内容保真检测 A vs D ──
    zk_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).to(DEVICE)
    rows = []
    for dior_cls, coco_name in CLASSES:
        print(f'\n--- {dior_cls} (COCO: {coco_name}) ---')
        images, gts = U.load_dior(dior_cls, n=N_PER_CLASS)
        if len(images) < 5:
            print('  图太少，跳过'); continue
        prompt = f'a photo of {coco_name}'
        dets_a = U.gdino_detect(images, prompt, 0.0)
        ap_a = per_img_ap(dets_a, gts)
        print(f'  AP50 raw(A)={ap_a:.4f}')
        d_imgs = US.harmonize_d(images, net, zk_mean)
        dets_d = U.gdino_detect(d_imgs, prompt, 0.0)
        ap_d = per_img_ap(dets_d, gts)
        print(f'  AP50 disentangle(D)={ap_d:.4f}  Δ={ap_d-ap_a:+.4f}')
        rows.append({'dior_cls': dior_cls, 'coco_name': coco_name, 'n_imgs': len(images),
                     'ap_raw': round(float(ap_a), 4), 'ap_disentangle': round(float(ap_d), 4),
                     'ap_delta_d_vs_a': round(float(ap_d - ap_a), 4)})
    ma = float(np.mean([r['ap_raw'] for r in rows])) if rows else 0
    md = float(np.mean([r['ap_disentangle'] for r in rows])) if rows else 0
    print(f'\n{"="*72}')
    print(f'均值 AP50  raw(A)={ma:.4f}  disentangle(D)={md:.4f}  Δ={md-ma:+.4f}')
    out = {'consistency': {'ours_zc': round(ours_mean, 4), 'clip_baseline': round(clip_mean, 4),
                           'delta': round(ours_mean - clip_mean, 4),
                           'zc_more_invariant': ours_mean > clip_mean},
           'cond_mse': round(cond_mse, 4),
           'detection': {'rows': rows, 'ap_raw_mean': round(ma, 4),
                         'ap_disentangle_mean': round(md, 4), 'ap_delta': round(md - ma, 4)},
           'note': '信息统一框架 v0: 条件解耦(content/condition)。①跨条件一致性高=分解成功; ②条件MSE低=显式; ③D用zc重建(条件取均值)喂检测, 保真不消除.'}
    with open(SAVE, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'结果 -> {SAVE}')


if __name__ == '__main__':
    main()
