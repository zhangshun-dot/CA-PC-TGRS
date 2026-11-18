# -*- coding: utf-8 -*-
"""probe_c_learned.py — (c) 学习型文本条件映射: A/B/C 全量对比 + bootstrap 显著性

A 路: 原始 DIOR 图 -> GDINO-tiny -> 逐图 AP50
B 路: Reinhard 色彩迁移(COCO 自然分布) -> 同检测器
C 路: TextConditionedEnhancer(文本锚定统一, 学习型) -> 同检测器

评估: 三类 DIOR->COCO (airplane/boat/car), 计算 AP50 + CLIP 对齐,
      并对 C vs A 做 bootstrap 显著性 (1000 重采样, 95% CI)。

若 NET 已存在则跳过训练直接评估。
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import json
import numpy as np
import torch
from PIL import Image, ImageEnhance
import clip as clip_module
import unified_eval as U
import text_unifier as TU

DEVICE = U.DEVICE
PRE_FILTER = U.PRE_FILTER
N_PER_CLASS = 20
SAVE = U.SAVE_DIR / 'probe_c_learned.json'
NET_PATH = U.SAVE_DIR / 'text_unifier.pt'
CLASSES = [('airplane', 'airplane'), ('ship', 'boat'), ('vehicle', 'car')]


# ── B 路工具(与之前一致) ──
def coco_natural_stats(n=80):
    from pycocotools.coco import COCO
    coco = COCO(U.COCO_ANN_FILE)
    iids = coco.getImgIds()[:600]
    ms, ss = [], []
    for iid in iids[:n]:
        info = coco.loadImgs([iid])[0]
        p = os.path.join(U.COCO_IMG_DIR, info['file_name'])
        try:
            im = np.asarray(Image.open(p).convert('RGB')).reshape(-1, 3) / 255.0
        except Exception:
            continue
        ms.append(im.mean(0)); ss.append(im.std(0))
    return np.mean(ms, 0), np.mean(ss, 0)


def reinhard(img, tgt_mean, tgt_std):
    im = np.asarray(img).astype(np.float32) / 255.0
    src_mean = im.reshape(-1, 3).mean(0)
    src_std = im.reshape(-1, 3).std(0)
    out = (im - src_mean) / (src_std + 1e-6) * tgt_std + tgt_mean
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


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


def clip_align(images, cls, clip_m, clip_pre):
    tok = clip_module.tokenize([f'a photo of {cls}']).to(DEVICE)
    with torch.no_grad():
        tf = clip_m.encode_text(tok).float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
    sims = []
    for im in images:
        with torch.no_grad():
            f = clip_m.encode_image(clip_pre(im).unsqueeze(0).to(DEVICE)).float()
            f = f / f.norm(dim=-1, keepdim=True)
        sims.append(float((f @ tf.T).item()))
    return float(np.mean(sims))


def bootstrap_delta(images, gts, prompt, anchor, net, n=1000):
    """C vs A 的 AP50 差异 bootstrap 95% CI (重采样图像, 保持同图 GT 一致)。"""
    dets_a = U.gdino_detect(images, prompt, 0.0)
    dets_c = U.gdino_detect(TU.harmonize(images, net, anchor), prompt, 0.0)
    pa = [filter_one(d, g) for d, g in zip(dets_a, gts)]
    pc = [filter_one(d, g) for d, g in zip(dets_c, gts)]
    m = len(images)
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        ap_a = U.compute_ap50_img([pa[i] for i in idx], [gts[i] for i in idx])
        ap_c = U.compute_ap50_img([pc[i] for i in idx], [gts[i] for i in idx])
        deltas.append(ap_c - ap_a)
    deltas = np.array(deltas)
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main():
    print('=' * 72)
    print('(c) 学习型文本条件映射 — A/B/C 全量对比 + bootstrap 显著性')
    print(f'检测器=GDINO-tiny  协议=pre-filter {PRE_FILTER}  逐图 AP50  每类 {N_PER_CLASS} 图')
    print('=' * 72)

    clip_m, clip_pre = U.get_clip()
    clip_m.eval()
    anchor = TU.build_anchor(clip_m)

    # 加载/训练网络
    net = TU.TextConditionedEnhancer().to(DEVICE)
    if os.path.exists(NET_PATH):
        net.load_state_dict(torch.load(NET_PATH, map_location=DEVICE))
        print(f'[net] 加载已训练 -> {NET_PATH}')
    else:
        print('[net] 未找到权重, 开始自监督训练...')
        TU.train_net(net, anchor, epochs=40, bs=4, save_path=NET_PATH)

    tgt_mean, tgt_std = coco_natural_stats()
    rows = []
    for dior_cls, coco_name in CLASSES:
        print(f'\n--- {dior_cls} (COCO: {coco_name}) ---')
        images, gts = U.load_dior(dior_cls, n=N_PER_CLASS)
        if len(images) < 5:
            print('  图太少，跳过'); continue
        prompt = f'a photo of {coco_name}'

        dets_a = U.gdino_detect(images, prompt, 0.0)
        ap_a = per_img_ap(dets_a, gts)
        sim_a = clip_align(images, coco_name, clip_m, clip_pre)
        print(f'  AP50 raw(A)={ap_a:.4f}  CLIP(A)={sim_a:.4f}')

        mig = [reinhard(im, tgt_mean, tgt_std) for im in images]
        dets_b = U.gdino_detect(mig, prompt, 0.0)
        ap_b = per_img_ap(dets_b, gts)
        sim_b = clip_align(mig, coco_name, clip_m, clip_pre)
        print(f'  AP50 reinhard(B)={ap_b:.4f} (Δ{ap_b-ap_a:+.4f})  CLIP(B)={sim_b:.4f}')

        dets_c = U.gdino_detect(TU.harmonize(images, net, anchor), prompt, 0.0)
        ap_c = per_img_ap(dets_c, gts)
        sim_c = clip_align(TU.harmonize(images, net, anchor), coco_name, clip_m, clip_pre)
        print(f'  AP50 learned(C)={ap_c:.4f} (ΔA{ap_c-ap_a:+.4f} ΔB{ap_c-ap_b:+.4f})  CLIP(C)={sim_c:.4f}')

        mean_d, lo, hi = bootstrap_delta(images, gts, prompt, anchor, net, n=1000)
        print(f'  C vs A bootstrap ΔAP50={mean_d:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  '
              f'{"显著 beat" if lo>0 else ("显著差" if hi<0 else "不显著")}')

        rows.append({
            'dior_cls': dior_cls, 'coco_name': coco_name, 'n_imgs': len(images),
            'ap_raw': round(float(ap_a), 4), 'ap_reinhard': round(float(ap_b), 4),
            'ap_learned': round(float(ap_c), 4),
            'ap_delta_c_vs_a': round(float(ap_c - ap_a), 4),
            'ap_delta_c_vs_b': round(float(ap_c - ap_b), 4),
            'align_raw': round(sim_a, 4), 'align_reinhard': round(sim_b, 4),
            'align_learned': round(sim_c, 4), 'align_delta_c_vs_a': round(sim_c - sim_a, 4),
            'boot_delta_mean': round(mean_d, 4), 'boot_ci_lo': round(lo, 4), 'boot_ci_hi': round(hi, 4),
            'boot_sig': ('beat' if lo > 0 else ('worse' if hi < 0 else 'ns')),
        })

    if rows:
        def mean(k):
            return float(np.mean([r[k] for r in rows]))
        ma, mb, mc = mean('ap_raw'), mean('ap_reinhard'), mean('ap_learned')
        sa, sb, sc = mean('align_raw'), mean('align_reinhard'), mean('align_learned')
        print(f'\n{"="*72}')
        print(f'均值 AP50   raw(A)={ma:.4f}  reinhard(B)={mb:.4f}  learned(C)={mc:.4f}')
        print(f'         ΔC-A={mc-ma:+.4f}   ΔC-B={mc-mb:+.4f}')
        print(f'均值 CLIP   raw(A)={sa:.4f}  reinhard(B)={sb:.4f}  learned(C)={sc:.4f}  ΔC-A={sc-sa:+.4f}')
        sig = [r['boot_sig'] for r in rows]
        print(f'C vs A 显著性: {sig}')
        out = {'rows': rows, 'mean': {
            'ap_raw': round(ma, 4), 'ap_reinhard': round(mb, 4), 'ap_learned': round(mc, 4),
            'ap_delta_c_vs_a': round(mc - ma, 4), 'ap_delta_c_vs_b': round(mc - mb, 4),
            'align_raw': round(sa, 4), 'align_reinhard': round(sb, 4), 'align_learned': round(sc, 4),
            'align_delta_c_vs_a': round(sc - sa, 4),
        }, 'note': 'C=学习型文本条件映射(TextConditionedEnhancer, FiLM文本条件+残差, 自监督). A=raw, B=Reinhard. bootstrap 1000 重采样测 C vs A 显著性.'}
        with open(SAVE, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'结果 -> {SAVE}')


if __name__ == '__main__':
    main()
