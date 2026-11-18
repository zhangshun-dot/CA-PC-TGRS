# -*- coding: utf-8 -*-
"""probe_c_text_anchored.py — C 路原型: (c) 文本锚定语义统一

A 路: 原始 DIOR 图 -> GDINO-tiny("a photo of {cls}") -> 逐图 AP50
B 路: Reinhard 色彩迁移(拉向 COCO 自然分布, 零训练) -> 同检测器
C 路: 文本锚定语义统一 —— 逐图在 CLIP 空间朝「类文本嵌入」做温和光度搜索
      (gamma / brightness / contrast 小网格, 选使 图像·文本 余弦相似度最大的那张),
      不搬 COCO 像素统计, 不破坏检测器依赖的局部几何特征。

对齐代理: CLIP 图像-文本余弦相似度(整图 vs "a photo of {cls}"), A/B/C 对比。

成功判据(论文级): C 的 AP50 应胜过 B (理想 >= A), 且 Δ对齐 >= 0;
                  若 C 对齐升但 AP50 反降, 则坐实"图像层干预反噬检测"风险。

复用 unified_eval 的 load_dior / gdino_detect / compute_ap50_img / get_clip。
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

DEVICE = U.DEVICE
PRE_FILTER = U.PRE_FILTER
N_PER_CLASS = 20
SAVE = U.SAVE_DIR / 'probe_abc_text_anchored.json'

CLASSES = [('airplane', 'airplane'), ('ship', 'ship'), ('vehicle', 'vehicle')]


# ── B 路工具(从 probe_first_layer_ab 复制, 保持一致) ──
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
    out = np.clip(out, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))


def per_img_ap(dets, gts):
    im_preds, im_gts = [], []
    for (boxes, scores), g in zip(dets, gts):
        keep = np.asarray(scores) > PRE_FILTER
        if keep.any():
            im_preds.append((np.asarray(boxes)[keep], np.asarray(scores)[keep]))
            im_gts.append(g)
        else:
            im_preds.append((np.zeros((0, 4)), np.zeros(0)))
            im_gts.append(g)
    return U.compute_ap50_img(im_preds, im_gts)


# ── C 路: 文本锚定语义统一 ──
def photometric(img, gamma=1.0, bright=0.0, contrast=1.0):
    """温和光度变换: gamma -> brightness(add) -> contrast。保持局部结构, 不搬全局统计。"""
    im = np.asarray(img).astype(np.float32) / 255.0
    im = np.clip(im ** gamma, 0, 1)
    im = np.clip(im + bright, 0, 1)
    out = Image.fromarray((im * 255).astype(np.uint8))
    if contrast != 1.0:
        out = ImageEnhance.Contrast(out).enhance(contrast)
    return out


def text_anchored_unify(images, text_emb, clip_m, clip_pre):
    """逐图: 在 CLIP 空间朝 text_emb 方向选最优温和光度变换。

    返回 (harmonized_images, per_img_best_sim)。
    text_emb: (1,512) 归一化类文本嵌入。
    """
    grid = [(g, b, c) for g in (0.85, 1.0, 1.15)
            for b in (-0.04, 0.0, 0.04)
            for c in (0.9, 1.0, 1.1)]  # 27 组合, 含恒等
    out_imgs, out_sims = [], []
    for img in images:
        best_sim, best_img = -1e9, img
        # 预编码所有变换
        for g, b, c in grid:
            t = photometric(img, g, b, c)
            with torch.no_grad():
                f = clip_m.encode_image(clip_pre(t).unsqueeze(0).to(DEVICE)).float()
                f = f / f.norm(dim=-1, keepdim=True)
            sim = float((f @ text_emb.T).item())
            if sim > best_sim:
                best_sim, best_img = sim, t
        out_imgs.append(best_img)
        out_sims.append(best_sim)
    return out_imgs, out_sims


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


def main():
    print('=' * 72)
    print('A/B/C 探针: 第一层图像优化 (c)文本锚定语义统一 方向性验证 (小样本)')
    print(f'检测器=GDINO-tiny  协议=pre-filter {PRE_FILTER}  逐图 AP50  每类 {N_PER_CLASS} 图')
    print('=' * 72)

    clip_m, clip_pre = U.get_clip()
    clip_m.eval()
    tgt_mean, tgt_std = coco_natural_stats()
    print(f'COCO 自然分布目标均值={tgt_mean.round(3)} 标准差={tgt_std.round(3)}')

    rows = []
    for dior_cls, coco_name in CLASSES:
        print(f'\n--- {dior_cls} (COCO: {coco_name}) ---')
        images, gts = U.load_dior(dior_cls, n=N_PER_CLASS)
        if len(images) < 5:
            print('  图太少，跳过'); continue
        prompt = f'a photo of {coco_name}'

        # 文本锚(类文本嵌入)
        tok = clip_module.tokenize([f'a photo of {coco_name}']).to(DEVICE)
        with torch.no_grad():
            text_emb = clip_m.encode_text(tok).float()
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # A 路: 原始
        dets_a = U.gdino_detect(images, prompt, 0.0)
        ap_a = per_img_ap(dets_a, gts)
        sim_a = clip_align(images, coco_name, clip_m, clip_pre)
        print(f'  AP50 raw(A)={ap_a:.4f}  CLIP_align(A)={sim_a:.4f}')

        # B 路: 色彩迁移
        mig = [reinhard(im, tgt_mean, tgt_std) for im in images]
        dets_b = U.gdino_detect(mig, prompt, 0.0)
        ap_b = per_img_ap(dets_b, gts)
        sim_b = clip_align(mig, coco_name, clip_m, clip_pre)
        print(f'  AP50 migrated(B)={ap_b:.4f}  Δ={ap_b-ap_a:+.4f}  CLIP_align(B)={sim_b:.4f}  Δ={sim_b-sim_a:+.4f}')

        # C 路: 文本锚定语义统一
        uni, uni_sims = text_anchored_unify(images, text_emb, clip_m, clip_pre)
        dets_c = U.gdino_detect(uni, prompt, 0.0)
        ap_c = per_img_ap(dets_c, gts)
        sim_c = clip_align(uni, coco_name, clip_m, clip_pre)
        print(f'  AP50 unified(C)={ap_c:.4f}  Δ={ap_c-ap_a:+.4f} (vsB {ap_c-ap_b:+.4f})  CLIP_align(C)={sim_c:.4f}  Δ={sim_c-sim_a:+.4f}')

        rows.append({
            'dior_cls': dior_cls, 'coco_name': coco_name, 'n_imgs': len(images),
            'ap_raw': round(float(ap_a), 4), 'ap_migrated': round(float(ap_b), 4),
            'ap_unified': round(float(ap_c), 4),
            'ap_delta_c_vs_a': round(float(ap_c - ap_a), 4),
            'ap_delta_c_vs_b': round(float(ap_c - ap_b), 4),
            'align_raw': round(sim_a, 4), 'align_migrated': round(sim_b, 4),
            'align_unified': round(sim_c, 4),
            'align_delta_c_vs_a': round(sim_c - sim_a, 4),
            'mean_best_sim_c': round(float(np.mean(uni_sims)), 4),
        })

    if rows:
        def mean(key):
            return float(np.mean([r[key] for r in rows]))
        ma, mb, mc = mean('ap_raw'), mean('ap_migrated'), mean('ap_unified')
        sa, sb, sc = mean('align_raw'), mean('align_migrated'), mean('align_unified')
        print(f'\n{"="*72}')
        print(f'均值  AP50   raw(A)={ma:.4f}  migrated(B)={mb:.4f}  unified(C)={mc:.4f}')
        print(f'            ΔC-A={mc-ma:+.4f}   ΔC-B={mc-mb:+.4f}')
        print(f'均值  CLIP   raw(A)={sa:.4f}  migrated(B)={sb:.4f}  unified(C)={sc:.4f}')
        print(f'            ΔC-A={sc-sa:+.4f}')
        out = {'rows': rows, 'mean': {
            'ap_raw': round(ma, 4), 'ap_migrated': round(mb, 4), 'ap_unified': round(mc, 4),
            'ap_delta_c_vs_a': round(mc - ma, 4), 'ap_delta_c_vs_b': round(mc - mb, 4),
            'align_raw': round(sa, 4), 'align_migrated': round(sb, 4), 'align_unified': round(sc, 4),
            'align_delta_c_vs_a': round(sc - sa, 4),
        }, 'note': 'A=原始DIOR; B=Reinhard迁COCO自然分布; C=文本锚定语义统一(CLIP空间朝类文本做温和光度搜索). 小样本方向性探针, 非显著性结论'}
        with open(SAVE, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'结果 -> {SAVE}')


if __name__ == '__main__':
    main()
