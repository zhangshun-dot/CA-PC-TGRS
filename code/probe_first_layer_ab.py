# -*- coding: utf-8 -*-
"""probe_first_layer_ab.py — A/B 小样本探针（第一层图像优化方向性验证）

A 路: 原始 DIOR 图 -> GDINO-tiny("a photo of {cls}") -> 逐图 AP50
B 路: DIOR 图经 Reinhard 色彩迁移(拉向 COCO 自然分布, 零训练) -> 同检测器 -> 同指标
对齐代理: CLIP 图像-文本余弦相似度(整图 vs "a photo of {cls}")，A/B 对比

目的: 验证"在第一层(图像)做优化"对 OVOD 检测有无方向性增益，
      并观察增益是否伴随 V-L 对齐相似度变化（框架核心问题）。

复用 unified_eval 的 load_dior / gdino_detect / compute_ap50_img / get_clip。
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import json
import numpy as np
import torch
from PIL import Image
import unified_eval as U

DEVICE = U.DEVICE
PRE_FILTER = U.PRE_FILTER
N_PER_CLASS = 20
SAVE = U.SAVE_DIR / 'probe_first_layer_ab.json'

# DIOR 中可用的、可映射到 COCO 6 类的三类（motorcycle 在 DIOR 无对应）
CLASSES = [('airplane', 'airplane'), ('ship', 'ship'), ('vehicle', 'vehicle')]


def coco_natural_stats(n=80):
    """COCO val2017 全局自然图像色彩统计（RGB, 0-1），作为迁移目标分布。"""
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
        ms.append(im.mean(0))
        ss.append(im.std(0))
    return np.mean(ms, 0), np.mean(ss, 0)


def reinhard(img, tgt_mean, tgt_std):
    """单图色彩迁移: 归一化到零均值单位标准差后施加 COCO 目标均值/标准差。"""
    im = np.asarray(img).astype(np.float32) / 255.0
    src_mean = im.reshape(-1, 3).mean(0)
    src_std = im.reshape(-1, 3).std(0)
    out = (im - src_mean) / (src_std + 1e-6) * tgt_std + tgt_mean
    out = np.clip(out, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))


def per_img_ap(dets, gts):
    """对单类: 每图 pre-filter 后逐图匹配 AP50 (compute_ap50_img 口径)。"""
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


_CLIP = {}

def get_clip_model():
    if 'm' not in _CLIP:
        import clip
        m, pre = clip.load(U.CLIP_PATH, device='cpu')
        m.eval()
        _CLIP['m'] = m
        _CLIP['pre'] = pre
    return _CLIP['m'], _CLIP['pre']


def clip_align(images, cls):
    """对齐代理: CLIP 图像-文本余弦相似度(整图 vs 'a photo of {cls}')。"""
    try:
        import clip
        m, pre = get_clip_model()
        tok = clip.tokenize([f'a photo of {cls}'])
        with torch.no_grad():
            tf = m.encode_text(tok).float()
            tf = tf / tf.norm(dim=-1, keepdim=True)
        sims = []
        for im in images:
            with torch.no_grad():
                f = m.encode_image(pre(im).unsqueeze(0)).float()
                f = f / f.norm(dim=-1, keepdim=True)
            sims.append(float((f @ tf.T).item()))
        return float(np.mean(sims))
    except Exception as e:
        print(f'  [对齐代理跳过] CLIP 加载失败: {repr(e)[:120]}')
        return None


def main():
    print('=' * 72)
    print('A/B 探针: 第一层图像优化对 OVOD 检测的方向性影响 (小样本)')
    print(f'检测器=GDINO-tiny  协议=pre-filter {PRE_FILTER}  逐图 AP50  每类 {N_PER_CLASS} 图')
    print('=' * 72)

    tgt_mean, tgt_std = coco_natural_stats()
    print(f'COCO 自然分布目标均值={tgt_mean.round(3)} 标准差={tgt_std.round(3)}')

    rows = []
    for dior_cls, coco_name in CLASSES:
        print(f'\n--- {dior_cls} (COCO: {coco_name}) ---')
        images, gts = U.load_dior(dior_cls, n=N_PER_CLASS)
        if len(images) < 5:
            print('  图太少，跳过')
            continue
        prompt = f'a photo of {coco_name}'

        # A 路: 原始
        dets_a = U.gdino_detect(images, prompt, 0.0)
        ap_a = per_img_ap(dets_a, gts)
        print(f'  AP50 raw(A)={ap_a:.4f}')
        sim_a = clip_align(images, coco_name)

        # B 路: 色彩迁移
        mig = [reinhard(im, tgt_mean, tgt_std) for im in images]
        dets_b = U.gdino_detect(mig, prompt, 0.0)
        ap_b = per_img_ap(dets_b, gts)
        print(f'  AP50 migrated(B)={ap_b:.4f}   Δ={ap_b-ap_a:+.4f}')
        sim_b = clip_align(mig, coco_name)

        row = {
            'dior_cls': dior_cls, 'coco_name': coco_name,
            'n_imgs': len(images),
            'ap_raw': round(float(ap_a), 4),
            'ap_migrated': round(float(ap_b), 4),
            'ap_delta': round(float(ap_b - ap_a), 4),
            'align_raw': (round(sim_a, 4) if sim_a is not None else None),
            'align_migrated': (round(sim_b, 4) if sim_b is not None else None),
            'align_delta': (round(sim_b - sim_a, 4) if (sim_a is not None and sim_b is not None) else None),
        }
        rows.append(row)
        if sim_a is not None and sim_b is not None:
            print(f'  CLIP raw={sim_a:.4f}  migrated={sim_b:.4f}  Δ={sim_b-sim_a:+.4f}')
        else:
            print('  CLIP 对齐代理未计算')

    if rows:
        ma = float(np.mean([r['ap_raw'] for r in rows]))
        mb = float(np.mean([r['ap_migrated'] for r in rows]))
        print(f'\n{"="*72}')
        print(f'均值  AP50   raw={ma:.4f}  migrated={mb:.4f}  Δ={mb-ma:+.4f}')
        align_rows = [r for r in rows if r['align_raw'] is not None]
        if align_rows:
            sa = float(np.mean([r['align_raw'] for r in align_rows]))
            sb = float(np.mean([r['align_migrated'] for r in align_rows]))
            print(f'均值  CLIP  raw={sa:.4f}  migrated={sb:.4f}  Δ={sb-sa:+.4f}')
            align_mean = {'align_raw': round(sa, 4), 'align_migrated': round(sb, 4),
                          'align_delta': round(sb - sa, 4)}
        else:
            sa = sb = None
            print('均值  CLIP  未计算')
            align_mean = {'align_raw': None, 'align_migrated': None, 'align_delta': None}
        out = {'rows': rows,
               'mean': {'ap_raw': round(ma, 4), 'ap_migrated': round(mb, 4),
                        'ap_delta': round(mb - ma, 4), **align_mean},
               'note': 'A=原始DIOR, B=Reinhard迁移至COCO自然分布; 小样本方向性探针, 非显著性结论'}
        with open(SAVE, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'结果 -> {SAVE}')


if __name__ == '__main__':
    main()
