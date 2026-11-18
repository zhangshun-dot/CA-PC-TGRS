# -*- coding: utf-8 -*-
"""validate_voc_rule.py — 用**冻结的残差选择规则**在第二个目标域 (DIOR→VOC2007, 5 类) 上做验证。

A) 源域可得部分 (零目标域数据): Res_c(β) = 1−cos(norm(norm(p_c+βΔ_c)−βΔ_c), p_c),
   仅用 DIOR 源原型 + 两个域的文本模板 -> 按冻结规则 (逐 β 取六/五类最小残差曲线,
   取 <0.15 的最大 β) 得到 β̂_VOC。
B) 目标域 oracle: 用缓存检测在 VOC 上扫 β 的逐图 AP50 增益, 找 β*_VOC 与 β=2.0(COCO 冻结值)。
对比 β̂ vs β* 即可回答"规则是否在第二个目标上选出该目标的最优外推强度"。
"""
import sys, json
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T
import tge_probe_voc as V

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'voc_rule_validation.json'
BETAS = T.BETAS
TARGETS = V.TARGETS  # (eval name, VOC class, DIOR source class)

# ---------- A) 源域原型 + 残差曲线 ----------
print('[A] DIOR 20 类 CLIP 源原型 ...', flush=True)
protos = {}
for cls in U.DIOR_ID2NAME.values():
    imgs, gts = U.load_dior(cls)
    p = V.clip_proto(V.gt_crops(imgs, gts))
    if p is not None:
        protos[cls] = p
print('    built %d prototypes' % len(protos), flush=True)

res = {}
delta_cache = {}
for name, voc_cls, src_cls in TARGETS:
    P_src = protos[src_cls].to(U.DEVICE)
    s_xs = T.text_avg(T.SRC_TMPLS, src_cls)
    d_y = T.text_avg(T.TGT_TMPLS, voc_cls)
    Delta = d_y - s_xs
    delta_cache[name] = Delta.detach().cpu()
    res[name] = [round(T.roundtrip_residual(P_src, Delta, b), 4) for b in BETAS]

names = list(res.keys())
agg = [round(min(res[n][j] for n in names), 4) for j in range(len(BETAS))]
sel_voc = None
for j, b in enumerate(BETAS):
    if agg[j] < 0.15:
        sel_voc = b

print('\n[A] 逐类残差 Res_c(beta), beta grid = %s' % BETAS)
hdr = '%-16s' % 'class' + ''.join('%9s' % b for b in BETAS)
print(hdr); print('-' * len(hdr))
for n in names:
    print('%-16s' % n + ''.join('%9.4f' % v for v in res[n]))
print('%-16s' % 'min over cls' + ''.join('%9.4f' % v for v in agg))
print('\n[A] rule-selected beta on VOC = %s   (COCO 冻结值 = 2.0)' % sel_voc, flush=True)

# ---------- B) VOC AP50 beta 扫描 (检测缓存, 特征每类只编码一次) ----------
def build_pool(images, dets):
    crops, boxes_all, gd_all, img_ids = [], [], [], []
    for ii, ((boxes, scores), img) in enumerate(zip(dets, images)):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            continue
        for box, sc in zip(boxes[keep], scores[keep]):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
            boxes_all.append([x1, y1, x2, y2]); gd_all.append(float(sc)); img_ids.append(ii)
    gd = np.clip(np.array(gd_all), 1e-6, 1.0) if gd_all else np.zeros(0)
    F = T.encode_clip_batch(crops).cpu().numpy() if crops else np.zeros((0, 512))
    return boxes_all, gd, img_ids, F, len(images)

def ap_of(boxes_all, gd, img_ids, F, n_img, gts, p_pos, p_neg):
    if p_neg is not None:
        vis = np.maximum(0.0, F @ p_pos - U.NEG_LAM * (F @ p_neg))
    else:
        vis = np.maximum(0.0, F @ p_pos)
    sc = np.sqrt(gd * np.clip(vis, 1e-6, 1.0))
    return U.compute_ap50_img(U.split_by_img(boxes_all, sc, img_ids, n_img), gts)

print('\n[B] VOC beta 扫描 (逐图 AP50 增益) ...', flush=True)
sweep = {}
means = {b: [] for b in BETAS}
means['base'] = []
for name, voc_cls, src_cls in TARGETS:
    imgs, gts = V.load_voc(voc_cls)
    dets = T.cached_dets(imgs, 'a photo of %s' % voc_cls, name)
    boxes_all, gd, img_ids, F, n_img = build_pool(imgs, dets)
    P_src = protos[src_cls].to(U.DEVICE)
    P_neg = T.get_neg_proto(protos, src_cls)
    p_src_np = P_src.cpu().numpy(); p_neg_np = P_neg.cpu().numpy()
    Delta = delta_cache[name].to(U.DEVICE)
    base = ap_of(boxes_all, gd, img_ids, F, n_img, gts, p_src_np, p_neg_np)
    row = {'baseline': round(float(base), 4), 'n_images': n_img, 'beta': {}}
    means['base'].append(base)
    for b in BETAS:
        P_hat = P_src + b * Delta
        P_hat = (P_hat / P_hat.norm()).cpu().numpy()
        g = ap_of(boxes_all, gd, img_ids, F, n_img, gts, P_hat, p_neg_np) - base
        row['beta'][str(b)] = round(float(g), 4)
        means[b].append(g)
    sweep[name] = row
    print('  %-14s N=%3d base=%.4f | ' % (name, n_img, base)
          + '  '.join('b=%s:%+.4f' % (b, row['beta'][str(b)]) for b in BETAS), flush=True)

mean_gain = {b: round(float(np.mean(means[b])), 4) for b in BETAS}
best_beta = max(BETAS, key=lambda b: mean_gain[b])
print('\n[B] 五类均值增益: ' + '  '.join('b=%s:%+.4f' % (b, mean_gain[b]) for b in BETAS))
print('   beta*_VOC(目标域 oracle) = %s' % best_beta)
print('   beta=2.0 (COCO 冻结值) 增益 = %+.4f   | 峰值 %+.4f' % (mean_gain[2.0], mean_gain[best_beta]))

out = {'config': {'protocol': 'DIOR->VOC2007, 5 classes, AP50 11pt per-image, cached detections',
                  'betas': BETAS, 'src_tmpls': T.SRC_TMPLS, 'tgt_tmpls': T.TGT_TMPLS,
                  'rule': 'aggregate = min over classes of Res_c(beta); pick largest beta with aggregate < 0.15'},
       'residual_per_class': res, 'residual_aggregate_min': agg,
       'rule_selected_beta_voc': sel_voc, 'coco_frozen_beta': 2.0,
       'sweep_per_class': sweep, 'mean_gain_by_beta': mean_gain, 'voc_oracle_beta': best_beta}
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print('\n-> %s' % OUT)
