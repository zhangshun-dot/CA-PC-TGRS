# -*- coding: utf-8 -*-
"""vlm_attribute_probe.py — Gate 2：生成式 VLM 属性 → 区域级判别 的最小判定实验

回答：VLM 生成的结构化属性，能否在未参与属性生成的测试图像上提升区域级 AP？
（比继续调原型权重更可能是真机制创新的那件事）

两个子实验（协议全 inductive，VLM 只看 K 张校准图的 GT 框，测试图绝不入 VLM）：

Sub-A 文本原型诊断（CLIP 双编码器空间，clip-vit-base-patch32 本地，现在就能跑）：
  对 boat/bus/motorcycle × K∈{1,3,5} × 10 splits（SEED=42，与冻结协议同源）：
    1. category-name prompt      —— 类别名文本
    2. human-attr prompt         —— 人工通用属性描述
    3. vlm-attr prompt           —— VLM 结构化属性（待模型，--vlm 后启用）
    4. vlm-attr+fix prompt       —— VLM 属性经人工筛错（同上，人工修正件）
    5. visual target-mean        —— 朴素目标视觉原型（DINOv2 空间，缓存重放）
  评分：变体 1-4 = CLIP 区域裁剪特征 × 文本嵌入（L2 cos，仅排顺序，AP50）；
        变体 5 = 现有冻结口径 sqrt(gd·F·P_tgt)。全部同 split 同测试集配对。
  判定：① VLM/人工属性文本是否 > 类别名文本（CLIP 空间内）② 文本原型 vs 视觉原型差距。

Sub-B VLM 分组标签器（--vlm 启用，模型下载后）：
  VLM 给 K 个目标框打「视角/形态」标签 → 分组 DINOv2 原型 → 混合评分
  s(r)=Σ_a w_a(r)·cos(f_r,π_c^a)（w_a=softmax）vs 朴素 target-mean。
  这是「属性条件原型库」的最小实现，π_a 全在 DINOv2 空间（不用 CLIP 匹配文本）。

输出：实验数据/vlm_attribute_probe_report.md + vlm_attribute_probe.json
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
import unified_eval as U

SEED = 42
N_SPLITS = 10
K_LIST = [1, 3, 5]
NEG_LAM = 0.5
MIN_SIDE = 10

PROBE = [
    ('COCO-boat', 'boat'),
    ('COCO-bus', 'bus'),
    ('COCO-motorcycle', 'motorcycle'),
]

# 文本变体（Sub-A）。VLM 变体在 --vlm 后从 vlm_attrs.json 读取
HUMAN_ATTRS = {
    'boat': 'a boat, side view, elongated hull, cabin, deck, floating on water surface, seen from ground level',
    'bus': 'a bus, city transit bus, long rectangular body, windows along the side, wheels below, on a road',
    'motorcycle': 'a motorcycle, two wheels, rider seat, handlebars, on a road, side view',
}
CAT_PROMPT = {'boat': 'a boat', 'bus': 'a bus', 'motorcycle': 'a motorcycle'}


def reproduce_images(cls_name, seed):
    """复现 expanded_cross_domain.load_coco_random 的图序（SEED+hash 派生 seed）。"""
    coco = COCO(U.COCO_ANN_FILE)
    cids = coco.getCatIds(catNms=[cls_name])
    cat_id = cids[0]
    iids = coco.getImgIds(catIds=[cat_id])
    rng = np.random.default_rng(seed)
    iids = rng.permutation(iids)
    out = []
    for iid in iids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[iid], catIds=[cat_id], iscrowd=False))
        if not anns:
            continue
        info = coco.loadImgs([iid])[0]
        p = os.path.join(U.COCO_IMG_DIR, info['file_name'])
        if not os.path.exists(p):
            continue
        try:
            Image.open(p).convert('RGB')
        except Exception:
            continue
        out.append((iid, p, [[a['bbox'][0], a['bbox'][1],
                              a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                             for a in anns]))
    return out


def load_clip():
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(U.DEVICE).eval()
    p = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    return m, p


def clip_encode_crops(m, p, image, boxes, bs=128):
    """裁剪候选框 → CLIP 特征（L2 归一化）。boxes: (N,4) xyxy。"""
    if len(boxes) == 0:
        return np.empty((0, 512), np.float32)
    crops = [image.crop([int(x1), int(y1), int(x2), int(y2)])
             for (x1, y1, x2, y2) in boxes]
    feats = []
    with torch.no_grad():
        for i in range(0, len(crops), bs):
            batch = crops[i:i + bs]
            enc = p(images=batch, return_tensors='pt').to(U.DEVICE)
            f = m.get_image_features(**enc)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def clip_text_emb(m, p, texts):
    enc = p(text=texts, return_tensors='pt', padding=True).to(U.DEVICE)
    with torch.no_grad():
        t = m.get_text_features(**enc)
    t = t / t.norm(dim=-1, keepdim=True)
    return t.cpu().numpy().astype(np.float32)


def eval_ap(test_idx, boxes, scores, gt_boxes):
    all_b, all_s, all_g = [], [], []
    for i in test_idx:
        b = boxes[i]
        if len(b) == 0:
            all_g.extend(gt_boxes[i].tolist())
            continue
        all_b.extend(b.tolist())
        all_s.extend(scores[i].tolist())
        all_g.extend(gt_boxes[i].tolist())
    return U.compute_ap50(all_b, all_s, all_g)


def visualize_ap(test_idx, cand_boxes, cand_gd, cand_feats, gt_boxes, P_np):
    """变体5 视觉原型：score=sqrt(gd·F·P)，同冻结口径。返回 per-image 分数表。"""
    scores = [[] for _ in cand_boxes]
    for i in range(len(cand_boxes)):
        cb = cand_boxes[i]
        if len(cb) == 0:
            scores[i] = np.empty(0, np.float32)
            continue
        gd = np.clip(cand_gd[i], 1e-6, 1.0)
        F = cand_feats[i]
        if len(F):
            vis = np.clip(F @ P_np, 1e-6, 1.0)
            scores[i] = np.sqrt(gd * vis)
        else:
            scores[i] = gd
    return eval_ap(test_idx, cand_boxes, scores, gt_boxes)


def clip_ap(test_idx, cand_boxes, cand_gd, clip_feats, gt_boxes, t_emb):
    """变体1-4：score = CLIP 文本 cos（纯文本匹配）。返回 per-image 分数表。"""
    scores = [[] for _ in cand_boxes]
    for i in range(len(cand_boxes)):
        cb = cand_boxes[i]
        if len(cb) == 0:
            scores[i] = np.empty(0, np.float32)
            continue
        F = clip_feats[i]
        if len(F):
            c = F @ t_emb  # (N,) 文本 cos
            scores[i] = np.clip(c, 1e-6, 1.0)
        else:
            scores[i] = np.full(len(cb), 1e-6, np.float32)
    return eval_ap(test_idx, cand_boxes, scores, gt_boxes)


def reproduce_splits(n):
    rng = np.random.default_rng(SEED)
    return [rng.permutation(n).tolist() for _ in range(N_SPLITS)]


def load_cache(name):
    _npz = np.load(U.SAVE_DIR / 'expanded_cache' / f'{name}.npz', allow_pickle=True)
    return {
        'cb': list(_npz['cand_boxes']), 'cgd': list(_npz['cand_gd']),
        'cf': list(_npz['cand_feats']), 'gt_boxes': list(_npz['gt_boxes']),
        'gt_feats': list(_npz['gt_feats']), 'n': len(_npz['gt_feats']),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--vlm', action='store_true', help='启用 Sub-B / VLM 文本变体（需模型+attrs）')
    ap.add_argument('--attrs', default=r'C:\Users\张三\Desktop\实验数据\vlm_attrs.json')
    args = ap.parse_args()

    clip_model, clip_proc = load_clip()
    # 文本嵌入：类别名 + 人工属性（VLM 变体若 attrs 文件存在则并入）
    text_pool = {}
    for name, tgt in PROBE:
        text_pool[name] = {
            'category': CAT_PROMPT[tgt],
            'human': HUMAN_ATTRS[tgt],
        }
    vlm_texts = {}
    if os.path.exists(args.attrs):
        vlm_texts = json.loads(open(args.attrs, encoding='utf-8').read())
        for name in text_pool:
            if name in vlm_texts:
                text_pool[name]['vlm'] = vlm_texts[name].get('prompt', '')
                text_pool[name]['vlm_fix'] = vlm_texts[name].get('prompt_fix', '')

    results = {}
    all_text_embs = {}
    for name, tgt in PROBE:
        print(f'\n[class] {name}')
        cache = load_cache(name)
        n = cache['n']
        imgs = reproduce_images(tgt, SEED + hash(name) % 1000)
        assert len(imgs) >= n, f'{name}: cache {n} > reproduced {len(imgs)}'
        imgs = imgs[:n]
        # CLIP 区域特征（一次性，全图）
        clip_feats = []
        for i in range(n):
            im = Image.open(imgs[i][1]).convert('RGB')
            f = clip_encode_crops(clip_model, clip_proc, im, cache['cb'][i])
            clip_feats.append(f)
        print(f'  clip features done, mean crops/img = '
              f'{np.mean([len(x) for x in clip_feats]):.1f}')
        # 文本嵌入
        emb = {}
        for variant, text in text_pool[name].items():
            emb[variant] = clip_text_emb(clip_model, clip_proc, [text])[0]
        all_text_embs[name] = emb
        perms = reproduce_splits(n)
        per = {v: {K: [[] for _ in range(N_SPLITS)] for K in K_LIST}
               for v in list(text_pool[name].keys()) + ['visual']}
        for s in range(N_SPLITS):
            for K in K_LIST:
                sup = perms[s][:K]
                test = perms[s][K:]
                # 变体5 视觉原型
                sf = [f for i in sup for f in cache['gt_feats'][i]]
                if len(sf):
                    P_tgt = np.mean(sf, axis=0)
                    P_tgt = P_tgt / np.linalg.norm(P_tgt)
                else:
                    P_tgt = None
                for variant, t_emb in emb.items():
                    apv = clip_ap(test, cache['cb'], cache['cgd'], clip_feats,
                                  cache['gt_boxes'], t_emb)
                    per[variant][K][s] = apv
                # visual：若 P_tgt 空则跳过（K 校准图无 GT 框）
                if P_tgt is not None:
                    apv = visualize_ap(test, cache['cb'], cache['cgd'], cache['cf'],
                                       cache['gt_boxes'], P_tgt)
                    per['visual'][K][s] = apv
        results[name] = {
            'n_img': n,
            'variants': {v: {str(K): [per[v][K][s] for s in range(N_SPLITS)]
                             for K in K_LIST} for v in per},
        }
        for v in per:
            for K in K_LIST:
                vals = [x for x in per[v][K] if x is not None]
                if vals:
                    print(f'  {v:8s} K={K}: {np.mean(vals):.4f}')

    # ── 报告 ──
    L = []
    L.append('## VLM 属性 Gate 探针报告（Sub-A 文本原型诊断）\n')
    L.append('协议：boat/bus/motorcycle，SEED=42 冻结 10 划分，同 split 同测试集配对；'
             '测试图绝不入属性生成。变体1-4=CLIP 文本匹配（区域裁剪×文本 cos，纯排序）；'
             '变体5=朴素目标视觉原型（DINOv2 空间，冻结口径）。\n')
    for name, tgt in PROBE:
        L.append(f'### {name}\n')
        L.append('| 变体 | K=1 | K=3 | K=5 |')
        L.append('|---|---|---|---|')
        r = results[name]
        for v in r['variants']:
            row = []
            for K in K_LIST:
                vals = [x for x in r['variants'][v][str(K)] if x is not None]
                row.append(f'{np.mean(vals):.4f}' if vals else '—')
            L.append(f'| {v} | ' + ' | '.join(row) + ' |')
        L.append('')
    L.append('### 判定\n')
    L.append('- ① 文本变体（category/human）在 boat 上是否 >0.1 AP（CLIP 空间跨域判别力下限）？')
    L.append('- ② visual target-mean 是否显著 > 任何文本变体 → 文本原型路线是否可弃。')
    L.append('- ③（--vlm 后）VLM 属性文本 vs 类别名/人工属性 vs visual 的相对位置。')
    L.append('- Sub-B（VLM 分组原型）独立于本表，--vlm 后追加。')
    (U.SAVE_DIR / 'vlm_attribute_probe_report.md').write_text('\n'.join(L), encoding='utf-8')
    summary = {
        'protocol': 'inductive, VLM/CLIP only sees K support GT boxes; SEED=42; 10 splits; AP50',
        'texts': text_pool,
        'results': results,
        'subB_ready': args.vlm,
    }
    (U.SAVE_DIR / 'vlm_attribute_probe.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\n✅ 报告 → {U.SAVE_DIR / "vlm_attribute_probe_report.md"}')


if __name__ == '__main__':
    main()
