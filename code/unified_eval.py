# -*- coding: utf-8 -*-
"""unified_eval.py — CD-TDM 跨域方法族统一评估

解决 P0: 之前各脚本评估协议不一致（阈值偷跑 / AP口径不同 / 图集不同）。
这里把所有方法拉到同一条起跑线上，只比"方法本身"的增益。

统一协议（所有方法强制一致）:
  1. GDINO-tiny, post_process threshold=0.0 取全部候选框
  2. 统一 pre-filter = 0.03 —— 含 baseline，杜绝"方法低阈值偷跑"
  3. 统一数据集级 11-point AP50（所有图所有框拼一个 PR 曲线，COCO 官方口径，
     不用 per-image AP 平均）
  4. 数据固定: COCO val {airplane, boat, car} 各 30 张；DIOR {airplane, ship, vehicle} 各 20 张

方法（核心 5 个）:
  baseline — 固定前缀 + GDINO score
  tdm      — OFAT 知识库推荐 top-2 修饰词（静态 TDM）
  dps      — CLIP 域分类逐图选前缀
  vep      — DINOv2 L11 余弦重打分（跨域 DIOR 原型）
  cdvm     — DINOv2 L11 余弦 + 类间负原型抑制（跨域 DIOR 原型）

关键公平性设计:
  * 重打分类（vep/cdvm）与 baseline 用【同一个候选框集合】，只改分数排序
    → AP 提升 = 排序改善，而不是"低阈值多召回"
  * 融合一律用几何平均 sqrt(gd * visual)，不引入 alpha 调参
  * 原型统一来自 DIOR 源域 bank（exemplar_bank_k5_dinov2_layers.pt），
    评估在 COCO 时是真正的跨域 zero/few-shot（不用目标域 GT）
  * CDVM 的马氏距离在无协方差时降级为"余弦 + 类间负原型"，保留负原型抑制思想

输出: 实验数据/unified_eval_results.json + 控制台对比表
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
import pandas as pd

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')

# ── 路径 ──
GDINO_CACHE = (r'C:\Users\张三\.cache\huggingface\hub'
               r'\models--IDEA-Research--grounding-dino-tiny'
               r'\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71')
DINOV2_PATH = r'C:\Users\张三\.cache\modelscope\tiansz\dinov2-base'
CLIP_PATH = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
BANK_PATH = SAVE_DIR / 'exemplar_bank_k5_dinov2_layers.pt'
COCO_IMG_DIR = r'C:\Users\张三\modeling\coco_data\val2017'
COCO_ANN_FILE = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'
DIOR_IMG_DIR = r'C:\Users\张三\modeling\dior_data\full20_images'
DIOR_PARQUET = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'

PRE_FILTER = 0.03        # 统一候选框阈值
VEP_LAYER = 10           # bank 层索引 0-11 → 对应 "L11"
NEG_LAM = 0.5            # 类间负原型抑制系数
N_COCO = 30
N_DIOR = 20

# 跨域类别映射 (bank key 是 DIOR 类名)
DIOR_ID2NAME = {
    0: 'airplane', 1: 'airport', 2: 'baseball field', 3: 'basketball court',
    4: 'bridge', 5: 'chimney', 6: 'dam', 7: 'expressway service area',
    8: 'expressway toll station', 9: 'golf course', 10: 'ground track field',
    11: 'harbor', 12: 'overpass', 13: 'ship', 14: 'stadium',
    15: 'storage tank', 16: 'tennis court', 17: 'train station',
    18: 'vehicle', 19: 'windmill',
}

# 评估实例: (name, dataset, cls_name, domain, prefix, dior_cls)
EVALS = [
    ('COCO-airplane', 'coco', 'airplane', 'natural_scene', 'a photo of', 'airplane'),
    ('COCO-boat',     'coco', 'boat',     'natural_scene', 'a photo of', 'ship'),
    ('COCO-car',      'coco', 'car',      'natural_scene', 'a photo of', 'vehicle'),
    ('DIOR-airplane', 'dior', 'airplane', 'remote_sensing', 'an overhead image of', 'airplane'),
    ('DIOR-ship',     'dior', 'ship',     'remote_sensing', 'an overhead image of', 'ship'),
    ('DIOR-vehicle',  'dior', 'vehicle',  'remote_sensing', 'an overhead image of', 'vehicle'),
]


# ══════════════════════════ 模型（懒加载 + 缓存） ══════════════════════════

_gd = {}

def get_gdino():
    if 'm' not in _gd:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        _gd['p'] = AutoProcessor.from_pretrained(GDINO_CACHE)
        _gd['m'] = AutoModelForZeroShotObjectDetection.from_pretrained(
            GDINO_CACHE, dtype=torch.float32).eval().to(DEVICE)
    return _gd['p'], _gd['m']


def get_dinov2():
    if 'dino' not in _gd:
        from transformers import AutoModel, AutoConfig
        from torchvision import transforms as T
        cfg = AutoConfig.from_pretrained(DINOV2_PATH)
        cfg.output_hidden_states = True
        _gd['dino'] = AutoModel.from_pretrained(DINOV2_PATH, config=cfg).eval().to(DEVICE)
        _gd['dino_pre'] = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return _gd['dino'], _gd['dino_pre']


def get_clip():
    if 'clip' not in _gd:
        import clip
        _gd['clip'], _gd['clip_pre'] = clip.load(CLIP_PATH, device=DEVICE)
        _gd['clip'].eval()
    return _gd['clip'], _gd['clip_pre']


def get_bank():
    if 'bank' not in _gd:
        _gd['bank'] = torch.load(BANK_PATH, map_location='cpu')
    return _gd['bank']


# ══════════════════════════ 数据加载 ══════════════════════════

def load_coco(cls_name, n=N_COCO):
    from pycocotools.coco import COCO
    coco = COCO(COCO_ANN_FILE)
    cids = coco.getCatIds(catNms=[cls_name])
    if not cids:
        return [], []
    cat_id = cids[0]
    iids = coco.getImgIds(catIds=[cat_id])
    images, gts = [], []
    for iid in iids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[iid], catIds=[cat_id], iscrowd=False))
        if not anns:
            continue
        info = coco.loadImgs([iid])[0]
        p = os.path.join(COCO_IMG_DIR, info['file_name'])
        if not os.path.exists(p):
            continue
        try:
            im = Image.open(p).convert('RGB')
        except Exception:
            continue
        g = [[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
             for a in anns]
        images.append(im)
        gts.append(g)
        if len(images) >= n:
            break
    return images, gts


def load_dior(cls_name, n=N_DIOR):
    # 从 full20_images 文件名提取 test image_id
    test_ids = set()
    for f in glob.glob(os.path.join(DIOR_IMG_DIR, '*.jpg')):
        parts = Path(f).stem.split('_')
        for part in reversed(parts):
            if part.isdigit():
                test_ids.add(int(part))
                break

    name2id = {v: k for k, v in DIOR_ID2NAME.items()}
    cls_id = name2id.get(cls_name)
    if cls_id is None:
        return [], []

    df = pd.read_parquet(DIOR_PARQUET)
    gt_index = defaultdict(list)
    for _, row in df.iterrows():
        img_id = int(row['image_id'])
        if img_id not in test_ids:
            continue
        for i, cat in enumerate(row['objects']['category']):
            if int(cat) == cls_id:
                x, y, w, h = [int(v) for v in row['objects']['bbox'][i]]
                gt_index[img_id].append([x, y, x + w, y + h])

    patterns = [
        os.path.join(DIOR_IMG_DIR, f"{cls_name}_*.jpg"),
        os.path.join(DIOR_IMG_DIR, f"{cls_name.replace(' ', '_')}_*.jpg"),
    ]
    paths = []
    for pat in patterns:
        paths = sorted(glob.glob(pat))
        if paths:
            break

    images, gts = [], []
    for p in paths:
        parts = Path(p).stem.split('_')
        img_id = None
        for part in reversed(parts):
            if part.isdigit():
                img_id = int(part)
                break
        if img_id is None or img_id not in test_ids:
            continue
        g = gt_index.get(img_id, [])
        if not g:
            continue
        try:
            im = Image.open(p).convert('RGB')
        except Exception:
            continue
        images.append(im)
        gts.append(g)
        if len(images) >= n:
            break
    return images, gts


# ══════════════════════════ 指标 ══════════════════════════

def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-8)


def compute_ap50(pred_boxes, pred_scores, gt_boxes):
    """⚠️ 全局拼接版 AP50（所有图框与 GT 一次匹配）——仅供历史归档对比，论文不用。

    2026-08-23 审计发现: COCO 图尺寸统一, 跨图框坐标可直接覆盖别图 GT,
    全局匹配产生跨图误匹配, 失效类 AP 系统性虚高 (car 0.21->0.11)。
    所有论文数字改用 compute_ap50_img (标准逐图匹配)。
    """
    if len(gt_boxes) == 0:
        return 0.0 if len(pred_boxes) > 0 else 1.0
    if len(pred_boxes) == 0:
        return 0.0
    idx = np.argsort(-np.array(pred_scores))
    pb_sorted = [pred_boxes[i] for i in idx]
    matched = [False] * len(gt_boxes)
    tp, fp = np.zeros(len(pb_sorted)), np.zeros(len(pb_sorted))
    for i, pb in enumerate(pb_sorted):
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if not matched[j]:
                v = iou(pb, gb)
                if v > best_iou:
                    best_iou, best_j = v, j
        if best_iou >= 0.5 and best_j >= 0:
            tp[i] = 1
            matched[best_j] = True
        else:
            fp[i] = 1
    tc, fc = np.cumsum(tp), np.cumsum(fp)
    recalls = tc / len(gt_boxes)
    precs = tc / (tc + fc + 1e-8)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        m = recalls >= t
        if m.any():
            ap += float(np.max(precs[m]))
    return ap / 11.0


def compute_ap50_img(im_preds, im_gts):
    """标准逐图匹配 AP50 (COCO 官方口径): 每图内独立贪心匹配, 全局排序画 PR。

    im_preds: list of (boxes (N_i,4), scores (N_i,)), 与 im_gts 逐图对齐
    im_gts:   list of gt box lists (每图一个 list)
    11-point max-precision 插值, 与 pycocotools 一致:
      1) 每图内按分数降序贪心匹配 (GT 逐图互斥)
      2) 所有图的 (score, tp) 按分数全局降序排序 → PR 曲线
    """
    n_gt = sum(len(g) for g in im_gts)
    if n_gt == 0:
        return 0.0 if any(len(p[0]) for p in im_preds) else 1.0
    evals = []  # (score, tp)
    for (boxes, scores), gts in zip(im_preds, im_gts):
        if len(boxes) == 0:
            continue
        matched = [False] * len(gts)
        idx = np.argsort(-np.asarray(scores))
        for i in idx:
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gts):
                if not matched[j]:
                    v = iou(boxes[i], gb)
                    if v > best_iou:
                        best_iou, best_j = v, j
            if best_iou >= 0.5 and best_j >= 0:
                evals.append((float(scores[i]), 1))
                matched[best_j] = True
            else:
                evals.append((float(scores[i]), 0))
    if not evals:
        return 0.0
    evals.sort(key=lambda x: -x[0])
    tp = np.array([e[1] for e in evals])
    tc = np.cumsum(tp)
    fc = np.cumsum(1 - tp)
    rec = tc / n_gt
    prec = tc / (tc + fc + 1e-8)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        m = rec >= t
        if m.any():
            ap += float(prec[m].max())
    return ap / 11.0


def compute_ap_coco(im_preds, im_gts, iou_thres=None):
    """COCO 风格 AP@[0.5:0.95]（v2 冻结协议 §4 补充指标）。

    与 compute_ap50_img（11 点 VOC 风格）不同：本函数在 10 个 IoU 阈值
    (0.5:0.05:0.95) 上各做逐图贪心匹配 + 全局 PR + **101 点全召回插值**，
    再对阈值取平均 —— 对齐 pycocotools 的 COCO 惯例（论文仍以 11 点 AP50
    为主表，本函数只用于 v2 补充/期刊投稿口径）。
    """
    if iou_thres is None:
        iou_thres = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    aps = []
    for th in iou_thres:
        aps.append(_ap_at_iou(im_preds, im_gts, float(th)))
    return float(np.mean(aps))


def _ap_at_iou(im_preds, im_gts, iou_th):
    """单 IoU 阈值下的逐图匹配 AP，101 点全召回插值。"""
    n_gt = sum(len(g) for g in im_gts)
    if n_gt == 0:
        return 0.0 if any(len(p[0]) for p in im_preds) else 1.0
    evals = []
    for (boxes, scores), gts in zip(im_preds, im_gts):
        if len(boxes) == 0:
            continue
        matched = [False] * len(gts)
        idx = np.argsort(-np.asarray(scores))
        for i in idx:
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gts):
                if not matched[j]:
                    v = iou(boxes[i], gb)
                    if v > best_iou:
                        best_iou, best_j = v, j
            if best_iou >= iou_th and best_j >= 0:
                evals.append((float(scores[i]), 1))
                matched[best_j] = True
            else:
                evals.append((float(scores[i]), 0))
    if not evals:
        return 0.0
    evals.sort(key=lambda x: -x[0])
    tp = np.array([e[1] for e in evals])
    tc = np.cumsum(tp)
    fc = np.cumsum(1 - tp)
    rec = tc / n_gt
    prec = tc / (tc + fc + 1e-8)
    # 101 点全召回插值
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        m = rec >= t
        if m.any():
            ap += float(prec[m].max())
    return ap / 101.0


def split_by_img(boxes, scores, img_ids, n_imgs):
    """把扁平框/分数按图归属拆成 per-image (boxes, scores) 列表 (供 compute_ap50_img)。"""
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    img_ids = np.asarray(img_ids)
    out = []
    for i in range(n_imgs):
        m = img_ids == i
        out.append((boxes[m], scores[m]))
    return out


# ══════════════════════════ GDINO 检测 ══════════════════════════

def gdino_detect(images, prompt, thresh=0.0):
    """对一批图用同一 prompt 检测，返回每图 (boxes, scores)。"""
    gd_p, gd_m = get_gdino()
    out = []
    for img in images:
        ins = gd_p(images=img, text=prompt, return_tensors='pt')
        ins = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in ins.items()}
        with torch.no_grad():
            o = gd_m(**ins)
        ts = torch.tensor([img.size[::-1]]).to(DEVICE)
        r = gd_p.post_process_grounded_object_detection(o, threshold=thresh, target_sizes=ts)[0]
        out.append((r['boxes'].cpu().numpy(), r['scores'].cpu().numpy()))
    return out


def filter_flatten(dets, gts):
    """pre-filter 后把每图框拼平，返回 (boxes_xyxy, scores, gts_flat)。"""
    all_b, all_s, all_g = [], [], []
    for (boxes, scores), g in zip(dets, gts):
        keep = scores > PRE_FILTER
        if not keep.any():
            continue
        all_b.extend(boxes[keep].tolist())
        all_s.extend(scores[keep].tolist())
        all_g.extend(g)
    return all_b, all_s, all_g


# ══════════════════════════ DINOv2 编码 ══════════════════════════

def encode_dinov2(crop, layer_idx=VEP_LAYER):
    dino, pre = get_dinov2()
    t = pre(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        o = dino(t)
    f = o.hidden_states[layer_idx + 1][0, 0, :]
    return f / f.norm(dim=-1, keepdim=True)


# ══════════════════════════ 各方法 ══════════════════════════

def method_baseline(images, gts, prefix, cls_name):
    dets = gdino_detect(images, f'{prefix} {cls_name}', 0.0)
    boxes, scores, gts_flat = filter_flatten(dets, gts)
    return compute_ap50(boxes, scores, gts_flat)


def method_tdm(images, gts, domain, cls_name, prefix):
    """静态 TDM：OFAT 知识库推荐 top-2 修饰词。"""
    from ofat_knowledge_base import OFATKnowledgeBase
    from dtdm_axis_descriptions import AXIS_MODIFIERS
    kb = OFATKnowledgeBase()
    recs = kb.get_recommended_axes(domain, cls_name)  # [(axis, delta), ...]
    mods = [AXIS_MODIFIERS[a] for a, _ in recs[:2]]
    prompt = f'{prefix} {cls_name}'
    if mods:
        prompt += ', ' + ', '.join(mods)
    dets = gdino_detect(images, prompt, 0.0)
    boxes, scores, gts_flat = filter_flatten(dets, gts)
    return compute_ap50(boxes, scores, gts_flat), prompt


def method_dps(images, gts, cls_name, prefix_ns, prefix_rs):
    """DPS：CLIP 域分类逐图选前缀。"""
    clip_m, clip_pre = get_clip()
    import clip as clip_module

    # 2-way 域分类文本嵌入
    dom_prompts = {
        'remote_sensing': ['a remote sensing image', 'a satellite image of terrain',
                           'an aerial view photograph', 'an overhead image from above'],
        'natural_scene': ['a natural scene photograph', 'a ground-level photo',
                          'an everyday scene image', 'a picture taken from eye level'],
    }
    prefix_map = {'remote_sensing': prefix_rs, 'natural_scene': prefix_ns}

    text_feats = {}
    for dom, prompts in dom_prompts.items():
        tok = clip_module.tokenize(prompts).to(DEVICE)
        with torch.no_grad():
            f = clip_m.encode_text(tok).float()
            f = f / f.norm(dim=-1, keepdim=True)
        text_feats[dom] = f  # (N, 512)

    gd_p, gd_m = get_gdino()
    all_b, all_s, all_g = [], [], []
    for img, g in zip(images, gts):
        # 分类
        imgt = clip_pre(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            imf = clip_m.encode_image(imgt).float()
            imf = imf / imf.norm(dim=-1, keepdim=True)
        best_dom, best_sim = None, -1
        for dom, tf in text_feats.items():
            s = float((imf @ tf.T).max())
            if s > best_sim:
                best_sim, best_dom = s, dom
        prompt = f'{prefix_map[best_dom]} {cls_name}'
        ins = gd_p(images=img, text=prompt, return_tensors='pt')
        ins = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in ins.items()}
        with torch.no_grad():
            o = gd_m(**ins)
        ts = torch.tensor([img.size[::-1]]).to(DEVICE)
        r = gd_p.post_process_grounded_object_detection(o, threshold=0.0, target_sizes=ts)[0]
        boxes, scores = r['boxes'].cpu().numpy(), r['scores'].cpu().numpy()
        keep = scores > PRE_FILTER
        if keep.any():
            all_b.extend(boxes[keep].tolist())
            all_s.extend(scores[keep].tolist())
            all_g.extend(g)
    return compute_ap50(all_b, all_s, all_g)


def rerank_visual(images, dets, gts, dior_cls, use_neg=False):
    """视觉重打分：同一候选框集合，fused = sqrt(gd * visual)。

    use_neg=False → VEP（纯正原型余弦）
    use_neg=True  → CDVM（正原型 + 类间负原型抑制）
    """
    bank = get_bank()
    pos_proto = bank[dior_cls][VEP_LAYER]['prototype'].to(DEVICE)

    neg_proto = None
    if use_neg:
        # 类间负原型：与正原型最接近的 top-3 其他类原型平均
        others = []
        for cls in bank:
            if cls == dior_cls:
                continue
            others.append(bank[cls][VEP_LAYER]['prototype'])
        others = torch.stack(others).to(DEVICE)  # (19, 768)
        sims = others @ pos_proto
        topk = sims.topk(3).indices
        neg_proto = others[topk].mean(dim=0)
        neg_proto = neg_proto / neg_proto.norm()

    all_b, all_s = [], []
    for (boxes, scores), img in zip(dets, images):
        keep = scores > PRE_FILTER
        if not keep.any():
            continue
        boxes_f = boxes[keep]
        scores_f = scores[keep]
        vis = []
        for box in boxes_f:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                vis.append(0.0)
                continue
            f = encode_dinov2(img.crop((x1, y1, x2, y2)), VEP_LAYER)
            pos = float((f @ pos_proto).item())
            if neg_proto is not None:
                neg = float((f @ neg_proto).item())
                vis.append(max(0.0, pos - NEG_LAM * neg))
            else:
                vis.append(max(0.0, pos))
        gd = np.clip(scores_f, 1e-6, 1.0)
        vs = np.clip(np.array(vis), 1e-6, 1.0)
        fused = np.sqrt(gd * vs)
        all_b.extend(boxes_f.tolist())
        all_s.extend(fused.tolist())
    gt_flat = [g for img_gts in gts for g in img_gts]
    return compute_ap50(all_b, all_s, gt_flat)


# ══════════════════════════ 主流程 ══════════════════════════

def run_eval():
    print('=' * 70)
    print('统一评估: 5方法 × (COCO 3类 + DIOR 3类)')
    print(f'协议: GDINO th=0.0 → pre-filter={PRE_FILTER} → 数据集级 AP50')
    print('=' * 70)

    results = {}
    for name, dataset, cls_name, domain, prefix, dior_cls in EVALS:
        print(f'\n{"─"*70}')
        print(f'[{name}] 加载数据...')
        if dataset == 'coco':
            images, gts = load_coco(cls_name)
        else:
            images, gts = load_dior(cls_name)
        n_gt = sum(len(g) for g in gts)
        print(f'  {len(images)} 图, {n_gt} GT 框')

        if len(images) < 5:
            print('  图太少，跳过')
            continue

        row = {}

        # 1) baseline（候选框缓存，供重打分类复用）
        base_dets = gdino_detect(images, f'{prefix} {cls_name}', 0.0)
        bb, bs, bgt = filter_flatten(base_dets, gts)
        row['baseline'] = compute_ap50(bb, bs, bgt)
        print(f'  baseline        AP50={row["baseline"]:.4f}')

        # 2) tdm
        tdm_ap, tdm_prompt = method_tdm(images, gts, domain, cls_name, prefix)
        row['tdm'] = tdm_ap
        print(f'  tdm             AP50={tdm_ap:.4f}  ("{tdm_prompt}")')

        # 3) dps
        prefix_ns = 'a photo of'
        prefix_rs = 'an overhead image of'
        row['dps'] = method_dps(images, gts, cls_name, prefix_ns, prefix_rs)
        print(f'  dps             AP50={row["dps"]:.4f}')

        # 4) vep（重打分，同候选集）
        row['vep'] = rerank_visual(images, base_dets, gts, dior_cls, use_neg=False)
        print(f'  vep             AP50={row["vep"]:.4f}')

        # 5) cdvm（重打分 + 负原型，同候选集）
        row['cdvm'] = rerank_visual(images, base_dets, gts, dior_cls, use_neg=True)
        print(f'  cdvm            AP50={row["cdvm"]:.4f}')

        row['deltas'] = {k: round(v - row['baseline'], 4)
                         for k, v in row.items() if k != 'baseline' and k != 'deltas'}
        results[name] = row

    # ── 汇总表 ──
    methods = ['baseline', 'tdm', 'dps', 'vep', 'cdvm']
    print(f'\n\n{"="*90}')
    print(f'{"Test":<16s}' + ''.join(f'{m:>10s}' for m in methods))
    print(f'{"─"*90}')
    for name, r in results.items():
        print(f'{name:<16s}' + ''.join(f'{r[m]:>10.4f}' for m in methods))
    print(f'{"─"*90}')
    # 平均
    for label, sel in [('COCO 均值', [n for n in results if n.startswith('COCO')]),
                       ('DIOR 均值', [n for n in results if n.startswith('DIOR')]),
                       ('总均值', list(results))]:
        if not sel:
            continue
        print(f'{label:<16s}' + ''.join(
            f'{np.mean([results[n][m] for n in sel]):>10.4f}' for m in methods))

    # Δ 汇总
    print(f'\nΔ vs baseline:')
    print(f'{"Test":<16s}' + ''.join(f'{m:>10s}' for m in methods[1:]))
    print(f'{"─"*70}')
    for name, r in results.items():
        print(f'{name:<16s}' + ''.join(f'{r["deltas"][m]:>+10.4f}' for m in methods[1:]))

    out = {n: {m: float(r[m]) for m in methods} | {'deltas': r['deltas']}
           for n, r in results.items()}
    out['config'] = {'pre_filter': PRE_FILTER, 'vep_layer': VEP_LAYER,
                     'neg_lam': NEG_LAM, 'n_coco': N_COCO, 'n_dior': N_DIOR,
                     'protocol': 'gdino th=0.0 -> pre-filter 0.03 -> dataset-level AP50'}
    with open(SAVE_DIR / 'unified_eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n结果 → {SAVE_DIR / "unified_eval_results.json"}')
    print('✅ 统一评估完成')


if __name__ == '__main__':
    run_eval()
