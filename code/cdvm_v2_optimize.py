# -*- coding: utf-8 -*-
"""CDVM v2: 一轮系统优化

基于 CDVM v1 结果:
  - airplane: 0.759→0.808 (+0.049, cos_dino_dior 最佳)
  - boat:     0.212→0.249 (+0.037, cos_clip_proj 最佳)
  - car:      0.134→0.266 (+0.133, late_fusion 最佳)

优化方向:
  1. [温度缩放] cosine → exp(cosine/τ), τ 小→信号锐化
  2. [融合权重] fused = gd^α × vs^(1-α), per-class 搜索最优 α
  3. [多层集成] L10+L11+L12 加权平均, 替代单层
  4. [DINOv2 patch tokens] CLS + mean(patch) 结合, 捕获局部信息
  5. [投影头 v2] 更大 hidden, 更多 epoch, cosine annealing LR
  6. [hard negative] IoU [0.1, 0.5) 的 GDINO boxes 作为难负样本
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')

DINOV2_PATH = r'C:\Users\张三\.cache\modelscope\tiansz\dinov2-base'
CLIP_PATH = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
GDINO_CACHE = (r'C:\Users\张三\.cache\huggingface\hub'
               r'\models--IDEA-Research--grounding-dino-tiny'
               r'\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71')
COCO_IMG_DIR = r'C:\Users\张三\modeling\coco_data\val2017'
COCO_ANN_FILE = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'

BANK_PATH = SAVE_DIR / 'exemplar_bank_k5_dinov2_layers.pt'

N_IMAGES = 30

# ══════════════════════════════════════════════════════
# 模型加载 (复用 v1)
# ══════════════════════════════════════════════════════

def get_dinov2():
    from transformers import AutoModel, AutoConfig
    config = AutoConfig.from_pretrained(DINOV2_PATH)
    config.output_hidden_states = True
    model = AutoModel.from_pretrained(DINOV2_PATH, config=config).eval().to(DEVICE)
    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def get_clip():
    import clip
    model, preprocess = clip.load(CLIP_PATH, device=DEVICE)
    return model.eval(), preprocess


def get_gdino():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    proc = AutoProcessor.from_pretrained(GDINO_CACHE)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_CACHE, dtype=torch.float32
    ).eval().to(DEVICE)
    return proc, model


# ══════════════════════════════════════════════════════
# 特征提取 v2: 多层 + patch tokens
# ══════════════════════════════════════════════════════

def extract_dinov2_multi(crop, dino_model, dino_transform, layers=[9, 10, 11]):
    """提取 DINOv2 多层 CLS + mean(patch) 特征。

    Returns:
      cls_feats: list of (768,) tensors, 每层 CLS token
      patch_mean_feats: list of (768,) tensors, 每层 patch mean
    """
    img_t = dino_transform(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        o = dino_model(img_t)
    cls_feats, patch_feats = [], []
    for l in layers:
        hs = o.hidden_states[l + 1]  # (1, 257, 768) for ViT-B/14
        cls_feats.append(hs[0, 0, :] / hs[0, 0, :].norm())       # CLS
        patch_mean = hs[0, 1:, :].mean(dim=0)                     # mean patch
        patch_feats.append(patch_mean / patch_mean.norm())
    return cls_feats, patch_feats


def extract_clip_feat(crop, clip_model, clip_preprocess):
    img_t = clip_preprocess(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = clip_model.encode_image(img_t).float()
    return (feat / feat.norm(dim=-1, keepdim=True)).squeeze(0)


# ══════════════════════════════════════════════════════
# 温度缩放相似度
# ══════════════════════════════════════════════════════

def tempered_cosine(feat, proto, tau=0.1):
    """cosine / tau → 温度缩放, 低 τ 锐化分布"""
    cos = (feat @ proto).clamp(min=-1, max=1)
    return (cos / tau).exp()  # 温度缩放 → 正区间变宽


# ══════════════════════════════════════════════════════
# 投影头 v2: 更大 capacity, 无 BatchNorm (用 LayerNorm)
# ══════════════════════════════════════════════════════

class ProjectionHeadV2(nn.Module):
    def __init__(self, in_dim, hidden=1024, out_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        h = self.net(x)
        return h / h.norm(dim=-1, keepdim=True)


# ══════════════════════════════════════════════════════
# Mahalanobis (复用 v1)
# ══════════════════════════════════════════════════════

def compute_mahalanobis_params(exemplar_feats, shrinkage=0.3):
    K, D = exemplar_feats.shape
    mu = exemplar_feats.mean(dim=0)
    centered = exemplar_feats - mu
    cov_sample = (centered.T @ centered) / (K - 1) if K > 1 else torch.eye(D, device=centered.device) * 1e-3
    cov_shrink = shrinkage * cov_sample + (1 - shrinkage) * torch.eye(D, device=cov_sample.device)
    cov_shrink = cov_shrink + 1e-4 * torch.eye(D, device=cov_shrink.device)
    try:
        L = torch.linalg.cholesky(cov_shrink)
        inv_cov = torch.cholesky_inverse(L)
        log_det = 2 * torch.sum(torch.log(torch.diag(L) + 1e-8))
    except RuntimeError:
        inv_cov = torch.eye(D, device=cov_shrink.device)
        log_det = 0.0
    return mu, inv_cov, log_det


def mahalanobis_sim(feat, mu, inv_cov):
    diff = feat - mu.to(feat.device)
    if diff.dim() == 1:
        diff = diff.unsqueeze(0)
    m = diff @ inv_cov.to(feat.device)
    dist_sq = (m * diff).sum(dim=1).clamp(min=0)
    return torch.exp(-0.5 * dist_sq).squeeze(0) if feat.dim() == 1 else torch.exp(-0.5 * dist_sq)


# ══════════════════════════════════════════════════════
# 数据
# ══════════════════════════════════════════════════════

def load_coco_data(class_name, n=N_IMAGES):
    from pycocotools.coco import COCO
    coco = COCO(COCO_ANN_FILE)
    cat_ids = coco.getCatIds(catNms=[class_name])
    if not cat_ids:
        return [], []
    cat_id = cat_ids[0]
    img_ids = coco.getImgIds(catIds=[cat_id])
    images, gts_all = [], []
    for iid in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[iid], catIds=[cat_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue
        img_info = coco.loadImgs([iid])[0]
        path = os.path.join(COCO_IMG_DIR, img_info['file_name'])
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            continue
        gts = [[a['bbox'][0], a['bbox'][1],
                a['bbox'][0] + a['bbox'][2],
                a['bbox'][1] + a['bbox'][3]] for a in anns]
        images.append(img)
        gts_all.append(gts)
        if len(images) >= n:
            break
    return images, gts_all


def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-8)


def compute_ap50(pred_boxes, pred_scores, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    idx = np.argsort(-np.array(pred_scores))
    pred_boxes = [pred_boxes[i] for i in idx]
    matched = [False] * len(gt_boxes)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    for i, pb in enumerate(pred_boxes):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            if matched[j]:
                continue
            iou = compute_iou(pb, gt)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= 0.5 and best_j >= 0:
            tp[i] = 1
            matched[best_j] = 1
        else:
            fp[i] = 1
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / (len(gt_boxes) + 1e-8)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            ap += float(np.max(precisions[mask]))
    return ap / 11.0


def collect_training_pairs_v2(images, gts_per_image, class_name, gd_proc, gd_model):
    """v2: 区分 easy negative 和 hard negative。

    Hard negative: GDINO检测框 IoU∈[0.1, 0.5) → 真正需要区分的
    Easy negative: IoU < 0.1 → 简单负例
    """
    positives, hard_neg, easy_neg = [], [], []

    for img, gts in zip(images, gts_per_image):
        for gt in gts:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            if x2 > x1 + 10 and y2 > y1 + 10:
                positives.append(img.crop((x1, y1, x2, y2)))

        prompt = f'a photo of {class_name}'
        inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            o = gd_model(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(DEVICE)
        r = gd_proc.post_process_grounded_object_detection(o, threshold=0.0, target_sizes=ts)[0]
        boxes = r['boxes'].cpu().numpy()
        scores = r['scores'].cpu().numpy()

        keep = scores > 0.1
        boxes = boxes[keep]

        for box in boxes:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            max_iou = max((compute_iou([x1, y1, x2, y2], gt) for gt in gts), default=0.0)
            crop = img.crop((x1, y1, x2, y2))
            if 0.1 <= max_iou < 0.5:
                hard_neg.append(crop)
            elif max_iou < 0.1:
                easy_neg.append(crop)

    return positives, hard_neg, easy_neg


# ══════════════════════════════════════════════════════
# 投影头训练 v2
# ══════════════════════════════════════════════════════

def train_projection_head_v2(pos_crops, hard_neg_crops, easy_neg_crops,
                               extract_fn, in_dim, proj_head, pos_feats_raw,
                               epochs=150, lr=1e-3, margin=0.3):
    """v2: cosine annealing LR, 更多 epoch, hard negative weighting。

    损失 = margin_ranking(TP, hard_neg) + 0.3 * margin_ranking(TP, easy_neg)
    """
    if len(pos_crops) < 3:
        return proj_head, None

    optimizer = torch.optim.AdamW(proj_head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    all_neg = hard_neg_crops + easy_neg_crops
    all_neg_feats = torch.stack([extract_fn(c) for c in all_neg])
    pos_feats = pos_feats_raw.clone()

    n_pos, n_neg = len(pos_feats), len(all_neg_feats)
    print(f"    训练投影头 v2: TP={n_pos}, HN={len(hard_neg_crops)}, EN={len(easy_neg_crops)}, "
          f"epochs={epochs}")

    best_margin = -1

    for epoch in range(epochs):
        bidx_p = torch.randint(0, n_pos, (min(48, n_pos),))
        bidx_n = torch.randint(0, n_neg, (min(48, n_neg),))

        p_feat = pos_feats[bidx_p].to(DEVICE)
        n_feat = all_neg_feats[bidx_n].to(DEVICE)

        p_proj = proj_head(p_feat)
        n_proj = proj_head(n_feat)

        # 在投影空间计算 Mahalanobis
        mp = compute_mahalanobis_params(p_proj.detach())
        mu_b, inv_b, _ = mp

        p_sim = mahalanobis_sim(p_proj, mu_b, inv_b)
        n_sim = mahalanobis_sim(n_proj, mu_b, inv_b)

        # Margin ranking
        loss = F.relu(margin - p_sim.unsqueeze(1) + n_sim.unsqueeze(0)).mean()

        # 正则化: 保持投影后特征在单位球面上
        reg = ((p_proj.norm(dim=-1) - 1.0).abs().mean() +
               (n_proj.norm(dim=-1) - 1.0).abs().mean())
        loss = loss + 0.001 * reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proj_head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 30 == 0:
            with torch.no_grad():
                pm = p_sim.mean().item()
                nm = n_sim.mean().item()
            cur_margin = pm - nm
            if cur_margin > best_margin:
                best_margin = cur_margin
            print(f"      epoch {epoch:3d}: loss={loss.item():.4f}, "
                  f"TP_sim={pm:.4f}, FP_sim={nm:.4f}, margin={cur_margin:+.4f}, "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    # 训练完后在投影空间计算 Mahalanobis 参数
    with torch.no_grad():
        all_pos_proj = proj_head(pos_feats.to(DEVICE)).cpu()
    mah_proj = compute_mahalanobis_params(all_pos_proj)

    # 投影后 prototype
    with torch.no_grad():
        proto_proj = proj_head(pos_feats[:5].to(DEVICE)).mean(dim=0).cpu()
        proto_proj = proto_proj / proto_proj.norm()

    return proj_head, mah_proj, proto_proj


# ══════════════════════════════════════════════════════
# 融合权重网格搜索
# ══════════════════════════════════════════════════════

def grid_search_fusion_weights(all_gd_scores, all_vs_values, all_gts,
                                 all_boxes, vs_keys):
    """对每个 vs 信号搜索最优融合权重 α。

    fused = gd^α × vs^(1-α), α∈[0.1, 0.9], step=0.1

    Returns: {key: (best_ap, best_alpha)}
    """
    results = {}
    for key in vs_keys:
        vs_vals = all_vs_values[key]
        best_ap, best_alpha = 0, 0.5
        for alpha in np.arange(0.1, 1.0, 0.05):
            fused = [g**alpha * max(v, 1e-8)**(1-alpha)
                     for g, v in zip(all_gd_scores, vs_vals)]
            ap = compute_ap50(all_boxes, np.array(fused), all_gts)
            if ap > best_ap:
                best_ap, best_alpha = ap, alpha
        results[key] = (best_ap, best_alpha)
    return results


# ══════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("CDVM v2: 系统优化 — 温度缩放 + 多层 + 融合权重 + 投影头 v2")
    print("=" * 70)

    gd_proc, gd_model = get_gdino()
    dino_model, dino_transform = get_dinov2()
    clip_model, clip_preprocess = get_clip()
    bank = torch.load(BANK_PATH, map_location='cpu')

    CLASS_MAP = {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle'}
    DINO_LAYERS = [9, 10, 11]  # L10, L11, L12

    all_results = {}

    for coco_cls in ['airplane', 'boat', 'car']:
        dior_cls = CLASS_MAP[coco_cls]
        images, gts = load_coco_data(coco_cls)

        print(f"\n{'='*70}")
        print(f"  {coco_cls} ({len(images)} images, {sum(len(g) for g in gts)} GT)")
        print(f"{'='*70}")

        # ── 收集训练数据 ──
        print("  收集训练样本 (v2: hard negatives)...")
        pos_crops, hard_neg, easy_neg = collect_training_pairs_v2(
            images, gts, coco_cls, gd_proc, gd_model)
        all_neg = hard_neg + easy_neg
        print(f"  TP={len(pos_crops)}, HN={len(hard_neg)}, EN={len(easy_neg)}")

        # ── 提取特征 ──
        print("  提取特征...")
        # DINOv2 多层
        dino_cls = {l: [] for l in DINO_LAYERS}
        dino_patch = {l: [] for l in DINO_LAYERS}
        for c in pos_crops[:50]:
            cls_f, patch_f = extract_dinov2_multi(c, dino_model, dino_transform, DINO_LAYERS)
            for i, l in enumerate(DINO_LAYERS):
                dino_cls[l].append(cls_f[i])
                dino_patch[l].append(patch_f[i])

        clip_coco_feats = torch.stack([extract_clip_feat(c, clip_model, clip_preprocess)
                                        for c in pos_crops[:50]])

        # ── 多层 DINOv2 prototype (DIOR) ──
        dior_protos = {}
        for l in DINO_LAYERS:
            dior_protos[l] = bank[dior_cls][l]['prototype']  # L10/L11/L12

        # ── 多层 COCO prototype ──
        coco_protos_d = {}
        for l in DINO_LAYERS:
            feats = torch.stack(dino_cls[l][:5])
            coco_protos_d[l] = (feats.mean(dim=0) / feats.mean(dim=0).norm())

        # CLIP COCO proto
        coco_proto_c = clip_coco_feats[:5].mean(dim=0)
        coco_proto_c = coco_proto_c / coco_proto_c.norm()

        # ── Mahalanobis ──
        mah_d = {}
        for l in DINO_LAYERS:
            feats = torch.stack(dino_cls[l])
            if len(feats) >= 3:
                mah_d[l] = compute_mahalanobis_params(feats)
            else:
                mah_d[l] = None

        mah_c = compute_mahalanobis_params(clip_coco_feats) if len(clip_coco_feats) >= 3 else None

        # ── 训练投影头 v2 ──
        print("  训练投影头 v2...")

        # DINOv2 (L11, 最佳层)
        extract_d_l11 = lambda c: extract_dinov2_multi(c, dino_model, dino_transform, [10])[0][0]
        proj_d = ProjectionHeadV2(in_dim=768).to(DEVICE)
        pos_feats_d_l11 = torch.stack([extract_d_l11(c) for c in pos_crops[:50]])
        if len(pos_crops) >= 3 and len(all_neg) >= 3:
            proj_d, mah_proj_d, proto_proj_d = train_projection_head_v2(
                pos_crops, hard_neg, easy_neg, extract_d_l11, 768,
                proj_d, pos_feats_d_l11)
            proj_d = proj_d.eval()
        else:
            proj_d, mah_proj_d, proto_proj_d = None, None, None

        # CLIP
        extract_c = lambda c: extract_clip_feat(c, clip_model, clip_preprocess)
        proj_c = ProjectionHeadV2(in_dim=512).to(DEVICE)
        pos_feats_c = clip_coco_feats.clone()
        if len(pos_crops) >= 3 and len(all_neg) >= 3:
            proj_c, mah_proj_c, proto_proj_c = train_projection_head_v2(
                pos_crops, hard_neg, easy_neg, extract_c, 512,
                proj_c, pos_feats_c)
            proj_c = proj_c.eval()
        else:
            proj_c, mah_proj_c, proto_proj_c = None, None, None

        # ── 收集所有检测框的 vs 信号 ──
        print("  评估 (含温度缩放 + 多层集成)...")
        all_boxes, all_gd, all_gts = [], [], []
        vs_buffers = defaultdict(list)

        # 温度候选
        taus = [0.05, 0.07, 0.1, 0.15, 0.2, 0.3]

        for img, img_gts in zip(images, gts):
            prompt = f'a photo of {coco_cls}'
            inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
            inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            with torch.no_grad():
                o = gd_model(**inputs)
            ts = torch.tensor([img.size[::-1]]).to(DEVICE)
            r = gd_proc.post_process_grounded_object_detection(o, threshold=0.0, target_sizes=ts)[0]
            boxes = r['boxes'].cpu().numpy()
            scores = r['scores'].cpu().numpy()

            keep = scores > 0.03
            boxes, scores = boxes[keep], scores[keep]

            for bi, box in enumerate(boxes):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                gd = float(scores[bi])
                all_boxes.append(box.tolist())
                all_gd.append(gd)

                bad = x2 <= x1 + 10 or y2 <= y1 + 10

                if not bad:
                    crop = img.crop((x1, y1, x2, y2))
                    cls_f, patch_f = extract_dinov2_multi(crop, dino_model, dino_transform, DINO_LAYERS)
                    fc = extract_clip_feat(crop, clip_model, clip_preprocess)

                # ── 单层 cosine (L11, 原版 VEP) ──
                if not bad:
                    vs_buffers['cos_dino_l11_dior'].append(
                        max(0.0, float((cls_f[1].cpu() @ dior_protos[10].cpu()).item())))
                    vs_buffers['cos_dino_l11_coco'].append(
                        max(0.0, float((cls_f[1].cpu() @ coco_protos_d[10].cpu()).item())))
                else:
                    vs_buffers['cos_dino_l11_dior'].append(0.0)
                    vs_buffers['cos_dino_l11_coco'].append(0.0)

                # ── 多层加权集成: mean(L10_CLS, L11_CLS, L12_CLS) ──
                if not bad:
                    multi_feat = torch.stack(cls_f, dim=0).mean(dim=0)  # (768,)
                    multi_feat = multi_feat / multi_feat.norm()
                    multi_proto_dior = torch.stack([dior_protos[l] for l in DINO_LAYERS]).mean(dim=0)
                    multi_proto_dior = multi_proto_dior / multi_proto_dior.norm()
                    multi_proto_coco = torch.stack([coco_protos_d[l] for l in DINO_LAYERS]).mean(dim=0)
                    multi_proto_coco = multi_proto_coco / multi_proto_coco.norm()
                    vs_buffers['cos_dino_multi_dior'].append(
                        max(0.0, float((multi_feat.cpu() @ multi_proto_dior.cpu()).item())))
                    vs_buffers['cos_dino_multi_coco'].append(
                        max(0.0, float((multi_feat.cpu() @ multi_proto_coco.cpu()).item())))
                else:
                    vs_buffers['cos_dino_multi_dior'].append(0.0)
                    vs_buffers['cos_dino_multi_coco'].append(0.0)

                # ── CLS + patch 集成 ──
                if not bad:
                    cls_patch = (cls_f[1] + patch_f[1])
                    cls_patch = cls_patch / cls_patch.norm()
                    vs_buffers['cos_dino_cls_patch_dior'].append(
                        max(0.0, float((cls_patch.cpu() @ dior_protos[10].cpu()).item())))
                else:
                    vs_buffers['cos_dino_cls_patch_dior'].append(0.0)

                # ── 温度缩放 (L11 DIOR) ──
                if not bad:
                    for tau in taus:
                        vs_buffers[f'temp_dino_tau{tau:.2f}'].append(
                            float(tempered_cosine(cls_f[1], dior_protos[10].to(DEVICE), tau).cpu().item()))
                else:
                    for tau in taus:
                        vs_buffers[f'temp_dino_tau{tau:.2f}'].append(0.0)

                # ── CLIP ──
                if not bad:
                    vs_buffers['cos_clip'].append(
                        max(0.0, float((fc.cpu() @ coco_proto_c.cpu()).item())))
                else:
                    vs_buffers['cos_clip'].append(0.0)

                # ── DINOv2 投影头 ──
                if not bad and proj_d is not None:
                    fd_p = proj_d(cls_f[1].unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
                    vs_buffers['cos_dino_proj'].append(
                        max(0.0, float((fd_p @ proto_proj_d).item())))
                else:
                    vs_buffers['cos_dino_proj'].append(0.0)

                # ── CLIP 投影头 ──
                if not bad and proj_c is not None:
                    fc_p = proj_c(fc.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
                    vs_buffers['cos_clip_proj'].append(
                        max(0.0, float((fc_p @ proto_proj_c).item())))
                else:
                    vs_buffers['cos_clip_proj'].append(0.0)

                # ── DINOv2 Mahalanobis ──
                if not bad and mah_d[10] is not None:
                    mu, inv, _ = mah_d[10]
                    vs_buffers['mah_dino'].append(
                        float(mahalanobis_sim(cls_f[1], mu, inv).item()))
                else:
                    vs_buffers['mah_dino'].append(0.0)

                # ── 投影 Mahalanobis ──
                if not bad and mah_proj_d is not None and proj_d is not None:
                    fd_p2 = proj_d(cls_f[1].unsqueeze(0).to(DEVICE)).squeeze(0)
                    mu2, inv2, _ = mah_proj_d
                    vs_buffers['mah_dino_proj'].append(
                        float(mahalanobis_sim(fd_p2, mu2.to(DEVICE), inv2.to(DEVICE)).item()))
                else:
                    vs_buffers['mah_dino_proj'].append(0.0)

                # ── Late fusion ──
                if not bad:
                    vd = vs_buffers['cos_dino_multi_dior'][-1]
                    vc = vs_buffers['cos_clip'][-1]
                    vs_buffers['late_fusion'].append(vd * vc)
                else:
                    vs_buffers['late_fusion'].append(0.0)

                # ── Late fusion + proj ──
                if not bad and proj_d is not None and proj_c is not None:
                    vdp = vs_buffers['cos_dino_proj'][-1]
                    vcp = vs_buffers['cos_clip_proj'][-1]
                    vs_buffers['late_fusion_proj'].append(vdp * vcp)
                else:
                    vs_buffers['late_fusion_proj'].append(0.0)

            all_gts.extend(img_gts)

        # ── Baseline ──
        base_ap = compute_ap50(all_boxes, np.array(all_gd), all_gts)
        print(f"\n  Baseline: AP50 = {base_ap:.4f}")

        # ── 评估所有 vs 信号 ──
        vs_keys = list(vs_buffers.keys())
        print(f"  评估 {len(vs_keys)} 个 vs 信号...")

        # 先用默认融合 sqrt(gd * vs)
        default_results = {}
        for key in vs_keys:
            fused = [np.sqrt(g * max(v, 1e-8)) for g, v in zip(all_gd, vs_buffers[key])]
            default_results[key] = compute_ap50(all_boxes, np.array(fused), all_gts)

        # 再对每个关键信号搜索最优融合权重 α
        print("  网格搜索最优融合权重 α...")
        key_subset = [k for k in vs_keys if not k.startswith('temp_')]
        fusion_results = grid_search_fusion_weights(
            all_gd, vs_buffers, all_gts, all_boxes, key_subset)

        # ── 打印关键结果 ──
        print(f"\n  {'Signal':<30s} {'sqrt':>8s} {'Best α':>8s} {'Best AP':>8s} {'Δ-Base':>8s}")
        print(f"  {'-'*64}")

        key_signals = ['cos_dino_l11_dior', 'cos_dino_l11_coco', 'cos_dino_multi_dior',
                       'cos_dino_cls_patch_dior', 'cos_dino_proj', 'mah_dino', 'mah_dino_proj',
                       'cos_clip', 'cos_clip_proj', 'late_fusion', 'late_fusion_proj']
        best_overall_ap = base_ap
        best_overall_name = 'baseline'

        for key in key_signals:
            if key not in default_results:
                continue
            sqrt_ap = default_results[key]
            best_ap, best_alpha = fusion_results.get(key, (sqrt_ap, 0.5))
            delta = best_ap - base_ap if best_ap > sqrt_ap else sqrt_ap - base_ap
            best_val = max(sqrt_ap, best_ap)
            marker = 'α' if best_ap > sqrt_ap + 0.001 else '√'
            print(f"  {key:<30s} {sqrt_ap:8.4f} {best_alpha:8.3f} {best_val:8.4f} {best_val-base_ap:+8.4f} [{marker}]")
            if best_val > best_overall_ap:
                best_overall_ap = best_val
                best_overall_name = key

        # 温度缩放
        print(f"\n  温度缩放 (cos_dino_l11_dior):")
        for tau in taus:
            key = f'temp_dino_tau{tau:.2f}'
            if key in default_results:
                ap = default_results[key]
                print(f"    τ={tau:.3f}: AP50={ap:.4f} Δ={ap-base_ap:+.4f}")

        print(f"\n  🏆 {coco_cls} Best: {best_overall_name} = {best_overall_ap:.4f} (Δ={best_overall_ap-base_ap:+.4f})")

        all_results[coco_cls] = {
            'baseline': base_ap,
            'best': best_overall_ap,
            'best_name': best_overall_name,
            'default': default_results,
            'fusion': fusion_results,
        }

    # ──── 汇总 ────
    print(f"\n\n{'='*70}")
    print("CDVM v1 → v2 对比")
    print(f"{'='*70}")

    v1_data = {
        'airplane': {'base': 0.7590, 'best': 0.8082, 'name': 'cos_dino_dior'},
        'boat':     {'base': 0.2117, 'best': 0.2491, 'name': 'cos_clip_proj'},
        'car':      {'base': 0.1339, 'best': 0.2664, 'name': 'late_fusion'},
    }

    print(f"\n{'Class':<12s} {'v1 Base':>8s} {'v1 Best':>8s} {'v2 Best':>8s} "
          f"{'v1 Δ':>8s} {'v2 Δ':>8s} {'v2-v1':>8s} {'v2 方案'}")
    print("-" * 90)
    for c in ['airplane', 'boat', 'car']:
        v1b = v1_data[c]['base']
        v1best = v1_data[c]['best']
        v2best = all_results[c]['best']
        v2name = all_results[c]['best_name']
        print(f"{c:<12s} {v1b:8.4f} {v1best:8.4f} {v2best:8.4f} "
              f"{v1best-v1b:+8.4f} {v2best-v1b:+8.4f} {v2best-v1best:+8.4f} {v2name}")

    # 保存
    json_out = {}
    for k, v in all_results.items():
        entry = {'baseline': v['baseline'], 'best': v['best'], 'best_name': v['best_name']}
        json_out[k] = entry
    with open(SAVE_DIR / 'cdvm_v2_results.json', 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2)
    print(f"\n结果保存到 {SAVE_DIR / 'cdvm_v2_results.json'}")


if __name__ == '__main__':
    main()
