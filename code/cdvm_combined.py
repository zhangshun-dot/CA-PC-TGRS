# -*- coding: utf-8 -*-
"""CDVM: Cross-Domain Visual Matcher — VEP 改进版

三个改进点组合验证:
  1. [换函数] cosine → Mahalanobis 距离 (利用 K exemplar 协方差)
  2. [嵌入模块] 可训练投影头 (MLP: DINOv2 768d → 256d, 最大化 TP-FP margin)
  3. [换模型] DINOv2 → CLIP ViT 特征 (多模态对齐, 不同特征空间)

消融设计 (每个组合都测试):
  A. Baseline (GDINO only)                    — 下界
  B. cosine(DINOv2)                           — 原版 VEP, gd×vs 融合
  C. Mahalanobis(DINOv2)                      — 换函数, 无训练
  D. cosine(Proj(DINOv2))                     — 嵌入模块, train→cosine
  E. Mahalanobis(Proj(DINOv2))               — 函数+模块组合
  F. cosine(CLIP)                             — 换模型
  G. Mahalanobis(CLIP)                        — 模型+函数
  H. cosine(Proj(CLIP))                       — 模型+模块
  I. Mahalanobis(Proj(CLIP))                 — 三合一
  J. cosine(concat(DINOv2,CLIP))             — 双模型级联
  K. Late fusion: sqrt(gd × vs_dino × vs_clip) — 独立信号相乘
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
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from torchvision import transforms as T

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
N_RANDOM = 5
COV_SHRINKAGE = 0.3  # 协方差收缩系数
MARGIN = 0.3          # ranking loss margin
LR = 1e-3
EPOCHS = 50
PROJ_DIM = 256


# ══════════════════════════════════════════════════════
# 模型加载
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
    model = model.eval()
    return model, preprocess


def get_gdino():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    proc = AutoProcessor.from_pretrained(GDINO_CACHE)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_CACHE, dtype=torch.float32
    ).eval().to(DEVICE)
    return proc, model


# ══════════════════════════════════════════════════════
# 特征提取
# ══════════════════════════════════════════════════════

def extract_dinov2_feat(crop, dino_model, dino_transform, layer_idx=10):
    """提取 DINOv2 CLS token @ layer_idx (L11)"""
    img_t = dino_transform(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        o = dino_model(img_t)
    hs = o.hidden_states[layer_idx + 1]
    feat = hs[0, 0, :]  # CLS token, 768-d
    return feat / feat.norm(dim=-1, keepdim=True)


def extract_clip_feat(crop, clip_model, clip_preprocess):
    """提取 CLIP ViT image feature (512-d)"""
    img_t = clip_preprocess(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = clip_model.encode_image(img_t).float()  # CLIP outputs fp16
    return (feat / feat.norm(dim=-1, keepdim=True)).squeeze(0)


# ══════════════════════════════════════════════════════
# 可训练投影头
# ══════════════════════════════════════════════════════

class ProjectionHead(nn.Module):
    """将视觉特征投影到更适合 Mahalanobis 匹配的空间"""

    def __init__(self, in_dim, hidden=512, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        # x: (N, in_dim)
        h = self.net(x)
        return h / h.norm(dim=-1, keepdim=True)


# ══════════════════════════════════════════════════════
# Mahalanobis 距离 (带协方差收缩)
# ══════════════════════════════════════════════════════

def compute_mahalanobis_params(exemplar_feats, shrinkage=COV_SHRINKAGE):
    """从 K 个 exemplar 特征计算均值和收缩协方差。

    exemplar_feats: (K, D) normalized features
    Returns: mu (D,), inv_cov (D, D), log_det
    """
    K, D = exemplar_feats.shape
    mu = exemplar_feats.mean(dim=0)
    # 样本协方差
    centered = exemplar_feats - mu
    cov_sample = (centered.T @ centered) / (K - 1) if K > 1 else torch.eye(D, device=exemplar_feats.device) * 1e-3
    # 收缩: Σ_shrink = α·Σ_sample + (1-α)·I
    cov_shrink = shrinkage * cov_sample + (1 - shrinkage) * torch.eye(D, device=exemplar_feats.device)
    # 正则化求逆
    cov_shrink = cov_shrink + 1e-4 * torch.eye(D, device=exemplar_feats.device)
    try:
        L = torch.linalg.cholesky(cov_shrink)
        inv_cov = torch.cholesky_inverse(L)
        log_det = 2 * torch.sum(torch.log(torch.diag(L) + 1e-8))
    except RuntimeError:
        inv_cov = torch.eye(D, device=exemplar_feats.device)
        log_det = 0.0
    return mu, inv_cov, log_det


def mahalanobis_sim(feat, mu, inv_cov):
    """Mahalanobis 距离 → 相似度。

    feat: (N, D) or (D,)
    mu: (D,)
    inv_cov: (D, D)

    sim = exp(-0.5 * (x-μ)ᵀ Σ⁻¹ (x-μ))
    """
    diff = feat - mu.to(feat.device)
    if diff.dim() == 1:
        diff = diff.unsqueeze(0)
    # (N, D) @ (D, D) → (N, D), then (N, D) * (N, D) → sum → (N,)
    m = diff @ inv_cov.to(feat.device)
    dist_sq = (m * diff).sum(dim=1).clamp(min=0)
    return torch.exp(-0.5 * dist_sq).squeeze(0) if feat.dim() == 1 else torch.exp(-0.5 * dist_sq)


# ══════════════════════════════════════════════════════
# 数据加载
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


# ══════════════════════════════════════════════════════
# 训练数据: 收集 COCO GT crops (正) + GDINO 高分非GT crops (负)
# ══════════════════════════════════════════════════════

def collect_training_pairs(images, gts_per_image, class_name, gd_proc, gd_model):
    """Collect (crop, is_positive) pairs for training the projection head.

    Positive: GT bbox crops (IoU ≥ 0.5 with GT)
    Negative: GDINO检测框 IoU < 0.3 且 gd_score > 0.1 (hard negatives)
    """
    positives = []
    negatives = []

    for img, gts in zip(images, gts_per_image):
        w, h = img.size

        # Positives: GT crops
        for gt in gts:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            if x2 > x1 + 10 and y2 > y1 + 10:
                positives.append(img.crop((x1, y1, x2, y2)))

        # Negatives: high-score GDINO boxes not matching GT
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
        scores = scores[keep]

        for box in boxes:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            # Check IoU with all GTs
            max_iou = 0.0
            for gt in gts:
                iou = compute_iou([x1, y1, x2, y2], gt)
                if iou > max_iou:
                    max_iou = iou
            if max_iou < 0.3:  # hard negative
                negatives.append(img.crop((x1, y1, x2, y2)))

    return positives, negatives


# ══════════════════════════════════════════════════════
# 训练投影头
# ══════════════════════════════════════════════════════

def train_projection_head(pos_crops, neg_crops, extract_fn, in_dim, proj_head,
                           dior_exemplar_feats, mahalanobis_params):
    """训练投影头，使投影后 TP 的 Mahalanobis sim > FP。

    Loss = margin ranking: max(0, MARGIN - sim(TP) + sim(FP))
    + 正则化项保原有结构

    Mahalanobis 参数在投影空间中重新计算（不依赖预计算参数维度）。
    """
    if len(pos_crops) < 3 or len(neg_crops) < 3:
        return proj_head, None  # 数据不足，跳过训练

    optimizer = torch.optim.Adam(proj_head.parameters(), lr=LR)

    # 预提取所有特征 (DINOv2/CLIP 不更新)
    pos_feats_raw = torch.stack([extract_fn(c) for c in pos_crops])
    neg_feats_raw = torch.stack([extract_fn(c) for c in neg_crops])

    print(f"    训练投影头: TP={len(pos_feats_raw)}, FP={len(neg_feats_raw)}, epochs={EPOCHS}")

    n_pos = len(pos_feats_raw)
    n_neg = len(neg_feats_raw)

    for epoch in range(EPOCHS):
        # 随机采样 minibatch
        batch_pos_idx = torch.randint(0, n_pos, (min(32, n_pos),))
        batch_neg_idx = torch.randint(0, n_neg, (min(32, n_neg),))

        p_feat = pos_feats_raw[batch_pos_idx].to(DEVICE)
        n_feat = neg_feats_raw[batch_neg_idx].to(DEVICE)

        # 投影
        p_proj = proj_head(p_feat)
        n_proj = proj_head(n_feat)

        # 在投影空间中计算 Mahalanobis 参数 (用当前 batch 的 TP)
        mah_params_batch = compute_mahalanobis_params(p_proj.detach())
        mu_b, inv_b, _ = mah_params_batch

        # Mahalanobis similarity
        p_sim = mahalanobis_sim(p_proj, mu_b, inv_b)
        n_sim = mahalanobis_sim(n_proj, mu_b, inv_b)

        # Margin ranking loss
        loss = F.relu(MARGIN - p_sim.unsqueeze(1) + n_sim.unsqueeze(0)).mean()

        # 正则化: 投影后不应完全破坏原始结构
        p_cos = (p_proj @ p_proj.T).mean()
        loss = loss + 0.01 * (1 - p_cos)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            with torch.no_grad():
                p_sim_m = p_sim.mean().item()
                n_sim_m = n_sim.mean().item()
            print(f"      epoch {epoch:3d}: loss={loss.item():.4f}, "
                  f"TP_sim={p_sim_m:.4f}, FP_sim={n_sim_m:.4f}, margin={p_sim_m-n_sim_m:+.4f}")

    # 训练完成后, 在投影空间计算全局 Mahalanobis 参数
    with torch.no_grad():
        all_pos_proj = proj_head(pos_feats_raw.to(DEVICE)).cpu()
    mah_proj_params = compute_mahalanobis_params(all_pos_proj)

    return proj_head, mah_proj_params


# ══════════════════════════════════════════════════════
# 完整评估
# ══════════════════════════════════════════════════════

def evaluate_conditions(images, gts_per_image, class_name, gd_proc, gd_model,
                          dino_model, dino_transform, clip_model, clip_preprocess,
                          proto_dino_dior, proto_dino_coco, proto_clip_coco,
                          best_layer, mah_params_dict, proj_dict, proto_dict):
    """一次推理，评估所有 VEP 变体。

    proto_dino_dior: DIOR prototype (768-d)
    proto_dino_coco: COCO prototype (768-d)
    proto_clip_coco: COCO prototype (512-d)
    mah_params_dict: {key: (mu, inv_cov, log_det)}
    proj_dict: {key: ProjectionHead}
    """
    all_boxes, all_gd_scores, all_gts = [], [], []
    vs_buffers = defaultdict(list)

    for img, gts in zip(images, gts_per_image):
        prompt = f'a photo of {class_name}'
        inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
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
            all_gd_scores.append(gd)

            bad = x2 <= x1 + 10 or y2 <= y1 + 10
            if not bad:
                crop = img.crop((x1, y1, x2, y2))
                fd = extract_dinov2_feat(crop, dino_model, dino_transform, best_layer)  # 768-d
                fc = extract_clip_feat(crop, clip_model, clip_preprocess)  # 512-d

            # ── 每种条件独立计算 vs ──

            # DINOv2 + DIOR proto (跨域, 原版 VEP)
            if not bad:
                vs_buffers['cos_dino_dior'].append(max(0.0, float((fd.cpu() @ proto_dino_dior.cpu()).item())))
            else:
                vs_buffers['cos_dino_dior'].append(0.0)

            # DINOv2 + COCO proto (同域上界)
            if not bad and proto_dino_coco is not None:
                vs_buffers['cos_dino_coco'].append(max(0.0, float((fd.cpu() @ proto_dino_coco.cpu()).item())))
            else:
                vs_buffers['cos_dino_coco'].append(0.0)

            # DINOv2 Mahalanobis (同域, 用 COCO GT 计算)
            if not bad and 'mah_dino' in mah_params_dict:
                mu, inv, _ = mah_params_dict['mah_dino']
                vs_buffers['mah_dino'].append(float(mahalanobis_sim(fd, mu, inv).item()))
            else:
                vs_buffers['mah_dino'].append(0.0)

            # DINOv2 + Projection + cosine (用投影后的 COCO proto)
            if not bad and 'proj_dino' in proj_dict and 'proto_proj_dino' is not None:
                p_d = proj_dict['proj_dino']
                fd_p = p_d(fd.unsqueeze(0)).squeeze(0)
                proto_pd = proto_dict.get('proto_proj_dino')
                if proto_pd is not None:
                    vs_buffers['cos_dino_proj'].append(max(0.0, float((fd_p.cpu() @ proto_pd.cpu()).item())))
                else:
                    vs_buffers['cos_dino_proj'].append(0.0)
            else:
                vs_buffers['cos_dino_proj'].append(0.0)

            # DINOv2 + Projection + Mahalanobis
            if not bad and 'proj_dino' in proj_dict and 'mah_proj_dino' in mah_params_dict:
                p_d = proj_dict['proj_dino']
                fd_p = p_d(fd.unsqueeze(0)).squeeze(0)
                mu, inv, _ = mah_params_dict['mah_proj_dino']
                vs_buffers['mah_dino_proj'].append(float(mahalanobis_sim(fd_p, mu, inv).item()))
            else:
                vs_buffers['mah_dino_proj'].append(0.0)

            # CLIP + COCO proto (同域)
            if not bad and proto_clip_coco is not None:
                vs_buffers['cos_clip'].append(max(0.0, float((fc.cpu() @ proto_clip_coco.cpu()).item())))
            else:
                vs_buffers['cos_clip'].append(0.0)

            # CLIP Mahalanobis
            if not bad and 'mah_clip' in mah_params_dict:
                mu, inv, _ = mah_params_dict['mah_clip']
                vs_buffers['mah_clip'].append(float(mahalanobis_sim(fc.unsqueeze(0), mu, inv).item()))
            else:
                vs_buffers['mah_clip'].append(0.0)

            # CLIP + Projection + cosine
            if not bad and 'proj_clip' in proj_dict and 'proto_proj_clip' in proto_dict:
                p_c = proj_dict['proj_clip']
                fc_p = p_c(fc.unsqueeze(0)).squeeze(0)
                proto_pc = proto_dict['proto_proj_clip']
                vs_buffers['cos_clip_proj'].append(max(0.0, float((fc_p.cpu() @ proto_pc.cpu()).item())))
            else:
                vs_buffers['cos_clip_proj'].append(0.0)

            # CLIP + Projection + Mahalanobis
            if not bad and 'proj_clip' in proj_dict and 'mah_proj_clip' in mah_params_dict:
                p_c = proj_dict['proj_clip']
                fc_p = p_c(fc.unsqueeze(0)).squeeze(0)
                mu, inv, _ = mah_params_dict['mah_proj_clip']
                vs_buffers['mah_clip_proj'].append(float(mahalanobis_sim(fc_p, mu, inv).item()))
            else:
                vs_buffers['mah_clip_proj'].append(0.0)

            # Concat DINOv2+CLIP
            if not bad and proto_dino_coco is not None and proto_clip_coco is not None:
                f_concat = torch.cat([fd, fc])
                p_concat = torch.cat([proto_dino_coco, proto_clip_coco])
                f_concat = f_concat / f_concat.norm()
                p_concat = p_concat / p_concat.norm()
                vs_buffers['cos_concat'].append(max(0.0, float((f_concat.cpu() @ p_concat.cpu()).item())))
            else:
                vs_buffers['cos_concat'].append(0.0)

            # Dual late fusion: cos_dino * cos_clip
            vd = vs_buffers['cos_dino_coco'][-1] if vs_buffers['cos_dino_coco'] else 0.0
            vc = vs_buffers['cos_clip'][-1] if vs_buffers['cos_clip'] else 0.0
            vs_buffers['late_fusion'].append(vd * vc)

        all_gts.extend(gts)

    results = {}
    results['baseline'] = compute_ap50(all_boxes, np.array(all_gd_scores), all_gts)

    for key, vs_vals in vs_buffers.items():
        fused = [np.sqrt(g * max(v, 1e-8)) for g, v in zip(all_gd_scores, vs_vals)]
        results[key] = compute_ap50(all_boxes, np.array(fused), all_gts)

    return results


# ══════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("CDVM: Cross-Domain Visual Matcher — 组合消融")
    print("=" * 70)

    gd_proc, gd_model = get_gdino()
    dino_model, dino_transform = get_dinov2()
    clip_model, clip_preprocess = get_clip()
    bank = torch.load(BANK_PATH, map_location='cpu')

    CLASS_MAP = {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle'}
    BEST_LAYERS = {'airplane': 10, 'boat': 10, 'car': 11}

    all_results = {}

    for coco_cls in ['airplane', 'boat', 'car']:
        dior_cls = CLASS_MAP[coco_cls]
        best_layer = BEST_LAYERS[coco_cls]

        print(f"\n{'='*70}")
        print(f"  {coco_cls} (best_layer=L{best_layer+1})")
        print(f"{'='*70}")

        images, gts = load_coco_data(coco_cls)
        print(f"  数据: {len(images)} images, {sum(len(g) for g in gts)} GT boxes")

        # ── 收集训练数据 ──
        print("  收集训练样本...")
        pos_crops, neg_crops = collect_training_pairs(
            images, gts, coco_cls, gd_proc, gd_model)
        print(f"  TP crops: {len(pos_crops)}, FP crops: {len(neg_crops)}")

        # ── DIOR exemplar 特征 (用于 prototype & Mahalanobis) ──
        dior_proto_l11 = bank[dior_cls][10]['prototype']  # L11
        dior_proto_l12 = bank[dior_cls][11]['prototype']  # L12
        dior_proto = dior_proto_l11 if best_layer == 10 else dior_proto_l12

        # DIOR exemplar crops 的逐样本特征 (用于 Mahalanobis 协方差)
        # 注: bank 只保存了均值, 没有保存逐样本特征
        # 需要在 GDINO 上重新提取, 但 DIOR 数据集不在本地
        # 替代方案: 用 COCO GT crops 模拟, 或用近似协方差
        # 这里用 COCO GT crops 作为"同域 exemplar"计算 Mahalanobis 参数
        # 对于跨域场景, 用 L2-正则化的近似协方差

        # ── 提取特征 ──
        print("  提取特征...")
        extract_d = lambda c: extract_dinov2_feat(c, dino_model, dino_transform, best_layer)
        extract_c = lambda c: extract_clip_feat(c, clip_model, clip_preprocess)

        # COCO GT 特征 (用于构建同域原型 + Mahalanobis + 投影头训练)
        pos_feats_d = torch.stack([extract_d(c) for c in pos_crops[:50]]) if pos_crops else None
        pos_feats_c = torch.stack([extract_c(c) for c in pos_crops[:50]]) if pos_crops else None

        # ── 各域原型 ──
        # DIOR 跨域 (DINOv2 only — bank 不存在 CLIP)
        proto_dino_dior = dior_proto_l11 if best_layer == 10 else dior_proto_l12

        # COCO 同域 (DINOv2)
        proto_dino_coco = None
        if pos_feats_d is not None and len(pos_feats_d) >= 3:
            proto_dino_coco = pos_feats_d[:5].mean(dim=0)
            proto_dino_coco = proto_dino_coco / proto_dino_coco.norm()

        # COCO 同域 (CLIP)
        proto_clip_coco = None
        if pos_feats_c is not None and len(pos_feats_c) >= 3:
            proto_clip_coco = pos_feats_c[:5].mean(dim=0)
            proto_clip_coco = proto_clip_coco / proto_clip_coco.norm()

        # ── Mahalanobis 参数 ──
        mah_params_dict = {}
        if pos_feats_d is not None and len(pos_feats_d) >= 3:
            mah_params_dict['mah_dino'] = compute_mahalanobis_params(pos_feats_d)
        if pos_feats_c is not None and len(pos_feats_c) >= 3:
            mah_params_dict['mah_clip'] = compute_mahalanobis_params(pos_feats_c)

        # ── 训练投影头 ──
        proj_dict = {}
        proto_dict = {}  # 投影空间中的 prototype

        print("  训练 DINOv2 投影头...")
        proj_dino = ProjectionHead(in_dim=768).to(DEVICE)
        if pos_feats_d is not None and pos_feats_d.shape[0] >= 3 and neg_crops:
            proj_dino, mah_pd = train_projection_head(
                pos_crops, neg_crops, extract_d, 768,
                proj_dino, pos_feats_d, mah_params_dict.get('mah_dino'))
            proj_dino = proj_dino.eval()
            proj_dict['proj_dino'] = proj_dino
            if mah_pd is not None:
                mah_params_dict['mah_proj_dino'] = mah_pd
            # 投影后的 prototype
            with torch.no_grad():
                proto_dict['proto_proj_dino'] = proj_dino(pos_feats_d[:5].to(DEVICE)).mean(dim=0).cpu()
                proto_dict['proto_proj_dino'] = proto_dict['proto_proj_dino'] / proto_dict['proto_proj_dino'].norm()
        else:
            print("    训练数据不足, 跳过 DINOv2 投影头")

        print("  训练 CLIP 投影头...")
        proj_clip = ProjectionHead(in_dim=512).to(DEVICE)
        if pos_feats_c is not None and pos_feats_c.shape[0] >= 3 and neg_crops:
            proj_clip, mah_pc = train_projection_head(
                pos_crops, neg_crops, extract_c, 512,
                proj_clip, pos_feats_c, mah_params_dict.get('mah_clip'))
            proj_clip = proj_clip.eval()
            proj_dict['proj_clip'] = proj_clip
            if mah_pc is not None:
                mah_params_dict['mah_proj_clip'] = mah_pc
            with torch.no_grad():
                proto_dict['proto_proj_clip'] = proj_clip(pos_feats_c[:5].to(DEVICE)).mean(dim=0).cpu()
                proto_dict['proto_proj_clip'] = proto_dict['proto_proj_clip'] / proto_dict['proto_proj_clip'].norm()
        else:
            print("    训练数据不足, 跳过 CLIP 投影头")

        # ── 评估所有条件 ──
        print("  评估所有条件...")
        results = evaluate_conditions(
            images, gts, coco_cls, gd_proc, gd_model,
            dino_model, dino_transform, clip_model, clip_preprocess,
            proto_dino_dior, proto_dino_coco, proto_clip_coco,
            best_layer, mah_params_dict, proj_dict, proto_dict)

        all_results[coco_cls] = results

        # ── 打印 ──
        base = results['baseline']
        print(f"\n  {'Condition':<24s} {'AP50':>8s} {'Δ-Base':>8s} {'判定':>10s}")
        print(f"  {'-'*50}")
        # 按优先级排列
        key_order = ['baseline', 'cos_dino_dior', 'cos_dino_coco', 'mah_dino',
                     'cos_dino_proj', 'mah_dino_proj',
                     'cos_clip', 'mah_clip',
                     'cos_clip_proj', 'mah_clip_proj',
                     'cos_concat', 'late_fusion']
        for key in key_order:
            ap = results.get(key, 0)
            delta = ap - base
            tag = '✅+' if delta > 0.01 else ('≈' if abs(delta) <= 0.01 else '❌-')
            print(f"  {key:<24s} {ap:8.4f} {delta:+8.4f} {tag:>10s}")

        # 找出最佳
        best_key = max(results, key=lambda k: results[k])
        print(f"\n  🏆 Best: {best_key} = {results[best_key]:.4f} (Δ={results[best_key]-base:+.4f})")

    # ──── 汇总 ────
    print(f"\n{'='*70}")
    print("三合一消融汇总")
    print(f"{'='*70}")

    conditions = ['baseline', 'cos_dino_dior', 'cos_dino_coco', 'mah_dino',
                  'cos_dino_proj', 'mah_dino_proj',
                  'cos_clip', 'mah_clip',
                  'cos_clip_proj', 'mah_clip_proj',
                  'cos_concat', 'late_fusion']

    print(f"\n{'Condition':<24s}", end="")
    for c in ['airplane', 'boat', 'car']:
        print(f" {c:>10s}", end="")
    print(f" {'Avg':>10s} {'Avg Δ':>8s}")
    print("-" * 78)

    for key in conditions:
        print(f"  {key:<24s}", end="")
        vals = []
        base_vals = []
        for cls_name in ['airplane', 'boat', 'car']:
            ap = all_results[cls_name].get(key, 0)
            print(f" {ap:10.4f}", end="")
            vals.append(ap)
            base_vals.append(all_results[cls_name]['baseline'])
        avg_delta = np.mean([v - b for v, b in zip(vals, base_vals)])
        print(f" {np.mean(vals):10.4f} {avg_delta:+8.4f}")

    # 保存
    json_out = {}
    for k, v in all_results.items():
        json_out[k] = {k2: float(v2) for k2, v2 in v.items()}
    with open(SAVE_DIR / 'cdvm_ablation.json', 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2)
    print(f"\n结果保存到 {SAVE_DIR / 'cdvm_ablation.json'}")


if __name__ == '__main__':
    main()
