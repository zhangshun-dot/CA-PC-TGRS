# -*- coding: utf-8 -*-
"""VDA: Visual Domain Adapter — CLIP 视觉特征域适应

思路: 不同域的图像在 CLIP 特征空间中分布不同。
学习 FiLM adapter 对检测框的 CLIP 特征做域条件调制，提升跨域原型匹配。

    feat_raw = CLIP.encode_image(crop)           # 512-d, L2 normed
    feat_ada = FiLM(feat_raw, domain_prototype)  # 域条件调制
    sim = cos(feat_ada, class_prototype)          # 原型相似度

训练: DIOR exemplar crops → same-class prototype 应接近，cross-class 应远离
评估: COCO GDINO boxes → FiLM adapt → VEP rerank → AP50
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json, glob, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import pandas as pd

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')

# ── 路径配置 ──
CLIP_PATH = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
DIOR_PARQUET = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
DIOR_IMG_DIR = r'C:\Users\张三\modeling\dior_data\full20_images'
COCO_IMG_DIR = r'C:\Users\张三\modeling\coco_data\val2017'
COCO_ANN_FILE = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'
GDINO_CACHE = (r'C:\Users\张三\.cache\huggingface\hub'
               r'\models--IDEA-Research--grounding-dino-tiny'
               r'\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71')

# 跨域类别映射
DIOR_TO_COCO = {'airplane': 'airplane', 'ship': 'boat', 'vehicle': 'car'}
SHARED_CLASSES = ['airplane', 'ship', 'vehicle']
COCO_CLASSES = ['airplane', 'boat', 'car']

N_EXEMPLARS = 5
N_TRAIN_EPOCHS = 300
PRE_FILTER_THRESH = 0.03
DET_THRESHOLD = 0.25


# ══════════════════════════════════════════════════════
# FiLM Layer (复用 CDPS v2 的设计)
# ══════════════════════════════════════════════════════

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: γ ⊙ x + β. 初始化为 near-identity."""
    def __init__(self, in_dim, cond_dim, hidden=128):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim))
        self.beta_net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim))
        # 初始化为 near-identity: γ≈1, β≈0
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.zeros_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, x, cond):
        gamma = self.gamma_net(cond) + 1.0
        beta = self.beta_net(cond)
        return gamma * x + beta


class VisualDomainAdapter(nn.Module):
    """两阶段 FiLM: 域原型 → (γ₁,β₁,γ₂,β₂) → 调制 CLIP 视觉特征。"""
    def __init__(self, d_model=512, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.film1 = FiLMLayer(hidden, d_model, hidden=128)
        self.fc2 = nn.Linear(hidden, hidden)
        self.film2 = FiLMLayer(hidden, d_model, hidden=128)
        self.fc_out = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, domain_proto):
        """
        Args:
            x: (B, 512) CLIP image features (already L2-normed)
            domain_proto: (B, 512) or (512,) domain prototype vector
        Returns:
            (B, 512) adapted features (not L2-normed by default)
        """
        if domain_proto.dim() == 1:
            domain_proto = domain_proto.unsqueeze(0).expand(x.size(0), -1)

        h = F.relu(self.fc1(x))
        h = self.film1(h, domain_proto)
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.film2(h, domain_proto)
        out = self.fc_out(h)
        return out


# ══════════════════════════════════════════════════════
# 数据准备
# ══════════════════════════════════════════════════════

def get_clip():
    import clip
    model, preprocess = clip.load(CLIP_PATH, device=DEVICE)
    model.eval()
    return model, preprocess


def compute_domain_prototype(image_paths, clip_model, preprocess, n_max=20):
    """CLIP image mean → domain prototype (512-d, L2-normed)."""
    features = []
    for p in image_paths[:n_max]:
        try:
            img = preprocess(Image.open(p).convert('RGB')).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                f = clip_model.encode_image(img).float()
                f = f / f.norm(dim=-1, keepdim=True)
            features.append(f.squeeze(0))
        except Exception:
            continue
    if not features:
        return torch.zeros(512)
    proto = torch.stack(features).mean(dim=0)
    return proto / proto.norm(dim=-1, keepdim=True)


def extract_exemplar_features(clip_model, preprocess):
    """从 DIOR GT crops 提取 per-class exemplar features (CLIP 512-d)。

    Returns:
        exemplars: {class_name: Tensor[K, 512]}  (L2-normed crops)
        prototypes: {class_name: Tensor[512]}    (L2-normed mean)
    """
    df = pd.read_parquet(DIOR_PARQUET)

    DIOR_ID2NAME = {
        0: 'airplane', 1: 'airport', 2: 'baseball field', 3: 'basketball court',
        4: 'bridge', 5: 'chimney', 6: 'dam', 7: 'expressway service area',
        8: 'expressway toll station', 9: 'golf course', 10: 'ground track field',
        11: 'harbor', 12: 'overpass', 13: 'ship', 14: 'stadium',
        15: 'storage tank', 16: 'tennis court', 17: 'train station',
        18: 'vehicle', 19: 'windmill',
    }
    NAME2ID = {v: k for k, v in DIOR_ID2NAME.items()}

    # 收集所有 GT crops 的特征
    class_features = defaultdict(list)

    for _, row in df.iterrows():
        img_id = int(row['image_id'])
        # 找图像文件
        img_path = None
        for pattern in [os.path.join(DIOR_IMG_DIR, f'*_{img_id}.jpg'),
                        os.path.join(DIOR_IMG_DIR, f'*_{img_id}.JPG')]:
            matches = glob.glob(pattern)
            if matches:
                img_path = matches[0]
                break
        if img_path is None:
            continue

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            continue

        for i, cat in enumerate(row['objects']['category']):
            cls_name = DIOR_ID2NAME.get(int(cat))
            if cls_name not in SHARED_CLASSES:
                continue

            x, y, w, h = [int(v) for v in row['objects']['bbox'][i]]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img.width, x + w), min(img.height, y + h)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue

            crop = img.crop((x1, y1, x2, y2))
            img_t = preprocess(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                f = clip_model.encode_image(img_t).float()
                f = f / f.norm(dim=-1, keepdim=True)
            class_features[cls_name].append(f.squeeze(0).cpu())

    # 每类取最多 20 个 exemplar，用 maximin 选 K 个
    exemplars, prototypes = {}, {}
    for cls_name in SHARED_CLASSES:
        feats = torch.stack(class_features[cls_name][:20])  # (N, 512)
        # Maximin 选择 K 个
        n = len(feats)
        k = min(N_EXEMPLARS, n)
        if k == 0:
            continue

        # 第一个: 最接近均值的
        mean_feat = feats.mean(dim=0)
        sims = (feats @ mean_feat)
        selected_idx = [int(sims.argmax())]

        for _ in range(k - 1):
            # 对每个未选中的，计算它与已选中之间的最小距离
            min_dists = []
            for i in range(n):
                if i in selected_idx:
                    min_dists.append(-1e9)
                    continue
                # 与每个已选中的距离
                dists = [1 - (feats[i] @ feats[j]).item() for j in selected_idx]
                min_dists.append(min(dists))
            best = int(np.argmax(min_dists))
            selected_idx.append(best)

        selected = feats[selected_idx]
        exemplars[cls_name] = selected  # (K, 512)
        prototypes[cls_name] = selected.mean(dim=0)
        prototypes[cls_name] = prototypes[cls_name] / prototypes[cls_name].norm()

    return exemplars, prototypes


def get_domain_images():
    """获取 DIOR (遥感) 和 COCO (自然) 的代表图像路径。"""
    rs_images = []
    for cls_name in SHARED_CLASSES[:2]:  # airplane, ship
        pattern = os.path.join(DIOR_IMG_DIR, f'{cls_name}_*.jpg')
        rs_images.extend(glob.glob(pattern)[:3])

    ns_images = []
    coco_files = sorted(glob.glob(os.path.join(COCO_IMG_DIR, '*.jpg')))[:20]
    ns_images = coco_files

    return rs_images, ns_images


# ══════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════

def train_vda(adapter, exemplars, prototypes, domain_protos, clip_model, preprocess,
              epochs=N_TRAIN_EPOCHS, lr=1e-3, margin=0.3, verbose=True):
    """训练 Visual Domain Adapter。

    对比损失:
      - 正样本: 同类 DIOR exemplar → adapter → 应接近同名 prototype
      - 负样本: 不同类 exemplar → 应远离
    同时保留 identity: ||adapter(f) - f|| 不能太大
    """
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # 准备训练对: 每个 exemplar 标记其类别
    all_feats = []  # list of (feat, cls_name, domain)
    for cls_name in SHARED_CLASSES:
        if cls_name not in exemplars:
            continue
        for i in range(len(exemplars[cls_name])):
            all_feats.append((exemplars[cls_name][i], cls_name, 'remote_sensing'))

    rs_proto = domain_protos['remote_sensing'].to(DEVICE)
    proto_dict = {k: v.to(DEVICE) for k, v in prototypes.items()}

    n = len(all_feats)
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0.0
        n_pos, n_neg_correct = 0, 0

        # 随机 shuffle
        import random
        random.shuffle(all_feats)

        for feat, cls_name, domain in all_feats:
            x = feat.unsqueeze(0).to(DEVICE)  # (1, 512)
            x_adapted = adapter(x, rs_proto)
            x_adapted = x_adapted / x_adapted.norm(dim=-1, keepdim=True)

            # 正样本: 同类 prototype
            pos_proto = proto_dict[cls_name].unsqueeze(0)  # (1, 512)
            pos_sim = (x_adapted @ pos_proto.T).squeeze()

            # 负样本: 随机选一个不同的类
            neg_cls_list = [c for c in SHARED_CLASSES if c != cls_name and c in proto_dict]
            if not neg_cls_list:
                continue
            neg_cls = random.choice(neg_cls_list)
            neg_proto = proto_dict[neg_cls].unsqueeze(0)
            neg_sim = (x_adapted @ neg_proto.T).squeeze()

            # 对比损失: pos_sim 应 > neg_sim + margin
            loss = F.relu(margin + neg_sim - pos_sim)

            # Identity 正则: 适应后的特征不要偏离原始太远
            x_normed = x / x.norm(dim=-1, keepdim=True)
            id_loss = 0.1 * (1 - (x_adapted @ x_normed.T).squeeze())

            loss = loss + id_loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if pos_sim > neg_sim:
                n_neg_correct += 1
            n_pos += 1

        scheduler.step()
        avg_loss = total_loss / max(n_pos, 1)
        rank_acc = n_neg_correct / max(n_pos, 1)

        if epoch % 50 == 0 or epoch == epochs - 1:
            if verbose:
                print(f"  E{epoch:3d}: loss={avg_loss:.4f} rank_acc={rank_acc:.3f} "
                      f"lr={scheduler.get_last_lr()[0]:.6f}")

    return adapter


# ══════════════════════════════════════════════════════
# GDINO + AP50 工具 (复用 cdps_e2e_validate 模式)
# ══════════════════════════════════════════════════════

def get_gdino():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    proc = AutoProcessor.from_pretrained(GDINO_CACHE)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_CACHE, dtype=torch.float32
    ).eval().to(DEVICE)
    return proc, model


def load_coco_eval_data(class_name, n=30):
    from pycocotools.coco import COCO
    coco = COCO(COCO_ANN_FILE)
    cat_ids = coco.getCatIds(catNms=[class_name])
    if not cat_ids:
        return [], []
    cat_id = cat_ids[0]
    img_ids = coco.getImgIds(catIds=[cat_id])
    images, gt_boxes_all = [], []
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
        gt_boxes_all.append(gts)
        if len(images) >= n:
            break
    return images, gt_boxes_all


def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-8)


def compute_ap50(pred_boxes, pred_scores, gt_boxes):
    if len(gt_boxes) == 0:
        return 0.0 if len(pred_boxes) > 0 else 1.0
    if len(pred_boxes) == 0:
        return 0.0
    idx = np.argsort(pred_scores)[::-1]
    pb_sorted = [pred_boxes[i] for i in idx]
    gt_matched = [False] * len(gt_boxes)
    tp, fp = np.zeros(len(pb_sorted)), np.zeros(len(pb_sorted))
    for i, pb in enumerate(pb_sorted):
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if not gt_matched[j]:
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
        if best_iou >= 0.5 and best_j >= 0:
            tp[i] = 1
            gt_matched[best_j] = True
        else:
            fp[i] = 1
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recalls = tp_cum / len(gt_boxes)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            ap += float(np.max(precisions[mask]))
    return ap / 11.0


# ══════════════════════════════════════════════════════
# VDA-VEP 评估
# ══════════════════════════════════════════════════════

def evaluate_vda_vep(images, coco_class, gd_proc, gd_model,
                     adapter, clip_model, preprocess,
                     prototypes_clip, domain_protos, domain='natural_scene'):
    """
    用 VDA 适配后的 CLIP 特征做 VEP 重打分。

    Returns: AP50
    """
    # GDINO 检测
    prompt = f'a photo of {coco_class}'
    v_dom = domain_protos[domain].to(DEVICE)
    proto = prototypes_clip.get(coco_class)
    if proto is None:
        # 跨域映射
        dior_cls = {v: k for k, v in DIOR_TO_COCO.items()}.get(coco_class, coco_class)
        proto = prototypes_clip.get(dior_cls)
    if proto is None:
        return 0.0
    proto = proto.to(DEVICE)

    all_pred_boxes, all_pred_scores = [], []

    for i, img in enumerate(images):
        inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            o = gd_model(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(DEVICE)
        r = gd_proc.post_process_grounded_object_detection(o, threshold=0.0, target_sizes=ts)[0]
        boxes = r['boxes'].cpu().numpy()
        scores = r['scores'].cpu().numpy()

        keep = scores > PRE_FILTER_THRESH
        boxes_f = boxes[keep]
        scores_f = scores[keep]

        if len(boxes_f) == 0:
            continue

        visual_sims = []
        for box in boxes_f:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                visual_sims.append(0.0)
                continue

            crop = img.crop((x1, y1, x2, y2))
            img_t = preprocess(crop).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                f = clip_model.encode_image(img_t).float()
                f = f / f.norm(dim=-1, keepdim=True)
                if adapter is not None:
                    f_ada = adapter(f, v_dom)
                    f_ada = f_ada / f_ada.norm(dim=-1, keepdim=True)
                    sim = max(0.0, float((f_ada @ proto).item()))
                else:
                    sim = max(0.0, float((f @ proto).item()))
            visual_sims.append(sim)

        gd = np.clip(scores_f, 1e-6, 1.0)
        vs = np.clip(np.array(visual_sims), 1e-6, 1.0)
        fused = np.sqrt(gd * vs)

        all_pred_boxes.extend(boxes_f.tolist())
        all_pred_scores.extend(fused.tolist())

    # 收集 GT (flattened)
    # GT 从外部传入会更好，但这里简化处理
    return all_pred_boxes, all_pred_scores


# ══════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("VDA: Visual Domain Adapter — CLIP 特征域适应")
    print("=" * 65)

    # ── 加载模型 ──
    print("\n[1] 加载 CLIP + GDINO...")
    clip_model, preprocess = get_clip()
    gd_proc, gd_model = get_gdino()

    # ── 构建域原型 ──
    print("\n[2] 构建域原型...")
    rs_images, ns_images = get_domain_images()
    rs_proto = compute_domain_prototype(rs_images, clip_model, preprocess)
    ns_proto = compute_domain_prototype(ns_images, clip_model, preprocess)
    domain_protos = {'remote_sensing': rs_proto, 'natural_scene': ns_proto}
    print(f"  remote_sensing: |v|={rs_proto.norm():.3f}, {len(rs_images)} images")
    print(f"  natural_scene:  |v|={ns_proto.norm():.3f}, {len(ns_images)} images")
    print(f"  cos(rs, ns) = {(rs_proto @ ns_proto).item():.4f}")

    # ── 提取 exemplar features ──
    print("\n[3] 提取 DIOR exemplar features...")
    exemplars, prototypes_dior = extract_exemplar_features(clip_model, preprocess)
    for cls_name in SHARED_CLASSES:
        if cls_name in exemplars:
            print(f"  {cls_name}: {len(exemplars[cls_name])} exemplars")

    # COCO prototype: 用 DIOR 的跨域映射
    prototypes = {}
    for dior_cls, coco_cls in DIOR_TO_COCO.items():
        if dior_cls in prototypes_dior:
            prototypes[coco_cls] = prototypes_dior[dior_cls]

    # ── 训练 VDA ──
    print("\n[4] 训练 Visual Domain Adapter...")
    adapter = VisualDomainAdapter(d_model=512, hidden=256).to(DEVICE)
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"  参数量: {n_params:,}")

    adapter = train_vda(
        adapter, exemplars, prototypes_dior, domain_protos,
        clip_model, preprocess, epochs=N_TRAIN_EPOCHS, lr=1e-3
    )

    # ── 保存 ──
    torch.save({
        'adapter_state': adapter.state_dict(),
        'domain_protos': {k: v.cpu() for k, v in domain_protos.items()},
        'prototypes': {k: v.cpu() for k, v in prototypes.items()},
    }, SAVE_DIR / 'vda_weights.pt')

    # ── 评估: COCO 跨域 VEP ──
    print("\n[5] COCO 跨域 VDA-VEP 评估...")
    all_results = {}

    for coco_cls in COCO_CLASSES:
        print(f"\n  --- {coco_cls} ---")
        images, gts = load_coco_eval_data(coco_cls, n=30)
        if len(images) < 5:
            print(f"    跳过 (太少图像)")
            continue
        gts_flat = []
        for g in gts:
            gts_flat.extend(g)

        # A. Baseline: 纯 GDINO
        prompt = f'a photo of {coco_cls}'
        all_boxes, all_scores = [], []
        for img in images:
            inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
            inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            with torch.no_grad():
                o = gd_model(**inputs)
            ts = torch.tensor([img.size[::-1]]).to(DEVICE)
            r = gd_proc.post_process_grounded_object_detection(o, threshold=DET_THRESHOLD, target_sizes=ts)[0]
            all_boxes.extend(r['boxes'].cpu().numpy().tolist())
            all_scores.extend(r['scores'].cpu().numpy().tolist())
        ap_base = compute_ap50(all_boxes, np.array(all_scores), gts_flat)

        # B. VEP (no adapter): CLIP prototype 直接重打分
        pred_b, scores_b = evaluate_vda_vep(
            images, coco_cls, gd_proc, gd_model,
            None, clip_model, preprocess, prototypes, domain_protos
        )
        ap_vep = compute_ap50(pred_b, np.array(scores_b), gts_flat)

        # C. VDA-VEP: FiLM 适配后重打分
        pred_c, scores_c = evaluate_vda_vep(
            images, coco_cls, gd_proc, gd_model,
            adapter, clip_model, preprocess, prototypes, domain_protos
        )
        ap_vda = compute_ap50(pred_c, np.array(scores_c), gts_flat)

        print(f"    Baseline:  {ap_base:.4f}")
        print(f"    VEP:       {ap_vep:.4f}  (Δ={ap_vep - ap_base:+.4f})")
        print(f"    VDA-VEP:   {ap_vda:.4f}  (Δ={ap_vda - ap_base:+.4f})")

        all_results[coco_cls] = {
            'baseline': float(ap_base),
            'vep': float(ap_vep),
            'vda_vep': float(ap_vda),
            'vda_delta': float(ap_vda - ap_base),
            'vep_delta': float(ap_vep - ap_base),
        }

    # ── 汇总 ──
    print(f"\n{'='*65}")
    print("VDA 汇总")
    print(f"{'='*65}")
    print(f"{'Class':<12s} {'Baseline':>8s} {'VEP':>8s} {'VDA-VEP':>8s} {'VEPΔ':>7s} {'VDAΔ':>7s}")
    print(f"{'-'*52}")

    for cn in COCO_CLASSES:
        if cn in all_results:
            r = all_results[cn]
            print(f"{cn:<12s} {r['baseline']:8.4f} {r['vep']:8.4f} {r['vda_vep']:8.4f} "
                  f"{r['vep_delta']:+7.4f} {r['vda_delta']:+7.4f}")

    with open(SAVE_DIR / 'vda_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果: {SAVE_DIR / 'vda_results.json'}")
    print("✅ VDA 完成")


if __name__ == '__main__':
    main()
