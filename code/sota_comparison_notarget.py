# -*- coding: utf-8 -*-
r"""sota_comparison_notarget.py — SOTA对比实验（无目标域数据版本）
DANN/CORAL完全不使用目标域任何数据（包括无标签数据），信息预算与A-CSPC目标域无关配置一致。

DANN无目标域版：只用源域数据（DIOR GT框特征+类别标签）训练有监督MLP分类器，用中间层特征重排序
CORAL无目标域版：只用源域协方差矩阵做白化（不做目标域着色），用白化后特征重排序

对比方法：
  - Baseline: GDINO-tiny原始分数
  - B2源原型: 源域CLIP原型重排序（无目标域数据）
  - A-CSPC目标域无关: (β=1.0, λ=0.25, DIOR→VOC开发域选参)
  - DANN无目标域: 源域有监督特征提取
  - CORAL无目标域: 源域特征白化
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
PRE_FILTER = 0.03
K_CALIB = 10

TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]

# ══════════════════════════ 基础工具 ══════════════════════════

def load_coco_all(cls_name):
    return U.load_coco(cls_name, n=100000)

def clip_proto(crops):
    if not crops:
        return None
    F = T.encode_clip_batch(crops)
    p = F.mean(dim=0)
    return p / p.norm()

def gt_crops(images, gts):
    crops = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
    return crops

def cached_dets(images, prompt, name):
    p = DATA / f'tge_v2_dets_COCO-{name}.pkl'
    if p.exists():
        return pickle.load(open(p, 'rb'))
    dets = U.gdino_detect(images, prompt, 0.0)
    pickle.dump(dets, open(p, 'wb'))
    return dets

def collect_features(images, gts, dets):
    per_img = []
    crops_all = []
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < PRE_FILTER:
                continue
            bl.append([x1, y1, x2, y2]); sl.append(float(sc)); ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
                        'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
                        'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
                        'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64)})
    if crops_all:
        F = T.encode_clip_batch(crops_all).cpu().numpy()
    else:
        F = np.zeros((0, 512))
    return per_img, F

def dataset_ap(per_img, key):
    sel = [per_img[i] for i in range(K_CALIB, len(per_img))]
    preds = [(p['box'], p[key]) for p in sel]
    gts = [p['gt'] for p in sel]
    return U.compute_ap50_img(preds, gts)

# ══════════════════════════ DANN无目标域版 ══════════════════════════

class SupervisedFeatureExtractor(nn.Module):
    """有监督特征提取器：用源域类别标签训练MLP分类器，中间层作为特征"""
    def __init__(self, input_dim=512, hidden_dim=512, n_classes=20):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)
    def forward(self, x):
        feat = self.feature(x)
        logits = self.classifier(feat)
        return feat, logits

def train_supervised_extractor(src_features, src_labels, n_classes=20, epochs=100, lr=1e-3):
    """只用源域数据训练有监督特征提取器"""
    model = SupervisedFeatureExtractor(input_dim=src_features.shape[1], n_classes=n_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    src_tensor = torch.FloatTensor(src_features).to(DEVICE)
    label_tensor = torch.LongTensor(src_labels).to(DEVICE)

    for epoch in range(epochs):
        model.train()
        _, logits = model(src_tensor)
        loss = criterion(logits, label_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model

def dann_notarget_rerank(per_img, F, text_proto, src_features, src_labels, n_classes=20):
    """DANN无目标域版：源域有监督特征提取"""
    model = train_supervised_extractor(src_features, src_labels, n_classes)
    with torch.no_grad():
        F_tensor = torch.FloatTensor(F).to(DEVICE)
        feat, _ = model(F_tensor)
        feat = feat.cpu().numpy()
        norms = np.linalg.norm(feat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        feat_norm = feat / norms

    text_proto_np = text_proto.cpu().numpy() if torch.is_tensor(text_proto) else text_proto
    for p in per_img:
        n = len(p['gd'])
        if n:
            vis = np.maximum(0.0, feat_norm[p['crop_idx']] @ text_proto_np)
            p['dann_notarget'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p['dann_notarget'] = np.zeros(0)
    return per_img

# ══════════════════════════ CORAL无目标域版 ══════════════════════════

def coral_whiten(src_features):
    """只用源域协方差矩阵做白化（不做目标域着色）"""
    src_cov = np.cov(src_features.T) + 1e-3 * np.eye(src_features.shape[1])
    eigvals, eigvecs = np.linalg.eigh(src_cov)
    eigvals = np.clip(eigvals, 1e-3, None)
    # 只白化，不着色
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return W

def coral_notarget_rerank(per_img, F, text_proto, src_features):
    """CORAL无目标域版：源域特征白化"""
    W = coral_whiten(src_features)
    F_whitened = F @ W.T
    norms = np.linalg.norm(F_whitened, axis=1, keepdims=True)
    norms[norms == 0] = 1
    F_norm = F_whitened / norms

    text_proto_np = text_proto.cpu().numpy() if torch.is_tensor(text_proto) else text_proto
    for p in per_img:
        n = len(p['gd'])
        if n:
            vis = np.maximum(0.0, F_norm[p['crop_idx']] @ text_proto_np)
            p['coral_notarget'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p['coral_notarget'] = np.zeros(0)
    return per_img

# ══════════════════════════ 主流程 ══════════════════════════

def main():
    print('=' * 70)
    print('SOTA对比实验（无目标域数据版本）')
    print('=' * 70)

    # 加载DIOR源域原型和特征（带类别标签）
    print('[1/4] 加载DIOR源域原型和特征...')
    protos = {}
    src_all_features = []
    src_all_labels = []
    class_names = list(U.DIOR_ID2NAME.values())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for cls in class_names:
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p
        crops = gt_crops(images, gts)
        if crops:
            F_src = T.encode_clip_batch(crops).cpu().numpy()
            src_all_features.append(F_src)
            src_all_labels.extend([class_to_idx[cls]] * len(F_src))

    src_all_features = np.concatenate(src_all_features, axis=0)
    src_all_labels = np.array(src_all_labels)
    print(f'  源域特征总数: {len(src_all_features)}, 类别数: {len(class_names)}')

    # 对每个目标类运行对比
    print('[2/4] 逐类运行5种方法对比...')
    results = {}
    for name, coco_cls, src_cls in TARGETS:
        print(f'\n  --- {name} ---')
        images, gts = load_coco_all(coco_cls)
        if len(images) < K_CALIB + 10:
            print(f'    图太少 ({len(images)}), 跳过')
            continue

        dets = cached_dets(images, TGT_TMPL.format(coco_cls), name)
        per_img, F = collect_features(images, gts, dets)

        # 原型（统一到cpu）
        P_src = protos[src_cls].cpu()
        P_neg = T.get_neg_proto(protos, src_cls).cpu()
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        Delta = d_y - s_xs

        # 方法1: Baseline
        for p in per_img:
            p['baseline'] = p['gd'].copy()

        # 方法2: B2源原型
        P_src_np = P_src.numpy()
        for p in per_img:
            n = len(p['gd'])
            if n:
                vis = np.maximum(0.0, F[p['crop_idx']] @ P_src_np)
                p['b2'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
            else:
                p['b2'] = np.zeros(0)

        # 方法3: A-CSPC目标域无关 (β=1.0, λ=0.25)
        beta, lam = 1.0, 0.25
        P_hat = P_src + beta * Delta
        P_hat = P_hat / P_hat.norm()
        P_hat_np = P_hat.numpy()
        P_neg_np = P_neg.numpy()
        for p in per_img:
            n = len(p['gd'])
            if n:
                pos_sim = F[p['crop_idx']] @ P_hat_np
                neg_sim = F[p['crop_idx']] @ P_neg_np
                vis = np.maximum(0.0, pos_sim - lam * neg_sim)
                p['acspc_notarget'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
            else:
                p['acspc_notarget'] = np.zeros(0)

        # 方法4: DANN无目标域
        per_img = dann_notarget_rerank(per_img, F, d_y, src_all_features, src_all_labels, len(class_names))

        # 方法5: CORAL无目标域
        per_img = coral_notarget_rerank(per_img, F, d_y, src_all_features)

        # 计算AP
        ap_baseline = dataset_ap(per_img, 'baseline')
        ap_b2 = dataset_ap(per_img, 'b2')
        ap_acspc = dataset_ap(per_img, 'acspc_notarget')
        ap_dann = dataset_ap(per_img, 'dann_notarget')
        ap_coral = dataset_ap(per_img, 'coral_notarget')

        results[name] = {
            'baseline': ap_baseline,
            'b2': ap_b2, 'b2_gain': ap_b2 - ap_baseline,
            'acspc_notarget': ap_acspc, 'acspc_notarget_gain': ap_acspc - ap_baseline,
            'dann_notarget': ap_dann, 'dann_notarget_gain': ap_dann - ap_baseline,
            'coral_notarget': ap_coral, 'coral_notarget_gain': ap_coral - ap_baseline,
        }
        print(f'    Baseline:          {ap_baseline:.4f}')
        print(f'    B2源原型:          {ap_b2:.4f} (增益 {ap_b2-ap_baseline:+.4f})')
        print(f'    A-CSPC目标域无关:  {ap_acspc:.4f} (增益 {ap_acspc-ap_baseline:+.4f})')
        print(f'    DANN无目标域:      {ap_dann:.4f} (增益 {ap_dann-ap_baseline:+.4f})')
        print(f'    CORAL无目标域:     {ap_coral:.4f} (增益 {ap_coral-ap_baseline:+.4f})')

    # 汇总
    print('\n' + '=' * 70)
    print('汇总（6类平均，无目标域数据）')
    print('=' * 70)
    methods = ['baseline', 'b2', 'acspc_notarget', 'dann_notarget', 'coral_notarget']
    avg = {}
    for m in methods:
        vals = [results[name][m] for name in results]
        avg[m] = np.mean(vals)
    print(f'  Baseline:          {avg["baseline"]:.4f}')
    print(f'  B2源原型:          {avg["b2"]:.4f} (增益 {avg["b2"]-avg["baseline"]:+.4f})')
    print(f'  A-CSPC目标域无关:  {avg["acspc_notarget"]:.4f} (增益 {avg["acspc_notarget"]-avg["baseline"]:+.4f})')
    print(f'  DANN无目标域:      {avg["dann_notarget"]:.4f} (增益 {avg["dann_notarget"]-avg["baseline"]:+.4f})')
    print(f'  CORAL无目标域:     {avg["coral_notarget"]:.4f} (增益 {avg["coral_notarget"]-avg["baseline"]:+.4f})')

    # 保存
    out = {'per_class': results, 'average': avg}
    out_path = DATA / 'sota_comparison_notarget.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {out_path}')

if __name__ == '__main__':
    main()
