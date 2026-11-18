# -*- coding: utf-8 -*-
r"""sota_dann_rerun.py — DANN补数据重跑（目标域从10张提到200张）
原DANN只用COCO前K_CALIB=10张校准图的候选框特征训练，可能欠训练/过拟合。
本脚本用前200张图的候选框特征训练DANN，其余协议不变。
同时对比：10张 / 50张 / 100张 / 200张 四个规模，观察DANN性能随训练数据量的变化。
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

DANN_TRAIN_SIZES = [10, 50, 100, 200]

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

# ══════════════════════════ DANN ══════════════════════════

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DANNFeatureExtractor(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )
    def forward(self, x, alpha=1.0):
        feat = self.feature(x)
        reverse_feat = GradientReversal.apply(feat, alpha)
        domain_pred = self.domain_classifier(reverse_feat)
        return feat, domain_pred

def train_dann(src_features, tgt_features, epochs=100, lr=1e-3, batch_size=256):
    model = DANNFeatureExtractor(input_dim=src_features.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    src_tensor = torch.FloatTensor(src_features).to(DEVICE)
    tgt_tensor = torch.FloatTensor(tgt_features).to(DEVICE)
    src_labels = torch.zeros(len(src_features), dtype=torch.long).to(DEVICE)
    tgt_labels = torch.ones(len(tgt_features), dtype=torch.long).to(DEVICE)

    n_src = len(src_features)
    n_tgt = len(tgt_features)

    for epoch in range(epochs):
        model.train()
        idx_src = np.random.choice(n_src, min(batch_size, n_src), replace=False)
        idx_tgt = np.random.choice(n_tgt, min(batch_size, n_tgt), replace=False)

        src_batch = src_tensor[idx_src]
        tgt_batch = tgt_tensor[idx_tgt]
        src_lab = src_labels[idx_src]
        tgt_lab = tgt_labels[idx_tgt]

        _, src_domain = model(src_batch, alpha=1.0)
        _, tgt_domain = model(tgt_batch, alpha=1.0)
        loss_domain = criterion(src_domain, src_lab) + criterion(tgt_domain, tgt_lab)

        optimizer.zero_grad()
        loss_domain.backward()
        optimizer.step()

    model.eval()
    return model

def dann_rerank_with_size(per_img, F, text_proto, src_features, tgt_features, train_size):
    """用指定数量的目标域图训练DANN"""
    # 只取前train_size张图的候选框特征
    tgt_features_subset = []
    for i in range(min(train_size, len(per_img))):
        if len(per_img[i]['crop_idx']) > 0:
            tgt_features_subset.append(F[per_img[i]['crop_idx']])
    if tgt_features_subset:
        tgt_features_used = np.concatenate(tgt_features_subset, axis=0)
    else:
        tgt_features_used = F[:100]

    print(f'    DANN训练: 源域{len(src_features)}样本, 目标域{len(tgt_features_used)}样本 ({train_size}张图)')

    model = train_dann(src_features, tgt_features_used)
    with torch.no_grad():
        F_tensor = torch.FloatTensor(F).to(DEVICE)
        feat, _ = model(F_tensor, alpha=0.0)
        feat = feat.cpu().numpy()
        norms = np.linalg.norm(feat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        feat_norm = feat / norms

    text_proto_np = text_proto.cpu().numpy() if torch.is_tensor(text_proto) else text_proto
    key = f'dann_{train_size}'
    for p in per_img:
        n = len(p['gd'])
        if n:
            vis = np.maximum(0.0, feat_norm[p['crop_idx']] @ text_proto_np)
            p[key] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p[key] = np.zeros(0)
    return per_img, key

# ══════════════════════════ 主流程 ══════════════════════════

def main():
    print('=' * 70)
    print('DANN补数据重跑：目标域训练数据量扫描')
    print('=' * 70)

    # 加载DIOR源域原型和特征
    print('[1/4] 加载DIOR源域原型和特征...')
    protos = {}
    src_all_features = []
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p
        crops = gt_crops(images, gts)
        if crops:
            F_src = T.encode_clip_batch(crops).cpu().numpy()
            src_all_features.append(F_src)
    src_all_features = np.concatenate(src_all_features, axis=0)
    print(f'  源域特征总数: {len(src_all_features)}')

    # 对每个目标类运行
    print('[2/4] 逐类运行DANN不同训练规模对比...')
    results = {}
    for name, coco_cls, src_cls in TARGETS:
        print(f'\n  --- {name} ---')
        images, gts = load_coco_all(coco_cls)
        if len(images) < K_CALIB + 10:
            print(f'    图太少 ({len(images)}), 跳过')
            continue

        dets = cached_dets(images, TGT_TMPL.format(coco_cls), name)
        per_img, F = collect_features(images, gts, dets)

        # 原型
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]

        # Baseline
        for p in per_img:
            p['baseline'] = p['gd'].copy()

        # 不同训练规模的DANN
        class_results = {'baseline': dataset_ap(per_img, 'baseline')}
        for train_size in DANN_TRAIN_SIZES:
            per_img, key = dann_rerank_with_size(per_img, F, d_y, src_all_features, None, train_size)
            ap = dataset_ap(per_img, key)
            class_results[f'dann_{train_size}'] = ap
            print(f'    DANN({train_size}张图): AP50={ap:.4f}, 增益={ap-class_results["baseline"]:+.4f}')

        results[name] = class_results

    # 汇总
    print('\n' + '=' * 70)
    print('汇总（6类平均）')
    print('=' * 70)
    avg = {}
    methods = ['baseline'] + [f'dann_{s}' for s in DANN_TRAIN_SIZES]
    for m in methods:
        vals = [results[name][m] for name in results]
        avg[m] = np.mean(vals)

    print(f'  Baseline:          {avg["baseline"]:.4f}')
    for s in DANN_TRAIN_SIZES:
        key = f'dann_{s}'
        print(f'  DANN({s:3d}张图):     {avg[key]:.4f} (增益 {avg[key]-avg["baseline"]:+.4f})')

    # 保存
    out = {'per_class': results, 'average': avg, 'train_sizes': DANN_TRAIN_SIZES}
    out_path = DATA / 'sota_dann_rerun.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {out_path}')

if __name__ == '__main__':
    main()
