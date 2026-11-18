# -*- coding: utf-8 -*-
r"""sota_dann_multiseed.py — DANN 200张图配置多种子方差
同时自查：三个类(car/bus/motorcycle)AP与Baseline完全相同是否为bug
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
TRAIN_SIZE = 200
SEEDS = [1, 7, 42, 2026, 2027]

TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]

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

def train_dann(src_features, tgt_features, seed, epochs=100, lr=1e-3, batch_size=256):
    torch.manual_seed(seed)
    np.random.seed(seed)
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

def load_coco_all(cls_name):
    return U.load_coco(cls_name, n=100000)

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

def main():
    print('=' * 70)
    print(f'DANN {TRAIN_SIZE}张图配置多种子方差（seeds={SEEDS}）')
    print('=' * 70)

    # 加载DIOR源域特征
    print('[1/3] 加载DIOR源域特征...')
    src_all_features = []
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        crops = gt_crops(images, gts)
        if crops:
            F_src = T.encode_clip_batch(crops).cpu().numpy()
            src_all_features.append(F_src)
    src_all_features = np.concatenate(src_all_features, axis=0)
    print(f'  源域特征总数: {len(src_all_features)}')

    # 逐类运行
    print('[2/3] 逐类运行多种子DANN...')
    all_results = {}
    debug_info = {}

    for name, coco_cls, src_cls in TARGETS:
        print(f'\n  --- {name} ---')
        images, gts = load_coco_all(coco_cls)
        dets = cached_dets(images, TGT_TMPL.format(coco_cls), name)
        per_img, F = collect_features(images, gts, dets)

        # 文本原型
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        text_proto_np = d_y.numpy()

        # Baseline
        for p in per_img:
            p['baseline'] = p['gd'].copy()
        baseline_ap = dataset_ap(per_img, 'baseline')

        # 收集目标域训练特征（前TRAIN_SIZE张图）
        tgt_features_list = []
        for i in range(min(TRAIN_SIZE, len(per_img))):
            if len(per_img[i]['crop_idx']) > 0:
                tgt_features_list.append(F[per_img[i]['crop_idx']])
        tgt_features = np.concatenate(tgt_features_list, axis=0) if tgt_features_list else F[:100]

        # 多种子运行
        seed_aps = []
        for seed in SEEDS:
            model = train_dann(src_all_features, tgt_features, seed)
            with torch.no_grad():
                F_tensor = torch.FloatTensor(F).to(DEVICE)
                feat, _ = model(F_tensor, alpha=0.0)
                feat = feat.cpu().numpy()
                norms = np.linalg.norm(feat, axis=1, keepdims=True)
                norms[norms == 0] = 1
                feat_norm = feat / norms

            key = f'dann_seed{seed}'
            for p in per_img:
                n = len(p['gd'])
                if n:
                    vis = np.maximum(0.0, feat_norm[p['crop_idx']] @ text_proto_np)
                    p[key] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
                else:
                    p[key] = np.zeros(0)
            ap = dataset_ap(per_img, key)
            seed_aps.append(ap)
            print(f'    seed={seed:5d}: AP50={ap:.4f}, 增益={ap-baseline_ap:+.4f}')

        mean_ap = np.mean(seed_aps)
        std_ap = np.std(seed_aps)
        all_results[name] = {
            'baseline': baseline_ap,
            'seed_aps': seed_aps,
            'mean': mean_ap,
            'std': std_ap,
            'mean_gain': mean_ap - baseline_ap,
        }
        print(f'    均值: {mean_ap:.4f} ± {std_ap:.4f}, 增益: {mean_ap-baseline_ap:+.4f}')

        # 自查：DANN变换后的特征与原始特征的差异
        # 用最后一个seed的模型做诊断
        with torch.no_grad():
            F_tensor = torch.FloatTensor(F).to(DEVICE)
            feat, _ = model(F_tensor, alpha=0.0)
            feat = feat.cpu().numpy()
        F_norm = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
        feat_norm_diag = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
        cos_sim = np.sum(F_norm * feat_norm_diag, axis=1)
        vis_orig = np.maximum(0.0, F_norm @ text_proto_np)
        vis_dann = np.maximum(0.0, feat_norm_diag @ text_proto_np)
        debug_info[name] = {
            'feat_cos_sim_mean': float(np.mean(cos_sim)),
            'feat_cos_sim_std': float(np.std(cos_sim)),
            'vis_orig_mean': float(np.mean(vis_orig)),
            'vis_dann_mean': float(np.mean(vis_dann)),
            'vis_diff_mean': float(np.mean(vis_dann - vis_orig)),
            'vis_diff_std': float(np.std(vis_dann - vis_orig)),
        }
        print(f'    [诊断] 特征余弦相似度: {np.mean(cos_sim):.4f}±{np.std(cos_sim):.4f}')
        print(f'    [诊断] vis变化: {np.mean(vis_dann-vis_orig):+.6f}±{np.std(vis_dann-vis_orig):.6f}')

    # 汇总
    print('\n' + '=' * 70)
    print(f'汇总（6类平均，{TRAIN_SIZE}张图，{len(SEEDS)}个种子）')
    print('=' * 70)
    baseline_mean = np.mean([all_results[n]['baseline'] for n in all_results])
    dann_mean = np.mean([all_results[n]['mean'] for n in all_results])
    dann_std_pooled = np.sqrt(np.mean([all_results[n]['std']**2 for n in all_results]))
    print(f'  Baseline:          {baseline_mean:.4f}')
    print(f'  DANN({TRAIN_SIZE}张图):  {dann_mean:.4f} ± {dann_std_pooled:.4f} (pooled std)')
    print(f'  平均增益:          {dann_mean-baseline_mean:+.4f}')

    # 逐类均值±std
    print('\n逐类结果:')
    for name in all_results:
        r = all_results[name]
        print(f'  {name}: {r["mean"]:.4f} ± {r["std"]:.4f} (基线 {r["baseline"]:.4f}, 增益 {r["mean_gain"]:+.4f})')

    # 保存
    out = {
        'config': {'train_size': TRAIN_SIZE, 'seeds': SEEDS, 'epochs': 100},
        'per_class': all_results,
        'debug': debug_info,
        'summary': {
            'baseline_mean': baseline_mean,
            'dann_mean': dann_mean,
            'dann_std_pooled': dann_std_pooled,
            'mean_gain': dann_mean - baseline_mean,
        }
    }
    out_path = DATA / 'sota_dann_multiseed.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {out_path}')

if __name__ == '__main__':
    main()
