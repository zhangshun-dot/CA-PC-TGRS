# -*- coding: utf-8 -*-
"""第二篇SOTA对比实验（简化版，一次一个域对，减少内存）
对比方法：Baseline / TGE / CORAL / DANN / 原始USC / C4 / Oracle
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import unified_eval as U

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

from usc_experiment_v12 import (
    DATA, K_PCA, ALPHAS, LAM, SRC_TMPL_OVERHEAD, TGT_TMPL,
    SOURCE_LOADERS, load_target_coco, build_neg_proto, evaluate, evaluate_baseline,
)
import tge_probe as T
from usc_rank1_affine_gpu_v2 import AffineUSC, load_domain_data

OUT = DATA / 'usc_sota_comparison_v2_results.json'

SENSOR_CONFIG = {
    'DIOR': {'gamma': -2.0, 'template': 'an overhead image of {}'},
    'NWPU': {'gamma': -2.0, 'template': 'an overhead image of {}'},
    'DOTA': {'gamma': -2.0, 'template': 'an overhead image of {}'},
    'VisDrone': {'gamma': -1.0, 'template': 'a drone view of {}'},
}


# ══════════════════════ DANN ══════════════════════
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DANNModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )
    def forward(self, x, alpha=1.0):
        feat = self.feature(x)
        reverse_feat = GradientReversal.apply(feat, alpha)
        domain_pred = self.domain_classifier(reverse_feat)
        return feat, domain_pred

def train_dann(src_features, tgt_features, epochs=20, lr=1e-3, batch_size=256):
    model = DANNModel(input_dim=src_features.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    src_tensor = torch.FloatTensor(src_features).to(device)
    tgt_tensor = torch.FloatTensor(tgt_features).to(device)
    src_labels = torch.zeros(len(src_features), dtype=torch.long).to(device)
    tgt_labels = torch.ones(len(tgt_features), dtype=torch.long).to(device)
    n_src, n_tgt = len(src_features), len(tgt_features)
    for epoch in range(epochs):
        model.train()
        idx_src = np.random.choice(n_src, min(batch_size, n_src), replace=False)
        idx_tgt = np.random.choice(n_tgt, min(batch_size, n_tgt), replace=False)
        _, src_domain = model(src_tensor[idx_src], alpha=1.0)
        _, tgt_domain = model(tgt_tensor[idx_tgt], alpha=1.0)
        loss = criterion(src_domain, src_labels[idx_src]) + criterion(tgt_domain, tgt_labels[idx_tgt])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


# ══════════════════════ CORAL ══════════════════════
def coral_align(src_features, tgt_features):
    src_cov = np.cov(src_features.T) + 1e-3 * np.eye(src_features.shape[1])
    tgt_cov = np.cov(tgt_features.T) + 1e-3 * np.eye(tgt_features.shape[1])
    src_eigvals, src_eigvecs = np.linalg.eigh(src_cov)
    tgt_eigvals, tgt_eigvecs = np.linalg.eigh(tgt_cov)
    src_eigvals = np.clip(src_eigvals, 1e-3, None)
    tgt_eigvals = np.clip(tgt_eigvals, 1e-3, None)
    src_whiten = src_eigvecs @ np.diag(1.0 / np.sqrt(src_eigvals)) @ src_eigvecs.T
    tgt_color = tgt_eigvecs @ np.diag(np.sqrt(tgt_eigvals)) @ tgt_eigvecs.T
    return src_whiten @ tgt_color


# ══════════════════════ 评估函数 ══════════════════════
def evaluate_method(domain_data, src_cls, coco_cls, method, gamma=0.0, alpha=0.0,
                    coral_W=None, dann_model=None):
    """用不同方法评估单个类别"""
    usc = domain_data['usc']
    protos_gpu = domain_data['protos_gpu']
    all_protos = domain_data['all_protos']
    text_deltas = domain_data['text_deltas']
    target_data = domain_data['target_data']

    if method == 'baseline':
        return float(evaluate_baseline(target_data[coco_cls]))

    # 构造原型
    if method in ['usc_original', 'usc_c4', 'usc_oracle']:
        p_corr_gpu = usc.correct_prototype_gpu(protos_gpu[src_cls], gamma, alpha)
        p_corr = p_corr_gpu.cpu().numpy()
        neg_raw = build_neg_proto(all_protos, src_cls)
        if neg_raw is not None:
            neg_gpu = torch.tensor(neg_raw, dtype=torch.float32, device=device)
            neg_corr_gpu = usc.correct_prototype_gpu(neg_gpu, gamma, alpha)
            neg = neg_corr_gpu.cpu().numpy()
        else:
            neg = None
    else:
        p_corr = all_protos[src_cls]
        neg = build_neg_proto(all_protos, src_cls)

    delta = text_deltas[src_cls]

    if method == 'tge':
        # TGE：纯目标域文本原型
        d_emb = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
        p_hat = d_emb / (np.linalg.norm(d_emb) + 1e-8)
    elif method in ['coral', 'dann']:
        p_hat = p_corr + 1.0 * delta
        p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)
    else:
        p_hat = p_corr + 1.0 * delta
        p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)

    # CORAL/DANN：对齐候选框特征后重排序
    if method in ['coral', 'dann']:
        data = target_data[coco_cls]
        all_boxes = []
        all_scores = []
        all_features = []
        for img in data['per_img']:
            if len(img['box']) > 0:
                all_boxes.append(img['box'])
                all_scores.append(img['gd'])
                if 'crop_idx' in img and len(img['crop_idx']) > 0:
                    feats = data['features'][img['crop_idx']]
                    all_features.append(feats)
                else:
                    all_features.append(np.zeros((0, 512)))

        fused_scores = []
        if method == 'coral' and coral_W is not None:
            for boxes, scores, feats in zip(all_boxes, all_scores, all_features):
                if len(boxes) > 0 and len(feats) > 0:
                    feats_aligned = feats @ coral_W.T
                    feats_norm = feats_aligned / (np.linalg.norm(feats_aligned, axis=1, keepdims=True) + 1e-8)
                    vis = np.maximum(0.0, feats_norm @ p_hat)
                    fused = np.sqrt(scores * np.clip(vis, 1e-6, 1.0))
                else:
                    fused = scores
                fused_scores.append(fused)
        elif method == 'dann' and dann_model is not None:
            with torch.no_grad():
                for boxes, scores, feats in zip(all_boxes, all_scores, all_features):
                    if len(boxes) > 0 and len(feats) > 0:
                        feats_tensor = torch.FloatTensor(feats).to(device)
                        feat_aligned, _ = dann_model(feats_tensor, alpha=0.0)
                        feat_aligned = feat_aligned.cpu().numpy()
                        feat_norm = feat_aligned / (np.linalg.norm(feat_aligned, axis=1, keepdims=True) + 1e-8)
                        vis = np.maximum(0.0, feat_norm @ p_hat[:feat_norm.shape[1]])
                        fused = np.sqrt(scores * np.clip(vis, 1e-6, 1.0))
                    else:
                        fused = scores
                    fused_scores.append(fused)
        else:
            fused_scores = all_scores

        preds = [(b, s) for b, s in zip(all_boxes, fused_scores)]
        gts = [img['gt'] for img in data['per_img']]
        return float(U.compute_ap50_img(preds, gts))

    # USC类方法/TGE：用evaluate函数
    return float(evaluate(target_data[coco_cls], p_hat, neg))


def run_domain(src_name, det_stats_model=None, X_mean=None, X_std=None):
    """跑单个域对的所有方法"""
    print(f"\n{'='*60}")
    print(f"域对: {src_name}→COCO")
    print(f"{'='*60}")

    data = load_domain_data(src_name)
    if data is None:
        print(f"  加载失败，跳过")
        return None

    valid_targets = data['valid_targets']
    print(f"  有效类别: {len(valid_targets)}")

    # 准备源域和目标域特征（用于DANN/CORAL）
    src_features_list = []
    tgt_features_list = []
    for eval_name, coco_cls, src_cls in valid_targets:
        src_data = data['source_data'][src_cls]
        tgt_data = data['target_data'][coco_cls]
        src_features_list.append(src_data['features'])
        tgt_features_list.append(tgt_data['features'])
    src_features = np.vstack(src_features_list)
    tgt_features = np.vstack(tgt_features_list)
    print(f"  源域特征: {src_features.shape}, 目标域特征: {tgt_features.shape}")

    # 训练DANN和CORAL
    print(f"  训练DANN...")
    dann_model = train_dann(src_features, tgt_features, epochs=20)
    print(f"  计算CORAL变换...")
    coral_W = coral_align(src_features, tgt_features)

    methods = ['baseline', 'tge', 'coral', 'dann', 'usc_original', 'usc_c4', 'usc_oracle']
    class_aps = {m: [] for m in methods}

    gamma_sensor = SENSOR_CONFIG[src_name]['gamma']

    for eval_name, coco_cls, src_cls in valid_targets:
        # 提取det_stats
        tgt_data = data['target_data'][coco_cls]
        all_scores = []
        for img in tgt_data['per_img']:
            if 'gd' in img and len(img['gd']) > 0:
                scores = img['gd']
                if hasattr(scores, 'tolist'):
                    all_scores.extend(scores.tolist())
                else:
                    all_scores.extend(list(scores))
        n_boxes = len(all_scores)
        scores_arr = np.array(all_scores) if n_boxes > 0 else np.array([0.0])
        det_stats = np.array([[float(np.mean(scores_arr)), float(np.var(scores_arr)), n_boxes]])

        # 预测α
        if det_stats_model is not None and X_mean is not None and X_std is not None:
            X_norm = (det_stats - X_mean) / X_std
            pred_alpha_cont = det_stats_model.predict(X_norm)[0]
            alpha_array = np.array(ALPHAS)
            alpha_pred = float(alpha_array[np.argmin(np.abs(alpha_array - pred_alpha_cont))])
        else:
            alpha_pred = 0.5

        # 逐方法评估
        ap_base = evaluate_method(data, src_cls, coco_cls, 'baseline')
        ap_tge = evaluate_method(data, src_cls, coco_cls, 'tge')
        ap_coral = evaluate_method(data, src_cls, coco_cls, 'coral', coral_W=coral_W)
        ap_dann = evaluate_method(data, src_cls, coco_cls, 'dann', dann_model=dann_model)
        ap_usc_orig = evaluate_method(data, src_cls, coco_cls, 'usc_original', gamma=0.0, alpha=0.5)
        ap_c4 = evaluate_method(data, src_cls, coco_cls, 'usc_c4', gamma=gamma_sensor, alpha=alpha_pred)

        # Oracle：扫描α找最优
        best_ap = -1
        best_alpha = 0.0
        for alpha in ALPHAS:
            ap = evaluate_method(data, src_cls, coco_cls, 'usc_oracle', gamma=-2.0, alpha=alpha)
            if ap > best_ap:
                best_ap = ap
                best_alpha = alpha
        ap_oracle = best_ap

        class_aps['baseline'].append(ap_base)
        class_aps['tge'].append(ap_tge)
        class_aps['coral'].append(ap_coral)
        class_aps['dann'].append(ap_dann)
        class_aps['usc_original'].append(ap_usc_orig)
        class_aps['usc_c4'].append(ap_c4)
        class_aps['usc_oracle'].append(ap_oracle)

        print(f"  {coco_cls:<12}: base={ap_base:.4f} tge={ap_tge:.4f} coral={ap_coral:.4f} "
              f"dann={ap_dann:.4f} usc_orig={ap_usc_orig:.4f} c4={ap_c4:.4f} oracle={ap_oracle:.4f}")

    # 释放内存
    del data, dann_model, coral_W, src_features, tgt_features
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {m: float(np.mean(class_aps[m])) for m in methods}


def main():
    source_names = list(SOURCE_LOADERS.keys())
    print(f"域对列表: {source_names}")

    # 阶段1：训练det_stats自适应α模型（用所有域对的数据）
    print(f"\n{'='*60}")
    print("阶段1：收集所有域对的det_stats和Oracle α")
    print(f"{'='*60}")

    feature_names = ['det_mean', 'det_var', 'n_boxes']
    all_class_info = []

    for src in source_names:
        print(f"\n--- {src}→COCO ---")
        data = load_domain_data(src)
        if data is None:
            continue
        valid_targets = data['valid_targets']
        for eval_name, coco_cls, src_cls in valid_targets:
            tgt_data = data['target_data'][coco_cls]
            all_scores = []
            for img in tgt_data['per_img']:
                if 'gd' in img and len(img['gd']) > 0:
                    scores = img['gd']
                    if hasattr(scores, 'tolist'):
                        all_scores.extend(scores.tolist())
                    else:
                        all_scores.extend(list(scores))
            n_boxes = len(all_scores)
            scores_arr = np.array(all_scores) if n_boxes > 0 else np.array([0.0])
            det_stats = {
                'det_mean': float(np.mean(scores_arr)),
                'det_var': float(np.var(scores_arr)),
                'n_boxes': n_boxes,
            }
            # 扫描α找Oracle
            usc = data['usc']
            protos_gpu = data['protos_gpu']
            all_protos = data['all_protos']
            text_deltas = data['text_deltas']
            target_data = data['target_data']
            best_ap = -1
            best_alpha = 0.0
            for alpha in ALPHAS:
                ap = evaluate_method(data, src_cls, coco_cls, 'usc_oracle', gamma=-2.0, alpha=alpha)
                if ap > best_ap:
                    best_ap = ap
                    best_alpha = alpha
            all_class_info.append({
                'domain': src, 'class': coco_cls, 'src_class': src_cls,
                'det_stats': det_stats, 'oracle_alpha': best_alpha, 'oracle_ap': best_ap,
            })
        del data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n总共收集 {len(all_class_info)} 个类别")

    # 训练全局det_stats模型
    X = np.array([[c['det_stats'][f] for f in feature_names] for c in all_class_info])
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    y_alpha = np.array([c['oracle_alpha'] for c in all_class_info])

    det_stats_model = LinearRegression()
    det_stats_model.fit(X_norm, y_alpha)

    print(f"n_boxes与最优α相关性: ρ={spearmanr([c['det_stats']['n_boxes'] for c in all_class_info], y_alpha)[0]:.3f}")

    # 阶段2：逐域对跑SOTA对比
    print(f"\n{'='*60}")
    print("阶段2：逐域对SOTA对比")
    print(f"{'='*60}")

    results = {}
    for src in source_names:
        domain_result = run_domain(src, det_stats_model, X_mean, X_std)
        if domain_result is not None:
            results[src] = domain_result

    # 阶段3：汇总
    print(f"\n{'='*60}")
    print("阶段3：SOTA对比汇总")
    print(f"{'='*60}")
    method_names = {
        'baseline': 'Baseline', 'tge': 'TGE', 'coral': 'CORAL', 'dann': 'DANN',
        'usc_original': '原始USC', 'usc_c4': 'C4 (Ours)', 'usc_oracle': 'Oracle',
    }
    methods = list(method_names.keys())
    print(f"{'方法':<15} {'DIOR':>8} {'NWPU':>8} {'DOTA':>8} {'VisDrone':>10} {'总体':>8} {'vs Base':>10}")
    print("-" * 75)
    baseline_overall = np.mean([results[s]['baseline'] for s in results])
    for m in methods:
        overall = np.mean([results[s][m] for s in results])
        gain = overall - baseline_overall
        print(f"{method_names[m]:<15} {results.get('DIOR',{}).get(m,0):>8.4f} "
              f"{results.get('NWPU',{}).get(m,0):>8.4f} {results.get('DOTA',{}).get(m,0):>8.4f} "
              f"{results.get('VisDrone',{}).get(m,0):>10.4f} {overall:>8.4f} {gain:>+10.4f}")

    # 保存
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    final_results = {
        'results': convert(results),
        'method_names': method_names,
        'baseline_overall': float(baseline_overall),
        'n_classes': len(all_class_info),
        'n_domains': len(source_names),
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")


if __name__ == '__main__':
    main()
