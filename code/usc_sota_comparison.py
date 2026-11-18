# -*- coding: utf-8 -*-
"""第二篇SOTA对比实验：4个域对上对比
- Baseline: GDINO-tiny原始分数
- TGE: 文本引导增强（目标域文本嵌入重排序）
- CORAL: 二阶统计量对齐（闭式解）
- DANN: 域对抗特征对齐（训练）
- 原始USC: γ=0, α=0.5
- C4: 传感器感知γ + det_stats自适应α（我们的方法）
- Oracle: 逐类最优(γ=-2.0, α)
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

from usc_experiment_v12 import (
    DATA, K_PCA, ALPHAS, LAM, SRC_TMPL_OVERHEAD, TGT_TMPL,
    SOURCE_LOADERS, load_target_coco, build_neg_proto, evaluate, evaluate_baseline,
)
import tge_probe as T
from usc_rank1_affine_gpu_v2 import AffineUSC, load_domain_data

OUT = DATA / 'usc_sota_comparison_results.json'

# 传感器感知配置
SENSOR_CONFIG = {
    'DIOR': {'gamma': -2.0, 'template': 'an overhead image of {}'},
    'NWPU': {'gamma': -2.0, 'template': 'an overhead image of {}'},
    'DOTA': {'gamma': -2.0, 'template': 'an overhead image of {}'},
    'VisDrone': {'gamma': -1.0, 'template': 'a drone view of {}'},
}

PRE_FILTER = 0.03


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

def train_dann(src_features, tgt_features, epochs=30, lr=1e-3, batch_size=256):
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
def evaluate_with_rerank(domain_data, src_cls, coco_cls, rerank_type,
                         gamma=0.0, alpha=0.0, template=SRC_TMPL_OVERHEAD,
                         src_features=None, tgt_features=None, dann_model=None, coral_W=None):
    """用不同重排序方法评估单个类别"""
    usc = domain_data['usc']
    protos_gpu = domain_data['protos_gpu']
    all_protos = domain_data['all_protos']
    target_data = domain_data['target_data']

    # 校正正原型（USC类方法）
    if rerank_type in ['usc_original', 'usc_c4', 'usc_oracle']:
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
        # 非USC方法：用源域原型
        p_corr = all_protos[src_cls]
        neg_raw = build_neg_proto(all_protos, src_cls)
        neg = neg_raw

    # 文本外推
    s_emb = T.text_avg([template], src_cls).cpu().numpy()
    d_emb = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
    delta = d_emb - s_emb

    if rerank_type == 'tge':
        # TGE：纯文本原型（目标域文本嵌入）
        p_hat = d_emb
        p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)
    elif rerank_type in ['coral', 'dann']:
        # CORAL/DANN：源域原型 + 文本外推，然后用对齐后的特征重排序
        p_hat = p_corr + 1.0 * delta
        p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)
    else:
        # USC类方法：校正后的原型 + 文本外推
        p_hat = p_corr + 1.0 * delta
        p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)

    # 评估
    if rerank_type == 'baseline':
        # Baseline：只用检测器分数
        ap = evaluate_baseline(target_data[coco_cls])
    elif rerank_type in ['coral', 'dann']:
        # CORAL/DANN：对齐候选框特征后，用原型相似度重排序
        data = target_data[coco_cls]
        all_boxes = []
        all_scores = []
        all_features = []
        for img in data['per_img']:
            if len(img['box']) > 0:
                all_boxes.append(img['box'])
                all_scores.append(img['gd'])
                # 获取候选框特征
                if 'crop_idx' in img and len(img['crop_idx']) > 0:
                    feats = data['features'][img['crop_idx']]
                    all_features.append(feats)
                else:
                    all_features.append(np.zeros((0, 512)))

        if rerank_type == 'coral' and coral_W is not None:
            # 用CORAL对齐后的特征
            fused_scores = []
            for boxes, scores, feats in zip(all_boxes, all_scores, all_features):
                if len(boxes) > 0 and len(feats) > 0:
                    feats_aligned = feats @ coral_W.T
                    feats_norm = feats_aligned / (np.linalg.norm(feats_aligned, axis=1, keepdims=True) + 1e-8)
                    vis = np.maximum(0.0, feats_norm @ p_hat)
                    fused = np.sqrt(scores * np.clip(vis, 1e-6, 1.0))
                else:
                    fused = scores
                fused_scores.append(fused)
        elif rerank_type == 'dann' and dann_model is not None:
            # 用DANN对齐后的特征
            fused_scores = []
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

        # 计算AP
        from usc_experiment_v12 import compute_ap50
        preds = [(b, s) for b, s in zip(all_boxes, fused_scores)]
        gts = [img['gt'] for img in data['per_img']]
        ap = compute_ap50(preds, gts)
    else:
        # USC类方法/TGE：用evaluate函数
        ap = evaluate(target_data[coco_cls], p_hat, neg)

    return float(ap)


def main():
    source_names = list(SOURCE_LOADERS.keys())
    print(f"\n域对: {source_names}")

    # 阶段1：加载所有域对数据
    print(f"\n{'='*60}")
    print("阶段1：加载所有域对数据")
    print(f"{'='*60}")
    all_domain_data = {}
    for src in source_names:
        print(f"\n--- {src}→COCO ---")
        data = load_domain_data(src)
        if data is not None:
            all_domain_data[src] = data
            print(f"  完成，有效类别: {len(data['valid_targets'])}")

    # 阶段2：训练det_stats自适应α模型（全局γ=-2.0）
    print(f"\n{'='*60}")
    print("阶段2：训练det_stats自适应α模型")
    print(f"{'='*60}")

    feature_names = ['det_mean', 'det_var', 'n_boxes']
    all_class_info = []

    for src in all_domain_data:
        domain_data = all_domain_data[src]
        valid_targets = domain_data['valid_targets']
        for eval_name, coco_cls, src_cls in valid_targets:
            # 提取det_stats
            data = domain_data['target_data'][coco_cls]
            all_scores = []
            for img in data['per_img']:
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
            # 扫描α（γ=-2.0）找Oracle
            best_ap = -1
            best_alpha = 0.0
            for alpha in ALPHAS:
                ap = evaluate_with_rerank(domain_data, src_cls, coco_cls, 'usc_oracle',
                                          gamma=-2.0, alpha=alpha)
                if ap > best_ap:
                    best_ap = ap
                    best_alpha = alpha
            all_class_info.append({
                'domain': src, 'class': coco_cls, 'src_class': src_cls,
                'det_stats': det_stats, 'oracle_alpha': best_alpha, 'oracle_ap': best_ap,
            })

    # 训练LOO模型
    X = np.array([[c['det_stats'][f] for f in feature_names] for c in all_class_info])
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    y_alpha = np.array([c['oracle_alpha'] for c in all_class_info])

    pred_alpha = np.zeros(len(all_class_info))
    for i in range(len(all_class_info)):
        X_train = np.delete(X_norm, i, axis=0)
        y_train = np.delete(y_alpha, i)
        X_test = X_norm[i:i+1]
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred_alpha[i] = model.predict(X_test)[0]

    alpha_array = np.array(ALPHAS)
    pred_alpha_discrete = np.array([alpha_array[np.argmin(np.abs(alpha_array - a))] for a in pred_alpha])

    for i, c in enumerate(all_class_info):
        c['pred_alpha'] = float(pred_alpha_discrete[i])

    print(f"n_boxes与最优α相关性: ρ={spearmanr([c['det_stats']['n_boxes'] for c in all_class_info], y_alpha)[0]:.3f}")

    # 阶段3：SOTA对比
    print(f"\n{'='*60}")
    print("阶段3：SOTA对比实验")
    print(f"{'='*60}")

    methods = ['baseline', 'tge', 'coral', 'dann', 'usc_original', 'usc_c4', 'usc_oracle']
    results = {m: {} for m in methods}

    for src in all_domain_data:
        domain_data = all_domain_data[src]
        valid_targets = domain_data['valid_targets']
        print(f"\n--- {src}→COCO ({len(valid_targets)}类) ---")

        # 准备源域和目标域特征（用于DANN/CORAL）
        src_features_list = []
        tgt_features_list = []
        for eval_name, coco_cls, src_cls in valid_targets:
            src_data = domain_data['source_data'][src_cls]
            tgt_data = domain_data['target_data'][coco_cls]
            src_features_list.append(src_data['features'])
            tgt_features_list.append(tgt_data['features'])
        src_features = np.vstack(src_features_list)
        tgt_features = np.vstack(tgt_features_list)

        # 训练DANN和CORAL（每个域对一次）
        print(f"  训练DANN...")
        dann_model = train_dann(src_features, tgt_features, epochs=30)
        print(f"  计算CORAL变换...")
        coral_W = coral_align(src_features, tgt_features)

        class_aps = {m: [] for m in methods}
        for eval_name, coco_cls, src_cls in valid_targets:
            class_info = next(c for c in all_class_info if c['domain'] == src and c['class'] == coco_cls)
            gamma_sensor = SENSOR_CONFIG[src]['gamma']
            alpha_pred = class_info['pred_alpha']

            # Baseline
            ap_base = evaluate_with_rerank(domain_data, src_cls, coco_cls, 'baseline')
            class_aps['baseline'].append(ap_base)

            # TGE
            ap_tge = evaluate_with_rerank(domain_data, src_cls, coco_cls, 'tge')
            class_aps['tge'].append(ap_tge)

            # CORAL
            ap_coral = evaluate_with_rerank(domain_data, src_cls, coco_cls, 'coral',
                                             coral_W=coral_W, src_features=src_features, tgt_features=tgt_features)
            class_aps['coral'].append(ap_coral)

            # DANN
            ap_dann = evaluate_with_rerank(domain_data, src_cls, coco_cls, 'dann',
                                            dann_model=dann_model, src_features=src_features, tgt_features=tgt_features)
            class_aps['dann'].append(ap_dann)

            # 原始USC (γ=0, α=0.5)
            ap_usc_orig = evaluate_with_rerank(domain_data, src_cls, coco_cls, 'usc_original',
                                                gamma=0.0, alpha=0.5)
            class_aps['usc_original'].append(ap_usc_orig)

            # C4 (传感器感知γ + det_stats自适应α)
            ap_c4 = evaluate_with_rerank(domain_data, src_cls, coco_cls, 'usc_c4',
                                          gamma=gamma_sensor, alpha=alpha_pred)
            class_aps['usc_c4'].append(ap_c4)

            # Oracle (γ=-2.0, 逐类最优α)
            ap_oracle = class_info['oracle_ap']
            class_aps['usc_oracle'].append(ap_oracle)

            print(f"  {coco_cls:<12}: base={ap_base:.4f} tge={ap_tge:.4f} coral={ap_coral:.4f} "
                  f"dann={ap_dann:.4f} usc_orig={ap_usc_orig:.4f} c4={ap_c4:.4f} oracle={ap_oracle:.4f}")

        for m in methods:
            results[m][src] = float(np.mean(class_aps[m]))

    # 阶段4：汇总
    print(f"\n{'='*60}")
    print("阶段4：SOTA对比汇总")
    print(f"{'='*60}")
    print(f"{'方法':<15} {'DIOR':>8} {'NWPU':>8} {'DOTA':>8} {'VisDrone':>10} {'总体':>8} {'vs Baseline':>12}")
    print("-" * 75)
    baseline_overall = np.mean(list(results['baseline'].values()))
    method_names = {
        'baseline': 'Baseline',
        'tge': 'TGE',
        'coral': 'CORAL',
        'dann': 'DANN',
        'usc_original': '原始USC',
        'usc_c4': 'C4 (Ours)',
        'usc_oracle': 'Oracle',
    }
    for m in methods:
        d = results[m]
        overall = np.mean(list(d.values()))
        gain = overall - baseline_overall
        print(f"{method_names[m]:<15} {d.get('DIOR',0):>8.4f} {d.get('NWPU',0):>8.4f} "
              f"{d.get('DOTA',0):>8.4f} {d.get('VisDrone',0):>10.4f} {overall:>8.4f} {gain:>+12.4f}")

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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU显存已释放")


if __name__ == '__main__':
    main()
