# -*- coding: utf-8 -*-
"""DTDM 阈值校准

在验证集上网格搜索每个轴的最优 CLIP 相似度阈值 τ_a(D)。
使用 OFAT 类级别的 δ 作为 ground truth：
  - δ > 0: 该轴修饰词对该类有帮助 → DTDM 应激活
  - δ ≤ 0: 该轴修饰词对该类无帮助/有害 → DTDM 应跳过

将阈值选择转换为 per-axis 二分类问题的 F1 最大化。
"""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from dtdm_axis_descriptions import AXIS_ORDER, precompute_axis_embeddings
from ofat_knowledge_base import OFATKnowledgeBase

# ── 配置 ───────────────────────────────────────────────
DIOR_DIR = r'C:\Users\张三\modeling\dior_data\full20_images'
COCO_DIR = r'C:\Users\张三\modeling\coco_data\val2017'
DEFAULT_TAU_RANGE = np.arange(-0.10, 0.55, 0.05)  # 网格搜索范围


def calibrate_thresholds(dtdm, domain, tau_range=None):
    """校准指定域的每轴 CLIP 相似度阈值。

    对域中每个有 OFAT 数据的类，取一张代表图像，计算 CLIP 相似度。
    对每个轴 a，以 δ(a) > 0 为正类，搜索最大化 F1 的 τ_a。

    Args:
        dtdm: DynamicTDM 实例
        domain: 'natural_scene' | 'remote_sensing'
        tau_range: 搜索范围，默认 [-0.10, 0.50] step 0.05

    Returns:
        Dict[str, float]: axis → optimal τ
        Dict[str, dict]: per-axis calibration details
    """
    import os, glob
    from PIL import Image

    if tau_range is None:
        tau_range = DEFAULT_TAU_RANGE

    domain_norm = dtdm.kb._normalize_domain(domain)
    kb = dtdm.kb

    # 收集每个类的代表图像和 OFAT delta
    class_data = []  # List of (class_name, image, {axis: delta})

    # 确定数据源
    if domain_norm == 'natural_scene':
        delta_source = kb.coco_deltas
        img_base = COCO_DIR
    else:
        delta_source = kb.dior_deltas
        img_base = DIOR_DIR

    # 收集每类一张代表图像
    if not os.path.isdir(img_base):
        print(f"  WARNING: {img_base} 不存在")
    elif domain_norm == 'natural_scene':
        # COCO: 用 pycocotools annotations 映射 image_id → class
        try:
            from pycocotools.coco import COCO
            ann_file = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'
            coco_api = COCO(ann_file)
            # 获取 20 个 OFAT 类的 image ids
            coco_classes = list(kb.coco['classes'].keys())
            cat_ids = coco_api.getCatIds(catNms=coco_classes)
            for cat_id in cat_ids:
                cat_name = coco_api.loadCats(cat_id)[0]['name']
                img_ids = coco_api.getImgIds(catIds=[cat_id])[:5]  # 每类取5张
                for img_id in img_ids[:1]:  # 实际只用1张
                    img_info = coco_api.loadImgs(img_id)[0]
                    img_path = os.path.join(img_base, img_info['file_name'])
                    if not os.path.exists(img_path):
                        continue
                    try:
                        img = Image.open(img_path).convert('RGB')
                    except Exception:
                        continue
                    deltas = {axis: delta_source.get((cat_name, axis), 0.0)
                              for axis in AXIS_ORDER}
                    class_data.append((cat_name, img, deltas))
        except Exception as e:
            print(f"  COCO annotations 加载失败: {e}")
    else:
        # DIOR: flat directory，文件名编码类名（如 airplane_2003.jpg）
        class_images = defaultdict(list)
        for fname in sorted(os.listdir(img_base)):
            if not fname.endswith('.jpg'):
                continue
            # rsplit 从右侧分割，处理 expressway_toll_station_2006.jpg
            parts = fname.rsplit('_', 1)
            if len(parts) != 2:
                continue
            cls_name = parts[0]
            # 验证类名在 OFAT 数据中存在
            if delta_source and not any(k[0] == cls_name for k in delta_source.keys()):
                continue
            class_images[cls_name].append(os.path.join(img_base, fname))

        for cls_name, img_paths in class_images.items():
            if not img_paths:
                continue
            try:
                img = Image.open(img_paths[0]).convert('RGB')
            except Exception:
                continue
            deltas = {axis: delta_source.get((cls_name, axis), 0.0)
                      for axis in AXIS_ORDER}
            if all(abs(d) < 1e-8 for d in deltas.values()):
                continue
            class_data.append((cls_name, img, deltas))

    if not class_data:
        print(f"  WARNING: 无可用图像，使用 fallback 统计校准")
        # Fallback: 使用类均值作为启发式阈值
        result = {}
        for axis in AXIS_ORDER:
            mu = kb.get_domain_mean(domain_norm, axis)
            if mu is None: mu = 0.0
            result[axis] = 0.05 if abs(mu) < 0.02 else 0.0
        return result, {'method': 'fallback_heuristic'}

    print(f"  校准数据: {len(class_data)} 类, {len(AXIS_ORDER)} 轴")
    print(f"  τ 搜索范围: [{tau_range[0]:.2f}, {tau_range[-1]:.2f}] step {tau_range[1]-tau_range[0]:.2f}")

    # 预计算所有图像的 CLIP 嵌入
    img_embeddings = []
    for cls_name, img, deltas in class_data:
        img_tensor = dtdm.preprocess(img).unsqueeze(0).to(dtdm.device)
        with torch.no_grad():
            feat = dtdm.clip_model.encode_image(img_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        img_embeddings.append(feat.squeeze(0))

    # 对每个轴搜索最优 τ
    optimal_tau = {}
    calibration_details = {}

    for axis in AXIS_ORDER:
        axis_emb = dtdm.axis_embeddings[axis]

        # 收集 ground truth 和 prediction scores
        y_true = []  # 1 if δ > 0, 0 otherwise
        scores = []  # CLIP cosine similarity

        for i, (cls_name, img, deltas) in enumerate(class_data):
            delta = deltas.get(axis, 0.0)
            y_true.append(1 if delta > 0 else 0)
            cos_sim = torch.dot(img_embeddings[i], axis_emb).item()
            scores.append(cos_sim)

        y_true = np.array(y_true)
        scores = np.array(scores)

        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos
        print(f"\n  {axis}: {n_pos}+ / {n_neg}− (δ>0 / δ≤0)")

        # 网格搜索
        best_tau = 0.0
        best_f1 = 0.0
        search_results = []

        for tau in tau_range:
            y_pred = (scores > tau).astype(int)
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            search_results.append({'tau': tau, 'f1': f1, 'prec': prec, 'rec': rec,
                                   'tp': tp, 'fp': fp, 'fn': fn})

            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau

        optimal_tau[axis] = float(best_tau)
        calibration_details[axis] = {
            'optimal_tau': float(best_tau),
            'best_f1': float(best_f1),
            'n_positive': int(n_pos),
            'n_negative': int(n_neg),
            'search_results': [{k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                                for k, v in r.items()}
                               for r in search_results],
            'scores_stats': {
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max()),
            }
        }
        print(f"    optimal τ = {best_tau:.3f} (F1={best_f1:.3f})")
        print(f"    scores: mean={scores.mean():.3f} std={scores.std():.3f} "
              f"range=[{scores.min():.3f}, {scores.max():.3f}]")

    return optimal_tau, calibration_details


# ── 主程序 ──
if __name__ == '__main__':
    import json
    from dynamic_tdm import DynamicTDM

    print("=" * 60)
    print("DTDM 阈值校准")
    print("=" * 60)

    print("\n[1/3] 加载 DTDM...")
    dtdm = DynamicTDM()

    all_results = {}
    for domain in ['natural_scene', 'remote_sensing']:
        print(f"\n[2/3] 校准 {domain}...")
        opt_tau, details = calibrate_thresholds(dtdm, domain)

        # 写回 OFAT_KB
        for axis, tau in opt_tau.items():
            dtdm.kb.set_threshold(domain, axis, tau)

        all_results[domain] = {
            'optimal_tau': opt_tau,
            'details': details,
        }

    # 保存
    output_path = Path(r'C:\Users\张三\Desktop\实验数据\dtdm_calibration.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[3/3] 结果已保存至: {output_path}")

    # 汇总
    print("\n" + "=" * 60)
    print("校准结果汇总")
    print("=" * 60)
    for domain in ['natural_scene', 'remote_sensing']:
        print(f"\n{domain}:")
        details = all_results[domain]['details']
        if isinstance(details, dict) and details.get('method') == 'fallback_heuristic':
            print(f"  (fallback heuristic, no images available)")
            for axis in AXIS_ORDER:
                tau = all_results[domain]['optimal_tau'].get(axis, 'N/A')
                print(f"  {axis:14s} τ*={tau}")
            continue
        for axis in AXIS_ORDER:
            info = details.get(axis, {})
            tau = info.get('optimal_tau', 'N/A')
            f1 = info.get('best_f1', 'N/A')
            stats = info.get('scores_stats', {})
            if stats:
                print(f"  {axis:14s} τ*={tau}  F1={f1}  "
                      f"μ_score={stats.get('mean', 0):.3f}")
            else:
                print(f"  {axis:14s} τ*={tau}  F1={f1}")

    print("\n✅ 校准完成")
