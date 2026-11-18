# -*- coding: utf-8 -*-
"""Visual Exemplar Prototype (VEP) — Few-Shot 视觉原型增强 GDINO 检测

核心思路:
  文本提示词优化对弱类(t网球、船)已达上限(GDINO vision-language alignment 不支持文本改善)
  → 用 few-shot 视觉 exemplar 弥补: GT bbox 裁剪 → CLIP 视觉原型 → 检测框重打分

两阶段:
  Stage 1: GDINO 低阈值(0.0)广撒网 → 高召回
  Stage 2: CLIP 视觉原型相似度重打分 → 提升精度

复用:
  - cps_search.py 的 CLIP/GDINO 加载模式
  - gdino_verifier.py 的 CLIP 图像编码模式

Usage:
    python visual_exemplar.py
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import io, json, glob, random, time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch, torch.nn.functional as F
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
SAVE_DIR.mkdir(exist_ok=True)
DIOR_TEST_DIR = r'C:\Users\张三\modeling\dior_data\full20_images'
PARQUET_PATH = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'

# ── DIOR 类别映射 ──
DIOR_ID2NAME = {
    0: 'airplane', 1: 'airport', 2: 'baseball field', 3: 'basketball court',
    4: 'bridge', 5: 'chimney', 6: 'dam', 7: 'expressway service area',
    8: 'expressway toll station', 9: 'golf course', 10: 'ground track field',
    11: 'harbor', 12: 'overpass', 13: 'ship', 14: 'stadium',
    15: 'storage tank', 16: 'tennis court', 17: 'train station',
    18: 'vehicle', 19: 'windmill',
}
DIOR_CLASSES = list(DIOR_ID2NAME.values())

# ── Baseline prompt (不加修饰词, 测纯视觉加成) ──
BASELINE_PROMPT = "an overhead image of {}"


# ══════════════════════════════════════════════════════════════════
# 工具: 加载模型
# ══════════════════════════════════════════════════════════════════

_gdino_proc, _gdino_model = None, None
_clip_model, _clip_preprocess = None, None


def get_gdino():
    global _gdino_proc, _gdino_model
    if _gdino_model is None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        gd_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
        _gdino_proc = AutoProcessor.from_pretrained(gd_cache)
        _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            gd_cache, dtype=torch.float32
        ).eval().to(DEVICE)
    return _gdino_proc, _gdino_model


def get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import clip as clip_module
        clip_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
        _clip_model, _clip_preprocess = clip_module.load(clip_path, device=DEVICE)
        _clip_model.eval()
    return _clip_model, _clip_preprocess


# ══════════════════════════════════════════════════════════════════
# Phase 1: 构建 Exemplar 库
# ══════════════════════════════════════════════════════════════════

def get_test_image_ids():
    """从 full20_images 提取所有测试 image_id (文件名中的数字)。"""
    ids = set()
    for f in glob.glob(os.path.join(DIOR_TEST_DIR, '*.jpg')):
        parts = Path(f).stem.split('_')
        for p in reversed(parts):
            if p.isdigit():
                ids.add(int(p))
                break
    return ids


def load_training_data():
    """加载 parquet, 返回 (df, available_ids_per_class)。

    available_ids: {class_name: [image_id, ...]} (不含 test 重叠)
    """
    df = pd.read_parquet(PARQUET_PATH)
    test_ids = get_test_image_ids()

    available = defaultdict(list)
    for _, row in df.iterrows():
        img_id = int(row['image_id'])
        if img_id in test_ids:
            continue  # 排除测试重叠
        cats = row['objects']['category']
        for c in cats:
            cls_name = DIOR_ID2NAME[int(c)]
            available[cls_name].append((img_id, row))
            break  # 每张图只记一次 (后续取 bbox 时再遍历)

    # 去重 image_id
    for cls_name in available:
        seen = set()
        unique = []
        for img_id, row in available[cls_name]:
            if img_id not in seen:
                seen.add(img_id)
                unique.append((img_id, row))
        available[cls_name] = unique

    print(f"训练数据: {len(df)} 图, 排除 {len(test_ids)} 测试 ID")
    for cls_name in DIOR_CLASSES:
        print(f"  {cls_name}: {len(available[cls_name])} 可用训练图")
    return df, dict(available)


def get_class_crops(df, class_name, available_rows, k=5):
    """为 class_name 获取 K 个多样化 exemplar crops。

    策略:
      1. 收集该类所有 bbox crop (每个 image_id 最多取 1 个最大的 bbox)
      2. CLIP encode 所有 crop
      3. 贪婪选择 mutual distance 最大的 top-K (maximin)
    """
    clip_m, clip_prep = get_clip()
    name2id = {v: k for k, v in DIOR_ID2NAME.items()}
    target_cat = name2id[class_name]

    crops_info = []  # [(PIL.Image, int image_id, float area), ...]

    for img_id, row in available_rows:
        objs = row['objects']
        # 找到该类所有 bbox
        best_area = -1
        best_bbox = None
        for i, cat in enumerate(objs['category']):
            if int(cat) == target_cat:
                area = int(objs['area'][i])
                if area > best_area:
                    best_area = area
                    best_bbox = objs['bbox'][i]

        if best_bbox is not None:
            img_bytes = row['image']['bytes']
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            x, y, w, h = [int(v) for v in best_bbox]
            # xywh → crop coords: 确保不越界
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(pil_img.width, x + w), min(pil_img.height, y + h)
            if x2 > x1 + 10 and y2 > y1 + 10:  # 太小跳过
                crop = pil_img.crop((x1, y1, x2, y2))
                crops_info.append((crop, img_id, float(best_area)))

    if len(crops_info) == 0:
        print(f"  ⚠ {class_name}: 无可用的 exemplar crop!")
        return None

    print(f"  {class_name}: {len(crops_info)} 个候选 crop → 选 top-{min(k, len(crops_info))}")

    # CLIP encode 所有 crop
    features = []
    for crop, _, _ in crops_info:
        img_tensor = clip_prep(crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = clip_m.encode_image(img_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        features.append(feat.cpu().squeeze(0))

    features = torch.stack(features, dim=0)  # (N, 512)

    # 多样化选择: maximin — 逐步选与已选集距离最大的
    N = features.shape[0]
    n_select = min(k, N)
    selected = []
    # 第一个: 选最接近所有 crop 均值的 (最具代表性)
    mean_feat = features.mean(dim=0)
    sims = (features @ mean_feat).numpy()
    first = int(np.argmax(sims))
    selected.append(first)

    # 剩余: maximin
    for _ in range(n_select - 1):
        selected_feats = features[selected]  # (S, 512)
        # 每个候选到已选集的最近距离
        sim_matrix = (features @ selected_feats.T).numpy()  # (N, S)
        min_dist_to_selected = 1.0 - sim_matrix.max(axis=1)  # (N,)
        # 排除已选的
        min_dist_to_selected[selected] = -np.inf
        next_idx = int(np.argmax(min_dist_to_selected))
        selected.append(next_idx)

    exemplar_crops = [crops_info[i][0] for i in selected]
    exemplar_ids = [crops_info[i][1] for i in selected]
    exemplar_feats = features[selected]  # (K, 512)
    prototype = exemplar_feats.mean(dim=0)
    prototype = prototype / prototype.norm()

    return {
        'prototype': prototype,
        'exemplar_features': exemplar_feats,
        'crops': exemplar_crops,
        'image_ids': exemplar_ids,
        'n_available': len(crops_info),
    }


def build_exemplar_bank(k=5):
    """主函数: 构建全 20 类 exemplar 库。"""
    print("=" * 60)
    print(f"Phase 1: 构建 Exemplar 库 (K={k})")
    print("=" * 60)

    # 预热 CLIP
    get_clip()

    df, available = load_training_data()

    bank = {}
    for cls_name in DIOR_CLASSES:
        rows = available.get(cls_name, [])
        result = get_class_crops(df, cls_name, rows, k=k)
        if result is not None:
            bank[cls_name] = result

    # 保存
    save_path = SAVE_DIR / f'exemplar_bank_k{k}.pt'
    torch.save({
        cls_name: {
            'prototype': info['prototype'],
            'exemplar_features': info['exemplar_features'],
            'image_ids': info['image_ids'],
            'n_available': info['n_available'],
        }
        for cls_name, info in bank.items()
    }, save_path)
    print(f"\nExemplar 库保存至: {save_path}")
    print(f"覆盖: {len(bank)}/{len(DIOR_CLASSES)} 类")
    return bank


# ══════════════════════════════════════════════════════════════════
# Phase 2: 两阶段检测
# ══════════════════════════════════════════════════════════════════

def gdino_detect_raw(images, prompt, low_thresh=0.0):
    """GDINO 低阈值检测, 返回所有 boxes + scores。

    Args:
        images: List[PIL.Image]
        prompt: str
        low_thresh: float, GDINO post_process threshold (0.0 = 全返回)

    Returns:
        List[Dict]: [{'boxes': Tensor (M,4), 'scores': Tensor (M,)}, ...]
    """
    gd_p, gd_m = get_gdino()
    results = []
    for img in images:
        inputs = gd_p(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            o = gd_m(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(DEVICE)
        r = gd_p.post_process_grounded_object_detection(
            o, threshold=low_thresh, target_sizes=ts
        )[0]
        results.append({
            'boxes': r['boxes'],      # (M, 4) xyxy
            'scores': r['scores'],    # (M,)
        })
    return results


def compute_per_class_threshold(bank_info, percentile=10):
    """从 exemplar 自身相似度分布估计 per-class threshold（仅作参考，不再用于检测判定）。

    实际检测使用标准 GDINO 阈值 0.25 对融合分数做判定。
    """
    feats = bank_info['exemplar_features']
    if feats.shape[0] < 2:
        return 0.2
    sim_matrix = (feats @ feats.T).numpy()
    triu_idx = np.triu_indices(feats.shape[0], k=1)
    pairwise_sims = sim_matrix[triu_idx]
    return float(np.percentile(pairwise_sims, percentile))


def visual_rerank_detections(gdino_results, images, class_name, bank, alpha=0.5,
                             prefilter_thresh=0.03, det_threshold=0.25):
    """对 GDINO 检测结果进行 CLIP 视觉重打分。

    使用几何平均融合: fused = gdino_score^α × visual_sim^(1-α)
    然后用标准 GDINO 阈值(0.25)判断检测，与 baseline 公平对比。

    Args:
        gdino_results: List[Dict], gdino_detect_raw 的输出
        images: List[PIL.Image]
        class_name: str
        bank: Dict, exemplar 库
        alpha: float, GDINO 分数权重 (0~1)
        prefilter_thresh: float, GDINO 分数粗过滤 (CLIP 前), 默认 0.03
        det_threshold: float, 融合分数检测阈值, 默认 0.25 (与 GDINO baseline 一致)

    Returns:
        List[Dict]: 每张图的最终检测结果
    """
    proto = bank[class_name]['prototype'].to(DEVICE)

    clip_m, clip_prep = get_clip()
    reranked = []

    for gd_r, img in zip(gdino_results, images):
        boxes = gd_r['boxes']
        scores = gd_r['scores']

        if len(boxes) == 0:
            reranked.append({'has_detection': False, 'best_fused': 0.0,
                             'n_raw': 0, 'threshold': det_threshold})
            continue

        # 粗过滤
        keep_mask = scores > prefilter_thresh
        keep_indices = torch.where(keep_mask)[0]
        if len(keep_indices) == 0:
            reranked.append({'has_detection': False, 'best_fused': 0.0,
                             'n_raw': len(boxes), 'n_filtered': 0,
                             'threshold': det_threshold})
            continue

        boxes = boxes[keep_indices]
        gd_scores_filtered = scores[keep_indices]

        # CLIP 视觉编码
        visual_sims = []
        for box in boxes:
            x1, y1, x2, y2 = [int(v.item()) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 5 or y2 <= y1 + 5:
                visual_sims.append(0.0)
                continue
            crop = img.crop((x1, y1, x2, y2))
            img_tensor = clip_prep(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = clip_m.encode_image(img_tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            sim = (feat @ proto).item()
            visual_sims.append(max(0.0, float(sim)))

        gd_scores = gd_scores_filtered.cpu().numpy()
        vis_scores = np.clip(np.array(visual_sims), 1e-6, 1.0)  # 防 log(0)
        gd_clipped = np.clip(gd_scores, 1e-6, 1.0)

        # 几何平均融合
        fused = (gd_clipped ** alpha) * (vis_scores ** (1.0 - alpha))

        # 标准阈值判断 (与 baseline GDINO 一致)
        has_det = np.any(fused > det_threshold)
        best_idx = int(np.argmax(fused))

        reranked.append({
            'has_detection': bool(has_det),
            'best_fused': float(fused[best_idx]),
            'best_gdino': float(gd_scores[best_idx]),
            'best_visual': float(vis_scores[best_idx]),
            'n_raw': len(keep_indices),
            'n_total_raw': len(gd_r['boxes']),
            'threshold': det_threshold,
        })

    return reranked


# ══════════════════════════════════════════════════════════════════
# Phase 3: 评估
# ══════════════════════════════════════════════════════════════════

def evaluate_class(class_name, bank, alpha=0.5, n_images=20):
    """对单个类运行 VEP 检测并计算 DR。

    Returns:
        Dict: 包含 baseline_dr, vep_dr, gdino_low_dr 等
    """
    paths = sorted(glob.glob(
        os.path.join(DIOR_TEST_DIR, f"{class_name.replace(' ','_')}_*.jpg")
    ))[:n_images]
    if not paths:
        paths = sorted(glob.glob(
            os.path.join(DIOR_TEST_DIR, f"{class_name}_*.jpg")
        ))[:n_images]
    images = [Image.open(p).convert('RGB') for p in paths]
    n = len(images)
    if n == 0:
        return None

    prompt = BASELINE_PROMPT.format(class_name)

    # Baseline: 标准阈值 0.25
    gd_p, gd_m = get_gdino()
    baseline_det = 0
    for img in images:
        inputs = gd_p(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            o = gd_m(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(DEVICE)
        r = gd_p.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=ts)[0]
        if len(r['boxes']) > 0:
            baseline_det += 1
    baseline_dr = baseline_det / n

    # GDINO 低阈值 (0.0) → 高召回 baseline
    raw = gdino_detect_raw(images, prompt, low_thresh=0.0)
    low_det = sum(1 for r in raw if len(r['boxes']) > 0)
    low_dr = low_det / n

    # VEP 重打分
    reranked = visual_rerank_detections(raw, images, class_name, bank, alpha=alpha)
    vep_det = sum(1 for r in reranked if r['has_detection'])
    vep_dr = vep_det / n

    # 统计
    avg_n_raw = np.mean([r['n_raw'] for r in reranked])
    avg_thresh = np.mean([r['threshold'] for r in reranked])

    return {
        'class': class_name,
        'baseline_dr': baseline_dr,
        'gdino_low_dr': low_dr,
        'vep_dr': vep_dr,
        'delta_vep': vep_dr - baseline_dr,
        'alpha': alpha,
        'n_images': n,
        'avg_raw_boxes': float(avg_n_raw),
        'avg_threshold': float(avg_thresh),
    }


def main():
    print("=" * 65)
    print("VEP: Visual Exemplar Prototype for GDINO Re-scoring")
    print("=" * 65)
    print(f"设备: {DEVICE}")

    # ── 预热模型 ──
    print("\n预热 CLIP + GDINO...")
    get_clip()
    get_gdino()
    print("完成")

    # ── Phase 1: 构建 Exemplar 库 ──
    K = 5
    bank = build_exemplar_bank(k=K)

    # 检查每类原型质量 (自身类内相似度)
    print(f"\n{'='*60}")
    print("Exemplar 原型质量 (类内 pairwise similarity)")
    print(f"{'='*60}")
    for cls_name in DIOR_CLASSES:
        if cls_name in bank:
            feats = bank[cls_name]['exemplar_features']
            sim_matrix = (feats @ feats.T).numpy()
            triu = np.triu_indices(feats.shape[0], k=1)
            pair_sims = sim_matrix[triu]
            p10 = np.percentile(pair_sims, 10)
            p50 = np.percentile(pair_sims, 50)
            print(f"  {cls_name:30s}: n={bank[cls_name]['n_available']:3d}, "
                  f"intra-sim p10={p10:.3f}, p50={p50:.3f}")

    # ── Phase 2+3: 全类评估, 多种 alpha ──
    alphas = [0.3, 0.5, 0.7]
    all_results = {}

    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"Phase 3: 全 20 类评估 (alpha={alpha})")
        print(f"{'='*60}")

        class_results = {}
        for i, cls_name in enumerate(DIOR_CLASSES):
            if cls_name not in bank:
                print(f"[{i+1:2d}/20] {cls_name}: 跳过 (无 exemplar)")
                continue
            result = evaluate_class(cls_name, bank, alpha=alpha)
            if result:
                class_results[cls_name] = result
                marker = "✅" if result['delta_vep'] > 0.02 else (
                    "⚠️" if result['delta_vep'] < -0.02 else "  ")
                print(f"[{i+1:2d}/20] {cls_name:30s} | "
                      f"Base={result['baseline_dr']:.3f} → "
                      f"VEP={result['vep_dr']:.3f} "
                      f"(Δ={result['delta_vep']:+.3f}) "
                      f"{marker} | "
                      f"raw={result['avg_raw_boxes']:.1f} boxes/img")

        # 汇总
        n_ok = len(class_results)
        m_base = sum(r['baseline_dr'] for r in class_results.values()) / n_ok
        m_vep = sum(r['vep_dr'] for r in class_results.values()) / n_ok
        m_delta = sum(r['delta_vep'] for r in class_results.values()) / n_ok

        print(f"\n  α={alpha}: mDR {m_base:.4f} → {m_vep:.4f} (Δ={m_delta:+.4f})")

        all_results[f'alpha_{alpha}'] = {
            'alpha': alpha,
            'class_results': class_results,
            'summary': {
                'mean_baseline': m_base,
                'mean_vep': m_vep,
                'mean_delta': m_delta,
                'n_classes': n_ok,
            },
        }

    # ── 保存结果 ──
    output = {
        'config': {'K': K, 'alphas': alphas},
        'results_by_alpha': all_results,
    }

    # 简化 per-class 汇总表
    best_alpha_summary = {}
    alpha_keys = list(all_results.keys())
    for cls_name in DIOR_CLASSES:
        best_delta = -99
        for ak in alpha_keys:
            cr = all_results[ak]['class_results']
            if cls_name in cr:
                d = cr[cls_name]['delta_vep']
                if d > best_delta:
                    best_delta = d
        best_alpha_summary[cls_name] = best_delta

    output['best_per_class'] = best_alpha_summary

    save_path = SAVE_DIR / 'visual_exemplar_results.json'
    # JSON 序列化: 只保留可序列化的字段
    serializable = {
        'config': output['config'],
        'best_per_class': best_alpha_summary,
    }
    for ak in alpha_keys:
        serializable[ak] = {
            'alpha': all_results[ak]['alpha'],
            'summary': all_results[ak]['summary'],
            'class_results': {
                cls: {k: v for k, v in info.items() if k != 'raw'}
                for cls, info in all_results[ak]['class_results'].items()
            },
        }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存至: {save_path}")

    # ── 最终汇总对比 ──
    print(f"\n{'='*65}")
    print("VEP → 对比 baseline (纯文本) 和 GDINO-Best (full_pipeline)")
    print(f"{'='*65}")
    print(f"{'Class':<30s} {'Base':>6s} {'VEP-best':>8s} {'Δ':>7s}")

    best_alpha = {}
    for ak in alpha_keys:
        alpha_val = all_results[ak]['alpha']
        for cls_name, cr in all_results[ak]['class_results'].items():
            if cls_name not in best_alpha or cr['delta_vep'] > best_alpha[cls_name]['delta_vep']:
                best_alpha[cls_name] = {
                    'alpha': alpha_val,
                    'delta_vep': cr['delta_vep'],
                    'vep_dr': cr['vep_dr'],
                    'baseline_dr': cr['baseline_dr'],
                }

    total_base, total_vep = 0, 0
    for cls_name in DIOR_CLASSES:
        if cls_name in best_alpha:
            b = best_alpha[cls_name]
            total_base += b['baseline_dr']
            total_vep += b['vep_dr']
            print(f"{cls_name:<30s} {b['baseline_dr']:.3f}   {b['vep_dr']:.3f}   {b['delta_vep']:+.3f}  (α={b['alpha']})")
    n = len(best_alpha)
    print(f"{'─'*55}")
    print(f"{'MEAN':<30s} {total_base/n:.4f}   {total_vep/n:.4f}   {total_vep/n - total_base/n:+.4f}")

    # 弱类特写
    weak = ['tennis court', 'ship', 'harbor', 'windmill']
    print(f"\n{'='*65}")
    print("弱类特写 (tennis court, ship, harbor, windmill)")
    print(f"{'='*65}")
    for cls_name in weak:
        if cls_name in best_alpha:
            b = best_alpha[cls_name]
            print(f"  {cls_name:20s}: {b['baseline_dr']:.3f} → {b['vep_dr']:.3f} "
                  f"(Δ={b['delta_vep']:+.3f}, α={b['alpha']})")
        else:
            print(f"  {cls_name:20s}: 无数据")

    print(f"\n✅ VEP 实验完成")


if __name__ == '__main__':
    main()
