# -*- coding: utf-8 -*-
"""统一流水线: 扩大 OFAT 覆盖 + 积累 GDINO 反馈

两条腿走路:
  A. 弱类增强: tennis court / windmill / harbor / ship 用更多修饰词变体
  B. 全类搜索: 20 类 CPS 搜索 + GDINO 验证 → 自动积累训练数据
  A+B 合并 → 重训 CPS → 重训 Verifier → 迭代

Usage:
    python full_pipeline.py
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json, glob, random, time
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
SAVE_DIR.mkdir(exist_ok=True)
DIOR_DIR = r'C:\Users\张三\modeling\dior_data\full20_images'

# ── 全部 DIOR 类别 ──
DIOR_CLASSES = [
    'airplane','airport','baseball field','basketball court','bridge',
    'chimney','dam','expressway service area','expressway toll station',
    'golf course','ground track field','harbor','overpass','ship','stadium',
    'storage tank','tennis court','train station','vehicle','windmill'
]

# ── 扩展修饰词库 (A: 更多变体) ──
MODIFIER_VARIANTS_FULL = {
    'granularity': [
        'with fine details', 'showing intricate details',
        'with sharp edges', 'featuring detailed structures',
        'with crisp boundaries', 'clearly delineated',
    ],
    'spatial': [
        'in its natural setting', 'in context of surroundings',
        'within its environment', 'surrounded by typical terrain',
        'set against the background', 'in a characteristic location',
    ],
    'geometric': [
        'with distinct structural features', 'showing clear geometry',
        'with well-defined shape', 'featuring characteristic outlines',
        'with recognizable form', 'geometrically distinct',
    ],
    'scale': [
        'large and prominent', 'clearly visible in scale',
        'dominating the view', 'at full resolution',
        'occupying significant area', 'conspicuous in size',
    ],
    'spectral': [
        'with natural coloring', 'in true color',
        'with realistic tones', 'showing natural hues',
        'with accurate color rendition', 'in natural light',
    ],
    'texture': [
        'with clearly visible surface features', 'showing surface texture',
        'with visible material patterns', 'featuring surface details',
        'with distinct surface characteristics', 'texturally apparent',
    ],
}

PREFIXES = [
    'an overhead image of', 'a satellite image of', 'a photo of',
    'a remote sensing image of', 'an aerial view of',
    'a high-resolution overhead view of',
]

AXIS_ORDER = ['granularity', 'spatial', 'geometric', 'scale', 'spectral', 'texture']

# 弱类列表 (需要更多修饰词变体的类)
WEAK_CLASSES = ['tennis court', 'windmill', 'harbor', 'ship']

# ── GDINO ──
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
GD_CACHE = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
gd_proc = None
gd_model = None

def get_gdino():
    global gd_proc, gd_model
    if gd_model is None:
        gd_proc = AutoProcessor.from_pretrained(GD_CACHE)
        gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GD_CACHE, dtype=torch.float32
        ).eval().to(DEVICE)
    return gd_proc, gd_model


# ══════════════════════════════════════════════════════════════════
# A: 弱类增强 — 更多修饰词变体 + 更多图像
# ══════════════════════════════════════════════════════════════════

def run_ofat_enhanced(class_name, n_images=50):
    """对弱类运行增强 OFAT: 更多修饰词变体, 更多图像。

    Returns:
        Dict: {axis: {'deltas': {modifier: ΔF1, ...}, 'best': modifier}}
    """
    paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{class_name.replace(' ','_')}_*.jpg")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{class_name}_*.jpg")))
    paths = paths[:n_images]
    if not paths:
        print(f"    无图像, 跳过")
        return {}
    images = [Image.open(p).convert('RGB') for p in paths]

    gd_p, gd_m = get_gdino()

    def detect_rate(prompt):
        detections = []
        for img in images:
            inputs = gd_p(images=img, text=prompt, return_tensors='pt')
            inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            with torch.no_grad():
                o = gd_m(**inputs)
            ts = torch.tensor([img.size[::-1]]).to(DEVICE)
            r = gd_p.post_process_grounded_object_detection(
                o, threshold=0.25, target_sizes=ts
            )[0]
            detections.append(len(r['boxes']) > 0)
        return float(np.mean(detections))

    # 基线
    base_prompt = f'an overhead image of {class_name}'
    base_dr = detect_rate(base_prompt)
    print(f"    基线: {base_dr:.3f}")

    results = {}

    for axis in AXIS_ORDER:
        variants = MODIFIER_VARIANTS_FULL[axis]  # 6 个变体/轴 (不是 4 个)
        deltas = {}
        best_mod = None
        best_dr = base_dr

        for mod in variants:
            prompt = f'an overhead image of {class_name}, {mod}'
            dr = detect_rate(prompt)
            delta = dr - base_dr
            deltas[mod] = delta

            if dr > best_dr:
                best_dr = dr
                best_mod = mod

        mean_delta = np.mean(list(deltas.values()))
        results[axis] = {
            'deltas': deltas,
            'mean_delta': float(mean_delta),
            'best_modifier': best_mod,
            'best_dr': best_dr,
            'best_delta': best_dr - base_dr,
        }

        n_pos = sum(1 for d in deltas.values() if d > 0)
        print(f"    {axis:14s}: δ={mean_delta:+.4f} [{n_pos}/{len(variants)} pos] best='{best_mod[:40] if best_mod else 'N/A'}...'")

    return {
        'class': class_name,
        'baseline_dr': base_dr,
        'baseline_prompt': base_prompt,
        'n_images': n_images,
        'axes': results,
    }


# ══════════════════════════════════════════════════════════════════
# B: CPS 搜索 + GDINO 验证 + 数据积累
# ══════════════════════════════════════════════════════════════════

from cps_search import CPSScorer as CPSScorerV2


def generate_candidates_rich(class_name, kb, is_weak=False, n_total=50):
    """生成候选 prompt (弱类更多候选)。"""
    candidates = set()

    # 前缀
    for prefix in PREFIXES:
        candidates.add(f'{prefix} {class_name}')

    # 修饰词
    modifier_source = MODIFIER_VARIANTS_FULL if is_weak else {
        k: v[:4] for k, v in MODIFIER_VARIANTS_FULL.items()
    }

    for axis in AXIS_ORDER:
        for mod in modifier_source[axis]:
            for prefix in ['an overhead image of', 'a photo of']:
                candidates.add(f'{prefix} {class_name}, {mod}')

    # OFAT 推荐组合
    recs = kb.get_recommended_axes('remote_sensing', class_name)
    top_axes = [a for a, _ in recs[:3]]
    if len(top_axes) >= 2:
        for prefix in PREFIXES[:3]:
            for i, a1 in enumerate(top_axes):
                for a2 in top_axes[i+1:]:
                    m1_variants = modifier_source.get(a1, modifier_source.get(a1, []))
                    m2_variants = modifier_source.get(a2, modifier_source.get(a2, []))
                    for m1 in m1_variants[:2]:
                        for m2 in m2_variants[:2]:
                            candidates.add(f'{prefix} {class_name}, {m1}, {m2}')

    # 随机组合
    all_mods = []
    for axis in AXIS_ORDER:
        all_mods.extend(modifier_source[axis])

    rng = random.Random(hash(class_name) % 10000 + 42)
    while len(candidates) < n_total:
        n = rng.randint(1, 3)
        mods = rng.sample(all_mods, min(n, len(all_mods)))
        used_axes = set()
        filtered = []
        for m in mods:
            for axis, variants in modifier_source.items():
                if m in variants and axis not in used_axes:
                    filtered.append(m)
                    used_axes.add(axis)
                    break
        prefix = rng.choice(PREFIXES)
        prompt = f'{prefix} {class_name}'
        if filtered:
            prompt += ', ' + ', '.join(filtered)
        candidates.add(prompt)

    return list(candidates)


def gdino_verify_batch(class_name, prompts, n_images=20):
    """批量 GDINO 验证 — 返回 {prompt: DR}。"""
    paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{class_name.replace(' ','_')}_*.jpg")))[:n_images]
    if not paths:
        paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{class_name}_*.jpg")))[:n_images]
    images = [Image.open(p).convert('RGB') for p in paths]
    if not images:
        return {}

    gd_p, gd_m = get_gdino()
    results = {}

    for prompt in prompts:
        detections = []
        for img in images:
            inputs = gd_p(images=img, text=prompt, return_tensors='pt')
            inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            with torch.no_grad():
                o = gd_m(**inputs)
            ts = torch.tensor([img.size[::-1]]).to(DEVICE)
            r = gd_p.post_process_grounded_object_detection(
                o, threshold=0.25, target_sizes=ts
            )[0]
            detections.append(len(r['boxes']) > 0)
        results[prompt] = float(np.mean(detections))

    return results


def cps_search_and_verify(class_name, scorer, kb, n_images=20, is_weak=False):
    """CPS 搜索 + GDINO 验证 (单类)。

    Returns:
        Dict: 搜索结果 + GDINO 反馈数据
    """
    # 生成候选
    n_cands = 60 if is_weak else 40
    candidates = generate_candidates_rich(class_name, kb, is_weak=is_weak, n_total=n_cands)

    # CPS 评分
    cps_ranking = scorer.rank(candidates)
    cps_top = cps_ranking[:12]  # top-12

    # 选择 GDINO 验证候选: CPS top-5 + baseline + TDM
    baseline_prompt = f'an overhead image of {class_name}'

    recs = kb.get_recommended_axes('remote_sensing', class_name)
    mods = [MODIFIER_VARIANTS_FULL[a][0] for a, _ in recs[:2]]
    tdm_prompt = baseline_prompt
    if mods:
        tdm_prompt += ', ' + ', '.join(mods)

    verify_set = [baseline_prompt, tdm_prompt]
    for p, _ in cps_top[:5]:
        if p not in verify_set:
            verify_set.append(p)

    # 对弱类: 额外加 2 个随机候选 (探索)
    if is_weak:
        rng = random.Random(hash(class_name) % 10000)
        extra = [p for p, _ in cps_top[5:]]
        if extra:
            verify_set.extend(rng.sample(extra, min(2, len(extra))))
    verify_set = list(dict.fromkeys(verify_set))[:8]

    # GDINO 验证
    gdino_results = gdino_verify_batch(class_name, verify_set, n_images=n_images)
    gdino_ranking = sorted(gdino_results.items(), key=lambda x: -x[1])

    best_prompt, best_dr = gdino_ranking[0]
    baseline_dr = gdino_results.get(baseline_prompt, 0)
    tdm_dr_val = gdino_results.get(tdm_prompt, 0)
    cps_best_prompt = cps_top[0][0]
    cps_best_dr = gdino_results.get(cps_best_prompt, 0)
    cps_best_score = cps_top[0][1]

    # 反馈数据: 每个 GDINO 验证过的 prompt
    feedback = []
    for prompt, dr in gdino_ranking:
        feedback.append({
            'class': class_name,
            'prompt': prompt,
            'dr': dr,
            'cps_score': float(dict(cps_ranking).get(prompt, 0)),
            'is_weak': is_weak,
        })

    return {
        'class': class_name,
        'baseline_dr': baseline_dr,
        'tdm_dr': tdm_dr_val,
        'cps_best_dr': cps_best_dr,
        'cps_best_score': cps_best_score,
        'cps_best_prompt': cps_best_prompt,
        'best_dr': best_dr,
        'best_prompt': best_prompt,
        'gdino_ranking': gdino_ranking,
        'feedback': feedback,
    }


# ══════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("Full Pipeline: 扩大覆盖 + 积累反馈")
    print("=" * 65)
    print(f"设备: {DEVICE}")

    from ofat_knowledge_base import OFATKnowledgeBase
    kb = OFATKnowledgeBase()

    # ── A: 弱类增强 OFAT ──
    print("\n" + "=" * 60)
    print("A: 弱类增强 OFAT (更多修饰词变体, 更多图像)")
    print("=" * 60)

    enhanced_ofat = {}
    for cls in WEAK_CLASSES:
        print(f"\n  {cls} (50 张图, 6 轴 × 6 变体 = 36 个 prompt)...")
        result = run_ofat_enhanced(cls, n_images=50)
        enhanced_ofat[cls] = result

    # 保存 A 结果
    with open(SAVE_DIR / 'enhanced_ofat.json', 'w', encoding='utf-8') as f:
        json.dump(enhanced_ofat, f, indent=2, ensure_ascii=False)
    print(f"\n  增强 OFAT 保存至: {SAVE_DIR / 'enhanced_ofat.json'}")

    # ── B: 全 20 类 CPS 搜索 + GDINO 验证 ──
    print("\n" + "=" * 60)
    print("B: 全 20 类 CPS 搜索 + GDINO 验证 → 积累反馈")
    print("=" * 60)

    scorer = CPSScorerV2()

    all_search_results = {}
    all_feedback = []

    for i, cls in enumerate(DIOR_CLASSES):
        is_weak = cls in WEAK_CLASSES
        label = "[弱类]" if is_weak else ""

        print(f"\n[{i+1:2d}/20] {cls} {label}")

        try:
            result = cps_search_and_verify(cls, scorer, kb, n_images=20, is_weak=is_weak)
            all_search_results[cls] = {
                'baseline_dr': result['baseline_dr'],
                'tdm_dr': result['tdm_dr'],
                'cps_best_dr': result['cps_best_dr'],
                'best_dr': result['best_dr'],
                'best_prompt': result['best_prompt'],
                'cps_best_prompt': result['cps_best_prompt'],
                'gdino_ranking': result['gdino_ranking'],
            }
            all_feedback.extend(result['feedback'])

            gain = result['best_dr'] - result['baseline_dr']
            print(f"    Base={result['baseline_dr']:.3f} | "
                  f"CPS={result['cps_best_dr']:.3f} | "
                  f"Best={result['best_dr']:.3f} ({gain:+.3f}) | "
                  f"N_fb={len(result['feedback'])}")
        except Exception as e:
            print(f"    错误: {e}")
            import traceback
            traceback.print_exc()

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print("全 20 类汇总")
    print(f"{'='*60}")

    n_done = len(all_search_results)
    total_base = sum(r['baseline_dr'] for r in all_search_results.values())
    total_tdm = sum(r['tdm_dr'] for r in all_search_results.values())
    total_cps = sum(r['cps_best_dr'] for r in all_search_results.values())
    total_best = sum(r['best_dr'] for r in all_search_results.values())

    print(f"  完成: {n_done}/20 类")
    print(f"  反馈数据: {len(all_feedback)} 个 (prompt, GDINO_DR) 配对")
    print(f"  Mean Baseline:  {total_base/n_done:.4f}")
    print(f"  Mean Static TDM: {total_tdm/n_done:.4f}")
    print(f"  Mean CPS Best:  {total_cps/n_done:.4f}")
    print(f"  Mean GDINO Best: {total_best/n_done:.4f}")

    # ── 合并历史 + 新数据 ──
    print(f"\n{'='*60}")
    print("合并数据池")
    print(f"{'='*60}")

    # 加载历史
    try:
        with open(SAVE_DIR / 'gdino_training_data.json', 'r', encoding='utf-8') as f:
            history = json.load(f)
        print(f"  历史数据: {len(history)} 点")
    except:
        history = []
        print(f"  历史数据: 0 点 (新)")

    # 合并 (去重)
    seen = set()
    merged = []
    for d in history + all_feedback:
        key = (d['class'], d['prompt'].lower().strip())
        if key not in seen:
            seen.add(key)
            merged.append(d)

    print(f"  合并后: {len(merged)} 点 (新增 {len(merged) - len(history)})")

    # 保存
    with open(SAVE_DIR / 'gdino_training_data_v3.json', 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"  保存至: {SAVE_DIR / 'gdino_training_data_v3.json'}")

    # 保存搜索结果
    output = {
        'enhanced_ofat': enhanced_ofat,
        'search_results': all_search_results,
        'feedback_count': len(all_feedback),
        'total_training_data': len(merged),
        'summary': {
            'n_classes': n_done,
            'mean_baseline': total_base / n_done,
            'mean_tdm': total_tdm / n_done,
            'mean_cps_best': total_cps / n_done,
            'mean_gdino_best': total_best / n_done,
        },
    }
    with open(SAVE_DIR / 'full_pipeline_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n全结果保存至: {SAVE_DIR / 'full_pipeline_results.json'}")

    print(f"\n✅ Full Pipeline 完成: {len(merged)} 训练数据点可用于重训 Verifier")
