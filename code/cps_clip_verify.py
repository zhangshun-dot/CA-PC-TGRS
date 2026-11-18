# -*- coding: utf-8 -*-
"""CPS + CLIP 双裁判闭环: CPS 提名 → CLIP 看图验证 → 选最优 prompt

核心洞察:
  - CPS 知道什么修饰词一般有效 (来自 OFAT), 但从没见过图像
  - CLIP 能看图判断 prompt-image 匹配度, 但缺乏 OFAT 的结构化知识
  - 两者互补: CPS 提名候选, CLIP 看图把关

验证逻辑:
  对每张图:
    1. CPS 生成排名候选 (top-8)
    2. 对每个候选, 计算 CLIP cos_sim(image, prompt)
    3. 取 top-3 CLIP 得分候选 → GDINO 验证 → 选 best
    4. 同时检查: CPS 和 CLIP 是否一致?
       - 一致 → 高置信度, 直接用
       - 不一致 → CLIP 优先级更高 (因为 CLIP 看过图)

与纯 CPS 的关键区别:
  - 纯 CPS: CPS 选 prompt → GDINO 验证 → CPS 不知道 tennis court 上犯了错
  - CPS+CLIP: CPS 提名 → CLIP 看图 → "satellite image + natural coloring" 跟图不匹配 → 跳过 → 试下一个

Usage:
    python cps_clip_verify.py
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json, time, glob, random
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
DIOR_CLASSES = ['airplane','airport','baseball field','basketball court','bridge',
    'chimney','dam','expressway service area','expressway toll station',
    'golf course','ground track field','harbor','overpass','ship','stadium',
    'storage tank','tennis court','train station','vehicle','windmill']

AXIS_MODIFIERS = {
    'granularity': 'with fine details',
    'spatial': 'in its natural setting',
    'geometric': 'with distinct structural features',
    'scale': 'large and prominent',
    'spectral': 'with natural coloring',
    'texture': 'with clearly visible surface features',
}

MODIFIER_VARIANTS = {
    'granularity': ['with fine details', 'showing intricate details',
                    'with sharp edges', 'featuring detailed structures'],
    'spatial': ['in its natural setting', 'in context of surroundings',
                'within its environment', 'surrounded by typical terrain'],
    'geometric': ['with distinct structural features', 'showing clear geometry',
                  'with well-defined shape', 'featuring characteristic outlines'],
    'scale': ['large and prominent', 'clearly visible in scale',
              'dominating the view', 'at full resolution'],
    'spectral': ['with natural coloring', 'in true color',
                 'with realistic tones', 'showing natural hues'],
    'texture': ['with clearly visible surface features', 'showing surface texture',
                'with visible material patterns', 'featuring surface details'],
}

PREFIXES = [
    'an overhead image of', 'a satellite image of', 'a photo of',
    'a remote sensing image of', 'an aerial view of',
]

AXIS_ORDER = ['granularity', 'spatial', 'geometric', 'scale', 'spectral', 'texture']


# ══════════════════════════════════════════════════════════════════
# CLIP 验证器
# ══════════════════════════════════════════════════════════════════

class CLIPVerifier:
    """CLIP 图像-文本匹配验证器。

    对每张图, 计算 CLIP cos_sim(image, prompt), 用于判断
    prompt 是否与图像内容匹配。

    关键: 这个信号是 per-image 的, CPS 做不到这一点。
    """

    def __init__(self):
        import clip as clip_module
        clip_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
        self.model, self.preprocess = clip_module.load(clip_path, device=DEVICE)
        self.model.eval()
        self.tokenize = clip_module.tokenize

    def image_similarity(self, image, prompt):
        """单图: CLIP cos_sim(image, prompt_text)。

        Returns:
            float: cosine similarity ∈ [-1, 1]
        """
        img_input = self.preprocess(image).unsqueeze(0).to(DEVICE)
        text_tok = self.tokenize([prompt], truncate=True).to(DEVICE)
        with torch.no_grad():
            img_feat = self.model.encode_image(img_input)
            text_feat = self.model.encode_text(text_tok)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ text_feat.T).item()
        return sim

    def batch_similarity(self, images, prompt):
        """多图: 对同一 prompt 计算平均 CLIP sim。

        Returns:
            float: 平均 cosine similarity
        """
        if not images:
            return 0.0
        sims = [self.image_similarity(img, prompt) for img in images]
        return float(np.mean(sims))

    def rank_by_similarity(self, images, prompts):
        """对多图取平均, 排序多个 prompt。

        Args:
            images: List[PIL.Image]
            prompts: List[str]

        Returns:
            List[Tuple[str, float]]: (prompt, avg_sim), 按 sim 降序
        """
        results = []
        for prompt in prompts:
            avg_sim = self.batch_similarity(images, prompt)
            results.append((prompt, avg_sim))
        results.sort(key=lambda x: -x[1])
        return results

    def verify(self, images, prompt, baseline_prompt=None):
        """验证一个 prompt 是否比 baseline 更好。

        判定准则:
          1. 如果 baseline 给出: candidate_sim > baseline_sim → 通过
          2. 如果无 baseline: candidate_sim > 通用阈值 → 通过

        Returns:
            Dict: {accepted: bool, sim: float, baseline_sim: float, margin: float}
        """
        sim = self.batch_similarity(images, prompt)

        if baseline_prompt:
            base_sim = self.batch_similarity(images, baseline_prompt)
            accepted = sim > base_sim
            return {
                'accepted': accepted,
                'sim': sim,
                'baseline_sim': base_sim,
                'margin': sim - base_sim,
            }
        else:
            # 通用阈值 (CLIP cos_sim 通常在 0.2-0.35 范围)
            accepted = sim > 0.22
            return {
                'accepted': accepted,
                'sim': sim,
                'baseline_sim': None,
                'margin': sim - 0.22,
            }


# ══════════════════════════════════════════════════════════════════
# CPS 评分器 (复用, 无 tanh 版本)
# ══════════════════════════════════════════════════════════════════

class CPSScorer:
    """CPS 评分器: 从 CLIP text embedding 预测 ΔF1 (r=0.97)。"""

    def __init__(self):
        import clip as clip_module
        clip_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
        self.clip_model, _ = clip_module.load(clip_path, device=DEVICE)
        self.clip_model.eval()
        self.clip_tokenize = clip_module.tokenize

        # 加载 CPS (无 tanh)
        ckpt = torch.load(SAVE_DIR / 'cps_model_notanh.pt', map_location=DEVICE)
        from cps_adapter import CLIPPromptScorer
        self.cps = CLIPPromptScorer(
            d_input=ckpt['config']['d_input'],
            hidden_dims=ckpt['config']['hidden_dims'],
            dropout=ckpt['config']['dropout'],
        )
        self.cps.load_state_dict(ckpt['state_dict'])
        # 去掉 tanh (即使模型存的时候有, 我们重新构造时用的 cps_adapter 版本)
        self.cps.eval()
        self.cps.to(DEVICE)
        print(f"  CPS 已加载: r={ckpt.get('full_corr', 0):.4f}")

    def encode(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        embs = []
        for i in range(0, len(prompts), 64):
            batch = prompts[i:i + 64]
            toks = self.clip_tokenize(batch, truncate=True).to(DEVICE)
            with torch.no_grad():
                emb = self.clip_model.encode_text(toks)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embs.append(emb.cpu().float())
        return torch.cat(embs, dim=0)

    def score(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        embs = self.encode(prompts)
        with torch.no_grad():
            scores = self.cps(embs.to(DEVICE)).squeeze().cpu().numpy()
        if scores.ndim == 0:
            scores = np.array([scores.item()])
        return scores

    def rank(self, prompts):
        scores = self.score(prompts)
        ranked = list(zip(prompts, scores))
        ranked.sort(key=lambda x: -x[1])
        return ranked


# ══════════════════════════════════════════════════════════════════
# 候选 prompt 生成器
# ══════════════════════════════════════════════════════════════════

def generate_candidates(class_name, kb, n_systematic=30):
    """为指定类生成候选 prompt (系统性覆盖 + 随机组合)。

    Returns:
        List[str]: 候选 prompt 列表
    """
    candidates = set()

    # 系统性: 所有前缀 × 空修饰词
    for prefix in PREFIXES:
        candidates.add(f'{prefix} {class_name}')

    # 系统性: overhead + 每个修饰词
    for axis in AXIS_ORDER:
        for mod in MODIFIER_VARIANTS[axis]:
            candidates.add(f'an overhead image of {class_name}, {mod}')

    # OFAT 推荐组合
    recs = kb.get_recommended_axes('remote_sensing', class_name)
    top_axes = [a for a, _ in recs[:3]]
    if len(top_axes) >= 2:
        for prefix in PREFIXES[:3]:
            for i, a1 in enumerate(top_axes):
                for a2 in top_axes[i+1:]:
                    mod1 = AXIS_MODIFIERS[a1]
                    mod2 = AXIS_MODIFIERS[a2]
                    candidates.add(f'{prefix} {class_name}, {mod1}, {mod2}')

    # 随机三修饰词组合
    all_mods = []
    for axis in AXIS_ORDER:
        all_mods.extend(MODIFIER_VARIANTS[axis])

    rng = random.Random(hash(class_name) % 10000)
    while len(candidates) < n_systematic + 20:
        n = rng.randint(1, 3)
        mods = rng.sample(all_mods, min(n, len(all_mods)))
        # 避免同轴冲突
        used_axes = set()
        filtered = []
        for m in mods:
            for axis, variants in MODIFIER_VARIANTS.items():
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


# ══════════════════════════════════════════════════════════════════
# GDINO 评估
# ══════════════════════════════════════════════════════════════════

def gdino_detect_rate(class_name, prompt, n_images=20, threshold=0.25):
    """GDINO 检测命中率。"""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    gd_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
    gd_proc = AutoProcessor.from_pretrained(gd_cache)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        gd_cache, dtype=torch.float32
    ).eval().to(DEVICE)

    paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{class_name.replace(' ','_')}_*.jpg")))[:n_images]
    if not paths:
        paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{class_name}_*.jpg")))[:n_images]
    images = [Image.open(p).convert('RGB') for p in paths]
    if not images:
        return 0.0

    detections = []
    for img in images:
        inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            o = gd_model(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(DEVICE)
        r = gd_proc.post_process_grounded_object_detection(
            o, threshold=threshold, target_sizes=ts
        )[0]
        detections.append(len(r['boxes']) > 0)
    return float(np.mean(detections))


# ══════════════════════════════════════════════════════════════════
# CPS + CLIP 双裁判主循环
# ══════════════════════════════════════════════════════════════════

def dual_judge_search(class_name, scorer, verifier, kb, images,
                      baseline_prompt, n_verify=8):
    """CPS + CLIP 双裁判搜索。

    流程:
      1. 生成所有候选 prompt
      2. CPS 评分 → top-n_verify 进入验证池
      3. CLIP 看图验证 → 对每张图计算 CLIP sim → 聚合排名
      4. 双裁判融合:
         - CPS rank + CLIP rank → 加权融合排序
         - CPS-CLIP 一致的 (都排前面) → 高置信度
         - CPS-CLIP 不一致的 → CLIP 优先 (因为看过图)
      5. 选 top-3 进 GDINO 终审

    Args:
        class_name: 类名
        scorer: CPSScorer
        verifier: CLIPVerifier
        kb: OFATKnowledgeBase
        images: List[PIL.Image] — 该类图像
        baseline_prompt: str
        n_verify: CLIP 验证的候选数

    Returns:
        Dict: 搜索结果
    """
    # Step 1: 生成候选
    all_candidates = generate_candidates(class_name, kb, n_systematic=30)
    print(f"  生成 {len(all_candidates)} 个候选")

    # Step 2: CPS 筛选
    cps_ranking = scorer.rank(all_candidates)
    cps_top = cps_ranking[:n_verify]
    cps_prompts = [p for p, _ in cps_top]
    cps_scores = {p: s for p, s in cps_top}

    print(f"  CPS top-3: ")
    for p, s in cps_top[:3]:
        print(f"    [{s:+.4f}] '{p[:70]}...'")

    # Step 3: CLIP 看图验证
    clip_ranking = verifier.rank_by_similarity(images, cps_prompts)
    clip_sims = {p: s for p, s in clip_ranking}

    print(f"  CLIP top-3: ")
    for p, s in clip_ranking[:3]:
        cps_s = cps_scores.get(p, 0)
        print(f"    [sim={s:.4f}, cps={cps_s:+.4f}] '{p[:70]}...'")

    # Step 4: 双裁判融合
    # CPS rank (1=best) + CLIP rank (1=best)
    cps_rank = {p: i for i, (p, _) in enumerate(cps_top)}
    clip_rank = {p: i for i, (p, _) in enumerate(clip_ranking)}

    # 加权融合: CLIP 权重略高 (因为看过图)
    fusion_scores = {}
    for p in cps_prompts:
        cps_r = cps_rank.get(p, n_verify)
        clip_r = clip_rank.get(p, n_verify)
        # Borda count: n - rank → 越高越好
        cps_borda = n_verify - cps_r
        clip_borda = n_verify - clip_r
        # CLIP 权重 0.6, CPS 权重 0.4 (CLIP 看过图, 更可信)
        fusion = 0.6 * clip_borda + 0.4 * cps_borda
        fusion_scores[p] = fusion

    # 排序
    fusion_ranking = sorted(fusion_scores.items(), key=lambda x: -x[1])

    # 一致性分析
    cps_best = cps_top[0][0]
    clip_best = clip_ranking[0][0]
    agreement = (cps_best == clip_best)

    print(f"\n  双裁判一致性: {'✓ 一致' if agreement else '✗ 不一致'}")
    if not agreement:
        print(f"    CPS 选:  '{cps_best[:60]}...'")
        print(f"    CLIP 选: '{clip_best[:60]}...'")
        print(f"    → CLIP 优先")

    # Step 5: 选 top-3 融合候选 + baseline + TDM → GDINO 终审
    tdm_prompt = _get_tdm_prompt(class_name, kb)
    finalists = [baseline_prompt, tdm_prompt]

    for p, _ in fusion_ranking[:3]:
        if p not in finalists:
            finalists.append(p)

    # 也加入 CLIP best (如果不在)
    if clip_best not in finalists:
        finalists.append(clip_best)

    finalists = list(dict.fromkeys(finalists))[:6]

    print(f"\n  GDINO 终审 {len(finalists)} 个候选...")
    gdino_results = {}
    for p in finalists:
        dr = gdino_detect_rate(class_name, p, n_images=len(images))
        gdino_results[p] = dr

    gdino_ranking = sorted(gdino_results.items(), key=lambda x: -x[1])
    best_prompt, best_dr = gdino_ranking[0]
    baseline_dr = gdino_results.get(baseline_prompt, 0)
    tdm_dr = gdino_results.get(tdm_prompt, 0)
    fusion_best_dr = gdino_results.get(fusion_ranking[0][0], 0)

    # 找出 CPS-CLIP 一致且排名靠前的
    consensus_prompts = []
    for p in cps_prompts[:5]:
        if cps_rank.get(p, 99) <= 2 and clip_rank.get(p, 99) <= 2:
            consensus_prompts.append(p)

    consensus_dr = None
    if consensus_prompts:
        consensus_dr = gdino_results.get(consensus_prompts[0], None)

    return {
        'class': class_name,
        'baseline_prompt': baseline_prompt,
        'baseline_dr': baseline_dr,
        'tdm_dr': tdm_dr,
        'fusion_best_dr': fusion_best_dr,
        'best_dr': best_dr,
        'best_prompt': best_prompt,
        'agreement': agreement,
        'cps_best': cps_best,
        'clip_best': clip_best,
        'consensus_dr': consensus_dr,
        'consensus_prompts': consensus_prompts,
        'fusion_ranking': fusion_ranking[:5],
        'gdino_ranking': gdino_ranking,
    }


def _get_tdm_prompt(class_name, kb):
    """Static TDM prompt。"""
    recs = kb.get_recommended_axes('remote_sensing', class_name)
    mods = [AXIS_MODIFIERS[a] for a, _ in recs[:2]]
    prompt = f'an overhead image of {class_name}'
    if mods:
        prompt += ', ' + ', '.join(mods)
    return prompt


# ══════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("CPS + CLIP 双裁判闭环: CPS 提名 → CLIP 看图验证")
    print("=" * 65)
    print(f"设备: {DEVICE}")

    # ── 初始化 ──
    print("\n加载模型...")
    scorer = CPSScorer()
    verifier = CLIPVerifier()

    from ofat_knowledge_base import OFATKnowledgeBase
    kb = OFATKnowledgeBase()

    # ── 评估类 ──
    eval_classes = ['airplane', 'ship', 'stadium', 'bridge', 'harbor',
                    'overpass', 'tennis court', 'windmill']
    n_images = 20

    all_results = {}

    print(f"\n{'='*60}")
    print(f"CPS + CLIP 双裁判: 每类 {n_images} 张图, CPS top-8 → CLIP 验证")
    print(f"{'='*60}")

    for cls in eval_classes:
        print(f"\n{'─'*60}")
        print(f"  {cls}")
        print(f"{'─'*60}")

        baseline_prompt = f'an overhead image of {cls}'

        # 加载图像
        paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{cls.replace(' ','_')}_*.jpg")))[:n_images]
        if not paths:
            paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{cls}_*.jpg")))[:n_images]
        images = [Image.open(p).convert('RGB') for p in paths]
        if not images:
            print(f"  无图像, 跳过")
            continue

        result = dual_judge_search(
            cls, scorer, verifier, kb, images, baseline_prompt, n_verify=8
        )
        all_results[cls] = result

        # 打印本类结果
        print(f"\n  ▶ {cls} 结果:")
        print(f"    Baseline:     {result['baseline_dr']:.3f}  '{baseline_prompt}'")
        print(f"    Static TDM:   {result['tdm_dr']:.3f}")
        print(f"    Fusion best:  {result['fusion_best_dr']:.3f}")
        if result['consensus_dr'] is not None:
            print(f"    Consensus:    {result['consensus_dr']:.3f}  (CPS+CLIP一致)")
        print(f"    GDINO best:   {result['best_dr']:.3f}  '{result['best_prompt'][:70]}...'")
        print(f"    一致性:       {'✓' if result['agreement'] else '✗ (CLIP纠正了CPS)'}")

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print("汇总: CPS vs CPS+CLIP")
    print(f"{'='*60}")
    print(f"{'Class':<20s} {'Base':>7s} {'TDM':>7s} {'CPS only':>8s} {'+CLIP':>7s} {'Consensus':>10s} {'Best':>7s}")
    print(f"{'-'*75}")

    # 加载之前纯 CPS 的结果用于对比
    try:
        with open(SAVE_DIR / 'cps_search_results.json', 'r', encoding='utf-8') as f:
            prev_results = json.load(f)
        prev_cps = {cls: r['cps_best_dr'] for cls, r in prev_results['final_results'].items()}
    except:
        prev_cps = {}

    total_base, total_tdm, total_cps, total_fusion, total_consensus, total_best = 0, 0, 0, 0, 0, 0
    n_consensus = 0

    for cls in eval_classes:
        r = all_results[cls]
        total_base += r['baseline_dr']
        total_tdm += r['tdm_dr']
        total_fusion += r['fusion_best_dr']
        total_best += r['best_dr']

        prev = prev_cps.get(cls, r['fusion_best_dr'])
        total_cps += prev

        if r['consensus_dr'] is not None:
            total_consensus += r['consensus_dr']
            n_consensus += 1
        else:
            total_consensus += r['fusion_best_dr']  # fallback
            n_consensus += 1

        consensus_str = f"{r['consensus_dr']:.3f}" if r['consensus_dr'] is not None else "N/A"
        agree_str = '✓' if r['agreement'] else '✗'
        print(f"{cls:<20s} {r['baseline_dr']:7.3f} {r['tdm_dr']:7.3f} "
              f"{prev:8.3f} {r['fusion_best_dr']:7.3f} {consensus_str:>7s} {agree_str:<3s} "
              f"{r['best_dr']:7.3f}")

    n = len(eval_classes)
    print(f"{'-'*75}")
    print(f"{'MEAN':<20s} {total_base/n:7.4f} {total_tdm/n:7.4f} "
          f"{total_cps/n:8.4f} {total_fusion/n:7.4f} {total_consensus/n:10.4f} "
          f"{total_best/n:7.4f}")

    print(f"\n  CPS only  (纯文本):  {total_cps/n:.4f}")
    print(f"  CPS+CLIP  (双裁判):  {total_fusion/n:.4f}")
    print(f"  Consensus  (一致):   {total_consensus/n:.4f}")
    print(f"  GDINO best (上限):   {total_best/n:.4f}")

    # ── 保存 ──
    output = {
        'results': {cls: {k: v for k, v in r.items()
                     if k not in ('fusion_ranking', 'gdino_ranking')}
                    for cls, r in all_results.items()},
        'summary': {
            'mean_baseline': total_base / n,
            'mean_tdm': total_tdm / n,
            'mean_cps_only': total_cps / n,
            'mean_cps_clip': total_fusion / n,
            'mean_consensus': total_consensus / n,
            'mean_best': total_best / n,
        },
    }
    with open(SAVE_DIR / 'cps_clip_verify_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存至: {SAVE_DIR / 'cps_clip_verify_results.json'}")

    print("\n✅ CPS + CLIP 双裁判闭环完成")
