# -*- coding: utf-8 -*-
"""CPS-Guided Prompt Search: 用 CPS 评分器驱动进化搜索, 迭代优化 prompt 文本

核心闭环:
  CPS(你的 OFAT 训练) → 评分候选 prompt → 进化搜索(变异/交叉) → 新候选 → CPS 评分 → 迭代
  最终 top-3 用 GDINO 验证

与 GLOV 的同构:
  GLOV: LLM生成 → CLIP零样本评估 → 排名反馈 → LLM再生成
  我们:  进化生成 → CPS(OFAT训练)评估 → 排名反馈 → 进化再生成

优势:
  - 完全本地, 不需要 LLM API
  - CPS 在 OFAT 数据上 r=0.96, 比 CLIP 零样本准确
  - 进化搜索直接优化 GDINO 检测率, 不像 GLOV 优化 CLIP 分类

Pipeline:
  Round 0: 系统生成候选 (前缀×修饰词×轴组合)
  Round 1-N: 变异 + 交叉 → CPS 评分 → top-K 存活
  最终: top-3 候选 GDINO 验证

Usage:
    python cps_search.py
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json, time, glob, random, copy
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
SAVE_DIR.mkdir(exist_ok=True)

# ── 搜索空间定义 ──────────────────────────────────────
PREFIXES = [
    'a photo of', 'an image of', 'a picture of',
    'an overhead image of', 'a satellite image of',
    'an aerial view of', 'a remote sensing image of',
    'a high-resolution overhead view of',
]

AXIS_MODIFIERS = {
    'granularity': 'with fine details',
    'spatial': 'in its natural setting',
    'geometric': 'with distinct structural features',
    'scale': 'large and prominent',
    'spectral': 'with natural coloring',
    'texture': 'with clearly visible surface features',
}

# 扩展修饰词库 (同轴变体, 用于变异)
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

AXIS_ORDER = ['granularity', 'spatial', 'geometric', 'scale', 'spectral', 'texture']

DIOR_CLASSES = ['airplane','airport','baseball field','basketball court','bridge',
    'chimney','dam','expressway service area','expressway toll station',
    'golf course','ground track field','harbor','overpass','ship','stadium',
    'storage tank','tennis court','train station','vehicle','windmill']

DIOR_DIR = r'C:\Users\张三\modeling\dior_data\full20_images'


# ══════════════════════════════════════════════════════════════════
# CPS 评分器 (复用已训练的)
# ══════════════════════════════════════════════════════════════════

class CLIPPromptScorer(nn.Module):
    """CLIP Prompt Scorer — 从 CLIP text embedding 预测 ΔF1。"""
    def __init__(self, d_input=512, hidden_dims=[256, 128, 64], dropout=0.15):
        super().__init__()
        layers = []
        prev = d_input
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h),
                          nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # 无 tanh — 保留 r=0.96 的排序能力


class CPSScorer:
    """CPS 评分器封装: 加载模型 + CLIP, 提供 batch 评分接口。"""

    def __init__(self, cps_path=None):
        import clip as clip_module
        clip_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
        self.clip_model, _ = clip_module.load(clip_path, device=DEVICE)
        self.clip_model.eval()
        self.clip_tokenize = clip_module.tokenize

        # 加载 CPS
        if cps_path is None:
            cps_path = SAVE_DIR / 'cps_model_notanh.pt'
        ckpt = torch.load(cps_path, map_location=DEVICE)
        self.cps = CLIPPromptScorer(
            d_input=ckpt['config']['d_input'],
            hidden_dims=ckpt['config']['hidden_dims'],
            dropout=ckpt['config']['dropout'],
        )
        self.cps.load_state_dict(ckpt['state_dict'])
        self.cps.eval()
        self.cps.to(DEVICE)
        full_corr = ckpt.get('full_corr', None)
        if full_corr is not None:
            print(f"  CPS 已加载: r={full_corr:.4f}")
        else:
            print("  CPS 已加载 (corr unknown)")

    def encode(self, prompts):
        """批量编码 prompt 为 CLIP 嵌入。"""
        if isinstance(prompts, str):
            prompts = [prompts]
        embeddings = []
        for i in range(0, len(prompts), 64):
            batch = prompts[i:i + 64]
            toks = self.clip_tokenize(batch, truncate=True).to(DEVICE)
            with torch.no_grad():
                emb = self.clip_model.encode_text(toks)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().float())
        return torch.cat(embeddings, dim=0)

    def score(self, prompts):
        """评分一批 prompt → ΔF1 预测值。

        Args:
            prompts: str 或 List[str]

        Returns:
            np.ndarray: 预测 ΔF1 (越高越好)
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        embs = self.encode(prompts)
        with torch.no_grad():
            scores = self.cps(embs.to(DEVICE)).squeeze().cpu().numpy()
        if scores.ndim == 0:
            scores = np.array([scores.item()])
        return scores

    def rank(self, prompts, top_k=None):
        """对 prompt 列表评分并排序。

        Returns:
            List[Tuple[str, float]]: (prompt, score), 按分数降序
        """
        scores = self.score(prompts)
        ranked = list(zip(prompts, scores))
        ranked.sort(key=lambda x: -x[1])
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked


# ══════════════════════════════════════════════════════════════════
# Prompt 进化引擎
# ══════════════════════════════════════════════════════════════════

@dataclass
class PromptGenome:
    """Prompt 的 "基因组" 表示, 支持变异和交叉。"""
    prefix: str = 'an overhead image of'
    modifiers: List[str] = field(default_factory=list)  # 选中的修饰词列表
    class_name: str = ''

    def to_prompt(self):
        """基因组 → prompt 文本。"""
        base = f'{self.prefix} {self.class_name}'
        if self.modifiers:
            base += ', ' + ', '.join(self.modifiers)
        return base

    def copy(self):
        return PromptGenome(prefix=self.prefix,
                           modifiers=list(self.modifiers),
                           class_name=self.class_name)


class PromptEvolution:
    """CPS-guided 进化搜索引擎。

    搜索空间:
      - 前缀: PREFIXES (8种)
      - 修饰词: 每轴 4 个变体 × 6 轴 = 24 个可选修饰词
      - 组合: 1-3 个修饰词

    操作:
      - 变异: 换前缀 / 换/增/删修饰词
      - 交叉: 两个父代的修饰词混合
      - 选择: CPS 评分 top-K 存活
    """

    def __init__(self, scorer, class_name, population_size=30, top_k=10,
                 mutation_rate=0.3, crossover_rate=0.5, seed=42):
        self.scorer = scorer
        self.class_name = class_name
        self.population_size = population_size
        self.top_k = top_k
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = random.Random(seed)

        # 所有可选修饰词
        self.all_modifiers = []
        for axis in AXIS_ORDER:
            self.all_modifiers.extend(MODIFIER_VARIANTS[axis])

        # 修饰词→轴映射
        self.modifier_to_axis = {}
        for axis, variants in MODIFIER_VARIANTS.items():
            for v in variants:
                self.modifier_to_axis[v] = axis

    def _random_genome(self):
        """随机生成一个基因组。"""
        prefix = self.rng.choice(PREFIXES)
        n_mods = self.rng.randint(0, 3)
        mods = self.rng.sample(self.all_modifiers, min(n_mods, len(self.all_modifiers)))
        return PromptGenome(prefix=prefix, modifiers=mods,
                           class_name=self.class_name)

    def initialize_population(self):
        """初始化种群: 随机 + 系统性覆盖。"""
        population = []

        # 系统性覆盖: 每种前缀 + 空修饰词
        for prefix in PREFIXES:
            population.append(PromptGenome(
                prefix=prefix, modifiers=[], class_name=self.class_name
            ))

        # 系统性覆盖: overhead prefix + 单修饰词
        for mod in self.all_modifiers:
            population.append(PromptGenome(
                prefix='an overhead image of', modifiers=[mod],
                class_name=self.class_name
            ))

        # 随机填充
        while len(population) < self.population_size:
            g = self._random_genome()
            if g not in population:
                population.append(g)

        return population[:self.population_size]

    def mutate(self, genome):
        """变异操作。

        类型 (随机选一):
          1. 换前缀 (20%)
          2. 换一个修饰词为同轴变体 (30%)
          3. 加一个修饰词 (如果不冲突, 20%)
          4. 删一个修饰词 (如果有, 20%)
          5. 换一个修饰词为跨轴随机 (10%)
        """
        g = genome.copy()
        op = self.rng.random()

        # 换前缀
        if op < 0.2:
            g.prefix = self.rng.choice(PREFIXES)

        # 换修饰词为同轴变体
        elif op < 0.5 and g.modifiers:
            idx = self.rng.randrange(len(g.modifiers))
            old_mod = g.modifiers[idx]
            axis = self.modifier_to_axis.get(old_mod)
            if axis:
                variants = MODIFIER_VARIANTS[axis]
                new_mod = self.rng.choice([v for v in variants if v != old_mod])
                g.modifiers[idx] = new_mod

        # 加修饰词 (避免同轴冲突)
        elif op < 0.7 and len(g.modifiers) < 3:
            current_axes = {self.modifier_to_axis.get(m) for m in g.modifiers}
            available = [m for m in self.all_modifiers
                        if self.modifier_to_axis.get(m) not in current_axes]
            if available:
                g.modifiers.append(self.rng.choice(available))

        # 删修饰词
        elif op < 0.9 and g.modifiers:
            idx = self.rng.randrange(len(g.modifiers))
            g.modifiers.pop(idx)

        # 跨轴随机换
        else:
            if g.modifiers:
                idx = self.rng.randrange(len(g.modifiers))
                g.modifiers[idx] = self.rng.choice(self.all_modifiers)
            else:
                g.modifiers.append(self.rng.choice(self.all_modifiers))

        return g

    def crossover(self, g1, g2):
        """交叉: 从两个父代混合修饰词 (不重复轴)。"""
        child = PromptGenome(
            prefix=self.rng.choice([g1.prefix, g2.prefix]),
            modifiers=[],
            class_name=self.class_name,
        )

        # 合并两个父代的修饰词, 去重, 去同轴冲突
        all_mods = list(set(g1.modifiers + g2.modifiers))
        used_axes = set()
        for mod in all_mods:
            axis = self.modifier_to_axis.get(mod)
            if axis not in used_axes:
                child.modifiers.append(mod)
                used_axes.add(axis)
            if len(child.modifiers) >= 3:
                break

        # 如果修饰词太少, 随机补
        while len(child.modifiers) < min(len(g1.modifiers), len(g2.modifiers)):
            available = [m for m in self.all_modifiers
                        if self.modifier_to_axis.get(m) not in used_axes]
            if not available:
                break
            mod = self.rng.choice(available)
            child.modifiers.append(mod)
            used_axes.add(self.modifier_to_axis.get(mod))

        return child

    def evolve(self, n_generations=5):
        """运行进化搜索。

        Args:
            n_generations: 进化代数

        Returns:
            Dict: 搜索历史和最优结果
        """
        population = self.initialize_population()
        history = []

        for gen in range(n_generations):
            # 评分当前种群
            prompts = [g.to_prompt() for g in population]
            # 去重评分
            unique_prompts = list(set(prompts))
            scores = self.scorer.score(unique_prompts)
            prompt_to_score = dict(zip(unique_prompts, scores))

            # 排序
            ranked = [(g, prompt_to_score.get(g.to_prompt(), -999))
                     for g in population]
            ranked.sort(key=lambda x: -x[1])

            top = ranked[:self.top_k]
            best_genome, best_score = top[0]
            avg_score = np.mean([s for _, s in top])
            diversity = len(set(g.to_prompt() for g, _ in top))

            history.append({
                'generation': gen,
                'best_prompt': best_genome.to_prompt(),
                'best_score': float(best_score),
                'avg_top_score': float(avg_score),
                'diversity': diversity,
            })

            print(f"  Gen {gen}: best='{best_genome.to_prompt()[:80]}...' "
                  f"score={best_score:+.4f} | avg_top={avg_score:+.4f} | "
                  f"diversity={diversity}/{self.top_k}")

            if gen == n_generations - 1:
                break

            # 生成下一代
            new_population = []

            # 精英保留
            elite_count = max(2, self.top_k // 4)
            for g, _ in top[:elite_count]:
                new_population.append(g.copy())

            # 填充
            while len(new_population) < self.population_size:
                if self.rng.random() < self.crossover_rate and len(top) >= 2:
                    # 交叉
                    parents = self.rng.sample(top[:self.top_k], 2)
                    child = self.crossover(parents[0][0], parents[1][0])
                else:
                    # 变异
                    parent = self.rng.choice(top[:self.top_k])[0]
                    child = self.mutate(parent)

                # 变异后可能还需要变异
                if self.rng.random() < self.mutation_rate:
                    child = self.mutate(child)

                if child not in new_population:
                    new_population.append(child)

            population = new_population[:self.population_size]

        # 最终: 对所有出现过的 prompt 排序, 返回 top-N
        all_prompts = set()
        for g in population:
            all_prompts.add(g.to_prompt())
        final_ranking = self.scorer.rank(list(all_prompts))

        return {
            'class': self.class_name,
            'history': history,
            'final_ranking': [(p, float(s)) for p, s in final_ranking[:10]],
            'best_prompt': final_ranking[0][0] if final_ranking else '',
            'best_score': float(final_ranking[0][1]) if final_ranking else 0.0,
        }


# ══════════════════════════════════════════════════════════════════
# GDINO 验证
# ══════════════════════════════════════════════════════════════════

def validate_with_gdino(class_name, candidate_prompts, n_images=20, threshold=0.25):
    """用 GDINO 验证一组候选 prompt。

    Args:
        class_name: 类名
        candidate_prompts: List[str] — 待验证的 prompt 候选
        n_images: 每类图像数
        threshold: GDINO 检测阈值

    Returns:
        Dict: {prompt: detection_rate}
    """
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    gd_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
    gd_proc = AutoProcessor.from_pretrained(gd_cache)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        gd_cache, dtype=torch.float32
    ).eval().to(DEVICE)

    # 加载图像
    paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{class_name.replace(' ','_')}_*.jpg")))[:n_images]
    if not paths:
        paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{class_name}_*.jpg")))[:n_images]
    images = [Image.open(p).convert('RGB') for p in paths]
    if not images:
        return {}

    results = {}
    for prompt in candidate_prompts:
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
        results[prompt] = float(np.mean(detections))

    return results


# ══════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("CPS-Guided Prompt Search — 进化搜索 + GDINO 闭环")
    print("=" * 65)
    print(f"设备: {DEVICE}")

    # ── 加载 CPS ──
    print("\n加载 CPS 评分器...")
    scorer = CPSScorer()

    # ── 选择评估类 ──
    eval_classes = ['airplane', 'ship', 'stadium', 'bridge', 'harbor',
                    'overpass', 'tennis court', 'windmill']

    # ── 基线 ──
    print("\n" + "=" * 60)
    print('基线: 原始 "an overhead image of {cls}"')
    print("=" * 60)
    baseline_prompts = {cls: f'an overhead image of {cls}' for cls in eval_classes}
    # 批量 GDINO (逐类)
    baseline_results = {}
    for cls in eval_classes:
        dr = validate_with_gdino(cls, [baseline_prompts[cls]], n_images=20)
        baseline_results[cls] = list(dr.values())[0] if dr else 0.0
        print(f"  {cls:<20s} baseline DR={baseline_results[cls]:.3f}")

    # ── Static TDM 基线 ──
    from ofat_knowledge_base import OFATKnowledgeBase
    kb = OFATKnowledgeBase()

    print("\n" + "=" * 60)
    print("Static TDM 基线")
    print("=" * 60)
    tdm_prompts = {}
    tdm_results = {}
    for cls in eval_classes:
        recs = kb.get_recommended_axes('remote_sensing', cls)
        mods = [AXIS_MODIFIERS[a] for a, _ in recs[:2]]
        tdm_prompt = f'an overhead image of {cls}'
        if mods:
            tdm_prompt += ', ' + ', '.join(mods)
        tdm_prompts[cls] = tdm_prompt
        dr = validate_with_gdino(cls, [tdm_prompt], n_images=20)
        tdm_results[cls] = list(dr.values())[0] if dr else 0.0
        print(f"  {cls:<20s} TDM DR={tdm_results[cls]:.3f}  "
              f"({'+' if tdm_results[cls] > baseline_results[cls] else ''}"
              f"{tdm_results[cls] - baseline_results[cls]:+.3f})")

    # ── CPS 进化搜索 ──
    print("\n" + "=" * 60)
    print("CPS 进化搜索 (每类独立进化)")
    print("=" * 60)

    all_evolution_results = {}
    cps_best_prompts = {}
    cps_best_scores = {}

    for cls in eval_classes:
        print(f"\n── {cls} ──")
        evo = PromptEvolution(scorer, cls, population_size=30, top_k=10,
                             mutation_rate=0.35, crossover_rate=0.5, seed=hash(cls) % 10000)
        result = evo.evolve(n_generations=6)
        all_evolution_results[cls] = result
        cps_best_prompts[cls] = result['best_prompt']
        cps_best_scores[cls] = result['best_score']

    # ── GDINO 验证 CPS 选出的最佳 prompt ──
    print("\n" + "=" * 60)
    print("GDINO 端到端验证: CPS 进化搜索 best prompt")
    print("=" * 60)

    # 对每个类, 测试 top-3 CPS 候选 + baseline + TDM
    final_results = {}

    for cls in eval_classes:
        # 收集候选
        candidates = [baseline_prompts[cls], tdm_prompts[cls]]

        # CPS top-3
        evo_result = all_evolution_results[cls]
        for prompt, score in evo_result['final_ranking'][:3]:
            if prompt not in candidates:
                candidates.append(prompt)

        # 也加入 CPS 对所有 candidate 评分最高的 (可能不在进化结果里)
        all_systematic = []
        for prefix in ['an overhead image of', 'a satellite image of',
                       'a remote sensing image of']:
            all_systematic.append(f'{prefix} {cls}')
            for axis in AXIS_ORDER:
                for mod in MODIFIER_VARIANTS[axis]:
                    all_systematic.append(f'{prefix} {cls}, {mod}')

        # CPS 评分所有系统性候选, 取 top-2 不在 candidates 里的
        systematic_scores = scorer.rank(all_systematic)
        added = 0
        for prompt, score in systematic_scores:
            if prompt not in candidates:
                candidates.append(prompt)
                added += 1
                if added >= 2:
                    break

        # 去重
        candidates = list(dict.fromkeys(candidates))  # 保持顺序去重
        candidates = candidates[:8]  # 最多 8 个候选

        # GDINO 验证
        print(f"\n  {cls}: 测试 {len(candidates)} 个候选...")
        gdino_scores = validate_with_gdino(cls, candidates, n_images=20)

        # 按 GDINO DR 排序
        ranked = sorted(gdino_scores.items(), key=lambda x: -x[1])
        best_prompt, best_dr = ranked[0]

        final_results[cls] = {
            'baseline_dr': baseline_results[cls],
            'tdm_dr': tdm_results[cls],
            'cps_best_dr': gdino_scores.get(cps_best_prompts[cls], 0),
            'gdino_best_dr': best_dr,
            'gdino_best_prompt': best_prompt,
            'all_tested': ranked,
            'cps_best_prompt': cps_best_prompts[cls],
            'cps_score': cps_best_scores[cls],
        }

        print(f"    Baseline:     {baseline_results[cls]:.3f}  '{baseline_prompts[cls]}'")
        print(f"    Static TDM:   {tdm_results[cls]:.3f}  '{tdm_prompts[cls][:60]}...'")
        print(f"    CPS best:     {final_results[cls]['cps_best_dr']:.3f}  "
              f"'{cps_best_prompts[cls][:60]}...'")
        print(f"    GDINO best:   {best_dr:.3f}  '{best_prompt[:60]}...'")

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    print(f"{'Class':<20s} {'Baseline':>8s} {'TDM':>8s} {'CPS':>8s} {'Best':>8s} {'Gain':>8s}")
    print(f"{'-'*60}")

    total_base, total_tdm, total_cps, total_best = 0, 0, 0, 0
    for cls in eval_classes:
        r = final_results[cls]
        total_base += r['baseline_dr']
        total_tdm += r['tdm_dr']
        total_cps += r['cps_best_dr']
        total_best += r['gdino_best_dr']
        gain = r['gdino_best_dr'] - r['baseline_dr']
        print(f"{cls:<20s} {r['baseline_dr']:8.3f} {r['tdm_dr']:8.3f} "
              f"{r['cps_best_dr']:8.3f} {r['gdino_best_dr']:8.3f} {gain:+8.3f}")

    n = len(eval_classes)
    print(f"{'-'*60}")
    print(f"{'MEAN':<20s} {total_base/n:8.4f} {total_tdm/n:8.4f} "
          f"{total_cps/n:8.4f} {total_best/n:8.4f} {(total_best-total_base)/n:+8.4f}")

    # ── 保存结果 ──
    output = {
        'baseline_results': baseline_results,
        'tdm_results': tdm_results,
        'final_results': final_results,
        'evolution_results': {
            cls: {
                'history': r['history'],
                'final_ranking': r['final_ranking'],
                'best_prompt': r['best_prompt'],
            }
            for cls, r in all_evolution_results.items()
        },
        'summary': {
            'mean_baseline': total_base / n,
            'mean_tdm': total_tdm / n,
            'mean_cps_best': total_cps / n,
            'mean_gdino_best': total_best / n,
            'mean_gain': (total_best - total_base) / n,
        },
    }

    out_path = SAVE_DIR / 'cps_search_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存至: {out_path}")

    print("\n✅ CPS 进化搜索完成")
