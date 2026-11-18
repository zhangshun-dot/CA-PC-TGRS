# -*- coding: utf-8 -*-
"""DTDM: Dynamic Token Discovery and Modulation

核心创新：用 CLIP 图像-轴属性匹配替代静态 OFAT 修饰词选择，
从类级优化升级到实例级优化。

与 CD-TDM 的关系：
  - CD-TDM (static): 每个类固定使用 OFAT 最优 top-2 修饰词
  - DTDM (dynamic): 逐图用 CLIP 判断哪些修饰词对当前图像有效

算法流程：
  1. 域识别: DomainClassifier → D
  2. CLIP 图像-属性匹配: 对每个轴 a, cos(I_emb, desc_emb[a])
  3. 决策: s_clip > τ_a(D) AND δ(D,c,a) > -0.02 AND not conflict(a,D)
  4. 构造 prompt: π_D + " " + c + ", " + [激活的修饰词]

Usage:
    from dynamic_tdm import DynamicTDM
    dtdm = DynamicTDM()
    prompt = dtdm.build_prompt(image, 'airplane')
    # → "an overhead image of airplane" (如果 CLIP 判断不需要修饰词)
    # → "an overhead image of airplane, large and prominent, with natural coloring"
"""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from PIL import Image

from dtdm_axis_descriptions import (
    AXIS_MODIFIERS, AXIS_DESCRIPTIONS, AXIS_ORDER,
    precompute_axis_embeddings,
)
from ofat_knowledge_base import OFATKnowledgeBase


@dataclass
class DTDMDecision:
    """单张图像的 DTDM 决策记录（可解释性）"""
    domain: str
    class_name: str
    prefix: str
    prompt: str
    activated_modifiers: List[str]
    all_scores: Dict[str, float]   # axis → CLIP similarity
    all_deltas: Dict[str, float]   # axis → OFAT δ prior
    decisions: Dict[str, str]      # axis → 'activated' | 'skipped(clip)' | 'skipped(delta)' | 'skipped(conflict)'


class DynamicTDM:
    """动态 Token 发现与调制。

    与 CDTDM 的接口兼容（build_prompt 签名一致），可直接替换。
    """

    def __init__(self, device=None, clip_model_path=None,
                 clip_similarity_weight=0.6, ofat_prior_weight=0.4):
        """
        Args:
            device: 'cuda' or 'cpu'. Auto-detected if None.
            clip_model_path: CLIP ViT-B/32 权重路径。
            clip_similarity_weight: CLIP 相似度在决策中的权重 [0, 1]
            ofat_prior_weight: OFAT 先验在决策中的权重 [0, 1]
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_weight = clip_similarity_weight
        self.ofat_weight = ofat_prior_weight

        # ── 加载 CLIP ──
        import clip
        self._clip = clip
        if clip_model_path is None:
            clip_model_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'

        self.clip_model, self.preprocess = clip.load(clip_model_path, device=self.device)
        self.clip_model.eval()

        # ── 预计算轴描述嵌入 ──
        self.axis_embeddings = precompute_axis_embeddings(self.clip_model, self.device)

        # ── 加载 OFAT 知识库 ──
        self.kb = OFATKnowledgeBase()

        # ── 延迟加载域分类器 ──
        self._domain_classifier = None

        # ── 域前缀映射（与 domain_registry.py 一致） ──
        self._domain_prefixes = {
            'remote_sensing': 'an overhead image of',
            'natural_scene': 'a photo of',
            'infrared': 'a thermal image of',
            'sar': 'a SAR image of',
            'underwater': 'an underwater image of',
            'medical': 'a medical image of',
        }

    # ── 属性 ──────────────────────────────────────────

    @property
    def domain_classifier(self):
        """延迟加载域分类器。"""
        if self._domain_classifier is None:
            from domain_classifier import DomainClassifier
            self._domain_classifier = DomainClassifier(
                clip_model_path=r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt',
                device=self.device
            )
        return self._domain_classifier

    # ── 核心方法 ──────────────────────────────────────

    def select_modifiers(self, image, class_name, domain_hint=None,
                         return_decision=False):
        """为单张图像动态选择修饰词。

        Args:
            image: PIL.Image (RGB)
            class_name: 目标类别名称
            domain_hint: 可选的域提示（跳过域分类）
            return_decision: 是否返回完整的 DTDMDecision

        Returns:
            List[str]: 激活的修饰词列表
            或 Tuple[List[str], DTDMDecision] 如果 return_decision=True
        """
        # Step 1: 域识别
        if domain_hint is not None:
            domain = domain_hint
        else:
            domain = self.domain_classifier.classify(image)

        # Step 2: 获取域前缀
        domain_normalized = self.kb._normalize_domain(domain)
        prefix = self._domain_prefixes.get(
            domain_normalized,
            self._domain_prefixes.get(domain, 'an image of')
        )

        # Step 3: CLIP 图像编码
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_features = self.clip_model.encode_image(img_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        # Step 4: 对每个轴计算 CLIP 相似度
        clip_scores = {}
        ofat_deltas = {}
        decisions = {}

        for axis in AXIS_ORDER:
            # CLIP 图像-轴描述相似度
            axis_emb = self.axis_embeddings[axis]
            cos_sim = torch.dot(img_features.squeeze(0), axis_emb).item()
            clip_scores[axis] = cos_sim

            # OFAT 先验
            delta = self.kb.get_delta(domain_normalized, class_name, axis)
            if delta is None:
                delta = 0.0
            ofat_deltas[axis] = delta

        # Step 5: 决策
        activated = []
        candidates = []

        for axis in AXIS_ORDER:
            s_clip = clip_scores[axis]
            delta = ofat_deltas[axis]
            threshold = self.kb.get_threshold(domain_normalized, axis)

            # 域冲突检测
            # 对于 DTDM，冲突轴需要 CLIP 信号特别强才激活
            is_conflict = self.kb.is_conflict_axis(
                axis, 'natural_scene', domain_normalized
            )
            effective_threshold = threshold + (0.15 if is_conflict else 0.0)

            # 决策规则
            if s_clip < effective_threshold:
                decisions[axis] = 'skipped(clip)'
            elif delta < -0.02:
                decisions[axis] = 'skipped(delta)'  # OFAT 确认有害
            else:
                # 通过筛选 → 候选
                combined_score = (self.clip_weight * s_clip +
                                  self.ofat_weight * max(delta, 0.0) * 5.0)
                candidates.append((axis, combined_score))
                decisions[axis] = 'candidate'

        # Step 6: 排序 + S-N 约束（top-2）
        candidates.sort(key=lambda x: -x[1])
        for axis, score in candidates[:2]:
            decisions[axis] = 'activated'
            activated.append(AXIS_MODIFIERS[axis])

        # 标记落选的 candidate
        for axis, score in candidates[2:]:
            decisions[axis] = 'skipped(s-n cap)'

        # Step 7: 构造返回
        if return_decision:
            decision = DTDMDecision(
                domain=domain,
                class_name=class_name,
                prefix=prefix,
                prompt='',  # 由 build_prompt 填充
                activated_modifiers=activated,
                all_scores=clip_scores,
                all_deltas=ofat_deltas,
                decisions=decisions,
            )
            return activated, decision

        return activated

    def build_prompt(self, image_or_domain, class_name, domain_hint=None,
                     return_decision=False):
        """构造 DTDM 优化的 prompt。

        兼容 CDTDM.build_prompt 接口：
          - image_or_domain: PIL.Image 或域标签字符串
          - class_name: 目标类别
          - domain_hint: 可选域提示

        Returns:
            str: 优化后的 prompt
            或 Tuple[str, DTDMDecision] 如果 return_decision=True
        """
        # 处理字符串域标签（复用 CDTDM 的 fallback 路径）
        if isinstance(image_or_domain, str):
            domain = image_or_domain
            domain_normalized = self.kb._normalize_domain(domain)
            prefix = self._domain_prefixes.get(
                domain_normalized,
                self._domain_prefixes.get(domain, 'an image of')
            )

            # 字符串路径：使用 OFAT 推荐作为 fallback（静态 TDM 行为）
            recs = self.kb.get_recommended_axes(domain_normalized, class_name)
            modifiers = [AXIS_MODIFIERS[a] for a, _ in recs[:2]]
            prompt = f'{prefix} {class_name}'
            if modifiers:
                prompt += ', ' + ', '.join(modifiers)
            return prompt

        # 图像路径：完整 DTDM
        modifiers, decision = self.select_modifiers(
            image_or_domain, class_name, domain_hint, return_decision=True
        )

        prompt = f'{decision.prefix} {class_name}'
        if modifiers:
            prompt += ', ' + ', '.join(modifiers)
        decision.prompt = prompt

        if return_decision:
            return prompt, decision
        return prompt

    def get_domain_prefix(self, domain):
        """获取域的默认前缀。"""
        domain_normalized = self.kb._normalize_domain(domain)
        return self._domain_prefixes.get(
            domain_normalized,
            self._domain_prefixes.get(domain, 'an image of')
        )


# ── 快速验证 ──
if __name__ == '__main__':
    import sys, os, glob

    print("=" * 60)
    print("DynamicTDM — 快速验证")
    print("=" * 60)

    print("\n[1/3] 初始化...")
    dtdm = DynamicTDM()
    print("  CLIP + OFAT_KB + DomainClassifier 加载完成")

    # 找测试图像
    dior_dir = r'C:\Users\张三\modeling\dior_data\full20_images'
    coco_dir = r'C:\Users\张三\modeling\coco_data\val2017'

    dior_images = []
    if os.path.isdir(dior_dir):
        # Flat directory, filenames encode class: airplane_2003.jpg
        class_groups = {}
        for fname in sorted(os.listdir(dior_dir))[:60]:
            if not fname.endswith('.jpg'):
                continue
            cls_name = fname.rsplit('_', 1)[0]
            if cls_name not in class_groups:
                class_groups[cls_name] = []
            class_groups[cls_name].append(os.path.join(dior_dir, fname))

        for cls_name, paths in sorted(class_groups.items())[:3]:
            dior_images.append((cls_name, paths[0]))

    print(f"\n[2/3] 测试图像 ({len(dior_images)} 张):")
    for cls_name, img_path in dior_images:
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert('RGB')
        prompt, decision = dtdm.build_prompt(img, cls_name, return_decision=True)
        print(f"\n  类: {cls_name}")
        print(f"  域: {decision.domain}")
        print(f"  Prompt: {prompt}")
        print(f"  激活: {decision.activated_modifiers}")
        print(f"  CLIP scores: { {a: f'{s:.3f}' for a, s in decision.all_scores.items()} }")
        print(f"  OFAT deltas: { {a: f'{d:+.3f}' for a, d in decision.all_deltas.items()} }")
        print(f"  Decisions:   {decision.decisions}")

    print(f"\n[3/3] 静态 fallback (domain string input):")
    for cls_name in ['airplane', 'stadium', 'ship']:
        prompt = dtdm.build_prompt('remote_sensing', cls_name)
        print(f"  remote_sensing.{cls_name} → {prompt}")

    print("\n✅ DynamicTDM 验证完成")
