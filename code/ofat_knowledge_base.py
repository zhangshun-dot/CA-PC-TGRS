# -*- coding: utf-8 -*-
"""OFAT 知识库 — DTDM 的查询层

封装所有已有 OFAT 实验数据，提供统一查询接口。
支持四级降级策略：
  L1: 精确 (domain, class, axis) 匹配
  L2: 域级均值 μ_domain(axis)
  L3: 跨域估计 δ_target ≈ δ_source · (1 − η_d)
  L4: universal fallback (仅 geometric + spectral)

数据源:
  - coco_ofat_results.json    → natural_scene 域, 20 COCO 类
  - ofat_heatmap_f1.json       → remote_sensing 域, 20 DIOR 类
  - token_universality_results.json → 跨域 U 分 & η_d
"""

import json
import numpy as np
from pathlib import Path

# ── 数据路径 ─────────────────────────────────────────────
BASE = Path(r'C:\Users\张三\Desktop')
COCO_OFAT_PATH = BASE / '实验数据' / 'coco_ofat_results.json'
DIOR_HEATMAP_PATH = BASE / '论文版本历史' / '初稿4_论文文件' / '实验数据' / 'ofat_heatmap_f1.json'
UNIVERSALITY_PATH = BASE / '实验数据' / 'token_universality_results.json'

AXIS_ORDER = ['granularity', 'spatial', 'geometric', 'scale', 'spectral', 'texture']

# 域标签映射
DOMAIN_ALIASES = {
    'natural_scene': 'natural_scene',
    'coco': 'natural_scene',
    'natural': 'natural_scene',
    'remote_sensing': 'remote_sensing',
    'dior': 'remote_sensing',
    'overhead': 'remote_sensing',
    'satellite': 'remote_sensing',
}


class OFATKnowledgeBase:
    """OFAT 实验数据知识库。

    Usage:
        kb = OFATKnowledgeBase()
        delta = kb.get_delta('natural_scene', 'airplane', 'scale')
        # → 预期 ΔF1
    """

    def __init__(self):
        self._load_coco()
        self._load_dior()
        self._load_universality()

        # 默认阈值（与 domain_registry.py 一致）
        self._default_threshold = 0.03

        # 校准后的阈值（可在 calibrate 时覆写）
        self._thresholds = {}  # key: (domain, axis) → float

    # ── 数据加载 ──────────────────────────────────────

    def _load_coco(self):
        with open(COCO_OFAT_PATH, 'r', encoding='utf-8') as f:
            self.coco = json.load(f)

        self.coco_deltas = {}  # (class_name, axis) → delta
        for cls_name, cls_data in self.coco['classes'].items():
            for axis in AXIS_ORDER:
                self.coco_deltas[(cls_name, axis)] = cls_data['deltas'][axis]

        self.coco_mean_deltas = {}
        for axis in AXIS_ORDER:
            self.coco_mean_deltas[axis] = self.coco['mean_deltas'][axis]

    def _load_dior(self):
        with open(DIOR_HEATMAP_PATH, 'r', encoding='utf-8') as f:
            self.dior_raw = json.load(f)

        self.dior_baseline = self.dior_raw['baseline']
        self.dior_classes = self.dior_raw['classes']

        # 计算 delta = F1_heatmap(axis) − F1_baseline
        self.dior_deltas = {}  # (class_name, axis) → delta
        for cls_name in self.dior_classes:
            base_f1 = self.dior_baseline[cls_name]
            for axis in AXIS_ORDER:
                heatmap_f1 = self.dior_raw['heatmap'][axis][cls_name]
                self.dior_deltas[(cls_name, axis)] = heatmap_f1 - base_f1

        # 域级均值
        self.dior_mean_deltas = {}
        for axis in AXIS_ORDER:
            deltas = [self.dior_deltas[(c, axis)] for c in self.dior_classes]
            self.dior_mean_deltas[axis] = np.mean(deltas)

    def _load_universality(self):
        with open(UNIVERSALITY_PATH, 'r', encoding='utf-8') as f:
            self.univ = json.load(f)

        self.eta_d = self.univ['eta_d']
        self.retention = self.univ['cross_retention']
        self.universality_scores = {}
        self.axis_classes = {}
        for axis, info in self.univ['axis_classification'].items():
            self.universality_scores[axis] = info['universality']
            self.axis_classes[axis] = info['class']
        self.universal_axes = self.univ['universal_axes']
        self.mixed_axes = self.univ['mixed_axes']
        self.domain_specific_axes = self.univ['domain_specific_axes']

    # ── 域标签规范化 ──────────────────────────────────

    def _normalize_domain(self, domain):
        """将域标签映射到标准名称。"""
        domain_lower = domain.lower().replace(' ', '_')
        return DOMAIN_ALIASES.get(domain_lower, domain_lower)

    # ── 查询接口 ──────────────────────────────────────

    def get_delta(self, domain, class_name, axis):
        """获取指定域/类/轴的期望 ΔF1。

        Args:
            domain: 'natural_scene' | 'remote_sensing' | ...
            class_name: 类名
            axis: 轴名

        Returns:
            float: 期望 ΔF1（正=推荐激活，负=避免激活）
            None: 如果域不支持

        降级策略：精确匹配 → 域均值 → 跨域估计 → universal
        """
        domain = self._normalize_domain(domain)

        # L1: 精确 (domain, class, axis)
        if domain == 'natural_scene':
            result = self.coco_deltas.get((class_name, axis))
            if result is not None:
                return result
        elif domain == 'remote_sensing':
            result = self.dior_deltas.get((class_name, axis))
            if result is not None:
                return result

        # L2: 域级均值
        domain_mean = self.get_domain_mean(domain, axis)
        if domain_mean is not None:
            return domain_mean

        # L3: 跨域估计（尝试从另一个域借用，用 η_d 折扣）
        other_domain = 'remote_sensing' if domain == 'natural_scene' else 'natural_scene'
        other_result = None
        if other_domain == 'natural_scene':
            other_result = self.coco_deltas.get((class_name, axis))
        else:
            other_result = self.dior_deltas.get((class_name, axis))

        if other_result is not None:
            # 跨域转移：符号保留，幅度折扣
            return other_result * self.retention

        # L4: universal fallback — 仅对 universal axes 返回弱正信号
        if axis in self.universal_axes:
            return 0.01  # 弱正信号，允许 CLIP 匹配驱动
        return 0.0

    def get_domain_mean(self, domain, axis):
        """获取域级别的轴均值 ΔF1。"""
        domain = self._normalize_domain(domain)
        if domain == 'natural_scene':
            return self.coco_mean_deltas.get(axis)
        elif domain == 'remote_sensing':
            return self.dior_mean_deltas.get(axis)
        return None

    def get_threshold(self, domain, axis):
        """获取轴的激活阈值。

        Args:
            domain: 域标签
            axis: 轴名

        Returns:
            float: CLIP cosine similarity 阈值
        """
        domain = self._normalize_domain(domain)
        key = (domain, axis)
        if key in self._thresholds:
            return self._thresholds[key]
        return self._default_threshold

    def set_threshold(self, domain, axis, value):
        """设置校准后的阈值。"""
        domain = self._normalize_domain(domain)
        self._thresholds[(domain, axis)] = value

    def get_universality(self, axis):
        """获取轴的跨域 Universality Score。"""
        return self.universality_scores.get(axis, 0.5)

    def get_axis_class(self, axis):
        """获取轴的分类：'UNIVERSAL' | 'MIXED' | 'DOMAIN-SPECIFIC'"""
        return self.axis_classes.get(axis, 'MIXED')

    def is_conflict_axis(self, axis, src_domain, tgt_domain):
        """判断轴在源域和目标域之间是否有符号反转（域冲突）。

        Args:
            axis: 轴名
            src_domain: 源域标签
            tgt_domain: 目标域标签

        Returns:
            bool: True 如果该轴在跨域时符号反转
        """
        src = self._normalize_domain(src_domain)
        tgt = self._normalize_domain(tgt_domain)

        src_mean = self.get_domain_mean(src, axis)
        tgt_mean = self.get_domain_mean(tgt, axis)

        if src_mean is None or tgt_mean is None:
            # 无法判断 → 检查 universality 中的 sign_match
            info = self.univ['axis_classification'].get(axis, {})
            return not info.get('sign_match', True)

        # 符号反转 = 冲突
        return (src_mean > 0 and tgt_mean < 0) or (src_mean < 0 and tgt_mean > 0)

    def is_universal_axis(self, axis):
        """轴是否为通用轴（可跨域安全使用）。"""
        return axis in self.universal_axes

    def get_recommended_axes(self, domain, class_name, include_conflict=False):
        """获取推荐激活的轴列表（基于 OFAT δ > threshold）。

        Args:
            domain: 域标签
            class_name: 类名
            include_conflict: 是否包含域冲突轴

        Returns:
            List[Tuple[str, float]]: [(axis, delta), ...], 按 delta 降序
        """
        domain = self._normalize_domain(domain)
        recommendations = []
        for axis in AXIS_ORDER:
            delta = self.get_delta(domain, class_name, axis)
            if delta is not None and delta > self.get_threshold(domain, axis):
                recommendations.append((axis, delta))
        recommendations.sort(key=lambda x: -x[1])
        return recommendations

    # ── 域列表 ────────────────────────────────────────

    @property
    def known_domains(self):
        return ['natural_scene', 'remote_sensing']

    @property
    def known_classes(self):
        """合并 COCO + DIOR 的所有类名（去重）。"""
        coco_classes = set(self.coco_deltas.keys())
        dior_classes = set(c['class_name'] for c in coco_classes) if False else set()
        # 从 coco_deltas keys 提取
        all_coco = set(k[0] for k in self.coco_deltas.keys())
        all_dior = set(k[0] for k in self.dior_deltas.keys())
        return sorted(all_coco | all_dior)


# ── 快速验证 ──
if __name__ == '__main__':
    kb = OFATKnowledgeBase()

    print("=" * 60)
    print("OFAT Knowledge Base — 快速验证")
    print("=" * 60)

    # 测试精确查询
    print("\n[L1] 精确查询:")
    for cls in ['airplane', 'dog', 'ship', 'stadium']:
        for dom in ['natural_scene', 'remote_sensing']:
            for axis in ['scale', 'spatial']:
                delta = kb.get_delta(dom, cls, axis)
                if delta is not None:
                    print(f"  δ({dom}, {cls}, {axis}) = {delta:+.4f}")

    # 测试域级均值
    print("\n[L2] 域级均值:")
    for dom in ['natural_scene', 'remote_sensing']:
        for axis in AXIS_ORDER:
            mu = kb.get_domain_mean(dom, axis)
            print(f"  μ({dom}, {axis}) = {mu:+.4f}")

    # 测试冲突检测
    print("\n[冲突检测] spatial 和 texture 跨域符号反转:")
    for axis in ['spatial', 'texture', 'geometric', 'spectral']:
        conflict = kb.is_conflict_axis(axis, 'natural_scene', 'remote_sensing')
        print(f"  {axis}: conflict={conflict}")

    # 测试推荐
    print("\n[推荐激活] COCO airplane vs DIOR airplane:")
    for dom in ['natural_scene', 'remote_sensing']:
        recs = kb.get_recommended_axes(dom, 'airplane')
        print(f"  {dom}.airplane → {recs}")

    print("\n✅ OFAT_KB 验证完成")
