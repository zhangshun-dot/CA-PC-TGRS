# -*- coding: utf-8 -*-
"""physical_priors.py — 物理先验知识库（9个类别版本）
基于物体本质属性的零样本先验，用于门控和α缩放
"""
import numpy as np

# 物体典型尺寸（米）
# 基于常识和物理属性，用于判断目标大小和特征离散度
OBJECT_SIZE = {
    'airplane': 30.0,      # 飞机，大目标
    'boat': 15.0,          # 船，中大目标
    'car': 4.5,            # 汽车，中小目标
    'bus': 12.0,           # 公交车，大目标
    'motorcycle': 2.0,     # 摩托车，小目标
    'bicycle': 1.8,        # 自行车，小目标
    'truck': 12.0,         # 卡车，大目标
    'person': 1.7,         # 人，小目标（俯视视角下尺寸更小）
    'van': 5.0,            # 厢式货车，中小目标
    # VOC类别名
    'aeroplane': 30.0,
    'motorbike': 2.0,
    'people': 1.7,
    'pedestrian': 1.7,
}

# 视角差异度（0-1）
# 【定义澄清】这是类级别的视角敏感性属性，衡量"该类别在不同视角下的视觉差异程度"
# 不是源-目标差（无向），也不是有向差
# 在遥感→自然方向上，这个属性可以预测"该类别迁移后视角差异大不大"
# 因为源域（遥感）是统一俯视，目标域（自然）视角多样，类级视角敏感性等价于源-目标差
# 在自然→遥感方向上，所有类别的视角差异都很大（从侧视到俯视），这个属性不再有区分度
# 因此门控的适用边界是：源域视角统一、目标域视角多样的场景
VIEWPOINT_DIFF = {
    'airplane': 0.3,       # 飞机，俯视和普通视角差异较小（都是看整体）
    'boat': 0.4,           # 船，差异中等
    'car': 0.7,            # 汽车，差异较大（俯视看车顶，普通视角看车身）
    'bus': 0.8,            # 公交车，差异大（俯视看车顶，普通视角看车身侧面）
    'motorcycle': 0.6,     # 摩托车，差异中等偏大
    'bicycle': 0.5,        # 自行车，差异中等
    'truck': 0.75,         # 卡车，差异较大
    'person': 0.9,         # 人，差异最大（俯视看头顶，普通视角看全身）
    'van': 0.7,            # 厢式货车，差异较大
    # VOC类别名
    'aeroplane': 0.3,
    'motorbike': 0.6,
    'people': 0.9,
    'pedestrian': 0.9,
}

# 尺寸门控阈值
# 尺寸小于阈值的目标，特征离散度高，需要校正
SIZE_THRESHOLD = 10.0  # 米

# 视角差门控阈值
# 视角差大于阈值的目标，源域原型需要外推
VIEWPOINT_THRESHOLD = 0.5


def normalize_class(cls_name):
    """标准化类别名（VOC -> COCO）"""
    mapping = {
        'aeroplane': 'airplane',
        'motorbike': 'motorcycle',
        'people': 'person',
        'pedestrian': 'person',
    }
    return mapping.get(cls_name, cls_name)


def get_object_size(cls_name):
    """获取物体典型尺寸（米）"""
    cls_norm = normalize_class(cls_name)
    return OBJECT_SIZE.get(cls_norm, 10.0)


def get_viewpoint_diff(cls_name):
    """获取视角差异度（0-1）"""
    cls_norm = normalize_class(cls_name)
    return VIEWPOINT_DIFF.get(cls_norm, 0.5)


def get_size_gate(cls_name, threshold=None):
    """尺寸门控：尺寸小于阈值 -> 开启（需要校正）"""
    if threshold is None:
        threshold = SIZE_THRESHOLD
    size = get_object_size(cls_name)
    return 1.0 if size < threshold else 0.0


def get_viewpoint_gate(cls_name, threshold=None):
    """视角差门控：视角差大于阈值 -> 开启（需要校正）"""
    if threshold is None:
        threshold = VIEWPOINT_THRESHOLD
    vp = get_viewpoint_diff(cls_name)
    return 1.0 if vp > threshold else 0.0


def get_combined_gate(cls_name, gate_type='size_viewpoint_and'):
    """组合门控
    - size: 仅尺寸门控
    - viewpoint: 仅视角差门控
    - size_viewpoint_and: 尺寸 AND 视角差（两个条件都满足才开启）
    - size_viewpoint_or: 尺寸 OR 视角差（任一条件满足就开启）
    """
    size_gate = get_size_gate(cls_name)
    vp_gate = get_viewpoint_gate(cls_name)
    if gate_type == 'size':
        return size_gate
    elif gate_type == 'viewpoint':
        return vp_gate
    elif gate_type == 'size_viewpoint_and':
        return size_gate * vp_gate  # AND
    elif gate_type == 'size_viewpoint_or':
        return min(1.0, size_gate + vp_gate)  # OR
    else:
        return 1.0


def compute_physical_alpha(cls_name, alpha_base=2.5, gamma=0.3, size_ref=10.0, use_vp=True):
    """物理先验驱动的α缩放
    α = alpha_base * (size_ref / size) ^ gamma * (viewpoint_diff if use_vp else 1)
    """
    size = get_object_size(cls_name)
    vp = get_viewpoint_diff(cls_name)
    size_factor = (size_ref / max(size, 0.1)) ** gamma
    vp_factor = vp if use_vp else 1.0
    alpha = alpha_base * size_factor * vp_factor
    return float(np.clip(alpha, 0.0, 6.0))


def get_gate_info(cls_name):
    """获取门控详细信息（用于调试和论文）"""
    cls_norm = normalize_class(cls_name)
    size = get_object_size(cls_name)
    vp = get_viewpoint_diff(cls_name)
    size_gate = get_size_gate(cls_name)
    vp_gate = get_viewpoint_gate(cls_name)
    combined = get_combined_gate(cls_name, 'size_viewpoint_and')
    return {
        'class': cls_norm,
        'size_m': size,
        'viewpoint_diff': vp,
        'size_gate': size_gate,
        'viewpoint_gate': vp_gate,
        'combined_gate': combined,
        'gate_status': '开启' if combined > 0 else '关闭',
    }


if __name__ == '__main__':
    print('=' * 80)
    print('物理先验知识库（9个类别版本）')
    print('=' * 80)
    print(f'\n{"类别":<12} {"尺寸(m)":<10} {"视角差":<10} {"尺寸门控":<10} {"视角门控":<10} {"组合门控":<10}')
    print('-' * 70)
    for cls in ['airplane', 'boat', 'car', 'bus', 'motorcycle', 'bicycle', 'truck', 'person', 'van']:
        info = get_gate_info(cls)
        print(f'{info["class"]:<12} {info["size_m"]:<10.1f} {info["viewpoint_diff"]:<10.1f} '
              f'{"开" if info["size_gate"] > 0 else "关":<10} '
              f'{"开" if info["viewpoint_gate"] > 0 else "关":<10} '
              f'{"开" if info["combined_gate"] > 0 else "关":<10}')

    print(f'\n门控开启的类别: {[cls for cls in ["airplane", "boat", "car", "bus", "motorcycle", "bicycle", "truck", "person", "van"] if get_combined_gate(cls, "size_viewpoint_and") > 0]}')
    print(f'门控关闭的类别: {[cls for cls in ["airplane", "boat", "car", "bus", "motorcycle", "bicycle", "truck", "person", "van"] if get_combined_gate(cls, "size_viewpoint_and") == 0]}')
