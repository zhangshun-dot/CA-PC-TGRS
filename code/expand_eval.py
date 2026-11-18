# -*- coding: utf-8 -*-
"""expand_eval.py — 主表扩类: 6 → 12 类（复用 gain_decompose 精确分解协议）

COCO 跨域 6 类: airplane, boat, car, truck, bus, motorcycle
  · truck/bus/motorcycle 共享 DIOR 'vehicle' 源原型（粗粒度源类 → 细粒度目标类，
    遥感 vehicle 泛化到自然场景的 car/truck/bus/motorcycle，更贴近真实 OVD 需求）
DIOR 同域 6 类: airplane, ship, vehicle, harbor, storage tank, bridge

协议 / 打分 / telescoping 分解 与 gain_decomposition.json 完全一致
  (L12 / K=10 / α=0.5 / NEG_LAM=0.5 / 前10校准后90测试 / pre-filter 0.03 / 数据集级 AP50)
输出 -> final_results_12class.json（不覆盖原 gain_decomposition.json）

前置提示: naive_baseline 协议下 bus/motorcycle 的 tgt_cdvm vs src_cdvm 增益
+0.252/+0.231 —— 扩类可能同时提升主表总均值。
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import gain_decompose as G
import unified_eval as U

EVALS_12 = [
    ('COCO-airplane',    'coco', 'airplane',    'natural_scene',  'a photo of',         'airplane'),
    ('COCO-boat',        'coco', 'boat',        'natural_scene',  'a photo of',         'ship'),
    ('COCO-car',         'coco', 'car',         'natural_scene',  'a photo of',         'vehicle'),
    ('COCO-truck',       'coco', 'truck',       'natural_scene',  'a photo of',         'vehicle'),
    ('COCO-bus',         'coco', 'bus',         'natural_scene',  'a photo of',         'vehicle'),
    ('COCO-motorcycle',  'coco', 'motorcycle',  'natural_scene',  'a photo of',         'vehicle'),
    ('DIOR-airplane',    'dior', 'airplane',    'remote_sensing', 'an overhead image of', 'airplane'),
    ('DIOR-ship',        'dior', 'ship',        'remote_sensing', 'an overhead image of', 'ship'),
    ('DIOR-vehicle',     'dior', 'vehicle',     'remote_sensing', 'an overhead image of', 'vehicle'),
    ('DIOR-harbor',      'dior', 'harbor',      'remote_sensing', 'an overhead image of', 'harbor'),
    ('DIOR-storage tank', 'dior', 'storage tank', 'remote_sensing', 'an overhead image of', 'storage tank'),
    ('DIOR-bridge',      'dior', 'bridge',      'remote_sensing', 'an overhead image of', 'bridge'),
]

# 前置检查: DIOR bank 是否有新增类的源原型
bank = U.get_bank()
missing = [d for _, ds, _, _, _, d in EVALS_12 if ds == 'dior' and d not in bank]
if missing:
    raise SystemExit(f'!! DIOR bank 缺少类原型: {missing}')

U.EVALS = EVALS_12
G.OUT = r'C:\Users\张三\Desktop\实验数据\final_results_12class.json'
print(f'扩类评估: {len(EVALS_12)} 类 (COCO 6 跨域 + DIOR 6 同域)')
print(f'  注意: truck/bus/motorcycle 共享 DIOR vehicle 源原型')
print(f'输出 -> {G.OUT}')
print('=' * 95)
G.main()
