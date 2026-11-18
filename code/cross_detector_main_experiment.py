# -*- coding: utf-8 -*-
"""cross_detector_main_experiment.py — 通用跨检测器主实验
支持GDINO-base和YOLO-World，跑7个类别的α网格搜索。
用法:
  python cross_detector_main_experiment.py --detector gdino_base
  python cross_detector_main_experiment.py --detector yolo_world
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import argparse
import json
import os
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
import torch

# 强制离线模式，禁止访问huggingface.co
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import unified_eval as U
from sota_comparison import (
    SOURCE_LOADERS, build_proto_tge, evaluate_with_features,
    PRE_FILTER,
)

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# 检测器配置
DETECTOR_CONFIGS = {
    'gdino_base': {
        'model_id': 'IDEA-Research/grounding-dino-base',
        'prompt_tmpl': 'a satellite image of {}',
        'cache_prefix': 'gdino_base_dets',
    },
    'yolo_world': {
        'model_id': 'wondervictor/YOLO-World',
        'prompt_tmpl': 'a satellite image of {}',
        'cache_prefix': 'yolo_world_dets',
    },
}

# 主分析7类
MAIN_CLASSES = ['airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'truck']

# α网格
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

# 物理先验门控
OBJECT_SIZE = {'airplane': 30.0, 'boat': 15.0, 'car': 4.5, 'bus': 12.0,
               'truck': 12.0, 'motorcycle': 2.0, 'bicycle': 1.8, 'person': 1.7}
VIEWPOINT_DIFF = {'airplane': 0.3, 'boat': 0.4, 'car': 0.7, 'bus': 0.8,
                   'truck': 0.75, 'motorcycle': 0.6, 'bicycle': 0.5, 'person': 0.9}


def get_gate(cls):
    """门控决策：size < 10 AND vp > 0.5"""
    return 1 if (OBJECT_SIZE.get(cls, 999) < 10.0 and VIEWPOINT_DIFF.get(cls, 0) > 0.5) else 0


def normalize(v):
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8) if norm > 1e-8 else v


def get_detector(detector_name):
    """懒加载检测器模型"""
    config = DETECTOR_CONFIGS[detector_name]
    cache_key = f'_{detector_name}_model'
    if cache_key not in globals():
        print(f'加载{detector_name}模型: {config["model_id"]}')
        globals()[f'_{detector_name}_proc'] = AutoProcessor.from_pretrained(
            config['model_id'], local_files_only=True)
        globals()[cache_key] = (AutoModelForZeroShotObjectDetection
                                 .from_pretrained(config['model_id'], dtype=torch.float32, local_files_only=True)
                                 .eval().to(U.DEVICE))
    return globals()[f'_{detector_name}_proc'], globals()[cache_key]


def detect(images, prompt, detector_name, thresh=0.0):
    """用指定检测器检测一批图像，返回每图(boxes, scores)"""
    proc, model = get_detector(detector_name)
    out = []
    for img in images:
        inputs = proc(text=prompt, images=img, return_tensors='pt')
        inputs = {k: v.to(U.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]]).to(U.DEVICE)
        r = proc.post_process_grounded_object_detection(outputs, threshold=thresh, target_sizes=target_sizes)[0]
        out.append((r['boxes'].cpu().numpy(), r['scores'].cpu().numpy()))
    return out


def load_target_coco_detector(coco_cls, detector_name, n=200):
    """用指定检测器检测COCO目标域，返回per_img和features"""
    config = DETECTOR_CONFIGS[detector_name]
    images, gts = U.load_coco(coco_cls, n=n)
    if not images:
        return None
    cache_p = DATA / f'{config["cache_prefix"]}_COCO-{coco_cls}.pkl'
    if cache_p.exists():
        dets = pickle.load(open(cache_p, 'rb'))
    else:
        prompt = config['prompt_tmpl'].format(coco_cls)
        dets = detect(images, prompt, detector_name, 0.0)
        pickle.dump(dets, open(cache_p, 'wb'))
    per_img = []
    crops_all = []
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < PRE_FILTER:
                continue
            bl.append([x1, y1, x2, y2])
            sl.append(float(sc))
            ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({
            'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
            'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
            'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
            'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64),
        })
    # 用CLIP编码特征（与检测器无关）
    import tge_probe as T
    F = T.encode_clip_batch(crops_all).cpu().numpy() if crops_all else np.zeros((0, 512))
    return {'per_img': per_img, 'features': F, 'n_crops': len(crops_all)}


def run_experiment(detector_name, src_name, src_cls, target_cls):
    """跑单个类别的实验"""
    print(f'\n{"─" * 80}')
    print(f'{src_name}/{src_cls} → COCO/{target_cls} ({detector_name}检测器)')
    print(f'{"─" * 80}')

    # 加载源域数据
    print(f'  加载源域数据...')
    src_loader = SOURCE_LOADERS[src_name][0]
    src_data = src_loader(src_cls, n=150)
    if src_data is None:
        print(f'  源域数据加载失败，跳过')
        return None
    src_proto = src_data['prototype']
    src_features = src_data['features']
    print(f'  源域原型: ||p||={np.linalg.norm(src_proto):.4f}, n_crops={src_data["n_crops"]}')

    # 负原型
    sims = src_features @ src_proto
    neg_idx = np.argmin(sims)
    proto_neg = src_features[neg_idx]
    proto_neg = normalize(proto_neg)

    # 加载COCO目标域数据
    print(f'  加载COCO目标域数据（{detector_name}检测）...')
    target_data = load_target_coco_detector(target_cls, detector_name, n=200)
    if target_data is None:
        print(f'  目标域数据加载失败，跳过')
        return None
    print(f'  目标域: n_imgs={len(target_data["per_img"])}, n_crops={target_data["n_crops"]}')

    # α网格搜索（TGE算子）
    print(f'  α网格搜索（TGE算子）...')
    ap_curve = {}
    for alpha in ALPHAS:
        if alpha == 0.0:
            proto = src_proto
        else:
            proto = build_proto_tge(src_proto, src_cls, target_cls, beta=alpha)
        ap = evaluate_with_features(target_data, proto, neg_proto=proto_neg)
        ap_curve[str(alpha)] = float(ap)

    # 找最优α
    best_alpha = ALPHAS[0]
    best_ap = ap_curve[str(ALPHAS[0])]
    for alpha in ALPHAS:
        if ap_curve[str(alpha)] > best_ap:
            best_ap = ap_curve[str(alpha)]
            best_alpha = alpha

    baseline_ap = ap_curve['0.0']
    oracle_gain = best_ap - baseline_ap

    # 固定α=2.5的增益
    ap_25 = ap_curve.get('2.5', baseline_ap)
    gain_25 = ap_25 - baseline_ap

    # 门控决策
    gate = get_gate(target_cls)

    # 门控后的增益
    if gate == 1:
        gated_gain = gain_25
        gated_alpha = 2.5
    else:
        gated_gain = 0.0
        gated_alpha = 0.0

    result = {
        'src': src_name,
        'src_cls': src_cls,
        'target_cls': target_cls,
        'target_domain': 'COCO',
        'detector': detector_name,
        'gate': gate,
        'baseline_ap': float(baseline_ap),
        'best_alpha': float(best_alpha),
        'best_ap': float(best_ap),
        'oracle_gain': float(oracle_gain),
        'gain_25': float(gain_25),
        'gated_gain': float(gated_gain),
        'gated_alpha': float(gated_alpha),
        'ap_curve': ap_curve,
    }

    print(f'  baseline AP50: {baseline_ap:.4f}')
    print(f'  最优α: {best_alpha}, 最优AP50: {best_ap:.4f}, Oracle增益: {oracle_gain:+.4f}')
    print(f'  固定α=2.5增益: {gain_25:+.4f}')
    print(f'  门控: {"开" if gate == 1 else "关"}, 门控后增益: {gated_gain:+.4f}')

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', type=str, required=True,
                        choices=['gdino_base', 'yolo_world'],
                        help='检测器名称')
    args = parser.parse_args()

    detector_name = args.detector

    print('=' * 80)
    print(f'{detector_name}跨检测器主实验')
    print('=' * 80)

    # 实验配置：DIOR源域 → COCO目标域，7个类
    experiments = [
        ('DIOR', 'airplane', 'airplane'),
        ('DIOR', 'ship', 'boat'),
        ('DIOR', 'vehicle', 'car'),
        ('DIOR', 'vehicle', 'bus'),
        ('DIOR', 'vehicle', 'motorcycle'),
        ('DIOR', 'vehicle', 'truck'),
        ('VisDrone', 'bicycle', 'bicycle'),
    ]

    print(f'\n实验配置: {len(experiments)}个类别')
    for src, src_cls, tgt_cls in experiments:
        gate = get_gate(tgt_cls)
        print(f'  {src}/{src_cls} → COCO/{tgt_cls}, gate={"开" if gate == 1 else "关"}')

    # 跑每个类别
    all_results = []
    for src, src_cls, tgt_cls in experiments:
        result = run_experiment(detector_name, src, src_cls, tgt_cls)
        if result is not None:
            all_results.append(result)

    # 汇总
    print(f'\n{"=" * 80}')
    print(f'汇总结果（{detector_name}检测器）')
    print(f'{"=" * 80}')

    print(f'\n{"类别":<12} {"gate":>5} {"baseline":>10} {"best_alpha":>10} '
          f'{"oracle":>10} {"gain_25":>10} {"gated":>10}')
    print('-' * 75)

    for r in all_results:
        print(f'{r["target_cls"]:<12} {r["gate"]:>5} {r["baseline_ap"]:>10.4f} '
              f'{r["best_alpha"]:>10.1f} {r["oracle_gain"]:>+10.4f} '
              f'{r["gain_25"]:>+10.4f} {r["gated_gain"]:>+10.4f}')

    # 统计
    print(f'\n统计:')
    oracle_gains = [r['oracle_gain'] for r in all_results]
    gain_25s = [r['gain_25'] for r in all_results]
    gated_gains = [r['gated_gain'] for r in all_results]

    print(f'  平均Oracle增益: {np.mean(oracle_gains):+.4f}')
    print(f'  平均固定α=2.5增益: {np.mean(gain_25s):+.4f}')
    print(f'  平均门控后增益: {np.mean(gated_gains):+.4f}')
    print(f'  正例比例（固定α=2.5）: {np.mean([1 if g > 0.001 else 0 for g in gain_25s]):.1%}')
    print(f'  正例比例（门控后）: {np.mean([1 if g > 0.001 else 0 for g in gated_gains]):.1%}')

    # 门控开启 vs 关闭
    on_results = [r for r in all_results if r['gate'] == 1]
    off_results = [r for r in all_results if r['gate'] == 0]

    if on_results:
        print(f'\n门控开启类（{len(on_results)}个）:')
        for r in on_results:
            print(f'  {r["target_cls"]}: baseline={r["baseline_ap"]:.4f}, '
                  f'oracle={r["oracle_gain"]:+.4f}, gain_25={r["gain_25"]:+.4f}')
        print(f'  平均固定α=2.5增益: {np.mean([r["gain_25"] for r in on_results]):+.4f}')

    if off_results:
        print(f'\n门控关闭类（{len(off_results)}个）:')
        for r in off_results:
            print(f'  {r["target_cls"]}: baseline={r["baseline_ap"]:.4f}, '
                  f'oracle={r["oracle_gain"]:+.4f}, gain_25={r["gain_25"]:+.4f}')
        print(f'  平均固定α=2.5增益: {np.mean([r["gain_25"] for r in off_results]):+.4f}')

    # 保存结果
    out_file = DATA / f'{detector_name}_cross_detector_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {out_file}')


if __name__ == '__main__':
    main()
