# -*- coding: utf-8 -*-
"""voc_sota_comparison.py — VOC目标域SOTA对比实验
验证方法在第二个目标域上的泛化性
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
from pathlib import Path
import numpy as np
import unified_eval as U
from voc_loader import load_voc, VOC2COCO, REMOTE_VOC_CLASSES

# 从sota_comparison导入所有需要的函数和常量
from sota_comparison import (
    SOURCE_LOADERS,
    DIOR_TARGETS, NWPU_TARGETS, DOTA_TARGETS, VISDRONE_TARGETS,
    build_proto_tge, evaluate_with_features, evaluate_baseline,
    compute_box_size_variance, compute_text_confusability,
    load_dual_factor_model, predict_alpha_adaptive, USC,
    BETA, K_PCA, DATA
)

OUT = DATA / 'voc_sota_comparison_results.json'

# 源域类别映射（和sota_comparison一致）
REMOTE_TARGETS = {
    'DIOR': DIOR_TARGETS,
    'NWPU': NWPU_TARGETS,
    'DOTA': DOTA_TARGETS,
    'VisDrone': VISDRONE_TARGETS,
}


def load_target_voc(voc_cls, n=200):
    """加载VOC目标域数据，格式和load_target_coco完全一致"""
    images, gts = load_voc(voc_cls, n=n)
    if not images:
        return None

    # 使用GDINO检测VOC图像（和COCO目标域一致的检测流程）
    cache_p = DATA / f'tge_v2_dets_VOC-{voc_cls}.pkl'
    if cache_p.exists():
        dets = pickle.load(open(cache_p, 'rb'))
    else:
        dets = U.gdino_detect(images, voc_cls, 0.0)
        pickle.dump(dets, open(cache_p, 'wb'))

    # 构建和COCO目标域完全一致的数据结构
    per_img = []
    crops_all = []
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < 0.03:  # PRE_FILTER
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

    # 提取CLIP特征（和COCO目标域一致）
    if crops_all:
        from sota_comparison import T
        F = T.encode_clip_batch(crops_all).cpu().numpy()
    else:
        F = np.zeros((0, 512))

    return {'per_img': per_img, 'features': F, 'n_crops': len(crops_all)}


def run_voc_sota():
    print('=' * 80)
    print('VOC目标域SOTA对比实验')
    print('=' * 80)

    model_params = load_dual_factor_model()
    # 尝试加载合并训练的模型参数（COCO+VOC）
    combined_model_file = DATA / 'dual_factor_model_params_combined.json'
    if combined_model_file.exists():
        with open(combined_model_file, 'r', encoding='utf-8') as f:
            model_params = json.load(f)
        print(f'使用合并训练的双因子模型（COCO+VOC）: ρ={model_params.get("rho", "N/A")}')
    elif model_params:
        print(f'双因子模型已加载（COCO only）')
    else:
        print('警告: 双因子模型未找到')

    results = {}
    # all_categories包含源域类别名+VOC类别名+COCO类别名
    all_categories = set()
    for src_name in REMOTE_TARGETS:
        for item in REMOTE_TARGETS[src_name]:
            coco_cls, src_cls, _ = item
            all_categories.add(coco_cls)
            all_categories.add(src_cls)
    for voc_cls in REMOTE_VOC_CLASSES:
        all_categories.add(voc_cls)
        all_categories.add(VOC2COCO[voc_cls])
    all_categories = sorted(list(all_categories))
    print(f"所有类别: {all_categories}")

    for src_name in REMOTE_TARGETS:
        print(f'\n{"=" * 60}')
        print(f'源域: {src_name}')
        print(f'{"=" * 60}')

        src_results = {}
        src_cls_list = REMOTE_TARGETS[src_name]
        loader_fn = SOURCE_LOADERS[src_name][0]

        # 加载所有源域数据
        src_data = {}
        for item in src_cls_list:
            coco_cls, src_cls, _ = item
            # 只做VOC中存在的类别
            voc_cls = None
            for vc in REMOTE_VOC_CLASSES:
                if VOC2COCO[vc] == coco_cls:
                    voc_cls = vc
                    break
            if voc_cls is None:
                print(f'  跳过 {src_cls} -> {coco_cls} (VOC中无对应类别)')
                continue

            print(f'  加载源域 {src_name}/{src_cls} -> VOC/{voc_cls}...')
            data = loader_fn(src_cls, n=50)
            if data is not None and 'features' in data and len(data['features']) > 0:
                src_data[src_cls] = {'data': data, 'voc_cls': voc_cls, 'coco_cls': coco_cls}
                print(f'    成功: {len(data["features"])} 个裁剪')
            else:
                print(f'    警告: {src_cls} 无有效特征，跳过')

        for src_cls in src_data:
            voc_cls = src_data[src_cls]['voc_cls']
            coco_cls = src_data[src_cls]['coco_cls']

            print(f'\n  --- 类别: {src_cls} -> VOC/{voc_cls} ---')

            # 加载VOC目标域数据
            try:
                target_data = load_target_voc(voc_cls, n=200)
            except Exception as e:
                print(f'    VOC目标域加载失败: {e}')
                import traceback
                traceback.print_exc()
                continue

            if target_data is None or len(target_data['per_img']) == 0:
                print(f'    VOC目标域无数据，跳过')
                continue

            src_feat = src_data[src_cls]['data']['features']
            src_proto = src_data[src_cls]['data']['prototype']
            tgt_feat = target_data['features']

            cls_results = {}

            # 配置1: Baseline
            ap_base = evaluate_baseline(target_data)
            cls_results['Baseline'] = ap_base
            print(f'    Baseline: {ap_base:.4f}')

            # 配置2: TGE
            proto_tge = build_proto_tge(src_proto, src_cls, coco_cls, beta=BETA)
            ap_tge = evaluate_with_features(target_data, proto_tge, use_neg=False)
            cls_results['TGE'] = ap_tge
            print(f'    TGE: {ap_tge:.4f} (增益: {ap_tge-ap_base:+.4f})')

            # 配置3: USC固定α=1.0
            usc = USC(n_components=K_PCA)
            usc.fit(src_feat, tgt_feat)
            proto_usc_fixed = usc.correct_prototype(src_proto, alpha=1.0)
            ap_usc_fixed = evaluate_with_features(target_data, proto_usc_fixed, use_neg=False)
            cls_results['USC_fixed_alpha1.0'] = ap_usc_fixed
            print(f'    USC (α=1.0): {ap_usc_fixed:.4f} (增益: {ap_usc_fixed-ap_base:+.4f})')

            # 配置4: USC双因子自适应α
            box_size_var = compute_box_size_variance(target_data)
            text_conf = compute_text_confusability(coco_cls, all_categories)
            alpha_adaptive = predict_alpha_adaptive(box_size_var, text_conf, model_params)
            proto_usc_adaptive = usc.correct_prototype(src_proto, alpha=alpha_adaptive)
            ap_usc_adaptive = evaluate_with_features(target_data, proto_usc_adaptive, use_neg=False)
            cls_results['USC_adaptive_alpha'] = ap_usc_adaptive
            cls_results['alpha_adaptive'] = alpha_adaptive
            cls_results['box_size_variance'] = box_size_var
            cls_results['text_confusability'] = text_conf
            print(f'    USC (自适应α={alpha_adaptive:.2f}, sv={box_size_var:.0f}, tc={text_conf:.3f}): {ap_usc_adaptive:.4f} (增益: {ap_usc_adaptive-ap_base:+.4f})')

            # 配置5: USC Oracle
            best_alpha = 0.0
            best_ap = ap_base
            for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
                proto_scan = usc.correct_prototype(src_proto, alpha=alpha)
                ap_scan = evaluate_with_features(target_data, proto_scan, use_neg=False)
                if ap_scan > best_ap:
                    best_ap = ap_scan
                    best_alpha = alpha
            cls_results['USC_oracle_alpha'] = best_ap
            cls_results['oracle_alpha'] = best_alpha
            print(f'    USC (Oracle α={best_alpha:.1f}): {best_ap:.4f} (增益: {best_ap-ap_base:+.4f})')

            src_results[src_cls] = cls_results

        results[src_name] = src_results

    # 保存结果
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印汇总
    print(f'\n\n{"=" * 80}')
    print('VOC目标域SOTA对比汇总')
    print(f'{"=" * 80}')

    methods = ['Baseline', 'TGE', 'USC_fixed_alpha1.0', 'USC_adaptive_alpha', 'USC_oracle_alpha']
    method_names = {
        'Baseline': 'Baseline',
        'TGE': 'TGE (文本外推)',
        'USC_fixed_alpha1.0': 'USC (α=1.0)',
        'USC_adaptive_alpha': 'USC (双因子自适应)',
        'USC_oracle_alpha': 'USC (Oracle最优)',
    }

    total_n = sum(len(results[s]) for s in results)
    print(f'\n--- 总体平均 (N={total_n}) ---')
    for method in methods:
        all_aps = []
        all_base = []
        for src_name in results:
            for cls in results[src_name]:
                if method in results[src_name][cls]:
                    all_aps.append(results[src_name][cls][method])
                    all_base.append(results[src_name][cls]['Baseline'])
        if all_aps:
            mean_ap = np.mean(all_aps)
            mean_base = np.mean(all_base)
            gain = mean_ap - mean_base
            print(f'  {method_names[method]:25s}: AP50={mean_ap:.4f}, 增益={gain:+.4f}')

    # Oracle实现率
    print(f'\n--- Oracle实现率分析 ---')
    adaptive_gains = []
    oracle_gains = []
    for src_name in results:
        for cls in results[src_name]:
            if 'USC_adaptive_alpha' in results[src_name][cls] and 'USC_oracle_alpha' in results[src_name][cls]:
                base = results[src_name][cls]['Baseline']
                adapt_gain = results[src_name][cls]['USC_adaptive_alpha'] - base
                oracle_gain = results[src_name][cls]['USC_oracle_alpha'] - base
                if oracle_gain > 0.001:
                    adaptive_gains.append(adapt_gain)
                    oracle_gains.append(oracle_gain)
    if adaptive_gains and oracle_gains:
        mean_adapt = np.mean(adaptive_gains)
        mean_oracle = np.mean(oracle_gains)
        realization_rate = mean_adapt / mean_oracle * 100 if mean_oracle > 0 else 0
        print(f'  有效类别数: {len(adaptive_gains)}')
        print(f'  自适应平均增益: {mean_adapt:+.4f}')
        print(f'  Oracle平均增益: {mean_oracle:+.4f}')
        print(f'  Oracle实现率: {realization_rate:.1f}%')

    print(f'\n结果已保存: {OUT}')


if __name__ == '__main__':
    import pickle
    run_voc_sota()
