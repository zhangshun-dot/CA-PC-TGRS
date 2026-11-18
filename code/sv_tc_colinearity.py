# -*- coding: utf-8 -*-
"""sv_tc_colinearity.py — 补测sv和tc的共线性
1. 重新计算27个样本的sv（候选框尺寸方差）、tc（文本混淆度）
2. 分析sv/tc和sim_pos_std/sim_pos_mean的共线性
3. 边界分解（边界点vs内部点）
4. 平坦度混淆检验
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
from scipy import stats

from sota_comparison import (
    SOURCE_LOADERS,
    DIOR_TARGETS, NWPU_TARGETS, DOTA_TARGETS, VISDRONE_TARGETS,
    build_proto_tge, evaluate_with_features,
    T, SRC_TMPL_OVERHEAD, TGT_TMPL,
    compute_text_confusability,
)

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

REMOTE_TARGETS = {
    'DIOR': DIOR_TARGETS,
    'NWPU': NWPU_TARGETS,
    'DOTA': DOTA_TARGETS,
    'VisDrone': VISDRONE_TARGETS,
}

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]


def compute_sv(target_data):
    """计算sv = 候选框尺寸方差（面积的方差）"""
    per_img = target_data['per_img']
    all_areas = []
    for p in per_img:
        if 'boxes' in p and len(p['boxes']) > 0:
            boxes = p['boxes']
            # boxes格式: [x1, y1, x2, y2]
            for box in boxes:
                if len(box) >= 4:
                    area = max(0, (box[2] - box[0])) * max(0, (box[3] - box[1]))
                    if area > 0:
                        all_areas.append(area)

    if len(all_areas) < 2:
        return 0.0, 0

    all_areas = np.array(all_areas)
    # 使用log面积的方差（因为面积分布通常是长尾的）
    log_areas = np.log(all_areas + 1e-8)
    sv = float(np.var(log_areas))
    return sv, len(all_areas)


def compute_tc(src_cls, target_cls):
    """计算tc = 文本混淆度（目标类别和其他类别的最小文本嵌入距离）"""
    # 所有类别列表
    all_classes = ['airplane', 'boat', 'car', 'bus', 'motorcycle', 'bicycle', 'truck',
                   'aeroplane', 'motorbike', 'ship', 'vehicle', 'plane', 'small-vehicle', 'large-vehicle']

    # 使用sota_comparison里的compute_text_confusability函数
    tc = compute_text_confusability(target_cls, all_classes)
    return float(tc)


def compute_sim_factors(target_data, src_proto):
    """计算sim_pos_mean和sim_pos_std"""
    features = target_data['features']
    sim_pos = features @ src_proto
    return {
        'sim_pos_mean': float(np.mean(sim_pos)),
        'sim_pos_std': float(np.std(sim_pos)),
    }


def process_sample(src_name, src_cls, target_cls, target_domain, target_loader_func):
    """处理单个样本"""
    # 加载源域
    src_loader = SOURCE_LOADERS[src_name][0]
    src_data = src_loader(src_cls, n=150)
    if src_data is None:
        return None
    src_proto = src_data['prototype']
    src_features = src_data['features']

    # 负原型
    sims = src_features @ src_proto
    neg_idx = np.argmin(sims)
    proto_neg = src_features[neg_idx]
    proto_neg = proto_neg / (np.linalg.norm(proto_neg) + 1e-8)

    # 加载目标域
    target_data = target_loader_func(target_cls, n=200)
    if target_data is None:
        return None

    # 计算所有因子
    sim_factors = compute_sim_factors(target_data, src_proto)
    sv, n_boxes = compute_sv(target_data)
    tc = compute_tc(src_cls, target_cls)

    # α网格搜索
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

    # 可辨识性
    threshold = 0.005
    plateau_alphas = [a for a in ALPHAS if abs(ap_curve[str(a)] - best_ap) <= threshold]
    plateau_width = max(plateau_alphas) - min(plateau_alphas) if plateau_alphas else 0
    identifiable = plateau_width < 0.5

    # 边界点
    hit_upper = best_alpha == ALPHAS[-1]
    hit_lower = best_alpha == ALPHAS[0]
    is_interior = not hit_upper and not hit_lower

    baseline_ap = ap_curve['0.0']
    delta_ap_at_1 = ap_curve['1.0'] - baseline_ap  # 固定α=1.0的ΔAP

    result = {
        'src': src_name,
        'src_cls': src_cls,
        'target_cls': target_cls,
        'target_domain': target_domain,
        'sv': sv,
        'tc': tc,
        'sim_pos_mean': sim_factors['sim_pos_mean'],
        'sim_pos_std': sim_factors['sim_pos_std'],
        'n_boxes': n_boxes,
        'best_alpha': float(best_alpha),
        'best_ap': float(best_ap),
        'baseline_ap': float(baseline_ap),
        'plateau_width': float(plateau_width),
        'identifiable': identifiable,
        'hit_upper': hit_upper,
        'hit_lower': hit_lower,
        'is_interior': is_interior,
        'delta_ap_at_1': float(delta_ap_at_1),
        'ap_curve': ap_curve,
    }

    return result


def analyze_colinearity(samples):
    """分析共线性"""
    print(f'\n{"=" * 80}')
    print('1. 因子共线性矩阵')
    print(f'{"=" * 80}')

    factors = ['sv', 'tc', 'sim_pos_mean', 'sim_pos_std', 'baseline_ap', 'n_boxes']
    n = len(samples)

    print(f'\n{"因子":<20s}', end='')
    for f in factors:
        print(f'{f:>15s}', end='')
    print()
    print('-' * (20 + 15 * len(factors)))

    corr_matrix = {}
    for f1 in factors:
        print(f'{f1:<20s}', end='')
        corr_matrix[f1] = {}
        for f2 in factors:
            v1 = np.array([s[f1] for s in samples])
            v2 = np.array([s[f2] for s in samples])
            rho, _ = stats.spearmanr(v1, v2)
            corr_matrix[f1][f2] = rho
            print(f'{rho:>15.4f}', end='')
        print()

    # 重点关注的共线性
    print(f'\n重点共线性:')
    print(f'  ρ(sv, sim_pos_std) = {corr_matrix["sv"]["sim_pos_std"]:.4f}')
    print(f'  ρ(tc, sim_pos_mean) = {corr_matrix["tc"]["sim_pos_mean"]:.4f}')
    print(f'  ρ(sv, tc) = {corr_matrix["sv"]["tc"]:.4f}')
    print(f'  ρ(sim_pos_mean, sim_pos_std) = {corr_matrix["sim_pos_mean"]["sim_pos_std"]:.4f}')

    if abs(corr_matrix['sv']['sim_pos_std']) > 0.7:
        print(f'  → SV与sim_pos_std高度相关，SV可能只是离散度的间接代理')
    if abs(corr_matrix['tc']['sim_pos_mean']) > 0.7:
        print(f'  → TC与sim_pos_mean高度相关，"双因子"可能是"一个因子的两种写法"')
    if abs(corr_matrix['sv']['tc']) > 0.7:
        print(f'  → SV与TC高度相关，6个点根本分不开两个因子')

    return corr_matrix


def analyze_boundary_decomposition(samples):
    """边界分解：边界点vs内部点"""
    print(f'\n{"=" * 80}')
    print('2. 边界分解：边界点vs内部点')
    print(f'{"=" * 80}')

    interior = [s for s in samples if s['is_interior']]
    boundary = [s for s in samples if not s['is_interior']]

    print(f'\n内部点: {len(interior)}/{len(samples)} ({len(interior)/len(samples)*100:.1f}%)')
    print(f'边界点: {len(boundary)}/{len(samples)} ({len(boundary)/len(samples)*100:.1f}%)')

    factors = ['sv', 'tc', 'sim_pos_mean', 'sim_pos_std', 'baseline_ap']

    print(f'\n{"因子":<20s} {"全部ρ":>10s} {"内部ρ":>10s} {"边界ρ":>10s}')
    print('-' * 55)

    for factor in factors:
        # 全部
        v_all = np.array([s[factor] for s in samples])
        a_all = np.array([s['best_alpha'] for s in samples])
        rho_all, _ = stats.spearmanr(v_all, a_all)

        # 内部点
        if len(interior) >= 3:
            v_int = np.array([s[factor] for s in interior])
            a_int = np.array([s['best_alpha'] for s in interior])
            rho_int, _ = stats.spearmanr(v_int, a_int)
        else:
            rho_int = float('nan')

        # 边界点
        if len(boundary) >= 3:
            v_bnd = np.array([s[factor] for s in boundary])
            a_bnd = np.array([s['best_alpha'] for s in boundary])
            rho_bnd, _ = stats.spearmanr(v_bnd, a_bnd)
        else:
            rho_bnd = float('nan')

        print(f'{factor:<20s} {rho_all:>10.4f} {rho_int:>10.4f} {rho_bnd:>10.4f}')

    print(f'\n判读:')
    print(f'  如果某因子在内部点不相关、只在边界点相关 → 假象（机械相关）')
    print(f'  如果某因子在内部点也相关 → 可能是真实关系')


def analyze_flatness_confound(samples):
    """平坦度混淆检验"""
    print(f'\n{"=" * 80}')
    print('3. 平坦度混淆检验')
    print(f'{"=" * 80}')

    factors = ['sv', 'tc', 'sim_pos_mean', 'sim_pos_std', 'baseline_ap']

    print(f'\n{"因子":<20s} {"ρ(因子,α*)":>12s} {"ρ(因子,平台宽度)":>18s} {"ρ(因子,ΔAP@1.0)":>18s}')
    print('-' * 75)

    for factor in factors:
        v = np.array([s[factor] for s in samples])
        a = np.array([s['best_alpha'] for s in samples])
        pw = np.array([s['plateau_width'] for s in samples])
        d1 = np.array([s['delta_ap_at_1'] for s in samples])

        rho_a, _ = stats.spearmanr(v, a)
        rho_pw, _ = stats.spearmanr(v, pw)
        rho_d1, _ = stats.spearmanr(v, d1)

        print(f'{factor:<20s} {rho_a:>12.4f} {rho_pw:>18.4f} {rho_d1:>18.4f}')

    print(f'\n判读:')
    print(f'  如果ρ(因子,平台宽度)很强，而ρ(因子,ΔAP@1.0)很弱或符号相反 → 平坦度混淆')
    print(f'  如果ρ(因子,ΔAP@1.0)也很强且符号一致 → 可能是真实机制')


def analyze_delta_ap_target(samples):
    """换靶子：固定α的ΔAP"""
    print(f'\n{"=" * 80}')
    print('4. 换靶子：固定α=1.0的ΔAP（连续、可测、直连机制）')
    print(f'{"=" * 80}')

    factors = ['sv', 'tc', 'sim_pos_mean', 'sim_pos_std', 'baseline_ap']

    print(f'\n{"因子":<20s} {"ρ(因子,ΔAP@1.0)":>18s} {"p值":>10s} {"显著性":>8s}')
    print('-' * 60)

    for factor in factors:
        v = np.array([s[factor] for s in samples])
        d1 = np.array([s['delta_ap_at_1'] for s in samples])
        rho, p = stats.spearmanr(v, d1)
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f'{factor:<20s} {rho:>18.4f} {p:>10.4f}{sig}')

    print(f'\n判读:')
    print(f'  ΔAP@1.0是连续、可测、直连机制的靶子')
    print(f'  如果某因子和ΔAP@1.0显著相关 → 真实机制')
    print(f'  如果只和α*相关、和ΔAP@1.0不相关 → α*是假象')


def main():
    print('=' * 80)
    print('补测sv和tc的共线性 + 边界分解 + 平坦度混淆 + 换靶子')
    print('=' * 80)

    all_samples = []

    # COCO目标域
    from sota_comparison import load_target_coco
    print(f'\n--- COCO目标域 ---')
    for src_name, class_list in REMOTE_TARGETS.items():
        for coco_cls, src_cls, src_display in class_list:
            print(f'  处理 {src_name}/{src_cls} -> COCO/{coco_cls}...')
            result = process_sample(src_name, src_cls, coco_cls, 'COCO', load_target_coco)
            if result:
                all_samples.append(result)
                print(f'    sv={result["sv"]:.4f}, tc={result["tc"]:.4f}, '
                      f'sim_mean={result["sim_pos_mean"]:.4f}, sim_std={result["sim_pos_std"]:.4f}, '
                      f'best α={result["best_alpha"]:.1f}, interior={result["is_interior"]}')

    # VOC目标域
    from voc_sota_comparison import load_target_voc
    COCO2VOC = {
        'airplane': 'aeroplane', 'boat': 'boat', 'car': 'car',
        'bus': 'bus', 'motorcycle': 'motorbike', 'bicycle': 'bicycle',
    }
    print(f'\n--- VOC目标域 ---')
    for src_name, class_list in REMOTE_TARGETS.items():
        for coco_cls, src_cls, src_display in class_list:
            if coco_cls not in COCO2VOC:
                continue
            voc_cls = COCO2VOC[coco_cls]
            print(f'  处理 {src_name}/{src_cls} -> VOC/{voc_cls}...')
            result = process_sample(src_name, src_cls, voc_cls, 'VOC', load_target_voc)
            if result:
                all_samples.append(result)
                print(f'    sv={result["sv"]:.4f}, tc={result["tc"]:.4f}, '
                      f'sim_mean={result["sim_pos_mean"]:.4f}, sim_std={result["sim_pos_std"]:.4f}, '
                      f'best α={result["best_alpha"]:.1f}, interior={result["is_interior"]}')

    print(f'\n总样本数: {len(all_samples)}')

    # 保存数据
    with open(DATA / 'sv_tc_colinearity_samples.json', 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    # 分析
    corr_matrix = analyze_colinearity(all_samples)
    analyze_boundary_decomposition(all_samples)
    analyze_flatness_confound(all_samples)
    analyze_delta_ap_target(all_samples)

    # 保存结果
    with open(DATA / 'sv_tc_colinearity_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'n_samples': len(all_samples),
            'corr_matrix': corr_matrix,
        }, f, indent=2, ensure_ascii=False)

    print(f'\n{"=" * 80}')
    print('分析完成')
    print(f'{"=" * 80}')


if __name__ == '__main__':
    main()
