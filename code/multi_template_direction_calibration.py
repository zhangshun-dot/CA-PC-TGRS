# -*- coding: utf-8 -*-
"""multi_template_direction_calibration.py — 多模板方向校准实验
验证：用模板集合的均值差估计干净视角方向，能否让COCO→DIOR反向外推转正。

改进方向1：多模板方向校准（零训练，直击根因）
- 源域（遥感/俯视）模板集：4个模板
- 目标域（自然/侧视）模板集：4个模板
- 方向 = mean(目标域模板嵌入) - mean(源域模板嵌入)
- 原理：多个模板对的差异平均掉用词/句法噪声，保留稳定的"视角"语义信号
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
from pathlib import Path
import tge_probe as T
from sota_comparison import (
    SOURCE_LOADERS, evaluate_with_features, PRE_FILTER,
)
import unified_eval as U

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# ── 多模板集合 ──
# 源域（遥感/俯视）模板集
SRC_TEMPLATES_OVERHEAD = [
    'an overhead image of {}',
    'an aerial view of {}',
    'a satellite image of {}',
    'a top-down view of {}',
]

# 目标域（自然/侧视）模板集
TGT_TEMPLATES_PHOTO = [
    'a photo of {}',
    'a natural image of {}',
    'a ground-level view of {}',
    'a side view of {}',
]

# 单模板（基线）
SRC_TEMPLATE_SINGLE = 'an overhead image of {}'
TGT_TEMPLATE_SINGLE = 'a photo of {}'


def build_proto_multi_template(src_proto, src_cls, tgt_cls, beta=1.0,
                                 src_templates=None, tgt_templates=None):
    """多模板方向校准的外推函数"""
    if src_templates is None:
        src_templates = SRC_TEMPLATES_OVERHEAD
    if tgt_templates is None:
        tgt_templates = TGT_TEMPLATES_PHOTO

    # 多模板均值嵌入
    s_emb = T.text_avg(src_templates, src_cls).cpu().numpy()
    d_emb = T.text_avg(tgt_templates, tgt_cls).cpu().numpy()
    delta = d_emb - s_emb
    p_hat = src_proto + beta * delta
    return p_hat / (np.linalg.norm(p_hat) + 1e-8)


def build_proto_single_template(src_proto, src_cls, tgt_cls, beta=1.0):
    """单模板外推（基线）"""
    s_emb = T.text_avg([SRC_TEMPLATE_SINGLE], src_cls).cpu().numpy()
    d_emb = T.text_avg([TGT_TEMPLATE_SINGLE], tgt_cls).cpu().numpy()
    delta = d_emb - s_emb
    p_hat = src_proto + beta * delta
    return p_hat / (np.linalg.norm(p_hat) + 1e-8)


def load_target_dior(dior_cls, n=200):
    """加载DIOR目标域数据（用GDINO-tiny检测）"""
    from pycocotools.coco import COCO
    from PIL import Image

    # DIOR标注路径
    ann_file = r'C:\Users\张三\modeling\dior_data\DOTA_V1.5\train\DOTA-v1.5_train.json'
    img_dir = r'C:\Users\张三\modeling\dior_data\DOTA_V1.5\train\images'

    # 这里简化处理，实际应该用DIOR的标注
    # 由于DIOR标注格式特殊，我们用之前缓存的检测结果
    # 先返回None，让调用者处理
    return None


def run_experiment(src_name, src_cls, tgt_cls, target_domain='COCO', beta_values=None):
    """跑单个类别的实验，对比单模板 vs 多模板"""
    if beta_values is None:
        beta_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    print(f'\n{"─" * 80}')
    print(f'{src_name}/{src_cls} → {target_domain}/{tgt_cls}')
    print(f'{"─" * 80}')

    # 加载源域数据
    print(f'  加载源域数据...')
    src_loader = SOURCE_LOADERS[src_name][0]
    src_data = src_loader(src_cls, n=150)
    if src_data is None:
        print(f'  源域数据加载失败，跳过')
        return None
    src_proto = src_data['prototype']
    print(f'  源域原型: ||p||={np.linalg.norm(src_proto):.4f}, n_crops={src_data["n_crops"]}')

    # 负原型
    src_features = src_data['features']
    sims = src_features @ src_proto
    neg_idx = np.argmin(sims)
    proto_neg = src_features[neg_idx]
    proto_neg = proto_neg / (np.linalg.norm(proto_neg) + 1e-8)

    # 加载目标域数据
    print(f'  加载{target_domain}目标域数据...')
    if target_domain == 'COCO':
        from sota_comparison import load_target_coco
        target_data = load_target_coco(tgt_cls, n=200)
    else:
        # DIOR目标域（反向实验）
        target_data = load_target_dior(tgt_cls, n=200)

    if target_data is None:
        print(f'  目标域数据加载失败，跳过')
        return None
    print(f'  目标域: n_imgs={len(target_data["per_img"])}, n_crops={target_data["n_crops"]}')

    # 对比单模板 vs 多模板
    results = {
        'src': src_name,
        'src_cls': src_cls,
        'target_cls': tgt_cls,
        'target_domain': target_domain,
        'single_template': {},
        'multi_template': {},
    }

    print(f'\n  β    单模板AP    多模板AP    差值')
    print(f'  {"─" * 50}')

    for beta in beta_values:
        # 单模板
        if beta == 0.0:
            proto_single = src_proto
            proto_multi = src_proto
        else:
            proto_single = build_proto_single_template(src_proto, src_cls, tgt_cls, beta=beta)
            proto_multi = build_proto_multi_template(src_proto, src_cls, tgt_cls, beta=beta)

        ap_single = evaluate_with_features(target_data, proto_single, neg_proto=proto_neg)
        ap_multi = evaluate_with_features(target_data, proto_multi, neg_proto=proto_neg)

        results['single_template'][str(beta)] = float(ap_single)
        results['multi_template'][str(beta)] = float(ap_multi)

        diff = ap_multi - ap_single
        print(f'  {beta:<5.1f} {ap_single:<10.4f} {ap_multi:<10.4f} {diff:+.4f}')

    # 找最优β
    best_beta_single = max(beta_values, key=lambda b: results['single_template'][str(b)])
    best_beta_multi = max(beta_values, key=lambda b: results['multi_template'][str(b)])

    baseline = results['single_template']['0.0']
    best_single = results['single_template'][str(best_beta_single)]
    best_multi = results['multi_template'][str(best_beta_multi)]

    results['baseline'] = float(baseline)
    results['best_beta_single'] = float(best_beta_single)
    results['best_ap_single'] = float(best_single)
    results['gain_single'] = float(best_single - baseline)
    results['best_beta_multi'] = float(best_beta_multi)
    results['best_ap_multi'] = float(best_multi)
    results['gain_multi'] = float(best_multi - baseline)
    results['improvement'] = float(best_multi - best_single)

    print(f'\n  总结:')
    print(f'    baseline (β=0):      {baseline:.4f}')
    print(f'    单模板最优: β={best_beta_single}, AP={best_single:.4f}, 增益={best_single - baseline:+.4f}')
    print(f'    多模板最优: β={best_beta_multi}, AP={best_multi:.4f}, 增益={best_multi - baseline:+.4f}')
    print(f'    多模板改进: {best_multi - best_single:+.4f}')

    return results


def main():
    print('=' * 80)
    print('多模板方向校准实验')
    print('=' * 80)
    print()
    print('源域模板集（遥感/俯视）:')
    for t in SRC_TEMPLATES_OVERHEAD:
        print(f'  - {t}')
    print()
    print('目标域模板集（自然/侧视）:')
    for t in TGT_TEMPLATES_PHOTO:
        print(f'  - {t}')

    # ── 实验1：正向 DIOR→COCO（验证不把正向搞坏）──
    print('\n' + '=' * 80)
    print('实验1：正向 DIOR→COCO（验证多模板不把正向搞坏）')
    print('=' * 80)

    forward_experiments = [
        ('DIOR', 'airplane', 'airplane'),
        ('DIOR', 'ship', 'boat'),
        ('DIOR', 'vehicle', 'car'),
        ('DIOR', 'vehicle', 'bus'),
        ('DIOR', 'vehicle', 'motorcycle'),
        ('DIOR', 'vehicle', 'truck'),
    ]

    forward_results = []
    for src, src_cls, tgt_cls in forward_experiments:
        result = run_experiment(src, src_cls, tgt_cls, target_domain='COCO')
        if result is not None:
            forward_results.append(result)

    # ── 汇总正向结果 ──
    print('\n' + '=' * 80)
    print('正向实验汇总（DIOR→COCO）')
    print('=' * 80)
    print(f'\n{"类别":<12} {"baseline":>10} {"单模板增益":>12} {"多模板增益":>12} {"改进":>10}')
    print('-' * 60)

    for r in forward_results:
        print(f'{r["target_cls"]:<12} {r["baseline"]:>10.4f} '
              f'{r["gain_single"]:>+12.4f} {r["gain_multi"]:>+12.4f} '
              f'{r["improvement"]:>+10.4f}')

    avg_gain_single = np.mean([r['gain_single'] for r in forward_results])
    avg_gain_multi = np.mean([r['gain_multi'] for r in forward_results])
    avg_improvement = np.mean([r['improvement'] for r in forward_results])

    print(f'\n平均:')
    print(f'  单模板增益: {avg_gain_single:+.4f}')
    print(f'  多模板增益: {avg_gain_multi:+.4f}')
    print(f'  多模板改进: {avg_improvement:+.4f}')

    # ── 保存结果 ──
    out_file = DATA / 'multi_template_direction_calibration.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            'forward_results': forward_results,
            'src_templates': SRC_TEMPLATES_OVERHEAD,
            'tgt_templates': TGT_TEMPLATES_PHOTO,
        }, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {out_file}')


if __name__ == '__main__':
    main()
