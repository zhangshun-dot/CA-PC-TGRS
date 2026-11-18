# -*- coding: utf-8 -*-
r"""tge_v2_appg.py — A-CSPC + APPG: 物理先验指导的逐类自适应参数

基于 tge_v2_main.py，加入 APPG (Automated Physics Prior Generator) 的
逐类自适应 β/λ 调整，对比自适应参数 vs 固定参数的效果。

核心逻辑:
  risk(c) = 物理先验风险评分 [0, 1]
  β_eff(c)  = β_base  * (1 + alpha * risk(c))
  λ_eff(c)  = λ_base  * (1 + alpha * risk(c))

风险越高的类别（如 motorcycle, bicycle），校正强度越大；
风险越低的类别（如 airplane, storage tank），校正越保守。

运行: python tge_v2_appg.py --alpha 0.5
      python tge_v2_appg.py --alpha-sweep 0.0,0.25,0.5,0.75,1.0
      python tge_v2_appg.py --prior-source human   # 用论文人工赋值先验
      python tge_v2_appg.py --prior-source offline # 用离线常识先验（默认）

输出: 实验数据/tge_v2_appg_alpha*.json + 控制台对比表
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
import json
import pickle
import argparse
import time
from pathlib import Path
import numpy as np
import unified_eval as U
import tge_probe as T

# APPG imports
from acspc.appg import (
    PhysicsPriorGenerator,
    PhysicsPriorLibrary,
    LLMClient,
    load_human_priors_from_dict,
    HUMAN_PRIORS_FROM_PAPER,
)
from acspc.coco_categories import COCO_CATEGORIES, get_offline_prior
from acspc.appg import PhysicsPrior

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# ── 基础配置（与 tge_v2_main.py 一致）──
FROZEN_BETA = 1.0
FROZEN_LAM = 0.25
K_CALIB = 10
R_BOOT = 1000
SEEDS = [42, 20260823, 7]
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
DETECTOR = 'gdino-tiny'
PRE_FILTER = 0.03

TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]


# ── APPG 先验库构建 ──

def build_prior_library(prior_source='offline', prior_path=None):
    """构建物理先验库。

    Args:
        prior_source: 'offline' | 'human' | 'llm' | 'file'
        prior_path: 先验文件路径（当 prior_source='file' 时）

    Returns:
        PhysicsPriorLibrary 实例
    """
    if prior_source == 'file' and prior_path:
        print(f'[APPG] 从文件加载先验: {prior_path}')
        return PhysicsPriorLibrary.load(prior_path)

    if prior_source == 'human':
        print('[APPG] 使用论文人工赋值先验')
        return load_human_priors_from_dict(HUMAN_PRIORS_FROM_PAPER)

    if prior_source == 'llm':
        print('[APPG] 使用 LLM 生成先验（需要 API key）')
        import os
        api_key = os.environ.get('OPENAI_API_KEY', '')
        llm = LLMClient(api_key=api_key)
        gen = PhysicsPriorGenerator(llm_client=llm, use_offline_fallback=True)
        priors = gen.generate_batch([t[1] for t in TARGETS])
        lib = PhysicsPriorLibrary()
        lib.add_batch(priors)
        return lib

    # 默认: offline
    print('[APPG] 使用离线常识先验')
    lib = PhysicsPriorLibrary()
    for name, coco_cls, src_cls in TARGETS:
        v, m, s, i = get_offline_prior(coco_cls)
        prior = PhysicsPrior(
            category=coco_cls,
            V=v, M=m, S=s, I=i,
            source='offline',
            confidence=0.6,
        )
        lib.add(prior)
    return lib


def get_adaptive_params(category, prior_lib, beta_base, lam_base, alpha):
    """根据物理先验风险计算自适应 β/λ。

    β_eff = β_base * (1 + alpha * risk)
    λ_eff = λ_base * (1 + alpha * risk)

    Args:
        category: 类别名
        prior_lib: PhysicsPriorLibrary
        beta_base: 基础 β
        lam_base: 基础 λ
        alpha: 风险调制系数 [0, 1]

    Returns:
        (beta_eff, lambda_eff, risk)
    """
    prior = prior_lib.get(category)
    if prior is None:
        risk = 0.5  # 默认中等风险
    else:
        risk = prior.risk_score()

    beta_eff = beta_base * (1.0 + alpha * risk)
    lambda_eff = lam_base * (1.0 + alpha * risk)

    # 限制在合理范围
    beta_eff = min(max(beta_eff, 0.0), 6.0)
    lambda_eff = min(max(lambda_eff, 0.0), 2.0)

    return beta_eff, lambda_eff, risk


# ── 数据与特征（与 tge_v2_main.py 一致）──

def load_coco_all(cls_name):
    return U.load_coco(cls_name, n=100000)


def clip_proto(crops):
    if not crops:
        return None
    F = T.encode_clip_batch(crops)
    p = F.mean(dim=0)
    return p / p.norm()


def gt_crops(images, gts):
    crops = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
    return crops


def v2_cached_dets(images, prompt, name):
    p = DATA / f'tge_v2_dets_COCO-{name}.pkl'
    if p.exists():
        print(f'    [缓存] 检测结果 {p.name}')
        return pickle.load(open(p, 'rb'))
    dets = U.gdino_detect(images, prompt, 0.0)
    pickle.dump(dets, open(p, 'wb'))
    return dets


def collect_features(images, gts, dets):
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
            bl.append([x1, y1, x2, y2]); sl.append(float(sc)); ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
                        'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
                        'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
                        'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64)})
    if crops_all:
        F = T.encode_clip_batch(crops_all).cpu().numpy()
    else:
        F = np.zeros((0, 512))
    return per_img, F


def fused_arrays(per_img, F, W):
    for p in per_img:
        n = len(p['gd'])
        if n:
            vis = np.maximum(0.0, F[p['crop_idx']] @ W)
            p['fused'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p['fused'] = np.zeros(0)
    return per_img


def dataset_ap(per_img, key, coco_style=False):
    sel = [per_img[i] for i in range(K_CALIB, len(per_img))]
    preds = [(p['box'], p[key]) for p in sel]
    gts = [p['gt'] for p in sel]
    if coco_style:
        return U.compute_ap_coco(preds, gts)
    return U.compute_ap50_img(preds, gts)


def recall_proposals(per_img):
    n_gt, hit = 0, 0
    for p in per_img:
        gts, boxes = p['gt'], p['box']
        n_gt += len(gts)
        if len(boxes) == 0:
            continue
        matched = np.zeros(len(gts), dtype=bool)
        for b in boxes:
            if matched.all():
                break
            for j, gb in enumerate(gts):
                if matched[j]:
                    continue
                if U.iou(b.tolist(), gb.tolist()) >= 0.5:
                    matched[j] = True
        hit += int(matched.sum())
    return hit / n_gt if n_gt else float('nan')


# ── 主流程 ──

def run_single_alpha(alpha, prior_lib, beta_base, lam_base, class_data):
    """运行单个 alpha 值的实验。

    Args:
        alpha: 风险调制系数
        prior_lib: 物理先验库
        beta_base: 基础 β
        lam_base: 基础 λ
        class_data: 预加载的类别数据

    Returns:
        Dict with per-class and mean results
    """
    rows = {}
    for d in class_data:
        name = d['name']
        coco_cls = d['coco_cls']
        P_src, P_neg = d['P_src'], d['P_neg']

        # 自适应参数
        beta_eff, lambda_eff, risk = get_adaptive_params(
            coco_cls, prior_lib, beta_base, lam_base, alpha
        )

        # baseline
        base50 = dataset_ap(d['per_img'], 'gd')

        # 固定参数 A-CSPC (beta_base, lam_base)
        P_hat_fixed = P_src + beta_base * d['Delta'].to(U.DEVICE)
        P_hat_fixed = P_hat_fixed / P_hat_fixed.norm()
        W_fixed = (P_hat_fixed.cpu().numpy() - lam_base * P_neg.cpu().numpy())
        fused_arrays(d['per_img'], d['F'], W_fixed)
        fixed50 = dataset_ap(d['per_img'], 'fused')

        # 自适应参数 A-CSPC (beta_eff, lambda_eff)
        P_hat_adapt = P_src + beta_eff * d['Delta'].to(U.DEVICE)
        P_hat_adapt = P_hat_adapt / P_hat_adapt.norm()
        W_adapt = (P_hat_adapt.cpu().numpy() - lambda_eff * P_neg.cpu().numpy())
        fused_arrays(d['per_img'], d['F'], W_adapt)
        adapt50 = dataset_ap(d['per_img'], 'fused')

        rec = recall_proposals(d['per_img'])

        rows[name] = {
            'n_pool': d['n_pool'], 'n_test': d['n_test'],
            'recall': round(rec, 4),
            'risk': round(risk, 4),
            'beta_eff': round(beta_eff, 4),
            'lambda_eff': round(lambda_eff, 4),
            'base50': round(base50, 4),
            'fixed50': round(fixed50, 4),
            'adapt50': round(adapt50, 4),
            'gain_fixed': round(fixed50 - base50, 4),
            'gain_adapt': round(adapt50 - base50, 4),
            'adapt_vs_fixed': round(adapt50 - fixed50, 4),
        }
        print(f'  {name:<16} risk={risk:.3f} β={beta_eff:.3f} λ={lambda_eff:.3f} | '
              f'base={base50:.4f} fixed={fixed50:.4f} adapt={adapt50:.4f} | '
              f'Δfixed={fixed50-base50:+.4f} Δadapt={adapt50-base50:+.4f} '
              f'diff={adapt50-fixed50:+.4f}')

    # 汇总
    mean_row = {k: round(float(np.mean([rows[c][k] for c in rows])), 4)
                 for k in ['base50', 'fixed50', 'adapt50', 'gain_fixed', 'gain_adapt',
                           'adapt_vs_fixed', 'risk', 'beta_eff', 'lambda_eff']}

    return {'classes': rows, 'mean': mean_row}


def main():
    global FROZEN_BETA, FROZEN_LAM
    t0 = time.time()
    ap = argparse.ArgumentParser(description='A-CSPC + APPG: 物理先验指导的逐类自适应参数')
    ap.add_argument('--alpha', type=float, default=0.5, help='风险调制系数 (default: 0.5)')
    ap.add_argument('--alpha-sweep', type=str, default='',
                    help='逗号分隔的 alpha 值扫描 (e.g., "0.0,0.25,0.5,0.75,1.0")')
    ap.add_argument('--beta', type=float, default=FROZEN_BETA, help='基础 β (default: 1.0)')
    ap.add_argument('--lam', type=float, default=FROZEN_LAM, help='基础 λ (default: 0.25)')
    ap.add_argument('--prior-source', type=str, default='offline',
                    choices=['offline', 'human', 'llm', 'file'],
                    help='物理先验来源 (default: offline)')
    ap.add_argument('--prior-path', type=str, default='', help='先验文件路径')
    args = ap.parse_args()

    FROZEN_BETA, FROZEN_LAM = args.beta, args.lam

    print('=' * 100)
    print('A-CSPC + APPG: 物理先验指导的逐类自适应参数')
    print('=' * 100)
    print(f'配置: β_base={FROZEN_BETA} λ_base={FROZEN_LAM} prior_source={args.prior_source}')
    if args.alpha_sweep:
        print(f'Alpha 扫描: {args.alpha_sweep}')
    else:
        print(f'Alpha: {args.alpha}')

    # ── 1. 构建物理先验库 ──
    print('\n[1/4] 构建物理先验库 ...')
    prior_lib = build_prior_library(args.prior_source, args.prior_path or None)
    for name, coco_cls, src_cls in TARGETS:
        prior = prior_lib.get(coco_cls)
        if prior:
            print(f'  {coco_cls:<12} risk={prior.risk_score():.4f} '
                  f'V={prior.V} M={prior.M} S={prior.S} I={prior.I}')

    # ── 2. DIOR 源原型 ──
    print('\n[2/4] DIOR 20 类 CLIP 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p

    # ── 3. COCO 6 类检测 + 特征（只做一次）──
    print('\n[3/4] COCO 6 类全量检测 + 特征 ...')
    class_data = []
    for name, coco_cls, src_cls in TARGETS:
        images, gts = load_coco_all(coco_cls)
        if len(images) < K_CALIB + 10:
            print(f'  !! {name} 图太少 ({len(images)}), 跳过')
            continue
        dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), name)
        per_img, F = collect_features(images, gts, dets)
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        Delta = d_y - s_xs
        class_data.append({'name': name, 'per_img': per_img, 'F': F, 'gts': gts,
                           'P_src': P_src, 'P_neg': P_neg, 'Delta': Delta,
                           'd_y': d_y, 'coco_cls': coco_cls,
                           'n_pool': len(images), 'n_test': len(images) - K_CALIB})
        n_box = sum(len(p['gd']) for p in per_img)
        print(f'  {name:<16} N_pool={len(images):3d} N_test={len(images)-K_CALIB:3d} 框={n_box:6d}')

    # ── 4. 实验 ──
    print('\n[4/4] 打分 + AP ...')

    if args.alpha_sweep:
        # Alpha 扫描
        alphas = [float(x.strip()) for x in args.alpha_sweep.split(',')]
        all_results = {}
        print(f'\n{"="*100}')
        print(f'{"Alpha":<8} {"Base":>8} {"Fixed":>8} {"Adapt":>8} {"ΔFixed":>9} {"ΔAdapt":>9} {"Diff":>9} {"Pos(adapt)":>10}')
        print(f'{"-"*100}')

        for alpha in alphas:
            print(f'\n  --- alpha = {alpha} ---')
            result = run_single_alpha(alpha, prior_lib, FROZEN_BETA, FROZEN_LAM, class_data)
            all_results[f'alpha_{alpha:.2f}'] = result
            m = result['mean']
            n_pos = sum(1 for c in result['classes'].values() if c['gain_adapt'] > 0)
            print(f'  MEAN: base={m["base50"]:.4f} fixed={m["fixed50"]:.4f} adapt={m["adapt50"]:.4f} '
                  f'Δfixed={m["gain_fixed"]:+.4f} Δadapt={m["gain_adapt"]:+.4f} '
                  f'diff={m["adapt_vs_fixed"]:+.4f} pos={n_pos}/{len(result["classes"])}')

        # 汇总表
        print(f'\n{"="*100}')
        print(f'{"Alpha":<8} {"Base":>8} {"Fixed":>8} {"Adapt":>8} {"ΔFixed":>9} {"ΔAdapt":>9} {"Diff":>9} {"Pos(adapt)":>10}')
        print(f'{"-"*100}')
        for alpha in alphas:
            m = all_results[f'alpha_{alpha:.2f}']['mean']
            n_pos = sum(1 for c in all_results[f'alpha_{alpha:.2f}']['classes'].values() if c['gain_adapt'] > 0)
            print(f'{alpha:<8.2f} {m["base50"]:>8.4f} {m["fixed50"]:>8.4f} {m["adapt50"]:>8.4f} '
                  f'{m["gain_fixed"]:>+9.4f} {m["gain_adapt"]:>+9.4f} {m["adapt_vs_fixed"]:>+9.4f} '
                  f'{n_pos:>4d}/{len(all_results[f"alpha_{alpha:.2f}"]["classes"])}')
        print(f'{"="*100}')

        # 找最佳 alpha
        best_alpha = max(alphas, key=lambda a: all_results[f'alpha_{a:.2f}']['mean']['gain_adapt'])
        print(f'\n最佳 alpha = {best_alpha} (Δadapt = {all_results[f"alpha_{best_alpha:.2f}"]["mean"]["gain_adapt"]:+.4f})')

        # 保存
        out_path = DATA / 'tge_v2_appg_sweep.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {'beta_base': FROZEN_BETA, 'lam_base': FROZEN_LAM,
                           'prior_source': args.prior_source, 'alphas': alphas},
                'results': all_results,
                'best_alpha': best_alpha,
            }, f, indent=2, ensure_ascii=False)
        print(f'结果保存至: {out_path}')

    else:
        # 单个 alpha
        result = run_single_alpha(args.alpha, prior_lib, FROZEN_BETA, FROZEN_LAM, class_data)

        # 汇总表
        print(f'\n{"="*100}')
        print(f'{"Class":<16} {"Risk":>6} {"β_eff":>7} {"λ_eff":>7} {"Base":>8} {"Fixed":>8} {"Adapt":>8} '
              f'{"ΔFixed":>9} {"ΔAdapt":>9} {"Diff":>9}')
        print(f'{"-"*100}')
        for name, r in result['classes'].items():
            print(f'{name:<16} {r["risk"]:>6.3f} {r["beta_eff"]:>7.3f} {r["lambda_eff"]:>7.3f} '
                  f'{r["base50"]:>8.4f} {r["fixed50"]:>8.4f} {r["adapt50"]:>8.4f} '
                  f'{r["gain_fixed"]:>+9.4f} {r["gain_adapt"]:>+9.4f} {r["adapt_vs_fixed"]:>+9.4f}')
        print(f'{"-"*100}')
        m = result['mean']
        n_pos = sum(1 for c in result['classes'].values() if c['gain_adapt'] > 0)
        print(f'{"MEAN":<16} {m["risk"]:>6.3f} {m["beta_eff"]:>7.3f} {m["lambda_eff"]:>7.3f} '
              f'{m["base50"]:>8.4f} {m["fixed50"]:>8.4f} {m["adapt50"]:>8.4f} '
              f'{m["gain_fixed"]:>+9.4f} {m["gain_adapt"]:>+9.4f} {m["adapt_vs_fixed"]:>+9.4f}')
        print(f'正增益类别 (adapt): {n_pos}/{len(result["classes"])}')
        print(f'{"="*100}')

        # 保存
        tag = 'alpha%g' % args.alpha
        out_path = DATA / f'tge_v2_appg_{tag}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {'alpha': args.alpha, 'beta_base': FROZEN_BETA,
                           'lam_base': FROZEN_LAM, 'prior_source': args.prior_source},
                'classes': result['classes'],
                'mean': result['mean'],
            }, f, indent=2, ensure_ascii=False)
        print(f'结果保存至: {out_path}')

    print(f'\n耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
