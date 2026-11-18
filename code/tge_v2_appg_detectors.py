# -*- coding: utf-8 -*-
r"""tge_v2_appg_detectors.py — A-CSPC + APPG: 多检测器验证

在 GDINO-base 和 OWLv2 上验证物理先验自适应参数的效果。

检测器:
  gdino-base : 概率尺度, pre-filter 0.03
  owlv2      : raw-logit 尺度, pre-filter 0.03

运行:
  python tge_v2_appg_detectors.py --detector gdinob --alpha 1.5
  python tge_v2_appg_detectors.py --detector owlv2 --alpha 1.5
  python tge_v2_appg_detectors.py --detector all --alpha 1.5

输出: 实验数据/tge_v2_appg_detectors_*.json
"""
import sys, os, glob, json, pickle, argparse, time
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T
from tge_v2_main import (load_coco_all, clip_proto, gt_crops, collect_features,
                         fused_arrays, dataset_ap, recall_proposals,
                         TARGETS, FROZEN_BETA, FROZEN_LAM, K_CALIB,
                         SRC_TMPL, TGT_TMPL, DATA, PRE_FILTER)

# APPG imports
from acspc.appg import PhysicsPriorGenerator, PhysicsPriorLibrary, LLMClient
from acspc.coco_categories import get_offline_prior
from acspc.appg import PhysicsPrior

GDINOB_ROOT = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-base\snapshots'
OWLV2_ROOT = r'C:\Users\张三\.cache\huggingface\hub\models--google--owlv2-base-patch16-ensemble\snapshots'

_models = {}


def _snapshot(root):
    snaps = sorted(glob.glob(os.path.join(root, '*')))
    if not snaps:
        raise FileNotFoundError('找不到权重 snapshot: %s' % root)
    return snaps[0]


def gdinob_detect(images, prompt, thresh=0.0):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    if 'gdinob' not in _models:
        path = _snapshot(GDINOB_ROOT)
        _models['gdinob'] = (AutoProcessor.from_pretrained(path),
                             AutoModelForZeroShotObjectDetection.from_pretrained(
                                 path, dtype=torch.float32).eval().to(U.DEVICE))
        print('    [加载] GDINO-base 模型', flush=True)
    p, m = _models['gdinob']
    out = []
    for img in images:
        ins = p(images=img, text=prompt, return_tensors='pt')
        ins = {k: v.to(U.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in ins.items()}
        with torch.no_grad():
            o = m(**ins)
        ts = torch.tensor([img.size[::-1]]).to(U.DEVICE)
        r = p.post_process_grounded_object_detection(o, threshold=thresh, target_sizes=ts)[0]
        out.append((r['boxes'].cpu().numpy(), r['scores'].cpu().numpy()))
    return out


def owlv2_detect(images, prompt):
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    if 'owlv2' not in _models:
        path = _snapshot(OWLV2_ROOT)
        _models['owlv2'] = (Owlv2Processor.from_pretrained(path),
                            Owlv2ForObjectDetection.from_pretrained(
                                path, dtype=torch.float32).eval().to(U.DEVICE))
        print('    [加载] OWLv2 模型', flush=True)
    op, om = _models['owlv2']
    out = []
    for img in images:
        ins = op(text=prompt, images=img, return_tensors='pt')
        ins = {k: v.to(U.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in ins.items()}
        with torch.no_grad():
            o = om(**ins)
        ts = torch.tensor([img.size[::-1]]).to(U.DEVICE)
        r = op.post_process_object_detection(o, threshold=0.0, target_sizes=ts)[0]
        out.append((r['boxes'].cpu().numpy(), r['scores'].cpu().numpy()))
    return out


def cached_dets(images, prompt, tag, detect_fn):
    p = DATA / f'tge_v2_dets_{tag}.pkl'
    if p.exists():
        print(f'    [缓存] {p.name}')
        return pickle.load(open(p, 'rb'))
    dets = detect_fn(images, prompt)
    pickle.dump(dets, open(p, 'wb'))
    return dets


def build_prior_library(prior_source='offline', prior_path=None):
    """构建物理先验库。

    prior_source: 'offline'（默认，离线常识先验）| 'file'（从 JSON 加载冻结专家库）
    """
    from acspc.appg import load_human_priors_from_dict, HUMAN_PRIORS_FROM_PAPER
    if prior_source == 'file' and prior_path:
        print(f'[APPG] 从文件加载先验库: {prior_path}')
        return PhysicsPriorLibrary.load(prior_path)
    if prior_source == 'human':
        print('[APPG] 使用论文人工赋值先验')
        return load_human_priors_from_dict(HUMAN_PRIORS_FROM_PAPER)
    print('[APPG] 使用离线常识先验')
    lib = PhysicsPriorLibrary()
    for name, coco_cls, src_cls in TARGETS:
        v, m, s, i = get_offline_prior(coco_cls)
        prior = PhysicsPrior(category=coco_cls, V=v, M=m, S=s, I=i,
                             source='offline', confidence=0.6)
        lib.add(prior)
    return lib


def get_adaptive_params(category, prior_lib, beta_base, lam_base, alpha):
    """根据物理先验风险计算自适应 β/λ。"""
    prior = prior_lib.get(category)
    risk = prior.risk_score() if prior else 0.5
    beta_eff = min(max(beta_base * (1.0 + alpha * risk), 0.0), 6.0)
    lambda_eff = min(max(lam_base * (1.0 + alpha * risk), 0.0), 2.0)
    return beta_eff, lambda_eff, risk


def run_detector_appg(detector_name, detect_fn, tag, alpha, prior_lib, protos):
    """对单个检测器跑 APPG 自适应参数实验。"""
    print(f'\n{"="*100}')
    print(f'[检测器] {detector_name} (tag={tag}, alpha={alpha})')
    print(f'{"="*100}')

    rows = {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = load_coco_all(coco_cls)
        if len(images) < K_CALIB + 10:
            print(f'  !! {name} 图太少 ({len(images)}), 跳过')
            continue
        prompt = TGT_TMPL.format(coco_cls)
        dets = cached_dets(images, prompt, tag + '-' + name, detect_fn)
        per_img, F = collect_features(images, gts, dets)
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        Delta = d_y - s_xs

        # 自适应参数
        beta_eff, lambda_eff, risk = get_adaptive_params(
            coco_cls, prior_lib, FROZEN_BETA, FROZEN_LAM, alpha)

        # baseline
        base50 = dataset_ap(per_img, 'gd')

        # 固定参数 A-CSPC
        P_hat_fixed = P_src + FROZEN_BETA * Delta.to(U.DEVICE)
        P_hat_fixed = P_hat_fixed / P_hat_fixed.norm()
        W_fixed = (P_hat_fixed.cpu().numpy() - FROZEN_LAM * P_neg.cpu().numpy())
        fused_arrays(per_img, F, W_fixed)
        fixed50 = dataset_ap(per_img, 'fused')

        # 自适应参数 A-CSPC
        P_hat_adapt = P_src + beta_eff * Delta.to(U.DEVICE)
        P_hat_adapt = P_hat_adapt / P_hat_adapt.norm()
        W_adapt = (P_hat_adapt.cpu().numpy() - lambda_eff * P_neg.cpu().numpy())
        fused_arrays(per_img, F, W_adapt)
        adapt50 = dataset_ap(per_img, 'fused')

        rec = recall_proposals(per_img)

        rows[name] = {
            'n_pool': len(images), 'n_test': len(images) - K_CALIB,
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
    n_pos = sum(1 for c in rows.values() if c['gain_adapt'] > 0)

    print(f'\n{"-"*100}')
    print(f'{"MEAN":<16} risk={mean_row["risk"]:.3f} β={mean_row["beta_eff"]:.3f} '
          f'λ={mean_row["lambda_eff"]:.3f} | base={mean_row["base50"]:.4f} '
          f'fixed={mean_row["fixed50"]:.4f} adapt={mean_row["adapt50"]:.4f} | '
          f'Δfixed={mean_row["gain_fixed"]:+.4f} Δadapt={mean_row["gain_adapt"]:+.4f} '
          f'diff={mean_row["adapt_vs_fixed"]:+.4f}')
    print(f'正增益类别 (adapt): {n_pos}/{len(rows)}')
    print(f'{"="*100}')

    return {'classes': rows, 'mean': mean_row, 'n_pos_adapt': n_pos}


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description='A-CSPC + APPG: 多检测器验证')
    ap.add_argument('--detector', type=str, default='all',
                    choices=['gdinob', 'owlv2', 'all'],
                    help='检测器 (default: all)')
    ap.add_argument('--alpha', type=float, default=1.5, help='风险调制系数 (default: 1.5)')
    ap.add_argument('--prior-source', type=str, default='offline',
                    choices=['offline', 'human', 'file'], help='先验来源 (default: offline)')
    ap.add_argument('--prior-path', type=str, default='', help='先验文件路径 (prior-source=file 时)')
    args = ap.parse_args()

    print('=' * 100)
    print('A-CSPC + APPG: 多检测器验证')
    print('=' * 100)
    print(f'配置: α={args.alpha} β_base={FROZEN_BETA} λ_base={FROZEN_LAM} prior_source={args.prior_source}')
    print(f'检测器: {args.detector}')

    # 构建先验库
    prior_lib = build_prior_library(args.prior_source, args.prior_path or None)
    print('  逐类 risk:', {c: round(prior_lib.get(c).risk_score(), 4)
                          for _, c, _ in TARGETS if prior_lib.get(c)})

    # 构建DIOR源原型
    print('\n[DIOR 20 类 CLIP 源原型]')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p
    print(f'  原型池 {len(protos)} 类')

    results = {}

    if args.detector in ['gdinob', 'all']:
        results['gdinob'] = run_detector_appg(
            'GDINO-base', gdinob_detect, 'GDINOB', args.alpha, prior_lib, protos)

    if args.detector in ['owlv2', 'all']:
        results['owlv2'] = run_detector_appg(
            'OWLv2', owlv2_detect, 'OWLV2', args.alpha, prior_lib, protos)

    # 汇总对比
    print(f'\n{"="*100}')
    print('多检测器汇总对比')
    print(f'{"="*100}')
    print(f'{"检测器":<14} {"Base":>8} {"Fixed":>8} {"Adapt":>8} '
          f'{"ΔFixed":>9} {"ΔAdapt":>9} {"Diff":>9} {"Pos":>6}')
    print(f'{"-"*100}')
    for det_name, r in results.items():
        m = r['mean']
        print(f'{det_name:<14} {m["base50"]:>8.4f} {m["fixed50"]:>8.4f} '
              f'{m["adapt50"]:>8.4f} {m["gain_fixed"]:>+9.4f} {m["gain_adapt"]:>+9.4f} '
              f'{m["adapt_vs_fixed"]:>+9.4f} {r["n_pos_adapt"]:>4d}/6')
    print(f'{"="*100}')

    # 保存
    out_path = DATA / f'tge_v2_appg_detectors_{args.detector}_alpha{args.alpha}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {'alpha': args.alpha, 'beta_base': FROZEN_BETA,
                       'lam_base': FROZEN_LAM, 'detector': args.detector,
                       'prior_source': args.prior_source},
            'results': results,
        }, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至: {out_path}')
    print(f'耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
