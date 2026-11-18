# -*- coding: utf-8 -*-
r"""tge_selector_nwpu.py — 选择器跨域验证: DIOR 源原型 → NWPU VHR-10 目标域

背景 (2026-08-27): 4 域 (COCO/VOC/DOTA/VisDrone) LODO 未通过逐类二值路由预注册判据
(STOP D, 见 selector_exploration_protocol.md)。唯一能翻盘的实验 = 把 NWPU VHR-10
作为第 5 个【目标域】(DIOR→NWPU, 与 DOTA/VisDrone 行完全同构), 在它上面做
"4 域冻结规则 → NWPU out-of-sample" 验证。

████████████████████████  预注册冻结规则 (写本脚本前固定, 不读取 NWPU 增益)  ████████████████████
  R1 (主判定): low_band_recovery 方向 "高 → 外推有害" (4 域 ρ=-0.52 p=0.012, 唯一机制信号)。
              切分 = NWPU 自身中位数 (相对比较, 阈值只来自 NWPU 特征分布, 不读 NWPU 增益)。
  R1b(严格对照): 绝对阈值 = 4 域 7 个 fwd 负类 (gain_ext<0) 的 low_band_recovery 中位数
              (从 tge_soft_routing.json 动态计算, 不涉及 NWPU)。
  R2 (对照):  4 域 22 类逻辑回归冻结 (特征 6 维, 标签 gain_ext<0, 组内 z-score, C=1.0)。
  R3 (诚实对照): sep_src < 0.08 → 有害 (写死, 已知会误伤 DOTA-plane, 用于暴露 4 域事后拟合代价)。
████████████████████████  预注册判定标准 (主判定 R1)  ████████████████████████████████
  J1: 策略增益 > 全 ext 增益 且 > 0, bootstrap(R=1000) 95%CI 下界 > 0。
      (策略 = 预测"外推"类用 ext, 预测"回退"类用 baseline; 故策略-全ext = -Σ_回退类 gain_ext)
  J2: 预测"回退"类中, 真实 gain_ext<0 的比例 ≥ 1/2。
  两个都不满足 → 选择器永久关闭, §4.9 定稿措辞为终局。满足 → 第 5 域 case study。
████████████████████████  评估协议  █████████████████████████████████████████████████
  NWPU VHR-10 全部 10 类 (positive_image_set + ground_truth), DIOR 20 类 CLIP 源原型。
  检测: GDINO-tiny, prompt='a photo of {cls}', 缓存 tge_probe_dets_NWPU-*.pkl。
  评估: k0 全图测试 (NWPU 图少, 最大化统计力), 逐图 AP50。
  三档: baseline(gd) / B2(W0=P_src-λP_neg) / ext(W1=P_hat-λP_neg, P_hat=norm(P_src+β·Δ))。
  β=2.0, λ=0.5 (主表固定值, 不针对 NWPU 调)。
  特征: probe() 全套 (agree/sep_src/gd_hi/small/conf/entropy/bpi/
        high_band_ext_agree/low_band_recovery/low_band_lift) + g1_roundtrip。
  统计: 按图配对 bootstrap, R=1000, seed 42。

用法:
  python tge_selector_nwpu.py          # 全量 (检测 ~600 图 + CLIP 编码, RTX5060 ~30-60 分钟)
  python tge_selector_nwpu.py --dry    # 冒烟: 只 import + NWPU 加载 + 类映射 + R1b 阈值, 不跑检测

输出: 实验数据/tge_selector_nwpu.json + 控制台判定表
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import argparse
import json
import re
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr, entropy
import unified_eval as U
import tge_probe as T
from tge_v2_main import (clip_proto, gt_crops, collect_features, fused_arrays,
                         SRC_TMPL, TGT_TMPL, DATA)

BETA = 2.0
LAM = U.NEG_LAM                 # 0.5, 主表固定
HIGH_THR = 0.3                  # 高段一致性阈值 (冻结, 同 tge_soft_routing)
LOW_PCT = 30                    # 低分桶 (冻结)
SEP_HARD = 0.08                 # R3 诚实对照阈值 (写死)
R_BOOT = 1000
SEED = 42

NWPU_DIR = Path(r'C:\Users\张三\modeling\nwpu_data\NWPU-VHR-10')
NWPU_CLASSES = [
    'airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court',
    'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle',
]
NWPU_CLS_ID = {c: i + 1 for i, c in enumerate(NWPU_CLASSES)}
# NWPU 类别名 → DIOR 源原型类名 (全部可对齐, 见 unified_eval.DIOR_ID2NAME)
NWPU_TO_DIOR = {
    'airplane': 'airplane', 'ship': 'ship', 'storage tank': 'storage tank',
    'baseball diamond': 'baseball field', 'tennis court': 'tennis court',
    'basketball court': 'basketball court', 'ground track field': 'ground track field',
    'harbor': 'harbor', 'bridge': 'bridge', 'vehicle': 'vehicle',
}
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
FEATS6 = ['agree', 'sep_src', 'gd_hi_frac', 'small_frac', 'low_band_recovery', 'low_band_lift']


# ── NWPU 加载 (同 tge_probe_nwpu.load_nwpu, 取全部含 GT 图) ──
def _find_img(img_dir, stem):
    for ext in ('.jpg', '.JPG', '.jpeg', '.png', '.tif'):
        for cand in (stem, stem.lstrip('0') or '0', stem.zfill(4)):
            p = img_dir / (cand + ext)
            if p.exists():
                return p
    return None


def load_nwpu(cls_name, n=100000):
    ann_dir = NWPU_DIR / 'ground_truth'
    img_dir = NWPU_DIR / 'positive_image_set'
    cls_id = NWPU_CLS_ID[cls_name]
    images, gts = [], []
    if not ann_dir.exists():
        return images, gts
    for txt_path in sorted(ann_dir.glob('*.txt')):
        boxes = []
        for line in txt_path.read_text(encoding='utf-8', errors='ignore').splitlines():
            line = line.strip()
            if not line:
                continue
            nums = [int(v) for v in re.findall(r'-?\d+', line)]
            if len(nums) < 5:
                continue
            x1, y1, x2, y2, cat = nums[-5], nums[-4], nums[-3], nums[-2], nums[-1]
            if cat != cls_id:
                continue
            boxes.append([x1, y1, x2, y2])
        if not boxes:
            continue
        img_path = _find_img(img_dir, txt_path.stem)
        if img_path is None:
            continue
        from PIL import Image
        images.append(Image.open(img_path).convert('RGB'))
        gts.append(boxes)
        if len(images) >= n:
            break
    return images, gts


# ── 4 域冻结数据的准备 (R1b 阈值 + R2 训练) ──
def load_4dom():
    """从 tge_soft_routing.json 提取 4 域 22 个 fwd 类 (特征 + gain), 供 R1b/R2 冻结用。"""
    sr = json.load(open(DATA / 'tge_soft_routing.json', encoding='utf-8'))
    rows = {}
    for n, c in sr['classes'].items():
        if n.startswith('REV-'):
            continue
        f = c['feats']
        rows[n] = {k: f.get(k) for k in FEATS6}
        rows[n]['gain_ext'] = c['curve']['1.0'] - c['curve']['base']
    return rows


def r1b_threshold(rows4):
    """R1b 绝对阈值 = 4 域外推有害类 (gain_ext<0) 的 low_band_recovery 中位数。"""
    neg = [rows4[n]['low_band_recovery'] for n in rows4 if rows4[n]['gain_ext'] < 0]
    return float(np.median(neg)) if neg else None


# ── 冻结特征 (同 tge_soft_routing.probe) ──
def probe(per_img, F, P_src, P_hat):
    gds, srcs, hats, n_boxes, n_small = [], [], [], 0, 0
    for p in per_img:
        n = len(p['gd'])
        if n:
            gds.append(p['gd'])
            srcs.append(F[p['crop_idx']] @ P_src.cpu().numpy())
            hats.append(F[p['crop_idx']] @ P_hat.cpu().numpy())
            if p['box'].size:
                areas = (p['box'][:, 2] - p['box'][:, 0]) * (p['box'][:, 3] - p['box'][:, 1])
                n_small += int(np.sum(areas < 32 * 32))
        n_boxes += n
    gd = np.concatenate(gds) if gds else np.zeros(0)
    src = np.concatenate(srcs) if srcs else np.zeros(0)
    hat = np.concatenate(hats) if hats else np.zeros(0)
    n_img = max(1, len(per_img))

    feats = {}
    if len(gd) >= 10:
        feats['agree'] = round(float(spearmanr(gd, src)[0]), 4)
        feats['sep_src'] = round(float(src.std()), 4)
        feats['gd_hi_frac'] = round(float(np.mean(gd > HIGH_THR)), 4)
        feats['small_frac'] = round(n_small / max(1, n_boxes), 4)
        feats['gd_mean_conf'] = round(float(gd.mean()), 4)
        feats['gd_entropy'] = round(float(entropy(np.histogram(gd, bins=20, range=(0, 1))[0] + 1e-9)), 4)
        feats['boxes_per_img'] = round(n_boxes / n_img, 2)
        hi = gd > HIGH_THR
        feats['high_band_ext_agree'] = (round(float(spearmanr(gd[hi], hat[hi])[0]), 4)
                                        if hi.sum() >= 10 else None)
        lo = gd < np.percentile(gd, LOW_PCT)
        feats['low_band_recovery'] = (round(float(hat[lo].mean()), 4) if lo.sum() else None)
        feats['low_band_lift'] = (round(float(hat[lo].mean() - src[lo].mean()), 4)
                                  if lo.sum() else None)
        feats['n_boxes'] = int(len(gd))
    return feats


# ── 冻结路由器 ──
def predict_rules(nwpu_feats, median_thr, abs_thr, clf, scaler):
    """返回每个 NWPU 类在所有冻结规则下的预测 (True=外推有害→回退)。"""
    pred = {'R1': [], 'R1b': [], 'R2': [], 'R3': []}
    for n in nwpu_feats:
        f = nwpu_feats[n]
        lbr = f['low_band_recovery']
        pred['R1'].append(bool(lbr > median_thr))           # NWPU 自身中位数
        pred['R1b'].append(bool(lbr > abs_thr))             # 4 域绝对阈值
        x = np.array([[f[k] for k in FEATS6]], dtype=float)
        xs = (x - scaler.mean_) / scaler.scale_
        pred['R2'].append(bool(clf.predict_proba(xs)[0, 1] >= 0.5))
        pred['R3'].append(bool(f['sep_src'] < SEP_HARD))
    return pred


# ── 主流程 ──
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true', help='冒烟: 不跑检测, 只验证加载/映射/冻结阈值')
    args = ap.parse_args()

    print('[0/5] 加载 4 域冻结数据 (tge_soft_routing.json, 供 R1b/R2 冻结) ...')
    rows4 = load_4dom()
    abs_thr = r1b_threshold(rows4)
    assert abs_thr is not None
    print(f'  4 域外推有害类 {sum(1 for n in rows4 if rows4[n]["gain_ext"]<0)} 个, '
          f'low_band_recovery 中位数 (R1b 绝对阈值) = {abs_thr:.4f}')

    # R2: 4 域 22 类逻辑回归 (预注册: 特征 6 维, 标签 gain_ext<0, C=1.0, 组内 z-score)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X4 = np.array([[rows4[n][k] for k in FEATS6] for n in rows4], dtype=float)
    y4 = np.array([rows4[n]['gain_ext'] < 0 for n in rows4])
    scaler = StandardScaler().fit(X4)
    Xs4 = scaler.transform(X4)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xs4, y4)
    print(f'  R2 训练: 22 类, 正例 {y4.sum()} (外推有害), 特征 {FEATS6}')

    print('[1/5] DIOR 源原型 ...')
    if args.dry:
        # 静态映射校验 (不加载图片, dry 快速)
        miss = [c for c in NWPU_TO_DIOR.values() if c not in U.DIOR_ID2NAME.values()]
        print(f'  [dry] DIOR 类名映射缺失: {miss if miss else "无"} (静态校验, 未加载图片)')
        protos = None
    else:
        protos = {}
        for cls in U.DIOR_ID2NAME.values():
            images, gts = U.load_dior(cls)
            p = clip_proto(gt_crops(images, gts))
            if p is not None:
                protos[cls] = p
        miss = [c for c in NWPU_TO_DIOR.values() if c not in protos]
        print(f'  DIOR 原型 {len(protos)} 个, NWPU 映射缺失: {miss if miss else "无"}')

    print('[2/5] 加载 NWPU 10 类目标域图集 ...')
    nwpu_data = {}
    for cls in NWPU_CLASSES:
        images, gts = load_nwpu(cls)
        nwpu_data[cls] = (images, gts)
        print('  %-16s N=%3d 图' % (cls, len(images)))

    if args.dry:
        print('\n[dry] 冒烟通过: import / NWPU 加载 / 类映射 / R1b 阈值 / R2 训练均 OK。未跑检测。')
        print('  冻结规则汇总: R1 方向=high→harmful, R1b 阈值=%.4f, R3 阈值=sep<%.2f' %
              (abs_thr, SEP_HARD))
        return

    print('[3/5] GDINO-tiny 检测 (NWPU 10 类, ~600 图) ...')
    dets = {}
    for cls in NWPU_CLASSES:
        images, _ = nwpu_data[cls]
        dets[cls] = T.cached_dets(images, TGT_TMPL.format(cls), 'NWPU-' + cls)

    print('[4/5] 特征 + 三档打分 (baseline/B2/ext) ...')
    per_cls = {}
    for cls in NWPU_CLASSES:
        images, gts = nwpu_data[cls]
        if len(images) < 10:
            print('  !! %s 图太少 (%d), 跳过' % (cls, len(images)))
            continue
        per_img, F = collect_features(images, gts, dets[cls])
        src_cls = NWPU_TO_DIOR[cls]
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(cls)]).cpu()[0]
        Delta = d_y - s_xs
        P_hat = P_src + BETA * Delta.to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        feats = probe(per_img, F, P_src, P_hat)
        feats['g1_roundtrip'] = round(float(T.roundtrip_residual(P_src.cpu(), Delta, BETA)), 4)

        a_base = U.compute_ap50_img([(p['box'], p['gd']) for p in per_img],
                                    [p['gt'] for p in per_img])
        W0 = (P_src.cpu().numpy() - LAM * P_neg.cpu().numpy())
        fused_arrays(per_img, F, W0)
        a_b2 = U.compute_ap50_img([(p['box'], p['fused']) for p in per_img],
                                  [p['gt'] for p in per_img])
        W1 = (P_hat.cpu().numpy() - LAM * P_neg.cpu().numpy())
        fused_arrays(per_img, F, W1)
        a_ext = U.compute_ap50_img([(p['box'], p['fused']) for p in per_img],
                                   [p['gt'] for p in per_img])
        per_cls[cls] = {'per_img': per_img, 'F': F, 'feats': feats,
                        'base': a_base, 'b2': a_b2, 'ext': a_ext,
                        'gain_ext': a_ext - a_base, 'n_img': len(images)}
        print('  %-16s base=%.4f  B2=%.4f  ext=%.4f  (ext-base)%+.4f  lbr=%.4f sep=%.4f' %
              (cls, a_base, a_b2, a_ext, a_ext - a_base,
               feats['low_band_recovery'], feats['sep_src']))

    # R1 中位数 (NWPU 自身)
    lbr_vals = [per_cls[c]['feats']['low_band_recovery'] for c in per_cls]
    median_thr = float(np.median(lbr_vals))
    print(f'\n  R1 阈值 (NWPU 自身中位数) = {median_thr:.4f}')

    nwpu_feats = {c: per_cls[c]['feats'] for c in per_cls}
    preds = predict_rules(nwpu_feats, median_thr, abs_thr, clf, scaler)

    print('\n[5/5] 判定 (冻结规则 vs NWPU 真实增益) ...')
    print('=' * 100)
    print('%-18s %8s %8s %8s | %5s %5s %5s %5s | %8s' %
          ('class', 'base', 'ext', 'gain', 'R1', 'R1b', 'R2', 'R3', 'lbr'))
    for c in per_cls:
        d = per_cls[c]
        g = d['gain_ext']
        flag = '  <-- 外推有害' if g < 0 else ''
        print('%-18s %8.4f %8.4f %+8.4f | %5s %5s %5s %5s | %8.4f%s' %
              (c, d['base'], d['ext'], g,
               '回退' if preds['R1'][NWPU_CLASSES.index(c)] else '外推',
               '回退' if preds['R1b'][NWPU_CLASSES.index(c)] else '外推',
               '回退' if preds['R2'][NWPU_CLASSES.index(c)] else '外推',
               '回退' if preds['R3'][NWPU_CLASSES.index(c)] else '外推',
               d['feats']['low_band_recovery'], flag))
    print('=' * 100)

    # ── bootstrap (按图配对, 固定预测) ──
    rng = np.random.RandomState(SEED)
    cls_order = [c for c in NWPU_CLASSES if c in per_cls]
    idxR1 = [i for i, c in enumerate(cls_order) if preds['R1'][NWPU_CLASSES.index(c)]]
    strat_gains, allext_gains = [], []
    for _ in range(R_BOOT):
        g_ext, g_base = [], []
        for c in cls_order:
            d = per_cls[c]
            n = len(d['per_img'])
            sel = [d['per_img'][i] for i in rng.randint(0, n, size=n)]
            gb = U.compute_ap50_img([(p['box'], p['gd']) for p in sel], [p['gt'] for p in sel])
            ge = U.compute_ap50_img([(p['box'], p['fused']) for p in sel], [p['gt'] for p in sel])
            g_ext.append(ge - gb); g_base.append(0.0)
        g_ext = np.array(g_ext)
        strat = sum(g_ext[i] for i in range(len(cls_order)) if i not in idxR1)  # 预测外推类
        strat_gains.append(strat)
        allext_gains.append(g_ext.sum())
    strat_gains = np.array(strat_gains)
    allext_gains = np.array(allext_gains)
    delta = strat_gains - allext_gains   # = -Σ_回退类 gain_ext

    def ci95(x):
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    s_ci, a_ci, d_ci = ci95(strat_gains), ci95(allext_gains), ci95(delta)

    # J2 命中率
    back_cls = [c for c in cls_order if preds['R1'][NWPU_CLASSES.index(c)]]
    hit = sum(1 for c in back_cls if per_cls[c]['gain_ext'] < 0)
    hit_rate = hit / len(back_cls) if back_cls else 0.0

    print('\n─ 主判定 R1 (NWPU 自身中位数切分, 回退类 = 高 lbr 半) ─')
    print('  回退类: %s' % back_cls)
    print('  J2 命中率: %d/%d = %.2f  (需 >= 1/2)' % (hit, len(back_cls), hit_rate))
    print('  策略增益 mean=%.4f CI[%.4f, %.4f]  全ext增益 mean=%.4f CI[%.4f, %.4f]'
          % (strat_gains.mean(), s_ci[0], s_ci[1], allext_gains.mean(), a_ci[0], a_ci[1]))
    print('  策略−全ext Δ mean=%+.4f CI[%+.4f, %+.4f]   (需 上界<0 即回退类净贡献为正)'
          % (delta.mean(), d_ci[0], d_ci[1]))
    j1_ok = (delta.mean() > 0) and (strat_gains.mean() > 0) and (d_ci[1] < 0)
    j2_ok = hit_rate >= 0.5
    print('  J1: %s   J2: %s' % ('PASS' if j1_ok else 'FAIL', 'PASS' if j2_ok else 'FAIL'))
    verdict = ('PROCEED → 第5域 case study' if (j1_ok and j2_ok) else
               'STOP → §4.9 定稿措辞为终局')

    # ── 对照规则命中率 ──
    print('\n─ 对照规则 (均按同一判定口径: 预测回退类中真实 gain_ext<0 比例) ─')
    for rule in ['R1b', 'R2', 'R3']:
        bc = [c for c in cls_order if preds[rule][NWPU_CLASSES.index(c)]]
        h = sum(1 for c in bc if per_cls[c]['gain_ext'] < 0)
        print('  %-4s 回退类=%2d 命中=%d 命中率=%.2f' % (rule, len(bc), h,
                                                        h / len(bc) if bc else float('nan')))

    print('\n综合判定: %s' % verdict)

    out = {'config': {'protocol': 'DIOR→NWPU target-domain selector validation',
                      'beta': BETA, 'lam': LAM, 'high_thr': HIGH_THR, 'low_pct': LOW_PCT,
                      'r1_rule': 'low_band_recovery > NWPU-median (high→harmful)',
                      'r1b_thr': abs_thr, 'r2': '4dom logistic C=1.0', 'r3_thr': SEP_HARD,
                      'eval': 'k0 all images, per-image AP50, R=1000 seed 42',
                      'frozen_before_nwpu_gain': True},
           'classes': {c: {'n_img': per_cls[c]['n_img'], 'base': round(per_cls[c]['base'], 4),
                           'b2': round(per_cls[c]['b2'], 4), 'ext': round(per_cls[c]['ext'], 4),
                           'gain_ext': round(per_cls[c]['gain_ext'], 4),
                           'feats': per_cls[c]['feats']} for c in per_cls},
           'preds': {c: {r: bool(preds[r][NWPU_CLASSES.index(c)]) for r in ['R1', 'R1b', 'R2', 'R3']}
                     for c in cls_order},
           'judgment': {'r1_median_thr': median_thr, 'back_cls': back_cls,
                        'j2_hit': hit, 'j2_total': len(back_cls), 'j2_rate': hit_rate,
                        'strat_gain': [round(strat_gains.mean(), 4), [round(x, 4) for x in s_ci]],
                        'allext_gain': [round(allext_gains.mean(), 4), [round(x, 4) for x in a_ci]],
                        'delta': [round(delta.mean(), 4), [round(x, 4) for x in d_ci]],
                        'j1': bool(j1_ok), 'j2': bool(j2_ok), 'verdict': verdict}}
    with open(DATA / 'tge_selector_nwpu.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n→ 实验数据/tge_selector_nwpu.json')


if __name__ == '__main__':
    main()
