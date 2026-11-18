# -*- coding: utf-8 -*-
"""probe_reverse_cond_adaptive.py — 反向(photo→overhead)条件自适应校正 小样本探针

背景: A-CSPC 汇报稿已知边界: 域方向选择性 —— photo→overhead 反向失败 (-0.2394),
      且论文已判定主因是"文本外推方向失配"而非负原型污染。

本探针检验: 能否用**图像估计的 realized condition**(V4Net z_c 域身份) 决定外推方向
           (delta 的符号), 从而在反向任务上把失败翻正。

关键简化: delta_rev = -delta_fwd
  - 前向 DIOR→COCO: delta_fwd = E('a photo of c') - E('an overhead image of c')
  - 反向 COCO→DIOR: delta_rev = E('an overhead image of c') - E('a photo of c') = -delta_fwd
  所以"反向仍用 +delta_fwd"= 错误方向(论文失败的那条); "用 -delta_fwd"= 正确方向。

三部分:
  A. 门控可靠性: V4Net z_c 能否分离 COCO(natural) vs DIOR(overhead) —— 机制前提(AUC)。
  B. 反向校正:   3 个 DIOR 目标类, 比较
                 (0) baseline 原始检测器分
                 (1) wrong-direction: +delta_fwd (论文失败配置)
                 (2) correct-direction: -delta_fwd (正确反向)
  C. 条件自适应: 逐图用 z_c 门控预测域 → 决定 delta 符号 → 得到 (3) gate-driven。
                 若门控可靠, (3) 应≈(2) 且显著优于 (1), 且不需要被告知源/目标域。

用法:
  "C:/Users/张三/GroundingDINO/venv/Scripts/python.exe" probe_reverse_cond_adaptive.py [--n 30] [--selftest]
"""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ['AC_SPC_COCO_ANN'] = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'
os.environ['AC_SPC_COCO_IMG'] = r'C:\Users\张三\modeling\coco_data\val2017'
os.environ['AC_SPC_DIOR_IMG'] = r'C:\Users\张三\modeling\dior_data\full20_images'
os.environ['AC_SPC_DIOR_PARQUET'] = r'C:\Users\张三\modeling\dior_data\dior_combined.parquet'

import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')       # acspc 包
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')       # V4Net / unified_eval

import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from acspc.config import Config
from acspc import data as D
from acspc import models as M
from acspc import metrics as Met
from acspc import prototype as Proto

# ───────── 参数 ─────────
BETA = 2.0          # 主配置 extrapolation strength
LAM = 0.5           # 主配置 负原型抑制
NEG_TOPK = 3
PRE_FILTER = 0.03

# 反向评估对: (DIOR 目标类, COCO 源类)
REVERSE_PAIRS = [
    ('airplane', 'airplane'),
    ('ship', 'boat'),
    ('vehicle', 'car'),
]
# COCO 源原型池(用于负原型 top-k 近邻)——比评估类更全
COCO_SRC_POOL = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']

CACHE_DIR = Path(r'C:\Users\张三\Desktop\实验数据\probe_reverse_cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(*a, flush=True)


# ───────── numpy 度量 ─────────
def roc_auc(y, s):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    order = np.argsort(s, kind='mergesort')
    ranks = np.arange(1, len(s) + 1, dtype=float)
    pos = ranks[y[order] == 1]
    n_pos, n_neg = len(pos), len(y) - len(pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    u = pos.sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def fit_ls(X, y):
    """最小二乘线性判别: 返回权重 w (含偏置在最后一维)"""
    A = np.hstack([X, np.ones((len(X), 1))])
    w, *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
    return w


def pred_ls(X, w):
    A = np.hstack([X, np.ones((len(X), 1))])
    return A @ w


# ───────── CLIP (transformers 版) ─────────
# 注意: venv 内 `clip` 包为空壳(无 load), 且 ~/.cache/clip/ViT-B-32.pt 是 CUDA 绑定的 JIT
#       (CPU torch 跑不了) → 改用 transformers 的 openai/clip-vit-base-patch32(即 OpenAI CLIP
#       ViT-B/32), HF 缓存中已有, 可离线加载。特征 L2 归一化, 与 A-CSPC 用法一致。
_CLIP = {}
DEVICE = 'cpu'
CLIP_ID = 'openai/clip-vit-base-patch32'


def get_clip():
    if 'model' not in _CLIP:
        from transformers import CLIPModel, CLIPProcessor
        _CLIP['model'] = CLIPModel.from_pretrained(CLIP_ID).to(DEVICE).eval()
        _CLIP['proc'] = CLIPProcessor.from_pretrained(CLIP_ID)
    return _CLIP['model'], _CLIP['proc']


def clip_encode_imgs(crops, bs=64):
    """返回 L2 归一化图像特征 (N,512)"""
    model, proc = get_clip()
    feats = []
    for i in range(0, len(crops), bs):
        inputs = proc(images=crops[i:i + bs], return_tensors='pt', padding=True).to(DEVICE)
        with torch.no_grad():
            f = model.get_image_features(**inputs).float()
        feats.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(feats).cpu()


def text_emb(cfg, t):
    """返回单个 L2 归一化文本向量 (512,)"""
    model, proc = get_clip()
    inputs = proc(text=[t], return_tensors='pt', padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        f = model.get_text_features(**inputs).float()
    return (f / f.norm(dim=-1, keepdim=True)).cpu()[0]


def delta_fwd(cfg, cls):
    """前向 delta = E('a photo of c') - E('an overhead image of c')"""
    e_photo = text_emb(cfg, f'a photo of {cls}')
    e_over = text_emb(cfg, f'an overhead image of {cls}')
    return e_photo - e_over


# ───────── 检测 + 编码 (带缓存) ─────────
def detect_encode_reverse(cfg, images, gts, prompt, key):
    """在 DIOR 目标图上跑 GDINO + CLIP 编码候选框。"""
    cp = CACHE_DIR / f'{key}.pkl'
    if cp.exists():
        log(f'    [cache] {cp.name}')
        return pickle.load(open(cp, 'rb'))

    log(f'    running GDINO on {len(images)} DIOR images ...')
    dets = M.gdino_detect(cfg, images, prompt, cfg.model.detection_threshold)

    per_img, crops_all = [], []
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

    F = clip_encode_imgs(crops_all).numpy() if crops_all else np.zeros((0, 512))
    blob = {'per_img': per_img, 'F': F}
    pickle.dump(blob, open(cp, 'wb'))
    return blob


def fuse_scores(per_img, F, W, out_key='fused'):
    for p in per_img:
        n = len(p['gd'])
        if n:
            vis = np.maximum(0.0, F[p['crop_idx']] @ W)
            p[out_key] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p[out_key] = np.zeros(0)
    return per_img


def ap_of(per_img, key):
    preds = [(p['box'], p[key]) for p in per_img]
    gts = [p['gt'] for p in per_img]
    return Met.compute_ap50_img(preds, gts)


def np_W(P_hat, P_neg, lam):
    return (P_hat.cpu().numpy() - lam * P_neg.cpu().numpy())


# ───────── 主流程 ─────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30, help='每类 DIOR 图像数')
    ap.add_argument('--n-proto', type=int, default=20, help='每个 COCO 源类的原型图像数')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        log('selftest: 仅验证 import + 数据加载, 不跑检测')
        cfg = Config()
        cfg.model.device = 'cpu'
        imgs, gts = D.load_coco(cfg, 'airplane', n=3)
        log(f'  COCO airplane: {len(imgs)} imgs')
        dimgs, dgts = D.load_dior(cfg, 'airplane', n=3)
        log(f'  DIOR airplane: {len(dimgs)} imgs')
        log('SELFTEST OK')
        return

    cfg = Config()
    cfg.model.device = 'cpu'
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = PRE_FILTER
    # A-CSPC 已知 bug: get_gdino 读 cfg.model.gdino_cache, 但该字段定义在 cfg.paths
    # (run_table6 靠手动赋值绕过; 这里必须显式设置, 否则 AttributeError)
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    log('=' * 74)
    log('反向(photo→overhead) 条件自适应校正 小样本探针')
    log(f'  BETA={BETA}  LAM={LAM}  N={args.n}/类   检测器=gdino-tiny  CLIP=ViT-B/32')
    log(f'  反向对: {REVERSE_PAIRS}')
    log('=' * 74)

    # ── 1. COCO 源原型 (源域 = COCO/photo) ──
    log('\n[1/4] 构建 COCO 源原型 (源域=COCO) ...')
    protos = {}
    for cls in COCO_SRC_POOL:
        images, gts = D.load_coco(cfg, cls, n=args.n_proto)
        crops = D.gt_crops(images, gts)
        if not crops:
            log(f'  !! {cls}: 无 GT crops, 跳过')
            continue
        F = clip_encode_imgs(crops)
        p = F.mean(dim=0)
        protos[cls] = p / p.norm()
        log(f'  {cls:<12} {len(crops)} crops')
    log(f'  构建 {len(protos)} 个 COCO 源原型')

    # ── 2. V4Net 门控可靠性 (COCO vs DIOR 域分离) ──
    log('\n[2/4] 门控可靠性: V4Net z_c 分离 COCO vs DIOR ...')
    gate = {'available': False}
    net = None
    try:
        import phaseB_aux_probe as P
        import unified_eval as U
        net = P.V4Net(content_ch=64, n_class=20)
        sd = torch.load(U.SAVE_DIR / 'unified_space_v4.pt', map_location='cpu')
        net.load_state_dict(sd)
        net.eval()
        log('  V4Net 加载成功')
    except Exception as e:
        log(f'  !! V4Net 加载失败: {e}')
        log('     门控部分将跳过, 仅跑 (1)(2) 方向对照')

    if net is not None:
        coco_imgs = []
        for cls in ['airplane', 'boat', 'car']:
            ims, _ = D.load_coco(cfg, cls, n=20)
            coco_imgs += ims[:20]
        dior_imgs = []
        for cls in ['airplane', 'ship', 'vehicle']:
            ims, _ = D.load_dior(cfg, cls, n=20)
            dior_imgs += ims[:20]
        log(f'  COCO={len(coco_imgs)} 张, DIOR={len(dior_imgs)} 张')

        Zc_coco, _ = P.encode_full(net, coco_imgs, bs=8)
        Zc_dior, _ = P.encode_full(net, dior_imgs, bs=8)
        X = np.vstack([Zc_coco, Zc_dior])
        y = np.array([0] * len(Zc_coco) + [1] * len(Zc_dior))  # 1 = overhead(DIOR)
        w = fit_ls(X, y)
        s = pred_ls(X, w)
        auc = roc_auc(y, s)
        acc = float(((s > 0.5).astype(int) == y).mean())
        gate = {'available': True, 'auc_zc': round(auc, 4), 'acc_zc': round(acc, 4),
                'n_coco': len(Zc_coco), 'n_dior': len(Zc_dior), 'w': w.tolist()}
        log(f'  z_c 域分类: AUC={auc:.4f}  acc={acc:.4f} (随机 0.5)')

    # ── 3/4. 反向逐类评估 ──
    log('\n[3/4] 反向逐类评估 (目标=DIOR/overhead) ...')
    results = {}
    for dior_cls, coco_cls in REVERSE_PAIRS:
        log(f'\n  [DIOR-{dior_cls}] ← COCO-{coco_cls}')
        if coco_cls not in protos:
            log('    !! 源原型缺失, 跳过')
            continue
        images, gts = D.load_dior(cfg, dior_cls, n=args.n)
        if len(images) < 10:
            log(f'    !! DIOR {dior_cls} 图像不足 ({len(images)}), 跳过')
            continue

        prompt = f'an overhead image of {coco_cls}'
        key = f'rev_{dior_cls}_from_{coco_cls}_n{len(images)}'
        blob = detect_encode_reverse(cfg, images, gts, prompt, key)
        per_img, F = blob['per_img'], blob['F']
        log(f'    N={len(images)}  boxes={sum(len(p["gd"]) for p in per_img)}')

        P_src = protos[coco_cls]
        P_neg = Proto.get_negative_prototype(protos, coco_cls, NEG_TOPK)
        d_fwd = delta_fwd(cfg, coco_cls)          # +delta (前向/错误方向)
        d_rev = -d_fwd                            # -delta (反向/正确方向)

        # (0) baseline
        base50 = ap_of(per_img, 'gd')

        # (1) wrong-direction: +delta_fwd
        W_wrong = np_W(Proto.extrapolate_positive(P_src, d_fwd, BETA), P_neg, LAM)
        ap_wrong = ap_of(fuse_scores(per_img, F, W_wrong, 'f_wrong'), 'f_wrong')

        # (2) correct-direction: -delta_fwd
        W_corr = np_W(Proto.extrapolate_positive(P_src, d_rev, BETA), P_neg, LAM)
        ap_corr = ap_of(fuse_scores(per_img, F, W_corr, 'f_corr'), 'f_corr')

        # (3) condition-adaptive gate: 逐图由 z_c 预测域 → 决定 delta 符号
        ap_gate = None
        gate_acc_here = None
        if net is not None:
            Zc_t, _ = P.encode_full(net, images, bs=8)
            s_t = pred_ls(Zc_t, np.array(gate['w']))
            is_overhead = (s_t > 0.5)                    # 预测为 overhead → 用反向 delta
            gate_acc_here = float(is_overhead.mean())    # 目标全是 DIOR, 期望≈1.0
            for p, use_rev in zip(per_img, is_overhead):
                n = len(p['gd'])
                if n:
                    Wp = W_corr if use_rev else W_wrong
                    vis = np.maximum(0.0, F[p['crop_idx']] @ Wp)
                    p['f_gate'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
                else:
                    p['f_gate'] = np.zeros(0)
            ap_gate = ap_of(per_img, 'f_gate')

        r = {
            'dior_cls': dior_cls, 'coco_src': coco_cls, 'n_images': len(images),
            'baseline_ap50': round(base50, 4),
            'wrong_dir_ap50': round(ap_wrong, 4),
            'wrong_dir_gain': round(ap_wrong - base50, 4),
            'correct_dir_ap50': round(ap_corr, 4),
            'correct_dir_gain': round(ap_corr - base50, 4),
        }
        if ap_gate is not None:
            r['gate_ap50'] = round(ap_gate, 4)
            r['gate_gain'] = round(ap_gate - base50, 4)
            r['gate_pick_reverse_frac'] = round(gate_acc_here, 4)
        results[f'DIOR-{dior_cls}'] = r
        gs = f'  gate={ap_gate - base50:+.4f} (reverse frac {gate_acc_here:.2f})' if ap_gate is not None else ''
        log(f'    base={base50:.4f}  wrong(+δ)={ap_wrong - base50:+.4f}  '
            f'correct(-δ)={ap_corr - base50:+.4f}{gs}')

    # ── 4. 汇总 ──
    log('\n[4/4] 汇总')
    if not results:
        log('  !! 无任何结果')
        return
    log(f"  {'Class':<16}{'Base':>8}{'wrong+δ':>10}{'correct-δ':>11}{'gate':>10}")
    log('  ' + '-' * 55)
    for k, r in results.items():
        gv = f"{r['gate_gain']:+.4f}" if 'gate_gain' in r else '—'
        log(f"  {k:<16}{r['baseline_ap50']:>8.4f}{r['wrong_dir_gain']:>+10.4f}"
            f"{r['correct_dir_gain']:>+11.4f}{gv:>10}")
    log('  ' + '-' * 55)
    mean_wrong = float(np.mean([r['wrong_dir_gain'] for r in results.values()]))
    mean_corr = float(np.mean([r['correct_dir_gain'] for r in results.values()]))
    gg = [r['gate_gain'] for r in results.values() if 'gate_gain' in r]
    mean_gate = float(np.mean(gg)) if gg else None
    mgs = f'{mean_gate:+.4f}' if mean_gate is not None else '—'
    log(f"  {'MEAN':<16}{'':>8}{mean_wrong:>+10.4f}{mean_corr:>+11.4f}{mgs:>10}")

    out = {
        'config': {'beta': BETA, 'lam': LAM, 'neg_topk': NEG_TOPK, 'n_per_class': args.n,
                   'detector': 'gdino-tiny', 'reverse_pairs': REVERSE_PAIRS,
                   'note': 'delta_rev = -delta_fwd; wrong=+delta (论文失败方向), correct=-delta (正确反向)'},
        'gate_reliability': gate,
        'classes': results,
        'mean': {'wrong_dir_gain': round(mean_wrong, 4), 'correct_dir_gain': round(mean_corr, 4),
                 'gate_gain': round(mean_gate, 4) if mean_gate is not None else None,
                 'n_classes': len(results)},
    }
    op = Path(r'C:\Users\张三\Desktop\实验数据') / 'probe_reverse_cond_adaptive.json'
    json.dump(out, open(op, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    log(f'\n结果 → {op}')
    log('=' * 74)
    log('判定: correct/gate 应显著优于 wrong。若 wrong≈-0.24 而 correct 转正 → 机制成立。')
    log('=' * 74)


if __name__ == '__main__':
    main()
