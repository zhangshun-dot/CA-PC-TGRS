# -*- coding: utf-8 -*-
"""probe_direction_mechanism.py — 判别 A-CSPC 外推方向的真实机制

待判别的两个竞争假说:
  H1 目标对齐(target-alignment): delta 把源原型推向**目标域** → 只有朝目标的方向才有效。
  H2 规范校正(canonicalization): delta 把原型推向 CLIP 的**规范/自然参考系**
     (ExDet 的 DeltaSpace 设计本就是 "多域文本原型 → natural 文本原型"),
     与目标域无关 → 朝 natural(photo) 的方向在任何源/目标组合下都应有效。

判别设计(关键):
  - **同域任务** (DIOR→DIOR, COCO→COCO): 无域差可对齐。
      若 H1 成立 → +δ(朝 photo) 应无效甚至有害(把原型推离目标域)。
      若 H2 成立 → +δ 仍应有效(朝规范参考系), 且 −δ 有害。
  - **反向任务** (COCO→DIOR, 目标是 overhead): +δ 朝 photo = 背离目标域。
      H1 → 应有害; H2 → 应有效。(上一轮探针已测得 +0.0685, 支持 H2)
  - **随机方向对照** (与 δ 同模长, 多种子): 排除"任何扰动都有益"的平庸解释。
  - **β=0 对照**: 不外推(论文消融中此项 gain≈0)。

增量(相对论文): 若 H2 成立, 则"A-CSPC 朝目标域外推"的解释是**在 DIOR→COCO 上
恰好与目标域重合而得到的误读**, 真实不变量是"朝规范方向" → 单一固定方向即可双向通用,
直接解决论文局限性②(方向选择性/反向失败)。

用法:
  HF_HUB_OFFLINE=1 "C:/Users/张三/GroundingDINO/venv/Scripts/python.exe" probe_direction_mechanism.py
"""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ['AC_SPC_COCO_ANN'] = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'
os.environ['AC_SPC_COCO_IMG'] = r'C:\Users\张三\modeling\coco_data\val2017'
os.environ['AC_SPC_DIOR_IMG'] = r'C:\Users\张三\modeling\dior_data\eval_v2_images'
os.environ['AC_SPC_DIOR_PARQUET'] = r'C:\Users\张三\modeling\dior_data\dior_combined.parquet'

import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from acspc.config import Config
from acspc import data as D
from acspc import models as M
from acspc import metrics as Met
from acspc import prototype as Proto

BETA = 2.0
LAM = 0.5
NEG_TOPK = 3
PRE_FILTER = 0.03
N_PROTO = 20      # 原型用前 N_PROTO 张
N_EVAL = 30       # 评测用紧接着的 N_EVAL 张(与原型split不重叠, 防泄漏)
N_RAND_SEED = 5   # 随机方向对照种子数

# 源域原型池(负原型 top-k 需要 >=4 个类)
COCO_POOL = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
DIOR_POOL = ['airplane', 'ship', 'vehicle', 'harbor', 'storage tank', 'bridge']

# 任务定义: (源域, 目标域, [(源类, 评测类), ...])
TASKS = {
    'same_dior2dior': ('dior', 'dior', [('airplane', 'airplane'), ('ship', 'ship'), ('vehicle', 'vehicle')]),
    'same_coco2coco': ('coco', 'coco', [('airplane', 'airplane'), ('boat', 'boat'), ('car', 'car')]),
    'rev_coco2dior': ('coco', 'dior', [('airplane', 'airplane'), ('boat', 'ship'), ('car', 'vehicle')]),
    'fwd_dior2coco': ('dior', 'coco', [('airplane', 'airplane'), ('ship', 'boat'), ('vehicle', 'car')]),
}

CACHE_DIR = Path(r'C:\Users\张三\Desktop\实验数据\probe_dirmech_cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(*a, flush=True)


# ───────── CLIP (transformers 版; venv 内 clip 为空壳, 缓存 JIT 是 CUDA 绑定) ─────────
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
    model, proc = get_clip()
    feats = []
    for i in range(0, len(crops), bs):
        inputs = proc(images=crops[i:i + bs], return_tensors='pt', padding=True).to(DEVICE)
        with torch.no_grad():
            f = model.get_image_features(**inputs).float()
        feats.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(feats).cpu()


def text_emb(t):
    model, proc = get_clip()
    inputs = proc(text=[t], return_tensors='pt', padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        f = model.get_text_features(**inputs).float()
    return (f / f.norm(dim=-1, keepdim=True)).cpu()[0]


def delta_pos(cls):
    """+δ = E('a photo of c') - E('an overhead image of c')  朝 natural/photo"""
    return text_emb(f'a photo of {cls}') - text_emb(f'an overhead image of {cls}')


# ───────── 数据 ─────────
def load_split(cfg, domain, cls, offset, n):
    """取 [offset, offset+n) 段, 防原型/评测泄漏"""
    if domain == 'coco':
        ims, gts = D.load_coco(cfg, cls, n=offset + n)
    else:
        ims, gts = D.load_dior(cfg, cls, n=offset + n)
    ims = ims[offset:offset + n]
    gts = gts[offset:offset + n]
    keep = [(i, g) for i, g in zip(ims, gts) if len(g) > 0]
    if not keep:
        return [], []
    return [k[0] for k in keep], [k[1] for k in keep]


def detect_encode(cfg, images, gts, prompt, key):
    cp = CACHE_DIR / f'{key}.pkl'
    if cp.exists():
        log(f'    [cache] {cp.name}')
        return pickle.load(open(cp, 'rb'))
    log(f'    GDINO on {len(images)} imgs (prompt="{prompt}") ...')
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


def ap_of(per_img, F, W):
    for p in per_img:
        n = len(p['gd'])
        if n:
            vis = np.maximum(0.0, F[p['crop_idx']] @ W)
            p['_s'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p['_s'] = np.zeros(0)
    preds = [(p['box'], p['_s']) for p in per_img]
    gts = [p['gt'] for p in per_img]
    return Met.compute_ap50_img(preds, gts)


def W_of(P_src, d, beta, P_neg, lam=LAM):
    if beta == 0.0:
        P_hat = P_src
    else:
        P_hat = Proto.extrapolate_positive(P_src, d, beta)
    return (P_hat.cpu().numpy() - lam * P_neg.cpu().numpy())


# ───────── 主流程 ─────────
def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--tasks', nargs='+', default=list(TASKS.keys()), choices=list(TASKS.keys()))
    ap_.add_argument('--n-eval', type=int, default=N_EVAL)
    args = ap_.parse_args()

    cfg = Config()
    cfg.model.device = 'cpu'
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'   # A-CSPC 字段错位 bug 的绕法

    log('=' * 78)
    log('方向机制判别: 目标对齐(H1) vs 规范校正(H2)')
    log(f'  BETA={BETA} LAM={LAM}  N_PROTO={N_PROTO} N_EVAL={args.n_eval}  随机方向种子={N_RAND_SEED}')
    log('=' * 78)

    # 预建两类源原型池
    proto_bank = {}
    for domain, pool in (('coco', COCO_POOL), ('dior', DIOR_POOL)):
        log(f'\n[protos] 构建 {domain.upper()} 源原型池 ({len(pool)} 类, 前 {N_PROTO} 张) ...')
        bank = {}
        for cls in pool:
            ims, gts = load_split(cfg, domain, cls, 0, N_PROTO)
            crops = D.gt_crops(ims, gts)
            if not crops:
                log(f'  !! {cls}: 无 crops, 跳过')
                continue
            F = clip_encode_imgs(crops)
            p = F.mean(dim=0)
            bank[cls] = p / p.norm()
        proto_bank[domain] = bank
        log(f'  构建 {len(bank)} 个')

    results = {}
    for tname in args.tasks:
        src_dom, tgt_dom, pairs = TASKS[tname]
        log(f'\n{"=" * 78}\nTASK {tname}  ({src_dom} → {tgt_dom})')
        bank = proto_bank[src_dom]
        tres = {}
        for src_cls, eval_cls in pairs:
            if src_cls not in bank:
                log(f'  !! {src_cls} 原型缺失, 跳过')
                continue
            log(f'\n  [{eval_cls}] ← {src_dom}:{src_cls}')
            # eval split 与 proto split 不重叠(offset=N_PROTO)
            images, gts = load_split(cfg, tgt_dom, eval_cls, N_PROTO, args.n_eval)
            if len(images) < 8:
                log(f'    !! 图像不足 ({len(images)}), 跳过')
                continue
            prompt = (f'a photo of {eval_cls}' if tgt_dom == 'coco'
                      else f'an overhead image of {eval_cls}')
            key = f'{tname}_{eval_cls}_off{N_PROTO}_n{len(images)}'
            blob = detect_encode(cfg, images, gts, prompt, key)
            per_img, F = blob['per_img'], blob['F']
            log(f'    N={len(images)} boxes={sum(len(p["gd"]) for p in per_img)}')

            P_src = bank[src_cls]
            P_neg = Proto.get_negative_prototype(bank, src_cls, NEG_TOPK)
            d_pos = delta_pos(src_cls)                 # 朝 natural/photo
            d_neg = -d_pos                             # 朝 overhead

            base = None
            # baseline = 原始检测分
            preds = [(p['box'], p['gd']) for p in per_img]
            base50 = Met.compute_ap50_img(preds, [p['gt'] for p in per_img])

            g_pos = ap_of(per_img, F, W_of(P_src, d_pos, BETA, P_neg)) - base50
            g_neg = ap_of(per_img, F, W_of(P_src, d_neg, BETA, P_neg)) - base50
            g_b0 = ap_of(per_img, F, W_of(P_src, d_pos, 0.0, P_neg)) - base50

            # 随机方向对照(同模长)
            dn = float(d_pos.norm())
            rnds = []
            for sd in range(N_RAND_SEED):
                rng = np.random.RandomState(1000 + sd)
                u = torch.from_numpy(rng.randn(512).astype(np.float32))
                u = u / u.norm() * dn
                rnds.append(ap_of(per_img, F, W_of(P_src, u, BETA, P_neg)) - base50)
            g_rnd = float(np.mean(rnds))

            tres[eval_cls] = {
                'baseline_ap50': round(base50, 4),
                'gain_plus_delta': round(g_pos, 4),
                'gain_minus_delta': round(g_neg, 4),
                'gain_beta0': round(g_b0, 4),
                'gain_random_dir': round(g_rnd, 4),
                'random_dir_spread': round(float(np.max(rnds) - np.min(rnds)), 4),
                'n_images': len(images),
            }
            log(f'    base={base50:.4f}  +δ={g_pos:+.4f}  −δ={g_neg:+.4f}  '
                f'β=0={g_b0:+.4f}  rand={g_rnd:+.4f}')

        if tres:
            results[tname] = tres
            mp = float(np.mean([v['gain_plus_delta'] for v in tres.values()]))
            mn = float(np.mean([v['gain_minus_delta'] for v in tres.values()]))
            mb = float(np.mean([v['gain_beta0'] for v in tres.values()]))
            mr = float(np.mean([v['gain_random_dir'] for v in tres.values()]))
            log(f'  MEAN  +δ={mp:+.4f}  −δ={mn:+.4f}  β=0={mb:+.4f}  rand={mr:+.4f}')

    # 判别结论
    log('\n' + '=' * 78)
    log('判别判据')
    log('  H1 目标对齐 预测: 同域任务 +δ 应≈0或负(无域差可对齐); 反向 +δ 应为负。')
    log('  H2 规范校正 预测: 同域任务 +δ 仍为正; 反向 +δ 为正; −δ 与随机方向明显更差。')
    log('-' * 78)
    for tname, tres in results.items():
        mp = float(np.mean([v['gain_plus_delta'] for v in tres.values()]))
        mn = float(np.mean([v['gain_minus_delta'] for v in tres.values()]))
        mr = float(np.mean([v['gain_random_dir'] for v in tres.values()]))
        mb = float(np.mean([v['gain_beta0'] for v in tres.values()]))
        log(f'  {tname:<16} +δ={mp:+.4f}  −δ={mn:+.4f}  β=0={mb:+.4f}  rand={mr:+.4f}')
    log('=' * 78)

    out = {'config': {'beta': BETA, 'lam': LAM, 'n_proto': N_PROTO, 'n_eval': args.n_eval,
                      'n_rand_seed': N_RAND_SEED, 'detector': 'gdino-tiny',
                      'clip': CLIP_ID, 'tasks': {k: [v[0], v[1]] for k, v in TASKS.items()}},
           'results': results}
    op = Path(r'C:\Users\张三\Desktop\实验数据') / 'probe_direction_mechanism.json'
    json.dump(out, open(op, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    log(f'结果 → {op}')


if __name__ == '__main__':
    main()
