# -*- coding: utf-8 -*-
"""autoproc_kshot_probe.py — 少样本跨域投影(Regime C 少样本版) 监督预算扫描

元定调(2026-10-05 19:34): 放开 0 样本前提, 转"少样本/有监督跨域原型投影器"方法。
本探针回答: 最少多少目标标注能稳定破 +1.25% 天花板?

方法: 复用 B' 监督正控制 machinery (CoOp 软提示, CLIP 主体冻结, K=4 ctx,
文本编码器反传), 但调参集从"每类 N_TUNE 全量 GT crop"改为"每类仅 k 个 GT crop"。
k ∈ {1,3,5,10,30}; 每 k 跑 3 随机种子看方差。同 evaluate() 协议比 AP50。

天花板参照: 冻结文本原型均值(本探针 protos_frozen) ≈ 0.4384; +1.25% 天花板 ≈ 0.451。
"破顶" = 均值 tuned AP > 0.451。

输出: 实验数据/autoproc_kshot_results.json + 日志
"""
import sys, json
sys.path.insert(0, r'C:\Users/张三\Desktop\实验代码')
import numpy as np
import torch
import torch.nn.functional as F
import unified_eval as U
import tge_probe as T
from usc_experiment_v12 import (evaluate, build_neg_proto, load_target_coco)
from autoproc_tpt_probe import (make_ids, encode_text_ctx, proto_of,
                                COCO_CLASSES, N_CTX)

DEVICE = U.DEVICE
PREFIX_WORDS = ['a', 'photo', 'of', 'a']
LR = 5e-3
STEPS = 300
N_VAL = 80
N_TUNE_POOL = 30          # GT crop 池来源图数
KS = [1, 3, 5, 10, 30]
SEEDS = [0, 1, 2, 3, 4]


def load_gt_crop_pool():
    """每类收集 GT crop 的 CLIP 图像特征池(归一化), 供 k-shot 采样。"""
    pool = {}
    for c in COCO_CLASSES:
        imgs, gts = U.load_coco(c, n=N_TUNE_POOL)
        crops = []
        for im, g in zip(imgs, gts):
            for box in g:
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(im.width, x2), min(im.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                crops.append(im.crop((x1, y1, x2, y2)))
        if crops:
            pool[c] = T.encode_clip_batch(crops).cpu().numpy()
        print(f'  pool[{c}] n_crops={len(crops)}')
    return pool


def sample_k(pool, k, seed):
    rng = np.random.RandomState(seed)
    feats, labels = [], []
    for ci, c in enumerate(COCO_CLASSES):
        n = len(pool[c])
        idx = rng.choice(n, size=min(k, n), replace=False)
        feats.append(pool[c][idx])
        labels.extend([ci] * len(idx))
    return np.vstack(feats), np.array(labels)


def tune_ctx(model, tune_f, tune_l, ids):
    logit_scale = model.logit_scale.exp().detach()
    ctx = torch.nn.Parameter(torch.randn(1, N_CTX, 512, device=DEVICE) * 0.02)
    opt = torch.optim.Adam([ctx], lr=LR)
    tf = torch.tensor(tune_f, device=DEVICE)
    tl = torch.tensor(tune_l, device=DEVICE)
    for _ in range(STEPS):
        opt.zero_grad()
        protos = torch.stack([encode_text_ctx(model, ctx, ids[c][0], ids[c][1]).squeeze(0)
                              for c in COCO_CLASSES])
        protos = protos / protos.norm(dim=-1, keepdim=True)
        logits = (tf @ protos.T) * logit_scale
        loss = F.cross_entropy(logits, tl)
        loss.backward()
        opt.step()
    return ctx


def main():
    print('=' * 70)
    print('少样本跨域投影探针: k-shot 监督预算扫描 (CoOp 软提示)')
    print(f'KS={KS} SEEDS={SEEDS} N_CTX={N_CTX} steps={STEPS} N_VAL={N_VAL}')
    print('=' * 70)
    model, _ = U.get_clip()
    model.float()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ids = {c: make_ids(model, c) for c in COCO_CLASSES}

    print('\n[1] 加载 GT crop 池 ...')
    pool = load_gt_crop_pool()
    protos_frozen = {c: T.text_avg(['a photo of a {}'], c).cpu().numpy() for c in COCO_CLASSES}
    print('\n[2] 缓存目标域评测集 (N_VAL=%d) ...' % N_VAL)
    tds = {c: load_target_coco(c, n=N_VAL) for c in COCO_CLASSES}

    # 冻结基线(所有 k 共享)
    frozen_aps = []
    for c in COCO_CLASSES:
        neg = build_neg_proto(protos_frozen, c)
        frozen_aps.append(float(evaluate(tds[c], protos_frozen[c], neg)))
    mean_frozen = float(np.mean(frozen_aps))
    print(f'\n  冻结文本原型均值 AP50 = {mean_frozen:.4f}  → +1.25% 天花板 ≈ {mean_frozen+0.0125:.4f}')

    results = {}
    for k in KS:
        per_seed_delta, per_seed_tuned = [], []
        per_class_seed = {c: [] for c in COCO_CLASSES}
        for seed in SEEDS:
            tune_f, tune_l = sample_k(pool, k, seed)
            ctx = tune_ctx(model, tune_f, tune_l, ids)
            protos_tuned = {c: proto_of(model, ctx, ids[c][0], ids[c][1]) for c in COCO_CLASSES}
            aps_t, aps_f = [], []
            for c in COCO_CLASSES:
                neg_f = build_neg_proto(protos_frozen, c)
                neg_t = build_neg_proto(protos_tuned, c)
                af = float(evaluate(tds[c], protos_frozen[c], neg_f))
                at = float(evaluate(tds[c], protos_tuned[c], neg_t))
                aps_f.append(af); aps_t.append(at)
                per_class_seed[c].append(round(at - af, 4))
            per_seed_delta.append(float(np.mean(np.array(aps_t) - np.array(aps_f))))
            per_seed_tuned.append(float(np.mean(aps_t)))
            print(f'  k={k:2d} seed={seed}  均值Δ={per_seed_delta[-1]:+.4f}  '
                  f'tuned均值={per_seed_tuned[-1]:.4f}')
        results[str(k)] = {
            'mean_delta': round(float(np.mean(per_seed_delta)), 4),
            'std_delta': round(float(np.std(per_seed_delta)), 4),
            'mean_tuned_ap': round(float(np.mean(per_seed_tuned)), 4),
            'std_tuned_ap': round(float(np.std(per_seed_tuned)), 4),
            'per_seed_delta': [round(x, 4) for x in per_seed_delta],
            'per_class_delta': {c: per_class_seed[c] for c in COCO_CLASSES},
            'breaks_ceiling': bool(np.mean(per_seed_tuned) > mean_frozen + 0.0125),
        }

    print('\n' + '=' * 70)
    print('汇总: k-shot 监督预算 vs 破顶能力')
    print(f'  冻结均值={mean_frozen:.4f}  天花板≈{mean_frozen+0.0125:.4f}')
    print(f'  {"k":>3s}  {"均值Δ":>8s}  {"±std":>7s}  {"tuned均值":>9s}  破顶?')
    for k in KS:
        r = results[str(k)]
        print(f'  {k:>3d}  {r["mean_delta"]:>+8.4f}  {r["std_delta"]:>7.4f}  '
              f'{r["mean_tuned_ap"]:>9.4f}   {"是" if r["breaks_ceiling"] else "否"}')

    out = {'config': {'ks': KS, 'seeds': SEEDS, 'n_ctx': N_CTX, 'steps': STEPS,
                      'n_val': N_VAL, 'method': 'CoOp text-prompt tuning, k-shot GT crops',
                      'ceiling': round(mean_frozen + 0.0125, 4)},
           'mean_frozen': mean_frozen, 'per_k': results}
    save = r'C:\Users\张三\Desktop\实验数据\autoproc_kshot_results.json'
    json.dump(out, open(save, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n✅ -> {save}')


if __name__ == '__main__':
    main()
