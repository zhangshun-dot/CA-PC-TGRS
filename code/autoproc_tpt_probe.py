# -*- coding: utf-8 -*-
"""autoproc_tpt_probe.py — B' 测试时提示微调最小探针（监督正控制）

目的（元 2026-10-05 选定 B' 方向）:
  验证"对 CLIP 文本编码器做软提示反传"能否打破冻结空间 +1.25% 天花板。
  这是机制探针，用 GT 标签做监督（正控制）；机制成立后再上无标签测试时版。

协议（与 usc_experiment_v12.evaluate 完全一致）:
  - 文本原型 proto(512-d) 由 encode_text_ctx(可学习 ctx) 产出
  - 直接塞进 evaluate(target_data, proto, neg_proto) 算 COCO 6 类 AP50
  - 对比三档: GDINO-only(地板) / 冻结文本原型(当前方法) / 调过的文本原型(B')

注意:
  - 只训练 ctx（K=4 个可学习向量），CLIP 主体冻结 → 测试时/少量样本友好
  - SMOKE=True 时极速自检（1 类、3 step），用于先确认代码能跑
"""
import sys, os, json
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import torch.nn.functional as F
import clip
import unified_eval as U
import tge_probe as T
from usc_experiment_v12 import (evaluate, build_neg_proto,
                                load_target_coco, evaluate_baseline)

DEVICE = U.DEVICE
COCO_CLASSES = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
PREFIX_WORDS = ['a', 'photo', 'of', 'a']   # 固定自然语言前缀
N_CTX = 4
LR = 5e-3
STEPS = 300
N_TUNE = 30          # 每类调参用的 GT crop 数（少量样本）
N_VAL = 80           # 评测用目标域图数（探针用，足够看趋势）
SMOKE = False
UNSUP = True          # True=测试时无标签(用GDINO伪标签) ; False=监督正控制(用GT)
PSEUDO_THR = 0.30     # 无标签: GDINO 置信度阈值作伪标签

if SMOKE:
    COCO_CLASSES = ['car']
    STEPS = 3
    N_TUNE = 5
    N_VAL = 20
    UNSUP = True

SOS, EOT = 49406, 49407

# 缓存各 resblock 原始 77×77 因果 mask，自定义前向时临时替换再恢复
_ORIG_MASKS = None


def _install_masks(model, L):
    global _ORIG_MASKS
    if _ORIG_MASKS is None:
        _ORIG_MASKS = [b.attn_mask for b in model.transformer.resblocks]
    mask = torch.empty(L, L, device=DEVICE).fill_(float('-inf'))
    mask.triu_(1)  # 上三角 -inf = 因果
    for b in model.transformer.resblocks:
        b.attn_mask = mask


def _restore_masks(model):
    for b, m0 in zip(model.transformer.resblocks, _ORIG_MASKS):
        b.attn_mask = m0


def make_ids(model, class_name):
    """返回 (prompt_len, pref_ids, cls_ids)。ctx 位置用占位 0，forward 时替换。"""
    pref_ids = []
    for w in PREFIX_WORDS:
        t = clip.tokenize([w])[0]
        eot_p = (t == EOT).nonzero()[0][0].item()
        pref_ids.append(int(t[1:eot_p][0]))
    ct = clip.tokenize([class_name])[0]
    eot_c = (ct == EOT).nonzero()[0][0].item()
    cls_ids = [int(x) for x in ct[1:eot_c]]
    return pref_ids, cls_ids


def encode_text_ctx(model, ctx, pref_ids, cls_ids):
    """CoOp 式前向: [SOS][prefix(hard)][ctx(learnable)][class(hard)][EOT]"""
    sos_emb = model.token_embedding.weight[SOS].unsqueeze(0).unsqueeze(0)      # [1,1,512]
    eot_emb = model.token_embedding.weight[EOT].unsqueeze(0).unsqueeze(0)
    pref_emb = model.token_embedding(torch.tensor(pref_ids, device=DEVICE).unsqueeze(0))
    cls_emb = model.token_embedding(torch.tensor(cls_ids, device=DEVICE).unsqueeze(0))
    x = torch.cat([sos_emb, pref_emb, ctx, cls_emb, eot_emb], dim=1)           # [1,L,512]
    L = x.shape[1]
    x = x + model.positional_embedding[:L].unsqueeze(0).to(x.dtype)
    x = x.permute(1, 0, 2)
    _install_masks(model, L)
    try:
        x = model.transformer(x)
    finally:
        _restore_masks(model)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).type(x.dtype)
    x = x[:, -1, :] @ model.text_projection
    return x


@torch.no_grad()
def proto_of(model, ctx, pref_ids, cls_ids):
    v = encode_text_ctx(model, ctx, pref_ids, cls_ids)
    return (v / v.norm(dim=-1, keepdim=True)).squeeze(0).cpu().numpy()


def load_tune_feats():
    """每类 N_TUNE 张 GT crop 的 CLIP 图像特征 (归一化) + 类别索引。"""
    feats, labels = [], []
    for ci, cls in enumerate(COCO_CLASSES):
        if not UNSUP:
            imgs, gts = U.load_coco(cls, n=N_TUNE)
            crops = []
            for im, g in zip(imgs, gts):
                for box in g:
                    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                    x2, y2 = min(im.width, x2), min(im.height, y2)
                    if x2 <= x1 + 10 or y2 <= y1 + 10:
                        continue
                    crops.append(im.crop((x1, y1, x2, y2)))
            if not crops:
                continue
            f = T.encode_clip_batch(crops).cpu().numpy()
        else:
            # 无标签: 目标域 unlabeled 图, 用 GDINO 高置信 crop 作伪标签
            td = load_target_coco(cls, n=N_TUNE)
            if td is None:
                continue
            Ft = td['features']; per = td['per_img']
            crops_f = []
            for p in per:
                gds = p['gd']; cidx = p['crop_idx']
                if len(gds) == 0:
                    continue
                for k in range(len(gds)):
                    if gds[k] > PSEUDO_THR:
                        crops_f.append(Ft[cidx[k]])
            if not crops_f:
                continue
            f = np.stack([np.asarray(x) for x in crops_f])
        feats.append(f)
        labels.extend([ci] * len(f))
        print(f'  tune[{cls}] n_crops={len(f)}')
    if not feats:
        return None, None
    return np.vstack(feats), np.array(labels)


def main():
    print('=' * 70)
    print('B\' 测试时提示微调探针（监督正控制）')
    print(f'device={DEVICE} classes={COCO_CLASSES} N_CTX={N_CTX} steps={STEPS}')
    print('=' * 70)

    model, _ = U.get_clip()
    model.float()          # 转 fp32，避免软提示 ctx(fp32) 与 transformer(fp16) 冲突
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logit_scale = model.logit_scale.exp().detach()

    # 预计算每类的 prefix/class token id
    ids = {c: make_ids(model, c) for c in COCO_CLASSES}

    # 1) 调参集（GT crop 图像特征）
    print('\n[1] 加载调参集 (GT crops)...')
    tune_f, tune_l = load_tune_feats()
    if tune_f is None:
        print('  ✗ 无调参数据，退出')
        return
    tune_f_t = torch.tensor(tune_f, device=DEVICE)
    tune_l_t = torch.tensor(tune_l, device=DEVICE)
    n_cls = len(COCO_CLASSES)

    # 2) 训练可学习 ctx（共享跨类，CoOp global context）
    print('\n[2] 反传训练 ctx ...')
    ctx = torch.nn.Parameter(torch.randn(1, N_CTX, 512, device=DEVICE) * 0.02)
    opt = torch.optim.Adam([ctx], lr=LR)
    for step in range(STEPS):
        opt.zero_grad()
        # 当前 ctx 下各文本原型
        protos = torch.stack([
            encode_text_ctx(model, ctx, ids[c][0], ids[c][1]).squeeze(0)
            for c in COCO_CLASSES
        ])  # [n_cls, 512]
        protos = protos / protos.norm(dim=-1, keepdim=True)
        # 图像特征 × 文本原型 → 分类 logits
        logits = (tune_f_t @ protos.T) * logit_scale  # [N, n_cls]
        loss = F.cross_entropy(logits, tune_l_t)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == STEPS - 1:
            acc = (logits.argmax(1) == tune_l_t).float().mean().item()
            print(f'  step {step:3d}  loss={loss.item():.4f}  train_acc={acc:.3f}')

    # 3) 产出调过的文本原型
    protos_tuned = {c: proto_of(model, ctx, ids[c][0], ids[c][1]) for c in COCO_CLASSES}
    # 冻结文本原型（当前方法的文本打分）
    protos_frozen = {c: T.text_avg(['a photo of a {}'], c).cpu().numpy() for c in COCO_CLASSES}

    # 4) 评测：同一 evaluate() 协议，只改文本原型
    print('\n[3] 评测 (evaluate 协议, 同候选框集)...')
    rows = {}
    for c in COCO_CLASSES:
        td = load_target_coco(c, n=N_VAL) if not SMOKE else _smoke_target(c)
        if td is None:
            print(f'  {c}: 目标域加载失败，跳过'); continue
        ap_base = evaluate_baseline(td)
        neg_f = build_neg_proto(protos_frozen, c)
        neg_t = build_neg_proto(protos_tuned, c)
        ap_frozen = evaluate(td, protos_frozen[c], neg_f)
        ap_tuned = evaluate(td, protos_tuned[c], neg_t)
        rows[c] = {'gdino_only': float(ap_base) if ap_base is not None else None,
                   'frozen_text': float(ap_frozen), 'tuned_text': float(ap_tuned)}
        print(f'  {c:10s} gdino={rows[c]["gdino_only"]:.4f}  '
              f'frozen={rows[c]["frozen_text"]:.4f}  tuned={rows[c]["tuned_text"]:.4f}  '
              f'Δtuned-frozen={rows[c]["tuned_text"]-rows[c]["frozen_text"]:+.4f}')

    out = {
        'config': {'n_ctx': N_CTX, 'lr': LR, 'steps': STEPS, 'n_tune': N_TUNE,
                   'supervised_probe': not UNSUP, 'unsupervised_pseudo': UNSUP,
                   'pseudo_thr': PSEUDO_THR if UNSUP else None, 'device': DEVICE},
        'classes': rows,
        'mean_frozen': float(np.mean([r['frozen_text'] for r in rows.values()])) if rows else None,
        'mean_tuned': float(np.mean([r['tuned_text'] for r in rows.values()])) if rows else None,
    }
    if rows:
        out['mean_gdino_only'] = float(np.mean([r['gdino_only'] for r in rows.values()
                                                if r['gdino_only'] is not None]))
        out['delta_tuned_minus_frozen'] = out['mean_tuned'] - out['mean_frozen']
    print('\n=== 汇总 ===')
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in out.items() if k in ('mean_gdino_only','mean_frozen','mean_tuned','delta_tuned_minus_frozen')},
                     indent=2, ensure_ascii=False))
    save = r'C:\Users\张三\Desktop\实验数据\autoproc_tpt_probe_results.json'
    json.dump(out, open(save, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n结果 → {save}')


def _smoke_target(cls):
    """SMOKE 下用 GDINO 缓存或回退到 GT crop 检索 AP（不依赖完整管道）。"""
    try:
        return U.load_target_coco(cls, n=N_VAL)
    except Exception:
        return None


if __name__ == '__main__':
    main()
