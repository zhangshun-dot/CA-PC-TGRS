# -*- coding: utf-8 -*-
"""autoproc_tpt_free_probe.py — B' 标签无关自监督探针（无 GDINO 伪标签）

与 autoproc_tpt_probe.py 的无标签伪标签版对比: 前者用 "GDINO 逐类高置信 crop" 作伪标签
→ 弱类(car/boat)无伪标签 → 失败。本脚本改用**纯无标签熵最小化(自标注自训练)**:
  - 把所有目标域 crop 的图像特征(无标签)聚到 6 个文本原型, 最小化分配熵
  - 自标注来自 "图像-文本相似度" 的 argmax, 不依赖 GDINO 逐类检测强度
  - 信号密集(用全部 crop), 绕开弱类无伪标签的死穴

协议同前: 调过的文本原型直接塞 evaluate() 比 AP50。
"""
import sys, os, json
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import torch.nn.functional as F
import clip
import unified_eval as U
import tge_probe as T
from usc_experiment_v12 import evaluate, build_neg_proto, load_target_coco, evaluate_baseline

DEVICE = U.DEVICE
COCO_CLASSES = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
PREFIX_WORDS = ['a', 'photo', 'of', 'a']
N_CTX = 4
LR = 5e-3
STEPS = 300
N_VAL = 80
SMOKE = False
ENT_REG = 0.01        # ctx 偏离初始化 的 L2 正则(防崩塌/漂移)
if SMOKE:
    COCO_CLASSES = ['car']
    STEPS = 3
    N_VAL = 20

SOS, EOT = 49406, 49407
_ORIG_MASKS = None


def _install_masks(model, L):
    global _ORIG_MASKS
    if _ORIG_MASKS is None:
        _ORIG_MASKS = [b.attn_mask for b in model.transformer.resblocks]
    m = torch.empty(L, L, device=DEVICE).fill_(float('-inf'))
    m.triu_(1)
    for b in model.transformer.resblocks:
        b.attn_mask = m


def _restore_masks(model):
    for b, m0 in zip(model.transformer.resblocks, _ORIG_MASKS):
        b.attn_mask = m0


def make_ids(model, class_name):
    pref_ids = []
    for w in PREFIX_WORDS:
        t = clip.tokenize([w])[0]
        e = (t == EOT).nonzero()[0][0].item()
        pref_ids.append(int(t[1:e][0]))
    ct = clip.tokenize([class_name])[0]
    ec = (ct == EOT).nonzero()[0][0].item()
    cls_ids = [int(x) for x in ct[1:ec]]
    return pref_ids, cls_ids


def encode_text_ctx(model, ctx, pref_ids, cls_ids):
    sos = model.token_embedding.weight[SOS].unsqueeze(0).unsqueeze(0)
    eot = model.token_embedding.weight[EOT].unsqueeze(0).unsqueeze(0)
    pref = model.token_embedding(torch.tensor(pref_ids, device=DEVICE).unsqueeze(0))
    cls = model.token_embedding(torch.tensor(cls_ids, device=DEVICE).unsqueeze(0))
    x = torch.cat([sos, pref, ctx, cls, eot], dim=1)
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


def load_pool_free():
    """无标签池: 6 类目标域 crop 的 CLIP 图像特征(全量, 不筛阈值), 无标签。"""
    feats = []
    for c in COCO_CLASSES:
        td = load_target_coco(c, n=N_VAL)
        if td is None:
            continue
        Ft = td['features']
        if len(Ft) == 0:
            continue
        feats.append(Ft)
        print(f'  pool[{c}] n_crops={len(Ft)}')
    if not feats:
        return None
    return np.vstack(feats)


def main():
    print('=' * 70)
    print('B\' 标签无关自监督探针 (熵最小化 / 无 GDINO 伪标签)')
    print(f'device={DEVICE} classes={COCO_CLASSES} N_CTX={N_CTX} steps={STEPS}')
    print('=' * 70)
    model, _ = U.get_clip()
    model.float(); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logit_scale = model.logit_scale.exp().detach()
    ids = {c: make_ids(model, c) for c in COCO_CLASSES}

    print('\n[1] 加载无标签目标域 crop 池 ...')
    pool = load_pool_free()
    if pool is None:
        print('  ✗ 无数据'); return
    pool_t = torch.tensor(pool, device=DEVICE)
    print(f'  pool total n={len(pool_t)}')

    print('\n[2] 熵最小化训练 ctx (无标签)...')
    ctx = torch.nn.Parameter(torch.randn(1, N_CTX, 512, device=DEVICE) * 0.02)
    ctx_init = ctx.detach().clone()
    opt = torch.optim.Adam([ctx], lr=LR)
    for step in range(STEPS):
        opt.zero_grad()
        protos = torch.stack([encode_text_ctx(model, ctx, ids[c][0], ids[c][1]).squeeze(0)
                              for c in COCO_CLASSES])
        protos = protos / protos.norm(dim=-1, keepdim=True)
        logits = (pool_t @ protos.T) * logit_scale           # [N,6]
        p = F.softmax(logits, dim=1)
        ent = -(p * (p + 1e-8).log()).sum(1).mean()          # 熵最小化
        reg = ((ctx - ctx_init) ** 2).mean()
        loss = ent + ENT_REG * reg
        loss.backward(); opt.step()
        if step % 50 == 0 or step == STEPS - 1:
            assign = logits.argmax(1)
            conf = p.max(1).values.mean().item()
            print(f'  step {step:3d}  loss={loss.item():.4f}  '
                  f'entr={ent.item():.4f}  mean_conf={conf:.3f}')

    protos_tuned = {c: proto_of(model, ctx, ids[c][0], ids[c][1]) for c in COCO_CLASSES}
    protos_frozen = {c: T.text_avg(['a photo of a {}'], c).cpu().numpy() for c in COCO_CLASSES}

    print('\n[3] 评测 (evaluate 协议)...')
    rows = {}
    for c in COCO_CLASSES:
        td = load_target_coco(c, n=N_VAL)
        if td is None:
            print(f'  {c}: 跳过'); continue
        ap_base = evaluate_baseline(td)
        neg_f = build_neg_proto(protos_frozen, c)
        neg_t = build_neg_proto(protos_tuned, c)
        ap_frozen = evaluate(td, protos_frozen[c], neg_f)
        ap_tuned = evaluate(td, protos_tuned[c], neg_t)
        rows[c] = {'gdino_only': float(ap_base), 'frozen_text': float(ap_frozen),
                   'tuned_text': float(ap_tuned)}
        print(f'  {c:10s} gdino={rows[c]["gdino_only"]:.4f}  frozen={rows[c]["frozen_text"]:.4f}  '
              f'tuned={rows[c]["tuned_text"]:.4f}  Δ={rows[c]["tuned_text"]-rows[c]["frozen_text"]:+.4f}')

    out = {
        'config': {'method': 'label_free_entropy_min', 'n_ctx': N_CTX, 'lr': LR,
                   'steps': STEPS, 'ent_reg': ENT_REG, 'n_val': N_VAL, 'device': DEVICE},
        'classes': rows,
    }
    if rows:
        out['mean_frozen'] = float(np.mean([r['frozen_text'] for r in rows.values()]))
        out['mean_tuned'] = float(np.mean([r['tuned_text'] for r in rows.values()]))
        out['mean_gdino_only'] = float(np.mean([r['gdino_only'] for r in rows.values()]))
        out['delta_tuned_minus_frozen'] = out['mean_tuned'] - out['mean_frozen']
    print('\n=== 汇总 ===')
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in out.items() if k in ('mean_gdino_only','mean_frozen','mean_tuned','delta_tuned_minus_frozen')},
                     indent=2, ensure_ascii=False))
    save = r'C:\Users\张三\Desktop\实验数据\autoproc_tpt_free_probe_results.json'
    json.dump(out, open(save, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n结果 → {save}')


if __name__ == '__main__':
    main()
