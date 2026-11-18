# -*- coding: utf-8 -*-
"""Exp 2 (fixed): CoOp 20-class 4-fold CV — evaluate only on test classes."""
import json, torch, torch.nn as nn, numpy as np
from PIL import Image
from collections import defaultdict
import pyarrow.parquet as pq, io, sys, time

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

import clip
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.float().eval()

DIOR = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

pq_file = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
table = pq.read_table(pq_file); rows = table.to_pylist()
class_data = defaultdict(list)
for row in rows:
    img_bytes, objs = row['image']['bytes'], row['objects']
    if not objs or 'category' not in objs: continue
    for cat in objs['category']:
        if len(class_data[DIOR[cat]]) < 20: class_data[DIOR[cat]].append(img_bytes)

def load_img(b): return preprocess(Image.open(io.BytesIO(b)).convert('RGB'))

np.random.seed(42)
cls_list = sorted(class_data.keys())
np.random.shuffle(cls_list)
folds = [cls_list[i*5:(i+1)*5] for i in range(4)]

N_CTX = 8; PREFIX = "a photo of a "
B = 32

with torch.no_grad():
    cls_full_emb = {}
    for c in DIOR:
        tok = clip.tokenize([f'{PREFIX}{c}']).to(device)
        emb = model.token_embedding(tok).float()
        cls_full_emb[c] = emb.detach()
    prefix_tok = clip.tokenize([PREFIX]).to(device)
    prefix_emb = model.token_embedding(prefix_tok).float()
    init_ctx = prefix_emb[0, 1:N_CTX+1, :].clone()

pos_emb = model.positional_embedding

def get_text_feature(c, ctx_tokens):
    full = cls_full_emb[c].clone()
    full[0, 1:N_CTX+1, :] = ctx_tokens
    x = (full + pos_emb).permute(1, 0, 2)
    for block in model.transformer.resblocks:
        x = block(x)
    x = x.permute(1, 0, 2)
    tf = x[0, 0, :]
    return tf / tf.norm(dim=-1, keepdim=True)

# Zero-shot baseline on 5-class subsets
print("Zero-shot per-fold baselines:", flush=True)
zs_baselines = {}
for fold_idx, test_cls in enumerate(folds):
    test_indices = [DIOR.index(c) for c in test_cls]
    test_data = []
    for c in test_cls:
        for b in class_data[c]:
            test_data.append((load_img(b), test_indices.index(DIOR.index(c))))

    with torch.no_grad():
        zs_text = []
        for c in test_cls:
            tok = clip.tokenize([f'{PREFIX}{c}']).to(device)
            tf = model.encode_text(tok).float()
            zs_text.append(tf / tf.norm(dim=-1, keepdim=True))
        zs_text = torch.cat(zs_text, dim=0)

        correct = 0; total = 0
        for i in range(0, len(test_data), B):
            batch = test_data[i:i+B]
            imgs = torch.stack([b[0] for b in batch]).to(device)
            labels = [b[1] for b in batch]
            img_feat = model.encode_image(imgs).float()
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = (img_feat @ zs_text.T) * model.logit_scale.exp()
            preds = logits.argmax(dim=-1).cpu().tolist()
            for p, l in zip(preds, labels):
                total += 1; correct += (p == l)
    zs_baselines[fold_idx] = correct/total
    print(f"  Fold {fold_idx+1} ({', '.join(test_cls[:3])}...): ZS acc = {correct/total:.4f}", flush=True)

# CoOp 4-fold CV
fold_results = []
for fold_idx, test_cls in enumerate(folds):
    train_cls = [c for c in DIOR if c not in test_cls]
    test_indices = [DIOR.index(c) for c in test_cls]
    n_test_cls = len(test_cls)
    print(f"\nFold {fold_idx+1}/4: {n_test_cls} test classes", flush=True)

    train_data = []; test_data = []
    for c in train_cls:
        for b in class_data[c]: train_data.append((load_img(b), DIOR.index(c)))
    for c in test_cls:
        for b in class_data[c]: test_data.append((load_img(b), test_indices.index(DIOR.index(c))))

    ctx = nn.Parameter(init_ctx.clone())
    opt = torch.optim.Adam([ctx], lr=0.002)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

    for epoch in range(100):
        idx = torch.randperm(len(train_data))
        total_loss = 0; nb = 0
        for start in range(0, len(train_data), B):
            bi = idx[start:start+B]
            imgs = torch.stack([train_data[i][0] for i in bi]).to(device)
            labels = torch.tensor([train_data[i][1] for i in bi]).to(device)

            with torch.no_grad():
                img_feat = model.encode_image(imgs).float()
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            batch_cls = labels.unique().tolist()
            text_feats_batch = [get_text_feature(DIOR[k], ctx) for k in batch_cls]
            text_feats_batch = torch.stack(text_feats_batch)

            with torch.no_grad():
                zs_text_all = []
                for c in DIOR:
                    tok = clip.tokenize([f'{PREFIX}{c}']).to(device)
                    tf = model.encode_text(tok).float()
                    zs_text_all.append(tf / tf.norm(dim=-1, keepdim=True))
                zs_text_all = torch.cat(zs_text_all, dim=0)

            batch_set = set(batch_cls)
            other_indices = [k for k in range(20) if k not in batch_set]
            text_full = torch.zeros(20, 512, device=device)
            for i_k, k in enumerate(batch_cls):
                text_full[k] = text_feats_batch[i_k]
            for k in other_indices:
                text_full[k] = zs_text_all[k]

            logits = (img_feat @ text_full.T) * model.logit_scale.exp()
            loss = nn.CrossEntropyLoss()(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1
        sched.step()
        if (epoch+1) % 50 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/nb:.4f}", flush=True)

    # Eval: ONLY on test classes
    with torch.no_grad():
        text_test_only = []
        for c in test_cls:
            tf = get_text_feature(c, ctx)
            text_test_only.append(tf)
        text_test_only = torch.stack(text_test_only)

    # Also compute zero-shot on test classes
    with torch.no_grad():
        zs_test = []
        for c in test_cls:
            tok = clip.tokenize([f'{PREFIX}{c}']).to(device)
            tf = model.encode_text(tok).float()
            zs_test.append(tf / tf.norm(dim=-1, keepdim=True))
        zs_test = torch.cat(zs_test, dim=0)

    # Evaluate CoOp
    correct = 0; total = 0; per_class_correct = defaultdict(int); per_class_total = defaultdict(int)
    all_preds = []; all_labels = []
    for i in range(0, len(test_data), B):
        batch = test_data[i:i+B]
        imgs = torch.stack([b[0] for b in batch]).to(device)
        labels = [b[1] for b in batch]
        img_feat = model.encode_image(imgs).float()
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        logits = (img_feat @ text_test_only.T) * model.logit_scale.exp()
        preds = logits.argmax(dim=-1).cpu().tolist()
        for p, l in zip(preds, labels):
            total += 1; correct += (p == l)
            per_class_correct[l] += (p == l); per_class_total[l] += 1
            all_preds.append(p); all_labels.append(l)

    acc = correct / total if total else 0

    # Macro F1
    f1s = []
    for cls_idx in range(n_test_cls):
        tp = per_class_correct[cls_idx]
        fp = per_class_total[cls_idx] - tp if cls_idx in per_class_total else 0
        # For macro F1, need FN from other classes predicted as this one
        fn = sum(1 for p, l in zip(all_preds, all_labels) if l == cls_idx and p != cls_idx)
        prec = tp/(tp+fp) if (tp+fp) else 0
        rec = tp/(tp+fn) if (tp+fn) else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        f1s.append(f1)
    macro_f1 = np.mean(f1s) if f1s else 0

    fold_results.append({
        'fold': fold_idx+1, 'test_classes': test_cls,
        'accuracy': acc, 'macro_f1': macro_f1,
        'correct': correct, 'total': total,
        'zs_accuracy': zs_baselines[fold_idx],
    })
    print(f"  CoOp: acc={acc:.4f} macro_F1={macro_f1:.4f} | ZS: {zs_baselines[fold_idx]:.4f}", flush=True)

# Summary
accs = [f['accuracy'] for f in fold_results]
f1s = [f['macro_f1'] for f in fold_results]
print(f"\n{'='*50}", flush=True)
print(f"CoOp 4-fold CV (FIXED: test-only eval):", flush=True)
for f in fold_results:
    print(f"  Fold {f['fold']}: acc={f['accuracy']:.4f} F1={f['macro_f1']:.4f} (ZS={f['zs_accuracy']:.4f})", flush=True)
print(f"  Mean acc: {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}", flush=True)
print(f"  Mean F1:  {np.mean(f1s):.4f} ± {np.std(f1s, ddof=1):.4f}", flush=True)

result = {
    'method': 'CoOp_4fold_CV_fixed',
    'folds': fold_results,
    'mean_accuracy': float(np.mean(accs)),
    'std_accuracy': float(np.std(accs, ddof=1)),
    'mean_macro_f1': float(np.mean(f1s)),
    'std_macro_f1': float(np.std(f1s, ddof=1)),
    'n_ctx': N_CTX, 'epochs': 100,
}
with open(r'C:\Users\张三\Desktop\coop_20class_cv_results.json', 'w') as f:
    json.dump(result, f, indent=2)
print("Done.", flush=True)
