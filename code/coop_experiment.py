# -*- coding: utf-8 -*-
"""CoOp: Context Optimization for DIOR (proper CLIP text transformer)."""
import json, torch, torch.nn as nn, numpy as np
from PIL import Image
from collections import defaultdict
import pyarrow.parquet as pq, io, gc

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}")

import clip
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.float().eval()

DIOR = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

# Data
pq_file = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
table = pq.read_table(pq_file); rows = table.to_pylist()
class_data = defaultdict(list)
for row in rows:
    img_bytes, objs = row['image']['bytes'], row['objects']
    if not objs or 'category' not in objs: continue
    for cat in objs['category']:
        if len(class_data[DIOR[cat]]) < 20: class_data[DIOR[cat]].append(img_bytes)

np.random.seed(42); cls_list = sorted(class_data.keys())
np.random.shuffle(cls_list)
test_cls = cls_list[5:10]; train_cls = [c for c in cls_list if c not in test_cls][:15]
print(f"Train: {len(train_cls)}, Test: {len(test_cls)}")

def load_img(b): return preprocess(Image.open(io.BytesIO(b)).convert('RGB'))
train_data = []; test_data = []
for c in train_cls:
    for b in class_data[c]: train_data.append((load_img(b), DIOR.index(c)))
for c in test_cls:
    for b in class_data[c]: test_data.append((load_img(b), DIOR.index(c)))
print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# Pre-compute class name embeddings (SOS + class_name + EOS without prefix)
PREFIX = "a photo of a "
with torch.no_grad():
    cls_full_emb = {}
    for c in DIOR:
        tok = clip.tokenize([f'{PREFIX}{c}']).to(device)
        emb = model.token_embedding(tok).float()
        cls_full_emb[c] = emb.detach()  # Detach to avoid keeping in graph

N_CTX = 8

# Init context from real prefix token embeddings
with torch.no_grad():
    prefix_tok = clip.tokenize([PREFIX]).to(device)
    prefix_emb = model.token_embedding(prefix_tok).float()
    init_ctx = prefix_emb[0, 1:N_CTX+1, :].clone()

# Zero-shot baseline
print("Zero-shot baseline...")
with torch.no_grad():
    zs_text = []
    for c in DIOR:
        tok = clip.tokenize([f'{PREFIX}{c}']).to(device)
        tf = model.encode_text(tok).float()
        zs_text.append(tf / tf.norm(dim=-1, keepdim=True))
    zs_text = torch.cat(zs_text, dim=0)

correct = 0; total = 0
with torch.no_grad():
    for img, label in test_data:
        img = img.unsqueeze(0).to(device)
        img_feat = model.encode_image(img).float()
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        logits = (img_feat @ zs_text.T) * model.logit_scale.exp()
        if logits.argmax().item() == label: correct += 1; total += 1
print(f"  Zero-shot: {correct}/{total} = {correct/total:.4f}")

# ── CoOp Training ──
# Strategy: only compute text features for classes in the current batch (less VRAM)
# Use the zero-shot features for negative classes
ctx = nn.Parameter(init_ctx.clone())
opt = torch.optim.Adam([ctx], lr=0.002)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
pos_emb = model.positional_embedding

def get_text_feature(c, ctx_tokens):
    """Get text feature for one class WITH gradient flow through ctx."""
    full = cls_full_emb[c].clone()
    full[0, 1:N_CTX+1, :] = ctx_tokens
    x = (full + pos_emb).permute(1, 0, 2)
    for block in model.transformer.resblocks:
        x = block(x)
    x = x.permute(1, 0, 2)
    tf = x[0, 0, :]
    return tf / tf.norm(dim=-1, keepdim=True)

print("\nTraining CoOp (100 epochs, gradient through transformer)...")
B = 32
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

        # Get unique classes in this batch + enough others for contrastive
        batch_cls = labels.unique().tolist()
        # Compute text features for these classes (with gradient)
        text_feats_batch = []
        for k in batch_cls:
            tf = get_text_feature(DIOR[k], ctx)
            text_feats_batch.append(tf)
        text_feats_batch = torch.stack(text_feats_batch)  # [U, 512]

        # Use zero-shot features for all other classes (frozen, no gradient)
        all_indices = list(range(20))
        batch_set = set(batch_cls)
        other_indices = [k for k in all_indices if k not in batch_set]

        # Build full text feature matrix: [20, 512]
        text_full = torch.zeros(20, 512, device=device)
        for i, k in enumerate(batch_cls):
            text_full[k] = text_feats_batch[i]
        for k in other_indices:
            text_full[k] = zs_text[k]  # frozen zero-shot features

        logits = (img_feat @ text_full.T) * model.logit_scale.exp()
        loss = nn.CrossEntropyLoss()(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); nb += 1
    sched.step()
    if (epoch+1) % 20 == 0:
        print(f"  Epoch {epoch+1}: loss={total_loss/nb:.4f} lr={sched.get_last_lr()[0]:.6f}")

# ── Eval ──
print("\nEvaluating...")
with torch.no_grad():
    text_full_eval = []
    for c in DIOR:
        tf = get_text_feature(c, ctx)
        text_full_eval.append(tf)
    text_full_eval = torch.stack(text_full_eval)

correct = 0; total = 0
with torch.no_grad():
    for i in range(0, len(test_data), B):
        batch = test_data[i:i+B]
        imgs = torch.stack([b[0] for b in batch]).to(device)
        labels = [b[1] for b in batch]
        img_feat = model.encode_image(imgs).float()
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        logits = (img_feat @ text_full_eval.T) * model.logit_scale.exp()
        preds = logits.argmax(dim=-1).cpu().tolist()
        for p, l in zip(preds, labels):
            total += 1; correct += (p == l)

acc = correct / total
print(f"CoOp: {acc:.4f} ({correct}/{total}) | Zero-shot: 0.43")
with open(r'C:\Users\张三\Desktop\coop_results_new.json', 'w') as f:
    json.dump({'method':'CoOp','accuracy':acc,'zero_shot':0.43,'n_ctx':N_CTX,'epochs':100}, f, indent=2)
print("Done.")
