# -*- coding: utf-8 -*-
"""CoCoOp: Image-conditional prompt learning in CLIP embedding space."""
import json, torch, torch.nn as nn, numpy as np
from PIL import Image
from collections import defaultdict
import pyarrow.parquet as pq
import io

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}")

import clip
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

DIOR_CLASSES = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
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
        if len(class_data[DIOR_CLASSES[cat]]) < 20:
            class_data[DIOR_CLASSES[cat]].append(img_bytes)

np.random.seed(42)
cls_list = sorted(class_data.keys())
np.random.shuffle(cls_list)
test_cls = cls_list[5:10]; train_cls = [c for c in cls_list if c not in test_cls][:15]
print(f"Train: {len(train_cls)}, Test: {len(test_cls)}")

def load_img(b): return preprocess(Image.open(io.BytesIO(b)).convert('RGB'))

train_data = []; test_data = []
for c in train_cls:
    for b in class_data[c]: train_data.append((load_img(b), DIOR_CLASSES.index(c)))
for c in test_cls:
    for b in class_data[c]: test_data.append((load_img(b), DIOR_CLASSES.index(c)))
print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# Pre-compute CLIP text features for all classes
with torch.no_grad():
    text_feat_all = []
    for c in DIOR_CLASSES:
        tok = clip.tokenize([f'a photo of a {c}']).to(device)
        tf = model.encode_text(tok).float()
        text_feat_all.append(tf / tf.norm(dim=-1, keepdim=True))
    text_feat_all = torch.cat(text_feat_all, dim=0)  # [20, 512]

# ── CoCoOp Model: additive image-conditional modulation ──
class CoCoOpEmbed(nn.Module):
    def __init__(self, dim=512, hidden=256):
        super().__init__()
        self.img_to_ctx = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        self.ctx_to_text = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, img_feat):
        # img_feat: [B, 512]
        ctx = self.img_to_ctx(img_feat)  # [B, 512]
        ctx = ctx / ctx.norm(dim=-1, keepdim=True)
        delta = self.ctx_to_text(ctx)  # [B, 512]
        return delta

cocoop = CoCoOpEmbed().to(device)
opt = torch.optim.Adam(cocoop.parameters(), lr=0.001)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

# ── Training ──
print("\nTraining CoCoOp (100 epochs)...")
B = 64
for epoch in range(100):
    cocoop.train()
    idx = torch.randperm(len(train_data))
    total_loss = 0; nb = 0
    for start in range(0, len(train_data), B):
        bi = idx[start:start+B]
        imgs = torch.stack([train_data[i][0] for i in bi]).to(device)
        labels = torch.tensor([train_data[i][1] for i in bi]).to(device)

        with torch.no_grad():
            img_feat = model.encode_image(imgs).float()
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # Image-conditional additive modulation
        delta = cocoop(img_feat)  # [B, 512]
        alpha = 0.3
        # text_mod[b,k] = normalize(text_feat[k] + alpha * delta[b])
        text_mod = text_feat_all.unsqueeze(0) + alpha * delta.unsqueeze(1)  # [B, 20, 512]
        text_mod = text_mod / text_mod.norm(dim=-1, keepdim=True)

        # Logits: img[b] @ text_mod[b,k].T
        logits = torch.bmm(img_feat.unsqueeze(1), text_mod.transpose(1,2)).squeeze(1)  # [B, 20]
        logits = logits * model.logit_scale.exp()

        loss = nn.CrossEntropyLoss()(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); nb += 1

    sched.step()
    if (epoch+1) % 20 == 0:
        print(f"  Epoch {epoch+1}: loss={total_loss/nb:.4f}")

# ── Evaluate ──
print("\nEvaluating...")
cocoop.eval()
correct = 0; total = 0
with torch.no_grad():
    for i in range(0, len(test_data), B):
        batch = test_data[i:i+B]
        imgs = torch.stack([b[0] for b in batch]).to(device)
        labels = [b[1] for b in batch]
        img_feat = model.encode_image(imgs).float()
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        delta = cocoop(img_feat)
        text_mod = text_feat_all.unsqueeze(0) + 0.3 * delta.unsqueeze(1)
        text_mod = text_mod / text_mod.norm(dim=-1, keepdim=True)
        logits = torch.bmm(img_feat.unsqueeze(1), text_mod.transpose(1,2)).squeeze(1)
        logits = logits * model.logit_scale.exp()
        preds = logits.argmax(dim=-1).cpu().tolist()
        for p, l in zip(preds, labels):
            total += 1
            if p == l: correct += 1

acc = correct / total
print(f"CoCoOp accuracy: {acc:.4f} ({correct}/{total})")

with open(r'C:\Users\张三\Desktop\cocoop_results.json', 'w') as f:
    json.dump({'method':'CoCoOp','accuracy':acc,'n_cls':20,'epochs':100}, f, indent=2)
print("Done.")
