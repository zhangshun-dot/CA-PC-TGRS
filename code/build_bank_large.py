# -*- coding: utf-8 -*-
"""build_bank_large.py — 用 dinov2-large 重建逐层 exemplar bank

与 layer_granularity_scan.build_layer_exemplar_bank 逻辑一致, 仅换:
  DINOV2_PATH → dinov2-large (1024 维)
  BANK_PATH   → exemplar_bank_k5_dinov2_large_layers.pt
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'

import io, glob
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
DIOR_TEST_DIR = r'C:\Users\张三\modeling\dior_data\full20_images'
PARQUET_PATH = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
DINOV2_PATH = r'C:\Users\张三\.cache\modelscope\hub\models\facebook\dinov2-large'
BANK_PATH = SAVE_DIR / 'exemplar_bank_k5_dinov2_large_layers.pt'

DIOR_ID2NAME = {
    0: 'airplane', 1: 'airport', 2: 'baseball field', 3: 'basketball court',
    4: 'bridge', 5: 'chimney', 6: 'dam', 7: 'expressway service area',
    8: 'expressway toll station', 9: 'golf course', 10: 'ground track field',
    11: 'harbor', 12: 'overpass', 13: 'ship', 14: 'stadium',
    15: 'storage tank', 16: 'tennis court', 17: 'train station',
    18: 'vehicle', 19: 'windmill',
}
DIOR_CLASSES = list(DIOR_ID2NAME.values())


def get_dinov2_with_layers():
    from transformers import AutoModel, AutoConfig
    config = AutoConfig.from_pretrained(DINOV2_PATH)
    config.output_hidden_states = True
    model = AutoModel.from_pretrained(DINOV2_PATH, config=config).eval().to(DEVICE)
    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def get_test_ids():
    ids = set()
    for f in glob.glob(os.path.join(DIOR_TEST_DIR, '*.jpg')):
        parts = Path(f).stem.split('_')
        for part in reversed(parts):
            if part.isdigit():
                ids.add(int(part))
                break
    return ids


def build_layer_exemplar_bank(k=5):
    print('=' * 60)
    print(f'构建逐层 Exemplar 库 (dinov2-large, K={k})')
    print('=' * 60)

    model, transform = get_dinov2_with_layers()
    N_LAYERS = 12
    dim = model.config.hidden_size
    print(f'  hidden_size = {dim}')

    df = pd.read_parquet(PARQUET_PATH)
    test_ids = get_test_ids()

    train_rows = defaultdict(list)
    for _, row in df.iterrows():
        img_id = int(row['image_id'])
        if img_id in test_ids:
            continue
        cats = row['objects']['category']
        for i, cat in enumerate(cats):
            cls_name = DIOR_ID2NAME[int(cat)]
            train_rows[cls_name].append((img_id, row, i))

    print(f'  训练类别: {len(train_rows)}')

    bank = {}
    for cls_name in DIOR_CLASSES:
        rows = train_rows.get(cls_name, [])
        if len(rows) == 0:
            continue

        img_best = {}
        for img_id, row, cat_idx in rows:
            bbox = row['objects']['bbox'][cat_idx]
            area = bbox[2] * bbox[3]
            if img_id not in img_best or area > img_best[img_id][2]:
                img_best[img_id] = (row, cat_idx, area)

        all_layer_feats = defaultdict(list)
        for img_id, (row, cat_idx, _) in img_best.items():
            try:
                img_bytes = row['image']['bytes']
                pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                x, y, w, h = [int(v) for v in row['objects']['bbox'][cat_idx]]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(pil_img.width, x + w), min(pil_img.height, y + h)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                crop = pil_img.crop((x1, y1, x2, y2))
                img_t = transform(crop).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    o = model(img_t)
                for layer_idx in range(N_LAYERS):
                    hs = o.hidden_states[layer_idx + 1]
                    all_layer_feats[layer_idx].append(hs[0, 0, :].cpu())
            except Exception:
                continue

        if len(all_layer_feats[0]) < k:
            print(f'  {cls_name}: only {len(all_layer_feats[0])} crops (skip)')
            continue

        cls_bank = {}
        for layer_idx in range(N_LAYERS):
            feats = torch.stack(all_layer_feats[layer_idx])
            feats = feats / feats.norm(dim=-1, keepdim=True)

            selected = [0]
            for _ in range(1, k):
                dists = [-1.0] * len(feats)
                for j in range(len(feats)):
                    if j in selected:
                        continue
                    dists[j] = min(float(1.0 - (feats[j] @ feats[s]).item()) for s in selected)
                selected.append(max(range(len(dists)), key=lambda j: dists[j]))

            proto = feats[selected].mean(dim=0)
            proto = proto / proto.norm(dim=-1, keepdim=True)
            cls_bank[layer_idx] = {'prototype': proto, 'n_available': len(feats)}

        bank[cls_name] = cls_bank
        print(f'  {cls_name}: {k}/{len(all_layer_feats[0])} exemplars × {N_LAYERS} layers')

    torch.save(bank, BANK_PATH)
    print(f'\n保存至: {BANK_PATH}')
    print(f'类别数: {len(bank)}, 维度: {dim}')
    return bank


if __name__ == '__main__':
    build_layer_exemplar_bank(k=5)
