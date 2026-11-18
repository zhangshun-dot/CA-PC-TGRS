# -*- coding: utf-8 -*-
"""Token Projection — 快速对照验证

检查: Token Projection 1.000 是全检测还是真的改进了检测质量?
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import torch, glob, numpy as np
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
gd_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
DIOR_DIR = r'C:\Users\张三\modeling\dior_data\full20_images'

# ── 加载模型和数据 ──
print("加载 GDINO...")
processor = AutoProcessor.from_pretrained(gd_cache)
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    gd_cache, dtype=torch.float32
).eval().to(DEVICE)
gd_tokenizer = AutoTokenizer.from_pretrained(gd_cache)

# ── 训练 Token Projection (复用主脚本逻辑) ──
print("训练 Token Projection...")
import clip
clip_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
clip_model, _ = clip.load(clip_path, device=DEVICE)
clip_model.eval()

from token_projection import TokenProjection, compute_axis_directions, train_projection
from ofat_knowledge_base import OFATKnowledgeBase
from token_projection_gdino import (
    ProjectedClip2BertMapper, precompute_class_projections, TextBackboneWrapper
)
from transformers import BertModel

kb = OFATKnowledgeBase()
axis_dirs_clip = compute_axis_directions(clip_model, clip.tokenize, device=DEVICE)
proj_wd = TokenProjection(d_model=512, rank=4, domain='remote_sensing')
train_projection(proj_wd, kb, axis_directions=axis_dirs_clip,
                 epochs=200, lr=0.005, device=DEVICE, verbose=False)
proj_wd.eval()

bert_model = BertModel.from_pretrained('bert-base-uncased', local_files_only=True).eval().to(DEVICE)

all_classes = ['airplane','airport','baseball field','basketball court','bridge',
    'chimney','dam','expressway service area','expressway toll station',
    'golf course','ground track field','harbor','overpass','ship','stadium',
    'storage tank','tennis court','train station','vehicle','windmill',
    'bird', 'car', 'dog', 'cat', 'building', 'tree', 'road', 'person',
    'truck', 'bicycle', 'motorcycle', 'bus', 'train', 'chair', 'table', 'house']

mapper = ProjectedClip2BertMapper()
mapper.train(proj_wd, clip_model, bert_model, all_classes)

class_projections, class_token_ids = precompute_class_projections(
    proj_wd, mapper, clip_model, all_classes, gd_tokenizer
)

wrapper = TextBackboneWrapper(
    model.model.text_backbone, gd_tokenizer, class_projections, class_token_ids
)

# ── 对照实验 ──
print("\n" + "=" * 60)
print("对照实验")
print("=" * 60)

def detect_with_boxes(model, processor, images, prompt):
    """返回检测框数量和平均置信度。"""
    n_boxes = []
    avg_scores = []
    for img in images:
        try:
            inputs = processor(images=img, text=prompt, return_tensors='pt')
            inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            with torch.no_grad():
                o = model(**inputs)
            ts = torch.tensor([img.size[::-1]]).to(DEVICE)
            r = processor.post_process_grounded_object_detection(
                o, threshold=0.25, target_sizes=ts
            )[0]
            n_boxes.append(len(r['boxes']))
            if len(r['scores']) > 0:
                avg_scores.append(r['scores'].mean().item())
        except Exception as e:
            n_boxes.append(0)
    return np.mean(n_boxes), np.mean(avg_scores) if avg_scores else 0.0

def load_images(cls, n=5):
    paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{cls.replace(' ','_')}_*.jpg")))[:n]
    return [Image.open(p).convert('RGB') for p in paths]

test_classes = ['airplane', 'ship', 'stadium', 'bridge', 'harbor']
n_imgs = 5

print(f"\n{'Prompt':<50s} {'DR':>6s} {'AvgBox':>8s} {'AvgScore':>8s}")
print("-" * 75)

# 1. 无 wrapper, 正常 prompt
for cls in test_classes:
    imgs = load_images(cls, n_imgs)
    nbox, score = detect_with_boxes(model, processor, imgs, f'an overhead image of {cls}')
    dr = (nbox > 0).mean() if hasattr(nbox, '__iter__') else (nbox > 0)
    print(f"  an overhead image of {cls:<20s}      {dr:6.3f} {nbox:8.1f} {score:8.3f}")

# 2. 无 wrapper, "detect everything" prompt
print()
for cls in test_classes:
    imgs = load_images(cls, n_imgs)
    nbox, score = detect_with_boxes(model, processor, imgs, 'an overhead image of object')
    print(f"  an overhead image of object (on {cls} imgs) {0:6.3f} {nbox:8.1f} {score:8.3f}")

# 3. 装 wrapper, 正常 prompt
print("\n--- 安装 wrapper 后 ---")
model.model.text_backbone = wrapper

for cls in test_classes:
    imgs = load_images(cls, n_imgs)
    nbox, score = detect_with_boxes(model, processor, imgs, f'an overhead image of {cls}')
    print(f"  an overhead image of {cls:<20s} [TP]  {0:6.3f} {nbox:8.1f} {score:8.3f}")

# 4. 装 wrapper, 完全不相关类名
print("\n--- 负对照: 投影嵌入注入 + 不相关类名 ---")
for cls in test_classes[:2]:
    imgs = load_images(cls, n_imgs)
    # 用不相关类名 (airplane 图像对 "ship" prompt)
    wrong_cls = 'ship' if cls == 'airplane' else 'airplane'
    nbox, score = detect_with_boxes(model, processor, imgs, f'an overhead image of {wrong_cls}')
    print(f"  {cls} imgs + prompt '{wrong_cls}' [TP]     {0:6.3f} {nbox:8.1f} {score:8.3f}")

print("\n✅ 对照实验完成")
