# -*- coding: utf-8 -*-
"""Token Projection GDINO — 诊断脚本

检查:
  1. token 查找表是否覆盖了测试类名
  2. wrapper 实际修改了多少 token
  3. CD-TDM 推荐修饰词是否正确
  4. 投影前后的 BERT 嵌入差异
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import torch, torch.nn.functional as F
import numpy as np
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, AutoTokenizer
from token_projection import TokenProjection, AXIS_MODIFIERS, AXIS_ORDER
from ofat_knowledge_base import OFATKnowledgeBase

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
gd_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'

print("=" * 60)
print("Token Projection GDINO 诊断")
print("=" * 60)

# ── 1. 加载模型 ──
print("\n[1] 加载模型...")
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    gd_cache, dtype=torch.float32
).eval().to(DEVICE)
gd_tokenizer = AutoTokenizer.from_pretrained(gd_cache)

import clip
clip_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
clip_model, _ = clip.load(clip_path, device=DEVICE)
clip_model.eval()

kb = OFATKnowledgeBase()

# ── 2. 检查 CD-TDM 推荐 ──
print("\n[2] CD-TDM 推荐检查:")
test_classes = ['airplane', 'ship', 'stadium', 'bridge', 'harbor',
                'overpass', 'tennis court', 'windmill']

for cls in test_classes:
    recs = kb.get_recommended_axes('remote_sensing', cls)
    mods = [AXIS_MODIFIERS[a] for a, _ in recs[:2]]
    print(f"  {cls:<20s} → axes={[a for a,_ in recs[:2]]}  mods={mods}")
    # 也检查 OFAT delta 值
    for axis in AXIS_ORDER[:3]:
        d = kb.get_delta('remote_sensing', cls, axis)
        if d is not None:
            print(f"    {axis:<14s} δ={d:+.4f}")

# ── 3. 检查 tokenization ──
print("\n[3] Tokenization 检查:")
for cls in test_classes:
    prompt = f'an overhead image of {cls}'
    tok = gd_tokenizer(prompt, return_tensors='pt')
    ids = tok['input_ids'][0]
    tokens = [gd_tokenizer.decode([tid]) for tid in ids]
    print(f"  {cls:<20s} → ids={ids.tolist()}")
    print(f"    tokens={tokens}")

    # 类名单独 tokenize
    cls_ids = gd_tokenizer(cls, add_special_tokens=False)['input_ids']
    cls_tokens = [gd_tokenizer.decode([tid]) for tid in cls_ids]
    print(f"    class_only: ids={cls_ids}  tokens={cls_tokens}")

# ── 4. 检查投影嵌入的 token 查找表 ──
print("\n[4] Token 查找表覆盖:")
from token_projection_gdino import Clip2TokenMapper, precompute_class_projections
from transformers import BertModel

bert_model_mapping = BertModel.from_pretrained(
    'bert-base-uncased', local_files_only=True
).eval().to(DEVICE)

# 重新训练 W_D
from token_projection import compute_axis_directions, train_projection
axis_dirs_clip = compute_axis_directions(clip_model, clip.tokenize, device=DEVICE)
proj_wd = TokenProjection(d_model=512, rank=4, domain='remote_sensing')
train_projection(proj_wd, kb, axis_directions=axis_dirs_clip,
                 epochs=200, lr=0.005, device=DEVICE, verbose=False)

mapper = Clip2TokenMapper(clip_model, bert_model_mapping)

all_classes = list(set(
    ['airplane','airport','baseball field','basketball court','bridge',
     'chimney','dam','expressway service area','expressway toll station',
     'golf course','ground track field','harbor','overpass','ship','stadium',
     'storage tank','tennis court','train station','vehicle','windmill',
     'bird', 'car', 'dog', 'cat', 'building', 'tree', 'road', 'person',
     'truck', 'bicycle', 'motorcycle', 'bus', 'train', 'chair', 'table', 'house']
))
class_projections, class_token_ids = precompute_class_projections(
    proj_wd, mapper, clip_model, all_classes, DEVICE
)

# 构建查找表
from token_projection_gdino import TextBackboneWrapper
wrapper = TextBackboneWrapper(
    model.model.text_backbone, proj_wd, mapper, clip_model,
    gd_tokenizer, class_projections, class_token_ids
)

for cls in test_classes:
    cls_ids = class_token_ids.get(cls, [])
    matched = []
    unmatched = []
    for tid in cls_ids:
        if tid in wrapper.token_lookup:
            matched.append(gd_tokenizer.decode([tid]))
        else:
            unmatched.append(gd_tokenizer.decode([tid]))
    print(f"  {cls:<20s} matched={matched}  unmatched={unmatched}")

# ── 5. 检查投影嵌入与原始嵌入的差异 ──
print("\n[5] 投影 vs 原始 BERT word embedding 差异:")
with torch.no_grad():
    word_emb = model.model.text_backbone.embeddings.word_embeddings.weight

for cls in test_classes:
    cls_ids = class_token_ids.get(cls, [])
    if cls_ids:
        tid = cls_ids[0]  # 第一个 token
        orig_emb = word_emb[tid]  # (768,)
        proj_emb = wrapper.token_lookup.get(tid)
        if proj_emb is not None:
            orig_norm = orig_emb.norm().item()
            proj_norm = proj_emb.norm().item()
            cos = F.cosine_similarity(orig_emb.unsqueeze(0).to(DEVICE),
                                       proj_emb.unsqueeze(0).to(DEVICE)).item()
            diff_norm = (proj_emb.to(DEVICE) - orig_emb).norm().item()
            print(f"  {cls:<20s} |orig|={orig_norm:.2f} |proj|={proj_norm:.2f} "
                  f"cos={cos:+.4f} |Δ|={diff_norm:.2f}")
        else:
            print(f"  {cls:<20s} token {tid} ('{gd_tokenizer.decode([tid])}') NOT in lookup!")

# ── 6. 检查 BERT 输出差异 ──
print("\n[6] BERT 输出差异 (投影前后):")
tb = model.model.text_backbone  # 原始 BERT backbone
tp_proj = model.model.text_projection

for cls in test_classes[:4]:
    prompt = f'an overhead image of {cls}'
    tok = gd_tokenizer(prompt, return_tensors='pt').to(DEVICE)

    # 原始 BERT 输出
    with torch.no_grad():
        out_raw = tb(input_ids=tok['input_ids'],
                     attention_mask=tok['attention_mask'],
                     token_type_ids=torch.zeros_like(tok['input_ids']),
                     position_ids=torch.arange(tok['input_ids'].shape[1], device=DEVICE).unsqueeze(0),
                     return_dict=True)
        feat_raw = tp_proj(out_raw.last_hidden_state)[:, 0, :]  # [CLS] 256-d

    # 投影后 BERT 输出 (通过 wrapper)
    model.model.text_backbone = wrapper
    with torch.no_grad():
        out_proj = wrapper(input_ids=tok['input_ids'],
                          attention_mask=tok['attention_mask'],
                          token_type_ids=torch.zeros_like(tok['input_ids']),
                          position_ids=torch.arange(tok['input_ids'].shape[1], device=DEVICE).unsqueeze(0),
                          return_dict=True)
        feat_proj = tp_proj(out_proj.last_hidden_state)[:, 0, :]

    model.model.text_backbone = tb  # 恢复

    cos_feat = F.cosine_similarity(feat_raw, feat_proj, dim=-1).item()
    diff_norm = (feat_proj - feat_raw).norm().item()
    print(f"  {cls:<20s} GDINO 256-d [CLS]: cos={cos_feat:+.4f} |Δ|={diff_norm:.3f}")

print("\n✅ 诊断完成")
