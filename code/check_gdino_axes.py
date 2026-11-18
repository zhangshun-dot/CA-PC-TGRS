# -*- coding: utf-8 -*-
"""检查 GDINO 256-d 空间中轴方向的可分性"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from transformers import AutoModelForZeroShotObjectDetection

gd_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_cache, dtype=torch.float32)
model.eval().cuda()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(gd_cache)

tb = model.model.text_backbone
tp = model.model.text_projection

AXIS_MODIFIERS = {
    'granularity': 'with fine details', 'spatial': 'in its natural setting',
    'geometric': 'with distinct structural features', 'scale': 'large and prominent',
    'spectral': 'with natural coloring', 'texture': 'with clearly visible surface features',
}
AXIS_ORDER = list(AXIS_MODIFIERS.keys())

classes = ['airplane','bird','car','dog','cat','ship','bridge','stadium',
           'building','tree','road','person','truck','bicycle','motorcycle',
           'bus','train','chair','table','house']

def gdino_encode(prompt):
    tok = tokenizer(prompt, return_tensors='pt').to('cuda')
    with torch.no_grad():
        bert_out = tb(**{k:v for k,v in tok.items() if k in ['input_ids','attention_mask','token_type_ids']})
        feat = tp(bert_out.last_hidden_state)
        return feat[:, 0, :].squeeze(0)  # [CLS] 256-d

axis_dirs = {}
for axis in AXIS_ORDER:
    modifier = AXIS_MODIFIERS[axis]
    dirs = []
    for cls in classes:
        base = gdino_encode(f'a photo of {cls}')
        mod = gdino_encode(f'a photo of {cls}, {modifier}')
        dirs.append(mod - base)
    mean_dir = torch.stack(dirs).mean(dim=0)
    axis_dirs[axis] = mean_dir / (mean_dir.norm() + 1e-8)

print('GDINO 256-d 空间轴方向余弦相似度:')
for a1 in AXIS_ORDER:
    sims = []
    for a2 in AXIS_ORDER:
        sim = F.cosine_similarity(axis_dirs[a1].unsqueeze(0), axis_dirs[a2].unsqueeze(0)).item()
        sims.append(f'{sim:+.3f}')
    print(f'  {a1:14s} {" ".join(sims)}')

# 汇总对比
cos_vals = []
for i in range(len(AXIS_ORDER)):
    for j in range(i+1, len(AXIS_ORDER)):
        cos_vals.append(F.cosine_similarity(axis_dirs[AXIS_ORDER[i]].unsqueeze(0), axis_dirs[AXIS_ORDER[j]].unsqueeze(0)).item())

print(f'\nGDINO 256-d: min={min(cos_vals):.3f}  max={max(cos_vals):.3f}  mean={sum(cos_vals)/len(cos_vals):.3f}')
print('BERT  768-d:  min=0.575  max=0.921  mean=0.680')
print('CLIP  512-d:  min=0.160  max=0.646  mean=0.378')
