# -*- coding: utf-8 -*-
"""PCA Token Embedding Analysis — 验证七轴划分的完备性。

提取BERT文本编码器的Token嵌入 → PCA降维 → 报告前7个主成分的方差占比。
若前7个PC覆盖>80%方差，则从数据角度佐证七轴划分的完备性。
"""
import json, torch, numpy as np, io
from PIL import Image
from collections import defaultdict
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from datetime import datetime

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Start: {datetime.now()}", flush=True)

OUT = r'C:\Users\张三\Desktop\pca_analysis_results.json'

DIOR = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

# ── OFAT data ──
with open(r'C:\Users\张三\Desktop\初稿4_论文文件\实验数据\ofat_overhead_results.json') as f:
    ofat = json.load(f)['ofat_overhead']

AXM = {'granularity':'with fine details','spatial':'in its natural setting',
       'geometric':'with distinct structural features','scale':'large and prominent',
       'spectral':'with natural coloring','texture':'with clearly visible surface features'}

def build_tdm_prompt(cls):
    if cls not in ofat: return f'an overhead image of {cls}'
    d = ofat[cls]['deltas']
    pos = sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and v>0.03], key=lambda x:-x[1])
    mods = [AXM[ax] for ax,_ in pos[:2]] if pos else []
    p = f'an overhead image of {cls}'
    if mods: p += ', ' + ', '.join(mods)
    return p

# ── Load GDINO-tiny ──
print("Loading GDINO-tiny...", flush=True)
tc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
gd_proc = AutoProcessor.from_pretrained(tc)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(tc, dtype=torch.float32).eval().to(device)

# ── Also generate prompts for each OFAT axis individually ──
AXIS_PROMPTS = {
    'grammar_oh': 'an overhead image of {cls}',
    'grammar_sat': 'a satellite image of {cls}',
    'granularity': 'an overhead image of {cls}, with fine details',
    'spatial': 'an overhead image of {cls}, in its natural setting',
    'geometric': 'an overhead image of {cls}, with distinct structural features',
    'scale': 'an overhead image of {cls}, large and prominent',
    'spectral': 'an overhead image of {cls}, with natural coloring',
    'texture': 'an overhead image of {cls}, with clearly visible surface features',
}

# ── Collect token embeddings ──
all_embeddings = []  # list of [num_tokens, hidden_dim] for each prompt
embedding_metadata = []  # track which prompt/class each embedding came from

print(f"\nExtracting token embeddings for {len(DIOR)} classes × {len(AXIS_PROMPTS)+1} prompts...", flush=True)

# Load one sample image per class (just for tokenizer context — the text encoder doesn't need images)
pq_file = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
table = pq.read_table(pq_file); rows = table.to_pylist()
class_data = defaultdict(list)
for row in rows:
    img_bytes, objs = row['image']['bytes'], row['objects']
    if not objs or 'category' not in objs: continue
    for cat in set(objs.get('category', [])):
        if len(class_data[DIOR[cat]]) < 1:
            class_data[DIOR[cat]].append(img_bytes)

for cls_name in DIOR:
    if cls_name not in class_data: continue
    img_bytes = class_data[cls_name][0]
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # TDM prompt
    tdm_prompt = build_tdm_prompt(cls_name)
    inputs = gd_proc(images=img, text=tdm_prompt, return_tensors='pt')
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        o = gd_model(**inputs, output_hidden_states=True)

    # Get text encoder last hidden state: [1, num_tokens, hidden_dim]
    text_hidden = o.encoder_last_hidden_state_text[0].cpu().numpy()  # [num_tokens, 256]
    input_ids = o.input_ids[0].cpu().numpy()

    # Only keep non-padding tokens
    valid_mask = input_ids != gd_proc.tokenizer.pad_token_id
    valid_embeddings = text_hidden[valid_mask]  # [valid_tokens, 256]

    all_embeddings.append(valid_embeddings)
    embedding_metadata.append({'class': cls_name, 'prompt_type': 'tdm', 'n_tokens': len(valid_embeddings)})

    if len(DIOR) <= 5:
        print(f"  {cls_name}: TDM prompt: {len(valid_embeddings)} tokens")

    # Per-axis prompts
    for axis_name, template in AXIS_PROMPTS.items():
        axis_prompt = template.replace('{cls}', cls_name)
        inputs = gd_proc(images=img, text=axis_prompt, return_tensors='pt')
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            o = gd_model(**inputs, output_hidden_states=True)

        text_hidden = o.encoder_last_hidden_state_text[0].cpu().numpy()
        input_ids = o.input_ids[0].cpu().numpy()
        valid_mask = input_ids != gd_proc.tokenizer.pad_token_id
        valid_embeddings = text_hidden[valid_mask]
        all_embeddings.append(valid_embeddings)
        embedding_metadata.append({'class': cls_name, 'prompt_type': axis_name, 'n_tokens': len(valid_embeddings)})

# ── Concatenate and run PCA ──
all_tokens = np.concatenate(all_embeddings, axis=0)  # [total_tokens, 256]
print(f"\nTotal token embeddings: {all_tokens.shape[0]} × {all_tokens.shape[1]}", flush=True)

pca = PCA(n_components=min(20, all_tokens.shape[1]))
pca.fit(all_tokens)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

print(f"\n{'='*60}")
print("PCA RESULTS: Token Embedding Variance Explained")
print(f"{'='*60}")
for i in range(min(10, len(explained))):
    print(f"  PC{i+1:2d}: {explained[i]*100:5.1f}%  (cumulative: {cumulative[i]*100:5.1f}%)")

print(f"\n  First 7 PCs total: {cumulative[6]*100:.1f}%")
print(f"  First 10 PCs total: {cumulative[9]*100:.1f}%")

# ── Save results ──
results = {
    'total_tokens': int(all_tokens.shape[0]),
    'hidden_dim': int(all_tokens.shape[1]),
    'n_prompts': len(embedding_metadata),
    'n_classes': len(DIOR),
    'explained_variance_ratio': [float(v) for v in explained[:20]],
    'cumulative_variance': [float(v) for v in cumulative[:20]],
    'pc7_cumulative': float(cumulative[6]),
    'pc10_cumulative': float(cumulative[9]),
    'pca_components': pca.components_[:7].tolist(),
    'conclusion': f"前7个主成分解释{cumulative[6]*100:.1f}%方差——" +
        ("七轴划分从数据角度得到佐证" if cumulative[6] > 0.80 else
         "七轴划分解释了主要方差但存在未覆盖维度" if cumulative[6] > 0.60 else
         "人工七轴划分仅覆盖小部分语义方差，需重新审视"),
}

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n结论: {results['conclusion']}")
print(f"\nSaved: {OUT}")
print(f"Done: {datetime.now()}")
torch.cuda.empty_cache()
