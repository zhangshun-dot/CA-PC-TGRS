# -*- coding: utf-8 -*-
"""完整体验：COCO 20类 TDM全量 + GDINO-base DIOR 20类 TDM全量"""
import json, os, glob, numpy as np, torch, time
from PIL import Image
os.environ['HF_HUB_OFFLINE']='1'
device='cuda'
print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ── Models ──
print("Loading GDINO-tiny...")
tiny_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
tiny_proc = AutoProcessor.from_pretrained(tiny_cache)
tiny_model = AutoModelForZeroShotObjectDetection.from_pretrained(tiny_cache, dtype=torch.float32).eval().to(device)

print("Loading GDINO-base...")
base_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-base\snapshots'
base_path = os.path.join(base_cache, os.listdir(base_cache)[0])
base_proc = AutoProcessor.from_pretrained(base_path)
base_model = AutoModelForZeroShotObjectDetection.from_pretrained(base_path, dtype=torch.float32).eval().to(device)
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# ── Data ──
DIOR = sorted(['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
        'expressway service area','expressway toll station','golf course','ground track field',
        'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill'])
dior_dir = r'C:\Users\张三\modeling\dior_data\full20_images'

# COCO 20 common classes
COCO_CLS = ['person','car','dog','chair','bottle','cup','boat','airplane','bird','potted plant',
            'dining table','couch','motorcycle','cat','tv','laptop','sink','refrigerator','book','clock']
coco_dir = r'C:\Users\张三\modeling\coco_data\val2017'
from pycocotools.coco import COCO
coco = COCO(r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json')
cat_ids_all = coco.getCatIds(catNms=COCO_CLS)
cat_map = {coco.loadCats([c])[0]['name']: c for c in cat_ids_all}

# ── OFAT data for TDM ──
with open(r'C:\Users\张三\Desktop\ofat_overhead_results.json') as f:
    ofat_dior = json.load(f)['ofat_overhead']
AXM = {'granularity':'with fine details','spatial':'in its natural setting',
       'geometric':'with distinct structural features','scale':'large and prominent',
       'spectral':'with natural coloring','texture':'with clearly visible surface features'}

def build_tdm(cls, ofat):
    d = ofat[cls]['deltas']
    gd = d.get('grammar',0)
    prefix = 'a satellite image of' if gd>0.05 else 'an overhead image of'
    pos = sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and v>0.03], key=lambda x:-x[1])
    mods = [AXM[ax] for ax,_ in pos[:2]] if pos else []
    p = f'{prefix} {cls}'
    if mods: p += ', ' + ', '.join(mods)
    return p

def detect(images, prompt, proc, model):
    dets = []
    for img in images:
        inputs = proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad(): o = model(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(device)
        r = proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=ts)[0]
        dets.append(len(r['boxes'])>0)
    return np.mean(dets)

# ═══════════════════════════════════════════════
# EXP 1: DIOR 20-class GDINO-base full
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print("EXP 1: DIOR 20-class GDINO-base (TDM vs overhead vs sat)")
print(f"{'='*60}")

dior_base = {}
for cls in DIOR:
    paths = sorted(glob.glob(os.path.join(dior_dir, f"{cls.replace(' ','_')}_*.jpg")))[:20]
    imgs = [Image.open(p).convert('RGB') for p in paths]
    r_sat = detect(imgs, f'a satellite image of {cls}', base_proc, base_model)
    r_oh = detect(imgs, f'an overhead image of {cls}', base_proc, base_model)
    r_tdm = detect(imgs, build_tdm(cls, ofat_dior), base_proc, base_model)
    dior_base[cls] = {'satellite': float(r_sat), 'overhead': float(r_oh), 'tdm': float(r_tdm)}
    d = r_tdm - r_oh
    print(f"  {cls:<25} sat={r_sat:.2f} oh={r_oh:.2f} tdm={r_tdm:.2f} Δ={d:+.2f}")

dior_base['mean_sat'] = float(np.mean([v['satellite'] for v in dior_base.values() if isinstance(v,dict)]))
dior_base['mean_oh'] = float(np.mean([v['overhead'] for v in dior_base.values() if isinstance(v,dict)]))
dior_base['mean_tdm'] = float(np.mean([v['tdm'] for v in dior_base.values() if isinstance(v,dict)]))
print(f"\n  MEAN: sat={dior_base['mean_sat']:.3f} oh={dior_base['mean_oh']:.3f} tdm={dior_base['mean_tdm']:.3f}")

# ═══════════════════════════════════════════════
# EXP 2: COCO 20-class GDINO-tiny (TDM via same axes)
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print("EXP 2: COCO 20-class GDINO-tiny")
print(f"{'='*60}")

coco_tiny = {}
for cls in COCO_CLS:
    cat_id = cat_map.get(cls)
    if cat_id is None: continue
    img_ids = coco.getImgIds(catIds=[cat_id])[:20]
    imgs_data = coco.loadImgs(img_ids)
    paths = [os.path.join(coco_dir, img['file_name']) for img in imgs_data]
    imgs = []
    for p in paths:
        if os.path.exists(p) and len(imgs) < 20:
            imgs.append(Image.open(p).convert('RGB'))
    if len(imgs) < 10:
        print(f"  {cls}: only {len(imgs)} images, skip")
        continue

    r_sat = detect(imgs, f'a satellite image of {cls}', tiny_proc, tiny_model)
    r_oh = detect(imgs, f'an overhead image of {cls}', tiny_proc, tiny_model)
    # Use overhead prefix for TDM (same logic as DIOR but without per-class OFAT)
    r_tdm = detect(imgs, f'an overhead image of {cls}, large and prominent', tiny_proc, tiny_model)
    coco_tiny[cls] = {'satellite': float(r_sat), 'overhead': float(r_oh), 'tdm': float(r_tdm)}
    print(f"  {cls:<20} sat={r_sat:.2f} oh={r_oh:.2f} tdm={r_tdm:.2f}")

coco_tiny['mean_sat'] = float(np.mean([v['satellite'] for v in coco_tiny.values() if isinstance(v,dict)]))
coco_tiny['mean_oh'] = float(np.mean([v['overhead'] for v in coco_tiny.values() if isinstance(v,dict)]))
coco_tiny['mean_tdm'] = float(np.mean([v['tdm'] for v in coco_tiny.values() if isinstance(v,dict)]))
print(f"\n  MEAN: sat={coco_tiny['mean_sat']:.3f} oh={coco_tiny['mean_oh']:.3f} tdm={coco_tiny['mean_tdm']:.3f}")

# ═══════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════
results = {
    'dior_gdino_base': dior_base,
    'coco_gdino_tiny': coco_tiny,
}
with open(r'C:\Users\张三\Desktop\full_experiment_results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nDone. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
