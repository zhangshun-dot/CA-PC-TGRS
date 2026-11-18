# -*- coding: utf-8 -*-
"""COCO跨数据集 + GDINO-base 对比实验"""
import json, os, glob, numpy as np, torch, time, random
from PIL import Image
os.environ['HF_HUB_OFFLINE']='1'
device='cuda'
print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ── GDINO-base ──
print("Loading GDINO-base (fp16)...")
base_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-base\snapshots'
snaps = os.listdir(base_cache)
base_path = os.path.join(base_cache, snaps[0])
base_proc = AutoProcessor.from_pretrained(base_path)
base_model = AutoModelForZeroShotObjectDetection.from_pretrained(base_path, dtype=torch.float16).eval().to(device)
print(f"VRAM after loading: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# ── COCO data ──
# Use COCO val2017, 10 common classes, 20 images each
COCO_CLASSES = ['person','car','dog','chair','bottle','cup','boat','airplane','bird','potted plant']
coco_img_dir = r'C:\Users\张三\modeling\coco_data\val2017'

# Need annotations to map class names to images
# Quick approach: use COCO annotations
from pycocotools.coco import COCO
ann_file = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'
if not os.path.exists(ann_file):
    print("Downloading COCO annotations...")
    import urllib.request, zipfile
    ann_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    ann_zip = r'C:\Users\张三\modeling\coco_data\annotations.zip'
    if not os.path.exists(ann_zip):
        urllib.request.urlretrieve(ann_url, ann_zip)
    with zipfile.ZipFile(ann_zip) as zf:
        zf.extractall(r'C:\Users\张三\modeling\coco_data')

coco = COCO(ann_file)
cat_ids = coco.getCatIds(catNms=COCO_CLASSES)
cat_map = {coco.loadCats([cid])[0]['name']: cid for cid in cat_ids}

def load_coco_images(cls, n=20):
    """Load n COCO images containing class cls"""
    cat_id = cat_map.get(cls)
    if cat_id is None: return []
    img_ids = coco.getImgIds(catIds=[cat_id])[:n*3]  # oversample
    imgs = coco.loadImgs(img_ids[:n*3])
    paths = [os.path.join(coco_img_dir, img['file_name']) for img in imgs]
    loaded = []
    for p in paths:
        if os.path.exists(p) and len(loaded) < n:
            loaded.append(Image.open(p).convert('RGB'))
    return loaded

def detect_batch(images, prompt, proc=base_proc, model=base_model, threshold=0.25):
    dets = []
    for img in images:
        inputs = proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad(): o = model(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(device)
        r = proc.post_process_grounded_object_detection(o, threshold=threshold, target_sizes=ts)[0]
        dets.append(len(r['boxes'])>0)
    return np.mean(dets)

# ═══════════════════════════════════════════
# Quick test: overhead vs satellite on 10 COCO classes
# ═══════════════════════════════════════════
print(f"\n{'='*60}")
print("COCO 10-class: overhead vs satellite (GDINO-base)")
print(f"{'='*60}")

results = {}
for cls in COCO_CLASSES:
    imgs = load_coco_images(cls, 20)
    if len(imgs) < 10:
        print(f"  {cls}: only {len(imgs)} images, skipping")
        continue
    r_sat = detect_batch(imgs, f'a satellite image of {cls}')
    r_oh = detect_batch(imgs, f'an overhead image of {cls}')
    results[cls] = {'satellite': float(r_sat), 'overhead': float(r_oh)}
    print(f"  {cls:<20} sat={r_sat:.2f}  oh={r_oh:.2f}  Δ={r_oh-r_sat:+.2f}")

if results:
    sat_mean = np.mean([v['satellite'] for v in results.values()])
    oh_mean = np.mean([v['overhead'] for v in results.values()])
    print(f"\n  MEAN: sat={sat_mean:.3f}  oh={oh_mean:.3f}  Δ={oh_mean-sat_mean:+.3f}")
    print(f"  overhead > satellite: {sum(1 for v in results.values() if v['overhead']>=v['satellite'])}/{len(results)}")

# ═══════════════════════════════════════════
# Also test on DIOR 20 with GDINO-base (quick, 5 images per class)
# ═══════════════════════════════════════════
print(f"\n{'='*60}")
print("DIOR 20-class quick test (GDINO-base, 5 images/class)")
print(f"{'='*60}")

DIOR = sorted(['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
        'expressway service area','expressway toll station','golf course','ground track field',
        'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill'])
img_dir = r'C:\Users\张三\modeling\dior_data\full20_images'

dior_oh, dior_sat = [], []
for cls in DIOR:
    paths = sorted(glob.glob(os.path.join(img_dir, f"{cls.replace(' ','_')}_*.jpg")))[:5]
    imgs = [Image.open(p).convert('RGB') for p in paths]
    rs = detect_batch(imgs, f'a satellite image of {cls}')
    ro = detect_batch(imgs, f'an overhead image of {cls}')
    dior_sat.append(rs); dior_oh.append(ro)
    if abs(ro-rs) > 0.1:
        print(f"  {cls:<25} sat={rs:.2f} oh={ro:.2f} Δ={ro-rs:+.2f}")

print(f"  DIOR-base: sat={np.mean(dior_sat):.3f} oh={np.mean(dior_oh):.3f} Δ={np.mean(dior_oh)-np.mean(dior_sat):+.3f}")

# ═══════════════════════════════════════════
# Save
# ═══════════════════════════════════════════
out_data = {
    'coco_gdino_base': results,
    'dior_gdino_base_5img': {'sat_mean': float(np.mean(dior_sat)), 'oh_mean': float(np.mean(dior_oh))},
}
with open(r'C:\Users\张三\Desktop\coco_results.json', 'w') as f:
    json.dump(out_data, f, indent=2)
print(f"\nDone. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
