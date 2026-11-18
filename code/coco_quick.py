# -*- coding: utf-8 -*-
"""COCO跨数据集验证：overhead vs satellite + GDINO-base scaling"""
import json, os, glob, numpy as np, torch, time, random
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

print("Loading GDINO-base (fp16)...")
base_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-base\snapshots'
base_path = os.path.join(base_cache, os.listdir(base_cache)[0])
base_proc = AutoProcessor.from_pretrained(base_path)
base_model = AutoModelForZeroShotObjectDetection.from_pretrained(base_path, dtype=torch.float32).eval().to(device)
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# ── Data ──
coco_dir = r'C:\Users\张三\modeling\coco_data\val2017'
all_imgs = sorted(glob.glob(os.path.join(coco_dir, '*.jpg')))[:200]  # 200 random images
random.seed(42); random.shuffle(all_imgs)
print(f"COCO images: {len(all_imgs)}")

DIOR = sorted(['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
        'expressway service area','expressway toll station','golf course','ground track field',
        'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill'])
dior_dir = r'C:\Users\张三\modeling\dior_data\full20_images'

# ── COCO test: generic object detection ──
generic_prompts = [
    ('object', 'detect object'),
    ('vehicle', 'detect vehicle'),
    ('building', 'detect building'),
    ('animal', 'detect animal'),
]
prefixes = {
    'bare': '{cls}',
    'satellite': 'a satellite image of {cls}',
    'overhead': 'an overhead image of {cls}',
}

def detect_coco(images, prompt, proc, model, dtype=torch.float32):
    """Count total detections across images"""
    total = 0
    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        inputs = proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        # Convert to model dtype
        if dtype==torch.float16:
            for k in inputs:
                if isinstance(inputs[k], torch.Tensor) and inputs[k].dtype==torch.float32:
                    inputs[k] = inputs[k].half()
        if dtype == torch.float16:
            for k in list(inputs.keys()):
                if isinstance(inputs[k], torch.Tensor) and inputs[k].dtype == torch.float32:
                    inputs[k] = inputs[k].half()
        with torch.no_grad(): o = model(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(device)
        r = proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=ts)[0]
        total += len(r['boxes'])
    return total / len(images)  # avg detections per image

# ═══════════════════════════════════════════
# COCO: overhead vs satellite (GDINO-tiny)
# ═══════════════════════════════════════════
print(f"\n{'='*60}")
print("COCO 200 images: overhead vs satellite (GDINO-tiny)")
print(f"{'='*60}")

coco_tiny = {}
for pname, ptpl in prefixes.items():
    rates = []
    for cls_name, _ in generic_prompts:
        p = ptpl.replace('{cls}', cls_name)
        r = detect_coco(all_imgs, p, tiny_proc, tiny_model)
        rates.append(r)
    coco_tiny[pname] = {'mean': float(np.mean(rates)), 'per_prompt': dict(zip([p[0] for p in generic_prompts], [float(x) for x in rates]))}
    print(f"  {pname:<10}: {np.mean(rates):.2f} detections/image")

print(f"  oh/sat ratio: {coco_tiny['overhead']['mean']/max(coco_tiny['satellite']['mean'],0.01):.1f}x")

# ═══════════════════════════════════════════
# COCO: GDINO-base
# ═══════════════════════════════════════════
print(f"\n{'='*60}")
print("COCO 100 images: GDINO-base")
print(f"{'='*60}")

base_imgs = all_imgs[:100]
coco_base = {}
for pname in ['satellite', 'overhead']:
    ptpl = prefixes[pname]
    rates = []
    for cls_name, _ in generic_prompts:
        p = ptpl.replace('{cls}', cls_name)
        r = detect_coco(base_imgs, p, base_proc, base_model, dtype=torch.float16)
        rates.append(r)
    coco_base[pname] = float(np.mean(rates))
    print(f"  {pname:<10}: {np.mean(rates):.2f} detections/image")

# ═══════════════════════════════════════════
# DIOR: GDINO-base quick test
# ═══════════════════════════════════════════
print(f"\n{'='*60}")
print("DIOR 20-class: GDINO-base (5 images/class)")
print(f"{'='*60}")

dior_base_sat, dior_base_oh = [], []
for cls in DIOR:
    paths = sorted(glob.glob(os.path.join(dior_dir, f"{cls.replace(' ','_')}_*.jpg")))[:5]
    imgs = [Image.open(p).convert('RGB') for p in paths]
    rs = detect_coco(imgs, f'a satellite image of {cls}', base_proc, base_model, dtype=torch.float16)
    ro = detect_coco(imgs, f'an overhead image of {cls}', base_proc, base_model, dtype=torch.float16)
    dior_base_sat.append(rs); dior_base_oh.append(ro)

print(f"  GDINO-base DIOR: sat={np.mean(dior_base_sat):.3f} oh={np.mean(dior_base_oh):.3f}")

# Also GDINO-tiny DIOR for direct scaling comparison
dior_tiny_sat, dior_tiny_oh = [], []
for cls in DIOR[:5]:  # just 5 classes for tiny comparison
    paths = sorted(glob.glob(os.path.join(dior_dir, f"{cls.replace(' ','_')}_*.jpg")))[:5]
    imgs = [Image.open(p).convert('RGB') for p in paths]
    dior_tiny_sat.append(detect_coco(imgs, f'a satellite image of {cls}', tiny_proc, tiny_model))
    dior_tiny_oh.append(detect_coco(imgs, f'an overhead image of {cls}', tiny_proc, tiny_model))

# ═══════════════════════════════════════════
# Save
# ═══════════════════════════════════════════
results = {
    'coco_gdino_tiny': coco_tiny,
    'coco_gdino_base': coco_base,
    'dior_gdino_base': {'sat_mean': float(np.mean(dior_base_sat)), 'oh_mean': float(np.mean(dior_base_oh))},
    'dior_gdino_tiny_5cls': {'sat_mean': float(np.mean(dior_tiny_sat)), 'oh_mean': float(np.mean(dior_tiny_oh))},
}
with open(r'C:\Users\张三\Desktop\coco_results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"COCO GDINO-tiny: oh/sat = {coco_tiny['overhead']['mean']/max(coco_tiny['satellite']['mean'],0.01):.1f}x")
print(f"COCO GDINO-base: oh/sat = {coco_base['overhead']/max(coco_base['satellite'],0.01):.1f}x")
print(f"DIOR GDINO-base: oh={np.mean(dior_base_oh):.3f} sat={np.mean(dior_base_sat):.3f}")
print(f"DIOR GDINO-tiny: oh={np.mean(dior_tiny_oh):.3f} sat={np.mean(dior_tiny_sat):.3f}")
print(f"Done. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
