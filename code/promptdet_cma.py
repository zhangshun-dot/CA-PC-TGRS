# -*- coding: utf-8 -*-
"""
PromptDet for GDINO — CMA-ES 离散搜索
直接优化检测率, 搜索最優 Prompt 前缀
"""
import json, os, glob, numpy as np, torch, time, random, cma
from PIL import Image
os.environ['HF_HUB_OFFLINE']='1'
device='cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}")

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ── Models ──
print("Loading GDINO-tiny...")
gd_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
gd_proc = AutoProcessor.from_pretrained(gd_cache)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_cache, dtype=torch.float32).eval().to(device)

# ── Data ──
DIOR = sorted(['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
        'expressway service area','expressway toll station','golf course','ground track field',
        'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill'])
img_dir = r'C:\Users\张三\modeling\dior_data\full20_images'

random.seed(42); np.random.seed(42)
all_cls = DIOR.copy(); random.shuffle(all_cls)
train_cls = sorted(all_cls[:15])
test_cls = sorted(all_cls[15:])
print(f"Train ({len(train_cls)}): {train_cls}")
print(f"Test  ({len(test_cls)}): {test_cls}")

def load_images(cls, n=10):
    paths = sorted(glob.glob(os.path.join(img_dir, f"{cls.replace(' ','_')}_*.jpg")))[:n]
    return [Image.open(p).convert('RGB') for p in paths]

def detect_batch(images, prompt):
    dets = []
    for img in images:
        inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad(): o = gd_model(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(device)
        r = gd_proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=ts)[0]
        dets.append(len(r['boxes'])>0)
    return np.mean(dets)

# ── Candidate word pool ──
WORD_POOL = [
    'overhead','satellite','aerial','remote','sensing','photo','view','image',
    'large','small','prominent','clear','detailed','fine','sharp','bright','natural',
    'distinct','structural','geometric','surface','texture','pattern','color',
    'urban','rural','open','isolated','visible','high','resolution','background',
    'terrain','shadow','daylight','panoramic','surrounding','context','setting',
    'features','shape','outline','boundary','edge','contrast','dark','light',
]
N_WORDS = 4  # search for 4-word prefix
DIM = len(WORD_POOL)

# ── CMA-ES objective ──
# Pre-load training images for speed
print("\nLoading training images...")
train_imgs = {}
for cls in train_cls:
    train_imgs[cls] = load_images(cls, 10)

def objective(x):
    """CMA-ES objective: average detection rate across training classes"""
    # Convert continuous vector to discrete word selection
    # x shape: (N_WORDS * DIM), split into N_WORDS softmax groups
    x = x.reshape(N_WORDS, DIM)
    word_indices = np.argmax(x, axis=1)
    prefix = ' '.join([WORD_POOL[i] for i in word_indices])

    rates = []
    for cls in train_cls:
        prompt = f'{prefix} {cls}'
        r = detect_batch(train_imgs[cls], prompt)
        rates.append(r)

    return -np.mean(rates)  # CMA minimizes

# ── Run CMA-ES ──
print(f"\n{'='*60}")
print(f"CMA-ES: Searching {N_WORDS}-word prefix from {DIM} words")
print(f"{'='*60}")

t0 = time.time()
x0 = np.zeros(N_WORDS * DIM)
# Initialize: give slight bias to "overhead image of" words
for i, w in enumerate(['overhead','image','of','clear']):
    if w in WORD_POOL:
        idx = WORD_POOL.index(w)
        x0[i * DIM + idx] = 0.5

es = cma.CMAEvolutionStrategy(x0, 2.0, {
    'popsize': 15,
    'maxiter': 100,
    'tolx': 0.01,
    'verbose': -9,
})

best_prefix = None
best_rate = -1

while not es.stop():
    solutions = es.ask()
    fitness = [objective(x) for x in solutions]
    es.tell(solutions, fitness)

    # Track best
    current_best_idx = np.argmin(fitness)
    current_rate = -fitness[current_best_idx]
    if current_rate > best_rate:
        best_rate = current_rate
        x_best = solutions[current_best_idx].reshape(N_WORDS, DIM)
        best_indices = np.argmax(x_best, axis=1)
        best_prefix = ' '.join([WORD_POOL[i] for i in best_indices])

    if es.countiter % 10 == 0:
        elapsed = time.time() - t0
        print(f"  Iter {es.countiter:3d} | best_rate={best_rate:.3f} | prefix='{best_prefix}' | {elapsed:.0f}s")

elapsed = time.time() - t0
print(f"\n  CMA-ES done in {elapsed:.0f}s")
print(f"  Best prefix: '{best_prefix}'")
print(f"  Best training rate: {best_rate:.3f}")

# ── Final Evaluation ──
print(f"\n{'='*60}")
print("FINAL EVALUATION")
print(f"{'='*60}")

# PromptDet
pd_rates = []
for cls in test_cls:
    r = detect_batch(load_images(cls, 20), f'{best_prefix} {cls}')
    pd_rates.append(r)
pd_mean = np.mean(pd_rates)

# TDM
with open(r'C:\Users\张三\Desktop\ofat_overhead_results.json') as f:
    ofat = json.load(f)['ofat_overhead']
AXM = {'granularity':'with fine details','spatial':'in its natural setting',
       'geometric':'with distinct structural features','scale':'large and prominent',
       'spectral':'with natural coloring','texture':'with clearly visible surface features'}
tdm_rates = []
for cls in test_cls:
    d = ofat[cls]['deltas']
    gd_delta = d.get('grammar',0)
    prefix = 'a satellite image of' if gd_delta>0.05 else 'an overhead image of'
    pos = sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and v>0.03], key=lambda x:-x[1])
    mods = [AXM[ax] for ax,_ in pos[:2]] if pos else []
    p = f'{prefix} {cls}'
    if mods: p += ', ' + ', '.join(mods)
    tdm_rates.append(detect_batch(load_images(cls,20), p))
tdm_mean = np.mean(tdm_rates)

# Overhead
oh_rates = []
for cls in test_cls:
    r = detect_batch(load_images(cls, 20), f'an overhead image of {cls}')
    oh_rates.append(r)
oh_mean = np.mean(oh_rates)

print(f"\n  {'Class':<25} {'overhead':>8} {'PromptDet':>10} {'TDM':>8}")
print(f"  {'-'*55}")
for i, cls in enumerate(test_cls):
    print(f"  {cls:<25} {oh_rates[i]:8.2f} {pd_rates[i]:10.2f} {tdm_rates[i]:8.2f}")

print(f"  {'-'*55}")
print(f"  {'MEAN':<25} {oh_mean:8.3f} {pd_mean:10.3f} {tdm_mean:8.3f}")
print(f"\n  PromptDet vs TDM: {pd_mean-tdm_mean:+.3f}")

# Save
with open(r'C:\Users\张三\Desktop\promptdet_results.json','w') as f:
    json.dump({
        'method': 'CMA-ES',
        'learned_prefix': best_prefix,
        'promptdet_mean': float(pd_mean),
        'tdm_mean': float(tdm_mean),
        'overhead_mean': float(oh_mean),
        'elapsed_s': float(elapsed),
        'test_classes': test_cls,
        'per_class': {cls: {'oh': float(oh_rates[i]), 'pd': float(pd_rates[i]), 'tdm': float(tdm_rates[i])}
                      for i, cls in enumerate(test_cls)},
    }, f, indent=2, ensure_ascii=False)
print(f"\nDone.")
