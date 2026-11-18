# -*- coding: utf-8 -*-
"""PromptDet: CMA-ES 搜索最优 Prompt 前缀, 直接优化检测率"""
import json, os, glob, numpy as np, torch, time, random, cma
from PIL import Image
os.environ['HF_HUB_OFFLINE']='1'
device='cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}")

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ── Model ──
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
print(f"Train: {train_cls}")
print(f"Test:  {test_cls}")

def load_images(cls, n=1):
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

# ── Load training images ──
print("Loading training images (1/class)...")
train_imgs = {cls: load_images(cls, 1) for cls in train_cls}

# ── Word pool (top 30 most relevant) ──
WORDS = ['overhead','aerial','remote','sensing','satellite','clear','sharp','detailed',
         'large','prominent','visible','natural','bright','dark','distinct','structural',
         'geometric','texture','surface','pattern','urban','rural','open','isolated',
         'background','terrain','shadow','daylight','panoramic','context']
N = 3  # search 3-word prefix
D = len(WORDS)
print(f"Search space: {N} words from {D} candidates = {D**N:,} combinations")

# ── CMA-ES ──
def objective(x):
    x = x.reshape(N, D)
    idx = np.argmax(x, axis=1)
    prefix = ' '.join([WORDS[i] for i in idx])
    rates = [detect_batch(train_imgs[cls], f'{prefix} {cls}') for cls in train_cls]
    return -np.mean(rates)

t0 = time.time()
x0 = np.zeros(N * D)
# Bias toward "overhead" + "image" + "clear"
for i, w in enumerate(['overhead','clear','prominent']):
    if w in WORDS:
        x0[i * D + WORDS.index(w)] = 0.5

es = cma.CMAEvolutionStrategy(x0, 2.0, {'popsize': 6, 'maxiter': 30, 'tolx': 0.01, 'verbose': -9})
best_prefix, best_rate = None, -1

print(f"\nCMA-ES: max 30 iters, pop=6, {len(train_cls)} train classes, 1 img/class")
print(f"Estimated: ~{30*6*len(train_cls)*0.5/60:.0f} min")

while not es.stop():
    solutions = es.ask()
    fitness = [objective(x) for x in solutions]
    es.tell(solutions, fitness)
    best_idx = np.argmin(fitness)
    rate = -fitness[best_idx]
    if rate > best_rate:
        best_rate = rate
        xb = solutions[best_idx].reshape(N, D)
        best_prefix = ' '.join([WORDS[i] for i in np.argmax(xb, axis=1)])
    if es.countiter % 5 == 0:
        elapsed = time.time() - t0
        print(f"  Iter {es.countiter:2d} | best={best_rate:.3f} | '{best_prefix}' | {elapsed:.0f}s")

print(f"\nDone in {time.time()-t0:.0f}s. Best: '{best_prefix}' rate={best_rate:.3f}")

# ── Final Eval ──
print(f"\n{'='*60}")
print("FINAL: PromptDet vs TDM vs overhead")
print(f"{'='*60}")

with open(r'C:\Users\张三\Desktop\ofat_overhead_results.json') as f:
    ofat = json.load(f)['ofat_overhead']
AXM = {'granularity':'with fine details','spatial':'in its natural setting',
       'geometric':'with distinct structural features','scale':'large and prominent',
       'spectral':'with natural coloring','texture':'with clearly visible surface features'}

pd_rates, tdm_rates, oh_rates = [], [], []
for cls in test_cls:
    imgs = load_images(cls, 20)
    r_pd = detect_batch(imgs, f'{best_prefix} {cls}')
    r_oh = detect_batch(imgs, f'an overhead image of {cls}')
    d = ofat[cls]['deltas']; gd = d.get('grammar',0)
    prefix = 'a satellite image of' if gd>0.05 else 'an overhead image of'
    pos = sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and v>0.03], key=lambda x:-x[1])
    mods = [AXM[ax] for ax,_ in pos[:2]] if pos else []
    p = f'{prefix} {cls}'
    if mods: p += ', ' + ', '.join(mods)
    r_tdm = detect_batch(imgs, p)
    pd_rates.append(r_pd); tdm_rates.append(r_tdm); oh_rates.append(r_oh)
    print(f"  {cls:<25} oh={r_oh:.2f}  PD={r_pd:.2f}  TDM={r_tdm:.2f}")

print(f"\n  MEAN: overhead={np.mean(oh_rates):.3f}  PromptDet={np.mean(pd_rates):.3f}  TDM={np.mean(tdm_rates):.3f}")

with open(r'C:\Users\张三\Desktop\promptdet_results.json','w') as f:
    json.dump({'method':'CMA-ES','prefix':best_prefix,'train_rate':float(best_rate),
        'promptdet_mean':float(np.mean(pd_rates)),'tdm_mean':float(np.mean(tdm_rates)),
        'overhead_mean':float(np.mean(oh_rates)),'test_classes':test_cls}, f, indent=2)
print("Done")
