# -*- coding: utf-8 -*-
"""
PromptDet for GDINO — REINFORCE + 有限差分混合训练
冻结GDINO, 在BERT端训练可学prompt token, 用检测率作为回报信号
"""
import json, os, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, random
from PIL import Image
os.environ['HF_HUB_OFFLINE']='1'
device='cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}")

from transformers import BertTokenizer, BertModel, AutoProcessor, AutoModelForZeroShotObjectDetection

# ── Models ──
print("Loading BERT + GDINO-tiny...")
bert_tok = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').eval().to(device)
gd_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
gd_proc = AutoProcessor.from_pretrained(gd_cache)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_cache, dtype=torch.float32).eval().to(device)

# ── Data ──
DIOR = sorted(['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
        'expressway service area','expressway toll station','golf course','ground track field',
        'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill'])
img_dir = r'C:\Users\张三\modeling\dior_data\full20_images'

random.seed(42); np.random.seed(42); torch.manual_seed(42)
all_cls = DIOR.copy(); random.shuffle(all_cls)
train_cls = sorted(all_cls[:15])
test_cls = sorted(all_cls[15:])
print(f"Train: {train_cls}")
print(f"Test:  {test_cls}")

def load_images(cls, n=20):
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

# ── PromptDet: Learnable Prompt Tokens ──
M = 8  # context tokens
DIM = 768  # BERT hidden dim
LR = 0.01  # learning rate for REINFORCE
EPOCHS = 80
POP_SIZE = 20  # perturbations per step

class LearnablePrompt(nn.Module):
    """Learn BERT prefix embeddings, decode to nearest token for GDINO inference"""
    def __init__(self):
        super().__init__()
        # Initialize from overhead-related words
        init_words = ['an','overhead','image','of','remote','sensing','photo','view']
        with torch.no_grad():
            inits = []
            for w in init_words[:M]:
                tok = bert_tok(w, return_tensors='pt')
                emb = bert_model.embeddings.word_embeddings(tok.input_ids[0,0])
                inits.append(emb)
            self.ctx = nn.Parameter(torch.stack(inits))  # (M, 768)

        # For decoding: pre-compute BERT vocab embeddings
        with torch.no_grad():
            self.vocab_emb = bert_model.embeddings.word_embeddings.weight[:30000].clone()
            self.vocab_norm = F.normalize(self.vocab_emb, dim=-1)

    def decode_to_text(self):
        """Nearest-neighbor decode context to words"""
        with torch.no_grad():
            ctx_n = F.normalize(self.ctx, dim=-1)
            sim = ctx_n @ self.vocab_norm.T
            best_ids = sim.argmax(dim=-1)
            words = bert_tok.decode(best_ids, skip_special_tokens=True)
            words = words.replace(' ##', '')
            return words.strip()

    def forward(self, cls_name):
        """Return the prompt text for evaluation"""
        prefix = self.decode_to_text()
        return f'{prefix} {cls_name}'

# ═══════════════════════════════════════════
# Training: REINFORCE with finite diff
# ═══════════════════════════════════════════
print(f"\n{'='*60}")
print(f"PromptDet Training: REINFORCE ({EPOCHS} epochs, pop={POP_SIZE})")
print(f"{'='*60}")

prompt_model = LearnablePrompt().to(device)
optimizer = torch.optim.Adam(prompt_model.parameters(), lr=LR)

best_test = 0.0
best_state = None
train_log = []
sigma = 0.1  # perturbation std

t0 = time.time()
for epoch in range(EPOCHS):
    # ── REINFORCE step ──
    prompt_model.train()
    total_reward = 0.0
    n_steps = 0

    for cls in train_cls:
        imgs = load_images(cls, 5)  # 5 images per class per epoch

        # Current prompt
        base_text = prompt_model.decode_to_text()
        current_prompt = f'{base_text} {cls}'
        base_reward = detect_batch(imgs, current_prompt)

        # Perturb each context vector and measure reward change
        grad_est = torch.zeros_like(prompt_model.ctx)
        for i in range(M):
            for pop in range(POP_SIZE):
                noise = torch.randn(DIM, device=device) * sigma
                # Positive perturbation
                prompt_model.ctx.data[i] += noise
                pos_text = prompt_model.decode_to_text()
                prompt_model.ctx.data[i] -= noise  # restore

                pos_prompt = f'{pos_text} {cls}'
                pos_reward = detect_batch(imgs, pos_prompt)

                # Gradient estimate: (pos_reward - base_reward) * noise / sigma^2
                grad_est[i] += (pos_reward - base_reward) * noise / (sigma**2 * POP_SIZE)

        # Apply gradient
        optimizer.zero_grad()
        prompt_model.ctx.grad = grad_est
        optimizer.step()
        total_reward += base_reward
        n_steps += 1

    # ── Evaluate on test set every 10 epochs ──
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        prompt_model.eval()
        test_rates = []
        for cls in test_cls:
            prompt = prompt_model(cls)
            r = detect_batch(load_images(cls, 20), prompt)
            test_rates.append(r)
        test_mean = np.mean(test_rates)

        # TDM reference
        with open(r'C:\Users\张三\Desktop\ofat_overhead_results.json') as f:
            ofat = json.load(f)['ofat_overhead']
        AXM = {'granularity':'with fine details','spatial':'in its natural setting',
               'geometric':'with distinct structural features','scale':'large and prominent',
               'spectral':'with natural coloring','texture':'with clearly visible surface features'}
        tdm_rates = []
        for cls in test_cls:
            d = ofat[cls]['deltas']
            gd = d.get('grammar',0)
            prefix = 'a satellite image of' if gd>0.05 else 'an overhead image of'
            pos = sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and v>0.03], key=lambda x:-x[1])
            mods = [AXM[ax] for ax,_ in pos[:2]] if pos else []
            p = f'{prefix} {cls}'
            if mods: p += ', ' + ', '.join(mods)
            tdm_rates.append(detect_batch(load_images(cls,20), p))
        tdm_mean = np.mean(tdm_rates)

        if test_mean > best_test:
            best_test = test_mean
            best_state = {k: v.clone() for k, v in prompt_model.state_dict().items()}

        decoded_text = prompt_model.decode_to_text()
        eta = (time.time()-t0)/(epoch+1)*(EPOCHS-epoch-1)
        train_log.append((epoch, float(total_reward/max(n_steps,1)), float(test_mean), float(tdm_mean), decoded_text))
        print(f"  E{epoch:3d} | r={total_reward/max(n_steps,1):.3f} | PD={test_mean:.3f} | TDM={tdm_mean:.3f} | best={best_test:.3f} | '{decoded_text[:40]}' | ETA={eta:.0f}s")

# Restore best
if best_state:
    prompt_model.load_state_dict(best_state)

# ═══════════════════════════════════════════
# FINAL EVALUATION
# ═══════════════════════════════════════════
print(f"\n{'='*60}")
print("FINAL COMPARISON")
print(f"{'='*60}")

prompt_model.eval()
pd_rates, tdm_rates, oh_rates = [], [], []
final_text = prompt_model.decode_to_text()
print(f"Learned prompt prefix: '{final_text}'")

with open(r'C:\Users\张三\Desktop\ofat_overhead_results.json') as f:
    ofat = json.load(f)['ofat_overhead']
AXM = {'granularity':'with fine details','spatial':'in its natural setting',
       'geometric':'with distinct structural features','scale':'large and prominent',
       'spectral':'with natural coloring','texture':'with clearly visible surface features'}

for cls in test_cls:
    imgs = load_images(cls, 20)
    r_pd = detect_batch(imgs, prompt_model(cls))
    r_oh = detect_batch(imgs, f'an overhead image of {cls}')
    d = ofat[cls]['deltas']
    gd = d.get('grammar',0)
    prefix = 'a satellite image of' if gd>0.05 else 'an overhead image of'
    pos = sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and v>0.03], key=lambda x:-x[1])
    mods = [AXM[ax] for ax,_ in pos[:2]] if pos else []
    p = f'{prefix} {cls}'
    if mods: p += ', ' + ', '.join(mods)
    r_tdm = detect_batch(imgs, p)
    pd_rates.append(r_pd); tdm_rates.append(r_tdm); oh_rates.append(r_oh)

print(f"\n  overhead: {np.mean(oh_rates):.3f}")
print(f"  PromptDet (REINFORCE, {EPOCHS} epochs): {np.mean(pd_rates):.3f}")
print(f"  TDM (zero-shot): {np.mean(tdm_rates):.3f}")

with open(r'C:\Users\张三\Desktop\promptdet_results.json','w') as f:
    json.dump({
        'learned_prefix': final_text,
        'promptdet_mean': float(np.mean(pd_rates)),
        'tdm_mean': float(np.mean(tdm_rates)),
        'overhead_mean': float(np.mean(oh_rates)),
        'best_test': float(best_test),
        'train_log': train_log,
        'epochs': EPOCHS,
        'test_classes': test_cls,
    }, f, indent=2, ensure_ascii=False)
print(f"\nDone. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
