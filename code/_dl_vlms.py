import os, time
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
MODELS = [
    'openai/clip-vit-large-patch14',
    'laion/CLIP-ViT-L-14-laion2B-s32B-b82K',
    'laion/CLIP-ViT-B-32-laion2B-s34B-b79K',
    'google/siglip-so400m-patch14-384',
]
for m in MODELS:
    t0 = time.time()
    try:
        p = snapshot_download(m, allow_patterns=['*.json', '*.txt', '*.safetensors', '*.bin', '*.model'])
        print(f'OK   {m}  ({time.time()-t0:.0f}s)  {p}', flush=True)
    except Exception as e:
        print(f'FAIL {m}  {type(e).__name__} {str(e)[:200]}', flush=True)
print('DONE', flush=True)
