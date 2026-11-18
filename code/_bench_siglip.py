# -*- coding: utf-8 -*-
import os, sys, time
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com'); os.environ['HF_HUB_OFFLINE']='1'
os.environ.setdefault('AC_SPC_DET_DEVICE','cuda')
sys.path.insert(0, r'C:/Users/张三/Desktop/实验代码'); sys.path.insert(0, r'C:/Users/张三/Desktop/A-CSPC')
import numpy as np, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
rng=np.random.RandomState(0)
sizes=[(rng.randint(20,600), rng.randint(20,600)) for _ in range(512)]
imgs=[Image.fromarray(rng.randint(0,255,(h,w,3),dtype=np.uint8)) for (w,h) in sizes]
for name,path in [('clip_l14',r'C:/Users/张三/modeling/vlm_weights/clip_l14'),
                  ('siglip_so400m',r'C:/Users/张三/modeling/vlm_weights/siglip_so400m')]:
    t=time.time(); m=AutoModel.from_pretrained(path).to('cuda').eval(); p=AutoProcessor.from_pretrained(path)
    print(f'load {name} {time.time()-t:.0f}s', flush=True)
    for half in [True,False]:
        mm = m.half() if half else m.float()
        # 单批时序 (bs=32): 区分 CPU 预处理 vs GPU 前向
        bs=32; b=imgs[:bs]
        t=time.time(); inp=p(images=b, return_tensors='pt'); t_cpu=time.time()-t
        inp=inp.to('cuda')
        if half: inp={k:(v.half() if v.dtype.is_floating_point else v) for k,v in inp.items()}
        ctx=torch.autocast('cuda',dtype=torch.float16) if half else torch.autocast('cuda',enabled=False)
        with torch.no_grad(), ctx:
            mm.get_image_features(**inp); torch.cuda.synchronize()
            t=time.time()
            for i in range(0,len(imgs),bs):
                inp=p(images=imgs[i:i+bs], return_tensors='pt').to('cuda')
                if half: inp={k:(v.half() if v.dtype.is_floating_point else v) for k,v in inp.items()}
                out=mm.get_image_features(**inp)
            torch.cuda.synchronize(); tot=time.time()-t
        print(f'  {name} {"fp16" if half else "fp32"}: cpu_preproc(1st bs=32)={t_cpu:.2f}s  {len(imgs)}imgs {tot:.1f}s -> {len(imgs)/tot:.1f} img/s', flush=True)
    del m,p
    import gc; gc.collect(); torch.cuda.empty_cache()
