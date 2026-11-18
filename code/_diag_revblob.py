# -*- coding: utf-8 -*-
import os, sys, json, pickle, glob
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('AC_SPC_DET_DEVICE','cpu')
sys.path.insert(0, r'C:/Users/张三/Desktop/实验代码'); sys.path.insert(0, r'C:/Users/张三/Desktop/A-CSPC')
import numpy as np, torch
import run_bidirectional as RB
RB.DEVICE='cpu'
import run_closed_loop_balanced as RBAL   # 复用其 ap_of / load_blob / bank 构建? -> 直接抄
from pathlib import Path
D = json.load(open(r'C:/Users/张三/Desktop/实验数据/run_closed_loop_balanced.json',encoding='utf-8'))
cells=[c for c in D['cells_detail'] if c['det']=='gdino' and c['direction']=='rev']
CACHE=Path(r'C:/Users/张三/Desktop/实验数据/probe_bidir_cache')
for c in cells:
    ev=c['eval']; tname='rev_coco2dior'
    cands = sorted(glob.glob(str(CACHE/f'{tname}_{ev}_off20_n*.pkl'))) + sorted(glob.glob(str(CACHE/f'{tname}_{ev}_n*.pkl')))
    cands=list(dict.fromkeys(cands))
    print(f"{c['src']}->{ev}: base_json={c['base']:.6f}  cands={[Path(x).name for x in cands]}")
    for fp in cands:
        b=pickle.load(open(fp,'rb'))
        pi=b['per_img']
        ng=sum(len(np.asarray(g,float).reshape(-1,4)) for p in pi for g in [p['gt']])
        nb=sum(len(p['box']) for p in pi)
        print(f"    {Path(fp).name}: n_img={len(pi)} n_box={nb} n_gt={ng}")
