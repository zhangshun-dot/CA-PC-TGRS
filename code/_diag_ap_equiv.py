# -*- coding: utf-8 -*-
"""对照: 我的 ap_from_evals/per_image_evals vs 官方 Met.compute_ap50_img (任意 W, 不需 CLIP)"""
import os, sys, pickle, glob
os.environ.setdefault('AC_SPC_DET_DEVICE','cpu')
sys.path.insert(0, r'C:/Users/张三/Desktop/实验代码'); sys.path.insert(0, r'C:/Users/张三/Desktop/A-CSPC')
import numpy as np
from acspc import metrics as Met
import run_bidirectional as RB
sys.path.insert(0, r'C:/Users/张三/Desktop/实验代码')
import importlib.util
spec=importlib.util.spec_from_file_location('RSB', r'C:/Users/张三/Desktop/实验代码/run_sizebin_pairedci.py')
# 直接内联拷贝两个函数, 避免 import 触发 main
def iou_mat(b,g):
    b=np.asarray(b,float).reshape(-1,4); g=np.asarray(g,float).reshape(-1,4)
    if len(b)==0 or len(g)==0: return np.zeros((len(b),len(g)))
    ix1=np.maximum(b[:,None,0],g[None,:,0]); iy1=np.maximum(b[:,None,1],g[None,:,1])
    ix2=np.minimum(b[:,None,2],g[None,:,2]); iy2=np.minimum(b[:,None,3],g[None,:,3])
    inter=np.maximum(0.0,ix2-ix1)*np.maximum(0.0,iy2-iy1)
    ab=(b[:,2]-b[:,0])*(b[:,3]-b[:,1]); ag=(g[:,2]-g[:,0])*(g[:,3]-g[:,1])
    return inter/(ab[:,None]+ag[None,:]-inter+1e-8)
def per_image_evals(boxes, scores, gts, iou_th, gt_mask=None):
    evs=[]; ngt=[]
    for b,s,g in zip(boxes,scores,gts):
        gm=np.asarray(g,float).reshape(-1,4)
        if gt_mask is not None: gm=gm[gt_mask(gm)]
        ng=len(gm); ngt.append(ng)
        if len(b)==0: evs.append((np.zeros(0),np.zeros(0,int))); continue
        M=iou_mat(b,gm) if ng else np.zeros((len(b),0))
        idx=np.argsort(-np.asarray(s,float)); matched=np.zeros(ng,bool); tp=np.zeros(len(b),int)
        for k,i in enumerate(idx):
            if ng==0: tp[k]=0; continue
            row=M[i].copy(); row[matched]=-1.0; bj=int(row.argmax())
            if row[bj]>=iou_th and bj>=0: tp[k]=1; matched[bj]=True
        evs.append((np.asarray(s,float)[idx],tp))
    return evs,ngt
def ap_from_evals(evs,n_gt,pts=11):
    if n_gt==0: return 0.0 if any(len(e[0]) for e in evs) else 1.0
    if not evs: return 0.0
    S=np.concatenate([e[0] for e in evs]); T=np.concatenate([e[1] for e in evs])
    if len(S)==0: return 0.0
    o=np.argsort(-S,kind='stable'); T=T[o]
    tc=np.cumsum(T).astype(float); fc=np.cumsum(1-T).astype(float)
    rec=tc/n_gt; prec=tc/(tc+fc+1e-8); ap=0.0
    for t in np.linspace(0,1,pts):
        m=rec>=t
        if m.any(): ap+=float(prec[m].max())
    return ap/pts

for fp in [r'C:/Users/张三/Desktop/实验数据/probe_bidir_cache/rev_coco2dior_ship_off20_n100.pkl',
           r'C:/Users/张三/Desktop/实验数据/probe_bidir_cache/rev_coco2dior_vehicle_off20_n100.pkl',
           r'C:/Users/张三/Desktop/实验数据/probe_bidir_cache/rev_coco2dior_airplane_off20_n100.pkl',
           r'C:/Users/张三/Desktop/实验数据/probe_bidir_cache/fwd_dior2coco_car_off20_n100.pkl']:
    b=pickle.load(open(fp,'rb')); pi,F=b['per_img'],b['F']
    rng=np.random.RandomState(0); W=rng.normal(size=512); W=W/np.linalg.norm(W)
    # 官方
    rows=[]
    for p in pi:
        if len(p['gd']):
            w=F[p['crop_idx']]@W; s=np.sqrt(p['gd']*np.clip(np.maximum(0.0,w),1e-6,1.0))
        else: s=np.zeros(0)
        rows.append((p['box'],s))
    off=Met.compute_ap50_img(rows,[p['gt'] for p in pi])
    # 我的
    sb=[]
    for p in pi:
        if len(p['gd']):
            w=F[p['crop_idx']]@W; sb.append(np.sqrt(p['gd']*np.clip(np.maximum(0.0,w),1e-6,1.0)))
        else: sb.append(np.zeros(0))
    evs,ngt=per_image_evals([p['box'] for p in pi],sb,[p['gt'] for p in pi],0.5)
    mine=ap_from_evals(evs,sum(ngt),11)
    print(f"{os.path.basename(fp):<45} official={off:.10f} mine={mine:.10f} diff={abs(off-mine):.2e}")
