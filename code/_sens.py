# -*- coding: utf-8 -*-
import os, sys, json
os.environ.setdefault('HF_HUB_OFFLINE','1')
sys.path.insert(0,r'C:/Users/张三/Desktop/实验代码'); sys.path.insert(0,r'C:/Users/张三/Desktop/A-CSPC')
import numpy as np, run_bidirectional as RB
VEC_KEYS=['full','par','perp','tge','vocab','cond']
D=json.load(open(r'C:/Users/张三/Desktop/实验数据/probe_dir_decomp.json',encoding='utf-8'))
all_names=sorted(set(RB.COCO_POOL+RB.DIOR_POOL))
E_photo=RB.text_embs([f'a photo of {c}' for c in all_names]); E_over=RB.text_embs([f'an overhead image of {c}' for c in all_names])
idx={c:i for i,c in enumerate(all_names)}
def unit(x):
    x=np.asarray(x,float);n=np.linalg.norm(x);return x/n if n>1e-12 else x
t_bar=unit(E_photo.mean(0)); Bmat=np.stack([E_photo[idx[c]]-t_bar for c in all_names],axis=1)
U,S,_=np.linalg.svd(Bmat,full_matrices=False); U=U[:,S>S[0]*1e-8]; P=U@U.T
def dirs(src,ev):
    dc=E_photo[idx[ev]]-E_over[idx[src]];par=P@dc;perp=dc-par
    return {'full':unit(dc),'par':unit(par) if np.linalg.norm(par)>1e-9 else np.zeros_like(dc),
            'perp':unit(perp) if np.linalg.norm(perp)>1e-9 else np.zeros_like(dc),
            'tge':unit(E_photo[idx[ev]]-t_bar),'vocab':unit(E_over[idx[ev]]-E_over[idx[src]]),
            'cond':unit(E_photo[idx[ev]]-E_over[idx[ev]])},float(np.linalg.norm(dc))
def pear(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    return float('nan') if x.std()<1e-12 or y.std()<1e-12 else float(np.corrcoef(x,y)[0,1])
for det,cells in D['per_det'].items():
    print('='*72); print(det)
    for drop_deg in (False,True):
        G=[];Y=[];names=[]
        for c in cells:
            if drop_deg and c['src']==c['eval']: continue
            ds,_=dirs(c['src'],c['eval']); G.append(np.stack([ds[k] for k in VEC_KEYS])); Y.append(np.array([c[k+'_star']-c['base'] for k in VEC_KEYS])); names.append(c['src']+'->'+c['eval'])
        G=np.stack(G);Y=np.stack(Y)
        tag='全部9 cell' if not drop_deg else '剔除 src==eval 的 %d 个对照 cell'%(9-len(G))
        print(f'  [{tag}] n_cell={len(G)}')
        for lab,w in (('class_axis t_photo(eval)-t_bar','tge'),('d_vocab','vocab'),('d_c// (类子空间)','par'),('d_c (full)','full'),('d_cond (纯观测)','cond'),('d_c^perp','perp')):
            W=G[:,VEC_KEYS.index(w),:]; cosv=np.einsum('cdi,ci->cd',G,W)
            Cc=cosv-cosv.mean(1,keepdims=True); Yc=Y-Y.mean(1,keepdims=True)
            hit=sum(1 for i in range(len(G)) if VEC_KEYS[int(np.argmax(cosv[i]))]==VEC_KEYS[int(np.argmax(Y[i]))])
            print(f'    ŵ={lab:<28} within-cell r={pear(Cc.reshape(-1),Yc.reshape(-1)):+.3f}  top1 {hit}/{len(G)}')
