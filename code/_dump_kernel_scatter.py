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
    x=np.asarray(x,float); n=np.linalg.norm(x); return x/n if n>1e-12 else x
t_bar=unit(E_photo.mean(0)); Bmat=np.stack([E_photo[idx[c]]-t_bar for c in all_names],axis=1)
U,S,_=np.linalg.svd(Bmat,full_matrices=False); U=U[:,S>S[0]*1e-8]; P=U@U.T
def dirs(src,ev):
    dc=E_photo[idx[ev]]-E_over[idx[src]]; par=P@dc; perp=dc-par
    return {'full':unit(dc),'par':unit(par) if np.linalg.norm(par)>1e-9 else np.zeros_like(dc),
            'perp':unit(perp) if np.linalg.norm(perp)>1e-9 else np.zeros_like(dc),
            'tge':unit(E_photo[idx[ev]]-t_bar),'vocab':unit(E_over[idx[ev]]-E_over[idx[src]]),
            'cond':unit(E_photo[idx[ev]]-E_over[idx[ev]])}, float(np.linalg.norm(dc))
out={}
for det,cells in D['per_det'].items():
    G=[];Y=[]
    for c in cells:
        ds,_=dirs(c['src'],c['eval']); G.append(np.stack([ds[k] for k in VEC_KEYS])); Y.append(np.array([c[k+'_star']-c['base'] for k in VEC_KEYS]))
    G=np.stack(G);Y=np.stack(Y)
    pack={}
    for lab,w in (('class_axis','tge'),('d_c','full'),('obs_axis','cond')):
        W=G[:,VEC_KEYS.index(w),:]; cosv=np.einsum('cdi,ci->cd',G,W)
        Cc=cosv-cosv.mean(1,keepdims=True); Yc=Y-Y.mean(1,keepdims=True)
        pack[lab]={'pts':[[round(float(a),4),round(float(b),4)] for a,b in zip(Cc.reshape(-1),Yc.reshape(-1))],
                   'r':round(float(np.corrcoef(Cc.reshape(-1),Yc.reshape(-1))[0,1]),3)}
    out[det]=pack
json.dump(out,open(r'C:/Users/张三/Desktop/实验数据/kernel_chart_pts.json','w'),indent=1)
print(json.dumps({d:{k:out[d][k]['r'] for k in out[d]} for d in out}))
print('gdino class_axis pts sample:', out['gdino']['class_axis']['pts'][:3])
