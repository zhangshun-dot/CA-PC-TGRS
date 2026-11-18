# -*- coding: utf-8 -*-
"""probe_kernel_zp.py — 零参数核: 增益 ∝ cos(v̂, ŵ) 其中 ŵ 是【固定公式给的】方向, 无逐cell拟合

候选核方向 (全部零目标GT可算):  d_c / t_photo(eval)-t_bar / d_vocab / d_cond
判据: 留一 cell (其实无参数可留, 等价于全局直线) 折外相关 + top-1 方向命中率。
"""
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
def pear(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    return float('nan') if x.std()<1e-12 or y.std()<1e-12 else float(np.corrcoef(x,y)[0,1])
CAND={'w=d_c (论文自身主张)':'full','w=t_photo(eval)-t_bar (TGE型)':'tge',
      'w=d_vocab':'vocab','w=d_cond':'cond','w=d_c// (类子空间)':'par'}
out={}
for det,cells in D['per_det'].items():
    G=[];Y=[]
    for c in cells:
        ds,_=dirs(c['src'],c['eval']); G.append(np.stack([ds[k] for k in VEC_KEYS])); Y.append(np.array([c[k+'_star']-c['base'] for k in VEC_KEYS]))
    G=np.stack(G);Y=np.stack(Y)
    print(f"\n=== {det} ===   (9 cell x 6 方向 = 54 点)")
    for label,w in CAND.items():
        W=G[:,VEC_KEYS.index(w),:] if w in VEC_KEYS else None
        # 固定核方向 = 该 cell 的该向量 (无拟合参数)
        W=G[:,VEC_KEYS.index(w),:]           # 9 x 512
        cosv=G@W[:,None,:].transpose(0,2,1)[...,0] if False else np.einsum('cdi,ci->cd',G,W)
        # 全局单斜率
        k=float(np.dot(cosv.reshape(-1),Y.reshape(-1))/max(np.dot(cosv.reshape(-1),cosv.reshape(-1)),1e-12))
        r=pear(cosv.reshape(-1),Y.reshape(-1))
        # 逐 cell top-1 命中
        hit=sum(1 for i in range(len(cells)) if VEC_KEYS[int(np.argmax(cosv[i]))]==VEC_KEYS[int(np.argmax(Y[i]))])
        # within-cell 去均值相关 (剔除 headroom)
        Cc=cosv-cosv.mean(1,keepdims=True); Yc=Y-Y.mean(1,keepdims=True)
        out.setdefault(det,{})[label]=dict(corr_pooled=float(r),corr_within=float(pear(Cc.reshape(-1),Yc.reshape(-1))),top1=f"{hit}/{len(cells)}",slope=k)
        print(f"  {label:<34} pooled r={r:+.3f} | within-cell r={pear(Cc.reshape(-1),Yc.reshape(-1)):+.3f} | top1 {hit}/{len(cells)}")
json.dump(out,open(r'C:/Users/张三/Desktop/实验数据/probe_kernel_zp.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
print('\n→ probe_kernel_zp.json')
