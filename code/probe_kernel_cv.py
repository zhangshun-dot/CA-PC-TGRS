# -*- coding: utf-8 -*-
"""probe_kernel_cv.py — 核函数【留一 cell 交叉验证】(去掉上文 R^2 的 in-sample 膨胀)

留一 cell: 用 8 个 cell 的 (方向,增益) 拟合 ĝ, 预测【没见过的第 9 个 cell】的 6 个方向增益。
判据: 折外 (out-of-fold) 预测 vs 观测的相关 —— 若仍高, 则 "增益 = <ĝ,v̂>" 是真规律而非拟合假象。
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
def fit_g(X,yv):
    Usv,Ssv,Vt=np.linalg.svd(X,full_matrices=False)
    keep=Ssv>Ssv[0]*1e-10
    return unit(Vt[keep].T@((Usv[:,keep].T@yv)/Ssv[keep]))
res={}
for det,cells in D['per_det'].items():
    G=[];Y=[]
    for c in cells:
        ds,_=dirs(c['src'],c['eval']); G.append(np.stack([ds[k] for k in VEC_KEYS])); Y.append(np.array([c[k+'_star']-c['base'] for k in VEC_KEYS]))
    G=np.stack(G);Y=np.stack(Y)   # 9 x 6 x 512, 9 x 6
    n=len(cells); oof_x=[];oof_y=[];rank=[]
    for i in range(n):
        tr=[j for j in range(n) if j!=i]
        X=G[tr].reshape(-1,512); yv=Y[tr].reshape(-1)
        g=fit_g(X,yv)
        pred=G[i]@g; obs=Y[i]
        # 逐 cell 单参数标定 (斜率用训练集估计的 g 的尺度, 这里用全局训练斜率)
        k=float(np.dot(X@g,yv)/max(np.dot(X@g,X@g),1e-12))
        oof_x+=[round(float(v),5) for v in pred]; oof_y+=[round(float(v),5) for v in obs]
        rank.append((pear(pred,obs), c['src']+'->'+c['eval']))
    res[det]=dict(oof_corr=pear(oof_x,oof_y), per_cell=rank,
                  n_cell=n, k=k)
    pos=sum(1 for r,_ in rank if r>0)
    print(f"\n=== {det} ===")
    print(f"  ★ 留一 cell 折外相关 (预测 vs 观测, n={len(oof_x)}) = {pear(oof_x,oof_y):+.3f}")
    print(f"  逐折 (留出的 cell) 相关: " + '  '.join(f'{r:+.2f}[{c}]' for r,c in rank))
    print(f"  折内正号 {pos}/{n}")
    # 符号命中: 预测 top1 方向 == 观测 top1 方向
    hit=0
    for i in range(n):
        tr=[j for j in range(n) if j!=i]
        g=fit_g(G[tr].reshape(-1,512), Y[tr].reshape(-1))
        if VEC_KEYS[int(np.argmax(G[i]@g))]==VEC_KEYS[int(np.argmax(Y[i]))]: hit+=1
    print(f"  top-1 方向命中率 = {hit}/{n}")
    res[det]['top1_hit']=f"{hit}/{n}"
json.dump(res,open(r'C:/Users/张三/Desktop/实验数据/probe_kernel_cv.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
print('\n→ probe_kernel_cv.json')
