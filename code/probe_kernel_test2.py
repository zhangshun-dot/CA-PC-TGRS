# -*- coding: utf-8 -*-
"""probe_kernel_test2.py — 核函数检验 (去 cell 效应版)

关键: 上一版 pooled corr=0.905 部分由 cell 间 headroom 差异驱动 (高 headroom 的 cell
      cos 与 gain 同时大)。这里做【cell 内去均值】(within-cell centering),
      剔除 headroom/cell 主效应, 只检验: 同一 cell 内, 方向与 ĝ 越对齐是否增益越大。
"""
import os, sys, json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, r'C:/Users/张三/Desktop/实验代码'); sys.path.insert(0, r'C:/Users/张三/Desktop/A-CSPC')
import numpy as np, run_bidirectional as RB

VEC_KEYS = ['full', 'par', 'perp', 'tge', 'vocab', 'cond']
D = json.load(open(r'C:/Users/张三/Desktop/实验数据/probe_dir_decomp.json', encoding='utf-8'))
all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
E_over  = RB.text_embs([f'an overhead image of {c}' for c in all_names])
idx = {c: i for i, c in enumerate(all_names)}
def unit(x):
    x = np.asarray(x, float); n = np.linalg.norm(x); return x / n if n > 1e-12 else x
t_bar = unit(E_photo.mean(0))
Bmat = np.stack([E_photo[idx[c]] - t_bar for c in all_names], axis=1)
U, S, _ = np.linalg.svd(Bmat, full_matrices=False); U = U[:, S > S[0]*1e-8]; P = U @ U.T
def dirs(src, ev):
    dc = E_photo[idx[ev]] - E_over[idx[src]]
    par = P @ dc; perp = dc - par
    return {'full':unit(dc),'par':unit(par) if np.linalg.norm(par)>1e-9 else np.zeros_like(dc),
            'perp':unit(perp) if np.linalg.norm(perp)>1e-9 else np.zeros_like(dc),
            'tge':unit(E_photo[idx[ev]]-t_bar),'vocab':unit(E_over[idx[ev]]-E_over[idx[src]]),
            'cond':unit(E_photo[idx[ev]]-E_over[idx[ev]])}, np.linalg.norm(dc)
def pear(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    return float('nan') if x.std()<1e-12 or y.std()<1e-12 else float(np.corrcoef(x,y)[0,1])

res = {}
for det, cells in D['per_det'].items():
    G=[]; Y=[]; H=[]; M=[]
    for c in cells:
        b=c['base']; ds,mag = dirs(c['src'],c['eval'])
        G.append(np.stack([ds[k] for k in VEC_KEYS]))
        Y.append(np.array([c[k+'_star']-b for k in VEC_KEYS]))
        H.append(1-b); M.append(mag)
    G=np.stack(G); Y=np.stack(Y); H=np.array(H); M=np.array(M)   # ncell x 6 x 512 / ncell x 6
    X=G.reshape(-1,512); yv=Y.reshape(-1)
    Usv,Ssv,Vt=np.linalg.svd(X,full_matrices=False); keep=Ssv>Ssv[0]*1e-8
    g_hat=unit(Vt[keep].T@((Usv[:,keep].T@yv)/Ssv[keep]))
    Cg = G@g_hat                                    # ncell x 6  <ĝ, v̂>
    # ---- within-cell 去均值 ----
    Cc = Cg - Cg.mean(1,keepdims=True); Yc = Y - Y.mean(1,keepdims=True)
    r_within = pear(Cc.reshape(-1), Yc.reshape(-1))
    # 逐 cell 相关
    rc = [pear(Cg[i],Y[i]) for i in range(len(cells))]
    # ---- 两因子模型: ΔAP ≈ H * k * <ĝ,v̂> ----
    L = H[:,None]*Cg; k = float(np.dot(L.reshape(-1),yv)/max(np.dot(L.reshape(-1),L.reshape(-1)),1e-12))
    pred = k*L.reshape(-1)
    r2 = 1-np.sum((yv-pred)**2)/max(np.sum((yv-yv.mean())**2),1e-12)
    # 对照: 只用 headroom (无方向)
    predH = float(np.dot(H[:,None].repeat(6,1).reshape(-1),yv)/max(np.dot(H[:,None].repeat(6,1).reshape(-1),H[:,None].repeat(6,1).reshape(-1)),1e-12))*H[:,None].repeat(6,1).reshape(-1)
    r2H = 1-np.sum((yv-predH)**2)/max(np.sum((yv-yv.mean())**2),1e-12)
    # 对照: 只用方向 (无 headroom)
    kD = float(np.dot(Cg.reshape(-1),yv)/max(np.dot(Cg.reshape(-1),Cg.reshape(-1)),1e-12))
    r2D = 1-np.sum((yv-kD*Cg.reshape(-1))**2)/max(np.sum((yv-yv.mean())**2),1e-12)
    res[det]=dict(r_within=float(r_within), rc=[float(v) for v in rc], k=k, r2=r2,
                  r2_headonly=float(r2H), r2_dironly=float(r2D),
                  cos_g={k2: float(Cg[:,i].mean()) for i,k2 in enumerate(VEC_KEYS)},
                  gain={k2: float(Y[:,i].mean()) for i,k2 in enumerate(VEC_KEYS)},
                  mean_head=float(H.mean()), ncell=len(cells))
    print(f"\n=== {det}  (n_cell={len(cells)}, 平均 headroom={H.mean():.3f}) ===")
    print(f"  ★ within-cell corr(⟨ĝ,v̂>, ΔAP) = {r_within:+.3f}   (n={Y.size}, 已剔除 cell/headroom 主效应)")
    print(f"  逐 cell corr: " + ' '.join(f'{v:+.2f}' for v in rc) + f"   (中位 {np.median(rc):+.3f}, 正号 {sum(1 for v in rc if v>0)}/{len(rc)})")
    print(f"  R²:  两因子 H·⟨ĝ,v̂⟩ = {r2:.3f} | 仅 headroom = {r2H:.3f} | 仅方向 = {r2D:.3f}")
    print(f"  方向       <ĝ,v̂>     增益")
    for i,k2 in enumerate(VEC_KEYS):
        print(f"  {k2:<8} {Cg[:,i].mean():>+7.3f}  {Y[:,i].mean():>+8.4f}")
np.save(r'C:/Users/张三/Desktop/实验数据/kernel_g_hat.npy', np.stack([res[d]['cos_g'][k] for d in res for k in VEC_KEYS]))
json.dump(res, open(r'C:/Users/张三/Desktop/实验数据/probe_kernel_test.json','w',encoding='utf-8'), indent=2, ensure_ascii=False)
print('\n→ probe_kernel_test.json')
