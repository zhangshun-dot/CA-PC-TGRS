# -*- coding: utf-8 -*-
"""phaseC_v2.py — 修正训练塌缩后重测条件感知原型对齐

phaseC_full 结果: 所有 z_k 自适应 ≤ baseline, 且 shuffled==true(完全相同) → 但可见
"塌缩"伪象: 训练后 ship→1.0 / vehicle→0.0(多数类塌缩)。这可能是训练退化(类别不平衡+强ΔP),
未必证明"条件无用"。故本版修正后重测:
  - 类别加权 CE (抑制多数类塌缩)
  - λ_reg 扫描 {1e-3, 1e-1} (限制/放开 ΔP)
  - 步数 120 (防过拟合)
  - 省略 mAP(18图太噪且慢), 只看 region_acc(179区域, 主指标)
判定: 若修正后 z_k 组 > baseline 且 shuffled 明显更差 → 条件真在驱动 → 上全量;
      若仍 shuffled==true 且 ≤ baseline → 确认条件原型对齐无效, 收口 A+B。
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import json
import numpy as np
import torch
import torch.nn.functional as F
import phaseC_full as C

DEVICE = 'cpu'


def train_v2(calib, protos, mode='add', per_class=False, lam=1e-2, zk_use='true',
             n_iter=120, lr=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(calib)); rng.shuffle(idx)
    nval = max(1, int(0.2 * len(calib)))
    val = [calib[i] for i in idx[:nval]]
    tr = [calib[i] for i in idx[nval:]]
    # 类别加权 (只对出现过的类; 未出现类权重不参与)
    cnt = {}
    for r in tr:
        cnt[r['label']] = cnt.get(r['label'], 0) + 1
    weights = {c: len(tr) / (len(cnt) * max(cnt.get(c, 1), 1)) for c in C.COCO_CLS}

    cond = C.CondEncoder().to(DEVICE)
    mod = C.Modulator(per_class=per_class, mode=mode).to(DEVICE)
    opt = torch.optim.AdamW(list(cond.parameters()) + list(mod.parameters()),
                            lr=lr, weight_decay=1e-3)
    zk_tr = [r['zk'] for r in tr]
    if zk_use == 'shuffle':
        zk_tr = np.random.RandomState(seed).permutation(zk_tr).tolist()
    for _ in range(n_iter):
        opt.zero_grad(); loss = 0.0; reg = 0.0
        for r, zk in zip(tr, zk_tr):
            f = torch.tensor(r['f']); hk = cond(torch.tensor([[zk]]))
            logits = []
            for ci in range(6):
                Pc = C.make_phat(protos, cond, mod, hk, ci, mode)[0]
                logits.append(f @ Pc)
            t = C.COCO_CLS.index(r['label'])
            loss += weights[r['label']] * F.cross_entropy(torch.stack(logits).unsqueeze(0),
                                                          torch.tensor([t]))
            if mode == 'add':
                d = mod(hk, t) if mod.per_class else mod(hk)
                reg += (d ** 2).sum()
        (loss + lam * reg).backward(); opt.step()
    vacc, _ = C.region_acc(val, protos, cond, mod, mode, zk_use='true')
    return cond, mod, vacc


def main():
    C.NET = C.P.V4Net().to(DEVICE)
    C.NET.load_state_dict(torch.load(C.V4_PATH, map_location=DEVICE))
    C.NET.eval()

    print('[1] COCO 原型', flush=True)
    protos = C.build_prototypes()
    print('[2] DIOR 划分', flush=True)
    calib, test_regs, test_images = C.load_dior_split()
    print(f'  calib={len(calib)} test={len(test_regs)}')

    base_acc, base_pca = C.region_acc(test_regs, protos, None, None, 'add', zk_use='none')
    out = {'baseline': {'region_acc': base_acc, 'per_class': base_pca}}
    print(f'\nbaseline(z_k=0): acc={base_acc:.3f} {base_pca}', flush=True)

    for lam in (1e-3, 1e-1):
        for zk_use in ('true', 'shuffle'):
            tag = f'add-global lam={lam:g} zk={zk_use}'
            print(f'\n=== {tag} ===', flush=True)
            cond, mod, vacc = train_v2(calib, protos, mode='add', per_class=False,
                                       lam=lam, zk_use=zk_use)
            acc, pca = C.region_acc(test_regs, protos, cond, mod, 'add', zk_use='true')
            out[tag] = {'region_acc': acc, 'per_class': pca, 'val_acc': vacc}
            print(f'  val_acc={vacc:.3f}  test_acc={acc:.3f}  {pca}', flush=True)

    with open(C.U.SAVE_DIR / 'phaseC_v2.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n[完成] -> 实验数据/phaseC_v2.json')


if __name__ == '__main__':
    main()
