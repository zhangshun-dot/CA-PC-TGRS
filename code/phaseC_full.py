# -*- coding: utf-8 -*-
"""phaseC_full.py — 条件感知原型对齐 (用户选 A: CondEncoder+Modulator, 完整轮)

验证"统一空间 z_k 按 intended 方式用于原型层"能否改善跨域(DIOR)检测。
协议校正: 校准在 DIOR(目标域), 评估在 DIOR test (跨域真实价值); COCO 仅作原型源。

组件:
  CondEncoder: z_k(1) -> h_k(128) MLP
  Modulator:
    A additive: ΔP = MLP(h_k)->768 (per_class 或 global 共享); P_hat = norm(P_src + ΔP)
    B gate:    s = sigmoid(MLP(h_k))->1;                P_hat = norm(P_src * (1+s))
Scorer: cos(F_crop, P_hat[c])
Loss:   CE(region分类) + λ_reg||ΔP||²
Ablation: baseline(z_k=0) / z_k-add-global / z_k-add-perclass / z_k-gate-global / shuffled-z_k
Eval:    DIOR test 区域6-way分类acc + DIOR test 逐类AP50(mAP, GDINO框+原型打分)
成功: adaptive acc/mAP 显著 > baseline -> 潜力兑现, 进完整检测头; null -> 收口

环境: GroundingDINO/venv (CPU). 冻 backbone.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
sys_path = r'C:\Users\张三\Desktop\实验代码'
import sys
sys.path.insert(0, sys_path)

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import unified_eval as U
import phaseB_aux_probe as P

DEVICE = 'cpu'
V4_PATH = U.SAVE_DIR / 'unified_space_v4.pt'
COCO_CLS = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
DIOR_CLS = ['airplane', 'ship', 'vehicle']
DIOR2COCO = {'airplane': 'airplane', 'ship': 'boat', 'vehicle': 'car'}
N_PER = 30
SEED = 42
K_CALIB = 20          # DIOR 每类校准数 (其余10为 test)
LAM_REG = 1e-2


# ───────── 模型 ─────────
class CondEncoder(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, h), nn.ReLU(), nn.Linear(h, h))
    def forward(self, zk):
        return self.net(zk.unsqueeze(-1) if zk.dim() == 1 else zk)  # (B,h)


class Modulator(nn.Module):
    def __init__(self, h=128, out_dim=768, per_class=False, mode='add'):
        super().__init__()
        self.mode = mode
        self.per_class = per_class
        if per_class:
            self.delta = nn.ModuleList([nn.Linear(h, out_dim) for _ in range(len(COCO_CLS))])
        else:
            self.delta = nn.Linear(h, out_dim)
        if mode == 'gate':
            self.gate = nn.Linear(h, 1)
    def forward(self, hk, c_idx=None):
        if self.mode == 'add':
            if self.per_class:
                return self.delta[c_idx](hk)
            return self.delta(hk)
        else:  # gate
            return torch.sigmoid(self.gate(hk))  # (B,1)


def build_prototypes():
    protos = {}
    for c in COCO_CLS:
        imgs, gts = U.load_coco(c, n=N_PER)
        fs = [U.encode_dinov2(im.crop(tuple(map(int, b)))) for im, boxes in zip(imgs, gts) for b in boxes]
        fs = torch.stack(fs)
        protos[c] = fs.mean(0) / fs.mean(0).norm()
        print(f'  P[{c}] {len(fs)} crops')
    return protos


def load_dior_split():
    global NET
    rng = np.random.default_rng(SEED)
    calib, test_regs = [], []
    test_images = {dc: [] for dc in DIOR_CLS}
    for dc in DIOR_CLS:
        imgs, gts = U.load_dior(dc, n=N_PER)
        zc, zk = P.encode_full(NET, imgs, bs=16)
        idx = np.arange(len(imgs)); rng.shuffle(idx)
        n_tr = max(1, int(0.7 * len(imgs)))      # 保证 test 非空
        tr_set = set(idx[:n_tr].tolist())
        for i, (im, boxes) in enumerate(zip(imgs, gts)):
            if i in tr_set:
                for b in boxes:
                    f = U.encode_dinov2(im.crop(tuple(map(int, b))))
                    calib.append({'f': f.cpu().numpy(), 'zk': float(zk[i]),
                                  'label': DIOR2COCO[dc], 'dior_cls': dc})
            else:
                for b in boxes:
                    f = U.encode_dinov2(im.crop(tuple(map(int, b))))
                    test_regs.append({'f': f.cpu().numpy(), 'zk': float(zk[i]),
                                      'label': DIOR2COCO[dc], 'dior_cls': dc})
                test_images[dc].append((im, gts[i], float(zk[i])))
    return calib, test_regs, test_images


def make_phat(protos, cond_enc, mod, hk, c_idx, mode):
    Pc = protos[COCO_CLS[c_idx]]
    if mode == 'add':
        d = mod(hk, c_idx) if mod.per_class else mod(hk)
        return F.normalize(Pc + d, dim=-1)
    else:
        s = mod(hk)  # (B,1)
        return F.normalize((Pc * (1 + s)).reshape(1, -1), dim=-1)


def region_acc(test_regs, protos, cond_enc, mod, mode, zk_use='true'):
    Pmat = torch.stack([protos[c] for c in COCO_CLS])  # (6,768)
    corr = tot = 0
    per = {c: [0, 0] for c in DIOR_CLS}
    for r in test_regs:
        f = torch.tensor(r['f'])
        if zk_use == 'none' or mod is None:
            Pc = Pmat
        else:
            hk = cond_enc(torch.tensor([[r['zk']]]))
            Pc = torch.stack([make_phat(protos, cond_enc, mod, hk, ci, mode)[0] for ci in range(6)])
        logits = f @ Pc.T
        pred = COCO_CLS[int(logits.argmax())]
        tot += 1; per[r['dior_cls']][1] += 1
        if pred == r['label']:
            corr += 1; per[r['dior_cls']][0] += 1
    return corr / tot, {c: per[c][0]/per[c][1] for c in DIOR_CLS}


PROMPT = 'airplane . ship . car . truck . bus . motorcycle'


def dior_ap50(test_images, det_boxes, protos, cond_enc, mod, mode):
    """GDINO 框(预检测缓存) + 原型余弦打分 -> 逐类 AP50 (mAP)。

    test_images: {dc: [(im, g, zk), ...]}; det_boxes: {dc: [boxes_numpy, ...]} 与 test_images[dc] 对齐。
    检测只跑一次(调用方缓存), 此函数仅按 config 不同原型打分, 5 个 config 共享同批框。
    """
    aps = {}
    for dc in DIOR_CLS:
        preds, gt_c = [], []
        for (im, g, zk), boxes in zip(test_images[dc], det_boxes[dc]):
            if len(boxes) == 0:
                preds.append((np.zeros((0, 4)), [])); gt_c.append(g); continue
            sc = []
            for b in boxes:
                f = U.encode_dinov2(im.crop(tuple(map(int, b)))).unsqueeze(0)
                if mod is None:
                    logits = torch.stack([protos[COCO_CLS[ci]].unsqueeze(0) @ f.T for ci in range(6)])
                else:
                    hk = cond_enc(torch.tensor([[float(zk)]]))
                    logits = torch.stack([make_phat(protos, cond_enc, mod, hk, ci, mode)[0] @ f.T for ci in range(6)])
                sc.append(float(logits.max()))
            preds.append((np.array(boxes), sc))
            gt_c.append(g)
        aps[dc] = U.compute_ap50_img(preds, gt_c)
    return float(np.mean(list(aps.values()))), aps


def train(config, calib, protos):
    if config['zk_use'] == 'none' or config.get('mode') is None:
        return None, None
    cond_enc = CondEncoder().to(DEVICE)
    mod = Modulator(per_class=config.get('per_class', False), mode=config['mode']).to(DEVICE)
    opt = torch.optim.AdamW(list(cond_enc.parameters()) + list(mod.parameters()), lr=1e-4, weight_decay=1e-3)
    zk_order = [r['zk'] for r in calib]
    if config['zk_use'] == 'shuffle':
        zk_order = np.random.RandomState(0).permutation(zk_order).tolist()
    for _ in range(400):
        opt.zero_grad(); loss = 0.0; reg = 0.0
        for r, zk in zip(calib, zk_order):
            f = torch.tensor(r['f']); hk = cond_enc(torch.tensor([[zk]]))
            logits = []
            for ci in range(6):
                Pc = make_phat(protos, cond_enc, mod, hk, ci, config['mode'])[0]
                logits.append(f @ Pc)
            t = COCO_CLS.index(r['label'])
            loss += F.cross_entropy(torch.stack(logits).unsqueeze(0), torch.tensor([t]))
            if config['mode'] == 'add':
                d = mod(hk, t) if mod.per_class else mod(hk)
                reg += (d ** 2).sum()
        loss += LAM_REG * reg
        loss.backward(); opt.step()
    return cond_enc, mod


def main():
    global NET
    print('=' * 70); print('C 条件感知原型对齐 — 完整轮 (DIOR 校准/DIOR 测)'); print('=' * 70)
    global NET
    NET = P.V4Net().to(DEVICE); NET.load_state_dict(torch.load(V4_PATH, map_location=DEVICE)); NET.eval()

    print('[1] COCO 原型'); protos = build_prototypes()
    print('[2] DIOR 划分'); calib, test, test_images = load_dior_split()
    print(f'  calib={len(calib)} test={len(test)}')
    print('[2b] 预检测 GDINO 框(仅一次, 缓存复用)', flush=True)
    det_boxes = {dc: [U.gdino_detect([im], PROMPT, 0.0)[0][0] for (im, g, zk) in test_images[dc]]
                 for dc in DIOR_CLS}
    print(f'  test 图: ' + ', '.join(f'{dc}={len(test_images[dc])}' for dc in DIOR_CLS))

    configs = [
        {'name': 'baseline(z_k=0)', 'zk_use': 'none', 'mode': None},
        {'name': 'z_k-add-global', 'zk_use': 'true', 'mode': 'add', 'per_class': False},
        {'name': 'z_k-add-perclass', 'zk_use': 'true', 'mode': 'add', 'per_class': True},
        {'name': 'z_k-gate-global', 'zk_use': 'true', 'mode': 'gate', 'per_class': False},
        {'name': 'shuffled-z_k', 'zk_use': 'shuffle', 'mode': 'add', 'per_class': False},
    ]
    results = {}
    for cfg in configs:
        print(f'\n=== {cfg["name"]} ===', flush=True)
        cond_enc, mod = train(cfg, calib, protos)
        acc, pca = region_acc(test, protos, cond_enc, mod, cfg['mode'] if cfg['mode'] else 'add',
                              zk_use=('none' if cfg['mode'] is None else 'true'))
        # AP50 (仅对真 z_k 配置跑, 省时; baseline 也跑)
        mAP, aps = (None, None)
        if cfg['zk_use'] in ('none', 'true'):
            try:
                mAP, aps = dior_ap50(test_images, det_boxes, protos, cond_enc, mod, cfg['mode'] if cfg['mode'] else 'add')
            except Exception as e:
                print(f'  AP50 跳过: {e}')
        results[cfg['name']] = {'region_acc': acc, 'per_class_acc': pca, 'mAP': mAP, 'ap_per_class': aps}
        print(f'  region_acc={acc:.3f}  mAP={mAP}')

    out = {'configs': results, 'k_calib': K_CALIB, 'note': 'DIOR校准/DIOR测; adaptive>baseline=潜力兑现'}
    with open(U.SAVE_DIR / 'phaseC_full.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n[完成] -> 实验数据/phaseC_full.json')


if __name__ == '__main__':
    main()
