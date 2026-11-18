# -*- coding: utf-8 -*-
"""probe_dior_cond_detect.py — DIOR 退化域条件感知检测 head 探针 (a)

目的: 验"第一层条件 z_k 在训练时可被消费, 补退化域检测 headroom"。
战场: DIOR(遥感, GDINO 本弱, 退化更掉, headroom 大) + 全局退化(res/haze) — z_k 全局估计粒度匹配。
数据: DIOR 3类×N 图, 每图 clean + 退化(res k∈{2,3,4}, haze t=0.5) 版本; GT 框坐标同尺寸不变。
管线: 退化图 GDINO 候选框 → 框 crop encode_dinov2(768) 内容 + 全图 V4Net z_k 条件。
三档 MLP head(7类含背景, 仅输入差):
  baseline  : 768 region
  +zk       : 768 + z_k(标准化)
  +oracle   : 768 + 真实退化等级 onehot(5维: clean/res2/res3/res4/haze)
训练: clean+退化混合 region 样本(同 orig 分层 split); 评估 clean/deg 域 mAP + 正样本 region_acc。
指标: deg 域 mAP / region_acc 的 (档 - baseline) 增量。
判定: +zk > baseline(deg域) 可观 → z_k 直接补 headroom, 强正上全量; +zk≈base 且 oracle>base → 框架有效但 CondEncoder 弱; oracle≈base → 条件感知训练无用。
"""
import os, sys
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import unified_eval as U
import phaseB_aux_probe as P
import unified_space_v2 as US2
from PIL import Image

DEVICE = 'cpu'
DIOR_CLS = ['airplane', 'ship', 'vehicle']
DIOR2COCO = {'airplane': 'airplane', 'ship': 'boat', 'vehicle': 'car'}
COCO_CLS = [a for a, _ in P.COCO_MAP]
PROMPT = 'airplane . ship . car . truck . bus . motorcycle'
N_PER = 10
SEEDS = [0, 1, 2]
SAVE = U.SAVE_DIR / 'probe_dior_cond_detect.json'
DEG_SPECS = [('res', 2), ('res', 3), ('res', 4), ('haze', 0.5)]


def degrade_haze(pil, t, rng):
    arr = np.asarray(pil, float) / 255.0
    A = float(np.percentile(arr, 99))
    x = arr * t + A * (1 - t) + rng.normal(0, 0.01, arr.shape).astype(arr.dtype)
    return Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))


def match_label(box, gts):
    best = 0.0
    for g in gts:
        v = U.iou(box, g)
        if v > best:
            best = v
    return best >= 0.5


def build_data(net):
    rng = np.random.default_rng(0)
    all_vim, meta = [], []
    for dc in DIOR_CLS:
        imgs, gts = U.load_dior(dc, n=N_PER)
        for oi, (im, g) in enumerate(zip(imgs, gts)):
            versions = [('clean', im, [1, 0, 0, 0, 0])]
            for di, (dt, kv) in enumerate(DEG_SPECS):
                oh = [0, 0, 0, 0, 0]; oh[di + 1] = 1
                vim = US2.degrade_res(im, kv) if dt == 'res' else degrade_haze(im, kv, rng)
                versions.append((f'{dt}{kv}', vim, oh))
            for vname, vim, oh in versions:
                all_vim.append(vim); meta.append((dc, oi, vname, oh, g))
    print(f'  [编码] {len(all_vim)} 版本图, 批量提 z_k ...', flush=True)
    _, Zk_all = P.encode_full(net, all_vim, bs=16)
    entries = []
    for idx, (dc, oi, vname, oh, g) in enumerate(meta):
        vim = all_vim[idx]; zk = float(Zk_all[idx])
        dets = U.gdino_detect([vim], PROMPT, 0.0)[0]
        boxes, scores = dets
        keep = scores > U.PRE_FILTER
        boxes = np.asarray(boxes)[keep]; scores = scores[keep]
        f_list, labels = [], []
        for b in boxes:
            f = U.encode_dinov2(vim.crop(tuple(map(int, b))))
            f_list.append(f.cpu().numpy())
            labels.append(COCO_CLS.index(DIOR2COCO[dc]) if match_label(b, g) else 6)
        entries.append({'dc': dc, 'oi': oi, 'vname': vname, 'deg': oh, 'zk': zk,
                        'gts': g, 'boxes': boxes, 'f_list': f_list, 'labels': labels,
                        'isdeg': vname != 'clean'})
    return entries


class Head(nn.Module):
    def __init__(self, din, ncls=7):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Linear(128, ncls))

    def forward(self, x):
        return self.fc(x)


def build_X(f, zk, deg, mode, mu, sd):
    if mode == 'baseline':
        return f.astype(np.float32)
    if mode == '+zk':
        return np.concatenate([f, [(zk - mu) / sd]]).astype(np.float32)
    return np.concatenate([f, deg]).astype(np.float32)


def train_head(Xtr, ytr, seed, n_iter=300):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Head(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    xt = torch.tensor(Xtr, float); yt = torch.tensor(ytr, long)
    for _ in range(n_iter):
        opt.zero_grad(); F.cross_entropy(m(xt), yt).backward(); opt.step()
    return m


def eval_split(entries, head, mode, mu, sd):
    corr = tot = 0
    im_preds, im_gts = [], []
    for e in entries:
        if len(e['f_list']) == 0:
            im_preds.append((np.zeros((0, 4)), np.zeros((0, 6)))); im_gts.append(e['gts']); continue
        s6, lcorr, ltot = [], 0, 0
        for f, lab in zip(e['f_list'], e['labels']):
            X = torch.tensor(build_X(f, e['zk'], e['deg'], mode, mu, sd), float).unsqueeze(0)
            with torch.no_grad():
                logit = head(X)[0].numpy()
            s6.append(logit[:6])
            if lab < 6:
                ltot += 1
                if logit[:6].argmax() == lab:
                    lcorr += 1
        corr += lcorr; tot += ltot
        im_preds.append((np.array(e['boxes']), np.array(s6)))
        im_gts.append(e['gts'])
    aps = {}
    for c in DIOR_CLS:
        sci = COCO_CLS.index(DIOR2COCO[c])
        preds = [(b, s[:, sci]) for (b, s) in im_preds]
        aps[c] = U.compute_ap50_img(preds, im_gts)
    mAP = float(np.mean(list(aps.values())))
    racc = corr / tot if tot else float('nan')
    return mAP, racc


def main():
    print('=== DIOR 退化域条件感知检测 head 探针 (a) ===', flush=True)
    net = P.V4Net().to(DEVICE)
    net.load_state_dict(torch.load(P.V4_PATH, map_location=DEVICE))
    net.eval()
    entries = build_data(net)
    print(f'  样本图版本={len(entries)}', flush=True)

    uniq = sorted(set((e['dc'], e['oi']) for e in entries))
    rng = np.random.default_rng(1); rng.shuffle(uniq)
    ntr = int(0.7 * len(uniq)); tr_set = set(uniq[:ntr])
    tr_entries = [e for e in entries if (e['dc'], e['oi']) in tr_set]
    te_entries = [e for e in entries if (e['dc'], e['oi']) not in tr_set]
    te_clean = [e for e in te_entries if not e['isdeg']]
    te_deg = [e for e in te_entries if e['isdeg']]
    zk_all = np.array([e['zk'] for e in tr_entries for _ in e['f_list']])
    mu, sd = zk_all.mean(), zk_all.std() + 1e-8
    print(f'  train 图={len(tr_set)} test 图={len(uniq)-len(tr_set)} (clean={len(te_clean)} deg={len(te_deg)})', flush=True)

    results = {}
    for mode in ['baseline', '+zk', '+oracle']:
        cm, cra, dm, dra = [], [], [], []
        for seed in SEEDS:
            Xtr, ytr = [], []
            for e in tr_entries:
                for f, lab in zip(e['f_list'], e['labels']):
                    Xtr.append(build_X(f, e['zk'], e['deg'], mode, mu, sd)); ytr.append(lab)
            head = train_head(np.array(Xtr), np.array(ytr), seed)
            a1, r1 = eval_split(te_clean, head, mode, mu, sd)
            a2, r2 = eval_split(te_deg, head, mode, mu, sd)
            cm.append(a1); cra.append(r1); dm.append(a2); dra.append(r2)
            print(f'  {mode} seed{seed}: clean_mAP={a1:.3f} clean_racc={r1:.3f} | deg_mAP={a2:.3f} deg_racc={r2:.3f}', flush=True)
        results[mode] = {'clean_mAP': round(np.nanmean(cm), 3), 'clean_racc': round(np.nanmean(cra), 3),
                         'deg_mAP': round(np.nanmean(dm), 3), 'deg_racc': round(np.nanmean(dra), 3)}
    out = dict(results)
    out['delta_deg_mAP_zk_minus_base'] = round(results['+zk']['deg_mAP'] - results['baseline']['deg_mAP'], 3)
    out['delta_deg_mAP_oracle_minus_base'] = round(results['+oracle']['deg_mAP'] - results['baseline']['deg_mAP'], 3)
    out['delta_deg_racc_zk_minus_base'] = round(results['+zk']['deg_racc'] - results['baseline']['deg_racc'], 3)
    out['n_train_orig'] = len(tr_set); out['n_test_orig'] = len(uniq) - len(tr_set)
    out['note'] = ('退化域检测 headroom 增量 = deg 域 (档 - baseline); 内容=768dim DINOv2 region(强), '
                   '条件=全图 z_k(全局, 匹配全局退化)。zk>base 可观⇒强正; zk≈base&oracle>base⇒CondEncoder弱; oracle≈base⇒无用。')
    with open(SAVE, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n结果 -> {SAVE}')
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
