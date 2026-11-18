# -*- coding: utf-8 -*-
"""rep_diag_gpu.py — 提取逐框表示向量, 算"表示质量"代理因子(选项 A)

阶段1(GPU, 慢, 可缓存): 对 11 场景逐框提取
  f  : DINOv2 L11 768-d 嵌入(即当前管线里唯一的"内容表示" z_c 代理)
  zk : CondEncoder 标量(条件强度 z_k)
  facs: 7 个融合因子(gd/vis/clip/rel/unc/cond/cond_inv)
阶段2(CPU, 快): 算每场景
  z_c_probe_accuracy  : 留一场景 kNN(11类) 探针精度 = 内容可分辨度
  effective_rank_zc   : per-scene 嵌入矩阵 SVD 参与比(有效秩, 坍缩→低)
  z_k_content_leakage : z_k 区分"自身类 vs 其他类"的 AUC(高=条件分支泄漏内容)
  unc_score/cond_inv_score : per-scene 因子均值
写入 factor_matrix_extra.json(合并已有 GT 列), 供 factor_importance.py 重跑。

诚实边界(务必写入报告): f 是 DINOv2 预训练特征, 不是我们训的 z_c 网络;
这些指标测的是"骨干判别力/条件估计纯度", 非"统一空间学到的表示质量"。
"""
import os, sys, json, pickle
import numpy as np
import torch
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from unified_eval import (get_gdino, get_dinov2, get_bank, get_clip, gdino_detect,
                          encode_dinov2, load_coco, load_dior,
                          PRE_FILTER, VEP_LAYER, DEVICE, SAVE_DIR)
import clip as clip_module
from torchvision import transforms

BASE = r'C:\Users\张三\Desktop\实验数据'
CACHE = os.path.join(BASE, 'rep_cache.pkl')
N_COCO, N_DIOR = 80, 50
SCEN = [
    ('COCO-airplane', 'coco', 'airplane', 'airplane', 'a photo of'),
    ('COCO-boat',     'coco', 'boat',     'ship',     'a photo of'),
    ('COCO-car',      'coco', 'car',      'vehicle',  'a photo of'),
    ('DIOR-airplane', 'dior', 'airplane', 'airplane', 'an overhead image of'),
    ('DIOR-ship',     'dior', 'ship',     'ship',     'an overhead image of'),
    ('DIOR-vehicle',  'dior', 'vehicle',  'vehicle',  'an overhead image of'),
    ('DIOR-harbor',   'dior', 'harbor',   'harbor',   'an overhead image of'),
    ('DIOR-storage',  'dior', 'storage tank', 'storage tank', 'an overhead image of'),
    ('DIOR-tennis',   'dior', 'tennis court', 'tennis court', 'an overhead image of'),
    ('DIOR-baseball', 'dior', 'baseball field', 'baseball field', 'an overhead image of'),
    ('DIOR-bridge',   'dior', 'bridge',   'bridge',   'an overhead image of'),
]


def prep_scene_with_emb(images, base_dets, dior_cls, clip_text_feat, cond_enc, clip_m, clip_pre):
    bank = get_bank()
    pos_proto = bank[dior_cls][VEP_LAYER]['prototype'].to(DEVICE)
    pos_proto = pos_proto / pos_proto.norm()
    others = []
    for c in bank:
        if c == dior_cls:
            continue
        p = bank[c][VEP_LAYER]['prototype'].to(DEVICE)
        others.append(p / p.norm())
    neg_protos = torch.stack(others)
    out = []
    for (boxes, scores), img in zip(base_dets, images):
        keep = scores > PRE_FILTER
        if not keep.any():
            continue
        boxes_f = boxes[keep]; scores_f = scores[keep]
        for i, box in enumerate(boxes_f):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crop = img.crop((x1, y1, x2, y2))
            f = encode_dinov2(crop, VEP_LAYER)
            cos_pos = float((f @ pos_proto).item())
            cos_neg = float((f @ neg_protos.T).max().item())
            vis = (cos_pos + 1) / 2
            rel = (1 - cos_neg) / 2
            imt = clip_pre(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                imf = clip_m.encode_image(imt).float()
            imf = imf / imf.norm(dim=-1, keepdim=True)
            clipv = float((imf @ clip_text_feat).item())
            clipv = (clipv + 1) / 2
            xa = transforms.ToTensor()(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                zk = float(cond_enc(xa).item())
            cond = torch.sigmoid(torch.tensor(zk)).item()
            out.append({'f': f.cpu().numpy().astype(np.float32), 'zk': zk,
                        'vis': vis, 'clip': clipv, 'rel': rel,
                        'unc': 1 - abs(cos_pos - clipv) / 2, 'cond': cond,
                        'cond_inv': 1 - cond, 'gd': float(scores_f[i])})
    return out


def eff_rank(M):
    M = M - M.mean(0, keepdims=True)
    sv = np.linalg.svd(M, compute_uv=False)
    sv = sv[sv > 1e-9]
    if len(sv) == 0:
        return 0.0
    p = sv / sv.sum()
    return float((p ** 2).sum() ** -1)  # participation ratio


def rankdata(a):
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a))
    ranks[order] = np.arange(1, len(a) + 1)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def auc_leak(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    r = rankdata(allv)
    n_pos, n_neg = len(pos), len(neg)
    return float((r[:n_pos].mean() - (n_pos + 1) / 2) / n_neg)


def main():
    if os.path.exists(CACHE):
        print('[cache] 命中 rep_cache.pkl, 跳过 GPU 提取', flush=True)
        data = pickle.load(open(CACHE, 'rb'))
    else:
        from single_scenario_factor_expand import load_cond
        cond_enc = load_cond()
        data = {}
        for name, dset, cls, dior, prefix in SCEN:
            if dset == 'coco':
                images, gts = load_coco(cls, n=N_COCO)
            else:
                images, gts = load_dior(cls, n=N_DIOR)
            base_dets = gdino_detect(images, f'{prefix} {cls}', 0.0)
            clip_m, clip_pre = get_clip()
            tok = clip_module.tokenize([f'{prefix} {cls}']).to(DEVICE)
            with torch.no_grad():
                txt = clip_m.encode_text(tok).float().squeeze(0)
            txt = txt / txt.norm()
            crops = prep_scene_with_emb(images, base_dets, dior, txt, cond_enc, clip_m, clip_pre)
            data[name] = {
                'emb': np.array([c['f'] for c in crops]),
                'zk': np.array([c['zk'] for c in crops]),
                'vis': np.array([c['vis'] for c in crops]),
                'clip': np.array([c['clip'] for c in crops]),
                'rel': np.array([c['rel'] for c in crops]),
                'unc': np.array([c['unc'] for c in crops]),
                'cond': np.array([c['cond'] for c in crops]),
                'cond_inv': np.array([c['cond_inv'] for c in crops]),
                'gd': np.array([c['gd'] for c in crops]),
                'n_crop': len(crops),
            }
            print(f'  {name}: n_crop={len(crops)} emb={data[name]["emb"].shape} '
                  f'zk_mean={data[name]["zk"].mean():.3f}', flush=True)
        pickle.dump(data, open(CACHE, 'wb'))
        print('[cache] 写出 rep_cache.pkl', flush=True)

    names = [s[0] for s in SCEN]
    # 语义标签(COCO↔DIOR 同类合并): DINOv2 是语义级特征, 域不变,
    # 用场景名(带域)当标签会让同语义跨域实例互判为异类 → 精度恒0(假象)。
    SEM = {'COCO-airplane': 'airplane', 'COCO-boat': 'ship', 'COCO-car': 'vehicle',
           'DIOR-airplane': 'airplane', 'DIOR-ship': 'ship', 'DIOR-vehicle': 'vehicle',
           'DIOR-harbor': 'harbor', 'DIOR-storage': 'storage tank', 'DIOR-tennis': 'tennis court',
           'DIOR-baseball': 'baseball field', 'DIOR-bridge': 'bridge'}
    uniq = {c: i for i, c in enumerate(sorted(set(SEM.values())))}
    # 全量 kNN 探针(留一框交叉验证): 每框由全体其他框的 k 近邻投票 → 该语义类的"内容可分辨度"
    E = np.concatenate([data[n]['emb'] for n in names], 0)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    L = np.concatenate([np.full(len(data[n]['emb']), uniq[SEM[n]]) for n in names], 0).astype(int)
    zc_probe, effrz, leak, unc_m, condinv_m = {}, {}, {}, {}, {}
    for si, s in enumerate(names):
        sem_si = uniq[SEM[s]]
        idxs = np.where(L == sem_si)[0]  # 该语义类全部框(COCO+DIOR)
        if len(idxs) == 0:
            zc_probe[s] = float('nan'); continue
        sim = E[idxs] @ E.T
        np.fill_diagonal(sim, -np.inf)  # 仅排除每个框自身(不排同类其他框)
        k = min(10, len(L) - len(idxs))
        knn = np.argsort(-sim, axis=1)[:, :k]
        pred = np.array([np.bincount(L[r]).argmax() for r in knn])
        zc_probe[s] = float((pred == sem_si).mean())
        effrz[s] = eff_rank(data[s]['emb'])
        leak[s] = auc_leak(data[s]['zk'], np.concatenate([data[t]['zk'] for t in names if t != s]))
        unc_m[s] = float(data[s]['unc'].mean())
        condinv_m[s] = float(data[s]['cond_inv'].mean())
        print(f'  {s}: z_c_probe={zc_probe[s]:.3f} eff_rank={effrz[s]:.1f} '
              f'zk_leak_AUC={leak[s]:.3f} unc={unc_m[s]:.3f} cond_inv={condinv_m[s]:.3f}', flush=True)

    # 合并已有 GT 列
    extra_path = os.path.join(BASE, 'factor_matrix_extra.json')
    if os.path.exists(extra_path):
        ex = json.load(open(extra_path, encoding='utf-8'))
        cols = ex['columns']; dmap = ex['data']
    else:
        cols, dmap = [], {}
    new = {'z_c_probe_accuracy': zc_probe, 'effective_rank_zc': effrz,
           'z_k_content_leakage': leak, 'unc_score': unc_m, 'cond_inv_score': condinv_m}
    for c, d in new.items():
        if c not in cols:
            cols.append(c)
        dmap[c] = [round(float(d[n]), 4) for n in names]
    out = {'columns': cols, 'data': dmap}
    with open(extra_path, 'w', encoding='utf-8') as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print('\n写出 factor_matrix_extra.json 列:', cols, flush=True)
    # 备份指标
    json.dump({'z_c_probe_accuracy': zc_probe, 'effective_rank_zc': effrz,
               'z_k_content_leakage': leak, 'unc_score': unc_m, 'cond_inv_score': condinv_m,
               'n_crop': {n: int(data[n]['n_crop']) for n in names}},
              open(os.path.join(BASE, 'rep_diag_gpu_metrics.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
