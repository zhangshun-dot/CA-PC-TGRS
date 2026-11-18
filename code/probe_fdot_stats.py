# -*- coding: utf-8 -*-
"""probe_fdot_stats.py — F·δ 到底在区分什么? (纯缓存分析, 不跑检测)

问题: 加偏移等价于给每个候选框加一项 β(F·δ)。那 F·δ 在区分什么?
  候选 1: 域(RS vs 自然) → RS 候选的 F·δ 应系统性更负
  候选 2: 前景/目标性    → TP 的 F·δ 应更高(AUC>0.5)
  候选 3: 框大小         → 与面积相关
  候选 4: 检测器置信度   → 与 gd 相关
区分清楚才能写"δ 的机制"。
"""
import os, sys, json, pickle
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

from pathlib import Path
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

CACHE_DIR = Path(r'C:\Users\张三\Desktop\实验数据\probe_dirmech_cache')
OUT = Path(r'C:\Users\张三\Desktop\实验数据\probe_fdot_stats.json')

TASKS = {
    'same_dior2dior': ('dior', ['airplane', 'ship', 'vehicle']),
    'same_coco2coco': ('coco', ['airplane', 'boat', 'car']),
    'rev_coco2dior': ('dior', ['airplane', 'ship', 'vehicle']),
    'fwd_dior2coco': ('coco', ['airplane', 'boat', 'car']),
}

CLIP_ID = 'openai/clip-vit-base-patch32'


def _as_vec(x):
    for a in ('pooler_output', 'image_embeds', 'text_embeds', 'last_hidden_state'):
        if hasattr(x, a) and getattr(x, a) is not None:
            return getattr(x, a)
    return x


def text_emb(t):
    m = CLIPModel.from_pretrained(CLIP_ID).eval()
    p = CLIPProcessor.from_pretrained(CLIP_ID)
    with torch.no_grad():
        f = _as_vec(m.get_text_features(**p(text=[t], return_tensors='pt',
                                              padding=True, truncation=True))).float()
    return (f / f.norm(dim=-1, keepdim=True)).numpy()[0]


def unit(x):
    return x / (np.linalg.norm(x) + 1e-12)


def iou_mat(b, g):
    if len(b) == 0 or len(g) == 0:
        return np.zeros((len(b), len(g)))
    ix1 = np.maximum(b[:, None, 0], g[None, :, 0]); iy1 = np.maximum(b[:, None, 1], g[None, :, 1])
    ix2 = np.minimum(b[:, None, 2], g[None, :, 2]); iy2 = np.minimum(b[:, None, 3], g[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / np.maximum(ab[:, None] + ag[None, :] - inter, 1e-12)


def roc_auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    o = np.argsort(s, kind='mergesort'); r = np.arange(1, len(s) + 1, dtype=float)
    pos = r[y[o] == 1]; npos, nneg = len(pos), len(y) - len(pos)
    if npos == 0 or nneg == 0:
        return 0.5
    return float((pos.sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def main():
    names = ['airplane', 'ship', 'vehicle', 'boat', 'car', 'truck', 'bus', 'motorcycle',
             'harbor', 'storage tank', 'bridge']
    dbar = np.mean([text_emb(f'a photo of {c}') - text_emb(f'an overhead image of {c}')
                    for c in names], axis=0)
    ubar = unit(dbar)
    # 任意图像的"中心性"参照: photo/overhead 两类模板的中心
    t_photo = np.mean([text_emb(f'a photo of {c}') for c in names], axis=0)
    t_over = np.mean([text_emb(f'an overhead image of {c}') for c in names], axis=0)

    out = {}
    for tname, (dom, classes) in TASKS.items():
        out[tname] = {}
        for cls in classes:
            p = CACHE_DIR / f'{tname}_{cls}_off20_n30.pkl'
            if not p.exists():
                continue
            b = pickle.load(open(p, 'rb'))
            per_img, F = b['per_img'], b['F']
            tp, idx, area, gd = [], [], [], []
            for pi in per_img:
                if len(pi['gd']) == 0:
                    continue
                m = iou_mat(pi['box'], pi['gt'])
                best = m.max(1) if m.size else np.zeros(len(pi['gd']))
                tp.append((best >= 0.5).astype(int))
                idx.append(pi['crop_idx'])
                a = (pi['box'][:, 2] - pi['box'][:, 0]) * (pi['box'][:, 3] - pi['box'][:, 1])
                area.append(a)
                gd.append(pi['gd'])
            tp = np.concatenate(tp); idx = np.concatenate(idx)
            area = np.concatenate(area); gd = np.concatenate(gd)
            Fc = F[idx]
            fd = Fc @ dbar
            fphoto = Fc @ unit(t_photo)
            fover = Fc @ unit(t_over)
            r = {
                'eval_domain': dom,
                'n': int(len(tp)), 'n_tp': int(tp.sum()),
                'Fdot_delta_mean': float(fd.mean()), 'Fdot_delta_std': float(fd.std()),
                'Fdot_delta_tp': float(fd[tp == 1].mean()) if tp.sum() else None,
                'Fdot_delta_fp': float(fd[tp == 0].mean()) if (1 - tp).sum() else None,
                'auc_Fdot_delta': roc_auc(tp, fd),
                'auc_photo_minus_over': roc_auc(tp, fphoto - fover),
                'auc_gd': roc_auc(tp, gd),
                'corr_fd_area': float(np.corrcoef(fd, area)[0, 1]),
                'corr_fd_gd': float(np.corrcoef(fd, gd)[0, 1]),
                'corr_area_gd': float(np.corrcoef(area, gd)[0, 1]),
                'corr_area_TP': float(np.corrcoef(area, tp.astype(float))[0, 1]),
                'auc_area': roc_auc(tp, area),
            }
            out[tname][cls] = r
            print(f'{tname:<16} {cls:<10} dom={dom:<4} '
                  f'F·δ mean={r["Fdot_delta_mean"]:+.4f}±{r["Fdot_delta_std"]:.4f} '
                  f'TP={r["Fdot_delta_tp"]:+.4f} FP={r["Fdot_delta_fp"]:+.4f} '
                  f'AUC(δ)={r["auc_Fdot_delta"]:.3f} AUC(area)={r["auc_area"]:.3f} '
                  f'corr(δ,area)={r["corr_fd_area"]:+.3f}')

    json.dump(out, open(OUT, 'w'), indent=2, ensure_ascii=False)

    print('\n===== 按评测域汇总 =====')
    for dom in ['dior', 'coco']:
        vals = [v for t in out.values() for v in t.values() if v['eval_domain'] == dom]
        if not vals:
            continue
        print(f'\n[评测域={dom}]  n_task={len(vals)}')
        for k in ['Fdot_delta_mean', 'Fdot_delta_std', 'auc_Fdot_delta', 'auc_area',
                  'corr_fd_area', 'corr_fd_gd', 'auc_gd']:
            print(f'  {k:<22s} {np.mean([v[k] for v in vals]):+.4f}')
    print(f'\n已保存: {OUT}')


if __name__ == '__main__':
    main()
