# -*- coding: utf-8 -*-
r"""R1#10: qualitative comparison of individual modules and their combination.

For a few target images, draw the same proposals rescored under five scoring
prototypes and show which boxes survive at the top:
  (a) base      W = unit(P_src) - lam*P_neg
  (b) +vocab    W = unit(P_src + beta*d_vocab) - lam*P_neg
  (c) +cond     W = unit(P_src + beta*d_cond)  - lam*P_neg
  (d) joint     W = unit(P_src + beta*d_joint) - lam*P_neg
  (e) joint+gate  per-box a_i = clip(A*R*g(q_i),0,4), W_i = unit(P_src + a_i*d_joint) - lam*P_neg

d_joint = e("a photo of tgt") - e("an overhead image of src")   (= cache Delta)
d_vocab = e("an overhead image of tgt") - e("an overhead image of src")
d_cond  = e("a photo of tgt") - e("an overhead image of tgt")

No detector is re-run; boxes, detector scores, GT and CLIP crop features all
come from confirm_split_cache.
"""
import os, sys, pickle, json
import numpy as np
import torch
import clip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

CACHE = r'C:\Users\张三\Desktop\实验数据\confirm_split_cache'
IMG_DIR = r'C:\Users\张三\modeling\coco_data\val2017'
OUT_FIG = r'C:\Users\张三\WorkBuddy\2026-09-23-16-06-04\figs\fig_module_qual.pdf'
OUT_PNG = r'C:\Users\张三\WorkBuddy\2026-09-23-16-06-04\figs\fig_module_qual.png'
CLIP_PATH = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
OUT_JSON = r'C:\Users\张三\WorkBuddy\2026-09-23-16-06-04\figs\fig_module_qual_numbers.json'

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
SRC = {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle',
       'truck': 'vehicle', 'bus': 'vehicle', 'motorcycle': 'vehicle'}
BETA, LAM = 2.0, 0.5
C0, S_GATE, MU = 0.2974, 3.0, 0.6187
A_SCALE = BETA / MU
TOPK = 4

_model = None
def get_model():
    global _model
    if _model is None:
        _model, _ = clip.load(CLIP_PATH, device=DEV)
        _model.eval()
    return _model

def emb(text):
    m = get_model()
    tok = clip.tokenize([text]).to(DEV)
    with torch.no_grad():
        f = m.encode_text(tok).float()
    return (f / f.norm(dim=-1, keepdim=True))[0].cpu().numpy().astype(np.float64)

def iou_mat(b, g):
    if len(b) == 0 or len(g) == 0:
        return np.zeros((len(b), len(g)))
    x1 = np.maximum(b[:, None, 0], g[None, :, 0]); y1 = np.maximum(b[:, None, 1], g[None, :, 1])
    x2 = np.minimum(b[:, None, 2], g[None, :, 2]); y2 = np.minimum(b[:, None, 3], g[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    a2 = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-8)

def score_and_match(box, gd, gt, crop_idx, F, W=None, Ws=None):
    """W: fixed weight; Ws: per-box (n,512). Returns sorted (scores, tp flags)."""
    if Ws is not None:
        sim = np.einsum('ij,ij->i', F[crop_idx], Ws)
    else:
        sim = F[crop_idx] @ W
    s = np.sqrt(np.clip(gd, 0, None) * np.clip(np.maximum(0.0, sim), 1e-6, 1.0))
    order = np.argsort(-s)
    tp = np.zeros(len(order), dtype=np.int8)
    if len(gt):
        IOU = iou_mat(box, gt)
        matched = np.zeros(len(gt), dtype=bool)
        for k, ii in enumerate(order):
            row = IOU[ii].copy(); row[matched] = -1.0
            j = int(np.argmax(row))
            if row[j] >= 0.5:
                tp[k] = 1; matched[j] = True
    return s[order], tp, order

def unit(x):
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x

def build_modules(blob):
    """Return dict name -> (fixed W or callable producing per-box Ws)."""
    P_src = blob['P_src'].astype(np.float64)
    P_neg = blob['P_neg'].astype(np.float64)
    d_joint = blob['Delta'].astype(np.float64)
    e_p_t = emb(f"a photo of {blob['cls']}")
    e_o_s = emb(f"an overhead image of {blob['src_cls']}")
    e_o_t = emb(f"an overhead image of {blob['cls']}")
    d_vocab = e_o_t - e_o_s
    d_cond = e_p_t - e_o_t
    cos_check = float(d_joint @ (e_p_t - e_o_s) / (np.linalg.norm(d_joint) * np.linalg.norm(e_p_t - e_o_s) + 1e-12))

    def W(d):
        return unit(P_src + BETA * d) - LAM * P_neg

    R = float(np.linalg.norm(d_joint))
    dhat = d_joint / (R + 1e-12)

    def gate_Ws(gd):
        g = np.clip(1.0 - np.tanh(S_GATE * (gd - C0)), 0.0, 1.0)
        a = np.clip(A_SCALE * R * g, 0.0, 4.0)
        # gamma * dhat = (R*a) * (d/R) = a * d  -> injection length is a*R, not a
        P = P_src[None, :] + a[:, None] * d_joint[None, :]
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        return P - LAM * P_neg[None, :]

    return {
        'base': (unit(P_src) - LAM * P_neg, None),
        'vocab': (W(d_vocab), None),
        'cond': (W(d_cond), None),
        'joint': (W(d_joint), None),
        'joint+gate': (None, gate_Ws),
    }, dict(d_vocab=d_vocab, d_cond=d_cond, d_joint=d_joint,
            R=R, cos_delta_check=cos_check,
            norm_vocab=float(np.linalg.norm(d_vocab)),
            norm_cond=float(np.linalg.norm(d_cond)))

def scan(blobs):
    """Per-image TP counts under each module; pick representative images."""
    rows = []
    for cls, blob in blobs.items():
        mods, info = build_modules(blob)
        n_gt_all = sum(len(p['gt']) for p in blob['per_img'])
        for i in range(10, len(blob['per_img'])):
            p = blob['per_img'][i]
            if len(p['gd']) == 0 or len(p['gt']) == 0:
                continue
            rec = dict(cls=cls, idx=i, img_id=int(blob['img_ids'][i]),
                       n_gt=len(p['gt']), n_box=len(p['gd']))
            for name, (W, Wf) in mods.items():
                Ws = Wf(p['gd']) if Wf is not None else None
                s, tp, order = score_and_match(p['box'], p['gd'], p['gt'], p['crop_idx'],
                                               blob['F'], W=W, Ws=Ws)
                keep = nms_pick(p['box'][order], s)
                rec[name + '_tp'] = int(sum(tp[r] for r in keep))
                rec[name + '_fp'] = int(len(keep) - sum(tp[r] for r in keep))
            rows.append(rec)
    return rows

def nms_pick(box, scores, thr=0.55, k=TOPK):
    """Greedy de-duplication so one object is not drawn four times."""
    order = np.argsort(-scores)
    keep = []
    for i in order:
        ok = True
        for j in keep:
            if iou_mat(box[i:i+1], box[j:j+1])[0, 0] > thr:
                ok = False; break
        if ok:
            keep.append(i)
        if len(keep) >= k:
            break
    return keep

def draw_sample(ax_row, cls, idx, blob, mods, names):
    p = blob['per_img'][idx]
    img_id = int(blob['img_ids'][idx])
    path = os.path.join(IMG_DIR, f'{img_id:012d}.jpg')
    img = np.array(Image.open(path).convert('RGB'))
    gt = p['gt']
    H, W = img.shape[:2]
    # crop only when the objects occupy a small part of the frame; otherwise keep
    # the full image so the scene stays recognisable
    x0, y0 = gt[:, 0].min(), gt[:, 1].min()
    x1, y1 = gt[:, 2].max(), gt[:, 3].max()
    frac = ((x1 - x0) * (y1 - y0)) / float(W * H)
    if frac < 0.08:
        mx, my = 0.45 * (x1 - x0), 0.45 * (y1 - y0)
        cx0, cy0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
        cx1, cy1 = min(W, int(x1 + mx)), min(H, int(y1 + my))
        if cx1 - cx0 < 60 or cy1 - cy0 < 60:
            cx0, cy0, cx1, cy1 = 0, 0, W, H
    else:
        cx0, cy0, cx1, cy1 = 0, 0, W, H

    for ax, name in zip(ax_row, names):
        ax.imshow(img)
        ax.set_xlim(cx0, cx1); ax.set_ylim(cy1, cy0)
        ax.set_aspect('auto')
        for g in gt:
            ax.add_patch(Rectangle((g[0], g[1]), g[2]-g[0], g[3]-g[1],
                                   fill=False, ec='#1a7f37', lw=1.4, ls=(0, (5, 2)), zorder=3))
        Wt, Wf = mods[name]
        Ws = Wf(p['gd']) if Wf is not None else None
        s, tp, order = score_and_match(p['box'], p['gd'], p['gt'], p['crop_idx'],
                                       blob['F'], W=Wt, Ws=Ws)
        keep = nms_pick(p['box'][order], s)
        for r in keep:
            oi = order[r]
            b = p['box'][oi]
            # skip proposals that blanket the frame: they hide the scene and carry no localisation information
            if (b[2] - b[0]) * (b[3] - b[1]) > 0.45 * W * H:
                continue
            col = '#1f4e9c' if tp[r] else '#c0392b'
            ax.add_patch(Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1],
                                   fill=False, ec=col, lw=1.7, zorder=4))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6); sp.set_color('#888888')
        ntp = int(sum(tp[r] for r in keep)); nfp = len(keep) - ntp
        ax.set_title(f'{name}\nTP {ntp} / FP {nfp}', fontsize=7.5, pad=2.5)
    return img_id

def main():
    blobs = {c: pickle.load(open(os.path.join(CACHE, c + '.pkl'), 'rb')) for c in CLASSES}
    rows = scan(blobs)
    # rank by (joint_tp - base_tp), require base has headroom
    def key(r):
        return r['joint_tp'] - r['base_tp']
    rows.sort(key=key, reverse=True)
    print('=== 方向范数 ===')
    for c in CLASSES:
        _, info = build_modules(blobs[c])
        print(f"  {c:<11} |vocab|={info['norm_vocab']:.3f} |cond|={info['norm_cond']:.3f} "
              f"|joint|={info['R']:.3f} cos(Delta,recomputed)={info['cos_delta_check']:.4f}")
    print('\n=== top positive candidates ===')
    for r in rows[:10]:
        print(r)
    # pick per-class best, then take two positives from different classes + one boundary
    by_cls = {}
    for r in rows:
        d = r['joint_tp'] - r['base_tp']
        if r['cls'] not in by_cls or d > by_cls[r['cls']][0]:
            by_cls[r['cls']] = (d, r)
    ranked_cls = sorted(by_cls.items(), key=lambda kv: kv[1][0], reverse=True)
    picked, used = [], set()
    # prefer informative positives: at least 3 GT and a 2-box swing
    for cls, (d, r) in ranked_cls:
        if d < 2 or r['n_gt'] < 3:
            continue
        picked.append(r); used.add(cls)
        if len(picked) == 3:
            break
    # relax if fewer than 3
    for cls, (d, r) in ranked_cls:
        if len(picked) >= 3:
            break
        if cls in used or d < 1:
            continue
        picked.append(r); used.add(cls)
    print('\n=== picked ===')
    for r in picked:
        print(f"  {r['cls']:<10} img={r['img_id']} n_gt={r['n_gt']} "
              f"base {r['base_tp']}TP/{r['base_fp']}FP  joint {r['joint_tp']}TP/{r['joint_fp']}FP  "
              f"gate {r['joint+gate_tp']}TP/{r['joint+gate_fp']}FP")

    names = ['base', 'vocab', 'cond', 'joint', 'joint+gate']
    fig, axes = plt.subplots(len(picked), len(names), figsize=(7.6, 2.25 * len(picked)))
    if len(picked) == 1:
        axes = axes[None, :]
    out = []
    for ax_row, r in zip(axes, picked):
        blob = blobs[r['cls']]
        mods, info = build_modules(blob)
        img_id = draw_sample(ax_row, r['cls'], r['idx'], blob, mods, names)
        ax_row[0].set_ylabel(r['cls'], fontsize=8.5, labelpad=2)
        out.append(dict(cls=r['cls'], img_id=img_id, n_gt=r['n_gt'],
                        **{k: r[k] for k in r if k.endswith(('_tp', '_fp'))}))
    handles = [Rectangle((0, 0), 1, 1, fill=False, ec='#1a7f37', lw=1.4, ls=(0, (5, 2))),
               Rectangle((0, 0), 1, 1, fill=False, ec='#1f4e9c', lw=1.7),
               Rectangle((0, 0), 1, 1, fill=False, ec='#c0392b', lw=1.7)]
    fig.legend(handles, ['ground truth', 'true positive', 'false positive'],
               loc='lower center', ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.005), handlelength=1.6)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.93, bottom=0.075,
                        wspace=0.05, hspace=0.30)
    fig.savefig(OUT_FIG, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    print('\nSaved:', OUT_FIG)
    json.dump(dict(config=dict(beta=BETA, lam=LAM, topk=TOPK, c0=C0, s=S_GATE, mu=MU),
                   samples=out), open(OUT_JSON, 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('Saved:', OUT_JSON)

if __name__ == '__main__':
    main()
