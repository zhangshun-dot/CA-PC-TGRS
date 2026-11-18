# -*- coding: utf-8 -*-
"""gen_qual_panels.py — 4 面板遥感定性图（dense vehicle / small car / ship / failure）

口径（与论文 §sec:exp-setup 的 base 定义严格一致）：
  base  s_b(x) = sqrt(q * max(0, F(x)@W_base)),  W_base = unit(P_src) - λ P_neg   (β=0)
  CA-PC s_c(x) = sqrt(q * max(0, F(x)@W_ca  )),  W_ca   = unit(P_src + β dc) - λ P_neg,
        dc = E_photo[target] - E_over[source]   （规范锚定角色固定，不随方向翻转）
  —— 左列不再画检测器原始置信度 q（旧 gen_qualitative.py 的口径错配已修正）。

数据（GDINO-tiny 检测 blob，来自 eval_v2_images 数据根，off=20, n=15）：
  probe_bidir_cache/qual_{car,truck,bus}_off20_n15.pkl  ← 源 DIOR 'vehicle'
  probe_bidir_cache/qual_boat_off20_n15.pkl              ← 源 DIOR 'ship'

用法:
  python gen_qual_panels.py stats    → 候选盘点
  python gen_qual_panels.py render   → 选取 + 渲染 figs/qual_panels.pdf
env: C:/Users/张三/GroundingDINO/venv/Scripts/python.exe
"""
import os, sys, pickle, json

# ⚠️ 必须在 import acspc.config 之前设定数据根（dataclass 默认值在 import 时绑定）
os.environ['AC_SPC_DIOR_IMG'] = r'C:\Users\张三\modeling\dior_data\eval_v2_images'
os.environ['AC_SPC_DIOR_PARQUET'] = r'C:\Users\张三\modeling\dior_data\dior_combined.parquet'
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cpu')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import numpy as np

sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import run_bidirectional as RB
from acspc import prototype as Proto

CACHE = r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache'
OUTPKL = r'C:\Users\张三\Desktop\实验数据\qual_panels.pkl'
FIG_DIR = r'C:\Users\张三\WorkBuddy\2026-09-23-16-06-04\figs'
TOPK = 3
OFF, N = 20, 15

# eval_cls -> (blob 文件名, 源类, 源域, 目标域)
SRC = {
    'car':   ('qual_car_off20_n15.pkl',   'vehicle', 'dior', 'coco'),
    'truck': ('qual_truck_off20_n15.pkl', 'vehicle', 'dior', 'coco'),
    'bus':   ('qual_bus_off20_n15.pkl',   'vehicle', 'dior', 'coco'),
    'boat':  ('qual_boat_off20_n15.pkl',  'ship',    'dior', 'coco'),
}


def unit(x):
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def iou_one(a, g):
    ix1, iy1 = max(a[0], g[0]), max(a[1], g[1])
    ix2, iy2 = min(a[2], g[2]), min(a[3], g[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    un = (a[2] - a[0]) * (a[3] - a[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter
    return inter / un if un > 1e-12 else 0.0


def top1_iou(boxes, scores, gt):
    if len(boxes) == 0 or len(gt) == 0:
        return 0.0
    bi = int(np.argmax(scores))
    return max(iou_one(boxes[bi], g) for g in gt)


def best_iou_any(boxes, scores, gt, k=TOPK):
    if len(boxes) == 0 or len(gt) == 0:
        return 0.0
    order = np.argsort(-scores)[:k]
    return max(iou_one(boxes[bi], g) for bi in order for g in gt)


def build_bank(cfg):
    import torch
    bank = {}
    for dom, pool in (('coco', RB.COCO_POOL), ('dior', RB.DIOR_POOL)):
        b = {}
        for cls in pool:
            ims, gts = RB.load_split(cfg, dom, cls, 0, RB.N_PROTO)
            crops = RB.D.gt_crops(ims, gts)
            if not crops:
                continue
            Ft = RB.clip_encode_imgs(crops)
            pt = torch.from_numpy(np.asarray(Ft.mean(0)))
            b[cls] = pt / pt.norm()
        bank[dom] = b
    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    Ep = RB.text_embs([f'a photo of {c}' for c in all_names])
    Eo = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    return bank, {c: Ep[i] for i, c in enumerate(all_names)}, {c: Eo[i] for i, c in enumerate(all_names)}


def collect(cfg):
    bank, Ep_d, Eo_d = build_bank(cfg)
    recs = []
    for eval_cls, (fn, src_cls, sdom, tdom) in SRC.items():
        blob = pickle.load(open(os.path.join(CACHE, fn), 'rb'))
        per_img, F = blob['per_img'], blob['F']
        images, gts = RB.load_split(cfg, sdom, src_cls, OFF, N)
        assert len(images) == len(per_img), (eval_cls, len(images), len(per_img))
        bad = 0
        for i, p in enumerate(per_img):
            if len(p['gt']) and not np.allclose(p['gt'], gts[i], atol=1e-3):
                bad += 1
        print(f'[{eval_cls:5s}] images={len(images)} GT 失配={bad}', flush=True)
        assert bad == 0, f'{eval_cls} GT 顺序失配 {bad}'
        P_src = unit(bank[sdom][src_cls].numpy())
        P_neg = unit(Proto.get_negative_prototype(bank[sdom], src_cls, RB.NEG_TOPK).numpy())
        dc = Ep_d[eval_cls] - Eo_d[src_cls]
        Wb = P_src - RB.LAM * P_neg
        Wc = unit(P_src + RB.BETA * dc) - RB.LAM * P_neg
        for i, p in enumerate(per_img):
            box, gd, gt, ci = p['box'], p['gd'], p['gt'], p['crop_idx']
            if len(gd):
                sb = np.sqrt(gd * np.clip(np.maximum(0.0, F[ci] @ Wb), 1e-6, 1.0))
                sc = np.sqrt(gd * np.clip(np.maximum(0.0, F[ci] @ Wc), 1e-6, 1.0))
            else:
                sb = sc = np.zeros(0)
            t1b, t1c = top1_iou(box, sb, gt), top1_iou(box, sc, gt)
            kb, kc = best_iou_any(box, sb, gt), best_iou_any(box, sc, gt)
            # 全框集（不限 top-k）下的最优 IoU：自证 CA-PC 只是重排序，未新增框
            mall = 0.0
            if len(box) and len(gt):
                mall = max(iou_one(b, g) for b in box for g in gt)
            areas = [(g[2] - g[0]) * (g[3] - g[1]) for g in gt]
            nmiss = sum(1 for g in gt if not len(box) or max(iou_one(b, g) for b in box) < 0.5)
            tag = ('success' if (kc > kb and kc >= 0.5) else
                   'miss' if (kb < 0.5 and kc < 0.5) else 'neutral')
            recs.append(dict(eval_cls=eval_cls, src_cls=src_cls, idx=i, img=images[i],
                             gt=gt, box=box, sb=sb, sc=sc, t1b=t1b, t1c=t1c, kb=kb, kc=kc,
                             n_gt=len(gt), n_box=len(box),
                             mall=mall,
                             med_area=float(np.median(areas)) if areas else 0.0,
                             nmiss=nmiss, tag=tag))
        sub = [r for r in recs if r['eval_cls'] == eval_cls]
        print(f'        tag: succ={sum(r["tag"]=="success" for r in sub)} '
              f'miss={sum(r["tag"]=="miss" for r in sub)} '
              f'neutral={sum(r["tag"]=="neutral" for r in sub)}', flush=True)
    return recs


def stats(recs):
    print('\n=== 全表 (每类 15 图) ===')
    for ec in SRC:
        sub = sorted([r for r in recs if r['eval_cls'] == ec], key=lambda r: r['idx'])
        print(f'\n-- {ec} --')
        print('  idx n_gt n_box med_area  t1b  t1c   kb   kc  maxIoU_all  tag')
        for r in sub:
            print(f'  {r["idx"]:3d} {r["n_gt"]:4d} {r["n_box"]:5d} {r["med_area"]:8.0f} '
                  f'{r["t1b"]:.2f} {r["t1c"]:.2f} {r["kb"]:.2f} {r["kc"]:.2f}  {r["mall"]:.2f}  {r["tag"]}')

    print('\n=== 候选盘点 ===')
    for ec in SRC:
        sub = [r for r in recs if r['eval_cls'] == ec]
        print(f'\n-- {ec} (n={len(sub)}) --')
        dn = sorted(sub, key=lambda r: -r['n_gt'])[:4]
        print('  densest(n_gt):', [(r['idx'], r['n_gt'], r['tag'], round(r['t1b'], 2), round(r['t1c'], 2)) for r in dn])
        sm = sorted(sub, key=lambda r: r['med_area'])[:4]
        print('  smallest(med_area):', [(r['idx'], round(r['med_area']), r['n_gt'], r['tag'], round(r['t1b'], 2), round(r['t1c'], 2)) for r in sm])
        su = sorted([r for r in sub if r['tag'] == 'success'], key=lambda r: -(r['kc'] - r['kb']))[:4]
        print('  success(best ΔIoU):', [(r['idx'], r['n_gt'], round(r['t1b'], 2), round(r['t1c'], 2)) for r in su])
        ms = sorted([r for r in sub if r['tag'] == 'miss'], key=lambda r: -r['nmiss'])[:4]
        print('  miss(nmiss):', [(r['idx'], r['n_gt'], r['nmiss'], round(r['t1b'], 2)) for r in ms])


def choose(recs):
    """判据 = top-1 框 IoU 提升 (t1c - t1b)，即 CA-PC 把更高 IoU 的既有框提到首位。"""
    def promo(r):
        return r['t1c'] - r['t1b']
    veh = [r for r in recs if r['eval_cls'] in ('car', 'truck', 'bus')]
    car = [r for r in recs if r['eval_cls'] == 'car']
    boat = [r for r in recs if r['eval_cls'] == 'boat']
    # (a) dense vehicle: 目标数最多的「有效提升」车辆场景
    cand = [r for r in veh if r['t1c'] >= 0.5 and promo(r) > 0]
    dense = max(cand, key=lambda r: (r['n_gt'], promo(r))) if cand else max(veh, key=lambda r: r['n_gt'])
    # (b) small car: base top-3 无框达标而 CA-PC 把达标框纳入其中，且中位目标面积最小
    cand = [r for r in car if r['kb'] < 0.5 and r['kc'] >= 0.5]
    small = min(cand, key=lambda r: r['med_area']) if cand else min(car, key=lambda r: r['med_area'])
    # (c) ship: boat 中「有效提升」且目标数最多者
    cand = [r for r in boat if r['t1c'] >= 0.5 and promo(r) > 0]
    ship = max(cand, key=lambda r: (r['n_gt'], promo(r))) if cand else max(boat, key=lambda r: r['n_gt'])
    # (d) failure: 车辆场景漏检，且"任何候选框离 GT 都最远"（无可提升对象）
    cand = [r for r in veh if r['tag'] == 'miss' and r['n_gt'] >= 5]
    fail = min(cand, key=lambda r: (r['mall'], -r['nmiss'])) if cand else None
    return [('Dense overhead vehicles', dense), ('Small overhead cars', small),
            ('Ships', ship), ('Recall collapse', fail)]


def roi_of(r):
    """每个面板的兴趣区：优先锚在「被校正提到首位」的框；其次锚在任一达标框；再退回 GT 簇。"""
    W, H = r['img'].size
    if r['t1c'] >= 0.5:
        box = r['box'][int(np.argmax(r['sc']))]
    elif r['t1b'] >= 0.5:
        box = r['box'][int(np.argmax(r['sb']))]
    elif r['mall'] >= 0.5:
        cands = [(iou_one(r['box'][bi], g), tuple(r['box'][bi]))
                 for S in (r['sb'], r['sc']) for bi in np.argsort(-S)[:TOPK] for g in r['gt']]
        box = list(max(cands)[1])
    else:
        box = [min(g[0] for g in r['gt']), min(g[1] for g in r['gt']),
               max(g[2] for g in r['gt']), max(g[3] for g in r['gt'])]
    x1, y1, x2, y2 = box
    pad = max(0.9 * max(x2 - x1, y2 - y1), 45)
    return (max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad))


def render(panels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    panels = [(t, r) for t, r in panels if r is not None]
    fig, axes = plt.subplots(len(panels), 2, figsize=(8.8, 2.35 * len(panels)))
    if len(panels) == 1:
        axes = np.array([axes])
    cmap = {0: '#1f77b4', 1: '#ff7f0e', 2: '#9467bd'}

    def box_iou(b, gt):
        return max((iou_one(b, g) for g in gt), default=0.0)

    def draw(ax, r, scores, title, inset):
        W, H = r['img'].size
        x1, y1, x2, y2 = roi_of(r)
        ax.imshow(r['img'])
        ax.set_xlim(x1, x2); ax.set_ylim(y2, y1)
        ax.set_title(title, fontsize=7.8); ax.axis('off')
        ax.add_patch(Rectangle((0, 0), W, H, fill=False, edgecolor='#aaaaaa', lw=0.7))
        for rank, bi in enumerate(np.argsort(-scores)[:TOPK]):
            bx, by, bx2, by2 = r['box'][bi]
            ax.add_patch(Rectangle((bx, by), bx2 - bx, by2 - by, fill=False,
                                   edgecolor=cmap[rank], lw=1.6))
        for g in r['gt']:
            ax.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False,
                                   edgecolor='#2ca02c', lw=1.3))
        if inset:
            ia = ax.inset_axes([0.67, 0.04, 0.32, 0.32])
            ia.imshow(r['img']); ia.axis('off')
            ia.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', lw=1.3))

    def iou3(r, scores):
        o = np.argsort(-scores)[:TOPK] if len(r['box']) else []
        return ' / '.join(f'{box_iou(r["box"][bi], r["gt"]):.2f}' for bi in o) or '--'

    for k, (name, r) in enumerate(panels):
        letter = 'abcd'[k]
        info = (f'$n_{{obj}}$={r["n_gt"]}  $n_{{prop}}$={r["n_box"]}  '
                f'median {r["med_area"]:.0f} px$^2$  best-proposal IoU {r["mall"]:.2f}')
        draw(axes[k, 0], r, r['sb'],
             f'({letter}) {name} — base\n1st/2nd/3rd IoU: {iou3(r, r["sb"])}\n{info}', True)
        draw(axes[k, 1], r, r['sc'],
             f'({letter}) {name} — CA-PC\n1st/2nd/3rd IoU: {iou3(r, r["sc"])}\n{info}', False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor='none', edgecolor='#2ca02c', label='ground truth'),
                        Patch(facecolor='none', edgecolor=cmap[0], label='1st-ranked proposal (highest score)'),
                        Patch(facecolor='none', edgecolor=cmap[1], label='2nd-ranked proposal'),
                        Patch(facecolor='none', edgecolor=cmap[2], label='3rd-ranked proposal')],
               loc='lower center', ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'qual_panels.{ext}'), dpi=190, bbox_inches='tight')
    print('  -> figs/qual_panels.pdf / .png')


def main():
    cfg = RB.Config()
    cfg.model.device = 'cpu'
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'
    mode = sys.argv[1] if len(sys.argv) > 1 else 'stats'
    recs = collect(cfg)
    stats(recs)
    panels = choose(recs)
    print('\n=== 选取（判据: top-1 框 IoU 提升）===')
    for name, r in panels:
        if r is None:
            print(f'  - {name}: 无可用样本'); continue
        print(f'  - {name:26s} cls={r["eval_cls"]:5s} idx={r["idx"]:3d} n_gt={r["n_gt"]:2d} '
              f'n_box={r["n_box"]:3d} nmiss={r["nmiss"]:2d} tag={r["tag"]:8s} '
              f'top1IoU {r["t1b"]:.2f}->{r["t1c"]:.2f}  med_area={r["med_area"]:.0f}px2  '
              f'base全框集最优IoU={r["mall"]:.2f}')
    if mode == 'render':
        pickle.dump(recs, open(OUTPKL, 'wb'))
        print('\n=== 各面板 top-3 框明细 (框, IoU) ===')
        for name, r in panels:
            if r is None:
                continue
            for lab, s in (('base ', r['sb']), ('CA-PC', r['sc'])):
                if not len(r['box']):
                    print(f'  {name:26s} {lab}: 无框'); continue
                det = [f'#{int(bi)}(IoU {max(iou_one(r["box"][bi], g) for g in r["gt"]):.2f}, '
                       f's={s[bi]:.3f}, box=[{r["box"][bi][0]:.0f},{r["box"][bi][1]:.0f},'
                       f'{r["box"][bi][2]:.0f},{r["box"][bi][3]:.0f}])'
                       for bi in np.argsort(-s)[:TOPK]]
                print(f'  {name:26s} {lab}: ' + '  '.join(det))
            print(f'    GT([{r["gt"][0][0]:.0f},{r["gt"][0][1]:.0f},...] 共{len(r["gt"])}个)')
        render(panels)


if __name__ == '__main__':
    main()
