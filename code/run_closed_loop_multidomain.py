# -*- coding: utf-8 -*-
"""run_closed_loop_multidomain.py — #3 多域/数据域强度检验（第二源域对）

核心: 复用 COCO 目标域评测缓存 probe_bidir_cache（GDINO 在 COCO 上的检测 F + per_img），
只把"源原型"从 DIOR 换成 DOTA / NWPU，验证闭环在**不同源域对**上是否仍成立。
  - 源域 dior : airplane/ship/vehicle/truck/bus/motorcycle → COCO
  - 源域 dota : plane/ship/small-vehicle/large-vehicle → COCO（更高分辨率遥感）
  - 源域 nwpu : airplane/ship/vehicle → COCO
共用 3 类 airplane/boat/car 做三域同口径对比；其余类按各域可用范围报。
门控尺度(c0/PRIOR_SCALE)沿用 run_closed_loop_balanced.json（DIOR→COCO 零GT定出），即"尺度不随域对重调"。

用法:
  HF_HUB_OFFLINE=1 "C:/Users/张三/vg/Scripts/python.exe" -u run_closed_loop_multidomain.py
"""
import os, sys, json, pickle, glob, time, re
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
import numpy as np, torch
from PIL import Image
import run_bidirectional as RB
from pathlib import Path

RB.DEVICE = 'cuda'
DEVICE = 'cuda'
LAM = RB.LAM; NEG_TOPK = RB.NEG_TOPK; N_PROTO = RB.N_PROTO
COCO_CACHE = Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache')
bal = json.load(open(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced.json'))
c0 = bal['config']['gate']['c0']; PRIOR_SCALE = bal['config']['gate']['prior_scale']
OUT = Path(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_multidomain.json')
DOTA_ROOT = Path(r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\val')
NWPU_ROOT = Path(r'C:\Users\张三\modeling\nwpu_data\NWPU-VHR-10')

# (源域, 源类, COCO目标类, 提示词)  —— dior 源类仅 airplane/ship/vehicle(DIOR_POOL)，
# truck/bus/motorcycle 在 DIOR 侧均归为 vehicle（与 balanced 脚本 TASKS 一致）
UNITS = [
    ('dior', 'airplane', 'airplane', 'airplane'),
    ('dior', 'ship', 'boat', 'boat'),
    ('dior', 'vehicle', 'car', 'car'),
    ('dior', 'vehicle', 'truck', 'truck'),
    ('dior', 'vehicle', 'bus', 'bus'),
    ('dior', 'vehicle', 'motorcycle', 'motorcycle'),
    ('dota', 'plane', 'airplane', 'plane'),
    ('dota', 'ship', 'boat', 'ship'),
    ('dota', 'small-vehicle', 'car', 'small vehicle'),
    ('dota', 'large-vehicle', 'truck', 'large vehicle'),
    ('nwpu', '1', 'airplane', 'airplane'),
    ('nwpu', '2', 'boat', 'ship'),
    ('nwpu', '10', 'car', 'vehicle'),
]


def unit(x):
    x = np.asarray(x, float); n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def _sorted_num(files):
    return sorted(files, key=lambda p: int(re.sub(r'\D', '', p.stem) or 0))


def load_dota(cls, offset, n):
    imgdir = DOTA_ROOT / 'images'
    lbdir = DOTA_ROOT / 'labelTxt-v1.5' / 'DOTA-v1.5_val'
    if not lbdir.exists():
        cands = list((DOTA_ROOT / 'labelTxt-v1.5').iterdir())
        lbdir = cands[0] if cands else lbdir
    out = []
    for f in sorted(lbdir.glob('*.txt')):
        stem = f.stem
        ip = imgdir / f'{stem}.png'
        if not ip.exists():
            continue
        lines = [l.strip() for l in open(f, encoding='utf-8', errors='ignore') if l.strip()]
        boxes = []
        for l in lines:
            if l.startswith('imagesource') or l.startswith('gsd'):
                continue
            parts = l.split()
            if len(parts) < 9 or parts[8] != cls:
                continue
            try:
                coords = [float(v) for v in parts[:8]]
            except ValueError:
                continue
            xs, ys = coords[0::2], coords[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            if x2 - x1 > 4 and y2 - y1 > 4:
                boxes.append([x1, y1, x2, y2])
        if boxes:
            out.append((ip, boxes))
    sel = out[offset:offset + n]
    imgs = [Image.open(p).convert('RGB') for p, _ in sel]
    return imgs, [b for _, b in sel]


def load_nwpu(cls_idx, offset, n):
    gtdir = NWPU_ROOT / 'ground_truth'
    imgdir = NWPU_ROOT / 'positive_image_set'
    out = []
    for f in sorted(gtdir.glob('*.txt')):
        ip = imgdir / f'{f.stem}.jpg'
        if not ip.exists():
            continue
        boxes = []
        for l in open(f, encoding='utf-8', errors='ignore'):
            l = l.strip()
            if not l:
                continue
            m = re.match(r'\(([\d.]+),\s*([\d.]+)\),\s*\(([\d.]+),\s*([\d.]+)\),\s*(\d+)', l)
            if not m or int(m.group(5)) != int(cls_idx):
                continue
            x1, y1, x2, y2 = [float(m.group(i)) for i in range(1, 5)]
            if x2 - x1 > 4 and y2 - y1 > 4:
                boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        if boxes:
            out.append((ip, boxes))
    sel = out[offset:offset + n]
    imgs = [Image.open(p).convert('RGB') for p, _ in sel]
    return imgs, [b for _, b in sel]


def load_src_imgs(ds, cls, n):
    if ds == 'dior':
        ims, gts = RB.load_split(RB.Config(), 'dior', cls, 0, n)
        keep = [(i, g) for i, g in zip(ims, gts) if len(g) > 0][:n]
        return [k[0] for k in keep], [k[1] for k in keep]
    if ds == 'dota':
        return load_dota(cls, 0, n)
    if ds == 'nwpu':
        return load_nwpu(cls, 0, n)
    return [], []


def load_coco_cache(eval_cls):
    hits = glob.glob(str(COCO_CACHE / f'fwd_dior2coco_{eval_cls}_n*.pkl')) or \
           glob.glob(str(COCO_CACHE / f'fwd_dior2coco_{eval_cls}_off{N_PROTO}_n*.pkl'))
    return pickle.load(open(hits[0], 'rb')) if hits else None


def gate_conf(conf, c=c0, scale=3.0):
    return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))


def gate_count(n_pred, n0, n_iqr):
    return float(np.clip(1.0 / (1.0 + np.exp((n_pred - n0) / (n_iqr / 2 + 1e-9))), 0.0, 1.0))


def ap_of(per_img, F, W):
    rows = []
    for p in per_img:
        if len(p['gd']):
            w = F[p['crop_idx']] @ W
            s = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
        else:
            s = np.zeros(0)
        rows.append((p['box'], s))
    return RB.Met.compute_ap50_img(rows, [p['gt'] for p in per_img])


def main():
    t0 = time.time()
    cfg = RB.Config(); cfg.model.device = 'cpu'
    cfg.model.detection_threshold = 0.0; cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    COCO_EVAL = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
    E_photo = {c: RB.text_embs([f'a photo of {c}'])[0] for c in COCO_EVAL}
    E_over_src = {(ds, sc): RB.text_embs([f'an overhead image of {sc}'])[0]
                  for ds, sc, _, _ in UNITS}

    # ── 每源域原型 bank（CLIP GT crop）──
    src_banks = {'dior': {}, 'dota': {}, 'nwpu': {}}
    # DIOR
    for cls in RB.DIOR_POOL:
        ims, gts = RB.load_split(cfg, 'dior', cls, 0, N_PROTO)
        crops = RB.D.gt_crops(ims, gts)
        if not crops:
            continue
        Ft = RB.clip_encode_imgs(crops)
        src_banks['dior'][cls] = torch.from_numpy(Ft.mean(0)) / torch.from_numpy(Ft.mean(0)).norm()
    # DOTA / NWPU
    for ds in ('dota', 'nwpu'):
        for dsc in sorted({sc for d, sc, _, _ in UNITS if d == ds}):
            ims, gts = load_src_imgs(ds, dsc, N_PROTO)
            if not ims:
                print(f'  !! {ds}/{dsc} 无图像'); continue
            crops = RB.D.gt_crops(ims, gts)
            if not crops:
                print(f'  !! {ds}/{dsc} 无 crop'); continue
            Ft = RB.clip_encode_imgs(crops)
            src_banks[ds][dsc] = torch.from_numpy(Ft.mean(0)) / torch.from_numpy(Ft.mean(0)).norm()
            print(f'  [src proto] {ds}/{dsc}: {len(crops)} crops')

    # pooled n0/n_iqr（来自 COCO 评测缓存，三域共用同一 COCO 图）
    n_preds = []
    for eval_cls in COCO_EVAL:
        blob = load_coco_cache(eval_cls)
        if blob:
            n_preds.extend([len(p['box']) for p in blob['per_img']])
    n_preds = np.array(n_preds)
    n0 = float(np.median(n_preds)); n_iqr = float(np.subtract(*np.percentile(n_preds, [75, 25])) or 1.0)

    BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    results = {}
    per_domain = {}
    for ds, src_cls, eval_cls, pname in UNITS:
        blob = load_coco_cache(eval_cls)
        if blob is None or src_cls not in src_banks[ds]:
            continue
        per_img = blob['per_img']; F = blob['F']
        P_src = unit(src_banks[ds][src_cls].numpy())
        P_neg = unit(RB.Proto.get_negative_prototype(src_banks[ds], src_cls, NEG_TOPK).numpy())
        dc = E_photo[eval_cls] - E_over_src[(ds, src_cls)]
        dc_mag = float(np.linalg.norm(dc))
        max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
        n_pred = sum(len(p['box']) for p in per_img)
        base = ap_of(per_img, F, P_src - LAM * P_neg)
        best_b, best_ap = 0.0, -1
        for bb in BETA_GRID:
            ap = ap_of(per_img, F, unit(P_src + bb * dc) - LAM * P_neg)
            if ap > best_ap:
                best_ap, best_b = ap, bb
        g_conf = float(np.clip(PRIOR_SCALE * dc_mag * gate_conf(max_conf), 0.0, 4.0))
        g_cnt = float(np.clip(PRIOR_SCALE * dc_mag * gate_count(n_pred, n0, n_iqr), 0.0, 4.0))
        ap_dc = ap_of(per_img, F, unit(P_src + dc_mag * dc) - LAM * P_neg)
        ap_dc_conf = ap_of(per_img, F, unit(P_src + g_conf * dc) - LAM * P_neg)
        ap_dc_cnt = ap_of(per_img, F, unit(P_src + g_cnt * dc) - LAM * P_neg)
        rec = {'eval': eval_cls, 'base': base, 'ap_star': best_ap, 'beta_star': best_b,
               'dc_mag': dc_mag, 'max_conf': max_conf, 'n_pred': n_pred,
               'ap_dc': ap_dc, 'ap_dc_x_conf': ap_dc_conf, 'ap_dc_x_count': ap_dc_cnt,
               'oracle%_dc_conf': ap_dc_conf / best_ap * 100 if best_ap > 0 else None,
               'oracle%_dc': ap_dc / best_ap * 100 if best_ap > 0 else None,
               'gain_dc_conf_vs_base': ap_dc_conf - base}
        results[f'{ds}_{src_cls}->{eval_cls}'] = rec
        per_domain.setdefault(ds, []).append(rec)
        print(f'  [{ds}/{src_cls}->{eval_cls}] base={base:.4f} oracle={best_ap:.4f} '
              f'dc×gate={ap_dc_conf:.4f}({ap_dc_conf/best_ap*100:.1f}%) dc={ap_dc:.4f}')

    print('\n[#3 多域] 每源域 dc×gate(conf) 平均 oracle%')
    for ds in ['dior', 'dota', 'nwpu']:
        rs = per_domain.get(ds, [])
        if not rs:
            continue
        oc = np.mean([r['oracle%_dc_conf'] for r in rs])
        print(f'  {ds:<6} n={len(rs)}  dc×gate oracle% = {oc:.1f}%   '
              f'(base={np.mean([r["base"] for r in rs]):.4f}, '
              f'oracle={np.mean([r["ap_star"] for r in rs]):.4f})')
    print('\n  共用类(airplane/boat/car) 三域 oracle%:')
    for cls in ['airplane', 'boat', 'car']:
        line = f'    {cls:<10}'
        for ds in ['dior', 'dota', 'nwpu']:
            found = [k for k in results if k.endswith(f'->{cls}') and k.startswith(ds + '_')]
            line += f' {ds}={results[found[0]]["oracle%_dc_conf"]:.1f}%' if found else f' {ds}=--'
        print(line)
    json.dump({'config': {'c0': c0, 'prior_scale': PRIOR_SCALE, 'n0': n0, 'n_iqr': n_iqr,
                          'detector': 'gdino-tiny', 'target': 'COCO', 'reuse_cache': 'probe_bidir_cache'},
               'results': results, 'elapsed_s': time.time() - t0},
              open(OUT, 'w'), indent=2, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
