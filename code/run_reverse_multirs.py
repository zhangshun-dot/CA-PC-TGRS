# -*- coding: utf-8 -*-
"""run_reverse_multirs.py — 反向任务扩到 3 个遥感评测集

动机: 反向(COCO→遥感)目前只有 DIOR 的 3 个类(airplane/ship/vehicle),
      样本面太窄。本脚本把遥感侧扩到:
        DIOR   : airplane, ship, vehicle                        (3 类)
        DOTA v1.5: plane, ship, small-vehicle, large-vehicle     (4 类)
        NWPU VHR-10: airplane, ship, vehicle                    (3 类)
      共 10 个"评测类-数据集"单元。

判据: 若"模板锚定不变"在三个遥感集上都优于"按角色交换模板",
      则方向结论不是 DIOR 特有的, 可上升为跨域不变量。

变体(每个单元):
  base   原始检测分
  A      模板锚定不变:  δ = E_photo(tgt) − E_overhead(src)      ← 本文
  B      按角色交换:    δ = E_overhead(tgt) − E_photo(src)      ← 现有做法的适配
  beta0  不外推
  rand   随机方向 ×3 (与 δ 同模长)

用法:
  HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cuda \
  "C:/Users/张三/vg/Scripts/python.exe" -u run_reverse_multirs.py --n-eval 100
"""
import os, sys, json, pickle, time, re
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_COCO_ANN', r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json')
os.environ.setdefault('AC_SPC_COCO_IMG', r'C:\Users\张三\modeling\coco_data\val2017')
os.environ.setdefault('AC_SPC_DIOR_IMG', r'C:\Users\张三\modeling\dior_data\eval_v2_images')
os.environ.setdefault('AC_SPC_DIOR_PARQUET', r'C:\Users\张三\modeling\dior_data\dior_combined.parquet')

sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

from pathlib import Path
import argparse
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from acspc.config import Config
from acspc import data as D
from acspc import models as M
from acspc import metrics as Met

CACHE = Path(r'C:\Users\张三\Desktop\实验数据\probe_multirs_cache')
CACHE.mkdir(parents=True, exist_ok=True)
OUT = Path(r'C:\Users\张三\Desktop\实验数据\run_reverse_multirs.json')
PROTOS = Path(r'C:\Users\张三\Desktop\实验数据\probe_equiv_protos.pkl')

BETA, LAM, NEG_TOPK, PRE_FILTER, N_PROTO = 2.0, 0.5, 3, 0.03, 20
CLIP_ID = 'openai/clip-vit-base-patch32'
DEVICE = os.environ.get('AC_SPC_DET_DEVICE', 'cuda')

DOTA_ROOT = Path(r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\val')
NWPU_ROOT = Path(r'C:\Users\张三\modeling\nwpu_data\NWPU-VHR-10')
NWPU_CLS = {1: 'airplane', 2: 'ship', 3: 'storage tank', 4: 'baseball diamond',
            5: 'tennis court', 6: 'basketball court', 7: 'ground track field',
            8: 'harbor', 9: 'bridge', 10: 'vehicle'}

# (遥感数据集, 遥感评测类, COCO 源类)
UNITS = [
    ('dior', 'airplane', 'airplane'),
    ('dior', 'ship', 'boat'),
    ('dior', 'vehicle', 'car'),
    ('dota', 'plane', 'airplane'),
    ('dota', 'ship', 'boat'),
    ('dota', 'small-vehicle', 'car'),
    ('dota', 'large-vehicle', 'truck'),
    ('nwpu', '1', 'airplane'),
    ('nwpu', '2', 'boat'),
    ('nwpu', '10', 'car'),
]


def log(*a):
    print(*a, flush=True)


# ───────── 数据集加载 ─────────
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
        try:
            lines = [l.strip() for l in open(f, encoding='utf-8', errors='ignore') if l.strip()]
        except Exception:
            continue
        boxes = []
        for l in lines:
            if l.startswith('imagesource') or l.startswith('gsd'):
                continue
            parts = l.split()
            if len(parts) < 9:
                continue
            if parts[8] != cls:
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
    out = _sorted_num([o[0] for o in out]) and out
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
            if not m:
                continue
            if int(m.group(5)) != int(cls_idx):
                continue
            x1, y1, x2, y2 = [float(m.group(i)) for i in range(1, 5)]
            if x2 - x1 > 4 and y2 - y1 > 4:
                boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        if boxes:
            out.append((ip, boxes))
    sel = out[offset:offset + n]
    imgs = [Image.open(p).convert('RGB') for p, _ in sel]
    return imgs, [b for _, b in sel]


def load_rs(cfg, ds, cls, offset, n):
    if ds == 'dior':
        ims, gts = D.load_dior(cfg, cls, n=offset + n)
        keep = [(i, g) for i, g in zip(ims, gts) if len(g) > 0]
        ims = [k[0] for k in keep][offset:offset + n]
        gts = [k[1] for k in keep][offset:offset + n]
        return ims, gts
    if ds == 'dota':
        return load_dota(cls, offset, n)
    if ds == 'nwpu':
        return load_nwpu(cls, offset, n)
    return [], []


# ───────── CLIP ─────────
_CLIP = {}


def _as_vec(x):
    for a in ('pooler_output', 'image_embeds', 'text_embeds', 'last_hidden_state'):
        if hasattr(x, a) and getattr(x, a) is not None:
            return getattr(x, a)
    return x


def get_clip():
    if 'model' not in _CLIP:
        from transformers import CLIPModel, CLIPProcessor
        _CLIP['model'] = CLIPModel.from_pretrained(CLIP_ID).to(DEVICE).eval()
        _CLIP['proc'] = CLIPProcessor.from_pretrained(CLIP_ID)
    return _CLIP['model'], _CLIP['proc']


@torch.no_grad()
def clip_encode(crops, bs=128):
    model, proc = get_clip()
    out = []
    for i in range(0, len(crops), bs):
        f = _as_vec(model.get_image_features(
            **proc(images=crops[i:i + bs], return_tensors='pt', padding=True).to(DEVICE))).float()
        out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 512))


@torch.no_grad()
def text_emb(t):
    model, proc = get_clip()
    f = _as_vec(model.get_text_features(
        **proc(text=[t], return_tensors='pt', padding=True, truncation=True).to(DEVICE))).float()
    return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()[0]


def unit(x):
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def neg_of(bank, cls):
    sub = {k: v for k, v in bank.items() if k != cls}
    if not sub:
        return np.zeros(512)
    ks = list(sub); P = np.stack([sub[k] for k in ks])
    o = np.argsort(-(P @ bank[cls]))[:NEG_TOPK]
    return unit(P[o].mean(0))


# ───────── 检测 + 编码 ─────────
def detect_encode(cfg, images, gts, prompt, key):
    cp = CACHE / f'{key}.pkl'
    if cp.exists():
        log(f'    [cache] {cp.name}')
        return pickle.load(open(cp, 'rb'))
    t0 = time.time()
    dets = M.gdino_detect(cfg, images, prompt, cfg.model.detection_threshold)
    # 分批裁剪+编码, 及时释放 (整批一次性 crop 会在大候选池上耗尽系统内存)
    per_img, pending, pend_idx, Fs, CH = [], [], [], [], 1500

    def flush():
        if pending:
            Fs.append(clip_encode(pending))
            pending.clear()
        return

    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < PRE_FILTER:
                continue
            bl.append([x1, y1, x2, y2]); sl.append(float(sc))
            ci.append(len(pend_idx))
            pend_idx.append(len(pending))
            pending.append(img.crop((x1, y1, x2, y2)))
            if len(pending) >= CH:
                flush()
        per_img.append({
            'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
            'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
            'gt': np.array(g, dtype=np.float64).reshape(-1, 4),
            'local_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64),
        })
    flush()
    F = np.concatenate(Fs) if Fs else np.zeros((0, 512))
    # local_idx -> 全局 crop 下标
    for p in per_img:
        p['crop_idx'] = np.asarray(pend_idx, dtype=np.int64)[p['local_idx']] if len(p['local_idx']) \
            else np.zeros(0, dtype=np.int64)
        p.pop('local_idx', None)
    del pend_idx
    log(f'    N={len(images)} boxes={sum(len(p["gd"]) for p in per_img)} ({time.time()-t0:.0f}s)')
    pickle.dump({'per_img': per_img, 'F': F}, open(cp, 'wb'))
    return {'per_img': per_img, 'F': F}


def ap_of(per_img, F, W):
    for p in per_img:
        if len(p['gd']):
            p['_s'] = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, F[p['crop_idx']] @ W), 1e-6, 1.0))
        else:
            p['_s'] = np.zeros(0)
    return Met.compute_ap50_img([(p['box'], p['_s']) for p in per_img], [p['gt'] for p in per_img])


def ap_base(per_img, F):
    for p in per_img:
        p['_s'] = p['gd'].copy()
    return Met.compute_ap50_img([(p['box'], p['_s']) for p in per_img], [p['gt'] for p in per_img])


PROMPT_NAME = {'plane': 'plane', 'ship': 'ship', 'small-vehicle': 'small vehicle',
               'large-vehicle': 'large vehicle', '1': 'airplane', '2': 'ship', '10': 'vehicle'}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--n-eval', type=int, default=100)
    args = ap_.parse_args()

    cfg = Config()
    cfg.model.device = DEVICE
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    log('=' * 88)
    log(f'反向任务扩到 3 个遥感集   device={DEVICE}  N_EVAL={args.n_eval}')
    log('=' * 88)

    bank = pickle.load(open(PROTOS, 'rb'))['coco']     # 源域 = COCO
    log(f'COCO 源原型: {sorted(bank)}')

    rng = np.random.default_rng(0)
    res = {}
    for ds, rs_cls, src_cls in UNITS:
        key = f'{ds}_{rs_cls}_{src_cls}'
        if src_cls not in bank:
            log(f'!! 缺源原型 {src_cls}, 跳过 {key}')
            continue
        imgs, gts = load_rs(cfg, ds, rs_cls, 0, args.n_eval)
        if len(imgs) < 8:
            log(f'!! {key}: 图像不足 ({len(imgs)}), 跳过')
            continue
        pname = PROMPT_NAME.get(rs_cls, rs_cls)
        prompt = f'an overhead image of {pname}'
        log(f'\n[{key}] N={len(imgs)} prompt="{prompt}"')
        blob = detect_encode(cfg, imgs, gts, prompt, f'{key}_n{len(imgs)}')
        per_img, F = blob['per_img'], blob['F']

        P_src, P_neg = bank[src_cls], neg_of(bank, src_cls)
        base = ap_base(per_img, F)
        # A: 模板锚定不变（photo 永远在目标类、overhead 永远在源类）
        dA = text_emb(f'a photo of {pname}') - text_emb(f'an overhead image of {src_cls}')
        # B: 按源/目标角色交换（源=COCO→photo, 目标=遥感→overhead）
        dB = text_emb(f'an overhead image of {pname}') - text_emb(f'a photo of {src_cls}')
        rr = [unit(rng.normal(size=512)) * np.linalg.norm(dA) for _ in range(3)]
        r = {'baseline': round(base, 4), 'n_img': len(imgs),
             'n_box': int(sum(len(p['gd']) for p in per_img)),
             'A_anchored': round(ap_of(per_img, F, unit(P_src + BETA * dA) - LAM * P_neg) - base, 4),
             'B_roleswap': round(ap_of(per_img, F, unit(P_src + BETA * dB) - LAM * P_neg) - base, 4),
             'beta0': round(ap_of(per_img, F, P_src - LAM * P_neg) - base, 4),
             'rand_mean': round(float(np.mean(
                 [ap_of(per_img, F, unit(P_src + BETA * u) - LAM * P_neg) - base for u in rr])), 4)}
        res[key] = r
        log(f'    base={base:.4f}  A(锚定)={r["A_anchored"]:+.4f}  '
            f'B(交换)={r["B_roleswap"]:+.4f}  β=0={r["beta0"]:+.4f}  rand={r["rand_mean"]:+.4f}')

    json.dump({'config': {'device': DEVICE, 'beta': BETA, 'lam': LAM, 'n_eval': args.n_eval,
                          'detector': 'gdino-tiny', 'clip': CLIP_ID}, 'results': res},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    log('\n' + '=' * 88)
    log('汇总')
    log('=' * 88)
    log(f'  {"单元":<26s}{"baseline":>10s}{"A 锚定":>11s}{"B 交换":>11s}{"A−B":>10s}{"β=0":>10s}{"随机":>10s}')
    for k, v in res.items():
        log(f'  {k:<26s}{v["baseline"]:>10.4f}{v["A_anchored"]:>+11.4f}{v["B_roleswap"]:>+11.4f}'
            f'{v["A_anchored"]-v["B_roleswap"]:>+10.4f}{v["beta0"]:>+10.4f}{v["rand_mean"]:>+10.4f}')
    for ds in ['dior', 'dota', 'nwpu']:
        sub = {k: v for k, v in res.items() if k.startswith(ds)}
        if not sub:
            continue
        log(f'\n  [{ds}] n={len(sub)}  A={np.mean([v["A_anchored"] for v in sub.values()]):+.4f}  '
            f'B={np.mean([v["B_roleswap"] for v in sub.values()]):+.4f}  '
            f'A−B={np.mean([v["A_anchored"]-v["B_roleswap"] for v in sub.values()]):+.4f}')
    allv = list(res.values())
    if allv:
        log(f'\n  [全部 {len(allv)} 单元] A={np.mean([v["A_anchored"] for v in allv]):+.4f}  '
            f'B={np.mean([v["B_roleswap"] for v in allv]):+.4f}  '
            f'A−B={np.mean([v["A_anchored"]-v["B_roleswap"] for v in allv]):+.4f}  '
            f'β=0={np.mean([v["beta0"] for v in allv]):+.4f}  '
            f'rand={np.mean([v["rand_mean"] for v in allv]):+.4f}')
    log(f'\n已保存: {OUT}')


if __name__ == '__main__':
    main()
