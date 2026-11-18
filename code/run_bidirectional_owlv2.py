# -*- coding: utf-8 -*-
"""run_bidirectional_owlv2.py — 换检测器复现双向结论 (OWLv2)

汇报稿自陈边界: 「强检测器上收益趋近于0或转负」。
本脚本用 OWLv2(google/owlv2-base-patch16-ensemble, 强于 gdino-tiny)复跑双向实验,
检验"模板锚定不变 > 按角色交换"这一结论是否跨检测器成立。

与 GDINO 实验保持一致的部分: 原型 20 张/类(与评测不相交)、评测 100 张/类、
CLIP ViT-B/32、β=2.0、λ=0.5、负原型 top-3、11-point AP50。
差异: 检测器换成 OWLv2; 候选框由 post_process threshold=0.03 得到(等价于 GDINO 的预筛)。

用法:
  HF_HUB_OFFLINE=1 "C:/Users/张三/vg/Scripts/python.exe" -u run_bidirectional_owlv2.py --n-eval 100
"""
import os, sys, json, pickle, time
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
from transformers import CLIPModel, CLIPProcessor, Owlv2Processor, Owlv2ForObjectDetection

from acspc.config import Config
from acspc import data as D
from acspc import metrics as Met

CACHE = Path(r'C:\Users\张三\Desktop\实验数据\probe_owl_cache')
CACHE.mkdir(parents=True, exist_ok=True)
OUT = Path(r'C:\Users\张三\Desktop\实验数据\run_bidirectional_owlv2.json')
PROTOS = Path(r'C:\Users\张三\Desktop\实验数据\probe_equiv_protos.pkl')

BETA, LAM, NEG_TOPK, N_PROTO, PRE_FILTER = 2.0, 0.5, 3, 20, 0.03
CLIP_ID = 'openai/clip-vit-base-patch32'
DEVICE = os.environ.get('AC_SPC_DET_DEVICE', 'cuda')

CORE = [('airplane', 'airplane'), ('ship', 'boat'), ('vehicle', 'car')]
EXTRA = [('vehicle', 'truck'), ('vehicle', 'bus'), ('vehicle', 'motorcycle')]
JOBS = [('fwd_dior2coco', CORE + EXTRA, 'dior'), ('rev_coco2dior', [(c, d) for d, c in CORE], 'coco')]


def log(*a):
    print(*a, flush=True)


# ───────── CLIP ─────────
_CLIP = {}


def _as_vec(x):
    for a in ('pooler_output', 'image_embeds', 'text_embeds', 'last_hidden_state'):
        if hasattr(x, a) and getattr(x, a) is not None:
            return getattr(x, a)
    return x


def get_clip():
    if 'model' not in _CLIP:
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


# ───────── OWLv2 ─────────
_OWL = {}


def get_owl():
    if 'model' not in _OWL:
        base = Path(r'C:\Users\张三\.cache\huggingface\hub\models--google--owlv2-base-patch16-ensemble\snapshots')
        snap = sorted(base.iterdir())[0]
        _OWL['proc'] = Owlv2Processor.from_pretrained(str(snap))
        _OWL['model'] = Owlv2ForObjectDetection.from_pretrained(
            str(snap), dtype=torch.float32).eval().to(DEVICE)
    return _OWL['proc'], _OWL['model']


@torch.no_grad()
def owl_detect(images, prompt):
    proc, model = get_owl()
    out = []
    for img in images:
        try:
            inputs = proc(text=[prompt], images=img, return_tensors='pt')
        except Exception:
            inputs = proc(text=[prompt], images=img, return_tensors='pt', truncation=True)
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        o = model(**inputs)
        target = torch.tensor([img.size[::-1]]).to(DEVICE)
        # transformers 5.x 把 post_process_object_detection 移到了 image_processor 上
        pp = getattr(proc, 'post_process_object_detection', None) or \
             getattr(proc.image_processor, 'post_process_object_detection')
        r = pp(o, threshold=PRE_FILTER, target_sizes=target)[0]
        out.append((r['boxes'].cpu().numpy(), r['scores'].cpu().numpy()))
    return out


def load_split(cfg, domain, cls, offset, n):
    if domain == 'coco':
        ims, gts = D.load_coco(cfg, cls, n=offset + n)
    else:
        ims, gts = D.load_dior(cfg, cls, n=offset + n)
    ims, gts = ims[offset:offset + n], gts[offset:offset + n]
    keep = [(i, g) for i, g in zip(ims, gts) if len(g) > 0]
    if not keep:
        return [], []
    return [k[0] for k in keep], [k[1] for k in keep]


def detect_encode(images, gts, prompt, key):
    cp = CACHE / f'{key}.pkl'
    if cp.exists():
        log(f'    [cache] {cp.name}')
        return pickle.load(open(cp, 'rb'))
    t0 = time.time()
    dets = owl_detect(images, prompt)
    # 分批裁剪+编码, 及时释放 (一次性 crop 全部候选框会耗尽系统内存)
    per_img, pending, pend_idx, Fs, CH = [], [], [], [], 1500
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
            ci.append(len(pend_idx)); pend_idx.append(len(pending))
            pending.append(img.crop((x1, y1, x2, y2)))
            if len(pending) >= CH:
                Fs.append(clip_encode(pending)); pending.clear()
        per_img.append({
            'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
            'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
            'gt': np.array(g, dtype=np.float64).reshape(-1, 4),
            'local_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64),
        })
    if pending:
        Fs.append(clip_encode(pending)); pending.clear()
    F = np.concatenate(Fs) if Fs else np.zeros((0, 512))
    pend_arr = np.asarray(pend_idx, dtype=np.int64)
    for p in per_img:
        p['crop_idx'] = pend_arr[p['local_idx']] if len(p['local_idx']) else np.zeros(0, dtype=np.int64)
        p.pop('local_idx', None)
    log(f'    N={len(images)} boxes={sum(len(p["gd"]) for p in per_img)} ({time.time()-t0:.0f}s)')
    blob = {'per_img': per_img, 'F': F}
    pickle.dump(blob, open(cp, 'wb'))
    return blob


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


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--n-eval', type=int, default=100)
    ap_.add_argument('--selftest', action='store_true')
    args = ap_.parse_args()

    cfg = Config(); cfg.model.device = DEVICE

    if args.selftest:
        ims, gts = load_split(cfg, 'dior', 'airplane', 0, 2)
        d = owl_detect(ims[:2], 'an overhead image of airplane')
        log(f'selftest: imgs={len(ims)} boxes={[len(x[0]) for x in d]}')
        return

    log('=' * 88)
    log(f'双向实验 @ OWLv2   device={DEVICE}  N_EVAL={args.n_eval}')
    log('=' * 88)

    bank_all = pickle.load(open(PROTOS, 'rb'))
    rng = np.random.default_rng(0)
    res = {}

    for tname, pairs, dom in JOBS:
        bank = bank_all.get(dom, {})
        rows = {}
        for src_cls, eval_cls in pairs:
            if src_cls not in bank:
                continue
            imgs, gts = load_split(cfg, 'coco' if tname == 'fwd_dior2coco' else 'dior',
                                   eval_cls, N_PROTO, args.n_eval)
            if len(imgs) < 8:
                log(f'  !! {tname}/{eval_cls} 图像不足 {len(imgs)}')
                continue
            prompt = (f'a photo of {eval_cls}' if tname == 'fwd_dior2coco'
                      else f'an overhead image of {eval_cls}')
            log(f'\n[{tname}] {eval_cls} ← {dom}:{src_cls}  N={len(imgs)}')
            blob = detect_encode(imgs, gts, prompt, f'{tname}_{eval_cls}_n{len(imgs)}')
            per_img, F = blob['per_img'], blob['F']

            P_src, P_neg = bank[src_cls], neg_of(bank, src_cls)
            base = ap_base(per_img, F)
            # A 锚定不变: photo 在目标类, overhead 在源类
            dA = text_emb(f'a photo of {eval_cls}') - text_emb(f'an overhead image of {src_cls}')
            # B 按角色交换: 源域用 photo, 目标域用 overhead
            dB = text_emb(f'an overhead image of {eval_cls}') - text_emb(f'a photo of {src_cls}')
            rr = [unit(rng.normal(size=512)) * np.linalg.norm(dA) for _ in range(3)]
            rows[f'{src_cls}->{eval_cls}'] = {
                'baseline': round(base, 4), 'n_img': len(imgs),
                'n_box': int(sum(len(p['gd']) for p in per_img)),
                'A_anchored': round(ap_of(per_img, F, unit(P_src + BETA * dA) - LAM * P_neg) - base, 4),
                'B_roleswap': round(ap_of(per_img, F, unit(P_src + BETA * dB) - LAM * P_neg) - base, 4),
                'beta0': round(ap_of(per_img, F, P_src - LAM * P_neg) - base, 4),
                'rand_mean': round(float(np.mean(
                    [ap_of(per_img, F, unit(P_src + BETA * u) - LAM * P_neg) - base for u in rr])), 4),
            }
            r = rows[f'{src_cls}->{eval_cls}']
            log(f'    base={base:.4f}  A(锚定)={r["A_anchored"]:+.4f}  '
                f'B(交换)={r["B_roleswap"]:+.4f}  β=0={r["beta0"]:+.4f}  rand={r["rand_mean"]:+.4f}')
        res[tname] = rows

    json.dump({'config': {'detector': 'owlv2-base-patch16-ensemble', 'beta': BETA, 'lam': LAM,
                          'n_eval': args.n_eval, 'pre_filter': PRE_FILTER, 'clip': CLIP_ID},
               'results': res}, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    log('\n' + '=' * 88)
    log('汇总 @ OWLv2')
    log('=' * 88)
    for tname, rows in res.items():
        if not rows:
            continue
        log(f'\n[{tname}] n_class={len(rows)}')
        log(f'  {"单元":<22s}{"baseline":>10s}{"A 锚定":>11s}{"B 交换":>11s}{"A−B":>10s}{"β=0":>10s}{"随机":>10s}')
        for k, v in rows.items():
            log(f'  {k:<22s}{v["baseline"]:>10.4f}{v["A_anchored"]:>+11.4f}{v["B_roleswap"]:>+11.4f}'
                f'{v["A_anchored"]-v["B_roleswap"]:>+10.4f}{v["beta0"]:>+10.4f}{v["rand_mean"]:>+10.4f}')
        for kk in ['A_anchored', 'B_roleswap', 'beta0', 'rand_mean']:
            log(f'  均值 {kk:<15s}{np.mean([v[kk] for v in rows.values()]):+10.4f}')
        log(f'  均值 baseline      {np.mean([v["baseline"] for v in rows.values()]):>10.4f}')
    log(f'\n已保存: {OUT}')


if __name__ == '__main__':
    main()
