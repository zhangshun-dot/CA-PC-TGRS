# -*- coding: utf-8 -*-
"""probe_dirmech_v2.py — 重测「同域对照能否推出规范化(canonicalization)」

为什么重测 (v1 = probe_direction_mechanism.py 的两处口径问题):
  (1) v1 的 base = 检测器**原始置信度** p['gd']；论文全稿 (含 tab:decomp) 的 base 口径是
      **未修正原型** s = sqrt(q * max(0, F·W_base)), W_base = unit(P_src) − λP_neg (β=0)。
      v1 那一列 "gain_beta0" 实测 +0.04~+0.06 (本应为 0) 就是混进来的**换打分器**效应。
      => v1 的绝对增益不可与论文比。
  (2) 随机方向只跑 5 个种子、只报 max−min 的 range；n_eval=30，无法判断 +δ 是否高于噪声底。

v2:
  · base 改为论文口径 (β=0 未修正原型)
  · 零分布: N_RAND 个与 ‖δ‖ 同模长的随机方向 -> mean / sd / p95 / 单边 p
  · bootstrap (over images) 给 (+δ − base) 的 95% CI
  · Arm A = 论文 tab:decomp 口径 (β=2.0, λ=0.5, 未修正原型 base) —— 判读主臂
  · Arm C = 论文部署口径 γ = PRIOR_SCALE·‖δ‖·g(q_max) —— 诊断臂
  · 关键优化: 每个方向的逐图打分**预计算一次**, 重采样只重算 AP
  · 复用 v1 检测缓存 (probe_dirmech_cache), 不跑 GPU/不跑检测

用法:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "C:/Users/张三/GroundingDINO/venv/Scripts/python.exe" -u probe_dirmech_v2.py
"""
import os

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ['AC_SPC_COCO_ANN'] = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'
os.environ['AC_SPC_COCO_IMG'] = r'C:\Users\张三\modeling\coco_data\val2017'
os.environ['AC_SPC_DIOR_IMG'] = r'C:\Users\张三\modeling\dior_data\eval_v2_images'
os.environ['AC_SPC_DIOR_PARQUET'] = r'C:\Users\张三\modeling\dior_data\dior_combined.parquet'

import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import glob
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from acspc.config import Config
from acspc import data as D
from acspc import metrics as Met
from acspc import prototype as Proto

BETA, LAM, NEG_TOPK, N_PROTO = 2.0, 0.5, 3, 20
N_RAND, N_BOOT = 200, 0   # 缓存只有 n=30 张/格 -> 先不跑 bootstrap(它只会给 ±0.3 的宽区间)
CACHE_DIR = Path(r'C:\Users\张三\Desktop\实验数据\probe_dirmech_cache')
OUT = Path(r'C:\Users\张三\Desktop\实验数据\probe_dirmech_v2.json')

COCO_POOL = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
DIOR_POOL = ['airplane', 'ship', 'vehicle', 'harbor', 'storage tank', 'bridge']
TASKS = {
    'same_dior2dior': ('dior', 'dior', [('airplane', 'airplane'), ('ship', 'ship'), ('vehicle', 'vehicle')]),
    'same_coco2coco': ('coco', 'coco', [('airplane', 'airplane'), ('boat', 'boat'), ('car', 'car')]),
    'rev_coco2dior': ('coco', 'dior', [('airplane', 'airplane'), ('boat', 'ship'), ('car', 'vehicle')]),
    'fwd_dior2coco': ('dior', 'coco', [('airplane', 'airplane'), ('ship', 'boat'), ('vehicle', 'car')]),
}
DEVICE, CLIP_ID = 'cpu', 'openai/clip-vit-base-patch32'
_C = {}


def log(*a):
    print(*a, flush=True)


def clip_model():
    if 'm' not in _C:
        from transformers import CLIPModel, CLIPProcessor
        _C['m'] = CLIPModel.from_pretrained(CLIP_ID).to(DEVICE).eval()
        _C['p'] = CLIPProcessor.from_pretrained(CLIP_ID)
    return _C['m'], _C['p']


def clip_imgs(crops, bs=64):
    m, p = clip_model()
    out = []
    for i in range(0, len(crops), bs):
        inp = p(images=crops[i:i + bs], return_tensors='pt', padding=True).to(DEVICE)
        with torch.no_grad():
            f = m.get_image_features(**inp).float()
        out.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(out).cpu()


def text_emb(t):
    m, p = clip_model()
    inp = p(text=[t], return_tensors='pt', padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        f = m.get_text_features(**inp).float()
    return (f / f.norm(dim=-1, keepdim=True)).cpu()[0]


def delta_pos(cls):
    return text_emb(f'a photo of {cls}') - text_emb(f'an overhead image of {cls}')


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


# ───────── 逐图打分 (预计算一次; 重采样只重算 AP) ─────────
def scores_beta0(per_img, F, P_src, P_neg):
    W = (P_src - LAM * P_neg).numpy()
    return [_score_img(p, F, W) for p in per_img]


def scores_dir(per_img, F, P_src, d, P_neg, beta):
    ph = Proto.extrapolate_positive(P_src, d, beta)
    W = (ph - LAM * P_neg).numpy()
    return [_score_img(p, F, W) for p in per_img]


def scores_gated(per_img, F, P_src, d_hat, P_neg, dmag, prior_scale, c0):
    out = []
    for p in per_img:
        if len(p['gd']):
            qmax = float(np.max(p['gd']))
            g = float(np.clip(1.0 - np.tanh(3.0 * (qmax - c0)), 0.0, 1.0))
            ph = P_src + (prior_scale * dmag * g) * d_hat
            ph = ph / ph.norm()
            W = (ph - LAM * P_neg).numpy()
        else:
            W = None
        out.append(_score_img(p, F, W))
    return out


def _score_img(p, F, W):
    n = len(p['gd'])
    if n == 0 or W is None:
        return np.zeros(0)
    vis = np.maximum(0.0, F[p['crop_idx']] @ W)
    return np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))


def ap_of(per_img, scores, idx=None):
    rng = range(len(per_img)) if idx is None else idx
    return Met.compute_ap50_img([(per_img[i]['box'], scores[i]) for i in rng],
                                [per_img[i]['gt'] for i in rng])


def main():
    t0 = time.time()
    cfg = Config()
    cfg.model.device = 'cpu'
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = 0.03
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    gcfg = json.load(open(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced.json'))['config']['gate']
    PRIOR_SCALE, C0 = float(gcfg['prior_scale']), float(gcfg['c0'])

    log('=' * 88)
    log('probe_dirmech_v2: 同域对照 + 论文 base 口径(β=0 未修正原型) + 随机方向零分布')
    log(f'  BETA={BETA} LAM={LAM} N_PROTO={N_PROTO} N_RAND={N_RAND} N_BOOT={N_BOOT}')
    log(f'  Arm A = 论文 tab:decomp 口径(判读主臂);  Arm C = 部署门控口径(诊断)')
    log('=' * 88)

    proto_bank = {}
    for domain, pool in (('coco', COCO_POOL), ('dior', DIOR_POOL)):
        bank = {}
        for cls in pool:
            ims, gts = load_split(cfg, domain, cls, 0, N_PROTO)
            crops = D.gt_crops(ims, gts)
            if not crops:
                continue
            F = clip_imgs(crops)
            p = F.mean(dim=0)
            bank[cls] = p / p.norm()
        proto_bank[domain] = bank
        log(f'[protos] {domain.upper()}: {len(bank)} 类')

    res = {}
    for tname, (src_dom, tgt_dom, pairs) in TASKS.items():
        bank = proto_bank[src_dom]
        for src_cls, eval_cls in pairs:
            files = glob.glob(str(CACHE_DIR / f'{tname}_{eval_cls}_off{N_PROTO}_n*.pkl'))
            if not files or src_cls not in bank:
                log(f'  !! {tname}/{eval_cls} 缺缓存或原型')
                continue
            blob = pickle.load(open(files[0], 'rb'))
            per_img, F = blob['per_img'], blob['F']
            P_src = bank[src_cls]
            P_neg = Proto.get_negative_prototype(bank, src_cls, NEG_TOPK)
            d = delta_pos(src_cls)
            dmag = float(d.norm())
            d_hat = d / d.norm()
            n_img = len(per_img)

            # ---- base (论文口径) ----
            sc_base = scores_beta0(per_img, F, P_src, P_neg)
            base = ap_of(per_img, sc_base)

            # ---- Arm A: β=2 raw δ, 未修正原型 base (论文 tab:decomp 口径) ----
            sc_Ap = scores_dir(per_img, F, P_src, d, P_neg, BETA)
            sc_An = scores_dir(per_img, F, P_src, -d, P_neg, BETA)
            A_pos, A_neg = ap_of(per_img, sc_Ap) - base, ap_of(per_img, sc_An) - base
            A_rnd, rnd_sc = [], []
            for sd in range(N_RAND):
                rng = np.random.RandomState(20000 + sd)
                u = torch.from_numpy(rng.randn(512).astype(np.float32))
                u = u / u.norm() * dmag
                s = scores_dir(per_img, F, P_src, u, P_neg, BETA)
                rnd_sc.append(s)
                A_rnd.append(ap_of(per_img, s) - base)
            A_rnd = np.array(A_rnd)

            # ---- Arm C: 部署门控口径 (诊断) ----
            sc_Cp = scores_gated(per_img, F, P_src, d_hat, P_neg, dmag, PRIOR_SCALE, C0)
            sc_Cn = scores_gated(per_img, F, P_src, -d_hat, P_neg, dmag, PRIOR_SCALE, C0)
            C_pos, C_neg = ap_of(per_img, sc_Cp) - base, ap_of(per_img, sc_Cn) - base
            C_rnd = []
            for sd in range(N_RAND):
                rng = np.random.RandomState(30000 + sd)
                u = torch.from_numpy(rng.randn(512).astype(np.float32))
                u = u / u.norm()
                C_rnd.append(ap_of(per_img, scores_gated(per_img, F, P_src, u, P_neg, dmag, PRIOR_SCALE, C0)) - base)
            C_rnd = np.array(C_rnd)

            # ---- Arm A bootstrap (可选; N_BOOT=0 跳过) ----
            if N_BOOT > 0:
                boot = np.array([ap_of(per_img, sc_Ap, (rng := np.random.RandomState(50000 + b)).randint(0, n_img, n_img))
                                 for b in range(N_BOOT)])
                base_boot = np.array([ap_of(per_img, sc_base, (rng := np.random.RandomState(60000 + b)).randint(0, n_img, n_img))
                                      for b in range(N_BOOT)])
                A_ci = [round(float(np.percentile(boot - base_boot, 2.5)), 4),
                        round(float(np.percentile(boot - base_boot, 97.5)), 4)]
            else:
                A_ci = None

            def pv(g, arr):
                return float(np.mean(arr >= g))

            res[f'{tname}_{eval_cls}'] = {
                'task': tname, 'eval_cls': eval_cls, 'src_cls': src_cls, 'n_img': n_img,
                'delta_mag': round(dmag, 4), 'base_paper': round(base, 4),
                'A_plus': round(A_pos, 4), 'A_minus': round(A_neg, 4),
                'A_rand_mean': round(float(A_rnd.mean()), 4), 'A_rand_sd': round(float(A_rnd.std()), 4),
                'A_rand_p95': round(float(np.percentile(A_rnd, 95)), 4),
                'A_p_plus': round(pv(A_pos, A_rnd), 4), 'A_ci95': A_ci, 'n_rand': N_RAND,
                'C_plus': round(C_pos, 4), 'C_neg': round(C_neg, 4),
                'C_rand_mean': round(float(C_rnd.mean()), 4), 'C_rand_sd': round(float(C_rnd.std()), 4),
                'C_p_plus': round(pv(C_pos, C_rnd), 4),
            }
            ci_s = f' CI{A_ci}' if A_ci else ''
            log(f'  [{tname}/{eval_cls}] base={base:.4f} | '
                f'A: +δ={A_pos:+.4f}{ci_s} −δ={A_neg:+.4f} rand={A_rnd.mean():+.4f}±{A_rnd.std():.4f} p={pv(A_pos, A_rnd):.3f} | '
                f'C: +δ={C_pos:+.4f} rand={C_rnd.mean():+.4f}±{C_rnd.std():.4f}')

    log('\n' + '=' * 88)
    log('汇总 (Arm A = 论文 tab:decomp 口径; +δ 是否显著高于随机方向)')
    log('-' * 88)
    for arm, kp, km, kr in (('A', 'A_p_plus', 'A_plus', 'A_rand_mean'),
                            ('C', 'C_p_plus', 'C_plus', 'C_rand_mean')):
        for scope in ['same_dior2dior', 'same_coco2coco', 'rev_coco2dior', 'fwd_dior2coco']:
            rows = [v for v in res.values() if v['task'] == scope]
            if not rows:
                continue
            ps = [v[kp] for v in rows]
            gp = float(np.mean([v[km] for v in rows]))
            gr = float(np.mean([v[kr] for v in rows]))
            log(f'  Arm {arm} {scope:<16} +δ={gp:+.4f} rand={gr:+.4f} Δ={gp - gr:+.4f} | '
                f'p<0.05: {sum(1 for p in ps if p < 0.05)}/{len(ps)}  p={[round(p, 3) for p in ps]}')
    log('=' * 88)

    OUT.write_text(json.dumps({'config': {'beta': BETA, 'lam': LAM, 'n_rand': N_RAND, 'n_boot': N_BOOT,
                                          'prior_scale': PRIOR_SCALE, 'c0': C0,
                                          'base_note': 'paper beta=0 uncorrected prototype, NOT raw detector score',
                                          'armA_note': 'matches paper tab:decomp injection (beta=2.0, lam=0.5, norm-matched)'},
                               'results': res}, indent=2, ensure_ascii=False), encoding='utf-8')
    log(f'→ {OUT}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
