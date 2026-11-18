# -*- coding: utf-8 -*-
"""exp_preprocess_control.py — 预处理敏感性控制实验 (反驳"Level 1 只是硬件没处理好")

目的: 审稿人若说"两级污染的第一级(观测条件)只是你们硬件/图像没处理好",本实验正面堵死:
  只对**源原型 P_src** 的 DIOR GT crop 施加标准"硬件修复"预处理包
  (MedianFilter 去噪 + autocontrast + equalize, 即 RS 从业者会做的 大气校正/对比度/去噪),
  重建 P_src, 复用现成 COCO 检测缓存, 对比 RAW vs ENHANCED 的:
    - base AP50 (预处理能否关掉跨域残差 → 若大涨说明是质量伪影)
    - dc×gate(conf) 增益 (方向性机制是否在增强后仍成立)
    - 反向 negated 方向 (方向特异性是否在增强后仍存在)

注意: dc 是**文本锚定**("a photo of X" − "an overhead image of Y"), 预处理源图动不了它
  → 即便 base 因预处理微变, 结构性 canonical 方向仍由文本层决定, 这是理论备份。

源域: DIOR (主 RS 源) → COCO, 6 评测类, gdino-tiny, 复用 probe_bidir_cache。
预测: ENHANCED base ≈ RAW base, ENHANCED gain ≈ RAW gain (预处理关不掉结构性残差)。
反驳条件: 若 ENHANCED base 大涨使增益→0, 则部分残差是质量伪影 (如实报)。

用法: HF_HUB_OFFLINE=1 "C:/Users/张三/vg/Scripts/python.exe" -u exp_preprocess_control.py
"""
import os, sys, json, time, pickle
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
import run_bidirectional as RB

RB.DEVICE = DEVICE = os.environ.get('PC_DEVICE', 'cuda')
LAM = RB.LAM; NEG_TOPK = RB.NEG_TOPK; N_PROTO = RB.N_PROTO
COCO_CACHE = r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache'
bal = json.load(open(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced.json'))
c0 = bal['config']['gate']['c0']; PRIOR_SCALE = bal['config']['gate']['prior_scale']
OUT = r'C:\Users\张三\Desktop\实验数据\run_preprocess_control.json'

COCO_EVAL = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
# (源类, 评测类, overhead 提示词)
DIOR_UNITS = [
    ('airplane', 'airplane', 'airplane'),
    ('ship', 'boat', 'ship'),
    ('vehicle', 'car', 'vehicle'),
    ('vehicle', 'truck', 'vehicle'),
    ('vehicle', 'bus', 'vehicle'),
    ('vehicle', 'motorcycle', 'vehicle'),
]
SRC_CLS_SET = sorted({u[0] for u in DIOR_UNITS})


def unit(x):
    x = np.asarray(x, float); n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def preprocess(crop_pil):
    """标准 RS '硬件修复' 预处理包: 去噪 + 自动对比度 + 直方图均衡 (大气校正/对比度代理)。"""
    im = crop_pil.convert('RGB')
    im = im.filter(ImageFilter.MedianFilter(3))   # 传感器噪声
    im = ImageOps.autocontrast(im)                 # 对比度 / 大气校正代理
    im = ImageOps.equalize(im)                     # 直方图均衡 (雾霾/对比度)
    return im


def build_dior_bank(do_pp):
    cfg = RB.Config(); cfg.model.device = 'cpu'
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'
    bank = {}
    for cls in RB.DIOR_POOL:
        ims, gts = RB.load_split(cfg, 'dior', cls, 0, N_PROTO)
        keep = [(i, g) for i, g in zip(ims, gts) if len(g) > 0][:N_PROTO]
        if not keep:
            continue
        crops = RB.D.gt_crops([k[0] for k in keep], [k[1] for k in keep])
        if do_pp:
            crops = [preprocess(c) for c in crops]
        if not crops:
            continue
        Ft = RB.clip_encode_imgs(crops)
        bank[cls] = torch.from_numpy(Ft.mean(0)) / torch.from_numpy(Ft.mean(0)).norm()
        print(f'  [src proto {"ENH" if do_pp else "RAW"}] dior/{cls}: {len(crops)} crops', flush=True)
    return bank


def load_coco_cache_n100(eval_cls):
    import glob
    hits = glob.glob(os.path.join(COCO_CACHE, f'fwd_dior2coco_{eval_cls}_off{N_PROTO}_n100.pkl'))
    if not hits:
        hits = glob.glob(os.path.join(COCO_CACHE, f'fwd_dior2coco_{eval_cls}_n*.pkl'))
    return pickle.load(open(hits[0], 'rb')) if hits else None


def gate_conf(conf, c=c0, scale=3.0):
    return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))


def ap_of(per_img, F, W):
    rows = []
    for p in per_img:
        if len(p['gd']):
            w = F[p['crop_idx']] @ W
            s = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
        else:
            s = np.zeros(0)
        rows.append((p['box'], s))
    return float(RB.Met.compute_ap50_img(rows, [p['gt'] for p in per_img]))


def main():
    t0 = time.time()
    cfg = RB.Config(); cfg.model.device = 'cpu'
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    E_photo = {c: RB.text_embs([f'a photo of {c}'])[0] for c in COCO_EVAL}
    E_over = {sc: RB.text_embs([f'an overhead image of {sc}'])[0] for sc in SRC_CLS_SET}

    bank_raw = build_dior_bank(do_pp=False)
    bank_enh = build_dior_bank(do_pp=True)

    # n0 / n_iqr (COCO 评测缓存, 三域共用同一 COCO 图)
    n_preds = []
    for ec in COCO_EVAL:
        blob = load_coco_cache_n100(ec)
        if blob:
            n_preds.extend([len(p['box']) for p in blob['per_img']])
    n_preds = np.array(n_preds)
    n0 = float(np.median(n_preds))
    n_iqr = float(np.subtract(*np.percentile(n_preds, [75, 25])) or 1.0)
    gate_count = lambda n_pred: float(np.clip(1.0 / (1.0 + np.exp((n_pred - n0) / (n_iqr / 2 + 1e-9))), 0.0, 1.0))

    rows = []
    agg = {'raw_base': [], 'enh_base': [], 'raw_gain': [], 'enh_gain': []}
    for src_cls, eval_cls, pname in DIOR_UNITS:
        blob = load_coco_cache_n100(eval_cls)
        if blob is None or src_cls not in bank_raw:
            continue
        per_img = blob['per_img']; F = blob['F']
        dc = E_photo[eval_cls] - E_over[src_cls]
        dc_mag = float(np.linalg.norm(dc))
        max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
        n_pred = sum(len(p['box']) for p in per_img)

        for tag, bank in (('RAW', bank_raw), ('ENH', bank_enh)):
            P_src = unit(bank[src_cls].numpy())
            P_neg = unit(RB.Proto.get_negative_prototype(bank, src_cls, NEG_TOPK).numpy())
            base = ap_of(per_img, F, P_src - LAM * P_neg)
            g_conf = float(np.clip(PRIOR_SCALE * dc_mag * gate_conf(max_conf), 0.0, 4.0))
            ap_dc = ap_of(per_img, F, unit(P_src + dc_mag * dc) - LAM * P_neg)
            ap_dc_conf = ap_of(per_img, F, unit(P_src + g_conf * dc) - LAM * P_neg)
            ap_neg = ap_of(per_img, F, unit(P_src - dc_mag * dc) - LAM * P_neg)
            r = {'src': src_cls, 'eval': eval_cls, 'tag': tag, 'base': round(base, 4),
                 'dc_mag': round(dc_mag, 4), 'ap_dc': round(ap_dc, 4),
                 'ap_dc_conf': round(ap_dc_conf, 4), 'ap_neg': round(ap_neg, 4),
                 'gain_dc': round(ap_dc - base, 4), 'gain_dc_conf': round(ap_dc_conf - base, 4),
                 'neg_gain': round(ap_neg - base, 4)}
            rows.append(r)
            if tag == 'RAW':
                agg['raw_base'].append(base); agg['raw_gain'].append(ap_dc_conf - base)
            else:
                agg['enh_base'].append(base); agg['enh_gain'].append(ap_dc_conf - base)
            print(f'  [{tag}] {src_cls}->{eval_cls}: base={base:.4f} dc×gate={ap_dc_conf:.4f} '
                  f'(gain={ap_dc_conf-base:+.4f}) neg={ap_neg:.4f}({ap_neg-base:+.4f})', flush=True)

    def m(x):
        return float(np.mean(x)) if x else float('nan')
    summary = {
        'raw_mean_base': round(m(agg['raw_base']), 4),
        'enh_mean_base': round(m(agg['enh_base']), 4),
        'base_delta': round(m(agg['enh_base']) - m(agg['raw_base']), 4),
        'raw_mean_gain': round(m(agg['raw_gain']), 4),
        'enh_mean_gain': round(m(agg['enh_gain']), 4),
        'gain_delta': round(m(agg['enh_gain']) - m(agg['raw_gain']), 4),
    }
    json.dump({'config': {'source': 'DIOR', 'target': 'COCO', 'detector': 'gdino-tiny',
                          'n_proto': N_PROTO, 'preprocess': 'MedianFilter3+autocontrast+equalize',
                          'gate': {'c0': c0, 'prior_scale': PRIOR_SCALE}, 'device': DEVICE},
               'rows': rows, 'summary': summary, 'elapsed_s': round(time.time() - t0, 1)},
              open(OUT, 'w'), indent=2, ensure_ascii=False)
    print('\n=== 汇总 (DIOR→COCO, RAW vs ENHANCED 源原型) ===')
    print(f"  RAW : mean base={summary['raw_mean_base']:.4f}  mean dc×gate gain={summary['raw_mean_gain']:+.4f}")
    print(f"  ENH : mean base={summary['enh_mean_base']:.4f}  mean dc×gate gain={summary['enh_mean_gain']:+.4f}")
    print(f"  BASE Δ(enh-raw) = {summary['base_delta']:+.4f}   GAIN Δ(enh-raw) = {summary['gain_delta']:+.4f}")
    print(f"  → 若 |BASE Δ| 很小: 预处理关不掉跨域残差 (结构性); 若 BASE Δ 大: 部分残差是质量伪影")
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
