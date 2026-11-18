# -*- coding: utf-8 -*-
"""score_gsd_decomp.py — 决定性 A/B 收敛实验 (真实 RS 条件轴)

设计 (元 2026-10-13 批准):
  真实 RS 条件轴 = GSD 降采样作用于【RS 目标域】DIOR vehicle 图像 (4 档: 1.0/0.5/0.25/0.125)。
  非零 vocab 跨域对 = COCO→DIOR 的 car/truck/bus/motorcycle → vehicle (名称错配, 用户原例 DIOR→VOC truck→veh 的代码内可用等价物)。
  检测器 = grounding-dino-tiny (gdino), 零目标 GT。

机制归因判据 (同一 mag / 同一门控 γ 下, 只动方向):
  base      = unit(P_src) - LAM*P_neg                (β=0, 无修正)
  full      = dc = E_photo[veh] - E_over[src]         (vocab + cond, CA-PC 真实方向)
  vocab-only= unit(E_over[veh] - E_over[src]) * mag    (纯名称映射, 盲于 photo↔overhead 视角)
  cond-only = unit(E_photo[veh] - E_over[veh]) * mag    (纯观测视角偏移, 盲于名称)
  wrong     = unit(E_over[veh] - E_photo[src]) * mag    (反轴对照, overhead-photo)
  满足 full = vocab + cond (文本空间逐字相加)。

裁决:
  Q1 跟踪性 : r( (full_gain - vocab_gain), H ) 且 r( (full_gain - cond_gain), H )
              >0 且显著 => 观测条件项是驱动项, 跟条件走。
  Q2 vocab盲 : vocab_gain 跨 GSD 是否近似平坦 (它不看图像条件)。
  Q3 反轴    : wrong 增益是否 ≤ 0 / 不跟。
  H (严重度) = 1/gsd (k=1,2,4,8), 越大越严重。

口径严格对齐 probe_dir_decomp.py (范数全匹配到 ||dc||, 同门控)。
用法 (CPU, vg 环境):
  HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu "C:/Users/张三/vg/Scripts/python.exe" -u score_gsd_decomp.py
"""
import os, sys, json, pickle, glob, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cpu')
os.environ.setdefault('AC_SPC_COCO_ANN', r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json')
os.environ.setdefault('AC_SPC_COCO_IMG', r'C:\Users\张三\modeling\coco_data\val2017')
os.environ.setdefault('AC_SPC_DIOR_IMG', r'C:\Users\张三\modeling\dior_data\eval_v2_images')
os.environ.setdefault('AC_SPC_DIOR_PARQUET', r'C:\Users\张三\modeling\dior_data\dior_combined.parquet')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch
import run_bidirectional as RB

RB.DEVICE = 'cpu'
LAM = RB.LAM; NEG_TOPK = RB.NEG_TOPK; N_PROTO = RB.N_PROTO
BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
GSD_CACHE = __import__('pathlib').Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_gsd_cache')
EVAL_CLS = 'vehicle'
OFF, N = 20, 100
GSD_LEVELS = [1.0, 0.5, 0.25, 0.125]
SRC_CLASSES = ['car', 'truck', 'bus', 'motorcycle']   # → DIOR vehicle (非零 vocab)
OUT_JSON = __import__('pathlib').Path(r'C:\Users\张三\Desktop\实验数据\score_gsd_decomp.json')
OUT_MD = __import__('pathlib').Path(r'C:\Users\张三\Desktop\实验数据\score_gsd_decomp.md')


def unit(x):
    x = np.asarray(x, float); n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return float('nan'), float('nan')
    x = x - x.mean(); y = y - y.mean()
    denom = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    r = float((x * y).sum() / denom) if denom > 1e-12 else float('nan')
    n = len(x)
    if n > 2 and denom > 1e-12:
        t = r * np.sqrt((n - 2) / max(1e-12, 1 - r ** 2))
        # 双侧 p (近似, df=n-2)
        from math import erf, sqrt
        p = float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))
    else:
        p = float('nan')
    return r, p


def main():
    t0 = time.time()
    cfg = RB.Config(); cfg.model.device = 'cpu'
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    # ── 源原型池 (COCO, 零目标 GT) ──
    bank_coco = {}
    for cls in RB.COCO_POOL:
        ims, gts = RB.load_split(cfg, 'coco', cls, 0, N_PROTO)
        crops = RB.D.gt_crops(ims, gts)
        if not crops:
            continue
        Ft = RB.clip_encode_imgs(crops)
        pt = torch.from_numpy(Ft.mean(0))
        bank_coco[cls] = pt / pt.norm()
    print(f'[protos] COCO: {len(bank_coco)} 类', flush=True)

    # ── 文本原型 ──
    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    idx = {c: i for i, c in enumerate(all_names)}

    # ── 门控尺度 (与 tab:det / probe_dir_decomp 同源) ──
    gcfg = json.load(open(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced.json'))['config']['gate']
    PRIOR_SCALE, c0 = gcfg['prior_scale'], gcfg['c0']
    def gate_conf(conf, c=c0, scale=3.0):
        return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))

    results = {}
    for gsd in GSD_LEVELS:
        cp = GSD_CACHE / f'rev_coco2dior_{EVAL_CLS}_off{OFF}_n{N}_gsd{gsd}.pkl'
        if not cp.exists():
            print(f'  !! 缺 GSD blob {cp.name}', flush=True)
            continue
        blob = pickle.load(open(cp, 'rb'))
        per_img, F = blob['per_img'], blob['F']
        max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
        rows = []
        for sc in SRC_CLASSES:
            if sc not in bank_coco:
                print(f'  !! 缺 COCO 源原型 {sc}', flush=True); continue
            P_src = bank_coco[sc]
            P_neg = unit(RB.Proto.get_negative_prototype(bank_coco, sc, NEG_TOPK))
            dc = E_photo[idx[EVAL_CLS]] - E_over[idx[sc]]
            mag = float(np.linalg.norm(dc))
            gamma = float(np.clip(PRIOR_SCALE * mag * gate_conf(max_conf), 0.0, 4.0))
            Vs = {
                'full': dc,
                'vocab': unit(E_over[idx[EVAL_CLS]] - E_over[idx[sc]]) * mag,
                'cond': unit(E_photo[idx[EVAL_CLS]] - E_over[idx[EVAL_CLS]]) * mag,
                'wrong': unit(E_over[idx[EVAL_CLS]] - E_photo[idx[sc]]) * mag,
            }
            base_w = unit(P_src) - LAM * P_neg
            base = float(RB.ap_base(per_img, F))
            cell = {'src': sc, 'gsd': gsd, 'k': round(1.0 / gsd), 'n_img': len(per_img),
                    'dc_mag': round(mag, 4), 'max_conf': round(max_conf, 4),
                    'gamma': round(gamma, 4), 'base': round(base, 4)}
            for k, v in Vs.items():
                w = unit(P_src + gamma * v) - LAM * P_neg
                g = float(RB.ap_of(per_img, F, w)) - base
                cell[k + '_g'] = round(g, 4)
                # oracle β 峰值 (只给 full / vocab 用于对照天花板)
                if k in ('full', 'vocab'):
                    cur = []
                    for bb in BETA_GRID:
                        cur.append(float(RB.ap_of(per_img, F, unit(P_src + bb * v) - LAM * P_neg)) - base)
                    cell[k + '_star'] = round(max(cur), 4)
                    cell[k + '_star_beta'] = BETA_GRID[int(np.argmax(cur))]
            cell['margin_full_vocab'] = round(cell['full_g'] - cell['vocab_g'], 4)
            cell['margin_full_cond'] = round(cell['full_g'] - cell['vocab_g'], 4)  # full-vocab == full-cond? no
            # 修正: full = vocab + cond, 故 full_g - cond_g 度量 vocab 部分, full_g - vocab_g 度量 cond 部分
            cell['margin_full_cond'] = round(cell['full_g'] - cell['cond_g'], 4)
            rows.append(cell)
            print(f'  [gsd={gsd}] {sc}->veh  base={base:.4f}  full{cell["full_g"]:+.4f}  '
                  f'vocab{cell["vocab_g"]:+.4f}  cond{cell["cond_g"]:+.4f}  wrong{cell["wrong_g"]:+.4f}  '
                  f'Δ(full-vocab){cell["margin_full_vocab"]:+.4f}  γ={gamma:.3f}', flush=True)
        results[str(gsd)] = {'k': round(1.0 / gsd), 'cells': rows,
                             'n_box': int(sum(len(p['gd']) for p in per_img))}

    # ── 聚合 + 裁决 ──
    per_gsd = []
    for gsd in GSD_LEVELS:
        if str(gsd) not in results:
            continue
        cells = results[str(gsd)]['cells']
        if not cells:
            continue
        m = lambda key: float(np.mean([c[key] for c in cells]))
        per_gsd.append({
            'gsd': gsd, 'k': round(1.0 / gsd),
            'base': m('base'),
            'full_g': m('full_g'), 'vocab_g': m('vocab_g'), 'cond_g': m('cond_g'), 'wrong_g': m('wrong_g'),
            'margin_fv': m('margin_full_vocab'), 'margin_fc': m('margin_full_cond'),
            'full_star': m('full_star'), 'vocab_star': m('vocab_star'),
            'gamma': m('gamma'), 'max_conf': m('max_conf'),
        })

    H = [d['k'] for d in per_gsd]
    r_base, p_base = pearson(H, [d['base'] for d in per_gsd])
    r_full, p_full = pearson(H, [d['full_g'] for d in per_gsd])
    r_vocab, p_vocab = pearson(H, [d['vocab_g'] for d in per_gsd])
    r_cond, p_cond = pearson(H, [d['cond_g'] for d in per_gsd])
    r_mfv, p_mfv = pearson(H, [d['margin_fv'] for d in per_gsd])
    r_mfc, p_mfc = pearson(H, [d['margin_fc'] for d in per_gsd])
    r_wrong, p_wrong = pearson(H, [d['wrong_g'] for d in per_gsd])

    verdict = {
        'n_gsd_levels': len(per_gsd),
        'r_base_vs_H': [round(r_base, 4), round(p_base, 4)],
        'r_full_g_vs_H': [round(r_full, 4), round(p_full, 4)],
        'r_vocab_g_vs_H': [round(r_vocab, 4), round(p_vocab, 4)],
        'r_cond_g_vs_H': [round(r_cond, 4), round(p_cond, 4)],
        'r_margin_full_vocab_vs_H': [round(r_mfv, 4), round(p_mfv, 4)],
        'r_margin_full_cond_vs_H': [round(r_mfc, 4), round(p_mfc, 4)],
        'r_wrong_g_vs_H': [round(r_wrong, 4), round(p_wrong, 4)],
    }
    # 文字裁决
    verdict['interpretation'] = []
    verdict['interpretation'].append(
        f'base AP50 与 H 相关 r={r_base:+.4f}(p={p_base:.3g}): '
        + ('条件轴有效拉开 (base 随退化下降)' if r_base < -0.5 else 'base 未明显随退化变化'))
    verdict['interpretation'].append(
        f'vocab-only 增益与 H 相关 r={r_vocab:+.4f}(p={p_vocab:.3g}): '
        + ('vocab 盲于条件(增益不跟 H)' if abs(r_vocab) < 0.5 else 'vocab 增益也随 H 变化'))
    verdict['interpretation'].append(
        f'(full-vocab) 边际(=cond 项贡献) 与 H 相关 r={r_mfv:+.4f}(p={p_mfv:.3g}): '
        + ('观测条件项跟条件走、是驱动项' if (r_mfv > 0.5 and p_mfv < 0.1) else
           '观测条件项未显著随 H 走 (不支持强归因)'))
    verdict['interpretation'].append(
        f'wrong 反轴增益与 H 相关 r={r_wrong:+.4f}(p={p_wrong:.3g}), 均值 wrong_g='
        f'{float(np.mean([d["wrong_g"] for d in per_gsd])):+.4f}: '
        + ('反轴不增/有损 (符合方向正确)' if np.mean([d['wrong_g'] for d in per_gsd]) <= 0.005 else '反轴也有增益 (方向不敏感)'))

    out = {'config': {'eval_cls': EVAL_CLS, 'src_classes': SRC_CLASSES, 'gsd_levels': GSD_LEVELS,
                      'detector': 'grounding-dino-tiny', 'lam': LAM, 'neg_topk': NEG_TOPK,
                      'n_proto': N_PROTO, 'zero_target_gt': True,
                      'note': 'real RS condition axis (GSD on DIOR vehicle target); non-zero vocab '
                              'cross-domain pairs (COCO car/truck/bus/motorcycle -> DIOR vehicle); '
                              'same mag + same gate gamma across directions; only direction differs. '
                              'H=1/gsd severity.',
                      'gate': {'prior_scale': PRIOR_SCALE, 'c0': c0}},
           'per_gsd': per_gsd, 'per_cell': results, 'verdict': verdict,
           'elapsed_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT_JSON, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    # ── MD 报告 ──
    lines = []
    lines.append('# 决定性 A/B 收敛实验: 真实 RS 条件轴 (GSD on DIOR vehicle) + 非零 vocab 跨域对\n')
    lines.append(f'- 生成时间: {time.strftime("%Y-%m-%d %H:%M")}  检测器: grounding-dino-tiny  零目标 GT\n')
    lines.append('## 1. 设计\n')
    lines.append('- **真实 RS 条件轴**: 对 RS 目标域 DIOR `vehicle` 做 GSD 降采样 (1.0/0.5/0.25/0.125, '
                 'k=1/2/4/8), 重检测+CLIP 编码 (gen_dior_gsd_det.py)。')
    lines.append('- **非零 vocab 跨域对**: COCO→DIOR 的 car/truck/bus/motorcycle → vehicle (名称错配)。'
                 '用户原例 "DIOR→VOC truck→veh" 在代码池 (仅 COCO/DIOR) 内的可用等价物是 '
                 '"COCO→DIOR truck→vehicle", 本质同为「名称错配 + 跨域」, 已按此落地。')
    lines.append('- **方向正交分解** (同 probe_dir_decomp, 范数全匹配到 ||dc||, 同门控 γ): '
                 '`full=dc=E_photo[veh]-E_over[src]`, `vocab=unit(E_over[veh]-E_over[src])*mag`, '
                 '`cond=unit(E_photo[veh]-E_over[veh])*mag`, `wrong=unit(E_over[veh]-E_photo[src])*mag`; '
                 '满足 `full = vocab + cond` (文本空间逐字相加)。')
    lines.append('- **判据**: 同一 γ 下只动方向。若 `full-vocab` (cond 项贡献) 与严重度 H 正相关且显著 '
                 '→ 观测条件项跟条件走、是驱动项; vocab-only 盲于图像条件应近似平坦。\n')

    lines.append('## 2. 每 GSD 档聚合 (4 源类均值)\n')
    lines.append('| GSD | k(H) | base | full_g | vocab_g | cond_g | wrong_g | '
                 'Δ(full-vocab) | Δ(full-cond) | γ | max_conf |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for d in per_gsd:
        lines.append(f"| {d['gsd']} | {d['k']} | {d['base']:.4f} | {d['full_g']:+.4f} | "
                     f"{d['vocab_g']:+.4f} | {d['cond_g']:+.4f} | {d['wrong_g']:+.4f} | "
                     f"{d['margin_fv']:+.4f} | {d['margin_fc']:+.4f} | {d['gamma']:.3f} | "
                     f"{d['max_conf']:.4f} |")

    lines.append('\n## 3. 与严重度 H 的相关性 (决策依据)\n')
    lines.append('| 量 | r vs H | p |\n|---|---|---|')
    lines.append(f"| base AP50 | {r_base:+.4f} | {p_base:.3g} |")
    lines.append(f"| full 增益 | {r_full:+.4f} | {p_full:.3g} |")
    lines.append(f"| vocab-only 增益 | {r_vocab:+.4f} | {p_vocab:.3g} |")
    lines.append(f"| cond-only 增益 | {r_cond:+.4f} | {p_cond:.3g} |")
    lines.append(f"| **Δ(full-vocab) 边际** | {r_mfv:+.4f} | {p_mfv:.3g} |")
    lines.append(f"| Δ(full-cond) 边际 | {r_mfc:+.4f} | {p_mfc:.3g} |")
    lines.append(f"| wrong 反轴增益 | {r_wrong:+.4f} | {p_wrong:.3g} |")

    lines.append('\n## 4. 裁决\n')
    for s in verdict['interpretation']:
        lines.append(f'- {s}')
    lines.append('\n## 5. 与既有证据对照\n')
    lines.append('- `ab_convergence_pilot` (合成成像轴 fog/blur/brightness, vocab=identity): '
                 '不支持收敛 (r(H,g_CA)=-0.057 ns; CA 在重退化反被错轴击败)。该小样只隔离 cond-alone '
                 '且合成轴未必是 RS GSD 好代理 → 非反驳, 而是暴露需真实 RS 轴。')
    lines.append('- `probe_dir_decomp` (tiny 9 对): vocab-only +0.0243(5/9) ≥ cond-only +0.0159(6/9) '
                 '→ 文本空间分解里 vocab 项也强。本次用真实 RS GSD 轴直接检验 "vocab-only 是否跟条件"。')
    lines.append('\n## 6. 待办 / 口径红线\n')
    lines.append('- 若本实验也显示 vocab-only 跟条件且 full 优势不随 H 增长 → 观测条件思路确弱, '
                 '投稿前须重议定位 (战略 flag)。')
    lines.append('- 发表版 tab:decomp 数字 (Joint +0.0489 等) 与 probe_decomposition.json (full ≈+0.062) '
                 '待对齐 re-anchored 源 (投稿前)。')
    open(OUT_MD, 'w', encoding='utf-8').write('\n'.join(lines))

    print(f'\n→ {OUT_JSON}  ({time.time()-t0:.0f}s)', flush=True)
    print('裁决摘要:', flush=True)
    for s in verdict['interpretation']:
        print('  -', s, flush=True)


if __name__ == '__main__':
    main()
