# -*- coding: utf-8 -*-
r"""tge_v2_bidirectional_autoprior.py — (ii) 双向自适应 + (iii) 自动标注：用零标注量替换专家手写先验

## 元的问题（2026-09-18 20:0x）

「哪个A-CSPC方法不是单向？物理先验需要专家标注吗？能不能尝试改成双向自适应和自动标注」

## 已经测出来的事实（本轮）

**(a) A-CSPC 是不是单向的？** 算子是 `p_hat = unit(p_src + β·Δ) − λ·p_neg`，β 固定取 +2.0
⇒ **方向系数只有正号，是单向的**。但 `tge_v2_beta_negative_4sources.json` 刚测出：

    22/22 个 (源域, 类) 单元格在扩展网格 β∈[−4, 4] 上取到**内点最优**（0 格落在边界）；
    四个源域的均值最优 β 全部 = 2.0（β=2.0 是共同峰）；
    唯一例外：DIOR-airplane 最优 β=−1.5、DOTA-airplane 最优 β=−0.5（2 格取到 β<0）。

⇒ **单向不是"没搜索过"造成的，是搜过之后的结论**；但**符号本身存在类级差异**，
   而 airplane 是唯一要求反向的类。**双向自适应 = 把这个符号交出去，让数据/判据定。**

**(b) 物理先验需要专家标注吗？** 需要。`get_gate` 的 `OBJECT_SIZE`（8 个手写米数）
与 `VIEWPOINT_DIFF`（8 个手写视角差）都是**专家手打常数**。且实测（`gate_vs_22cells_audit.py`）：

    get_gate 作为路由器 = 11/22 = 0.500，**低于平凡基线「全开」15/22 = 0.6818**；
    recall_bad=6/7 但 recall_good=5/15 ⇒ 为了拒 6 个有害类，误关 10 个有益类。
    在论文自己的 22 格里，有 5 格（DOTA 的 storage-tank/harbor/bridge/large-vehicle/small-vehicle）
    根本没有先验 → 靠 `.get(cls, 999)` 默认关门。

⇒ **专家手写先验既贵又不管用。** 本脚本试**换成零专家、零目标域标注的自动量**。

## 本脚本做什么

对 22 个 (源域, 类) 单元格，算出若干**零目标域 GT** 的候选量（全部只依赖源域原型 + CLIP 文本），
然后检验它们能否预测两件事：

  L1 最优 β 的**符号**（双向自适应的核心：该正推还是反推）
  L2 外推是否**有益**（与 tge_soft_routing 的 gain_extB2<0 的 7 个类对齐）

★ 诚实边界：**规则本身零 GT；标签用 GT 只是为了评估规则**（这是验证零标注规则的唯一办法）。

候选量（全部零目标域 GT；其中带[源]的需要源域标注，源域标注在跨域设定下是允许的）：
  c1  cos(P_src, d_tgt)                     — 源原型已经有多像目标文本
  c2  cos(P_src, d_src_tmpl)                — 源原型有多像源文本
  c3  P_src·Δ  = c1 − c2                    — 外推还有多少余地（越大越该推）
  c4  |Δ| / cos(d_tgt, d_src_tmpl)          — 类名词义距离（纯 CLIP 文本，零图像）
  c5  cos(P_src, mean_others)  [源]         — 该类在源域池里有多"独特"
  c6  |P_neg| / cos(P_src, P_neg)  [源]     — 源域类间结构强度
  c7  spread(P_src 的类内裁剪)  [源]        — 源类内一致性
  c8  检测器侧零 GT 量：frac(vis>0) @β=2、gd 均值、框数   — 完全不碰源域

输出: 实验数据/tge_v2_bidirectional_autoprior.json
运行: C:\python-study\python39\python.exe tge_v2_bidirectional_autoprior.py
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json, pickle, time
import numpy as np
import torch

import unified_eval as U
import tge_probe as T
from tge_v2_lambda_4sources import (build_dior_protos, build_nwpu_protos,
                                    build_dota_protos, build_visdrone_protos,
                                    build_coco_cache, TARGETS, SRC_MAP)
from tge_v2_main import K_CALIB, SRC_TMPL, TGT_TMPL, DATA

BETA_JSON = DATA / 'tge_v2_beta_negative_4sources.json'
SOFT_JSON = DATA / 'tge_soft_routing.json'
OUT = DATA / 'tge_v2_bidirectional_autoprior.json'
LAM_SHOW = 0.5


def unit(x):
    x = np.asarray(x, float); n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def auc(scores, labels):
    """rank-based AUC（无 sklearn）。labels: 0/1。"""
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    if y.sum() == 0 or y.sum() == len(y):
        return float('nan')
    order = np.argsort(s)
    r = np.empty(len(s), float); r[order] = np.arange(1, len(s) + 1)
    # 平均秩处理并列
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    t0 = time.time()
    bj = json.load(open(BETA_JSON, encoding='utf-8'))
    soft = json.load(open(SOFT_JSON, encoding='utf-8'))['classes']

    print('=' * 100)
    print('双向自适应 + 自动标注：零标注量 vs 专家手写先验')
    print('=' * 100)

    print('[1/3] 源域原型池 ...')
    builders = [('DIOR', build_dior_protos), ('NWPU', build_nwpu_protos),
                ('DOTA', build_dota_protos), ('VisDrone', build_visdrone_protos)]
    protos_all = {}
    for name, fn in builders:
        ts = time.time(); protos_all[name] = fn()
        print('  %-8s %2d 类 (%.1fs)' % (name, len(protos_all[name]), time.time() - ts))

    print('[2/3] COCO 缓存 ...')
    cache = build_coco_cache()

    print('[3/3] 逐格特征 + 标签 ...')
    rows = []
    for src, _ in builders:
        protos = protos_all[src]
        for eval_name, coco_cls in TARGETS:
            src_cls = SRC_MAP[src][eval_name]
            if src_cls is None or src_cls not in protos or eval_name not in bj['sources'].get(src, {}):
                continue
            cell = bj['sources'][src][eval_name]
            P_src = protos[src_cls].cpu().numpy()
            P_src = unit(P_src)
            P_neg = T.get_neg_proto(protos, src_cls).cpu().numpy()
            P_neg = unit(P_neg)
            d_tgt = unit(T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0].numpy())
            d_src = unit(T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0].numpy())
            Delta = d_tgt - d_src
            others = np.stack([unit(v.cpu().numpy()) for k, v in protos.items() if k != src_cls])
            mu_oth = unit(others.mean(axis=0))

            # β 曲线的键在 json 里是 str(float)（如 '-4.0'），转回 float 再取值
            curve = {float(b): v for b, v in cell['beta_curve_lam%g' % LAM_SHOW].items()}
            betas = sorted(curve)
            best_beta = max(betas, key=lambda b: curve[b])

            # 源域类内一致性：原型池里该类与其它类的余弦分布
            cos_oth = others @ P_src

            # 检测器侧零 GT 量
            F = cache[eval_name]['F']
            box_sel = cache[eval_name]['per_img'][K_CALIB:]
            Fc_list = [F[p['crop_idx']] if len(p['gd']) else np.zeros((0, F.shape[1]), np.float32)
                       for p in box_sel]
            allX = np.concatenate([x for x in Fc_list if len(x)])
            gd_all = np.concatenate([np.asarray(p['gd'], float) for p in box_sel
                                     if len(p['gd'])])
            Ph2 = unit(P_src + 2.0 * Delta)
            fr2 = float(((allX @ (Ph2 - LAM_SHOW * P_neg)) > 0).mean())

            feat = {
                'c1_cos_psrc_dtgt': float(P_src @ d_tgt),
                'c2_cos_psrc_dsrc': float(P_src @ d_src),
                'c3_psrc_dot_delta': float(P_src @ Delta),
                'c4_cos_dtgt_dsrc': float(d_tgt @ d_src),
                'c4b_norm_delta': float(np.linalg.norm(Delta)),
                'c5_cos_psrc_meanothers': float(mu_oth @ P_src),
                'c6_cos_psrc_pneg': float(P_src @ P_neg),
                'c7_cos_oth_mean': float(cos_oth.mean()),
                'c7b_cos_oth_max': float(cos_oth.max()),
                'c8_frac_vis_pos_b2': fr2,
                'c8b_gd_mean': float(gd_all.mean()),
                'c8c_boxes_per_img': float(len(allX) / max(len(box_sel), 1)),
            }
            # 标签
            lab_sign = 1 if best_beta > 0 else 0
            lab_help = 1 if curve[2.0] > curve[0.0] else 0
            rows.append({'src': src, 'cls': coco_cls, 'src_cls': src_cls,
                         'best_beta': best_beta, 'lab_sign_pos': lab_sign,
                         'lab_ext_helpful': lab_help,
                         'gain_b2_vs_b0': round(curve[2.0] - curve[0.0], 4),
                         'gain_at_b2': curve[2.0],
                         'gain_at_neg': min(curve[b] for b in betas if b < 0),
                         'feats': feat})

    # ── 评估 ──
    keys = list(rows[0]['feats'].keys())
    print('\n' + '=' * 100)
    print('L1 预测「最优 β 的符号」(正=外推 / 负=反推)   正例 %d / %d'
          % (sum(r['lab_sign_pos'] for r in rows), len(rows)))
    print('=' * 100)
    print('%-26s %9s   %s' % ('候选量', 'AUC', '定向准确率(取使准确率最大的符号)'))
    l1 = {}
    for k in keys:
        v = [r['feats'][k] for r in rows]
        y = [r['lab_sign_pos'] for r in rows]
        a = auc(v, y)
        pos = sum(1 for vi, yi in zip(v, y) if (vi > 0) == (yi == 1)) / len(rows)
        l1[k] = {'auc': a, 'acc_pos_rule': pos, 'acc_max': max(pos, 1 - pos)}
        print('%-26s %9.4f   %.4f   (反向规则 %.4f)' % (k, a, pos, 1 - pos))

    print('\n' + '=' * 100)
    print('L2 预测「外推是否有益」(gain(β=2) > gain(β=0))   正例 %d / %d'
          % (sum(r['lab_ext_helpful'] for r in rows), len(rows)))
    print('=' * 100)
    print('%-26s %9s   %s' % ('候选量', 'AUC', '定向准确率'))
    l2 = {}
    for k in keys:
        v = [r['feats'][k] for r in rows]
        y = [r['lab_ext_helpful'] for r in rows]
        a = auc(v, y)
        pos = sum(1 for vi, yi in zip(v, y) if (vi > 0) == (yi == 1)) / len(rows)
        l2[k] = {'auc': a, 'acc_pos_rule': pos, 'acc_max': max(pos, 1 - pos)}
        print('%-26s %9.4f   %.4f   (反向规则 %.4f)' % (k, a, pos, 1 - pos))

    mk1 = max(l1, key=lambda k: abs(l1[k]['auc'] - 0.5))
    mk2 = max(l2, key=lambda k: abs(l2[k]['auc'] - 0.5))
    print('\n★ L1 最佳判别量 %s (AUC=%.4f)' % (mk1, l1[mk1]['auc']))
    print('★ L2 最佳判别量 %s (AUC=%.4f)' % (mk2, l2[mk2]['auc']))

    print('\n── 逐格明细 ──')
    print('%-9s %-12s %8s %8s %10s %10s %10s %9s' %
          ('src', 'cls', 'bestβ', 'sign', 'gainβ2-β0', 'c3_psrc·Δ', 'c1_psrc·dtgt', 'c8_frac'))
    for r in rows:
        print('%-9s %-12s %8g %8d %+10.4f %+10.4f %+10.4f %9.3f' %
              (r['src'], r['cls'], r['best_beta'], r['lab_sign_pos'], r['gain_b2_vs_b0'],
               r['feats']['c3_psrc_dot_delta'], r['feats']['c1_cos_psrc_dtgt'],
               r['feats']['c8_frac_vis_pos_b2']))

    json.dump({'rows': rows, 'L1_sign': l1, 'L2_helpful': l2,
               'best_L1': mk1, 'best_L2': mk2, 'elapsed_s': time.time() - t0},
              open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n→ %s  (%.0fs)' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
