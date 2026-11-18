# -*- coding: utf-8 -*-
"""dual_signal_probe.py — 双信号（GDINO文本 × CDVM视觉）一致性探测

问题（P10 后续，协议审计之后）:
  伪标签原型 P_top20 只用单一 vis 分数选框。route_proxy_probe 已证伪
  「无标签单分数统计量预测 AUC 路由」。本探测换角度：不预测路由，而是问
  双信号一致性（gd 文本分 ∩ vis 视觉分 都高的框）能否【无标签】选出更干净
  的原型框，从而提升下游 AP。

双信号直觉: gd（预训练 GDINO 置信度）与 vis（我们算的 CDVM 原型相似度）
  是两个独立判据。单判据 top-k 都混入大量负例（boat vis AUC≈0.48），
  但两判据【交集】的框更可能是真目标 → 伪标签 precision 更高 → 原型更纯。

协议（跑前写死，用户规定）:
  ① 独立无标签适配集定规则（选框策略/α 固定，不看测试集真值）
  ② 测试集只做最终 AP
  ③ 阈值/规则不碰测试真值
  ④ 报告 precision / 原型 cosine / 正负 gap / AP
  ⑤ 结果按 source-only / transductive / inductive 三区块展示

三区块定义:
  - source-only      : P_src（源域 bank），测试集 AP。基线，可部署。
  - transductive     : 原型在【测试集自身】选框计算，测试集 AP。用测试集
                       特征分布（无标签），不可部署，作上界对照。
  - inductive        : 原型在【独立适配集】选框计算，测试集 AP。可部署，
                       是双信号探测的真正战场。
  - inductive K-shot : P_tgt 从 few-shot 集真 GT 框提取。有监督上界。
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
import torch
import unified_eval as U

LAYER = 11
K = 10          # few-shot 校准图数
N_ADAPT = 45    # 无标签适配集图数
TOPK = 20       # 选框 k（对齐 P10 top-20）
N_COCO = 100

TARGETS = [
    ('COCO-airplane', 'airplane', 'airplane', 'a photo of'),
    ('COCO-boat',     'boat',     'ship',                         'a photo of'),
    ('COCO-car',      'car',      'vehicle',                      'a photo of'),
]


def get_neg_proto(bank, dior_cls, layer):
    pos = bank[dior_cls][layer]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][layer]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return neg / neg.norm()


def norm(p):
    return p / p.norm()


def encode(crop):
    return U.encode_dinov2(crop, LAYER)


def vis_score(f, P, neg):
    """CDVM 视觉分数（负原型相对化）。"""
    if f is None:
        return 0.0
    return max(0.0, float((f @ P).item()) - U.NEG_LAM * float((f @ neg).item()))


def detect_encode(images, prompt, P_src, neg):
    """对一组图跑 GDINO 检测 + DINOv2 编码，返回扁平候选框列表。

    返回: boxes(list), gd(list), vis(list), feats(list)
    gd = GDINO 文本分数（双信号之一），vis = CDVM source-only 视觉分数（双信号之二）。
    """
    dets = U.gdino_detect(images, prompt, 0.0)
    boxes_all, gd_all, vis_all, feats_all = [], [], [], []
    for (boxes, scores), img in zip(dets, images):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            continue
        for box, sc in zip(boxes[keep], scores[keep]):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                boxes_all.append([x1, y1, x2, y2]); gd_all.append(float(sc))
                vis_all.append(0.0); feats_all.append(None)
                continue
            f = encode(img.crop((x1, y1, x2, y2)))
            boxes_all.append([x1, y1, x2, y2]); gd_all.append(float(sc))
            feats_all.append(f)
            vis_all.append(vis_score(f, P_src, neg))
    return boxes_all, gd_all, vis_all, feats_all


def label_boxes(boxes, gts_flat):
    """用真值给候选框打标（仅诊断，不参与选框规则）。"""
    return [1 if any(U.iou(b, gt) >= 0.5 for gt in gts_flat) else 0 for b in boxes]


def gt_proto(images, gts):
    """从真 GT 框提取原型。"""
    feats = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            feats.append(encode(img.crop((x1, y1, x2, y2))))
    return norm(torch.stack(feats).mean(dim=0)) if feats else None


def select_proto(feats, idxs):
    """从选中索引的特征算原型，返回 (proto, n_selected)。"""
    vals = [feats[i] for i in idxs if feats[i] is not None]
    if not vals:
        return None, len(vals)
    return norm(torch.stack(vals).mean(dim=0)), len(vals)


def diagnose(proto, P_gt, idxs, feats_all, labs_all, neg):
    """诊断选中框的质量（仅报告，用真值）：precision、原型cosine、正负gap。"""
    n = len(idxs)
    n_pos = sum(1 for i in idxs if labs_all[i] == 1)
    prec = n_pos / n if n else float('nan')
    cos = float(proto @ P_gt) if (proto is not None and P_gt is not None) else float('nan')
    # gap: 原型 proto 对整个适配集框的判别力（正例 vis 均值 − 负例 vis 均值）
    if proto is None:
        gap = float('nan')
    else:
        vv = [vis_score(f, proto, neg) for f in feats_all]
        pos = [v for v, l in zip(vv, labs_all) if l == 1]
        negs = [v for v, l in zip(vv, labs_all) if l == 0]
        gap = (np.mean(pos) - np.mean(negs)) if (pos and negs) else float('nan')
    return prec, cos, gap, n


def cdvm_ap(boxes, feats, gd, P_cal, neg, gt_flat):
    """用 P_cal 重打分（fused=sqrt(gd*vis)），算数据集级 AP50。"""
    fused = []
    for f, g in zip(feats, gd):
        if f is None:
            fused.append(0.0)
        else:
            v = vis_score(f, P_cal, neg)
            fused.append(float(np.sqrt(np.clip(g, 1e-6, 1.0) * np.clip(v, 1e-6, 1.0))))
    return U.compute_ap50(boxes, fused, gt_flat)


def build_strategies(gd, vis):
    """在某个无标签集上构建四种选框策略（全部不用真值）。"""
    n = len(gd)
    order_vis = np.argsort(-np.asarray(vis))
    order_gd = np.argsort(-np.asarray(gd))
    fused = np.array([np.sqrt(np.clip(g, 1e-6, 1.0) * np.clip(v, 1e-6, 1.0)) for g, v in zip(gd, vis)])
    order_fused = np.argsort(-fused)
    set_vis = set(order_vis[:TOPK].tolist())
    set_gd = set(order_gd[:TOPK].tolist())
    return {
        'vis_top20':      list(order_vis[:TOPK]),
        'gd_top20':       list(order_gd[:TOPK]),
        'both_intersect': [i for i in range(n) if i in set_vis and i in set_gd],
        'fused_top20':    list(order_fused[:TOPK]),
    }


def eval_protos(strategies, feats, t_boxes, t_feats, t_gd, t_gt_flat, P_src, neg):
    """对一组选框策略算原型，返回各策略在测试集上的 AP（α=0 纯 / α=0.5 混合）。"""
    out = {}
    for sname, idxs in strategies.items():
        P_rule, _ = select_proto(feats, idxs)
        if P_rule is None:
            out[sname] = {'a0.0': None, 'a0.5': None}
            continue
        ap0 = cdvm_ap(t_boxes, t_feats, t_gd, P_rule, neg, t_gt_flat)
        ap05 = cdvm_ap(t_boxes, t_feats, t_gd, norm(0.5 * P_src + 0.5 * P_rule), neg, t_gt_flat)
        out[sname] = {'a0.0': round(ap0, 4), 'a0.5': round(ap05, 4)}
    return out


def main():
    bank = U.get_bank()
    report = {}

    for name, cls, dior_cls, prefix in TARGETS:
        print('\n' + '=' * 72)
        print(f'[{name}]')
        images, gts = U.load_coco(cls, n=N_COCO)
        fs_imgs, fs_gts = images[:K], gts[:K]
        rest_imgs, rest_gts = images[K:], gts[K:]
        adapt_imgs, adapt_gts = rest_imgs[:N_ADAPT], rest_gts[:N_ADAPT]
        test_imgs, test_gts = rest_imgs[N_ADAPT:], rest_gts[N_ADAPT:]
        print(f'  共 {len(images)} 图: few-shot {len(fs_imgs)} / 适配集 {len(adapt_imgs)} / 测试集 {len(test_imgs)}')

        P_src = bank[dior_cls][LAYER]['prototype'].to(U.DEVICE)
        neg = get_neg_proto(bank, dior_cls, LAYER)
        prompt = f'{prefix} {cls}'

        P_tgt = gt_proto(fs_imgs, fs_gts)                 # few-shot 有监督上界
        P_gt_adapt = gt_proto(adapt_imgs, adapt_gts)      # 适配集真 GT 原型（诊断基准）
        print(f'  源域-目标域原型 cosine = {float(P_src @ P_tgt):.4f}' if P_tgt is not None else '  P_tgt 缺失')

        # 适配集：检测+编码（无标签，用于选框算原型）
        a_boxes, a_gd, a_vis, a_feats = detect_encode(adapt_imgs, prompt, P_src, neg)
        a_gts_flat = [g for gs in adapt_gts for g in gs]
        a_labs = label_boxes(a_boxes, a_gts_flat)
        # 测试集：检测+编码（只算最终 AP）
        t_boxes, t_gd, t_vis, t_feats = detect_encode(test_imgs, prompt, P_src, neg)
        t_gt_flat = [g for gs in test_gts for g in gs]

        # 适配集选框策略 + 原型 + 诊断
        strat_a = build_strategies(a_gd, a_vis)
        protos_adapt = {}
        diag = {}
        for sname, idxs in strat_a.items():
            P_rule, nn = select_proto(a_feats, idxs)
            protos_adapt[sname] = P_rule
            prec, cos, gap, _ = diagnose(P_rule, P_gt_adapt, idxs, a_feats, a_labs, neg)
            diag[sname] = {'precision': round(prec, 3), 'cos_gt': round(cos, 3),
                           'pos_neg_gap': round(gap, 3), 'n_selected': nn}
            print(f'  [适配集 {sname:<16s}] n={nn:3d}  prec={prec:.3f}  cos_gt={cos:.3f}  gap={gap:.3f}')

        row = {'diag': diag}

        # --- source-only ---
        base = cdvm_ap(t_boxes, t_feats, t_gd, P_src, neg, t_gt_flat)
        row['source_only'] = round(base, 4)
        print(f'  [source-only]            AP={base:.4f}')

        # --- inductive（适配集算原型，测试集 AP，可部署）---
        row['inductive'] = eval_protos(strat_a, a_feats, t_boxes, t_feats, t_gd, t_gt_flat, P_src, neg)
        for sname, v in row['inductive'].items():
            print(f'  [inductive {sname:<16s}] α=0:{v["a0.0"]}  α=0.5:{v["a0.5"]}')

        # --- transductive（测试集自身选框，不可部署，上界对照）---
        strat_t = build_strategies(t_gd, t_vis)
        row['transductive'] = eval_protos(strat_t, t_feats, t_boxes, t_feats, t_gd, t_gt_flat, P_src, neg)
        for sname, v in row['transductive'].items():
            print(f'  [transductive {sname:<16s}] α=0:{v["a0.0"]}  α=0.5:{v["a0.5"]}')

        # --- inductive K-shot（few-shot 真 GT，有监督上界）---
        if P_tgt is not None:
            ap0 = cdvm_ap(t_boxes, t_feats, t_gd, P_tgt, neg, t_gt_flat)
            ap05 = cdvm_ap(t_boxes, t_feats, t_gd, norm(0.5 * P_src + 0.5 * P_tgt), neg, t_gt_flat)
            row['inductive_Kshot'] = {'a0.0': round(ap0, 4), 'a0.5': round(ap05, 4)}
            print(f'  [inductive K-shot真GT]   α=0:{ap0:.4f}  α=0.5:{ap05:.4f}')

        report[name] = row

    # ===== 汇总（三区块对比表）=====
    print('\n\n' + '=' * 72)
    print('汇总（测试集 AP50，三区块）:')
    strat_names = ['vis_top20', 'gd_top20', 'both_intersect', 'fused_top20']
    for name, r in report.items():
        print(f'\n【{name}】 source-only={r["source_only"]}')
        print('  区块          策略              α=0      α=0.5')
        for sname in strat_names:
            ind = r['inductive'].get(sname, {})
            trd = r['transductive'].get(sname, {})
            print(f'  inductive      {sname:<16s} {ind.get("a0.0")}  {ind.get("a0.5")}')
        for sname in strat_names:
            trd = r['transductive'].get(sname, {})
            print(f'  transductive   {sname:<16s} {trd.get("a0.0")}  {trd.get("a0.5")}')
        ks = r.get('inductive_Kshot', {})
        print(f'  K-shot(真GT)   {"—":<16s} {ks.get("a0.0")}  {ks.get("a0.5")}')

    SAVE = Path(r'C:\Users\张三\Desktop\实验数据') / 'dual_signal_probe.json'
    with open(SAVE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'\n结果 → {SAVE}')
    print('✅ 双信号一致性探测完成')


if __name__ == '__main__':
    main()
