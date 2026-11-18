# -*- coding: utf-8 -*-
"""tge_probe_owlv2.py — 检测器独立性: OWLv2 框池 + A-CSPC 重打分 (端到端稳健性)

回答的问题（⚠️ 表述边界）: 不是"原型重打分对不同检测器都有效"——候选框池已经变了,
本实验验证的是**检测器替换下的端到端稳健性**: 换掉 GDINO 后主结论 (A-CSPC 增益) 是否成立。

协议与主表完全一致 (固定, 不因检测器变化而调整):
  N_pool=100, K=10, 后 90 图评估; 逐图 IoU 匹配 + dataset-level 11-point AP50;
  pre-filter 0.03 (OWLv2 尺度注意: v2 曾把 OWLv2 scores sigmoid 到概率尺度再过滤,
  但诊断发现 OWLv2 logits 尺度仅 [0, 0.7]: sigmoid 后 [0.5, 0.65] 无区分度,
  pre-filter 0.03 全部失效 (3600 框/图 全保留), 海量 logit=0 噪声框被抬到分数中段,
  A-CSPC 增益假性崩到 -0.48 —— 已废弃;
  v1 曾直接 clip [1e-6,1] 计分且不滤 (logit=0 框 gd=0 垫底, 无害)。
  终版: gd = OWLv2 raw logits (0~0.7, 与 GDINO 概率同数量级, 可直接比较),
  pre-filter 0.03 在 logit 尺度过滤, 恰好滤掉 logit=0 的全部噪声框 (3600->~几十/图));
  β=2, λ=0.5 不调参; sqrt(gd*vis) 融合参数不变。
检测 prompt 'a photo of {cls}' (目标域, 同主表), OWLv2 结果缓存 tge_probe_dets_OWLV2-*.pkl。
同时报告: OWLv2 baseline 与 OWLv2+A-CSPC, 以及每类候选框召回 (区分检测器变化与原型排序变化)。
源域: DIOR (主对照, 与表 1 +0.0652 直接可比) + NWPU (第二对照)。
CLIP 框特征一次性编码, 两源共享 (F 与源无关)。

输出: 实验数据/tge_probe_owlv2.json + 控制台表
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T
import tge_probe_nwpu as NW

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_probe_owlv2.json'

K = 10
BETA = 2.0
PRE_FILTER = 0.03  # 与主表统一: sigmoid 概率尺度下过滤
N_COCO = 100
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
OWLV2_CACHE = (r'C:\Users\张三\.cache\huggingface\hub'
               r'\models--google--owlv2-base-patch16-ensemble\snapshots')

TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]

# 源域原型构建器: (名称, 构建 protos dict 的函数)
SOURCES = [
    ('DIOR', lambda: build_protos_dior()),
    ('NWPU', lambda: build_protos_nwpu()),
]


def build_protos_dior():
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        protos[cls] = clip_proto(gt_crops(images, gts))
    return protos


def build_protos_nwpu():
    protos = {}
    for cls in NW.NWPU_CLASSES:
        images, gts = NW.load_nwpu(cls)
        protos[cls] = clip_proto(gt_crops(images, gts))
    return protos


def clip_proto(crops):
    if not crops:
        return None
    F = T.encode_clip_batch(crops)
    p = F.mean(dim=0)
    return p / p.norm()


def gt_crops(images, gts):
    crops = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
    return crops


# ── OWLv2 检测 ──

_owl = {}

def get_owlv2():
    if 'm' not in _owl:
        import os
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        osnap = os.listdir(OWLV2_CACHE)[0]
        opath = os.path.join(OWLV2_CACHE, osnap)
        _owl['p'] = Owlv2Processor.from_pretrained(opath)
        _owl['m'] = Owlv2ForObjectDetection.from_pretrained(
            opath, dtype=torch.float32).eval().to(U.DEVICE)
    return _owl['p'], _owl['m']


def owlv2_detect(images, prompt):
    """OWLv2 检测, 返回每图 (boxes_xyxy, scores) —— 与 U.gdino_detect 同格式。"""
    op, om = get_owlv2()
    out = []
    for img in images:
        ins = op(text=prompt, images=img, return_tensors='pt')
        ins = {k: v.to(U.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in ins.items()}
        with torch.no_grad():
            o = om(**ins)
        ts = torch.tensor([img.size[::-1]]).to(U.DEVICE)
        r = op.post_process_object_detection(o, threshold=0.0, target_sizes=ts)[0]
        out.append((r['boxes'].cpu().numpy(), r['scores'].cpu().numpy()))
    return out


def owlv2_cached_dets(images, prompt, name):
    p = DATA / f'tge_probe_dets_OWLV2-{name}.pkl'
    if p.exists():
        print(f'    [缓存] OWLv2 检测结果 {p.name}', flush=True)
        return pickle.load(open(p, 'rb'))
    dets = owlv2_detect(images, prompt)
    pickle.dump(dets, open(p, 'wb'))
    return dets


# ── 评估 ──

def collect_boxes(images, dets, gts):
    """收集每图候选框 (logit 尺度 pre-filter 0.03 + 小框过滤), 一次 CLIP 编码。

    F: 全部框的 CLIP 特征 (512,) 行; 两源共享 (W_hat 不同 → vis 不同)。
    分数尺度 (2026-08-24 诊断确认, tge_diag_owlv2.py):
      OWLv2 scores 是 logits, 实测分布 [0, 0.7]: 噪声框 logit=0, GT 框 0.2~0.6,
      无任何 logit>=1 的框。因此 gd 直接用 raw logits (与 GDINO 概率同数量级),
      pre-filter 0.03 在 logit 尺度过滤 —— 恰好滤掉 logit=0 的全部噪声框。
      v2 曾 sigmoid -> 概率尺度: sigmoid 压缩到 [0.5,0.65] 无区分度, 且 pre-filter
      完全失效 (3600 框/图 全保留), 噪声框被抬到中段, 增益假性崩到 -0.48, 废弃。
    """
    crops_all, per_img = [], []
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl = [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < PRE_FILTER:  # logit 尺度 (OWLv2 logits 与 GDINO 概率同数量级)
                continue
            bl.append([x1, y1, x2, y2])
            sl.append(sc)
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
                        'gd': (np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0)
                               if sl else np.zeros(0)),
                        'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4))})
    if crops_all:
        print('    ... %d 个 crop 待 CLIP 编码 (bs=256, %d 批, 数分钟无进度输出) ...'
              % (len(crops_all), (len(crops_all) + 255) // 256), flush=True)
        F = T.encode_clip_batch(crops_all).cpu().numpy()
    else:
        F = np.zeros((0, 512))
    fi = 0
    for p in per_img:
        n = len(p['gd'])
        p['F'] = F[fi:fi + n] if n else np.zeros((0, 512))
        fi += n
    return per_img


def score_ours(per_img, W_hat):
    """A-CSPC 分数: ours = sqrt(gd * clip(max(0, F@W_hat), 1e-6, 1))。"""
    for p in per_img:
        n = len(p['gd'])
        if n:
            vis = np.maximum(0.0, p['F'] @ W_hat)
            p['ours'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p['ours'] = np.zeros(0)


def ap_90(per_img, key):
    """后 90 图逐图 AP50 (与主表同协议)。"""
    sel = [per_img[i] for i in range(K, len(per_img))]
    return U.compute_ap50_img([(p['box'], p[key]) for p in sel], [p['gt'] for p in sel])


def recall_pf(per_img):
    """pre-filter 0.03 + 小框过滤后的 GT 召回率 (IoU>=0.5, 逐图匹配, 同 tge_recall_probe 口径)。"""
    n_gt, hit = 0, 0
    for p in per_img:
        gts = p['gt']
        boxes = p['box']
        n_gt += len(gts)
        if len(boxes) == 0:
            continue
        matched = np.zeros(len(gts), dtype=bool)
        for b in boxes:
            if matched.all():
                break
            for j, gb in enumerate(gts):
                if matched[j]:
                    continue
                if U.iou(b.tolist(), gb.tolist()) >= 0.5:
                    matched[j] = True
        hit += int(matched.sum())
    return hit / n_gt if n_gt else float('nan')


def main():
    import faulthandler
    faulthandler.enable()   # 原生崩溃(如 CUDA 驱动被杀)时也能打出调用栈, 而不是静默退出
    t0 = time.time()
    print('[1/4] 源原型 (DIOR 20 类 + NWPU 10 类) ...')
    src_protos = {}
    for name, builder in SOURCES:
        src_protos[name] = builder()
        print('    %s: %d 类原型就绪' % (name, len(src_protos[name])))

    print('[2/4] COCO 6 类: OWLv2 检测 (缓存) + CLIP 特征一次性编码 ...')
    class_data = []
    for name, coco_cls, src_cls in TARGETS:
        images, gts = U.load_coco(coco_cls, n=N_COCO)
        dets = owlv2_cached_dets(images, TGT_TMPL.format(coco_cls), name)
        per_img = collect_boxes(images, dets, gts)
        n_boxes = sum(len(p['gd']) for p in per_img)
        print('    %-16s N=%3d 框=%6d (sigmoid + pre-filter 0.03 后)' % (name, len(images), n_boxes),
              flush=True)
        class_data.append({'name': name, 'coco_cls': coco_cls, 'src_cls': src_cls,
                           'per_img': per_img, 'n_test': len(images) - K})

    print('[3/4] A-CSPC 重打分 (β=%.1f, λ=%.1f 固定) ...' % (BETA, U.NEG_LAM))
    results = {sname: {} for sname, _ in SOURCES}
    for sname, _ in SOURCES:
        protos = src_protos[sname]
        for d in class_data:
            name, coco_cls, src_cls = d['name'], d['coco_cls'], d['src_cls']
            P_src = protos[src_cls].to(U.DEVICE)
            P_neg = T.get_neg_proto(protos, src_cls)
            s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
            d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
            P_hat = P_src + BETA * (d_y - s_xs).to(U.DEVICE)
            P_hat = P_hat / P_hat.norm()
            W_hat = (P_hat.cpu().numpy() - U.NEG_LAM * P_neg.cpu().numpy())
            score_ours(d['per_img'], W_hat)
            ap_base = ap_90(d['per_img'], 'gd')
            ap_ours = ap_90(d['per_img'], 'ours')
            results[sname][name] = {'baseline': round(ap_base, 4), 'acspc': round(ap_ours, 4),
                                    'gain': round(ap_ours - ap_base, 4)}
        # 召回与源无关, 在 DIOR 源时算一次
        if sname == 'DIOR':
            for d in class_data:
                results[sname][d['name']]['recall'] = round(recall_pf(d['per_img']), 4)

    print('[4/4] 汇总写盘 ...')
    out = {'config': {'detector': 'OWLv2-base-patch16-ensemble', 'target': 'COCO val2017',
                      'n_coco': N_COCO, 'k': K, 'beta': BETA, 'neg_lam': U.NEG_LAM,
                      'pre_filter': PRE_FILTER,
                      'score_scale': 'OWLv2 raw logits (实测 [0,0.7]: 噪声框 logit=0, GT 框 0.2~0.6, '
                                     '无 logit>=1; 与 GDINO 概率同数量级; '
                                     'sigmoid 会压缩到 [0.5,0.65] 致 pre-filter 失效, 已废弃)',
                      'protocol': 'OWLv2 boxes + A-CSPC rerank, N=100/类, K=10, 后90图, '
                                  '逐图 AP50, logit 尺度 pre-filter 0.03 (滤掉全部 logit=0 噪声框), '
                                  'sqrt(gd*vis), 参数与主表完全一致',
                      'question': 'end-to-end robustness under detector replacement '
                                  '(box pool changes; NOT "reranking works on any detector")'},
           'sources': {}}
    for sname, _ in SOURCES:
        rows = results[sname]
        mean_b = float(np.mean([rows[n]['baseline'] for n in rows]))
        mean_o = float(np.mean([rows[n]['acspc'] for n in rows]))
        out['sources'][sname] = {'classes': rows,
                                 'mean': {'baseline': round(mean_b, 4),
                                          'acspc': round(mean_o, 4),
                                          'gain': round(mean_o - mean_b, 4)}}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print()
    for sname, _ in SOURCES:
        rows = results[sname]
        print('=' * 100)
        print('%s 源 (OWLv2 检测器): %-16s %9s %9s %9s %9s' % (sname, 'class', 'baseline',
                                                                'A-CSPC', '增益', '召回'))
        print('-' * 100)
        for name in [t[0] for t in TARGETS]:
            r = rows[name]
            rec = r.get('recall', float('nan'))
            print('%-16s %9.4f %9.4f %+9.4f %9.4f' % (name, r['baseline'], r['acspc'],
                                                      r['gain'], rec))
        m = out['sources'][sname]['mean']
        print('-' * 100)
        print('%-16s %9.4f %9.4f %+9.4f %9s   <- mean' % ('mean', m['baseline'], m['acspc'],
                                                          m['gain'], '—'))
    print('=' * 100)
    print('GDINO 对照 (主表): DIOR 源 baseline 0.3809 / A-CSPC 0.4461 (+0.0652); NWPU 源 +0.0589')
    print('elapsed %.1f s -> %s' % (time.time() - t0, OUT))


if __name__ == '__main__':
    main()
