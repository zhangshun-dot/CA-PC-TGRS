# -*- coding: utf-8 -*-
"""tge_probe_ap.py — 阶段2 线A: CLIP 空间外推原型重打分 AP50 (最终裁决第一枪)

在 unified_eval 协议 (GDINO th=0 -> pre-filter 0.03 -> dataset-level AP50,
sqrt(gd*vis) 融合) 下, 把 CLIP 空间的外推原型当 CDVM 正原型用, 直接测 AP50。

回答核心问题: AUC 层面的外推增益能否穿过"绝对分数"这关?
  - 真判别: 正例绝对分数健康, sqrt(gd*vis) 后 AP50 提升
  - 假提升: 正例分数也塌 (只是负例塌更快), AP50 不升反降 ← fusion_ablation 教训

对照:
  baseline — GDINO 原始分数 AP50
  B2       — CLIP 真实源原型重打分 (控制换编码器)
  B3_joint — 外推 joint@2.0 (默认口径, 不挑)
  B3_best  — 每类最优变体@β (上限口径, 标注配置)

复用 tge_probe.py 的检测缓存 (tge_probe_dets_*.pkl), 只重算 CLIP 特征 (~15min)。
输出: 实验数据/tge_probe_ap.json + 控制台表
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
import torch
import clip
import unified_eval as U
import tge_probe as T   # 复用 encode_clip_batch / text_emb / get_neg_proto / cached_dets

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_probe_ap_st.json'     # 2026-08-22: 单模板主口径 + λ 消融 (多模板已判负回退)
PROBE = DATA / 'tge_probe_12.json'   # 读 B3_best 配置

N_COCO = 500
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'

TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]   # 2026-08-22 λ 消融 (负原型相对化权重, dinov2 遗产)


def get_clip():
    m, pre = U.get_clip()
    return m, pre


def main():
    # 1) DIOR CLIP 源原型 (同 tge_probe 协议)
    print('[1/3] DIOR 20 类 CLIP 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        crops = []
        for img, g in zip(images, gts):
            for box in g:
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 > x1 + 10 and y2 > y1 + 10:
                    crops.append(img.crop((x1, y1, x2, y2)))
        if not crops:
            continue
        F = T.encode_clip_batch(crops)
        p = F.mean(dim=0)
        protos[cls] = p / p.norm()

    # 2) 读 B3_best 配置 (变体@β)
    probe = json.load(open(PROBE, encoding='utf-8'))
    best_cfg = {}
    for name, coco_cls, src_cls in TARGETS:
        r = probe['classes'][name]
        v_best, b_best, a_best = None, None, -1
        for v in ['joint', 'cat', 'dom']:
            for b_str, a in r['B3_extrap'][v].items():
                if a > a_best:
                    v_best, b_best, a_best = v, float(b_str), a
        best_cfg[name] = (v_best, b_best, a_best)

    # 3) 每类: 特征 + vis + 融合 AP
    print('\n[2/3] COCO 6 类 AP50 ...')
    rows = {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = U.load_coco(coco_cls, n=N_COCO)
        dets = T.cached_dets(images, TGT_TMPL.format(coco_cls), name)

        # 候选框特征 + 标签 (记录 img_ids: 逐图 AP50 需要)
        crops, boxes_all, gd_all, lab_all, img_ids = [], [], [], [], []
        for ii, ((boxes, scores), img, g) in enumerate(zip(dets, images, gts)):
            keep = scores > U.PRE_FILTER
            if not keep.any():
                continue
            for box, score in zip(boxes[keep], scores[keep]):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                lab = 1 if any(U.iou([x1, y1, x2, y2], gt) >= 0.5 for gt in g) else 0
                boxes_all.append([x1, y1, x2, y2])
                gd_all.append(float(score))
                lab_all.append(lab)
                img_ids.append(ii)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    crops.append(None)
                else:
                    crops.append(img.crop((x1, y1, x2, y2)))
        real_idx = [i for i, c in enumerate(crops) if c is not None]
        enc = T.encode_clip_batch([crops[i] for i in real_idx]).cpu().numpy()
        F = np.zeros((len(crops), enc.shape[1]))
        for k, i in enumerate(real_idx):
            F[i] = enc[k]
        gd = np.array(gd_all)
        labels = np.array(lab_all)
        boxes_all = np.array(boxes_all)

        # 原型
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        # 2026-08-22: 回退单模板 (多模板负结果: 模板平均稀释域对比度)
        s_xs = T.text_emb([T.SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([T.TGT_TMPL.format(coco_cls)]).cpu()[0]
        d_xs = T.text_emb([T.TGT_TMPL.format(src_cls)]).cpu()[0]
        deltas = {'joint': d_y - s_xs, 'cat': d_y - d_xs, 'dom': d_xs - s_xs}

        def fused_ap(P_vec, beta=1.0, extra=0.0, neg_lam=U.NEG_LAM):
            """sqrt(gd * max(0, f·P − λ f·P_neg)) 融合 → AP50 (unified_eval 协议)。"""
            P_vec = P_vec.cpu() if isinstance(P_vec, torch.Tensor) else P_vec
            extra = extra.cpu() if isinstance(extra, torch.Tensor) else extra
            P = (P_vec + beta * extra).to(U.DEVICE)
            P = P / P.norm()
            vis = np.maximum(0.0, F @ P.cpu().numpy() - neg_lam * (F @ P_neg.cpu().numpy()))
            vc = np.clip(vis, 1e-6, 1.0)
            gc = np.clip(gd, 1e-6, 1.0)
            fused = np.sqrt(gc * vc)
            return U.compute_ap50_img(U.split_by_img(boxes_all, fused, img_ids, len(images)), gts)

        # baseline (GDINO 原始, 逐图匹配)
        ap_base = U.compute_ap50_img(U.split_by_img(boxes_all, gd, img_ids, len(images)), gts)
        # B2 (CLIP 真实原型 + 负原型)
        ap_b2 = fused_ap(P_src)
        # B2_noneg: 源原型无负原型 (2×2 归因矩阵左上角, 隔离"负原型本身"的贡献)
        ap_b2_noneg = fused_ap(P_src, neg_lam=0.0)
        # B3 joint@2.0 (默认口径)
        ap_j2 = fused_ap(P_src, 2.0, deltas['joint'])
        # B3_best (上限口径)
        v_b, b_b, _ = best_cfg[name]
        ap_best = fused_ap(P_src, b_b, deltas[v_b])
        # T4: 纯文本原型 d(y) 直接打分 = CLIP 零样本分类 (方法成立性关键对照!)
        ap_text = fused_ap(d_y / d_y.norm())
        # λ 消融: joint@2 扫负原型权重 (λ=0 → 无负原型项, 测 CDVM 遗产的存续价值)
        lam_scan = {str(l): round(fused_ap(P_src, 2.0, deltas['joint'], l), 4)
                    for l in LAMS}

        rows[name] = {
            'baseline': round(ap_base, 4), 'B2_clip': round(ap_b2, 4),
            'B2_noneg': round(ap_b2_noneg, 4),
            'B3_joint2': round(ap_j2, 4), 'B3_best': round(ap_best, 4),
            'T4_text': round(ap_text, 4), 'lam_scan': lam_scan,
            'best_cfg': f'{v_b}@{b_b}',
        }
        print(f'  {name:<16s} base={ap_base:.4f}  B2={ap_b2:.4f}  B2_noneg={ap_b2_noneg:.4f}  '
              f'joint2={ap_j2:.4f}  best({v_b}@{b_b})={ap_best:.4f}  T4_text={ap_text:.4f}')
        print(f'    λ 消融: ' + '  '.join(f'λ={l}:{lam_scan[str(l)]:.4f}' for l in LAMS))

    # 汇总
    print('\n' + '=' * 100)
    print(f'{"类":<16s} {"baseline":>9s} {"B2 clip":>9s} {"joint@2":>9s} {"best":>9s} {"T4 text":>9s}  cfg')
    print('-' * 100)
    for name, coco_cls, src_cls in TARGETS:
        r = rows[name]
        print(f'{name:<16s} {r["baseline"]:>9.4f} {r["B2_clip"]:>9.4f} '
              f'{r["B3_joint2"]:>9.4f} {r["B3_best"]:>9.4f} {r["T4_text"]:>9.4f}  {r["best_cfg"]}')

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'n_coco': N_COCO, 'neg_lam': U.NEG_LAM,
                              'protocol': 'gdino th=0 -> pre-filter 0.03 -> image-safe per-image AP50 '
                                          '(2026-08-23 修正), sqrt(gd*vis)'},
                   'classes': rows}, f, indent=2, ensure_ascii=False)
    print(f'\n✅ -> {OUT}')


if __name__ == '__main__':
    main()
