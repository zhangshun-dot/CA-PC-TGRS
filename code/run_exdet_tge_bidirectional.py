# -*- coding: utf-8 -*-
"""run_exdet_tge_bidirectional.py — 忠实复现 ExDet 的 TGE (DeltaSpace 域外推) 并在本协议双向跑

背景: ExDet (arXiv 2606.09360) 代码未公开 (CatalyzeX "Request Code", GitHub 无仓库)。
本文按原文公式复现其 TGE 模块, 在 CA-PC 的双向协议 (DIOR<->COCO, GDINO-tiny, 11pt AP50,
复用 run_bidirectional 检测缓存) 下评估, 验证"DeltaSpace/TGE 外推本质朝目标域、故天然双向"。

ExDet 公式 (ar5iv 2606.09360):
  Eq.1 类别外推:  v_hat_n = v_bar_b + (t_n - t_b)        # t 用通用/自然(photo)文本
  Eq.2 域外推:    v_hat^D = v_hat + (t^D_{c,j} - t_c)    # 目标域文本 - 源域文本, 加在源视觉原型
  => 总偏移 = (t_目标类 - t_源类) [类别] + (t^D_目标类 - t_源域文本_目标类) [域]

本协议映射: 通用文本=photo; 源/目标"域"由迁移方向定 (fwd: 源=overhead/DIOR, 目标=photo/COCO;
rev: 源=photo/COCO, 目标=overhead/DIOR)。

用法 (GPU, 原型编码仅一次, 评测重排走缓存):
  HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cuda \
  "C:/Users/张三/vg/Scripts/python.exe" -u run_exdet_tge_bidirectional.py
"""
import os, sys, json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cuda')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import numpy as np
import run_bidirectional as RB

OUT = r'C:\Users\张三\Desktop\实验数据\run_exdet_tge_bidirectional.json'


def main():
    cfg = RB.Config()
    cfg.model.device = RB.DEVICE
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    # 源原型池 (CLIP crop mean, 同 CA-PC)
    bank = {}
    for domain, pool in (('coco', RB.COCO_POOL), ('dior', RB.DIOR_POOL)):
        b = {}
        for cls in pool:
            ims, gts = RB.load_split(cfg, domain, cls, 0, RB.N_PROTO)
            crops = RB.D.gt_crops(ims, gts)
            if not crops:
                continue
            Ft = RB.clip_encode_imgs(crops)
            pt = RB.torch.from_numpy(Ft.mean(0))
            b[cls] = pt / pt.norm()
        bank[domain] = b
        print(f'[protos] {domain}: {len(b)} 类')

    # 文本嵌入: photo=通用, overhead=遥感域
    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    Ep = {c: E_photo[i] for i, c in enumerate(all_names)}
    Eo = {c: E_over[i] for i, c in enumerate(all_names)}

    # 检测缓存
    blobs = {}
    for tname in RB.TASKS:
        src_dom, tgt_dom, pairs = RB.TASKS[tname]
        for src_cls, eval_cls in pairs:
            images, gts = RB.load_split(cfg, tgt_dom, eval_cls, RB.N_PROTO, 100)
            if len(images) < 8:
                continue
            prompt = (f'a photo of {eval_cls}' if tgt_dom == 'coco'
                      else f'an overhead image of {eval_cls}')
            key = f'{tname}_{eval_cls}_off{RB.N_PROTO}_n{len(images)}'
            blobs[(tname, eval_cls)] = (src_cls, RB.detect_encode(cfg, images, gts, prompt, key))

    BETA, LAM, NEG_TOPK = RB.BETA, RB.LAM, RB.NEG_TOPK

    def make_TGE(src_dom, sc, tgt_cls):
        """ExDet TGE: 类别外推(photo文本差) + 域外推(目标域文本-源域文本), 加在源视觉原型。"""
        P_src = RB.unit(bank[src_dom][sc].numpy())
        P_neg = RB.unit(RB.Proto.get_negative_prototype(bank[src_dom], sc, NEG_TOPK).numpy())
        # 类别外推 (Eq.1): t_目标类 - t_源类, 均用 photo(通用)文本
        cat_off = Ep[tgt_cls] - Ep[sc]
        if src_dom == 'dior':   # fwd: 源=overhead, 目标=photo
            dom_off = Ep[tgt_cls] - Eo[tgt_cls]      # 目标域(photo) - 源域(overhead)
        else:                   # rev: 源=photo, 目标=overhead
            dom_off = Eo[tgt_cls] - Ep[tgt_cls]      # 目标域(overhead) - 源域(photo)
        W = RB.unit(P_src + BETA * (cat_off + dom_off)) - LAM * P_neg
        return W

    def make_CA_PC(src_dom, sc, tgt_cls):
        """CA-PC 规范锚定 (A): 类别迁移 + 全局域项 d_bar, 目标类钉 photo/overhead 由方向定。"""
        P_src = RB.unit(bank[src_dom][sc].numpy())
        P_neg = RB.unit(RB.Proto.get_negative_prototype(bank[src_dom], sc, NEG_TOPK).numpy())
        d_cls = Ep[tgt_cls] - Ep[sc]
        d_bar = np.mean([Ep[c] - Eo[c] for c in all_names], axis=0)
        W = RB.unit(P_src + BETA * d_cls + BETA * d_bar) - LAM * P_neg
        return W

    # 评估
    results = {}
    for tname in RB.TASKS:
        src_dom, tgt_dom, pairs = RB.TASKS[tname]
        results[tname] = {}
        for src_cls, eval_cls in pairs:
            if (tname, eval_cls) not in blobs:
                continue
            sc, blob = blobs[(tname, eval_cls)]
            per_img, F = blob['per_img'], blob['F']
            base = RB.ap_base(per_img, F)
            tge = RB.ap_of(per_img, F, make_TGE(src_dom, sc, eval_cls))
            capc = RB.ap_of(per_img, F, make_CA_PC(src_dom, sc, eval_cls))
            results[tname][f'{sc}->{eval_cls}'] = {
                'baseline': round(base, 4),
                'TGE_gain': round(tge - base, 4),
                'CA_PC_gain': round(capc - base, 4),
            }
            print(f'  {tname} {sc}->{eval_cls}: base={base:.4f} '
                  f'TGE={tge-base:+.4f}  CA_PC={capc-base:+.4f}', flush=True)

    def mean_gain(d, k):
        return round(float(np.mean([v[k] for v in d.values()])), 4)

    summary = {t: {'TGE_mean': mean_gain(results[t], 'TGE_gain'),
                   'CA_PC_mean': mean_gain(results[t], 'CA_PC_gain')}
               for t in results}
    out = {'config': {'method': 'ExDet TGE reimpl (Eq.1+Eq.2, arXiv 2606.09360)',
                      'note': 'ExDet 代码未公开, 此为按论文公式忠实复现; 与 CA-PC 同协议双向评测',
                      'detector': 'gdino-tiny', 'beta': BETA, 'lam': LAM},
            'per_class': results, 'summary': summary}
    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('\n' + '=' * 84)
    print('ExDet TGE (复现) vs CA-PC  双向对比 (相对 baseline 增益均值)')
    print('=' * 84)
    for t in summary:
        print(f'  [{t}]  TGE={summary[t]["TGE_mean"]:+.4f}   CA_PC={summary[t]["CA_PC_mean"]:+.4f}')
    print(f'\n已保存: {OUT}')


if __name__ == '__main__':
    main()
