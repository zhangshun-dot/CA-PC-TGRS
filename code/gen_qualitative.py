# -*- coding: utf-8 -*-
"""gen_qualitative.py — ISPRS 缺口3：CA-PC 定性检测图（成功 + 边界/作用域）

复用 run_bidirectional.py 的 CA-PC(B) 装置（bank/δ/W 完全一致，保证与论文表数字同源），
在 DIOR→COCO 正臂类 (vehicle→car/truck/bus) 上对 GDINO-tiny 跑小样本前向，渲染
base vs CA-PC 预测框 + GT，挑 success（CA-PC 把混淆框重排到正确类/位置）与
operating-range（检测器漏检、CA-PC 救不回，证适用域边界）。

用法（gdino CPU venv）:
  AC_SPC_DET_DEVICE=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "C:/Users/张三/GroundingDINO/venv/Scripts/python.exe" -u gen_qualitative.py
"""
import os, sys, json
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cpu')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import numpy as np
import torch

sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
import run_bidirectional as RB
from acspc import prototype as Proto

N_EVAL = 15          # 每类小样本（定性用，非指标）
CLS_PAIRS = [('vehicle', 'car'), ('vehicle', 'truck'), ('vehicle', 'bus')]
FIG_DIR = r'C:\Users\张三\WorkBuddy\2026-09-23-16-06-04\figs'


def build_bank_delta(cfg):
    """复刻 RB.main 阶段1-2：源原型池 + 文本偏移 δ。"""
    bank = {}
    for domain, pool in (('coco', RB.COCO_POOL), ('dior', RB.DIOR_POOL)):
        b = {}
        for cls in pool:
            ims, gts = RB.load_split(cfg, domain, cls, 0, RB.N_PROTO)
            crops = RB.D.gt_crops(ims, gts)
            if not crops:
                continue
            Ft = RB.clip_encode_imgs(crops)
            pt = torch.from_numpy(Ft.mean(0))
            b[cls] = pt / pt.norm()
        bank[domain] = b
    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    E_photo_d = {c: E_photo[i] for i, c in enumerate(all_names)}
    E_over_d = {c: E_over[i] for i, c in enumerate(all_names)}
    return bank, E_photo_d, E_over_d


def iou(box, gt):
    x1, y1, x2, y2 = box
    gx1, gy1, gx2, gy2 = gt
    ix1, iy1 = max(x1, gx1), max(y1, gy1)
    ix2, iy2 = min(x2, gx2), min(y2, gy2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(1e-9, (x2 - x1) * (y2 - y1) + (gx2 - gx1) * (gy2 - gy1) - inter)
    return inter / union


def best_iou(boxes, scores, gt, k=1):
    if len(boxes) == 0 or len(gt) == 0:
        return 0.0, -1
    order = np.argsort(-scores)[:k]
    best = 0.0
    for bi in order:
        for g in gt:
            best = max(best, iou(boxes[bi], g))
    return best, order[0]


def main():
    cfg = RB.Config()
    cfg.model.device = 'cpu'
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    bank, E_photo_d, E_over_d = build_bank_delta(cfg)
    examples = []  # (eval_cls, img, gt, box, gd, s_corr)

    for src_cls, eval_cls in CLS_PAIRS:
        # DIOR→COCO: 加载 DIOR 图像（DIOR 类 src_cls），用 COCO 类 eval_cls 的自然提示检测
        images, gts = RB.load_split(cfg, 'dior', src_cls, RB.N_PROTO, N_EVAL)
        if len(images) < 4:
            print(f'  !! {src_cls} 图像不足'); continue
        prompt = f'a photo of {eval_cls}'
        key = f'qual_{eval_cls}_off{RB.N_PROTO}_n{len(images)}'
        blob = RB.detect_encode(cfg, images, gts, prompt, key)
        per_img, F = blob['per_img'], blob['F']

        P_src = RB.unit(bank['dior'][src_cls].numpy())
        P_neg = RB.unit(Proto.get_negative_prototype(bank['dior'], src_cls, RB.NEG_TOPK).numpy())
        dc_anchor = E_photo_d[eval_cls] - E_over_d[src_cls]
        W = RB.unit(P_src + RB.BETA * dc_anchor) - RB.LAM * P_neg

        for i, p in enumerate(per_img):
            if len(p['gd']) == 0:
                examples.append((eval_cls, images[i], p['gt'], p['box'], p['gd'],
                                 np.zeros(0), 'miss'))  # 漏检/无框
                continue
            s = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, F[p['crop_idx']] @ W), 1e-6, 1.0))
            base_i, _ = best_iou(p['box'], p['gd'], p['gt'], k=1)
            corr_i, _ = best_iou(p['box'], s, p['gt'], k=1)
            tag = 'success' if (corr_i > base_i and corr_i >= 0.5) else ('miss' if base_i < 0.5 else 'neutral')
            examples.append((eval_cls, images[i], p['gt'], p['box'], p['gd'], s, tag))
        print(f'  [{eval_cls}] n={len(per_img)} '
              f'success={sum(1 for e in examples if e[-1]=="success" and e[0]==eval_cls)} '
              f'miss={sum(1 for e in examples if e[-1]=="miss" and e[0]==eval_cls)}')

    # 挑选 success 与 operating-range 代表
    succ = [e for e in examples if e[-1] == 'success']
    miss = [e for e in examples if e[-1] == 'miss']
    print(f'总 success={len(succ)} miss={len(miss)}')
    json.dump({'n_success': len(succ), 'n_miss': len(miss),
               'classes': [c for c, *_ in CLS_PAIRS]},
              open(r'C:\Users\张三\Desktop\实验数据\qualitative_summary.json', 'w'))
    # 渲染交给 draw_qual_figs.py（独立，避免 matplotlib 在本脚本拖慢）
    pickle_path = r'C:\Users\张三\Desktop\实验数据\qualitative_examples.pkl'
    import pickle
    pickle.dump(examples, open(pickle_path, 'wb'))
    print(f'  -> 实验数据/qualitative_examples.pkl ({len(examples)} 图候选)')


if __name__ == '__main__':
    main()
