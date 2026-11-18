# -*- coding: utf-8 -*-
"""final_method_table.py — 最终方法主表（few-shot 跨域 OVD）

设定:
  源域 DIOR(大量标注) → 目标域 COCO(每类 K=10 张 few-shot 标注) → 剩余测试

方法:
  baseline      — GDINO 原始分数
  CDVM 零样本   — 纯 DIOR 源域原型, α=1 (P_cal = P_src)
  CDVM few-shot — α=0.5 校准 (P_cal = 0.5·P_src + 0.5·P_tgt)

跨域(COCO): few-shot 用前 K 张, 测试用后 90 张
同域(DIOR):  源域=目标域, 零样本即最优, 20 张全测
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import unified_eval as U

LAYER = 11
K = 10
NEG_LAM = 0.5
ALPHA = 0.5
N_COCO = 100


def get_neg_proto(bank, dior_cls, layer):
    pos = bank[dior_cls][layer]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][layer]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return neg / neg.norm()


def encode_l12(crop):
    dino, pre = U.get_dinov2()
    t = pre(crop).unsqueeze(0).to(U.DEVICE)
    with torch.no_grad():
        o = dino(t)
    f = o.hidden_states[LAYER + 1][0, 0, :]
    return f / f.norm()


def extract_fewshot_proto(images, gts):
    feats = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            feats.append(encode_l12(img.crop((x1, y1, x2, y2))))
    if not feats:
        return None
    p = torch.stack(feats).mean(dim=0)
    return p / p.norm()


def eval_class(images, gts, dets, P_cal, neg):
    """逐图 AP50 (2026-08-23: 全局拼接匹配改为 per-image 匹配 + 全局排序)。"""
    boxes_all, gd_all, feats, img_ids = [], [], [], []
    for ii, ((boxes, scores), img) in enumerate(zip(dets, images)):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            continue
        for box, sc in zip(boxes[keep], scores[keep]):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            feats.append(encode_l12(img.crop((x1, y1, x2, y2))))
            boxes_all.append([x1, y1, x2, y2])
            gd_all.append(float(sc))
            img_ids.append(ii)
    gd = np.clip(np.array(gd_all), 1e-6, 1.0)
    base = U.compute_ap50_img(U.split_by_img(boxes_all, gd, img_ids, len(images)), gts)
    vis = [max(0.0, float((f @ P_cal).item()) - NEG_LAM * float((f @ neg).item())) for f in feats]
    fused = np.sqrt(gd * np.clip(vis, 1e-6, 1.0))
    ap = U.compute_ap50_img(U.split_by_img(boxes_all, fused, img_ids, len(images)), gts)
    return base, ap


# 2026-08-23: 扩展为 6 COCO 类 (truck/bus/moto 同 vehicle 源原型, 与主表口径一致),
# 与 U.EVALS 的 3+3 保持同协议 (prompt/前缀一致)
TARGETS = [
    ('COCO-airplane',   'coco', 'airplane',   'a photo of',          'airplane'),
    ('COCO-boat',       'coco', 'boat',       'a photo of',          'ship'),
    ('COCO-car',        'coco', 'car',        'a photo of',          'vehicle'),
    ('COCO-truck',      'coco', 'truck',      'a photo of',          'vehicle'),
    ('COCO-bus',        'coco', 'bus',        'a photo of',          'vehicle'),
    ('COCO-motorcycle', 'coco', 'motorcycle', 'a photo of',          'vehicle'),
    ('DIOR-airplane',   'dior', 'airplane',   'an overhead image of', 'airplane'),
    ('DIOR-ship',       'dior', 'ship',       'an overhead image of', 'ship'),
    ('DIOR-vehicle',    'dior', 'vehicle',    'an overhead image of', 'vehicle'),
]


def main():
    bank = U.get_bank()
    summary = {}
    for name, dataset, cls, prefix, dior_cls in TARGETS:
        print('\n' + '=' * 70)
        print(f'[{name}]  {"跨域" if dataset == "coco" else "同域"}')
        if dataset == 'coco':
            images, gts = U.load_coco(cls, n=N_COCO)
        else:
            images, gts = U.load_dior(cls)
        dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)

        P_src = bank[dior_cls][LAYER]['prototype'].to(U.DEVICE)
        neg = get_neg_proto(bank, dior_cls, LAYER)

        if dataset == 'coco':
            fs_imgs, fs_gts = images[:K], gts[:K]
            test_imgs, test_gts = images[K:], gts[K:]
            test_dets = dets[K:]
            P_tgt = extract_fewshot_proto(fs_imgs, fs_gts)
            # 零样本 (α=1)
            base, ap_zs = eval_class(test_imgs, test_gts, test_dets, P_src, neg)
            # few-shot 校准 (α=0.5)
            P_cal = (ALPHA * P_src + (1 - ALPHA) * P_tgt)
            P_cal = P_cal / P_cal.norm()
            _, ap_fs = eval_class(test_imgs, test_gts, test_dets, P_cal, neg)
            print(f'  baseline={base:.4f} | 零样本={ap_zs:.4f} | few-shot={ap_fs:.4f}')
            summary[name] = {'baseline': base, 'zero': ap_zs, 'fewshot': ap_fs}
        else:
            base, ap_zs = eval_class(images, gts, dets, P_src, neg)
            print(f'  baseline={base:.4f} | 零样本={ap_zs:.4f} (同域)')
            summary[name] = {'baseline': base, 'zero': ap_zs, 'fewshot': ap_zs}

    cols = ['baseline', 'zero', 'fewshot']
    print('\n\n' + '=' * 90)
    print(f'{"Test":<16s}' + ''.join(f'{c:>14s}' for c in ['baseline', 'CDVM(零样本)', 'CDVM(few-shot)']))
    print('-' * 90)
    for name, r in summary.items():
        print(f'{name:<16s}' + ''.join(f'{r[c]:>14.4f}' for c in cols))
    print('-' * 90)
    for label, sel in [('跨域均值', [n for n in summary if n.startswith('COCO')]),
                       ('同域均值', [n for n in summary if n.startswith('DIOR')]),
                       ('总均值', list(summary))]:
        print(f'{label:<16s}' + ''.join(f'{np.mean([summary[n][c] for n in sel]):>14.4f}' for c in cols))

    print('\nΔ vs baseline:')
    print(f'{"Test":<16s}' + ''.join(f'{c:>14s}' for c in ['zero', 'fewshot']))
    for name, r in summary.items():
        print(f'{name:<16s}' + ''.join(f'{r[c]-r["baseline"]:>+14.4f}' for c in ['zero', 'fewshot']))

    print('\n✅ 最终主表完成')


if __name__ == '__main__':
    main()
