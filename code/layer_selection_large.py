# -*- coding: utf-8 -*-
"""layer_selection_large.py — 层选择协议（大样本版, COCO 100 张/类）

与小样本版 layer_selection.py 的差异:
  COCO 跨域目标域扩到 100 张/类(airplane 97 全用), 验证集 VAL_K=20,
  测试集 77~80 张, 让 few-shot 选层更稳健, 逼近 oracle。
  DIOR 同域仍 20 张/类, 固定 L12(本地无完整测试集)。

主协议: AP 选层(验证集 AP50 最高层); AUC 选层作对照。
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import unified_eval as U

VAL_K = 20           # 跨域每类留 20 张验证图
N_COCO_TARGET = 100  # COCO 每类取 100 张
LAM = 0.5
FIXED_LAYER = 11


def get_neg_proto(bank, dior_cls, layer):
    pos = bank[dior_cls][layer]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][layer]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return neg / neg.norm()


def encode_all_layers(crop):
    dino, pre = U.get_dinov2()
    t = pre(crop).unsqueeze(0).to(U.DEVICE)
    with torch.no_grad():
        o = dino(t)
    return [o.hidden_states[l + 1][0, 0, :] for l in range(12)]


def encode_class(images, gts, dets):
    boxes_all, gd_all, feats_all, labels_all, img_idx = [], [], [], [], []
    for img_i, ((boxes, scores), img, g) in enumerate(zip(dets, images, gts)):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            continue
        for box, sc in zip(boxes[keep], scores[keep]):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            feats = [f / f.norm() for f in encode_all_layers(img.crop((x1, y1, x2, y2)))]
            lab = 0
            for gt in g:
                if U.iou([x1, y1, x2, y2], gt) >= 0.5:
                    lab = 1
                    break
            boxes_all.append([x1, y1, x2, y2])
            gd_all.append(float(sc))
            feats_all.append(feats)
            labels_all.append(lab)
            img_idx.append(img_i)
    return boxes_all, gd_all, feats_all, labels_all, img_idx


def vis_for_layer(feats_all, layer, pos, neg):
    return np.array([max(0.0, float((f[layer] @ pos).item()) - LAM * float((f[layer] @ neg).item()))
                     for f in feats_all])


def auc_rank(vis, labels):
    vis = np.array(vis); labels = np.array(labels)
    pos_mask = labels == 1
    if pos_mask.sum() == 0 or (~pos_mask).sum() == 0:
        return float('nan')
    order = np.argsort(vis)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(vis))
    n_pos = pos_mask.sum()
    return (rank[pos_mask].mean() - (n_pos - 1) / 2) / (~pos_mask).sum()


def main():
    bank = U.get_bank()
    summary = {}
    for name, dataset, cls, domain, prefix, dior_cls in U.EVALS:
        print('\n' + '=' * 70)
        print(f'[{name}]  {"跨域" if dataset == "coco" else "同域"}({domain})')
        if dataset == 'coco':
            images, gts = U.load_coco(cls, n=N_COCO_TARGET)
        else:
            images, gts = U.load_dior(cls)
        dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)
        n_img = len(images)

        boxes_all, gd_all, feats_all, labels_all, img_idx = encode_class(images, gts, dets)
        gt_flat = [g for gs in gts for g in gs]
        gd = np.clip(np.array(gd_all), 1e-6, 1.0)
        print(f'  {n_img} 图, {len(boxes_all)} 候选框, {len(gt_flat)} GT')

        def gt_of_img(img_mask):
            return [g for gi, gs in enumerate(gts) for g in gs if img_mask[gi]]

        base = U.compute_ap50(boxes_all, gd, gt_flat)
        row = {'baseline': base}
        print(f'  baseline          = {base:.4f}')

        if dataset == 'dior':
            l = FIXED_LAYER
            pos = bank[dior_cls][l]['prototype'].to(U.DEVICE)
            neg = get_neg_proto(bank, dior_cls, l)
            vis = vis_for_layer(feats_all, l, pos, neg)
            ap = U.compute_ap50(boxes_all, np.sqrt(gd * np.clip(vis, 1e-6, 1.0)), gt_flat)
            row.update({'L12固定': ap, 'AUC选层': ap, 'AP选层': ap, 'oracle': ap})
            print(f'  L12固定           = {ap:.4f}  (同域默认)')
        else:
            val_img = np.array([i < VAL_K for i in range(n_img)])
            test_img = ~val_img
            val_mask = np.array([val_img[i] for i in img_idx])
            test_mask = ~val_mask
            gt_val = gt_of_img(val_img)
            gt_test = gt_of_img(test_img)

            # 固定 L12 (test)
            l = FIXED_LAYER
            pos = bank[dior_cls][l]['prototype'].to(U.DEVICE)
            neg = get_neg_proto(bank, dior_cls, l)
            vis = vis_for_layer([f for i, f in enumerate(feats_all) if test_mask[i]], l, pos, neg)
            ap12 = U.compute_ap50(
                [boxes_all[i] for i in range(len(boxes_all)) if test_mask[i]],
                np.sqrt(gd[test_mask] * np.clip(vis, 1e-6, 1.0)), gt_test)
            print(f'  L12固定(test)     = {ap12:.4f}')

            layer_auc, layer_ap_val, layer_ap_all = [], [], []
            for l in range(12):
                pos = bank[dior_cls][l]['prototype'].to(U.DEVICE)
                neg = get_neg_proto(bank, dior_cls, l)
                vis_val = vis_for_layer([f for i, f in enumerate(feats_all) if val_mask[i]], l, pos, neg)
                labs_val = [lab for i, lab in enumerate(labels_all) if val_mask[i]]
                layer_auc.append(auc_rank(vis_val, labs_val))
                gd_val = gd[val_mask]
                b_val = [boxes_all[i] for i in range(len(boxes_all)) if val_mask[i]]
                layer_ap_val.append(U.compute_ap50(b_val, np.sqrt(gd_val * np.clip(vis_val, 1e-6, 1.0)), gt_val))
                vis_all_l = vis_for_layer(feats_all, l, pos, neg)
                layer_ap_all.append(U.compute_ap50(boxes_all, np.sqrt(gd * np.clip(vis_all_l, 1e-6, 1.0)), gt_flat))

            l_auc = int(np.nanargmax(np.array(layer_auc)))
            l_ap = int(np.argmax(layer_ap_val))
            l_oracle = int(np.argmax(layer_ap_all))
            print(f'  val AP 各层: ' + ' '.join(f'L{k+1}={v:.3f}' for k, v in enumerate(layer_ap_val)))
            print(f'  → AUC 选层 L{l_auc+1}, AP 选层 L{l_ap+1}, oracle L{l_oracle+1}')

            def test_ap(l):
                pos = bank[dior_cls][l]['prototype'].to(U.DEVICE)
                neg = get_neg_proto(bank, dior_cls, l)
                vis = vis_for_layer([f for i, f in enumerate(feats_all) if test_mask[i]], l, pos, neg)
                return U.compute_ap50(
                    [boxes_all[i] for i in range(len(boxes_all)) if test_mask[i]],
                    np.sqrt(gd[test_mask] * np.clip(vis, 1e-6, 1.0)), gt_test)

            ap_auc = test_ap(l_auc)
            ap_ap = test_ap(l_ap)
            ap_oracle = test_ap(l_oracle)
            print(f'  test AP: L12固定={ap12:.4f} | AUC选层={ap_auc:.4f}(L{l_auc+1}) | '
                  f'AP选层={ap_ap:.4f}(L{l_ap+1}) | oracle={ap_oracle:.4f}(L{l_oracle+1})')
            row.update({'L12固定': ap12, 'AUC选层': ap_auc, 'AP选层': ap_ap, 'oracle': ap_oracle})

        summary[name] = row

    cols = ['baseline', 'L12固定', 'AUC选层', 'AP选层', 'oracle']
    print('\n\n' + '=' * 90)
    print(f'{"Test":<16s}' + ''.join(f'{c:>12s}' for c in cols))
    print('-' * 90)
    for name, r in summary.items():
        print(f'{name:<16s}' + ''.join(f'{r[c]:>12.4f}' for c in cols))
    print('-' * 90)
    for label, sel in [('总均值', list(summary)),
                       ('跨域均值', [n for n in summary if n.startswith('COCO')])]:
        print(f'{label:<16s}' + ''.join(f'{np.mean([summary[n][c] for n in sel]):>12.4f}' for c in cols))

    print('\nΔ vs baseline:')
    print(f'{"Test":<16s}' + ''.join(f'{c:>12s}' for c in cols[1:]))
    for name, r in summary.items():
        print(f'{name:<16s}' + ''.join(f'{r[c]-r["baseline"]:>+12.4f}' for c in cols[1:]))

    print('\n✅ 层选择大样本版完成')


if __name__ == '__main__':
    main()
