# -*- coding: utf-8 -*-
"""layer_selection.py — 层选择组件（诚实协议）

协议:
  同域(DIOR→DIOR)   : 固定 L12（无监督默认, 源域语义深层最优）
  跨域(DIOR→COCO)   : 每类留 5 张目标域验证图, 扫 12 层选层, 剩余图评 AP

选层指标（两种对比）:
  AUC 选层 : 验证图候选框的 CDVM vis 分数区分正/负例框的分离度 AUC 最大层
  AP  选层 : 验证图数据集级 AP50 最大层（5 张图, 噪声大, 作对照）
  oracle   : 全部图 AP50 最大层（作弊上界, 仅参考）

一次 forward 编码 12 层特征, 选层/评 AP 纯矩阵运算, 不重复 encode。
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import unified_eval as U

VAL_K = 5            # 跨域每类留 5 张验证图
LAM = 0.5            # CDVM 负原型系数
FIXED_LAYER = 11     # 同域固定 L12


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
    """编码全部候选框 12 层特征 + 框级标签 + 图索引。"""
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
    vis = []
    for feats in feats_all:
        f = feats[layer]
        vis.append(max(0.0, float((f @ pos).item()) - LAM * float((f @ neg).item())))
    return np.array(vis)


def auc_rank(vis, labels):
    """rank-based AUC (1=完美分离, 0.5=随机)。"""
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
        images, gts = U.load_coco(cls) if dataset == 'coco' else U.load_dior(cls)
        dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)
        n_img = len(images)

        boxes_all, gd_all, feats_all, labels_all, img_idx = encode_class(images, gts, dets)
        gt_flat = [g for gs in gts for g in gs]
        gd = np.clip(np.array(gd_all), 1e-6, 1.0)
        print(f'  {n_img} 图, {len(boxes_all)} 候选框, {len(gt_flat)} GT')

        # 图级 GT（按图索引, img_mask 长度=n_img）
        def gt_of_img(img_mask):
            return [g for gi, gs in enumerate(gts) for g in gs if img_mask[gi]]

        # baseline（全部图）
        base = U.compute_ap50(boxes_all, gd, gt_flat)
        row = {'baseline': base}
        print(f'  baseline          = {base:.4f}')

        if dataset == 'dior':
            # 固定 L12, 不选层
            l = FIXED_LAYER
            pos = bank[dior_cls][l]['prototype'].to(U.DEVICE)
            neg = get_neg_proto(bank, dior_cls, l)
            vis = vis_for_layer(feats_all, l, pos, neg)
            ap = U.compute_ap50(boxes_all, np.sqrt(gd * np.clip(vis, 1e-6, 1.0)), gt_flat)
            row.update({'L12固定': ap, 'AUC选层': ap, 'AP选层': ap, 'oracle': ap})  # 同域固定 L12
            print(f'  L12固定           = {ap:.4f}  (同域默认)')
        else:
            # 跨域: val=前 VAL_K 张, test=其余（图级 mask 与框级 mask 分开）
            val_img = np.array([i < VAL_K for i in range(n_img)])
            test_img = ~val_img
            val_mask = np.array([val_img[i] for i in img_idx])   # 框级
            test_mask = ~val_mask
            gt_val = gt_of_img(val_img)
            gt_test = gt_of_img(test_img)

            # 固定 L12 (test 集)
            l = FIXED_LAYER
            pos = bank[dior_cls][l]['prototype'].to(U.DEVICE)
            neg = get_neg_proto(bank, dior_cls, l)
            vis = vis_for_layer([f for i, f in enumerate(feats_all) if test_mask[i]], l, pos, neg)
            ap12 = U.compute_ap50(
                [boxes_all[i] for i in range(len(boxes_all)) if test_mask[i]],
                np.sqrt(gd[test_mask] * np.clip(vis, 1e-6, 1.0)), gt_test)
            print(f'  L12固定(test)     = {ap12:.4f}')

            # 逐层 val AUC + val AP + 全图 AP(oracle)
            layer_auc, layer_ap_val, layer_ap_all = [], [], []
            for l in range(12):
                pos = bank[dior_cls][l]['prototype'].to(U.DEVICE)
                neg = get_neg_proto(bank, dior_cls, l)
                # val AUC
                vis_val = vis_for_layer([f for i, f in enumerate(feats_all) if val_mask[i]], l, pos, neg)
                labs_val = [lab for i, lab in enumerate(labels_all) if val_mask[i]]
                layer_auc.append(auc_rank(vis_val, labs_val))
                # val AP
                gd_val = gd[val_mask]
                b_val = [boxes_all[i] for i in range(len(boxes_all)) if val_mask[i]]
                layer_ap_val.append(U.compute_ap50(b_val, np.sqrt(gd_val * np.clip(vis_val, 1e-6, 1.0)), gt_val))
                # 全图 AP (oracle 用)
                vis_all_l = vis_for_layer(feats_all, l, pos, neg)
                layer_ap_all.append(U.compute_ap50(boxes_all, np.sqrt(gd * np.clip(vis_all_l, 1e-6, 1.0)), gt_flat))

            # 选层
            l_auc = int(np.nanargmax(np.array(layer_auc)))
            l_ap = int(np.argmax(layer_ap_val))
            l_oracle = int(np.argmax(layer_ap_all))
            print(f'  val AUC 各层: ' + ' '.join(f'L{k+1}={v:.2f}' for k, v in enumerate(layer_auc)))
            print(f'  → AUC 选层 L{l_auc+1}, AP 选层 L{l_ap+1}, oracle L{l_oracle+1}')

            # 各选层策略在 test 集评 AP
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

    # 汇总
    print('\n\n' + '=' * 90)
    cols = ['baseline', 'L12固定', 'AUC选层', 'AP选层', 'oracle']
    print(f'{"Test":<16s}' + ''.join(f'{c:>12s}' for c in cols))
    print('-' * 90)
    for name, r in summary.items():
        print(f'{name:<16s}' + ''.join(f'{r[c]:>12.4f}' for c in cols))
    print('-' * 90)
    for label, sel in [('总均值', list(summary)),
                       ('跨域均值', [n for n in summary if n.startswith('COCO')])]:
        print(f'{label:<16s}' + ''.join(
            f'{np.mean([summary[n][c] for n in sel]):>12.4f}' for c in cols))

    # Δ vs baseline
    print('\nΔ vs baseline:')
    print(f'{"Test":<16s}' + ''.join(f'{c:>12s}' for c in cols[1:]))
    for name, r in summary.items():
        print(f'{name:<16s}' + ''.join(f'{r[c]-r["baseline"]:>+12.4f}' for c in cols[1:]))

    print('\n✅ 层选择协议完成')


if __name__ == '__main__':
    main()
