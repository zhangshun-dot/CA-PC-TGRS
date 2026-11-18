# -*- coding: utf-8 -*-
"""draw_qual_figs.py — 渲染 CA-PC 定性图（成功 + 作用域边界）from qualitative_examples.pkl

每图三行（car/truck/bus 各一例），两列：左=base(GDINO-tiny 未校正)，右=CA-PC(dc×conf)。
绿框=GT；蓝/红框=top-3 预测（按分数），框旁标注分数。
输出 figs/qual_success.pdf（CA-PC 把混淆框重排到正确位置）与
      figs/qual_operating.pdf（检测器漏检/召回崩塌，CA-PC 救不回，证适用域边界）。
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PKL = r'C:\Users\张三\Desktop\实验数据\qualitative_examples.pkl'
FIG_DIR = r'C:\Users\张三\WorkBuddy\2026-09-23-16-06-04\figs'
NROW, NCOL = 3, 2
TOPK = 3


def draw_ax(ax, img, gt, boxes, scores, title):
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    # GT green
    for g in gt:
        x1, y1, x2, y2 = g
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor='#2ca02c', lw=2))
    # predictions
    if len(boxes):
        order = np.argsort(-scores)[:TOPK]
        cmap = {0: '#1f77b4', 1: '#ff7f0e', 2: '#9467bd'}
        for rank, bi in enumerate(order):
            x1, y1, x2, y2 = boxes[bi]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=cmap.get(rank, '#555555'), lw=1.8))
            ax.text(x1, max(2, y1 - 3), f'{scores[bi]:.2f}', color=cmap.get(rank, '#555'),
                    fontsize=7)


def collect(examples, tag, want, limit=3):
    out = []
    for e in examples:
        if e[-1] == tag and e[0] in want and len(out) < limit:
            out.append(e)
    # 若某类不足，补其他类
    if len(out) < limit:
        for e in examples:
            if e[-1] == tag and e not in out and len(out) < limit:
                out.append(e)
    return out


def render(examples, tag, fname, title):
    want = ['car', 'truck', 'bus']
    sel = collect(examples, tag, want, limit=NROW)
    if not sel:
        print(f'  无 {tag} 样本，跳过 {fname}'); return
    fig, axes = plt.subplots(len(sel), NCOL, figsize=(9, 3.0 * len(sel)))
    if len(sel) == 1:
        axes = np.array([axes])
    for r, (eval_cls, img, gt, box, gd, s, _) in enumerate(sel):
        draw_ax(axes[r, 0], img, gt, box, gd, f'{eval_cls}  base')
        draw_ax(axes[r, 1], img, gt, box, s, f'{eval_cls}  CA-PC')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  -> {path}')


def main():
    examples = pickle.load(open(PKL, 'rb'))
    render(examples, 'success', 'qual_success.pdf',
           'CA-PC success cases: prototype correction re-ranks confused overhead vehicles')
    render(examples, 'miss', 'qual_operating.pdf',
           'Operating range: detector recall collapse on overhead targets; CA-PC cannot recover missed detections')


if __name__ == '__main__':
    main()
