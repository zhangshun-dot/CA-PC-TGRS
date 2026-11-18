# -*- coding: utf-8 -*-
"""autoproc_atext_retrieval_probe.py — A' 检索增强文本原型探针(零训练, 无目标标签)

元 2026-10-05 在 A' 两版(朴素/精炼)全败后要求再试"检索增强":
  用目标域图像特征检索文本语料, 把文本原型拉向目标视觉分布。

实现(零训练/无标签, 公平):
  1. 每类建文本语料(模板+同义+属性短语)。
  2. 对目标域 crop(load_target_coco 全部 crop, 不筛阈值)算 CLIP 视觉质心 v_c。
  3. 软检索: aug_proto = softmax(v_c @ corpus_emb^T / tau) @ corpus_emb
     —— 按"视觉质心↔短语"余弦加权, 把文本原型拉向目标视觉分布。
  4. evaluate() 比 AP50, 与单模板基线对照。

预期: 检索出的仍是冻结 CLIP 文本嵌入→仍在冻结空间, 封顶~+1.25%; 但若能稳定为正,
      是零训练下真实 modest 增益(比朴素同义版中性偏负好)。
"""
import sys, json
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import unified_eval as U
import tge_probe as T
from usc_experiment_v12 import evaluate, build_neg_proto, load_target_coco, evaluate_baseline

COCO_CLASSES = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
N_VAL = 80
TAU = 0.07
SMOKE = False
if SMOKE:
    COCO_CLASSES = ['car']
    N_VAL = 20

CORPUS = {
    'airplane':   ['a photo of an airplane', 'an aeroplane', 'an aircraft', 'a plane in the sky',
                   'a passenger jet', 'a small propeller plane'],
    'boat':       ['a photo of a boat', 'a ship', 'a sailboat', 'a motorboat',
                   'a vessel on water', 'a fishing boat'],
    'car':        ['a photo of a car', 'an automobile', 'a sedan', 'an SUV',
                   'a car on a road', 'a vehicle'],
    'truck':      ['a photo of a truck', 'a lorry', 'a pickup truck', 'a delivery truck', 'a heavy vehicle'],
    'bus':        ['a photo of a bus', 'a coach', 'a public transport bus', 'a school bus'],
    'motorcycle': ['a photo of a motorcycle', 'a motorbike', 'a two-wheeled motor vehicle', 'a racing motorcycle'],
}


def main():
    print('=' * 70)
    print('A\' 检索增强文本原型探针 (零训练, 软检索拉向目标视觉分布)')
    print(f'classes={COCO_CLASSES} N_VAL={N_VAL} tau={TAU}')
    print('=' * 70)
    rows = {}
    for c in COCO_CLASSES:
        td = load_target_coco(c, n=N_VAL)
        if td is None:
            print(f'  {c}: 跳过'); continue
        # 视觉质心: 目标域全部 crop 的 CLIP 图像特征均值
        Ft = td['features']
        if len(Ft) == 0:
            print(f'  {c}: 无 crop'); continue
        v_c = Ft.mean(axis=0)
        v_c = v_c / (np.linalg.norm(v_c) + 1e-8)
        # 语料文本嵌入
        emb = T.text_emb(CORPUS[c]).cpu().numpy()           # [K,512]
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        # 软检索权重
        logits = (v_c @ emb.T) / TAU
        w = np.exp(logits - logits.max()); w = w / w.sum()
        aug = (w @ emb); aug = aug / (np.linalg.norm(aug) + 1e-8)

        ap_base = evaluate_baseline(td)
        p_single = T.text_avg(['a photo of a {}'], c).cpu().numpy()
        neg_s = build_neg_proto({c: p_single}, c)
        neg_a = build_neg_proto({c: aug}, c)
        ap_single = evaluate(td, p_single, neg_s)
        ap_aug = evaluate(td, aug, neg_a)
        rows[c] = {'gdino_only': float(ap_base), 'single': float(ap_single),
                   'retrieval_aug': float(ap_aug), 'delta': float(ap_aug - ap_single)}
        print(f'  {c:10s} gdino={rows[c]["gdino_only"]:.4f}  single={ap_single:.4f}  '
              f'aug={ap_aug:.4f}  Δ={rows[c]["delta"]:+.4f}  (top_w={w.max():.2f})')

    out = {'config': {'method': 'A_retrieval_augmented', 'tau': TAU, 'n_val': N_VAL}, 'classes': rows}
    if rows:
        out['mean_single'] = float(np.mean([r['single'] for r in rows.values()]))
        out['mean_aug'] = float(np.mean([r['retrieval_aug'] for r in rows.values()]))
        out['mean_delta'] = out['mean_aug'] - out['mean_single']
        out['n_pos'] = int(sum(1 for r in rows.values() if r['delta'] > 0))
        out['n_neg'] = int(sum(1 for r in rows.values() if r['delta'] < 0))
    print('\n=== 汇总 ===')
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in out.items() if k in ('mean_single','mean_aug','mean_delta','n_pos','n_neg')},
                     indent=2, ensure_ascii=False))
    save = r'C:\Users\张三\Desktop\实验数据\autoproc_atext_retrieval_results.json'
    json.dump(out, open(save, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n结果 → {save}')


if __name__ == '__main__':
    main()
