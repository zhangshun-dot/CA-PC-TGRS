# -*- coding: utf-8 -*-
"""autoproc_atext_probe.py — A' 文本侧富集最小探针（零训练, 守约束）

元 2026-10-05 选定 A': 多模板 + 同义/源概念扩充文本原型, 替换单句 prompt。
零训练、不碰特征空间, 定位=modest 且一致的文本原型增益(仍在冻结空间, 不破 +1.25% 天花板)。

协议: enriched 文本原型直接塞 evaluate() 比 AP50, 与单模板冻结原型对照。
公平点: 当前方法文本打分就是单模板(=基线), enriched 是其超集, delta=真实零训练增益。
"""
import sys, json
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import unified_eval as U
import tge_probe as T
from usc_experiment_v12 import evaluate, build_neg_proto, load_target_coco, evaluate_baseline

COCO_CLASSES = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
N_VAL = 80
SMOKE = False
if SMOKE:
    COCO_CLASSES = ['car']
    N_VAL = 20

# 每类: 单模板(基线) + 扩充模板(仅类专属近义词, 去掉跨类共享泛化词避免糊类)
SINGLE = ['a photo of a {}']
ENRICH = {
    'airplane':   ['a photo of an airplane', 'a photo of an aeroplane', 'an aircraft'],
    'boat':       ['a photo of a boat', 'a photo of a ship'],          # boat←源 ship
    'car':        ['a photo of a car', 'a photo of an automobile'],
    'truck':      ['a photo of a truck', 'a photo of a lorry'],
    'bus':        ['a photo of a bus', 'a coach'],
    'motorcycle': ['a photo of a motorcycle', 'a motorbike'],
}


def main():
    print('=' * 70)
    print('A\' 文本侧富集探针 (零训练)')
    print(f'classes={COCO_CLASSES} N_VAL={N_VAL}')
    print('=' * 70)
    rows = {}
    for c in COCO_CLASSES:
        td = load_target_coco(c, n=N_VAL)
        if td is None:
            print(f'  {c}: 跳过'); continue
        ap_base = evaluate_baseline(td)
        p_single = T.text_avg(SINGLE, c).cpu().numpy()
        p_enr = T.text_avg(ENRICH[c], c).cpu().numpy()
        neg_s = build_neg_proto({c: p_single}, c) if False else build_neg_proto({c: p_single}, c)
        neg_e = build_neg_proto({c: p_enr}, c)
        ap_single = evaluate(td, p_single, neg_s)
        ap_enr = evaluate(td, p_enr, neg_e)
        rows[c] = {'gdino_only': float(ap_base), 'single': float(ap_single),
                   'enriched': float(ap_enr), 'delta': float(ap_enr - ap_single)}
        print(f'  {c:10s} gdino={rows[c]["gdino_only"]:.4f}  single={ap_single:.4f}  '
              f'enriched={ap_enr:.4f}  Δ={rows[c]["delta"]:+.4f}')

    out = {'config': {'method': 'A_text_enrichment', 'n_val': N_VAL}, 'classes': rows}
    if rows:
        out['mean_single'] = float(np.mean([r['single'] for r in rows.values()]))
        out['mean_enriched'] = float(np.mean([r['enriched'] for r in rows.values()]))
        out['mean_delta'] = out['mean_enriched'] - out['mean_single']
        out['n_pos'] = int(sum(1 for r in rows.values() if r['delta'] > 0))
        out['n_neg'] = int(sum(1 for r in rows.values() if r['delta'] < 0))
    print('\n=== 汇总 ===')
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in out.items() if k in ('mean_single','mean_enriched','mean_delta','n_pos','n_neg')},
                     indent=2, ensure_ascii=False))
    save = r'C:\Users\张三\Desktop\实验数据\autoproc_atext_probe_results.json'
    json.dump(out, open(save, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n结果 → {save}')


if __name__ == '__main__':
    main()
