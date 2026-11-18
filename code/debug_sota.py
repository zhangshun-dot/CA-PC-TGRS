# -*- coding: utf-8 -*-
"""调试SOTA对比实验"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
print('Testing imports...')
from usc_experiment_v12 import DATA, K_PCA, ALPHAS, SOURCE_LOADERS, load_target_coco, build_neg_proto, evaluate, evaluate_baseline
import tge_probe as T
from usc_rank1_affine_gpu_v2 import AffineUSC, load_domain_data
print('All imports OK')

print('Loading DIOR data...')
data = load_domain_data('DIOR')
print(f'DIOR loaded: {len(data["valid_targets"])} classes')

print('Testing evaluate_baseline...')
ap = evaluate_baseline(data['target_data']['airplane'])
print(f'Baseline AP for airplane: {ap:.4f}')

print('Testing evaluate with prototype...')
usc = data['usc']
protos_gpu = data['protos_gpu']
all_protos = data['all_protos']
text_deltas = data['text_deltas']
target_data = data['target_data']

src_cls = 'airplane'
coco_cls = 'airplane'
p_corr_gpu = usc.correct_prototype_gpu(protos_gpu[src_cls], -2.0, 0.0)
p_corr = p_corr_gpu.cpu().numpy()
neg_raw = build_neg_proto(all_protos, src_cls)
if neg_raw is not None:
    import torch
    neg_gpu = torch.tensor(neg_raw, dtype=torch.float32)
    neg_corr_gpu = usc.correct_prototype_gpu(neg_gpu, -2.0, 0.0)
    neg = neg_corr_gpu.cpu().numpy()
else:
    neg = None
delta = text_deltas[src_cls]
p_hat = p_corr + 1.0 * delta
p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)
ap = evaluate(target_data[coco_cls], p_hat, neg)
print(f'USC AP for airplane: {ap:.4f}')

print('All tests passed!')
