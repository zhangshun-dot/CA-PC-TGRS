# -*- coding: utf-8 -*-
"""冒烟测试：COCO airplane 10 张，跑 baseline + vep + cdvm，验证 unified_eval 能跑通。"""
import sys, time
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import unified_eval as U

t0 = time.time()
images, gts = U.load_coco('airplane', 10)
print(f'[1/3] 数据: {len(images)} 图, {sum(len(g) for g in gts)} GT, {time.time()-t0:.0f}s')

base_dets = U.gdino_detect(images, 'a photo of airplane', 0.0)
bb, bs, bgt = U.filter_flatten(base_dets, gts)
ap_base = U.compute_ap50(bb, bs, bgt)
print(f'[2/3] baseline AP50={ap_base:.4f} ({len(bb)} 候选框), {time.time()-t0:.0f}s')

ap_vep = U.rerank_visual(images, base_dets, gts, 'airplane', use_neg=False)
print(f'[3/3] vep  AP50={ap_vep:.4f}, {time.time()-t0:.0f}s')

ap_cdvm = U.rerank_visual(images, base_dets, gts, 'airplane', use_neg=True)
print(f'      cdvm AP50={ap_cdvm:.4f}, {time.time()-t0:.0f}s')

print('SMOKE OK' if ap_vep >= 0 and ap_cdvm >= 0 else 'SMOKE FAIL')
