# -*- coding: utf-8 -*-
"""run_reverse_extra.py — 补齐反向 COCO→DIOR 全量主表缺失的 3 对 EXTRA
(memory: run_bidirectional 的 rev_coco2dior 只含 CORE 3 对, 缺 truck/bus/motorcycle→vehicle,
与正向 fwd_dior2coco 的 6 对不对称。)

做法: 复用 run_bidirectional 全部函数与常量, 仅对反向 eval_cls=vehicle 已存在的
检测缓存 (rev_coco2dior_vehicle_off20_n100.pkl, gdino-tiny) 重算 CA-PC AP,
零 GPU 重跑。输出到独立文件 run_reverse_extra.json, 不覆盖 run_bidirectional.json。
"""
import sys, json, pickle, numpy as np
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import run_bidirectional as RB
import torch

RB.DEVICE = 'cpu'
cfg = RB.Config()
cfg.model.device = 'cpu'
cfg.model.detection_threshold = 0.0
cfg.model.pre_filter_threshold = RB.PRE_FILTER
cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

# 源原型池 (GT-crop CLIP, 与 run_bidirectional.main 完全一致)
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

# 反向 vehicle 检测缓存 (gdino, off20, n100) — 已存在, 不重跑 GPU
tname, eval_cls = 'rev_coco2dior', 'vehicle'
cp = RB.CACHE_DIR / f'{tname}_{eval_cls}_off{RB.N_PROTO}_n100.pkl'
print('cache exists?', cp.exists())
blob = pickle.load(open(cp, 'rb'))
per_img, F = blob['per_img'], blob['F']

# 缺失的 3 对 (与正向 EXTRA 对称): COCO 源 → DIOR vehicle 目标
extra = [('truck', 'vehicle'), ('bus', 'vehicle'), ('motorcycle', 'vehicle')]
out = {'config': {'beta': RB.BETA, 'lam': RB.LAM, 'n_proto': RB.N_PROTO,
                 'note': 'reverse EXTRA pairs, reuse rev_coco2dior_vehicle cache (zero GPU)'},
       'results': {}}
for sc, ev in extra:
    P_src = RB.unit(bank['coco'][sc].numpy())
    P_neg = RB.unit(RB.Proto.get_negative_prototype(bank['coco'], sc, RB.NEG_TOPK).numpy())
    dc_anchor = E_photo_d[ev] - E_over_d[sc]   # photo(target) - overhead(source) = 反向 +δ
    base = RB.ap_base(per_img, F)
    Wplus = RB.unit(P_src + RB.BETA * dc_anchor) - RB.LAM * P_neg
    ap = RB.ap_of(per_img, F, Wplus)
    r = {'baseline': round(float(base), 4), 'n_img': len(per_img),
         'plus_delta': round(float(ap), 4)}
    r['gain'] = round(r['plus_delta'] - r['baseline'], 4)
    out['results'][f'{sc}->{ev}'] = r
    print(f'  {sc}->{ev}: base={r["baseline"]:.4f} +delta={r["plus_delta"]:.4f} '
          f'gain={r["gain"]:+.4f} n_img={r["n_img"]}')

json.dump(out, open(r'C:\Users\张三\Desktop\实验数据\run_reverse_extra.json', 'w'),
          indent=2, ensure_ascii=False)
print('saved run_reverse_extra.json')
