# -*- coding: utf-8 -*-
"""run_bidirectional_seeds.py — 主实验多种子稳健性 (bootstrap 评测图)

复用 run_bidirectional 的检测缓存 (probe_bidir_cache/*.pkl, 已含 per_img + F),
仅对评测图像做 bootstrap 重采样, 重算各变体 AP50 增益, 给出 mean±std。
目的: 检验主表关键数字是否显著而非噪声 (尤其反向方向特异增量 +0.026)。

用法 (GPU, 原型编码仅一次, 评测重排走缓存):
  HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cuda N_SEEDS=20 \
  "C:/Users/张三/vg/Scripts/python.exe" -u run_bidirectional_seeds.py
"""
import os, sys, json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cuda')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import numpy as np
import run_bidirectional as RB

K = int(os.environ.get('N_SEEDS', '20'))
VARIANTS = ['plus_delta', 'minus_delta', 'beta0', 'rand0', 'rand1', 'rand2']
OUT = r'C:\Users\张三\Desktop\实验数据\run_bidirectional_seeds.json'


def main():
    cfg = RB.Config()
    cfg.model.device = RB.DEVICE
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    # ── 源原型池 (offset 0, 与主实验一致) ──
    bank = {}
    for domain, pool in (('coco', RB.COCO_POOL), ('dior', RB.DIOR_POOL)):
        b = {}
        for cls in pool:
            ims, gts = RB.load_split(cfg, domain, cls, 0, RB.N_PROTO)
            crops = RB.D.gt_crops(ims, gts)
            if not crops:
                print(f'  !! {domain}:{cls} 无 crops'); continue
            Ft = RB.clip_encode_imgs(crops)
            pt = RB.torch.from_numpy(Ft.mean(0))
            b[cls] = pt / pt.norm()
        bank[domain] = b
        print(f'[protos] {domain}: {len(b)} 类')

    # ── δ ──
    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    E_photo_d = {c: E_photo[i] for i, c in enumerate(all_names)}
    E_over_d = {c: E_over[i] for i, c in enumerate(all_names)}
    nrm = float(np.mean([np.linalg.norm(E_photo_d[c] - E_over_d[c]) for c in all_names]))

    # ── 取检测缓存 (已落盘, 不重跑 GPU 检测) ──
    blobs = {}
    for tname in RB.TASKS:
        src_dom, tgt_dom, pairs = RB.TASKS[tname]
        for src_cls, eval_cls in pairs:
            images, gts = RB.load_split(cfg, tgt_dom, eval_cls, RB.N_PROTO, 100)
            if len(images) < 8:
                print(f'  !! {tname}/{eval_cls} 图像不足'); continue
            prompt = (f'a photo of {eval_cls}' if tgt_dom == 'coco'
                      else f'an overhead image of {eval_cls}')
            key = f'{tname}_{eval_cls}_off{RB.N_PROTO}_n{len(images)}'
            blobs[(tname, eval_cls)] = (src_cls, RB.detect_encode(cfg, images, gts, prompt, key))

    BETA, LAM, NEG_TOPK = RB.BETA, RB.LAM, RB.NEG_TOPK
    rng_rand = np.random.default_rng(0)
    rand_vecs = [RB.unit(rng_rand.normal(size=512)) * nrm for _ in range(RB.N_RAND)]

    def make_W(src_dom, sc, eval_cls):
        P_src = RB.unit(bank[src_dom][sc].numpy())
        P_neg = RB.unit(RB.Proto.get_negative_prototype(bank[src_dom], sc, NEG_TOPK).numpy())
        dc_anchor = E_photo_d[eval_cls] - E_over_d[sc]
        dc_roleswap = E_photo_d[sc] - E_over_d[eval_cls]
        W = {
            'plus_delta': RB.unit(P_src + BETA * dc_anchor) - LAM * P_neg,
            'minus_delta': RB.unit(P_src + BETA * dc_roleswap) - LAM * P_neg,
            'beta0': P_src - LAM * P_neg,
        }
        for i, u in enumerate(rand_vecs):
            W[f'rand{i}'] = RB.unit(P_src + BETA * u) - LAM * P_neg
        return W

    # ── bootstrap ──
    rec = {t: {f'{sc}->{ec}': {k: [] for k in VARIANTS}
               for sc, ec in RB.TASKS[t][2]} for t in RB.TASKS}
    for b in range(K):
        rng = np.random.default_rng(b + 1)
        for tname in RB.TASKS:
            src_dom, tgt_dom, pairs = RB.TASKS[tname]
            for src_cls, eval_cls in pairs:
                if (tname, eval_cls) not in blobs:
                    continue
                sc, blob = blobs[(tname, eval_cls)]
                per_img, F = blob['per_img'], blob['F']
                idx = rng.integers(0, len(per_img), len(per_img))
                sub = [per_img[i] for i in idx]
                base = RB.ap_base(sub, F)
                W = make_W(src_dom, sc, eval_cls)
                for k in VARIANTS:
                    gain = RB.ap_of(sub, F, W[k]) - base
                    rec[tname][f'{sc}->{eval_cls}'][k].append(round(float(gain), 4))
        print(f'  bootstrap {b + 1}/{K} done', flush=True)

    # ── 聚合 ──
    def stats(vals):
        a = np.array(vals, float)
        return {'mean': round(float(a.mean()), 4), 'std': round(float(a.std(ddof=1)), 4)}

    agg = {}
    for tname in rec:
        agg[tname] = {}
        for key, d in rec[tname].items():
            agg[tname][key] = {k: stats(v) for k, v in d.items()}

    # 任务级 (跨类) 均值增益的 mean±std
    task = {}
    for tname in rec:
        task[tname] = {}
        for k in VARIANTS:
            per_class_mean = [np.mean(rec[tname][kk][k]) for kk in rec[tname]]
            a = np.array(per_class_mean, float)
            task[tname][k] = {'mean_of_class_means': round(float(a.mean()), 4),
                              'std_across_seeds': round(float(a.std(ddof=1)), 4)}

    out = {'config': {'n_seeds': K, 'method': 'bootstrap over eval images (cached detections)',
                      'variants': VARIANTS, 'detector': 'gdino-tiny',
                      'note': 'DIOR 源域每类仅~20图, 原型无法换种子; 故用评测图 bootstrap 估计 AP 增益方差'},
           'per_class': agg, 'task_level': task}
    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    # ── 打印关键结论 ──
    print('\n' + '=' * 84)
    print(f'多种子稳健性 (K={K} bootstrap)  ·  任务级跨类均值增益 mean±std(seed)')
    print('=' * 84)
    for tname in task:
        print(f'\n[{tname}]')
        for k in VARIANTS:
            s = task[tname][k]
            print(f'  {k:<12s} gain={s["mean_of_class_means"]:+.4f}  ±{s["std_across_seeds"]:.4f}')
    # 反向方向特异增量 (rev plus_delta - rev rand均值) 的 CI
    if 'rev_coco2dior' in task:
        rev = task['rev_coco2dior']
        plus = np.mean([np.mean(rec['rev_coco2dior'][kk]['plus_delta']) for kk in rec['rev_coco2dior']])
        rnd = np.mean([np.mean(rec['rev_coco2dior'][kk][f'rand{i}']) for kk in rec['rev_coco2dior'] for i in range(RB.N_RAND)])
        print(f'\n[反向方向特异增量] rev plus_delta({plus:+.4f}) − rev rand均值({rnd:+.4f}) '
              f'= {plus - rnd:+.4f}  (主实验原为 +0.026)')
    print(f'\n已保存: {OUT}')


if __name__ == '__main__':
    main()
