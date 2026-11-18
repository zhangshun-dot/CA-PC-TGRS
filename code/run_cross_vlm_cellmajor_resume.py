# -*- coding: utf-8 -*-
"""run_cross_vlm_cellmajor.py — ISPRS 缺口#96 (cell-major 版, 低内存/快)

问题: per-VLM-major 会把 36 个 cell 的 crop 全存内存 (≈20GB) → 爆页文件/卡死;
      --stream 版每 (cell,VLM) 都重载图像 → 5 遍图像 I/O, 太慢。
方案: cell-major —— 外层遍历 cell, 每 cell 只加载一次图像并裁一次 crop, 用【全部 VLM】
      依次编码+评估, 然后释放; 5 个模型同时常驻 GPU。图像只加载 1 遍, 内存峰值=单 cell crops。

口径: 与 run_cross_vlm.py 完全一致 (复用其 VLM/load_blob/rebuild_images 等) ——
      缓存检测框逐位不变, 只换 VLM 编码器; AP50/mAP 评估一致。

用法:
  HF_HUB_OFFLINE=1 "C:/Users/张三/vg/Scripts/python.exe" -u run_cross_vlm_cellmajor.py
"""
import os, sys, json, time, argparse
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cuda')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch
import run_bidirectional as RB
import run_cross_vlm as CV          # 复用 VLM / VLMS / LOCAL / detect 缓存工具 / 常量
os.environ['HF_HUB_OFFLINE'] = '1'  # 已在 import 后被 CV 置 '0', 这里改回离线


def _install_coco_cache():
    """acspc.data.load_coco 每次调用都重新 COCO(ann) 解析 ~25MB json (实测卡死主因)。
    给 pycocotools.COCO 打补丁: 同一 annFile 只解析一次。"""
    import pycocotools.coco as pc
    if getattr(pc, '_cached_patch', False):
        return
    _Orig = pc.COCO

    class _CachedCOCO(_Orig):
        _cache = {}

        def __init__(self, annFile, *a, **k):
            if getattr(self, '_inited', False):
                return
            _Orig.__init__(self, annFile, *a, **k)
            self._inited = True

        def __new__(cls, annFile, *a, **k):
            if annFile in cls._cache:
                return cls._cache[annFile]
            o = super().__new__(cls)
            cls._cache[annFile] = o
            return o

    pc.COCO = _CachedCOCO
    pc._cached_patch = True


_install_coco_cache()

# ── fp16 编码加速 ──
# 本机是 RTX 5060 **Laptop**, torch 默认 matmul.allow_tf32=False → ViT-L/14 fp32 仅 ~2.3 TFLOPS
# (实测 1692 crop 需 58.7s)。fp16 autocast 实测 ~2.9x 提速。论文计算复杂度表本就以 fp16 报告。
# 但 fp16 会把 CLIP ViT-B/32 的 base 偏 +0.0001 / CA-PC 偏 -0.0003 (4 位小数可见) →
# 为保住与 tab:det/tab:sota 的逐位一致, clip_b32 强制 fp32 (它很小: 全程仅 ~9min),
# 其余 4 个 backbone 用 fp16。CVLM_HALF=0 全部 fp32 对照; CVLM_FP32_TAGS 可覆盖。
USE_HALF = os.environ.get('CVLM_HALF', '1') == '1'
FP32_TAGS = set(t for t in os.environ.get('CVLM_FP32_TAGS', 'clip_b32').split(',') if t)
BS_CLIP = int(os.environ.get('CVLM_BS', '128'))
BS_SIGLIP = int(os.environ.get('CVLM_BS_SIGLIP', '64'))


def _install_half():
    if not USE_HALF:
        return

    @torch.no_grad()
    def enc_img_half(self, crops, bs=None):
        if not crops:
            return np.zeros((0, 1))
        if bs is None:
            bs = BS_SIGLIP if self.fam == 'siglip' else BS_CLIP
        use_half = self.tag not in FP32_TAGS
        m = self.model.half() if use_half else self.model.float()
        out = []
        for i in range(0, len(crops), bs):
            inp = self.proc(images=crops[i:i + bs], return_tensors='pt').to(CV.DEVICE)
            if use_half:
                inp = {k: (v.half() if v.dtype.is_floating_point else v) for k, v in inp.items()}
                with torch.autocast('cuda', dtype=torch.float16):
                    f = RB._as_vec(m.get_image_features(**inp)).float()
            else:
                f = RB._as_vec(m.get_image_features(**inp)).float()
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu())
        return torch.cat(out).numpy()

    CV.VLM.enc_img = enc_img_half


_install_half()

BETA, LAM, NEG_TOPK, N_PROTO, BIG_N = CV.BETA, CV.LAM, CV.NEG_TOPK, CV.N_PROTO, CV.BIG_N
BETA_GRID = CV.BETA_GRID
OUT = CV.OUT


def aggregate(all_cells):
    summary = {}
    for tag, vcells in all_cells.items():
        dets = []
        for c in vcells:
            if c['det'] not in dets:
                dets.append(c['det'])
        for det in dets:
            cc = [c for c in vcells if c['det'] == det]
            row = {k: float(np.mean([c[k] for c in cc]))
                   for k in ('base', 'base_m', 'ca_pc', 'ca_pc_m', 'ca_pc_g', 'ca_pc_g_m',
                             'ca_pc_star', 'ca_pc_star_m', 'dc_mag')}
            row['n_cells'] = len(cc)
            row['gain'] = row['ca_pc'] - row['base']
            row['gain_m'] = row['ca_pc_m'] - row['base_m']
            row['gain_g'] = row['ca_pc_g'] - row['base']
            row['gain_g_m'] = row['ca_pc_g_m'] - row['base_m']
            summary[f'{tag}|{det}'] = row
        row = {k: float(np.mean([c[k] for c in vcells]))
               for k in ('base', 'base_m', 'ca_pc', 'ca_pc_m', 'ca_pc_g', 'ca_pc_g_m',
                         'ca_pc_star', 'ca_pc_star_m')}
        row['n_cells'] = len(vcells)
        row['gain'] = row['ca_pc'] - row['base']
        row['gain_m'] = row['ca_pc_m'] - row['base_m']
        row['gain_g'] = row['ca_pc_g'] - row['base']
        row['gain_g_m'] = row['ca_pc_g_m'] - row['base_m']
        summary[f'{tag}|ALL'] = row
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vlms', nargs='+', default=list(CV.VLMS.keys()))
    ap.add_argument('--dets', nargs='+', default=list(CV.DETECTORS.keys()))
    ap.add_argument('--tasks', nargs='+', default=list(RB.TASKS))
    ap.add_argument('--merge', action='store_true',
                    help='把本轮 VLM 的 cells 合并进已有 json (分轮跑, 每轮只驻留几个模型以适配 8GB 显存)')
    args = ap.parse_args()
    t0 = time.time()

    cfg = RB.Config()
    cfg.model.device = CV.DEVICE
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'
    _gate = json.load(open(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced.json'))['config']['gate']
    PRIOR_SCALE, c0 = _gate['prior_scale'], _gate['c0']

    def gate_conf(conf, c=c0, scale=3.0):
        return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))

    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))

    # ── cell 元信息 (不含 crops) ──
    cells = []
    for det in args.dets:
        cdir = CV.DETECTORS[det]
        for tname in args.tasks:
            src_dom, tgt_dom, pairs = RB.TASKS[tname]
            for src_cls, eval_cls in pairs:
                blob = CV.load_blob(cdir, tname, eval_cls)
                if blob is None:
                    print(f'  !! {det}/{tname}/{eval_cls} 缺缓存', flush=True)
                    continue
                per_img = blob['per_img']
                nb = int(sum(len(p['gd']) for p in per_img))
                cells.append({'det': det, 'tname': tname, 'src': src_cls, 'eval': eval_cls,
                              'src_dom': src_dom, 'tgt_dom': tgt_dom,
                              'per_img': per_img, 'n_img': len(per_img), 'n_box': nb})
        print(f'[cells] {det}: 累计 {len(cells)} cells', flush=True)
    print(f'[cells] 共 {len(cells)} cells, {sum(c["n_box"] for c in cells)} crops', flush=True)

    # ── 加载全部 VLM (常驻 GPU; 非 FP32_TAGS 立即转 half 以省显存) ──
    vlms, prep = {}, {}
    for tag in args.vlms:
        if tag not in CV.VLMS:
            continue
        v = CV.VLM(tag)
        if USE_HALF and tag not in FP32_TAGS:
            v.model.half()
            torch.cuda.empty_cache()
            mib = torch.cuda.memory_allocated() / 2**30
            print(f'  [VLM] {tag} -> fp16  (alloc {mib:.2f} GiB)', flush=True)
        vlms[tag] = v
    # ── 原型池 GT crop 与 VLM 无关 → 只加载一次 (避免 5× 冗余 load_split) ──
    proto_crops = {}
    for domain, pool in (('coco', RB.COCO_POOL), ('dior', RB.DIOR_POOL)):
        d = {}
        for cls in pool:
            ims, gts = RB.load_split(cfg, domain, cls, 0, N_PROTO)
            gc_ = RB.D.gt_crops(ims, gts)
            if gc_:
                d[cls] = gc_
        proto_crops[domain] = d
    print(f'[prep] 原型池 GT crop 就绪 '
          f'({sum(len(v) for v in proto_crops.values())} 类)', flush=True)

    for tag, vlm in vlms.items():
        bank = {}
        for domain in ('coco', 'dior'):
            b = {}
            for cls, gc_ in proto_crops[domain].items():
                Ft = vlm.enc_img(gc_)
                pt = torch.from_numpy(Ft.mean(0))
                b[cls] = pt / pt.norm()
            bank[domain] = b
        Ep = vlm.enc_txt([f'a photo of {c}' for c in all_names])
        Eo = vlm.enc_txt([f'an overhead image of {c}' for c in all_names])
        prep[tag] = (bank, {c: Ep[i] for i, c in enumerate(all_names)},
                     {c: Eo[i] for i, c in enumerate(all_names)})
        print(f'[prep] {tag} 原型池/文本就绪', flush=True)
    if torch.cuda.is_available():
        print(f'[gpu] allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB', flush=True)

    # ── 分轮合并: 载入已有其它 VLM 的 cells (本轮只跑 --vlms 指定的模型) ──
    carry = {}
    if args.merge and os.path.exists(OUT):
        try:
            _old = json.load(open(OUT, encoding='utf-8'))
            carry = {k: v for k, v in (_old.get('cells', {}) or {}).items() if k not in args.vlms}
            print(f'[merge] 载入已有 {list(carry.keys())} (本轮 {args.vlms})', flush=True)
        except Exception as _e:
            print(f'[merge] 读取失败, 从空开始: {_e}', flush=True)
            carry = {}

    # ── cell-major 主循环 ──
    res = {tag: [] for tag in vlms}
    # --- 断点续跑 (2026-10-10) ---
    try:
        _old_full = json.load(open(OUT, encoding='utf-8')).get('cells', {}) or {}
    except Exception:
        _old_full = {}
    DONE = {}
    for tag in vlms:
        _prev = [c for c in (_old_full.get(tag) or [])]
        res[tag] = _prev
        DONE[tag] = {(c['det'], c['tname'], c['src'], c['eval'], c['n_img']) for c in _prev}
    if any(len(v) for v in DONE.values()):
        print('  [resume] ' + ', '.join(f'{t}:{len(DONE[t])}/{len(cells)}' for t in vlms),
              flush=True)
    for ci, c in enumerate(cells):
        ims = CV.rebuild_images(cfg, c['tgt_dom'], c['eval'], c['n_img'])
        if len(ims) != c['n_img']:
            print(f'  !! {c["det"]}/{c["tname"]}/{c["eval"]} 图像数不符, 跳过', flush=True)
            continue
        crops = []
        for img, p in zip(ims, c['per_img']):
            crops.extend(CV.crops_from_cached(img, p))
        assert len(crops) == c['n_box'], f'crop {len(crops)} != box {c["n_box"]}'
        per_img = c['per_img']
        for tag, vlm in vlms.items():
            _k = (c['det'], c['tname'], c['src'], c['eval'], c['n_img'])
            if _k in DONE[tag]:
                continue
            bank, Ep, Eo = prep[tag]
            F = vlm.enc_img(crops)
            P_src = CV.unit(bank[c['src_dom']][c['src']].numpy())
            P_neg = CV.unit(RB.Proto.get_negative_prototype(bank[c['src_dom']], c['src'], NEG_TOPK).numpy())
            dc = Ep[c['eval']] - Eo[c['src']]
            b50, bm = CV.eval_pair(per_img, F, CV.unit(P_src) - LAM * P_neg)
            c50, cm = CV.eval_pair(per_img, F, CV.unit(P_src + BETA * dc) - LAM * P_neg)
            max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
            gamma = float(np.clip(PRIOR_SCALE * float(np.linalg.norm(dc)) * gate_conf(max_conf), 0.0, 4.0))
            g50, gm = CV.eval_pair(per_img, F, CV.unit(P_src + gamma * dc) - LAM * P_neg)
            best50, bestm, bestb = -1, 0, 0
            for bb in BETA_GRID:
                a50, m = CV.eval_pair(per_img, F, CV.unit(P_src + bb * dc) - LAM * P_neg)
                if a50 > best50:
                    best50, bestm, bestb = a50, m, bb
            res[tag].append({'det': c['det'], 'tname': c['tname'], 'src': c['src'], 'eval': c['eval'],
                             'n_img': c['n_img'], 'n_box': c['n_box'],
                             'dc_mag': round(float(np.linalg.norm(dc)), 4),
                             'base': round(b50, 4), 'base_m': round(bm, 4),
                             'ca_pc': round(c50, 4), 'ca_pc_m': round(cm, 4),
                             'ca_pc_g': round(g50, 4), 'ca_pc_g_m': round(gm, 4), 'gamma': round(gamma, 4),
                             'ca_pc_star': round(best50, 4), 'ca_pc_star_m': round(bestm, 4), 'ca_pc_beta': bestb})
            del F
        print(f'  (cell {ci+1}/{len(cells)}) {c["det"]} {c["src"]}->{c["eval"]} '
              f'nbox={c["n_box"]}  [ok {[len(res[t]) for t in vlms]}]', flush=True)
        del crops, ims
        # 增量落盘 (合并本轮 + 已有其它 VLM)
        try:
            _all = dict(carry); _all.update(res)
            json.dump({'config': {'beta': BETA, 'lam': LAM, 'cell_major': True, 'partial': True},
                       'cells': _all}, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        except Exception as _e:
            print(f'  [warn] 增量落盘失败: {_e}', flush=True)

    # ── 并入已有其它 VLM 的 cells ──
    all_cells = dict(carry)
    all_cells.update(res)
    summary = aggregate(all_cells)
    res = all_cells

    print('\n' + '=' * 104, flush=True)
    print('跨 VLM backbone 鲁棒性 (DIOR<->COCO, 零目标 GT; AP50 / mAP@[.5:.95]; BETA=2.0 固定)', flush=True)
    print('=' * 104, flush=True)
    print(f'{"VLM":<20}{"base AP50":>11}{"CA-PC(fix)":>12}{"Δfix":>9}{"CA-PC(g)":>11}{"Δg":>9}'
          f'{"base mAP":>11}{"CA-PC mAP":>12}{"Δ mAP":>10}', flush=True)
    print('-' * 110, flush=True)
    for tag in res:
        r = summary[f'{tag}|ALL']
        print(f'{CV.VLMS[tag][1]:<20}{r["base"]:>11.4f}{r["ca_pc"]:>12.4f}{r["gain"]:>+9.4f}'
              f'{r["ca_pc_g"]:>11.4f}{r["gain_g"]:>+9.4f}'
              f'{r["base_m"]:>11.4f}{r["ca_pc_m"]:>12.4f}{r["gain_m"]:>+10.4f}', flush=True)
    print('\n每检测器明细:', flush=True)
    for tag in res:
        for det in CV.DET_LABEL:
            key = f'{tag}|{det}'
            if key not in summary:
                continue
            r = summary[key]
            print(f'  {CV.VLMS[tag][1]:<20} {CV.DET_LABEL[det]:<14} base={r["base"]:.4f} '
                  f'ca_pc={r["ca_pc"]:.4f}(Δ{r["gain"]:+.4f}) ca_pc_g={r["ca_pc_g"]:.4f}(Δ{r["gain_g"]:+.4f}) '
                  f'mAP Δ{r["gain_m"]:+.4f} n={r["n_cells"]}', flush=True)

    json.dump({'config': {'beta': BETA, 'lam': LAM, 'neg_topk': NEG_TOPK, 'n_proto': N_PROTO,
                          'zero_target_gt': True, 'big_n': BIG_N, 'cell_major': True,
                          'note': 'identical cached detector boxes re-cropped once per cell + encoded by all VLMs; '
                                  'only the VLM backbone varies; AP50/mAP eval identical across VLMs',
                          'vlm_ids': {t: CV.VLMS[t][0] for t in CV.VLMS},
                          'excluded': 'EVA-CLIP: only 8B mirrored (too large for 8GB); L-size not on hub'},
               'summary': summary, 'cells': res, 'elapsed_s': round(time.time() - t0, 1)},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
