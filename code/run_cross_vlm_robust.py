# -*- coding: utf-8 -*-
"""run_cross_vlm.py — ISPRS 缺口#96: CA-PC 跨 VLM backbone 鲁棒性

目的: 把 CA-PC (以及 base) 的「检测器输出之后、用 VLM 原型重打分」这一层,
      在多个视觉-语言 backbone 上重跑, 检验 CA-PC 的增益是否 VLM 无关。
      backbone: CLIP ViT-B/32 (基线), CLIP ViT-L/14, OpenCLIP ViT-L/14,
                OpenCLIP ViT-B/32, SigLIP SO400M。

为什么口径完全一致 (关键设计, 直接复用缺口#95 的思路):
  检测框来自【缓存】(per_img['box']), 我【不重新检测】, 用缓存框在原图上重新裁剪
  同一批 crop 像素, 只更换 VLM 编码器 → 框/NMS/AP50/mAP 评估全部逐位一致, 唯一变量是 VLM。
  图像重建: 用 RB.load_split(offset=N_PROTO, n=BIG) 重建同一顺序的图像, 取前 len(per_img) 张
  (前缀性质保证与建缓存时完全一致)。

方法 (打分 s = sqrt(gd * clip(max(0, F·W))), 与 CA-PC 完全相同):
  base  : W = unit(P_src)              - LAM·P_neg
  ca_pc : W = unit(P_src + BETA·dc)    - LAM·P_neg     # dc = E_photo[eval] - E_over[src]
  BETA=2.0 固定 (单一超参, 不做任何 per-VLM 调参 → 干净的鲁棒性检验);
  另报 per-cell grid 最优 (*) 作为各 VLM 的上界, 不 handicap 任何一方。

红线: 纯 B-pipeline (零目标 GT); 绝不混入 A-CSPC 缓存/数字。
所有 VLM 特征 (图像/文本) 统一 L2 归一化, 使 F·W = 余弦相似度。

用法 (GPU; 需先联网下载权重 → hf-mirror):
  "C:/Users/张三/vg/Scripts/python.exe" -u run_cross_vlm.py
  # 小样本验证: --dets gdino --vlms clip_b32 clip_l14 --tasks fwd_dior2coco
"""
import os, sys, json, pickle, glob, time, argparse
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cuda')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch
import run_bidirectional as RB

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BETA = RB.BETA            # 2.0 固定
LAM = RB.LAM              # 0.5
NEG_TOPK = RB.NEG_TOPK    # 3
N_PROTO = RB.N_PROTO      # 20
BIG_N = 400               # 重建图像时请求的 n (>= 建缓存时用的 100, 前缀性质 → 前 len(per_img) 张一致)
BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

DETECTORS = {
    'gdino':  r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache',
    'owlv2':  r'C:\Users\张三\Desktop\实验数据\probe_owl_cache',
    'yolo':   r'C:\Users\张三\Desktop\实验数据\probe_yolo_cache',
    'rtdetr': r'C:\Users\张三\Desktop\实验数据\probe_rtdetr_cache',
}
DET_LABEL = {'gdino': 'grounding-tiny', 'owlv2': 'OWL-ViT', 'yolo': 'YOLO-style', 'rtdetr': 'RT-DETR'}

# VLM backbone: tag -> (HF id, display, family)
VLMS = {
    'clip_b32':      ('openai/clip-vit-base-patch32',           'CLIP ViT-B/32',    'clip'),
    'clip_l14':      ('openai/clip-vit-large-patch14',          'CLIP ViT-L/14',    'clip'),
    'openclip_l14':  ('laion/CLIP-ViT-L-14-laion2B-s32B-b82K',  'OpenCLIP ViT-L/14','clip'),
    'openclip_b32':  ('laion/CLIP-ViT-B-32-laion2B-s34B-b79K',  'OpenCLIP ViT-B/32','clip'),
    'siglip_so400m': ('google/siglip-so400m-patch14-384',       'SigLIP SO400M',    'siglip'),
}
# 本地权重目录 (curl 从 hf-mirror 直下; Python 端可正常处理非 ASCII 路径)
LOCAL = {
    'clip_l14':      r'C:\Users\张三\modeling\vlm_weights\clip_l14',
    'openclip_l14':  r'C:\Users\张三\modeling\vlm_weights\openclip_l14',
    'openclip_b32':  r'C:\Users\张三\modeling\vlm_weights\openclip_b32',
    'siglip_so400m': r'C:\Users\张三\modeling\vlm_weights\siglip_so400m',
}
OUT = r'C:\Users\张三\Desktop\实验数据\run_cross_vlm.json'
LOG = r'C:\Users\张三\Desktop\实验数据\run_cross_vlm.log'


def unit(x):
    x = np.asarray(x, float)
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


class VLM:
    def __init__(self, tag):
        self.tag = tag
        hf_id, disp, fam = VLMS[tag]
        src = LOCAL.get(tag, hf_id)          # 优先本地权重目录
        self.disp, self.fam = disp, fam
        from transformers import AutoModel, AutoProcessor
        t0 = time.time()
        self.model = AutoModel.from_pretrained(src).to(DEVICE).eval()
        self.proc = AutoProcessor.from_pretrained(src)
        print(f'  [VLM] loaded {disp} ({src}) fam={fam}  {time.time()-t0:.0f}s', flush=True)

    def free(self):
        del self.model, self.proc
        import gc; gc.collect(); torch.cuda.empty_cache()

    @torch.no_grad()
    def enc_img(self, crops, bs=None):
        if not crops:
            return np.zeros((0, 1))
        if bs is None:
            bs = _DEFAULT_VLM_BS or (64 if self.fam == 'siglip' else 128)
        tries = 0
        while True:
            try:
                out = []
                for i in range(0, len(crops), bs):
                    batch = crops[i:i + bs]
                    inp = self.proc(images=batch, return_tensors='pt').to(DEVICE)
                    f = RB._as_vec(self.model.get_image_features(**inp)).float()
                    f = f / f.norm(dim=-1, keepdim=True)
                    out.append(f.cpu())
                    del inp, f
                return torch.cat(out).numpy()
            except Exception as _e:                      # 显存不足 -> 折半重试
                if not (isinstance(_e, torch.AcceleratorError) or
                        isinstance(_e, RuntimeError)) or 'memory' not in str(_e).lower():
                    raise
                try:
                    del out
                except Exception:
                    pass
                import gc as _gc; _gc.collect(); torch.cuda.empty_cache()
                tries += 1
                if bs <= 1 or tries >= 5:
                    raise
                bs = max(1, bs // 2)
                print(f'  [oom] 编码 batch 折半 -> {bs} (第 {tries} 次重试)', flush=True)

    @torch.no_grad()
    def enc_txt(self, ts, bs=64):
        out = []
        for i in range(0, len(ts), bs):
            batch = ts[i:i + bs]
            if self.fam == 'siglip':
                inp = self.proc(text=batch, return_tensors='pt', padding='max_length',
                                truncation=True).to(DEVICE)
            else:
                inp = self.proc(text=batch, return_tensors='pt', padding=True,
                                truncation=True).to(DEVICE)
            f = RB._as_vec(self.model.get_text_features(**inp)).float()
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu())
        return torch.cat(out).numpy()



# --- OOM-hardening (2026-10-10): VLM 编码 batch 可经环境变量调小 ---
_DEFAULT_VLM_BS = int(os.environ.get('AC_SPC_VLM_BS', '0')) or None
if _DEFAULT_VLM_BS:
    print(f'[bs] AC_SPC_VLM_BS={_DEFAULT_VLM_BS} (覆盖默认 batch)', flush=True)

def load_blob(cache_dir, tname, eval_cls):
    """与论文表(tab:det/tab:sota 来源 run_closed_loop_balanced.load_blob)严格同口径:
    取 glob 结果的字典序最小者 (hits[0])。复现真实选取: gdino 'n77'<'n8'(取 n77),
    rtdetr 'n20'<'n77'(airplane 取 n20)、'n100'<'n20'(其余取 n100)。"""
    hits = glob.glob(os.path.join(cache_dir, f'{tname}_{eval_cls}_n*.pkl')) or \
           glob.glob(os.path.join(cache_dir, f'{tname}_{eval_cls}_off{N_PROTO}_n*.pkl'))
    if not hits:
        return None
    return pickle.load(open(sorted(hits)[0], 'rb'))


def rebuild_images(cfg, tgt_dom, eval_cls, k):
    """用与建缓存时相同的方式重建前 k 张图像 (前缀性质保证一致)。"""
    ims, gts = RB.load_split(cfg, tgt_dom, eval_cls, N_PROTO, BIG_N)
    return ims[:k]


def crops_from_cached(img, p):
    out = []
    for (x1, y1, x2, y2) in p['box'].astype(int):
        out.append(img.crop((int(x1), int(y1), int(x2), int(y2))))
    return out


def score_rows(per_img, F, W):
    rows = []
    for p in per_img:
        if len(p['gd']):
            w = F[p['crop_idx']] @ W
            s = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
        else:
            s = np.zeros(0)
        rows.append((p['box'], s))
    return rows


def eval_pair(per_img, F, W):
    rows = score_rows(per_img, F, W)
    return (float(RB.Met.compute_ap50_img(rows, [p['gt'] for p in per_img])),
            float(RB.Met.compute_ap_coco(rows, [p['gt'] for p in per_img])))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--dets', nargs='+', default=list(DETECTORS.keys()))
    ap_.add_argument('--vlms', nargs='+', default=list(VLMS.keys()))
    ap_.add_argument('--tasks', nargs='+', default=list(RB.TASKS))
    ap_.add_argument('--only-blobs', action='store_true', help='只重建缓存(含框与图像计数), 不编码')
    ap_.add_argument('--merge', action='store_true', help='合并已有 json 的 cells (增量补跑某个 VLM)')
    ap_.add_argument('--stream', action='store_true', help='逐 cell 重建 crops (降低内存峰值, 供大模型如 SigLIP)')
    args = ap_.parse_args()
    t0 = time.time()

    cfg = RB.Config()
    cfg.model.device = DEVICE
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    # dc×conf 门控尺度 (与 tab:det 同源, 零GT; 直接施于各 VLM 的 ||dc||)
    _gate = json.load(open(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced.json'))['config']['gate']
    PRIOR_SCALE, c0 = _gate['prior_scale'], _gate['c0']

    def gate_conf(conf, c=c0, scale=3.0):
        return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))

    # ── 收集所有 cell 的 (框, 图像) ──
    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    cells = []
    for det in args.dets:
        cdir = DETECTORS[det]
        for tname in args.tasks:
            src_dom, tgt_dom, pairs = RB.TASKS[tname]
            for src_cls, eval_cls in pairs:
                blob = load_blob(cdir, tname, eval_cls)
                if blob is None:
                    print(f'  !! {det}/{tname}/{eval_cls} 缺缓存', flush=True)
                    continue
                per_img = blob['per_img']
                ims = rebuild_images(cfg, tgt_dom, eval_cls, len(per_img))
                if len(ims) != len(per_img):
                    print(f'  !! {det}/{tname}/{eval_cls} 图像重建数不符 '
                          f'{len(ims)}!={len(per_img)}, 跳过', flush=True)
                    continue
                nb = int(sum(len(p['gd']) for p in per_img))
                crops = None
                if not args.stream:                       # stream: 不预存 crops (低内存)
                    crops = []
                    for img, p in zip(ims, per_img):
                        crops.extend(crops_from_cached(img, p))
                    if len(crops) != nb:
                        print(f'  !! {det}/{tname}/{eval_cls} crop 数 {len(crops)} != box 数 {nb}, 跳过', flush=True)
                        continue
                cells.append({'det': det, 'tname': tname, 'src': src_cls, 'eval': eval_cls,
                              'src_dom': src_dom, 'tgt_dom': tgt_dom,
                              'per_img': per_img, 'crops': crops,
                              'n_img': len(per_img), 'n_box': nb})
        print(f'[cells] {det}: 累计 {len(cells)} cells', flush=True)

    total_crops = sum(c['n_box'] for c in cells)
    print(f'[cells] 共 {len(cells)} cells, {total_crops} crops', flush=True)
    if args.only_blobs:
        print('only-blobs: 缓存重建自检通过', flush=True)
        return

    data = {}
    for tag in args.vlms:
        if tag not in VLMS:
            continue
        print('\n' + '=' * 90, flush=True)
        print(f'VLM = {tag} ({VLMS[tag][1]})', flush=True)
        print('=' * 90, flush=True)
        vlm = VLM(tag)

        # 源原型池 (该 VLM 空间)
        bank = {}
        for domain, pool in (('coco', RB.COCO_POOL), ('dior', RB.DIOR_POOL)):
            b = {}
            for cls in pool:
                ims, gts = RB.load_split(cfg, domain, cls, 0, N_PROTO)
                gc_ = RB.D.gt_crops(ims, gts)
                if not gc_:
                    continue
                Ft = vlm.enc_img(gc_)
                pt = torch.from_numpy(Ft.mean(0))
                b[cls] = pt / pt.norm()
            bank[domain] = b
        # 文本原型
        E_photo = vlm.enc_txt([f'a photo of {c}' for c in all_names])
        E_over = vlm.enc_txt([f'an overhead image of {c}' for c in all_names])
        E_photo_d = {c: E_photo[i] for i, c in enumerate(all_names)}
        E_over_d = {c: E_over[i] for i, c in enumerate(all_names)}

        vcells = []
        # --- 断点续跑: 已有该 tag 的 cell 直接跳过 ---
        done_keys = set()
        try:
            _prev = json.load(open(OUT, encoding='utf-8')).get('cells', {}) or {}
            for _pc in _prev.get(tag, []) or []:
                done_keys.add((_pc['det'], _pc['tname'], _pc['src'], _pc['eval'], _pc['n_img']))
                vcells.append(_pc)
        except Exception:
            vcells = []
        if done_keys:
            print(f'  [resume] {tag} 已有 {len(done_keys)}/{len(cells)} cells, 跳过这些',
                  flush=True)

        def _dump_incr():
            try:
                _d = {}
                if os.path.exists(OUT):
                    try:
                        _d = json.load(open(OUT, encoding='utf-8')).get('cells', {}) or {}
                    except Exception:
                        _d = {}
                _d[tag] = vcells
                json.dump({'config': {'beta': BETA, 'lam': LAM, 'neg_topk': NEG_TOPK,
                                      'n_proto': N_PROTO, 'zero_target_gt': True,
                                      'stream': args.stream,
                                      'note': 'per-cell incremental dump'},
                           'cells': _d}, open(OUT, 'w', encoding='utf-8'),
                          indent=2, ensure_ascii=False)
            except Exception as _e:
                print(f'  [warn] 增量落盘失败: {_e}', flush=True)

        for ci, c in enumerate(cells):
            _key = (c['det'], c['tname'], c['src'], c['eval'], c['n_img'])
            if _key in done_keys:
                continue
            if c.get('crops') is None:                    # stream: 逐 cell 重建 crops (低内存)
                _ims = rebuild_images(cfg, c['tgt_dom'], c['eval'], c['n_img'])
                _crops = []
                for _img, _p in zip(_ims, c['per_img']):
                    _crops.extend(crops_from_cached(_img, _p))
                F = vlm.enc_img(_crops)
                del _crops, _ims
            else:
                F = vlm.enc_img(c['crops'])
            per_img = c['per_img']
            P_src = unit(bank[c['src_dom']][c['src']].numpy())
            P_neg = unit(RB.Proto.get_negative_prototype(bank[c['src_dom']], c['src'], NEG_TOPK).numpy())
            dc = E_photo_d[c['eval']] - E_over_d[c['src']]
            W_base = unit(P_src) - LAM * P_neg
            W_ca = unit(P_src + BETA * dc) - LAM * P_neg
            # gate-balanced (与 tab:det 同一零GT门控公式; 同一 PRIOR_SCALE/c0 施于各 VLM 的 ||dc||)
            max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
            gamma = float(np.clip(PRIOR_SCALE * float(np.linalg.norm(dc)) * gate_conf(max_conf), 0.0, 4.0))
            W_cg = unit(P_src + gamma * dc) - LAM * P_neg
            b50, bm = eval_pair(per_img, F, W_base)
            c50, cm = eval_pair(per_img, F, W_ca)
            g50, gm = eval_pair(per_img, F, W_cg)
            # per-cell oracle grid
            best50, bestm, bestb = -1, 0, 0
            for bb in BETA_GRID:
                a50, m = eval_pair(per_img, F, unit(P_src + bb * dc) - LAM * P_neg)
                if a50 > best50:
                    best50, bestm, bestb = a50, m, bb
            cell = {'det': c['det'], 'tname': c['tname'], 'src': c['src'], 'eval': c['eval'],
                    'n_img': c['n_img'], 'n_box': c['n_box'], 'dc_mag': round(float(np.linalg.norm(dc)), 4),
                    'base': round(b50, 4), 'base_m': round(bm, 4),
                    'ca_pc': round(c50, 4), 'ca_pc_m': round(cm, 4),
                    'ca_pc_g': round(g50, 4), 'ca_pc_g_m': round(gm, 4), 'gamma': round(gamma, 4),
                    'ca_pc_star': round(best50, 4), 'ca_pc_star_m': round(bestm, 4), 'ca_pc_beta': bestb}
            vcells.append(cell)
            print(f'  [{c["det"]}] {c["src"]}->{c["eval"]}: base={b50:.4f} ca_pc={c50:.4f} '
                  f'(gain {c50-b50:+.4f}) *{best50:.4f}  mAP base={bm:.4f} ca={cm:.4f}  '
                  f'||dc||={cell["dc_mag"]:.3f}  ({len(vcells)}/{len(cells)})', flush=True)
            _dump_incr()
            import gc as _gc2; _gc2.collect(); torch.cuda.empty_cache()
        data[tag] = vcells
        vlm.free()
        print(f'  [VLM] {tag} done, 释放显存', flush=True)
        # 增量落盘 (防止后续 VLM 崩溃丢结果)
        try:
            json.dump({'config': {'beta': BETA, 'lam': LAM, 'neg_topk': NEG_TOPK, 'n_proto': N_PROTO,
                                  'zero_target_gt': True, 'stream': args.stream,
                                  'note': 'incremental partial dump'},
                       'cells': data}, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
            print(f'  [save] 增量已写 {OUT}', flush=True)
        except Exception as _e:
            print(f'  [warn] 增量落盘失败: {_e}', flush=True)

    # ── 合并已有 json (增量补 VLM) ──
    all_data = {}
    if args.merge and os.path.exists(OUT):
        try:
            all_data = json.load(open(OUT, encoding='utf-8')).get('cells', {}) or {}
            print(f'[merge] 载入已有 {list(all_data.keys())}', flush=True)
        except Exception as e:
            print(f'[merge] 读取失败, 从空开始: {e}', flush=True)
            all_data = {}
    all_data.update(data)

    # ── 汇总: 每 (vlm, det) 宏平均 + 全局 ──
    summary = {}
    for tag, vcells in all_data.items():
        dets_present = []
        for c in vcells:
            if c['det'] not in dets_present:
                dets_present.append(c['det'])
        for det in dets_present:
            cc = [c for c in vcells if c['det'] == det]
            if not cc:
                continue
            row = {k: float(np.mean([c[k] for c in cc]))
                   for k in ('base', 'base_m', 'ca_pc', 'ca_pc_m', 'ca_pc_g', 'ca_pc_g_m',
                             'ca_pc_star', 'ca_pc_star_m', 'dc_mag')}
            row['n_cells'] = len(cc)
            row['gain'] = row['ca_pc'] - row['base']
            row['gain_m'] = row['ca_pc_m'] - row['base_m']
            row['gain_g'] = row['ca_pc_g'] - row['base']
            row['gain_g_m'] = row['ca_pc_g_m'] - row['base_m']
            summary[f'{tag}|{det}'] = row
        # 全局 (跨检测器平均)
        row = {k: float(np.mean([c[k] for c in vcells]))
               for k in ('base', 'base_m', 'ca_pc', 'ca_pc_m', 'ca_pc_g', 'ca_pc_g_m',
                         'ca_pc_star', 'ca_pc_star_m')}
        row['n_cells'] = len(vcells)
        row['gain'] = row['ca_pc'] - row['base']
        row['gain_m'] = row['ca_pc_m'] - row['base_m']
        row['gain_g'] = row['ca_pc_g'] - row['base']
        row['gain_g_m'] = row['ca_pc_g_m'] - row['base_m']
        summary[f'{tag}|ALL'] = row

    print('\n' + '=' * 100, flush=True)
    print('跨 VLM backbone 鲁棒性 (DIOR<->COCO, 零目标 GT; AP50 / mAP@[.5:.95]; BETA=2.0 固定)', flush=True)
    print('=' * 100, flush=True)
    hdr = (f'{"VLM":<20}{"base AP50":>11}{"CA-PC(fix)":>12}{"Δfix":>9}'
           f'{"CA-PC(g)":>11}{"Δg":>9}{"base mAP":>11}{"CA-PC mAP":>12}{"Δ mAP":>10}')
    print(hdr, flush=True)
    print('-' * 110, flush=True)
    for tag in all_data:
        key = f'{tag}|ALL'
        if key not in summary:
            continue
        r = summary[key]
        print(f'{VLMS[tag][1]:<20}{r["base"]:>11.4f}{r["ca_pc"]:>12.4f}{r["gain"]:>+9.4f}'
              f'{r["ca_pc_g"]:>11.4f}{r["gain_g"]:>+9.4f}'
              f'{r["base_m"]:>11.4f}{r["ca_pc_m"]:>12.4f}{r["gain_m"]:>+10.4f}', flush=True)
    # 每检测器明细
    print('\n每检测器明细 (VLM × detector):', flush=True)
    for tag in all_data:
        for det in DET_LABEL:
            key = f'{tag}|{det}'
            if key not in summary:
                continue
            r = summary[key]
            print(f'  {VLMS[tag][1]:<20} {DET_LABEL[det]:<14} base={r["base"]:.4f} '
                  f'ca_pc={r["ca_pc"]:.4f} (Δ{r["gain"]:+.4f})  ca_pc_g={r["ca_pc_g"]:.4f} '
                  f'(Δ{r["gain_g"]:+.4f})  mAP Δ{r["gain_m"]:+.4f}  n={r["n_cells"]}', flush=True)

    json.dump({'config': {'beta': BETA, 'lam': LAM, 'neg_topk': NEG_TOPK, 'n_proto': N_PROTO,
                          'zero_target_gt': True, 'big_n': BIG_N,
                          'note': 'identical cached detector boxes re-cropped + re-encoded per VLM; '
                                  'only the VLM backbone varies; AP50/mAP eval identical across VLMs',
                          'vlm_ids': {t: VLMS[t][0] for t in VLMS},
                          'excluded': 'EVA-CLIP: only 8B checkpoint mirrored (too large for 8GB); L-size not on hub'},
               'summary': summary, 'cells': all_data, 'elapsed_s': round(time.time() - t0, 1)},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
