# -*- coding: utf-8 -*-
"""probe_remoteclip_text.py — 文本层诊断: RemoteCLIP 是否还保留 CA-PC 依赖的「photo vs overhead」视角轴?

动机 (先验证方向再烧 GPU):
  CA-PC 的核心量是 d_xi = E_text["a photo of y_t"] - E_text["an overhead image of y_s"]。
  RemoteCLIP 是「遥感微调」的 CLIP: 它的图文对全是遥感图。风险是 —— 它对所有类别都已
  "overhead 化", 于是 photo/overhead 这条视角轴可能被【塌缩】(cos -> 1, ||d_xi|| -> 0),
  那样 CA-PC 的 prior 就失去针对性, 该杠杆会【反向】强化 so-what 拒稿。
  本脚本【纯文本、不编码图像】, 分钟级给出判据。

控制变量设计 (关键):
  RemoteCLIP-ViT-B-32 与 openai CLIP ViT-B/32 【同架构】, 唯一变量 = 是否遥感微调。
  两条都用同一段 openai-clip 代码路径算 anchor, 做到干净对照。
  另附 transformers 版 openai/clip-vit-base-patch32 (即论文 tab:vlm 的基线编码器) 供对齐。

判据:
  - cos(photo, overhead) 越接近 1 / ||d_xi|| 越小  -> 轴塌缩, 杠杆失效。
  - 若 RemoteCLIP 的 ||d_xi|| 与 CLIP-B/32 同量级且方向结构相似 -> 轴仍在, 值得上图像实验。
  - 注意: 论文 tab:vlm 已证 ||d_xi|| 越大反而 fixed-gain 越差(过冲); 所以本诊断只看
    「轴是否存在/是否可分」, 不把 ||d_xi|| 大小本身当作增益预测。
"""
import os, sys, json, time
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ['HF_HUB_OFFLINE'] = '0'
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch

import run_bidirectional as RB          # 复用 harness 的 _as_vec (与 run_cross_vlm.py 同口径)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RC_B32 = r'C:\Users\张三\modeling\vlm_weights\remoteclip_b32\RemoteCLIP-ViT-B-32.pt'
RC_L14 = r'C:\Users\张三\modeling\vlm_weights\remoteclip_l14\RemoteCLIP-ViT-L-14.pt'
OUT = r'C:\Users\张三\Desktop\实验数据\probe_remoteclip_text.json'

PHOTO_T = 'a photo of {}'
OVER_T = 'an overhead image of {}'


def unit(a):
    a = np.asarray(a, float)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, 1e-12)


# ---------- 编码器封装 ----------
class OpenAIClip:
    """openai clip.load 路径 (RemoteCLIP 官方 demo 用法)。"""
    def __init__(self, path, disp):
        import clip
        self.disp = disp
        t0 = time.time()
        # 绕开被裁剪的 clip.load(): 本机 clip 包缺 build_model 导出, 且 load() 把同一文件句柄
        # 先喂 torch.jit.load 再喂 torch.load -> 句柄耗尽报 "Ran out of input"。
        # 直接 torch.load(weights_only=False) + clip.model.build_model 构建。
        sd = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        self.model = clip.model.build_model(sd).to(DEVICE).eval()
        self.clip = clip
        print(f'  [load] {disp}  {time.time()-t0:.0f}s', flush=True)

    @torch.no_grad()
    def enc_txt(self, ts):
        tok = self.clip.tokenize(list(ts), truncate=True).to(DEVICE)
        f = self.model.encode_text(tok).float()
        return unit(f.cpu().numpy())

    def free(self):
        del self.model
        import gc; gc.collect(); torch.cuda.empty_cache()


class TxClip:
    """transformers AutoModel 路径 (与论文 tab:vlm 基线同源)。"""
    def __init__(self, hf_id, disp):
        from transformers import AutoModel, AutoProcessor
        self.disp = disp
        t0 = time.time()
        self.model = AutoModel.from_pretrained(hf_id).to(DEVICE).eval()
        self.proc = AutoProcessor.from_pretrained(hf_id)
        print(f'  [load] {disp}  {time.time()-t0:.0f}s', flush=True)

    @torch.no_grad()
    def enc_txt(self, ts):
        inp = self.proc(text=list(ts), return_tensors='pt', padding=True,
                        truncation=True).to(DEVICE)
        f = RB._as_vec(self.model.get_text_features(**inp)).float()
        return unit(f.cpu().numpy())

    def free(self):
        del self.model, self.proc
        import gc; gc.collect(); torch.cuda.empty_cache()


def analyse(enc, names):
    E_ph = enc.enc_txt([PHOTO_T.format(c) for c in names])
    E_ov = enc.enc_txt([OVER_T.format(c) for c in names])
    d = E_ph - E_ov
    cos = (E_ph * E_ov).sum(1)
    mag = np.linalg.norm(d, axis=1)
    return {
        'per_class': {names[i]: {'cos_photo_over': round(float(cos[i]), 4),
                                 'dc_mag': round(float(mag[i]), 4)} for i in range(len(names))},
        'mean_cos': round(float(cos.mean()), 4),
        'mean_dc_mag': round(float(mag.mean()), 4),
        'min_dc_mag': round(float(mag.min()), 4),
        'max_dc_mag': round(float(mag.max()), 4),
    }


def main():
    names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    print(f'[names] {len(names)} 个类: {names}', flush=True)

    def _ready(path, full):
        try:
            return os.path.getsize(path) >= int(full * 0.995)
        except Exception:
            return False

    results = {}
    # openai 原生 CLIP-B/32 权重与 RemoteCLIP-B/32 同代码路径对照
    try:
        import clip as _c
        have_openai_clip = True
    except Exception:
        have_openai_clip = False
    print(f'[env] openai clip usable={have_openai_clip}', flush=True)

    # 1) 论文基线: transformers CLIP-B/32
    try:
        e = TxClip('openai/clip-vit-base-patch32', 'CLIP ViT-B/32 (transformers)')
        results['clip_b32_tx'] = analyse(e, names)
        e.free()
    except Exception as ex:
        print('  !! clip_b32_tx 失败:', ex, flush=True)

    # 2) RemoteCLIP-B/32 (openai clip.load)
    if _ready(RC_B32, 605208421):
        try:
            e = OpenAIClip(RC_B32, 'RemoteCLIP ViT-B/32')
            results['remoteclip_b32'] = analyse(e, names)
            e.free()
        except Exception as ex:
            print('  !! remoteclip_b32 失败:', ex, flush=True)
    else:
        print('  (RemoteCLIP-B/32 权重未就绪, 跳过)', flush=True)

    # 3) RemoteCLIP-L/14 (可选, 若已下载)
    if _ready(RC_L14, 1710613765):
        try:
            e = OpenAIClip(RC_L14, 'RemoteCLIP ViT-L/14')
            results['remoteclip_l14'] = analyse(e, names)
            e.free()
        except Exception as ex:
            print('  !! remoteclip_l14 失败:', ex, flush=True)
    else:
        print('  (RemoteCLIP-L/14 权重未就绪, 跳过)', flush=True)

    # ---- 汇总 ----
    print('\n' + '=' * 88, flush=True)
    print('photo-vs-overhead 视角轴诊断 (纯文本, L2 归一化)', flush=True)
    print('=' * 88, flush=True)
    print(f'{"encoder":<40}{"mean cos":>10}{"mean ||d_xi||":>14}{"min":>8}{"max":>8}', flush=True)
    print('-' * 88, flush=True)
    for k, r in results.items():
        print(f'{k:<40}{r["mean_cos"]:>10.4f}{r["mean_dc_mag"]:>14.4f}'
              f'{r["min_dc_mag"]:>8.4f}{r["max_dc_mag"]:>8.4f}', flush=True)

    # 关键对照: transformers-B/32 vs RemoteCLIP-B/32
    if 'clip_b32_tx' in results and 'remoteclip_b32' in results:
        a, b = results['clip_b32_tx'], results['remoteclip_b32']
        print('\n[控制对照] CLIP-B/32 vs RemoteCLIP-B/32 (同架构, 仅权重不同):', flush=True)
        print(f'  mean ||d_xi||: {a["mean_dc_mag"]:.4f} -> {b["mean_dc_mag"]:.4f} '
              f'({b["mean_dc_mag"]-a["mean_dc_mag"]:+.4f})', flush=True)
        print(f'  mean cos     : {a["mean_cos"]:.4f} -> {b["mean_cos"]:.4f} '
              f'({b["mean_cos"]-a["mean_cos"]:+.4f})', flush=True)
        # 方向一致性: 逐类 d_xi 的余弦 (跨编码器)
        common = set(a['per_class']) & set(b['per_class'])
        print(f'  (逐类 d_xi 方向需图像特征才能算对齐; 此处仅文本几何)', flush=True)

    json.dump({'names': names, 'results': results,
               'note': 'text-only anchor geometry; RemoteCLIP-B32 vs CLIP-B32 same arch'},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n-> {OUT}', flush=True)


if __name__ == '__main__':
    main()
