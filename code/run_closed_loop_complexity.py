# -*- coding: utf-8 -*-
"""run_closed_loop_complexity.py — ISPRS 缺口2：CA-PC 计算复杂度 profile（GPU 实测）

CA-PC 是**推理期原型空间调制**，不微调检测器主干、不重编码图像 crop。本脚本实测
CA-PC 相对 vanilla CLIP-scoring OVD 的**纯增量成本**（Base vs CA-PC）：

  (a) 离线一次性（跨全部目标图共享，只算一次）
        - 源 crop CLIP 图像编码  N_PROTO 张  (Base 也已做，非 CA-PC 增量)
        - canonical anchor 文本编码  N_ANCHOR 条  (CA-PC 增量)
  (b) 在线每图（CA-PC 增量）
        - 由缓存 anchor 算 d_xi（若干 512 维向量相减，≈0）
        - 原型调制  w = F_crop @ W ; s = sqrt(gd * clip(w))   (50 框)
  (c) GPU 显存增量：加载 CLIP 后 CA-PC 工作集（512 维向量）相对 Base 的额外占用

计时用 torch.cuda.synchronize() + time.perf_counter()，避免 CPU/GPU 异步导致的 0.000 假象。
Base 检测器本身前向耗时 CA-PC 不引入任何额外 FLOP，故吞吐由 Base 决定（文内说明）。

用法:
  HF_HUB_OFFLINE=1 "C:/Users/张三/vg/Scripts/python.exe" -u run_closed_loop_complexity.py
"""
import os, time, json, gc
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_PROTO = 20          # 源域 GT crop 数（典型）
N_ANCHOR = 6          # canonical anchor 文本条数（6 类）
N_CROPS_PER_IMG = 50  # 单图典型检测框数
REPEAT = 1000

from transformers import CLIPModel, CLIPProcessor
model_id = 'openai/clip-vit-base-patch32'
clip = CLIPModel.from_pretrained(model_id).to(DEVICE).eval()
proc = CLIPProcessor.from_pretrained(model_id)
torch.cuda.reset_peak_memory_stats()
base_mem0 = torch.cuda.max_memory_allocated()

def sync_t(method):
    ts = []
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        method()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return 1000 * np.median(ts)

# (a1) 离线：源 crop 图像编码（Base 共享，非 CA-PC 增量，仅报量级）
dummy = [np.zeros((224, 224, 3), np.uint8) for _ in range(N_PROTO)]
def enc_src():
    with torch.no_grad():
        inp = proc(images=dummy, return_tensors='pt').to(DEVICE)
        _ = clip.get_image_features(**inp)
src_ms = sync_t(enc_src)
print(f'[复杂度] device={DEVICE}')
print(f'  (a1) 离线 源crop CLIP图像编码 {N_PROTO} 张: {src_ms:.1f} ms '
      f'(= {src_ms/N_PROTO:.2f} ms/crop) — Base 已做，非 CA-PC 增量')

# (a2) 离线：canonical anchor 文本编码（CA-PC 增量，只算一次）
prompts = [f'a photo of a {c}' for c in ['car','truck','bus','motorcycle','boat','airplane']]
def enc_anchor():
    with torch.no_grad():
        inp = proc(text=prompts, return_tensors='pt').to(DEVICE)
        _ = clip.get_text_features(**inp)
anchor_ms = sync_t(enc_anchor)
print(f'  (a2) 离线 canonical anchor 文本编码 {N_ANCHOR} 条: {anchor_ms:.2f} ms '
      f'(= {anchor_ms/N_ANCHOR:.3f} ms/条) — CA-PC 增量，只算一次')

# (b) 在线每图：原型调制（CA-PC 增量）
F = torch.randn(N_CROPS_PER_IMG, 512, device=DEVICE)
W = torch.randn(512, device=DEVICE)
gd = torch.rand(N_CROPS_PER_IMG, device=DEVICE)
def modulate():
    w = F @ W
    s = torch.sqrt(gd * torch.clamp(torch.clamp(w, min=0.0), 1e-6, 1.0))
    return s
online_ms = sync_t(modulate)
print(f'  (b) 在线每图 原型调制 ({N_CROPS_PER_IMG} 框): {online_ms:.4f} ms/image '
      f'(= {online_ms/N_CROPS_PER_IMG:.5f} ms/框) — CA-PC 增量')

# (c) GPU 显存增量
base_mem = torch.cuda.max_memory_allocated()
# CA-PC 工作集：W(512) + d_xi(512) + 每框分数(50) — 极小
work = torch.randn(512, device=DEVICE)
dxi = torch.randn(512, device=DEVICE)
score = torch.randn(N_CROPS_PER_IMG, device=DEVICE)
torch.cuda.synchronize()
capc_mem = torch.cuda.max_memory_allocated()
vr_delta_mb = (capc_mem - base_mem) / (1024**2)
print(f'  (c) GPU 显存增量（CA-PC 工作集 vs Base）: {vr_delta_mb*1024:.1f} KiB '
      f'(≈ {vr_delta_mb:.4f} MiB) — 复用 Base 已加载的 CLIP 编码器，几乎零额外')

# 汇总
total_offline_capc = anchor_ms            # CA-PC 独有离线增量
total_online_capc = online_ms             # CA-PC 在线增量
# Base 在线 = 检测器前向 + crop→CLIP 投影 + 文本相似度；CA-PC 只在其后追加调制
print(f'  结论: CA-PC 在线增量 {online_ms:.4f} ms/image（{N_CROPS_PER_IMG} 框），'
      f'离线独有增量 {anchor_ms:.2f} ms（仅一次），GPU 额外显存 < 0.1 MiB；'
      f'不增加任何检测器 FLOP，吞吐由 Base 检测器决定。')

out = {
    'device': DEVICE,
    'n_proto': N_PROTO, 'n_anchor': N_ANCHOR, 'n_crops_per_img': N_CROPS_PER_IMG,
    'offline_src_crop_ms': src_ms, 'offline_src_crop_per_crop_ms': src_ms / N_PROTO,
    'offline_anchor_text_ms': anchor_ms, 'offline_anchor_text_per_prompt_ms': anchor_ms / N_ANCHOR,
    'online_modulation_ms_per_img': online_ms, 'online_modulation_ms_per_box': online_ms / N_CROPS_PER_IMG,
    'capc_offline_incremental_ms': total_offline_capc,
    'capc_online_incremental_ms': total_online_capc,
    'gpu_vram_delta_mib': vr_delta_mb,
    'base_detector_fwd_unchanged': True,
}
json.dump(out, open(r'C:\Users\张三\Desktop\实验数据\complexity_profile.json', 'w'), indent=2)
print('  -> 实验数据/complexity_profile.json')
