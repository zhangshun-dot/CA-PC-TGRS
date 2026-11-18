# -*- coding: utf-8 -*-
"""r2_cost_memory_measure.py -- R2#3 补齐: baseline 参数量 / 绝对显存峰值 / 多图计时中位数

审稿人 R2#3 明确要求 baseline 的 训练时间、推理时间、参数量、GPU 内存。
已有: GDINO 单图计时 734.2 ms (tge_cost_analysis.json, 单次)、CA-PC 增量显存 0.0 MiB。
缺: ① baseline 自身参数量 ② 绝对显存峰值 (measured peak, 不是增量)
本脚本在同一硬件上实测，输出 实验数据/r2_cost_memory_20261118.json
"""
import sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'r2_cost_memory_20261118.json'
TGT_TMPL = 'a photo of {}'
N_IMG = 20


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def main():
    assert torch.cuda.is_available(), 'CUDA required'
    dev = torch.device('cuda')
    props = torch.cuda.get_device_properties(0)
    print('[env]', props.name, '%.1f GB' % (props.total_memory / 2**30), 'torch', torch.__version__)

    print('[1/4] load GDINO-tiny ...')
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    gd_p, gd_m = U.get_gdino()
    w_mib = torch.cuda.memory_allocated() / 2**20
    n_par = count_params(gd_m)
    print('  params = %.2f M, weights on GPU = %.1f MiB' % (n_par / 1e6, w_mib))

    print('[2/4] COCO car 图像 %d 张 ...' % N_IMG)
    images, gts = U.load_coco('car', n=N_IMG)

    # warmup (第一次 forward 含 kernel 编译/cuDNN autotune，不计入中位数)
    ins0 = gd_p(images=images[0], text=TGT_TMPL.format('car'), return_tensors='pt')
    ins0 = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in ins0.items()}
    with torch.no_grad():
        gd_m(**ins0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    lat = []
    for i, img in enumerate(images):
        ins = gd_p(images=img, text=TGT_TMPL.format('car'), return_tensors='pt')
        ins = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in ins.items()}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            o = gd_m(**ins)
        ts = torch.tensor([img.size[::-1]]).to(dev)
        gd_p.post_process_grounded_object_detection(o, threshold=0.0, target_sizes=ts)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)

    peak_alloc = torch.cuda.max_memory_allocated() / 2**20
    peak_resv = torch.cuda.max_memory_reserved() / 2**20
    lat = np.array(lat)
    print('  latency median %.1f ms  mean %.1f  p90 %.1f  min %.1f  max %.1f'
          % (np.median(lat), lat.mean(), np.percentile(lat, 90), lat.min(), lat.max()))
    print('  peak allocated %.0f MiB, peak reserved %.0f MiB' % (peak_alloc, peak_resv))

    print('[3/4] CA-PC 打分增量 (512 维点积, 1291 框) ...')
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated() / 2**20
    F = torch.rand(1291, 512, device=dev)
    W = torch.rand(512, device=dev)
    for _ in range(20):
        s = torch.sqrt(torch.clamp(F @ W, min=0))
    torch.cuda.synchronize()
    after = torch.cuda.max_memory_allocated() / 2**20
    print('  score-step peak delta %.2f MiB' % (after - before))

    print('[4/4] 写盘 ...')
    out = {
        'config': {
            'device': props.name,
            'total_vram_gb': round(props.total_memory / 2**30, 1),
            'torch': torch.__version__,
            'n_img_timed': int(N_IMG),
            'warmup_excluded': True,
            'detector': 'grounding-dino-tiny (transformers AutoModelForZeroShotObjectDetection)',
            'prompt': TGT_TMPL.format('car'),
            'date': time.strftime('%Y-%m-%d %H:%M'),
        },
        'baseline_detector': {
            'n_params': n_par,
            'n_params_million': round(n_par / 1e6, 2),
            'weights_gpu_mib': round(w_mib, 1),
            'latency_ms_median': round(float(np.median(lat)), 1),
            'latency_ms_mean': round(float(lat.mean()), 1),
            'latency_ms_p90': round(float(np.percentile(lat, 90)), 1),
            'latency_ms_min': round(float(lat.min()), 1),
            'latency_ms_max': round(float(lat.max()), 1),
            'peak_allocated_mib': round(peak_alloc, 1),
            'peak_reserved_mib': round(peak_resv, 1),
        },
        'capc_added': {
            'trainable_params': 0,
            'score_step_peak_delta_mib': round(float(after - before), 2),
            'note': 'online modulation is a 512-d dot product on the cached crop features; '
                    'adds no detector forward pass and no trainable parameter',
        },
    }
    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('saved', OUT)


if __name__ == '__main__':
    main()
