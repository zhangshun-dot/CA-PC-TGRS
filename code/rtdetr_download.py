# -*- coding: utf-8 -*-
"""rtdetr_download.py — 稳健下载 RT-DETR 权重到本地平铺目录（绕过 blob 缓存 + Windows 符号链接坑）

做法：对仓库每个文件单独 hf_hub_download(local_dir=..., local_dir_use_symlinks=False)，
带重试与长超时。权重文件 model.safetensors 体积大，单独多试几次。
下载完成后用 transformers 从 local_dir 直接加载做最终校验。
"""
import os, sys, time, shutil
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import hf_hub_download
from pathlib import Path

REPO = 'PekingU/rtdetr_r50vd_coco_o365'
LOCAL = Path(r'C:\Users\张三\Desktop\实验数据\rtdetr_r50vd_coco_o365')
LOCAL.mkdir(parents=True, exist_ok=True)

FILES = ['config.json', 'preprocessor_config.json', 'model.safetensors']
MAX_RETRY = 5
TIMEOUT = 600  # 秒


def download_one(fname):
    last = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            print(f'  [{fname}] attempt {attempt}/{MAX_RETRY} ...', flush=True)
            p = hf_hub_download(
                repo_id=REPO,
                filename=fname,
                local_dir=str(LOCAL),
                local_dir_use_symlinks=False,
                local_files_only=False,
            )
            sz = os.path.getsize(p)
            print(f'  [{fname}] OK -> {p}  ({sz/1e6:.1f} MB)', flush=True)
            return p
        except Exception as e:  # noqa
            last = e
            print(f'  [{fname}] attempt {attempt} FAILED: {type(e).__name__}: {e}', flush=True)
            # 清掉可能残留的 .incomplete 半成品，避免下一轮读到坏文件
            for bad in LOCAL.glob(f'{fname}.incomplete*'):
                try:
                    bad.unlink()
                except Exception:
                    pass
            time.sleep(3)
    raise RuntimeError(f'{fname} 多次重试仍失败: {last}')


if __name__ == '__main__':
    print('=' * 70)
    print(f'RT-DETR 稳健下载  repo={REPO}  local={LOCAL}', flush=True)
    print('=' * 70)
    for f in FILES:
        download_one(f)
    # 校验：从本地目录加载
    print('\n[校验] 从 local_dir 加载 transformers 模型 ...', flush=True)
    try:
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        proc = RTDetrImageProcessor.from_pretrained(str(LOCAL))
        model = RTDetrForObjectDetection.from_pretrained(str(LOCAL))
        cfg = model.config
        print('config id2label n =', len(cfg.id2label), flush=True)
        print('sample:', {k: cfg.id2label[k] for k in list(cfg.id2label)[:5]}, flush=True)
        print('RT-DETR LOAD OK from local_dir', flush=True)
    except Exception as e:  # noqa
        print('LOAD VERIFY FAILED:', type(e).__name__, e, flush=True)
        sys.exit(1)
