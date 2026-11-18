# -*- coding: utf-8 -*-
"""download_sar_datasets.py — 元 2026-10-15

通过 huggingface_hub（走用户代理 HTTPS_PROXY）把 Risk① 需要的真实 SAR 数据集拉到本地，
并**扁平化**到 score_sar_sensor_multiclass.py 期望的目录布局：

  HRSID             -> sar_data/HRSID/images/*.jpg  + sar_data/HRSID/labels/*.txt   (YOLO, class0=ship)
  SAR-AIRcraft-1.0  -> sar_data/SAR-AIRcraft-1.0/JPEGImages/*.jpg + .../Annotations/*.xml (VOC, 机型名)

HF 布局是嵌套的 (data/images/train/...)，loader 的 anndir/(stem+'.txt') 找不到，
所以这里全部拍平到 images/ 与 labels/（或 JPEGImages/ 与 Annotations/）。

用法:
  "C:/Users/张三/vg/Scripts/python.exe" -u download_sar_datasets.py
"""
import os, shutil, sys, time, glob
from pathlib import Path

# 禁用 huggingface_hub 的 safe-delete 交互确认（Windows 沙箱无 trash，会卡死等输入）
os.environ['HF_HUB_DISABLE_SAFE_DELETE'] = '1'
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

import huggingface_hub as hh

SAR_DATA = Path(r'C:\Users\张三\modeling\sar_data')
TMP = SAR_DATA / '_hf_tmp'
TMP.mkdir(parents=True, exist_ok=True)

REPOS = {
    'HRSID': {
        'repo_id': 'dronefreak/HRSID',
        'allow_patterns': ['data/images/train/**', 'data/labels/train/**', 'data/data.yaml', 'README.md'],
        'flatten': [
            ('data/images/train/**/*.jpg', 'HRSID/images'),
            ('data/labels/train/**/*.txt', 'HRSID/labels'),
        ],
    },
    'SAR-AIRcraft': {
        'repo_id': 'biubushy/sar-aircraft-1.0',
        'allow_patterns': ['Pascal-VOC/JPEGImages/**', 'Pascal-VOC/Annotations/**'],
        'flatten': [
            ('Pascal-VOC/JPEGImages/**/*.jpg', 'SAR-AIRcraft-1.0/JPEGImages'),
            ('Pascal-VOC/Annotations/**/*.xml', 'SAR-AIRcraft-1.0/Annotations'),
        ],
    },
}


def flatten_one(name, spec):
    t0 = time.time()
    local = TMP / name
    print(f'[1/3] snapshot_download {spec["repo_id"]} ...', flush=True)
    hh.snapshot_download(
        repo_id=spec['repo_id'],
        repo_type='dataset',
        local_dir=str(local),
        allow_patterns=spec['allow_patterns'],
        max_workers=8,
    )
    print(f'      snapshot done in {time.time()-t0:.0f}s -> {local}', flush=True)

    for pat, rel in spec['flatten']:
        srcs = sorted(glob.glob(str(local / pat)))
        dst_dir = SAR_DATA / rel
        dst_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for sp in srcs:
            sp = Path(sp)
            dp = dst_dir / sp.name
            if dp.exists():
                continue
            shutil.move(str(sp), str(dp))
            n += 1
        print(f'      moved {n:5d} -> {dst_dir}  (total now {len(list(dst_dir.iterdir()))})', flush=True)

    # 清理临时目录
    shutil.rmtree(str(local), ignore_errors=True)
    print(f'[done] {name} flattened.\n', flush=True)


def sanity():
    print('=== format sanity ===', flush=True)
    # HRSID: 第一个 label 应是归一化 YOLO (值 0..1)
    lbs = sorted((SAR_DATA / 'HRSID' / 'labels').glob('*.txt'))
    if lbs:
        lines = [l for l in open(lbs[0], encoding='utf-8', errors='ignore') if l.strip()]
        print(f'HRSID label sample {lbs[0].name}: {len(lines)} objs; first line -> {lines[0].strip() if lines else "(empty)"}')
        allnorm = all(max(float(v) for v in l.split()[1:5]) <= 1.05 for l in lines if len(l.split()) >= 5)
        print(f'  HRSID labels normalized(<=1.05)? {allnorm}')
    # SAR-AIRcraft: 第一个 xml 的 <name>
    import xml.etree.ElementTree as ET, re
    xls = sorted((SAR_DATA / 'SAR-AIRcraft-1.0' / 'Annotations').glob('*.xml'))
    if xls:
        t = ET.parse(str(xls[0])).getroot()
        names = [o.findtext('name').strip() for o in t.iter('object')]
        print(f'SAR-AIRcraft xml sample {xls[0].name}: names={names}')
    imgs = len(list((SAR_DATA / 'HRSID' / 'images').glob('*.jpg')))
    labs = len(list((SAR_DATA / 'HRSID' / 'labels').glob('*.txt')))
    sj = len(list((SAR_DATA / 'SAR-AIRcraft-1.0' / 'JPEGImages').glob('*.jpg')))
    sa = len(list((SAR_DATA / 'SAR-AIRcraft-1.0' / 'Annotations').glob('*.xml')))
    print(f'counts: HRSID images={imgs} labels={labs} | SAR-AIRcraft jpg={sj} xml={sa}', flush=True)


if __name__ == '__main__':
    t0 = time.time()
    for name, spec in REPOS.items():
        flatten_one(name, spec)
    sanity()
    print(f'\n[ALL DONE {time.time()-t0:.0f}s]', flush=True)
