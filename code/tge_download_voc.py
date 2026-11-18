# -*- coding: utf-8 -*-
"""tge_download_voc.py — 下载 VOC2007 trainval (多镜像兜底) + 解压

候选源 (按顺序尝试):
  1. pjreddie 镜像  http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
  2. ox.ac.uk 官方  http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  3. HF pascal_voc  (匿名, verify=False; 上次 401 则跳过)

下载到 modeling/voc_data/VOCdevkit/VOC2007/ (JPEGImages + Annotations), ~500MB。
"""
import sys
import tarfile
import urllib.request
import ssl
from pathlib import Path

LOCAL = Path(r'C:\Users\张三\modeling\voc_data')
TAR = LOCAL / 'VOCtrainval_06-Nov-2007.tar'
OK_DIR = LOCAL / 'VOCdevkit' / 'VOC2007'

SOURCES = [
    'http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar',
    'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
]

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def try_download(url):
    print(f'  尝试 {url}')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=CTX, timeout=300) as r, open(TAR, 'wb') as f:
        total = int(r.headers.get('Content-Length', 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f'    {done / 1e6:.0f}/{total / 1e6:.0f} MB', end='\r')
    print()
    return TAR.stat().st_size > 50e6      # 至少 50MB 才算成功


def extract():
    print('解压 ...')
    with tarfile.open(TAR) as tf:
        # Python 3.9 无 filter 参数 (3.12+ 才有); 数据来自可信官方源
        tf.extractall(LOCAL)
    jpg = OK_DIR / 'JPEGImages'
    xml = OK_DIR / 'Annotations'
    n_jpg = len(list(jpg.glob('*.jpg'))) if jpg.exists() else 0
    n_xml = len(list(xml.glob('*.xml'))) if xml.exists() else 0
    print(f'  JPEGImages: {n_jpg}, Annotations: {n_xml}')
    return n_jpg > 4000 and n_xml > 4000


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    if (OK_DIR / 'JPEGImages').exists() and len(list((OK_DIR / 'JPEGImages').glob('*.jpg'))) > 4000:
        print('✅ VOC2007 已就绪, 跳过')
        return
    if TAR.exists() and TAR.stat().st_size > 50e6:
        print('tar 已存在 (%d MB), 直接解压' % (TAR.stat().st_size / 1e6))
        if extract():
            print(f'✅ VOC2007 -> {OK_DIR}')
            return
    for url in SOURCES:
        try:
            if try_download(url):
                if extract():
                    print(f'✅ VOC2007 -> {OK_DIR}')
                    return
        except Exception as e:
            print(f'  ❌ {e}')
    print('❌ 全部源失败, 请手动下载 VOCtrainval_06-Nov-2007.tar 到 %s' % LOCAL)


if __name__ == '__main__':
    main()
