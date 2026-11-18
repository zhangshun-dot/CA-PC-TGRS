# -*- coding: utf-8 -*-
"""由 run_closed_loop_multidomain.py 生成 v2：加入第 4 个源域 VEDAI。

原脚本与其输出 run_closed_loop_multidomain.json 是论文 tab:multi 的权威来源，**保持不动**；
v2 写到 run_closed_loop_multidomain_v2.json。
"""
import sys, io

SRC = r'C:\Users\张三\Desktop\实验代码\run_closed_loop_multidomain.py'
DST = r'C:\Users\张三\Desktop\实验代码\run_closed_loop_multidomain_v2.py'

s = open(SRC, encoding='utf-8').read()
orig = s
n = 0


def rep(old, new, label):
    global s, n
    assert old in s, f'未命中: {label}'
    assert s.count(old) == 1, f'多处命中({s.count(old)}): {label}'
    s = s.replace(old, new); n += 1


# 1) 输出文件（绝不覆盖权威 json）+ 设备可配
rep("OUT = Path(r'C:\\Users\\张三\\Desktop\\实验数据\\run_closed_loop_multidomain.json')",
    "OUT = Path(r'C:\\Users\\张三\\Desktop\\实验数据\\run_closed_loop_multidomain_v2.json')",
    'OUT')

rep("RB.DEVICE = 'cuda'\nDEVICE = 'cuda'",
    "RB.DEVICE = DEVICE = os.environ.get('MD_DEVICE', 'cuda')\n"
    "# MD_ONLY 可只跑指定源域（小样本验证用），逗号分隔，如 MD_ONLY=vedai",
    'DEVICE')

# 2) VEDAI 常量与 label->name 映射（本机实测 counts 与两份公开来源逐类吻合）
rep("NWPU_ROOT = Path(r'C:\\Users\\张三\\modeling\\nwpu_data\\NWPU-VHR-10')",
    "NWPU_ROOT = Path(r'C:\\Users\\张三\\modeling\\nwpu_data\\NWPU-VHR-10')\n"
    "VEDAI_ROOT = Path(r'C:\\Users\\张三\\modeling\\dior_data\\VEDAI')\n"
    "# VEDAI 官方 label（anno 第 13 列）。1/2/5/9/10/11/23/31 与 moark 数据集页、\n"
    "# github long1014/vedai 两份独立来源 + 本机实测 counts 逐类吻合。\n"
    "VEDAI_LABEL2NAME = {1: 'car', 2: 'truck', 4: 'tractor', 5: 'camping car',\n"
    "                    9: 'van', 10: 'other', 11: 'pickup', 23: 'boat', 31: 'plane'}",
    'VEDAI_ROOT')

# 3) UNITS 增加 VEDAI 4 个类对
rep("    ('nwpu', '10', 'car', 'vehicle'),\n]",
    "    ('nwpu', '10', 'car', 'vehicle'),\n"
    "    ('vedai', 'plane', 'airplane', 'plane'),\n"
    "    ('vedai', 'boat', 'boat', 'boat'),\n"
    "    ('vedai', 'car', 'car', 'car'),\n"
    "    ('vedai', 'truck', 'truck', 'truck'),\n]",
    'UNITS')

# 4) load_vedai：读官方 annotation512.txt（15 列，与 DevKit watch_image.m 逐列吻合）
LOAD_VEDAI = '''
def load_vedai(name, offset, n):
    """VEDAI 512 RGB(_co) 子集。annotation512.txt 每行 15 列:
        imgid cx cy angle X1..X4 Y1..Y4 label occ crop
       oriented 四点 -> AABB(min/max)，>4px 过滤（与 load_dota/load_nwpu 一致）。"""
    name2lab = {v: k for k, v in VEDAI_LABEL2NAME.items()}
    if name not in name2lab:
        return [], []
    target = name2lab[name]
    ann = VEDAI_ROOT / 'annotations' / 'Annotations512' / 'annotation512.txt'
    imgdir = VEDAI_ROOT / 'images' / 'Vehicules512'
    if not ann.exists():
        print(f'  !! VEDAI 缺 {ann}'); return [], []
    per = {}
    for l in open(ann, encoding='utf-8', errors='ignore'):
        p = l.strip().split()
        if len(p) != 15:
            continue
        try:
            iid = int(p[0]); lab = int(float(p[12]))
            xs = [float(v) for v in p[4:8]]; ys = [float(v) for v in p[8:12]]
        except ValueError:
            continue
        if lab != target:
            continue
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if x2 - x1 > 4 and y2 - y1 > 4:
            per.setdefault(iid, []).append([x1, y1, x2, y2])
    out = []
    for iid in sorted(per):
        ip = imgdir / f'{iid:08d}_co.png'
        if ip.exists():
            out.append((ip, per[iid]))
    sel = out[offset:offset + n]
    imgs = [Image.open(p).convert('RGB') for p, _ in sel]
    return imgs, [b for _, b in sel]


def load_src_imgs(ds, cls, n):'''
rep("\ndef load_src_imgs(ds, cls, n):", LOAD_VEDAI, 'load_vedai')

# 5) load_src_imgs 分支
rep("    if ds == 'nwpu':\n        return load_nwpu(cls, 0, n)\n    return [], []",
    "    if ds == 'nwpu':\n        return load_nwpu(cls, 0, n)\n"
    "    if ds == 'vedai':\n        return load_vedai(cls, 0, n)\n    return [], []",
    'load_src_imgs 分支')

# 6) bank 域集合
rep("    src_banks = {'dior': {}, 'dota': {}, 'nwpu': {}}",
    "    src_banks = {'dior': {}, 'dota': {}, 'nwpu': {}, 'vedai': {}}", 'src_banks')

rep("    for ds in ('dota', 'nwpu'):\n        for dsc in sorted({sc for d, sc, _, _ in UNITS if d == ds}):",
    "    for ds in ('dota', 'nwpu', 'vedai'):\n"
    "        if ds not in ONLY:\n            continue\n"
    "        for dsc in sorted({sc for d, sc, _, _ in units if d == ds}):",
    'bank 循环')

# 7) MD_ONLY 过滤
rep("    COCO_EVAL = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']",
    "    only = os.environ.get('MD_ONLY', '').strip()\n"
    "    ONLY = [d for d in only.split(',') if d] if only else ['dior', 'dota', 'nwpu', 'vedai']\n"
    "    units = [u for u in UNITS if u[0] in ONLY]\n"
    "    print(f'[v2] 源域={ONLY}  units={len(units)}  device={DEVICE}')\n"
    "    COCO_EVAL = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']",
    'MD_ONLY')

rep("    for ds, src_cls, eval_cls, pname in UNITS:",
    "    for ds, src_cls, eval_cls, pname in units:", '主循环')

# 8) 汇总
rep("    for ds in ['dior', 'dota', 'nwpu']:\n        rs = per_domain.get(ds, [])",
    "    for ds in ['dior', 'dota', 'nwpu', 'vedai']:\n        rs = per_domain.get(ds, [])",
    'per_domain 汇总')

rep("        for ds in ['dior', 'dota', 'nwpu']:\n"
    "            found = [k for k in results if k.endswith(f'->{cls}') and k.startswith(ds + '_')]",
    "        for ds in ['dior', 'dota', 'nwpu', 'vedai']:\n"
    "            found = [k for k in results if k.endswith(f'->{cls}') and k.startswith(ds + '_')]",
    '共用类对比')

# 9) config 标注版本
rep("'detector': 'gdino-tiny', 'target': 'COCO', 'reuse_cache': 'probe_bidir_cache'}",
    "'detector': 'gdino-tiny', 'target': 'COCO', 'reuse_cache': 'probe_bidir_cache',\n"
    "                          'version': 'v2 (+VEDAI 4th source)', 'sources_ran': ONLY, 'device': DEVICE}",
    'config')

open(DST, 'w', encoding='utf-8').write(s)
print(f'替换 {n} 处 -> {DST}')
print(f'原文件未改动: {len(orig)} chars; 新文件 {len(s)} chars')
