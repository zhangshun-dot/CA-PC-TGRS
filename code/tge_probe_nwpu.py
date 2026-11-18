# -*- coding: utf-8 -*-
"""tge_probe_nwpu.py — 留一源域验证: NWPU VHR-10 → COCO (6 类, 与主表同协议)

期刊补强 (源域独立性): 主结论 (外推修复失效类) 在第一个独立 RS 源域上复现。
β=2, λ=0.5 全部沿用主表固定值, 不做任何调整 = 真留一源域验证 (参数基于 DIOR→COCO 定)。
类映射 (NWPU 源原型 → COCO 目标类, 与 DIOR→COCO 同结构):
  airplane → airplane (同名), ship → boat (粗→细), vehicle → car/truck/bus/motorcycle (粗→细)
行: baseline / B2(CLIP 真实源原型) / TGE(λ=0 纯外推) / A-CSPC(外推+负原型 λ=0.5)
协议与 tge_probe_fs 完全一致 (N=100/类, 前 K=10, 后 90 图测试, pre-filter 0.03,
小框过滤, 逐图 AP50, sqrt(gd*vis))。
COCO 检测结果复用 tge_probe_fs 的缓存 (tge_probe_dets_COCO-*.pkl, 检测与源域无关)。
NWPU VHR-10 标注格式: GroundTruth/*.txt, 每行 "(x1,y1),(x2,y2),类别号"
类别号: 1-airplane 2-ship 3-storage tank 4-baseball diamond 5-tennis court
6-basketball court 7-ground track field 8-harbor 9-bridge 10-vehicle
输出: 实验数据/tge_probe_nwpu.json + 控制台表
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import re
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_probe_nwpu.json'
NWPU_DIR = Path(r'C:\Users\张三\modeling\nwpu_data\NWPU-VHR-10')

K = 10
BETA = 2.0           # 主表固定, 不挑参
N_COCO = 100         # 与主表协议一致
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'

# 与主表 (tge_probe_fs TARGETS) 完全一致的缓存名 → 直接命中 COCO 检测缓存
TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]

# NWPU VHR-10 全部 10 类 (负原型需要"其他类"池, 保留源域结构)
NWPU_CLASSES = [
    'airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court',
    'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle',
]
NWPU_CLS_ID = {c: i + 1 for i, c in enumerate(NWPU_CLASSES)}  # 1-10


def _find_img(img_dir, stem):
    """文件名回退: 处理前导零/扩展名大小写差异。"""
    for ext in ('.jpg', '.JPG', '.jpeg', '.png', '.tif'):
        for cand in (stem, stem.lstrip('0') or '0', stem.zfill(4)):
            p = img_dir / (cand + ext)
            if p.exists():
                return p
    return None


def load_nwpu(cls_name, n=20):
    """NWPU 该类含 GT 的图 (GroundTruth txt 解析), 取前 n 张。"""
    ann_dir = NWPU_DIR / 'ground_truth'
    img_dir = NWPU_DIR / 'positive_image_set'
    cls_id = NWPU_CLS_ID[cls_name]
    images, gts = [], []
    if not ann_dir.exists():
        print(f'  ❌ 标注目录不存在: {ann_dir}')
        return images, gts
    for txt_path in sorted(ann_dir.glob('*.txt')):
        boxes = []
        for line in txt_path.read_text(encoding='utf-8', errors='ignore').splitlines():
            line = line.strip()
            if not line:
                continue
            nums = [int(v) for v in re.findall(r'-?\d+', line)]
            if len(nums) < 5:  # x1 y1 x2 y2 a (最后一项是类别号)
                continue
            x1, y1, x2, y2, cat = nums[-5], nums[-4], nums[-3], nums[-2], nums[-1]
            if cat != cls_id:
                continue
            boxes.append([x1, y1, x2, y2])
        if not boxes:
            continue
        img_path = _find_img(img_dir, txt_path.stem)
        if img_path is None:
            continue
        try:
            from PIL import Image
            images.append(Image.open(img_path).convert('RGB'))
        except Exception:
            continue
        gts.append(boxes)
        if len(images) >= n:
            break
    return images, gts


def clip_proto(crops):
    if not crops:
        return None
    F = T.encode_clip_batch(crops)
    p = F.mean(dim=0)
    return p / p.norm()


def gt_crops(images, gts):
    crops = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
    return crops


def eval_class(images, gts, dets, P_cal, P_neg):
    """逐图 AP50 (同 tge_probe_fs.eval_class): P_neg=None → TGE 行 (无负原型)。"""
    boxes_all, gd_all, feats, img_ids = [], [], [], []
    for ii, ((boxes, scores), img) in enumerate(zip(dets, images)):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            continue
        for box, sc in zip(boxes[keep], scores[keep]):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            feats.append(img.crop((x1, y1, x2, y2)))
            boxes_all.append([x1, y1, x2, y2])
            gd_all.append(float(sc))
            img_ids.append(ii)
    gd = np.clip(np.array(gd_all), 1e-6, 1.0)
    base = U.compute_ap50_img(U.split_by_img(boxes_all, gd, img_ids, len(images)), gts)
    if feats:
        F = T.encode_clip_batch(feats).cpu().numpy()
        if P_neg is not None:
            vis = np.maximum(0.0, F @ P_cal.cpu().numpy()
                             - U.NEG_LAM * (F @ P_neg.cpu().numpy()))
        else:
            vis = np.maximum(0.0, F @ P_cal.cpu().numpy())
        fused = np.sqrt(gd * np.clip(vis, 1e-6, 1.0))
        ap = U.compute_ap50_img(U.split_by_img(boxes_all, fused, img_ids, len(images)), gts)
    else:
        ap = base
    return base, ap


def main():
    if not NWPU_DIR.exists():
        print(f'❌ NWPU VHR-10 未就绪: {NWPU_DIR}')
        print('   先下载解压: https://data.source.coop/opengeos/geoai/NWPU-VHR-10.zip')
        return

    print('[1/2] NWPU VHR-10 10 类 CLIP 源原型 ...')
    protos = {}
    for cls in NWPU_CLASSES:
        images, gts = load_nwpu(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p
            print('  %-20s N=%3d 图, 原型 ✓' % (cls, len(images)))
        else:
            print('  %-20s 无图, 原型 ✗' % cls)

    print('[2/2] COCO 6 类零样本 (N=100, K=10, 后 90 测试, 逐图 AP50) ...')
    rows = {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = U.load_coco(coco_cls, n=N_COCO)
        dets = T.cached_dets(images, TGT_TMPL.format(coco_cls), name)  # 命中 fs 缓存

        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        P_hat = P_src + BETA * (d_y - s_xs).to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()

        # 与主表相同切分: 前 K=10 图校准段, 后 90 图测试
        test_imgs, test_gts, test_dets = images[K:], gts[K:], dets[K:]

        base, _ = eval_class(test_imgs, test_gts, test_dets, P_src, P_neg)
        _, ap_b2 = eval_class(test_imgs, test_gts, test_dets, P_src, P_neg)
        _, ap_tge = eval_class(test_imgs, test_gts, test_dets, P_hat, None)
        _, ap_cspc = eval_class(test_imgs, test_gts, test_dets, P_hat, P_neg)

        rows[name] = {'src_cls': src_cls, 'baseline': round(base, 4),
                      'b2': round(ap_b2, 4), 'tge_lam0': round(ap_tge, 4),
                      'acspc': round(ap_cspc, 4), 'tge_gain': round(ap_tge - base, 4),
                      'acspc_gain': round(ap_cspc - base, 4),
                      'n_test': len(test_imgs)}
        print('  %-16s base=%.4f  B2=%.4f  TGE=%.4f  A-CSPC=%.4f  [TGE %+.4f | A %+.4f]'
              % (name, base, ap_b2, ap_tge, ap_cspc, ap_tge - base, ap_cspc - base))

    if not rows:
        print('❌ 无有效类')
        return

    mean_row = {k: round(float(np.mean([rows[n][k] for n in rows])), 4)
                for k in ['baseline', 'b2', 'tge_lam0', 'acspc', 'tge_gain', 'acspc_gain']}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'source': 'NWPU VHR-10', 'target': 'COCO val2017',
                              'n_coco': N_COCO, 'k': K, 'beta': BETA,
                              'neg_lam': U.NEG_LAM,
                              'protocol': 'NWPU->COCO, N=100/类, K=10, 后90图, 逐图 AP50, 同主表口径',
                              'note': 'β/λ 沿用 DIOR->COCO 主表固定值, 未针对 NWPU 调整 (留一源域验证)'},
                   'classes': rows, 'mean': mean_row}, f, indent=2, ensure_ascii=False)

    print()
    print('=' * 100)
    print('%-16s %9s %9s %9s %9s %9s %9s' %
          ('class', 'baseline', 'B2', 'TGE(λ0)', 'A-CSPC', 'TGE增益', 'A增益'))
    print('-' * 100)
    for name in [t[0] for t in TARGETS]:
        r = rows.get(name)
        if not r:
            continue
        print('%-16s %9.4f %9.4f %9.4f %9.4f %+9.4f %+9.4f' %
              (name, r['baseline'], r['b2'], r['tge_lam0'], r['acspc'],
               r['tge_gain'], r['acspc_gain']))
    print('-' * 100)
    print('%-16s %9.4f %9.4f %9.4f %9.4f %+9.4f %+9.4f   <- mean' %
          ('mean', mean_row['baseline'], mean_row['b2'], mean_row['tge_lam0'],
           mean_row['acspc'], mean_row['tge_gain'], mean_row['acspc_gain']))
    print('=' * 100)
    print('→ %s' % OUT)
    print('主表对照 (DIOR->COCO): baseline 0.3809 / A-CSPC 0.4461 (+0.0652)')


if __name__ == '__main__':
    main()
