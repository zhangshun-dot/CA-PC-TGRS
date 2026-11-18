# -*- coding: utf-8 -*-
"""autoproc_spacemigrate_object_probe.py — 形态 A: object-centric CLIP (SAM 提前景, 零训练)

用户新方向(空间迁移/平行时空):
  "在同一个冻结 CLIP 空间里重排/替换原型被锁在 ~+1.25% 天花板, 能不能空间迁移——
   保留物体本身但场景换了, 也就是天花板换了但还是原图。"

本探针测的假设: 跨域差距里"场景偏置"占多大头?
  - 冻结 CLIP 图像编码器不变(还是同一个空间), 只把送入编码器的 crop 从
    "整框(含场景背景)" 换成 "SAM 掩出的物体本身"(背景置灰 / 紧裁物体)。
  - 若 object-centric 特征在【相同 DIOR 文本原型】下 AP50 明显高于 full-crop,
    说明场景偏置是跨域对齐差距的重要来源 → 指向"训练一个 object-centric 投影器"
    才是真正的空间迁移(后续 C'-like)。
  - 若≈持平, 说明场景不是主因, 天花板是 CLIP 空间对齐本身的结构性限制。

协议 100% 复刻 usc_experiment_v12.load_target_coco:
  GDINO th=0.0 → pre-filter 0.03 → per_img{crop_idx} → evaluate(features, proto, neg_proto)。
只用同一候选框集合, 只换图像侧特征, 所以 AP 差 = 场景偏置效应。

输出: 实验数据/autoproc_spacemigrate_object_results.json + 控制台表
"""
import sys
sys.path.insert(0, r'C:\Users/张三/Desktop/实验代码')
import json
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import usc_experiment_v12 as E
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users/张三/Desktop/实验数据')
OUT = DATA / 'autoproc_spacemigrate_object_results.json'
SAM_CKPT = r'C:\Users/张三/Downloads/sam_vit_b_01ec64.pth'
N_VAL = 80                       # 小样本先验证 (用户原则)
PRE_FILTER = 0.03
CLASSES = [                      # (eval_name, coco_cls, src_cls)
    ('airplane', 'airplane', 'airplane'),
    ('boat',     'boat',     'ship'),
    ('car',      'car',      'vehicle'),
    ('truck',    'truck',    'vehicle'),
    ('bus',      'bus',      'vehicle'),
    ('motorcycle','motorcycle','vehicle'),
]

DEVICE = U.DEVICE

_sam = None
def get_sam():
    global _sam
    if _sam is None:
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry['vit_b'](checkpoint=SAM_CKPT)
        sam = sam.to(DEVICE).eval()
        _sam = SamPredictor(sam)
        print(f'  SAM ViT-B 加载完成 ({SAM_CKPT})')
    return _sam


def build_target(coco_cls, n=N_VAL):
    """复刻 load_target_coco 的候选框提取, 额外返回原图+逐框坐标供 SAM。"""
    images, gts = U.load_coco(coco_cls, n=n)
    if not images:
        return None
    cache_p = DATA / f'tge_v2_dets_COCO-{coco_cls}.pkl'
    if cache_p.exists():
        dets = pickle.load(open(cache_p, 'rb'))
    else:
        dets = U.gdino_detect(images, f'a photo of {coco_cls}', 0.0)
        pickle.dump(dets, open(cache_p, 'wb'))
    per_img = []
    crop_records = []          # (img_idx, box_xyxy, full_crop_pil)
    for img_idx, ((boxes, scores), img, g) in enumerate(zip(dets, images, gts)):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < PRE_FILTER:
                continue
            bl.append([x1, y1, x2, y2]); sl.append(float(sc))
            ci.append(len(crop_records))
            crop_records.append((img_idx, (x1, y1, x2, y2),
                                 img.crop((x1, y1, x2, y2))))
        per_img.append({
            'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
            'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
            'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
            'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64),
        })
    return {'images': images, 'gts': gts, 'per_img': per_img,
            'crop_records': crop_records}


def sam_object_variants(crop_records, images):
    """对每张原图 set_image 一次, 逐框 predict 得到掩码, 产出两种 object-only crop。

    返回: obj_crops (紧裁 mask bbox), comp_crops (背景置灰, 保留原位置/尺度)
    """
    predictor = get_sam()
    obj_crops, comp_crops = [], []
    n_fallback = 0
    # 按图分组
    by_img = {}
    for ridx, (img_idx, box, full_crop) in enumerate(crop_records):
        by_img.setdefault(img_idx, []).append((ridx, box, full_crop))
    for img_idx, recs in by_img.items():
        img = images[img_idx]
        arr = np.array(img.convert('RGB'))
        predictor.set_image(arr)
        for ridx, box, full_crop in recs:
            x1, y1, x2, y2 = box
            # 注意: 这套 segment_anything 版本 predict 内部对 box 调 .astype(),
            # 必须传 numpy 数组 (不能传 torch tensor, 否则 AttributeError 全退化)
            box_np = np.array([[x1, y1, x2, y2]], dtype=np.float32)
            try:
                masks, scores, _ = predictor.predict(box=box_np, multimask_output=False)
                mask = (masks[0] > 0.5)
            except Exception:
                mask = None
            if mask is None or mask.sum() < 20:
                n_fallback += 1
                # 退化: 退化为整框
                obj_crops.append(full_crop)
                comp_crops.append(full_crop)
                continue
            ys, xs = np.where(mask)
            bx1, by1, bx2, by2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            # ① 紧裁物体
            tight = full_crop.crop((bx1 - x1, by1 - y1, bx2 - x1, by2 - y1))
            obj_crops.append(tight)
            # ② 背景置灰 (保留原位置/尺度)
            m_crop = mask[y1:y2, x1:x2]
            gray = Image.new('RGB', full_crop.size, (128, 128, 128))
            comp = Image.composite(full_crop,
                                   gray,
                                   Image.fromarray((m_crop * 255).astype('uint8')))
            comp_crops.append(comp)
    print(f'    [SAM] 总 crop={len(crop_records)} 成功抠前景={len(crop_records)-n_fallback} '
          f'退化={n_fallback}')
    return obj_crops, comp_crops


def main():
    print('=' * 70)
    print('形态 A 探针: object-centric CLIP (SAM 提前景, 零训练)')
    print(f'N_VAL={N_VAL}, 设备={DEVICE}, 协议=GDINO th0→pre-filter{PRE_FILTER}→evaluate()')
    print('=' * 70)

    clip_m, _ = T.get_clip()
    rows = {}
    for eval_name, coco_cls, src_cls in CLASSES:
        print(f'\n[{eval_name}] 加载候选框 + SAM 提前景 ...')
        td = build_target(coco_cls, n=N_VAL)
        if td is None or len(td['crop_records']) == 0:
            print('  无候选框, 跳过')
            continue
        full_crops = [r[2] for r in td['crop_records']]
        obj_crops, comp_crops = sam_object_variants(td['crop_records'], td['images'])

        F_full = T.encode_clip_batch(full_crops).cpu().numpy()
        F_obj = T.encode_clip_batch(obj_crops).cpu().numpy()
        F_comp = T.encode_clip_batch(comp_crops).cpu().numpy()

        # DIOR 源原型 (两种口径)
        src = E.load_dior_src(src_cls, n=50)
        proto_raw = src['prototype']                       # 冻结 CLIP 源原型 (无文本delta)
        all_protos = {src_cls: proto_raw}
        neg_raw = E.build_neg_proto(all_protos, src_cls)
        proto_tge = E.build_proto(proto_raw, src_cls, coco_cls,
                                  E.SRC_TMPL_OVERHEAD, use_tge=True)  # TGE 外推 (alpha=0)

        def ap_of(F):
            tgt = {'per_img': td['per_img'], 'features': F,
                   'n_crops': F.shape[0]}
            a_raw = E.evaluate(tgt, proto_raw, neg_raw)
            a_tge = E.evaluate(tgt, proto_tge, neg_raw)
            return float(a_raw), float(a_tge)

        ap_full_raw, ap_full_tge = ap_of(F_full)
        ap_obj_raw, ap_obj_tge = ap_of(F_obj)
        ap_comp_raw, ap_comp_tge = ap_of(F_comp)
        base = E.evaluate_baseline(td)        # GDINO 原始分数 (无CLIP)

        rows[eval_name] = {
            'n_crops': int(F_full.shape[0]),
            'gdino_only': round(base, 4),
            'fullcrop_raw_proto': round(ap_full_raw, 4),
            'fullcrop_tge_proto': round(ap_full_tge, 4),
            'objectcrop_raw_proto': round(ap_obj_raw, 4),
            'objectcrop_tge_proto': round(ap_obj_tge, 4),
            'composite_raw_proto': round(ap_comp_raw, 4),
            'composite_tge_proto': round(ap_comp_tge, 4),
            'delta_obj_vs_full_raw': round(ap_obj_raw - ap_full_raw, 4),
            'delta_obj_vs_full_tge': round(ap_obj_tge - ap_full_tge, 4),
            'delta_comp_vs_full_raw': round(ap_comp_raw - ap_full_raw, 4),
            'delta_comp_vs_full_tge': round(ap_comp_tge - ap_full_tge, 4),
        }
        print(f"  候选框={rows[eval_name]['n_crops']} GDINOonly={base:.4f}")
        print(f"  full  raw={ap_full_raw:.4f} tge={ap_full_tge:.4f}")
        print(f"  obj   raw={ap_obj_raw:.4f} (Δ{ap_obj_raw-ap_full_raw:+.4f}) "
              f"tge={ap_obj_tge:.4f} (Δ{ap_obj_tge-ap_full_tge:+.4f})")
        print(f"  comp  raw={ap_comp_raw:.4f} (Δ{ap_comp_raw-ap_full_raw:+.4f}) "
              f"tge={ap_comp_tge:.4f} (Δ{ap_comp_tge-ap_full_tge:+.4f})")

    # 汇总
    print('\n' + '=' * 70)
    print('汇总 (6 类均值)')
    print('=' * 70)
    def mean(key):
        return float(np.mean([rows[c][key] for c in rows]))
    summary = {k: round(mean(k), 4) for k in next(iter(rows.values()))}
    for k, v in summary.items():
        print(f'  {k:<26s} {v:+.4f}' if 'delta' in k or k == 'gdino_only' else f'  {k:<26s} {v:.4f}')

    print('\n  → 关键结论 (object-crop 相对 full-crop 的增益, 跨域对齐):')
    print(f"    raw 原型: Δobj={summary['delta_obj_vs_full_raw']:+.4f}  "
          f"Δcomp={summary['delta_comp_vs_full_raw']:+.4f}")
    print(f"    tge 原型: Δobj={summary['delta_obj_vs_full_tge']:+.4f}  "
          f"Δcomp={summary['delta_comp_vs_full_tge']:+.4f}")

    json.dump({'config': {'n_val': N_VAL, 'pre_filter': PRE_FILTER,
                          'sam': 'vit_b', 'protocol': 'evaluate() 同候选框, 仅换图像特征',
                          'hypothesis': '场景偏置是否构成跨域 CLIP 对齐差距主因'},
               'per_class': rows, 'summary': summary},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n✅ -> {OUT}')


if __name__ == '__main__':
    main()
