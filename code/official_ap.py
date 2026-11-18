# -*- coding: utf-8 -*-
"""official_ap.py v2 — 用 pycocotools COCOeval 复算 DIOR→COCO 6 类 official AP

阶段 A (baseline, 纯 CPU 离线):
  raw GDINO det -> COCO det 格式 -> COCOeval AP50 / AP@[.5:.95].

阶段 B (GPU/CLIP, 编码一次): 复算三个 A-CSPC 变体 (备忘录第1条):
  - target_tuned (β=2.0, λ=0.5)  -> Oracle / upper-bound  (降级, 非主角)
  - fixed       (β=1.0, λ=0.25) -> A-CSPC-fixed (annotation-free, dev域选参)
  - adaptive (β_eff(c),λ_eff(c), α=1.5, 物理先验R(c)) -> A-CSPC-adaptive (新主角)
并额外用 11-point (K_CALIB=10 skip) 复算各变体, 与稿中 +0.0646/+0.0461 交叉验证.

运行(阶段A):  python official_ap.py
运行(阶段B):  python official_ap.py --adaptive
"""
import os, sys, json, pickle, argparse
import numpy as np

DATA = r"C:\Users\张三\Desktop\实验数据"
COCO_ANN = r"C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json"
COCO_IMG_DIR = r"C:\Users\张三\modeling\coco_data\val2017"

DEV_DIOR = [
    ('airplane',   'airplane'),
    ('boat',       'ship'),
    ('car',        'vehicle'),
    ('truck',      'vehicle'),
    ('bus',        'vehicle'),
    ('motorcycle', 'vehicle'),
]
PRE_FILTER = 0.03
K_CALIB = 10

# 物理先验 (源域DIOR类别级 G/S/I/C), 来自 usc_experiment_v3.py
PHYSICAL_PRIORS = {
    'airplane':   {'G': 0.1, 'S': 0.4, 'I': 0.3, 'C': 0.1},
    'boat':       {'G': 0.5, 'S': 0.7, 'I': 0.8, 'C': 0.4},
    'car':        {'G': 0.9, 'S': 0.8, 'I': 0.7, 'C': 0.7},
    'truck':      {'G': 0.9, 'S': 0.5, 'I': 0.5, 'C': 0.6},
    'bus':        {'G': 0.9, 'S': 0.4, 'I': 0.3, 'C': 0.5},
    'motorcycle': {'G': 0.9, 'S': 0.6, 'I': 0.5, 'C': 0.5},
}
PRIOR_W = {'G': 0.25, 'S': 0.25, 'I': 0.25, 'C': 0.25}  # 等权 = 自动化赋值, annotation-free

# 三个变体配置
TARGET_TUNED = (2.0, 0.5)   # Oracle / upper-bound
FIXED        = (1.0, 0.25)  # A-CSPC-fixed
ADAPT_BASE   = (1.0, 0.25)  # adaptive 的 β_base, λ_base
ADAPT_ALPHA  = 1.5


def risk_score(coco_cls):
    p = PHYSICAL_PRIORS[coco_cls]
    return sum(PRIOR_W[f] * p[f] for f in 'GSIC')


def adaptive_beta_lam(coco_cls):
    R = risk_score(coco_cls)
    b = ADAPT_BASE[0] * (1.0 + ADAPT_ALPHA * R)
    l = ADAPT_BASE[1] * (1.0 + ADAPT_ALPHA * R)
    return b, l


def xyxy_to_xywh(b):
    x1, y1, x2, y2 = [float(v) for v in b]
    return [x1, y1, x2 - x1, y2 - y1]


def load_valid_iids(coco, cat_id):
    iids = coco.getImgIds(catIds=[cat_id])
    valid = []
    for iid in iids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[iid], catIds=[cat_id], iscrowd=False))
        if not anns:
            continue
        info = coco.loadImgs([iid])[0]
        if not os.path.exists(os.path.join(COCO_IMG_DIR, info['file_name'])):
            continue
        valid.append(iid)
    return valid


def baseline_dets(coco, cat_id, valid_iids, coco_cls):
    pkl = os.path.join(DATA, f"tge_v2_dets_COCO-COCO-{coco_cls}.pkl")
    with open(pkl, "rb") as f:
        dets = pickle.load(f)
    if len(dets) != len(valid_iids):
        print(f"  ⚠ 长度不符 {coco_cls}: pkl={len(dets)} valid_iids={len(valid_iids)}")
        n = min(len(dets), len(valid_iids))
        dets = dets[:n]; valid_iids = valid_iids[:n]
    out = []
    for iid, (boxes, scores) in zip(valid_iids, dets):
        boxes = np.array(boxes, dtype=float); scores = np.array(scores, dtype=float)
        for b, s in zip(boxes, scores):
            out.append({"image_id": int(iid), "category_id": int(cat_id),
                        "bbox": xyxy_to_xywh(b), "score": float(s)})
    return out


def collect_features(images, gts, dets, T):
    per_img, crops_all = [], []
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2 = min(img.width, x2); y2 = min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < PRE_FILTER:
                continue
            bl.append([x1, y1, x2, y2]); sl.append(float(sc)); ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({
            "box": np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
            "gd": np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
            "gt": np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
            "crop_idx": np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64),
        })
    F = T.encode_clip_batch(crops_all).cpu().numpy() if crops_all else np.zeros((0, 512))
    return per_img, F


def prepare_class(coco, cat_id, coco_cls, src_cls, protos, valid_iids, T, U):
    """编码一次 CLIP, 返回该类的 (per_img, F, P_src, P_neg, Delta)."""
    images, gts = U.load_coco(coco_cls, n=100000)
    assert len(images) == len(valid_iids), \
        f"{coco_cls}: load_coco 图数 {len(images)} ≠ valid_iids {len(valid_iids)}"
    pkl = os.path.join(DATA, f"tge_v2_dets_COCO-COCO-{coco_cls}.pkl")
    with open(pkl, "rb") as f:
        dets = pickle.load(f)
    per_img, F = collect_features(images, gts, dets, T)
    P_src = protos[src_cls].to(U.DEVICE)
    P_neg = T.get_neg_proto(protos, src_cls)
    s_xs = T.text_emb([f"an overhead image of {src_cls}"]).cpu()[0]
    d_y = T.text_emb([f"a photo of {coco_cls}"]).cpu()[0]
    Delta = d_y - s_xs
    return per_img, F, P_src, P_neg, Delta


def build_W(P_src, P_neg, Delta, beta, lam, U):
    P_hat = P_src + beta * Delta.to(U.DEVICE)
    P_hat = P_hat / P_hat.norm()
    return P_hat.cpu().numpy() - lam * P_neg.cpu().numpy()


def fusion_dets(per_img, F, W, valid_iids, cat_id):
    """几何平均融合 (给定 W) -> COCO det 列表."""
    out = []
    for iid, p in zip(valid_iids, per_img):
        n = len(p["gd"])
        if n == 0:
            continue
        vis = np.maximum(0.0, F[p["crop_idx"]] @ W)
        fused = np.sqrt(p["gd"] * np.clip(vis, 1e-6, 1.0))
        for b, s in zip(p["box"], fused):
            out.append({"image_id": int(iid), "category_id": int(cat_id),
                        "bbox": xyxy_to_xywh(b), "score": float(s)})
    return out


def eval_cat(coco, dets, cat_id, valid_iids):
    from pycocotools.cocoeval import COCOeval
    coco_dt = coco.loadRes(dets)
    e = COCOeval(coco, coco_dt, "bbox")
    e.params.catIds = [cat_id]
    e.params.imgIds = valid_iids
    e.evaluate(); e.accumulate(); e.summarize()
    return {"AP": float(e.stats[0]), "AP50": float(e.stats[1]), "AP75": float(e.stats[2])}


def ap11(per_img, key):
    """11-point AP (K_CALIB skip), 与稿中协议一致."""
    import unified_eval as U
    sel = per_img[K_CALIB:] if len(per_img) > K_CALIB else per_img
    preds = [(p["box"], p[key]) for p in sel]
    gts = [p["gt"] for p in sel]
    return U.compute_ap50_img(preds, gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptive", action="store_true",
                    help="复算 fixed/adaptive/target_tuned 三个 A-CSPC 变体 official AP (需 GPU/CLIP)")
    args = ap.parse_args()

    from pycocotools.coco import COCO
    coco = COCO(COCO_ANN)
    catid_of = {cls: coco.getCatIds(catNms=[cls])[0] for cls, _ in DEV_DIOR}

    print("=" * 78)
    print("阶段A: baseline (raw GDINO) official AP — DIOR→COCO 6 类")
    print("=" * 78)
    base, valid_of = {}, {}
    for coco_cls, src_cls in DEV_DIOR:
        cat_id = catid_of[coco_cls]
        valid_iids = load_valid_iids(coco, cat_id)
        valid_of[coco_cls] = valid_iids
        dets = baseline_dets(coco, cat_id, valid_iids, coco_cls)
        r = eval_cat(coco, dets, cat_id, valid_iids)
        base[coco_cls] = r
        print(f"  {coco_cls:<12} n_img={len(valid_iids):>4}  AP50={r['AP50']:.4f}  AP[.5:.95]={r['AP']:.4f}")
    base50 = float(np.mean([base[c]["AP50"] for c, _ in DEV_DIOR]))
    base_all = float(np.mean([base[c]["AP"] for c, _ in DEV_DIOR]))
    print(f"\n  baseline 均值: AP50={base50:.4f}  AP[.5:.95]={base_all:.4f}")

    if args.adaptive:
        print("\n" + "=" * 78)
        print("阶段B: A-CSPC 三变体 official AP (GPU/CLIP, 编码一次)")
        print("=" * 78)
        import unified_eval as U
        import tge_probe as T
        import tge_v2_paper_main_recompute as M
        print("[build_protos] DIOR 20 类 CLIP 原型 ...", flush=True)
        protos = M.build_protos()

        variants = {
            "target_tuned": TARGET_TUNED,
            "fixed":        FIXED,
            "adaptive":     None,  # 逐类 β_eff/λ_eff
        }
        res, res11 = {n: {} for n in variants}, {n: {} for n in variants}

        for coco_cls, src_cls in DEV_DIOR:
            cat_id = catid_of[coco_cls]
            valid_iids = valid_of[coco_cls]
            per_img, F, P_src, P_neg, Delta = prepare_class(
                coco, cat_id, coco_cls, src_cls, protos, valid_iids, T, U)
            base11 = ap11(per_img, "gd")

            for vname, cfg in variants.items():
                beta, lam = (adaptive_beta_lam(coco_cls) if vname == "adaptive" else cfg)
                W = build_W(P_src, P_neg, Delta, beta, lam, U)
                dets = fusion_dets(per_img, F, W, valid_iids, cat_id)
                r = eval_cat(coco, dets, cat_id, valid_iids)
                res[vname][coco_cls] = r
                # 11-point 交叉验证 (同 W)
                for p in per_img:
                    n = len(p["gd"])
                    if n == 0:
                        p["fused"] = np.zeros(0); continue
                    vis = np.maximum(0.0, F[p["crop_idx"]] @ W)
                    p["fused"] = np.sqrt(p["gd"] * np.clip(vis, 1e-6, 1.0))
                v11 = ap11(per_img, "fused")
                res11[vname][coco_cls] = {"ap11": float(v11), "gain11": float(v11 - base11)}
                print(f"  [{vname:<11}] {coco_cls:<12} β={beta:.3f} λ={lam:.3f} "
                      f"AP50={r['AP50']:.4f} AP={r['AP']:.4f} | 11pt={v11:.4f} (+{v11-base11:+.4f})",
                      flush=True)

        print("\n" + "=" * 78)
        print("汇总: 各变体 official AP 增益 (vs baseline)")
        print("=" * 78)
        summary = {"baseline_AP50": {c: base[c]["AP50"] for c, _ in DEV_DIOR},
                   "baseline_AP": {c: base[c]["AP"] for c, _ in DEV_DIOR},
                   "mean_baseline_AP50": base50, "mean_baseline_AP": base_all}
        for vname in variants:
            m50 = float(np.mean([res[vname][c]["AP50"] for c, _ in DEV_DIOR]))
            mall = float(np.mean([res[vname][c]["AP"] for c, _ in DEV_DIOR]))
            g50, gall = m50 - base50, mall - base_all
            g11 = float(np.mean([res11[vname][c]["gain11"] for c, _ in DEV_DIOR]))
            summary[f"{vname}_AP50"] = {c: res[vname][c]["AP50"] for c, _ in DEV_DIOR}
            summary[f"{vname}_AP"] = {c: res[vname][c]["AP"] for c, _ in DEV_DIOR}
            summary[f"mean_{vname}_AP50"] = m50
            summary[f"mean_{vname}_AP"] = mall
            summary[f"gain_{vname}_AP50"] = g50
            summary[f"gain_{vname}_AP"] = gall
            summary[f"gain_{vname}_11pt"] = g11
            print(f"  {vname:<11}: AP50 gain={g50:+.4f}  AP gain={gall:+.4f}  11pt gain={g11:+.4f}")
        print(f"\n  稿声称: target_tuned 11pt +0.0646 / official +0.0654; adaptive 11pt +0.0461")
        out = os.path.join(DATA, "official_ap_recompute.json")
        with open(out, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  结果 -> {out}")


if __name__ == "__main__":
    main()
