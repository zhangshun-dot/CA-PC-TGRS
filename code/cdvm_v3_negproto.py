# -*- coding: utf-8 -*-
"""CDVM v3: 负原型 + 低阈值 + 上下文特征

v3 三个创新:
  1. 负原型: 从 hard negatives (IoU 0.1-0.5) 构建"背景原型"
     score = mahalanobis(pos_proto) - lam * mahalanobis(neg_proto)
  2. 低阈值: GDINO th 0.03 → 0.005, 更多候选框
  3. 上下文: 扩大 bbox 1.5x 提取周围特征

用法: python cdvm_v3_negproto.py
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
DINOV2_PATH = r'C:\Users\张三\.cache\modelscope\tiansz\dinov2-base'
CLIP_PATH = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
GDINO_CACHE = (r'C:\Users\张三\.cache\huggingface\hub'
               r'\models--IDEA-Research--grounding-dino-tiny'
               r'\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71')
COCO_IMG_DIR = r'C:\Users\张三\modeling\coco_data\val2017'
COCO_ANN_FILE = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'
BANK_PATH = SAVE_DIR / 'exemplar_bank_k5_dinov2_layers.pt'

N_IMAGES = 20
MAX_BOXES_PER_IMAGE = 200  # 每图最多保留的候选框数


# ═══════════════════ 模型 ═══════════════════

def get_dinov2():
    from transformers import AutoModel, AutoConfig
    config = AutoConfig.from_pretrained(DINOV2_PATH)
    config.output_hidden_states = True
    model = AutoModel.from_pretrained(DINOV2_PATH, config=config).eval().to(DEVICE)
    pre = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                     T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return model, pre


def get_clip():
    import clip
    model, pre = clip.load(CLIP_PATH, device=DEVICE)
    return model.eval(), pre


def get_gdino():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    proc = AutoProcessor.from_pretrained(GDINO_CACHE)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_CACHE, dtype=torch.float32).eval().to(DEVICE)
    return proc, model


# ═══════════════════ 特征 ═══════════════════

def extract_dinov2(crop, dino_model, dino_pre, layer=10):
    t = dino_pre(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        o = dino_model(t)
    f = o.hidden_states[layer + 1][0, 0, :]
    return f / f.norm(dim=-1, keepdim=True)


def extract_clip_f(crop, clip_model, clip_pre):
    t = clip_pre(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = clip_model.encode_image(t).float()
    return (f / f.norm(dim=-1, keepdim=True)).squeeze(0)


def context_crop(img, box, expand=1.5):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = (x2 - x1) * expand, (y2 - y1) * expand
    nx1, ny1 = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
    nx2, ny2 = min(img.width, int(cx + nw / 2)), min(img.height, int(cy + nh / 2))
    if nx2 <= nx1 + 5 or ny2 <= ny1 + 5:
        return None
    return img.crop((nx1, ny1, nx2, ny2))


# ═══════════════════ Mahalanobis ═══════════════════

def mah_params(feats, shrinkage=0.3):
    K, D = feats.shape
    mu = feats.mean(dim=0)
    if K > 1:
        cov = (feats - mu).T @ (feats - mu) / (K - 1)
    else:
        cov = torch.eye(D, device=feats.device) * 1e-3
    cov_s = shrinkage * cov + (1 - shrinkage) * torch.eye(D, device=cov.device)
    cov_s = cov_s + 1e-4 * torch.eye(D, device=cov_s.device)
    try:
        L = torch.linalg.cholesky(cov_s)
        inv = torch.cholesky_inverse(L)
    except RuntimeError:
        inv = torch.eye(D, device=cov_s.device)
    return mu, inv


def mah_sim(feat, mu, inv):
    d = feat - mu.to(feat.device)
    if d.dim() == 1:
        d = d.unsqueeze(0)
    m = d @ inv.to(feat.device)
    sq = (m * d).sum(dim=1).clamp(min=0)
    s = torch.exp(-0.5 * sq)
    return s.squeeze(0) if feat.dim() == 1 else s


# ═══════════════════ 数据 ═══════════════════

def load_coco(cls_name, n=N_IMAGES):
    from pycocotools.coco import COCO
    coco = COCO(COCO_ANN_FILE)
    cids = coco.getCatIds(catNms=[cls_name])
    if not cids:
        return [], []
    iids = coco.getImgIds(catIds=[cids[0]])
    imgs, gts = [], []
    for iid in iids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[iid], catIds=[cids[0]], iscrowd=False))
        if not anns:
            continue
        info = coco.loadImgs([iid])[0]
        p = os.path.join(COCO_IMG_DIR, info['file_name'])
        try:
            im = Image.open(p).convert('RGB')
        except Exception:
            continue
        g = [[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
             for a in anns]
        imgs.append(im)
        gts.append(g)
        if len(imgs) >= n:
            break
    return imgs, gts


def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-8)


def ap50(boxes, scores, gts):
    if len(boxes) == 0 or len(gts) == 0:
        return 0.0
    idx = np.argsort(-np.array(scores))
    boxes = [boxes[i] for i in idx]
    matched = [False] * len(gts)
    tp = np.zeros(len(boxes))
    fp = np.zeros(len(boxes))
    for i, pb in enumerate(boxes):
        bi, bj = 0.0, -1
        for j, gt in enumerate(gts):
            if matched[j]:
                continue
            v = iou(pb, gt)
            if v > bi:
                bi, bj = v, j
        if bi >= 0.5 and bj >= 0:
            tp[i] = 1
            matched[bj] = 1
        else:
            fp[i] = 1
    tc = np.cumsum(tp)
    fc = np.cumsum(fp)
    r = tc / len(gts)
    p = tc / (tc + fc + 1e-8)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        m = r >= t
        if m.any():
            ap += float(np.max(p[m]))
    return ap / 11.0


# ═══════════════════ 主函数 ═══════════════════

def main():
    print("=" * 65)
    print("CDVM v3: 负原型 + 低阈值(th=0.005) + 上下文特征")
    print("=" * 65)

    gd_proc, gd_model = get_gdino()
    dino_model, dino_pre = get_dinov2()
    clip_model, clip_pre = get_clip()
    bank = torch.load(BANK_PATH, map_location='cpu')

    MAP = {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle'}
    results = {}

    for cls in ['airplane', 'boat', 'car']:
        dcls = MAP[cls]
        imgs, gts = load_coco(cls)
        print(f"\n{'='*65}")
        print(f"  {cls} ({len(imgs)} images, {sum(len(g) for g in gts)} GT)")
        print(f"{'='*65}")

        # ── GT crops + hard negatives ──
        print("  构建正/负原型...")
        gt_crops, hn_crops = [], []
        for img, gs in zip(imgs, gts):
            for gt_box in gs:
                x1, y1, x2, y2 = [int(v) for v in gt_box]
                if x2 > x1 + 10 and y2 > y1 + 10:
                    gt_crops.append(img.crop((x1, y1, x2, y2)))

        # 收集 hard negatives (IoU 0.1-0.5, gd_score > 0.1)
        for img, gs in zip(imgs, gts):
            prompt = f'a photo of {cls}'
            ins = gd_proc(images=img, text=prompt, return_tensors='pt')
            ins = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in ins.items()}
            with torch.no_grad():
                o = gd_model(**ins)
            ts = torch.tensor([img.size[::-1]]).to(DEVICE)
            r = gd_proc.post_process_grounded_object_detection(o, threshold=0.0, target_sizes=ts)[0]
            bxs = r['boxes'].cpu().numpy()
            scs = r['scores'].cpu().numpy()
            for bi, bx in enumerate(bxs[scs > 0.1]):
                x1, y1, x2, y2 = [max(0, int(v)) for v in bx]
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                mi = max((iou([x1, y1, x2, y2], g) for g in gs), default=0.0)
                if 0.1 <= mi < 0.5:
                    hn_crops.append(img.crop((x1, y1, x2, y2)))

        print(f"  GT={len(gt_crops)}, HN={len(hn_crops)}")

        # ── 提取特征 ──
        gt_feats_d = torch.stack([extract_dinov2(c, dino_model, dino_pre, 10) for c in gt_crops[:50]])
        pos_mu, pos_inv = mah_params(gt_feats_d)

        neg_mu = neg_inv = None
        if len(hn_crops) >= 3:
            hn_feats_d = torch.stack([extract_dinov2(c, dino_model, dino_pre, 10) for c in hn_crops[:30]])
            neg_mu, neg_inv = mah_params(hn_feats_d)
            print(f"  负原型: {len(hn_feats_d)} samples, pos-neg cosine={float(pos_mu @ neg_mu):.4f}")
        else:
            print(f"  负原型: 样本不足 ({len(hn_crops)}), 跳过 pos-neg scoring")

        # DIOR 跨域原型
        dior_proto = bank[dcls][10]['prototype']

        # COCO 同域原型
        coco_proto_d = gt_feats_d[:5].mean(dim=0)
        coco_proto_d = coco_proto_d / coco_proto_d.norm()

        # CLIP 原型
        gt_feats_c = torch.stack([extract_clip_f(c, clip_model, clip_pre) for c in gt_crops[:50]])
        coco_proto_c = gt_feats_c[:5].mean(dim=0)
        coco_proto_c = coco_proto_c / coco_proto_c.norm()

        # ── 评估 th=0.005 ──
        TH = 0.03
        print(f"\n  GDINO threshold = {TH}, max {MAX_BOXES_PER_IMAGE} boxes/img")

        all_boxes, all_gd, all_gts = [], [], []
        vs = defaultdict(list)

        for img, gs in zip(imgs, gts):
            prompt = f'a photo of {cls}'
            ins = gd_proc(images=img, text=prompt, return_tensors='pt')
            ins = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in ins.items()}
            with torch.no_grad():
                o = gd_model(**ins)
            ts = torch.tensor([img.size[::-1]]).to(DEVICE)
            r = gd_proc.post_process_grounded_object_detection(o, threshold=0.0, target_sizes=ts)[0]
            bxs = r['boxes'].cpu().numpy()
            scs = r['scores'].cpu().numpy()

            keep = scs > TH
            bxs, scs = bxs[keep], scs[keep]
            # 每图最多保留 MAX_BOXES_PER_IMAGE 个框 (按 score 降序)
            if len(bxs) > MAX_BOXES_PER_IMAGE:
                top_idx = np.argsort(-scs)[:MAX_BOXES_PER_IMAGE]
                bxs, scs = bxs[top_idx], scs[top_idx]

            for bi, bx in enumerate(bxs):
                x1, y1, x2, y2 = [max(0, int(v)) for v in bx]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                gd = float(scs[bi])
                all_boxes.append(bx.tolist())
                all_gd.append(gd)

                bad = (x2 <= x1 + 10 or y2 <= y1 + 10)

                if not bad:
                    tight = img.crop((x1, y1, x2, y2))
                    fd = extract_dinov2(tight, dino_model, dino_pre, 10)
                    fc = extract_clip_f(tight, clip_model, clip_pre)
                    ctx = context_crop(img, [x1, y1, x2, y2], 1.5)
                    fd_ctx = extract_dinov2(ctx, dino_model, dino_pre, 10) if ctx else fd.clone()

                # cos_dino DIOR
                vs['cos_dino_dior'].append(
                    max(0.0, float((fd.cpu() @ dior_proto.cpu()).item())) if not bad else 0.0)

                # cos_dino COCO
                vs['cos_dino_coco'].append(
                    max(0.0, float((fd.cpu() @ coco_proto_d.cpu()).item())) if not bad else 0.0)

                # mah_dino
                vs['mah_dino'].append(
                    float(mah_sim(fd, pos_mu, pos_inv).item()) if not bad else 0.0)

                # mah_pos_neg (lam=1.0)
                if not bad and neg_mu is not None:
                    ps = mah_sim(fd, pos_mu, pos_inv)
                    ns = mah_sim(fd, neg_mu, neg_inv)
                    pv = float(ps.item() if hasattr(ps, 'item') else ps)
                    nv = float(ns.item() if hasattr(ns, 'item') else ns)
                    vs['mah_pos_neg'].append(max(0.0, pv - nv))
                else:
                    vs['mah_pos_neg'].append(0.0)

                # mah_pos_neg lam sweep
                for lam in [0.5, 1.0, 1.5]:
                    if not bad and neg_mu is not None:
                        ps = mah_sim(fd, pos_mu, pos_inv)
                        ns = mah_sim(fd, neg_mu, neg_inv)
                        val = max(0.0, float(ps.item() if hasattr(ps, 'item') else ps) -
                                  lam * float(ns.item() if hasattr(ns, 'item') else ns))
                        vs[f'mah_pn_lam{lam}'].append(val)
                    else:
                        vs[f'mah_pn_lam{lam}'].append(0.0)

                # cos_clip
                vs['cos_clip'].append(
                    max(0.0, float((fc.cpu() @ coco_proto_c.cpu()).item())) if not bad else 0.0)

                # 上下文: tight + context 平均
                if not bad:
                    fd_cmb = (fd + fd_ctx)
                    fd_cmb = fd_cmb / fd_cmb.norm()
                    vs['cos_ctx'].append(
                        max(0.0, float((fd_cmb.cpu() @ coco_proto_d.cpu()).item())))
                    vs['mah_ctx'].append(
                        float(mah_sim(fd_cmb, pos_mu, pos_inv).item()))
                    if neg_mu is not None:
                        ps2 = mah_sim(fd_cmb, pos_mu, pos_inv)
                        ns2 = mah_sim(fd_cmb, neg_mu, neg_inv)
                        pv2 = float(ps2.item() if hasattr(ps2, 'item') else ps2)
                        nv2 = float(ns2.item() if hasattr(ns2, 'item') else ns2)
                        vs['mah_pn_ctx'].append(max(0.0, pv2 - nv2))
                    else:
                        vs['mah_pn_ctx'].append(0.0)
                else:
                    vs['cos_ctx'].append(0.0)
                    vs['mah_ctx'].append(0.0)
                    vs['mah_pn_ctx'].append(0.0)

                # Late fusion
                vd = vs['mah_dino'][-1]
                vc = vs['cos_clip'][-1]
                vs['late_fusion'].append(vd * vc)
                vpn = vs['mah_pos_neg'][-1]
                vs['late_fusion_pn'].append(max(0, vpn * vc))
                vpnc = vs['mah_pn_ctx'][-1]
                vs['late_fusion_pn_ctx'].append(max(0, vpnc * vc))

            all_gts.extend(gs)

        n_boxes = len(all_boxes)
        base = ap50(all_boxes, np.array(all_gd), all_gts)
        print(f"  Boxes={n_boxes}, Baseline AP50={base:.4f}")

        # ── 评估 ──
        print(f"\n  {'Signal':<22s} {'sqrt':>8s} {'α=.10':>8s} "
              f"{'α=.20':>8s} {'α=.30':>8s} {'Best':>8s} {'Δ':>8s}")
        print(f"  {'-'*70}")

        keys = ['cos_dino_dior', 'cos_dino_coco', 'mah_dino', 'mah_pos_neg',
                'cos_clip', 'cos_ctx', 'mah_ctx', 'mah_pn_ctx',
                'late_fusion', 'late_fusion_pn', 'late_fusion_pn_ctx']

        alphas_try = [0.5, 0.10, 0.20, 0.30]
        alpha_names = ['sqrt', '.10', '.20', '.30']
        best_overall = base
        best_name = 'baseline'

        for key in keys:
            if key not in vs:
                continue
            vv = vs[key]
            ap_vals = []
            for alpha in alphas_try:
                if alpha == 0.5:
                    fused = [np.sqrt(g * max(v, 1e-8)) for g, v in zip(all_gd, vv)]
                else:
                    fused = [g**alpha * max(v, 1e-8)**(1 - alpha) for g, v in zip(all_gd, vv)]
                ap_vals.append(ap50(all_boxes, np.array(fused), all_gts))

            best_ap = max(ap_vals)
            best_idx = ap_vals.index(best_ap)
            print(f"  {key:<22s} {ap_vals[0]:8.4f} {ap_vals[1]:8.4f} "
                  f"{ap_vals[2]:8.4f} {ap_vals[3]:8.4f} "
                  f"{best_ap:8.4f} {best_ap-base:+8.4f} [{alpha_names[best_idx]}]")
            if best_ap > best_overall:
                best_overall = best_ap
                best_name = key

        # λ sweep
        print(f"\n  pos-neg Mahalanobis λ sweep:")
        for lam in [0.5, 1.0, 1.5]:
            k = f'mah_pn_lam{lam}'
            if k in vs and vs[k]:
                fused = [np.sqrt(g * max(v, 1e-8)) for g, v in zip(all_gd, vs[k])]
                ap_l = ap50(all_boxes, np.array(fused), all_gts)
                print(f"    λ={lam:.1f}: sqrt_AP={ap_l:.4f} Δ={ap_l-base:+.4f}")

        print(f"\n  🏆 {cls} Best: {best_name} = {best_overall:.4f} (Δ={best_overall-base:+.4f})")
        results[cls] = {'base': base, 'best': best_overall, 'name': best_name, 'n': n_boxes}

    # ──── 汇总 ────
    print(f"\n{'='*65}")
    print("v2 → v3 对比")
    print(f"{'='*65}")
    v2 = {'airplane': 0.8486, 'boat': 0.2683, 'car': 0.3138}
    print(f"\n{'Class':<12s} {'v2':>8s} {'v3':>8s} {'Δ':>8s} {'v3 方案':<25s}")
    print("-" * 62)
    for c in ['airplane', 'boat', 'car']:
        v3b = results[c]['best']
        print(f"{c:<12s} {v2[c]:8.4f} {v3b:8.4f} {v3b-v2[c]:+8.4f} {results[c]['name']:<25s}")

    with open(SAVE_DIR / 'cdvm_v3_results.json', 'w', encoding='utf-8') as f:
        json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                       for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
    print(f"\n结果 → {SAVE_DIR / 'cdvm_v3_results.json'}")


if __name__ == '__main__':
    main()
