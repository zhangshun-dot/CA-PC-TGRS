# -*- coding: utf-8 -*-
"""expanded_cross_domain.py — 跨域评估扩展（3 源语义概念 × 6 目标类 + 2 语义错配负对照）

论文身份（2026-08-18 用户拍板，跑前锁定，禁止跑后按结果改归属）：
  Evaluation covers SIX target-domain categories derived from THREE source
  semantic concepts, including FOUR target manifestations sharing the same
  source vehicle prototype.

  分组：
    strict            — airplane→airplane, ship→boat, vehicle→car     （近严格跨域）
    vehicle_pressure  — vehicle→car/truck/bus/motorcycle               （上位类→子类压力测试）
    neg_control       — tennis court→sports ball, train station→train  （semantic-mismatch 负对照，仅附录，
                        不进主表均值，不参与"正增益"结论，无论正负都完整报告）

协议（继承 unified_eval）：
  GDINO-tiny th=0.0 取全部候选框 → pre-filter=0.03 → DINOv2-base L12 特征
  → CDVM(负原型相对化, NEG_LAM=0.5) 重打分 → 数据集级 11-point AP50
  K = {0,1,2,3,5,10}；α = {0,0.25,0.5,0.75,1}；N_SPLITS=10 嵌套随机划分
  K=0 = 零样本（P_cal=P_src，与划分无关）
  公平性：每个 (split, K) 下，baseline / 源原型(α=1) / few-shot(各α) 全部在
          同一测试图集（池−校准 K 张）上评估；K=0 时测试=全池。

GPU 每类只跑一次：GDINO 检测 + DINOv2 编码候选框与 GT 框 → 落盘缓存 npz。
  之后所有 K/α/划分 在内存完成（毫秒级），断点续跑只补缺失类。

输出：
  实验数据/expanded_eval_results.json   — 完整 AP 矩阵（class×split×K×α）
  实验数据/expanded_cache/{name}.npz    — 每类逐图候选框/GT 框特征缓存（逐图预测复现用）
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import unified_eval as U

LAYER = 11           # hidden_states[LAYER+1] = L12（final_method_table 固化口径）
NEG_LAM = 0.5
K_LIST = [0, 1, 2, 3, 5, 10]
ALPHA_LIST = [0.0, 0.25, 0.5, 0.75, 1.0]
N_SPLITS = 10
SEED = 42
TEST_POOL = 100
PRE_FILTER = U.PRE_FILTER
CACHE_DIR = U.SAVE_DIR / 'expanded_cache'
CACHE_DIR.mkdir(exist_ok=True)
OUT_JSON = U.SAVE_DIR / 'expanded_eval_results.json'

# (name, src_dior_cls, tgt_coco_cls, group)
MAIN_EVAL = [
    ('COCO-airplane',   'airplane',  'airplane',   'strict'),
    ('COCO-boat',       'ship',      'boat',       'strict'),
    ('COCO-car',        'vehicle',   'car',        'vehicle_pressure'),
    ('COCO-truck',      'vehicle',   'truck',      'vehicle_pressure'),
    ('COCO-bus',        'vehicle',   'bus',        'vehicle_pressure'),
    ('COCO-motorcycle', 'vehicle',   'motorcycle', 'vehicle_pressure'),
]
# (name, src_dior_cls, tgt_coco_cls) — 语义错配负对照，附录
NEG_CONTROL = [
    ('NEG-sports_ball', 'tennis court', 'sports ball'),
    ('NEG-train',       'train station', 'train'),
]


def load_coco_random(cls_name, n, seed):
    """从全部本地图随机抽 n 张含 GT 的图（固定 seed）。"""
    from pycocotools.coco import COCO
    coco = COCO(U.COCO_ANN_FILE)
    cids = coco.getCatIds(catNms=[cls_name])
    if not cids:
        return [], []
    cat_id = cids[0]
    iids = coco.getImgIds(catIds=[cat_id])
    rng = np.random.default_rng(seed)
    iids = rng.permutation(iids)
    images, gts = [], []
    for iid in iids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[iid], catIds=[cat_id], iscrowd=False))
        if not anns:
            continue
        info = coco.loadImgs([iid])[0]
        p = os.path.join(U.COCO_IMG_DIR, info['file_name'])
        if not os.path.exists(p):
            continue
        try:
            im = Image.open(p).convert('RGB')
        except Exception:
            continue
        g = [[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
             for a in anns]
        images.append(im)
        gts.append(g)
        if len(images) >= n:
            break
    return images, gts


def encode_crop(crop):
    dino, pre = U.get_dinov2()
    t = pre(crop).unsqueeze(0).to(U.DEVICE)
    with torch.no_grad():
        o = dino(t)
    f = o.hidden_states[LAYER + 1][0, 0, :]
    return f / f.norm()


def get_neg_proto(bank, dior_cls):
    pos = bank[dior_cls][LAYER]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][LAYER]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return neg / neg.norm()


def build_cache(name, images, gts, prompt):
    """检测+编码，落盘 npz（逐图）。已有缓存跳过。返回 npz 路径。"""
    cache = CACHE_DIR / f'{name}.npz'
    if cache.exists():
        print(f'  [cache hit] {name}')
        return cache

    dets = U.gdino_detect(images, prompt, 0.0)
    cand_boxes, cand_gd, cand_feats = [], [], []
    gt_boxes, gt_feats = [], []

    for img, (boxes, scores), gts_img in zip(images, dets, gts):
        # 候选框
        cb, cg, cf = [], [], []
        keep = scores > PRE_FILTER
        for box, sc in zip(boxes[keep], scores[keep]):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            cb.append([x1, y1, x2, y2])
            cg.append(float(sc))
            cf.append(encode_crop(img.crop((x1, y1, x2, y2))).cpu().numpy())
        # GT 框（校准原型用）
        gb, gf = [], []
        for g in gts_img:
            x1, y1, x2, y2 = [max(0, int(v)) for v in g]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            gb.append([x1, y1, x2, y2])
            gf.append(encode_crop(img.crop((x1, y1, x2, y2))).cpu().numpy())
        cand_boxes.append(np.array(cb, dtype=np.float32) if cb else np.empty((0, 4), np.float32))
        cand_gd.append(np.array(cg, dtype=np.float32) if cg else np.empty(0, np.float32))
        cand_feats.append(np.array(cf, dtype=np.float32) if cf else np.empty((0, 768), np.float32))
        gt_boxes.append(np.array(gb, dtype=np.float32) if gb else np.empty((0, 4), np.float32))
        gt_feats.append(np.array(gf, dtype=np.float32) if gf else np.empty((0, 768), np.float32))

    np.savez_compressed(cache,
                        cand_boxes=np.array(cand_boxes, dtype=object),
                        cand_gd=np.array(cand_gd, dtype=object),
                        cand_feats=np.array(cand_feats, dtype=object),
                        gt_boxes=np.array(gt_boxes, dtype=object),
                        gt_feats=np.array(gt_feats, dtype=object))
    print(f'  [cached] {name}  ({len(images)} 图)')
    return cache


def load_cache(name):
    d = np.load(CACHE_DIR / f'{name}.npz', allow_pickle=True)
    return (list(d['cand_boxes']), list(d['cand_gd']), list(d['cand_feats']),
            list(d['gt_boxes']), list(d['gt_feats']))


def ap_on(test_idx, cand_boxes, cand_gd, cand_feats, gt_boxes, P_cal, neg, use_vis=True):
    """在 test_idx 图上评估数据集级 AP50。P_cal=None → 纯 gd（baseline）。"""
    all_b, all_s, all_g = [], [], []
    for i in test_idx:
        boxes = cand_boxes[i]
        if len(boxes) == 0:
            all_g.extend(gt_boxes[i].tolist())
            continue
        gd = np.clip(cand_gd[i], 1e-6, 1.0)
        if use_vis and len(cand_feats[i]):
            F = cand_feats[i]
            vis = F @ P_cal.cpu().numpy() - NEG_LAM * (F @ neg.cpu().numpy())
            vis = np.clip(vis, 1e-6, 1.0)
            score = np.sqrt(gd * vis)
        else:
            score = gd
        all_b.extend(boxes.tolist())
        all_s.extend(score.tolist())
        all_g.extend(gt_boxes[i].tolist())
    return U.compute_ap50(all_b, all_s, all_g)


def eval_class(name, src_cls, tgt_cls, bank):
    P_src = bank[src_cls][LAYER]['prototype'].to(U.DEVICE)
    neg = get_neg_proto(bank, src_cls)

    n = min(TEST_POOL, 100000)  # load_coco_random 内部按实际图数截断
    images, gts = load_coco_random(tgt_cls, n, seed=SEED + hash(name) % 1000)
    if len(images) < 30:
        print(f'  !! {name} 图数不足 {len(images)}，跳过')
        return None

    cache = build_cache(name, images, gts, f'a photo of {tgt_cls}')
    cand_boxes, cand_gd, cand_feats, gt_boxes, gt_feats = load_cache(name)

    rng = np.random.default_rng(SEED)
    n_img = len(images)

    row = {'group': None, 'n_images': n_img}
    # K=0 零样本（测试=全池，与划分无关）
    test_all = list(range(n_img))
    row['baseline'] = ap_on(test_all, cand_boxes, cand_gd, cand_feats, gt_boxes, None, None, use_vis=False)
    row['zero_shot'] = ap_on(test_all, cand_boxes, cand_gd, cand_feats, gt_boxes, P_src, neg)

    row['splits'] = {}
    for s in range(N_SPLITS):
        perm = rng.permutation(n_img).tolist()
        row['splits'][str(s)] = {}
        for K in K_LIST:
            if K == 0:
                continue  # 已在 zero_shot
            fs_idx = perm[:K]
            test_idx = perm[K:]
            # 校准原型：校准集 GT 框特征平均
            # 注意：GT 框经 build_cache 的 10px 过滤后可能为空（小目标类如 sports ball），
            # 校准集恰好无有效框时回退到纯源原型（等价 K=0 零样本，但测试集排除校准图）
            gtf_list = [gt_feats[i] for i in fs_idx if len(gt_feats[i])]
            gtf = np.concatenate(gtf_list, axis=0) if gtf_list else np.empty((0, 768), np.float32)
            if len(gtf) == 0:
                P_tgt = None
            else:
                P_tgt = torch.tensor(gtf.mean(axis=0), device=U.DEVICE)
                P_tgt = P_tgt / P_tgt.norm()
            # baseline / 源原型对照 在同一测试集
            row['splits'][str(s)][f'K{K}_baseline'] = ap_on(test_idx, cand_boxes, cand_gd, cand_feats, gt_boxes, None, None, use_vis=False)
            row['splits'][str(s)][f'K{K}_alpha1'] = ap_on(test_idx, cand_boxes, cand_gd, cand_feats, gt_boxes, P_src, neg)
            if P_tgt is None:
                for a in ALPHA_LIST:
                    row['splits'][str(s)][f'K{K}_a{a}'] = ap_on(test_idx, cand_boxes, cand_gd, cand_feats, gt_boxes, P_src, neg)
                continue
            for a in ALPHA_LIST:
                P_cal = (a * P_src + (1 - a) * P_tgt)
                P_cal = P_cal / P_cal.norm()
                row['splits'][str(s)][f'K{K}_a{a}'] = ap_on(test_idx, cand_boxes, cand_gd, cand_feats, gt_boxes, P_cal, neg)
    return row


def main():
    bank = U.get_bank()
    print(f'DEVICE = {U.DEVICE}')
    results = {
        'config': {
            'layer': 'L12', 'neg_lam': NEG_LAM, 'pre_filter': PRE_FILTER,
            'k_list': K_LIST, 'alpha_list': ALPHA_LIST, 'n_splits': N_SPLITS,
            'test_pool': TEST_POOL, 'seed': SEED,
            'identity': ('six target-domain categories derived from three source '
                         'semantic concepts, four sharing the vehicle prototype; '
                         'neg_controls answer scene-association-only question, appendix only'),
        },
        'main': {}, 'neg_controls': {},
    }
    for name, src, tgt, group in MAIN_EVAL:
        print(f'\n[MAIN] {name}  ({src} → {tgt})  [{group}]')
        r = eval_class(name, src, tgt, bank)
        if r is None:
            continue
        r['group'] = group
        r['src_concept'] = src
        results['main'][name] = r
    for name, src, tgt in NEG_CONTROL:
        print(f'\n[NEG]  {name}  ({src} → {tgt})')
        r = eval_class(name, src, tgt, bank)
        if r is None:
            continue
        r['group'] = 'neg_control'
        r['src_concept'] = src
        results['neg_controls'][name] = r

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n✅ 结果 → {OUT_JSON}')


if __name__ == '__main__':
    main()
