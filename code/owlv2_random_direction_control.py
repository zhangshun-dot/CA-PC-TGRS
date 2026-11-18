# -*- coding: utf-8 -*-
"""owlv2_random_direction_control.py — OWLv2上的随机方向对照实验
预登记单一操作点s=1.46（α=2.5），m=49个随机方向。
重点验证car和motorcycle在OWLv2上是不是仍然排1/50。
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import unified_eval as U
import tge_probe as T
from sota_comparison import (
    SOURCE_LOADERS, build_proto_tge, evaluate_with_features,
    SRC_TMPL_OVERHEAD, TGT_TMPL, PRE_FILTER,
)

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# OWLv2模型路径
OWLV2_CACHE = r'C:\Users\张三\.cache\huggingface\hub\models--google--owlv2-base-patch16-ensemble\snapshots'

# 主分析7类
MAIN_CLASSES = ['airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'truck']

# 预登记操作点：s=1.46（α=2.5, ||d||/||p||=0.5834）
S_TARGET = 1.4585

# 随机方向采样数（m=49，p_c下限0.02）
N_RANDOM = 49

# 随机种子（保证可复现）
RANDOM_SEED = 42


def normalize(v):
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8) if norm > 1e-8 else v


def get_owlv2():
    """懒加载OWLv2模型"""
    if 'owl' not in globals():
        snapshots = sorted(Path(OWLV2_CACHE).glob('*'))
        if snapshots:
            cache_path = str(snapshots[-1])
        else:
            cache_path = 'google/owlv2-base-patch16-ensemble'
        print(f'加载OWLv2模型: {cache_path}')
        globals()['owl_proc'] = AutoProcessor.from_pretrained(cache_path, local_files_only=True)
        globals()['owl_model'] = (AutoModelForZeroShotObjectDetection
                                   .from_pretrained(cache_path, dtype=torch.float32, local_files_only=True)
                                   .eval().to(U.DEVICE))
    return globals()['owl_proc'], globals()['owl_model']


def owlv2_detect(images, prompt, thresh=0.0):
    """用OWLv2检测一批图像，返回每图(boxes, scores)"""
    owl_p, owl_m = get_owlv2()
    out = []
    for img in images:
        if len(prompt) > 60:
            prompt = prompt[:60]
        inputs = owl_p(text=prompt, images=img, return_tensors='pt')
        inputs = {k: v.to(U.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = owl_m(**inputs)
        target_sizes = torch.tensor([img.size[::-1]]).to(U.DEVICE)
        r = owl_p.post_process_object_detection(outputs, threshold=thresh, target_sizes=target_sizes)[0]
        out.append((r['boxes'].cpu().numpy(), r['scores'].cpu().numpy()))
    return out


def load_target_coco_owlv2(coco_cls, n=200):
    """用OWLv2检测COCO目标域，返回per_img和features"""
    images, gts = U.load_coco(coco_cls, n=n)
    if not images:
        return None
    cache_p = DATA / f'owlv2_dets_COCO-{coco_cls}.pkl'
    if cache_p.exists():
        dets = pickle.load(open(cache_p, 'rb'))
    else:
        dets = owlv2_detect(images, f'a satellite image of {coco_cls}', 0.0)
        pickle.dump(dets, open(cache_p, 'wb'))
    per_img = []
    crops_all = []
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < PRE_FILTER:
                continue
            bl.append([x1, y1, x2, y2])
            sl.append(float(sc))
            ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({
            'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
            'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
            'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
            'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64),
        })
    F = T.encode_clip_batch(crops_all).cpu().numpy() if crops_all else np.zeros((0, 512))
    return {'per_img': per_img, 'features': F, 'n_crops': len(crops_all)}


def get_tge_direction(src_cls, target_cls):
    """获取TGE的真实方向向量"""
    s_emb = T.text_avg([SRC_TMPL_OVERHEAD], src_cls)
    d_emb = T.text_avg([TGT_TMPL], target_cls)
    if hasattr(s_emb, 'cpu'):
        s_emb = s_emb.cpu().numpy()
    if hasattr(d_emb, 'cpu'):
        d_emb = d_emb.cpu().numpy()
    d = d_emb - s_emb
    return normalize(d)


def evaluate_with_direction(target_data, src_proto, d_direction, displacement, neg_proto):
    """用指定方向和位移评估AP50"""
    proto = normalize(src_proto + displacement * d_direction)
    return evaluate_with_features(target_data, proto, neg_proto=neg_proto)


def run_experiment(src_name, src_cls, target_cls):
    """跑单个类别的随机方向对照实验"""
    print(f'\n{"─" * 80}')
    print(f'{src_name}/{src_cls} → COCO/{target_cls} (OWLv2检测器, s={S_TARGET:.4f})')
    print(f'{"─" * 80}')

    # 加载源域数据
    print(f'  加载源域数据...')
    src_loader = SOURCE_LOADERS[src_name][0]
    src_data = src_loader(src_cls, n=150)
    if src_data is None:
        print(f'  源域数据加载失败，跳过')
        return None
    src_proto = src_data['prototype']
    src_features = src_data['features']
    print(f'  源域原型: ||p||={np.linalg.norm(src_proto):.4f}, n_crops={src_data["n_crops"]}')

    # 负原型
    sims = src_features @ src_proto
    neg_idx = np.argmin(sims)
    proto_neg = src_features[neg_idx]
    proto_neg = normalize(proto_neg)

    # 加载COCO目标域数据（OWLv2检测）
    print(f'  加载COCO目标域数据（OWLv2检测）...')
    target_data = load_target_coco_owlv2(target_cls, n=200)
    if target_data is None:
        print(f'  目标域数据加载失败，跳过')
        return None
    print(f'  目标域: n_imgs={len(target_data["per_img"])}, n_crops={target_data["n_crops"]}')

    # baseline (displacement=0)
    baseline_ap = evaluate_with_features(target_data, src_proto, neg_proto=proto_neg)
    print(f'  baseline AP50: {baseline_ap:.4f}')

    # 真实方向
    real_d = get_tge_direction(src_cls, target_cls)
    real_ap = evaluate_with_direction(target_data, src_proto, real_d, S_TARGET, proto_neg)
    real_gain = real_ap - baseline_ap
    print(f'  真实方向: AP={real_ap:.4f}, 增益={real_gain:+.4f}')

    # 随机方向（N_RANDOM个）
    print(f'  随机方向（{N_RANDOM}个）...')
    rng = np.random.RandomState(RANDOM_SEED)
    random_gains = []
    for i in range(N_RANDOM):
        random_d = rng.randn(len(src_proto))
        random_d = normalize(random_d)
        random_ap = evaluate_with_direction(target_data, src_proto, random_d, S_TARGET, proto_neg)
        random_gain = random_ap - baseline_ap
        random_gains.append(random_gain)
        if (i + 1) % 10 == 0:
            print(f'    已完成 {i+1}/{N_RANDOM}')

    random_mean = np.mean(random_gains)
    random_std = np.std(random_gains)

    # 排名：真实方向在N_RANDOM个随机方向中的排名（1=最好）
    rank = 1 + sum(1 for rg in random_gains if rg > real_gain)
    p_c = rank / (N_RANDOM + 1)

    print(f'  随机方向: 均值={random_mean:+.4f}, 标准差={random_std:.4f}')
    print(f'  排名: {rank}/{N_RANDOM+1}, p_c={p_c:.4f}')

    # 反向方向
    reverse_d = -real_d
    reverse_ap = evaluate_with_direction(target_data, src_proto, reverse_d, S_TARGET, proto_neg)
    reverse_gain = reverse_ap - baseline_ap
    print(f'  反向方向: AP={reverse_ap:.4f}, 增益={reverse_gain:+.4f}')

    result = {
        'src': src_name,
        'src_cls': src_cls,
        'target_cls': target_cls,
        'target_domain': 'COCO',
        'detector': 'OWLv2',
        's_target': S_TARGET,
        'n_random': N_RANDOM,
        'baseline_ap': float(baseline_ap),
        'real_ap': float(real_ap),
        'real_gain': float(real_gain),
        'random_mean': float(random_mean),
        'random_std': float(random_std),
        'random_gains': [float(g) for g in random_gains],
        'rank': int(rank),
        'p_c': float(p_c),
        'reverse_ap': float(reverse_ap),
        'reverse_gain': float(reverse_gain),
    }

    return result


def main():
    print('=' * 80)
    print('OWLv2上的随机方向对照实验')
    print(f'预登记操作点 s={S_TARGET:.4f}（α=2.5）, m={N_RANDOM}个随机方向')
    print('=' * 80)

    # 实验配置：DIOR源域 → COCO目标域，7个类
    experiments = [
        ('DIOR', 'airplane', 'airplane'),
        ('DIOR', 'ship', 'boat'),
        ('DIOR', 'vehicle', 'car'),
        ('DIOR', 'vehicle', 'bus'),
        ('DIOR', 'vehicle', 'motorcycle'),
        ('DIOR', 'vehicle', 'truck'),
        ('VisDrone', 'bicycle', 'bicycle'),
    ]

    print(f'\n实验配置: {len(experiments)}个类别')
    for src, src_cls, tgt_cls in experiments:
        print(f'  {src}/{src_cls} → COCO/{tgt_cls}')

    # 跑每个类别
    all_results = []
    for src, src_cls, tgt_cls in experiments:
        result = run_experiment(src, src_cls, tgt_cls)
        if result is not None:
            all_results.append(result)

    # 汇总
    print(f'\n{"=" * 80}')
    print('汇总结果（OWLv2检测器, s={:.4f}）'.format(S_TARGET))
    print(f'{"=" * 80}')

    print(f'\n{"类别":<12} {"baseline":>10} {"real_gain":>10} {"random_mean":>12} '
          f'{"random_std":>10} {"rank":>8} {"p_c":>8} {"reverse":>10}')
    print('-' * 90)

    for r in all_results:
        print(f'{r["target_cls"]:<12} {r["baseline_ap"]:>10.4f} {r["real_gain"]:>+10.4f} '
              f'{r["random_mean"]:>+12.4f} {r["random_std"]:>10.4f} '
              f'{r["rank"]:>4}/{r["n_random"]+1} {r["p_c"]:>8.4f} {r["reverse_gain"]:>+10.4f}')

    # Fisher合并检验
    print(f'\nFisher合并检验:')
    chi2 = -2 * sum(np.log(r['p_c'] + 1e-10) for r in all_results)
    df = 2 * len(all_results)
    from scipy.stats import chi2 as chi2_dist
    fisher_p = 1 - chi2_dist.cdf(chi2, df)
    print(f'  χ²({df}) = {chi2:.4f}, p = {fisher_p:.6f}')

    # car和motorcycle的特别验证
    print(f'\ncar和motorcycle的特别验证:')
    for cls in ['car', 'motorcycle']:
        r = next((r for r in all_results if r['target_cls'] == cls), None)
        if r:
            print(f'  {cls}: rank={r["rank"]}/{r["n_random"]+1}, p_c={r["p_c"]:.4f}')
            if r['rank'] == 1:
                print(f'    ✅ 真实方向优于全部{r["n_random"]}个随机方向')
            else:
                print(f'    ⚠️ 真实方向排名第{r["rank"]}，不是第1')

    # 保存结果
    out_file = DATA / 'owlv2_random_direction_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {out_file}')


if __name__ == '__main__':
    main()
