# -*- coding: utf-8 -*-
"""probe_delta_structure.py — δ 的结构探针 (纯文本编码器, 无需检测)

核心问题: A-CSPC 的 δ_c = t_photo(c) - t_overhead(c) 究竟是
  (H1) 一个"域搬运方向"(应当依赖类别, 且把原型推向目标域), 还是
  (H6) 一个"类别无关的全局偏移"(所有类的 δ_c 几乎共线)?

若 H6 成立, 则:
  - 反向任务 (+δ 把 COCO 原型继续推向 photo) 也涨分就不再是悖论;
  - A-CSPC 的增益来源应是"正原型平移 + 负原型不平移"这一非对称,
    而非域对齐;
  - δ 可由一个全局常数向量替代 → 方法不再需要任何目标域知识。

输出: 类别间 pairwise cosine 矩阵统计量 / 全局成分占比 / 残差占比 /
      不同模板对的对比 / 与随机方向的对照。
"""
import os, sys, json, itertools
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import numpy as np
import torch

OUT = r'C:\Users\张三\Desktop\实验数据\probe_delta_structure.json'

CLASSES = ['airplane', 'ship', 'vehicle', 'car', 'truck', 'bus', 'motorcycle', 'boat',
           'bridge', 'harbor', 'dam', 'storage tank', 'tennis court', 'baseball field',
           'basketball court', 'ground track field', 'stadium', 'train station',
           'airport', 'overpass', 'windmill', 'chimney', 'golf course', 'expressway-service area',
           'expressway-toll-station', 'swimming pool']

TEMPLATES = {
    'photo':     'a photo of {}',
    'overhead':  'an overhead image of {}',
    'satellite': 'a satellite image of {}',
    'aerial':    'an aerial photograph of {}',
    'painting':  'a painting of {}',
    'sketch':    'a sketch of {}',
    'plain':     '{}',
    'natural':   'a natural image of {}',
    'remote':    'a remote sensing image of {}',
}


def load_clip():
    from transformers import CLIPModel, CLIPTokenizer
    name = 'openai/clip-vit-base-patch32'
    m = CLIPModel.from_pretrained(name)
    tok = CLIPTokenizer.from_pretrained(name)
    m.eval()
    return m, tok


@torch.no_grad()
def text_emb(m, tok, texts, bs=64):
    outs = []
    for i in range(0, len(texts), bs):
        b = texts[i:i + bs]
        enc = tok(b, padding=True, truncation=True, return_tensors='pt')
        e = m.get_text_features(**enc)
        e = torch.nn.functional.normalize(e, dim=-1)
        outs.append(e.numpy())
    return np.concatenate(outs, 0)


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def analyze_delta(D, tag):
    """D: (C, d) 每类一个 delta(未归一化或已归一化均可, 内部只看方向)."""
    U = l2(D)
    C = U.shape[0]
    cosine = U @ U.T
    iu = np.triu_indices(C, 1)
    off = cosine[iu]
    mean_dir = l2(U.mean(0))
    align_to_mean = (U @ mean_dir)
    # 分解: 每类 delta = 全局成分 + 残差
    proj = (D @ mean_dir)[:, None] * mean_dir[None, :]
    resid = D - proj
    gfrac = np.linalg.norm(proj, axis=1) / (np.linalg.norm(D, axis=1) + 1e-12)
    rfrac = np.linalg.norm(resid, axis=1) / (np.linalg.norm(D, axis=1) + 1e-12)
    return dict(
        tag=tag, n_class=int(C),
        pairwise_cos_mean=float(off.mean()),
        pairwise_cos_min=float(off.min()),
        pairwise_cos_max=float(off.max()),
        pairwise_cos_std=float(off.std()),
        align_to_global_mean=float(align_to_mean.mean()),
        align_to_global_min=float(align_to_mean.min()),
        global_component_frac=float(gfrac.mean()),
        residual_frac=float(rfrac.mean()),
        delta_norm_mean=float(np.linalg.norm(D, axis=1).mean()),
        delta_norm_std=float(np.linalg.norm(D, axis=1).std()),
    )


def main():
    m, tok = load_clip()
    res = {'classes': CLASSES, 'templates': TEMPLATES}

    # ---- 1. 所有模板下各类别的文本 embedding ----
    emb = {}
    for name, tpl in TEMPLATES.items():
        emb[name] = text_emb(m, tok, [tpl.format(c) for c in CLASSES])

    # ---- 2. 模板空间的"中心性":每个模板到全类别质心的平均余弦 ----
    allcat = np.concatenate([emb[k] for k in TEMPLATES], 0)
    centroid = l2(allcat.mean(0))[None, :]
    centrality = {}
    for name in TEMPLATES:
        centrality[name] = float((emb[name] @ centroid.T).mean())
    res['template_centrality'] = centrality

    # ---- 3. 各类模板对的 delta 结构 ----
    deltas = {}
    pairs = [('photo', 'overhead'), ('overhead', 'photo'),
             ('natural', 'remote'), ('photo', 'satellite'),
             ('photo', 'aerial'), ('painting', 'sketch'),
             ('photo', 'plain'), ('remote', 'photo')]
    for a, b in pairs:
        D = emb[a] - emb[b]
        tag = f'{a} - {b}'
        deltas[tag] = analyze_delta(D, tag)
    res['delta_structure'] = deltas

    # ---- 4. 全局方向之间的对齐 (不同模板对得到的"全局方向"是不是同一个?) ----
    glob = {}
    for tag in deltas:
        a, b = tag.split(' - ')
        D = emb[a] - emb[b]
        glob[tag] = l2(D.mean(0))
    ks = list(glob)
    M = np.stack([glob[k] for k in ks])
    cos = M @ M.T
    res['global_dir_pairwise'] = {f'{ks[i]}|{ks[j]}': float(cos[i, j])
                                  for i, j in itertools.combinations(range(len(ks)), 2)}

    # ---- 5. 随机方向对照: 同范数随机向量的 pairwise cosine ----
    rng = np.random.default_rng(0)
    d = emb['photo'].shape[1]
    R = rng.normal(size=(len(CLASSES), d))
    R = l2(R)
    cr = R @ R.T
    iu = np.triu_indices(R.shape[0], 1)
    res['random_dir_pairwise_cos_mean'] = float(cr[iu].mean())
    res['random_dir_pairwise_cos_abs_mean'] = float(np.abs(cr[iu]).mean())

    # ---- 6. photo/overhead 两个模板 embedding 本身的类间离散度 ----
    for name in ['photo', 'overhead', 'satellite', 'plain']:
        U = emb[name] @ emb[name].T
        iu = np.triu_indices(U.shape[0], 1)
        res.setdefault('template_intra_spread', {})[name] = float(U[iu].mean())

    json.dump(res, open(OUT, 'w'), indent=2, ensure_ascii=False)

    print('=' * 78)
    print('δ 结构探针结果')
    print('=' * 78)
    print('\n[1] 模板中心性 (到全体类别质心的平均余弦, 越大越"规范"):')
    for k, v in sorted(centrality.items(), key=lambda x: -x[1]):
        print(f'    {k:<10s} {v:+.4f}')
    print('\n[2] delta 结构 (pairwise cosine 越高 → δ 越类别无关):')
    print(f'    {"模板对":<26s} {"pwcos":>8s} {"alignμ":>8s} {"全局占比":>9s} {"残差":>7s} {"||δ||":>7s}')
    for tag, d_ in deltas.items():
        print(f'    {tag:<26s} {d_["pairwise_cos_mean"]:+.4f} '
              f'{d_["align_to_global_mean"]:+.4f} {d_["global_component_frac"]:>9.3f} '
              f'{d_["residual_frac"]:>7.3f} {d_["delta_norm_mean"]:>7.4f}')
    print(f'\n[3] 随机方向对照 pairwise cosine = {res["random_dir_pairwise_cos_mean"]:+.4f} '
          f'(|·| = {res["random_dir_pairwise_cos_abs_mean"]:.4f})')
    print('\n[4] 模板自身类间离散度 (平均余弦, 越低说明类别区分越好):')
    for k, v in res['template_intra_spread'].items():
        print(f'    {k:<10s} {v:+.4f}')
    print('\n[5] 不同模板对得到的"全局方向"两两余弦:')
    for k, v in res['global_dir_pairwise'].items():
        print(f'    {k:<40s} {v:+.4f}')
    print(f'\n已保存: {OUT}')


if __name__ == '__main__':
    main()
