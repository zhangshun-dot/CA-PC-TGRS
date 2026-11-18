# -*- coding: utf-8 -*-
r"""tge_v2_margin_probe.py — 想法 A 的证伪测试（2 分钟级）

背景（我自己的发现，见 AutoProCorrector_γ路线双重否定_2026-09-18.md）：
  USC 的校正是  Z + γ(Z·v)v + αv = Z + s·v，**只剩一个标量 s**。
  而上一版用的"免标注内目标" cos(USC(s), p_tgt) 关于 s **单调**（因为 v 指向目标，
  越往 +v 走越像目标）⇒ argmax 永远在网格端点 ⇒ 代理没有排序能力。

想法 A：换内目标——把"像不像目标原型"换成"**选出来的框像不像一个类簇**"。
  p̂(s) = unit(p_src + s·v)
  z_i  = p̂(s)·f_i                      （f_i = 目标域检测器框特征，零标注）
  取 top-k 框，算它们的**平均特征模长** ||mean(f_topk)|| —— 这就是"类簇紧致度"
  （单元特征下，随机 k 个框的期望 ≈ 1/√k；越紧致越大）。
  quality(s) = ||mean(f_topk)|| − 1/√k

  为什么可能有内部最优：
    s 太小（≈0）→ p̂ 还是源域原型，可能根本没选中目标类 ⇒ 簇松散
    s 太大       → p̂ → ±v，退化成"协议方向"，在噪声/尺寸等 nuisance 上排序 ⇒ 簇也松散
    中间应有峰。**如果 quality(s) 也在端点取到最大，想法 A 当场死掉。**

本脚本只做一件事：**检查 quality(s) 有没有内部最优**（证伪测试），不做 AP 验证。
运行: C:\python-study\python39\python.exe tge_v2_margin_probe.py
"""
import sys, os, json, pickle, time
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np

from tge_v2_lambda_4sources import (
    build_dior_protos, build_nwpu_protos, build_dota_protos, build_visdrone_protos,
    SRC_MAP, FEAT_CACHE,
)

DATA = os.path.dirname(FEAT_CACHE)
OUT = os.path.join(DATA, 'tge_v2_margin_probe.json')

S_GRID = np.arange(-1.5, 1.5001, 0.05)      # 直接扫标量 s（不是 γ）
TOP_FRAC = 0.10


def unit(x):
    n = np.linalg.norm(x)
    return x / (n + 1e-12)


def coherence(F, scores, k):
    idx = np.argsort(-scores)[:k]
    m = F[idx].mean(axis=0)
    return float(np.linalg.norm(m))


def main():
    t0 = time.time()
    cache = pickle.load(open(FEAT_CACHE, 'rb'))
    tgt_proto, boxF = {}, {}
    for eval_name, blob in cache.items():
        F = blob['F'].astype(np.float64)
        F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
        boxF[eval_name] = F
        tgt_proto[eval_name] = unit(F.mean(axis=0))
    print('目标侧：%d 类，框数 %s' % (len(boxF), {k: v.shape[0] for k, v in boxF.items()}))

    src_protos = {}
    for name, fn in [('DIOR', build_dior_protos), ('NWPU', build_nwpu_protos),
                     ('DOTA', build_dota_protos), ('VisDrone', build_visdrone_protos)]:
        src_protos[name] = fn()
    print('源域原型就绪')

    cells = []
    for src in ['DIOR', 'NWPU', 'DOTA', 'VisDrone']:
        for coco_cls, src_cls in SRC_MAP[src].items():
            if src_cls is None or src_cls not in src_protos[src] or coco_cls not in boxF:
                continue
            _p = src_protos[src][src_cls]
            if hasattr(_p, 'detach'):
                _p = _p.detach().cpu().numpy()
            cells.append(dict(src=src, coco=coco_cls, src_cls=src_cls,
                              p_src=unit(np.asarray(_p, dtype=np.float64)),
                              F=boxF[coco_cls], p_tgt=tgt_proto[coco_cls]))
    print('有效格数 %d\n' % len(cells))

    for src in ['DIOR', 'NWPU', 'DOTA', 'VisDrone']:
        sub = [c for c in cells if c['src'] == src]
        if sub:
            v = unit(np.mean([c['p_tgt'] - c['p_src'] for c in sub], axis=0))
            for c in sub:
                c['v'] = v

    print('%-22s %8s %9s %10s %10s %10s' %
          ('格', 'k_top', 'argmax_s', '内部最优?', '曲线跨度', '端点值'))
    out_cells, n_interior = [], 0
    for c in cells:
        F, ps, v = c['F'], c['p_src'], c['v']
        N = F.shape[0]
        k = max(8, int(N * TOP_FRAC))
        null = 1.0 / np.sqrt(k)
        qs = []
        for s in S_GRID:
            ph = unit(ps + s * v)
            sc = F @ ph
            qs.append(coherence(F, sc, k) - null)
        qs = np.array(qs)
        am = int(np.argmax(qs))
        s_star = float(S_GRID[am])
        interior = 0 < am < len(S_GRID) - 1
        n_interior += interior
        print('%-22s %8d %9.2f %10s %10.4f %10.4f' %
              ('%s-%s' % (c['src'], c['coco'].replace('COCO-', '')), k,
               s_star, '是' if interior else '否(端点)', qs.max() - qs.min(), qs[am]))
        out_cells.append(dict(src=c['src'], coco=c['coco'], k_top=k,
                              s_star=s_star, interior=bool(interior),
                              curve_span=float(qs.max() - qs.min()),
                              f4_proj_v=float(ps @ v),
                              curve={float(s): float(q) for s, q in zip(S_GRID, qs)}))

    print('\n内部最优的格数: %d / %d' % (n_interior, len(cells)))
    if n_interior == 0:
        print('⇒ 想法 A 的证伪测试：**未通过**（与 cos 代理一样落在端点，仍然是退化目标）')
    else:
        print('⇒ 想法 A **通过证伪测试**：目标函数存在内部最优，值得接 AP 验证')
    print('s 的扫描范围 [%.2f, %.2f]；note: 端点命中说明真峰可能更远，需扩范围复核'
          % (S_GRID[0], S_GRID[-1]))

    json.dump(dict(s_grid=[float(x) for x in S_GRID], n_interior=n_interior,
                   n_cells=len(cells), cells=out_cells),
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('\n→ %s  (%.1f 分钟)' % (OUT, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
