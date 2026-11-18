# -*- coding: utf-8 -*-
"""
聚合 CA-PC 三个实验产物的方向性结论，验证两条主张：
  (1) A(锚定) > B(角色交换) 的方向优势在 DIOR 之外也成立（反向扩到 DOTA/NWPU）
  (2) 方向排序（A>B）在强检测器 OWLv2 上仍存在，即便绝对增益归零
所有数字直接来自 实验数据/*.json，不手填。
"""
import json, os, numpy as np
from collections import defaultdict

DATA = r'C:\Users\张三\Desktop\实验数据'

def load(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding='utf-8'))

def norm_unit(v):
    """把不同脚本的单元 dict 统一成 {A_anchored, B_roleswap, beta0, rand_mean}"""
    if 'A_anchored' in v:
        return v
    if 'plus_delta' in v:  # run_bidirectional.json (GDINO 主双向): 存的是绝对 AP，转成 delta
        base = v['baseline']
        rk = [v[k] - base for k in ('rand0', 'rand1', 'rand2') if k in v]
        return {'A_anchored': v['plus_delta'] - base, 'B_roleswap': v['minus_delta'] - base,
                'beta0': v.get('beta0', base) - base,
                'rand_mean': float(np.mean(rk)) if rk else 0.0}
    return v

def stats(rows):
    """rows: list of dict，含 A_anchored/B_roleswap/beta0/rand_mean"""
    if not rows:
        return None
    rows = [norm_unit(r) for r in rows]
    A = np.array([r['A_anchored'] for r in rows])
    B = np.array([r['B_roleswap'] for r in rows])
    b0 = np.array([r['beta0'] for r in rows])
    rd = np.array([r['rand_mean'] for r in rows])
    return {
        'n': len(rows),
        'A_mean': float(A.mean()), 'B_mean': float(B.mean()),
        'A_min': float(A.min()), 'A_max': float(A.max()),
        'B_min': float(B.min()), 'B_max': float(B.max()),
        'A_gt_B': int((A > B).sum()),
        'A_ge_B': int((A >= B).sum()),
        'A_minus_B_mean': float((A - B).mean()),
        'beta0_mean': float(b0.mean()),
        'rand_mean': float(rd.mean()),
        'A_beats_beta0': int((A > b0).sum()),
        'A_beats_rand': int((A > rd).sum()),
    }

def fmt(s):
    if s is None:
        return '  (无数据)'
    return (f"  n={s['n']}  A均={s['A_mean']:+.4f}  B均={s['B_mean']:+.4f}  "
            f"A−B均={s['A_minus_B_mean']:+.4f}\n"
            f"  A>B: {s['A_gt_B']}/{s['n']}  β=0均={s['beta0_mean']:+.4f}  "
            f"随机均={s['rand_mean']:+.4f}")

report = {'claims': {}, 'detail': {}}

# ---------- 1) GDINO 主双向 (run_bidirectional.json) ----------
gd = load('run_bidirectional.json')
if gd:
    out = {}
    for tname, units in gd['results'].items():
        out[tname] = stats(list(units.values()))
    report['detail']['gdino_bidirectional'] = out
    print('=== GDINO 主双向 (run_bidirectional.json) ===')
    for t, s in out.items():
        print(f'[{t}]')
        print(fmt(s))

# ---------- 2) 反向扩类 (run_reverse_multirs.json, 扁平 units) ----------
mr = load('run_reverse_multirs.json')
if mr:
    flat = mr['results']
    by_ds = defaultdict(list)
    for key, v in flat.items():
        ds = key.split('_')[0]
        by_ds[ds].append(v)
    out = {ds: stats(rows) for ds, rows in by_ds.items()}
    out['ALL'] = stats(list(flat.values()))
    report['detail']['multirs_by_dataset'] = out
    print('\n=== 反向扩类 (run_reverse_multirs.json) ===')
    for ds, s in out.items():
        print(f'[{ds}]')
        print(fmt(s))

# ---------- 3) OWLv2 双向 (run_bidirectional_owlv2.json, 嵌套) ----------
owl = load('run_bidirectional_owlv2.json')
if owl:
    out = {}
    for tname, units in owl['results'].items():
        out[tname] = stats(list(units.values()))
    report['detail']['owlv2_bidirectional'] = out
    print('\n=== OWLv2 双向 (run_bidirectional_owlv2.json) ===')
    for t, s in out.items():
        print(f'[{t}]')
        print(fmt(s))

# ---------- 汇总判定 ----------
print('\n' + '=' * 70)
print('方向优势主张判定')

def agg_units(*stats_list):
    """把多个 stats 的单元数累加，算总体 A>B 比例与 A/b0/rand 均值"""
    n = A = B = b0 = rd = 0
    agt = age = ab0 = ard = 0
    for s in stats_list:
        if not s:
            continue
        n += s['n']; A += s['A_mean']*s['n']; B += s['B_mean']*s['n']
        b0 += s['beta0_mean']*s['n']; rd += s['rand_mean']*s['n']
        agt += s['A_gt_B']; age += s['A_ge_B']
        ab0 += s['A_beats_beta0']; ard += s['A_beats_rand']
    if n == 0:
        return None
    return {'n': n, 'A_mean': A/n, 'B_mean': B/n, 'beta0_mean': b0/n, 'rand_mean': rd/n,
            'A_gt_B': agt, 'A_ge_B': age, 'A_beats_beta0': ab0, 'A_beats_rand': ard}

gdino_s = list(report['detail'].get('gdino_bidirectional', {}).values())
mr_s = list(report['detail'].get('multirs_by_dataset', {}).values())
owl_s = list(report['detail'].get('owlv2_bidirectional', {}).values())

# 主张1：反向任务里 A>B（GDINO 反向 + MultiRS 全部）
rev_units = []
if gd: rev_units += list(gd['results'].get('rev_coco2dior', {}).values())
if mr: rev_units += list(mr['results'].values())
s1 = stats(rev_units) if rev_units else None
if s1:
    print(f"[主张1] 反向任务 (GDINO反向 + MultiRS全类): n={s1['n']} A>B={s1['A_gt_B']}/{s1['n']} "
          f"A均={s1['A_mean']:+.4f} B均={s1['B_mean']:+.4f} A−B均={s1['A_minus_B_mean']:+.4f}")
    report['claims']['reverse_A_beats_B'] = s1

# 主张2：OWLv2 上 A>B 方向排序仍在（绝对增益可归零）
if owl_s:
    s2 = agg_units(*owl_s)
    print(f"[主张2] OWLv2 双向: n={s2['n']} A>B={s2['A_gt_B']}/{s2['n']} "
          f"A均={s2['A_mean']:+.4f} B均={s2['B_mean']:+.4f}")
    report['claims']['owlv2_direction_holds'] = s2

# 主张0（基准）：前向 GDINO A>B 且 A 明显>β0/随机
if gdino_s:
    s0 = agg_units(*gdino_s)
    print(f"[基准] GDINO 双向全任务: n={s0['n']} A>B={s0['A_gt_B']}/{s0['n']} "
          f"A均={s0['A_mean']:+.4f} β=0均={s0['beta0_mean']:+.4f} 随机均={s0['rand_mean']:+.4f}")
    report['claims']['gdino_overall'] = s0

outp = os.path.join(DATA, 'aggregate_ca_pc.json')
json.dump(report, open(outp, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print(f'\n已保存聚合: {outp}')
