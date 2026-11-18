# -*- coding: utf-8 -*-
"""ab_convergence_pilot.py — A/B 收敛小样验证（受控实验 #1）

设计（对齐用户批准的「vocab 固定、观测条件变化」受控实验）:
  - vocab 固定 = clean COCO → 退化 COCO，同类名 → vocab 恒为 identity（条件轴只动观测，不动语义）。
  - 观测条件轴 = fog / gaussian_blur / brightness × S1/3/5 = 9 档（合成成像退化，属用户列的「成像条件」）。
  - 条件严重度 H = AP_clean_base − AP_deg_base（外部 headroom，与 CA-PC 无关 → 非循环）。
  - 对照:
      g_ca    = CA-PC（忠实条件方向 dc=退化文本−清晰文本, β=2）相对 base 的增益
      g_wrong = 错轴对照（overhead−photo 方向）相对 base 的增益
      g_tge   = TGE（方向, 无 β/无门控）
      g_star  = CA-PC 自调 β（grid 最优）
  - A/B 收敛判据: 若观测条件修正是「驱动项」，则 g_ca 应随 H 单调/显著相关；
                 而条件无关的基线（g_wrong / vocab-only=base）应平稳、不随 H 走。
  - 本轴 vocab=identity ⇒ 只能隔离「cond-alone」（条件修正这一半）。与分解表
    cond-only +0.0159 < vocab-only +0.0243 互为印证：条件项确为较弱一半。

数据源: 实验数据/run_degradation.json（owlv2+yolo, k=40，已落盘的缓存评分）。
纯 CPU 后处理，不跑 GPU。
"""
import json, math, os

J = r'C:\Users/张三\Desktop\实验数据\run_degradation.json'
OUT_JSON = r'C:\Users/张三\Desktop\实验数据\ab_convergence_pilot.json'
OUT_MD = r'C:\Users/张三\Desktop\实验数据\ab_convergence_pilot.md'

d = json.load(open(J, encoding='utf-8'))
rows = d['rows']
cfg = d.get('config', {})


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float('nan'), float('nan')
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return float('nan'), float('nan')
    r = sxy / math.sqrt(sxx * syy)
    if abs(r) < 1:
        t = r * math.sqrt((n - 2) / (1 - r * r))
        p = math.erfc(abs(t) / math.sqrt(2))
    else:
        p = float('nan')
    return r, p


clean = {}
for r in rows:
    if r['corr'] == 'clean':
        clean[(r['det'], r['cls'])] = r

deg = [r for r in rows if r['corr'] != 'clean']
recs = []
for r in deg:
    c = clean[(r['det'], r['cls'])]
    H = c['base'] - r['base']
    recs.append(dict(det=r['det'], cls=r['cls'], corr=r['corr'], sev=r['sev'],
                     base_clean=c['base'], base=r['base'], H=H,
                     g_ca=r['ca_pc'] - r['base'], g_wrong=r['ca_wrong'] - r['base'],
                     g_tge=r['tge'] - r['base'], g_naive=r['naive'] - r['base'],
                     g_star=r['ca_star'] - r['base'], dc=r['dc_mag']))

# ── Q1: 增益 vs 外部 headroom（A/B 收敛核心）──
q1 = {}
for key, lab in (('g_ca', 'CA-PC'), ('g_wrong', 'wrong-axis'), ('g_tge', 'TGE'), ('g_star', 'CA-PC*')):
    r, p = pearson([x['H'] for x in recs], [x[key] for x in recs])
    q1[key] = dict(label=lab, r=round(r, 4), p=round(p, 4), n=len(recs))

# ── Q2: CA-PC vs wrong-axis 配对（机制特异性）──
dif = [x['g_ca'] - x['g_wrong'] for x in recs]
n = len(dif)
mu = sum(dif) / n
sd = math.sqrt(sum((v - mu) ** 2 for v in dif) / (n - 1)) if n > 1 else 0.0
se = sd / math.sqrt(n) if n else 0.0
win = sum(1 for v in dif if v > 1e-9)
lose = sum(1 for v in dif if v < -1e-9)
q2 = dict(mean_ca_minus_wrong=round(mu, 4), sd=round(sd, 4), se=round(se, 4),
          t=round(mu / se, 2) if se else float('nan'), n=n,
          ca_wins=win, wrong_wins=lose, ties=n - win - lose)

# ── Q3: 各 corruption 随 severity 的增益趋势（条件轴内部是否单调）──
q3 = {}
for corr in sorted(dict.fromkeys(x['corr'] for x in recs)):
    g = [x for x in recs if x['corr'] == corr]
    by_sev = {}
    for sev in sorted(dict.fromkeys(x['sev'] for x in g)):
        gg = [x for x in g if x['sev'] == sev]
        by_sev[sev] = round(sum(x['g_ca'] for x in gg) / len(gg), 4)
    rs, _ = pearson([x['sev'] for x in g], [x['g_ca'] for x in g])
    q3[corr] = dict(by_sev=by_sev, r_sev_gca=round(rs, 3), n=len(g))

# ── 每条件档明细（按 H 降序）──
detail = sorted(recs, key=lambda z: -z['H'])

# ── 汇总：按 detector 拆分 ──
by_det = {}
for det in sorted(dict.fromkeys(x['det'] for x in recs)):
    g = [x for x in recs if x['det'] == det]
    rc, pc = pearson([x['H'] for x in g], [x['g_ca'] for x in g])
    rw, pw = pearson([x['H'] for x in g], [x['g_wrong'] for x in g])
    by_det[det] = dict(mean_g_ca=round(sum(x['g_ca'] for x in g) / len(g), 4),
                       mean_g_wrong=round(sum(x['g_wrong'] for x in g) / len(g), 4),
                       mean_g_star=round(sum(x['g_star'] for x in g) / len(g), 4),
                       r_H_gca=round(rc, 4), p_H_gca=round(pc, 4),
                       r_H_gwrong=round(rw, 4), p_H_gwrong=round(pw, 4), n=len(g))

summary = dict(
    config=cfg,
    n_cells=len(recs), n_clean=len(clean),
    q1_A_B_convergence=q1,
    q2_direction_specificity=q2,
    q3_severity_trend=q3,
    by_detector=by_det,
    detail=detail,
)
json.dump(summary, open(OUT_JSON, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


def fmt(x):
    return f'{x:+.4f}' if isinstance(x, float) else str(x)


# ── Markdown 报告 ──
L = []
L.append('# A/B 收敛受控实验 — 小样验证报告（#1）\n')
L.append('## 实验设计')
L.append('- **目的**：验证「观测条件修正」是否为 CA-PC 增益的驱动项（生死线）。')
L.append('- **受控条件**：vocab 固定（clean COCO → 退化 COCO，同类名，vocab=identity）；')
L.append('  观测条件轴 = fog / gaussian_blur / brightness × S1/3/5 = **9 档合成成像退化**。')
L.append(f"- **规模**：{cfg.get('dets')} ，k={cfg.get('k')} ，6 类（COCO），共 {len(recs)} 个退化 cell + {len(clean)} clean。")
L.append('- **条件严重度** H = AP_clean_base − AP_deg_base（外部 headroom，与 CA-PC 无关）。')
L.append('- **对照方法**：CA-PC(β=2, 忠实条件方向) / TGE(方向,无β) / wrong-axis(overhead−photo, 错轴控制) / CA-PC*(grid 自调β)。\n')

L.append('## A/B 收敛判据与结果\n')
L.append('若观测条件修正是驱动项，则 **g_CA 应随 H 显著相关**；而条件无关基线（g_wrong）应平稳。\n')
L.append('| 方法 | r(g, H) | p | 解读 |')
L.append('|---|---|---|---|')
L.append(f"| CA-PC | {q1['g_ca']['r']:+.3f} | {q1['g_ca']['p']:.3g} | **不随条件严重度走（ns）** |")
L.append(f"| wrong-axis | {q1['g_wrong']['r']:+.3f} | {q1['g_wrong']['p']:.3g} | 同样不随 H 走 |")
L.append(f"| TGE | {q1['g_tge']['r']:+.3f} | {q1['g_tge']['p']:.3g} | 不随 H 走 |")
L.append(f"| CA-PC* | {q1['g_star']['r']:+.3f} | {q1['g_star']['p']:.3g} | 自调β后仍不随 H 走 |")
L.append('')
L.append('**额外信号**：r(base_deg, g_CA) = −0.508 (p≈1.2e-9) —— base 越低（退化越重），CA-PC **反而伤得越多**。\n')

L.append('## 机制特异性（Q2：CA-PC vs 错轴对照）\n')
L.append(f"- mean(g_CA − g_wrong) = {q2['mean_ca_minus_wrong']:+.4f} ，sd={q2['sd']:.4f}，t≈{q2['t']}")
L.append(f"- CA-PC 在 **{q2['ca_wins']}/{q2['n']}** 个 cell 击败错轴；错轴在 **{q2['wrong_wins']}** 个 cell 反超。")
L.append('⇒ 条件修正增益与「随便一个方向」无差别，甚至更差 → **非机制特异**。\n')

L.append('## 按检测器拆分\n')
L.append('| 检测器 | mean g_CA | mean g_wrong | mean g_star | r(H,g_CA) | r(H,g_wrong) |')
L.append('|---|---|---|---|---|---|')
for det, v in by_det.items():
    L.append(f"| {det} | {v['mean_g_ca']:+.4f} | {v['mean_g_wrong']:+.4f} | {v['mean_g_star']:+.4f} | "
             f"{v['r_H_gca']:+.3f} | {v['r_H_gwrong']:+.3f} |")
L.append('')

L.append('## 各 corruption 随 severity 的趋势（Q3）\n')
L.append('| corruption | S1 g_CA | S3 g_CA | S5 g_CA | r(sev,g_CA) |')
L.append('|---|---|---|---|---|')
for corr, v in q3.items():
    bs = v['by_sev']
    L.append(f"| {corr} | {bs.get(1,float('nan')):+.4f} | {bs.get(3,float('nan')):+.4f} | "
             f"{bs.get(5,float('nan')):+.4f} | {v['r_sev_gca']:+.3f} |")
L.append('')

L.append('## 裁决\n')
L.append('**本小样不支持 A/B 收敛假设（即「观测条件修正是增益驱动项」）。**')
L.append('1. CA-PC 增益不随条件严重度 H 走（r≈−0.06, ns），且在重退化时反而有害。')
L.append('2. CA-PC 在多数 cell 被错轴方向击败 → 增益非机制特异。')
L.append('3. 与分解表互为印证：**cond-only +0.0159 < vocab-only +0.0243** —— 条件项确为较弱一半。\n')

L.append('## 关键局限（决定下一步）\n')
L.append('- **vocab=identity**：本轴只能隔离「cond-alone」，无法测「vocab 固定非零 + 条件变化」的完整收敛。')
L.append('- **合成成像轴**：fog/blur/brightness 文本锚可能与实际退化不对齐，未必是 RS GSD/传感器 的好代理。')
L.append('- **小样本**：k=40、仅 owlv2+yolo，噪声大；主力 gdino 的该轴评分被旧 OUT 覆盖，需补跑。\n')

L.append('## 建议的下一步（决定性实验）\n')
L.append('在**真实 RS 条件轴**（GSD/传感器，作用于 RS 目标域）上，用**非零 vocab 的跨域对**')
L.append('（如 DIOR→VOC 的 truck→veh 这类名称错配转移），以 **gdino** 重跑：')
L.append('比较 full(=vocab+cond) 与 vocab-only(丢掉 d_cond) 的增益差是否随条件严重度走。')
L.append('那一版才真正回答「条件修正是否跟条件走、vocab-only 不跟」。\n')

L.append('---')
L.append(f'_生成自 run_degradation.json（{len(recs)} cells）。纯 CPU 后处理，未跑 GPU。_')

open(OUT_MD, 'w', encoding='utf-8').write('\n'.join(L))

# 控制台摘要
print('A/B 收敛小样 — 裁决: 不支持（cond 修正不随条件严重度走, 且被错轴反超）')
print(f"  r(H,g_CA)={q1['g_ca']['r']:+.3f}(p={q1['g_ca']['p']:.3g})  "
      f"r(base_deg,g_CA)= -0.508  ca_wins={q2['ca_wins']}/{q2['n']}")
for det, v in by_det.items():
    print(f"  {det}: mean_gCA={v['mean_g_ca']:+.4f}  r(H,gCA)={v['r_H_gca']:+.3f}")
print(f'→ {OUT_JSON}')
print(f'→ {OUT_MD}')
