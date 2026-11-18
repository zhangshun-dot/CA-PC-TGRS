# -*- coding: utf-8 -*-
"""gen_tab_vlm.py — 从 run_cross_vlm.json 生成 ISPRS 论文 tab:vlm 的 LaTeX

表形: 行 = 5 个 VLM backbone + Mean; 列 = AP50(Base/CA-PC/Δ/Δ*oracle) + mAP(Base/CA-PC/Δ)
口径: CA-PC 列 = ca_pc_g (dc×conf gate 变体), 与全文所有表一致; Δ* = ca_pc_star - base
      (oracle 强度, 仅作"方向可迁移"证据, β* 在目标 GT 上调, 不是可比方法)。
用法: python gen_tab_vlm.py  (默认纯 python, 无需 numpy)
"""
import json

JSON = r'C:\Users\张三\Desktop\实验数据\run_cross_vlm.json'
OUT = r'C:\Users\张三\Desktop\实验数据\tab_vlm.tex'

ORDER = ['clip_b32', 'clip_l14', 'openclip_l14', 'openclip_b32', 'siglip_so400m']
LABEL = {
    'clip_b32': 'CLIP ViT-B/32',
    'clip_l14': 'CLIP ViT-L/14',
    'openclip_l14': 'OpenCLIP ViT-L/14',
    'openclip_b32': 'OpenCLIP ViT-B/32',
    'siglip_so400m': 'SigLIP SO400m',
}

d = json.load(open(JSON, encoding='utf-8'))
cs = d['cells']

MIN_N = 36   # 只纳入跑满的 backbone; 部分运行(如 siglip 7/36)不计入, 保证各列 n 一致


def mean(v, k):
    return sum(c[k] for c in v) / len(v)


rows = {}
partial = {}
for tag in ORDER:
    v = cs.get(tag)
    if not v or len(v) < MIN_N:
        rows[tag] = None
        if v:
            partial[tag] = len(v)
        continue
    rows[tag] = dict(
        base=mean(v, 'base'), ca=mean(v, 'ca_pc_g'), base_m=mean(v, 'base_m'),
        ca_m=mean(v, 'ca_pc_g_m'), orc=mean(v, 'ca_pc_star'), n=len(v),
        dc=mean(v, 'dc_mag'),
    )

avail = [t for t in ORDER if rows[t]]
mean_row = {
    'base': sum(rows[t]['base'] for t in avail) / len(avail),
    'ca': sum(rows[t]['ca'] for t in avail) / len(avail),
    'base_m': sum(rows[t]['base_m'] for t in avail) / len(avail),
    'ca_m': sum(rows[t]['ca_m'] for t in avail) / len(avail),
    'orc': sum(rows[t]['orc'] for t in avail) / len(avail),
}

lines = []
lines.append(r'\begin{table}[ht]')
lines.append(r'\centering')
lines.append(r'\footnotesize')
lines.append(r'\setlength{\tabcolsep}{3.5pt}')
lines.append(r'\caption{Backbone robustness of CA-PC across five vision--language '
             r'encoders, evaluated on the identical four-detector DIOR$\leftrightarrow$COCO '
             r'closed loop (36 detector--class cells) with the identical cached detector boxes '
             r'and crop set, so only the encoder that produces the crop features changes. '
             r'CA-PC is the deployed zero-target-annotation variant of '
             r'Section~\ref{sec:correct}; $\Delta^{\star}$ is the AP50 reached with the '
             r'per-cell correction strength tuned on target ground truth, which serves only as '
             r'evidence that the canonical-shift \emph{direction} is encoder-transferable, not '
             r'as a deployable configuration. The canonical-shift magnitude grows with encoder '
             r'capacity ($\|d_\xi\|$ from $%.2f$ to $%.2f$), so the encoder operating range is '
             r'made explicit: the direction transfers to all encoders, whereas the zero-GT '
             r'strength rule calibrated on the compact CLIP-B/32 space is not strength-neutral '
             r'and must be re-calibrated for the higher-capacity spaces.}'
             % (min(rows[t]['dc'] for t in avail), max(rows[t]['dc'] for t in avail)))
lines.append(r'\label{tab:vlm}')

lines.append(r'\begin{tabular}{@{}lrrrrrrr@{}}')
lines.append(r'\toprule')
lines.append(r' & \multicolumn{4}{c}{AP50 (eleven-point)} & \multicolumn{3}{c}{mAP@[0.50:0.95]} \\')
lines.append(r'\cmidrule(l){2-5} \cmidrule(l){6-8}')
lines.append(r'Backbone & Base & CA-PC & $\Delta$ & $\Delta^{\star}$ & Base & CA-PC & $\Delta$ \\')
lines.append(r'\midrule')


def fmt(r):
    return ('%.4f & %.4f & $%+.4f$ & $%+.4f$ & %.4f & %.4f & $%+.4f$'
            % (r['base'], r['ca'], r['ca'] - r['base'], r['orc'] - r['base'],
               r['base_m'], r['ca_m'], r['ca_m'] - r['base_m']))


for tag in ORDER:
    r = rows.get(tag)
    if r is None:
        note = ('\\emph{(partial run: %d/%d cells, excluded)}' % (partial[tag], MIN_N)) if tag in partial \
            else '\\emph{(pending)}'
        lines.append(r'%s & \multicolumn{7}{c}{%s} \\' % (LABEL[tag], note))
        continue
    lines.append(r'%s & %s \\' % (LABEL[tag], fmt(r)))
lines.append(r'\midrule')
lines.append(r'Mean & %s \\' % fmt(mean_row))
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')

open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

# ── 同时打印速查 ──
print('tab:vlm ->', OUT)
print('%-20s %8s %8s %8s %8s | %8s %8s %8s' % ('backbone', 'b50', 'ca50', 'd50', 'd50*', 'bm', 'cam', 'dm'))
for tag in ORDER:
    r = rows.get(tag)
    if r is None:
        print('%-20s  (%s)' % (LABEL[tag], ('partial %d/%d' % (partial[tag], MIN_N)) if tag in partial else 'pending'))
        continue
    print('%-20s %8.4f %8.4f %+8.4f %+8.4f | %8.4f %8.4f %+8.4f' % (
        LABEL[tag], r['base'], r['ca'], r['ca'] - r['base'], r['orc'] - r['base'],
        r['base_m'], r['ca_m'], r['ca_m'] - r['base_m']))
print('%-20s %8.4f %8.4f %+8.4f %+8.4f | %8.4f %8.4f %+8.4f' % (
    'MEAN', mean_row['base'], mean_row['ca'], mean_row['ca'] - mean_row['base'],
    mean_row['orc'] - mean_row['base'], mean_row['base_m'], mean_row['ca_m'],
    mean_row['ca_m'] - mean_row['base_m']))
print('  n_cells per backbone:', {t: (rows[t]['n'] if rows[t] else 0) for t in ORDER})
print('  dp(dc_mag):', {t: (round(rows[t]['dc'], 3) if rows[t] else None) for t in ORDER})
