# -*- coding: utf-8 -*-
"""给 run_cross_vlm_cellmajor.py 补「断点续跑」(不改口径/不改计算):
原脚本每跑完一个 cell 就增量落盘, 重启时却从 cell 1 重来并覆盖已有结果。
本补丁在启动时把 json 里该 tag 已完成的 cell 读回, 并跳过这些 cell。

生成 run_cross_vlm_cellmajor_resume.py (原脚本一个字节不动)。
batch 无需改: 原脚本支持 CVLM_BS_SIGLIP 环境变量。
"""
import io, os, ast

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_cross_vlm_cellmajor.py')
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_cross_vlm_cellmajor_resume.py')

s = io.open(SRC, encoding='utf-8').read()
orig = s

# ---- 1) 启动时读回已完成的 cell ----
old1 = "    res = {tag: [] for tag in vlms}\n"
new1 = """    res = {tag: [] for tag in vlms}
    # --- 断点续跑 (2026-10-10) ---
    try:
        _old_full = json.load(open(OUT, encoding='utf-8')).get('cells', {}) or {}
    except Exception:
        _old_full = {}
    DONE = {}
    for tag in vlms:
        _prev = [c for c in (_old_full.get(tag) or [])]
        res[tag] = _prev
        DONE[tag] = {(c['det'], c['tname'], c['src'], c['eval'], c['n_img']) for c in _prev}
    if any(len(v) for v in DONE.values()):
        print('  [resume] ' + ', '.join(f'{t}:{len(DONE[t])}/{len(cells)}' for t in vlms),
              flush=True)
"""
assert s.count(old1) == 1, 'anchor res= not unique'
s = s.replace(old1, new1, 1)

# ---- 2) cell 循环内跳过已完成项 ----
old2 = """        for tag, vlm in vlms.items():
            bank, Ep, Eo = prep[tag]"""
new2 = """        for tag, vlm in vlms.items():
            _k = (c['det'], c['tname'], c['src'], c['eval'], c['n_img'])
            if _k in DONE[tag]:
                continue
            bank, Ep, Eo = prep[tag]"""
assert s.count(old2) == 1, 'anchor per-tag loop not unique'
s = s.replace(old2, new2, 1)

# ---- 3) 打印进度时不要索引 res[t][-1] (跳过时长度会错位) ----
old3 = """        print(f'  (cell {ci+1}/{len(cells)}) {c["det"]} {c["src"]}->{c["eval"]} '
              f'nbox={c["n_box"]}  ' + '  '.join(
                  f'{t}: base={res[t][-1]["base"]:.4f} ca_g={res[t][-1]["ca_pc_g"]:.4f}' for t in vlms),
              flush=True)"""
new3 = """        print(f'  (cell {ci+1}/{len(cells)}) {c["det"]} {c["src"]}->{c["eval"]} '
              f'nbox={c["n_box"]}  [ok {[len(res[t]) for t in vlms]}]', flush=True)"""
assert s.count(old3) == 1, 'anchor progress print not unique'
s = s.replace(old3, new3, 1)

assert s != orig
io.open(DST, 'w', encoding='utf-8').write(s)
ast.parse(s)
print('WROTE', DST)
print('  patch 1: resume seed from existing json')
print('  patch 2: skip completed cells')
print('  patch 3: progress print uses lengths (skip-safe)')
print('syntax OK')
