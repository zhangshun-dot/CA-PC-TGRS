# -*- coding: utf-8 -*-
"""给 run_cross_vlm.py 打 OOM 加固补丁 (不改逻辑/不改口径, 只加三件事):
1. VLM 编码 batch 可经环境变量 AC_SPC_VLM_BS 调小 (SigLIP SO400M 在 8GB 卡上 bs=64 会 OOM);
2. 每跑完一个 cell 立即增量落盘 (原代码只在整轮 VLM 跑完后才写, 崩了全丢);
3. 支持断点续跑: --merge 时若 json 里已有该 tag 的部分 cell, 跳过已完成项。

生成 run_cross_vlm_robust.py (原脚本一个字节不动)。
"""
import io, sys, os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_cross_vlm.py')
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_cross_vlm_robust.py')

s = io.open(SRC, encoding='utf-8').read()
orig = s
reps = []

# ---- (1) 可调 batch: 在 VLMS 定义之后插入模块级默认 bs ----
old = "VLMS = {"
assert s.count(old) == 1, 'VLMS anchor not unique'
# 找到 VLMS dict 结束后的第一行
anchor_after = "\ndef load_blob("
assert s.count(anchor_after) == 1
new_bs = """

# --- OOM-hardening (2026-10-10): VLM 编码 batch 可经环境变量调小 ---
_DEFAULT_VLM_BS = int(os.environ.get('AC_SPC_VLM_BS', '0')) or None
if _DEFAULT_VLM_BS:
    print(f'[bs] AC_SPC_VLM_BS={_DEFAULT_VLM_BS} (覆盖默认 batch)', flush=True)
"""
s = s.replace(anchor_after, new_bs + anchor_after, 1)
reps.append('insert _DEFAULT_VLM_BS')

# ---- (2) enc_img 使用可调 batch + OOM 自动折半重试 ----
old_enc = """        if bs is None:
            bs = 64 if self.fam == 'siglip' else 128
        out = []
        for i in range(0, len(crops), bs):
            batch = crops[i:i + bs]
            inp = self.proc(images=batch, return_tensors='pt').to(DEVICE)
            f = RB._as_vec(self.model.get_image_features(**inp)).float()
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu())
        return torch.cat(out).numpy()"""
new_enc = """        if bs is None:
            bs = _DEFAULT_VLM_BS or (64 if self.fam == 'siglip' else 128)
        tries = 0
        while True:
            try:
                out = []
                for i in range(0, len(crops), bs):
                    batch = crops[i:i + bs]
                    inp = self.proc(images=batch, return_tensors='pt').to(DEVICE)
                    f = RB._as_vec(self.model.get_image_features(**inp)).float()
                    f = f / f.norm(dim=-1, keepdim=True)
                    out.append(f.cpu())
                    del inp, f
                return torch.cat(out).numpy()
            except Exception as _e:                      # 显存不足 -> 折半重试
                if not (isinstance(_e, torch.AcceleratorError) or
                        isinstance(_e, RuntimeError)) or 'memory' not in str(_e).lower():
                    raise
                try:
                    del out
                except Exception:
                    pass
                import gc as _gc; _gc.collect(); torch.cuda.empty_cache()
                tries += 1
                if bs <= 1 or tries >= 5:
                    raise
                bs = max(1, bs // 2)
                print(f'  [oom] 编码 batch 折半 -> {bs} (第 {tries} 次重试)', flush=True)"""
assert s.count(old_enc) == 1, 'enc_img body not found'
s = s.replace(old_enc, new_enc, 1)
reps.append('enc_img adaptive bs')

# ---- (3) 断点续跑 + 逐 cell 增量落盘 ----
old_loop = """        vcells = []
        for ci, c in enumerate(cells):
            if c.get('crops') is None:"""
new_loop = """        vcells = []
        # --- 断点续跑: 已有该 tag 的 cell 直接跳过 ---
        done_keys = set()
        try:
            _prev = json.load(open(OUT, encoding='utf-8')).get('cells', {}) or {}
            for _pc in _prev.get(tag, []) or []:
                done_keys.add((_pc['det'], _pc['tname'], _pc['src'], _pc['eval'], _pc['n_img']))
                vcells.append(_pc)
        except Exception:
            vcells = []
        if done_keys:
            print(f'  [resume] {tag} 已有 {len(done_keys)}/{len(cells)} cells, 跳过这些',
                  flush=True)

        def _dump_incr():
            try:
                _d = {}
                if os.path.exists(OUT):
                    try:
                        _d = json.load(open(OUT, encoding='utf-8')).get('cells', {}) or {}
                    except Exception:
                        _d = {}
                _d[tag] = vcells
                json.dump({'config': {'beta': BETA, 'lam': LAM, 'neg_topk': NEG_TOPK,
                                      'n_proto': N_PROTO, 'zero_target_gt': True,
                                      'stream': args.stream,
                                      'note': 'per-cell incremental dump'},
                           'cells': _d}, open(OUT, 'w', encoding='utf-8'),
                          indent=2, ensure_ascii=False)
            except Exception as _e:
                print(f'  [warn] 增量落盘失败: {_e}', flush=True)

        for ci, c in enumerate(cells):
            _key = (c['det'], c['tname'], c['src'], c['eval'], c['n_img'])
            if _key in done_keys:
                continue
            if c.get('crops') is None:"""
assert s.count(old_loop) == 1, 'cell loop anchor not found'
s = s.replace(old_loop, new_loop, 1)
reps.append('resume + _dump_incr')

# 每 cell 末尾 pequeno incremento
old_tail = """                  f'||dc||={cell["dc_mag"]:.3f}  ({ci+1}/{len(cells)})', flush=True)"""
new_tail = """                  f'||dc||={cell["dc_mag"]:.3f}  ({len(vcells)}/{len(cells)})', flush=True)
            _dump_incr()
            import gc as _gc2; _gc2.collect(); torch.cuda.empty_cache()"""
assert s.count(old_tail) == 1, 'cell tail anchor not found'
s = s.replace(old_tail, new_tail, 1)
reps.append('per-cell dump + empty_cache')

# 收尾处的整轮 dump 保留 (幂等, 无害)
assert s != orig
io.open(DST, 'w', encoding='utf-8').write(s)
print('WROTE', DST)
for r in reps:
    print('  patch:', r)
# 语法自检
import ast
ast.parse(s)
print('syntax OK')
