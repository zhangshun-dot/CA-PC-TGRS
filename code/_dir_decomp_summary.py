# -*- coding: utf-8 -*-
"""方向分解 (probe_dir_decomp.json) 汇总: full vs par vs perp vs tge, 部署口径(门控) + oracle 峰值.
用法: python _dir_decomp_summary.py
输出: 4 检测器的前向/反向表 + 一条 par-vs-full 一致性判定.
"""
import json
import pathlib
import statistics as st

ROOT = pathlib.Path(r"C:\Users\张三\Desktop\实验数据")
D = json.loads((ROOT / "probe_dir_decomp.json").read_text(encoding="utf-8"))
PD = D["per_det"]
VECS = ["full", "par", "perp", "tge"]


def agg(det, tname):
    rows = [r for r in PD[det] if r["tname"] == tname]
    if not rows:
        return None
    out = {}
    for v in VECS:
        g = [r[f"{v}_g"] - r["base"] for r in rows]
        s = [r[f"{v}_star"] - r["base"] for r in rows]
        out[v] = dict(gate=st.mean(g), oracle=st.mean(s),
                      pos=sum(1 for x in g if x > 0), n=len(g))
    return out


print("=" * 78)
print("方向分解汇总 (probe_dir_decomp.json)  —— 所有向量范数匹配 ||d_c||, 同 β grid, 同门控")
print("=" * 78)
verdict = {}
for det in PD:
    for tname in ["fwd_dior2coco", "rev_coco2dior"]:
        a = agg(det, tname)
        if not a:
            continue
        tag = "前向 DIOR→COCO" if tname.startswith("fwd") else "反向 COCO→DIOR"
        print(f"\n[{det}] {tag}  (n={a['full']['n']})")
        print(f"  {'向量':<6} {'门控增益':>10} {'oracle峰值':>11}  {'门控正':>7}")
        for v in VECS:
            m = a[v]
            print(f"  {v:<6} {m['gate']:+10.4f} {m['oracle']:+11.4f}  {m['pos']:>3d}/{m['n']}")
        d = a["par"]["gate"] - a["full"]["gate"]
        verdict.setdefault(det, {})[tname] = d
        flag = "par>full ✅" if d > 0 else "par≤full"
        print(f"  → par − full (门控) = {d:+.4f}   {flag}")

print("\n" + "=" * 78)
print("前向 par-vs-full 一致性判定 (预登记规则 d: 至少一个检测器同向)")
print("=" * 78)
ok = 0
tot = 0
for det, d in verdict.items():
    v = d.get("fwd_dior2coco")
    if v is None:
        continue
    tot += 1
    if v > 0:
        ok += 1
    print(f"  {det:<8} 前向 par−full = {v:+.4f}  {'✅' if v > 0 else '❌'}")
print(f"  → 前向一致为正: {ok}/{tot}  " + ("通过" if ok >= 1 else "未通过"))
