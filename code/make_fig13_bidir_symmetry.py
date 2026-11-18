"""Render fig13: bidirectional symmetry of CA-PC gains (forward vs reverse, 6 paired pairs).
Reuses already-computed numbers from run_bidirectional.json + run_reverse_extra.json. ZERO GPU.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = "C:/Users/张三/Desktop/实验数据"

d = json.load(open(f"{DATA}/run_bidirectional.json"))
fwd = d["results"]["fwd_dior2coco"]
rev3 = d["results"]["rev_coco2dior"]
ext = json.load(open(f"{DATA}/run_reverse_extra.json"))["results"]
def g(e):
    return e["plus_delta"] - e["baseline"]

# paired by COCO target class
pairs = [
    ("airplane", "airplane->airplane", "airplane->airplane"),
    ("boat/ship", "ship->boat",        "boat->ship"),
    ("car",       "vehicle->car",      "car->vehicle"),
    ("truck",     "vehicle->truck",    "truck->vehicle"),
    ("bus",       "vehicle->bus",      "bus->vehicle"),
    ("motorcycle","vehicle->motorcycle","motorcycle->vehicle"),
]
fg = [g(fwd[a]) for _, a, _ in pairs]
rg = [g(rev3[b]) if b in rev3 else ext[b]["gain"] for _, _, b in pairs]

fwd_mean = float(np.mean(fg))
rev_mean = float(np.mean(rg))

x = np.arange(len(pairs))
w = 0.38
fig, ax = plt.subplots(figsize=(9.2, 4.6))
b1 = ax.bar(x - w/2, fg, w, label=f"Forward DIOR→COCO (mean +{fwd_mean:.4f})", color="#2c6fbb")
b2 = ax.bar(x + w/2, rg, w, label=f"Reverse COCO→DIOR (mean +{rev_mean:.4f})", color="#c0392b")
ax.axhline(fwd_mean, ls="--", lw=1, color="#2c6fbb", alpha=0.7)
ax.axhline(rev_mean, ls="--", lw=1, color="#c0392b", alpha=0.7)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([p[0] for p in pairs], fontsize=10)
ax.set_ylabel("AP50 gain (canonical-anchored CA-PC)", fontsize=10)
ax.set_title("Bidirectional symmetry: same fixed config (β=2.0, λ=0.5), both directions +0.065, 6/6 positive",
             fontsize=10.5)
ax.legend(fontsize=8.5, loc="upper right")
for bars in (b1, b2):
    for r in bars:
        h = r.get_height()
        ax.annotate(f"{h:+.3f}", (r.get_x()+r.get_width()/2, h),
                    ha="center", va="bottom" if h >= 0 else "top",
                    fontsize=7.2, rotation=0)
ax.set_ylim(-0.01, 0.13)
ax.grid(axis="y", ls=":", alpha=0.4)
fig.tight_layout()
out = f"{DATA}/figs/fig13_bidirectional_symmetry.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
print(f"FWD mean +{fwd_mean:.4f} | REV mean +{rev_mean:.4f}")
