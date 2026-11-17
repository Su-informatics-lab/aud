#!/usr/bin/env python3
"""Fig. 3b: the 18 group differences for the AUD diagnosis and MedDx, sensitivity and specificity.
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, "figures"); sys.path.insert(0, "figures")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import npj_style as S
import panel_reader_swarm as R
OUT = "figures/out/fig3"
d = pd.read_csv(R.SRC_M)
S.apply_npj_style()
W, H = 88.0, 44.0
fig = plt.figure(figsize=S.mm(W, H)); fig.set_layout_engine("none")
ax = fig.add_axes([12.0 / W, 10.0 / H, 74.0 / W, 22.0 / H])
ax.axhspan(-R.BAND, R.BAND, color=S.RULE, alpha=0.28, lw=0, zorder=0); ax.axhline(0, color=S.RULE, lw=S.LW_MIN, zorder=1)
cols = []
for metric in ("sensitivity", "specificity"):
    for reader, col in ((R.CODE, S.C_CODE), (R.PRIMARY, S.C_INDEX)):
        v = d[d["metric"].eq(metric) & d["reader"].eq(reader)]["difference"].to_numpy(dtype=float); assert len(v) == 18
        cols.append((metric, "Diagnosis" if reader == R.CODE else "MedDx", v, col))
for x, (metric, label, v, col) in enumerate(cols):
    xx = x + (0.6 if x >= 2 else 0.0)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.add_patch(Rectangle((xx - 0.28, q1), 0.56, q3 - q1, facecolor="none", edgecolor=col, lw=S.LW_MIN * 0.7, alpha=0.3, zorder=2))
    ax.plot([xx - 0.32, xx + 0.32], [med, med], color=col, lw=S.LW_MIN * 0.9, alpha=0.4, zorder=2)
    ax.plot(xx + R.swarm(v), v, marker="o", ms=R.MS * 1.6, mfc=col, mec="white", mew=S.LW_MIN * 0.5, linestyle="none", zorder=3)
    ax.text(xx, 25.5, "%d" % int(np.sum(np.abs(v) > R.BAND)), ha="center", va="bottom", fontsize=S.FS_BODY, fontweight="bold", color=col)
ax.axvline(1.8, color=S.RULE, lw=S.LW_MIN)
ax.set_xlim(-0.6, 4.2); ax.set_ylim(-25, 25); ax.set_yticks([-20, -10, 0, 10, 20])
ax.set_xticks([0, 1, 2.6, 3.6]); ax.set_xticklabels(["Diagnosis", "MedDx", "Diagnosis", "MedDx"], fontsize=S.FS_SMALL)
for t_, c in zip(ax.get_xticklabels(), (S.C_CODE, S.C_INDEX, S.C_CODE, S.C_INDEX)): t_.set_color(c)
ax.text(0.5, -34, "Sensitivity", ha="center", va="top", fontsize=S.FS_SMALL, color=S.INK)
ax.text(3.1, -34, "Specificity", ha="center", va="top", fontsize=S.FS_SMALL, color=S.INK)
ax.set_ylabel("Group minus rest\n(points)", fontsize=S.FS_SMALL, labelpad=2, linespacing=1.1)
fig.text(12.0 / W, 36.3 / H, "Of 18 groups, beyond 5 points:", ha="left", va="bottom", fontsize=S.FS_SMALL, color=S.MUTED)
ax.tick_params(axis="x", length=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.text(7.0 / W, 41.5 / H, "The AUD diagnosis violates equity across groups, MedDx holds", fontsize=S.FS_BODY, fontweight="bold", va="center", color=S.INK)
S.check_text_fits(fig, W, H)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "panel_fig3_d." + ext), dpi=600, bbox_inches=None, pad_inches=0, facecolor="white")
print("wrote panel_fig3_d")
