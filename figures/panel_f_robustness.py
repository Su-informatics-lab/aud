#!/usr/bin/env python3
"""Supplementary Fig. 3: the group differences for MedDx, its four skill variations and the other ten engines.
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, "figures"); sys.path.insert(0, "figures")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import npj_style as S
import panel_reader_swarm as R
OUT = "figures/out/sf"
def load(src, metric):
    d = pd.read_csv(src); d = d[d["metric"].eq(metric)]
    return {n: g["difference"].to_numpy(dtype=float) for n, g in d.groupby("reader")}
S.apply_npj_style()
W, H = 183.0, 65.0
fig = plt.figure(figsize=S.mm(W, H)); fig.set_layout_engine("none")
for j, (metric, y0, lab, ylim, yt) in enumerate([("sensitivity", 38.0, "Sensitivity,\ngroup minus rest", (-14, 14), [-10, 0, 10]),
                                                 ("specificity", 15.0, "Specificity,\ngroup minus rest", (-30, 30), [-20, 0, 20])]):
    cols = [c for c in R.columns(load(R.SRC_M, metric), load(R.SRC_P, metric)) if not c[2]]
    _REL = {"MedDx, T = 0": "MedDx, temperature 0", "T = 0.3": "Temperature 0.3", "T = 0.7": "Temperature 0.7", "Paraphrase A": "Paraphrased skill A", "Paraphrase B": "Paraphrased skill B"}
    cols = [(_REL.get(l, l), v, z) for (l, v, z) in cols]
    ax = fig.add_axes([21.0 / W, y0 / H, 158.0 / W, 17.0 / H])
    ax.axhspan(-R.BAND, R.BAND, color=S.RULE, alpha=0.28, lw=0, zorder=0); ax.axhline(0, color=S.INK, lw=S.LW_MIN, zorder=1)
    for x, (label, v, _) in enumerate(cols):
        q1, med, q3 = np.percentile(v, [25, 50, 75]); col = S.C_INDEX
        ax.add_patch(Rectangle((x - 0.28, q1), 0.56, q3 - q1, facecolor="white", edgecolor=col, lw=S.LW_MIN * 0.8, zorder=2))
        ax.plot([x - 0.32, x + 0.32], [med, med], color=col, lw=S.LW_MIN * 1.4, zorder=4)
        ax.plot(x + R.swarm(v), v, marker="o", ms=R.MS, mfc=col, mec="white", mew=S.LW_MIN * 0.35, linestyle="none", zorder=3)
        ax.text(x, ylim[1] * 1.02, "%d" % int(np.sum(np.abs(v) > R.BAND)), ha="center", va="bottom", fontsize=S.FS_SMALL, color=S.INK)
    ax.axvline(4.5, color=S.RULE, lw=S.LW_MIN, zorder=1)
    ax.set_xlim(-0.62, len(cols) - 0.38); ax.set_ylim(*ylim); ax.set_yticks(yt)
    ax.set_xticks(range(len(cols)))
    if j == 1:
        ax.set_xticklabels([c[0] for c in cols], rotation=R.ROT, ha="right", rotation_mode="anchor", fontsize=R.FS_TICK)
        for t_, (label, _, _) in zip(ax.get_xticklabels(), cols):
            if label in R.BOLD: t_.set_fontweight("bold")
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="x", pad=1.5, length=2); ax.set_ylabel(lab, fontsize=S.FS_SMALL, labelpad=3, linespacing=1.1)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.text(20.0 / W, (y0 + 17.0 + 0.6) / H, "Of 18 groups,\nbeyond 5 points", ha="right", va="bottom", fontsize=S.FS_SMALL, color=S.MUTED, linespacing=1.05)
fig.text(7.0 / W, 63.0 / H, "Parity across patient groups holds across engines and skill variations", fontsize=S.FS_BODY, fontweight="bold", va="center", color=S.INK)
S.check_text_fits(fig, W, H)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "panel_f_robustness." + ext), dpi=600, bbox_inches=None, pad_inches=0, facecolor="white")
print("wrote panel_f_robustness")
