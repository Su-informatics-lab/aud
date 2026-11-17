#!/usr/bin/env python3
"""Fig. 3e: supervised comparators trained at the academic health system and applied to the other hospital systems.
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "downstream"); sys.path.insert(0, "downstream/npj")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import npj_style as S
SRC = "downstream/npj/out/figure4/site_transfer.csv"
W_MM, H_MM = 183.0, 44.0
SITES = [("IUH", "Academic system, training"), ("Eskenazi", "Safety-net hospital"), ("Other", "Other hospitals")]
READERS = [("MedDx", "MedDx, training-free", S.C_INDEX, "", 1.0),
           ("XGBoost, with F10.x", "XGBoost, structured record, trained at the academic system", S.C_CODE, "", 1.0),
           ("TF-IDF logistic regression", "TF-IDF regression, notes, trained at the academic system", S.C_INDEX, "////", 0.0)]
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="downstream/npj/out/figure4"); ap.add_argument("--letter", default="e"); ap.add_argument("--letter-visible", action="store_true"); a = ap.parse_args()
d = pd.read_csv(SRC)
S.apply_npj_style(); fig = plt.figure(figsize=S.mm(W_MM, H_MM)); fig.set_layout_engine("none")
AX_Y0, AX_H = 9.5, 24.5
axes = [fig.add_axes([12.0 / W_MM, AX_Y0 / H_MM, 78.0 / W_MM, AX_H / H_MM]), fig.add_axes([103.0 / W_MM, AX_Y0 / H_MM, 78.0 / W_MM, AX_H / H_MM])]
bw = 0.24
for ax, met, title in zip(axes, ("sens", "spec"), ("Sensitivity", "Specificity")):
    for si, (site, slab) in enumerate(SITES):
        for ri, (key, lab, col, hatch, fillalpha) in enumerate(READERS):
            r = d[(d["reader"] == key) & (d["site"] == site)].iloc[0]
            x = si + (ri - 1.0) * bw
            ax.bar(x, r[met], width=bw * 0.92, facecolor=col if fillalpha else "white", edgecolor=col, hatch=hatch, linewidth=S.LW_MIN, zorder=2)
            ax.errorbar(x, r[met], yerr=[[r[met] - r[met + "_lo"]], [r[met + "_hi"] - r[met]]], fmt="none", ecolor=S.INK, elinewidth=S.LW_MIN, capsize=1.2, zorder=3)
            ax.text(x, r[met + "_hi"] + 0.025, "%.0f" % (100 * r[met]), ha="center", va="bottom", fontsize=S.FS_SMALL - 1.5, color=S.INK, zorder=4)
    ax.set_xticks(range(len(SITES))); ax.set_xticklabels([s[1] for s in SITES], fontsize=S.FS_SMALL)
    ax.set_ylim(0, 1.14); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "25", "50", "75", "100"], fontsize=S.FS_SMALL)
    ax.set_ylabel(title + " (%)", fontsize=S.FS_SMALL); ax.tick_params(axis="x", length=0, pad=2); ax.tick_params(axis="y", pad=2)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.axhline(0.5, color=S.RULE, lw=S.LW_MIN, ls=(0, (2.4, 2.0)), zorder=1)
fig.text(S.LETTER_BOX_W / W_MM, 41.5 / H_MM, "MedDx is more specific than supervised models applied outside the system that trained them", fontsize=S.FS_BODY, color=S.INK, ha="left", va="center", fontweight="bold")
handles = [Patch(facecolor=c if f else "white", edgecolor=c, hatch=h) for _, _, c, h, f in READERS]
leg = fig.legend(handles, [l for _, l, _, _, _ in READERS], loc="upper left", bbox_to_anchor=(S.LETTER_BOX_W / W_MM, 38.0 / H_MM), ncol=3, frameon=False, fontsize=S.FS_SMALL, handlelength=1.4, handleheight=0.9, handletextpad=0.6, columnspacing=1.6, borderpad=0.0, borderaxespad=0.0)
for t_ in leg.get_texts(): t_.set_color(S.INK)
fig.text(S.LETTER_BOX_W / W_MM, 1.6 / H_MM, "Wilson 95% CI; academic-system values are 5-fold out-of-fold; thresholds set at the training-site prevalence", fontsize=S.FS_SMALL - 1.0, color=S.MUTED, ha="left", va="bottom")
S.panel_letter(fig, a.letter, a.letter_visible)
os.makedirs(a.out, exist_ok=True)
for ext in ("pdf", "png"): fig.savefig(os.path.join(a.out, "panel_transfer." + ext), dpi=600, bbox_inches=None, pad_inches=0, facecolor="white")
print("wrote %s/panel_transfer.pdf and .png at %.0f by %.0f mm" % (a.out, W_MM, H_MM))
