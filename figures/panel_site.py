#!/usr/bin/env python3
"""Fig. 3c: sensitivity and specificity by hospital system, MedDx and the AUD diagnosis side by side.
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "downstream/npj"); sys.path.insert(0, "downstream")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import npj_style as S
SRC = "downstream/npj/out/figure4/subgroup_rates.csv"
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="downstream/npj/out/figure4"); ap.add_argument("--letter", default="c"); ap.add_argument("--letter-visible", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
d = pd.read_csv(SRC); d = d[d["axis"] == "Hospital system"].set_index("level")
ORDER = ["Academic", "Safety net", "Other hospitals"]
LAB = {"Academic": "Academic system", "Safety net": "Safety-net hospital", "Other hospitals": "Other hospitals"}
W_MM, H_MM = 88.0, 54.0
S.apply_npj_style()
fig = plt.figure(figsize=S.mm(W_MM, H_MM)); fig.set_layout_engine("none")
axes = [fig.add_axes([30.0 / W_MM, 10.0 / H_MM, 25.0 / W_MM, 31.0 / H_MM]), fig.add_axes([59.0 / W_MM, 10.0 / H_MM, 25.0 / W_MM, 31.0 / H_MM])]
for ax, met, title in zip(axes, ("sens", "spec"), ("Sensitivity", "Specificity")):
    for k, lvl in enumerate(ORDER):
        y = len(ORDER) - 1 - k; r = d.loc[lvl]
        for reader, col, off, mk in (("code", S.C_CODE, 0.18, "s"), ("llm", S.C_INDEX, -0.18, "o")):
            v, lo, hi = r["%s_%s" % (met, reader)], r["%s_%s_lo" % (met, reader)], r["%s_%s_hi" % (met, reader)]
            ax.errorbar(v, y + off, xerr=[[v - lo], [hi - v]], fmt=mk, ms=3.0, lw=S.LW_MIN, capsize=1.4, color=col, mfc=col, mec=col, zorder=3)
            ax.text(101.5, y + off, "%.0f" % v, fontsize=S.FS_SMALL - 0.5, color=col, va="center", ha="left", clip_on=False)
        n = int(r["n_sens" if met == "sens" else "n_spec"])
        if met == "sens":
            fig.text(28.0 / W_MM, (10.0 + 31.0 * (y + 0.6) / len(ORDER)) / H_MM, "%s\nn = %d / %d" % (LAB[lvl], int(r["n_sens"]), int(r["n_spec"])), fontsize=S.FS_SMALL, ha="right", va="center", linespacing=1.15)
    ax.set_yticks([]); ax.set_ylim(-0.6, len(ORDER) - 0.4); ax.set_xlim(45, 101); ax.set_xticks([50, 75, 100])
    ax.set_title(title + " (%)", fontsize=S.FS_BODY, pad=3)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    for k in range(1, len(ORDER)): ax.axhline(k - 0.5, color=S.HAIR, lw=S.LW_MIN, ls=(0, (2, 2)))
fig.text(S.LETTER_BOX_W / W_MM, 51.0 / H_MM, "The reader gap is largest at the safety-net hospital", fontsize=S.FS_BODY, fontweight="bold", va="center")
handles = [Line2D([], [], ls="none", marker="o", ms=3.0, color=S.C_INDEX), Line2D([], [], ls="none", marker="s", ms=3.0, color=S.C_CODE)]
leg = fig.legend(handles, ["MedDx", "AUD diagnosis"], loc="upper left", bbox_to_anchor=(S.LETTER_BOX_W / W_MM, 47.5 / H_MM), ncol=2, frameon=False, fontsize=S.FS_SMALL, handletextpad=0.4, columnspacing=1.4, borderpad=0, borderaxespad=0)
fig.text(S.LETTER_BOX_W / W_MM, 3.0 / H_MM, "n = ground-truth-positive / ground-truth-negative charts; Wilson 95% CI", fontsize=S.FS_SMALL - 0.5, color=S.MUTED, va="center")
S.panel_letter(fig, a.letter, a.letter_visible); S.check_text_fits(fig, W_MM, H_MM)
for ext in ("pdf", "png"): fig.savefig(os.path.join(a.out, "panel_site." + ext), dpi=600, bbox_inches=None, pad_inches=0, facecolor="white")
print(d[["n_sens","sens_code","sens_llm","n_spec","spec_code","spec_llm"]].round(1).to_string())
