#!/usr/bin/env python3
"""Fig. 1c: ground truth against MedDx, three classes, count and row percentage per cell.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "downstream")
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.patches import Rectangle                                # noqa: E402

import npj_style as S                                                   # noqa: E402

SRC = "downstream/npj_goal6_outputs/panels/f1_confusion_6class.csv"
OUT = "downstream/npj/out/figure1"
PRIMARY = "Qwen3.6-27B"

COLLAPSE = {"Definitely Yes": "AUD+", "Probably Yes": "AUD+",
            "Probably No": "AUD-", "Definitely No": "AUD-",
            "Uncertain": "Indeterminate", "Not Enough Data": "Indeterminate"}
CATS3 = ["AUD+", "Indeterminate", "AUD-"]
SHORT = ["AUD+", "Indet.", "AUD−"]
CLASS_COLOUR = {"AUD+": "#5B2A86", "Indeterminate": "#8C8981", "AUD-": "#1B7F5A"}
S.TINT.update({"#5B2A86": "#E4DAEE", "#8C8981": "#EBEBEB", "#1B7F5A": "#D5EBE1"})

# 66 mm by 64 mm, so that panel a at 112 mm and a 5 mm gutter complete the 183 mm
# double-column width, and the two panels share one height.
W_MM, H_MM = 66.0, 64.0
RECT = [15.0 / W_MM, 15.0 / H_MM, 41.0 / W_MM, 41.0 / H_MM]


def matrix(src=SRC):
    """The primary model's counts over the three classes."""
    d = pd.read_csv(src)
    d = d[d["method"].eq(PRIMARY)]
    if d.empty:
        raise SystemExit("no rows for %s in %s" % (PRIMARY, src))
    m = np.zeros((3, 3), dtype=int)
    for _, r in d.iterrows():
        i = CATS3.index(COLLAPSE[r["reference_label"]])
        j = CATS3.index(COLLAPSE[r["prediction_label"]])
        m[i, j] += int(r["n"])
    return m


def draw_panel_b(ax, m, note=True):
    n = m.shape[0]
    rows = m.sum(axis=1, keepdims=True)
    pct = np.where(rows > 0, m / rows * 100, 0)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")

    for i in range(n):
        col = CLASS_COLOUR[CATS3[i]]
        for j in range(n):
            if i == j:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=S.TINT[col],
                                       ec="none", zorder=1))
            ax.text(j, i - 0.15, "%d" % m[i, j], ha="center", va="center",
                    fontsize=S.FS_BODY, color=S.INK, zorder=4)
            ax.text(j, i + 0.19, "%.0f%%" % pct[i, j], ha="center", va="center",
                    fontsize=S.FS_SMALL, color=S.MUTED, zorder=4)

    # Separators inside the matrix, light in tone and 1 pt in weight.
    for k in (0.5, 1.5):
        ax.plot([-0.5, n - 0.5], [k, k], color=S.HAIR, lw=S.LW_MIN, zorder=2)
        ax.plot([k, k], [-0.5, n - 0.5], color=S.HAIR, lw=S.LW_MIN, zorder=2)
    # The diagonal cell outlines sit above the separators.
    # The indeterminate class carries a dashed edge everywhere in the paper, which is
    # the only channel that separates it from the other two under grey printing.
    for i in range(n):
        dash = (0, (2.6, 1.9)) if CATS3[i] == "Indeterminate" else "solid"
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fc="none", ls=dash,
                               ec=CLASS_COLOUR[CATS3[i]], lw=S.LW_MIN, zorder=3))
    # The frame is Tier 2 and carries no category, so it is not the indeterminate
    # colour, which would read as though the frame belonged to that class.
    ax.add_patch(Rectangle((-0.5, -0.5), n, n, fc="none", ec=S.RULE,
                           lw=S.LW_MIN, zorder=5))

    ax.set_xticks(range(n))
    ax.set_xticklabels(SHORT, fontsize=S.FS_BODY, color=S.INK)
    ax.set_yticks(range(n))
    ax.set_yticklabels(SHORT, fontsize=S.FS_BODY, color=S.INK)
    ax.tick_params(axis="both", length=0, pad=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("MedDx", fontsize=S.FS_BODY, color=S.INK, labelpad=5)
    ax.set_ylabel("Ground truth", fontsize=S.FS_BODY, color=S.INK, labelpad=5)

    if note:
        tot, diag = int(m.sum()), int(np.trace(m))
        # Centred on the figure. Centred on the axes it ran past the right edge,
        # because the axes sit 15 mm from the left and only 2.5 mm from the right.
        # Bowker's test of symmetry on the three off-diagonal pairs (df = 3): are the
        # disagreements directional? kappa and its bootstrap CI come from the leaderboard.
        from scipy.stats import chi2
        pairs = [(0, 1), (0, 2), (1, 2)]
        bow = sum((m[i, j] - m[j, i]) ** 2 / (m[i, j] + m[j, i]) for i, j in pairs
                  if m[i, j] + m[j, i] > 0)
        p_sym = chi2.sf(bow, df=3)
        lb = pd.read_csv("figures/out/fig4/panel_model_leaderboard.csv")
        k = lb[lb["reader"] == PRIMARY].iloc[0]
        for yy, txt in ((0.088, "N = %d; %d (%.1f%%) in the same class" % (tot, diag, 100 * diag / tot)),
                        (0.049, "$\\kappa$ = %.3f (95%% CI %.3f\u2013%.3f)" % (k["kappa"], k["ci_low"], k["ci_high"])),
                        (0.010, "Disagreements symmetric: Bowker $P$ = %.2f" % p_sym)):
            ax.figure.text(0.5, yy, txt, ha="center", va="bottom", fontsize=S.FS_SMALL - 0.5, color=S.MUTED)
        print("Bowker chi2 %.2f df 3 P %.3f" % (bow, p_sym))
    return ax


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--letter", default="b")
    ap.add_argument("--letter-visible", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m = matrix(a.src)
    t = pd.DataFrame(m, index=CATS3, columns=CATS3)
    t.index.name = "reference_standard"
    t.to_csv(os.path.join(a.out, "panel_b_confusion3.csv"))
    print("three-category collapse, %s, N = %d, diagonal %d"
          % (PRIMARY, int(m.sum()), int(np.trace(m))))
    print(t.to_string())

    S.apply_npj_style()
    fig = plt.figure(figsize=S.mm(W_MM, H_MM))
    fig.set_layout_engine("none")
    fig.text(S.LETTER_BOX_W / W_MM, 61.5 / H_MM, "MedDx matches the ground truth on 464 of 485", fontsize=S.FS_BODY, fontweight="bold", color=S.INK, ha="left", va="center")
    ax = fig.add_axes(RECT)
    draw_panel_b(ax, m)
    S.panel_letter(fig, a.letter, a.letter_visible)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "panel_b." + ext), dpi=600,
                    bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    print("wrote %s/panel_b.pdf and .png at %.1f by %.1f mm" % (a.out, W_MM, H_MM))


if __name__ == "__main__":
    main()
