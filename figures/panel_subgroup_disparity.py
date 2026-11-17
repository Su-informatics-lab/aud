#!/usr/bin/env python3
"""Fig. 3a: sensitivity and specificity of each patient group minus the rest, for both classifiers.
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
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402

import npj_style as S                                                    # noqa: E402

SRC = "downstream/npj/out/figure4/subgroup_disparity.csv"
OUT = "downstream/npj/out/figure4"

W_MM, H_MM = 90.0, 162.0
X_LAB, X_INDENT = 1.0, 3.0
AX1_X0, AX1_W = 24.0, 20.0
NUM1_X = AX1_X0 + AX1_W + 1.0
AX2_X0, AX2_W = 57.0, 20.0
NUM2_X = AX2_X0 + AX2_W + 1.0
Y_AX0, AX_H = 15.0, 130.0
Y_LEGEND, Y_NOTE = 154.5, 150.0
XLIM = (-40.0, 25.0)
XTICKS = [-30, -15, 0, 15]
BAND = 5.0
OFF = 0.22
MIN_N = 20
# The three axes Fig. 4a also carries come first and in its order, so the two panels can
# be read against each other. Hospital system sits beside neighbourhood deprivation
# because the section compares them from its third paragraph onward.
ORDER = ["Age", "Sex", "Race", "Residence", "Neighbourhood deprivation"]


def load(src=SRC):
    d = pd.read_csv(src)
    rows = []
    for axis in ORDER:
        rows.append({"kind": "head", "label": axis})
        for _, r in d[d["axis"].eq(axis)].iterrows():
            rows.append({"kind": "data", "label": r["level"], "r": r})
    return rows, d


def half(ax, rows, ys, tag, num_x, xlabel):
    # alternate groups carry a faint band, so the six axes read as six blocks
    heads = [k for k, r in enumerate(rows) if r["kind"] == "head"] + [len(rows)]
    for g, (k0, k1) in enumerate(zip(heads[:-1], heads[1:])):
        if g % 2 == 1:
            ax.axhspan(ys[k1 - 1] - 0.5, ys[k0] + 0.5, color=S.HAIR, alpha=0.35, lw=0, zorder=0)
    ax.axvspan(-BAND, BAND, color=S.RULE, alpha=0.28, lw=0, zorder=0)
    ax.axvline(0.0, color=S.INK, lw=S.LW_MIN, zorder=1)
    for k, row in enumerate(rows):
        if row["kind"] != "data":
            continue
        r, yy = row["r"], ys[k]
        n = int(r["n_" + tag])
        tot = {"sens": 250, "spec": 173}[tag]
        grey = n < MIN_N
        for reader, col, sign in (("code", S.C_CODE, +1.0), ("llm", S.C_INDEX, -1.0)):
            v = float(r["%s_%s" % (tag, reader)])
            lo = float(r["%s_%s_lo" % (tag, reader)])
            hi = float(r["%s_%s_hi" % (tag, reader)])
            if not np.isfinite(v):
                continue
            ax.errorbar(v, yy + sign * OFF, xerr=[[v - lo], [hi - v]], fmt="o", ms=2.6,
                        lw=S.LW_MIN, capsize=1.0, color=col,
                        mfc="white" if grey else col, mec=col,
                        mew=S.LW_MIN * 1.4, zorder=3)
        ax.annotate("%d (%.0f%%)" % (n, 100.0 * n / tot), xy=(1.0, yy),
                    xytext=((num_x - ax.get_position().x1 * W_MM) * 72.0 / 25.4, 0),
                    textcoords="offset points", xycoords=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=S.FS_SMALL,
                    color=S.MUTED if grey else S.INK, annotation_clip=False)

    ax.set_yticks([])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(*XLIM)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([str(t) for t in XTICKS])
    ax.set_xlabel(xlabel, fontsize=S.FS_SMALL, color=S.INK, labelpad=3, linespacing=1.1)
    ax.tick_params(axis="x", pad=2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--letter", default="a")
    ap.add_argument("--letter-visible", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not Path(SRC).exists():
        raise SystemExit("%s is missing; run f04_subgroup_disparity.py first" % SRC)
    rows, d = load()
    n = len(rows)
    ys = {k: n - 1 - k for k in range(n)}

    for tag, what in (("sens", "sensitivity"), ("spec", "specificity")):
        for reader, name in (("code", "code"), ("llm", "model")):
            v = d["%s_%s" % (tag, reader)]
            out = (v.abs() > BAND).sum()
            excl = ((d["%s_%s_lo" % (tag, reader)] > 0)
                    | (d["%s_%s_hi" % (tag, reader)] < 0)).sum()
            print("  %-11s %-5s largest %5.1f points; outside the band %2d of %d; "
                  "intervals excluding zero %2d"
                  % (what, name, v.abs().max(), out, len(d), excl))

    S.apply_npj_style()
    fig = plt.figure(figsize=S.mm(W_MM, H_MM))
    fig.set_layout_engine("none")
    ax1 = fig.add_axes([AX1_X0 / W_MM, Y_AX0 / H_MM, AX1_W / W_MM, AX_H / H_MM])
    ax2 = fig.add_axes([AX2_X0 / W_MM, Y_AX0 / H_MM, AX2_W / W_MM, AX_H / H_MM])
    fig.canvas.draw()
    half(ax1, rows, ys, "sens", NUM1_X, "Sensitivity\ndifference (points)")
    half(ax2, rows, ys, "spec", NUM2_X, "Specificity\ndifference (points)")
    fig.text(X_LAB / W_MM, 2.2 / H_MM, "Difference: this group minus all other groups of the same variable, on the chart-reviewed sample", fontsize=S.FS_SMALL - 0.5, color=S.MUTED, ha="left", va="center")

    for k, row in enumerate(rows):
        head = row["kind"] == "head"
        y_fig = (Y_AX0 + AX_H * (ys[k] + 0.6) / n) / H_MM
        x_mm = X_LAB if head else X_LAB + X_INDENT
        tx = fig.text(x_mm / W_MM, y_fig, row["label"], fontsize=S.FS_SMALL,
                      color=S.INK, ha="left", va="center",
                      fontweight="bold" if head else "normal")
        bb = tx.get_window_extent(renderer=fig.canvas.get_renderer())
        if not head and bb.x1 / fig.dpi * 25.4 > AX1_X0 - 0.8:
            print("  WARNING label reaches the left plot: %.1f mm  %r"
                  % (bb.x1 / fig.dpi * 25.4, row["label"]))

    for x_mm in (NUM1_X, NUM2_X):
        fig.text(x_mm / W_MM, Y_NOTE / H_MM, "n (%)", fontsize=S.FS_SMALL,
                 color=S.MUTED, ha="left", va="center")

    handles = [Line2D([], [], marker="o", ms=2.6, linestyle="none", color=S.C_CODE,
                      label="AUD diagnosis"),
               Line2D([], [], marker="o", ms=2.6, linestyle="none", color=S.C_INDEX,
                      label="MedDx")]
    leg = fig.legend(handles=handles, loc="upper left",
                     bbox_to_anchor=(X_LAB / W_MM, (Y_LEGEND + 2.4) / H_MM),
                     ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.6,
                     fontsize=S.FS_SMALL, borderpad=0.0, handlelength=0.9)
    for txt in leg.get_texts():
        txt.set_color(S.INK)

    fig.text((X_LAB + 5.0) / W_MM, 160.0 / H_MM, "MedDx keeps parity across groups, the diagnosis does not", fontsize=S.FS_BODY, color=S.INK, ha="left", va="center", fontweight="bold")
    S.panel_letter(fig, a.letter, a.letter_visible)
    S.check_text_fits(fig, W_MM, H_MM)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "panel_subgroup_disparity." + ext), dpi=600,
                    bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    print("wrote %s/panel_subgroup_disparity.pdf and .png at %.1f by %.1f mm"
          % (a.out, W_MM, H_MM))


if __name__ == "__main__":
    main()
