#!/usr/bin/env python3
"""Fig. 3d: each patient group as an operating point, sensitivity against the false positive rate, with the convex hull of each classifier's points.
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
from matplotlib.patches import Polygon                                   # noqa: E402
from scipy.spatial import ConvexHull                                     # noqa: E402

import npj_style as S                                                    # noqa: E402

SRC = "downstream/npj/out/figure4/subgroup_rates.csv"
OUT = "downstream/npj/out/figure4"

W_MM, H_MM = 88.0, 64.0
X_AX0, AX_W = 13.0, 60.0
Y_AX0, AX_H = 11.0, 38.5
Y_LEGEND, Y_NOTE = 55.5, 51.5
XLIM, YLIM = (-1.0, 29.0), (60.0, 103.0)   # the safety-net point of the AUD diagnosis is at 62.5%
XTICKS, YTICKS = [0, 10, 20], [70, 80, 90, 100]
MIN_N = 20
# The strata whose position is discussed, and where their label goes relative to the point.
CALL = {("code", "Safety net"): (2.2, -0.4, "left"),
        ("code", "65 and over"): (2.2, -0.4, "left"),
        ("code", "Most deprived"): (-2.2, 0.0, "right"),
        ("code", "Under 50"): (-2.2, 0.0, "right")}
READERS = [("code", "AUD diagnosis", S.C_CODE),
           ("llm", "MedDx", S.C_INDEX)]
# Colour names the reader and shape names the stratification, so a point can be placed
# without a label on every one of the 36.
SHAPES = {"Hospital system": "o", "Sex": "s", "Race": "^", "Age": "D",
          "Residence": "v", "Neighbourhood deprivation": "P"}
SHORT = {"Hospital system": "System", "Sex": "Sex", "Race": "Race",
         "Age": "Age", "Residence": "Residence",
         "Neighbourhood deprivation": "Deprivation"}


def load(src=SRC):
    d = pd.read_csv(src)
    out = []
    for tag, label, col in READERS:
        x = 100.0 - d["spec_" + tag].to_numpy(dtype=float)
        y = d["sens_" + tag].to_numpy(dtype=float)
        small = ((d["n_sens"] < MIN_N) | (d["n_spec"] < MIN_N)).to_numpy()
        out.append({"tag": tag, "label": label, "col": col, "x": x, "y": y,
                    "small": small, "level": d["level"].to_numpy(),
                    "axis": d["axis"].to_numpy()})
    return d, out


def draw(ax, readers):
    # a hairline joins the two readers' points of the same stratum, so the pairs read
    code, llm = readers[0], readers[1]
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("pair", [S.C_CODE, S.C_INDEX])
    for x0, y0, x1, y1 in zip(code["x"], code["y"], llm["x"], llm["y"]):
        if np.isfinite([x0, y0, x1, y1]).all():
            n = 24
            xs = np.linspace(x0, x1, n + 1); ys = np.linspace(y0, y1, n + 1)
            segs = [((xs[i], ys[i]), (xs[i + 1], ys[i + 1])) for i in range(n)]
            lc = LineCollection(segs, colors=cmap(np.linspace(0, 1, n)), linewidths=1.1, alpha=0.24, zorder=2)
            ax.add_collection(lc)
    for r in readers:
        pts = np.column_stack([r["x"], r["y"]])
        ok = np.isfinite(pts).all(axis=1)
        pts = pts[ok]
        if len(pts) >= 3:
            h = ConvexHull(pts)
            ax.add_patch(Polygon(pts[h.vertices], closed=True, facecolor=r["col"],
                                 edgecolor=r["col"], lw=S.LW_MIN, alpha=0.13,
                                 zorder=1))
        for xx, yy, sm, axis in zip(r["x"], r["y"], r["small"], r["axis"]):
            if not (np.isfinite(xx) and np.isfinite(yy)):
                continue
            ax.plot([xx], [yy], marker=SHAPES[axis], ms=2.9,
                    mfc="white" if sm else r["col"], mec=r["col"],
                    mew=S.LW_MIN * 1.4, linestyle="none", zorder=3)
        for xx, yy, lvl in zip(r["x"], r["y"], r["level"]):
            key = (r["tag"], lvl)
            if key not in CALL:
                continue
            dx, dy, ha = CALL[key]
            ax.annotate(lvl, xy=(xx + dx, yy + dy), ha=ha, va="center",
                        fontsize=S.FS_SMALL, color=S.INK, zorder=4)

    # three regions a reader can land in, named so the plane reads without the legend
    from matplotlib.patches import Rectangle as _R
    ax.add_patch(_R((XLIM[0], 95.0), 10.0 - XLIM[0], XLIM[1] - 95.0, facecolor="#F3F2EF", edgecolor="none", zorder=0))
    ax.text(XLIM[0] + 0.4, 101.0, "finds nearly all, few false positives", fontsize=S.FS_SMALL - 1.0, color=S.MUTED, ha="left", va="bottom", zorder=5)
    ax.text(XLIM[1] - 0.4, 95.4, "finds nearly all, more false positives", fontsize=S.FS_SMALL - 1.0, color=S.MUTED, ha="right", va="bottom", zorder=5)
    ax.text(XLIM[0] + 0.4, YLIM[0] + 0.6, "misses patients", fontsize=S.FS_SMALL - 0.5, color=S.MUTED, ha="left", va="bottom", zorder=5)
    ax.axhline(95.0, color=S.HAIR, lw=S.LW_MIN, ls=(0, (2, 2)), zorder=0)
    ax.axvline(10.0, ymin=(95.0 - YLIM[0]) / (YLIM[1] - YLIM[0]), color=S.HAIR, lw=S.LW_MIN, ls=(0, (2, 2)), zorder=0)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks(XTICKS)
    ax.set_yticks(YTICKS)
    ax.set_xlabel("False positive rate (%)", fontsize=S.FS_BODY, color=S.INK, labelpad=4)
    ax.set_ylabel("Sensitivity (%)", fontsize=S.FS_BODY, color=S.INK, labelpad=3)
    ax.tick_params(axis="both", pad=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return ax


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--letter", default="b")
    ap.add_argument("--letter-visible", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not Path(SRC).exists():
        raise SystemExit("%s is missing; run f04_subgroup_rates.py first" % SRC)
    d, readers = load()

    for r in readers:
        pts = np.column_stack([r["x"], r["y"]])
        h = ConvexHull(pts[np.isfinite(pts).all(axis=1)])
        print("  %-19s false positive rate %.1f to %.1f, sensitivity %.1f to %.1f, "
              "hull area %.1f" % (r["label"], np.nanmin(r["x"]), np.nanmax(r["x"]),
                                  np.nanmin(r["y"]), np.nanmax(r["y"]), h.volume))
    t = pd.DataFrame({"axis": readers[0]["axis"], "level": readers[0]["level"],
                      "code_fpr": readers[0]["x"], "code_sens": readers[0]["y"],
                      "llm_fpr": readers[1]["x"], "llm_sens": readers[1]["y"],
                      "under_20": readers[0]["small"]})
    t.to_csv(os.path.join(a.out, "panel_roc_space.csv"), index=False)

    S.apply_npj_style()
    fig = plt.figure(figsize=S.mm(W_MM, H_MM))
    fig.set_layout_engine("none")
    ax = fig.add_axes([X_AX0 / W_MM, Y_AX0 / H_MM, AX_W / W_MM, AX_H / H_MM])
    fig.canvas.draw()
    draw(ax, readers)

    readers = [Line2D([], [], marker="o", ms=2.9, linestyle="none", color=c, label=lab)
               for _, lab, c in READERS]
    leg1 = fig.legend(handles=readers, loc="upper left",
                      bbox_to_anchor=(1.5 / W_MM, (Y_LEGEND + 2.4) / H_MM),
                      ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.6,
                      fontsize=S.FS_SMALL, borderpad=0.0, handlelength=0.9)
    for txt in leg1.get_texts():
        txt.set_color(S.INK)
    fig.add_artist(leg1)
    axes = [Line2D([], [], marker=m, ms=2.9, linestyle="none", mfc="white",
                   mec=S.INK, mew=S.LW_MIN * 1.4, label=SHORT[k])
            for k, m in SHAPES.items()]
    leg2 = fig.legend(handles=axes, loc="upper left",
                      bbox_to_anchor=(1.5 / W_MM, (Y_NOTE + 2.2) / H_MM),
                      ncol=6, frameon=False, handletextpad=0.25, columnspacing=0.7,
                      fontsize=S.FS_SMALL, borderpad=0.0, handlelength=0.7)
    for txt in leg2.get_texts():
        txt.set_color(S.INK)

    fig.text(7.0 / W_MM, 62.0 / H_MM, "MedDx stays where few patients are missed", fontsize=S.FS_BODY, fontweight="bold", color=S.INK, ha="left", va="center")
    S.panel_letter(fig, a.letter, a.letter_visible)
    S.check_text_fits(fig, W_MM, H_MM)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "panel_roc_space." + ext), dpi=600,
                    bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    print("wrote %s/panel_roc_space.pdf and .png at %.1f by %.1f mm"
          % (a.out, W_MM, H_MM))


if __name__ == "__main__":
    main()
