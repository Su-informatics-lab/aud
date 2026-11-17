#!/usr/bin/env python3
"""Fig. 1e: whether MedDx's classification matched the ground truth against whether the comparator's did.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "downstream")
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.patches import Rectangle                                # noqa: E402

import npj_style as S                                                   # noqa: E402
import panels_de as DE                                                  # noqa: E402

OUT = "downstream/npj/out/figure1"
W_MM, H_MM = 40.0, 70.0

# With rotation_mode anchor and va bottom the rotated title occupies [X - 2.5, X],
# so X is its right edge. At 1.8 it ran off the left of the panel.
X_YTITLE, X_YLAB = 3.4, 14.0
X_P0, CELL = 15.5, 9.0
# n sits rotated at the right of each matrix. It is a denominator rather than a result, and
# under the comparator name it took a line of height and read with the same weight as the
# counts. X_NLAB is its right edge, as X_YTITLE is for the rotated axis title.
X_NLAB = 38.5
TOP_PLOT = (37.5, 55.5)
BOT_PLOT = (5.0, 23.0)
T1_OFF = 2.6
# The rule between the two blocks. Its clearance above the column labels of the upper
# block and below the name of the lower one is about a millimetre at this panel size, so
# both are measured at the end of the draw rather than assumed.
Y_RULE = 31.5

# The comparator, the name printed over its matrix, and the colour its own cell carries.
PANES = [(DE.CODE, "AUD diagnosis", S.C_CODE),
         (DE.FELLOW, "Hepatologist A", S.C_REF)]
OUTPUTS = {"AUD diagnosis": "present / absent", "Hepatologist A": "yes / no"}
ROWS, COLS = ["Correct", "Wrong"], ["Correct", "Wrong"]


def _text(ax, x, y, s, size, colour, ha, va="center", rotation=0):
    # rotation is passed at creation. Setting it after the artist exists leaves the
    # extent below measured on the unrotated text, which reported a false overflow for
    # the rotated axis title.
    t = ax.text(x, y, s, fontsize=size, color=colour, ha=ha, va=va, zorder=5,
                rotation=rotation, rotation_mode="anchor")
    fig = ax.figure
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    x0, x1 = bb.x0 / fig.dpi * 25.4, bb.x1 / fig.dpi * 25.4
    if x0 < -0.2 or x1 > W_MM + 0.2:
        print("  WARNING text runs outside the panel: %.1f to %.1f mm  %r" % (x0, x1, s))
    return t


def _mm(ax, t):
    bb = t.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    return bb.y0 / ax.figure.dpi * 25.4, bb.y1 / ax.figure.dpi * 25.4


def draw_panel_e(ax, disc):
    ax.set_xlim(0, W_MM); ax.set_ylim(0, H_MM); ax.set_aspect("equal"); ax.axis("off")
    spans = []

    for (other, name, own), (y0, y1) in zip(PANES, (TOP_PLOT, BOT_PLOT)):
        d = disc[other]
        mat = d["mat"]
        spans.append(("the name %r" % name,
                      _mm(ax, _text(ax, X_P0 + CELL, y1 + T1_OFF + 2.7, name, S.FS_BODY,
                                    S.INK, "center", va="baseline"))))
        spans.append(("the outputs of %r" % name,
                      _mm(ax, _text(ax, X_P0 + CELL, y1 + 1.0, OUTPUTS[name], S.FS_SMALL,
                                    S.MUTED, "center", va="baseline"))))
        _text(ax, X_NLAB, (y0 + y1) / 2, "n = %d" % d["n"], S.FS_SMALL, S.MUTED,
              "center", va="bottom", rotation=90)

        for i in range(2):                       # 0 LLM correct, 1 LLM wrong
            for j in range(2):                   # 0 comparator correct, 1 wrong
                x = X_P0 + j * CELL
                y = y1 - (i + 1) * CELL
                fill, ink, bold = "white", S.INK, False
                if i == 0 and j == 1:            # MedDx alone was right
                    ink, bold = S.C_INDEX, True
                elif i == 1 and j == 0:          # the comparator alone was right
                    ink, bold = own, True
                ax.add_patch(Rectangle((x, y), CELL, CELL, fc=fill, ec="none", zorder=1))
                t_ = _text(ax, x + CELL / 2, y + CELL / 2, "%d" % mat[i, j], S.FS_BODY,
                           ink, "center")
                if bold:
                    t_.set_fontweight("bold")

        ax.plot([X_P0, X_P0 + 2 * CELL], [y1 - CELL] * 2, color=S.HAIR,
                lw=S.LW_MIN, zorder=2)
        ax.plot([X_P0 + CELL] * 2, [y0, y1], color=S.HAIR, lw=S.LW_MIN, zorder=2)
        ax.add_patch(Rectangle((X_P0, y0), 2 * CELL, 2 * CELL, fc="none", ec=S.RULE,
                               lw=S.LW_MIN, zorder=4))

        for i, lab in enumerate(ROWS):
            _text(ax, X_YLAB, y1 - (i + 0.5) * CELL, lab, S.FS_BODY, S.INK, "right")
        _text(ax, X_YTITLE, (y0 + y1) / 2, "MedDx", S.FS_BODY, S.INK, "center",
              va="bottom", rotation=90)
        for j, lab in enumerate(COLS):
            spans.append(("the column label %r under %r" % (lab, name),
                          _mm(ax, _text(ax, X_P0 + (j + 0.5) * CELL, y0 - 1.6, lab,
                                        S.FS_BODY, S.INK, "center", va="top"))))

    ax.plot([1.0, W_MM - 1.0], [Y_RULE] * 2, color=S.RULE, lw=S.LW_MIN,
            ls=(0, (2.4, 2.0)), zorder=2)
    for what, (lo, hi) in spans:
        if lo - 0.2 < Y_RULE < hi + 0.2:
            print("  WARNING the rule at %.1f mm crosses %s" % (Y_RULE, what))
        if lo < -0.2:
            print("  WARNING %s runs off the foot of the panel: %.1f mm" % (what, lo))
    return ax


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--letter", default="e")
    ap.add_argument("--letter-visible", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    frame = DE.load()
    disc = DE.discordance(frame)

    rows = []
    for other, name, _ in PANES:
        d = disc[other]
        m = d["mat"]
        rows.append({"comparator": name, "n": d["n"],
                     "both_correct": int(m[0, 0]), "llm_only": int(m[0, 1]),
                     "comparator_only": int(m[1, 0]), "both_wrong": int(m[1, 1]),
                     "discordant": d["disagree"], "test": d["test"]["method"],
                     "statistic": round(d["test"]["statistic"], 2),
                     "p_value": d["test"]["p_value"]})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.out, "panel_e_discordance.csv"), index=False)
    print(t.to_string(index=False))

    S.apply_npj_style()
    _TITLE = True
    fig = plt.figure(figsize=S.mm(W_MM, H_MM))
    fig.set_layout_engine("none")
    fig.text(S.LETTER_BOX_W / W_MM, 67.5 / H_MM, "MedDx right in 52 of 57", fontsize=S.FS_BODY, fontweight="bold", color=S.INK, ha="left", va="center")
    fig.text(S.LETTER_BOX_W / W_MM, 63.6 / H_MM, "disagreements with the diagnosis", fontsize=S.FS_SMALL, color=S.MUTED, ha="left", va="center")
    ax = fig.add_axes([0, 0, 1, 1])
    fig.canvas.draw()
    draw_panel_e(ax, disc)
    S.panel_letter(fig, a.letter, a.letter_visible)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "panel_e." + ext), dpi=600,
                    bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    print("wrote %s/panel_e.pdf and .png at %.1f by %.1f mm" % (a.out, W_MM, H_MM))


if __name__ == "__main__":
    main()
