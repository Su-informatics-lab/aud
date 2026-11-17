#!/usr/bin/env python3
"""Fig. 1d: sensitivity and specificity of MedDx and of the AUD diagnosis, in the study cohort and weighted to the registry.
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
from matplotlib.legend_handler import HandlerTuple                      # noqa: E402
from matplotlib.lines import Line2D                                     # noqa: E402

import npj_style as S                                                   # noqa: E402

SRC = "downstream/npj/out/f01_accuracy/reader_estimates.csv"
OUT = "downstream/npj/out/figure1"

W_MM, H_MM = 112.0, 64.0
# Measured at 8 pt in Liberation Sans: the widest row label is 25.4 mm and the widest
# numeric string is 28.7 mm. Everything below follows from those two.
X_AX0, AX_W = 27.5, 46.0
X_NUM = X_AX0 + AX_W + 2.0
Y_AX0, AX_H = 8.5, 41.0
# The legend sits at the top, at the same height as in panels f and g, and the weighted
# Youden line sits under it. The line was above the legend until 24 August, which made it
# read as a title for the panel rather than as a note on the values in it, and put a
# running sentence where every other panel of this figure carries none.
Y_LEGEND, Y_SENT = 57.3, 52.3

MODEL, CODE = "M", "I"
NAME = {MODEL: "MedDx", CODE: "AUD diagnosis"}
COLOUR = {MODEL: S.C_INDEX, CODE: S.C_CODE}
MARK = {MODEL: "o", CODE: "s"}
# The code has no call of its own. In the sensitivity block what is scored is the presence
# of a code among reference-positive patients, and in the specificity block the absence of
# one among reference-negative patients. The row says which.
CODE_ROW = {"sensitivity": "AUD diagnosis", "specificity": "AUD diagnosis"}
MINUS = "−"


def load(src=SRC):
    d = pd.read_csv(src)
    d = d[d["reader"].isin([MODEL, CODE])].copy()
    d["weighted"] = d["weighted"].astype(str).str.lower().eq("true")
    return d


def _num(v):
    return ("%.0f" % (100 * v)).replace("-", MINUS)


def _signed(v):
    return ("%+.0f" % (100 * v)).replace("-", MINUS)


def _fit(fig, artist, name):
    bb = artist.get_window_extent(renderer=fig.canvas.get_renderer())
    x0, x1 = bb.x0 / fig.dpi * 25.4, bb.x1 / fig.dpi * 25.4
    if x0 < -0.2 or x1 > W_MM + 0.2:
        print("  WARNING %s runs outside the panel: %.1f to %.1f mm" % (name, x0, x1))


def draw_panel_d(ax, d):
    """Two metrics; within each, MedDx then the AUD diagnosis. Filled marker with its Wilson 95% CI is the study
    cohort, open marker with its bootstrap 95% CI the registry estimate. The MedDx-minus-diagnosis gap is printed
    under each metric pair, for the cohort and for the registry."""
    rows, y = [], 0.0
    jv = {}
    marks = []
    gaps = []
    for met in ("specificity", "sensitivity"):              # bottom-up
        vals = {}
        for who in (CODE, MODEL):
            sub = d[(d["reader"] == who) & (d["metric"] == met)]
            s_ = sub[~sub["weighted"]].iloc[0]
            w = sub[sub["weighted"]].iloc[0]
            jv.setdefault(who, {})[met] = float(w["value"])
            vals[who] = (float(s_["value"]), float(w["value"]))
            rows.append((y, who, met, float(s_["value"]), float(s_["lo"]), float(s_["hi"]),
                         float(w["value"]), float(w["lo"]), float(w["hi"])))
            y += 1.0
        marks.append((y - 2.0, met))
        gaps.append((y - 2.0, met, vals[MODEL][0] - vals[CODE][0], vals[MODEL][1] - vals[CODE][1]))
        y += 1.1

    for yy, who, met, sv, slo, shi, wv, lo, hi in rows:
        col = COLOUR[who]
        ax.plot([slo, shi], [yy + 0.18] * 2, color=col, lw=S.LW_MIN, solid_capstyle="butt", zorder=2)
        for e in (slo, shi):
            ax.plot([e] * 2, [yy + 0.18 - 0.12, yy + 0.18 + 0.12], color=col, lw=S.LW_MIN, zorder=2)
        ax.plot(sv, yy + 0.18, MARK[who], ms=3.4, mfc=col, mec=col, mew=S.LW_MIN, zorder=3)
        ax.errorbar(wv, yy - 0.18, xerr=[[wv - lo], [hi - wv]], fmt=MARK[who], ms=3.8,
                    lw=S.LW_MIN, capsize=1.4, color=col, mfc="white", mec=col,
                    mew=S.LW_MIN, zorder=4)
        txt = "%s   %s" % (_num(sv), _num(wv))
        t = ax.annotate(txt, xy=(1.0, yy), xytext=(2.0, 0), textcoords="offset points",
                        xycoords=ax.get_yaxis_transform(), ha="left", va="center",
                        fontsize=S.FS_SMALL, color=S.INK, annotation_clip=False)
        _fit(ax.figure, t, "numeric label %r" % txt)
    # column heads for the numbers
    t = ax.annotate("cohort   registry", xy=(1.0, y - 0.55), xytext=(2.0, 0), textcoords="offset points",
                    xycoords=ax.get_yaxis_transform(), ha="left", va="center", fontsize=S.FS_SMALL,
                    color=S.MUTED, annotation_clip=False)
    for ypos, met, g_s, g_w in gaps:
        txt = "MedDx \u2212 diagnosis: %s cohort, %s registry (points)" % (_signed(g_s), _signed(g_w))
        ax.text(0.556, ypos - 0.72, txt, fontsize=S.FS_SMALL, color=S.MUTED, va="center", ha="left")

    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([CODE_ROW[r[2]] if r[1] == CODE else NAME[MODEL] for r in rows],
                       fontsize=S.FS_BODY, color=S.INK)
    ax.set_ylim(-1.3, y - 0.2)
    ax.set_xlim(0.55, 1.01)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_xticklabels(["60", "70", "80", "90", "100"])
    ax.set_xlabel("Value against the ground truth (%)", fontsize=S.FS_BODY,
                  color=S.INK, labelpad=4)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", pad=2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    for ypos, met in marks:
        ax.text(0.556, ypos + 1.52, met.capitalize(), fontsize=S.FS_BODY,
                fontweight="bold", color=S.INK, va="center", ha="left")
    return jv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--letter", default="d")
    ap.add_argument("--letter-visible", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not Path(a.src).exists():
        raise SystemExit("run panels_de.py first; %s is missing" % a.src)
    d = load(a.src)

    S.apply_npj_style()
    fig = plt.figure(figsize=S.mm(W_MM, H_MM))
    fig.set_layout_engine("none")
    fig.text(S.LETTER_BOX_W / W_MM, 62.0 / H_MM, "MedDx is more sensitive than the AUD diagnosis, in cohort and registry", fontsize=S.FS_BODY, fontweight="bold", color=S.INK, ha="left", va="center")
    ax = fig.add_axes([X_AX0 / W_MM, Y_AX0 / H_MM, AX_W / W_MM, AX_H / H_MM])
    fig.canvas.draw()
    jv = draw_panel_d(ax, d)

    S.panel_letter(fig, a.letter, a.letter_visible)

    # The two states are a property of the marker and not of a reader, so each legend
    # handle carries both readers' shapes. Neutral would have been a third colour; the
    # readers' own colours keep the vocabulary closed.
    fill_h = tuple(Line2D([], [], ls="none", marker=MARK[k], ms=4.0, mfc=COLOUR[k],
                          mec=COLOUR[k], mew=S.LW_MIN) for k in (MODEL, CODE))
    open_h = tuple(Line2D([], [], ls="none", marker=MARK[k], ms=4.2, mfc="white",
                          mec=COLOUR[k], mew=S.LW_MIN) for k in (MODEL, CODE))
    leg = fig.legend([fill_h, open_h],
                     ["Study cohort, Wilson 95% CI", "Registry estimate, bootstrap 95% CI"],
                     handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
                     loc="upper left",
                     bbox_to_anchor=(S.LETTER_BOX_W / W_MM, Y_LEGEND / H_MM),
                     ncol=2, frameon=False, fontsize=S.FS_BODY, handlelength=1.8,
                     handletextpad=0.6, columnspacing=2.2, borderpad=0.0,
                     borderaxespad=0.0)
    for t_ in leg.get_texts():
        t_.set_color(S.INK)

    j = {k: jv[k]["sensitivity"] + jv[k]["specificity"] - 1 for k in jv}
    t = fig.text(S.LETTER_BOX_W / W_MM, Y_SENT / H_MM,
                 "Registry estimate, Youden's J: MedDx %.2f, AUD diagnosis %.2f"
                 % (j[MODEL], j[CODE]),
                 fontsize=S.FS_SMALL, color=S.MUTED, ha="left", va="baseline")
    _fit(fig, t, "sentence")

    print("%-20s %-12s %8s %8s %10s" % ("reader", "metric", "sample", "weighted", "95% CI"))
    for _, r in d.iterrows():
        if r["weighted"]:
            continue
        w = d[(d["reader"] == r["reader"]) & (d["metric"] == r["metric"])
              & d["weighted"]].iloc[0]
        print("%-20s %-12s %8.3f %8.3f   %.3f to %.3f"
              % (NAME[r["reader"]], r["metric"], r["value"], w["value"], w["lo"], w["hi"]))
    print("Youden's J weighted: LLM %.3f, code %.3f" % (j[MODEL], j[CODE]))

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "panel_d." + ext), dpi=600,
                    bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    print("wrote %s/panel_d.pdf and .png at %.1f by %.1f mm" % (a.out, W_MM, H_MM))


if __name__ == "__main__":
    main()
