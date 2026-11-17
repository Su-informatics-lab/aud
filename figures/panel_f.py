#!/usr/bin/env python3
"""Fig. 1f: positive and negative predictive value against prevalence, from the registry-weighted rates of positive and
negative classification (registry/predictive_values.py). At the registry prevalence the curves give the registry values.
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
from matplotlib.lines import Line2D                                     # noqa: E402

import npj_style as S                                                   # noqa: E402
import panels_de as DE                                                  # noqa: E402

RATES = "output/registry/predictive_values.json"
PREV = "downstream/npj/out/f01_accuracy/registry_prevalence.csv"
OUT = "downstream/npj/out/figure1"

W_MM, H_MM = 52.0, 64.0
X_AX0, AX_W = 12.5, 37.5
Y_AX0, AX_H = 10.5, 38.0
Y_LEGEND = 57.5

READERS = [("M", "MedDx", S.C_INDEX), ("I", "AUD diagnosis", S.C_CODE)]
P_LO, P_HI = 0.01, 0.50


def load(rates=RATES, prev=PREV):
    import json
    j = json.load(open(rates))
    r = pd.DataFrame([{"reader": k, **{q: j["%s_%s" % (q, name)]["point"] for q in ("tpr", "fpr", "tnr", "fnr")}}
                      for k, name in (("M", "MedDx"), ("I", "diagnosis"))])
    v = pd.read_csv(prev).iloc[0]
    return r, float(v["value"]), float(v["lo"]), float(v["hi"])


def draw_panel_f(ax, r, q, lo, hi):
    ax.axvspan(lo, hi, color=S.HAIR, lw=0, zorder=0)
    for x in (DE.NSDUH_2024_ADULTS, q):
        ax.axvline(x, color=S.RULE, lw=S.LW_MIN, ls=(0, (2.4, 2.0)), zorder=1)

    p = np.linspace(P_LO, P_HI, 400)
    for key, _, col in READERS:
        row = r[r["reader"] == key].iloc[0]
        tpr, fpr = float(row["tpr"]), float(row["fpr"])
        tnr, fnr = float(row["tnr"]), float(row["fnr"])
        ax.plot(p, tpr * p / (tpr * p + fpr * (1 - p)), "-", color=col,
                lw=S.LW_MIN, zorder=3)
        ax.plot(p, tnr * (1 - p) / (tnr * (1 - p) + fnr * p), color=col,
                lw=S.LW_MIN, ls=(0, (2.6, 1.8)), zorder=3)

    ax.set_xlim(0.0, 0.52)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "10", "20", "30", "40", "50"]); ax.set_yticklabels(["0", "20", "40", "60", "80", "100"])
    ax.set_xlabel("Prevalence in the population read (%)", fontsize=S.FS_BODY,
                  color=S.INK, labelpad=4)
    ax.set_ylabel("Predictive value (PPV, NPV; %)", fontsize=S.FS_BODY, color=S.INK, labelpad=4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # The two populations that bound the range the paper cares about. The labels sit in
    # the lower part of the plot, which no curve enters at either prevalence: the lowest
    # curve there is the code positive predictive value, at 0.50 and 0.86 respectively.
    # The two values are in the caption and their positions are on the axis, so the
    # labels carry the population only. Carrying the value as well made the longer of the
    # two 18 mm tall, which brought its top within 0.25 mm of the code curve.
    for x, lab in ((DE.NSDUH_2024_ADULTS, "US adults"), (q, "ICR")):
        ax.text(x - 0.010, 0.03, lab, fontsize=S.FS_SMALL, color=S.MUTED, ha="right",
                va="bottom", rotation=90, zorder=5)
    return ax


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--letter", default="f")
    ap.add_argument("--letter-visible", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in (RATES, PREV):
        if not Path(f).exists():
            raise SystemExit("run registry/predictive_values.py and panels_de.py first; %s is missing" % f)
    r, q, lo, hi = load()

    print("registry prevalence %.4f (%.4f to %.4f)" % (q, lo, hi))
    rows = []
    for key, name, _ in READERS:
        row = r[r["reader"] == key].iloc[0]
        tpr, fpr = float(row["tpr"]), float(row["fpr"])
        tnr, fnr = float(row["tnr"]), float(row["fnr"])
        for pp, where in ((DE.NSDUH_2024_ADULTS, "US adults"),
                          (q, "ICR")):
            rows.append({"reader": name, "prevalence_at": where, "prevalence": pp,
                         "ppv": tpr * pp / (tpr * pp + fpr * (1 - pp)),
                         "npv": tnr * (1 - pp) / (tnr * (1 - pp) + fnr * pp)})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.out, "panel_f_predictive_value.csv"), index=False)
    print(t.to_string(index=False))

    S.apply_npj_style()
    fig = plt.figure(figsize=S.mm(W_MM, H_MM))
    fig.set_layout_engine("none")
    fig.text(S.LETTER_BOX_W / W_MM, 62.3 / H_MM, "Higher PPV and NPV at any prevalence", fontsize=S.FS_BODY, fontweight="bold", color=S.INK, ha="left", va="center")
    ax = fig.add_axes([X_AX0 / W_MM, Y_AX0 / H_MM, AX_W / W_MM, AX_H / H_MM])
    fig.canvas.draw()
    draw_panel_f(ax, r, q, lo, hi)
    S.panel_letter(fig, a.letter, a.letter_visible)

    handles = [Line2D([], [], color=c, lw=S.LW_MIN) for _, _, c in READERS]
    labels = [n for _, n, _ in READERS]
    handles += [Line2D([], [], color=S.C_FORE, lw=S.LW_MIN),
                Line2D([], [], color=S.C_FORE, lw=S.LW_MIN, ls=(0, (2.6, 1.8)))]
    labels += ["PPV", "NPV"]
    leg = fig.legend(handles, labels, loc="upper left",
                     bbox_to_anchor=(S.LETTER_BOX_W / W_MM, Y_LEGEND / H_MM),
                     ncol=2, frameon=False, fontsize=S.FS_SMALL, handlelength=1.5,
                     handletextpad=0.6, columnspacing=1.8, labelspacing=0.35,
                     borderpad=0.0, borderaxespad=0.0)
    for t_ in leg.get_texts():
        t_.set_color(S.INK)
    fig.canvas.draw()
    bb = leg.get_window_extent()
    if bb.x1 / fig.dpi * 25.4 > W_MM + 0.2:
        print("  WARNING legend runs outside the panel: right edge %.1f mm"
              % (bb.x1 / fig.dpi * 25.4))

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "panel_f." + ext), dpi=600,
                    bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    print("wrote %s/panel_f.pdf and .png at %.1f by %.1f mm" % (a.out, W_MM, H_MM))


if __name__ == "__main__":
    main()
