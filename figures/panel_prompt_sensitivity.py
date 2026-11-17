#!/usr/bin/env python3
"""Extended Data Fig. 5: agreement with the ground truth under the skill and four variations.
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

import npj_style as S                                                   # noqa: E402

COND = "downstream/npj/out/figure4/panel_d_conditions.csv"
OUT = "downstream/npj/out/figure3"

W_MM, H_MM = 57.0, 58.0
X_AX0, AX_W = 20.0, 27.0
Y_AX0, AX_H = 10.5, 36.5
GAP = 0.9
BASE = "v1_t0p0"

# The key in the producer, the row label, and which block the row belongs to.
ROWS = [(BASE, "Temperature 0", "temp"),
        ("v1_t0p3", "Temperature 0.3", "temp"),
        ("v1_t0p7", "Temperature 0.7", "temp"),
        ("paraphrase_a_t0p0", "Paraphrased skill A", "wording"),
        ("paraphrase_b_t0p0", "Paraphrased skill B", "wording")]
BLOCKS = [("Temperature", "temp"), ("Wording", "wording")]


def load(cond=COND):
    d = pd.read_csv(cond).set_index("condition")
    rows = []
    for key, label, block in ROWS:
        h = d.loc[key]
        rows.append({"key": key, "label": label, "block": block,
                     "kappa": float(h["kappa_w"]), "lo": float(h["ci_low"]),
                     "hi": float(h["ci_high"]), "n": int(h["n"]),
                     "d": float(h["paired_difference_from_baseline"]),
                     "d_lo": float(h["paired_ci_low"]),
                     "d_hi": float(h["paired_ci_high"]),
                     "changes": int(h["bucket_changes_from_baseline"]),
                     "ned": int(h["not_enough_data"]),
                     "uncertain": int(h["uncertain"])})
    stale = sorted(set(str(v) for v in d["status"]) - {"nan"})
    return rows, d.loc[BASE, "kappa_w"], stale


def draw(ax, rows, base):
    """Five rows in two blocks, the upper block on top, with a gap between them."""
    y, ys, heads = 0.0, [], []
    for _, block in reversed(BLOCKS):                       # bottom block first
        members = [k for k, r in enumerate(rows) if r["block"] == block]
        for k in reversed(members):                         # bottom row of the block first
            ys.append((k, y))
            y += 1.0
        heads.append((next(n for n, b in BLOCKS if b == block), y - 1.0))
        y += GAP
    pos = dict(ys)

    ax.axvline(base, color=S.RULE, lw=S.LW_MIN, ls=(0, (1.4, 1.6)), zorder=1)
    for k, r in enumerate(rows):
        yy = pos[k]
        ax.errorbar(r["kappa"], yy,
                    xerr=[[r["kappa"] - r["lo"]], [r["hi"] - r["kappa"]]],
                    fmt="o", ms=3.8, lw=S.LW_MIN, capsize=1.5, color=S.C_INDEX,
                    mfc=S.C_INDEX, mec=S.C_INDEX, mew=S.LW_MIN, zorder=3)
        ax.annotate("%.3f" % r["kappa"], xy=(1.0, yy), xytext=(2.0, 0),
                    textcoords="offset points", xycoords=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=S.FS_SMALL, color=S.INK,
                    annotation_clip=False)

    ax.set_yticks([pos[k] for k in range(len(rows))])
    ax.set_yticklabels([r["label"] for r in rows], fontsize=S.FS_SMALL, color=S.INK)
    ax.set_ylim(-0.75, y - GAP - 0.25)
    ax.set_xlim(0.84, 0.98)
    ax.set_xticks([0.85, 0.90, 0.95])
    ax.set_xlabel("Agreement with the ground truth, $\\kappa$", 
                  color=S.INK, labelpad=4, fontsize=6.5)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", pad=2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    for name, ytop in heads:
        ax.text(0.8425, ytop + 0.62, name, fontsize=S.FS_SMALL, fontweight="bold",
                color=S.INK, va="center", ha="left")
    return ax


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--letter", default="c")
    ap.add_argument("--letter-visible", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not Path(COND).exists():
        raise SystemExit("%s is missing" % COND)
    rows, base, stale = load()

    t = pd.DataFrame(rows)[["label", "block", "kappa", "lo", "hi", "n", "d", "d_lo",
                            "d_hi", "changes"]]
    t.columns = ["condition", "block", "kappa", "ci_low", "ci_high", "n_kappa",
                 "paired_difference", "paired_ci_low", "paired_ci_high",
                 "charts_changed"]
    t.to_csv(os.path.join(a.out, "panel_prompt_sensitivity.csv"), index=False)
    print(t.round(4).to_string(index=False))
    ks = [r["kappa"] for r in rows]
    print("agreement %.3f to %.3f; denominators %d to %d"
          % (min(ks), max(ks), min(r["n"] for r in rows), max(r["n"] for r in rows)))
    var = [r for r in rows if r["key"] != BASE]
    print("paired change against the frozen protocol:")
    for r in var:
        print("  %-14s %+.4f (%+.4f to %+.4f)%s  %d charts change bucket"
              % (r["label"], r["d"], r["d_lo"], r["d_hi"],
                 "" if r["d_lo"] <= 0 <= r["d_hi"] else "  DOES NOT COVER ZERO",
                 r["changes"]))
    print("largest paired change %s at %+.4f"
          % (max(var, key=lambda r: abs(r["d"]))["label"],
             max(var, key=lambda r: abs(r["d"]))["d"]))
    print("not enough data %d to %d charts; uncertain %d to %d"
          % (min(r["ned"] for r in rows), max(r["ned"] for r in rows),
             min(r["uncertain"] for r in rows), max(r["uncertain"] for r in rows)))
    if stale:
        print("  NOTE panel_d_conditions.csv still carries status %s from before the "
              "reference standard closed; it is not read by this panel" % stale)

    S.apply_npj_style()
    fig = plt.figure(figsize=S.mm(W_MM, H_MM))
    fig.set_layout_engine("none")
    ax = fig.add_axes([X_AX0 / W_MM, Y_AX0 / H_MM, AX_W / W_MM, AX_H / H_MM])
    fig.canvas.draw()
    draw(ax, rows, base)
    S.panel_letter(fig, a.letter, a.letter_visible)
    fig.text(1.5 / W_MM, 55.5 / H_MM, "Skill variations change agreement little", fontsize=S.FS_BODY, fontweight="bold", color=S.INK, ha="left", va="center")
    S.check_text_fits(fig, W_MM, H_MM)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "panel_prompt_sensitivity." + ext), dpi=600,
                    bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    print("wrote %s/panel_prompt_sensitivity.pdf and .png at %.1f by %.1f mm"
          % (a.out, W_MM, H_MM))


if __name__ == "__main__":
    main()
