#!/usr/bin/env python3
"""Fig. 4d: pairwise agreement between the 11 engines, with the ground truth.
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
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize        # noqa: E402

import npj_style as S                                                   # noqa: E402

KAPPA = "downstream/npj/out/figure4/panel_b_kappa_matrix.csv"
NMAT = "downstream/npj/out/figure4/panel_b_n_matrix.csv"
OUT = "downstream/npj/out/figure3"

W_MM, H_MM = 83.0, 58.0
X_AX0, AX_W = 27.0, 54.5
Y_AX0, AX_H = 10.0, 42.0
FS_CELL = 6.5
REF = "Reference standard"
PRIMARY = "Qwen3.6-27B"
DISPLAY = {PRIMARY: "Qwen3.6-27B", REF: "Ground truth"}
VMIN, VMAX = 0.42, 1.00




def load(kappa=KAPPA, nmat=NMAT):
    m = pd.read_csv(kappa, index_col=0)
    mods = [c for c in m.index if c != REF]
    order = [REF] + list(m.loc[REF, mods].astype(float)
                         .sort_values(ascending=False).index)
    m = m.loc[order, order].astype(float)
    n = pd.read_csv(nmat, index_col=0).loc[order, order]
    return m, n, order


def draw(ax, m, order):
    k = m.values.copy()
    np.fill_diagonal(k, np.nan)
    # A warm sequential ramp, to sit with the figure's own palette rather than against
    # it. magma is viridis's warm sibling: perceptually uniform and safe under the
    # common colour vision deficiencies. The ends are trimmed because the near-black
    # low end reads as a hole between white gridlines and the extreme high end loses
    # contrast against white text. Light is high in both maps, so nothing about how the
    # panel is read changes.
    cmap = ListedColormap(plt.get_cmap("magma")(np.linspace(0.12, 0.95, 256)))
    cmap.set_bad("white")
    norm = Normalize(VMIN, VMAX)
    q = ax.pcolormesh(np.arange(len(order) + 1), np.arange(len(order) + 1),
                      np.ma.masked_invalid(k[::-1]), cmap=cmap, norm=norm,
                      edgecolors="white", linewidth=0.5)

    # ".93" rather than "0.93": three characters instead of four in a 5 mm cell.
    texts = []
    for i in range(len(order)):
        for j in range(len(order)):
            if i == j:
                continue
            v = m.values[i, j]
            r, g, b, _ = cmap(norm(v))
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            texts.append(ax.text(j + 0.5, len(order) - i - 0.5,
                                 ("%.2f" % v).lstrip("0"),
                                 ha="center", va="center", fontsize=FS_CELL,
                                 color="white" if lum < 0.55 else S.INK, zorder=3))

    # The reference standard is the target, not a twelfth reader, so it is set off.
    top = len(order)
    ax.plot([0, top], [top - 1, top - 1], color=S.INK, lw=S.LW_MIN, zorder=4,
            solid_capstyle="butt")
    ax.plot([1, 1], [0, top], color=S.INK, lw=S.LW_MIN, zorder=4,
            solid_capstyle="butt")

    ax.set_yticks(np.arange(len(order)) + 0.5)
    ax.set_yticklabels([DISPLAY.get(n, n) for n in order[::-1]], fontsize=S.FS_SMALL, color=S.INK)
    for t_, name in zip(ax.get_yticklabels(), order[::-1]):
        if name == PRIMARY:
            t_.set_fontweight("bold")
    # Columns carry panel a's rank rather than a repeated name: R is the reference
    # standard, 1 to 11 are the models in the order panel a draws them. The axis label
    # names the statistic and the legend says what the column numbers stand for, which
    # keeps the ordering note out of the panel.
    ax.set_xticks(np.arange(len(order)) + 0.5)
    ax.set_xticklabels(["R"] + [str(i) for i in range(1, len(order))],
                       fontsize=S.FS_SMALL, color=S.INK)
    ax.set_xlabel("Pairwise agreement, $\\kappa$",
                  fontsize=S.FS_BODY, color=S.INK, labelpad=3)
    ax.set_xlim(0, len(order))
    ax.set_ylim(0, len(order))
    ax.tick_params(axis="both", length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return q, texts


def check_cells(fig, ax, texts, n):
    """Every number has to sit inside its own cell, which check_text_fits cannot see."""
    fig.canvas.draw()
    bb = ax.get_window_extent()
    cw, ch = bb.width / n, bb.height / n
    worst_w = worst_h = 1.0
    for t in texts:
        e = t.get_window_extent()
        worst_w = min(worst_w, 1.0 - e.width / cw)
        worst_h = min(worst_h, 1.0 - e.height / ch)
    print("  cell %.2f by %.2f mm; the tightest number leaves %.0f%% of the cell width "
          "and %.0f%% of its height" % (cw / fig.dpi * 25.4, ch / fig.dpi * 25.4,
                                        100 * worst_w, 100 * worst_h))
    if worst_w < 0 or worst_h < 0:
        print("  WARNING a number is wider or taller than its cell")
    return worst_w, worst_h


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--letter", default="b")
    ap.add_argument("--letter-visible", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in (KAPPA, NMAT):
        if not Path(f).exists():
            raise SystemExit("%s is missing" % f)
    m, n, order = load()
    mods = order[1:]

    m.round(4).to_csv(os.path.join(a.out, "panel_model_matrix.csv"))
    sub = m.loc[mods, mods].values
    pairs = sub[np.triu_indices(len(mods), 1)]
    vs_ref = m.loc[REF, mods].values
    print("order: %s" % " > ".join(mods))
    print("all %d models: median pair %.4f, median against the standard %.4f"
          % (len(mods), np.median(pairs), np.median(vs_ref)))
    for k in (3, 8):
        idx = mods[:k]
        s2 = m.loc[idx, idx].values[np.triu_indices(k, 1)]
        print("top %d: pairs %.3f to %.3f, median pair %.4f, median against the "
              "standard %.4f" % (k, s2.min(), s2.max(), np.median(s2),
                                 np.median(m.loc[REF, idx].values)))
    print("pair range %.3f to %.3f" % (pairs.min(), pairs.max()))
    print("DeepSeek-V4-Flash against the standard %.3f, against GPT-OSS-120B %.3f"
          % (m.loc[REF, "DeepSeek-V4-Flash"],
             m.loc["DeepSeek-V4-Flash", "GPT-OSS-120B"]))
    print("panel_b_n_matrix.csv reports %s for every cell, which is the count "
          "before Not Enough Data is dropped; the effective denominators are 425 to 445"
          % sorted(set(n.values.ravel().tolist())))

    S.apply_npj_style()
    fig = plt.figure(figsize=S.mm(W_MM, H_MM))
    fig.set_layout_engine("none")
    ax = fig.add_axes([X_AX0 / W_MM, Y_AX0 / H_MM, AX_W / W_MM, AX_H / H_MM])
    fig.canvas.draw()
    q, texts = draw(ax, m, order)
    check_cells(fig, ax, texts, len(order))

    fig.text(7.0 / W_MM, 56.0 / H_MM, "The strong engines agree with one another", fontsize=S.FS_BODY, fontweight="bold", color=S.INK, ha="left", va="center")
    S.panel_letter(fig, a.letter, a.letter_visible)
    S.check_text_fits(fig, W_MM, H_MM)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "panel_model_matrix." + ext), dpi=600,
                    bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    print("wrote %s/panel_model_matrix.pdf and .png at %.1f by %.1f mm"
          % (a.out, W_MM, H_MM))


if __name__ == "__main__":
    main()
