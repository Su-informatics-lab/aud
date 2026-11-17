#!/usr/bin/env python3
"""Fig. 4b: agreement with the ground truth for the 11 engines and for the AUD diagnosis.
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

MODELS = "downstream/npj/out/figure4/panel_a_leaderboard.csv"
CODE = "downstream/npj/out/figure4/diagnosis_code.csv"
OUT = "downstream/npj/out/figure3"

W_MM, H_MM = 90.0, 64.0
X_AX0, AX_W = 28.0, 52.0
Y_AX0, AX_H = 10.5, 46.5
GAP = 0.9
PRIMARY = "Qwen3.6-27B"


def load(models=MODELS, code=CODE):
    d = pd.read_csv(models).sort_values("kappa_w", ascending=False)
    PARAMS = {"DeepSeek-V4-Flash": "DeepSeek-V4-Flash (304B)", "GLM-5.2": "GLM-5.2 (753B)"}
    rows = [{"label": (r["name"] + " (primary)" if r["name"] == PRIMARY else PARAMS.get(r["name"], r["name"])), "kappa": float(r["kappa_w"]), "lo": float(r["ci_low"]),
             "hi": float(r["ci_high"]), "n": int(r["n"]), "kind": "model"}
            for _, r in d.iterrows()]
    c = pd.read_csv(code).iloc[0]
    rows.append({"label": "AUD diagnosis", "kappa": float(c["kappa_w"]),
                 "lo": float(c["ci_low"]), "hi": float(c["ci_high"]),
                 "n": int(c["n"]), "kind": "code"})
    return rows


def draw(ax, rows):
    """Eleven model rows, then a rule, then the comparator."""
    n_model = sum(1 for r in rows if r["kind"] == "model")
    pos, y = {}, 0.0
    for k in range(len(rows) - 1, -1, -1):                  # bottom row first
        if k == n_model - 1:                                # the gap holding the rule
            y += GAP
        pos[k] = y
        y += 1.0
    y_rule = pos[n_model - 1] - (1.0 + GAP) / 2.0
    ax.axhline(y_rule, color=S.RULE, lw=S.LW_MIN, zorder=1)
    # the code is the comparator: its estimate and interval run through every model row
    c = next(r for r in rows if r["kind"] == "code")
    ax.axvspan(c["lo"], c["hi"], color=S.TINT[S.C_CODE], lw=0, zorder=0)
    ax.axvline(c["kappa"], color=S.C_CODE, lw=S.LW_MIN, zorder=1)

    RECOMMEND = 3                                            # the top three engines are recommended
    NOT_RECOMMENDED_FROM = "GPT-OSS-20B"                     # this engine and every one below it: pale, negative results
    pale_from = next(k for k, r in enumerate(rows) if r["label"].startswith(NOT_RECOMMENDED_FROM))
    for k, r in enumerate(rows):
        yy = pos[k]
        col = (S.C_INDEX if r["label"].startswith(PRIMARY)
               else S.C_CODE if r["kind"] == "code" else ("#E3B3C9" if k >= pale_from else "#B5366E"))
        if r["kind"] == "model" and k < RECOMMEND:
            ax.annotate("*", xy=(0.0, yy), xytext=(-2.5, 0), textcoords="offset points", xycoords=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=S.FS_SMALL, color=S.INK, annotation_clip=False)
        ax.errorbar(r["kappa"], yy,
                    xerr=[[r["kappa"] - r["lo"]], [r["hi"] - r["kappa"]]],
                    fmt="o", ms=3.4, lw=S.LW_MIN, capsize=1.4, color=col,
                    mfc=col, mec=col, mew=S.LW_MIN, zorder=3)
        ax.annotate("%.3f" % r["kappa"], xy=(1.0, yy), xytext=(2.0, 0),
                    textcoords="offset points", xycoords=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=S.FS_SMALL, color=S.INK,
                    annotation_clip=False)

    ax.set_yticks([pos[k] for k in range(len(rows))])
    ax.set_yticklabels([r["label"] for r in rows], fontsize=S.FS_SMALL, color=S.INK)
    ax.annotate("* recommended engines; pale, not recommended (negative results)", xy=(-0.52, -1.25), xycoords=("axes fraction", "data"), fontsize=S.FS_SMALL - 0.5, color=S.MUTED, va="center", ha="left", annotation_clip=False)
    for t_, r in zip(ax.get_yticklabels(), rows):
        if r["label"].startswith(PRIMARY):
            t_.set_fontweight("bold")
    ax.set_ylim(-1.7, y - 0.3)
    ax.set_xlim(0.42, 1.00)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("Three-class Cohen's $\\kappa$ with the ground truth (95% CI)",
                  fontsize=S.FS_BODY, color=S.INK, labelpad=4)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", pad=2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    return ax


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--letter", default="a")
    ap.add_argument("--letter-visible", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in (MODELS, CODE):
        if not Path(f).exists():
            raise SystemExit("%s is missing" % f)
    rows = load()

    t = pd.DataFrame(rows)[["label", "kind", "kappa", "lo", "hi", "n"]]
    t.columns = ["reader", "kind", "kappa", "ci_low", "ci_high", "n_kappa"]
    t.to_csv(os.path.join(a.out, "panel_model_leaderboard.csv"), index=False)
    print(t.round(4).to_string(index=False))

    code = next(r for r in rows if r["kind"] == "code")
    models = [r for r in rows if r["kind"] == "model"]
    print("denominators %d to %d" % (min(r["n"] for r in models),
                                     max(r["n"] for r in models)))
    print("models above the code's point estimate of %.3f: %d of %d"
          % (code["kappa"], sum(1 for r in models if r["kappa"] > code["kappa"]),
             len(models)))
    for thr in (0.68, 0.85):
        print("models above %.2f: %d" % (thr, sum(1 for r in models
                                                  if r["kappa"] > thr)))
    print("models whose lower bound clears the code's upper bound of %.3f: %d"
          % (code["hi"], sum(1 for r in models if r["lo"] > code["hi"])))

    S.apply_npj_style()
    fig = plt.figure(figsize=S.mm(W_MM, H_MM))
    fig.set_layout_engine("none")
    ax = fig.add_axes([X_AX0 / W_MM, Y_AX0 / H_MM, AX_W / W_MM, AX_H / H_MM])
    fig.canvas.draw()
    draw(ax, rows)
    fig.text(7.0 / W_MM, 61.5 / H_MM, "Agreement with the ground truth: 11 engines and the AUD diagnosis", fontsize=S.FS_BODY, fontweight="bold", va="center", color=S.INK)
    S.panel_letter(fig, a.letter, a.letter_visible)
    S.check_text_fits(fig, W_MM, H_MM)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "panel_model_leaderboard." + ext), dpi=600,
                    bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    print("wrote %s/panel_model_leaderboard.pdf and .png at %.1f by %.1f mm"
          % (a.out, W_MM, H_MM))


if __name__ == "__main__":
    main()
