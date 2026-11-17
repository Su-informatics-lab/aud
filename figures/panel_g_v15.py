#!/usr/bin/env python3
"""Fig. 1g: every comparator on the same charts, in four groups. Values from the comparator table (build_supp_table6.py).
"""
import argparse, os, sys
import pandas as pd
sys.path.insert(0, "downstream"); sys.path.insert(0, "downstream/npj")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import npj_style as S
TABLE = "downstream/npj/out/supp_table6.csv"; OUT = "downstream/npj/out/figure1"
W_MM, H_MM = 86.0, 64.0
X_AX0, AX_W, GAP = 46.0, 18.0, 3.0
Y_AX0, AX_H = 8.0, 42.0
READS = {"notes": S.C_INDEX, "structured record": S.C_CODE}
GROUPS = [
    ("Training-free, reads the notes", [("LLM (Qwen3.6-27B)", "MedDx")]),
    ("Diagnosis-code rules, training-free", [("AUD or liver-disease code", "AUD or alcoholic liver disease"),
                                           ("Any alcohol code", "Any alcohol-related diagnosis"),
                                           ("ICD F10.x", "AUD diagnosis (F10.x)")]),
    ("Supervised on the structured record", [("Gradient boosting, with code", "XGBoost, with F10.x"),
                                             ("Gradient boosting, no code", "XGBoost, without F10.x")]),
    ("Supervised on the notes", [("Fine-tuned Bio_ClinicalBERT", "Bio_ClinicalBERT, fine-tuned"),
                                 ("TF-IDF logistic regression", "TF-IDF logistic regression")]),
]
SUB = {"Supervised on the structured record": "codes + demographics, comorbidities, utilisation"}

def load(table):
    d = pd.read_csv(table); d["short"] = d["short"].str.replace(r"\s+", " ", regex=True).str.strip()
    d["short"] = d["short"].str.replace("\\", "", regex=False)
    rows = []
    for g, members in GROUPS:
        rows.append({"kind": "header", "label": g})
        for key, lab in members:
            r = d[d["short"] == key]
            if len(r) != 1: raise SystemExit("missing comparator %r" % key)
            r = r.iloc[0].to_dict(); r.update(kind="row", label=lab); rows.append(r)
    return pd.DataFrame(rows)

def column(ax, d, met, title, show_labels, fig):
    n = len(d)
    for k, r in d.iterrows():
        y = n - 1 - k
        if r["kind"] == "header":
            continue
        col = READS[r["reads"]]; v = r[met] / 100.0; lo, hi = r[met + "_lo"] / 100.0, r[met + "_hi"] / 100.0
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", ms=3.0, lw=S.LW_MIN, capsize=1.4, color=col, mfc=col, mec=col, mew=S.LW_MIN, zorder=3)
    ax.set_yticks(range(n))
    labels = [("" if r["kind"] == "header" else r["label"]) for _, r in d.iterrows()][::-1]
    ax.set_yticklabels(labels if show_labels else [""] * n, fontsize=S.FS_BODY, color=S.INK)
    if show_labels:
        for t, name in zip(ax.get_yticklabels(), labels):
            if name == "MedDx": t.set_fontweight("bold")
        for k, r in d.iterrows():
            y = n - 1 - k
            if r["kind"] == "header":
                dy = 0.30 if r["label"] in SUB else 0.0
                ax.text(-0.02, y + dy, r["label"], transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=S.FS_SMALL, color=S.MUTED, style="italic", clip_on=False)
                if r["label"] in SUB:
                    ax.text(-0.02, y - 0.22, SUB[r["label"]], transform=ax.get_yaxis_transform(), ha="right", va="center",
                            fontsize=S.FS_SMALL - 1.0, color=S.MUTED, clip_on=False)
    ax.set_ylim(-0.7, n - 0.3); ax.set_xlim(0.65, 1.035)
    ax.set_xticks([0.7, 0.85, 1.0]); ax.set_xticklabels(["70", "85", "100"], fontsize=S.FS_SMALL - 1.0)
    ax.set_title(title, fontsize=S.FS_BODY, color=S.INK, pad=1, loc="center")
    ax.tick_params(axis="y", length=0, pad=3); ax.tick_params(axis="x", pad=2)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.patch.set_alpha(0.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--table", default=TABLE); ap.add_argument("--out", default=OUT)
    ap.add_argument("--letter", default="g"); ap.add_argument("--letter-visible", action="store_true"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); d = load(a.table)
    d[d["kind"] == "row"][["label", "reads", "youden", "sens", "sens_lo", "sens_hi", "spec", "spec_lo", "spec_hi", "n"]].to_csv(os.path.join(a.out, "panel_g_ladder.csv"), index=False)
    S.apply_npj_style(); fig = plt.figure(figsize=S.mm(W_MM, H_MM)); fig.set_layout_engine("none")
    fig.text(S.LETTER_BOX_W / W_MM, 63.0 / H_MM, "MedDx outperforms every comparator", fontsize=S.FS_BODY, fontweight="bold", color=S.INK, ha="left", va="center")
    ax1 = fig.add_axes([X_AX0 / W_MM, Y_AX0 / H_MM, AX_W / W_MM, AX_H / H_MM])
    ax2 = fig.add_axes([(X_AX0 + AX_W + GAP) / W_MM, Y_AX0 / H_MM, AX_W / W_MM, AX_H / H_MM])
    fig.canvas.draw(); column(ax1, d, "sens", "Sensitivity (%)", True, fig); column(ax2, d, "spec", "Specificity (%)", False, fig)
    # pale bands behind alternate groups, spanning the whole panel width
    from matplotlib.patches import Rectangle
    n = len(d); starts = [k for k, r in d.iterrows() if r["kind"] == "header"] + [n]
    for gi in range(len(starts) - 1):
        if gi % 2 == 1:
            y_top = n - 1 - starts[gi] + 0.5; y_bot = n - 1 - (starts[gi + 1] - 1) - 0.5
            f_top = fig.transFigure.inverted().transform(ax1.transData.transform((0.7, y_top)))[1]
            f_bot = fig.transFigure.inverted().transform(ax1.transData.transform((0.7, y_bot)))[1]
            fig.patches.append(Rectangle((0.0, f_bot), 1.0, f_top - f_bot, transform=fig.transFigure, facecolor="#F1F0ED", edgecolor="none", zorder=-5))
    fig.text((X_AX0 + AX_W + GAP / 2) / W_MM, 2.2 / H_MM, "Against the ground truth", ha="center", va="baseline", fontsize=S.FS_SMALL, color=S.INK)
    S.panel_letter(fig, a.letter, a.letter_visible)
    handles = [Patch(facecolor=READS["notes"], edgecolor="none"), Patch(facecolor=READS["structured record"], edgecolor="none")]
    leg = fig.legend(handles, ["Input: clinical notes", "Input: structured record"], loc="upper left", bbox_to_anchor=(S.LETTER_BOX_W / W_MM, 59.3 / H_MM),
                     ncol=2, frameon=False, fontsize=S.FS_SMALL, handlelength=1.4, handleheight=0.9, handletextpad=0.6, columnspacing=1.6, borderpad=0.0, borderaxespad=0.0)
    for t_ in leg.get_texts(): t_.set_color(S.INK)
    fig.text(S.LETTER_BOX_W / W_MM, 55.3 / H_MM, "Supervised rows are trained on the ground truth and scored out of fold", fontsize=S.FS_SMALL - 1.0, color=S.MUTED, ha="left", va="center")
    fig.canvas.draw()
    for t_ in ax1.get_yticklabels():
        bb = t_.get_window_extent()
        if bb.x0 / fig.dpi * 25.4 < -0.2: print("  WARNING row label runs outside the panel: %.1f mm  %r" % (bb.x0 / fig.dpi * 25.4, t_.get_text()))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "panel_g." + ext), dpi=600, bbox_inches=None, pad_inches=0, facecolor="white")
    print("wrote %s/panel_g.pdf and .png at %.1f by %.1f mm" % (a.out, W_MM, H_MM))

if __name__ == "__main__":
    main()
