"""Fig. 4e: agreement (Cohen's kappa, 95% bootstrap interval) between two readings of the same chart on the 186 charts
hepatologist B read, drawn from the table stability.py writes. The axis and the printed column are the same quantity.
"""
import argparse
import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import npj_local as S
S.apply()
ap = argparse.ArgumentParser(); ap.add_argument("--csv", default="q/stability.csv")
ap.add_argument("--out", default="q/fig4"); a = ap.parse_args()
t = pd.read_csv(a.csv)
W, H = 57.0, 58.0
fig = plt.figure(figsize=(W * S.MM, H * S.MM))
ax = fig.add_axes([27.0 / W, 10.0 / H, 18.0 / W, 38.0 / H])
GROUPS = [("Two hepatologists", "Two hepatologists", "human"),
          ("LLM with itself", "MedDx with itself vs temperature 0", "self"),
          ("LLM and a hepatologist", "MedDx and a hepatologist", "mixed")]
PAIR = {"T = 0.3": "Temperature 0.3", "T = 0.7": "Temperature 0.7", "Paraphrase A": "Paraphrased skill A", "Paraphrase B": "Paraphrased skill B", "T=0.3": "Temperature 0.3", "T=0.7": "Temperature 0.7", "M.A. and S.N.": "A and B", "LLM and M.A.": "MedDx and A", "LLM and S.N.": "MedDx and B"}
STY = {"self": dict(color=S.C_INDEX, mfc=S.C_INDEX), "human": dict(color=S.C_REF, mfc=S.C_REF),
       "mixed": dict(color=S.C_INDEX, mfc="white")}
y, ticks, labs, heads = 0.0, [], [], []
for g, head, kind in GROUPS:
    heads.append((y, head)); y -= 0.85
    for _, r in t[t["group"] == g].iterrows():
        ax.errorbar(r["kappa"], y, xerr=[[r["kappa"] - r["k_lo"]], [r["k_hi"] - r["kappa"]]],
                    fmt="o", ms=3.2, lw=0.9, capsize=1.6, mew=0.9, **STY[kind])
        ticks.append(y); labs.append(PAIR.get(r["pair"], r["pair"]))
        ax.text(1.04, y, "%.2f" % r["kappa"], transform=ax.get_yaxis_transform(), fontsize=S.FS_SMALL,
                va="center", ha="left", color=S.INK)
        y -= 1.0
    y -= 0.35
ax.set_ylim(y + 0.6, heads[0][0] + 0.6)
for yy, head in heads:
    xf = ax.transData.transform((ax.get_xlim()[0], yy))
    yf = fig.transFigure.inverted().transform(xf)[1]
    fig.text(2.0 / W, yf, head, fontsize=S.FS_SMALL, ha="left", va="center", fontweight="bold",
             color=S.INK)
ax.text(1.04, heads[0][0], "κ", transform=ax.get_yaxis_transform(), fontsize=S.FS_SMALL,
        ha="left", va="center", color=S.MUTED, style="italic")
ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=S.FS_SMALL)
ax.set_ylim(y + 0.6, heads[0][0] + 0.6)
ax.set_xlim(0.68, 1.0); ax.set_xticks([0.7, 0.8, 0.9, 1.0])
ax.tick_params(axis="y", length=0, labelsize=S.FS_SMALL); ax.tick_params(axis="x", labelsize=S.FS_SMALL)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_xlabel("Cohen's κ (95% CI)", fontsize=S.FS_SMALL, labelpad=2)
fig.text(7.0 / W, 55.5 / H, "Two readings of the same chart", fontsize=S.FS_BODY, fontweight="bold", va="center", color=S.INK)
assert S.check(fig, W, H) == 0, "text leaves the panel"
os.makedirs(a.out, exist_ok=True)
fig.savefig(os.path.join(a.out, "panel_stability.png"), dpi=600, transparent=True)
fig.savefig(os.path.join(a.out, "panel_stability.pdf"), transparent=True)
print("panel_stability %.0f x %.0f mm" % (W, H))
