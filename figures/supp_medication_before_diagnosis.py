#!/usr/bin/env python3
"""Supplementary Fig. 4: medication recorded before the cirrhosis diagnosis, by stratum.

Registry patients with a first record of the medication dated before t0, over the registry patients of each
stratum: AUD-specific medication (the prior users), gabapentin, baclofen and topiramate. The counts are registry
aggregates from registry_frame.build() and the drug records (ingredient with descendants), entered here.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import npj_style as S
MM = S.MM; W, H = 183.0, 62.0
LAB = ["AUD diagnosis", "Other alcohol-related diagnosis", "No alcohol-related diagnosis", "All registry patients"]
N = [2627, 1591, 10188, 14406]
DRUG = [("AUD-specific medication", [144, 18, 25, 187], 6), ("Gabapentin", [584, 196, 1859, 2639], 24),
        ("Baclofen", [85, 22, 239, 346], 4), ("Topiramate", [64, 13, 172, 249], 4)]
COL = [S.RAMP_CODE[0], S.RAMP_CODE[2], S.RAMP_REF[1], S.C_BACK]
assert sum(N[:3]) == N[3] and all(sum(c[:3]) == c[3] for _, c, _ in DRUG)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": S.SANS, "font.size": S.FS_BODY, "svg.fonttype": "none"})
fig = plt.figure(figsize=(W * MM, H * MM)); X = [0, 1, 2, 3.3]
for k, (name, cnt, ymax) in enumerate(DRUG):
    ax = fig.add_axes([0.058 + k * 0.239, 0.20, 0.20, 0.55])
    for i in range(4):
        v = 100 * cnt[i] / N[i]
        ax.bar(X[i], v, 0.72, color=COL[i], edgecolor=COL[i], lw=0.6)
        ax.text(X[i], v + ymax * 0.02, "%.1f%%\n(%s)" % (v, format(cnt[i], ",d")), ha="center", va="bottom", fontsize=S.FS_SMALL - 0.5, color=S.INK, linespacing=0.95)
    ax.axvline(2.65, color=S.RULE, lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks([])
    ax.set_ylim(0, ymax); ax.set_xlim(-0.6, 3.9); ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, integer=True))
    ax.tick_params(labelsize=S.FS_SMALL, width=0.7, length=2.0, colors=S.INK); ax.tick_params(axis="x", length=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_linewidth(0.7); ax.spines[s].set_color(S.INK)
    ax.set_title(name, fontsize=S.FS_BODY, color=S.INK, loc="left", pad=11)
    if k == 0: ax.set_ylabel("Registry patients with a record\nbefore $t_0$ (%)", fontsize=S.FS_SMALL, color=S.INK)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor=c, edgecolor=c, label="%s (%s)" % (l, format(n, ",d"))) for c, l, n in zip(COL, LAB, N)], loc="lower center", bbox_to_anchor=(0.5, 0.01),
           ncol=4, frameon=False, fontsize=S.FS_SMALL, handlelength=1.2, handleheight=0.8, columnspacing=1.8)
fig.text(0.008, 0.965, "AUD-specific medication follows the alcohol-related diagnoses; gabapentin, baclofen and topiramate do not", fontsize=S.FS_BODY, fontweight="bold", color=S.INK, va="top")
import os; os.makedirs("figures/out/sf", exist_ok=True)
for ext in ("png", "pdf", "svg"): fig.savefig("figures/out/sf/supp_medication_before_diagnosis." + ext, dpi=600)
print("saved")
