#!/usr/bin/env python3
"""Fig. 2d, e: the share of patients with AUD on an AUD-specific drug from t0 to five years, by stratum, and the gap below the AUD diagnosis stratum. No model: counts over the registry estimate.
"""
import argparse, json, os
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.patches import Patch
import npj_local as S
S.apply(); INK, MUTED = S.INK, S.MUTED; FS, FS_S = S.FS_BODY, S.FS_SMALL; MM = S.MM
ap = argparse.ArgumentParser(); ap.add_argument("--json", required=True); ap.add_argument("--aj", required=True)
ap.add_argument("--band", required=True); ap.add_argument("--out", default="."); ap.add_argument("--dpi", type=int, default=600); a = ap.parse_args()
J = json.load(open(a.json)); T = J["treatment"]
c = pd.read_csv(a.aj); c = c[c["censoring"] == "admin"]
bd = pd.read_csv(a.band)
SENS = {"Other alcohol-related diagnosis": 0.9818, "No alcohol-related diagnosis": 0.9868}   # MedDx sensitivity on the reviewed charts of the stratum (Fig. 2c)
SN = ["AUD diagnosis", "Other alcohol-related diagnosis", "No alcohol-related diagnosis"]
COL = {SN[0]: S.RAMP_CODE[0], SN[1]: S.RAMP_CODE[2], SN[2]: S.C_SILENT}
W, H = S.W_DOUBLE, 64.0
fig = plt.figure(figsize=(W * MM, H * MM))
# ---- d
ax = fig.add_axes([12.0 / W, 14.5 / H, 60.0 / W, 38.0 / H])
ax.spines[["top", "right"]].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_linewidth(S.LW_MIN)
ax.tick_params(labelsize=FS_S, width=0.7, length=2.0, colors=INK)
TMAX = 5.0
curves = {}
for s in SN:
    t = T[s]; den = t["aud_denominator"]
    x = c[c["stratum"] == s].sort_values("t")
    y = 100 * (t["prior"] + t["new_users"] * x["cif"].values) / den
    tt = np.r_[0, x["t"].values]; yy = np.r_[100 * t["prior"] / den, y]; curves[s] = (tt, yy)
    b = bd[bd["stratum"] == s].sort_values("t")
    ax.fill_between(b["t"].values, b["lo"].values, b["hi"].values, color=COL[s], alpha=0.16, lw=0, zorder=1)
    ax.plot(tt, yy, color=COL[s], lw=1.2, zorder=3)
    ax.plot([0], [100 * t["prior"] / den], marker="o", ms=3.0, color=COL[s], clip_on=False, zorder=4)
    lab = "%.1f%% %s\n(%.1f%% at $t_0$)" % (t["share_5y"], s.replace("alcohol-related ", "alcohol-related\n"), t["share_t0"])
    va = {SN[0]: "center", SN[1]: "bottom", SN[2]: "top"}[s]; dy = {SN[0]: 0.0, SN[1]: 0.35, SN[2]: -0.35}[s]
    ax.text(TMAX + 0.12, t["share_5y"] + dy, lab, fontsize=FS_S, color=COL[s], va=va, clip_on=False, linespacing=1.1)
# scenario: the patients MedDx identifies in an undiagnosed stratum treated at the AUD-diagnosis stratum's rate
ref_t, ref_y = curves[SN[0]]
for s in SN[1:]:
    tt, yy = curves[s]; yref = np.interp(tt, ref_t, ref_y); sc = SENS[s] * yref + (1 - SENS[s]) * yy
    ax.plot(tt, sc, color=S.C_INDEX, lw=0.9, ls=(0, (2.0, 1.4)), zorder=3, alpha=0.9)
ax.text(0.15, 13.3, "dotted: if screened by MedDx", fontsize=FS_S, color=S.C_INDEX, va="center", ha="left")
ax.set_xlim(0, TMAX); ax.set_xticks(range(0, 6)); ax.set_xlabel("Years since $t_0$", fontsize=FS, labelpad=2)
ax.set_ylim(0, 14); ax.set_ylabel("Patients with AUD on an\nAUD-specific drug (%)", fontsize=FS_S)
fig.text(S.LETTER_BOX_W / W, 61.0 / H, "Few patients with AUD are treated at $t_0$ or in the five years after it", fontsize=FS, color=INK, fontweight="bold", va="center")
fig.text(S.LETTER_BOX_W / W, 4.0 / H, "Dot at $t_0$: prior users over the registry estimate of patients with AUD. Line: adds new users' first AUD-specific drug (Aalen–Johansen, death competing); band, bootstrap 95% CI.", fontsize=FS_S - 1.0, color=MUTED, va="bottom")
fig.text(S.LETTER_BOX_W / W, 1.6 / H, "Dotted: scenario in which the patients MedDx identifies in an undiagnosed stratum are treated at the AUD-diagnosis stratum's rate.", fontsize=FS_S - 1.0, color=MUTED, va="bottom")
# ---- e
h = [Patch(facecolor="white", edgecolor=INK), Patch(facecolor=INK, edgecolor=INK)]
X_E = S.W_SINGLE + S.GUTTER
ax2 = fig.add_axes([(X_E + 16.0) / W, 14.5 / H, 68.0 / W, 38.0 / H])
ax2.spines[["top", "right"]].set_visible(False)
for sp in ("left", "bottom"): ax2.spines[sp].set_linewidth(S.LW_MIN)
ax2.tick_params(labelsize=FS_S, width=0.7, length=2.0, colors=INK)
for j, s in enumerate(SN[1:]):
    t = T[s]; g0 = t["gap_t0"]; ga = t["gap_added_after_t0"]; g5 = t["gap_5y"]; lo, hi = t["gap_5y_ci"]
    ax2.bar(j, g0, width=0.5, facecolor="white", edgecolor=COL[s], linewidth=S.LW_MIN, zorder=2)
    ax2.bar(j, ga, bottom=g0, width=0.5, facecolor=COL[s], edgecolor=COL[s], linewidth=S.LW_MIN, zorder=2)
    ax2.errorbar([j], [g5], yerr=[[g5 - lo], [hi - g5]], fmt="none", ecolor=INK, elinewidth=S.LW_MIN, capsize=1.2, zorder=3)
    ax2.text(j + 0.28, g0 / 2, "%.1f" % g0, ha="left", va="center", fontsize=FS_S - 0.5, color=INK)
    ax2.text(j + 0.28, g0 + ga / 2, "+%.1f" % ga, ha="left", va="center", fontsize=FS_S - 0.5, color=INK)
    ax2.text(j, hi + 0.15, "%.1f" % g5, ha="center", va="bottom", fontsize=FS_S, color=INK, fontweight="bold")
    assert abs(g0 + ga - g5) < 0.01, s
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Other alcohol-related diagnosis", "No alcohol-related diagnosis"], fontsize=FS_S)
ax2.set_ylim(0, 11.5); ax2.set_ylabel("Points below the\nAUD-diagnosis stratum", fontsize=FS_S)
ax2.legend(h, ["Already treated at $t_0$", "Newly treated within 5 years"], loc="upper left", frameon=False, fontsize=FS_S, handlelength=1.4, handleheight=0.9, borderpad=0.0, borderaxespad=0.2)
fig.text((X_E + 12.0) / W, 61.0 / H, "The gap below the AUD-diagnosis stratum widens after $t_0$", fontsize=FS, color=INK, fontweight="bold", va="center")
os.makedirs(a.out, exist_ok=True)
for ext in ("png", "pdf"): fig.savefig(os.path.join(a.out, "fig2nh_de_v15." + ext), dpi=a.dpi, facecolor="white")
print("fig2nh_de_v16 %g x %g mm" % (W, H))
for s in SN: print(s, {k: round(T[s][k], 2) for k in ("share_t0", "share_1y", "share_5y")}, T[s].get("gap_5y"))
