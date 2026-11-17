#!/usr/bin/env python3
"""Extended Data Fig. 4: the nine cells of the registry estimate.
"""
import argparse, json, os
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import npj_local as S
S.apply(); INK, MUTED = S.INK, S.MUTED; FS, FS_S = S.FS_BODY, S.FS_SMALL; MM = S.MM
ap = argparse.ArgumentParser(); ap.add_argument("--json", required=True); ap.add_argument("--out", default="."); ap.add_argument("--dpi", type=int, default=600); a = ap.parse_args()
J = json.load(open(a.json)); cells = J["cells"]
SN = {1: "AUD diagnosis (any time)", 2: "Other alcohol-related diagnosis (any time)", 3: "No alcohol-related diagnosis (any time)"}
HN = ["Academic health system", "Safety-net hospital", "Other hospitals"]
SC = {1: S.RAMP_CODE[0], 2: S.RAMP_CODE[2], 3: S.C_SILENT}
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h
W, H = S.W_DOUBLE, 64.0
fig = plt.figure(figsize=(W * MM, H * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
C_LAB, C_REG, C_CH, C_POS, C_PCT, C_AUD = 48.0, 68.0, 86.0, 96.0, 106.0, 178.0
XA0, XA1 = 114.0, 160.0
def sx(v): return XA0 + (XA1 - XA0) * v
HB = 52.5; y = HB - 3.4; ROW = 3.1; AX_Y = 6.0
ax.text(S.LETTER_BOX_W, 61.0, "Nine cells carry chart review to the registry; the share with AUD rises at the safety-net hospital", fontsize=FS, va="center", color=INK, fontweight="bold")
for x, s_, ha in ((C_LAB, "Hospital type", "right"), (C_REG, "Registry\npatients", "right"), (C_CH, "Charts:\ndeterminate\n(reviewed)", "right"), (C_POS, "AUD+\ncharts", "right"), (C_AUD, "Patients\nwith AUD", "right")):
    ax.text(x, HB, s_, fontsize=FS_S, ha=ha, va="bottom", linespacing=1.2)
ax.text((XA0 + XA1) / 2, HB, "Share with AUD, % (Wilson 95% CI)", fontsize=FS_S, ha="center", va="bottom")
ax.plot([S.LETTER_BOX_W, W - 1.0], [HB - 0.8] * 2, color=INK, lw=0.8)
tot_N = tot_A = 0.0
for s in (1, 2, 3):
    sN = sA = 0.0; rows = []
    for h in HN:
        c = [x for x in cells if x["h"] == h and x["s"] == s][0]
        p, lo, hi = wilson(c["a_pos"], c["n_det"]); est = c["N"] * p; sN += c["N"]; sA += est; rows.append((h, c, p, lo, hi, est))
    # stratum line carries the stratum subtotal
    ax.text(S.LETTER_BOX_W, y, SN[s], fontsize=FS, ha="left", va="center", color=SC[s], fontweight="bold")
    ax.text(C_REG, y, format(int(sN), ",d"), fontsize=FS, ha="right", va="center", color=SC[s])
    ax.text(C_PCT, y, "%.1f" % (100 * sA / sN), fontsize=FS, ha="right", va="center", color=SC[s])
    ax.text(C_AUD, y, format(int(round(sA)), ",d"), fontsize=FS, ha="right", va="center", color=SC[s], fontweight="bold")
    y -= ROW
    for h, c, p, lo, hi, est in rows:
        ax.text(C_LAB, y, h, fontsize=FS, ha="right", va="center")
        ax.text(C_REG, y, format(c["N"], ",d"), fontsize=FS, ha="right", va="center")
        ax.text(C_CH, y, "%d (%d)" % (c["n_det"], c["n_rev"]), fontsize=FS, ha="right", va="center")
        ax.text(C_POS, y, "%d" % c["a_pos"], fontsize=FS, ha="right", va="center")
        ax.text(C_PCT, y, "%.1f" % (100 * p), fontsize=FS, ha="right", va="center")
        ax.plot([sx(lo), sx(hi)], [y] * 2, color=SC[s], lw=S.LW_MIN, zorder=3)
        for e in (lo, hi): ax.plot([sx(e)] * 2, [y - 0.6, y + 0.6], color=SC[s], lw=S.LW_MIN, zorder=3)
        ax.plot([sx(p)], [y], marker="o", ms=3.2, mfc=SC[s], mec=SC[s], zorder=4)
        ax.text(C_AUD, y, format(int(round(est)), ",d"), fontsize=FS, ha="right", va="center")
        y -= ROW
    tot_N += sN; tot_A += sA
    ax.plot([S.LETTER_BOX_W, W - 1.0], [y + ROW * 0.5] * 2, color=MUTED, lw=S.LW_HAIR, ls=S.DOT, zorder=1)
ax.text(C_LAB, y, "All registry patients", fontsize=FS, ha="right", va="center", style="italic")
ax.text(C_REG, y, format(int(tot_N), ",d"), fontsize=FS, ha="right", va="center")
ax.text(C_PCT, y, "%.1f" % (100 * tot_A / tot_N), fontsize=FS, ha="right", va="center")
ax.text(C_AUD, y, format(int(round(tot_A)), ",d"), fontsize=FS, ha="right", va="center", fontweight="bold")
for v in (0, 0.25, 0.5, 0.75, 1.0):
    ax.plot([sx(v)] * 2, [AX_Y - 0.8, AX_Y], color=INK, lw=0.7)
    ax.text(sx(v), AX_Y - 1.2, "%d" % (100 * v), fontsize=FS_S, color=MUTED, ha="center", va="top")
ax.plot([sx(0), sx(1)], [AX_Y] * 2, color=INK, lw=S.LW_MIN)
assert y > AX_Y + 2.0, y
print("E1 from cells %.1f" % tot_A, "json E1 %.1f" % J["point"]["E1"]); assert abs(tot_A - J["point"]["E1"]) < 1.0
os.makedirs(a.out, exist_ok=True)
for ext in ("png", "pdf"): fig.savefig(os.path.join(a.out, "ed2_cells_v16." + ext), dpi=a.dpi, facecolor="white")
print("ed2_cells_v16 written")
