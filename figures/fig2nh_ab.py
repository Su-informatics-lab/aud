#!/usr/bin/env python3
"""Fig. 2b, c: the AUD diagnosis against the registry estimate, and the three strata at t0.
"""
from __future__ import annotations

import argparse
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

import npj_local as S

C_INDEX, C_CODE = S.C_INDEX, S.C_CODE
RAMP_INDEX, RAMP_CODE, RAMP_REF = S.RAMP_INDEX, S.RAMP_CODE, S.RAMP_REF
INK, MUTED, RULE = S.INK, S.MUTED, S.RULE
MM = S.MM
FS, FS_S, FS_P = S.FS_BODY, S.FS_SMALL, S.FS_PANEL
S.apply()

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--out", default=".")
_ap.add_argument("--dpi", type=int, default=600)
ARGS = _ap.parse_args()

N_REG = 14406
# ---- panel a, panel_registry_burden.csv and panel_registry_cascade.csv ------------------
BURDEN = [
    ("Named by the AUD diagnosis", [
        ("AUD diagnosis at $t_0$", 2627, None, None, "code"),
        ("AUD diagnosis at any time", 4210, None, None, "code")]),
    ("Estimated to have AUD, projected to the registry by hospital type and diagnosis", [
        ("Ground truth", 6223, 5744, 6749, "ref"),
        ("MedDx classification", 6466, 5906, 6981, "llm")]),
]
FILL = {"code": (RAMP_CODE[2], RAMP_CODE[0]), "ref": (RAMP_REF[2], RAMP_REF[0]),
        "model": ("white", RAMP_REF[0]), "llm": (RAMP_INDEX[2], RAMP_INDEX[0])}
# Registry estimate (Model 1: hospital type x whole-record stratum, determinate charts; master_numbers_2026-10-26.json):
# patients with AUD, and how many of them carry the AUD diagnosis at t0 and at any time.
N_AUD_MODEL = 6223
AUD_CODED = {0: 2417, 1: 3900}   # design-weighted (hospital type x diagnosis) patients with AUD among those diagnosed at t0 / at any time
GAIN = [(0, 2627, "%d%% undocumented" % round(100 * (1 - AUD_CODED[0] / N_AUD_MODEL))),
        (1, 4210, "%d%% undocumented" % round(100 * (1 - AUD_CODED[1] / N_AUD_MODEL)))]
GAIN_TO = N_AUD_MODEL

# ---- panel b ------------------------------------------------------------------------------
# stratum, registry patients, estimated to have AUD (%, charts weighted by cell), patients with AUD,
# reference-positive charts, LLM (sens, lo, hi), code (sens, lo, hi): the AUD diagnosis at any time in the
# registry record, sensitivity_rows.json fig2c dx_omop (127/127, 32/47, 43/76), Wilson intervals
STRATA = [
    ("AUD diagnosis", 2627, 91.99, 2417, 127,
     (0.9843, 0.9444, 0.9957), (1.0000, 0.9706, 1.0000)),
    ("Other alcohol-related diagnosis", 1591, 81.31, 1294, 47,
     (0.9787, 0.8889, 0.9962), (0.6809, 0.5383, 0.7960)),
    ("No alcohol-related diagnosis", 10188, 24.66, 2513, 76,
     (0.9868, 0.9292, 0.9977), (0.5658, 0.4539, 0.6714)),
]
TOTAL = ("All registry patients", 14406, 43.20, 6223)
HILITE = 2                                   # the stratum the paper's claim rests on

W, H = S.W_DOUBLE, 64.0
PW = S.W_SINGLE
X_B = PW + S.GUTTER                          # panel b's left edge on the canvas
AX_Y = 12.0
TICK_Y, LAB_Y = AX_Y - 1.2, AX_Y - 5.4
fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# ======================== panel a, the registry yield ===================================
XB0, XB1 = 32.0, 74.0
BMAX = 7200.0


def bx(v):
    return XB0 + (XB1 - XB0) * v / BMAX


HEAD_Y = {0: 53.0, 1: 31.0}
ax.text(S.LETTER_BOX_W, 59.6, "Most patients with AUD carry no AUD diagnosis at $t_0$", fontsize=FS, va="bottom", color=INK, fontweight="bold")
BAR_Y = [50.0, 42.0, 27.0, 20.0]
bar_h = 3.4
k = 0
for g, (head, rows) in enumerate(BURDEN):
    ax.text(S.LETTER_BOX_W - 4.0, HEAD_Y[g], head, fontsize=FS_S, va="bottom", color=MUTED)
    for lab, m, lo, hi, kind in rows:
        y = BAR_Y[k]
        k += 1
        fill, edge = FILL[kind]
        ax.add_patch(Rectangle((XB0, y - bar_h / 2), bx(m) - XB0, bar_h,
                               facecolor=fill, edgecolor=edge, lw=0.8, zorder=2))
        if lo is not None:
            ax.plot([bx(lo), bx(hi)], [y] * 2, color=edge, lw=S.LW_MIN, zorder=4)
            for e in (lo, hi):
                ax.plot([bx(e)] * 2, [y - 1.0, y + 1.0], color=edge, lw=S.LW_MIN,
                        zorder=4)
        ax.text(XB0 - 1.6, y, lab, fontsize=FS, va="center", ha="right")
        ax.text(XB1 + 2.0, y, format(m, ",d"), fontsize=FS, va="center", color=INK)
assert k == len(BAR_Y)
ax.plot([bx(GAIN_TO)] * 2, [BAR_Y[-1] + bar_h / 2, BAR_Y[0]], color=RAMP_REF[1],
        lw=0.5, ls=(0, (1.6, 1.6)), zorder=1)
for _row, _from, _lab in GAIN:
    _y = BAR_Y[_row]
    ax.annotate("", xy=(bx(GAIN_TO), _y), xytext=(bx(_from), _y), zorder=5,
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.6, shrinkA=0.0,
                                shrinkB=0.0, mutation_scale=4.5))
    ax.text((bx(_from) + bx(GAIN_TO)) / 2, _y + bar_h / 2 + 0.25, _lab, fontsize=FS_S - 0.5,
            color=INK, ha="center", va="bottom", zorder=6)
for v in (0, 2000, 4000, 6000):
    ax.plot([bx(v)] * 2, [AX_Y - 0.8, AX_Y], color=INK, lw=0.7)
    ax.text(bx(v), TICK_Y, format(v, ",d"), fontsize=FS_S, color=MUTED, ha="center",
            va="top")
ax.plot([XB0, bx(6000)], [AX_Y] * 2, color=INK, lw=S.LW_MIN)
ax.text((XB0 + XB1) / 2, LAB_Y, "Registry patients, of %s" % format(N_REG, ",d"),
        fontsize=FS, ha="center", va="top")

# ======================== panel b, the strata ===========================================
# Columns in millimetres from panel b's left edge.
C_LAB = 30.0          # stratum label, right-aligned
C_REG = 40.0          # registry patients, right-aligned
C_PCT = 48.3          # chart-documented AUD, %, right-aligned
C_AUD = 58.0          # patients with AUD, right-aligned
XA0, XA1 = 63.0, 86.0  # sensitivity axis, 0.50 to 1.00
ROW_Y = [44.0, 35.0, 26.0]
TOT_Y = 18.0
HB = 49.5             # header baseline
ROW_H = 9.0


def bxx(v):
    return X_B + v


def sx(v):
    return bxx(XA0 + (XA1 - XA0) * (v - 0.5) / 0.5)


ax.text(bxx(S.LETTER_BOX_W), 59.6, "Most patients with AUD lie outside the AUD diagnosis stratum", fontsize=FS, va="bottom", color=INK, fontweight="bold")
# header: the two AUD columns share a spanning head, as the old table's did
ax.text(bxx(C_REG), HB, "Registry\npatients", fontsize=FS_S, ha="right", va="bottom",
        linespacing=1.25)
ax.text(bxx((C_PCT - 6.0 + C_AUD) / 2), HB + 5.3, "Estimated to have AUD", fontsize=FS_S,
        ha="center", va="bottom")
ax.plot([bxx(C_PCT - 6.0), bxx(C_AUD)], [HB + 4.9] * 2, color=INK, lw=0.5)
ax.text(bxx(C_PCT), HB, "%", fontsize=FS_S, ha="right", va="bottom")
ax.text(bxx(C_AUD), HB, "Patients", fontsize=FS_S, ha="right", va="bottom")
ax.text(bxx(C_LAB), HB, "Diagnoses at $t_0$", fontsize=FS_S, ha="right",
        va="bottom")
ax.legend(handles=[
    Line2D([0], [0], marker="o", color="none", mfc=C_INDEX, mec=RAMP_INDEX[0], ms=3.6,
           label="MedDx"),
    Line2D([0], [0], marker="s", color="none", mfc=C_CODE, mec=RAMP_CODE[0], ms=3.4,
           label="AUD diagnosis")],
    loc="lower left", bbox_to_anchor=(bxx(XA0 + 1.0) / W, (HB - 0.2) / H),
    frameon=False, fontsize=FS_S, handletextpad=0.3, labelspacing=0.25, borderpad=0.0,
    handlelength=1.0)
ax.plot([bxx(S.LETTER_BOX_W), bxx(PW - 0.5)], [HB - 1.0] * 2, color=INK, lw=0.8)

for i, (lab, n, th, p, npos, llm, code) in enumerate(STRATA):
    y = ROW_Y[i]
    if i == HILITE:
        ax.add_patch(Rectangle((bxx(S.LETTER_BOX_W), y - ROW_H / 2),
                               PW - 0.5 - S.LETTER_BOX_W, ROW_H,
                               facecolor=S.TINT_SILENT, edgecolor="none", zorder=0))
    _two = "\n" in lab.replace("alcohol-related ", "alcohol-related\n") and lab != "AUD diagnosis"
    ax.text(bxx(C_LAB), y + (1.9 if _two else 1.1), lab.replace("alcohol-related ", "alcohol-related\n") if _two else lab, fontsize=FS, ha="right", va="center", linespacing=0.95,
            color=[RAMP_CODE[0], RAMP_CODE[2], S.C_SILENT][i], fontweight="bold")
    ax.text(bxx(C_LAB), y - (3.0 if _two else 2.1), "$n$ = %d" % npos, fontsize=FS_S, ha="right",
            va="center", color=MUTED)
    ax.text(bxx(C_REG), y, format(n, ",d"), fontsize=FS, ha="right", va="center")
    ax.text(bxx(C_PCT), y, "%.1f" % th, fontsize=FS, ha="right", va="center")
    ax.text(bxx(C_AUD), y, format(p, ",d"), fontsize=FS, ha="right", va="center")
    for (m, lo, hi), col, edge, mk, ms, off in (
            (llm, C_INDEX, RAMP_INDEX[0], "o", 3.6, 1.0),
            (code, C_CODE, RAMP_CODE[0], "s", 3.4, -1.0)):
        ax.plot([sx(lo), sx(hi)], [y + off] * 2, color=edge, lw=S.LW_MIN,
                solid_capstyle="butt", zorder=3)
        for e in (lo, hi):
            ax.plot([sx(e)] * 2, [y + off - 0.7, y + off + 0.7], color=edge, lw=S.LW_MIN,
                    zorder=3)
        ax.plot([sx(m)], [y + off], marker=mk, ms=ms, mfc=col, mec=edge, mew=0.8,
                zorder=4)
    if i < len(STRATA) - 1:
        ax.plot([bxx(S.LETTER_BOX_W), bxx(PW - 0.5)], [y - ROW_H / 2] * 2, color=MUTED,
                lw=S.LW_HAIR, ls=S.DOT, zorder=1)
ax.plot([bxx(S.LETTER_BOX_W), bxx(PW - 0.5)], [ROW_Y[-1] - ROW_H / 2] * 2, color=INK,
        lw=0.8)
tl, tn, tth, tp = TOTAL
for x, s_, style in ((C_LAB, tl, "italic"), (C_REG, format(tn, ",d"), "normal"),
                     (C_PCT, "%.1f" % tth, "normal"), (C_AUD, format(tp, ",d"), "normal")):
    ax.text(bxx(x), TOT_Y, s_, fontsize=FS, ha="right", va="center", color=MUTED,
            style=style)

for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    ax.plot([sx(v)] * 2, [AX_Y - 0.8, AX_Y], color=INK, lw=0.7)
    ax.text(sx(v), TICK_Y, ("%.1f" % v).lstrip("0") if v < 1 else "1", fontsize=FS_S,
            color=MUTED, ha="center", va="top")
ax.plot([sx(0.5), sx(1.0)], [AX_Y] * 2, color=INK, lw=S.LW_MIN)
ax.text((sx(0.5) + sx(1.0)) / 2, LAB_Y, "Sensitivity (95% CI)", fontsize=FS, ha="center",
        va="top")

# --- guards on the arithmetic
assert sum(s[1] for s in STRATA) == TOTAL[1] == N_REG
assert abs(sum(s[3] for s in STRATA) - TOTAL[3]) <= 1   # each count is rounded from its own unrounded value
for s in STRATA + [TOTAL]:
    assert abs(s[1] * s[2] / 100.0 - s[3]) < 1.0 + s[1] * 5e-5, s[0]
assert TOTAL[3] == [r[1] for r in BURDEN[1][1] if r[0] == "Ground truth"][0]
assert BURDEN[1][1][1][1] == 6466, "MedDx projection (meddx_estimate_2026-10-26.json)"
assert STRATA[0][1] == BURDEN[0][1][0][1], "the AUD-code stratum is the code count at t0"
for _row, _from, _lab in GAIN:
    assert _lab in ("61% undocumented", "37% undocumented"), _lab
assert min(r[1] for r in BURDEN[1][1]) > max(r[1] for r in BURDEN[0][1])
# --- guards on the layout
assert ROW_Y[-1] - ROW_H / 2 - 1.0 > TOT_Y, "the total row runs into the last stratum"
assert TOT_Y - 2.0 > AX_Y, "the total row crowds the axis"
assert ROW_Y[0] + ROW_H / 2 <= HB - 1.0 + 1e-9, "the first row runs into the header"
assert S.check(fig, W, H) == 0, "text leaves the canvas"
fig.canvas.draw()
for t in ax.texts:
    bb = t.get_window_extent()
    x0t, x1t = bb.x0 / fig.dpi * 25.4, bb.x1 / fig.dpi * 25.4
    assert not (x0t < PW + S.GUTTER / 2 < x1t), ("text crosses the cut", t.get_text())
    if x0t >= X_B and x0t < X_B + S.LETTER_BOX_W:
        assert bb.y1 / fig.dpi * 25.4 < H - S.LETTER_BOX_W / 2, ("letter box", t.get_text())

fig.savefig(os.path.join(ARGS.out, "fig2nh_ab.pdf"), transparent=True)
fig.savefig(os.path.join(ARGS.out, "fig2nh_ab.png"), dpi=ARGS.dpi, transparent=True)
for key, x0 in (("a", 0.0), ("b", X_B)):
    bb = Bbox.from_bounds(x0 * MM, 0.0, PW * MM, H * MM)
    for ext, kw in (("pdf", {}), ("png", {"dpi": ARGS.dpi})):
        fig.savefig(os.path.join(ARGS.out, "fig3v_panel_%s.%s" % (key, ext)),
                    transparent=True, bbox_inches=bb, **kw)
print("  fig2nh_ab  %.0f x %.0f mm; panels a and b cut at %.0f mm each" % (W, H, PW))
