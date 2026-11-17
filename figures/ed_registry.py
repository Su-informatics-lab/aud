#!/usr/bin/env python3
"""Extended Data Fig. 7: AUD-specific medication after t0 in the registry, as recorded.
"""
import argparse, os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import npj_local as S

S.apply()
INK, MUTED = S.INK, S.MUTED
FS, FS_S = S.FS_BODY, S.FS_SMALL
MM = S.MM
SN = ["AUD code", "alcohol coded only", "record silent"]
LAB = {"AUD code": "AUD diagnosis", "alcohol coded only": "Other alcohol-related diagnosis", "record silent": "No alcohol-related diagnosis"}
NEW = {"AUD ICD code": "AUD code", "alcohol ICD code only": "alcohol coded only", "no alcohol ICD code": "record silent"}
STYLE = {"AUD code": (S.RAMP_CODE[0], "-"), "alcohol coded only": (S.RAMP_CODE[2], (0, (3.0, 1.6))),
         "record silent": (S.C_SILENT, "-")}
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--cif", required=True); ap.add_argument("--prior", required=True); ap.add_argument("--counts", required=True)
ap.add_argument("--out", default="."); ap.add_argument("--dpi", type=int, default=600)
a = ap.parse_args()
c = pd.read_csv(a.cif); pu = pd.read_csv(a.prior); k = pd.read_csv(a.counts)

W, H = S.W_DOUBLE, 82.0
TMAX = 5.0
fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([14.0 / W, 31.0 / H, 52.0 / W, 44.0 / H])
ax.spines[["top", "right"]].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_linewidth(S.LW_MIN)
ax.tick_params(labelsize=FS_S, width=0.7, length=2.0, colors=INK)
ax.set_xlim(0, TMAX); ax.set_xticks(range(0, 6))
ax.set_xlabel("Years since $t_0$", fontsize=FS, labelpad=2)
ax.set_ylabel("First AUD-specific drug (%)", fontsize=FS, labelpad=2)
ax.set_title("Every new user, as recorded; no model", fontsize=FS, color=INK, pad=3, loc="left")
ymax = 0
for s in SN:
    col, ls = STYLE[s]
    x = c[(c["quantity"] == "crude_aj_rx") & (c["stratum"] == s) & (c["t"] <= TMAX + 1e-9)].sort_values("t")
    t, y = np.r_[0, x["t"]], np.r_[0, x["mean"]] * 100
    ax.step(t, y, where="post", color=col, lw=1.2, ls=ls)
    ax.text(TMAX + 0.12, y[-1], "%.1f%% %s" % (y[-1], LAB[s]), fontsize=FS_S, color=col, va="center", clip_on=False)
    ymax = max(ymax, y.max())
ax.set_ylim(0, float(np.ceil(ymax)))
ROW0, STEP, HEAD = -0.40, 0.085, -0.29
tr = ax.get_xaxis_transform()
ax.text(-0.95, HEAD, "Number at risk", transform=tr, fontsize=FS_S, color=MUTED, ha="left", va="center", clip_on=False)
for i, s in enumerate(SN):
    col, _ = STYLE[s]
    ax.text(-1.2, ROW0 - STEP * i, {"AUD code": "AUD diagnosis", "alcohol coded only": "Other alcohol-related\ndiagnosis", "record silent": "No alcohol-related\ndiagnosis"}[s], transform=tr, fontsize=FS_S - 1.0, color=col, ha="left", va="center", clip_on=False)
    x = c[(c["quantity"] == "at_risk") & (c["stratum"] == s)].sort_values("t")
    for t, n in zip(x["t"], x["mean"]):
        if 0.5 <= t <= TMAX:
            ax.text(t, ROW0 - STEP * i, format(int(n), ",d"), transform=tr, fontsize=FS_S - 0.5, color=col, ha="center", va="center", clip_on=False)

# ---- b and c: tables on the right ----
axt = fig.add_axes([0, 0, 1, 1]); axt.set_xlim(0, W); axt.set_ylim(0, H); axt.axis("off")
X0 = 107.0


def table(y_top, title, sub, heads, rows, widths):
    axt.text(X0, y_top, title, fontsize=FS, color=INK, va="bottom")
    axt.text(X0, y_top - 3.4, sub, fontsize=FS_S, color=MUTED, va="bottom")
    yh = y_top - 8.0
    xs = np.cumsum([0] + widths[:-1]) + X0
    for x, h, w in zip(xs, heads, widths):
        axt.text(x + (w if h != heads[0] else 0), yh, h, fontsize=FS_S, color=MUTED, ha="left" if h == heads[0] else "right", va="center")
    axt.plot([X0, X0 + sum(widths)], [yh - 2.2] * 2, color=INK, lw=0.5)
    y = yh - 5.3
    for col, cells in rows:
        for j, (x, v, w) in enumerate(zip(xs, cells, widths)):
            axt.text(x + (w if j else 0), y, v, fontsize=FS_S, color=col, ha="left" if j == 0 else "right", va="center")
        y -= 4.3
    axt.plot([X0, X0 + sum(widths)], [y + 2.1] * 2, color=INK, lw=0.5)
    return y


rows = []
for _, r in pu.iterrows():
    s = NEW[r["stratum"]]
    rows.append((STYLE[s][0], [LAB[s], "%d" % r["prior_users"], "%d" % r["rx_after_t0"], "%d" % r["death"], "%.1f" % r["cif5"]]))
y = table(74.5, "Prior users, by stratum", "a drug before $t_0$; a further drug on or after $t_0$, death competing",
          ["Stratum", "Patients", "Drug after", "Died", "% by 5 y"], rows, [29, 10, 12, 8, 12])
axt.text(X0 + 1, 74.5 + 2.6, "b", fontsize=S.FS_PANEL, fontweight="bold", va="bottom", ha="right")

rows = []
for s in SN:
    for lab in ("model AUD+", "model not AUD+"):
        sub = k[(k["group"] == "reviewed, %s, model" % {"AUD code": "AUD ICD code", "alcohol coded only": "alcohol ICD code only", "record silent": "no alcohol ICD code"}[s]) & (k["label"] == lab) & (k["t"] == 5)]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        rows.append((STYLE[s][0], [LAB[s] + ", MedDx " + lab.replace("model ", ""), "%d" % r["n"], "%d" % r["rx"], "%d" % r["death"]]))
y2 = table(y - 5.0, "Reviewed charts by stratum and MedDx label", "new users; first AUD-specific drug after $t_0$; counts, no model",
           ["Stratum, MedDx label", "Patients", "Treated", "Died"], rows, [43, 10, 10, 8])
axt.text(X0 + 1, y - 5.0 + 2.6, "c", fontsize=S.FS_PANEL, fontweight="bold", va="bottom", ha="right")
for ext, kw in (("pdf", {}), ("png", {"dpi": a.dpi})):
    fig.savefig(os.path.join(a.out, "ed_registry_v9.%s" % ext), transparent=True, **kw)
print("ed_registry_v9 %.0f x %.0f mm; table bottom at %.1f mm" % (W, H, y2))
