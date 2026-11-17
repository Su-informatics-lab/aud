#!/usr/bin/env python3
"""The quotation audit counts drawn in Fig. 4f.
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from scipy.stats import beta

import npj_style as S
import f05_figure5 as F
import run_npj_provenance as prov
from grounding_finish import fragments

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

rates, trail, ct, ab, man, stmt, hand = F.load(argparse.Namespace(
    detail=F.DETAIL, panel=F.PANEL, abstain=F.ABSTAIN))
p = pd.read_csv(F.PANEL, dtype={"omop_person_id": str})
p = p[p["model"].eq(F.PRIMARY_TAG)].set_index("omop_person_id")
notes = prov.load_patient_notes("resources/mapped_report_ids.csv",
    ["/N/project/depot/hw56/aud/cirrhosis_regv20/redacted_txt",
     "/N/project/depot/hw56/aud/cirrhosis_regv20/redacted_pdf-text"])
pieces = words_all = words_missing = 0
for pid in hand:
    q = str(p.loc[pid, "supporting_quote"]); nt = notes.get(pid, "")
    frs = [prov.normalize(f) for f in fragments(q)]
    if re.search(r"\.\.\.|…", q) and frs and all(f in nt for f in frs):
        pieces += 1
    elif all((" %s " % t) in (" %s " % nt) for t in prov.normalize(q).split()):
        words_all += 1
    else:
        words_missing += 1
d = {"n": trail["n"], "verbatim": trail["auto"], "verbatim_in_pieces": pieces,
     "not_verbatim_all_words_present": words_all,
     "not_verbatim_words_absent": words_missing, "no_quotation": trail["statement"]}
d["quoting"] = d["n"] - d["no_quotation"]
d["verbatim_total"] = d["verbatim"] + d["verbatim_in_pieces"]
k, n = d["verbatim_total"], d["quoting"]
d["pct"] = round(100.0 * k / n, 1)
d["ci_low"] = round(100.0 * beta.ppf(0.025, k, n - k + 1), 1)
d["ci_high"] = round(100.0 * beta.ppf(0.975, k + 1, n - k), 1) if k < n else 100.0
assert d["verbatim"] + pieces + words_all + words_missing + d["no_quotation"] == d["n"]
pd.DataFrame([d]).to_csv(os.path.join(a.out, "panel_audit_v8.csv"), index=False)
print(d)

S.apply_npj_style()
W, H = 57.0, 58.0
fig = plt.figure(figsize=S.mm(W, H)); fig.set_layout_engine("none")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
NV = words_all + words_missing
segs = [(d["verbatim"], S.RAMP_INDEX[3], S.RAMP_INDEX[2], "-"),
        (pieces, S.RAMP_INDEX[2], S.RAMP_INDEX[2], "-"),
        (NV, "white", S.RAMP_INDEX[0], "-"),
        (d["no_quotation"], S.TINT[S.C_INDET], S.RAMP_REF[2], (0, (1.5, 1.0)))]
x0, x1, y, h = 3.0, 54.0, 25.0, 9.0
kx = (x1 - x0) / d["n"]
xs, x = [], x0
for w_, fc, ec, ls in segs:
    ax.add_patch(Rectangle((x, y), w_ * kx, h, fc=fc, ec=ec, lw=0.6, ls=ls))
    xs.append((x, x + w_ * kx)); x += w_ * kx
FS, FSS = S.FS_BODY, S.FS_SMALL
ax.text(sum(xs[0]) / 2, y + h / 2, "%d verbatim" % d["verbatim"], ha="center", va="center",
        fontsize=FSS, color=S.INK)
# the three narrow segments are labelled below the bar, each on its own leader
labs = [("%d verbatim in pieces" % pieces, S.RAMP_INDEX[0], 4.0),
        ("%d not verbatim" % NV, S.RAMP_INDEX[0], 9.5),
        ("%d no quotation" % d["no_quotation"], S.MUTED, 15.0)]
for (xa, xb), (s, col, drop) in zip(xs[1:], labs):
    xm = (xa + xb) / 2
    ax.plot([xm, xm], [y, y - drop + 0.8], color=col, lw=0.5)
    ax.text(xm + 0.6, y - drop, s, ha="right", va="top", fontsize=FSS, color=col)
yb = y + h + 2.0; xb = xs[1][1]
ax.plot([x0, x0, xb, xb], [yb - 0.8, yb, yb, yb - 0.8], color=S.INK, lw=0.6)
ax.text((x0 + xb) / 2, yb + 1.2, "%d of %d supporting\nquotations verbatim" % (k, n),
        ha="center", va="bottom", fontsize=FS, color=S.INK, linespacing=1.1)
ax.text(x0, 2.0, "%d patients" % d["n"], fontsize=FSS, color=S.MUTED)
fig.savefig(os.path.join(a.out, "panel_audit_v8.png"), dpi=600, transparent=True)
fig.savefig(os.path.join(a.out, "panel_audit_v8.pdf"), transparent=True)
print("panel_audit_v8 %.0f x %.0f mm" % (W, H))
