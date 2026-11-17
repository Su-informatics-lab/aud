#!/usr/bin/env python3
"""Fig. 4b-f and Extended Data Fig. 6.

Runs panel_model_leaderboard_v16.py (4b), panel_stability_v15.py (4e) and panel_model_matrix_v15.py (4d), and
draws the rest: sensitivity and specificity under the skill variations and across engines (4c), the quotation
audit (4f), and the findings and note types MedDx cites by class and by hospital system (Extended Data Fig. 6).
Reads the tables written by acc_readers.py, evid_by_site.py, panel_audit_v8.py, panel_trigger_section.py and
stability.py. Fig. 4a is a diagram drawn by hand. Run from the analysis root with
PYTHONPATH=figures:downstream/npj:downstream.
"""
import os, sys, subprocess
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import npj_style as S
C = "figures"; OUT = C + "/out/fig4"; os.makedirs(OUT, exist_ok=True)
S.apply_npj_style(); O = S.RAMP_INDEX; INK, MUTED = S.INK, S.MUTED
PLUM, GREEN = "#5B2A86", "#1B7F5A"; SAFETY, ACAD = "#8C510A", "#5E5E5E"; SKILL, HARN, ENG = "#B8860B", "#2A7F8F", "#B5366E"
def save(fig, name, W, H, out=OUT):
    S.check_text_fits(fig, W, H)
    for ext in ("pdf", "png"): fig.savefig(os.path.join(out, name + "." + ext), dpi=600, bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig); print("wrote", name, W, H)
def title(fig, W, H, s, y_mm, x_mm=7.0): fig.text(x_mm / W, y_mm / H, s, fontsize=S.FS_BODY, fontweight="bold", va="center", color=INK)
NAME = {"social history": "Social history", "hpi narrative": "HPI narrative", "discharge diagnosis": "Discharge diagnosis", "problem list": "Problem list", "assessment/plan": "Assessment/plan", "structured field": "Structured field", "discharge summary": "Discharge summary", "progress note": "Progress note", "hospital course": "Hospital course", "consultation note": "Consultation note"}

# ---- b: leaderboard of engines ----
subprocess.run([sys.executable, C + "/panel_model_leaderboard_v16.py", "--out", OUT], check=True)

# ---- c: stability, skill terms ----
subprocess.run([sys.executable, C + "/panel_stability_v15.py", "--csv", "figures/out/stability/stability.csv", "--out", OUT], check=True)
print("stability pairs:", sorted(set(pd.read_csv("figures/out/stability/stability.csv")["pair"])))

# ---- e: quotation audit strip + where the evidence is deposited ----
d = pd.read_csv("figures/out/fig4/panel_audit_v8.csv").iloc[0]
exact = int(d["verbatim"]) + int(d["verbatim_in_pieces"]); fuzzy = int(d["not_verbatim_all_words_present"]) + int(d["not_verbatim_words_absent"]); none = int(d["no_quotation"])
assert exact + fuzzy + none == 485 and exact == 447 and fuzzy == 8 and none == 30, (exact, fuzzy, none)
ev = pd.read_csv(C + "/out/new/evidence_by_gt_site.csv"); ev["lab"] = ev["category"].map(NAME)
ev = ev.sort_values("n_AUD+", ascending=False).reset_index(drop=True)
W, H = 100.0, 52.0
fig = plt.figure(figsize=S.mm(W, H)); fig.set_layout_engine("none")
# audit strip
ax = fig.add_axes([7.0 / W, 36.5 / H, 86.0 / W, 4.0 / H]); ax.set_xlim(0, 485); ax.set_ylim(0, 1); ax.axis("off")
x0 = 0
for n, fc, ec, lab in [(exact, O[2], O[1], "%d exact" % exact), (fuzzy, "white", O[0], "%d fuzzy" % fuzzy), (none, "#ECEBE8", "#9A9893", "%d no quotation" % none)]:
    ax.add_patch(Rectangle((x0, 0), n, 1, facecolor=fc, edgecolor=ec, lw=0.5))
    x0 += n
ax.text(exact / 2, 0.5, "%d exact, word for word" % exact, ha="center", va="center", fontsize=S.FS_SMALL, color="white")
ax.text(485, -0.45, "%d fuzzy, %d no quotation" % (fuzzy, none), ha="right", va="top", fontsize=S.FS_SMALL - 0.5, color=MUTED)
fig.text(7.0 / W, 42.3 / H, "Quotation audit, 485 classifications", fontsize=S.FS_SMALL, color=INK, va="bottom")
# three classes explained below the strip
rows3 = [(exact, O[2], O[1], "Exact match, word for word in the notes: %d as one passage, %d as passages joined by an ellipsis" % (d["verbatim"], d["verbatim_in_pieces"])),
         (fuzzy, "white", O[0], "Fuzzy match, in the notes but not as one passage: %d every word present, %d with one to four words absent" % (d["not_verbatim_all_words_present"], d["not_verbatim_words_absent"])),
         (none, "#ECEBE8", "#9A9893", "No quotation: MedDx reported that the chart held nothing on alcohol; the ground truth was indeterminate for all %d" % none)]
for k, (n, fc, ec, lab) in enumerate(rows3):
    yk = 27.0 - k * 7.5
    axk = fig.add_axes([7.0 / W, (yk - 1.6) / H, 3.2 / W, 3.2 / H]); axk.axis("off"); axk.set_xlim(0, 1); axk.set_ylim(0, 1)
    axk.add_patch(Rectangle((0.05, 0.05), 0.9, 0.9, facecolor=fc, edgecolor=ec, lw=0.6))
    fig.text(12.0 / W, yk / H, "%d" % n, fontsize=S.FS_BODY, fontweight="bold", color=INK, va="center", ha="left")
    fig.text(19.0 / W, yk / H, lab, fontsize=S.FS_SMALL, color=INK, va="center", ha="left", wrap=True)
title(fig, W, H, "Fidelity check: %d of %d quotations verbatim (%.1f%%)" % (exact, d["quoting"], d["pct"]), 48.5)
save(fig, "panel_c_waffle", W, H)

# ---- e: heatmap ----
t = pd.read_csv("figures/out/fig4/panel_trigger_section_positive.csv", index_col=0)
ROWN = {"no standalone trigger": "Other evidence, outside the rules", "explicit aud diagnosis": "Explicit AUD diagnosis",
        "alcoholic cirrhosis": "Alcoholic cirrhosis", "alcohol withdrawal / ciwa": "Alcohol withdrawal"}
ORDER = ["explicit aud diagnosis", "alcoholic cirrhosis", "alcohol withdrawal / ciwa", "no standalone trigger"]
t = t.loc[ORDER]; cols = [c for c in t.columns if c != "row_n"]
W, H = 83.0, 52.0
fig = plt.figure(figsize=S.mm(W, H)); fig.set_layout_engine("none")
ax = fig.add_axes([38.0 / W, 16.0 / H, 44.0 / W, 24.0 / H])
nr, nc = len(t), len(cols)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("plum", ["#FFFFFF", "#E4DAEE", "#5B2A86"])
M = np.array([[t.loc[i, c] / t.loc[i, "row_n"] for c in cols] for i in t.index])
ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
for r in range(nr):
    for c in range(nc):
        v = int(t.iloc[r][cols[c]]); share = M[r, c]
        ax.text(c, r, "%d" % v, ha="center", va="center", fontsize=S.FS_SMALL, color="white" if share > 0.55 else INK)
ax.set_xticks(range(nc)); ax.set_xticklabels([NAME[c] for c in cols], rotation=30, ha="right", rotation_mode="anchor", fontsize=S.FS_SMALL)
ax.set_yticks(range(nr)); ax.set_yticklabels(["%s (%d)" % (ROWN[i], int(t.loc[i, "row_n"])) for i in t.index], fontsize=S.FS_SMALL)
ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_xticks(np.arange(-0.5, nc, 1), minor=True); ax.set_yticks(np.arange(-0.5, nr, 1), minor=True)
ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0)
fig.text(60.0 / W, 2.5 / H, "Note type or section cited", ha="center", va="center", fontsize=S.FS_SMALL, color=MUTED)
title(fig, W, H, "AUD+ patients (n = 250): rule finding reported,", 49.0)
fig.text(7.0 / W, 45.0 / H, "by note type or section cited. Number, patients; shade, share of the row", fontsize=S.FS_SMALL, color=MUTED, va="center")
save(fig, "panel_e_bubbles", W, H)

# ---- f: two mirrored butterflies ----
e = pd.read_csv(C + "/out/new/evidence_by_gt_site.csv"); e["lab"] = e["category"].map(NAME); e["diff"] = e["rate_AUD+"] - e["rate_AUD-"]
e = e.sort_values("diff", ascending=False).reset_index(drop=True)
nL, nR = int(e["N_AUD-"][0]), int(e["N_AUD+"][0]); nS, nA = int(e["N_AUD+_Safety net"][0]), int(e["N_AUD+_Academic"][0])
def butterfly(ax, df, lcol, rcol, lc, rc, lhead, rhead, labels):
    ys = np.arange(len(df))[::-1].astype(float)
    for yy, (_, r) in zip(ys, df.iterrows()):
        l, rr = 100 * r[lcol], 100 * r[rcol]
        ax.barh(yy, -l, height=0.62, color=lc, lw=0); ax.barh(yy, rr, height=0.62, color=rc, lw=0)
        ax.text(-l - 1.5, yy, "%.0f" % l, ha="right", va="center", fontsize=S.FS_SMALL - 0.5, color=INK)
        ax.text(rr + 1.5, yy, "%.0f" % rr, ha="left", va="center", fontsize=S.FS_SMALL - 0.5, color=INK)
    ax.axvline(0, color=INK, lw=0.5); ax.set_xlim(-112, 112); ax.set_ylim(-0.7, len(df) - 0.3)
    ax.set_xticks([-100, -50, 0, 50, 100]); ax.set_xticklabels(["100", "50", "0", "50", "100"], fontsize=S.FS_SMALL)
    ax.set_yticks(ys); ax.set_yticklabels(list(df["lab"]) if labels else [], fontsize=S.FS_SMALL); ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.text(-54, len(df) - 0.1, lhead, ha="center", va="bottom", fontsize=S.FS_SMALL, color=INK, linespacing=1.0)
    ax.text(54, len(df) - 0.1, rhead, ha="center", va="bottom", fontsize=S.FS_SMALL, color=INK, linespacing=1.0)
W, H = 100.0, 52.0
fig = plt.figure(figsize=S.mm(W, H)); fig.set_layout_engine("none")
ax1 = fig.add_axes([27.0 / W, 11.0 / H, 33.0 / W, 29.0 / H])
butterfly(ax1, e, "rate_AUD-", "rate_AUD+", GREEN, PLUM, "AUD−\n(n = %d)" % nL, "AUD+\n(n = %d)" % nR, True)
ax2 = fig.add_axes([65.0 / W, 11.0 / H, 33.0 / W, 29.0 / H])
butterfly(ax2, e, "rate_AUD+_Safety net", "rate_AUD+_Academic", SAFETY, ACAD, "Safety-net\nhospital (n = %d)" % nS, "Academic\nsystem (n = %d)" % nA, False)
ax1.text(-112, len(e) - 0.1, "Note type or section", ha="right", va="bottom", fontsize=S.FS_SMALL, color=MUTED, transform=ax1.transData, clip_on=False)
fig.text(43.5 / W, 4.6 / H, "By ground-truth class, all systems", ha="center", va="center", fontsize=S.FS_SMALL, color=INK)
fig.text(81.5 / W, 4.6 / H, "AUD+ patients, by hospital system", ha="center", va="center", fontsize=S.FS_SMALL, color=INK)
fig.text(62.5 / W, 1.8 / H, "Patients whose classification cited it (%)", ha="center", va="center", fontsize=S.FS_SMALL, color=MUTED)
title(fig, W, H, "Evidence is spread over the chart, and the safety-net hospital differs", 49.0)
save(fig, "panel_f_evidence", W, H)

# ---- g: robustness, compressed ----
acc = pd.read_csv(C + "/out/new/reader_accuracy.csv").set_index("tag")
md = pd.read_csv("downstream/npj/out/figure4/model_disparity.csv")[["reader", "tag"]].drop_duplicates(); T2N = dict(zip(md["tag"], md["reader"]))
lead = pd.read_csv("downstream/npj/out/figure4/panel_a_leaderboard.csv").sort_values("kappa_w", ascending=False)
N2T = {v: k for k, v in T2N.items()}
rows = [("MedDx: skill, temperature 0", "aud_qwen36_27b", True), ("Temperature 0.3", "goal5_qwen_v1_t0p3", False), ("Temperature 0.7", "goal5_qwen_v1_t0p7", False), ("Paraphrased skill A", "goal5_qwen_paraphrase_a_t0p0", False), ("Paraphrased skill B", "goal5_qwen_paraphrase_b_t0p0", False)]
rows += [(n, N2T[n], False) for n in lead["name"] if n != "Qwen3.6-27B"]
assert all(t in acc.index for _, t, _ in rows), [t for _, t, _ in rows if t not in acc.index]
code = acc.loc["code"]
W, H = 98.0, 54.0
fig = plt.figure(figsize=S.mm(W, H)); fig.set_layout_engine("none")
for k, (metric, x0, lab, xlim) in enumerate([("sens", 33.0, "Sensitivity (%), 250 GT+", (40, 100)), ("spec", 68.0, "Specificity (%), 173 GT−", (40, 100))]):
    ax = fig.add_axes([x0 / W, 8.0 / H, 24.0 / W, 39.0 / H])
    ys = []; y = 0.0
    for i, (label, tag, bold) in enumerate(rows):
        if i == 5: y -= 0.6
        r = acc.loc[tag]; v, lo, hi = r[metric], r[metric + "_lo"], r[metric + "_hi"]
        cc = S.C_INDEX if i < 5 else ENG
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", ms=2.6, lw=0.7, capsize=1.2, color=cc, mfc=cc if i < 5 else "white", mec=cc, mew=0.7, zorder=3)
        ax.text(1.03, y, "%.1f" % v, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=S.FS_SMALL - 0.5, color=INK)
        ys.append(y); y -= 1.0
    ax.axvline(code[metric], color=S.C_CODE, lw=0.8, zorder=1); ax.axvspan(code[metric + "_lo"], code[metric + "_hi"], color=S.TINT[S.C_CODE], lw=0, zorder=0)
    ax.text(code[metric], ys[0] + 0.8, "Diagnosis %.1f" % code[metric], ha="center", va="bottom", fontsize=S.FS_SMALL - 0.5, color=S.C_CODE)
    ax.axhline(ys[4] - 0.8, color=S.RULE, lw=0.5)
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 1.8); ax.set_xlim(*xlim); ax.set_xticks([40, 60, 80, 100]); ax.tick_params(axis="x", labelsize=S.FS_SMALL - 0.5)
    ax.set_yticks(ys); ax.set_yticklabels([l for l, _, _ in rows] if k == 0 else [], fontsize=S.FS_SMALL - 0.5); ax.tick_params(axis="y", length=0)
    if k == 0: ax.get_yticklabels()[0].set_fontweight("bold")
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.set_xlabel(lab, fontsize=S.FS_SMALL - 0.5, labelpad=2)
title(fig, W, H, "Performance robustness across engines and skill variations", 51.5)
save(fig, "panel_g_robustness", W, H)

# ---- h: matrix, larger ----
subprocess.run([sys.executable, C + "/panel_model_matrix_v15.py", "--out", OUT], check=True)
