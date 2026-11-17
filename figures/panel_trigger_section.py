#!/usr/bin/env python3
"""Findings by note type or section, the table behind Extended Data Fig. 6.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

import npj_style as S
import f05_supp_alluvial as A

W_MM, H_MM = 119.0, 58.0
X_AX0, AX_W = 33.0, 82.0
Y_AX0, AX_H = 19.0, 30.0
SHOW_T = {"no standalone trigger": "No sufficient finding",
          "explicit aud diagnosis": "Explicit AUD diagnosis",
          "alcoholic cirrhosis": "Alcoholic cirrhosis",
          "alcohol withdrawal / ciwa": "Alcohol withdrawal"}
SHOW_E = {"social history": "Social history", "hpi narrative": "HPI narrative",
          "problem list": "Problem list", "discharge diagnosis": "Discharge diagnosis",
          "assessment/plan": "Assessment/plan", "structured field": "Structured field",
          "discharge summary": "Discharge summary", "hospital course": "Hospital course",
          "progress note": "Progress note"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel", default="output/panel.csv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    # v8: reference-positive patients only (the author, 2 Oct). Over all 487 the rows were
    # close to the reference categories themselves, so the panel repeated e; restricted to
    # the 251 with AUD it shows what each positive classification rested on, including the
    # patients with no sufficient finding.
    import pandas as pd
    from collections import Counter
    r = pd.read_csv("output/raters/rater_long.csv", dtype=str)
    gold = r[r["in_analysis_cohort"].eq("True") & r["rater"].eq("gold")].set_index("omop_person_id")["label_3"]
    p = pd.read_csv(a.panel, dtype={"omop_person_id": str})
    p = p[p["model"].eq(A.PRIMARY_TAG) & p["omop_person_id"].isin(gold.index[gold.eq("AUD+")])]
    c, ltrig, levid = Counter(), Counter(), Counter()
    grade_none = Counter()
    for _, x in p.iterrows():
        tr = sorted(set(A.parse_list(x["rule_findings"]))) or [A.NONE_TRIGGER]
        ev = sorted(set(A.parse_list(x["evidence_categories"])))
        for t in tr:
            ltrig[t] += 1
            for e in ev:
                c[(t, e)] += 1
        for e in ev:
            levid[e] += 1
        if tr == [A.NONE_TRIGGER]:
            grade_none[x["model_label"]] += 1
    n = len(p)
    assert n == 250, n
    evids = [e for e, _ in levid.most_common(8)]
    trigs = [A.NONE_TRIGGER] + [t for t, _ in ltrig.most_common() if t != A.NONE_TRIGGER]
    M = np.array([[c.get((t, e), 0) for e in evids] for t in trigs])
    rown = np.array([ltrig[t] for t in trigs])
    share = M / rown[:, None]
    print("reference-positive patients", n, "| rows", dict(zip(trigs, rown)), "| sections", evids)
    print("LLM grade of those with no sufficient finding:", dict(grade_none))
    print(M)
    pd.DataFrame(M, index=trigs, columns=evids).assign(row_n=rown).to_csv(
        os.path.join(a.out, "panel_trigger_section_positive.csv"))
    S.apply_npj_style()
    fig = plt.figure(figsize=S.mm(W_MM, H_MM))
    fig.set_layout_engine("none")
    ax = fig.add_axes([X_AX0 / W_MM, Y_AX0 / H_MM, AX_W / W_MM, AX_H / H_MM])
    cmap = LinearSegmentedColormap.from_list("idx", ["#FFFFFF", S.RAMP_INDEX[3], S.RAMP_INDEX[1]])
    ax.imshow(share, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, "%d" % v, ha="center", va="center", fontsize=S.FS_SMALL - 0.5,
                    color="white" if share[i, j] > 0.6 else S.INK)
    ax.set_xticks(range(len(evids)))
    ax.set_xticklabels([SHOW_E.get(e, e) for e in evids], fontsize=S.FS_SMALL - 0.5,
                       rotation=35, ha="right", rotation_mode="anchor", color=S.INK)
    ax.set_yticks(range(len(trigs)))
    labs = ax.set_yticklabels(["%s (%d)" % (SHOW_T.get(t, t), k) for t, k in zip(trigs, rown)], fontsize=S.FS_SMALL, color=S.INK)
    for t, lab in zip(trigs, labs):
        if t == A.NONE_TRIGGER:
            lab.set_fontweight("bold")
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(evids), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(trigs), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    fig.text(X_AX0 / W_MM - 0.005, (Y_AX0 + AX_H + 2.0) / H_MM, "Sufficient finding", fontsize=S.FS_BODY,
             ha="right", va="bottom", color=S.INK)
    fig.text((X_AX0 + AX_W / 2) / W_MM, (Y_AX0 + AX_H + 2.0) / H_MM,
             "Evidence category the LLM reported (patients with AUD)", fontsize=S.FS_BODY, ha="center", va="bottom",
             color=S.INK)
    S.check_text_fits(fig, W_MM, H_MM)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "panel_trigger_section." + ext), dpi=600,
                    bbox_inches=None, pad_inches=0, facecolor="white")
    print("wrote panel_trigger_section at %.0f x %.0f mm" % (W_MM, H_MM))


if __name__ == "__main__":
    main()
