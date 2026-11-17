#!/usr/bin/env python3
"""The chart-level patient-group table: the grouping variables come from icd_fairness.build(), so age, sex
and race have one definition across the chart-scale and registry-scale analyses.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "downstream")
import stats_core as sc                                                 # noqa: E402
from icd_fairness import build                                          # noqa: E402

RATERS = "output/raters/rater_long.csv"
ROSTER = "resources/patient_roster.csv"
EXISTING = "downstream/npj/out/subgroups/subgroup_feasibility.csv"
OUT = "downstream/npj/out/subgroups"
PRIMARY = "aud_qwen36_27b"
# The axis in m, the name for the output, and the display order of its levels.
AXES = [("sex", "Gender", ["Male", "Female"]),
        ("race", "Race, two levels", ["Other than White", "White"]),
        ("agegrp", "Age bracket", ["Under 50", "50 to 64", "65 and over"])]
# What the existing table calls the same levels, for the reproduction check.
OLD = {("Gender", "Male"): ("Gender", "Man"),
       ("Gender", "Female"): ("Gender", "Woman"),
       ("Race, two levels", "Other than White"): ("Race, two levels", "Non-White"),
       ("Race, two levels", "White"): ("Race, two levels", "White")}


def charts():
    r = pd.read_csv(RATERS, dtype=str)
    r = r[r["in_analysis_cohort"].eq("True")]
    w = r.pivot_table(index="omop_person_id", columns="rater", values="label_3",
                      aggfunc="first")
    if PRIMARY not in w.columns or "gold" not in w.columns:
        raise SystemExit("rater_long.csv lacks gold or %s" % PRIMARY)
    d = pd.DataFrame({"pid": w.index.astype(str), "gold": w["gold"].values,
                      "llm": w[PRIMARY].values})
    ros = pd.read_csv(ROSTER, dtype={"omop_person_id": str})
    d = d.merge(ros[["omop_person_id", "has_aud_icd"]].rename(
        columns={"omop_person_id": "pid"}), on="pid", how="left")
    import aud_dx as DX
    d["code"] = DX.attach(d, key="pid")["aud_dx"].eq(1).to_numpy()
    print("analysis cohort %d charts, reference-positive %d"
          % (len(d), int(d["gold"].eq("AUD+").sum())))
    return d


def row(sub, axis_name, level):
    """Sensitivity of each reader on the reference-positive charts of one stratum."""
    pos = sub[sub["gold"].eq("AUD+")]
    n = len(pos)
    if n == 0:
        return None
    kc = int(pos["code"].sum())
    kl = int(pos["llm"].eq("AUD+").sum())
    c_lo, c_hi = sc.wilson_ci(kc, n)
    l_lo, l_hi = sc.wilson_ci(kl, n)
    half = 100.0 * (c_hi - c_lo) / 2.0
    mc = sc.mcnemar(np.ones(n, dtype=bool), pos["llm"].eq("AUD+").to_numpy(),
                    pos["code"].to_numpy())
    # The rule is stated here rather than inherited: a stratum can support a 10-point
    # comparison when the half-width of the code's Wilson interval is at most 10 points.
    # On this rule "Other than White" does not clear it, at 10.2, where the table with no
    # producer labelled it as clearing. Its numbers reproduce exactly; only the label moves.
    verdict = ("too few positives to estimate" if n < 20 else
               "can support a 10-point comparison" if half <= 10.0 else
               "estimable, interval too wide to compare")
    return {"axis": axis_name, "level": level, "source": "structured",
            "charts": len(sub), "reference_positive": n,
            "code_sens": 100.0 * kc / n, "code_lo": 100.0 * c_lo,
            "code_hi": 100.0 * c_hi, "code_halfwidth": half,
            "llm_sens": 100.0 * kl / n, "llm_lo": 100.0 * l_lo,
            "llm_hi": 100.0 * l_hi, "gain": 100.0 * (kl - kc) / n,
            "llm_only": mc["b"], "code_only": mc["c"],
            "discordant": mc["b"] + mc["c"], "p_paired": mc["p_value"],
            "paired_method": mc["method"], "verdict": verdict}


def main():
    os.makedirs(OUT, exist_ok=True)
    d = charts()
    m = build()
    m["pid"] = m["person_id"].astype("int64").astype(str)
    keep = ["pid", "sex", "race", "agegrp"]
    d = d.merge(m[keep], on="pid", how="left")
    outside = int(d["sex"].isna().sum())
    print("charts outside the registry frame: %d" % outside)

    rows = []
    for col, name, levels in AXES:
        for lvl in levels:
            r = row(d[d[col].eq(lvl)], name, lvl)
            if r:
                rows.append(r)
        r = row(d[d[col].isna()], name, "outside the registry frame")
        if r:
            rows.append(r)
    t = pd.DataFrame(rows)
    t["recomputed"] = True

    # Everything this script does not recompute is carried forward verbatim from the table
    # with no producer, marked as carried. Its method is confirmed: sex and race reproduce
    # to zero below, on the same inputs and the same estimand. Panels read one file.
    old_all = pd.read_csv(EXISTING)
    done = {a for _, a, _ in AXES}
    carried = old_all[~old_all["axis"].isin(done)].copy()
    carried["recomputed"] = False
    for c in ("llm_only", "code_only", "discordant", "p_paired", "paired_method"):
        if c not in carried.columns:
            carried[c] = np.nan
    t = pd.concat([t, carried[t.columns]], ignore_index=True)
    t.to_csv(os.path.join(OUT, "subgroup_feasibility_v2.csv"), index=False)
    print("recomputed %d rows on %d axes; carried %d rows on %d axes forward"
          % (int(t["recomputed"].sum()), len(AXES), int((~t["recomputed"]).sum()),
             carried["axis"].nunique()))
    print()
    r_ = t[t["recomputed"]]
    print(r_[["axis", "level", "charts", "reference_positive", "code_sens", "llm_sens",
              "gain", "llm_only", "code_only", "p_paired", "paired_method",
              "verdict"]].round(3).to_string(index=False))
    print()
    print("paired comparison, LLM against the code on the same reference-positive charts:")
    for _, r in t[t["recomputed"]].iterrows():
        if r["level"] == "outside the registry frame":
            continue
        print("  %-18s %-18s gain %+5.1f points; the LLM alone found %d, the code alone "
              "%d, %s p = %.3f" % (r["axis"], r["level"], r["gain"], r["llm_only"],
                                   r["code_only"], r["paired_method"], r["p_paired"]))
    print()
    print("verdicts that differ from the table with no producer:")
    old_v = pd.read_csv(EXISTING).set_index(["axis", "level"])["verdict"].to_dict()
    for _, r in t[t["recomputed"]].iterrows():
        k = OLD.get((r["axis"], r["level"]))
        if k and k in old_v and old_v[k] != r["verdict"]:
            print("  %s %s: was %r, now %r" % (r["axis"], r["level"], old_v[k],
                                               r["verdict"]))

    print()
    print("reproduction check against the table with no producer:")
    old = pd.read_csv(EXISTING).set_index(["axis", "level"])
    worst = 0.0
    for _, r in t[t["recomputed"]].iterrows():
        k = OLD.get((r["axis"], r["level"]))
        if k is None or k not in old.index:
            continue
        o = old.loc[k]
        for f in ("charts", "reference_positive", "code_sens", "llm_sens"):
            diff = abs(float(r[f]) - float(o[f]))
            worst = max(worst, diff)
            flag = "" if diff < 1e-6 else "   DIFFERS"
            print("  %-18s %-18s %-18s %14.6f vs %14.6f%s"
                  % (r["axis"], r["level"], f, float(r[f]), float(o[f]), flag))
    print("largest absolute difference on the reproduced axes: %.6f" % worst)
    print()
    print("age, old cut against new:")
    for lvl in ("under 40", "40 to 64", "65 and over"):
        if ("Age bracket", lvl) in old.index:
            o = old.loc[("Age bracket", lvl)]
            print("  old  %-12s charts %4d  positives %4d  code %.1f  llm %.1f"
                  % (lvl, o["charts"], o["reference_positive"], o["code_sens"],
                     o["llm_sens"]))
    for _, r in t[t["axis"].eq("Age bracket") & t["recomputed"]].iterrows():
        print("  new  %-12s charts %4d  positives %4d  code %.1f  llm %.1f"
              % (r["level"], r["charts"], r["reference_positive"], r["code_sens"],
                 r["llm_sens"]))
    print()
    print("wrote %s/subgroup_feasibility_v2.csv" % OUT)


if __name__ == "__main__":
    main()
