#!/usr/bin/env python3
"""Sensitivity and specificity of MedDx and of the AUD diagnosis by patient group, on the reviewed charts.
An indeterminate output counts against the classifier that gave it.
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, "downstream")
sys.path.insert(0, "downstream/npj")
import stats_core as sc                                                   # noqa: E402
import f04_sdi as F4                                                      # noqa: E402
import f06_subgroup_frame as SF                                           # noqa: E402

RATERS = "output/raters/rater_long.csv"
ROSTER = "resources/patient_roster.csv"
OUT = "downstream/npj/out/figure4"
PRIMARY = "aud_qwen36_27b"

# The axis in the frame, its heading, and its levels in drawing order.
AXES = [
    ("site", "Hospital system", ["Academic", "Safety net", "Other hospitals"]),
    ("gender", "Sex", ["Men", "Women"]),
    ("race", "Race", ["Other than White", "White"]),
    ("agegrp", "Age", ["Under 50", "50 to 64", "65 and over"]),
    ("residence", "Residence", ["Urban", "Suburban", "Rural"]),
    ("sdi_q", "Neighbourhood deprivation", F4.Q5),
]
RELABEL = {"Man": "Men", "Woman": "Women", "Non-White": "Other than White",
           "urban": "Urban", "suburban": "Suburban", "rural": "Rural",
           "Other": "Other hospitals"}


def charts():
    r = pd.read_csv(RATERS, dtype=str)
    r = r[r["in_analysis_cohort"].eq("True")]
    w = r.pivot_table(index="omop_person_id", columns="rater", values="label_3",
                      aggfunc="first")
    d = pd.DataFrame({"pid": w.index.astype(str), "gold": w["gold"].values,
                      "llm": w[PRIMARY].values})
    ros = pd.read_csv(ROSTER, dtype={"omop_person_id": str})
    d = d.merge(ros[["omop_person_id", "has_aud_icd"]].rename(
        columns={"omop_person_id": "pid"}), on="pid", how="left")
    import aud_dx as DX
    d["code"] = DX.attach(d, key="pid")["aud_dx"].eq(1).to_numpy()
    d["person_id"] = d["pid"].astype("int64")
    return d


def attach(d):
    s, _, _ = SF.build_axes(SimpleNamespace(panel=SF.PANEL))
    s = s[["pid", "site", "gender", "race_2", "ruca_category", "age"]].copy()
    s["person_id"] = s["pid"].astype("int64")
    d = d.merge(s.drop(columns=["pid"]), on="person_id", how="left", validate="one_to_one")
    d["site"] = d["site"].replace(RELABEL)
    d["gender"] = d["gender"].replace(RELABEL)
    d["race"] = d["race_2"].replace(RELABEL)
    d["residence"] = d["ruca_category"].replace(RELABEL)
    d["agegrp"] = np.where(d["age"] < 50, "Under 50",
                  np.where(d["age"] < 65, "50 to 64", "65 and over"))
    ref = F4.sdi_reference()
    zp = F4.zip5_by_person()
    d = F4.attach(d, ref, zp)
    for col, name, levels in AXES:
        miss = int((~d[col].isin(levels)).sum())
        if miss:
            print("  %s: %d charts outside the drawn levels" % (name, miss))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = attach(charts())
    pos = d[d["gold"].eq("AUD+")].copy()
    neg = d[d["gold"].eq("AUD-")].copy()
    pos["code_ok"] = pos["code"].astype(bool)
    pos["llm_ok"] = pos["llm"].eq("AUD+")
    neg["code_ok"] = ~neg["code"].astype(bool)
    neg["llm_ok"] = neg["llm"].eq("AUD-")
    print("  reference-positive %d, reference-negative %d, indeterminate %d"
          % (len(pos), len(neg), len(d) - len(pos) - len(neg)))
    print("  pooled: sensitivity code %.4f model %.4f; specificity code %.4f model %.4f"
          % (pos["code_ok"].mean(), pos["llm_ok"].mean(),
             neg["code_ok"].mean(), neg["llm_ok"].mean()))

    rows = []
    for col, name, levels in AXES:
        for lvl in levels:
            rec = {"axis": name, "level": lvl}
            for tag, frame in (("sens", pos), ("spec", neg)):
                sub = frame[frame[col].eq(lvl)]
                rec["n_" + tag] = len(sub)
                for reader, flag in (("code", "code_ok"), ("llm", "llm_ok")):
                    n = len(sub)
                    k = int(sub[flag].sum()) if n else 0
                    if n == 0:
                        rec["%s_%s" % (tag, reader)] = np.nan
                        rec["%s_%s_lo" % (tag, reader)] = np.nan
                        rec["%s_%s_hi" % (tag, reader)] = np.nan
                        continue
                    lo, hi = sc.wilson_ci(k, n)
                    rec["%s_%s" % (tag, reader)] = 100.0 * k / n
                    rec["%s_%s_lo" % (tag, reader)] = 100.0 * lo
                    rec["%s_%s_hi" % (tag, reader)] = 100.0 * hi
            rows.append(rec)
            print("    %-26s %-17s sens n %3d code %5.1f model %5.1f | "
                  "spec n %3d code %5.1f model %5.1f"
                  % (name, lvl, rec["n_sens"], rec["sens_code"], rec["sens_llm"],
                     rec["n_spec"], rec["spec_code"], rec["spec_llm"]))

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.out, "subgroup_rates.csv"), index=False)
    print("\n  lowest sensitivity: code %.1f, model %.1f" % (t["sens_code"].min(),
                                                             t["sens_llm"].min()))
    print("  lowest specificity: code %.1f, model %.1f" % (t["spec_code"].min(),
                                                            t["spec_llm"].min()))
    print("  strata with under 20 charts: sensitivity %d, specificity %d of %d"
          % ((t["n_sens"] < 20).sum(), (t["n_spec"] < 20).sum(), len(t)))
    print("  wrote %s/subgroup_rates.csv" % a.out)


if __name__ == "__main__":
    main()
