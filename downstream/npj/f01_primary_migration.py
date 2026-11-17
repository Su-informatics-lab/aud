#!/usr/bin/env python3
"""Every agreement value under the primary definition: three classes, a chart dropped when either classifier
returned not enough data, unweighted Cohen's kappa with a percentile bootstrap over patients.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "downstream")
sys.path.insert(0, "downstream/npj")

import stats_core as sc                                            # noqa: E402
import ref_standard as RS                                          # noqa: E402
import registry_frame as B                                      # noqa: E402

OUT = "downstream/npj/out/primary"
PANEL = "output/panel.csv"
RATERS = "output/raters/rater_long.csv"
INST = "output/fairness/patient_institution.csv"

POS = {"Definitely Yes", "Probably Yes"}
NEG = {"Definitely No", "Probably No"}
NED = "Not Enough Data"
ORD3 = ["AUD-", "Uncertain", "AUD+"]
# The ordering the manuscript has carried until now, for the before column.
NED_LAST = ["Definitely Yes", "Probably Yes", "Uncertain", "Probably No",
            "Definitely No", "Not Enough Data"]
N_BOOT, SEED = 2000, 20260814

DISPLAY = {
    "aud_qwen36_27b": "Qwen3.6-27B", "aud_nemotron_fp8": "Nemotron-3-120B",
    "aud_oss120b": "GPT-OSS-120B", "aud_glm52": "GLM-5.2",
    "aud_nemotron_llama": "Nemotron-Llama-49B", "aud_gptoss_20b": "GPT-OSS-20B",
    "aud_deepseek_v4_flash_0731": "DeepSeek-V4-Flash", "aud_gemma4": "Gemma4-31B",
    "aud_llama4_scout": "Llama-4-Scout", "aud_mistral32_24b": "Mistral-3.2-24B",
    "aud_llama33_70b": "Llama-3.3-70B",
}
PRIMARY = "Qwen3.6-27B"


def collapse(s):
    """Three categories, preserving absence. Not enough data becomes absent, so the
    pair-wise filter drops the chart rather than scoring it."""
    out = pd.Series(np.where(s.isin(POS), "AUD+",
                    np.where(s.isin(NEG), "AUD-",
                    np.where(s.eq("Uncertain"), "Uncertain", None))), index=s.index)
    return out.where(s.notna() & s.ne(NED))


def boot(fn, a, b):
    rng = np.random.default_rng(SEED)
    a, b = np.asarray(a), np.asarray(b)
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(a), len(a))
        try:
            out.append(fn(a[i], b[i]))
        except Exception:
            continue
    return np.nanpercentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def primary(a6, b6, weights=None):
    a, b = collapse(a6), collapse(b6)
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    k = sc.cohens_kappa(a, b, ORD3, weights)
    lo, hi = boot(lambda x, y: sc.cohens_kappa(x, y, ORD3, weights), a, b)
    return len(a), k, lo, hi


def before(a6, b6):
    m = a6.notna() & b6.notna()
    return int(m.sum()), sc.cohens_kappa(a6[m], b6[m], NED_LAST, "linear")


def load(a):
    ref, man = RS.build()
    assert len(ref) == man["cohort"]["analysed"] == 485
    ref["omop_person_id"] = ref["omop_person_id"].astype(str)
    ref["pid"] = pd.to_numeric(ref["omop_person_id"], errors="coerce").astype("int64")

    p = pd.read_csv(a.panel, dtype={"omop_person_id": str}, low_memory=False)
    p = p[p["model"].isin(DISPLAY)].copy()
    p["name"] = p["model"].map(DISPLAY)
    w = p.pivot(index="omop_person_id", columns="name", values="model_label")
    d = ref.set_index("omop_person_id").join(w)

    inst = pd.read_csv(a.inst)
    inst["omop_person_id"] = inst["person_id"].astype(str)
    d = d.join(inst.set_index("omop_person_id")["institution"])

    reg = B.build(); reg["pid"] = reg["person_id"].astype("int64")
    d = d.reset_index().merge(reg[["pid", "t0"]], on="pid", how="left")
    d["period"] = pd.cut(pd.to_datetime(d["t0"]).dt.year, [2014, 2016, 2018, 2021],
                         labels=["2015-2016", "2017-2018", "2019-2020"])

    r = pd.read_csv(a.raters, dtype={"omop_person_id": str})
    r = r[r["in_analysis_cohort"].astype(str).str.lower().eq("true")]
    r6 = r.pivot_table(index="omop_person_id", columns="rater", values="label_6",
                       aggfunc="first")
    d = d.merge(r6[["maaz", "sai"]], left_on="omop_person_id", right_index=True,
                how="left")
    SITE = {"IUH": "academic", "Eskenazi": "safety net"}
    d["site"] = d["institution"].map(SITE).fillna("other hospitals")
    return d, man


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--raters", default=RATERS)
    ap.add_argument("--inst", default=INST)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d, man = load(a)
    g = d["gold_label"]
    rows = []

    def add(where, what, x, y):
        n0, k0 = before(x, y)
        n1, k1, lo, hi = primary(x, y)
        _, kw, _, _ = primary(x, y, "linear")
        rows.append({"where": where, "quantity": what,
                     "n_before": n0, "kappa_before": k0,
                     "n_primary": n1, "kappa_primary": k1, "lo": lo, "hi": hi,
                     "kappa_primary_weighted": kw})

    for name in sorted(DISPLAY.values()):
        add("Fig 4a, Supp Table 3", name, g, d[name])
    add("Abstract, Findings 1", "index test against the reference", g, d[PRIMARY])
    add("Findings 1, Supp Table 8", "M.A. against S.N.", d["maaz"], d["sai"])
    add("Findings 1, Supp Table 8", "index test against S.N.", d[PRIMARY], d["sai"])
    for s in ("academic", "safety net", "other hospitals"):
        m = d["site"].eq(s)
        add("Fig 2a, Supp Table 9", "index test, %s" % s, g[m], d.loc[m, PRIMARY])
    rf = d["gold_source"].ne("maaz_unreviewed")
    add("Fig 4c", "never referred", g[~rf], d.loc[~rf, PRIMARY])
    add("Fig 4c", "referred", g[rf], d.loc[rf, PRIMARY])
    for band in ("2015-2016", "2017-2018", "2019-2020"):
        m = d["period"].eq(band)
        add("Supp Fig 8", "index test, %s" % band, g[m], d.loc[m, PRIMARY])

    t = pd.DataFrame(rows)
    t["delta"] = t["kappa_primary"] - t["kappa_before"]
    t.to_csv(os.path.join(a.out, "primary_migration.csv"), index=False)

    show = t.copy()
    for c in ("kappa_before", "kappa_primary", "lo", "hi",
              "kappa_primary_weighted", "delta"):
        show[c] = show[c].round(3)
    print(show[["where", "quantity", "n_before", "kappa_before", "n_primary",
                "kappa_primary", "lo", "hi", "delta"]].to_string(index=False))
    print()
    print("unweighted is the reported primary; the weighted column is in the csv")
    print("charts dropped because a reader marked not enough data: %d of 487 for the "
          "index test against the reference"
          % (487 - int(t.loc[t["quantity"].eq("index test against the reference"),
                             "n_primary"].iloc[0])))
    print()
    lb = t[t["where"].eq("Fig 4a, Supp Table 3")].copy()
    lb["r_before"] = lb["kappa_before"].rank(ascending=False)
    lb["r_after"] = lb["kappa_primary"].rank(ascending=False)
    moved = lb[lb["r_before"].ne(lb["r_after"])]
    print("leaderboard: %d of 11 change rank" % len(moved))
    if len(moved):
        print(moved[["quantity", "kappa_before", "r_before",
                     "kappa_primary", "r_after"]].round(3).to_string(index=False))
    print("\nwrote %s/primary_migration.csv" % a.out)


if __name__ == "__main__":
    main()
