#!/usr/bin/env python3
"""Agreement between two readings of the same chart (Fig. 4e): the two hepatologists, MedDx against each, and MedDx against itself under two temperatures and two paraphrases of the skill.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "downstream")
import stats_core as sc                                              # noqa: E402
from analyze_goal5_robustness import kappa3, pairs3                   # noqa: E402

OUT = "figures/out/stability"
os.makedirs(OUT, exist_ok=True)
POS, NEG = ("Definitely Yes", "Probably Yes"), ("Definitely No", "Probably No")
r = pd.read_csv("output/raters/rater_long.csv", dtype=str)
r = r[r["in_analysis_cohort"].eq("True")]
W = r.pivot_table(index="omop_person_id", columns="rater", values="label_6", aggfunc="first")
assert len(W) == 485, len(W)
BASE = "goal5_qwen_v1_t0p0"
assert (W[BASE] == W["aud_qwen36_27b"]).all(), "goal5 baseline differs from the primary run"


def bucket(v):
    v = np.asarray(v, dtype=object)
    return np.where(np.isin(v, POS), "+", np.where(np.isin(v, NEG), "-", "i"))


PAIRS = [("LLM with itself", "T = 0.3", BASE, "goal5_qwen_v1_t0p3"),
         ("LLM with itself", "T = 0.7", BASE, "goal5_qwen_v1_t0p7"),
         ("LLM with itself", "Paraphrase A", BASE, "goal5_qwen_paraphrase_a_t0p0"),
         ("LLM with itself", "Paraphrase B", BASE, "goal5_qwen_paraphrase_b_t0p0"),
         ("Two hepatologists", "M.A. and S.N.", "maaz", "sai"),
         ("LLM and a hepatologist", "LLM and M.A.", BASE, "maaz"),
         ("LLM and a hepatologist", "LLM and S.N.", BASE, "sai")]
# the hepatologist comparisons sit on the charts S.N. read, so all three share one set
SN = set(W.index[W["sai"].notna()])
rng = np.random.default_rng(42)
rows = []
for group, name, x, y in PAIRS:
    d = W[[x, y]].dropna()
    d = d[d.index.isin(SN)]          # every pair on the same 186 charts
    a, b = d[x].to_numpy(object), d[y].to_numpy(object)
    n = len(d)
    same = bucket(a) == bucket(b)
    idx = rng.integers(0, n, size=(2000, n))
    k_boot = np.array([kappa3(a[i], b[i]) for i in idx])
    s_boot = same[idx].mean(axis=1)
    rows.append({"group": group, "pair": name, "n": n, "n_same": int(same.sum()),
                 "pct_same": 100 * same.mean(),
                 "pct_lo": 100 * np.percentile(s_boot, 2.5), "pct_hi": 100 * np.percentile(s_boot, 97.5),
                 "kappa": kappa3(a, b), "k_n": pairs3(a, b),
                 "k_lo": np.nanpercentile(k_boot, 2.5), "k_hi": np.nanpercentile(k_boot, 97.5)})
t = pd.DataFrame(rows)
t.to_csv(os.path.join(OUT, "stability.csv"), index=False)
print(t.to_string(index=False, float_format=lambda v: "%.3f" % v))

# paired difference on the charts S.N. read, with S.N.'s independent first pass as the yardstick
# (as in f01_kappa_inventory: M.A. against S.N., LLM against S.N.): kappa(LLM, S.N.) minus
# kappa(M.A., S.N.)
d = W[[BASE, "maaz", "sai"]].dropna()
L, M, S = (d[c].to_numpy(object) for c in (BASE, "maaz", "sai"))
n = len(d)
idx = np.random.default_rng(7).integers(0, n, size=(2000, n))
diff = np.array([kappa3(L[i], S[i]) - kappa3(M[i], S[i]) for i in idx])
pt = kappa3(L, S) - kappa3(M, S)
lo, hi = np.nanpercentile(diff, [2.5, 97.5])
pd.DataFrame([{"n": n, "kappa_llm_sn": kappa3(L, S), "kappa_ma_sn": kappa3(M, S),
               "difference": pt, "ci_low": lo, "ci_high": hi}]).to_csv(
    os.path.join(OUT, "paired_llm_vs_fellow.csv"), index=False)
print("\npaired, against S.N., LLM minus M.A., %d charts: %.3f (%.3f to %.3f)" % (n, pt, lo, hi))
