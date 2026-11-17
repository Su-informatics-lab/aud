#!/usr/bin/env python3
"""Each patient group against the rest of its variable, for both classifiers and both rates (Fig. 3a;
Supplementary Table 6). The quantity is the difference in percentage points, with a percentile bootstrap
resampled within sampling stratum.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "downstream")
sys.path.insert(0, "downstream/npj")
from icd_fairness import NBOOT, SEED                                      # noqa: E402
import f04_subgroup_rates as SR                                           # noqa: E402

OUT = "downstream/npj/out/figure4"



def level_rates(codes, ok, k):
    """Mean of ok within each of k integer level codes; nan where the level is empty."""
    n = np.bincount(codes, minlength=k).astype(float)
    s = np.bincount(codes, weights=ok, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan)


def vs_complement(codes, ok, k):
    """Rate in each level minus the rate over the other levels of the same axis."""
    n = np.bincount(codes, minlength=k).astype(float)
    s = np.bincount(codes, weights=ok, minlength=k)
    tot_n, tot_s = n.sum(), s.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        inside = np.where(n > 0, s / n, np.nan)
        outside = np.where(tot_n - n > 0, (tot_s - s) / (tot_n - n), np.nan)
    return 100.0 * (inside - outside)


def differences(frame, col, levels, flags, rng):
    """{reader: (diff array, lo array, hi array)} against the rest of the axis."""
    idx = {lvl: i for i, lvl in enumerate(levels)}
    f = frame[frame[col].isin(levels)]
    codes = f[col].map(idx).to_numpy().astype(int)
    k = len(levels)
    out = {}
    for reader, flag in flags:
        ok = f[flag].to_numpy(dtype=float)
        obs = vs_complement(codes, ok, k)
        boots = np.empty((NBOOT, k))
        n = len(ok)
        for b in range(NBOOT):
            j = rng.integers(0, n, n)
            boots[b] = vs_complement(codes[j], ok[j], k)
        lo = np.nanpercentile(boots, 2.5, axis=0)
        hi = np.nanpercentile(boots, 97.5, axis=0)
        out[reader] = (obs, lo, hi)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = SR.attach(SR.charts())
    pos = d[d["gold"].eq("AUD+")].copy()
    neg = d[d["gold"].eq("AUD-")].copy()
    pos["code_ok"] = pos["code"].astype(bool)
    pos["llm_ok"] = pos["llm"].eq("AUD+")
    neg["code_ok"] = ~neg["code"].astype(bool)
    neg["llm_ok"] = neg["llm"].eq("AUD-")
    flags = [("code", "code_ok"), ("llm", "llm_ok")]

    rows = []
    for col, name, levels in SR.AXES:
        rng = np.random.default_rng(SEED)
        got = {"sens": differences(pos, col, levels, flags, rng),
               "spec": differences(neg, col, levels, flags, rng)}
        print("\n  %s, each level against the rest of the axis" % name)
        for i, lvl in enumerate(levels):
            rec = {"axis": name, "level": lvl, "reference": "rest of the axis",
                   "n_sens": int(pos[col].eq(lvl).sum()),
                   "n_spec": int(neg[col].eq(lvl).sum())}
            for tag in ("sens", "spec"):
                for reader, _ in flags:
                    obs, lo, hi = got[tag][reader]
                    rec["%s_%s" % (tag, reader)] = obs[i]
                    rec["%s_%s_lo" % (tag, reader)] = lo[i]
                    rec["%s_%s_hi" % (tag, reader)] = hi[i]
            rows.append(rec)
            print("    %-17s sensitivity code %+6.1f (%+6.1f to %+6.1f)  "
                  "model %+5.1f (%+5.1f to %+5.1f) | specificity code %+6.1f "
                  "(%+6.1f to %+6.1f)  model %+5.1f (%+5.1f to %+5.1f)"
                  % (lvl, rec["sens_code"], rec["sens_code_lo"], rec["sens_code_hi"],
                     rec["sens_llm"], rec["sens_llm_lo"], rec["sens_llm_hi"],
                     rec["spec_code"], rec["spec_code_lo"], rec["spec_code_hi"],
                     rec["spec_llm"], rec["spec_llm_lo"], rec["spec_llm_hi"]))

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.out, "subgroup_disparity.csv"), index=False)
    nz = t
    print("\n  strata %d" % len(nz))
    for tag in ("sens", "spec"):
        for reader in ("code", "llm"):
            v = nz["%s_%s" % (tag, reader)].abs()
            excl = ((nz["%s_%s_lo" % (tag, reader)] > 0)
                    | (nz["%s_%s_hi" % (tag, reader)] < 0)).sum()
            print("  %-4s %-4s largest absolute difference %5.1f points; intervals "
                  "excluding zero %d" % (tag, reader, v.max(), excl))
    print("  wrote %s/subgroup_disparity.csv" % a.out)


if __name__ == "__main__":
    main()
