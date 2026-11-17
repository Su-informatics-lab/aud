#!/usr/bin/env python3
"""Which findings MedDx cites, by ground-truth class. A classification may cite more than one finding, so
citations exceed classifications; the count and the rate per classification are both written.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "downstream")
sys.path.insert(0, str(HERE))

from f05_figure5 import TOPCATS, parse_cats, PRIMARY_TAG                # noqa: E402

PANEL = "output/panel.csv"
RATERS = "output/raters/rater_long.csv"
OUT = "downstream/npj/out/figure3"
GROUPS = ["AUD+", "AUD-", "Indeterminate"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in (PANEL, RATERS):
        if not Path(f).exists():
            raise SystemExit("%s is missing" % f)

    r = pd.read_csv(RATERS, dtype=str)
    r = r[r["in_analysis_cohort"].eq("True") & r["rater"].eq("gold")]
    gold = r.set_index("omop_person_id")["label_3"]

    p = pd.read_csv(PANEL, dtype={"omop_person_id": str})
    p = p[p["model"].eq(PRIMARY_TAG)].set_index("omop_person_id")
    cats = p["evidence_categories"].reindex(gold.index)
    if cats.isna().all():
        raise SystemExit("no evidence_categories for %s" % PRIMARY_TAG)

    rows = []
    for pid, g in gold.items():
        for c in parse_cats(cats.get(pid)):
            if c in TOPCATS:
                rows.append({"category": c, "reference": g})
    ct = (pd.DataFrame(rows).groupby(["category", "reference"]).size()
          .unstack(fill_value=0).reindex(columns=GROUPS, fill_value=0)
          .reindex(TOPCATS).fillna(0).astype(int))
    n = gold.value_counts().reindex(GROUPS).astype(int)

    out = ct.copy()
    out.columns = ["n_" + c for c in GROUPS]
    for c in GROUPS:
        out["rate_" + c] = (ct[c] / n[c]).round(4)
    out["total"] = ct.sum(axis=1)
    out.to_csv(os.path.join(a.out, "evidence_categories.csv"))
    pd.DataFrame({"reference": n.index, "classifications": n.values}).to_csv(
        os.path.join(a.out, "evidence_categories_n.csv"), index=False)

    print("classifications by reference label: %s"
          % ", ".join("%s %d" % (k, v) for k, v in n.items()))
    print("citations %d over %d classifications, %.2f per classification"
          % (int(ct.values.sum()), int(n.sum()), ct.values.sum() / n.sum()))
    print()
    print(pd.concat([ct, ct.sum(axis=1).rename("total")], axis=1).to_string())
    print()
    print("citations per classification:")
    print(out[["rate_" + c for c in GROUPS]].to_string())
    print()
    for c in ("structured field", "social history", "hpi narrative"):
        row = ct.loc[c]
        rate = {g: row[g] / n[g] for g in GROUPS}
        print("  %-18s counts %s | per classification %s"
              % (c, dict(row), {k: round(v, 3) for k, v in rate.items()}))
    print("wrote %s/evidence_categories.csv" % a.out)


if __name__ == "__main__":
    main()
