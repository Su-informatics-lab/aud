#!/usr/bin/env python3
"""The AUD diagnosis as a classifier: an AUD diagnosis code at any time in the registry record.

One definition for every analysis that scores the AUD diagnosis, stratifies by it or uses it as a
feature: registry_frame.build()'s U (condition source values matching registry_frame.AUD_CODE,
dated 2005 or later), read from the table registry/source_table.py writes
(output/registry/source_2026-10-26.csv, column aud_icd_any_time).

The roster's has_aud_icd is the flag the study cohort was sampled on. It stays where it describes
the sampling design (the sampling strata, their weights and the resampling within them) and is not
the classifier.
"""
import pandas as pd

SRC = "output/registry/source_2026-10-26.csv"
COL = "aud_dx"


def flag():
    """Series indexed by omop_person_id (str): 1 with an AUD diagnosis at any time, 0 without."""
    s = pd.read_csv(SRC, usecols=["omop_person_id", "aud_icd_any_time", "reviewed"],
                    dtype={"omop_person_id": str})
    if s["omop_person_id"].duplicated().any() or s["aud_icd_any_time"].isna().any():
        raise SystemExit("%s: duplicated patients or a missing AUD diagnosis flag" % SRC)
    if int((s["reviewed"] == 1).sum()) != 485:
        raise SystemExit("%s does not hold the 485 reviewed charts" % SRC)
    return s.set_index("omop_person_id")["aud_icd_any_time"].astype(int)


def attach(df, key="omop_person_id", col=COL):
    """Return df with the registry flag as column `col` (float; missing outside the registry frame).

    Every reviewed chart must be matched: a key of the wrong type would otherwise turn the whole
    cohort into patients without the diagnosis at the next fillna(0).
    """
    f = flag()
    f.index = f.index.astype("int64")
    k = pd.to_numeric(df[key], errors="coerce")
    out = df.copy()
    out[col] = k.map(f).astype(float).to_numpy()
    s = pd.read_csv(SRC, usecols=["omop_person_id", "reviewed"])
    rev = set(s.loc[s["reviewed"] == 1, "omop_person_id"].astype("int64"))
    inrev = k.isin(rev)
    if inrev.any() and out.loc[inrev, col].isna().any():
        raise SystemExit("a reviewed chart has no registry AUD diagnosis flag")
    return out
