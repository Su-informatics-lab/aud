#!/usr/bin/env python3
"""The registry frame: one row per registry patient, with t0, the strata and the medication dates.

build() returns every patient of the Indiana Cirrhosis Registry with a cirrhosis diagnosis:
t0 (the first cirrhosis diagnosis), the first AUD diagnosis and the first alcohol-related
diagnosis, the stratum at t0 (S), whether an AUD diagnosis appears at any time (U), the first
AUD-specific drug and whether it precedes t0, and, for the patients of the study cohort, the
ground truth. Every registry-scale analysis and every table that needs t0 calls it, so the
diagnosis code sets and the drug definition are stated once, here.

followup() adds the exit time and status used for the cumulative incidence of a first
AUD-specific drug after t0, with death as a competing event.

Run from the analysis root on the cluster; the registry tables are not in this repository.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "downstream")
sys.path.insert(0, "downstream/npj")

D = "/N/project/depot/hw56/aud/cirrhosis_regv20/data_tables_v1/parquet"
MIN_DATE = pd.Timestamp("2005-01-01")
AUD_CODE = r"^(F10|303|305\.0)"
CIRR_CODE = r"^(K70|K74|571)"
# t0 is the first code specific to cirrhosis: the registry's own cirrhosis codes (K70.30, K70.31,
# K71.7, K74.3 to K74.6) and their ICD-9 equivalents (571.2, 571.5, 571.6). A patient with none of
# them takes the first code of the wider families above.
CIRR_T0 = r"^(K70\.?3|K71\.?7|K74\.?[3-6]|571\.?2|571\.?5|571\.?6)"
# The alcohol-related diagnosis code set (Supplementary Table 15): alcohol-specific codes only.
ALC_SIGNAL = (r"^(K70|571\.0|571\.1|571\.2|571\.3|K85\.2|G62\.1|357\.5|"
              r"I42\.6|425\.5|K29\.2|535\.3|E24\.4|291|T51\.?0|T51\.?9|980\.?0|980\.?9|Z71\.4|"
              r"V11\.3|F10)")
NARROW_SIGNAL = r"^(K70|571\.0|571\.1|571\.2|571\.3)"
AUD_DRUGS = {1714319, 19043959, 735850}   # naltrexone, acamprosate, disulfiram (ingredients)
EXCLUDE_DRUG = {909841}
COMBO_INGREDIENTS = {1110410, 750982}   # morphine, bupropion (ingredients)
POS = {"definitely yes", "probably yes"}
NEG = {"definitely no", "probably no"}
AB = {"uncertain", "not enough data"}
PNAMES = ["2015-16", "2017-18", "2019+"]
SNAMES = ["AUD code", "alcohol coded only", "record silent"]


def cat(x):
    x = (x or "").strip().lower()
    return "AUD+" if x in POS else "AUD-" if x in NEG else "Ind" if x in AB else ""


def build(collapse=0, narrow=0, acamp=0, ever_rx=0, u_source="parquet",
          anchor="keep"):
    # The ground truth comes from ref_standard.build(), the one place it is derived.
    import ref_standard as R
    g, _ = R.build()
    ref = {int(pid): cat(lab) for pid, lab in
           zip(g["omop_person_id"].astype(str), g["gold_label"])
           if str(pid).strip().isdigit()}

    pe = pd.read_parquet(D + "/person.parquet",
                         columns=["person_id", "year_of_birth", "gender_concept_id"])
    pe["person_id"] = pe["person_id"].astype("int64")
    co = pd.read_parquet(D + "/condition_occurrence.parquet",
                         columns=["person_id", "condition_source_value", "condition_start_date"])
    co["person_id"] = co["person_id"].astype("int64")
    co["condition_start_date"] = pd.to_datetime(co["condition_start_date"], errors="coerce")
    co = co[co["condition_start_date"].notna() & (co["condition_start_date"] >= MIN_DATE)]
    co["icd"] = co["condition_source_value"].astype(str).str.extract(r"\^\^([^\^]+)\^")[0]
    co.loc[co["icd"].isna(), "icd"] = co.loc[co["icd"].isna(), "condition_source_value"].astype(str)
    ic = co["icd"].astype(str)
    f_fam = co[ic.str.match(CIRR_CODE, na=False)].groupby("person_id")["condition_start_date"].min()
    f_spec = co[ic.str.match(CIRR_T0, na=False)].groupby("person_id")["condition_start_date"].min()
    f_cirr = f_spec.reindex(f_fam.index).fillna(f_fam)
    f_aud = co[ic.str.match(AUD_CODE, na=False)].groupby("person_id")["condition_start_date"].min()
    sig = NARROW_SIGNAL if narrow else ALC_SIGNAL
    f_alc = co[ic.str.match(sig, na=False)].groupby("person_id")["condition_start_date"].min()
    f_ald = co[ic.str.match(r"^(K70|571\.0|571\.1|571\.2|571\.3)", na=False)].groupby(
        "person_id")["condition_start_date"].min()
    del co, ic

    m = pe.copy()
    m["t0"] = m["person_id"].map(f_cirr)
    m = m[m["t0"].notna()].copy()
    m["aud"] = m["person_id"].map(f_aud)
    m["alc"] = m["person_id"].map(f_alc)
    # Strata at t0: 0 AUD diagnosis, 1 other alcohol-related diagnosis, 2 neither.
    c1 = m["aud"].notna() & (m["aud"] <= m["t0"])
    c2 = (~c1) & m["alc"].notna() & (m["alc"] <= m["t0"])
    m["S"] = np.where(c1, 0, np.where(c2, 1, 2))
    if collapse:
        m["S"] = np.where(c1, 0, 1)
    fa = m["person_id"].map(f_ald)
    m["alcld"] = (fa.notna() & (fa <= m["t0"])).astype(float)
    m["age"] = m["t0"].dt.year - m["year_of_birth"]
    m["age"] = m["age"].fillna(m["age"].median())
    m["male"] = (m["gender_concept_id"] == 8507).astype(float)

    vo = pd.read_parquet(D + "/visit_occurrence.parquet",
                         columns=["person_id", "visit_start_date"])
    vo["person_id"] = vo["person_id"].astype("int64")
    vo["visit_start_date"] = pd.to_datetime(vo["visit_start_date"], errors="coerce")
    vo = vo[vo["visit_start_date"] >= MIN_DATE]
    j = vo.merge(m[["person_id", "t0"]], on="person_id", how="inner")
    m["nvis"] = m["person_id"].map(
        j[j["visit_start_date"] <= j["t0"]].groupby("person_id").size()).fillna(0)
    del vo, j

    de = pd.read_parquet(D + "/drug_exposure.parquet",
                         columns=["person_id", "drug_concept_id",
                                  "drug_exposure_start_date"])
    de["person_id"] = de["person_id"].astype("int64")
    de["drug_concept_id"] = de["drug_concept_id"].astype("int64")
    anc = pd.read_parquet(D + "/concept_ancestor.parquet",
                          columns=["ancestor_concept_id", "descendant_concept_id"])
    anc["ancestor_concept_id"] = anc["ancestor_concept_id"].astype("int64")
    conc = AUD_DRUGS | set(anc.loc[anc["ancestor_concept_id"].isin(AUD_DRUGS),
                                   "descendant_concept_id"].astype("int64"))
    conc -= EXCLUDE_DRUG
    # Combination products are not AUD treatment (morphine-naltrexone, bupropion-naltrexone):
    # every descendant of those co-ingredients is dropped.
    conc -= set(anc.loc[anc["ancestor_concept_id"].isin(COMBO_INGREDIENTS),
                        "descendant_concept_id"].astype("int64"))
    anc2 = anc
    del anc
    if acamp:
        seeds = {19043959, 735850}
        conc = seeds | set(anc2.loc[anc2["ancestor_concept_id"].isin(seeds),
                                    "descendant_concept_id"].astype("int64"))
        conc -= EXCLUDE_DRUG
        conc -= set(anc2.loc[anc2["ancestor_concept_id"].isin(COMBO_INGREDIENTS), "descendant_concept_id"].astype("int64"))
    # Receipt is dated relative to t0: Y is a first drug on or after t0, rx_before_t0 a first
    # drug before it (the prior users).
    de["d"] = pd.to_datetime(de["drug_exposure_start_date"], errors="coerce")
    de = de[de["d"].notna()]
    de = de[de["drug_concept_id"].isin(conc)]
    first = de.groupby("person_id")["d"].min().rename("first_rx")
    m["first_rx"] = m["person_id"].map(first)
    if ever_rx:
        m["Y"] = m["first_rx"].notna().astype(float)
    else:
        m["Y"] = ((m["first_rx"].notna()) & (m["first_rx"] >= m["t0"])).astype(float)
    m["rx_before_t0"] = ((m["first_rx"].notna()) & (m["first_rx"] < m["t0"])).astype(float)
    del de

    m["period"] = np.where(m["t0"].dt.year <= 2016, 0,
                           np.where(m["t0"].dt.year <= 2018, 1, 2))
    m["per1"] = (m["period"] == 1).astype(float)
    m["per2"] = (m["period"] == 2).astype(float)
    lab = m["person_id"].map(lambda p: ref.get(p, ""))
    m["Aobs"] = np.where(lab.eq("AUD+"), 1.0,
                         np.where(lab.eq("AUD-"), 0.0, np.nan))
    m["val"] = m["Aobs"].notna()
    # U: an AUD diagnosis at any time. Charts were sampled on it, while S is defined at t0.
    m["U"] = m["aud"].notna().astype(int)
    if u_source == "roster":
        r = pd.read_csv("resources/patient_roster.csv")
        r["pid"] = pd.to_numeric(r["omop_person_id"], errors="coerce")
        r = r[r["gpids_era_a"].notna() & r["pid"].notna()]
        h = dict(zip(r["pid"].astype("int64"), r["has_aud_icd"].astype(int)))
        u = m["person_id"].astype("int64").map(h)
        n_moved = int(((u.notna()) & (u != m["U"])).sum())
        m["U"] = u.fillna(m["U"]).astype(int)
        print("u_source=roster: %d sampled patients moved cell" % n_moved)

    # Uth equals U under anchor="keep", which every analysis in the paper uses.
    m["Uth"] = m["U"]
    if anchor != "keep":
        hit = m["val"].to_numpy(bool) & (m["S"].to_numpy(int) >= 1) \
            & (m["U"].to_numpy(int) == 1)
        n_hit = int(hit.sum())
        print("anchor=%s: %d validated charts sit in a cell whose window is centred on "
              "the AUD code that places them there" % (anchor, n_hit))
        for s_ in (1, 2):
            k = hit & (m["S"].to_numpy(int) == s_)
            print("  stratum %d %-20s %3d charts, %3d reference positive"
                  % (s_ + 1, SNAMES[s_], int(k.sum()),
                     int(np.nansum(m["Aobs"].to_numpy(float)[k]))))
        m.loc[hit, "Aobs"] = np.nan
        m["val"] = m["Aobs"].notna()
        if anchor == "asuncoded":
            m.loc[m["S"] >= 1, "Uth"] = 0
        elif anchor != "drop":
            raise SystemExit("unknown anchor mode: %s" % anchor)
    return m


def followup(m, censor):
    """Add exit time T (years since t0) and status E (1 rx, 2 death, 0 censored)."""
    de = pd.read_parquet(D + "/death.parquet", columns=["person_id", "death_date"])
    de["person_id"] = de["person_id"].astype("int64")
    de["death_date"] = pd.to_datetime(de["death_date"], errors="coerce")
    m["death"] = m["person_id"].map(de.groupby("person_id")["death_date"].min())
    last = []
    for tab, col in (("visit_occurrence", "visit_start_date"),
                     ("condition_occurrence", "condition_start_date"),
                     ("drug_exposure", "drug_exposure_start_date")):
        x = pd.read_parquet(D + "/%s.parquet" % tab, columns=["person_id", col])
        x["person_id"] = x["person_id"].astype("int64")
        x[col] = pd.to_datetime(x[col], errors="coerce")
        last.append(x.groupby("person_id")[col].max().rename(tab))
        del x
    m["last_rec"] = m["person_id"].map(pd.concat(last, axis=1).max(axis=1))
    admin = m["last_rec"].max()
    cens = (m["last_rec"].clip(upper=admin) if censor == "last_record"
            else pd.Series(admin, index=m.index))
    rx = m["first_rx"].where(m["first_rx"] >= m["t0"])
    dth = m["death"].where(m["death"] >= m["t0"])
    end = pd.concat([rx, dth, cens], axis=1).min(axis=1)
    m["E"] = np.where(rx.notna() & (rx <= end), 1,
                      np.where(dth.notna() & (dth <= end), 2, 0))
    # A death on the day of t0 would give zero exposure and an event: half a day at risk.
    m["T"] = np.maximum((end - m["t0"]).dt.days.to_numpy(float), 0.5) / 365.25
    return m, admin
