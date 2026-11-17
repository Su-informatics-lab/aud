"""Registry patients with a first record of a medication before t0, by stratum at t0 (Supplementary Fig. 4).

AUD-specific medication (the prior users of the treatment analysis), gabapentin, baclofen and topiramate; each drug is
its RxNorm ingredient with descendants. The counts are entered in figures/supp_medication_before_diagnosis.py.
Run from the analysis root on the cluster. Aggregates only.
"""
import sys, json, pandas as pd
sys.path.insert(0, "downstream"); sys.path.insert(0, "downstream/npj")
import registry_frame as FS
D = FS.D; m = FS.build(anchor="keep")[["person_id", "t0", "S", "rx_before_t0"]]; m["person_id"] = m["person_id"].astype("int64")
anc = pd.read_parquet(D + "/concept_ancestor.parquet", columns=["ancestor_concept_id", "descendant_concept_id"]).astype("int64")
de = pd.read_parquet(D + "/drug_exposure.parquet", columns=["person_id", "drug_concept_id", "drug_exposure_start_date"])
de["person_id"] = de["person_id"].astype("int64"); de["drug_concept_id"] = de["drug_concept_id"].astype("int64")
de["d"] = pd.to_datetime(de["drug_exposure_start_date"], errors="coerce"); de = de[de["d"].notna() & (de["d"] >= FS.MIN_DATE)]
out = {"registry_patients": [int((m.S == s).sum()) for s in range(3)], "AUD-specific medication": [int(m[m.S == s].rx_before_t0.sum()) for s in range(3)]}
for name, cid in (("Gabapentin", 797399), ("Baclofen", 715233), ("Topiramate", 742267)):
    desc = set(anc.loc[anc["ancestor_concept_id"] == cid, "descendant_concept_id"]) | {cid}
    first = de[de["drug_concept_id"].isin(desc)].groupby("person_id")["d"].min().rename("first")
    j = m.merge(first, left_on="person_id", right_index=True, how="inner"); pr = j[j["first"] < j["t0"]]
    out[name] = [int((pr.S == s).sum()) for s in range(3)]
json.dump(out, open("output/registry/medication_before_t0.json", "w"), indent=1); print(json.dumps(out))
