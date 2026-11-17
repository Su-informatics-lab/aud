"""One row per registry patient: t0, hospital type, whole-record AUD diagnosis, stratum at t0 and, for the
study cohort, the ground truth and MedDx's classification.

Hospital type is assigned from the sites of a patient's inpatient and emergency visits: safety-net hospital if
any visit was there, otherwise academic health system if any was there, otherwise other hospitals. The table is
the input of every script in registry/. It is written on the cluster and stays there.
"""
import sys, numpy as np, pandas as pd, warnings, os; warnings.filterwarnings("ignore")
sys.path.insert(0, "downstream"); sys.path.insert(0, "downstream/npj")
import registry_frame as FS
D = "/N/project/depot/hw56/aud/cirrhosis_regv20/data_tables_v1/parquet"
m = FS.build(anchor="keep"); m["pid"] = m["person_id"].astype("int64")
vo = pd.read_parquet(D + "/visit_occurrence.parquet", columns=["person_id", "visit_concept_id", "care_site_id"])
cs = pd.read_parquet(D + "/care_site.parquet", columns=["care_site_id", "care_site_name", "care_site_source_value"])
cs["care_site_id"]=pd.to_numeric(cs["care_site_id"],errors="coerce").astype("Int64"); vo["care_site_id"]=pd.to_numeric(vo["care_site_id"],errors="coerce").astype("Int64")
cs["inst"] = pd.to_numeric(cs["care_site_source_value"].str.extract(r"INST#(\d+)")[0], errors="coerce")
cs["lab"] = cs["care_site_name"].str.contains(r"\bLAB\b|LABORATORY|XRAY|X-RAY|RADIOLOG|IMAGING|\bMRI\b|\bCT\b|ULTRASOUND|PHARMACY", case=False, na=False)
vo = vo.merge(cs[["care_site_id", "inst", "lab"]], on="care_site_id", how="left"); vo["lab"] = vo["lab"].fillna(False).astype(bool)
vo = vo[vo["visit_concept_id"].isin([9201, 9203]) & ~vo["lab"]]; vo["person_id"] = vo["person_id"].astype("int64")
vo["site"] = np.where(vo["inst"] == 1, 0, np.where(vo["inst"] == 2, 1, 2))
site = vo.groupby("person_id")["site"].min().map({0: "Safety-net hospital", 1: "Academic health system", 2: "Other hospitals"})
g = pd.read_csv("resources/gold_reference_v3.csv", dtype=str); g["pid"] = g["omop_person_id"].astype("int64")
p = pd.read_csv("output/panel.csv", dtype=str); p = p[p["model"] == "aud_qwen36_27b"]
g["qwen36_27b"] = g["omop_person_id"].map(p.set_index("omop_person_id")["model_label"])
POS={"Definitely Yes","Probably Yes"}; NEG={"Definitely No","Probably No"}
g["llm"] = np.where(g["qwen36_27b"].isin(POS),"AUD+",np.where(g["qwen36_27b"].isin(NEG),"AUD-","indet"))
g["gt"] = g["gold_three"].replace({"indeterminate":"indet","Indeterminate":"indet"})
t = pd.DataFrame({"omop_person_id": m["pid"], "t0_first_cirrhosis_diagnosis": pd.to_datetime(m["t0"]).dt.strftime("%Y-%m-%d"),
                  "hospital_type_cascade": m["pid"].map(site).fillna("Other hospitals"),
                  "aud_icd_any_time": m["U"].astype(int), "stratum_at_t0": m["S"].map(dict(enumerate(["AUD diagnosis","Other alcohol-related diagnosis","No alcohol-related diagnosis"])))})
t = t.merge(g[["pid","gt","llm"]].rename(columns={"pid":"omop_person_id","gt":"ground_truth","llm":"llm_classification"}), on="omop_person_id", how="left")
t["reviewed"] = t["ground_truth"].notna().astype(int)
t["ground_truth"] = t["ground_truth"].fillna("NA"); t["llm_classification"] = t["llm_classification"].fillna("NA")
out = "output/registry/source_2026-10-26.csv"; os.makedirs(os.path.dirname(out), exist_ok=True)
t.to_csv(out, index=False); os.chmod(out, 0o660); os.chmod(os.path.dirname(out), 0o770)
print(out, t.shape); print(t["reviewed"].value_counts().to_dict())
print(pd.crosstab(t["hospital_type_cascade"], t["aud_icd_any_time"], margins=True).to_string())
print(t[t.reviewed==1]["ground_truth"].value_counts().to_dict(), t[t.reviewed==1]["llm_classification"].value_counts().to_dict())
