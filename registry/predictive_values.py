"""Registry-weighted predictive values of MedDx and of the AUD diagnosis (Fig. 1f and the Results sentence beside it).

The study cohort was drawn in roughly equal numbers from registry patients with and without an AUD diagnosis, so rates
computed on the reviewed charts without weights do not describe the registry. Each chart with an AUD+ or AUD- ground
truth carries the weight of the registry estimate (hospital type by whole-record diagnosis status). The positive
predictive value is the weighted share of charts classified AUD+ whose ground truth is AUD+, the negative predictive
value the weighted share of those classified AUD- whose ground truth is AUD-; the four weighted rates of classification
draw the curves of Fig. 1f. Intervals: percentile bootstrap, 2,000 resamples of the charts within cells.
Run from the analysis root on the cluster after registry/source_table.py. Aggregates only.
"""
import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "downstream"); sys.path.insert(0, "downstream/npj")
import registry_frame as FS
m = FS.build(anchor="keep"); m["pid"] = m["person_id"].astype("int64")
src = pd.read_csv("output/registry/source_2026-10-26.csv")
t = src.merge(m[["pid", "U", "aud", "alc"]].rename(columns={"pid": "omop_person_id"}), on="omop_person_id")
t["h"] = t["hospital_type_cascade"]; t["s"] = np.where(t["aud"].notna(), 1, np.where(t["alc"].notna(), 2, 3))
cells = ["h", "s"]; N = t.groupby(cells).size().rename("N")
r = t[(t.reviewed == 1) & (t.ground_truth != "indet")].copy(); r["pos"] = (r.ground_truth == "AUD+").astype(int)


def stats(rv):
    n = rv.groupby(cells).size(); rv = rv.merge((N / n).rename("w").reset_index(), on=cells); w = rv.w; pos = rv.pos == 1; o = {}
    for rd, P, Ng in (("MedDx", rv.llm_classification == "AUD+", rv.llm_classification == "AUD-"), ("diagnosis", rv.U == 1, rv.U == 0)):
        o["ppv_" + rd] = float((w * (P & pos)).sum() / (w * P).sum()); o["npv_" + rd] = float((w * (Ng & ~pos)).sum() / (w * Ng).sum())
        o["tpr_" + rd] = float((w * (P & pos)).sum() / (w * pos).sum()); o["fpr_" + rd] = float((w * (P & ~pos)).sum() / (w * ~pos).sum())
        o["fnr_" + rd] = float((w * (Ng & pos)).sum() / (w * pos).sum()); o["tnr_" + rd] = float((w * (Ng & ~pos)).sum() / (w * ~pos).sum())
    return o


pt = stats(r); rng = np.random.default_rng(2026)
B = pd.DataFrame([stats(r.groupby(cells, group_keys=False).apply(lambda d: d.sample(len(d), replace=True, random_state=int(rng.integers(1e9))))) for _ in range(2000)])
out = {k: {"point": pt[k], "lo": float(np.nanpercentile(B[k], 2.5)), "hi": float(np.nanpercentile(B[k], 97.5))} for k in pt}
json.dump(out, open("output/registry/predictive_values.json", "w"), indent=1)
for k in ("ppv_MedDx", "npv_MedDx", "ppv_diagnosis", "npv_diagnosis"): print("%-14s %.3f (%.3f-%.3f)" % (k, out[k]["point"], out[k]["lo"], out[k]["hi"]))
