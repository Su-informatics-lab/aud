#!/usr/bin/env python3
"""The gradient-boosted comparator over the structured record, by hospital system. Refits the comparator
(with and without the AUD diagnosis) exactly as structured_baselines.py does and reports sensitivity and
specificity by hospital system with Wilson 95% CI, beside the AUD diagnosis and MedDx on the same charts.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "downstream"); sys.path.insert(0, "downstream/npj")
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from xgboost import XGBClassifier
import ref_standard as R
SEED = 20250823
ref, _ = R.build(); ref["omop_person_id"] = ref["omop_person_id"].astype(str)
ros = pd.read_csv("resources/patient_roster.csv", dtype={"omop_person_id": str})
pan = pd.read_csv("output/panel.csv", dtype={"omop_person_id": str})
pan = pan[pan["model"] == "aud_qwen36_27b"][["omop_person_id", "model_label"]]
feat = pd.read_parquet("output/features_structured.parquet"); feat["omop_person_id"] = feat["person_id"].astype("int64").astype(str)
inst = pd.read_csv("output/fairness/patient_institution.csv", dtype=str).rename(columns={"person_id": "omop_person_id"})
d = (ref.merge(ros[["omop_person_id", "has_aud_icd", "n_notes"]], on="omop_person_id").merge(pan, on="omop_person_id")
        .merge(feat, on="omop_person_id").merge(inst[["omop_person_id", "institution"]], on="omop_person_id", how="left"))
import aud_dx as DX
det = DX.attach(d[d["gold_three"].isin(["AUD+", "AUD-"])].copy().reset_index(drop=True))
y = (det["gold_three"] == "AUD+").astype(int).to_numpy()
# alcohol flags and visit counts exactly as structured_baselines.py builds them
import registry_frame as B
m = B.build(0)
co = pd.read_parquet(B.D + "/condition_occurrence.parquet", columns=["person_id", "condition_source_value"])
co["icd"] = co["condition_source_value"].astype(str).str.extract(r"\^\^([^\^]+)\^")[0]
co.loc[co["icd"].isna(), "icd"] = co.loc[co["icd"].isna(), "condition_source_value"].astype(str)
ic = co["icd"].astype(str)
any_alc = set(co.loc[ic.str.match(B.ALC_SIGNAL, na=False), "person_id"].astype("int64"))
ald = set(co.loc[ic.str.match(B.NARROW_SIGNAL, na=False), "person_id"].astype("int64"))
nvis = m.set_index("person_id")["nvis"]
pid = det["omop_person_id"].astype("int64")
det["any_alcohol_code"] = pid.isin(any_alc).astype(int); det["alcoholic_liver_disease"] = pid.isin(ald).astype(int)
det["n_visits"] = pid.map(nvis).fillna(0)
base = ["age", "CCI", "n_visits", "n_notes", "any_alcohol_code", "alcoholic_liver_disease"]
charl = [c for c in feat.columns if c not in ("person_id", "omop_person_id", "age", "gender", "race", "ethnicity", "care_site_id", "CCI")]
X = det[base + charl].copy()
for c in ("gender", "race", "ethnicity"):
    X = pd.concat([X, pd.get_dummies(det[c].astype(str), prefix=c)], axis=1)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
grid = {"max_depth": [2, 3], "n_estimators": [150, 300], "learning_rate": [0.05, 0.1]}
outer = StratifiedKFold(5, shuffle=True, random_state=SEED); inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
def wil(k, n):
    if n == 0: return (np.nan, np.nan, np.nan)
    z = 1.959964; p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h
preds = {}
for label, cols in (("XGBoost, with F10.x", X.assign(aud_code=(det["aud_dx"] == 1).astype(int))), ("XGBoost, without F10.x", X)):
    gs = GridSearchCV(XGBClassifier(eval_metric="logloss", random_state=SEED, tree_method="hist", n_jobs=1), grid, cv=inner, scoring="roc_auc", n_jobs=1)
    p = cross_val_predict(gs, cols, y, cv=outer, method="predict_proba", n_jobs=1)[:, 1]
    thr = float(np.quantile(p, 1 - y.mean())); preds[label] = (p >= thr).astype(int)
preds["AUD diagnosis (F10.x)"] = (det["aud_dx"] == 1).astype(int).to_numpy()
preds["MedDx"] = det["model_label"].isin(["Definitely Yes", "Probably Yes"]).astype(int).to_numpy()   # indeterminate counts against
rows = []
for reader, yhat in preds.items():
    for site in ("ALL", "IUH", "Eskenazi", "Other"):
        m = np.ones(len(det), bool) if site == "ALL" else (det["institution"] == site).to_numpy()
        pos = m & (y == 1); neg = m & (y == 0)
        s = wil(int(yhat[pos].sum()), int(pos.sum())); sp = wil(int((1 - yhat[neg]).sum()), int(neg.sum()))
        rows.append(dict(reader=reader, site=site, n_pos=int(pos.sum()), n_neg=int(neg.sum()), sens=s[0], sens_lo=s[1], sens_hi=s[2], spec=sp[0], spec_lo=sp[1], spec_hi=sp[2]))
out = pd.DataFrame(rows); out.to_csv("downstream/npj/out/figure4/supervised_by_site.csv", index=False)
pd.set_option("display.width", 200); print(out.round(3).to_string(index=False))
