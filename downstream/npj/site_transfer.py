#!/usr/bin/env python3
"""Supervised comparators trained at the academic health system only and applied to the safety-net hospital
and the other hospitals (Fig. 3e; Supplementary Table 8), beside MedDx and the AUD diagnosis on the same charts.
"""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "downstream"); sys.path.insert(0, "downstream/npj")
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import ref_standard as R, registry_frame as B
SEED = 20250823
ref, _ = R.build(); ref["omop_person_id"] = ref["omop_person_id"].astype(str)
ros = pd.read_csv("resources/patient_roster.csv", dtype={"omop_person_id": str})
pan = pd.read_csv("output/panel.csv", dtype={"omop_person_id": str}); pan = pan[pan["model"] == "aud_qwen36_27b"][["omop_person_id", "model_label"]]
feat = pd.read_parquet("output/features_structured.parquet"); feat["omop_person_id"] = feat["person_id"].astype("int64").astype(str)
inst = pd.read_csv("output/fairness/patient_institution.csv", dtype=str).rename(columns={"person_id": "omop_person_id"})
d = (ref.merge(ros[["omop_person_id", "has_aud_icd", "n_notes"]], on="omop_person_id").merge(pan, on="omop_person_id")
        .merge(feat, on="omop_person_id").merge(inst[["omop_person_id", "institution"]], on="omop_person_id", how="left"))
import aud_dx as DX
det = DX.attach(d[d["gold_three"].isin(["AUD+", "AUD-"])].copy().reset_index(drop=True))
y = (det["gold_three"] == "AUD+").astype(int).to_numpy(); site = det["institution"].to_numpy()
m = B.build(0)
co = pd.read_parquet(B.D + "/condition_occurrence.parquet", columns=["person_id", "condition_source_value"])
co["icd"] = co["condition_source_value"].astype(str).str.extract(r"\^\^([^\^]+)\^")[0]
co.loc[co["icd"].isna(), "icd"] = co.loc[co["icd"].isna(), "condition_source_value"].astype(str); ic = co["icd"].astype(str)
any_alc = set(co.loc[ic.str.match(B.ALC_SIGNAL, na=False), "person_id"].astype("int64")); ald = set(co.loc[ic.str.match(B.NARROW_SIGNAL, na=False), "person_id"].astype("int64"))
nvis = m.set_index("person_id")["nvis"]; pid = det["omop_person_id"].astype("int64")
det["any_alcohol_code"] = pid.isin(any_alc).astype(int); det["alcoholic_liver_disease"] = pid.isin(ald).astype(int); det["n_visits"] = pid.map(nvis).fillna(0)
base = ["age", "CCI", "n_visits", "n_notes", "any_alcohol_code", "alcoholic_liver_disease"]
charl = [c for c in feat.columns if c not in ("person_id", "omop_person_id", "age", "gender", "race", "ethnicity", "care_site_id", "CCI")]
X = det[base + charl].copy()
for c in ("gender", "race", "ethnicity"): X = pd.concat([X, pd.get_dummies(det[c].astype(str), prefix=c)], axis=1)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).assign(aud_code=(det["aud_dx"] == 1).astype(int))
notes = pickle.loads(Path("output/npj_baselines/notes_full.pkl").read_bytes())
texts = np.array([notes.get(i, "") for i in det["omop_person_id"]], dtype=object)
has_text = np.array([len(t) > 0 for t in texts])
grid_x = {"max_depth": [2, 3], "n_estimators": [150, 300], "learning_rate": [0.05, 0.1]}
inner = StratifiedKFold(3, shuffle=True, random_state=SEED); outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
def wil(k, n):
    if n == 0: return (np.nan, np.nan, np.nan)
    z = 1.959964; p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h
def score(rows, reader, setting, yhat, mask):
    for s in ("IUH", "Eskenazi", "Other"):
        mm = mask & (site == s); pos = mm & (y == 1); neg = mm & (y == 0)
        se = wil(int((yhat[pos] == 1).sum()), int(pos.sum())); sp = wil(int((yhat[neg] == 0).sum()), int(neg.sum()))   # an indeterminate output (-1) counts against the reader
        rows.append(dict(reader=reader, setting=setting, site=s, n_pos=int(pos.sum()), n_neg=int(neg.sum()), sens=se[0], sens_lo=se[1], sens_hi=se[2], spec=sp[0], spec_lo=sp[1], spec_hi=sp[2]))
rows = []
tr = site == "IUH"
# XGBoost: fit on IUH (grid by inner CV), threshold at IUH prevalence quantile of the IUH out-of-fold scores
gs = GridSearchCV(XGBClassifier(eval_metric="logloss", random_state=SEED, tree_method="hist", n_jobs=1), grid_x, cv=inner, scoring="roc_auc", n_jobs=1)
p_oof = cross_val_predict(gs, X[tr], y[tr], cv=outer, method="predict_proba", n_jobs=1)[:, 1]
thr = float(np.quantile(p_oof, 1 - y[tr].mean()))
gs.fit(X[tr], y[tr]); p_all = gs.predict_proba(X)[:, 1]
yhat = np.zeros(len(det), int); yhat[tr] = (p_oof >= thr).astype(int); yhat[~tr] = (p_all[~tr] >= thr).astype(int)
score(rows, "XGBoost, with F10.x", "trained on IUH", yhat, np.ones(len(det), bool))
# TF-IDF logistic regression on notes: same recipe, C by inner CV
tr_t = tr & has_text
pipe = Pipeline([("v", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=3, max_features=200000, lowercase=True)), ("m", LogisticRegression(max_iter=2000))])
gs_t = GridSearchCV(pipe, {"m__C": [0.1, 0.3, 1.0, 3.0, 10.0]}, cv=inner, scoring="roc_auc", n_jobs=1)
p_oof_t = cross_val_predict(gs_t, texts[tr_t], y[tr_t], cv=outer, method="predict_proba", n_jobs=1)[:, 1]
thr_t = float(np.quantile(p_oof_t, 1 - y[tr_t].mean()))
gs_t.fit(texts[tr_t], y[tr_t]); p_all_t = gs_t.predict_proba(texts[has_text])[:, 1]
yhat_t = np.zeros(len(det), int); tmp = np.zeros(has_text.sum(), int)
idx_tr_in_text = tr[has_text]; tmp[idx_tr_in_text] = (p_oof_t >= thr_t).astype(int); tmp[~idx_tr_in_text] = (p_all_t[~idx_tr_in_text] >= thr_t).astype(int)
yhat_t[has_text] = tmp
score(rows, "TF-IDF logistic regression", "trained on IUH", yhat_t, has_text)
# references on the same charts
score(rows, "AUD diagnosis (F10.x)", "no training", (det["aud_dx"] == 1).astype(int).to_numpy(), np.ones(len(det), bool))
lab3 = np.where(det["model_label"].isin(["Definitely Yes", "Probably Yes"]), 1, np.where(det["model_label"].isin(["Definitely No", "Probably No"]), 0, -1))
score(rows, "MedDx", "no training", lab3, np.ones(len(det), bool))
out = pd.DataFrame(rows); out.to_csv("downstream/npj/out/figure4/site_transfer.csv", index=False)
print("charts with note text:", int(has_text.sum()), "of", len(det), "| IUH training charts:", int(tr.sum()), "(text:", int(tr_t.sum()), ")")
pd.set_option("display.width", 220); print(out.round(3).to_string(index=False))
