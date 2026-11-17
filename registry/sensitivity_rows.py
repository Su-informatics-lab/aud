"""The rows of Supplementary Table 16 and the sensitivities of Fig. 2c.

The registry estimate under one change at a time: indeterminate charts counted as AUD-, as AUD+, or at one
quarter, one half and three quarters of their cell's share with AUD; six cells in place of nine; and the hospital
type of a reviewed chart taken from the source of its notes. Run from the analysis root on the cluster after registry/source_table.py. Aggregates only.
"""
import sys, os, json, re, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"downstream"); sys.path.insert(0,"downstream/npj")
import registry_frame as FS
m=FS.build(anchor="keep"); m["pid"]=m["person_id"].astype("int64")
src=pd.read_csv("output/registry/source_2026-10-26.csv")[["omop_person_id","hospital_type_cascade","ground_truth","llm_classification","reviewed"]]
t=src.merge(m[["pid","S","U","aud","alc"]].rename(columns={"pid":"omop_person_id"}),on="omop_person_id")
t["h"]=t["hospital_type_cascade"]; t["s"]=np.where(t["aud"].notna(),1,np.where(t["alc"].notna(),2,3))
ros=pd.read_csv("resources/patient_roster.csv",dtype=str).set_index("omop_person_id")["has_aud_icd"]
t["roster"]=t.omop_person_id.astype(str).map(ros).map(lambda v: 1 if str(v).lower() in ("1","true","yes","1.0") else 0)
# Hospital type of a reviewed chart from the source of its notes, by the rule used for visits: safety-net
# hospital if any note was written there, otherwise academic health system if any was, otherwise other hospitals.
rep=pd.read_excel("resources/institution_report.xlsx"); mp=pd.read_csv("resources/mapped_report_ids.csv",dtype=str)
idcol=[c for c in mp.columns if "OMOP" in c.upper()][0]; rcol=[c for c in mp.columns if "REPORT" in c.upper()][0]
mp=mp[[idcol,rcol]].dropna(); mp.columns=["pid","REPORT_ID"]; mp["pid"]=pd.to_numeric(mp["pid"],errors="coerce").astype("Int64"); mp["REPORT_ID"]=pd.to_numeric(mp["REPORT_ID"],errors="coerce").astype("Int64")
rep["REPORT_ID"]=pd.to_numeric(rep["REPORT_ID"],errors="coerce").astype("Int64"); notes=rep.merge(mp,on="REPORT_ID"); notes["pid"]=notes["pid"].astype("int64")
casc=notes.assign(r=notes["ACTUAL_INSTITUTION_CATEGORY"].map({"Eskenazi":0,"IUH":1,"Other":2})).groupby("pid")["r"].min().map({0:"Safety-net hospital",1:"Academic health system",2:"Other hospitals"})
t["h_note"]=t.omop_person_id.map(casc)
N=t.groupby(["h","s"]).size()
rev=t[t.reviewed==1].copy(); det=rev[rev.ground_truth!="indet"].copy(); det["pos"]=(det.ground_truth=="AUD+").astype(float)
def est(charts,cells=("h","s"),Nreg=None,hcol="h"):
    cells=list(cells); Nr=(t.groupby(cells).size() if Nreg is None else Nreg)
    c=charts.copy(); c["h"]=c[hcol]
    n=c.groupby(cells).size(); w=(Nr/n).rename("w").reset_index(); c=c.merge(w,on=cells)
    W=c.pos*c.w; return float(W.sum()), float(W[c.U==0].sum()) if "U" in c else np.nan
out={}
E,E2=est(det); out["primary"]=(E,E2)
allc=rev.copy(); share=det.groupby(["h","s"]).pos.mean().rename("p").reset_index(); allc=allc.merge(share,on=["h","s"],how="left")
for nm,q in (("indet_neg",0.0),("indet_q25",.25),("indet_q50",.5),("indet_q75",.75)):
    a=allc.copy(); a["pos"]=np.where(a.ground_truth=="AUD+",1.0,np.where(a.ground_truth=="AUD-",0.0,q*a.p)); out[nm]=est(a)
a=allc.copy(); a["pos"]=np.where(a.ground_truth=="AUD-",0.0,1.0); out["indet_pos"]=est(a)
out["six_cells"]=est(det,cells=("h","U"))
d2=det[det.h_note.notna()]; out["note_source_h"]=est(d2,hcol="h_note"); out["note_source_missing"]=int(det.h_note.isna().sum())
# Fig 2c: sensitivity on reviewed charts by t0 stratum (diagnosis any time = roster flag as scored in Fig. 1; MedDx)
f2c={}
for S in (0,1,2):
    g=det[(det.S==S)&(det.pos==1)]
    f2c[S]=dict(n=int(len(g)),dx_roster=float((g.roster==1).mean()),dx_omop=float((g.U==1).mean()),meddx=float((g.llm_classification=="AUD+").mean()),
                n_det=int(((det.S==S)).sum()),n_rev=int((rev.S==S).sum()))
out["fig2c"]=f2c
out["strata_t0"]={int(k):int(v) for k,v in t.S.value_counts().items()}; out["strata_ever"]={int(k):int(v) for k,v in t.s.value_counts().items()}
out["no_dx_ever_registry"]=int((t.U==0).sum())
print(json.dumps(out,indent=1)); json.dump(out,open("output/registry/sensitivity_rows.json","w"),indent=1)
