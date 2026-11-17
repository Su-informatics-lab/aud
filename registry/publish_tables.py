"""Write the registry results into the tables the figure producers read: the registry prevalence, the paired
bootstrap of the registry-weighted accuracy of Fig. 1d, and the inputs of Supplementary Fig. 2.
Run from the analysis root on the cluster after registry/source_table.py. Aggregates only.
"""
import sys, os, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"downstream"); sys.path.insert(0,"downstream/npj")
import registry_frame as FS
O="downstream/npj/out"; J=json.load(open("output/registry/master_numbers_2026-10-26.json")); p,c=J["point"],J["ci"]
# prevalence
v=pd.read_csv(O+"/f01_accuracy/registry_prevalence.csv"); v.loc[0,["value","lo","hi"]]=[p["E1"]/14406,c["E1"][0]/14406,c["E1"][1]/14406]; v.loc[0,"estimator"]="registry estimate, Model 1 (hospital type x whole-record stratum, determinate charts)"; v.to_csv(O+"/f01_accuracy/registry_prevalence.csv",index=False)
# paired bootstrap for panel d: MedDx and the AUD diagnosis (registry flag) under the nine-cell weights
m=FS.build(anchor="keep"); m["pid"]=m["person_id"].astype("int64")
src=pd.read_csv("output/registry/source_2026-10-26.csv")[["omop_person_id","hospital_type_cascade","ground_truth","llm_classification","reviewed"]]
ros=pd.read_csv("resources/patient_roster.csv",dtype=str).set_index("omop_person_id")["has_aud_icd"]
t=src.merge(m[["pid","aud","alc"]].rename(columns={"pid":"omop_person_id"}),on="omop_person_id")
t["h"]=t["hospital_type_cascade"]; t["s"]=np.where(t["aud"].notna(),1,np.where(t["alc"].notna(),2,3)); cells=["h","s"]; N=t.groupby(cells).size().rename("N")
r=t[(t.reviewed==1)&(t.ground_truth!="indet")].copy(); r["pos"]=(r.ground_truth=="AUD+").astype(int)
r["roster"]=r["aud"].notna().astype(int)   # the AUD diagnosis at any time in the registry record (registry_frame U), the classifier of Fig. 1d
def stats(rv):
    n=rv.groupby(cells).size(); w=(N/n).rename("w").reset_index(); rv=rv.merge(w,on=cells); pos=rv.pos==1; neg=rv.pos==0; o={}
    o["Msens"]=float((rv.w[pos]*(rv.llm_classification[pos]=="AUD+")).sum()/rv.w[pos].sum()); o["Mspec"]=float((rv.w[neg]*(rv.llm_classification[neg]=="AUD-")).sum()/rv.w[neg].sum())
    o["Isens"]=float((rv.w[pos]*(rv.roster[pos]==1)).sum()/rv.w[pos].sum()); o["Ispec"]=float((rv.w[neg]*(rv.roster[neg]==0)).sum()/rv.w[neg].sum())
    o["dsens"]=o["Msens"]-o["Isens"]; o["dspec"]=o["Mspec"]-o["Ispec"]; o["JM"]=o["Msens"]+o["Mspec"]-1; o["JI"]=o["Isens"]+o["Ispec"]-1; return o
pt=stats(r); rng=np.random.default_rng(2026); B=[stats(r.groupby(cells,group_keys=False).apply(lambda d: d.sample(len(d),replace=True,random_state=int(rng.integers(1e9))))) for b in range(2000)]
bd=pd.DataFrame(B); ci={k:(float(np.percentile(bd[k],2.5)),float(np.percentile(bd[k],97.5))) for k in pt}
json.dump({"point":pt,"ci":ci},open("output/registry/panel_d_weighted.json","w"),indent=1); print(json.dumps({"point":pt,"ci":ci},indent=1))
d=pd.read_csv(O+"/f01_accuracy/reader_estimates.csv")
def setrow(reader,metric,val,lo,hi,dv=None,dlo=None,dhi=None):
    k=(d.reader==reader)&(d.metric==metric)&(d.weighted==True); assert k.sum()==1
    d.loc[k,["value","lo","hi"]]=[val,lo,hi]
    if dv is not None: d.loc[k,["diff_vs_model","diff_lo","diff_hi"]]=[dv,dlo,dhi]
setrow("M","sensitivity",pt["Msens"],*ci["Msens"]); setrow("M","specificity",pt["Mspec"],*ci["Mspec"])
setrow("I","sensitivity",pt["Isens"],*ci["Isens"],pt["dsens"],*ci["dsens"]); setrow("I","specificity",pt["Ispec"],*ci["Ispec"],pt["dspec"],*ci["dspec"])
d.loc[d["weighted"].astype(str).str.lower().eq("true")&d["reader"].isin(["M","I"]),"basis"]="weighted to the registry (hospital type x whole-record stratum)"
d.to_csv(O+"/f01_accuracy/reader_estimates.csv",index=False)
# Supp Fig. 4 inputs
g=pd.read_csv("output/registry/registry_disparity_model1.csv")
g[["axis","level","coverage_pct","ratio","ci_low","ci_high","registry_n"]].to_csv(O+"/figure4/registry_disparity.csv",index=False)
b=pd.DataFrame({"axis":g.axis,"level":g.level,"registry_n":g.registry_n,"with_aud":g.with_aud,"coded_by_t0":g.with_aud-g.uncoded_at_t0,"uncoded_at_t0":g.uncoded_at_t0,"ci_low":g.uncoded_lo,"ci_high":g.uncoded_hi,"pct_coded":g.coverage_pct})
b.to_csv(O+"/figure4/registry_burden_subgroup.csv",index=False); print("converted")
