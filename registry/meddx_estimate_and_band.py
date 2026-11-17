"""(1) The registry estimate from MedDx's classifications of the same charts, on the same nine cells (Fig. 2b).
(2) A check of the AUD-specific drug extraction: patients with naltrexone, acamprosate or disulfiram by concept
descendants against a search of concept names. (3) The band of Fig. 2d: 500 paired draws of the share-on-a-drug
curve. Run from the analysis root on the cluster after registry/source_table.py. Aggregates only.
"""
import sys, os, json, numpy as np, pandas as pd, warnings, importlib.util; warnings.filterwarnings("ignore")
sys.path.insert(0,"downstream"); sys.path.insert(0,"downstream/npj")
import registry_frame as FS
import registry_frame as ttr
m=FS.build(anchor="keep"); m["pid"]=m["person_id"].astype("int64")
src=pd.read_csv("output/registry/source_2026-10-26.csv")
t=src.merge(m[["pid","S","U","aud","alc","rx_before_t0","first_rx","t0"]].rename(columns={"pid":"omop_person_id"}),on="omop_person_id")
t["h"]=t["hospital_type_cascade"]; t["s"]=np.where(t["aud"].notna(),1,np.where(t["alc"].notna(),2,3))
cells=["h","s"]; N=t.groupby(cells).size().rename("N"); SN=["AUD diagnosis","Other alcohol-related diagnosis","No alcohol-related diagnosis"]
out={}
# ---- (1) MedDx-based estimate
rv=t[(t.reviewed==1)&(t.llm_classification.isin(["AUD+","AUD-"]))].copy(); rv["pos"]=(rv.llm_classification=="AUD+").astype(int)
print("MedDx determinate charts", len(rv), rv.pos.sum())
def est(d):
    n=d.groupby(cells).size(); d=d.merge((N/n).rename("w").reset_index(),on=cells); W=d.pos*d.w
    o={"E1":float(W.sum())}
    for s in range(3): o["E1_S%d"%s]=float(W[d.S==s].sum())
    o["E2_t0"]=float(W[d.S!=0].sum()); o["E2_ever"]=float(W[d.s!=1].sum()); return o
pt=est(rv); rng=np.random.default_rng(11); B=pd.DataFrame([est(rv.groupby(cells,group_keys=False).apply(lambda g:g.sample(len(g),replace=True,random_state=int(rng.integers(1e9))))) for _ in range(2000)])
out["meddx_estimate"]={"point":pt,"ci":{k:[float(np.percentile(B[k],2.5)),float(np.percentile(B[k],97.5))] for k in pt},"n_charts":int(len(rv))}
print(json.dumps(out["meddx_estimate"],indent=1))
# ---- (2) drug extraction check
D=ttr.D if hasattr(ttr,"D") else None
print("data dir", D)
de=pd.read_parquet(D+"/drug_exposure.parquet",columns=["person_id","drug_concept_id","drug_source_value","drug_exposure_start_date","drug_type_concept_id"])
de["person_id"]=de["person_id"].astype("int64"); de=de[de.person_id.isin(m.pid)]
con=pd.read_parquet(D+"/concept.parquet",columns=["concept_id","concept_name","domain_id","vocabulary_id"]) if os.path.exists(D+"/concept.parquet") else None
res={}
if con is not None:
    de=de.merge(con.rename(columns={"concept_id":"drug_concept_id"}),on="drug_concept_id",how="left")
    for drug in ("naltrexone","acamprosate","disulfiram"):
        byname=de[de.concept_name.str.lower().str.contains(drug,na=False)|de.drug_source_value.astype(str).str.lower().str.contains(drug,na=False)]
        res[drug]={"patients_by_name":int(byname.person_id.nunique()),"rows_by_name":int(len(byname)),"top_concepts":byname.concept_name.value_counts().head(8).to_dict(),"type_concepts":byname.drug_type_concept_id.value_counts().head(5).to_dict(),"by_year":byname.drug_exposure_start_date.astype(str).str[:4].value_counts().sort_index().to_dict()}
        if drug=="naltrexone": res[drug]["methylnaltrexone_rows"]=int(byname.concept_name.str.lower().str.contains("methylnaltrexone",na=False).sum())
else:
    res["note"]="no concept.parquet; source values only"
    for drug in ("naltrexone","acamprosate","disulfiram"):
        byname=de[de.drug_source_value.astype(str).str.lower().str.contains(drug,na=False)]; res[drug]={"patients_by_source_value":int(byname.person_id.nunique())}
res["outcome_definition"]={"patients_any_rx":int(m.first_rx.notna().sum()),"prior_users":int(m.rx_before_t0.sum())}
out["drug_check"]=res; print(json.dumps(res,indent=1,default=str))
# ---- (3) band for Fig 2d
J=json.load(open("output/registry/master_numbers_2026-10-26.json")); mm,_=ttr.followup(m.copy(),"admin"); grid=np.round(np.arange(0,5.0001,0.02),3)
def aj_curve(T,E):
    o=np.argsort(T); T=T[o]; E=E[o]; n=len(T); surv=1.0; cif=0.0; at=n; i=0; res=[]
    for h in grid:
        while i<n and T[i]<=h:
            j=i
            while j<n and T[j]==T[i]: j+=1
            d1=(E[i:j]==1).sum(); d2=(E[i:j]==2).sum(); cif+=surv*d1/at; surv*=1-(d1+d2)/at; at-=(j-i); i=j
        res.append(cif)
    return np.array(res)
# denominator draws: re-bootstrap the GT estimate per stratum (same scheme as master_numbers)
r=t[(t.reviewed==1)&(t.ground_truth!="indet")].copy(); r["pos"]=(r.ground_truth=="AUD+").astype(int)
rng=np.random.default_rng(2026); den=[]
for b in range(500):
    d=r.groupby(cells,group_keys=False).apply(lambda g:g.sample(len(g),replace=True,random_state=int(rng.integers(1e9)))); n=d.groupby(cells).size(); d=d.merge((N/n).rename("w").reset_index(),on=cells); W=d.pos*d.w
    den.append([float(W[d.S==s].sum()) for s in range(3)])
den=np.array(den); rows=[]
for s in range(3):
    g=mm[(mm.S==s)&(mm.rx_before_t0==0)]; T=g["T"].to_numpy(float); E=g["E"].to_numpy(int); prior=int(mm[(mm.S==s)].rx_before_t0.sum()); nn=len(T)
    base=100*(prior+nn*aj_curve(T,E))/J["point"]["E1_S%d"%s]; rng2=np.random.default_rng(7+s); draws=[]
    for b in range(500):
        idx=rng2.integers(0,nn,nn); draws.append(100*(prior+nn*aj_curve(T[idx],E[idx]))/den[b,s])
    draws=np.array(draws)
    for k,h in enumerate(grid): rows.append(dict(stratum=SN[s],t=h,share=base[k],lo=float(np.percentile(draws[:,k],2.5)),hi=float(np.percentile(draws[:,k],97.5))))
pd.DataFrame(rows).to_csv("output/registry/share_band_2026-10-26.csv",index=False); json.dump(out,open("output/registry/meddx_estimate_2026-10-26.json","w"),indent=1,default=str); print("done")
