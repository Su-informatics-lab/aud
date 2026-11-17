"""AUD-specific drug records by ingredient, concept and source value, and the band of the share-on-a-drug
curve after combination products are excluded. Run from the analysis root on the cluster after registry/source_table.py. Aggregates only.
"""
import sys, os, json, numpy as np, pandas as pd, warnings, importlib.util; warnings.filterwarnings("ignore")
sys.path.insert(0,"downstream"); sys.path.insert(0,"downstream/npj")
import registry_frame as FS
import registry_frame as ttr
m=FS.build(anchor="keep"); m["pid"]=m["person_id"].astype("int64"); D=ttr.D
import pyarrow.parquet as pq, pyarrow.compute as pc
ct=pq.read_table(D+"/concept.parquet",columns=["concept_id","concept_name"])
low=pc.utf8_lower(ct["concept_name"])
res={}
import pyarrow.parquet as pq
pids=set(m.pid.tolist()); parts=[]
pf=pq.ParquetFile(D+"/drug_exposure.parquet")
for batch in pf.iter_batches(batch_size=500000, columns=["person_id","drug_concept_id","drug_source_value","drug_source_concept_id","drug_exposure_start_date"]):
    b=batch.to_pandas(); b["person_id"]=b["person_id"].astype("int64"); b=b[b.person_id.isin(pids)]
    if len(b): parts.append(b)
de=pd.concat(parts,ignore_index=True); del parts; print("rows cohort", len(de))
for drug in ("naltrexone","acamprosate","disulfiram"):
    mask=pc.match_substring(low,drug)
    if drug=="naltrexone": mask=pc.and_(mask,pc.invert(pc.match_substring(low,"methyl")))
    ids=ct.filter(mask).to_pandas(); idset=set(ids.concept_id)
    byc=de[de.drug_concept_id.isin(idset)|de.drug_source_concept_id.isin(idset)]
    bys=de[de.drug_source_value.astype(str).str.lower().str.contains(drug,na=False)]
    both=pd.concat([byc,bys]).drop_duplicates()
    nm=dict(zip(ids.concept_id,ids.concept_name))
    res[drug]={"patients_by_concept":int(byc.person_id.nunique()),"patients_by_source_string":int(bys.person_id.nunique()),"patients_union":int(both.person_id.nunique()),
      "top_concepts_union":{str(nm.get(k,k)):int(v) for k,v in both.drug_concept_id.value_counts().head(8).items()},
      "top_source_values":{str(k):int(v) for k,v in both.drug_source_value.value_counts().head(8).items()},
      "by_year_union":{str(k):int(v) for k,v in both.drug_exposure_start_date.astype(str).str[:4].value_counts().sort_index().items()},
      "unmapped_rows (concept 0) among string hits":int((bys.drug_concept_id==0).sum())}
    print(drug, json.dumps(res[drug],indent=1))
# how the outcome definition counts
res["outcome_definition"]={"patients_any_rx_in_m":int(m.first_rx.notna().sum()),"prior_users":int(m.rx_before_t0.sum()),"with_first_rx_on_or_after_t0":int(((m.first_rx>=m.t0)).sum())}
print(res["outcome_definition"])
json.dump(res,open("output/registry/drug_check_2026-10-26.json","w"),indent=1)
# ---- band
J=json.load(open("output/registry/master_numbers_2026-10-26.json")); mm,_=ttr.followup(m.copy(),"admin"); grid=np.round(np.arange(0,5.0001,0.02),3)
src=pd.read_csv("output/registry/source_2026-10-26.csv"); t=src.merge(m[["pid","S","aud","alc"]].rename(columns={"pid":"omop_person_id"}),on="omop_person_id")
t["h"]=t["hospital_type_cascade"]; t["s"]=np.where(t["aud"].notna(),1,np.where(t["alc"].notna(),2,3)); cells=["h","s"]; N=t.groupby(cells).size().rename("N"); SN=["AUD diagnosis","Other alcohol-related diagnosis","No alcohol-related diagnosis"]
def aj_curve(T,E):
    o=np.argsort(T); T=T[o]; E=E[o]; n=len(T); surv=1.0; cif=0.0; at=n; i=0; out=[]
    for h in grid:
        while i<n and T[i]<=h:
            j=i
            while j<n and T[j]==T[i]: j+=1
            d1=(E[i:j]==1).sum(); d2=(E[i:j]==2).sum(); cif+=surv*d1/at; surv*=1-(d1+d2)/at; at-=(j-i); i=j
        out.append(cif)
    return np.array(out)
r=t[(t.reviewed==1)&(t.ground_truth!="indet")].copy(); r["pos"]=(r.ground_truth=="AUD+").astype(int); rng=np.random.default_rng(2026); den=[]
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
pd.DataFrame(rows).to_csv("output/registry/share_band_2026-10-26.csv",index=False); print("band done")
