"""Counts behind Extended Data Fig. 7 and Extended Data Table 2: crude cumulative incidence with numbers at
risk, prior users by stratum, the study-cohort charts by MedDx classification, and follow-up per stratum (new
users, person-years, first treatments, deaths, median follow-up). Run from the analysis root on the cluster after registry/source_table.py. Aggregates only.
"""
import sys, os, json, numpy as np, pandas as pd, warnings, importlib.util; warnings.filterwarnings("ignore")
sys.path.insert(0,"downstream"); sys.path.insert(0,"downstream/npj")
import registry_frame as FS, pyarrow.parquet as pq
import registry_frame as ttr
m=FS.build(anchor="keep"); m["pid"]=m["person_id"].astype("int64"); D=ttr.D
mm,_=ttr.followup(m.copy(),"admin"); mm["pid"]=mm["person_id"].astype("int64")
OLD={0:"AUD code",1:"alcohol coded only",2:"record silent"}; OLD2={0:"AUD ICD code",1:"alcohol ICD code only",2:"no alcohol ICD code"}
def aj(T,E,grid):
    o=np.argsort(T); T=T[o]; E=E[o]; n=len(T); surv=1.0; cif=0.0; at=n; i=0; out=[]
    for h in grid:
        while i<n and T[i]<=h:
            j=i
            while j<n and T[j]==T[i]: j+=1
            d1=(E[i:j]==1).sum(); d2=(E[i:j]==2).sum(); cif+=surv*d1/at; surv*=1-(d1+d2)/at; at-=(j-i); i=j
        out.append((h,cif,at))
    return out
grid=np.round(np.arange(0.02,6.0001,0.02),3); rows=[]; facts=[]
for s in range(3):
    g=mm[(mm.S==s)&(mm.rx_before_t0==0)]; T=g["T"].to_numpy(float); E=g["E"].to_numpy(int)
    for h,c,at in aj(T,E,grid): rows.append(dict(quantity="crude_aj_rx",stratum=OLD[s],t=h,mean=c,ci_low=np.nan,ci_high=np.nan,rhat=np.nan))
    for h,c,at in aj(T,E,[0.5,1,2,3,4,5,6]): rows.append(dict(quantity="at_risk",stratum=OLD[s],t=h,mean=at,ci_low=np.nan,ci_high=np.nan,rhat=np.nan))
    facts.append(dict(stratum=OLD2[s],new_users=len(g),person_years=round(float(np.minimum(T,9.0).sum())),first_treatment=int((E==1).sum()),deaths=int((E==2).sum()),median_followup=round(float(np.median(T)),1),death5=round(100*aj(T,np.where(E==2,1,np.where(E==1,2,0)),[5.0])[0][1],1)))
pd.DataFrame(rows).to_csv("output/registry/cif_crude_2026-10-26.csv",index=False)
pd.DataFrame(facts).to_csv("output/registry/followup_facts_2026-10-26.csv",index=False); print(pd.DataFrame(facts).to_string())
# prior users: a further drug on or after t0
conc=set(); anc=pd.read_parquet(D+"/concept_ancestor.parquet",columns=["ancestor_concept_id","descendant_concept_id"])
seeds=FS.AUD_DRUGS; conc=seeds|set(anc.loc[anc.ancestor_concept_id.isin(seeds),"descendant_concept_id"].astype("int64")); conc-=FS.EXCLUDE_DRUG
conc-=set(anc.loc[anc.ancestor_concept_id.isin(FS.COMBO_INGREDIENTS),"descendant_concept_id"].astype("int64")); del anc
pids=set(m.pid); parts=[]; pf=pq.ParquetFile(D+"/drug_exposure.parquet")
for b in pf.iter_batches(batch_size=500000, columns=["person_id","drug_concept_id","drug_exposure_start_date"]):
    b=b.to_pandas(); b["person_id"]=b["person_id"].astype("int64"); b=b[b.person_id.isin(pids)&b.drug_concept_id.isin(conc)]
    if len(b): parts.append(b)
de=pd.concat(parts); de["d"]=pd.to_datetime(de.drug_exposure_start_date,errors="coerce")
t0=mm.set_index("pid")["t0"]; de["t0"]=de.person_id.map(t0); after=de[de.d>=de.t0].groupby("person_id").d.min()
pu=[]
for s in range(3):
    g=mm[(mm.S==s)&(mm.rx_before_t0==1)].copy(); g["after"]=g.pid.map(after); g["death"]=mm.loc[g.index,"death"] if "death" in mm else pd.NaT
    end=mm.loc[g.index,"last_rec"].max()
    ev=np.where(g["after"].notna(),1,np.where(g["death"].notna()&(g["death"]>=g["t0"]),2,0)); tt=np.where(ev==1,(g["after"]-g["t0"]).dt.days,np.where(ev==2,(g["death"]-g["t0"]).dt.days,(end-g["t0"]).dt.days)).astype(float)/365.25; tt=np.maximum(tt,0.5/365.25)
    c=aj(tt,ev,[1,3,5]); pu.append(dict(stratum=OLD2[s],prior_users=len(g),rx_after_t0=int((ev==1).sum()),death=int((ev==2).sum()),cif1=round(100*c[0][1],1),cif3=round(100*c[1][1],1),cif5=round(100*c[2][1],1)))
pd.DataFrame(pu).to_csv("output/registry/prior_users_by_stratum_2026-10-26.csv",index=False); print(pd.DataFrame(pu).to_string())
# reviewed charts by MedDx label (new users only), plus registry rows
src=pd.read_csv("output/registry/source_2026-10-26.csv"); lab=src[src.reviewed==1].set_index("omop_person_id")["llm_classification"]
mm["mdx"]=mm.pid.map(lab); out=[]
def block(group,label,g):
    T=g["T"].to_numpy(float); E=g["E"].to_numpy(int); c=aj(T,E,[1,3,5]) if len(g) else [(1,0,0),(3,0,0),(5,0,0)]
    for (h,cif,_) in c: out.append(dict(group=group,label=label,n=len(g),rx=int((E==1).sum()),death=int((E==2).sum()),t=h,cif_pct=round(100*cif,2)))
for s in range(3):
    g=mm[(mm.S==s)&(mm.rx_before_t0==0)&mm.mdx.notna()]
    block("reviewed, %s, model"%OLD2[s],"model AUD+",g[g.mdx=="AUD+"]); block("reviewed, %s, model"%OLD2[s],"model not AUD+",g[g.mdx!="AUD+"])
    block("registry new users",OLD2[s],mm[(mm.S==s)&(mm.rx_before_t0==0)])
g=mm[(mm.rx_before_t0==0)&mm.mdx.notna()]; block("reviewed, all strata, model","model AUD+",g[g.mdx=="AUD+"]); block("reviewed, all strata, model","model not AUD+",g[g.mdx!="AUD+"])
pd.DataFrame(out).to_csv("output/registry/aj_by_label_summary_2026-10-26.csv",index=False); print("written")
