"""(a) Registry-weighted accuracy of the AUD diagnosis as recorded when the study cohort was drawn, the
classifier scored in Fig. 1, under the nine-cell weights with 2,000 bootstrap draws within cells. (b)
Aalen-Johansen curves of a first AUD-specific drug among new users by stratum at t0, on a time grid, and the
share-on-a-drug curve among patients with AUD (Fig. 2d). (c) Three-year shares with intervals.
Run from the analysis root on the cluster after registry/source_table.py. Aggregates only.
"""
import sys, os, json, numpy as np, pandas as pd, warnings, importlib.util; warnings.filterwarnings("ignore")
sys.path.insert(0,"downstream"); sys.path.insert(0,"downstream/npj")
import registry_frame as FS
import registry_frame as ttr
m=FS.build(anchor="keep"); m["pid"]=m["person_id"].astype("int64")
src=pd.read_csv("output/registry/source_2026-10-26.csv")
ros=pd.read_csv("resources/patient_roster.csv",dtype=str).set_index("omop_person_id")["has_aud_icd"]
t=src.merge(m[["pid","S","U","Uth","aud","alc","rx_before_t0","first_rx","t0"]].rename(columns={"pid":"omop_person_id"}),on="omop_person_id")
t["h"]=t["hospital_type_cascade"]; t["s"]=np.where(t["aud"].notna(),1,np.where(t["alc"].notna(),2,3))
cells=["h","s"]; N=t.groupby(cells).size().rename("N")
r=t[(t.reviewed==1)&(t.ground_truth!="indet")].copy(); r["pos"]=(r.ground_truth=="AUD+").astype(int)
rr=ros.reindex(r.omop_person_id.astype(str)).values
r["roster"]=pd.Series(rr).map(lambda v: 1 if str(v).lower() in ("1","true","yes") else 0).values
print("roster sens sample", (r.roster[r.pos==1]==1).mean(), "Uth sens", (r.Uth[r.pos==1]==1).mean(), "U sens", (r.U[r.pos==1]==1).mean(), "disagree roster vs U", int((r.roster!=r.U).sum()))
def wt(rv):
    n=rv.groupby(cells).size(); w=(N/n).rename("w").reset_index(); return rv.merge(w,on=cells)
def stats(rv):
    rv=wt(rv); pos=rv.pos==1; neg=rv.pos==0; o={}
    for nm,yh in (("roster",rv.roster),("omop",rv.U.astype(int))):
        o["sens_"+nm]=float((rv.w[pos]*(yh[pos]==1)).sum()/rv.w[pos].sum()); o["spec_"+nm]=float((rv.w[neg]*(yh[neg]==0)).sum()/rv.w[neg].sum())
        o["youden_"+nm]=o["sens_"+nm]+o["spec_"+nm]-1
    return o
pt=stats(r); rng=np.random.default_rng(2026); B=[]
for b in range(2000): B.append(stats(r.groupby(cells,group_keys=False).apply(lambda d: d.sample(len(d),replace=True,random_state=int(rng.integers(1e9))))))
bd=pd.DataFrame(B); out={"point":pt,"ci":{k:[float(np.percentile(bd[k],2.5)),float(np.percentile(bd[k],97.5))] for k in pt},"sample":{"sens_roster":float((r.roster[r.pos==1]==1).mean()),"spec_roster":float((r.roster[r.pos==0]==0).mean()),"sens_omop":float((r.U[r.pos==1]==1).mean()),"spec_omop":float((r.U[r.pos==0]==0).mean())}}
print(json.dumps(out,indent=1))
# ---- AJ curves
def aj_curve(T,E,grid):
    o=np.argsort(T); T=T[o]; E=E[o]; n=len(T); surv=1.0; cif=0.0; at=n; i=0; out=[]
    gi=0
    while gi<len(grid):
        h=grid[gi]
        while i<n and T[i]<=h:
            j=i
            while j<n and T[j]==T[i]: j+=1
            d1=(E[i:j]==1).sum(); d2=(E[i:j]==2).sum(); cif+=surv*d1/at; surv*=1-(d1+d2)/at; at-=(j-i); i=j
        out.append((h,cif,at)); gi+=1
    return out
SN={0:"AUD diagnosis",1:"Other alcohol-related diagnosis",2:"No alcohol-related diagnosis"}
grid=np.round(np.arange(0,5.0001,0.02),3); rows=[]
for cens in ("admin","last_record"):
    mm,_=ttr.followup(m.copy(),cens)
    for s in (0,1,2):
        d=mm[(mm.S==s)&(mm.rx_before_t0==0)]
        T=d["T"].values.astype(float); E=d["event"].values.astype(int) if "event" in d else d["E"].values.astype(int)
        base=aj_curve(T,E,grid)
        # bootstrap band (300)
        bs=[]; rng2=np.random.default_rng(7+s)
        for b in range(300):
            idx=rng2.integers(0,len(T),len(T)); bs.append([c for _,c,_ in aj_curve(T[idx],E[idx],grid)])
        bs=np.array(bs)
        for k,(h,c,at) in enumerate(base):
            rows.append(dict(censoring=cens,stratum=SN[s],t=h,cif=c,lo=float(np.percentile(bs[:,k],2.5)),hi=float(np.percentile(bs[:,k],97.5)),at_risk=int(at),new_users=len(T)))
pd.DataFrame(rows).to_csv("output/registry/aj_curves_by_stratum_2026-10-26.csv",index=False); print("curves written")
json.dump(out,open("output/registry/code_roster_acc_2026-10-26.json","w"),indent=1)
