"""What the no alcohol-related diagnosis stratum at t0 is made of, and the sensitivity of the registry estimate
to the skill's alcohol-associated cirrhosis finding (Supplementary Note 2). Run from the analysis root on the cluster after registry/source_table.py. Aggregates only.
"""
import sys, numpy as np, pandas as pd, warnings, importlib.util, json; warnings.filterwarnings("ignore")
sys.path.insert(0,"downstream"); sys.path.insert(0,"downstream/npj"); import registry_frame as FS
import registry_frame as ttr
m=FS.build(anchor="keep"); m["pid"]=m["person_id"].astype("int64")
t=pd.read_csv("output/registry/source_2026-10-26.csv").merge(m[["pid","S","U","aud","alc","rx_before_t0","first_rx","t0"]].rename(columns={"pid":"omop_person_id"}),on="omop_person_id")
t["h"]=t["hospital_type_cascade"]; t["s"]=np.where(t["aud"].notna(),1,np.where(t["alc"].notna(),2,3))
SN={0:"AUD diagnosis",1:"Other alcohol-related diagnosis",2:"No alcohol-related diagnosis"}
sN={1:"AUD code at any time",2:"other alcohol code only, at any time",3:"no alcohol code ever"}
cells=["h","s"]; N=t.groupby(cells).size().rename("N")
r0=t[(t.reviewed==1)&(t.ground_truth!="indet")].copy(); r0["pos"]=(r0.ground_truth=="AUD+").astype(int)
def weights(rv):
    n=rv.groupby(cells).size(); w=(N/n).rename("w").reset_index(); return rv.merge(w,on=cells)
def summarise(rv):
    rv=weights(rv); out={}
    out["E1"]=float((rv.pos*rv.w).sum())
    for S,d in rv.groupby("S"): out["t0_%d"%S]=float((d.pos*d.w).sum())
    for (S,s),d in rv.groupby(["S","s"]): out["t0_%d_s%d"%(S,s)]=float((d.pos*d.w).sum())
    for s,d in rv.groupby("s"): out["ever_s%d"%s]=float((d.pos*d.w).sum())
    return out
def boot(r,B=400,seed=11):
    pt=summarise(r); rng=np.random.default_rng(seed); bs=[]
    for b in range(B): bs.append(summarise(r.groupby(cells,group_keys=False).apply(lambda d: d.sample(len(d),replace=True,random_state=int(rng.integers(1e9))))))
    bd=pd.DataFrame(bs); return pt,{k:(float(bd[k].quantile(.025)),float(bd[k].quantile(.975))) for k in pt},bd
# ---------- treatment machinery
mm,admin=ttr.followup(m.copy(),"admin"); mm=mm.merge(t[["omop_person_id","s"]].rename(columns={"omop_person_id":"pid"}),on="pid",how="left")
def aj(T,E,h=5.0):
    o=np.argsort(T); T=T[o]; E=E[o]; n=len(T); surv=1.0; cif=0.0; at=n; i=0
    while i<n and T[i]<=h:
        j=i
        while j<n and T[j]==T[i]: j+=1
        d1=(E[i:j]==1).sum(); d2=(E[i:j]==2).sum()
        cif+=surv*d1/at; surv*=1-(d1+d2)/at; at-=(j-i); i=j
    return cif
def treat(g,seed=5,B=300):
    prior=int(g.rx_before_t0.sum()); new=g[g.rx_before_t0==0]; T=new["T"].to_numpy(float); E=new["E"].to_numpy(int)
    c5=aj(T,E); rng=np.random.default_rng(seed); bt=[]
    for b in range(B): idx=rng.integers(0,len(new),len(new)); bt.append(aj(T[idx],E[idx]))
    return dict(N=int(len(g)),prior=prior,new_users=int(len(new)),cif5=float(c5),treated5=float(prior+c5*len(new)),cif5_boot=bt)
def share(tr,den,den_boot):
    sh=100*(tr["prior"]+np.array(tr["cif5_boot"])[:,None]*tr["new_users"])/den_boot[None,:300]
    return dict(treated5=tr["treated5"],share=100*tr["treated5"]/den,share_ci=(float(np.percentile(sh,2.5)),float(np.percentile(sh,97.5))))
out={}
# ================= comment 15: decomposition of the t0 no-alcohol stratum
pt,ci,bd=boot(r0)
dec={}
for s in (1,2,3):
    sub=t[(t.S==2)&(t.s==s)]; rv=r0[(r0.S==2)&(r0.s==s)]
    k="t0_2_s%d"%s
    tr=treat(mm[(mm.S==2)&(mm.s==s)])
    dec[sN[s]]=dict(registry_patients=int(len(sub)),charts_determinate=int(len(rv)),charts_positive=int(rv.pos.sum()),
        est_aud=pt[k],ci=ci[k],prevalence=100*pt[k]/len(sub),treatment={kk:v for kk,v in tr.items() if kk!="cif5_boot"},
        treated_share=share(tr,pt[k],bd[k].to_numpy()))
dec["whole stratum"]=dict(registry_patients=int((t.S==2).sum()),est_aud=pt["t0_2"],ci=ci["t0_2"],
    never_aud_code=int(((t.S==2)&(t.aud.isna())).sum()),never_any_alcohol_code=int(((t.S==2)&(t.s==3)).sum()))
out["c15_decomposition"]=dec
out["c15_whole_record_never_any_code"]=dict(est=pt["ever_s3"],ci=ci["ever_s3"],registry=int((t.s==3).sum()),
    charts_det=int((r0.s==3).sum()),charts_pos=int(r0[r0.s==3].pos.sum()))
# ================= comment 17: alcoholic-cirrhosis trigger
p=pd.read_csv("output/panel.csv",low_memory=False); q=p[p.model=="aud_qwen36_27b"][["omop_person_id","rule_findings","model_label"]]
q["trig"]=q.rule_findings.fillna("[]").apply(lambda s: sorted(json.loads(s)))
q["sole_ac"]=q.trig.apply(lambda l: l==["alcoholic cirrhosis"])
q["sole_dx"]=q.trig.apply(lambda l: l==["explicit AUD diagnosis"])
q["any_ac"]=q.trig.apply(lambda l: "alcoholic cirrhosis" in l)
rr=r0.merge(q[["omop_person_id","sole_ac","sole_dx","any_ac"]],on="omop_person_id",how="left")
for c in ["sole_ac","sole_dx","any_ac"]: rr[c]=rr[c].fillna(False).astype(bool)
def tab(flag):
    d=rr[rr[flag]]
    return dict(n_charts=int(len(d)),n_pos=int(d.pos.sum()),
        by_t0_stratum={SN[int(S)]:dict(n=int(len(g)),pos=int(g.pos.sum())) for S,g in d.groupby("S")},
        by_hospital={h:int(len(g)) for h,g in d.groupby("h")})
out["c17_flags"]={"sole alcoholic cirrhosis":tab("sole_ac"),"sole explicit AUD diagnosis":tab("sole_dx"),"any alcoholic cirrhosis":tab("any_ac")}
# all reviewed charts incl. indeterminate, where MedDx sole trigger is alcoholic cirrhosis
allrev=t[t.reviewed==1].merge(q[["omop_person_id","sole_ac"]],on="omop_person_id",how="left")
out["c17_sole_ac_all_reviewed"]={k:int(v) for k,v in allrev[allrev.sole_ac==True].ground_truth.value_counts().items()}
def scenario(name,drop_flag,mode):
    r=rr.copy()
    if mode=="indet": r=r[~r[drop_flag]]
    else: r.loc[r[drop_flag],"pos"]=0
    ptx,cix,bdx=boot(r)
    res=dict(E1=ptx["E1"],E1_ci=cix["E1"])
    for S in range(3):
        k="t0_%d"%S; tr=treat(mm[mm.S==S]); den=ptx[k]; nS=int((t.S==S).sum())
        res[SN[S]]=dict(est_aud=den,ci=cix[k],prevalence=100*den/nS,**share(tr,den,bdx[k].to_numpy()))
    out["c17_"+name]=res
scenario("base","sole_ac","none")
scenario("sole_ac_indet","sole_ac","indet"); scenario("sole_ac_neg","sole_ac","neg")
scenario("sole_dx_indet","sole_dx","indet"); scenario("sole_dx_neg","sole_dx","neg")
scenario("any_ac_indet","any_ac","indet")
json.dump(out,open("output/registry/stratum_decomposition_2026-11-04.json","w"),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
