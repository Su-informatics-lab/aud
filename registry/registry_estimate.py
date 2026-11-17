"""The registry estimate and everything counted over it.

Each chart with a determinate ground truth is weighted to the registry patients of its cell, hospital type by
whole-record diagnosis status (nine cells); the weighted sum of ground-truth-positive charts is the number of
registry patients with AUD. From the same weights: the estimate by stratum at t0 and by hospital type, the
sensitivity of the AUD diagnosis at t0 within each patient group and its ratio to the rest, the registry-weighted
sensitivity and specificity of MedDx and of the AUD diagnosis, and the share of patients with AUD on an
AUD-specific drug at t0, one year and five years (prior users plus the Aalen-Johansen cumulative incidence among
new users, death competing). Intervals are percentile bootstrap, charts resampled within cell.

usage: registry_estimate.py [bootstrap draws, default 2000]
Run from the analysis root on the cluster after registry/source_table.py. Aggregates only.
"""
import sys, os, json, numpy as np, pandas as pd, warnings, importlib.util; warnings.filterwarnings("ignore")

sys.path.insert(0,"downstream"); sys.path.insert(0,"downstream/npj")
import registry_frame as FS
from icd_fairness import build as ibuild
import f04_sdi as F4, f04_registry_burden_subgroup as RB
import registry_frame as ttr
NB=int(sys.argv[1]) if len(sys.argv)>1 else 2000; NAJ=500
m=FS.build(anchor="keep"); m["pid"]=m["person_id"].astype("int64")
mg=F4.attach(RB.attach_charlson(ibuild()), F4.sdi_reference(), F4.zip5_by_person()); mg["pid"]=mg["person_id"].astype("int64")
AX=[("agegrp","Age",["Under 50","50 to 64","65 and over"]),("sex","Sex",["Male","Female"]),("race","Race",["Other than White","White"]),("charlson_noliver","Charlson index",["0","1 to 2","3 or more"]),("sdi_q","Neighbourhood deprivation",F4.Q5)]
gcols=[a for a,_,_ in AX]
src=pd.read_csv("output/registry/source_2026-10-26.csv")
t=src.merge(m[["pid","S","U","aud","alc","rx_before_t0","first_rx","t0"]].rename(columns={"pid":"omop_person_id"}),on="omop_person_id")
t=t.merge(mg[["pid"]+gcols].rename(columns={"pid":"omop_person_id"}),on="omop_person_id",how="left")
t["h"]=t["hospital_type_cascade"]; t["s"]=np.where(t["aud"].notna(),1,np.where(t["alc"].notna(),2,3))
SN=["AUD diagnosis","Other alcohol-related diagnosis","No alcohol-related diagnosis"]; HN=["Academic health system","Safety-net hospital","Other hospitals"]
cells=["h","s"]; N=t.groupby(cells).size().rename("N")
r=t[(t.reviewed==1)&(t.ground_truth!="indet")].copy(); r["pos"]=(r.ground_truth=="AUD+").astype(int)
r["mdx"]=r["llm_classification"]; r["code"]=r["U"].astype(int)
def wt(rv):
    n=rv.groupby(cells).size(); w=(N/n).rename("w").reset_index(); return rv.merge(w,on=cells)
def q(xs,lo=2.5,hi=97.5): return (float(np.nanpercentile(xs,lo)),float(np.nanpercentile(xs,hi)))
def stats(rv):
    rv=wt(rv); o={}; W=rv.pos*rv.w
    o["E1"]=float(W.sum()); o["E2_ever"]=float(W[rv.s!=1].sum()); o["E2_t0"]=float(W[rv.S!=0].sum())
    for i,h in enumerate(HN): o["E1_h%d"%i]=float(W[rv.h==h].sum())
    for s in range(3): o["E1_S%d"%s]=float(W[rv.S==s].sum()); o["E1_sever%d"%(s+1)]=float(W[rv.s==s+1].sum())
    # per-group code sensitivity at t0 among AUD+ (coded at t0 = S==0)
    a=rv[rv.pos==1]
    for ax,_,levels in AX:
        for lv in levels:
            g=a[a[ax]==lv]; c=a[a[ax]!=lv]
            cov=float((g.w*(g.S==0)).sum()/g.w.sum()) if g.w.sum()>0 else np.nan
            covc=float((c.w*(c.S==0)).sum()/c.w.sum()) if c.w.sum()>0 else np.nan
            o["E4_%s|%s|cov"%(ax,lv)]=cov; o["E4_%s|%s|ratio"%(ax,lv)]=cov/covc if covc else np.nan
            o["E4_%s|%s|with_aud"%(ax,lv)]=float(g.w.sum()); o["E4_%s|%s|uncoded"%(ax,lv)]=float((g.w*(g.S!=0)).sum())
    # registry-weighted accuracy, indeterminate counts against the reader
    for rd,yh in (("MedDx",(rv.mdx=="AUD+").astype(int)),("code",rv.code)):
        okpos=(rv.mdx!="indet") if rd=="MedDx" else np.ones(len(rv),bool)
        pos=rv.pos==1; neg=rv.pos==0
        o["sens_reg_"+rd]=float((rv.w[pos]*(yh[pos]==1)).sum()/rv.w[pos].sum()); o["spec_reg_"+rd]=float((rv.w[neg]*((yh[neg]==0)&okpos[neg])).sum()/rv.w[neg].sum())
        o["sens_smp_"+rd]=float((yh[pos]==1).mean()); o["spec_smp_"+rd]=float(((yh[neg]==0)&okpos[neg]).mean())
    return o
pt=stats(r); rng=np.random.default_rng(2026); boots=[]
for b in range(NB):
    boots.append(stats(r.groupby(cells,group_keys=False).apply(lambda d: d.sample(len(d),replace=True,random_state=int(rng.integers(1e9))))))
bd=pd.DataFrame(boots); ci={k:q(bd[k]) for k in pt}
# ---- treatment
mm,admin=ttr.followup(m.copy(),"admin"); ml,_=ttr.followup(m.copy(),"last_record")
def aj(T,E,h):
    o=np.argsort(T); T=T[o]; E=E[o]; n=len(T); surv=1.0; cif=0.0; at=n; i=0
    while i<n and T[i]<=h:
        j=i
        while j<n and T[j]==T[i]: j+=1
        d1=(E[i:j]==1).sum(); d2=(E[i:j]==2).sum(); cif+=surv*d1/at; surv*=1-(d1+d2)/at; at-=(j-i); i=j
    return cif
tr={}; ajb={}
rng2=np.random.default_rng(9)
for S in range(3):
    g=mm[mm.S==S]; new=g[g.rx_before_t0==0]; T=new["T"].to_numpy(float); E=new["E"].to_numpy(int)
    newl=ml[(ml.S==S)&(ml.rx_before_t0==0)]
    tr[S]=dict(N=int(len(g)),prior=int(g.rx_before_t0.sum()),new_users=int(len(new)),cif1=aj(T,E,1.0),cif3=aj(T,E,3.0),cif5=aj(T,E,5.0),cif5_lastrec=aj(newl["T"].to_numpy(float),newl["E"].to_numpy(int),5.0))
    draws=[]
    for b in range(NAJ):
        idx=rng2.integers(0,len(T),len(T)); draws.append((aj(T[idx],E[idx],1.0),aj(T[idx],E[idx],5.0)))
    ajb[S]=np.array(draws)
out={"cells":[dict(h=h,s=int(s),N=int(N[(h,s)]),n_det=int(((r.h==h)&(r.s==s)).sum()),a_pos=int(r[(r.h==h)&(r.s==s)].pos.sum()),n_rev=int(((t.reviewed==1)&(t.h==h)&(t.s==s)).sum())) for (h,s) in N.index],
     "point":pt,"ci":ci,"registry":{"N":14406,"strata_t0":{SN[s]:int((t.S==s).sum()) for s in range(3)},"strata_ever":{SN[s-1]:int((t.s==s).sum()) for s in (1,2,3)},"hospital":{h:int((t.h==h).sum()) for h in HN},"coded_any_time":int(t.U.sum()),"coded_at_t0":int((t.S==0).sum())},"treatment":{}}
den=np.array([bd["E1_S%d"%S].to_numpy() for S in range(3)])   # NB draws
for S in range(3):
    x=tr[S]; dS=pt["E1_S%d"%S]
    sh0=100*x["prior"]/dS; sh5=100*(x["prior"]+x["cif5"]*x["new_users"])/dS; sh1=100*(x["prior"]+x["cif1"]*x["new_users"])/dS
    k=min(NB,NAJ); d0=100*x["prior"]/den[S,:k]; d5=100*(x["prior"]+ajb[S][:k,1]*x["new_users"])/den[S,:k]; d1=100*(x["prior"]+ajb[S][:k,0]*x["new_users"])/den[S,:k]
    out["treatment"][SN[S]]=dict(**x,aud_denominator=dS,share_t0=sh0,share_t0_ci=q(d0),share_1y=sh1,share_1y_ci=q(d1),share_5y=sh5,share_5y_ci=q(d5),cif1_ci=q(100*ajb[S][:,0]),cif5_ci=q(100*ajb[S][:,1]),
        share_5y_lastrec=100*(x["prior"]+x["cif5_lastrec"]*x["new_users"])/dS, _d0=d0.tolist(), _d5=d5.tolist())
ref0=np.array(out["treatment"][SN[0]]["_d0"]); ref5=np.array(out["treatment"][SN[0]]["_d5"])
for S in (1,2):
    o=out["treatment"][SN[S]]; d0=np.array(o["_d0"]); d5=np.array(o["_d5"])
    o["gap_t0"]=out["treatment"][SN[0]]["share_t0"]-o["share_t0"]; o["gap_t0_ci"]=q(ref0-d0)
    o["gap_5y"]=out["treatment"][SN[0]]["share_5y"]-o["share_5y"]; o["gap_5y_ci"]=q(ref5-d5)
    o["gap_added_after_t0"]=o["gap_5y"]-o["gap_t0"]; o["gap_added_ci"]=q((ref5-d5)-(ref0-d0))
    o["ratio_5y"]=out["treatment"][SN[0]]["share_5y"]/o["share_5y"]; o["ratio_5y_ci"]=q(ref5/d5)
for S in range(3):
    for k in ("_d0","_d5"): out["treatment"][SN[S]].pop(k)
out["prevalence"]=pt["E1"]/14406
def ppv(se,sp,p): return se*p/(se*p+(1-sp)*(1-p))
def npv(se,sp,p): return sp*(1-p)/(sp*(1-p)+(1-se)*p)
p=out["prevalence"]; out["predictive_at_prevalence"]={rd:dict(ppv=ppv(pt["sens_smp_"+rd],pt["spec_smp_"+rd],p),npv=npv(pt["sens_smp_"+rd],pt["spec_smp_"+rd],p)) for rd in ("MedDx","code")}
out["youden_reg"]={rd:pt["sens_reg_"+rd]+pt["spec_reg_"+rd]-1 for rd in ("MedDx","code")}
json.dump(out,open("output/registry/master_numbers_2026-10-26.json","w"),indent=1,default=float)
# csv for Supp Fig 4 producer
rows=[]
for ax,disp,levels in AX:
    for lv in levels:
        k="E4_%s|%s|"%(ax,lv); rows.append(dict(axis=disp,level=lv,coverage_pct=100*pt[k+"cov"],ratio=pt[k+"ratio"],ci_low=ci[k+"ratio"][0],ci_high=ci[k+"ratio"][1],registry_n=int((t[ax]==lv).sum()),with_aud=pt[k+"with_aud"],uncoded_at_t0=pt[k+"uncoded"],uncoded_lo=ci[k+"uncoded"][0],uncoded_hi=ci[k+"uncoded"][1]))
pd.DataFrame(rows).to_csv("output/registry/registry_disparity_model1.csv",index=False)
print("done E1 %.0f (%.0f-%.0f)"%(pt["E1"],*ci["E1"]))
