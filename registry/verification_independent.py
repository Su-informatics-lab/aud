"""Registry-weighted sensitivity and specificity against the verification-independent ground truth, under the
same nine-cell weights as the registry estimate, and the predictive values those rates imply (Supplementary
Note 1). Run from the analysis root on the cluster after registry/source_table.py. Aggregates only.
"""
import sys, os, json, argparse, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"downstream"); sys.path.insert(0,"downstream/npj")
import f01_restricted as FR, registry_frame as FS
a=argparse.Namespace(raters=FR.RATERS,panel=FR.PANEL,roster=FR.ROSTER)
d,man=FR.build(a)
src=pd.read_csv("output/registry/source_2026-10-26.csv"); src["omop_person_id"]=src["omop_person_id"].astype(str)
m=FS.build(anchor="keep"); m["omop_person_id"]=m["person_id"].astype("int64").astype(str)
m["s"]=np.where(m["aud"].notna(),1,np.where(m["alc"].notna(),2,3))
reg=src.merge(m[["omop_person_id","s"]],on="omop_person_id"); reg["h"]=reg["hospital_type_cascade"]
N=reg.groupby(["h","s"]).size().rename("N")
d=d.merge(reg[["omop_person_id","h","s"]],on="omop_person_id",how="left")
def weighted(sub,gold_col):
    det=sub[sub[gold_col].isin(["AUD+","AUD-"])].copy()
    n=det.groupby(["h","s"]).size(); det=det.merge((N/n).rename("w").reset_index(),on=["h","s"])
    out={}
    for name,col in (("MedDx","llm3"),("code","code3")):
        pos=det[det[gold_col]=="AUD+"]; neg=det[det[gold_col]=="AUD-"]
        out["wsens_"+name]=float((pos.w*(pos[col]=="AUD+")).sum()/pos.w.sum()); out["wspec_"+name]=float((neg.w*(neg[col]=="AUD-")).sum()/neg.w.sum())
    return out
res={"full":weighted(d,"gold3"),"restricted":weighted(d[d["restricted"].notna()],"restricted")}
# bootstrap the restricted within cells
rng=np.random.default_rng(3); dr=d[d["restricted"].notna()&d["restricted"].isin(["AUD+","AUD-"])]
B=[weighted(dr.groupby(["h","s"],group_keys=False).apply(lambda g: g.sample(len(g),replace=True,random_state=int(rng.integers(1e9)))),"restricted") for _ in range(2000)]
bd=pd.DataFrame(B); res["restricted_ci"]={k:[float(np.percentile(bd[k],2.5)),float(np.percentile(bd[k],97.5))] for k in bd}
# predictive values at the new prevalence, from the sample four rates of each standard (as Fig. 1f and Supp Note 1 do)
rates=pd.read_csv("downstream/npj/out/restricted/restricted_rates.csv")
p=6201.859432432364/14406
def pv(r,p):
    return dict(ppv=r.tpr*p/(r.tpr*p+r.fpr*(1-p)), npv=r.tnr*(1-p)/(r.tnr*(1-p)+r.fnr*p), ppv10=r.tpr*0.1/(r.tpr*0.1+r.fpr*0.9), npv10=r.tnr*0.9/(r.tnr*0.9+r.fnr*0.1))
res["predictive"]={f"{r.standard}|{r.reader}":pv(r,p) for r in rates.itertuples()}
res["prevalence"]=p; print(json.dumps(res,indent=1)); json.dump(res,open("output/registry/vi_ninecell_2026-10-26.json","w"),indent=1)
