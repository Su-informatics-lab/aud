#!/usr/bin/env python3
"""Sensitivity and specificity of every engine, of MedDx under the skill variations and of the AUD diagnosis: the table behind Fig. 4c.
"""
import sys, pandas as pd, numpy as np
from scipy.stats import norm
sys.path.insert(0,"downstream/npj"); sys.path.insert(0,"downstream")
r = pd.read_csv("output/raters/rater_long.csv", dtype=str); r = r[r["in_analysis_cohort"].eq("True")]
gold = r[r["rater"].eq("gold")].set_index("omop_person_id")["label_3"]; det = gold[gold.isin(["AUD+","AUD-"])]
import aud_dx as DX
ros = DX.flag()
lead = pd.read_csv("downstream/npj/out/figure4/panel_a_leaderboard.csv")
TAG = dict(zip(lead["tag"] if "tag" in lead else [], lead["name"])) if "tag" in lead else {}
print(lead.columns.tolist())
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return 100*p,100*(c-h),100*(c+h)
rows=[]
readers=[(t,t) for t in sorted(r[r["rater_type"].eq("model")]["rater"].unique())]+[("code","code")]
for tag,_ in readers:
    if tag=="code": lab=ros.reindex(det.index).astype(float).fillna(0).map({1.0:"AUD+",0.0:"AUD-"})
    else: lab=r[r["rater"].eq(tag)].set_index("omop_person_id")["label_3"].reindex(det.index)
    pos=det.eq("AUD+"); neg=det.eq("AUD-")
    se=wilson(int((lab[pos]=="AUD+").sum()),int(pos.sum())); sp=wilson(int((lab[neg]=="AUD-").sum()),int(neg.sum()))
    rows.append(dict(tag=tag,sens=se[0],sens_lo=se[1],sens_hi=se[2],spec=sp[0],spec_lo=sp[1],spec_hi=sp[2],n_pos=int(pos.sum()),n_neg=int(neg.sum())))
o=pd.DataFrame(rows).round(2); o.to_csv("figures/out/new/reader_accuracy.csv",index=False); print(o.to_string())
# safety-net minus academic, AUD+
e=pd.read_csv("figures/out/new/evidence_by_gt_site.csv")
def newcombe(k1,n1,k2,n2,z=1.96):
    p1,l1,u1=[v/100 for v in wilson(k1,n1)]; p2,l2,u2=[v/100 for v in wilson(k2,n2)]
    d=p1-p2; return 100*d, 100*(d-np.sqrt((p1-l1)**2+(u2-p2)**2)), 100*(d+np.sqrt((u1-p1)**2+(p2-l2)**2))
out=[]
for _,x in e.iterrows():
    d,lo,hi=newcombe(x["n_AUD+_Safety net"],x["N_AUD+_Safety net"],x["n_AUD+_Academic"],x["N_AUD+_Academic"])
    out.append(dict(category=x["category"],safety=100*x["rate_AUD+_Safety net"],academic=100*x["rate_AUD+_Academic"],other=100*x["rate_AUD+_Other"],diff=d,lo=lo,hi=hi))
d=pd.DataFrame(out).round(1).sort_values("diff",ascending=False); d.to_csv("figures/out/new/evidence_safety_minus_academic.csv",index=False); print(d.to_string())
