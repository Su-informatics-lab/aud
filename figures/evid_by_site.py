#!/usr/bin/env python3
"""Findings cited by MedDx, by ground-truth class and by hospital system: the table behind Extended Data Fig. 6.
"""
import argparse, sys, pandas as pd
sys.path.insert(0,"downstream/npj"); sys.path.insert(0,"downstream")
import f05_figure5 as F
from grounding_finish import is_absence_statement
r = pd.read_csv("output/raters/rater_long.csv", dtype=str)
gold = r[r["in_analysis_cohort"].eq("True") & r["rater"].eq("gold")].set_index("omop_person_id")["label_3"]
p = pd.read_csv(F.PANEL, dtype={"omop_person_id": str}); p = p[p["model"].eq(F.PRIMARY_TAG)].set_index("omop_person_id")
inst = pd.read_csv("output/fairness/patient_institution.csv", dtype=str)
site = inst.set_index("person_id")["institution"].map({"IUH":"Academic","Eskenazi":"Safety net"}).fillna("Other")
df = pd.DataFrame({"gt": gold}); df["site"] = site.reindex(df.index).fillna("Other")
df["mdx"] = p["model_label"].reindex(df.index)
print(df.groupby(["site","gt"]).size().unstack(fill_value=0))
rows=[]
for pid,g in gold.items():
    for c in F.parse_cats(p["evidence_categories"].get(pid)):
        if c in F.TOPCATS: rows.append({"pid":pid,"category":c,"gt":g,"site":df.loc[pid,"site"]})
e=pd.DataFrame(rows)
n_all=df.groupby("gt").size(); n_site=df.groupby(["site","gt"]).size()
out=[]
for c in F.TOPCATS:
    sub=e[e.category==c]
    rec={"category":c}
    for g in ("AUD+","AUD-","Indeterminate"):
        k=sub[sub["gt"]==g].pid.nunique(); rec["n_%s"%g]=k; rec["N_%s"%g]=int(n_all[g]); rec["rate_%s"%g]=round(k/n_all[g],4)
        for s in ("Academic","Safety net","Other"):
            k2=sub[(sub["gt"]==g)&(sub["site"]==s)].pid.nunique(); N2=int(n_site.get((s,g),0))
            rec["n_%s_%s"%(g,s)]=k2; rec["N_%s_%s"%(g,s)]=N2; rec["rate_%s_%s"%(g,s)]=round(k2/N2,4) if N2 else float("nan")
    out.append(rec)
o=pd.DataFrame(out); o.to_csv("figures/out/new/evidence_by_gt_site.csv",index=False)
pd.set_option("display.width",250)
print(o[["category"]+[c for c in o.columns if c.startswith("rate_")]].to_string())
# no-quotation patients
q=p["supporting_quote"].reindex(gold.index)
stmt=q.apply(lambda v: pd.isna(v) or not str(v).strip() or is_absence_statement(v))
print("no quotation:", int(stmt.sum()), "by gt:", df[stmt]["gt"].value_counts().to_dict(), "by MedDx:", df[stmt].mdx.value_counts().to_dict())
print("MedDx label of GT-indeterminate with a quotation:", df[~stmt & df["gt"].eq("Indeterminate")].mdx.value_counts().to_dict())
