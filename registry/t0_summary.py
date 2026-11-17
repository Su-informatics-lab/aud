"""The range of t0 and the size of the three strata at t0 (Methods; Fig. 2a).
Run from the analysis root on the cluster. Aggregates only.
"""
import sys, json
sys.path.insert(0, "downstream"); sys.path.insert(0, "downstream/npj")
import registry_frame as FS
m = FS.build(anchor="keep")
out = {"t0_min": str(m.t0.min())[:10], "t0_max": str(m.t0.max())[:10], "t0_by_end_2020_pct": float(100 * (m.t0.dt.year <= 2020).mean()),
       "strata_at_t0": {int(k): int(v) for k, v in m.S.value_counts().sort_index().items()}}
json.dump(out, open("output/registry/t0_summary.json", "w"), indent=1); print(json.dumps(out, indent=1))
