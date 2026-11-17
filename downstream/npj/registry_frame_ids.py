#!/usr/bin/env python3
"""Write the identifiers of the patients in the registry frame (resources/registry_frame_ids.csv).

The study cohort is restricted to sampled patients who have a cirrhosis diagnosis, hence a t0, in the registry.
ref_standard.py reads membership from this file.
"""
import sys, pandas as pd
sys.path.insert(0, "downstream"); sys.path.insert(0, "downstream/npj")
import registry_frame as FS
m = FS.build(); m = m[0] if isinstance(m, tuple) else m
out = pd.DataFrame({"omop_person_id": sorted(set(m["person_id"].astype("int64").astype(str)))})
out.to_csv("resources/registry_frame_ids.csv", index=False)
print(len(out), "patients in the registry frame")
