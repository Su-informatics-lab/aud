#!/usr/bin/env python3
"""Build the tidy tables the figure and table producers read, from the ground truth and the canonical panel.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats_core as sc


# The pending marker is gone: the reference standard is built and versioned, and the
# marker was being rendered into figure panels where a reader would see it.
GOLD_MARKER = ""
GOLD_ESTIMAND = "performance against the reference standard"
MODEL_CANDIDATES = [
    "Qwen3.6-27B",
    "Nemotron-3-120B-FP8",
    "GPT-OSS-120B",
    "GPT-OSS-20B",
]


def collapse3(value, cfg: dict) -> str:
    if value in set(cfg["aud_pos"]):
        return "+"
    if value in set(cfg["aud_neg"]):
        return "-"
    if value in set(cfg["abstain"]):
        return "I"
    return "missing"


def load_rating(path: Path, labels: list[str], reviewer: str) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    rows = []
    for review_id, record in payload.items():
        rows.append(
            {
                "omop_person_id": str(record["omop_person_id"]),
                reviewer: sc.normalize_label(
                    record["classification"], labels, strict=True
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out["omop_person_id"].duplicated().any():
        raise RuntimeError(f"duplicate person in {path}")
    return out


def binary_performance(
    reference: pd.Series, prediction: pd.Series, cfg: dict
) -> tuple[dict, int, float]:
    ref3 = reference.map(lambda value: collapse3(value, cfg))
    pred3 = prediction.map(lambda value: collapse3(value, cfg))
    deterministic = ref3.isin(["+", "-"]) & pred3.isin(["+", "-"])
    metrics = sc.binary_metrics(
        ref3[deterministic].eq("+"), pred3[deterministic].eq("+")
    )
    abstention = float(pred3.eq("I").mean())
    return metrics, int(deterministic.sum()), abstention


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-yaml", default="downstream/data.yaml")
    ap.add_argument("--out", default="downstream/npj_goal6_outputs/panels")
    ap.add_argument("--desai", default="chart/ratings/desai.json")
    ap.add_argument("--gawrieh", default="chart/ratings/gawrieh.json")
    args = ap.parse_args()
    cfg = sc.load_yaml_with_scope(args.data_yaml, "ref_gold")
    labels = cfg["labels_6"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    roster = pd.read_csv(
        cfg["paths"]["roster"], dtype={"omop_person_id": "string"}
    )
    roster["Maaz"] = sc.normalize_labels(
        roster["ndepth_maaz_rating"], labels, strict=True
    )
    gold = pd.read_csv(
        cfg["paths"]["gold"], dtype={"omop_person_id": "string"}
    )
    gold["reference_label"] = sc.normalize_labels(
        gold[cfg["raters"]["gold"]["col"]], labels, strict=True
    )
    # The gate is kept but its expectation is read from the manifest the reference
    # standard was built with, rather than written here as two literals. Hard-coding the
    # cohort size meant that changing the cohort broke this file, and the fix was to edit
    # the numbers, which is the same as having no gate at all.
    _man = json.loads(Path("resources/gold_manifest_v3.json").read_text())
    _want = _man["cohort"]["analysed"]
    _labelled = int(gold["reference_label"].notna().sum())
    if len(gold) != _want or _labelled != _want:
        raise RuntimeError(
            "the reference standard should hold %d rows each carrying a label, "
            "per %s; found %d rows and %d labels"
            % (_want, _man["gold_version"], len(gold), _labelled)
        )

    panel = pd.read_csv(
        cfg["paths"]["canonical_panel"],
        dtype={"omop_person_id": "string"},
    )
    model_specs = {name: spec["model"] for name, spec in cfg["models"].items()}
    tag_to_name = {tag: name for name, tag in model_specs.items()}
    panel = panel[panel["model"].isin(tag_to_name)].copy()
    panel["model_name"] = panel["model"].map(tag_to_name)
    panel["model_label"] = sc.normalize_labels(
        panel["model_label"], labels, strict=True
    )
    model_counts = panel.groupby("model_name")["omop_person_id"].agg(
        ["size", "nunique"]
    )
    if len(model_counts) != 11 or not model_counts.eq(499).all().all():
        raise RuntimeError(f"11-model panel coverage failed:\n{model_counts}")
    wide = panel.pivot(
        index="omop_person_id", columns="model_name", values="model_label"
    ).reset_index()

    base = (
        roster[["omop_person_id", "has_aud_icd", "Maaz"]]
        .merge(
            gold[["omop_person_id", "reference_label", "gold_source"]],
            on="omop_person_id",
            how="left",
            validate="one_to_one",
        )
        .merge(wide, on="omop_person_id", how="left", validate="one_to_one")
    )
    analysis = base[base["reference_label"].notna()].copy()

    # ------------------------------------------------------------------ F1
    flow = pd.DataFrame(
        [
            {
                "stage": "expanded extraction cohort",
                "n": 499,
                "detail": "full roster",
                "estimand_label": GOLD_ESTIMAND,
            },
            {
                "stage": "reference standard available",
                "n": 496,
                "detail": "three roster patients lack any human label",
                "estimand_label": GOLD_ESTIMAND,
            },
            {
                "stage": "expert reviewed by Desai",
                "n": 58,
                "detail": "selected review subset; not a random reliability sample",
                "estimand_label": GOLD_ESTIMAND,
            },
            {
                "stage": "Maaz/Desai same collapse bucket",
                "n": 24,
                "detail": "Desai supplies ordinal label under senior-available rule",
                "estimand_label": GOLD_ESTIMAND,
            },
            {
                "stage": "Gawrieh adjudicated",
                "n": 34,
                "detail": "contested subset",
                "estimand_label": GOLD_ESTIMAND,
            },
        ]
    )
    flow.to_csv(out / "f1_reference_flow.csv", index=False)

    confusion_rows = []
    for method in MODEL_CANDIDATES:
        cm = sc.confusion_matrix(
            analysis["reference_label"], analysis[method], labels
        )
        for i, reference_label in enumerate(labels):
            for j, prediction_label in enumerate(labels):
                confusion_rows.append(
                    {
                        "method": method,
                        "reference_label": reference_label,
                        "prediction_label": prediction_label,
                        "n": int(cm[i, j]),
                        # the scored denominator, not a literal
                        "paired_n": _want,
                        "primary_model_selected": False,
                        "estimand_label": GOLD_ESTIMAND,
                    }
                )
    pd.DataFrame(confusion_rows).to_csv(out / "f1_confusion_6class.csv", index=False)
    print("wrote", out / "f1_reference_flow.csv", "and", out / "f1_confusion_6class.csv")


if __name__ == "__main__":
    main()
