"""Independent integrity audit for securities-lending pressure v1."""

import hashlib
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/locked_securities_lending_pressure_v1.json"
RESULT = ROOT / "reports/securities_lending_pressure.json"
MANIFEST = ROOT / "data/processed/factors/securities_lending_fetched_dates.csv"
FEATURES = ROOT / "data/processed/factors/securities_lending_pressure.csv"
OUTPUT = ROOT / "reports/securities_lending_pressure_integrity.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    manifest = pd.read_csv(MANIFEST)
    features = pd.read_csv(FEATURES)
    checks = {
        "all_locked_hashes_match": all(
            sha256(ROOT / relative) == expected
            for relative, expected in config["implementation_hashes"].items()
        ),
        "result_data_hash_matches": result["data_sha256"] == sha256(FEATURES),
        "manifest_4038_unique": len(manifest) == manifest["date"].nunique() == 4038,
        "features_4038_unique": len(features) == features["date"].nunique() == 4038,
        "manifest_all_success": manifest["status"].eq("success").all(),
        "manifest_feature_dates_equal": set(manifest["date"]) == set(features["date"]),
        "date_range_locked": (
            features["date"].min() == "2010-01-04"
            and features["date"].max() == "2026-07-23"
        ),
        "numeric_finite": bool(
            pd.DataFrame({
                column: pd.to_numeric(features[column], errors="coerce")
                for column in features.columns if column != "date"
            }).notna().all().all()
        ),
    }
    for name, variant in result["variants"].items():
        records = variant["records"]
        positions = [int(record["position"]) for record in records]
        hits = sum(int(record["prediction"]) == int(record["target"]) for record in records)
        checks[f"{name}_unique_dates"] = len(records) == len(
            {record["date"] for record in records}
        )
        checks[f"{name}_spacing"] = all(
            right - left >= 20 for left, right in zip(positions, positions[1:])
        )
        checks[f"{name}_cases"] = len(records) == int(variant["cases"])
        checks[f"{name}_hits"] = hits == int(variant["hits"])
        checks[f"{name}_accuracy"] = math.isclose(
            hits / len(records), float(variant["accuracy"]), abs_tol=1e-15
        )
        checks[f"{name}_binary"] = all(
            record["prediction"] in (-1, 1) and record["target"] in (-1, 1)
            for record in records
        )
    left = {
        record["date"]: bool(record["hit"])
        for record in result["variants"]["index_aggregate_control"]["records"]
    }
    right = {
        record["date"]: bool(record["hit"])
        for record in result["variants"]["plus_securities_lending"]["records"]
    }
    common = set(left).intersection(right)
    increment = result["lending_increment"]
    checks["paired_common_dates"] = len(common) == increment["identical_dates"]
    checks["paired_left_only"] = sum(
        left[date] and not right[date] for date in common
    ) == increment["left_only"]
    checks["paired_right_only"] = sum(
        right[date] and not left[date] for date in common
    ) == increment["right_only"]
    checks = {name: bool(value) for name, value in checks.items()}
    audit = {"passed": all(checks.values()), "checks": checks}
    OUTPUT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not audit["passed"]:
        raise RuntimeError(json.dumps(audit))


if __name__ == "__main__":
    main()
