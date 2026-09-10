"""Independent integrity audit for foreign-shareholding structure v1."""

import hashlib
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/locked_foreign_shareholding_structure_v1.json"
RESULT = ROOT / "reports/foreign_shareholding_structure.json"
MANIFEST = ROOT / "data/processed/factors/foreign_shareholding_fetched_dates.csv"
FEATURES = ROOT / "data/processed/factors/foreign_shareholding_structure.csv"
OUTPUT = ROOT / "reports/foreign_shareholding_structure_integrity.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    manifest = pd.read_csv(MANIFEST)
    features = pd.read_csv(FEATURES)
    expected = config["data_semantics"]["daily_rows"]
    checks = {
        "all_locked_hashes_match": all(
            sha256(ROOT / relative) == digest
            for relative, digest in config["implementation_hashes"].items()
        ),
        "result_data_hash_matches": result["data_sha256"] == sha256(FEATURES),
        "manifest_count_unique": len(manifest) == manifest["date"].nunique() == expected,
        "features_count_unique": len(features) == features["date"].nunique() == expected,
        "manifest_all_success": manifest["status"].eq("success").all(),
        "manifest_feature_dates_equal": set(manifest["date"]) == set(features["date"]),
        "date_range_locked": (
            features["date"].min() == config["data_semantics"]["earliest_date"]
            and features["date"].max() == config["data_semantics"]["latest_date"]
        ),
        "numeric_finite": pd.DataFrame({
            column: pd.to_numeric(features[column], errors="coerce")
            for column in features.columns if column != "date"
        }).notna().all().all(),
    }
    for name, variant in result["variants"].items():
        records = variant["records"]
        positions = [int(record["position"]) for record in records]
        hits = sum(record["prediction"] == record["target"] for record in records)
        checks[f"{name}_unique_dates"] = len(records) == len(
            {record["date"] for record in records}
        )
        checks[f"{name}_spacing"] = all(
            right - left >= 20 for left, right in zip(positions, positions[1:])
        )
        checks[f"{name}_cases"] = len(records) == variant["cases"]
        checks[f"{name}_hits"] = hits == variant["hits"]
        checks[f"{name}_accuracy"] = math.isclose(
            hits / len(records), variant["accuracy"], abs_tol=1e-15
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
        for record in result["variants"]["plus_foreign_shareholding"]["records"]
    }
    common = set(left).intersection(right)
    increment = result["foreign_shareholding_increment"]
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
