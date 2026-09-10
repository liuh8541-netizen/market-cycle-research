"""Unlabeled feasibility audit for the credit-short inventory source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "research/phase2_credit_short_inventory"
FEATURES = WORKSPACE / "data/credit_short_inventory.csv"
MANIFEST = WORKSPACE / "data/credit_short_inventory_fetched_dates.csv"
TRADING_DATES = (
    ROOT / "data/processed/factors/cross_sectional_breadth_fetched_dates.csv"
)
OUTPUT = WORKSPACE / "reports/credit_short_inventory_feasibility.json"
REPORT = WORKSPACE / "reports/credit_short_inventory_feasibility.md"
HISTORY = 1512
HORIZON = 20
NON_MODEL_NUMERIC_COLUMNS = {"source_duplicate_stock_rows"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2005-07-01")
    parser.add_argument("--end-date", required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(MANIFEST)
    features = pd.read_csv(FEATURES) if FEATURES.exists() else pd.DataFrame()
    cash = pd.read_csv(TRADING_DATES)
    cash["price_source_rows"] = pd.to_numeric(
        cash["price_source_rows"], errors="coerce"
    ).fillna(0)
    expected = (
        cash.loc[
            cash["price_source_rows"].gt(0)
            & cash["date"].astype(str).between(args.start_date, args.end_date),
            "date",
        ]
        .drop_duplicates()
        .astype(str)
        .sort_values()
        .tolist()
    )
    expected_set = set(expected)
    manifest["date"] = manifest["date"].astype(str)
    terminal = {
        "success",
        "excluded_no_common_rows",
        "excluded_low_common_stock_coverage",
    }
    successful = set(
        manifest.loc[manifest["status"].eq("success"), "date"].astype(str)
    )
    terminal_dates = set(
        manifest.loc[manifest["status"].isin(terminal), "date"].astype(str)
    )
    failed_dates = sorted(
        set(manifest.loc[manifest["status"].eq("failed"), "date"].astype(str))
    )
    missing_dates = sorted(expected_set - terminal_dates)
    if not features.empty:
        features["date"] = features["date"].astype(str)
        numeric = features.drop(columns=["date"]).apply(
            pd.to_numeric, errors="coerce"
        )
        nonfinite = int((~np.isfinite(numeric.to_numpy(float))).sum())
        constants = [
            column
            for column in numeric.columns
            if numeric[column].nunique(dropna=True) <= 1
        ]
        model_eligible_constants = [
            column
            for column in constants
            if column not in NON_MODEL_NUMERIC_COLUMNS
        ]
        duplicate_features = int(features["date"].duplicated().sum())
        feature_dates = set(features["date"])
        min_stock_count = int(features["credit_short_stock_count"].min())
        median_stock_count = float(features["credit_short_stock_count"].median())
        date_min = str(features["date"].min())
        date_max = str(features["date"].max())
    else:
        nonfinite = 0
        constants = []
        model_eligible_constants = []
        duplicate_features = 0
        feature_dates = set()
        min_stock_count = 0
        median_stock_count = 0.0
        date_min = None
        date_max = None
    aligned_dates = [day for day in expected if day in feature_dates]
    usable_rows = max(0, len(aligned_dates) - 1)
    maximum_nonoverlap_cases = max(
        0, (usable_rows - HISTORY - HORIZON) // HORIZON
    )
    completed = (
        not missing_dates
        and not failed_dates
        and expected_set == terminal_dates
    )
    contains_target_like = (
        [
            column
            for column in features.columns
            if any(
                token in column.lower()
                for token in ["target", "label", "future", "forward", "hit"]
            )
        ]
        if not features.empty
        else []
    )
    checks = {
        "dataset": "TaiwanDailyShortSaleBalances",
        "decision_availability": "next actual trading day after the 21:00 source update",
        "expected_trading_dates": len(expected_set),
        "terminal_dates": len(terminal_dates),
        "successful_dates": len(successful),
        "failed_dates": failed_dates,
        "missing_dates": missing_dates,
        "backfill_complete": completed,
        "feature_rows": len(features),
        "feature_unique_dates": len(feature_dates),
        "duplicate_feature_dates": duplicate_features,
        "success_feature_date_mismatch": sorted(successful ^ feature_dates),
        "feature_date_min": date_min,
        "feature_date_max": date_max,
        "nonfinite_numeric_cells": nonfinite,
        "constant_numeric_columns": constants,
        "constant_model_eligible_columns": model_eligible_constants,
        "minimum_common_stock_count": min_stock_count,
        "median_common_stock_count": median_stock_count,
        "target_like_columns": contains_target_like,
        "usable_next_day_rows": usable_rows,
        "maximum_independent_20d_cases_after_1512_history": (
            maximum_nonoverlap_cases
        ),
    }
    passed = (
        completed
        and len(features) == len(successful)
        and not checks["success_feature_date_mismatch"]
        and duplicate_features == 0
        and nonfinite == 0
        and not model_eligible_constants
        and min_stock_count >= 50
        and not contains_target_like
        and maximum_nonoverlap_cases >= 100
    )
    payload = {
        "passed": bool(passed),
        "eligible_to_lock_formal_experiment": bool(passed),
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Credit-short inventory unlabeled feasibility",
        "",
        f"- Backfill complete: {completed}",
        f"- Successful dates: {len(successful)} / {len(expected_set)}",
        f"- Feature range: {date_min} to {date_max}",
        f"- Non-finite numeric cells: {nonfinite}",
        f"- Minimum common-stock count: {min_stock_count}",
        f"- Maximum independent 20-day cases: {maximum_nonoverlap_cases}",
        f"- Eligible to lock: {passed}",
        "",
        "No prediction target or future-return label was read by this audit.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
