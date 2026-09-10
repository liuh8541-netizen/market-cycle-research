"""Unlabeled feasibility audit for market absorption geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "research/phase3_market_absorption_geometry"
FEATURES = WORKSPACE / "data/market_absorption_geometry.csv"
MANIFEST = WORKSPACE / "data/market_absorption_geometry_fetched_dates.csv"
TWII = ROOT / "data/processed/twii_daily.csv"
OUTPUT = WORKSPACE / "reports/market_absorption_geometry_feasibility.json"
REPORT = WORKSPACE / "reports/market_absorption_geometry_feasibility.md"
NON_MODEL = {"source_duplicate_stock_rows"}
CONFIRM_START = "2014-01-01"
CONFIRM_END = "2025-12-31"
HORIZON = 20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="1997-07-02")
    parser.add_argument("--end-date", required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(MANIFEST)
    features = pd.read_csv(FEATURES) if FEATURES.exists() else pd.DataFrame()
    twii = pd.read_csv(TWII, usecols=["date"])
    expected_dates = pd.to_datetime(twii["date"], errors="coerce").dropna()
    expected = set(
        expected_dates.loc[
            expected_dates.between(
                pd.Timestamp(args.start_date), pd.Timestamp(args.end_date)
            )
        ].dt.strftime("%Y-%m-%d")
    )
    manifest["date"] = manifest["date"].astype(str)
    terminal_statuses = {
        "success",
        "excluded_no_common_rows",
        "excluded_low_common_stock_coverage",
    }
    terminal = set(
        manifest.loc[
            manifest["status"].isin(terminal_statuses), "date"
        ].astype(str)
    )
    successful = set(
        manifest.loc[manifest["status"].eq("success"), "date"].astype(str)
    )
    failed = sorted(
        set(manifest.loc[manifest["status"].eq("failed"), "date"].astype(str))
    )
    missing = sorted(expected - terminal)
    if not features.empty:
        features["date"] = features["date"].astype(str)
        numeric = features.drop(columns=["date"]).apply(
            pd.to_numeric, errors="coerce"
        )
        nonfinite = int((~np.isfinite(numeric.to_numpy(float))).sum())
        constants = [
            column
            for column in numeric
            if numeric[column].nunique(dropna=True) <= 1
        ]
        model_constants = [column for column in constants if column not in NON_MODEL]
        feature_dates = set(features["date"])
        duplicates = int(features["date"].duplicated().sum())
        minimum_stocks = int(features["geometry_stock_count"].min())
        median_stocks = float(features["geometry_stock_count"].median())
        date_min = features["date"].min()
        date_max = features["date"].max()
    else:
        nonfinite = 0
        constants = []
        model_constants = []
        feature_dates = set()
        duplicates = 0
        minimum_stocks = 0
        median_stocks = 0.0
        date_min = None
        date_max = None
    confirmation_dates = sorted(
        day
        for day in feature_dates
        if CONFIRM_START <= day <= CONFIRM_END
    )
    maximum_confirmation_cases = max(
        0, (len(confirmation_dates) - HORIZON) // HORIZON
    )
    target_like = (
        [
            column
            for column in features
            if any(
                token in column.lower()
                for token in ["target", "label", "future", "forward", "hit"]
            )
        ]
        if not features.empty
        else []
    )
    complete = not missing and not failed and expected == terminal
    checks = {
        "dataset": "TaiwanStockPrice",
        "factor_formation_uses_direction_labels": False,
        "decision_availability": "next trading decision after the completed source session",
        "expected_dates": len(expected),
        "terminal_dates": len(terminal),
        "successful_dates": len(successful),
        "failed_dates": failed,
        "missing_dates": missing,
        "backfill_complete": complete,
        "feature_rows": len(features),
        "feature_unique_dates": len(feature_dates),
        "duplicate_feature_dates": duplicates,
        "success_feature_date_mismatch": sorted(successful ^ feature_dates),
        "feature_date_min": date_min,
        "feature_date_max": date_max,
        "nonfinite_numeric_cells": nonfinite,
        "constant_numeric_columns": constants,
        "constant_model_eligible_columns": model_constants,
        "minimum_common_stock_count": minimum_stocks,
        "median_common_stock_count": median_stocks,
        "target_like_columns": target_like,
        "independent_confirmation_start": CONFIRM_START,
        "independent_confirmation_end": CONFIRM_END,
        "independent_confirmation_rows": len(confirmation_dates),
        "maximum_independent_20d_confirmation_cases": maximum_confirmation_cases,
    }
    passed = (
        complete
        and len(features) == len(successful)
        and not checks["success_feature_date_mismatch"]
        and duplicates == 0
        and nonfinite == 0
        and not model_constants
        and minimum_stocks >= 100
        and not target_like
        and maximum_confirmation_cases >= 100
    )
    payload = {
        "passed": bool(passed),
        "eligible_for_prelabel_formal_design": bool(passed),
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    REPORT.write_text(
        "\n".join(
            [
                "# Market absorption geometry unlabeled feasibility",
                "",
                f"- Backfill complete: {complete}",
                f"- Successful dates: {len(successful)} / {len(expected)}",
                f"- Feature range: {date_min} to {date_max}",
                f"- Non-finite numeric cells: {nonfinite}",
                f"- Minimum common-stock count: {minimum_stocks}",
                f"- Independent historical confirmation cases: {maximum_confirmation_cases}",
                f"- Eligible for pre-label formal design: {passed}",
                "",
                "No prediction target or forward-return label was read.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
