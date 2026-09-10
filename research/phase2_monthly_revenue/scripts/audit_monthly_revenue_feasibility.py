"""Unlabeled feasibility audit for monthly-revenue breadth."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "research/phase2_monthly_revenue"
DATA = WORKSPACE / "data/monthly_revenue_breadth.csv"
MANIFEST = WORKSPACE / "data/monthly_revenue_fetched_months.csv"
TWII = ROOT / "data/processed/twii_daily.csv"
OUT_JSON = WORKSPACE / "reports/monthly_revenue_feasibility.json"
OUT_MD = WORKSPACE / "reports/monthly_revenue_feasibility.md"


def main() -> None:
    data = pd.read_csv(DATA)
    manifest = pd.read_csv(MANIFEST)
    twii = pd.read_csv(TWII, usecols=["date"])
    data["source_date"] = pd.to_datetime(data["source_date"], errors="coerce")
    data["available_date"] = pd.to_datetime(data["available_date"], errors="coerce")
    twii["date"] = pd.to_datetime(twii["date"], errors="coerce")
    numeric = [column for column in data if column not in {"source_date", "available_date"}]
    numbers = data[numeric].apply(pd.to_numeric, errors="coerce")
    constant = [
        column for column in numeric
        if numbers[column].dropna().nunique() <= 1
    ]
    invalid = {
        column: int((~np.isfinite(numbers[column])).sum())
        for column in numeric
        if (~np.isfinite(numbers[column])).any()
    }
    aligned = pd.merge_asof(
        twii.dropna().sort_values("date"),
        data.sort_values("available_date"),
        left_on="date",
        right_on="available_date",
        direction="backward",
    )
    aligned = aligned.dropna(subset=["source_date"])
    minimum_rows = 1512 + 252 + 252 + 20
    maximum_effective_cases = max(0, (len(aligned) - 1512 - 252 - 20) // 20)
    checks = {
        "manifest_all_success": bool(manifest["status"].eq("success").all()),
        "unique_source_months": bool(len(data) == data["source_date"].nunique()),
        "dates_valid": bool(data[["source_date", "available_date"]].notna().all().all()),
        "availability_strictly_lagged": bool(
            (data["available_date"] > data["source_date"]).all()
        ),
        "numeric_finite": not invalid,
        "no_constant_fields": not constant,
        "sufficient_aligned_rows": len(aligned) >= minimum_rows,
        "maximum_effective_cases_at_least_100": maximum_effective_cases >= 100,
        "early_yoy_coverage": bool(
            pd.to_numeric(data["revenue_yoy_valid_count"], errors="coerce")
            .head(min(60, len(data)))
            .median()
            >= 100
        ),
    }
    payload = {
        "status": "unlabeled feasibility audit; no forecast target was read",
        "source_months": len(data),
        "source_start": data["source_date"].min().date().isoformat(),
        "source_end": data["source_date"].max().date().isoformat(),
        "available_start": data["available_date"].min().date().isoformat(),
        "available_end": data["available_date"].max().date().isoformat(),
        "manifest_months": len(manifest),
        "aligned_twii_rows": len(aligned),
        "minimum_required_rows": minimum_rows,
        "maximum_effective_20d_cases": maximum_effective_cases,
        "constant_fields": constant,
        "nonfinite_counts": invalid,
        "checks": checks,
        "passed": all(checks.values()),
        "decision": (
            "Eligible to define and lock one formal treatment before reading labels."
            if all(checks.values())
            else "Do not fit a target; repair or reject the source first."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Monthly-revenue breadth feasibility",
        "",
        payload["status"],
        "",
        f"- Source months: {payload['source_months']} "
        f"({payload['source_start']} to {payload['source_end']})",
        f"- Available-date range: {payload['available_start']} to {payload['available_end']}",
        f"- Aligned TWII rows: {payload['aligned_twii_rows']}",
        f"- Maximum independent 20-day cases: {maximum_effective_cases}",
        f"- Constant fields: {constant}",
        f"- Non-finite counts: {invalid}",
        "",
        "| Check | Passed |",
        "| --- | --- |",
    ]
    lines.extend(f"| {name} | {value} |" for name, value in checks.items())
    lines += ["", f"Decision: {payload['decision']}"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not payload["passed"]:
        raise RuntimeError(payload["decision"])


if __name__ == "__main__":
    main()
