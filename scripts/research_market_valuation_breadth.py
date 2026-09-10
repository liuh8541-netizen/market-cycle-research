"""Locked purged OOS test of all-market valuation breadth for TWII direction."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_cross_sectional_breadth_cycle as core


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/locked_market_valuation_breadth_v1.json"
DATA = ROOT / "data/processed/factors/market_valuation_breadth.csv"
OUTPUT_JSON = ROOT / "reports/market_valuation_breadth.json"
OUTPUT_MD = ROOT / "reports/market_valuation_breadth.md"

CONTROL = core.CONTROL
RAW_VALUATION = [
    "valuation_stock_count",
    "per_valid_fraction",
    "per_log_median",
    "per_log_iqr",
    "per_below_10_fraction",
    "per_above_30_fraction",
    "pbr_valid_fraction",
    "pbr_log_median",
    "pbr_log_iqr",
    "pbr_below_1_fraction",
    "yield_valid_fraction",
    "yield_median",
    "yield_above_4_fraction",
]
VALUATION = [f"valuation_{column}_z" for column in RAW_VALUATION]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def causal_z(series, window=252, minimum=126):
    series = pd.to_numeric(series, errors="coerce")
    mean = series.rolling(window, min_periods=minimum).mean()
    std = series.rolling(window, min_periods=minimum).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def prepare():
    frame = core.prepare()
    valuation = pd.read_csv(DATA)
    valuation["date"] = pd.to_datetime(valuation["date"], errors="raise")
    frame = frame.merge(valuation, on="date", how="inner").sort_values("date")
    for raw, transformed in zip(RAW_VALUATION, VALUATION):
        frame[transformed] = causal_z(frame[raw])
    required = CONTROL + VALUATION
    frame[required] = frame[required].replace([np.inf, -np.inf], np.nan)
    return frame.dropna(subset=required).reset_index(drop=True)


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for relative, expected in config["implementation_hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Locked hash mismatch: {relative}")
    core.BASELINES = config["baselines"]
    frame = prepare()
    variants = {
        "index_aggregate_control": core.evaluate(frame, CONTROL),
        "plus_market_valuation_breadth": core.evaluate(frame, CONTROL + VALUATION),
    }
    increment = core.paired(
        variants["index_aggregate_control"],
        variants["plus_market_valuation_breadth"],
    )
    increment_passed = (
        increment["right_only"] > increment["left_only"]
        and increment["mcnemar_exact_p"] < config["gates"]["paired_p"]
    )
    variants["index_aggregate_control"]["passed"] = variants[
        "index_aggregate_control"
    ]["passed_base_gate"]
    variants["plus_market_valuation_breadth"]["passed"] = (
        variants["plus_market_valuation_breadth"]["passed_base_gate"]
        and increment_passed
    )
    payload = {
        "experiment": config["experiment"],
        "aligned_rows": len(frame),
        "data_sha256": sha256(DATA),
        "variants": variants,
        "valuation_increment": increment,
        "valuation_increment_passed": increment_passed,
        "passing": [
            name for name, result in variants.items() if result["passed"]
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Market valuation breadth v1",
        "",
        f"- Aligned causal rows: {len(frame)}",
        "",
        "| Variant | Cases | Accuracy | Coverage | Wilson lower | Strongest baseline | Model/base only | Min block | Passed |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for name, result in variants.items():
        lines.append(
            f"| {name} | {result['cases']} | {result['accuracy']:.2%} | "
            f"{result['coverage']:.2%} | {result['wilson_95_lower']:.2%} | "
            f"{result['strongest_baseline']} "
            f"{result['strongest_baseline_accuracy']:.2%} | "
            f"{result['model_only']}/{result['baseline_only']} "
            f"(p={result['mcnemar_exact_p']:.4g}) | "
            f"{result['minimum_material_window_accuracy']:.2%} | "
            f"{result['passed']} |"
        )
    lines += [
        "",
        f"Common dates: {increment['identical_dates']}; control/treatment "
        f"{increment['left_accuracy']:.2%}/{increment['right_accuracy']:.2%}; "
        f"exclusive control/treatment {increment['left_only']}/"
        f"{increment['right_only']}; p={increment['mcnemar_exact_p']:.6g}.",
        "",
        f"Passing variants: {payload['passing']}",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
