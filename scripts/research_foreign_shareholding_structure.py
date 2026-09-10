"""Locked purged OOS test of foreign-shareholding structure for TWII direction."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_cross_sectional_breadth_cycle as core


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/locked_foreign_shareholding_structure_v1.json"
DATA = ROOT / "data/processed/factors/foreign_shareholding_structure.csv"
OUTPUT_JSON = ROOT / "reports/foreign_shareholding_structure.json"
OUTPUT_MD = ROOT / "reports/foreign_shareholding_structure.md"

CONTROL = core.CONTROL
HOLDING_FEATURES = [
    "fh_stock_count_z",
    "fh_valid_fraction_z",
    "fh_issued_weighted_ratio_z",
    "fh_ratio_median_z",
    "fh_ratio_iqr_z",
    "fh_zero_fraction_z",
    "fh_above10_fraction_z",
    "fh_above20_fraction_z",
    "fh_above40_fraction_z",
    "fh_log_total_shares_z",
    "fh_top10_share_z",
    "fh_hhi_z",
    "fh_remaining_ratio_median_z",
    "fh_restricted_fraction_z",
    "fh_weighted_ratio_change20_z",
    "fh_median_change20_z",
    "fh_above20_change20_z",
    "fh_top10_change20_z",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def causal_z(series, window=252, minimum=126):
    series = pd.to_numeric(series, errors="coerce")
    mean = series.rolling(window, min_periods=minimum).mean()
    std = series.rolling(window, min_periods=minimum).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def prepare():
    frame = core.prepare()
    holding = pd.read_csv(DATA)
    holding["date"] = pd.to_datetime(holding["date"], errors="raise")
    frame = frame.merge(holding, on="date", how="inner").sort_values("date")
    transforms = {
        "fh_stock_count_z": frame["foreign_holding_stock_count"],
        "fh_valid_fraction_z": frame["foreign_holding_valid_fraction"],
        "fh_issued_weighted_ratio_z": frame["foreign_holding_issued_weighted_ratio"],
        "fh_ratio_median_z": frame["foreign_holding_ratio_median"],
        "fh_ratio_iqr_z": frame["foreign_holding_ratio_iqr"],
        "fh_zero_fraction_z": frame["foreign_holding_zero_fraction"],
        "fh_above10_fraction_z": frame["foreign_holding_above10_fraction"],
        "fh_above20_fraction_z": frame["foreign_holding_above20_fraction"],
        "fh_above40_fraction_z": frame["foreign_holding_above40_fraction"],
        "fh_log_total_shares_z": np.log1p(frame["foreign_holding_total_shares"]),
        "fh_top10_share_z": frame["foreign_holding_top10_share"],
        "fh_hhi_z": frame["foreign_holding_hhi"],
        "fh_remaining_ratio_median_z": frame["foreign_remaining_ratio_median"],
        "fh_restricted_fraction_z": frame["foreign_upper_limit_restricted_fraction"],
        "fh_weighted_ratio_change20_z": frame[
            "foreign_holding_issued_weighted_ratio"
        ].diff(20),
        "fh_median_change20_z": frame["foreign_holding_ratio_median"].diff(20),
        "fh_above20_change20_z": frame[
            "foreign_holding_above20_fraction"
        ].diff(20),
        "fh_top10_change20_z": frame["foreign_holding_top10_share"].diff(20),
    }
    for name, values in transforms.items():
        frame[name] = causal_z(values)
    required = CONTROL + HOLDING_FEATURES
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
        "plus_foreign_shareholding": core.evaluate(frame, CONTROL + HOLDING_FEATURES),
    }
    increment = core.paired(
        variants["index_aggregate_control"],
        variants["plus_foreign_shareholding"],
    )
    increment_passed = (
        increment["right_only"] > increment["left_only"]
        and increment["mcnemar_exact_p"] < config["gates"]["paired_p"]
    )
    variants["index_aggregate_control"]["passed"] = variants[
        "index_aggregate_control"
    ]["passed_base_gate"]
    variants["plus_foreign_shareholding"]["passed"] = (
        variants["plus_foreign_shareholding"]["passed_base_gate"]
        and increment_passed
    )
    payload = {
        "experiment": config["experiment"],
        "aligned_rows": len(frame),
        "data_sha256": sha256(DATA),
        "variants": variants,
        "foreign_shareholding_increment": increment,
        "foreign_shareholding_increment_passed": increment_passed,
        "passing": [
            name for name, result in variants.items() if result["passed"]
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Foreign-shareholding structure v1",
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
