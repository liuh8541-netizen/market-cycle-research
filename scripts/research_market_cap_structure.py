"""Locked purged OOS test of market-cap structure for TWII direction."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_cross_sectional_breadth_cycle as core


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/locked_market_cap_structure_v1.json"
DATA = ROOT / "data/processed/factors/market_cap_structure.csv"
OUTPUT_JSON = ROOT / "reports/market_cap_structure.json"
OUTPUT_MD = ROOT / "reports/market_cap_structure.md"

CONTROL = core.CONTROL
CAP_FEATURES = [
    "cap_stock_count_z",
    "cap_log_total_z",
    "cap_top1_share_z",
    "cap_top5_share_z",
    "cap_top10_share_z",
    "cap_top20_share_z",
    "cap_top50_share_z",
    "cap_bottom_half_share_z",
    "cap_hhi_z",
    "cap_log_median_z",
    "cap_log_iqr_z",
    "cap_log_q90_q10_z",
    "cap_top10_change20_z",
    "cap_hhi_change20_z",
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
    cap = pd.read_csv(DATA)
    cap["date"] = pd.to_datetime(cap["date"], errors="raise")
    frame = frame.merge(cap, on="date", how="inner").sort_values("date")
    transforms = {
        "cap_stock_count_z": frame["market_cap_stock_count"],
        "cap_log_total_z": np.log1p(frame["market_cap_total"]),
        "cap_top1_share_z": frame["market_cap_top1_share"],
        "cap_top5_share_z": frame["market_cap_top5_share"],
        "cap_top10_share_z": frame["market_cap_top10_share"],
        "cap_top20_share_z": frame["market_cap_top20_share"],
        "cap_top50_share_z": frame["market_cap_top50_share"],
        "cap_bottom_half_share_z": frame["market_cap_bottom_half_share"],
        "cap_hhi_z": frame["market_cap_hhi"],
        "cap_log_median_z": frame["market_cap_log_median"],
        "cap_log_iqr_z": frame["market_cap_log_iqr"],
        "cap_log_q90_q10_z": frame["market_cap_log_q90_q10"],
        "cap_top10_change20_z": frame["market_cap_top10_share"].diff(20),
        "cap_hhi_change20_z": frame["market_cap_hhi"].diff(20),
    }
    for name, values in transforms.items():
        frame[name] = causal_z(values)
    required = CONTROL + CAP_FEATURES
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
        "plus_market_cap_structure": core.evaluate(frame, CONTROL + CAP_FEATURES),
    }
    increment = core.paired(
        variants["index_aggregate_control"],
        variants["plus_market_cap_structure"],
    )
    increment_passed = (
        increment["right_only"] > increment["left_only"]
        and increment["mcnemar_exact_p"] < config["gates"]["paired_p"]
    )
    variants["index_aggregate_control"]["passed"] = variants[
        "index_aggregate_control"
    ]["passed_base_gate"]
    variants["plus_market_cap_structure"]["passed"] = (
        variants["plus_market_cap_structure"]["passed_base_gate"]
        and increment_passed
    )
    payload = {
        "experiment": config["experiment"],
        "aligned_rows": len(frame),
        "data_sha256": sha256(DATA),
        "variants": variants,
        "market_cap_increment": increment,
        "market_cap_increment_passed": increment_passed,
        "passing": [
            name for name, result in variants.items() if result["passed"]
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Market-cap structure v1",
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
