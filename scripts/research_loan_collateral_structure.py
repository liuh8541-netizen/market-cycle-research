"""Locked purged OOS test of nontraditional loan-collateral structure."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_cross_sectional_breadth_cycle as core


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/locked_loan_collateral_structure_v1.json"
DATA = ROOT / "data/processed/factors/loan_collateral_structure.csv"
OUTPUT_JSON = ROOT / "reports/loan_collateral_structure.json"
OUTPUT_MD = ROOT / "reports/loan_collateral_structure.md"

CONTROL = core.CONTROL
LOAN_FEATURES = [
    "lc_log_other_balance_z",
    "lc_other_change_z",
    "lc_other_active_fraction_z",
    "lc_other_utilization_z",
    "lc_other_top10_share_z",
    "lc_other_hhi_z",
    "lc_other_credit_share_z",
    "lc_securities_firm_share_z",
    "lc_unrestricted_share_z",
    "lc_secured_finance_share_z",
    "lc_settlement_share_z",
    "lc_other_balance_change20_z",
    "lc_utilization_change20_z",
    "lc_credit_share_change20_z",
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
    loan = pd.read_csv(DATA)
    loan["date"] = pd.to_datetime(loan["date"], errors="raise")
    frame = frame.merge(loan, on="date", how="inner").sort_values("date")
    transforms = {
        "lc_log_other_balance_z": np.log1p(frame["loan_other_balance"]),
        "lc_other_change_z": frame["loan_other_change"],
        "lc_other_active_fraction_z": frame["loan_other_active_fraction"],
        "lc_other_utilization_z": frame["loan_other_utilization"],
        "lc_other_top10_share_z": frame["loan_other_top10_share"],
        "lc_other_hhi_z": frame["loan_other_hhi"],
        "lc_other_credit_share_z": frame["loan_other_share_of_total_credit"],
        "lc_securities_firm_share_z": frame["loan_securities_firm_share"],
        "lc_unrestricted_share_z": frame["loan_unrestricted_share"],
        "lc_secured_finance_share_z": frame["loan_secured_finance_share"],
        "lc_settlement_share_z": frame["loan_settlement_share"],
        "lc_other_balance_change20_z": np.log1p(frame["loan_other_balance"]).diff(20),
        "lc_utilization_change20_z": frame["loan_other_utilization"].diff(20),
        "lc_credit_share_change20_z": frame[
            "loan_other_share_of_total_credit"
        ].diff(20),
    }
    for name, values in transforms.items():
        frame[name] = causal_z(values)
    required = CONTROL + LOAN_FEATURES
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
        "plus_nontraditional_credit": core.evaluate(frame, CONTROL + LOAN_FEATURES),
    }
    increment = core.paired(
        variants["index_aggregate_control"],
        variants["plus_nontraditional_credit"],
    )
    increment_passed = (
        increment["right_only"] > increment["left_only"]
        and increment["mcnemar_exact_p"] < config["gates"]["paired_p"]
    )
    variants["index_aggregate_control"]["passed"] = variants[
        "index_aggregate_control"
    ]["passed_base_gate"]
    variants["plus_nontraditional_credit"]["passed"] = (
        variants["plus_nontraditional_credit"]["passed_base_gate"]
        and increment_passed
    )
    payload = {
        "experiment": config["experiment"],
        "aligned_rows": len(frame),
        "data_sha256": sha256(DATA),
        "variants": variants,
        "credit_increment": increment,
        "credit_increment_passed": increment_passed,
        "passing": [
            name for name, result in variants.items() if result["passed"]
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Nontraditional loan-collateral structure v1",
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
