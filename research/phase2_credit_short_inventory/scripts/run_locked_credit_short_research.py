"""Single locked formal test of the credit-short inventory increment."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "research/phase2_credit_short_inventory"
MONTHLY_ENGINE_DIR = ROOT / "research/phase2_monthly_revenue/scripts"
sys.path.insert(0, str(MONTHLY_ENGINE_DIR))

import run_locked_revenue_research as engine


CONFIG = WORKSPACE / "config/locked_credit_short_inventory_v1.json"
SNAPSHOT = WORKSPACE / "data/locked_daily_feature_snapshot.csv"
RESULT = WORKSPACE / "reports/credit_short_inventory_v1.json"
REPORT = WORKSPACE / "reports/credit_short_inventory_v1.md"
RECEIPT = WORKSPACE / "reports/formal_run_receipt.json"
CONTROL = [
    "ret_5",
    "ret_20",
    "ret_60",
    "ret_120",
    "ret_240",
    "vol_ratio",
    "drawdown_120",
    "pulse_rate",
    "inst_foreign_net_20d_z",
    "inst_trust_net_20d_z",
    "inst_dealer_net_20d_z",
]
CREDIT_FEATURES = [
    "cs_log_combined_balance_z",
    "cs_log_margin_balance_z",
    "cs_log_sbl_balance_z",
    "cs_margin_balance_share_z",
    "cs_combined_balance_change_rate_z",
    "cs_margin_increase_breadth_z",
    "cs_sbl_increase_breadth_z",
    "cs_margin_new_short_rate_z",
    "cs_margin_cover_rate_z",
    "cs_sbl_new_short_rate_z",
    "cs_sbl_return_rate_z",
    "cs_positive_balance_breadth_z",
    "cs_top10_share_z",
    "cs_top50_share_z",
    "cs_hhi_z",
    "cs_log_iqr_z",
]
BASELINES = [
    "prior_majority",
    "momentum_20",
    "reversal_20",
    "trend_120",
    "short_build_bearish",
    "short_build_contrarian",
    "margin_build_bearish",
    "sbl_build_bearish",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def assert_config(config: dict) -> None:
    if config["features"]["control"] != CONTROL:
        raise RuntimeError("Control features differ from locked config")
    if config["features"]["treatment_increment"] != CREDIT_FEATURES:
        raise RuntimeError("Credit features differ from locked config")
    if config["baselines"] != BASELINES:
        raise RuntimeError("Baselines differ from locked config")
    if config["walk_forward"] != {
        "history_rows": 1512,
        "inner_validation_rows": 252,
        "test_rows": 252,
        "purge_rows": 20,
        "effective_case_minimum_spacing_rows": 20,
        "window_step_rows": 252,
        "post_hoc_subgroups_allowed": False,
    }:
        raise RuntimeError("Walk-forward settings differ from locked config")
    required_gates = {
        "accuracy": 0.9,
        "effective_cases": 100,
        "coverage": 0.1,
        "wilson_95_lower": 0.8,
        "minimum_material_window_accuracy": 0.8,
        "material_window_cases": 10,
        "must_beat_strongest_simple_baseline": True,
        "paired_p": 0.05,
        "treatment_must_significantly_beat_control": True,
    }
    if config["gates"] != required_gates:
        raise RuntimeError("Formal gates differ from locked requirements")


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert_config(config)
    for relative, digest in config["hashes"].items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError(f"Locked hash mismatch: {relative}")
    receipt = (
        json.loads(RECEIPT.read_text(encoding="utf-8"))
        if RECEIPT.exists()
        else {}
    )
    if receipt.get("status") == "complete":
        raise RuntimeError("Formal run already completed; refusing a second run")
    receipt = {
        "experiment": config["experiment"],
        "status": "running",
        "started_at": receipt.get("started_at")
        or datetime.now(timezone.utc).isoformat(),
        "resume_count": int(receipt.get("resume_count", -1)) + 1,
        "config_sha256": sha256(CONFIG),
        "snapshot_sha256": sha256(SNAPSHOT),
    }
    atomic_json(receipt, RECEIPT)

    frame = pd.read_csv(SNAPSHOT)
    frame["information_date"] = pd.to_datetime(
        frame["information_date"], errors="raise"
    )
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise")
    frame["date"] = frame["decision_date"]
    future = frame["close"].shift(-20) / frame["close"] - 1
    frame["target"] = np.where(future > 0, 1, -1).astype(float)
    frame.loc[future.isna(), "target"] = np.nan

    engine.BASELINES = BASELINES
    control = engine.evaluate(frame, CONTROL)
    treatment = engine.evaluate(frame, CONTROL + CREDIT_FEATURES)
    increment = engine.paired(control, treatment)
    increment_passed = (
        increment["treatment_only"] > increment["control_only"]
        and increment["mcnemar_exact_p"] < config["gates"]["paired_p"]
    )
    control["passed"] = control["passed_base_gate"]
    treatment["passed"] = treatment["passed_base_gate"] and increment_passed
    payload = {
        "experiment": config["experiment"],
        "status": "single locked formal run",
        "target_anchor": (
            "information-date cash close to the cash close 20 trading rows later; "
            "decision occurs the next trading morning after the 21:00 source update"
        ),
        "snapshot_sha256": sha256(SNAPSHOT),
        "aligned_rows": len(frame),
        "variants": {
            "index_aggregate_control": control,
            "plus_credit_short_inventory": treatment,
        },
        "credit_increment": increment,
        "credit_increment_passed": increment_passed,
        "passing": [
            name
            for name, result in {
                "index_aggregate_control": control,
                "plus_credit_short_inventory": treatment,
            }.items()
            if result["passed"]
        ],
    }
    atomic_json(payload, RESULT)
    lines = [
        "# Credit-short inventory v1",
        "",
        "Single locked purged non-overlapping historical OOS run.",
        "",
        "| Variant | Cases | Accuracy | Coverage | Wilson | Strongest baseline | Model/base only | Min block | Passed |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for name, result in payload["variants"].items():
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
        f"Common decisions: {increment['identical_dates']}; control/treatment "
        f"{increment['control_accuracy']:.2%}/{increment['treatment_accuracy']:.2%}; "
        f"exclusive {increment['control_only']}/{increment['treatment_only']}; "
        f"p={increment['mcnemar_exact_p']:.6g}.",
        "",
        f"Passing: {payload['passing']}",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt["status"] = "complete"
    receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
    receipt["result_sha256"] = sha256(RESULT)
    atomic_json(receipt, RECEIPT)
    print(
        json.dumps(
            {"passing": payload["passing"], "increment": increment}, indent=2
        )
    )


if __name__ == "__main__":
    main()

