"""Single locked confirmation test of market-absorption geometry."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "research/phase3_market_absorption_geometry"
ENGINE_DIR = ROOT / "research/phase2_monthly_revenue/scripts"
sys.path.insert(0, str(ENGINE_DIR))

import run_locked_revenue_research as engine


CONFIG = WORKSPACE / "config/locked_market_absorption_geometry_v1.json"
SNAPSHOT = WORKSPACE / "data/locked_daily_feature_snapshot.csv"
RESULT = WORKSPACE / "reports/market_absorption_geometry_v1.json"
REPORT = WORKSPACE / "reports/market_absorption_geometry_v1.md"
RECEIPT = WORKSPACE / "reports/formal_run_receipt.json"
HORIZON = 20
HISTORY = 1512
TEST = 252
CONFIRM_START = pd.Timestamp("2014-01-01")
CONFIRM_END = pd.Timestamp("2025-12-31")
CONTROL = [
    "ret_5",
    "ret_20",
    "ret_60",
    "ret_120",
    "ret_240",
    "realized_vol_20",
    "vol_ratio_20_120",
    "drawdown_120",
]
RAW_GEOMETRY = [
    "zero_range_fraction",
    "return_up_fraction",
    "return_down_fraction",
    "return_flat_fraction",
    "return_sign_entropy",
    "return_direction_synchronization",
    "close_location_mean",
    "close_location_median",
    "close_location_dispersion",
    "close_location_money_weighted",
    "low_close_fraction",
    "high_close_fraction",
    "low_close_money_fraction",
    "high_close_money_fraction",
    "down_but_recovered_fraction",
    "up_but_faded_fraction",
    "down_but_recovered_money_fraction",
    "up_but_faded_money_fraction",
    "path_efficiency_median",
    "path_efficiency_q75",
    "one_way_path_fraction",
    "churn_path_fraction",
    "signed_efficiency_median",
    "signed_efficiency_money_weighted",
    "range_median",
    "range_dispersion",
    "range_q90",
    "upper_excursion_q90",
    "lower_excursion_q90",
    "excursion_tail_asymmetry",
    "negative_return_money_fraction",
    "positive_return_money_fraction",
]
GEOMETRY = [f"mag_{column}_z" for column in RAW_GEOMETRY]
BASELINES = [
    "prior_majority",
    "momentum_20",
    "reversal_20",
    "trend_120",
    "breadth_sign",
    "close_location_sign",
    "signed_efficiency_sign",
    "absorption_balance_sign",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def assert_locked(config: dict) -> None:
    if config["features"]["control"] != CONTROL:
        raise RuntimeError("Control features differ from locked config")
    if config["features"]["treatment_increment"] != GEOMETRY:
        raise RuntimeError("Geometry features differ from locked config")
    if config["baselines"] != BASELINES:
        raise RuntimeError("Baselines differ from locked config")
    required_walk = {
        "confirmation_start": "2014-01-01",
        "confirmation_end": "2025-12-31",
        "history_rows": 1512,
        "inner_validation_rows": 252,
        "test_rows": 252,
        "purge_rows": 20,
        "effective_case_minimum_spacing_rows": 20,
        "window_step_rows": 252,
        "post_hoc_subgroups_allowed": False,
    }
    if config["walk_forward"] != required_walk:
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


def evaluate_confirmation(frame: pd.DataFrame, features: list[str]) -> dict:
    first = int(np.flatnonzero(frame["decision_date"].ge(CONFIRM_START))[0])
    last = int(np.flatnonzero(frame["decision_date"].le(CONFIRM_END))[-1]) + 1
    daily_records: list[dict] = []
    tested_positions: list[int] = []
    windows: list[dict] = []
    for test_start in range(first, last, TEST):
        test_end = min(test_start + TEST, last)
        history_start = test_start - HISTORY
        if history_start < 0:
            raise RuntimeError("Insufficient pre-confirmation training history")
        history = frame.iloc[history_start:test_start]
        test = frame.iloc[test_start:test_end]
        selection = engine.select_config(history, features)
        tested_positions.extend(test["_twii_position"].astype(int))
        if selection is None:
            windows.append(
                {
                    "test_start": str(test["decision_date"].min().date()),
                    "test_end": str(test["decision_date"].max().date()),
                    "daily_selected": 0,
                    "effective_cases": 0,
                    "effective_accuracy": 0.0,
                    "selection_failed": True,
                }
            )
            continue
        _, lam, threshold, quantile, val_accuracy, val_cases = selection
        model = engine.fit_model(history.iloc[:-HORIZON], features, lam)
        prediction, confidence = engine.predict(model, test, features)
        use = confidence >= threshold
        prior = history.iloc[:-HORIZON]["target"].dropna()
        majority = 1 if (prior == 1).mean() >= 0.5 else -1
        block: list[dict] = []
        for local in np.flatnonzero(use):
            row = test.iloc[local]
            if pd.isna(row["target"]):
                continue
            record = {
                "date": str(row["decision_date"].date()),
                "information_date": str(row["information_date"].date()),
                "position": int(row["_twii_position"]),
                "prediction": int(prediction[local]),
                "target": int(row["target"]),
                "hit": int(prediction[local] == row["target"]),
                "confidence": float(confidence[local]),
                "lambda": float(lam),
                "quantile": float(quantile),
                "validation_accuracy": float(val_accuracy),
                "validation_cases": int(val_cases),
                "prior_majority": int(majority),
            }
            for baseline in BASELINES[1:]:
                record[baseline] = int(row[baseline])
            block.append(record)
            daily_records.append(record)
        effective = engine.greedy_nonoverlap(block)
        windows.append(
            {
                "test_start": str(test["decision_date"].min().date()),
                "test_end": str(test["decision_date"].max().date()),
                "daily_selected": len(block),
                "effective_cases": len(effective),
                "effective_accuracy": (
                    sum(item["hit"] for item in effective) / len(effective)
                    if effective
                    else 0.0
                ),
                "validation_accuracy": float(val_accuracy),
                "validation_cases": int(val_cases),
                "selection_failed": False,
            }
        )
    records = engine.greedy_nonoverlap(daily_records)
    cases = len(records)
    hits = sum(record["hit"] for record in records)
    maximum = engine.count_nonoverlap(tested_positions)
    baselines = {}
    for name in BASELINES:
        baseline_hits = sum(record[name] == record["target"] for record in records)
        baselines[name] = {
            "hits": baseline_hits,
            "accuracy": baseline_hits / cases if cases else 0.0,
        }
    strongest = max(baselines, key=lambda item: baselines[item]["accuracy"])
    model_only = sum(
        record["hit"] and record[strongest] != record["target"]
        for record in records
    )
    baseline_only = sum(
        not record["hit"] and record[strongest] == record["target"]
        for record in records
    )
    material = [window for window in windows if window["effective_cases"] >= 10]
    result = {
        "confirmation_start": str(CONFIRM_START.date()),
        "confirmation_end": str(CONFIRM_END.date()),
        "tested_daily_rows": len(tested_positions),
        "maximum_effective_opportunities": maximum,
        "daily_selected_rows": len(daily_records),
        "cases": cases,
        "hits": hits,
        "accuracy": hits / cases if cases else 0.0,
        "coverage": cases / maximum if maximum else 0.0,
        "wilson_95_lower": engine.wilson(hits, cases),
        "baselines": baselines,
        "strongest_baseline": strongest,
        "strongest_baseline_accuracy": baselines[strongest]["accuracy"],
        "model_only": model_only,
        "baseline_only": baseline_only,
        "mcnemar_exact_p": engine.mcnemar(model_only, baseline_only),
        "minimum_material_window_accuracy": min(
            (window["effective_accuracy"] for window in material), default=0.0
        ),
        "windows": windows,
        "records": records,
    }
    result["passed_base_gate"] = (
        result["accuracy"] >= 0.90
        and cases >= 100
        and result["coverage"] >= 0.10
        and result["wilson_95_lower"] >= 0.80
        and result["minimum_material_window_accuracy"] >= 0.80
        and result["accuracy"] > result["strongest_baseline_accuracy"]
        and model_only > baseline_only
        and result["mcnemar_exact_p"] < 0.05
    )
    return result


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert_locked(config)
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
    future = frame["close"].shift(-HORIZON) / frame["close"] - 1
    frame["target"] = np.where(future > 0, 1, -1).astype(float)
    frame.loc[future.isna(), "target"] = np.nan
    engine.BASELINES = BASELINES
    control = evaluate_confirmation(frame, CONTROL)
    treatment = evaluate_confirmation(frame, CONTROL + GEOMETRY)
    increment = engine.paired(control, treatment)
    increment_passed = (
        increment["treatment_only"] > increment["control_only"]
        and increment["mcnemar_exact_p"] < config["gates"]["paired_p"]
    )
    control["passed"] = control["passed_base_gate"]
    treatment["passed"] = treatment["passed_base_gate"] and increment_passed
    variants = {
        "price_control": control,
        "plus_market_absorption_geometry": treatment,
    }
    payload = {
        "experiment": config["experiment"],
        "status": "single locked formal confirmation run",
        "target_anchor": (
            "completed information-date TWII close to the TWII close 20 "
            "trading rows later; decision availability begins next session"
        ),
        "snapshot_sha256": sha256(SNAPSHOT),
        "aligned_rows": len(frame),
        "variants": variants,
        "geometry_increment": increment,
        "geometry_increment_passed": increment_passed,
        "passing": [name for name, result in variants.items() if result["passed"]],
    }
    atomic_json(payload, RESULT)
    lines = [
        "# Market absorption geometry v1",
        "",
        "Single locked purged non-overlapping historical confirmation run.",
        "",
        "| Variant | Cases | Accuracy | Coverage | Wilson | Strongest baseline | Model/base only | Min block | Passed |",
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
    print(json.dumps({"passing": payload["passing"], "increment": increment}, indent=2))


if __name__ == "__main__":
    main()
