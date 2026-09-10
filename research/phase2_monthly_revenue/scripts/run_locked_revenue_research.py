"""Single locked purged walk-forward test of monthly-revenue breadth."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "research/phase2_monthly_revenue"
CONFIG = WORKSPACE / "config/locked_monthly_revenue_breadth_v1.json"
SNAPSHOT = WORKSPACE / "data/locked_daily_feature_snapshot.csv"
RESULT = WORKSPACE / "reports/monthly_revenue_breadth_v1.json"
REPORT = WORKSPACE / "reports/monthly_revenue_breadth_v1.md"
RECEIPT = WORKSPACE / "reports/formal_run_receipt.json"

HORIZON = 20
HISTORY = 1512
VALIDATION = 252
TEST = 252
LAMBDAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
QUANTILES = [0.0, 0.25, 0.5, 0.65, 0.75, 0.85]
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
REVENUE_FEATURES = [
    "rev_stock_count_z",
    "rev_log_total_z",
    "rev_log_median_z",
    "rev_log_iqr_z",
    "rev_top10_share_z",
    "rev_top50_share_z",
    "rev_hhi_z",
    "rev_yoy_coverage_z",
    "rev_yoy_median_z",
    "rev_yoy_iqr_z",
    "rev_yoy_positive_z",
    "rev_yoy_above20_z",
    "rev_yoy_below_minus20_z",
    "rev_total_yoy_z",
    "rev_positive_change3_z",
    "rev_median_change3_z",
]
BASELINES = [
    "prior_majority",
    "momentum_20",
    "reversal_20",
    "trend_120",
    "revenue_breadth_sign",
    "revenue_median_sign",
    "revenue_total_yoy_sign",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def wilson(hits: int, cases: int, z: float = 1.959963984540054) -> float:
    if not cases:
        return 0.0
    proportion = hits / cases
    denominator = 1 + z * z / cases
    center = proportion + z * z / (2 * cases)
    spread = z * math.sqrt(
        proportion * (1 - proportion) / cases + z * z / (4 * cases * cases)
    )
    return (center - spread) / denominator


def mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if not discordant:
        return 1.0
    smaller = min(left_only, right_only)
    tail = sum(math.comb(discordant, i) for i in range(smaller + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def fit_model(frame: pd.DataFrame, features: list[str], lam: float):
    raw = frame[features].replace([np.inf, -np.inf], np.nan)
    median = raw.median()
    raw = raw.fillna(median).fillna(0)
    mean = raw.mean()
    std = raw.std().replace(0, 1)
    matrix = ((raw - mean) / std).to_numpy(float)
    matrix = np.column_stack([np.ones(len(matrix)), matrix])
    target = frame["target"].to_numpy(float)
    penalty = np.eye(matrix.shape[1]) * lam
    penalty[0, 0] = 0
    weights = np.linalg.pinv(matrix.T @ matrix + penalty) @ matrix.T @ target
    return weights, median, mean, std


def predict(model, frame: pd.DataFrame, features: list[str]):
    weights, median, mean, std = model
    raw = (
        frame[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(median)
        .fillna(0)
    )
    matrix = ((raw - mean) / std).to_numpy(float)
    matrix = np.column_stack([np.ones(len(matrix)), matrix])
    score = matrix @ weights
    return np.where(score >= 0, 1, -1), np.abs(score)


def select_config(history: pd.DataFrame, features: list[str]):
    split = len(history) - VALIDATION
    fit = history.iloc[: split - HORIZON]
    validation = history.iloc[split : len(history) - HORIZON]
    if len(fit) < 500 or len(validation) < 100:
        return None
    best = None
    for lam in LAMBDAS:
        model = fit_model(fit, features, lam)
        prediction, confidence = predict(model, validation, features)
        target = validation["target"].to_numpy(int)
        for quantile in QUANTILES:
            threshold = float(np.quantile(confidence, quantile))
            use = confidence >= threshold
            cases = int(use.sum())
            if cases < 25 or cases / len(validation) < 0.10:
                continue
            accuracy = float((prediction[use] == target[use]).mean())
            rank = (accuracy, cases / len(validation), -lam, -quantile)
            if best is None or rank > best[0]:
                best = (rank, lam, threshold, quantile, accuracy, cases)
    return best


def greedy_nonoverlap(records: list[dict]) -> list[dict]:
    chosen = []
    last = -10**12
    for record in sorted(records, key=lambda value: value["position"]):
        if record["position"] - last >= HORIZON:
            chosen.append(record)
            last = record["position"]
    return chosen


def count_nonoverlap(positions: list[int]) -> int:
    chosen = []
    last = -10**12
    for position in sorted(set(positions)):
        if position - last >= HORIZON:
            chosen.append(position)
            last = position
    return len(chosen)


def evaluate(frame: pd.DataFrame, features: list[str]) -> dict:
    daily_records = []
    windows = []
    tested_positions: list[int] = []
    tested_rows = 0
    start = 0
    while start + HISTORY + TEST + HORIZON <= len(frame):
        history = frame.iloc[start : start + HISTORY]
        test = frame.iloc[start + HISTORY : start + HISTORY + TEST]
        selection = select_config(history, features)
        tested_rows += len(test)
        tested_positions.extend(test["_twii_position"].astype(int))
        if selection:
            _, lam, threshold, quantile, validation_accuracy, validation_cases = selection
            model = fit_model(history.iloc[:-HORIZON], features, lam)
            prediction, confidence = predict(model, test, features)
            use = confidence >= threshold
            prior = history.iloc[:-HORIZON]["target"].dropna()
            majority = 1 if (prior == 1).mean() >= 0.5 else -1
            block = []
            for local in np.flatnonzero(use):
                row = test.iloc[local]
                if pd.isna(row["target"]):
                    continue
                record = {
                    "date": str(row["date"].date()),
                    "position": int(row["_twii_position"]),
                    "prediction": int(prediction[local]),
                    "target": int(row["target"]),
                    "hit": int(prediction[local] == row["target"]),
                    "confidence": float(confidence[local]),
                    "lambda": lam,
                    "quantile": quantile,
                    "validation_accuracy": validation_accuracy,
                    "validation_cases": validation_cases,
                    "prior_majority": majority,
                }
                for baseline in BASELINES[1:]:
                    record[baseline] = int(row[baseline])
                block.append(record)
                daily_records.append(record)
            effective = greedy_nonoverlap(block)
            windows.append(
                {
                    "test_start": str(test["date"].min().date()),
                    "test_end": str(test["date"].max().date()),
                    "daily_selected": len(block),
                    "effective_cases": len(effective),
                    "effective_accuracy": (
                        sum(item["hit"] for item in effective) / len(effective)
                        if effective
                        else 0.0
                    ),
                    "validation_accuracy": validation_accuracy,
                    "validation_cases": validation_cases,
                }
            )
        start += TEST
    records = greedy_nonoverlap(daily_records)
    cases = len(records)
    hits = sum(record["hit"] for record in records)
    maximum = count_nonoverlap(tested_positions)
    baselines = {}
    for name in BASELINES:
        baseline_hits = sum(record[name] == record["target"] for record in records)
        baselines[name] = {
            "hits": baseline_hits,
            "accuracy": baseline_hits / cases if cases else 0.0,
        }
    strongest = max(baselines, key=lambda name: baselines[name]["accuracy"])
    model_only = sum(
        record["hit"] and record[strongest] != record["target"] for record in records
    )
    baseline_only = sum(
        not record["hit"] and record[strongest] == record["target"] for record in records
    )
    material = [window for window in windows if window["effective_cases"] >= 10]
    result = {
        "tested_daily_rows": tested_rows,
        "maximum_effective_opportunities": maximum,
        "daily_selected_rows": len(daily_records),
        "cases": cases,
        "hits": hits,
        "accuracy": hits / cases if cases else 0.0,
        "coverage": cases / maximum if maximum else 0.0,
        "wilson_95_lower": wilson(hits, cases),
        "baselines": baselines,
        "strongest_baseline": strongest,
        "strongest_baseline_accuracy": baselines[strongest]["accuracy"],
        "model_only": model_only,
        "baseline_only": baseline_only,
        "mcnemar_exact_p": mcnemar(model_only, baseline_only),
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


def paired(control: dict, treatment: dict) -> dict:
    left = {record["date"]: bool(record["hit"]) for record in control["records"]}
    right = {record["date"]: bool(record["hit"]) for record in treatment["records"]}
    dates = sorted(set(left).intersection(right))
    left_only = sum(left[day] and not right[day] for day in dates)
    right_only = sum(right[day] and not left[day] for day in dates)
    return {
        "identical_dates": len(dates),
        "control_accuracy": (
            sum(left[day] for day in dates) / len(dates) if dates else 0.0
        ),
        "treatment_accuracy": (
            sum(right[day] for day in dates) / len(dates) if dates else 0.0
        ),
        "control_only": left_only,
        "treatment_only": right_only,
        "mcnemar_exact_p": mcnemar(left_only, right_only),
    }


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for relative, digest in config["hashes"].items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError(f"Locked hash mismatch: {relative}")
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8")) if RECEIPT.exists() else {}
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
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    future = frame["close"].shift(-HORIZON) / frame["close"] - 1
    frame["target"] = np.where(future > 0, 1, -1).astype(float)
    frame.loc[future.isna(), "target"] = np.nan
    control = evaluate(frame, CONTROL)
    treatment = evaluate(frame, CONTROL + REVENUE_FEATURES)
    increment = paired(control, treatment)
    increment_passed = (
        increment["treatment_only"] > increment["control_only"]
        and increment["mcnemar_exact_p"] < config["gates"]["paired_p"]
    )
    control["passed"] = control["passed_base_gate"]
    treatment["passed"] = treatment["passed_base_gate"] and increment_passed
    payload = {
        "experiment": config["experiment"],
        "status": "single locked formal run",
        "snapshot_sha256": sha256(SNAPSHOT),
        "aligned_rows": len(frame),
        "variants": {
            "index_aggregate_control": control,
            "plus_monthly_revenue_breadth": treatment,
        },
        "revenue_increment": increment,
        "revenue_increment_passed": increment_passed,
        "passing": [
            name
            for name, result in {
                "index_aggregate_control": control,
                "plus_monthly_revenue_breadth": treatment,
            }.items()
            if result["passed"]
        ],
    }
    atomic_json(payload, RESULT)
    lines = [
        "# Monthly-revenue breadth v1",
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
        f"Common dates: {increment['identical_dates']}; control/treatment "
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
