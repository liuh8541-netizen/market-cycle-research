"""Strict OOS ablation of TX night path microstructure for TWII opening gap."""

import json
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCKED_CONFIG = ROOT / "config" / "locked_night_microstructure_experiment.json"
HISTORY, VALID, TEST = 756, 126, 126
LAMBDAS = [0.1, 1.0, 10.0, 100.0]
QUANTILES = [0.0, 0.25, 0.50, 0.65, 0.75, 0.85]
CONTROL_FEATURES = [
    "tx_night_spread_per", "tx_night_return", "tx_night_range", "log_night_volume",
]
PATH_FEATURES = [
    "first_hour_return", "last_hour_return", "return_acceleration",
    "realized_volatility", "trend_efficiency", "up_bar_share", "close_location",
    "max_drawdown", "max_runup", "first_hour_volume_share", "last_hour_volume_share",
    "signed_volume_ratio", "log_tick_count",
]
VARIANTS = {
    "daily_aggregate_control": CONTROL_FEATURES,
    "daily_plus_microstructure": CONTROL_FEATURES + PATH_FEATURES,
}


def wilson(hits, cases):
    if not cases:
        return 0.0
    z = 1.959963984540054
    p = hits / cases
    den = 1 + z*z/cases
    return (p + z*z/(2*cases) - z*math.sqrt((p*(1-p)+z*z/(4*cases))/cases)) / den


def mcnemar(left_only, right_only):
    n = left_only + right_only
    if not n:
        return 1.0
    low = min(left_only, right_only)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(low + 1)) / 2**n)


def prepare():
    micro = pd.read_csv(ROOT / "data" / "processed" / "factors" / "night_microstructure.csv")
    micro["date"] = pd.to_datetime(micro["signal_date"])
    night = pd.read_csv(ROOT / "data" / "processed" / "taiwan_futures_night.csv")
    night["date"] = pd.to_datetime(night["signal_date"])
    cash = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv", usecols=["date", "open", "close"])
    cash["date"] = pd.to_datetime(cash["date"])
    cash = cash.sort_values("date").drop_duplicates("date", keep="last")
    cash["gap_return"] = pd.to_numeric(cash["open"], errors="coerce") / pd.to_numeric(cash["close"], errors="coerce").shift() - 1
    frame = cash[["date", "gap_return"]].merge(
        night[["date", "tx_night_spread_per", "tx_night_return", "tx_night_range", "tx_night_volume"]],
        on="date", how="inner",
    ).merge(micro.drop(columns=["signal_date", "contract_date"]), on="date", how="inner")
    frame["log_night_volume"] = np.log1p(pd.to_numeric(frame["tx_night_volume"], errors="coerce"))
    frame["log_tick_count"] = np.log1p(pd.to_numeric(frame["tick_count"], errors="coerce"))
    frame["target"] = np.where(frame["gap_return"] >= 0, 1, -1)
    frame["night_spread_baseline"] = np.where(frame["tx_night_spread_per"] >= 0, 1, -1)
    frame["night_return_baseline"] = np.where(frame["tx_night_return"] >= 0, 1, -1)
    return frame.replace([np.inf, -np.inf], np.nan).sort_values("date").reset_index(drop=True)


def fit(frame, features, lam):
    raw = frame[features]
    med = raw.median().fillna(0)
    filled = raw.fillna(med)
    mean = filled.mean()
    std = filled.std().replace(0, 1).fillna(1)
    matrix = ((filled - mean) / std).to_numpy(float)
    matrix = np.column_stack([np.ones(len(matrix)), matrix])
    y = frame["target"].to_numpy(float)
    penalty = np.eye(matrix.shape[1]) * lam
    penalty[0, 0] = 0
    weights = np.linalg.pinv(matrix.T @ matrix + penalty) @ matrix.T @ y
    return weights, med, mean, std


def predict(model, frame, features):
    weights, med, mean, std = model
    raw = frame[features].fillna(med).fillna(0)
    matrix = ((raw - mean) / std).to_numpy(float)
    score = np.column_stack([np.ones(len(matrix)), matrix]) @ weights
    return np.where(score >= 0, 1, -1), np.abs(score)


def choose(history, features):
    fit_frame, valid = history.iloc[:-VALID], history.iloc[-VALID:]
    actual = valid["target"].to_numpy(int)
    best = None
    for lam in LAMBDAS:
        model = fit(fit_frame, features, lam)
        pred, confidence = predict(model, valid, features)
        fit_confidence = predict(model, fit_frame, features)[1]
        for quantile in QUANTILES:
            threshold = float(np.quantile(fit_confidence, quantile))
            use = confidence >= threshold
            cases = int(use.sum())
            if cases < 15 or cases / len(valid) < .10:
                continue
            accuracy = float((pred[use] == actual[use]).mean())
            rank = (accuracy >= .90, accuracy, cases, -lam, -quantile)
            if best is None or rank > best[0]:
                best = (rank, lam, quantile, accuracy, cases)
    return best


def evaluate(frame, features):
    records, windows, tested = [], [], 0
    start = 0
    while start + HISTORY + TEST <= len(frame):
        history = frame.iloc[start:start + HISTORY]
        test = frame.iloc[start + HISTORY:start + HISTORY + TEST]
        tested += len(test)
        config = choose(history, features)
        if config:
            _, lam, quantile, validation_accuracy, validation_cases = config
            model = fit(history, features, lam)
            pred, confidence = predict(model, test, features)
            threshold = float(np.quantile(predict(model, history, features)[1], quantile))
            use = confidence >= threshold
            for pos in np.flatnonzero(use):
                row = test.iloc[pos]
                records.append({
                    "date": str(row["date"].date()), "prediction": int(pred[pos]),
                    "target": int(row["target"]), "hit": int(pred[pos] == row["target"]),
                    "night_spread": int(row["night_spread_baseline"]),
                    "night_return": int(row["night_return_baseline"]),
                    "confidence": float(confidence[pos]), "threshold": threshold,
                    "lambda": lam, "quantile": quantile,
                    "validation_accuracy": validation_accuracy,
                    "validation_cases": validation_cases,
                })
            cases = int(use.sum())
            hits = int((pred[use] == test.loc[use, "target"].to_numpy()).sum())
            windows.append({
                "test_start": str(test["date"].min().date()), "test_end": str(test["date"].max().date()),
                "lambda": lam, "quantile": quantile, "test_cases": cases,
                "test_accuracy": hits / cases if cases else 0,
                "validation_accuracy": validation_accuracy, "validation_cases": validation_cases,
            })
        start += TEST
    cases = len(records)
    hits = sum(row["hit"] for row in records)
    accuracy = hits / cases if cases else 0
    coverage = cases / tested if tested else 0
    baselines = {}
    for name in ["night_spread", "night_return"]:
        baseline_hits = sum(row[name] == row["target"] for row in records)
        baselines[name] = {"hits": baseline_hits, "accuracy": baseline_hits / cases if cases else 0}
    strongest_name = max(baselines, key=lambda name: baselines[name]["accuracy"]) if baselines else ""
    strongest = baselines.get(strongest_name, {"accuracy": 0})
    model_only = sum(row["hit"] and row[strongest_name] != row["target"] for row in records) if records else 0
    baseline_only = sum(not row["hit"] and row[strongest_name] == row["target"] for row in records) if records else 0
    p_value = mcnemar(model_only, baseline_only)
    material = [window for window in windows if window["test_cases"] >= 10]
    minimum_block = min((window["test_accuracy"] for window in material), default=0)
    lower = wilson(hits, cases)
    passed = (
        cases >= 100 and coverage >= .10 and accuracy >= .90 and lower >= .80
        and accuracy > strongest["accuracy"] and model_only > baseline_only
        and p_value < .05 and minimum_block >= .80
    )
    return {
        "tested_rows": tested, "cases": cases, "hits": hits, "accuracy": accuracy,
        "coverage": coverage, "wilson_95_lower": lower, "baselines": baselines,
        "strongest_baseline": strongest_name, "strongest_baseline_accuracy": strongest["accuracy"],
        "model_only": model_only, "baseline_only": baseline_only,
        "mcnemar_exact_p": p_value, "minimum_material_window_accuracy": minimum_block,
        "passed": passed, "windows": windows, "records": records,
    }


def paired(left, right):
    a = {row["date"]: row for row in left["records"]}
    b = {row["date"]: row for row in right["records"]}
    dates = sorted(set(a) & set(b))
    left_only = sum(a[day]["hit"] and not b[day]["hit"] for day in dates)
    right_only = sum(not a[day]["hit"] and b[day]["hit"] for day in dates)
    return {
        "identical_dates": len(dates),
        "left_accuracy": sum(a[day]["hit"] for day in dates) / len(dates) if dates else 0,
        "right_accuracy": sum(b[day]["hit"] for day in dates) / len(dates) if dates else 0,
        "left_only": left_only, "right_only": right_only,
        "mcnemar_exact_p": mcnemar(left_only, right_only),
    }


def main():
    registration = json.loads(LOCKED_CONFIG.read_text(encoding="utf-8"))
    actual_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()
    expected_hash = registration["implementation"]["sha256_at_registration"].upper()
    if actual_hash != expected_hash:
        raise RuntimeError(
            "Locked experiment implementation changed. Register a new experiment id "
            f"instead of rerunning v1 (expected {expected_hash}, actual {actual_hash})."
        )
    frame = prepare()
    results = {name: evaluate(frame, features) for name, features in VARIANTS.items()}
    path_vs_control = paired(results["daily_plus_microstructure"], results["daily_aggregate_control"])
    path_increment_passed = (
        results["daily_plus_microstructure"]["passed"]
        and path_vs_control["left_accuracy"] > path_vs_control["right_accuracy"]
        and path_vs_control["left_only"] > path_vs_control["right_only"]
        and path_vs_control["mcnemar_exact_p"] < .05
    )
    enough_history = len(frame) >= HISTORY + TEST
    payload = {
        "status": (
            "strict historical OOS microstructure experiment; not production unless all gates pass"
            if enough_history else
            f"pending: only {len(frame)} aligned rows cached; at least {HISTORY + TEST} required"
        ),
        "hypothesis": "The within-night TX price/volume path adds opening-gap direction beyond daily night OHLCV and night-spread sign.",
        "causal_rule": "Only ticks from the completed session beginning at prior TWII cash date 15:00 and ending next calendar day 05:00 enter target date D.",
        "method": "630-day ridge fit, following 126-day nested penalty/abstention selection, frozen 126-day OOS blocks; exact-date simple night baselines and path-versus-daily ablation.",
        "gate": ">=90% accuracy, >=100 cases, >=10% coverage, Wilson lower >=80%, significant superiority over strongest identical-date night baseline, every material block >=80%.",
        "history": HISTORY, "validation": VALID, "test": TEST,
        "aligned_rows": len(frame), "enough_history_to_test": enough_history, "variants": {
            name: {"features": VARIANTS[name], "result": result} for name, result in results.items()
        }, "path_vs_control": path_vs_control, "path_increment_passed": path_increment_passed,
        "passing": [name for name, result in results.items() if result["passed"]],
    }
    out = ROOT / "reports"
    (out / "night_microstructure_gap.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# TX night microstructure to TWII opening gap", "", payload["status"], "",
        payload["hypothesis"], "", payload["causal_rule"], "", payload["method"], "",
        payload["gate"], "",
        "| Variant | Cases | Accuracy | Coverage | Wilson lower | Strongest night baseline | Model/base only | p | Min block | Passed |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['cases']} | {result['accuracy']:.2%} | {result['coverage']:.2%} | "
            f"{result['wilson_95_lower']:.2%} | {result['strongest_baseline']} {result['strongest_baseline_accuracy']:.2%} | "
            f"{result['model_only']}/{result['baseline_only']} | {result['mcnemar_exact_p']:.4g} | "
            f"{result['minimum_material_window_accuracy']:.2%} | {result['passed']} |"
        )
    row = path_vs_control
    lines += [
        "", "## Exact-date path ablation", "",
        f"Common dates: {row['identical_dates']}; path {row['left_accuracy']:.2%}; daily control "
        f"{row['right_accuracy']:.2%}; path-only/control-only {row['left_only']}/{row['right_only']}; "
        f"McNemar p={row['mcnemar_exact_p']:.4g}.",
        f"Microstructure increment passed: {path_increment_passed}.",
        "", "Every selected OOS prediction and frozen configuration is retained in the JSON companion file.",
    ]
    (out / "night_microstructure_gap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "aligned_rows": len(frame), "passing": payload["passing"], "path_vs_control": path_vs_control,
        "results": {
            name: {key: value for key, value in result.items() if key not in ["records", "windows", "baselines"]}
            for name, result in results.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
