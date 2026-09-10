"""Locked nested OOS ablation of the first 15 TAIEX minutes for remaining-day direction."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HISTORY, VALID, TEST = 756, 126, 126
LAMBDAS = [0.1, 1.0, 10.0, 100.0]
QUANTILES = [0.0, 0.25, 0.5, 0.65, 0.75, 0.85]
CONTROL = [
    "opening_gap", "night_spread", "night_return",
    "prior_cash_1d", "prior_cash_5d", "prior_cash_20d", "prior_cash_vol20",
]
PULSE = [
    "early_return", "first5_return", "last5_return", "return_acceleration",
    "early_range", "realized_volatility", "trend_efficiency", "close_location",
    "linear_slope_15m", "gap_x_early", "night_x_early",
]
VARIANTS = {"open_control": CONTROL, "open_plus_early_pulse": CONTROL + PULSE}


def prepare():
    cash = pd.read_csv(ROOT / "data/processed/twii_daily.csv")
    cash["date"] = pd.to_datetime(cash["date"])
    cash["close"] = pd.to_numeric(cash["close"], errors="coerce")
    cash = cash.sort_values("date").drop_duplicates("date", keep="last")
    cash["cash_return"] = cash["close"].pct_change()
    cash["prior_close"] = cash["close"].shift(1)
    cash["prior_cash_1d"] = cash["cash_return"].shift(1)
    cash["prior_cash_5d"] = cash["cash_return"].shift(1).rolling(5).sum()
    cash["prior_cash_20d"] = cash["cash_return"].shift(1).rolling(20).sum()
    cash["prior_cash_vol20"] = cash["cash_return"].shift(1).rolling(20).std()

    night = pd.read_csv(ROOT / "data/processed/taiwan_futures_night.csv")
    night["date"] = pd.to_datetime(night["signal_date"])
    night = night.rename(columns={
        "tx_night_spread_per": "night_spread",
        "tx_night_return": "night_return",
    })

    pulse = pd.read_csv(ROOT / "data/processed/factors/taiex_early_pulse.csv")
    pulse["date"] = pd.to_datetime(pulse["date"])
    pulse = pulse.sort_values("date").drop_duplicates("date", keep="last")
    numeric = [
        "open_0900", "price_0915", "early_return", "first5_return",
        "last5_return", "return_acceleration", "early_range",
        "realized_volatility", "trend_efficiency", "close_location",
        "linear_slope_15m",
    ]
    pulse[numeric] = pulse[numeric].apply(pd.to_numeric, errors="coerce")

    frame = cash.merge(
        night[["date", "night_spread", "night_return"]], on="date", how="inner"
    ).merge(pulse[["date"] + numeric], on="date", how="inner")
    frame["opening_gap"] = frame["open_0900"] / frame["prior_close"] - 1
    frame["remaining_return"] = frame["close"] / frame["price_0915"] - 1
    frame["gap_x_early"] = frame["opening_gap"] * frame["early_return"]
    frame["night_x_early"] = frame["night_spread"] * frame["early_return"]
    frame["target"] = np.where(frame["remaining_return"] >= 0, 1, -1)
    frame["early_momentum"] = np.where(frame["early_return"] >= 0, 1, -1)
    frame["early_reversal"] = -frame["early_momentum"]
    frame["gap_continuation"] = np.where(frame["opening_gap"] >= 0, 1, -1)
    frame["night_direction"] = np.where(frame["night_spread"] >= 0, 1, -1)
    return frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["remaining_return", "price_0915", "early_return", "opening_gap", "night_spread"]
    ).sort_values("date").reset_index(drop=True)


def fit(frame, features, lam):
    raw = frame[features].replace([np.inf, -np.inf], np.nan)
    median = raw.median().fillna(0)
    filled = raw.fillna(median)
    mean, std = filled.mean(), filled.std().replace(0, 1).fillna(1)
    x = np.column_stack([np.ones(len(filled)), ((filled - mean) / std).to_numpy(float)])
    y = frame["target"].to_numpy(float)
    penalty = np.eye(x.shape[1]) * lam
    penalty[0, 0] = 0
    weights = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y
    return weights, median, mean, std


def predict(model, frame, features):
    weights, median, mean, std = model
    raw = frame[features].replace([np.inf, -np.inf], np.nan).fillna(median).fillna(0)
    x = np.column_stack([np.ones(len(raw)), ((raw - mean) / std).to_numpy(float)])
    score = x @ weights
    return np.where(score >= 0, 1, -1), np.abs(score)


def choose(history, features):
    train, valid = history.iloc[:-VALID], history.iloc[-VALID:]
    actual = valid["target"].to_numpy(int)
    best = None
    for lam in LAMBDAS:
        model = fit(train, features, lam)
        pred, confidence = predict(model, valid, features)
        train_confidence = predict(model, train, features)[1]
        for quantile in QUANTILES:
            cutoff = float(np.quantile(train_confidence, quantile))
            use = confidence >= cutoff
            cases = int(use.sum())
            if cases < 15 or use.mean() < 0.10:
                continue
            accuracy = float((pred[use] == actual[use]).mean())
            rank = (accuracy >= 0.90, accuracy, cases, -lam, -quantile)
            if best is None or rank > best[0]:
                best = (rank, lam, quantile, accuracy, cases)
    return best


def wilson(hits, cases):
    if not cases:
        return 0
    z, probability = 1.959963984540054, hits / cases
    denominator = 1 + z * z / cases
    return (
        probability + z * z / (2 * cases)
        - z * math.sqrt((probability * (1 - probability) + z * z / (4 * cases)) / cases)
    ) / denominator


def mcnemar(left_only, right_only):
    discordant = left_only + right_only
    if not discordant:
        return 1
    lower = min(left_only, right_only)
    return min(1, 2 * sum(math.comb(discordant, k) for k in range(lower + 1)) / 2**discordant)


def evaluate(frame, features):
    records, windows, tested = [], [], 0
    baseline_names = ["early_momentum", "early_reversal", "gap_continuation", "night_direction"]
    for start in range(0, len(frame) - HISTORY - TEST + 1, TEST):
        history = frame.iloc[start:start + HISTORY]
        test = frame.iloc[start + HISTORY:start + HISTORY + TEST]
        tested += len(test)
        selection = choose(history, features)
        if not selection:
            continue
        _, lam, quantile, validation_accuracy, validation_cases = selection
        model = fit(history, features, lam)
        prediction, confidence = predict(model, test, features)
        cutoff = float(np.quantile(predict(model, history, features)[1], quantile))
        use = confidence >= cutoff
        for position in np.flatnonzero(use):
            row = test.iloc[position]
            record = {
                "date": str(row.date.date()),
                "prediction": int(prediction[position]),
                "target": int(row.target),
                "hit": int(prediction[position] == row.target),
                "confidence": float(confidence[position]),
                "lambda": lam,
                "quantile": quantile,
            }
            record.update({name: int(row[name]) for name in baseline_names})
            records.append(record)
        cases = int(use.sum())
        windows.append({
            "test_start": str(test.date.min().date()),
            "test_end": str(test.date.max().date()),
            "cases": cases,
            "accuracy": float((prediction[use] == test.loc[use, "target"]).mean()) if cases else 0,
            "validation_accuracy": validation_accuracy,
            "validation_cases": validation_cases,
        })
    cases, hits = len(records), sum(row["hit"] for row in records)
    baselines = {}
    for name in baseline_names:
        base_hits = sum(row[name] == row["target"] for row in records)
        baselines[name] = {"hits": base_hits, "accuracy": base_hits / cases if cases else 0}
    strongest = max(baselines, key=lambda name: baselines[name]["accuracy"])
    model_only = sum(row["hit"] and row[strongest] != row["target"] for row in records)
    baseline_only = sum(not row["hit"] and row[strongest] == row["target"] for row in records)
    accuracy = hits / cases if cases else 0
    material = [window for window in windows if window["cases"] >= 10]
    result = {
        "tested_rows": tested,
        "cases": cases,
        "hits": hits,
        "accuracy": accuracy,
        "coverage": cases / tested if tested else 0,
        "wilson_95_lower": wilson(hits, cases),
        "baselines": baselines,
        "strongest_baseline": strongest,
        "strongest_baseline_accuracy": baselines[strongest]["accuracy"],
        "model_only": model_only,
        "baseline_only": baseline_only,
        "mcnemar_exact_p": mcnemar(model_only, baseline_only),
        "minimum_material_window_accuracy": min(
            (window["accuracy"] for window in material), default=0
        ),
        "windows": windows,
        "records": records,
    }
    result["passed"] = (
        cases >= 100
        and result["coverage"] >= 0.10
        and accuracy >= 0.90
        and result["wilson_95_lower"] >= 0.80
        and accuracy > result["strongest_baseline_accuracy"]
        and model_only > baseline_only
        and result["mcnemar_exact_p"] < 0.05
        and result["minimum_material_window_accuracy"] >= 0.80
    )
    return result


def paired(control, treatment):
    left, right = ({row["date"]: row for row in result["records"]} for result in [control, treatment])
    dates = sorted(left.keys() & right.keys())
    control_only = sum(left[day]["hit"] and not right[day]["hit"] for day in dates)
    pulse_only = sum(right[day]["hit"] and not left[day]["hit"] for day in dates)
    return {
        "identical_dates": len(dates),
        "control_accuracy": sum(left[d]["hit"] for d in dates) / len(dates) if dates else 0,
        "pulse_accuracy": sum(right[d]["hit"] for d in dates) / len(dates) if dates else 0,
        "control_only": control_only,
        "pulse_only": pulse_only,
        "mcnemar_exact_p": mcnemar(control_only, pulse_only),
    }


def main():
    frame = prepare()
    results = {name: evaluate(frame, features) for name, features in VARIANTS.items()}
    comparison = paired(results["open_control"], results["open_plus_early_pulse"])
    increment = (
        comparison["pulse_only"] > comparison["control_only"]
        and comparison["mcnemar_exact_p"] < 0.05
    )
    results["open_plus_early_pulse"]["passed"] &= increment
    payload = {
        "experiment_id": "taiex_early_pulse_close_v1",
        "status": "strict historical OOS; not production unless every gate passes",
        "target": "same-day TWII close versus the observed 09:15 TAIEX level",
        "decision_time": "09:15:00 Asia/Taipei after exactly fifteen cash-session minutes",
        "hypothesis": "The first 15-minute TAIEX path adds remaining-day direction beyond the opening gap, TX night direction, and lagged cash state.",
        "method": "630/126/126 nested ridge/abstention, fixed 09:15 cutoff, exact-date ablation.",
        "gate": ">=90% accuracy, >=100 cases, >=10% coverage, Wilson >=80%, min block >=80%, significant superiority over strongest simple 09:15 baseline and open-only control.",
        "aligned_rows": len(frame),
        "variants": results,
        "pulse_increment": comparison,
        "pulse_increment_passed": increment,
        "passing": [name for name, result in results.items() if result["passed"]],
    }
    out = ROOT / "reports"
    (out / "taiex_early_pulse_close.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# TAIEX early-pulse remaining-day research", "", payload["status"], "",
        payload["hypothesis"], "",
        "| Variant | Cases | Accuracy | Coverage | Wilson lower | Strongest baseline | Model/base only | Min block | Passed |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['cases']} | {result['accuracy']:.2%} | "
            f"{result['coverage']:.2%} | {result['wilson_95_lower']:.2%} | "
            f"{result['strongest_baseline']} {result['strongest_baseline_accuracy']:.2%} | "
            f"{result['model_only']}/{result['baseline_only']} "
            f"(p={result['mcnemar_exact_p']:.4g}) | "
            f"{result['minimum_material_window_accuracy']:.2%} | {result['passed']} |"
        )
    lines += [
        "",
        f"Common control/pulse: {comparison['control_accuracy']:.2%}/"
        f"{comparison['pulse_accuracy']:.2%}; exclusive wins "
        f"{comparison['control_only']}/{comparison['pulse_only']}; "
        f"p={comparison['mcnemar_exact_p']:.4g}.",
        f"Independent early-pulse increment passed: {increment}.",
    ]
    (out / "taiex_early_pulse_close.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    concise = {
        "aligned_rows": len(frame),
        "variants": {
            name: {key: value for key, value in result.items() if key not in ("records", "windows", "baselines")}
            for name, result in results.items()
        },
        "pulse_increment": comparison,
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
