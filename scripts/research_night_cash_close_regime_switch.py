"""Causal regime-switch test for night-spread transmission to TWII cash close."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_long_history_night_external as source
from research_balanced_external_failure import fit_balanced, score


HISTORY, VALID, TEST = 756, 126, 126
ROUNDS = [10, 25, 50]
LEARNING_RATES = [0.05, 0.10, 0.20]
CONFIDENCE_QUANTILES = [0.0, 0.25, 0.50, 0.65, 0.75, 0.85]
FLIP_QUANTILES = [0.0, 0.25, 0.50, 0.75]
BASE_FEATURES = [
    "tx_night_return", "tx_night_range", "tx_night_volume", "tx_night_spread_per",
    "tx_night_return_z", "tx_night_range_z", "tx_night_volume_z", "tx_night_spread_per_z",
    "night_external_divergence", "night_range_external_state",
] + source.EXTERNAL + [
    "external_lag_days", "weekday", "external_consensus_abs",
    "external_return_mean", "external_return_dispersion",
]
REGIME_FEATURES = [
    "prior_cash_return_1d", "prior_cash_return_5d", "prior_cash_return_20d",
    "prior_cash_vol_20d", "prior_cash_vol_60d",
    "past_night_hit_20d", "past_night_hit_60d", "past_night_hit_252d",
    "night_external_agreement",
]
FEATURES = BASE_FEATURES + REGIME_FEATURES


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
    frame = source.prepare().copy()
    cash = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv", usecols=["date", "close"])
    cash["date"] = pd.to_datetime(cash["date"])
    cash = cash.sort_values("date").drop_duplicates("date", keep="last")
    cash["cash_return"] = pd.to_numeric(cash["close"], errors="coerce").pct_change()
    frame = frame.merge(cash[["date", "cash_return"]], on="date", how="left")
    frame["close_target"] = np.where(frame["cash_return"] >= 0, 1, -1)
    frame["night_baseline"] = np.where(frame["tx_night_spread_per"] >= 0, 1, -1)
    frame["night_return_baseline"] = np.where(frame["tx_night_return"] >= 0, 1, -1)
    frame["external_baseline"] = frame["baseline"].astype(int)
    frame["failure_target"] = np.where(frame["night_baseline"] == frame["close_target"], 1, -1)

    # Only outcomes strictly before date D enter the state observed for D.
    hit = (frame["night_baseline"] == frame["close_target"]).astype(float)
    for window in [20, 60, 252]:
        frame[f"past_night_hit_{window}d"] = hit.shift(1).rolling(window, min_periods=max(10, window // 4)).mean()
    ret = frame["cash_return"]
    frame["prior_cash_return_1d"] = ret.shift(1)
    frame["prior_cash_return_5d"] = ret.shift(1).rolling(5, min_periods=5).sum()
    frame["prior_cash_return_20d"] = ret.shift(1).rolling(20, min_periods=20).sum()
    frame["prior_cash_vol_20d"] = ret.shift(1).rolling(20, min_periods=20).std()
    frame["prior_cash_vol_60d"] = ret.shift(1).rolling(60, min_periods=30).std()
    frame["night_external_agreement"] = frame["night_baseline"] * frame["external_baseline"]
    return frame.dropna(subset=["cash_return", "tx_night_spread_per"]).reset_index(drop=True)


def choose(fit_frame, valid):
    actual = valid["close_target"].to_numpy(int)
    baseline = valid["night_baseline"].to_numpy(int)
    fit_abs = fit_frame["tx_night_spread_per"].abs()
    best = None
    for learning_rate in LEARNING_RATES:
        full = fit_balanced(fit_frame, FEATURES, max(ROUNDS), learning_rate)
        for rounds in ROUNDS:
            model = {"stumps": full["stumps"][:rounds], "median": full["median"]}
            validation_score = score(model, valid, FEATURES)
            negative = -validation_score[validation_score < 0]
            flip_cuts = np.unique(np.quantile(negative, FLIP_QUANTILES)) if len(negative) else [np.inf]
            for confidence_quantile in CONFIDENCE_QUANTILES:
                confidence_cut = float(fit_abs.quantile(confidence_quantile))
                use = valid["tx_night_spread_per"].abs().to_numpy() >= confidence_cut
                if use.sum() < 15 or use.mean() < .10:
                    continue
                for flip_cut in flip_cuts:
                    flip = validation_score <= -float(flip_cut)
                    prediction = baseline.copy()
                    prediction[flip] *= -1
                    cases = int(use.sum())
                    accuracy = float((prediction[use] == actual[use]).mean())
                    baseline_accuracy = float((baseline[use] == actual[use]).mean())
                    model_only = int(((prediction == actual) & (baseline != actual) & use).sum())
                    baseline_only = int(((prediction != actual) & (baseline == actual) & use).sum())
                    gate = accuracy >= .90 and accuracy > baseline_accuracy and model_only > baseline_only
                    rank = (
                        gate, accuracy, accuracy - baseline_accuracy, model_only - baseline_only,
                        -int((flip & use).sum()), cases, -rounds, -learning_rate,
                    )
                    if best is None or rank > best[0]:
                        best = (
                            rank, model, float(flip_cut), confidence_cut, confidence_quantile,
                            rounds, learning_rate, accuracy, baseline_accuracy, cases,
                            model_only, baseline_only,
                        )
    return best


def evaluate(frame):
    records, windows, tested = [], [], 0
    start = 0
    while start + HISTORY + TEST <= len(frame):
        history = frame.iloc[start:start + HISTORY]
        fit_frame, valid = history.iloc[:-VALID], history.iloc[-VALID:]
        test = frame.iloc[start + HISTORY:start + HISTORY + TEST]
        tested += len(test)
        config = choose(fit_frame, valid)
        if config:
            (
                _, model, flip_cut, confidence_cut, confidence_quantile,
                rounds, learning_rate, validation_accuracy, validation_baseline,
                validation_cases, validation_model_only, validation_baseline_only,
            ) = config
            failure_score = score(model, test, FEATURES)
            use = test["tx_night_spread_per"].abs().to_numpy() >= confidence_cut
            flip = failure_score <= -flip_cut
            baseline = test["night_baseline"].to_numpy(int)
            prediction = baseline.copy()
            prediction[flip] *= -1
            actual = test["close_target"].to_numpy(int)
            for pos in np.flatnonzero(use):
                row = test.iloc[pos]
                records.append({
                    "date": str(row["date"].date()), "prediction": int(prediction[pos]),
                    "target": int(actual[pos]), "hit": int(prediction[pos] == actual[pos]),
                    "night_baseline": int(baseline[pos]),
                    "night_return_baseline": int(row["night_return_baseline"]),
                    "external_baseline": int(row["external_baseline"]),
                    "flipped": bool(flip[pos]), "failure_score": float(failure_score[pos]),
                    "night_confidence": float(abs(row["tx_night_spread_per"])),
                    "past_night_hit_20d": float(row["past_night_hit_20d"]) if pd.notna(row["past_night_hit_20d"]) else None,
                    "past_night_hit_60d": float(row["past_night_hit_60d"]) if pd.notna(row["past_night_hit_60d"]) else None,
                    "rounds": rounds, "learning_rate": learning_rate,
                    "flip_cut": flip_cut, "confidence_cut": confidence_cut,
                    "confidence_quantile": confidence_quantile,
                })
            cases = int(use.sum())
            windows.append({
                "test_start": str(test["date"].min().date()), "test_end": str(test["date"].max().date()),
                "cases": cases,
                "model_accuracy": float((prediction[use] == actual[use]).mean()) if cases else 0,
                "night_baseline_accuracy": float((baseline[use] == actual[use]).mean()) if cases else 0,
                "flips": int((flip & use).sum()), "rounds": rounds,
                "learning_rate": learning_rate, "flip_cut": flip_cut,
                "confidence_quantile": confidence_quantile, "confidence_cut": confidence_cut,
                "validation_accuracy": validation_accuracy,
                "validation_baseline_accuracy": validation_baseline,
                "validation_cases": validation_cases,
                "validation_model_only": validation_model_only,
                "validation_baseline_only": validation_baseline_only,
            })
        start += TEST

    cases = len(records)
    hits = sum(row["hit"] for row in records)
    accuracy = hits / cases if cases else 0
    coverage = cases / tested if tested else 0
    baselines = {}
    for name in ["night_baseline", "night_return_baseline", "external_baseline"]:
        baseline_hits = sum(row[name] == row["target"] for row in records)
        baselines[name] = {"hits": baseline_hits, "accuracy": baseline_hits / cases if cases else 0}
    strongest_name = max(baselines, key=lambda name: baselines[name]["accuracy"])
    strongest = baselines[strongest_name]
    model_only = sum(row["hit"] and row[strongest_name] != row["target"] for row in records)
    baseline_only = sum(not row["hit"] and row[strongest_name] == row["target"] for row in records)
    p_value = mcnemar(model_only, baseline_only)
    material = [window for window in windows if window["cases"] >= 10]
    minimum_block = min((window["model_accuracy"] for window in material), default=0)
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


def main():
    frame = prepare()
    result = evaluate(frame)
    payload = {
        "status": "exploratory strict OOS; target history previously inspected; not production",
        "hypothesis": "A causally observed reliability/volatility/overseas regime identifies when the night-spread direction should continue, reverse, or be abstained for the same-day TWII cash close.",
        "causal_rule": "All reliability and cash-regime features are shifted at least one completed cash day; overseas observations are strictly earlier than the Taiwan target date.",
        "method": "630-day balanced failure classifier, following 126-day nested calibration of flip and abstention rules, then frozen 126-day OOS blocks. Prediction normally retains night-spread sign and flips only in a learned failure state.",
        "gate": ">=90% accuracy, >=100 cases, >=10% coverage, Wilson lower >=80%, significant superiority over strongest identical-date simple baseline, every material block >=80%.",
        "history": HISTORY, "validation": VALID, "test": TEST, "features": FEATURES,
        "aligned_rows": len(frame), "result": result,
    }
    out = ROOT / "reports"
    (out / "night_cash_close_regime_switch.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Night-to-cash-close regime-switch research", "", payload["status"], "",
        payload["hypothesis"], "", payload["causal_rule"], "", payload["method"], "",
        payload["gate"], "",
        "| Cases | Accuracy | Coverage | Wilson lower | Strongest baseline | Model/base only | p | Min block | Passed |",
        "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
        f"| {result['cases']} | {result['accuracy']:.2%} | {result['coverage']:.2%} | "
        f"{result['wilson_95_lower']:.2%} | {result['strongest_baseline']} {result['strongest_baseline_accuracy']:.2%} | "
        f"{result['model_only']}/{result['baseline_only']} | {result['mcnemar_exact_p']:.4g} | "
        f"{result['minimum_material_window_accuracy']:.2%} | {result['passed']} |",
        "", "## Walk-forward windows", "",
        "| Test period | Cases | Model | Night baseline | Flips | Validation model/base |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for window in result["windows"]:
        lines.append(
            f"| {window['test_start']} to {window['test_end']} | {window['cases']} | "
            f"{window['model_accuracy']:.2%} | {window['night_baseline_accuracy']:.2%} | "
            f"{window['flips']} | {window['validation_accuracy']:.2%}/{window['validation_baseline_accuracy']:.2%} |"
        )
    lines += ["", "Every selected OOS prediction and frozen configuration is retained in the JSON companion file."]
    (out / "night_cash_close_regime_switch.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "aligned_rows": len(frame),
        "result": {key: value for key, value in result.items() if key not in ["records", "windows", "baselines"]},
    }, indent=2))


if __name__ == "__main__":
    main()
