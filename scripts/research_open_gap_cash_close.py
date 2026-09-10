"""At-open TWII opening-gap acceptance model for same-day close direction."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HISTORY, VALID, TEST = 756, 126, 126
QUANTILES = np.linspace(0, .9, 19)


def prepare():
    cash = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv", usecols=["date", "open", "close"])
    cash["date"] = pd.to_datetime(cash["date"])
    cash["open"] = pd.to_numeric(cash["open"], errors="coerce")
    cash["close"] = pd.to_numeric(cash["close"], errors="coerce")
    cash = cash.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    cash["prior_close"] = cash["close"].shift(1)
    cash["gap_return"] = cash["open"] / cash["prior_close"] - 1
    cash["cash_return"] = cash["close"] / cash["prior_close"] - 1
    cash["prior_cash_return"] = cash["close"].pct_change().shift(1)
    cash["target"] = np.where(cash["cash_return"] >= 0, 1, -1)

    night = pd.read_csv(ROOT / "data" / "processed" / "taiwan_futures_night.csv")
    night["date"] = pd.to_datetime(night["signal_date"])
    night["night_spread_prediction"] = np.where(pd.to_numeric(night["tx_night_spread_per"], errors="coerce") >= 0, 1, -1)
    night["night_return_prediction"] = np.where(pd.to_numeric(night["tx_night_return"], errors="coerce") >= 0, 1, -1)
    frame = cash.merge(night[["date", "night_spread_prediction", "night_return_prediction"]], on="date", how="inner")
    frame = frame.dropna(subset=["gap_return", "cash_return"]).reset_index(drop=True)
    frame["prediction"] = np.where(frame["gap_return"] >= 0, 1, -1)
    frame["confidence"] = frame["gap_return"].abs()
    return frame


def select(history):
    valid = history.iloc[-VALID:]
    best = None
    for quantile in QUANTILES:
        threshold = float(valid["confidence"].quantile(quantile))
        use = valid["confidence"] >= threshold
        cases = int(use.sum())
        if cases < 15 or cases / len(valid) < .10:
            continue
        accuracy = float((valid.loc[use, "prediction"] == valid.loc[use, "target"]).mean())
        rank = (accuracy, cases / len(valid), -quantile)
        if best is None or rank > best[0]:
            best = (rank, threshold, accuracy, cases, float(quantile))
    return best


def wilson(hits, cases):
    if not cases:
        return 0.0
    z = 1.959963984540054
    p = hits / cases
    den = 1 + z*z/cases
    return (p + z*z/(2*cases) - z*math.sqrt((p*(1-p)+z*z/(4*cases))/cases))/den


def mcnemar(a_only, b_only):
    n = a_only + b_only
    if not n:
        return 1.0
    low = min(a_only, b_only)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(low + 1)) / (2**n))


def main():
    frame = prepare()
    records, windows, tested = [], [], 0
    start = 0
    while start + HISTORY + TEST <= len(frame):
        history = frame.iloc[start : start + HISTORY]
        test = frame.iloc[start + HISTORY : start + HISTORY + TEST]
        tested += len(test)
        config = select(history)
        if config:
            _, threshold, va, vn, quantile = config
            use = test["confidence"] >= threshold
            for _, row in test.loc[use].iterrows():
                target, prediction = int(row["target"]), int(row["prediction"])
                records.append({
                    "date": str(row["date"].date()), "prediction": prediction,
                    "target": target, "hit": int(prediction == target),
                    "gap_return": float(row["gap_return"]), "cash_return": float(row["cash_return"]),
                    "threshold": threshold, "selected_quantile": quantile,
                    "validation_accuracy": va, "validation_cases": vn,
                    "baselines": {
                        "always_up": 1,
                        "night_spread": int(row["night_spread_prediction"]),
                        "night_return": int(row["night_return_prediction"]),
                        "prior_cash": 1 if row["prior_cash_return"] >= 0 else -1,
                    },
                })
            hits = int((test.loc[use, "prediction"] == test.loc[use, "target"]).sum())
            cases = int(use.sum())
            windows.append({
                "test_end": str(test["date"].max().date()), "threshold": threshold,
                "selected_quantile": quantile, "validation_accuracy": va, "validation_cases": vn,
                "test_cases": cases, "test_accuracy": hits / cases if cases else 0,
            })
        start += TEST

    cases, hits = len(records), sum(row["hit"] for row in records)
    accuracy, coverage = (hits / cases if cases else 0), (cases / tested if tested else 0)
    baselines = {}
    for name in ["always_up", "night_spread", "night_return", "prior_cash"]:
        bh = sum(row["baselines"][name] == row["target"] for row in records)
        baselines[name] = {"hits": bh, "accuracy": bh / cases if cases else 0}
    strongest_name = max(baselines, key=lambda name: baselines[name]["accuracy"])
    strongest = baselines[strongest_name]
    model_only = sum(row["hit"] and row["baselines"][strongest_name] != row["target"] for row in records)
    baseline_only = sum(not row["hit"] and row["baselines"][strongest_name] == row["target"] for row in records)
    p_value = mcnemar(model_only, baseline_only)
    lower = wilson(hits, cases)
    material_windows = [window for window in windows if window["test_cases"] >= 10]
    stable = bool(material_windows) and min(window["test_accuracy"] for window in material_windows) >= .80
    # Full-coverage gap sign is the same directional rule, evaluated without
    # the causally selected abstention threshold. It is reported as an ablation,
    # not as an identical-date competing prediction (which would be identical).
    oos_frame = frame.iloc[HISTORY:HISTORY + tested]
    raw_gap_accuracy = float((oos_frame["prediction"] == oos_frame["target"]).mean()) if len(oos_frame) else 0
    passed = (
        accuracy >= .90 and cases >= 100 and coverage >= .10 and lower >= .80
        and accuracy > strongest["accuracy"] and model_only > baseline_only and p_value < .05
        and stable and accuracy > raw_gap_accuracy
    )
    result = {
        "aligned_rows": len(frame), "tested_rows": tested, "cases": cases, "hits": hits,
        "accuracy": accuracy, "coverage": coverage, "wilson_95_lower": lower,
        "full_coverage_gap_sign_accuracy": raw_gap_accuracy,
        "baselines": baselines, "strongest_baseline": strongest_name,
        "strongest_baseline_accuracy": strongest["accuracy"], "model_only": model_only,
        "baseline_only": baseline_only, "mcnemar_exact_p": p_value,
        "minimum_material_window_accuracy": min((w["test_accuracy"] for w in material_windows), default=0),
        "stable_windows": stable, "passed": passed, "windows": windows, "records": records,
    }
    payload = {
        "scope": "At the TWII cash open, predict same-day close versus prior cash close.",
        "hypothesis": "A sufficiently large observed opening gap represents price acceptance and retains its sign through the cash close.",
        "method": "756-day rolling history; last 126 days choose only an absolute-gap abstention threshold; frozen following 126-day OOS block; compare identical selected dates with night spread/return, prior cash direction and always-up.",
        "gate": ">=90% accuracy, >=100 cases, >=10% coverage, Wilson lower >=80%, significant paired superiority over strongest alternative, improvement over unfiltered gap sign, and every material OOS block >=80%.",
        "result": result,
    }
    out = ROOT / "reports"
    (out / "open_gap_cash_close.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Opening-gap acceptance to cash-close direction", "", payload["scope"], "",
        payload["hypothesis"], "", payload["method"], "", payload["gate"], "",
        f"OOS: {hits}/{cases} = {accuracy:.2%}; coverage {coverage:.2%}; Wilson lower {lower:.2%}.",
        f"Full-coverage gap-sign ablation: {raw_gap_accuracy:.2%}.",
        f"Strongest selected-date alternative: {strongest_name} {strongest['accuracy']:.2%}.",
        f"Paired model-only/baseline-only: {model_only}/{baseline_only}; p={p_value:.4g}.",
        f"Minimum material block: {result['minimum_material_window_accuracy']:.2%}; passed={passed}.", "",
        "| Test end | Threshold | Cases | Accuracy | Validation cases/accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for window in windows:
        lines.append(
            f"| {window['test_end']} | {window['threshold']:.3%} | {window['test_cases']} | "
            f"{window['test_accuracy']:.2%} | {window['validation_cases']}/{window['validation_accuracy']:.2%} |"
        )
    (out / "open_gap_cash_close.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ["windows", "records"]}, indent=2))


if __name__ == "__main__":
    main()
