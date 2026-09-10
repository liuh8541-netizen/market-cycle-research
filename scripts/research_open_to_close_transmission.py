"""Strict OOS test of the night -> cash open -> cash close transmission chain."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HISTORY = 756
VALID = 126
TEST = 126
LAMBDAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
QUANTILES = [0.0, 0.25, 0.5, 0.65, 0.75, 0.85, 0.9]
BASE_FEATURES = [
    "gap_return", "night_return", "night_spread", "night_range", "night_volume_z",
    "prior_cash_1d", "prior_cash_5d", "prior_intraday", "prior_gap",
    "prior_volatility20", "external_breadth", "prior_vix_change",
]
FEATURES = BASE_FEATURES + [
    "gap_x_night", "gap_x_prior_trend", "night_x_external", "gap_abs",
]


def causal_z(series, window=252, minimum=80):
    mean = series.rolling(window, min_periods=minimum).mean().shift(1)
    std = series.rolling(window, min_periods=minimum).std().shift(1).replace(0, np.nan)
    return (series - mean) / std


def prepare():
    cash = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
    cash["date"] = pd.to_datetime(cash["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        cash[col] = pd.to_numeric(cash[col], errors="coerce")
    cash = cash.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    cash["gap_return"] = cash["open"] / cash["close"].shift(1) - 1
    cash["intraday_return"] = cash["close"] / cash["open"] - 1
    cash["prior_cash_1d"] = cash["close"].pct_change().shift(1)
    cash["prior_cash_5d"] = cash["close"].pct_change(5).shift(1)
    cash["prior_intraday"] = cash["intraday_return"].shift(1)
    cash["prior_gap"] = cash["gap_return"].shift(1)
    cash["prior_volatility20"] = np.log(cash["close"]).diff().rolling(20).std().shift(1)

    night = pd.read_csv(ROOT / "data" / "processed" / "taiwan_futures_night.csv")
    night["date"] = pd.to_datetime(night["signal_date"])
    night = night.rename(columns={
        "tx_night_return": "night_return", "tx_night_spread_per": "night_spread",
        "tx_night_range": "night_range", "tx_night_volume": "night_volume",
    })
    night["night_volume_z"] = causal_z(pd.to_numeric(night["night_volume"], errors="coerce"))

    external = pd.read_csv(ROOT / "data" / "processed" / "external_markets.csv")
    external["date"] = pd.to_datetime(external["date"])
    closes = []
    for col in ["sp500_close", "nasdaq_close", "sox_close", "dow_close", "tsm_adr_close", "micron_close"]:
        value = pd.to_numeric(external[col], errors="coerce")
        ret = value.pct_change()
        closes.append(np.sign(ret).replace(0, np.nan))
    # shift(1): only an overseas close strictly before the Taiwan signal date.
    external["external_breadth"] = pd.concat(closes, axis=1).mean(axis=1).shift(1)
    external["prior_vix_change"] = pd.to_numeric(external["vix_close"], errors="coerce").pct_change().shift(1)
    external = external[["date", "external_breadth", "prior_vix_change"]]

    frame = cash.merge(night[["date", "night_return", "night_spread", "night_range", "night_volume_z"]], on="date", how="inner")
    frame = pd.merge_asof(
        frame.sort_values("date"), external.sort_values("date"), on="date",
        direction="backward", tolerance=pd.Timedelta("5D"),
    )
    frame["gap_x_night"] = frame["gap_return"] * frame["night_spread"]
    frame["gap_x_prior_trend"] = frame["gap_return"] * frame["prior_cash_5d"]
    frame["night_x_external"] = frame["night_spread"] * frame["external_breadth"]
    frame["gap_abs"] = frame["gap_return"].abs()
    frame["target"] = np.where(frame["intraday_return"] >= 0, 1, -1)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["intraday_return", "gap_return", "night_spread"])
    return frame.reset_index(drop=True)


def fit(frame, lam):
    raw = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    med = raw.median().fillna(0)
    raw = raw.fillna(med)
    mean = raw.mean()
    std = raw.std().replace(0, 1).fillna(1)
    x = ((raw - mean) / std).to_numpy(float)
    x = np.column_stack([np.ones(len(x)), x])
    y = frame["target"].to_numpy(float)
    penalty = np.eye(x.shape[1]) * lam
    penalty[0, 0] = 0
    weights = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y
    return weights, med, mean, std


def predict(model, frame):
    weights, med, mean, std = model
    raw = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0)
    x = ((raw - mean) / std).to_numpy(float)
    score = np.column_stack([np.ones(len(x)), x]) @ weights
    return np.where(score >= 0, 1, -1), np.abs(score)


def select(history):
    train = history.iloc[: -VALID]
    valid = history.iloc[-VALID:]
    best = None
    for lam in LAMBDAS:
        model = fit(train, lam)
        pred, confidence = predict(model, valid)
        target = valid["target"].to_numpy(int)
        for quantile in QUANTILES:
            threshold = float(np.quantile(confidence, quantile))
            use = confidence >= threshold
            cases = int(use.sum())
            if cases < 15 or cases / len(valid) < 0.10:
                continue
            accuracy = float((pred[use] == target[use]).mean())
            rank = (accuracy, cases / len(valid), -lam, -quantile)
            if best is None or rank > best[0]:
                best = (rank, lam, threshold, accuracy, cases, quantile)
    return best


def exact_mcnemar(a_only, b_only):
    n = a_only + b_only
    if n == 0:
        return 1.0
    low = min(a_only, b_only)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(low + 1)) / (2**n))


def wilson(hits, cases):
    if not cases:
        return 0.0
    z = 1.959963984540054
    p = hits / cases
    den = 1 + z * z / cases
    return (p + z * z / (2 * cases) - z * math.sqrt((p * (1 - p) + z * z / (4 * cases)) / cases)) / den


def main():
    frame = prepare()
    records, windows, tested_rows = [], [], 0
    start = 0
    while start + HISTORY + TEST <= len(frame):
        history = frame.iloc[start : start + HISTORY]
        test = frame.iloc[start + HISTORY : start + HISTORY + TEST]
        config = select(history)
        tested_rows += len(test)
        if config:
            _, lam, threshold, valid_accuracy, valid_cases, quantile = config
            model = fit(history, lam)
            pred, confidence = predict(model, test)
            use = confidence >= threshold
            for offset in np.flatnonzero(use):
                row = test.iloc[offset]
                gap_sign = 1 if row["gap_return"] >= 0 else -1
                records.append({
                    "date": str(row["date"].date()), "prediction": int(pred[offset]),
                    "target": int(row["target"]), "hit": int(pred[offset] == row["target"]),
                    "confidence": float(confidence[offset]), "lambda": lam,
                    "threshold": threshold, "selected_quantile": quantile,
                    "validation_accuracy": valid_accuracy, "validation_cases": valid_cases,
                    "gap_return": float(row["gap_return"]), "night_spread": float(row["night_spread"]),
                    "baselines": {
                        "always_up": 1,
                        "gap_continuation": gap_sign,
                        "gap_reversion": -gap_sign,
                        "night_spread_sign": 1 if row["night_spread"] >= 0 else -1,
                        "prior_cash_trend": 1 if row["prior_cash_5d"] >= 0 else -1,
                    },
                })
            valid = use
            windows.append({
                "test_end": str(test["date"].max().date()), "lambda": lam,
                "threshold": threshold, "selected_quantile": quantile,
                "validation_accuracy": valid_accuracy, "validation_cases": valid_cases,
                "test_cases": int(valid.sum()),
                "test_accuracy": float((pred[valid] == test.loc[valid, "target"]).mean()) if valid.any() else 0,
            })
        start += TEST

    cases = len(records)
    hits = sum(row["hit"] for row in records)
    accuracy = hits / cases if cases else 0
    coverage = cases / tested_rows if tested_rows else 0
    baseline_results = {}
    for name in ["always_up", "gap_continuation", "gap_reversion", "night_spread_sign", "prior_cash_trend"]:
        bh = sum(row["baselines"][name] == row["target"] for row in records)
        baseline_results[name] = {"hits": bh, "accuracy": bh / cases if cases else 0}
    strongest_name = max(baseline_results, key=lambda key: baseline_results[key]["accuracy"])
    strongest = baseline_results[strongest_name]
    model_only = sum(row["hit"] and row["baselines"][strongest_name] != row["target"] for row in records)
    baseline_only = sum(not row["hit"] and row["baselines"][strongest_name] == row["target"] for row in records)
    p_value = exact_mcnemar(model_only, baseline_only)
    lower = wilson(hits, cases)
    passed = (
        accuracy >= 0.90 and cases >= 100 and coverage >= 0.10 and lower >= 0.80
        and accuracy > strongest["accuracy"] and model_only > baseline_only and p_value < 0.05
        and all(window["test_accuracy"] >= 0.80 for window in windows if window["test_cases"] >= 10)
    )
    result = {
        "aligned_rows": len(frame), "tested_rows": tested_rows, "cases": cases, "hits": hits,
        "accuracy": accuracy, "coverage": coverage, "wilson_95_lower": lower,
        "baselines": baseline_results, "strongest_baseline": strongest_name,
        "strongest_baseline_accuracy": strongest["accuracy"], "model_only": model_only,
        "baseline_only": baseline_only, "mcnemar_exact_p": p_value,
        "all_material_windows_at_least_80pct": all(window["test_accuracy"] >= 0.80 for window in windows if window["test_cases"] >= 10),
        "passed": passed, "windows": windows, "records": records,
    }
    payload = {
        "scope": "Prediction made at the TWII cash open: same-day close versus same-day open.",
        "hypothesis": "Night displacement and the observed opening gap interact with lagged constitution and complexion to determine intraday continuation versus gap absorption.",
        "method": "756-day rolling history; last 126 days nested validation; frozen following 126-day OOS block; causal inputs only; identical-date simple baseline and exact paired audit.",
        "gate": ">=90% accuracy, >=100 cases, >=10% coverage, Wilson lower >=80%, significant superiority over strongest identical-date baseline, and every material OOS block >=80%.",
        "features": FEATURES, "result": result,
    }
    out = ROOT / "reports"
    (out / "open_to_close_transmission.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Night-to-open-to-close transmission research", "", payload["scope"], "",
        payload["hypothesis"], "", payload["method"], "", payload["gate"], "",
        f"OOS: {hits}/{cases} = {accuracy:.2%}; coverage {coverage:.2%}; Wilson lower {lower:.2%}.",
        f"Strongest identical-date baseline: {strongest_name} {strongest['accuracy']:.2%}.",
        f"Paired model-only/baseline-only: {model_only}/{baseline_only}; exact McNemar p={p_value:.4g}.",
        f"All material blocks >=80%: {result['all_material_windows_at_least_80pct']}; passed={passed}.", "",
        "| Test end | Cases | Accuracy | Validation cases/accuracy | Lambda | Threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for window in windows:
        lines.append(
            f"| {window['test_end']} | {window['test_cases']} | {window['test_accuracy']:.2%} | "
            f"{window['validation_cases']}/{window['validation_accuracy']:.2%} | {window['lambda']} | {window['threshold']:.6g} |"
        )
    (out / "open_to_close_transmission.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ["records", "windows"]}, indent=2))


if __name__ == "__main__":
    main()
