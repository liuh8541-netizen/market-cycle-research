import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DAYS = 1260
VALID_DAYS = 252
TEST_DAYS = 252
HORIZONS = [20, 60, 120]
FAST_WINDOWS = [10, 20, 40]
SLOW_WINDOWS = [60, 120, 240]
PHASE_BINS = [8, 12, 16]
MIN_COUNTS = [20, 40]
CONFIDENCES = [0.60, 0.70, 0.80, 0.90]


def wilson_lower(hits, cases, z=1.959963984540054):
    if not cases:
        return 0.0
    p = hits / cases
    den = 1 + z * z / cases
    center = p + z * z / (2 * cases)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * cases)) / cases)
    return (center - margin) / den


def add_phase_features(price, fast, slow, bins):
    x = price.copy()
    logp = np.log(x["close"])
    fast_line = logp.ewm(span=fast, adjust=False).mean()
    slow_line = logp.ewm(span=slow, adjust=False).mean()
    oscillator = fast_line - slow_line
    velocity = oscillator.diff(5) / 5
    # Both scales use trailing observations only.
    osc_scale = oscillator.rolling(slow, min_periods=slow // 2).std().replace(0, np.nan)
    vel_scale = velocity.rolling(slow, min_periods=slow // 2).std().replace(0, np.nan)
    phase = np.arctan2(velocity / vel_scale, oscillator / osc_scale)
    x["phase_bin"] = np.floor((phase + np.pi) / (2 * np.pi) * bins).clip(0, bins - 1)
    x["macro"] = np.where(logp > slow_line, 1, -1)
    x["state"] = x["phase_bin"].astype("Int64").astype(str) + "|" + pd.Series(x["macro"], index=x.index).astype(str)
    return x


def direction_map(train, min_count, confidence):
    result = {}
    for state, group in train.dropna(subset=["target"]).groupby("state"):
        if len(group) < min_count:
            continue
        counts = group["target"].value_counts()
        direction = int(counts.idxmax())
        purity = float(counts.max() / len(group))
        if purity >= confidence:
            result[state] = direction
    return result


def score(train, sample, min_count, confidence):
    mapping = direction_map(train, min_count, confidence)
    predicted = sample["state"].map(mapping)
    actionable = predicted.notna() & sample["target"].notna()
    cases = int(actionable.sum())
    hits = int((predicted[actionable].astype(int) == sample.loc[actionable, "target"].astype(int)).sum())
    return hits, cases, int(len(sample))


def select_config(history, horizon):
    fit = history.iloc[: -VALID_DAYS]
    valid = history.iloc[-VALID_DAYS:]
    candidates = []
    for fast in FAST_WINDOWS:
        for slow in SLOW_WINDOWS:
            if fast >= slow:
                continue
            for bins in PHASE_BINS:
                featured = add_phase_features(history, fast, slow, bins)
                train_part = featured.iloc[: -VALID_DAYS]
                valid_part = featured.iloc[-VALID_DAYS:]
                for min_count in MIN_COUNTS:
                    for confidence in CONFIDENCES:
                        hits, cases, rows = score(train_part, valid_part, min_count, confidence)
                        coverage = cases / rows if rows else 0
                        accuracy = hits / cases if cases else 0
                        # Selection demands useful coverage; accuracy is primary, cases break ties.
                        if coverage >= 0.10:
                            candidates.append((accuracy, cases, fast, slow, bins, min_count, confidence))
    if not candidates:
        return None
    selected = max(candidates, key=lambda z: (z[0], z[1]))
    return {
        "validation_accuracy": selected[0], "validation_cases": selected[1],
        "fast": selected[2], "slow": selected[3], "bins": selected[4],
        "min_count": selected[5], "confidence": selected[6],
    }


def walk_forward(price, horizon):
    work = price.copy()
    future = work["close"].shift(-horizon) / work["close"] - 1
    work["target"] = np.where(future > 0, 1, -1).astype(float)
    work.loc[future.isna(), "target"] = np.nan
    all_hits = all_cases = all_rows = 0
    windows = []
    start = 0
    minimum = TRAIN_DAYS + VALID_DAYS
    while start + minimum + TEST_DAYS + horizon <= len(work):
        history = work.iloc[start:start + minimum].copy()
        test = work.iloc[start + minimum:start + minimum + TEST_DAYS].copy()
        config = select_config(history, horizon)
        if config is None:
            windows.append({"test_end": str(test["date"].max().date()), "cases": 0, "accuracy": 0.0, "config": None})
            all_rows += len(test)
            start += TEST_DAYS
            continue
        combined = pd.concat([history, test], ignore_index=True)
        featured = add_phase_features(combined, config["fast"], config["slow"], config["bins"])
        train_featured = featured.iloc[:len(history)]
        test_featured = featured.iloc[len(history):]
        hits, cases, rows = score(train_featured, test_featured, config["min_count"], config["confidence"])
        all_hits += hits
        all_cases += cases
        all_rows += rows
        windows.append({
            "test_end": str(test["date"].max().date()), "cases": cases,
            "accuracy": hits / cases if cases else 0.0, "config": config,
        })
        start += TEST_DAYS
    accuracy = all_hits / all_cases if all_cases else 0
    coverage = all_cases / all_rows if all_rows else 0
    majority = majority_baseline(work, horizon, start=minimum, end=minimum + all_rows)
    lower = wilson_lower(all_hits, all_cases)
    return {
        "horizon_days": horizon, "hits": all_hits, "cases": all_cases,
        "accuracy": accuracy, "coverage": coverage, "wilson_95_lower": lower,
        "majority_baseline": majority, "edge": accuracy - majority,
        "passed": accuracy >= .90 and all_cases >= 100 and coverage >= .10 and lower >= .80 and accuracy > majority,
        "windows": windows,
    }


def majority_baseline(work, horizon, start, end):
    actual = work.iloc[start:end]["target"].dropna()
    if actual.empty:
        return 0.0
    up = float((actual == 1).mean())
    return max(up, 1 - up)


def main():
    price = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv", usecols=["date", "close"])
    price["date"] = pd.to_datetime(price["date"])
    results = [walk_forward(price, horizon) for horizon in HORIZONS]
    payload = {
        "hypothesis": "A causal oscillator phase captures the recurring macro cycle while consolidation is a temporary low-velocity phase.",
        "method": "Nested yearly walk-forward; each year's phase parameters and state rules are selected using earlier fit/validation data only.",
        "results": results,
        "passing": [r["horizon_days"] for r in results if r["passed"]],
    }
    out = ROOT / "reports"
    (out / "causal_phase_research.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Causal phase research", "", payload["method"], "", "| Horizon | Cases | Accuracy | Coverage | Baseline | Edge | Wilson lower | Passed |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results:
        lines.append(f"| {r['horizon_days']} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['majority_baseline']:.2%} | {r['edge']:.2%} | {r['wilson_95_lower']:.2%} | {r['passed']} |")
    (out / "causal_phase_research.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "results"} | {"results": [{k: v for k, v in r.items() if k != "windows"} for r in results]}, indent=2))


if __name__ == "__main__":
    main()
