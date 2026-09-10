import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_causal_phase import add_phase_features, wilson_lower


TRAIN_DAYS = 1260
TEST_DAYS = 252
HORIZONS = [20, 60, 120]


def add_pulse(price):
    x = add_phase_features(price, fast=20, slow=120, bins=12)
    r = np.log(x["close"]).diff()
    vol20 = r.rolling(20).std()
    vol120 = r.rolling(120).std()
    x["amplitude_ratio"] = vol20 / vol120.replace(0, np.nan)
    x["pulse_rate"] = r.gt(0).ne(r.shift().gt(0)).rolling(20).mean()
    x["ret20"] = x["close"].pct_change(20)
    x["ret60"] = x["close"].pct_change(60)
    x["ret240"] = x["close"].pct_change(240)
    x["shock_z"] = (r.abs() / vol120.replace(0, np.nan)).rolling(20).max()
    # Every diagnostic threshold is based only on the preceding 5 years.
    amp_hi = x["amplitude_ratio"].rolling(1260, min_periods=500).quantile(.85).shift(1)
    amp_lo = x["amplitude_ratio"].rolling(1260, min_periods=500).quantile(.20).shift(1)
    rate_hi = x["pulse_rate"].rolling(1260, min_periods=500).quantile(.80).shift(1)
    shock_hi = x["shock_z"].rolling(1260, min_periods=500).quantile(.90).shift(1)
    state = pd.Series("NORMAL", index=x.index, dtype="object")
    state[(x["amplitude_ratio"] <= amp_lo) & (x["pulse_rate"] >= rate_hi)] = "STAGNATION"
    state[(x["pulse_rate"] >= rate_hi) & (x["amplitude_ratio"] > 1.0)] = "ARRHYTHMIA"
    state[(x["ret240"] > 0) & (x["ret60"] > 0) & (x["ret20"] < 0)] = "EXHAUSTION"
    state[(x["ret240"] < 0) & (x["ret20"] < 0) & (x["amplitude_ratio"] >= amp_hi)] = "PANIC"
    state[(x["amplitude_ratio"] >= amp_hi) | (x["shock_z"] >= shock_hi)] = "EXTERNAL_SHOCK"
    x["pulse_state"] = state
    x["diagnostic_state"] = x["state"] + "|" + x["pulse_state"]
    return x


def build_map(train, column, min_count=30, confidence=.58):
    mapping = {}
    for state, group in train.dropna(subset=["target"]).groupby(column):
        if len(group) < min_count:
            continue
        counts = group["target"].value_counts()
        purity = float(counts.max() / len(group))
        if purity >= confidence:
            mapping[state] = int(counts.idxmax())
    return mapping


def walk_forward(price, horizon):
    x = add_pulse(price)
    future = x["close"].shift(-horizon) / x["close"] - 1
    x["target"] = np.where(future > 0, 1, -1).astype(float)
    x.loc[future.isna(), "target"] = np.nan
    records = []
    start = 0
    while start + TRAIN_DAYS + TEST_DAYS + horizon <= len(x):
        train = x.iloc[start:start + TRAIN_DAYS]
        test = x.iloc[start + TRAIN_DAYS:start + TRAIN_DAYS + TEST_DAYS].copy()
        mapping = build_map(train, "diagnostic_state")
        test["prediction"] = test["diagnostic_state"].map(mapping)
        usable = test["prediction"].notna() & test["target"].notna()
        for row in test.loc[usable].itertuples():
            records.append({
                "date": str(row.date.date()), "pulse_state": row.pulse_state,
                "prediction": int(row.prediction), "target": int(row.target),
                "hit": int(row.prediction == row.target),
            })
        start += TEST_DAYS
    frame = pd.DataFrame(records)
    total_rows = (start // TEST_DAYS) * TEST_DAYS
    overall = metrics(frame, total_rows)
    by_pulse = {state: metrics(group, len(group)) for state, group in frame.groupby("pulse_state")} if not frame.empty else {}
    normal = frame[frame["pulse_state"] == "NORMAL"] if not frame.empty else frame
    normal_metrics = metrics(normal, total_rows)
    majority_actual = x.iloc[TRAIN_DAYS:TRAIN_DAYS + total_rows]["target"].dropna()
    up = float((majority_actual == 1).mean()) if len(majority_actual) else 0.5
    baseline = max(up, 1 - up)
    normal_metrics["baseline"] = baseline
    normal_metrics["edge"] = normal_metrics["accuracy"] - baseline
    normal_metrics["passed"] = (
        normal_metrics["accuracy"] >= .90 and normal_metrics["cases"] >= 100
        and normal_metrics["coverage"] >= .10 and normal_metrics["wilson_95_lower"] >= .80
        and normal_metrics["accuracy"] > baseline
    )
    return {"horizon_days": horizon, "overall": overall, "normal_only": normal_metrics, "by_pulse": by_pulse}


def metrics(frame, denominator):
    cases = len(frame)
    hits = int(frame["hit"].sum()) if cases else 0
    return {
        "cases": cases, "hits": hits, "accuracy": hits / cases if cases else 0.0,
        "coverage": cases / denominator if denominator else 0.0,
        "wilson_95_lower": wilson_lower(hits, cases),
    }


def main():
    price = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv", usecols=["date", "close"])
    price["date"] = pd.to_datetime(price["date"])
    results = [walk_forward(price, h) for h in HORIZONS]
    payload = {
        "hypothesis": "Market pulse diagnoses causal rhythm disorders; macro-cycle phase is trusted only under a normal pulse.",
        "pulse_states": ["NORMAL", "STAGNATION", "ARRHYTHMIA", "EXHAUSTION", "PANIC", "EXTERNAL_SHOCK"],
        "method": "Fixed causal diagnostics and yearly walk-forward state mapping; all rolling thresholds lagged one day.",
        "results": results,
        "passing": [r["horizon_days"] for r in results if r["normal_only"]["passed"]],
    }
    out = ROOT / "reports"
    (out / "market_pulse_research.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Market pulse research", "", payload["method"], "", "| Horizon | Scope | Cases | Accuracy | Coverage | Baseline | Edge | Wilson lower | Passed |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results:
        for scope, key in [("all diagnosed", "overall"), ("normal pulse only", "normal_only")]:
            m = r[key]
            lines.append(f"| {r['horizon_days']} | {scope} | {m['cases']} | {m['accuracy']:.2%} | {m['coverage']:.2%} | {m.get('baseline', 0):.2%} | {m.get('edge', 0):.2%} | {m['wilson_95_lower']:.2%} | {m.get('passed', False)} |")
    lines.extend(["", "## Accuracy by diagnosed pulse"])
    for r in results:
        lines.append(f"\n### {r['horizon_days']} days")
        for state, m in r["by_pulse"].items():
            lines.append(f"- {state}: {m['accuracy']:.2%} ({m['hits']}/{m['cases']})")
    (out / "market_pulse_research.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
