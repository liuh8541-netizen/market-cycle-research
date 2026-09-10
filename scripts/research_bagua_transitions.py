import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from research_bagua_cycle import build_multiframe_states
from research_selective_signals import wilson_lower


HORIZONS = [20, 60, 120]


def main():
    price = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
    price["date"] = pd.to_datetime(price["date"])
    data = build_multiframe_states(price)
    results = []
    for timeframe in ["daily_state", "weekly_state", "monthly_state"]:
        transitions = data[data[timeframe].ne(data[timeframe].shift())].copy()
        for horizon in HORIZONS:
            results.extend(evaluate_hypotheses(data, transitions, horizon, timeframe))
    payload = {
        "method": "月線卦位轉換日事件驗證",
        "results": results,
        "passing": [item for item in results if item["accuracy"] >= 0.90 and item["cases"] >= 100 and item["wilson_95_lower"] >= 0.80],
    }
    (ROOT / "reports" / "bagua_transition_research.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "reports" / "bagua_transition_research.md").write_text(render(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def evaluate_hypotheses(data, transitions, horizon, timeframe):
    close = data["close"].reset_index(drop=True)
    future_final = close.shift(-horizon) / close - 1
    future_max = pd.concat([close.shift(-step) / close - 1 for step in range(1, horizon + 1)], axis=1).max(axis=1)
    future_min = pd.concat([close.shift(-step) / close - 1 for step in range(1, horizon + 1)], axis=1).min(axis=1)
    aligned = data[["date", timeframe]].copy()
    aligned["future_final"] = future_final
    aligned["future_max"] = future_max
    aligned["future_min"] = future_min
    sample = aligned.loc[transitions.index].dropna(subset=["future_final"])
    definitions = [
        ("坎艮後反彈", ["KAN", "GEN"], sample["future_max"] >= threshold_for(horizon)),
        ("巽離後延續", ["XUN", "LI"], sample["future_final"] > 0),
        ("兌乾後回撤", ["DUI", "QIAN"], sample["future_min"] <= -threshold_for(horizon)),
    ]
    output = []
    for name, states, outcome in definitions:
        mask = sample[timeframe].isin(states)
        cases = int(mask.sum())
        hits = int(outcome[mask].sum())
        output.append({
            "hypothesis": name,
            "timeframe": timeframe.replace("_state", ""),
            "horizon_days": horizon,
            "cases": cases,
            "hits": hits,
            "accuracy": hits / cases if cases else 0.0,
            "wilson_95_lower": wilson_lower(hits, cases),
        })
    return output


def threshold_for(horizon):
    return {20: 0.04, 60: 0.07, 120: 0.10}[horizon]


def render(payload):
    lines = ["# 八卦轉折事件研究", "", "| 尺度 | 假設 | 週期 | 案例 | 命中率 | 95%下限 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for item in payload["results"]:
        lines.append(f"| {item['timeframe']} | {item['hypothesis']} | {item['horizon_days']}日 | {item['cases']} | {item['accuracy']:.2%} | {item['wilson_95_lower']:.2%} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
