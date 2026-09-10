import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predict_market import classify_bagua_row, prepare_bagua_features, resample_market_frame
from research_selective_signals import TARGET_ACCURACY, TARGET_COVERAGE, MIN_TOTAL_TEST_CASES, wilson_lower


HORIZONS = [20, 60, 120]
TRAIN_DAYS = 1260
TEST_DAYS = 252
MIN_GROUP = 30


def main():
    price = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
    price["date"] = pd.to_datetime(price["date"])
    states = build_multiframe_states(price)
    results = [walk_forward(states, horizon) for horizon in HORIZONS]
    payload = {
        "hypothesis": "八卦量化狀態與多週期共振可預測大方向",
        "method": "expanding walk-forward; training-only state direction mapping",
        "results": results,
        "passing_horizons": [item["horizon_days"] for item in results if item["passed"]],
    }
    (ROOT / "reports" / "bagua_cycle_research.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "reports" / "bagua_cycle_research.md").write_text(render(payload), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


def build_multiframe_states(price):
    daily = state_frame(price, 120, 20, 60, "daily_state")
    weekly_raw = resample_market_frame(price, "W-FRI")
    weekly = state_frame(weekly_raw, 104, 8, 26, "weekly_state")[["date", "weekly_state"]]
    monthly_raw = resample_market_frame(price, "ME")
    monthly = state_frame(monthly_raw, 60, 3, 12, "monthly_state")[["date", "monthly_state"]]
    output = pd.merge_asof(daily.sort_values("date"), weekly.sort_values("date"), on="date", direction="backward")
    output = pd.merge_asof(output.sort_values("date"), monthly.sort_values("date"), on="date", direction="backward")
    output["state_combo"] = output[["monthly_state", "weekly_state", "daily_state"]].fillna("NA").agg("|".join, axis=1)
    output["weekly_daily"] = output[["weekly_state", "daily_state"]].fillna("NA").agg("|".join, axis=1)
    return output.dropna(subset=["monthly_state", "weekly_state"]).reset_index(drop=True)


def state_frame(frame, window, short_window, mid_window, name):
    prepared = prepare_bagua_features(frame, window, short_window, mid_window).copy()
    prepared[name] = prepared.apply(classify_bagua_row, axis=1)
    return prepared


def walk_forward(data, horizon):
    work = data.copy()
    future = work["close"].shift(-horizon) / work["close"] - 1
    neutral = {20: 0.02, 60: 0.04, 120: 0.06}[horizon]
    work["target"] = np.where(future > neutral, 1, np.where(future < -neutral, -1, 0))
    hits = []
    total_rows = 0
    windows = []
    start = 0
    while start + TRAIN_DAYS + TEST_DAYS + horizon <= len(work):
        train = work.iloc[start:start + TRAIN_DAYS]
        test = work.iloc[start + TRAIN_DAYS:start + TRAIN_DAYS + TEST_DAYS]
        predictions = predict_from_state_map(train, test)
        actionable = predictions != 0
        actual = test["target"].to_numpy()
        window_hits = predictions[actionable] == actual[actionable]
        hits.extend(window_hits.astype(int).tolist())
        total_rows += len(test)
        windows.append({
            "test_end": str(test["date"].max().date()),
            "cases": int(actionable.sum()),
            "accuracy": float(window_hits.mean()) if len(window_hits) else 0.0,
        })
        start += TEST_DAYS
    cases = len(hits)
    hit_count = sum(hits)
    accuracy = hit_count / cases if cases else 0.0
    coverage = cases / total_rows if total_rows else 0.0
    lower = wilson_lower(hit_count, cases)
    passed = accuracy >= TARGET_ACCURACY and cases >= MIN_TOTAL_TEST_CASES and coverage >= TARGET_COVERAGE and lower >= 0.80
    return {"horizon_days": horizon, "cases": cases, "accuracy": accuracy, "coverage": coverage, "wilson_95_lower": lower, "passed": passed, "windows": windows}


def predict_from_state_map(train, test):
    predictions = np.zeros(len(test), dtype=int)
    levels = ["state_combo", "weekly_daily", "monthly_state", "weekly_state", "daily_state"]
    maps = {level: build_direction_map(train, level) for level in levels}
    for position, (_, row) in enumerate(test.iterrows()):
        for level in levels:
            signal = maps[level].get(row[level])
            if signal:
                predictions[position] = signal
                break
    return predictions


def build_direction_map(train, column):
    output = {}
    for value, group in train.groupby(column):
        counts = group["target"].value_counts()
        count = len(group)
        if count < MIN_GROUP:
            continue
        direction = int(counts.idxmax())
        accuracy = counts.max() / count
        if direction != 0 and accuracy >= 0.60:
            output[value] = direction
    return output


def render(payload):
    lines = ["# 八卦大方向週期研究", "", f"- 假設：{payload['hypothesis']}", f"- 達標週期：{payload['passing_horizons'] or '無'}", "", "| 週期 | 案例 | 命中率 | 覆蓋率 | 95%下限 | 達標 |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for item in payload["results"]:
        lines.append(f"| {item['horizon_days']}日 | {item['cases']} | {item['accuracy']:.2%} | {item['coverage']:.2%} | {item['wilson_95_lower']:.2%} | {'是' if item['passed'] else '否'} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
