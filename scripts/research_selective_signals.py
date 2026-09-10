import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_lifecycle.external_market import add_external_market_features
from market_lifecycle.factor_features import add_factor_features
from market_lifecycle.features import add_features


HORIZONS = [1, 5, 20, 60]
THRESHOLDS = [0.5, 1.0, 1.5, 2.0]
TRAIN_DAYS = 756
TEST_DAYS = 126
MIN_TRAIN_CASES = 30
MIN_TOTAL_TEST_CASES = 100
TARGET_ACCURACY = 0.90
TARGET_COVERAGE = 0.10


def main():
    data = build_research_frame()
    results = [walk_forward_horizon(data, horizon) for horizon in HORIZONS]
    ensemble_results = [walk_forward_ensemble(data, horizon) for horizon in HORIZONS]
    payload = {
        "method": "training-selected conjunctive signals with abstention",
        "requirements": {
            "target_accuracy": TARGET_ACCURACY,
            "min_test_cases": MIN_TOTAL_TEST_CASES,
            "min_coverage": TARGET_COVERAGE,
        },
        "data_start": str(data["date"].min().date()),
        "data_end": str(data["date"].max().date()),
        "rows": int(len(data)),
        "results": results,
        "ensemble_results": ensemble_results,
        "passing_horizons": [item["horizon_days"] for item in results + ensemble_results if item["passed"]],
    }
    out_json = ROOT / "reports" / "selective_signal_research.json"
    out_md = ROOT / "reports" / "selective_signal_research.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"passing_horizons": payload["passing_horizons"], "results": results}, ensure_ascii=False, indent=2))


def build_research_frame():
    price = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
    price = add_external_market_features(
        price,
        ROOT / "data" / "processed" / "external_markets.csv",
        ROOT / "data" / "processed" / "taiwan_futures_night.csv",
    )
    data = add_factor_features(add_features(price), str(ROOT / "data" / "processed" / "factors"))
    data["date"] = pd.to_datetime(data["date"])
    feature_candidates = [
        "return_5d", "return_20d", "rsi_14", "volatility_20", "drawdown",
        "inst_net_5d", "inst_net_20d", "margin_change_20d", "short_change_20d",
        "futures_inst_net_5d", "futures_inst_net_20d", "futures_oi_change_20d",
        "option_put_call_proxy", "option_oi_put_call", "option_inst_net_5d",
        "option_inst_net_20d", "option_vix", "option_vix_20d_change",
        "nasdaq_return_1d", "sox_return_1d", "tsm_adr_return_1d", "vix_return_1d",
        "tx_night_return", "tx_night_gap_vs_spot",
    ]
    features = [col for col in feature_candidates if col in data and data[col].notna().sum() >= 300]
    for col in features:
        values = pd.to_numeric(data[col], errors="coerce")
        mean = values.rolling(252, min_periods=60).mean()
        std = values.rolling(252, min_periods=60).std().replace(0, np.nan)
        data["z_" + col] = (values - mean) / std
    data.attrs["features"] = ["z_" + col for col in features]
    return data.sort_values("date").reset_index(drop=True)


def walk_forward_horizon(data, horizon):
    work = data.copy()
    future_return = work["close"].shift(-horizon) / work["close"] - 1
    neutral = 0.005 if horizon <= 5 else 0.01
    work["target"] = np.where(future_return > neutral, 1, np.where(future_return < -neutral, -1, 0))
    atoms = make_atoms(data.attrs["features"])
    rows = []
    windows = []
    start = 0
    while start + TRAIN_DAYS + TEST_DAYS + horizon <= len(work):
        train = work.iloc[start : start + TRAIN_DAYS]
        test = work.iloc[start + TRAIN_DAYS : start + TRAIN_DAYS + TEST_DAYS]
        selected = select_rule_fast(train, atoms)
        mask = apply_rule(test, selected["rule"])
        actionable = test[mask]
        hits = int((actionable["target"] == selected["direction"]).sum())
        count = int(len(actionable))
        for _, row in actionable.iterrows():
            rows.append({"date": str(row["date"].date()), "hit": int(row["target"] == selected["direction"])})
        windows.append({
            "train_end": str(train["date"].max().date()),
            "test_end": str(test["date"].max().date()),
            "rule": selected["label"],
            "direction": "up" if selected["direction"] == 1 else "down",
            "train_accuracy": selected["accuracy"],
            "test_cases": count,
            "test_accuracy": hits / count if count else 0.0,
        })
        start += TEST_DAYS
    total = len(rows)
    hits = sum(row["hit"] for row in rows)
    eligible_test_rows = len(windows) * TEST_DAYS
    accuracy = hits / total if total else 0.0
    coverage = total / eligible_test_rows if eligible_test_rows else 0.0
    lower = wilson_lower(hits, total)
    passed = accuracy >= TARGET_ACCURACY and total >= MIN_TOTAL_TEST_CASES and coverage >= TARGET_COVERAGE and lower >= 0.80
    return {
        "horizon_days": horizon,
        "test_cases": total,
        "hits": hits,
        "accuracy": accuracy,
        "coverage": coverage,
        "wilson_95_lower": lower,
        "passed": passed,
        "windows": windows,
    }


def walk_forward_ensemble(data, horizon):
    work = data.copy()
    future_return = work["close"].shift(-horizon) / work["close"] - 1
    neutral = 0.005 if horizon <= 5 else 0.01
    work["target"] = np.where(future_return > neutral, 1, np.where(future_return < -neutral, -1, 0))
    atoms = make_atoms(data.attrs["features"])
    all_hits = []
    total_eligible = 0
    window_summaries = []
    start = 0
    while start + TRAIN_DAYS + TEST_DAYS + horizon <= len(work):
        train = work.iloc[start : start + TRAIN_DAYS]
        test = work.iloc[start + TRAIN_DAYS : start + TRAIN_DAYS + TEST_DAYS]
        selected = select_rules_fast(train, atoms, limit=20)
        votes = np.zeros(len(test), dtype=int)
        active = np.zeros(len(test), dtype=int)
        for rule in selected:
            mask = apply_rule(test, rule["rule"]).to_numpy()
            votes[mask] += rule["direction"]
            active[mask] += 1
        prediction = np.sign(votes)
        target = test["target"].to_numpy()
        actionable = (active >= 2) & (prediction != 0)
        hits = prediction[actionable] == target[actionable]
        all_hits.extend(hits.astype(int).tolist())
        total_eligible += len(test)
        window_summaries.append({
            "test_end": str(test["date"].max().date()),
            "selected_rules": len(selected),
            "test_cases": int(actionable.sum()),
            "test_accuracy": float(hits.mean()) if len(hits) else 0.0,
        })
        start += TEST_DAYS
    total = len(all_hits)
    hit_count = sum(all_hits)
    accuracy = hit_count / total if total else 0.0
    coverage = total / total_eligible if total_eligible else 0.0
    lower = wilson_lower(hit_count, total)
    passed = accuracy >= TARGET_ACCURACY and total >= MIN_TOTAL_TEST_CASES and coverage >= TARGET_COVERAGE and lower >= 0.80
    return {
        "mode": "top20_consensus_vote",
        "horizon_days": horizon,
        "test_cases": total,
        "hits": hit_count,
        "accuracy": accuracy,
        "coverage": coverage,
        "wilson_95_lower": lower,
        "passed": passed,
        "windows": window_summaries,
    }


def make_atoms(features):
    atoms = []
    for feature in features:
        for threshold in THRESHOLDS:
            atoms.append((feature, ">=", threshold))
            atoms.append((feature, "<=", -threshold))
    return atoms


def select_rule_fast(train, atoms):
    candidates = select_rules_fast(train, atoms, limit=1)
    return candidates[0] if candidates else {"rule": tuple(), "direction": 1, "accuracy": 0.0, "score": 0.0, "label": "no_rule"}


def select_rules_fast(train, atoms, limit=20):
    matrix = np.column_stack([
        (train[feature].to_numpy() >= threshold) if operator == ">="
        else (train[feature].to_numpy() <= threshold)
        for feature, operator, threshold in atoms
    ])
    valid = np.ones(len(train), dtype=bool)
    up_mask = train["target"].to_numpy() == 1
    down_mask = train["target"].to_numpy() == -1
    m = matrix.astype(np.int16)
    total_pair = m[valid].T @ m[valid]
    up_pair = m[up_mask].T @ m[up_mask]
    down_pair = m[down_mask].T @ m[down_mask]
    candidates = []
    count_atoms = len(atoms)
    for left in range(count_atoms):
        for right in range(left, count_atoms):
            if left != right and atoms[left][0] == atoms[right][0]:
                continue
            count = int(total_pair[left, right])
            if count < MIN_TRAIN_CASES:
                continue
            up = int(up_pair[left, right])
            down = int(down_pair[left, right])
            direction = 1 if up >= down else -1
            hits = max(up, down)
            accuracy = hits / count
            score = wilson_lower(hits, count) + min(count, 200) / 20000
            rule = (atoms[left],) if left == right else (atoms[left], atoms[right])
            candidates.append({"rule": rule, "direction": direction, "accuracy": accuracy, "score": score, "label": rule_label(rule)})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    seen_signatures = set()
    for item in candidates:
        signature = tuple(sorted(atom[0] for atom in item["rule"])) + (item["direction"],)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def select_rule(train, rules):
    best = None
    for rule in rules:
        mask = apply_rule(train, rule) & (train["target"] != 0)
        targets = train.loc[mask, "target"]
        count = len(targets)
        if count < MIN_TRAIN_CASES:
            continue
        up = int((targets == 1).sum())
        down = int((targets == -1).sum())
        direction = 1 if up >= down else -1
        hits = max(up, down)
        accuracy = hits / count
        lower = wilson_lower(hits, count)
        score = lower + min(count, 200) / 20000
        if best is None or score > best["score"]:
            best = {"rule": rule, "direction": direction, "accuracy": accuracy, "score": score, "label": rule_label(rule)}
    return best or {"rule": tuple(), "direction": 1, "accuracy": 0.0, "score": 0.0, "label": "no_rule"}


def apply_rule(frame, rule):
    mask = pd.Series(True, index=frame.index)
    for feature, operator, threshold in rule:
        values = frame[feature]
        mask &= values >= threshold if operator == ">=" else values <= threshold
    return mask.fillna(False)


def rule_label(rule):
    return " AND ".join(f"{feature} {operator} {threshold}" for feature, operator, threshold in rule)


def wilson_lower(hits, total, z=1.96):
    if total == 0:
        return 0.0
    p = hits / total
    denominator = 1 + z * z / total
    center = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (center - margin) / denominator


def render_report(payload):
    lines = [
        "# 選擇性高命中訊號研究",
        "",
        f"- 資料期間：{payload['data_start']} 至 {payload['data_end']}",
        f"- 90% 達標週期：{payload['passing_horizons'] or '無'}",
        "",
        "| 週期 | 樣本外案例 | 命中率 | 覆蓋率 | 95% 下限 | 是否達標 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in payload["results"]:
        lines.append(f"| {item['horizon_days']}日 | {item['test_cases']} | {item['accuracy']:.2%} | {item['coverage']:.2%} | {item['wilson_95_lower']:.2%} | {'是' if item['passed'] else '否'} |")
    lines.extend(["", "## 多規則一致投票", "", "| 週期 | 樣本外案例 | 命中率 | 覆蓋率 | 95% 下限 | 是否達標 |", "| --- | ---: | ---: | ---: | ---: | --- |"]) 
    for item in payload["ensemble_results"]:
        lines.append(f"| {item['horizon_days']}日 | {item['test_cases']} | {item['accuracy']:.2%} | {item['coverage']:.2%} | {item['wilson_95_lower']:.2%} | {'是' if item['passed'] else '否'} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
