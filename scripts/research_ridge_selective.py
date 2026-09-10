import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_selective_signals import (
    HORIZONS, MIN_TOTAL_TEST_CASES, TARGET_ACCURACY, TARGET_COVERAGE,
    TEST_DAYS, TRAIN_DAYS, build_research_frame, wilson_lower,
)


LAMBDAS = [0.1, 1.0, 10.0, 100.0]
MIN_INNER_CASES = 20


def main():
    data = build_research_frame()
    features = data.attrs["features"]
    results = [evaluate_horizon(data, features, horizon) for horizon in HORIZONS]
    payload = {"method": "nested walk-forward ridge classifier with abstention", "results": results}
    (ROOT / "reports" / "ridge_selective_research.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "reports" / "ridge_selective_research.md").write_text(render(payload), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


def evaluate_horizon(data, features, horizon):
    work = data.copy()
    future = work["close"].shift(-horizon) / work["close"] - 1
    neutral = 0.005 if horizon <= 5 else 0.01
    work["target"] = np.where(future > neutral, 1, np.where(future < -neutral, -1, 0))
    all_hits = []
    total_rows = 0
    windows = []
    start = 0
    while start + TRAIN_DAYS + TEST_DAYS + horizon <= len(work):
        train = work.iloc[start:start + TRAIN_DAYS]
        test = work.iloc[start + TRAIN_DAYS:start + TRAIN_DAYS + TEST_DAYS]
        inner_cut = int(len(train) * 0.80)
        fit = train.iloc[:inner_cut]
        validation = train.iloc[inner_cut:]
        config = choose_config(fit, validation, features)
        model = fit_ridge(train, features, config["lambda"])
        pred, confidence = predict_ridge(model, test, features)
        actionable = confidence >= config["threshold"]
        target = test["target"].to_numpy()
        hits = pred[actionable] == target[actionable]
        all_hits.extend(hits.astype(int).tolist())
        total_rows += len(test)
        windows.append({
            "test_end": str(test["date"].max().date()),
            "lambda": config["lambda"],
            "threshold": config["threshold"],
            "inner_accuracy": config["accuracy"],
            "test_cases": int(actionable.sum()),
            "test_accuracy": float(hits.mean()) if len(hits) else 0.0,
        })
        start += TEST_DAYS
    cases = len(all_hits)
    hit_count = sum(all_hits)
    accuracy = hit_count / cases if cases else 0.0
    coverage = cases / total_rows if total_rows else 0.0
    lower = wilson_lower(hit_count, cases)
    passed = accuracy >= TARGET_ACCURACY and cases >= MIN_TOTAL_TEST_CASES and coverage >= TARGET_COVERAGE and lower >= 0.80
    return {"horizon_days": horizon, "test_cases": cases, "accuracy": accuracy, "coverage": coverage, "wilson_95_lower": lower, "passed": passed, "windows": windows}


def choose_config(fit, validation, features):
    best = None
    for ridge_lambda in LAMBDAS:
        model = fit_ridge(fit, features, ridge_lambda)
        pred, confidence = predict_ridge(model, validation, features)
        target = validation["target"].to_numpy()
        thresholds = np.unique(np.quantile(confidence, np.linspace(0, 0.9, 19)))
        for threshold in thresholds:
            mask = confidence >= threshold
            count = int(mask.sum())
            if count < MIN_INNER_CASES or count / len(validation) < TARGET_COVERAGE:
                continue
            accuracy = float((pred[mask] == target[mask]).mean())
            score = accuracy + min(count / len(validation), 0.30) * 0.02
            if best is None or score > best["score"]:
                best = {"lambda": ridge_lambda, "threshold": float(threshold), "accuracy": accuracy, "score": score}
    return best or {"lambda": 10.0, "threshold": 0.0, "accuracy": 0.0, "score": 0.0}


def fit_ridge(frame, features, ridge_lambda):
    x = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    x = np.column_stack([np.ones(len(x)), x])
    labels = frame["target"].to_numpy()
    y = np.column_stack([(labels == value).astype(float) for value in [-1, 0, 1]])
    penalty = np.eye(x.shape[1]) * ridge_lambda
    penalty[0, 0] = 0
    weights = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y
    return weights


def predict_ridge(weights, frame, features):
    x = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    x = np.column_stack([np.ones(len(x)), x])
    scores = x @ weights
    order = np.argsort(scores, axis=1)
    prediction = np.array([-1, 0, 1])[order[:, -1]]
    confidence = scores[np.arange(len(scores)), order[:, -1]] - scores[np.arange(len(scores)), order[:, -2]]
    return prediction, confidence


def render(payload):
    lines = ["# Ridge 選擇性預測研究", "", "| 週期 | 案例 | 命中率 | 覆蓋率 | 95% 下限 | 達標 |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for item in payload["results"]:
        lines.append(f"| {item['horizon_days']}日 | {item['test_cases']} | {item['accuracy']:.2%} | {item['coverage']:.2%} | {item['wilson_95_lower']:.2%} | {'是' if item['passed'] else '否'} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
