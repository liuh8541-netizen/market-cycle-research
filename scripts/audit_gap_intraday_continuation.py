"""Post-hoc simple-baseline audit prompted by the transmission-model result."""

import json
import math
from pathlib import Path

import numpy as np

from research_open_to_close_transmission import HISTORY, TEST, VALID, prepare, wilson, exact_mcnemar


ROOT = Path(__file__).resolve().parents[1]
RULES = ["gap", "night", "agreement"]
THRESHOLDS = [0.0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01]


def predictions(frame, rule):
    gap = np.where(frame["gap_return"].to_numpy() >= 0, 1, -1)
    night = np.where(frame["night_spread"].to_numpy() >= 0, 1, -1)
    if rule == "gap":
        return gap, np.abs(frame["gap_return"].to_numpy())
    if rule == "night":
        return night, np.abs(frame["night_spread"].to_numpy())
    agree = gap == night
    confidence = np.minimum(np.abs(frame["gap_return"].to_numpy()), np.abs(frame["night_spread"].to_numpy()))
    return gap, np.where(agree, confidence, -1.0)


def select(history):
    valid = history.iloc[-VALID:]
    target = valid["target"].to_numpy(int)
    best = None
    for rule in RULES:
        pred, confidence = predictions(valid, rule)
        for threshold in THRESHOLDS:
            use = confidence >= threshold
            cases = int(use.sum())
            if cases < 15 or cases / len(valid) < 0.10:
                continue
            accuracy = float((pred[use] == target[use]).mean())
            rank = (accuracy, cases / len(valid), -threshold, -RULES.index(rule))
            if best is None or rank > best[0]:
                best = (rank, rule, threshold, accuracy, cases)
    return best


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
            _, rule, threshold, valid_accuracy, valid_cases = config
            pred, confidence = predictions(test, rule)
            use = confidence >= threshold
            for offset in np.flatnonzero(use):
                row = test.iloc[offset]
                records.append({
                    "date": str(row["date"].date()), "rule": rule, "threshold": threshold,
                    "validation_accuracy": valid_accuracy, "validation_cases": valid_cases,
                    "prediction": int(pred[offset]), "target": int(row["target"]),
                    "hit": int(pred[offset] == row["target"]),
                    "gap_return": float(row["gap_return"]), "night_spread": float(row["night_spread"]),
                })
            windows.append({
                "test_end": str(test["date"].max().date()), "rule": rule, "threshold": threshold,
                "validation_accuracy": valid_accuracy, "validation_cases": valid_cases,
                "test_cases": int(use.sum()),
                "test_accuracy": float((pred[use] == test.loc[use, "target"].to_numpy()).mean()) if use.any() else 0,
            })
        start += TEST
    cases = len(records)
    hits = sum(row["hit"] for row in records)
    accuracy = hits / cases if cases else 0
    coverage = cases / tested if tested else 0
    alternatives = {
        "always_up": [1] * cases,
        "unfiltered_gap": [1 if row["gap_return"] >= 0 else -1 for row in records],
        "unfiltered_night": [1 if row["night_spread"] >= 0 else -1 for row in records],
    }
    baselines = {}
    for name, values in alternatives.items():
        bh = sum(pred == row["target"] for pred, row in zip(values, records))
        baselines[name] = {"hits": bh, "accuracy": bh / cases if cases else 0}
    strongest_name = max(baselines, key=lambda name: baselines[name]["accuracy"])
    strongest_values = alternatives[strongest_name]
    model_only = sum(row["hit"] and pred != row["target"] for pred, row in zip(strongest_values, records))
    baseline_only = sum(not row["hit"] and pred == row["target"] for pred, row in zip(strongest_values, records))
    p_value = exact_mcnemar(model_only, baseline_only)
    lower = wilson(hits, cases)
    # This is post-hoc after inspecting the parent study, so it cannot be promoted
    # from the same dates even if numeric gates happen to pass.
    numeric_gate = (
        accuracy >= .90 and cases >= 100 and coverage >= .10 and lower >= .80
        and accuracy > baselines[strongest_name]["accuracy"] and model_only > baseline_only and p_value < .05
    )
    result = {
        "cases": cases, "hits": hits, "accuracy": accuracy, "coverage": coverage,
        "wilson_95_lower": lower, "baselines": baselines,
        "strongest_baseline": strongest_name,
        "strongest_baseline_accuracy": baselines[strongest_name]["accuracy"],
        "model_only": model_only, "baseline_only": baseline_only,
        "mcnemar_exact_p": p_value, "numeric_gate": numeric_gate,
        "promotable": False, "reason": "Post-hoc family identified after inspecting the parent OOS result; requires untouched future confirmation.",
        "windows": windows, "records": records,
    }
    payload = {
        "scope": "Same-day TWII close versus open, prediction at cash open.",
        "method": "Post-hoc simple audit: gap continuation, night continuation, or their agreement; fixed threshold grid selected on prior 126 days and frozen for following 126-day block.",
        "result": result,
    }
    out = ROOT / "reports"
    (out / "gap_intraday_continuation_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Gap intraday continuation audit", "", payload["scope"], "", payload["method"], "",
        "This family was identified after inspecting the parent OOS result and is therefore exploratory, not promotable from the same historical dates.", "",
        f"OOS: {hits}/{cases} = {accuracy:.2%}; coverage {coverage:.2%}; Wilson lower {lower:.2%}.",
        f"Strongest identical-date alternative: {strongest_name} {baselines[strongest_name]['accuracy']:.2%}.",
        f"Paired model-only/baseline-only {model_only}/{baseline_only}; p={p_value:.4g}; numeric gate={numeric_gate}; promotable=False.", "",
        "| Test end | Rule | Threshold | Cases | Accuracy | Validation cases/accuracy |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for window in windows:
        lines.append(
            f"| {window['test_end']} | {window['rule']} | {window['threshold']:.3%} | "
            f"{window['test_cases']} | {window['test_accuracy']:.2%} | "
            f"{window['validation_cases']}/{window['validation_accuracy']:.2%} |"
        )
    (out / "gap_intraday_continuation_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ["windows", "records"]}, indent=2))


if __name__ == "__main__":
    main()
