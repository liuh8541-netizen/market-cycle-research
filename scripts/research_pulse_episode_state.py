"""Causal event-level test of a time-varying pulse/constitution/complexion state.

Each prediction is made from earlier, fully resolved episodes only.  Event origins
are separated by the forecast horizon, so overlapping daily labels cannot inflate
the effective sample size.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_causal_phase import wilson_lower
from research_three_layer_purged import build_frame


HORIZONS = [5, 10, 20]
MIN_TRAIN = 80
VALID_FRACTION = 0.25
LAMBDAS = [0.1, 1.0, 10.0, 100.0]
QUANTILES = [0.0, 0.25, 0.5, 0.7]
FEATURES = [
    "ret_5", "ret_20", "ret_60", "ret_120", "ret_240", "vol_ratio",
    "drawdown_120", "pulse_rate", "body_strength", "body_momentum",
    "color_breadth", "vix_pressure", "inst_foreign_net_20d_z",
    "inst_trust_net_20d_z", "inst_dealer_net_20d_z",
    "futures_foreign_net_20d_z", "option_foreign_net_20d_z",
]


def ridge_fit(frame, features, lam):
    raw = frame[features].replace([np.inf, -np.inf], np.nan)
    med = raw.median().fillna(0.0)
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


def ridge_predict(model, frame, features):
    weights, med, mean, std = model
    raw = frame[features].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0)
    x = ((raw - mean) / std).to_numpy(float)
    score = np.column_stack([np.ones(len(x)), x]) @ weights
    return np.where(score >= 0, 1, -1), np.abs(score)


def make_events(base, horizon):
    x = base.copy().sort_values("date").reset_index(drop=True)
    # A pulse disorder is observable at the close: price displacement or a
    # volatility acceleration.  Entry-only prevents repeated rows in one pulse.
    disorder = (
        (x["ret_5"].abs() >= 0.025)
        | (x["vol_ratio"] >= 1.15)
        | ((x["ret_20"].abs() >= 0.06) & (x["pulse_rate"] >= 0.45))
    )
    onset = disorder & ~disorder.shift(1).fillna(False)
    positions = []
    last = -10**9
    for pos in x.index[onset]:
        if pos - last >= horizon:
            positions.append(int(pos))
            last = int(pos)
    events = x.loc[positions, ["date", *FEATURES]].copy()
    events["position"] = positions
    future = x["close"].shift(-horizon) / x["close"] - 1
    events["future_return"] = events["position"].map(future)
    events = events.dropna(subset=["future_return"]).reset_index(drop=True)
    events["target"] = np.where(events["future_return"] > 0, 1, -1)
    for col in FEATURES:
        if col not in events:
            events[col] = np.nan
    return events


def select_config(history):
    split = max(MIN_TRAIN // 2, int(len(history) * (1 - VALID_FRACTION)))
    train, valid = history.iloc[:split], history.iloc[split:]
    if len(train) < 40 or len(valid) < 20:
        return None
    best = None
    for lam in LAMBDAS:
        model = ridge_fit(train, FEATURES, lam)
        pred, confidence = ridge_predict(model, valid, FEATURES)
        actual = valid["target"].to_numpy(int)
        for quantile in QUANTILES:
            threshold = float(np.quantile(confidence, quantile))
            use = confidence >= threshold
            cases = int(use.sum())
            if cases < 20 or cases / len(valid) < 0.10:
                continue
            accuracy = float((pred[use] == actual[use]).mean())
            # Accuracy first; coverage only breaks nearly equal configurations.
            rank = (accuracy, cases / len(valid), -lam, -quantile)
            if best is None or rank > best[0]:
                best = (rank, lam, threshold, accuracy, cases, quantile)
    return best


def mcnemar_exact(model_only, baseline_only):
    discordant = model_only + baseline_only
    if not discordant:
        return 1.0
    smaller = min(model_only, baseline_only)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def evaluate(base, horizon):
    events = make_events(base, horizon)
    records = []
    for idx in range(MIN_TRAIN, len(events)):
        current = events.iloc[idx]
        # With non-overlap spacing, the immediately preceding episode can still
        # finish on the current origin. Exclude any outcome ending after origin.
        history = events.iloc[:idx].loc[
            events.iloc[:idx]["position"] + horizon <= current["position"]
        ]
        if len(history) < MIN_TRAIN:
            continue
        config = select_config(history)
        if config is None:
            continue
        _, lam, threshold, validation_accuracy, validation_cases, quantile = config
        model = ridge_fit(history, FEATURES, lam)
        pred, confidence = ridge_predict(model, events.iloc[[idx]], FEATURES)
        if confidence[0] < threshold:
            continue
        target = int(current["target"])
        prior_up = float((history["target"] == 1).mean())
        baselines = {
            "prior_majority": 1 if prior_up >= 0.5 else -1,
            "ret5_continuation": 1 if current["ret_5"] >= 0 else -1,
            "ret5_reversal": -1 if current["ret_5"] >= 0 else 1,
            "macro240": 1 if current["ret_240"] >= 0 else -1,
        }
        records.append({
            "date": str(current["date"].date()), "position": int(current["position"]),
            "prediction": int(pred[0]), "target": target,
            "hit": int(pred[0] == target), "confidence": float(confidence[0]),
            "lambda": lam, "threshold": threshold, "selected_quantile": quantile,
            "validation_accuracy": validation_accuracy, "validation_cases": validation_cases,
            "training_episodes": len(history), "future_return": float(current["future_return"]),
            "baselines": {name: int(value) for name, value in baselines.items()},
        })
    eligible = max(0, len(events) - MIN_TRAIN)
    cases = len(records)
    hits = sum(row["hit"] for row in records)
    accuracy = hits / cases if cases else 0.0
    baseline_results = {}
    for name in ["prior_majority", "ret5_continuation", "ret5_reversal", "macro240"]:
        baseline_hits = sum(row["baselines"][name] == row["target"] for row in records)
        baseline_results[name] = {
            "hits": baseline_hits,
            "accuracy": baseline_hits / cases if cases else 0.0,
        }
    strongest_name = max(baseline_results, key=lambda name: baseline_results[name]["accuracy"])
    strongest = baseline_results[strongest_name]
    model_only = sum(
        row["hit"] and row["baselines"][strongest_name] != row["target"] for row in records
    )
    baseline_only = sum(
        not row["hit"] and row["baselines"][strongest_name] == row["target"] for row in records
    )
    coverage = cases / eligible if eligible else 0.0
    lower = wilson_lower(hits, cases)
    passed = (
        accuracy >= 0.90 and cases >= 100 and coverage >= 0.10 and lower >= 0.80
        and accuracy > strongest["accuracy"] and model_only > baseline_only
        and mcnemar_exact(model_only, baseline_only) < 0.05
    )
    return {
        "horizon_days": horizon, "raw_nonoverlap_events": len(events),
        "eligible_oos_events": eligible, "cases": cases, "hits": hits,
        "accuracy": accuracy, "coverage": coverage, "wilson_95_lower": lower,
        "baselines": baseline_results, "strongest_baseline": strongest_name,
        "strongest_baseline_accuracy": strongest["accuracy"],
        "model_only": model_only, "baseline_only": baseline_only,
        "mcnemar_exact_p": mcnemar_exact(model_only, baseline_only),
        "passed": passed, "records": records,
    }


def main():
    base, _ = build_frame()
    results = [evaluate(base, horizon) for horizon in HORIZONS]
    payload = {
        "hypothesis": "Pulse direction is not fixed; onset constitution and lagged external complexion identify continuation versus recovery.",
        "method": "Non-overlapping causal pulse onsets; per-event expanding walk-forward; only fully resolved prior episodes; nested temporal ridge/abstention selection; identical-date simple baselines.",
        "gate": "accuracy >=90%, cases >=100, coverage >=10%, Wilson lower >=80%, and significant paired superiority over strongest identical-date simple baseline.",
        "features": FEATURES, "results": results,
        "passing": [row["horizon_days"] for row in results if row["passed"]],
    }
    out = ROOT / "reports"
    (out / "pulse_episode_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Pulse episode latent-state research", "", payload["hypothesis"], "",
        payload["method"], "", payload["gate"], "",
        "| Horizon | Raw events | OOS cases | Accuracy | Coverage | Wilson lower | Strongest baseline | Paired model/base only | p | Passed |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in results:
        lines.append(
            f"| {row['horizon_days']} | {row['raw_nonoverlap_events']} | {row['cases']} | "
            f"{row['accuracy']:.2%} | {row['coverage']:.2%} | {row['wilson_95_lower']:.2%} | "
            f"{row['strongest_baseline']} {row['strongest_baseline_accuracy']:.2%} | "
            f"{row['model_only']}/{row['baseline_only']} | {row['mcnemar_exact_p']:.4g} | {row['passed']} |"
        )
    lines += ["", "All event records and frozen per-event configurations are stored in the JSON companion file."]
    (out / "pulse_episode_state.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "passing": payload["passing"],
        "results": [{key: value for key, value in row.items() if key != "records"} for row in results],
    }, indent=2))


if __name__ == "__main__":
    main()
