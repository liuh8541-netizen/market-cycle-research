"""Locked ablation of TX intranight path for material opening-gap risk."""

import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_night_microstructure_gap as source  # noqa: E402

HISTORY, VALID, TEST = 756, 126, 126
LAMBDAS = [0.1, 1.0, 10.0, 100.0]
ABSTENTION = [0.0, 0.25, 0.5, 0.65, 0.75, 0.85]
EVENT_QUANTILE = 0.75
VARIANTS = {
    "daily_amplitude_control": source.CONTROL_FEATURES,
    "daily_plus_microstructure": source.CONTROL_FEATURES + source.PATH_FEATURES,
}


def wilson(h, n):
    if not n:
        return 0.0
    z, p = 1.959963984540054, h / n
    den = 1 + z * z / n
    return (p + z*z/(2*n) - z*math.sqrt((p*(1-p)+z*z/(4*n))/n)) / den


def mcnemar(a, b):
    n = a + b
    if not n:
        return 1.0
    low = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(low + 1)) / 2**n)


def fit(frame, features, target, lam):
    raw = frame[features]
    med = raw.median().fillna(0)
    filled = raw.fillna(med)
    mean, std = filled.mean(), filled.std().replace(0, 1).fillna(1)
    x = np.column_stack([np.ones(len(filled)), ((filled - mean) / std).to_numpy(float)])
    penalty = np.eye(x.shape[1]) * lam
    penalty[0, 0] = 0
    weights = np.linalg.pinv(x.T @ x + penalty) @ x.T @ target
    return weights, med, mean, std


def predict(model, frame, features):
    weights, med, mean, std = model
    raw = frame[features].fillna(med).fillna(0)
    x = np.column_stack([np.ones(len(raw)), ((raw - mean) / std).to_numpy(float)])
    score = x @ weights
    return np.where(score >= 0, 1, -1), np.abs(score)


def choose(history, features):
    train, valid = history.iloc[:-VALID], history.iloc[-VALID:]
    event_cut = float(train["gap_return"].abs().quantile(EVENT_QUANTILE))
    y_train = np.where(train["gap_return"].abs() >= event_cut, 1, -1)
    y_valid = np.where(valid["gap_return"].abs() >= event_cut, 1, -1)
    best = None
    for lam in LAMBDAS:
        model = fit(train, features, y_train, lam)
        pred, conf = predict(model, valid, features)
        train_conf = predict(model, train, features)[1]
        for q in ABSTENTION:
            cut = float(np.quantile(train_conf, q))
            use = conf >= cut
            cases = int(use.sum())
            if cases < 15 or use.mean() < 0.10:
                continue
            acc = float((pred[use] == y_valid[use]).mean())
            rank = (acc >= 0.90, acc, cases, -lam, -q)
            if best is None or rank > best[0]:
                best = (rank, lam, q, acc, cases)
    return best


def evaluate(frame, features):
    records, windows, tested = [], [], 0
    for start in range(0, len(frame) - HISTORY - TEST + 1, TEST):
        history = frame.iloc[start:start+HISTORY]
        test = frame.iloc[start+HISTORY:start+HISTORY+TEST]
        tested += len(test)
        chosen = choose(history, features)
        if chosen is None:
            continue
        _, lam, q, va, vn = chosen
        event_cut = float(history["gap_return"].abs().quantile(EVENT_QUANTILE))
        y_history = np.where(history["gap_return"].abs() >= event_cut, 1, -1)
        y_test = np.where(test["gap_return"].abs() >= event_cut, 1, -1)
        model = fit(history, features, y_history, lam)
        pred, conf = predict(model, test, features)
        conf_cut = float(np.quantile(predict(model, history, features)[1], q))
        use = conf >= conf_cut
        # Frozen simple scalar baseline: top quartile absolute night spread in
        # the same history predicts a material gap.
        spread_cut = float(history["tx_night_spread_per"].abs().quantile(EVENT_QUANTILE))
        scalar = np.where(test["tx_night_spread_per"].abs() >= spread_cut, 1, -1)
        for pos in np.flatnonzero(use):
            records.append({
                "date": str(test.iloc[pos].date.date()),
                "prediction": int(pred[pos]), "target": int(y_test[pos]),
                "hit": int(pred[pos] == y_test[pos]),
                "scalar_baseline": int(scalar[pos]), "always_small": -1,
                "event_cut": event_cut, "spread_cut": spread_cut,
                "confidence": float(conf[pos]), "lambda": lam, "quantile": q,
            })
        cases = int(use.sum())
        windows.append({
            "test_start": str(test.date.min().date()), "test_end": str(test.date.max().date()),
            "cases": cases,
            "accuracy": float((pred[use] == y_test[use]).mean()) if cases else 0,
            "validation_accuracy": va, "validation_cases": vn,
        })
    cases, hits = len(records), sum(r["hit"] for r in records)
    baselines = {}
    for name in ["scalar_baseline", "always_small"]:
        bh = sum(r[name] == r["target"] for r in records)
        baselines[name] = {"hits": bh, "accuracy": bh/cases if cases else 0}
    strongest = max(baselines, key=lambda k: baselines[k]["accuracy"])
    model_only = sum(r["hit"] and r[strongest] != r["target"] for r in records)
    base_only = sum(not r["hit"] and r[strongest] == r["target"] for r in records)
    accuracy = hits/cases if cases else 0
    material = [w for w in windows if w["cases"] >= 10]
    result = {
        "tested_rows": tested, "cases": cases, "hits": hits, "accuracy": accuracy,
        "coverage": cases/tested if tested else 0, "wilson_95_lower": wilson(hits, cases),
        "baselines": baselines, "strongest_baseline": strongest,
        "strongest_baseline_accuracy": baselines[strongest]["accuracy"],
        "model_only": model_only, "baseline_only": base_only,
        "mcnemar_exact_p": mcnemar(model_only, base_only),
        "minimum_material_window_accuracy": min((w["accuracy"] for w in material), default=0),
        "windows": windows, "records": records,
    }
    result["passed"] = (
        cases >= 100 and result["coverage"] >= .10 and accuracy >= .90
        and result["wilson_95_lower"] >= .80
        and accuracy > result["strongest_baseline_accuracy"]
        and model_only > base_only and result["mcnemar_exact_p"] < .05
        and result["minimum_material_window_accuracy"] >= .80
    )
    return result


def paired(left, right):
    a, b = ({r["date"]: r for r in x["records"]} for x in [left, right])
    dates = sorted(a.keys() & b.keys())
    lo = sum(a[d]["hit"] and not b[d]["hit"] for d in dates)
    ro = sum(b[d]["hit"] and not a[d]["hit"] for d in dates)
    return {
        "identical_dates": len(dates),
        "control_accuracy": sum(a[d]["hit"] for d in dates)/len(dates) if dates else 0,
        "path_accuracy": sum(b[d]["hit"] for d in dates)/len(dates) if dates else 0,
        "control_only": lo, "path_only": ro, "mcnemar_exact_p": mcnemar(lo, ro),
    }


def main():
    frame = source.prepare().dropna(subset=["gap_return"]).reset_index(drop=True)
    variants = {name: evaluate(frame, features) for name, features in VARIANTS.items()}
    pair = paired(variants["daily_amplitude_control"], variants["daily_plus_microstructure"])
    increment = pair["path_only"] > pair["control_only"] and pair["mcnemar_exact_p"] < .05
    variants["daily_plus_microstructure"]["passed"] &= increment
    payload = {
        "experiment_id": "microstructure_gap_amplitude_v1",
        "status": "strict historical OOS amplitude-risk experiment; not production unless every gate passes",
        "target": "whether absolute TWII opening gap exceeds the frozen prior-history 75th percentile",
        "decision_time": "after completed TX night session and before TWII cash open",
        "event_quantile": EVENT_QUANTILE,
        "method": "630/126/126 nested ridge/abstention ablation; event and scalar-baseline cuts frozen from prior history.",
        "gate": ">=90% accuracy, >=100 cases, >=10% coverage, Wilson lower >=80%, min block >=80%, significant superiority over scalar baseline and daily control.",
        "aligned_rows": len(frame), "variants": variants,
        "path_increment": pair, "path_increment_passed": increment,
        "passing": [n for n, r in variants.items() if r["passed"]],
    }
    out = ROOT / "reports"
    (out/"microstructure_gap_amplitude.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Microstructure opening-gap amplitude research", "", payload["status"], "",
             payload["target"], "",
             "| Variant | Cases | Accuracy | Coverage | Wilson lower | Strongest baseline | Model/base only | Min block | Passed |",
             "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |"]
    for name, r in variants.items():
        lines.append(f"| {name} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | "
                     f"{r['wilson_95_lower']:.2%} | {r['strongest_baseline']} {r['strongest_baseline_accuracy']:.2%} | "
                     f"{r['model_only']}/{r['baseline_only']} (p={r['mcnemar_exact_p']:.4g}) | "
                     f"{r['minimum_material_window_accuracy']:.2%} | {r['passed']} |")
    lines += ["", f"Common-date control/path: {pair['control_accuracy']:.2%}/{pair['path_accuracy']:.2%}; "
              f"exclusive wins {pair['control_only']}/{pair['path_only']}; p={pair['mcnemar_exact_p']:.4g}.",
              f"Independent path increment passed: {increment}."]
    (out/"microstructure_gap_amplitude.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({
        "aligned_rows": len(frame),
        "variants": {n: {k:v for k,v in r.items() if k not in ("records","windows","baselines")} for n,r in variants.items()},
        "path_increment": pair,
    }, indent=2))


if __name__ == "__main__":
    main()
