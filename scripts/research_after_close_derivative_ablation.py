"""Registered after-close derivatives ablation for the next TWII opening gap."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_after_close_next_gap import (
    HISTORY, VALID, TEST, LAMBDAS, QUANTILES, FEATURES as CASH_FEATURES,
    prepare, wilson, mcnemar,
)


DERIVATIVE_FEATURES = [
    "futures_volume_proxy", "futures_volume_20d_z", "futures_oi_proxy", "futures_oi_change_20d",
    "futures_inst_net_proxy", "futures_inst_net_5d", "futures_inst_net_20d",
    "futures_foreign_net_proxy", "futures_foreign_net_5d", "futures_foreign_net_20d",
    "futures_trust_net_proxy", "futures_trust_net_5d", "futures_trust_net_20d",
    "futures_dealer_net_proxy", "futures_dealer_net_5d", "futures_dealer_net_20d",
    "option_put_call_proxy", "option_put_call_20d_z", "option_oi_put_call",
    "option_inst_net_proxy", "option_inst_net_5d", "option_inst_net_20d",
    "option_foreign_net_proxy", "option_foreign_net_5d", "option_foreign_net_20d",
    "option_trust_net_proxy", "option_trust_net_5d", "option_trust_net_20d",
    "option_dealer_net_proxy", "option_dealer_net_5d", "option_dealer_net_20d",
]
VARIANTS = {
    "derivatives_only": DERIVATIVE_FEATURES,
    "cash_margin_only_same_period": CASH_FEATURES,
    "combined": CASH_FEATURES + DERIVATIVE_FEATURES,
}


def fit(frame, features, lam):
    raw = frame[features].replace([np.inf, -np.inf], np.nan)
    med = raw.median().fillna(0)
    raw = raw.fillna(med)
    mean, std = raw.mean(), raw.std().replace(0, 1).fillna(1)
    x = ((raw - mean) / std).to_numpy(float)
    x = np.column_stack([np.ones(len(x)), x])
    y = frame["target"].to_numpy(float)
    penalty = np.eye(x.shape[1]) * lam
    penalty[0, 0] = 0
    weights = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y
    return weights, med, mean, std


def predict(model, frame, features):
    weights, med, mean, std = model
    raw = frame[features].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0)
    x = ((raw - mean) / std).to_numpy(float)
    score = np.column_stack([np.ones(len(x)), x]) @ weights
    return np.where(score >= 0, 1, -1), np.abs(score)


def select(history, features):
    train, valid = history.iloc[:-VALID], history.iloc[-VALID:]
    actual = valid["target"].to_numpy(int)
    best = None
    for lam in LAMBDAS:
        model = fit(train, features, lam)
        pred, confidence = predict(model, valid, features)
        for quantile in QUANTILES:
            threshold = float(np.quantile(confidence, quantile))
            use = confidence >= threshold
            cases = int(use.sum())
            if cases < 15 or cases / len(valid) < .10:
                continue
            accuracy = float((pred[use] == actual[use]).mean())
            rank = (accuracy, cases / len(valid), -lam, -quantile)
            if best is None or rank > best[0]:
                best = (rank, lam, threshold, accuracy, cases, quantile)
    return best


def evaluate(frame, features):
    records, windows, tested = [], [], 0
    start = 0
    while start + HISTORY + TEST <= len(frame):
        history = frame.iloc[start : start + HISTORY]
        test = frame.iloc[start + HISTORY : start + HISTORY + TEST]
        tested += len(test)
        config = select(history, features)
        if config:
            _, lam, threshold, va, vn, quantile = config
            model = fit(history, features, lam)
            pred, confidence = predict(model, test, features)
            use = confidence >= threshold
            for offset in np.flatnonzero(use):
                row = test.iloc[offset]
                records.append({
                    "signal_date": str(row["date"].date()), "target_date": str(row["target_date"].date()),
                    "prediction": int(pred[offset]), "target": int(row["target"]),
                    "hit": int(pred[offset] == row["target"]), "confidence": float(confidence[offset]),
                    "lambda": lam, "threshold": threshold, "selected_quantile": quantile,
                    "validation_accuracy": va, "validation_cases": vn,
                    "baselines": {
                        "always_up": 1,
                        "prior_gap": 1 if row["gap_return"] >= 0 else -1,
                        "prior_cash": 1 if row["cash_return"] >= 0 else -1,
                        "institution_sign": 1 if row["inst_net_proxy"] >= 0 else -1,
                        "futures_foreign_sign": 1 if row["futures_foreign_net_proxy"] >= 0 else -1,
                        "option_foreign_sign": 1 if row["option_foreign_net_proxy"] >= 0 else -1,
                    },
                    "later_night_prediction": int(row["later_night_prediction"]) if pd.notna(row["later_night_prediction"]) else None,
                })
            windows.append({
                "test_end": str(test["target_date"].max().date()), "lambda": lam,
                "threshold": threshold, "validation_accuracy": va, "validation_cases": vn,
                "test_cases": int(use.sum()),
                "test_accuracy": float((pred[use] == test.loc[use, "target"].to_numpy()).mean()) if use.any() else 0,
            })
        start += TEST
    cases, hits = len(records), sum(row["hit"] for row in records)
    accuracy, coverage = (hits / cases if cases else 0), (cases / tested if tested else 0)
    baseline_results = {}
    baseline_names = list(records[0]["baselines"]) if records else []
    for name in baseline_names:
        bh = sum(row["baselines"][name] == row["target"] for row in records)
        baseline_results[name] = {"hits": bh, "accuracy": bh / cases if cases else 0}
    strongest_name = max(baseline_results, key=lambda name: baseline_results[name]["accuracy"])
    strongest = baseline_results[strongest_name]
    model_only = sum(row["hit"] and row["baselines"][strongest_name] != row["target"] for row in records)
    baseline_only = sum(not row["hit"] and row["baselines"][strongest_name] == row["target"] for row in records)
    p_value = mcnemar(model_only, baseline_only)
    lower = wilson(hits, cases)
    later = [row for row in records if row["later_night_prediction"] is not None]
    later_hits = sum(row["later_night_prediction"] == row["target"] for row in later)
    passed = (
        accuracy >= .90 and cases >= 100 and coverage >= .10 and lower >= .80
        and accuracy > strongest["accuracy"] and model_only > baseline_only and p_value < .05
    )
    return {
        "tested_rows": tested, "cases": cases, "hits": hits, "accuracy": accuracy,
        "coverage": coverage, "wilson_95_lower": lower, "baselines": baseline_results,
        "strongest_baseline": strongest_name, "strongest_baseline_accuracy": strongest["accuracy"],
        "model_only": model_only, "baseline_only": baseline_only, "mcnemar_exact_p": p_value,
        "later_night_benchmark": {"cases": len(later), "hits": later_hits, "accuracy": later_hits / len(later) if later else 0},
        "passed": passed, "windows": windows, "records": records,
    }


def paired_variants(results, left, right):
    a = {row["target_date"]: row for row in results[left]["records"]}
    b = {row["target_date"]: row for row in results[right]["records"]}
    dates = sorted(set(a) & set(b))
    left_only = sum(a[d]["hit"] and not b[d]["hit"] for d in dates)
    right_only = sum(not a[d]["hit"] and b[d]["hit"] for d in dates)
    return {
        "left": left, "right": right, "identical_dates": len(dates),
        "left_accuracy": sum(a[d]["hit"] for d in dates) / len(dates) if dates else 0,
        "right_accuracy": sum(b[d]["hit"] for d in dates) / len(dates) if dates else 0,
        "left_only": left_only, "right_only": right_only,
        "mcnemar_exact_p": mcnemar(left_only, right_only),
    }


def main():
    all_frame = prepare()
    frame = all_frame.loc[all_frame["futures_inst_net_proxy"].notna() & all_frame["option_inst_net_proxy"].notna()].reset_index(drop=True)
    results = {name: evaluate(frame, features) for name, features in VARIANTS.items()}
    paired = [
        paired_variants(results, "derivatives_only", "cash_margin_only_same_period"),
        paired_variants(results, "combined", "derivatives_only"),
        paired_variants(results, "combined", "cash_margin_only_same_period"),
    ]
    payload = {
        "scope": "At TWII close D, predict D+1 cash opening gap before D+1 night price is known.",
        "hypothesis": "D-day futures/options positioning provides next-gap information independent of cash/margin state.",
        "method": "Same 2018+ dates for three registered variants; 756/126/126 nested rolling walk-forward; no option VIX due only 98 dates; identical-date causal simple baselines and paired ablations.",
        "gate": ">=90% accuracy, >=100 cases, >=10% coverage, Wilson lower >=80%, and significant paired superiority over strongest simple baseline.",
        "aligned_rows": len(frame), "variants": {name: {"features": VARIANTS[name], "result": result} for name, result in results.items()},
        "paired_variant_audits": paired,
        "passing": [name for name, result in results.items() if result["passed"]],
    }
    out = ROOT / "reports"
    (out / "after_close_derivative_ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# After-close derivatives next-gap ablation", "", payload["scope"], "",
        payload["hypothesis"], "", payload["method"], "", payload["gate"], "",
        "| Variant | Cases | Accuracy | Coverage | Wilson lower | Strongest baseline | Paired model/base only | p | Passed |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['cases']} | {result['accuracy']:.2%} | {result['coverage']:.2%} | "
            f"{result['wilson_95_lower']:.2%} | {result['strongest_baseline']} {result['strongest_baseline_accuracy']:.2%} | "
            f"{result['model_only']}/{result['baseline_only']} | {result['mcnemar_exact_p']:.4g} | {result['passed']} |"
        )
    lines += ["", "## Paired variant ablations", "", "| Left | Right | Dates | Accuracy left/right | Left/right only | p |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in paired:
        lines.append(f"| {row['left']} | {row['right']} | {row['identical_dates']} | {row['left_accuracy']:.2%}/{row['right_accuracy']:.2%} | {row['left_only']}/{row['right_only']} | {row['mcnemar_exact_p']:.4g} |")
    lines += ["", "Every selected OOS prediction and frozen window configuration is retained in the JSON companion file."]
    (out / "after_close_derivative_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "aligned_rows": len(frame), "passing": payload["passing"],
        "variants": {name: {key: value for key, value in result.items() if key not in ["records", "windows", "baselines", "later_night_benchmark"]} for name, result in results.items()},
        "paired": paired,
    }, indent=2))


if __name__ == "__main__":
    main()
