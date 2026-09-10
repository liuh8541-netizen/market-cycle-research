"""Locked OOS ablation of 09:15 order-book pressure for remaining-day tail risk."""

import json
from pathlib import Path

import numpy as np

import research_orderbook_early_pulse_close as direction


ROOT = Path(__file__).resolve().parents[1]
CONTROL = direction.CONTROL
ORDERBOOK = direction.ORDERBOOK
VARIANTS = {"price_risk_control": CONTROL, "price_plus_orderbook_risk": CONTROL + ORDERBOOK}
BASELINES = [
    "always_normal", "early_range_risk", "early_volatility_risk",
    "opening_gap_risk", "deal_activity_risk",
]


def causal_q75(series):
    return series.rolling(252, min_periods=126).quantile(0.75).shift(1)


def prepare():
    frame = direction.prepare()
    frame["absolute_remaining_return"] = frame["remaining_return"].abs()
    frame["material_threshold"] = causal_q75(frame["absolute_remaining_return"])
    frame["target"] = np.where(
        frame["absolute_remaining_return"] > frame["material_threshold"], 1, -1
    )
    frame["always_normal"] = -1
    frame["early_range_risk"] = np.where(
        frame["early_range"] > causal_q75(frame["early_range"]), 1, -1
    )
    frame["early_volatility_risk"] = np.where(
        frame["realized_volatility"] > causal_q75(frame["realized_volatility"]), 1, -1
    )
    frame["opening_gap_risk"] = np.where(
        frame["opening_gap"].abs() > causal_q75(frame["opening_gap"].abs()), 1, -1
    )
    frame["deal_activity_risk"] = np.where(
        frame["log_deal_volume_15m"] > causal_q75(frame["log_deal_volume_15m"]), 1, -1
    )
    return frame.dropna(subset=["material_threshold"]).reset_index(drop=True)


def main():
    frame = prepare()
    original_baselines = direction.BASELINES
    direction.BASELINES = BASELINES
    try:
        results = {
            name: direction.evaluate(frame, features)
            for name, features in VARIANTS.items()
        }
    finally:
        direction.BASELINES = original_baselines
    comparison = direction.base.paired(
        results["price_risk_control"], results["price_plus_orderbook_risk"]
    )
    comparison["orderbook_accuracy"] = comparison.pop("pulse_accuracy")
    comparison["orderbook_only"] = comparison.pop("pulse_only")
    increment = (
        comparison["orderbook_only"] > comparison["control_only"]
        and comparison["mcnemar_exact_p"] < 0.05
    )
    results["price_plus_orderbook_risk"]["passed"] &= increment
    payload = {
        "experiment_id": "orderbook_remaining_risk_v1",
        "status": "strict historical OOS; not production unless every gate passes",
        "target": "absolute TWII 09:15-to-close return exceeds the causal prior-252-session 75th percentile",
        "decision_time": "09:15:00 Asia/Taipei",
        "hypothesis": "Market-wide order and trade pressure through 09:15 adds remaining-day material-move risk beyond the observed TAIEX price path.",
        "method": "630/126/126 nested ridge/abstention; causal shifted rolling risk threshold; exact-date ablation.",
        "gate": ">=90% accuracy, >=100 cases, >=10% coverage, Wilson >=80%, min block >=80%, significant superiority over strongest simple risk baseline and price-path control.",
        "aligned_rows": len(frame),
        "risk_prevalence": float((frame.target == 1).mean()),
        "variants": results,
        "orderbook_increment": comparison,
        "orderbook_increment_passed": increment,
        "passing": [name for name, result in results.items() if result["passed"]],
    }
    out = ROOT / "reports"
    (out / "orderbook_remaining_risk.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Order-book remaining-day risk research", "", payload["status"], "",
        payload["hypothesis"], "",
        f"Material-move prevalence: {payload['risk_prevalence']:.2%}.", "",
        "| Variant | Cases | Accuracy | Coverage | Wilson lower | Strongest baseline | Model/base only | Min block | Passed |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['cases']} | {result['accuracy']:.2%} | "
            f"{result['coverage']:.2%} | {result['wilson_95_lower']:.2%} | "
            f"{result['strongest_baseline']} {result['strongest_baseline_accuracy']:.2%} | "
            f"{result['model_only']}/{result['baseline_only']} "
            f"(p={result['mcnemar_exact_p']:.4g}) | "
            f"{result['minimum_material_window_accuracy']:.2%} | {result['passed']} |"
        )
    lines += [
        "",
        f"Common control/order-book: {comparison['control_accuracy']:.2%}/"
        f"{comparison['orderbook_accuracy']:.2%}; exclusive wins "
        f"{comparison['control_only']}/{comparison['orderbook_only']}; "
        f"p={comparison['mcnemar_exact_p']:.4g}.",
        f"Independent order-book increment passed: {increment}.",
    ]
    (out / "orderbook_remaining_risk.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "aligned_rows": len(frame),
        "risk_prevalence": payload["risk_prevalence"],
        "variants": {
            name: {key: value for key, value in result.items()
                   if key not in ("records", "windows", "baselines")}
            for name, result in results.items()
        },
        "orderbook_increment": comparison,
    }, indent=2))


if __name__ == "__main__":
    main()
