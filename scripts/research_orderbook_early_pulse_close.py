"""Locked OOS ablation of 09:15 market-wide order/trade pressure."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_taiex_early_pulse_close as base


ROOT = Path(__file__).resolve().parents[1]
CONTROL = base.CONTROL + base.PULSE
ORDERBOOK = [
    "order_count_imbalance", "order_volume_imbalance", "order_size_log_ratio",
    "order_count_imbalance_change", "order_volume_imbalance_change",
    "order_growth_imbalance", "volume_growth_imbalance",
    "late_order_growth_imbalance", "late_volume_growth_imbalance",
    "log_deal_order_15m", "log_deal_volume_15m", "log_deal_money_15m",
    "price_x_order_volume", "gap_x_order_count",
]
VARIANTS = {"price_pulse_control": CONTROL, "price_plus_orderbook": CONTROL + ORDERBOOK}
BASELINES = [
    "early_momentum", "early_reversal", "gap_continuation", "night_direction",
    "order_count_direction", "order_volume_direction",
]


def imbalance(left, right):
    denominator = left.abs() + right.abs()
    return (left - right) / denominator.replace(0, np.nan)


def prepare():
    frame = base.prepare()
    orderbook = pd.read_csv(ROOT / "data/processed/factors/orderbook_early_pulse.csv")
    orderbook["date"] = pd.to_datetime(orderbook["date"])
    orderbook = orderbook.sort_values("date").drop_duplicates("date", keep="last")
    numeric = [
        "order_count_imbalance", "order_volume_imbalance", "order_size_log_ratio",
        "order_count_imbalance_change", "order_volume_imbalance_change",
        "buy_order_growth", "sell_order_growth", "buy_volume_growth",
        "sell_volume_growth", "late_buy_order_growth", "late_sell_order_growth",
        "late_buy_volume_growth", "late_sell_volume_growth", "deal_order_15m",
        "deal_volume_15m", "deal_money_15m",
    ]
    orderbook[numeric] = orderbook[numeric].apply(pd.to_numeric, errors="coerce")
    orderbook["order_growth_imbalance"] = imbalance(
        orderbook.buy_order_growth, orderbook.sell_order_growth
    )
    orderbook["volume_growth_imbalance"] = imbalance(
        orderbook.buy_volume_growth, orderbook.sell_volume_growth
    )
    orderbook["late_order_growth_imbalance"] = imbalance(
        orderbook.late_buy_order_growth, orderbook.late_sell_order_growth
    )
    orderbook["late_volume_growth_imbalance"] = imbalance(
        orderbook.late_buy_volume_growth, orderbook.late_sell_volume_growth
    )
    for source in ["deal_order_15m", "deal_volume_15m", "deal_money_15m"]:
        orderbook["log_" + source] = np.log1p(orderbook[source].clip(lower=0))
    frame = frame.merge(orderbook[["date"] + ORDERBOOK[:-2]], on="date", how="inner")
    frame["price_x_order_volume"] = (
        frame["early_return"] * frame["order_volume_imbalance"]
    )
    frame["gap_x_order_count"] = frame["opening_gap"] * frame["order_count_imbalance"]
    frame["order_count_direction"] = np.where(
        frame["order_count_imbalance"] >= 0, 1, -1
    )
    frame["order_volume_direction"] = np.where(
        frame["order_volume_imbalance"] >= 0, 1, -1
    )
    return frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["order_count_imbalance", "order_volume_imbalance"]
    ).sort_values("date").reset_index(drop=True)


def evaluate(frame, features):
    records, windows, tested = [], [], 0
    for start in range(0, len(frame) - base.HISTORY - base.TEST + 1, base.TEST):
        history = frame.iloc[start:start + base.HISTORY]
        test = frame.iloc[start + base.HISTORY:start + base.HISTORY + base.TEST]
        tested += len(test)
        selection = base.choose(history, features)
        if not selection:
            continue
        _, lam, quantile, validation_accuracy, validation_cases = selection
        model = base.fit(history, features, lam)
        prediction, confidence = base.predict(model, test, features)
        cutoff = float(np.quantile(base.predict(model, history, features)[1], quantile))
        use = confidence >= cutoff
        for position in np.flatnonzero(use):
            row = test.iloc[position]
            record = {
                "date": str(row.date.date()),
                "prediction": int(prediction[position]),
                "target": int(row.target),
                "hit": int(prediction[position] == row.target),
                "confidence": float(confidence[position]),
                "lambda": lam,
                "quantile": quantile,
            }
            record.update({name: int(row[name]) for name in BASELINES})
            records.append(record)
        cases = int(use.sum())
        windows.append({
            "test_start": str(test.date.min().date()),
            "test_end": str(test.date.max().date()),
            "cases": cases,
            "accuracy": float((prediction[use] == test.loc[use, "target"]).mean()) if cases else 0,
            "validation_accuracy": validation_accuracy,
            "validation_cases": validation_cases,
        })
    cases, hits = len(records), sum(row["hit"] for row in records)
    baselines = {}
    for name in BASELINES:
        base_hits = sum(row[name] == row["target"] for row in records)
        baselines[name] = {"hits": base_hits, "accuracy": base_hits / cases if cases else 0}
    strongest = max(baselines, key=lambda name: baselines[name]["accuracy"])
    model_only = sum(row["hit"] and row[strongest] != row["target"] for row in records)
    baseline_only = sum(not row["hit"] and row[strongest] == row["target"] for row in records)
    accuracy = hits / cases if cases else 0
    material = [window for window in windows if window["cases"] >= 10]
    result = {
        "tested_rows": tested, "cases": cases, "hits": hits, "accuracy": accuracy,
        "coverage": cases / tested if tested else 0,
        "wilson_95_lower": base.wilson(hits, cases),
        "baselines": baselines, "strongest_baseline": strongest,
        "strongest_baseline_accuracy": baselines[strongest]["accuracy"],
        "model_only": model_only, "baseline_only": baseline_only,
        "mcnemar_exact_p": base.mcnemar(model_only, baseline_only),
        "minimum_material_window_accuracy": min(
            (window["accuracy"] for window in material), default=0
        ),
        "windows": windows, "records": records,
    }
    result["passed"] = (
        cases >= 100 and result["coverage"] >= 0.10 and accuracy >= 0.90
        and result["wilson_95_lower"] >= 0.80
        and accuracy > result["strongest_baseline_accuracy"]
        and model_only > baseline_only and result["mcnemar_exact_p"] < 0.05
        and result["minimum_material_window_accuracy"] >= 0.80
    )
    return result


def main():
    frame = prepare()
    results = {name: evaluate(frame, features) for name, features in VARIANTS.items()}
    comparison = base.paired(
        results["price_pulse_control"], results["price_plus_orderbook"]
    )
    # base.paired labels the right-hand accuracy as pulse_accuracy; here it is
    # the order-book treatment and is renamed in the saved evidence.
    comparison["orderbook_accuracy"] = comparison.pop("pulse_accuracy")
    comparison["orderbook_only"] = comparison.pop("pulse_only")
    increment = (
        comparison["orderbook_only"] > comparison["control_only"]
        and comparison["mcnemar_exact_p"] < 0.05
    )
    results["price_plus_orderbook"]["passed"] &= increment
    payload = {
        "experiment_id": "orderbook_early_pulse_close_v1",
        "status": "strict historical OOS; not production unless every gate passes",
        "target": "same-day TWII close versus observed 09:15 TAIEX",
        "decision_time": "09:15:00 Asia/Taipei",
        "hypothesis": "Market-wide order and trade pressure through 09:15 adds remaining-day direction beyond the observed TAIEX price path.",
        "method": "630/126/126 nested ridge/abstention, fixed 09:15 cutoff, exact-date ablation.",
        "gate": ">=90% accuracy, >=100 cases, >=10% coverage, Wilson >=80%, min block >=80%, significant superiority over strongest simple baseline and price-path control.",
        "aligned_rows": len(frame), "variants": results,
        "orderbook_increment": comparison,
        "orderbook_increment_passed": increment,
        "passing": [name for name, result in results.items() if result["passed"]],
    }
    out = ROOT / "reports"
    (out / "orderbook_early_pulse_close.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Order-book early-pulse remaining-day research", "", payload["status"], "",
        payload["hypothesis"], "",
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
    (out / "orderbook_early_pulse_close.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "aligned_rows": len(frame),
        "variants": {
            name: {key: value for key, value in result.items()
                   if key not in ("records", "windows", "baselines")}
            for name, result in results.items()
        },
        "orderbook_increment": comparison,
    }, indent=2))


if __name__ == "__main__":
    main()
