"""Post-hoc fixed-quantile sensitivity audit for opening-gap acceptance."""

import json
from pathlib import Path

import numpy as np

from research_open_gap_cash_close import HISTORY, TEST, prepare, wilson, mcnemar


ROOT = Path(__file__).resolve().parents[1]
QUANTILES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def evaluate(frame, quantile):
    records, windows, tested = [], [], 0
    start = 0
    while start + HISTORY + TEST <= len(frame):
        history = frame.iloc[start : start + HISTORY]
        test = frame.iloc[start + HISTORY : start + HISTORY + TEST]
        tested += len(test)
        threshold = float(history["confidence"].quantile(quantile))
        use = test["confidence"] >= threshold
        for _, row in test.loc[use].iterrows():
            records.append({
                "date": str(row["date"].date()), "prediction": int(row["prediction"]),
                "target": int(row["target"]), "night_spread": int(row["night_spread_prediction"]),
                "threshold": threshold, "gap_return": float(row["gap_return"]),
            })
        cases = int(use.sum())
        hits = int((test.loc[use, "prediction"] == test.loc[use, "target"]).sum())
        windows.append({
            "test_end": str(test["date"].max().date()), "threshold": threshold,
            "cases": cases, "hits": hits, "accuracy": hits / cases if cases else 0,
        })
        start += TEST
    cases, hits = len(records), sum(row["prediction"] == row["target"] for row in records)
    night_hits = sum(row["night_spread"] == row["target"] for row in records)
    model_only = sum(row["prediction"] == row["target"] and row["night_spread"] != row["target"] for row in records)
    night_only = sum(row["prediction"] != row["target"] and row["night_spread"] == row["target"] for row in records)
    material = [row for row in windows if row["cases"] >= 10]
    return {
        "quantile": quantile, "cases": cases, "hits": hits,
        "accuracy": hits / cases if cases else 0, "coverage": cases / tested if tested else 0,
        "wilson_95_lower": wilson(hits, cases),
        "night_spread_accuracy": night_hits / cases if cases else 0,
        "edge_over_night": (hits - night_hits) / cases if cases else 0,
        "model_only": model_only, "night_only": night_only,
        "mcnemar_exact_p": mcnemar(model_only, night_only),
        "minimum_material_window_accuracy": min((row["accuracy"] for row in material), default=0),
        "numeric_gate_without_significance": hits / cases >= .90 and cases >= 100 and cases / tested >= .10 and wilson(hits, cases) >= .80 and hits > night_hits,
        "full_gate_with_paired_significance": hits / cases >= .90 and cases >= 100 and cases / tested >= .10 and wilson(hits, cases) >= .80 and hits > night_hits and mcnemar(model_only, night_only) < .05 and min((row["accuracy"] for row in material), default=0) >= .80,
        "windows": windows, "records": records,
    }


def main():
    frame = prepare()
    results = [evaluate(frame, quantile) for quantile in QUANTILES]
    payload = {
        "status": "post_hoc_sensitivity_not_promotable_from_inspected_history",
        "scope": "At cash open, predict same-day close versus prior close using opening-gap sign.",
        "method": "For each frozen 126-day test block, threshold is the fixed absolute-gap quantile of the preceding 756 days; no validation optimization.",
        "reason": "This quantile family was examined after the registered adaptive-threshold result was known. It may generate a prospective candidate but cannot retroactively pass.",
        "results": results,
    }
    out = ROOT / "reports"
    (out / "open_gap_cash_close_fixed_quantiles.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Opening-gap cash-close fixed-quantile sensitivity", "", payload["status"], "",
        payload["scope"], "", payload["method"], "", payload["reason"], "",
        "| Quantile | Cases | Accuracy | Coverage | Wilson lower | Night spread | Edge | Model/night only | p | Min material block | Numeric/full gate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results:
        lines.append(
            f"| {row['quantile']:.0%} | {row['cases']} | {row['accuracy']:.2%} | {row['coverage']:.2%} | "
            f"{row['wilson_95_lower']:.2%} | {row['night_spread_accuracy']:.2%} | {row['edge_over_night']:.2%} | "
            f"{row['model_only']}/{row['night_only']} | {row['mcnemar_exact_p']:.4g} | "
            f"{row['minimum_material_window_accuracy']:.2%} | {row['numeric_gate_without_significance']}/{row['full_gate_with_paired_significance']} |"
        )
    (out / "open_gap_cash_close_fixed_quantiles.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps([{key: value for key, value in row.items() if key not in ["windows", "records"]} for row in results], indent=2))


if __name__ == "__main__":
    main()
