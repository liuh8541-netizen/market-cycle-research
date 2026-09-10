"""Locked OOS ablation of 09:15 core-sector breadth for remaining-day direction."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_orderbook_early_pulse_close as direction


ROOT = Path(__file__).resolve().parents[1]
CONTROL = direction.CONTROL + direction.ORDERBOOK
SECTOR = [
    "twse_electronic_return", "twse_electronic_last5_return",
    "tpex_electronic_return", "tpex_electronic_last5_return",
    "twse_finance_return", "twse_finance_last5_return",
    "tpex_index_return", "tpex_index_last5_return",
    "sector_mean_return", "sector_dispersion", "sector_breadth",
    "sector_last5_breadth", "electronic_finance_rotation",
    "tpex_twse_electronic_rotation", "tpex_taiex_rotation",
]
VARIANTS = {"price_orderbook_control": CONTROL, "plus_sector_breadth": CONTROL + SECTOR}
BASELINES = [
    "early_momentum", "early_reversal", "gap_continuation", "night_direction",
    "order_count_direction", "order_volume_direction", "sector_breadth_direction",
    "twse_electronic_direction", "twse_finance_direction", "tpex_direction",
]


def prepare():
    frame = direction.prepare()
    sector = pd.read_csv(ROOT / "data/processed/factors/sector_early_pulse.csv")
    sector["date"] = pd.to_datetime(sector.date)
    sector = sector.sort_values("date").drop_duplicates("date", keep="last")
    numeric = [
        column for column in sector.columns
        if column not in {"date", "first_timestamp", "cutoff_timestamp"}
        and not column.endswith("_observations")
    ]
    sector[numeric] = sector[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.merge(sector[["date"] + numeric], on="date", how="inner")
    frame["tpex_taiex_rotation"] = frame["tpex_index_return"] - frame["early_return"]
    frame["sector_breadth_direction"] = np.where(frame.sector_mean_return >= 0, 1, -1)
    frame["twse_electronic_direction"] = np.where(
        frame.twse_electronic_return >= 0, 1, -1
    )
    frame["twse_finance_direction"] = np.where(frame.twse_finance_return >= 0, 1, -1)
    frame["tpex_direction"] = np.where(frame.tpex_index_return >= 0, 1, -1)
    return frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=SECTOR
    ).sort_values("date").reset_index(drop=True)


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
        results["price_orderbook_control"], results["plus_sector_breadth"]
    )
    comparison["sector_accuracy"] = comparison.pop("pulse_accuracy")
    comparison["sector_only"] = comparison.pop("pulse_only")
    increment = (
        comparison["sector_only"] > comparison["control_only"]
        and comparison["mcnemar_exact_p"] < 0.05
    )
    results["plus_sector_breadth"]["passed"] &= increment
    payload = {
        "experiment_id": "sector_early_pulse_close_v1",
        "status": "strict historical OOS; not production unless every gate passes",
        "target": "same-day TWII close versus observed 09:15 TAIEX",
        "decision_time": "09:15:00 Asia/Taipei",
        "hypothesis": "Core TWSE/TPEx sector breadth, dispersion and rotation through 09:15 adds remaining-day direction beyond aggregate price and order-book pressure.",
        "method": "630/126/126 nested ridge/abstention, fixed 09:15 cutoff, exact-date ablation.",
        "gate": ">=90% accuracy, >=100 cases, >=10% coverage, Wilson >=80%, min block >=80%, significant superiority over strongest simple baseline and aggregate control.",
        "aligned_rows": len(frame), "variants": results,
        "sector_increment": comparison, "sector_increment_passed": increment,
        "passing": [name for name, result in results.items() if result["passed"]],
    }
    out = ROOT / "reports"
    (out / "sector_early_pulse_close.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Sector early-pulse remaining-day research", "", payload["status"], "",
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
        f"Common control/sector: {comparison['control_accuracy']:.2%}/"
        f"{comparison['sector_accuracy']:.2%}; exclusive wins "
        f"{comparison['control_only']}/{comparison['sector_only']}; "
        f"p={comparison['mcnemar_exact_p']:.4g}.",
        f"Independent sector increment passed: {increment}.",
    ]
    (out / "sector_early_pulse_close.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "aligned_rows": len(frame),
        "variants": {
            name: {key: value for key, value in result.items()
                   if key not in ("records", "windows", "baselines")}
            for name, result in results.items()
        },
        "sector_increment": comparison,
    }, indent=2))


if __name__ == "__main__":
    main()
