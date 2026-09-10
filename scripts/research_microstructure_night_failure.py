"""Locked ablation: does the completed intranight path identify night-signal failures?"""

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_night_cash_close_regime_switch as base  # noqa: E402

MICRO = [
    "first_hour_return", "last_hour_return", "return_acceleration",
    "realized_volatility", "trend_efficiency", "up_bar_share",
    "close_location", "max_drawdown", "max_runup",
    "first_hour_volume_share", "last_hour_volume_share",
    "signed_volume_ratio", "log_tick_count",
]


def prepare():
    frame = base.prepare()
    micro = pd.read_csv(ROOT / "data/processed/factors/night_microstructure.csv")
    micro["date"] = pd.to_datetime(micro["signal_date"])
    micro["log_tick_count"] = pd.to_numeric(micro["tick_count"], errors="coerce").clip(lower=0)
    micro["log_tick_count"] = (micro["log_tick_count"] + 1).map(__import__("math").log)
    return frame.merge(micro[["date"] + MICRO], on="date", how="inner").reset_index(drop=True)


def paired(left, right):
    a = {row["date"]: row for row in left["records"]}
    b = {row["date"]: row for row in right["records"]}
    dates = sorted(a.keys() & b.keys())
    left_only = sum(a[d]["hit"] and not b[d]["hit"] for d in dates)
    right_only = sum(b[d]["hit"] and not a[d]["hit"] for d in dates)
    return {
        "identical_dates": len(dates),
        "regime_accuracy": sum(a[d]["hit"] for d in dates) / len(dates) if dates else 0,
        "microstructure_accuracy": sum(b[d]["hit"] for d in dates) / len(dates) if dates else 0,
        "regime_only": left_only,
        "microstructure_only": right_only,
        "mcnemar_exact_p": base.mcnemar(left_only, right_only),
    }


def main():
    frame = prepare()
    original = list(base.FEATURES)
    base.FEATURES = original
    control = base.evaluate(frame)
    base.FEATURES = original + MICRO
    micro = base.evaluate(frame)
    base.FEATURES = original
    comparison = paired(control, micro)
    increment = (
        comparison["microstructure_only"] > comparison["regime_only"]
        and comparison["mcnemar_exact_p"] < 0.05
    )
    micro["passed"] = bool(micro["passed"] and increment)
    payload = {
        "experiment_id": "microstructure_night_failure_v1",
        "status": "registered microstructure-feature experiment; target family was previously studied",
        "hypothesis": "A pathological completed 15-minute TX night path identifies when the otherwise strong night-spread direction will fail to transmit to the same-day TWII cash close.",
        "causal_rule": "Microstructure ends by 05:00 on target date; cash reliability features lag at least one completed session; overseas data are strictly earlier than target date.",
        "method": "Exact-date base-regime versus base-plus-path ablation using the same 630/126/126 balanced failure-classifier walk-forward and frozen flip/abstention rules.",
        "gate": ">=90% accuracy, >=100 cases, >=10% coverage, Wilson lower >=80%, minimum block >=80%, significant superiority over strongest simple baseline and base-regime control.",
        "aligned_rows": len(frame),
        "variants": {
            "regime_control": {"features": original, "result": control},
            "regime_plus_microstructure": {"features": original + MICRO, "result": micro},
        },
        "path_increment": comparison,
        "path_increment_passed": increment,
        "passing": ["regime_plus_microstructure"] if micro["passed"] else [],
    }
    out = ROOT / "reports"
    (out / "microstructure_night_failure.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Microstructure night-failure research", "", payload["status"], "",
        payload["hypothesis"], "", payload["causal_rule"], "",
        "| Variant | Cases | Accuracy | Coverage | Wilson lower | Strongest baseline | Model/base only | Minimum block | Passed |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for name, result in [("regime_control", control), ("regime_plus_microstructure", micro)]:
        lines.append(
            f"| {name} | {result['cases']} | {result['accuracy']:.2%} | "
            f"{result['coverage']:.2%} | {result['wilson_95_lower']:.2%} | "
            f"{result['strongest_baseline']} {result['strongest_baseline_accuracy']:.2%} | "
            f"{result['model_only']}/{result['baseline_only']} (p={result['mcnemar_exact_p']:.4g}) | "
            f"{result['minimum_material_window_accuracy']:.2%} | {result['passed']} |"
        )
    lines += [
        "",
        f"Common dates: {comparison['identical_dates']}; control/path accuracy "
        f"{comparison['regime_accuracy']:.2%}/{comparison['microstructure_accuracy']:.2%}.",
        f"Control-only/path-only wins: {comparison['regime_only']}/"
        f"{comparison['microstructure_only']}; p={comparison['mcnemar_exact_p']:.4g}.",
        f"Independent path increment passed: {increment}.",
    ]
    (out / "microstructure_night_failure.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "aligned_rows": len(frame),
        "control": {k: v for k, v in control.items() if k not in ("records", "windows", "baselines")},
        "microstructure": {k: v for k, v in micro.items() if k not in ("records", "windows", "baselines")},
        "path_increment": comparison,
    }, indent=2))


if __name__ == "__main__":
    main()
