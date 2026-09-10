"""Audit all after-hours candidates against night-price baselines on exact dates."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def mcnemar(left_only, right_only):
    n = left_only + right_only
    if not n:
        return 1.0
    low = min(left_only, right_only)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(low + 1)) / 2**n)


def load_variants():
    candidate = json.loads((ROOT / "reports" / "afterhours_candidate_audit.json").read_text(encoding="utf-8"))
    ablation = json.loads((ROOT / "reports" / "afterhours_amount_ablation.json").read_text(encoding="utf-8"))
    variants = {"pulse_complexion_candidate": candidate["records"]}
    variants.update({f"amount_{name}": value["records"] for name, value in ablation["variants"].items()})
    return variants


def audit(rows, night):
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.merge(night, on="date", how="left")
    frame = frame.dropna(subset=["model", "target", "night_spread", "night_return"])
    records = []
    for row in frame.itertuples():
        records.append({
            "date": str(row.date.date()), "target": int(row.target), "model": int(row.model),
            "night_spread": int(row.night_spread), "night_return": int(row.night_return),
        })
    cases = len(records)
    model_hits = sum(row["model"] == row["target"] for row in records)
    baselines = {}
    for name in ["night_spread", "night_return"]:
        hits = sum(row[name] == row["target"] for row in records)
        model_only = sum(row["model"] == row["target"] and row[name] != row["target"] for row in records)
        baseline_only = sum(row["model"] != row["target"] and row[name] == row["target"] for row in records)
        baselines[name] = {
            "hits": hits, "accuracy": hits / cases if cases else 0,
            "model_only": model_only, "baseline_only": baseline_only,
            "mcnemar_exact_p": mcnemar(model_only, baseline_only),
        }
    strongest_name = max(baselines, key=lambda name: baselines[name]["accuracy"])
    strongest = baselines[strongest_name]
    return {
        "cases": cases, "model_hits": model_hits,
        "model_accuracy": model_hits / cases if cases else 0,
        "baselines": baselines, "strongest_baseline": strongest_name,
        "strongest_baseline_accuracy": strongest["accuracy"],
        "edge": model_hits / cases - strongest["accuracy"] if cases else 0,
        "model_only": strongest["model_only"], "baseline_only": strongest["baseline_only"],
        "mcnemar_exact_p": strongest["mcnemar_exact_p"],
        "beats_strongest_baseline": model_hits / cases > strongest["accuracy"] if cases else False,
        "records": records,
    }


def main():
    night = pd.read_csv(ROOT / "data" / "processed" / "taiwan_futures_night.csv")
    night["date"] = pd.to_datetime(night["signal_date"])
    night["night_spread"] = np.where(pd.to_numeric(night["tx_night_spread_per"], errors="coerce") >= 0, 1, -1)
    night["night_return"] = np.where(pd.to_numeric(night["tx_night_return"], errors="coerce") >= 0, 1, -1)
    night = night[["date", "night_spread", "night_return"]].drop_duplicates("date", keep="last")
    results = {name: audit(rows, night) for name, rows in load_variants().items()}
    payload = {
        "status": "retrospective strong-baseline audit of previously inspected candidates",
        "question": "Did any FinMind after-hours candidate add direction beyond simple night price on its exact selected dates?",
        "method": "Join every retained OOS candidate record to corrected signal_date night spread and night return; exact-date paired McNemar audit.",
        "results": results,
        "passing": [name for name, row in results.items() if row["beats_strongest_baseline"] and row["mcnemar_exact_p"] < .05],
    }
    out = ROOT / "reports"
    (out / "afterhours_night_strong_baseline_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# After-hours candidates versus night-price strong baseline", "", payload["status"], "",
        payload["question"], "", payload["method"], "",
        "| Candidate | Cases | Model | Strongest night baseline | Edge | Model/base only | p | Beats baseline |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for name, row in results.items():
        lines.append(
            f"| {name} | {row['cases']} | {row['model_accuracy']:.2%} | "
            f"{row['strongest_baseline']} {row['strongest_baseline_accuracy']:.2%} | "
            f"{row['edge']:.2%} | {row['model_only']}/{row['baseline_only']} | "
            f"{row['mcnemar_exact_p']:.4g} | {row['beats_strongest_baseline']} |"
        )
    lines += [
        "", "A high headline accuracy is not an independent FinMind effect when the same-date night-price rule is equal or better.",
    ]
    (out / "afterhours_night_strong_baseline_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "passing": payload["passing"],
        "results": {
            name: {key: value for key, value in row.items() if key not in ["records", "baselines"]}
            for name, row in results.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
