"""Locked event study of large cross-sectional breadth sign transitions."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research_cross_sectional_breadth_cycle import prepare


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/locked_breadth_transition_episode_v1.json"
OUTPUT_JSON = ROOT / "reports/breadth_transition_episode.json"
OUTPUT_MD = ROOT / "reports/breadth_transition_episode.md"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def wilson(hits, cases, z=1.959963984540054):
    if not cases:
        return 0.0
    p = hits / cases
    d = 1 + z * z / cases
    c = p + z * z / (2 * cases)
    s = z * math.sqrt(p * (1 - p) / cases + z * z / (4 * cases * cases))
    return (c - s) / d


def mcnemar(a_only, b_only):
    n = a_only + b_only
    if not n:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, j) for j in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def greedy(frame, mask, gap):
    selected = []
    last = -10**9
    for index in np.flatnonzero(mask.to_numpy()):
        position = int(frame.iloc[index]["_twii_position"])
        if position - last >= gap:
            selected.append(index)
            last = position
    return selected


def score(records, prediction_column, baselines, gates):
    cases = len(records)
    truth = records["target"].astype(int)
    prediction = records[prediction_column].astype(int)
    correct = prediction.eq(truth)
    result = {
        "hits": int(correct.sum()),
        "cases": cases,
        "accuracy": float(correct.mean()) if cases else 0.0,
        "wilson_95_lower": wilson(int(correct.sum()), cases),
        "records": [],
        "baselines": {},
    }
    chunks = np.array_split(np.arange(cases), 5)
    result["chronological_blocks"] = [
        {
            "cases": int(len(chunk)),
            "accuracy": float(correct.iloc[chunk].mean()) if len(chunk) else 0.0,
        }
        for chunk in chunks
    ]
    result["minimum_material_block_accuracy"] = min(
        block["accuracy"] for block in result["chronological_blocks"]
        if block["cases"] >= 10
    )
    for name in baselines:
        bp = records[name].astype(int)
        bc = bp.eq(truth)
        model_only = int((correct & ~bc).sum())
        baseline_only = int((~correct & bc).sum())
        result["baselines"][name] = {
            "accuracy": float(bc.mean()),
            "model_only": model_only,
            "baseline_only": baseline_only,
            "mcnemar_exact_p": mcnemar(model_only, baseline_only),
        }
    strongest_name = max(
        result["baselines"], key=lambda key: result["baselines"][key]["accuracy"]
    )
    strongest = result["baselines"][strongest_name]
    result["strongest_baseline"] = strongest_name
    result["passes_absolute_gates"] = (
        result["accuracy"] >= gates["accuracy"]
        and cases >= gates["cases"]
        and result["wilson_95_lower"] >= gates["wilson_95_lower"]
        and result["minimum_material_block_accuracy"] >= gates["block_accuracy"]
    )
    result["beats_strongest_baseline"] = (
        result["accuracy"] > strongest["accuracy"]
        and strongest["mcnemar_exact_p"] < gates["paired_p"]
    )
    for _, row in records.iterrows():
        result["records"].append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "position": int(row["_twii_position"]),
            "price_breadth": float(row["price_advance_decline_breadth"]),
            "institutional_breadth": float(row["inst_combined_breadth"]),
            "prediction": int(row[prediction_column]),
            "target": int(row["target"]),
        })
    return result


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    script_path = ROOT / config["implementation"]["script"]
    actual_hash = sha256(script_path)
    if actual_hash != config["implementation"]["sha256"]:
        raise RuntimeError("Locked implementation hash mismatch")

    frame = prepare().sort_values("_twii_position").reset_index(drop=True)
    frame = frame.loc[
        frame["_twii_position"].ge(config["eligibility"]["minimum_position"])
        & frame["target"].notna()
    ].copy().reset_index(drop=True)
    price = frame["price_advance_decline_breadth"]
    prior_price = price.shift(1)
    institutional = frame["inst_combined_breadth"]
    threshold = config["event"]["absolute_breadth_threshold"]
    event = (
        np.sign(price).ne(np.sign(prior_price))
        & price.abs().ge(threshold)
        & prior_price.abs().ge(threshold)
    )
    confirmed = (
        event
        & np.sign(institutional).eq(np.sign(price))
        & institutional.abs().ge(config["event"]["institutional_threshold"])
    )
    gap = config["event"]["minimum_origin_spacing"]
    base_indices = greedy(frame, event, gap)
    confirmed_indices = greedy(frame, confirmed, gap)

    frame["price_transition"] = np.where(price.ge(0), 1, -1)
    frame["institutional_override"] = np.where(
        institutional.abs().ge(config["event"]["institutional_threshold"]),
        np.where(institutional.ge(0), 1, -1),
        frame["price_transition"],
    )
    frame["momentum_20"] = np.where(frame["ret_20"].ge(0), 1, -1)
    frame["reversal_20"] = -frame["momentum_20"]
    frame["trend_120"] = np.where(frame["ret_120"].ge(0), 1, -1)
    prior_up = frame["target"].shift(20).rolling(252, min_periods=126).mean()
    frame["prior_majority"] = np.where(prior_up.ge(0), 1, -1)

    baseline_names = config["baselines"]
    variants = {
        "price_transition": score(
            frame.iloc[base_indices].copy(), "price_transition",
            baseline_names, config["gates"],
        ),
        "institutional_override": score(
            frame.iloc[base_indices].copy(), "institutional_override",
            baseline_names, config["gates"],
        ),
        "institutional_confirmation": score(
            frame.iloc[confirmed_indices].copy(), "price_transition",
            baseline_names, config["gates"],
        ),
    }
    denominator = len(base_indices)
    for name, result in variants.items():
        result["coverage"] = result["cases"] / denominator if denominator else 0.0
        result["passed"] = (
            result["passes_absolute_gates"]
            and result["coverage"] >= config["gates"]["coverage"]
            and result["beats_strongest_baseline"]
        )

    control = variants["price_transition"]
    treatment = variants["institutional_override"]
    control_correct = {
        record["date"]: record["prediction"] == record["target"]
        for record in control["records"]
    }
    treatment_correct = {
        record["date"]: record["prediction"] == record["target"]
        for record in treatment["records"]
    }
    treatment_only = sum(
        treatment_correct[date] and not control_correct[date]
        for date in control_correct
    )
    control_only = sum(
        control_correct[date] and not treatment_correct[date]
        for date in control_correct
    )
    increment = {
        "identical_dates": len(control_correct),
        "treatment_only": treatment_only,
        "control_only": control_only,
        "mcnemar_exact_p": mcnemar(treatment_only, control_only),
        "supported": (
            treatment["accuracy"] > control["accuracy"]
            and mcnemar(treatment_only, control_only) < config["gates"]["paired_p"]
        ),
    }
    for result in variants.values():
        result["records"] = result["records"]
    output = {
        "experiment": config["experiment"],
        "locked_implementation_hash": actual_hash,
        "eligible_rows": len(frame),
        "raw_event_cases": int(event.sum()),
        "raw_confirmed_cases": int(confirmed.sum()),
        "nonoverlap_event_cases": len(base_indices),
        "nonoverlap_confirmed_cases": len(confirmed_indices),
        "variants": variants,
        "institutional_increment": increment,
        "passing_variants": [
            name for name, result in variants.items() if result["passed"]
        ],
    }
    OUTPUT_JSON.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Breadth transition episode v1",
        "",
        f"- Eligible rows: {len(frame)}",
        f"- Raw events / confirmed: {int(event.sum())} / {int(confirmed.sum())}",
        f"- Non-overlap events / confirmed: {len(base_indices)} / {len(confirmed_indices)}",
        "",
    ]
    for name, result in variants.items():
        lines += [
            f"## {name}",
            "",
            f"- Accuracy: {result['hits']}/{result['cases']} = {result['accuracy']:.2%}",
            f"- Coverage: {result['coverage']:.2%}",
            f"- Wilson 95% lower: {result['wilson_95_lower']:.2%}",
            f"- Minimum material block: {result['minimum_material_block_accuracy']:.2%}",
            f"- Strongest baseline: {result['strongest_baseline']} "
            f"({result['baselines'][result['strongest_baseline']]['accuracy']:.2%})",
            f"- Passed: {result['passed']}",
            "",
        ]
    lines += [
        "## Institutional increment",
        "",
        f"- Treatment-only / control-only: {treatment_only} / {control_only}",
        f"- Exact McNemar p: {increment['mcnemar_exact_p']:.6g}",
        f"- Supported: {increment['supported']}",
        "",
        f"Passing variants: {output['passing_variants']}",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
