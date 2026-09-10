"""Locked purged OOS test of point-in-time stock breadth for 20-day TWII direction."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_three_layer_purged import build_frame, fit_model, predict


HORIZON = 20
HISTORY = 1512
VALIDATION = 252
TEST = 252
LAMBDAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
QUANTILES = [0.0, 0.25, 0.5, 0.65, 0.75, 0.85]

CONTROL = [
    "ret_5", "ret_20", "ret_60", "ret_120", "ret_240",
    "vol_ratio", "drawdown_120", "pulse_rate",
    "inst_foreign_net_20d_z", "inst_trust_net_20d_z",
    "inst_dealer_net_20d_z",
]
PRICE_BREADTH = [
    "price_advancing_fraction", "price_declining_fraction",
    "price_unchanged_fraction", "price_advance_decline_breadth",
    "price_equal_weight_return", "price_median_return",
    "price_return_dispersion", "price_upper_tail_fraction",
    "price_lower_tail_fraction", "price_up_volume_fraction",
    "price_volume_weighted_return", "price_money_top10_share",
    "price_median_log_money",
]
INSTITUTIONAL_BREADTH = [
    "inst_foreign_positive_fraction", "inst_foreign_negative_fraction",
    "inst_foreign_median_intensity", "inst_foreign_aggregate_intensity",
    "inst_trust_positive_fraction", "inst_trust_negative_fraction",
    "inst_trust_median_intensity", "inst_trust_aggregate_intensity",
    "inst_dealer_positive_fraction", "inst_dealer_negative_fraction",
    "inst_dealer_median_intensity", "inst_dealer_aggregate_intensity",
    "inst_combined_positive_fraction", "inst_combined_negative_fraction",
    "inst_combined_median_intensity", "inst_combined_aggregate_intensity",
    "inst_absolute_net_top10_share", "inst_foreign_trust_disagreement",
    "inst_combined_breadth",
]
VARIANTS = {
    "index_aggregate_control": CONTROL,
    "plus_price_breadth": CONTROL + PRICE_BREADTH,
    "plus_price_and_institutional_breadth": (
        CONTROL + PRICE_BREADTH + INSTITUTIONAL_BREADTH
    ),
}
BASELINES = [
    "prior_majority", "momentum_20", "reversal_20", "trend_120",
    "price_breadth_sign", "institutional_breadth_sign",
]


def wilson(hits, cases, z=1.959963984540054):
    if not cases:
        return 0.0
    proportion = hits / cases
    denominator = 1 + z * z / cases
    center = proportion + z * z / (2 * cases)
    spread = z * math.sqrt(
        proportion * (1 - proportion) / cases + z * z / (4 * cases * cases)
    )
    return (center - spread) / denominator


def mcnemar(model_only, baseline_only):
    discordant = model_only + baseline_only
    if not discordant:
        return 1.0
    smaller = min(model_only, baseline_only)
    tail = sum(
        math.comb(discordant, index) for index in range(smaller + 1)
    ) / (2 ** discordant)
    return min(1.0, 2 * tail)


def prepare():
    base, _ = build_frame()
    manifest = pd.read_csv(
        ROOT / "data/processed/factors/cross_sectional_breadth_fetched_dates.csv"
    )
    cash_dates = pd.to_datetime(
        manifest.loc[
            pd.to_numeric(
                manifest["price_source_rows"], errors="coerce"
            ).fillna(0).gt(0),
            "date",
        ],
        errors="raise",
    )
    # Canonical TWII can contain a copied OHLCV placeholder on a cash-market
    # closure. FinMind price presence defines the actual trading-day clock.
    base = base.loc[pd.to_datetime(base["date"]).isin(set(cash_dates))].copy()
    base = base.sort_values("date").reset_index(drop=True)
    base["_twii_position"] = np.arange(len(base))
    future = base["close"].shift(-HORIZON) / base["close"] - 1
    base["target"] = np.where(future > 0, 1, -1).astype(float)
    base.loc[future.isna(), "target"] = np.nan
    breadth = pd.read_csv(
        ROOT / "data/processed/factors/cross_sectional_breadth.csv"
    )
    breadth["date"] = pd.to_datetime(breadth["date"], errors="raise")
    breadth = breadth.sort_values("date").drop_duplicates("date", keep="last")
    frame = base.merge(breadth, on="date", how="inner").sort_values("date")
    frame["momentum_20"] = np.where(frame["ret_20"] >= 0, 1, -1)
    frame["reversal_20"] = -frame["momentum_20"]
    frame["trend_120"] = np.where(frame["ret_120"] >= 0, 1, -1)
    frame["price_breadth_sign"] = np.where(
        frame["price_advance_decline_breadth"] >= 0, 1, -1
    )
    frame["institutional_breadth_sign"] = np.where(
        frame["inst_combined_breadth"] >= 0, 1, -1
    )
    required = sorted(set(sum(VARIANTS.values(), [])))
    for column in required:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    counts_ok = (
        pd.to_numeric(frame["price_stock_count"], errors="coerce").ge(100)
        & pd.to_numeric(frame["inst_stock_count"], errors="coerce").ge(50)
    )
    return frame.loc[counts_ok].replace(
        [np.inf, -np.inf], np.nan
    ).reset_index(drop=True)


def select_config(history, features):
    split = len(history) - VALIDATION
    fit = history.iloc[:split - HORIZON]
    validation = history.iloc[split:len(history) - HORIZON]
    if len(fit) < 500 or len(validation) < 100:
        return None
    best = None
    for lam in LAMBDAS:
        model = fit_model(fit, features, lam)
        prediction, confidence = predict(model, validation, features)
        target = validation["target"].to_numpy(int)
        for quantile in QUANTILES:
            threshold = float(np.quantile(confidence, quantile))
            use = confidence >= threshold
            cases = int(use.sum())
            if cases < 25 or cases / len(validation) < 0.10:
                continue
            accuracy = float((prediction[use] == target[use]).mean())
            rank = (accuracy, cases / len(validation), -lam, -quantile)
            if best is None or rank > best[0]:
                best = (
                    rank, lam, threshold, quantile, accuracy, cases
                )
    return best


def greedy_nonoverlap(records):
    chosen = []
    last_position = -10**12
    for record in sorted(records, key=lambda item: item["position"]):
        if record["position"] - last_position >= HORIZON:
            chosen.append(record)
            last_position = record["position"]
    return chosen


def count_nonoverlap_positions(positions):
    count = 0
    last_position = -10**12
    for position in sorted(set(int(value) for value in positions)):
        if position - last_position >= HORIZON:
            count += 1
            last_position = position
    return count


def evaluate(frame, features):
    daily_records = []
    windows = []
    tested_rows = 0
    tested_positions = []
    start = 0
    while start + HISTORY + TEST + HORIZON <= len(frame):
        history = frame.iloc[start:start + HISTORY]
        test = frame.iloc[start + HISTORY:start + HISTORY + TEST]
        selection = select_config(history, features)
        tested_rows += len(test)
        tested_positions.extend(test["_twii_position"].astype(int).tolist())
        if selection:
            _, lam, threshold, quantile, validation_accuracy, validation_cases = selection
            # Outcomes in the final 20 rows have not completed at the test origin.
            model = fit_model(history.iloc[:-HORIZON], features, lam)
            prediction, confidence = predict(model, test, features)
            use = confidence >= threshold
            block_candidates = []
            prior_resolved = history.iloc[:-HORIZON]["target"].dropna()
            prior_majority = 1 if (prior_resolved == 1).mean() >= 0.5 else -1
            for local_position in np.flatnonzero(use):
                row = test.iloc[local_position]
                if pd.isna(row["target"]):
                    continue
                record = {
                    "date": str(row["date"].date()),
                    "position": int(row["_twii_position"]),
                    "prediction": int(prediction[local_position]),
                    "target": int(row["target"]),
                    "hit": int(prediction[local_position] == row["target"]),
                    "confidence": float(confidence[local_position]),
                    "lambda": lam,
                    "quantile": quantile,
                    "validation_accuracy": validation_accuracy,
                    "validation_cases": validation_cases,
                    "prior_majority": prior_majority,
                }
                for baseline in BASELINES[1:]:
                    record[baseline] = int(row[baseline])
                block_candidates.append(record)
                daily_records.append(record)
            effective = greedy_nonoverlap(block_candidates)
            windows.append({
                "test_start": str(test["date"].min().date()),
                "test_end": str(test["date"].max().date()),
                "daily_selected": len(block_candidates),
                "effective_cases": len(effective),
                "effective_accuracy": (
                    sum(item["hit"] for item in effective) / len(effective)
                    if effective else 0.0
                ),
                "validation_accuracy": validation_accuracy,
                "validation_cases": validation_cases,
            })
        start += TEST
    records = greedy_nonoverlap(daily_records)
    cases = len(records)
    hits = sum(record["hit"] for record in records)
    accuracy = hits / cases if cases else 0.0
    maximum_effective = count_nonoverlap_positions(tested_positions)
    baselines = {}
    for name in BASELINES:
        baseline_hits = sum(
            record[name] == record["target"] for record in records
        )
        baselines[name] = {
            "hits": baseline_hits,
            "accuracy": baseline_hits / cases if cases else 0.0,
        }
    strongest = max(baselines, key=lambda name: baselines[name]["accuracy"])
    model_only = sum(
        record["hit"] and record[strongest] != record["target"]
        for record in records
    )
    baseline_only = sum(
        not record["hit"] and record[strongest] == record["target"]
        for record in records
    )
    material = [
        window for window in windows if window["effective_cases"] >= 10
    ]
    result = {
        "tested_daily_rows": tested_rows,
        "maximum_effective_opportunities": maximum_effective,
        "daily_selected_rows": len(daily_records),
        "cases": cases,
        "hits": hits,
        "accuracy": accuracy,
        "coverage": cases / maximum_effective if maximum_effective else 0.0,
        "wilson_95_lower": wilson(hits, cases),
        "baselines": baselines,
        "strongest_baseline": strongest,
        "strongest_baseline_accuracy": baselines[strongest]["accuracy"],
        "model_only": model_only,
        "baseline_only": baseline_only,
        "mcnemar_exact_p": mcnemar(model_only, baseline_only),
        "minimum_material_window_accuracy": min(
            (window["effective_accuracy"] for window in material), default=0.0
        ),
        "windows": windows,
        "records": records,
    }
    result["passed_base_gate"] = (
        accuracy >= 0.90
        and cases >= 100
        and result["coverage"] >= 0.10
        and result["wilson_95_lower"] >= 0.80
        and result["minimum_material_window_accuracy"] >= 0.80
        and accuracy > result["strongest_baseline_accuracy"]
        and model_only > baseline_only
        and result["mcnemar_exact_p"] < 0.05
    )
    return result


def paired(left, right):
    left_records = {record["date"]: record for record in left["records"]}
    right_records = {record["date"]: record for record in right["records"]}
    dates = sorted(set(left_records).intersection(right_records))
    left_only = right_only = left_hits = right_hits = 0
    for day in dates:
        left_hit = bool(left_records[day]["hit"])
        right_hit = bool(right_records[day]["hit"])
        left_hits += left_hit
        right_hits += right_hit
        left_only += left_hit and not right_hit
        right_only += right_hit and not left_hit
    return {
        "identical_dates": len(dates),
        "left_accuracy": left_hits / len(dates) if dates else 0.0,
        "right_accuracy": right_hits / len(dates) if dates else 0.0,
        "left_only": left_only,
        "right_only": right_only,
        "mcnemar_exact_p": mcnemar(left_only, right_only),
    }


def main():
    frame = prepare()
    results = {
        name: evaluate(frame, features) for name, features in VARIANTS.items()
    }
    price_increment = paired(
        results["index_aggregate_control"], results["plus_price_breadth"]
    )
    institutional_increment = paired(
        results["plus_price_breadth"],
        results["plus_price_and_institutional_breadth"],
    )
    price_increment_passed = (
        price_increment["right_only"] > price_increment["left_only"]
        and price_increment["mcnemar_exact_p"] < 0.05
    )
    institutional_increment_passed = (
        institutional_increment["right_only"]
        > institutional_increment["left_only"]
        and institutional_increment["mcnemar_exact_p"] < 0.05
    )
    results["plus_price_breadth"]["passed"] = (
        results["plus_price_breadth"]["passed_base_gate"]
        and price_increment_passed
    )
    results["plus_price_and_institutional_breadth"]["passed"] = (
        results["plus_price_and_institutional_breadth"]["passed_base_gate"]
        and institutional_increment_passed
    )
    results["index_aggregate_control"]["passed"] = results[
        "index_aggregate_control"
    ]["passed_base_gate"]
    payload = {
        "experiment_id": "cross_sectional_breadth_cycle_v1",
        "status": "strict purged non-overlapping historical OOS",
        "target": "20-trading-day TWII close direction from date-t cash close",
        "decision_time": "after date-t cash and institutional data availability",
        "hypothesis": (
            "Point-in-time individual-stock participation breadth adds "
            "20-day TWII direction beyond index trend and aggregate factors."
        ),
        "method": (
            "1512/252/252 nested ridge-abstention, 20-row purge at every "
            "fit boundary, and effective origins separated by 20 TWII rows."
        ),
        "gate": (
            ">=90% accuracy, >=100 effective cases, >=10% effective coverage, "
            "Wilson >=80%, min material block >=80%, significant superiority "
            "over strongest simple baseline and preceding ablation."
        ),
        "aligned_rows": len(frame),
        "variants": results,
        "price_increment": price_increment,
        "price_increment_passed": price_increment_passed,
        "institutional_increment": institutional_increment,
        "institutional_increment_passed": institutional_increment_passed,
        "passing": [
            name for name, result in results.items() if result["passed"]
        ],
    }
    report_dir = ROOT / "reports"
    (report_dir / "cross_sectional_breadth_cycle.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Cross-sectional breadth cycle research",
        "",
        payload["status"],
        "",
        payload["hypothesis"],
        "",
        "| Variant | Effective cases | Accuracy | Coverage | Wilson lower | Strongest baseline | Model/base only | Min block | Passed |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['cases']} | {result['accuracy']:.2%} | "
            f"{result['coverage']:.2%} | {result['wilson_95_lower']:.2%} | "
            f"{result['strongest_baseline']} "
            f"{result['strongest_baseline_accuracy']:.2%} | "
            f"{result['model_only']}/{result['baseline_only']} "
            f"(p={result['mcnemar_exact_p']:.4g}) | "
            f"{result['minimum_material_window_accuracy']:.2%} | "
            f"{result['passed']} |"
        )
    lines += [
        "",
        f"Price increment common dates: {price_increment['identical_dates']}; "
        f"control/price {price_increment['left_accuracy']:.2%}/"
        f"{price_increment['right_accuracy']:.2%}; exclusive "
        f"{price_increment['left_only']}/{price_increment['right_only']}; "
        f"p={price_increment['mcnemar_exact_p']:.4g}.",
        f"Institutional increment common dates: "
        f"{institutional_increment['identical_dates']}; price/full "
        f"{institutional_increment['left_accuracy']:.2%}/"
        f"{institutional_increment['right_accuracy']:.2%}; exclusive "
        f"{institutional_increment['left_only']}/"
        f"{institutional_increment['right_only']}; "
        f"p={institutional_increment['mcnemar_exact_p']:.4g}.",
    ]
    (report_dir / "cross_sectional_breadth_cycle.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "aligned_rows": len(frame),
        "passing": payload["passing"],
        "variants": {
            name: {
                key: value for key, value in result.items()
                if key not in ("records", "windows", "baselines")
            }
            for name, result in results.items()
        },
        "price_increment": price_increment,
        "institutional_increment": institutional_increment,
    }, indent=2))


if __name__ == "__main__":
    main()
