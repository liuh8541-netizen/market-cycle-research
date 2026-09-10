"""Strict OOS test of currency-adjusted EWT overnight price discovery."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HISTORY, VALID, TEST = 756, 126, 126
WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
QUANTILES = np.linspace(0, .9, 19)


def asof_return(dates, external, close_col, output_col):
    source = external.loc[external[close_col].notna(), ["date", close_col]].copy().sort_values("date")
    source[output_col] = pd.to_numeric(source[close_col], errors="coerce").pct_change()
    source = source.dropna(subset=[output_col]).rename(columns={"date": output_col + "_date"})
    return pd.merge_asof(
        dates.sort_values("date"), source[[output_col + "_date", output_col]].sort_values(output_col + "_date"),
        left_on="date", right_on=output_col + "_date", direction="backward",
        allow_exact_matches=False, tolerance=pd.Timedelta("5D"),
    )


def prepare():
    cash = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv", usecols=["date", "open", "close"])
    cash["date"] = pd.to_datetime(cash["date"])
    cash["open"] = pd.to_numeric(cash["open"], errors="coerce")
    cash["close"] = pd.to_numeric(cash["close"], errors="coerce")
    cash = cash.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    cash["prior_close"] = cash["close"].shift(1)
    cash["gap_return"] = cash["open"] / cash["prior_close"] - 1
    cash["cash_close_return"] = cash["close"] / cash["prior_close"] - 1

    external = pd.read_csv(ROOT / "data" / "processed" / "external_markets.csv")
    external["date"] = pd.to_datetime(external["date"])
    base = cash[["date", "gap_return", "cash_close_return"]].copy()
    for close_col, output_col in [
        ("ewt_close", "ewt_return"), ("usd_twd_close", "usd_twd_return"),
        ("tsm_adr_close", "tsm_return"), ("sp500_close", "sp500_return"),
        ("nasdaq_close", "nasdaq_return"), ("sox_close", "sox_return"),
        ("dow_close", "dow_return"),
    ]:
        feature = asof_return(base[["date"]].copy(), external, close_col, output_col)
        base = base.merge(feature[["date", output_col + "_date", output_col]], on="date", how="left")

    # EWT is priced in USD while its underlying assets are TWD. Approximate the
    # local-currency Taiwan equity move as EWT_USD return + USD/TWD return.
    base["ewt_fx_adjusted_return"] = base["ewt_return"] + base["usd_twd_return"]
    votes = base[["sp500_return", "nasdaq_return", "sox_return", "dow_return", "tsm_return"]]
    base["external_majority"] = np.where(np.sign(votes.fillna(0)).sum(axis=1) >= 0, 1, -1)

    night = pd.read_csv(ROOT / "data" / "processed" / "taiwan_futures_night.csv")
    night["date"] = pd.to_datetime(night["signal_date"])
    for col in ["tx_night_spread_per", "tx_night_return"]:
        night[col] = pd.to_numeric(night[col], errors="coerce")
    base = base.merge(night[["date", "tx_night_spread_per", "tx_night_return"]], on="date", how="inner")
    base = base.dropna(subset=["gap_return", "cash_close_return", "ewt_fx_adjusted_return", "tx_night_spread_per"])
    base["gap_target"] = np.where(base["gap_return"] >= 0, 1, -1)
    base["close_target"] = np.where(base["cash_close_return"] >= 0, 1, -1)
    base["ewt_prediction"] = np.where(base["ewt_fx_adjusted_return"] >= 0, 1, -1)
    base["night_prediction"] = np.where(base["tx_night_spread_per"] >= 0, 1, -1)
    base["night_return_prediction"] = np.where(base["tx_night_return"] >= 0, 1, -1)
    return base.reset_index(drop=True)


def scaled_score(frame, night_scale, ewt_scale, weight):
    night = frame["tx_night_spread_per"].to_numpy(float) / night_scale
    ewt = frame["ewt_fx_adjusted_return"].to_numpy(float) / ewt_scale
    return night + weight * ewt


def select_config(history, target_col):
    train, valid = history.iloc[:-VALID], history.iloc[-VALID:]
    night_scale = max(float(train["tx_night_spread_per"].abs().median()), 1e-9)
    ewt_scale = max(float(train["ewt_fx_adjusted_return"].abs().median()), 1e-9)
    actual = valid[target_col].to_numpy(int)
    best = None
    for weight in WEIGHTS:
        train_conf = np.abs(scaled_score(train, night_scale, ewt_scale, weight))
        score = scaled_score(valid, night_scale, ewt_scale, weight)
        pred, confidence = np.where(score >= 0, 1, -1), np.abs(score)
        for quantile in QUANTILES:
            threshold = float(np.quantile(train_conf, quantile))
            use = confidence >= threshold
            cases = int(use.sum())
            if cases < 15 or cases / len(valid) < .10:
                continue
            accuracy = float((pred[use] == actual[use]).mean())
            rank = (accuracy, cases / len(valid), -weight, -quantile)
            if best is None or rank > best[0]:
                best = (rank, weight, float(quantile), accuracy, cases)
    return best


def wilson(hits, cases):
    if not cases:
        return 0.0
    z = 1.959963984540054
    p = hits / cases
    den = 1 + z*z/cases
    return (p + z*z/(2*cases) - z*math.sqrt((p*(1-p)+z*z/(4*cases))/cases))/den


def mcnemar(a_only, b_only):
    n = a_only + b_only
    if not n:
        return 1.0
    low = min(a_only, b_only)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(low + 1)) / (2**n))


def evaluate(frame, target_col):
    records, windows, tested = [], [], 0
    start = 0
    while start + HISTORY + TEST <= len(frame):
        history = frame.iloc[start:start + HISTORY]
        test = frame.iloc[start + HISTORY:start + HISTORY + TEST]
        tested += len(test)
        config = select_config(history, target_col)
        if config:
            _, weight, quantile, va, vn = config
            night_scale = max(float(history["tx_night_spread_per"].abs().median()), 1e-9)
            ewt_scale = max(float(history["ewt_fx_adjusted_return"].abs().median()), 1e-9)
            history_conf = np.abs(scaled_score(history, night_scale, ewt_scale, weight))
            threshold = float(np.quantile(history_conf, quantile))
            score = scaled_score(test, night_scale, ewt_scale, weight)
            pred, confidence = np.where(score >= 0, 1, -1), np.abs(score)
            use = confidence >= threshold
            for offset in np.flatnonzero(use):
                row = test.iloc[offset]
                target = int(row[target_col])
                records.append({
                    "date": str(row["date"].date()), "prediction": int(pred[offset]),
                    "target": target, "hit": int(pred[offset] == target),
                    "weight": weight, "quantile": quantile, "threshold": threshold,
                    "validation_accuracy": va, "validation_cases": vn,
                    "ewt_fx_adjusted_return": float(row["ewt_fx_adjusted_return"]),
                    "night_spread": float(row["tx_night_spread_per"]),
                    "baselines": {
                        "night_spread": int(row["night_prediction"]),
                        "night_return": int(row["night_return_prediction"]),
                        "ewt_fx": int(row["ewt_prediction"]),
                        "external_majority": int(row["external_majority"]),
                        "always_up": 1,
                    },
                })
            cases = int(use.sum())
            hits = int((pred[use] == test.loc[use, target_col].to_numpy()).sum())
            windows.append({
                "test_end": str(test["date"].max().date()), "weight": weight,
                "quantile": quantile, "threshold": threshold, "validation_accuracy": va,
                "validation_cases": vn, "test_cases": cases,
                "test_accuracy": hits / cases if cases else 0,
            })
        start += TEST
    cases, hits = len(records), sum(row["hit"] for row in records)
    accuracy, coverage = (hits / cases if cases else 0), (cases / tested if tested else 0)
    baselines = {}
    for name in ["night_spread", "night_return", "ewt_fx", "external_majority", "always_up"]:
        bh = sum(row["baselines"][name] == row["target"] for row in records)
        baselines[name] = {"hits": bh, "accuracy": bh / cases if cases else 0}
    strongest_name = max(baselines, key=lambda name: baselines[name]["accuracy"])
    strongest = baselines[strongest_name]
    model_only = sum(row["hit"] and row["baselines"][strongest_name] != row["target"] for row in records)
    baseline_only = sum(not row["hit"] and row["baselines"][strongest_name] == row["target"] for row in records)
    p_value = mcnemar(model_only, baseline_only)
    lower = wilson(hits, cases)
    material = [window for window in windows if window["test_cases"] >= 10]
    minimum_block = min((window["test_accuracy"] for window in material), default=0)
    passed = (
        accuracy >= .90 and cases >= 100 and coverage >= .10 and lower >= .80
        and accuracy > strongest["accuracy"] and model_only > baseline_only and p_value < .05
        and minimum_block >= .80
    )
    return {
        "target": target_col, "tested_rows": tested, "cases": cases, "hits": hits,
        "accuracy": accuracy, "coverage": coverage, "wilson_95_lower": lower,
        "baselines": baselines, "strongest_baseline": strongest_name,
        "strongest_baseline_accuracy": strongest["accuracy"],
        "model_only": model_only, "baseline_only": baseline_only,
        "mcnemar_exact_p": p_value, "minimum_material_window_accuracy": minimum_block,
        "passed": passed, "windows": windows, "records": records,
    }


def main():
    frame = prepare()
    results = [evaluate(frame, "gap_target"), evaluate(frame, "close_target")]
    payload = {
        "hypothesis": "Currency-adjusted EWT provides independent overnight Taiwan price discovery that can correct the TX night-spread direction.",
        "causal_alignment": "For Taiwan date D, every US/FX input uses the latest completed external observation with external date strictly earlier than D. Current-day FX is excluded.",
        "method": "756-day rolling history; last 126 days select a low-degree EWT weight and abstention quantile; freeze for following 126-day OOS block; exact identical-date strong-baseline audit.",
        "gate": ">=90% accuracy, >=100 cases, >=10% coverage, Wilson lower >=80%, significant superiority over strongest identical-date baseline, and every material block >=80%.",
        "weights": WEIGHTS, "quantiles": QUANTILES.tolist(), "aligned_rows": len(frame),
        "results": results, "passing": [row["target"] for row in results if row["passed"]],
    }
    out = ROOT / "reports"
    (out / "ewt_fx_transmission.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Currency-adjusted EWT transmission research", "", payload["hypothesis"], "",
        payload["causal_alignment"], "", payload["method"], "", payload["gate"], "",
        "| Target | Cases | Accuracy | Coverage | Wilson lower | Strongest baseline | Model/base only | p | Min block | Passed |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in results:
        lines.append(
            f"| {row['target']} | {row['cases']} | {row['accuracy']:.2%} | {row['coverage']:.2%} | "
            f"{row['wilson_95_lower']:.2%} | {row['strongest_baseline']} {row['strongest_baseline_accuracy']:.2%} | "
            f"{row['model_only']}/{row['baseline_only']} | {row['mcnemar_exact_p']:.4g} | "
            f"{row['minimum_material_window_accuracy']:.2%} | {row['passed']} |"
        )
    for row in results:
        lines += ["", f"## {row['target']} windows", "", "| Test end | EWT weight | Quantile | Cases | Accuracy | Validation cases/accuracy |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for window in row["windows"]:
            lines.append(f"| {window['test_end']} | {window['weight']} | {window['quantile']:.0%} | {window['test_cases']} | {window['test_accuracy']:.2%} | {window['validation_cases']}/{window['validation_accuracy']:.2%} |")
    (out / "ewt_fx_transmission.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "aligned_rows": len(frame), "passing": payload["passing"],
        "results": [{key: value for key, value in row.items() if key not in ["windows", "records", "baselines"]} for row in results],
    }, indent=2))


if __name__ == "__main__":
    main()
