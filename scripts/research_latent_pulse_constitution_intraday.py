"""Locked nested walk-forward test of pre-open pulse and lagged constitution.

The experiment is intentionally low-degree.  All cash and FinMind fields are
shifted by one completed TWII session; only the completed night session may use
target-date information.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from market_lifecycle.factor_features import (  # noqa: E402
    build_institutional_features,
    build_margin_features,
)

HISTORY, VALID, TEST = 756, 126, 126
LAMBDAS = [0.1, 1.0, 10.0, 100.0]
QUANTILES = [0.0, 0.25, 0.5, 0.65, 0.75, 0.85]

PULSE = [
    "night_spread", "night_return", "night_range", "night_volume_z",
    "prior_cash_1d", "prior_cash_5d", "prior_intraday", "prior_volatility20",
    "first_hour_return", "last_hour_return", "return_acceleration",
    "realized_volatility", "trend_efficiency", "up_bar_share",
    "close_location", "max_drawdown", "max_runup",
    "first_hour_volume_share", "last_hour_volume_share",
    "signed_volume_ratio", "log_tick_count",
]
CONSTITUTION = [
    "inst_net_z", "inst_5d_z", "foreign_net_z", "trust_net_z", "dealer_net_z",
    "margin_change_20d_z", "short_change_20d_z", "margin_short_ratio_z",
]
VARIANTS = {
    "pulse_control": PULSE,
    "pulse_plus_lagged_constitution": PULSE + CONSTITUTION,
}


def causal_z(s: pd.Series, window: int = 252, minimum: int = 80) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mean = s.rolling(window, min_periods=minimum).mean().shift(1)
    std = s.rolling(window, min_periods=minimum).std().shift(1).replace(0, np.nan)
    return (s - mean) / std


def prepare() -> pd.DataFrame:
    cash = pd.read_csv(ROOT / "data/processed/twii_daily.csv")
    cash["date"] = pd.to_datetime(cash["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        cash[col] = pd.to_numeric(cash[col], errors="coerce")
    cash = cash.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    cash["intraday"] = cash["close"] / cash["open"] - 1
    cash["prior_cash_1d"] = cash["close"].pct_change().shift(1)
    cash["prior_cash_5d"] = cash["close"].pct_change(5).shift(1)
    cash["prior_intraday"] = cash["intraday"].shift(1)
    cash["prior_volatility20"] = np.log(cash["close"]).diff().rolling(20).std().shift(1)

    inst = build_institutional_features(
        pd.read_csv(ROOT / "data/processed/factors/institutional_total.csv")
    ).set_index("date")
    margin = build_margin_features(
        pd.read_csv(ROOT / "data/processed/factors/margin_total.csv")
    ).set_index("date")
    factors = inst.join(margin, how="outer").sort_index()
    # Merge onto the cash calendar first, then shift exactly one completed cash
    # row.  This prevents a weekend/holiday calendar shift from becoming D-day.
    factor_cols = list(factors.columns)
    cash = cash.merge(factors.reset_index(), on="date", how="left")
    cash[factor_cols] = cash[factor_cols].shift(1)
    z_sources = {
        "inst_net_z": "inst_net_proxy",
        "inst_5d_z": "inst_net_5d",
        "foreign_net_z": "inst_foreign_net_proxy",
        "trust_net_z": "inst_trust_net_proxy",
        "dealer_net_z": "inst_dealer_net_proxy",
        "margin_change_20d_z": "margin_change_20d",
        "short_change_20d_z": "short_change_20d",
        "margin_short_ratio_z": "margin_short_ratio",
    }
    for out, source in z_sources.items():
        cash[out] = causal_z(cash[source]) if source in cash else np.nan

    night = pd.read_csv(ROOT / "data/processed/taiwan_futures_night.csv")
    night["date"] = pd.to_datetime(night["signal_date"])
    night = night.rename(columns={
        "tx_night_spread_per": "night_spread",
        "tx_night_return": "night_return",
        "tx_night_range": "night_range",
        "tx_night_volume": "night_volume",
    })
    night["night_volume_z"] = causal_z(night["night_volume"])

    micro = pd.read_csv(ROOT / "data/processed/factors/night_microstructure.csv")
    micro["date"] = pd.to_datetime(micro["signal_date"])
    micro["log_tick_count"] = np.log1p(pd.to_numeric(micro["tick_count"], errors="coerce"))

    daily = ["date", "night_spread", "night_return", "night_range", "night_volume_z"]
    path = ["date"] + [c for c in PULSE if c in micro.columns or c == "log_tick_count"]
    # Remove daily/prior fields from the microstructure selection.
    path = ["date"] + [c for c in path if c not in daily and not c.startswith("prior_") and c != "date"]
    frame = cash.merge(night[daily], on="date", how="inner").merge(micro[path], on="date", how="inner")
    frame["target"] = np.where(frame["intraday"] >= 0, 1, -1)
    return frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["intraday", "night_spread", "night_return"]
    ).reset_index(drop=True)


def fit(frame: pd.DataFrame, features: list[str], lam: float):
    raw = frame[features].replace([np.inf, -np.inf], np.nan)
    med = raw.median().fillna(0)
    raw = raw.fillna(med)
    mean, std = raw.mean(), raw.std().replace(0, 1).fillna(1)
    x = np.column_stack([np.ones(len(raw)), ((raw - mean) / std).to_numpy(float)])
    y = frame["target"].to_numpy(float)
    penalty = np.eye(x.shape[1]) * lam
    penalty[0, 0] = 0
    weights = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y
    return weights, med, mean, std


def predict(model, frame: pd.DataFrame, features: list[str]):
    weights, med, mean, std = model
    raw = frame[features].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0)
    x = np.column_stack([np.ones(len(raw)), ((raw - mean) / std).to_numpy(float)])
    score = x @ weights
    return np.where(score >= 0, 1, -1), np.abs(score)


def select(history: pd.DataFrame, features: list[str]):
    train, valid = history.iloc[:-VALID], history.iloc[-VALID:]
    best = None
    for lam in LAMBDAS:
        model = fit(train, features, lam)
        pred, confidence = predict(model, valid, features)
        target = valid["target"].to_numpy(int)
        for quantile in QUANTILES:
            threshold = float(np.quantile(confidence, quantile))
            use = confidence >= threshold
            cases = int(use.sum())
            if cases < 15 or cases / len(valid) < 0.10:
                continue
            accuracy = float((pred[use] == target[use]).mean())
            rank = (accuracy, cases / len(valid), -lam, -quantile)
            if best is None or rank > best[0]:
                best = (rank, lam, threshold, quantile, accuracy, cases)
    return best


def wilson(hits: int, cases: int) -> float:
    if not cases:
        return 0.0
    z, p = 1.959963984540054, hits / cases
    den = 1 + z * z / cases
    return (p + z * z / (2 * cases) - z * math.sqrt(
        (p * (1 - p) + z * z / (4 * cases)) / cases
    )) / den


def mcnemar(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if not n:
        return 1.0
    low = min(a_only, b_only)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(low + 1)) / 2**n)


def run_variant(frame: pd.DataFrame, features: list[str]) -> dict:
    records, windows, tested = [], [], 0
    for start in range(0, len(frame) - HISTORY - TEST + 1, TEST):
        history = frame.iloc[start:start + HISTORY]
        test = frame.iloc[start + HISTORY:start + HISTORY + TEST]
        tested += len(test)
        chosen = select(history, features)
        if chosen is None:
            continue
        _, lam, threshold, quantile, va, vn = chosen
        pred, confidence = predict(fit(history, features, lam), test, features)
        use = confidence >= threshold
        for offset in np.flatnonzero(use):
            row = test.iloc[offset]
            records.append({
                "date": str(row.date.date()), "prediction": int(pred[offset]),
                "target": int(row.target), "hit": int(pred[offset] == row.target),
                "confidence": float(confidence[offset]), "lambda": lam,
                "quantile": quantile, "validation_accuracy": va,
                "validation_cases": vn,
                "night_spread": 1 if row.night_spread >= 0 else -1,
                "prior_intraday": 1 if row.prior_intraday >= 0 else -1,
            })
        cases = int(use.sum())
        windows.append({
            "test_start": str(test.date.min().date()), "test_end": str(test.date.max().date()),
            "cases": cases,
            "accuracy": float((pred[use] == test.loc[use, "target"]).mean()) if cases else 0.0,
            "lambda": lam, "quantile": quantile,
        })
    cases, hits = len(records), sum(r["hit"] for r in records)
    baselines = {}
    for name in ["always_up", "night_spread", "prior_intraday"]:
        bh = sum((1 if name == "always_up" else r[name]) == r["target"] for r in records)
        baselines[name] = {"hits": bh, "accuracy": bh / cases if cases else 0.0}
    strongest = max(baselines, key=lambda k: baselines[k]["accuracy"])
    model_only = sum(r["hit"] and (1 if strongest == "always_up" else r[strongest]) != r["target"] for r in records)
    base_only = sum(not r["hit"] and (1 if strongest == "always_up" else r[strongest]) == r["target"] for r in records)
    accuracy = hits / cases if cases else 0.0
    material = [w for w in windows if w["cases"] >= 10]
    passed = (
        accuracy >= 0.90 and cases >= 100 and cases / tested >= 0.10
        and wilson(hits, cases) >= 0.80
        and accuracy > baselines[strongest]["accuracy"]
        and model_only > base_only and mcnemar(model_only, base_only) < 0.05
        and all(w["accuracy"] >= 0.80 for w in material)
    )
    return {
        "tested_rows": tested, "cases": cases, "hits": hits, "accuracy": accuracy,
        "coverage": cases / tested if tested else 0.0,
        "wilson_95_lower": wilson(hits, cases), "baselines": baselines,
        "strongest_baseline": strongest,
        "strongest_baseline_accuracy": baselines[strongest]["accuracy"],
        "model_only": model_only, "baseline_only": base_only,
        "mcnemar_exact_p": mcnemar(model_only, base_only),
        "minimum_material_window_accuracy": min((w["accuracy"] for w in material), default=0.0),
        "passed": passed, "windows": windows, "records": records,
    }


def common_pair(left: dict, right: dict) -> dict:
    a = {r["date"]: r for r in left["records"]}
    b = {r["date"]: r for r in right["records"]}
    dates = sorted(a.keys() & b.keys())
    left_only = sum(a[d]["hit"] and not b[d]["hit"] for d in dates)
    right_only = sum(b[d]["hit"] and not a[d]["hit"] for d in dates)
    return {
        "identical_dates": len(dates),
        "pulse_accuracy": sum(a[d]["hit"] for d in dates) / len(dates) if dates else 0.0,
        "constitution_accuracy": sum(b[d]["hit"] for d in dates) / len(dates) if dates else 0.0,
        "pulse_only": left_only, "constitution_only": right_only,
        "mcnemar_exact_p": mcnemar(left_only, right_only),
    }


def main():
    frame = prepare()
    variants = {
        name: {"features": features, "result": run_variant(frame, features)}
        for name, features in VARIANTS.items()
    }
    pair = common_pair(
        variants["pulse_control"]["result"],
        variants["pulse_plus_lagged_constitution"]["result"],
    )
    constitution = variants["pulse_plus_lagged_constitution"]["result"]
    increment = (
        pair["constitution_only"] > pair["pulse_only"]
        and pair["mcnemar_exact_p"] < 0.05
    )
    constitution["passed"] = bool(constitution["passed"] and increment)
    payload = {
        "experiment_id": "latent_pulse_constitution_intraday_v1",
        "status": "strict historical OOS; not production unless every gate passes",
        "target": "sign of TWII same-day close versus same-day open",
        "decision_time": "after completed TX night session and before TWII cash open",
        "causal_rule": "Cash and FinMind inputs lag one completed TWII session; target-date inputs are restricted to the completed night session.",
        "method": "630-day ridge fit, following 126-day nested penalty/abstention selection, frozen 126-day OOS blocks.",
        "gate": ">=90% accuracy, >=100 cases, >=10% coverage, Wilson lower >=80%, every material block >=80%, significant superiority over strongest simple baseline and pulse-only control.",
        "aligned_rows": len(frame), "variants": variants,
        "constitution_increment": pair, "constitution_increment_passed": increment,
        "passing": [
            name for name, value in variants.items() if value["result"]["passed"]
        ],
    }
    out = ROOT / "reports"
    (out / "latent_pulse_constitution_intraday.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Latent pulse + lagged constitution intraday research", "",
        payload["status"], "", payload["target"], "", payload["causal_rule"], "",
        "| Variant | Cases | Accuracy | Coverage | Wilson lower | Strongest baseline | Model/base only | Minimum block | Passed |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for name, value in variants.items():
        r = value["result"]
        lines.append(
            f"| {name} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | "
            f"{r['wilson_95_lower']:.2%} | {r['strongest_baseline']} "
            f"{r['strongest_baseline_accuracy']:.2%} | {r['model_only']}/{r['baseline_only']} "
            f"(p={r['mcnemar_exact_p']:.4g}) | {r['minimum_material_window_accuracy']:.2%} | {r['passed']} |"
        )
    lines += [
        "",
        f"Common-date pulse/constitution accuracy: {pair['pulse_accuracy']:.2%} / {pair['constitution_accuracy']:.2%}.",
        f"Pulse-only/constitution-only wins: {pair['pulse_only']}/{pair['constitution_only']}; p={pair['mcnemar_exact_p']:.4g}.",
        f"Independent constitution increment passed: {increment}.",
    ]
    (out / "latent_pulse_constitution_intraday.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "aligned_rows": len(frame),
        "results": {
            name: {k: v for k, v in value["result"].items() if k not in ("records", "windows")}
            for name, value in variants.items()
        },
        "constitution_increment": pair,
    }, indent=2))


if __name__ == "__main__":
    main()
