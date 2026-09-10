"""After-close FinMind cash/margin factors -> next TWII opening gap."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from market_lifecycle.factor_features import add_factor_features


HISTORY, VALID, TEST = 756, 126, 126
LAMBDAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
QUANTILES = [0.0, 0.25, 0.5, 0.65, 0.75, 0.85, 0.9]
RAW_FACTORS = [
    "inst_net_proxy", "inst_net_5d", "inst_net_20d",
    "inst_foreign_net_proxy", "inst_foreign_net_5d", "inst_foreign_net_20d",
    "inst_trust_net_proxy", "inst_trust_net_5d", "inst_trust_net_20d",
    "inst_dealer_net_proxy", "inst_dealer_net_5d", "inst_dealer_net_20d",
    "margin_balance_proxy", "margin_change_20d", "short_balance_proxy",
    "short_change_20d", "margin_short_ratio",
]
PRICE_FEATURES = ["cash_return", "cash_return_5d", "gap_return", "intraday_return", "volatility20"]
FEATURES = RAW_FACTORS + ["margin_change_1d", "short_change_1d"] + PRICE_FEATURES


def prepare():
    cash = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
    cash["date"] = pd.to_datetime(cash["date"])
    for col in ["open", "close", "high", "low", "volume"]:
        cash[col] = pd.to_numeric(cash[col], errors="coerce")
    cash = cash.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    cash["cash_return"] = cash["close"].pct_change()
    cash["cash_return_5d"] = cash["close"].pct_change(5)
    cash["gap_return"] = cash["open"] / cash["close"].shift(1) - 1
    cash["intraday_return"] = cash["close"] / cash["open"] - 1
    cash["volatility20"] = np.log(cash["close"]).diff().rolling(20).std()
    frame = add_factor_features(cash, str(ROOT / "data" / "processed" / "factors"))
    for col in RAW_FACTORS:
        if col not in frame:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["margin_change_1d"] = frame["margin_balance_proxy"].pct_change()
    frame["short_change_1d"] = frame["short_balance_proxy"].pct_change()
    # Row D target is D+1 cash open versus D close. All factor inputs are D
    # after-close publications, with no D+1 night information in FEATURES.
    frame["target_gap_return"] = frame["open"].shift(-1) / frame["close"] - 1
    frame["target"] = np.where(frame["target_gap_return"] >= 0, 1, -1)
    frame.loc[frame["target_gap_return"].isna(), "target"] = np.nan
    frame["target_date"] = frame["date"].shift(-1)

    # Later-information benchmark only: D+1 night session is not available at
    # the registered D-close decision time and never enters model selection.
    night = pd.read_csv(ROOT / "data" / "processed" / "taiwan_futures_night.csv")
    night["target_date"] = pd.to_datetime(night["signal_date"])
    night["later_night_prediction"] = np.where(
        pd.to_numeric(night["tx_night_spread_per"], errors="coerce") >= 0, 1, -1
    )
    frame = frame.merge(night[["target_date", "later_night_prediction"]], on="target_date", how="left")
    usable = frame["inst_net_proxy"].notna() & frame["target"].notna()
    return frame.loc[usable].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def fit(frame, lam):
    raw = frame[FEATURES].copy()
    med = raw.median().fillna(0)
    raw = raw.fillna(med)
    mean, std = raw.mean(), raw.std().replace(0, 1).fillna(1)
    x = ((raw - mean) / std).to_numpy(float)
    x = np.column_stack([np.ones(len(x)), x])
    y = frame["target"].to_numpy(float)
    penalty = np.eye(x.shape[1]) * lam
    penalty[0, 0] = 0
    weights = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y
    return weights, med, mean, std


def predict(model, frame):
    weights, med, mean, std = model
    raw = frame[FEATURES].fillna(med).fillna(0)
    x = ((raw - mean) / std).to_numpy(float)
    score = np.column_stack([np.ones(len(x)), x]) @ weights
    return np.where(score >= 0, 1, -1), np.abs(score)


def select(history):
    train, valid = history.iloc[:-VALID], history.iloc[-VALID:]
    best = None
    for lam in LAMBDAS:
        model = fit(train, lam)
        pred, confidence = predict(model, valid)
        actual = valid["target"].to_numpy(int)
        for quantile in QUANTILES:
            threshold = float(np.quantile(confidence, quantile))
            use = confidence >= threshold
            cases = int(use.sum())
            if cases < 15 or cases / len(valid) < .10:
                continue
            accuracy = float((pred[use] == actual[use]).mean())
            rank = (accuracy, cases / len(valid), -lam, -quantile)
            if best is None or rank > best[0]:
                best = (rank, lam, threshold, accuracy, cases, quantile)
    return best


def wilson(hits, cases):
    if not cases:
        return 0.0
    z = 1.959963984540054
    p = hits / cases
    den = 1 + z * z / cases
    return (p + z*z/(2*cases) - z * math.sqrt((p*(1-p) + z*z/(4*cases))/cases)) / den


def mcnemar(a_only, b_only):
    n = a_only + b_only
    if not n:
        return 1.0
    low = min(a_only, b_only)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(low + 1)) / (2**n))


def main():
    frame = prepare()
    records, windows, tested = [], [], 0
    start = 0
    while start + HISTORY + TEST <= len(frame):
        history = frame.iloc[start : start + HISTORY]
        test = frame.iloc[start + HISTORY : start + HISTORY + TEST]
        tested += len(test)
        config = select(history)
        if config:
            _, lam, threshold, valid_accuracy, valid_cases, quantile = config
            model = fit(history, lam)
            pred, confidence = predict(model, test)
            use = confidence >= threshold
            for offset in np.flatnonzero(use):
                row = test.iloc[offset]
                prior_gap = 1 if row["gap_return"] >= 0 else -1
                prior_cash = 1 if row["cash_return"] >= 0 else -1
                inst_sign = 1 if row["inst_net_proxy"] >= 0 else -1
                records.append({
                    "signal_date": str(row["date"].date()),
                    "target_date": str(row["target_date"].date()),
                    "prediction": int(pred[offset]), "target": int(row["target"]),
                    "hit": int(pred[offset] == row["target"]), "confidence": float(confidence[offset]),
                    "lambda": lam, "threshold": threshold, "selected_quantile": quantile,
                    "validation_accuracy": valid_accuracy, "validation_cases": valid_cases,
                    "baselines": {"always_up": 1, "prior_gap": prior_gap, "prior_cash": prior_cash, "institution_sign": inst_sign},
                    "later_night_prediction": int(row["later_night_prediction"]) if pd.notna(row["later_night_prediction"]) else None,
                })
            windows.append({
                "test_end": str(test["target_date"].max().date()), "lambda": lam,
                "threshold": threshold, "validation_accuracy": valid_accuracy,
                "validation_cases": valid_cases, "test_cases": int(use.sum()),
                "test_accuracy": float((pred[use] == test.loc[use, "target"].to_numpy()).mean()) if use.any() else 0,
            })
        start += TEST
    cases, hits = len(records), sum(row["hit"] for row in records)
    accuracy = hits / cases if cases else 0
    coverage = cases / tested if tested else 0
    baselines = {}
    for name in ["always_up", "prior_gap", "prior_cash", "institution_sign"]:
        bh = sum(row["baselines"][name] == row["target"] for row in records)
        baselines[name] = {"hits": bh, "accuracy": bh / cases if cases else 0}
    strongest_name = max(baselines, key=lambda name: baselines[name]["accuracy"])
    model_only = sum(row["hit"] and row["baselines"][strongest_name] != row["target"] for row in records)
    baseline_only = sum(not row["hit"] and row["baselines"][strongest_name] == row["target"] for row in records)
    later = [row for row in records if row["later_night_prediction"] is not None]
    later_hits = sum(row["later_night_prediction"] == row["target"] for row in later)
    p_value = mcnemar(model_only, baseline_only)
    lower = wilson(hits, cases)
    passed = (
        accuracy >= .90 and cases >= 100 and coverage >= .10 and lower >= .80
        and accuracy > baselines[strongest_name]["accuracy"]
        and model_only > baseline_only and p_value < .05
    )
    result = {
        "aligned_rows": len(frame), "tested_rows": tested, "cases": cases, "hits": hits,
        "accuracy": accuracy, "coverage": coverage, "wilson_95_lower": lower,
        "baselines_at_decision_time": baselines, "strongest_baseline": strongest_name,
        "strongest_baseline_accuracy": baselines[strongest_name]["accuracy"],
        "model_only": model_only, "baseline_only": baseline_only, "mcnemar_exact_p": p_value,
        "later_information_night_benchmark": {"cases": len(later), "hits": later_hits, "accuracy": later_hits/len(later) if later else 0},
        "passed": passed, "windows": windows, "records": records,
    }
    payload = {
        "scope": "At TWII close D, predict the D+1 cash opening gap before the D+1 night session is known.",
        "hypothesis": "After-close cash-institution and margin state contains an independent next-opening-gap signal.",
        "method": "756/126/126 nested rolling walk-forward ridge and abstention; D factors map only to D+1; no night data in model; identical-date causal simple baselines.",
        "gate": ">=90% accuracy, >=100 cases, >=10% coverage, Wilson lower >=80%, and significant paired superiority over strongest decision-time simple baseline.",
        "features": FEATURES, "result": result,
    }
    out = ROOT / "reports"
    (out / "after_close_next_gap.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# After-close FinMind factors to next opening gap", "", payload["scope"], "",
        payload["hypothesis"], "", payload["method"], "", payload["gate"], "",
        f"OOS: {hits}/{cases} = {accuracy:.2%}; coverage {coverage:.2%}; Wilson lower {lower:.2%}.",
        f"Strongest decision-time baseline: {strongest_name} {baselines[strongest_name]['accuracy']:.2%}.",
        f"Paired model-only/baseline-only: {model_only}/{baseline_only}; p={p_value:.4g}; passed={passed}.",
        f"Later-information night benchmark (not available at D close): {later_hits}/{len(later)} = {later_hits/len(later) if later else 0:.2%}.", "",
        "| Test end | Cases | Accuracy | Validation cases/accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for window in windows:
        lines.append(f"| {window['test_end']} | {window['test_cases']} | {window['test_accuracy']:.2%} | {window['validation_cases']}/{window['validation_accuracy']:.2%} |")
    (out / "after_close_next_gap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ["windows", "records"]}, indent=2))


if __name__ == "__main__":
    main()
