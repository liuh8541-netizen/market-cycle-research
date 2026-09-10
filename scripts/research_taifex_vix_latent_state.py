"""Nested walk-forward test of lagged Taiwan VIX as a latent risk-state proxy."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HISTORY, VALID, TEST = 378, 126, 126
LAMBDAS = [0.1, 1.0, 10.0, 100.0]
QUANTILES = [0.0, 0.25, 0.5, 0.65, 0.75, 0.85]
FEATURES = [
    "night_spread", "night_return", "vix_z_20", "vix_change_1d", "vix_change_5d",
    "night_x_vix_level", "night_x_vix_change",
]


def wilson(hits, cases):
    if not cases:
        return 0.0
    z = 1.959963984540054
    p = hits / cases
    den = 1 + z * z / cases
    return (p + z*z/(2*cases) - z*math.sqrt((p*(1-p)+z*z/(4*cases))/cases)) / den


def mcnemar(left_only, right_only):
    n = left_only + right_only
    if not n:
        return 1.0
    low = min(left_only, right_only)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(low + 1)) / 2**n)


def prepare():
    cash = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv", usecols=["date", "open", "close"])
    cash["date"] = pd.to_datetime(cash["date"])
    cash = cash.sort_values("date").drop_duplicates("date", keep="last")
    cash["prior_close"] = pd.to_numeric(cash["close"], errors="coerce").shift()
    cash["gap_return"] = pd.to_numeric(cash["open"], errors="coerce") / cash["prior_close"] - 1
    cash["close_return"] = pd.to_numeric(cash["close"], errors="coerce") / cash["prior_close"] - 1

    night = pd.read_csv(ROOT / "data" / "processed" / "taiwan_futures_night.csv")
    night["date"] = pd.to_datetime(night["signal_date"])
    night = night.rename(columns={"tx_night_spread_per": "night_spread", "tx_night_return": "night_return"})

    vix = pd.read_csv(ROOT / "data" / "processed" / "factors" / "taifex_vix_daily.csv")
    vix["vix_date"] = pd.to_datetime(vix["date"])
    vix["vix_close"] = pd.to_numeric(vix["vix_close"], errors="coerce")
    vix = vix.sort_values("vix_date").drop_duplicates("vix_date", keep="last")
    # All rolling features are formed on the VIX series itself. The as-of join
    # below forbids an equal date, so Taiwan date D can only use VIX close D-1.
    prior_mean = vix["vix_close"].rolling(20, min_periods=20).mean().shift(1)
    prior_std = vix["vix_close"].rolling(20, min_periods=20).std().shift(1)
    vix["vix_z_20"] = (vix["vix_close"] - prior_mean) / prior_std
    vix["vix_change_1d"] = vix["vix_close"].pct_change()
    vix["vix_change_5d"] = vix["vix_close"].pct_change(5)

    x = cash[["date", "gap_return", "close_return"]].merge(
        night[["date", "night_spread", "night_return"]], on="date", how="inner"
    ).sort_values("date")
    x = pd.merge_asof(
        x, vix[["vix_date", "vix_close", "vix_z_20", "vix_change_1d", "vix_change_5d"]],
        left_on="date", right_on="vix_date", direction="backward",
        allow_exact_matches=False, tolerance=pd.Timedelta("7D"),
    )
    x["night_x_vix_level"] = x["night_spread"] * x["vix_z_20"]
    x["night_x_vix_change"] = x["night_spread"] * x["vix_change_1d"]
    x["gap_target"] = np.where(x["gap_return"] >= 0, 1, -1)
    x["close_target"] = np.where(x["close_return"] >= 0, 1, -1)
    x["night_spread_baseline"] = np.where(x["night_spread"] >= 0, 1, -1)
    x["night_return_baseline"] = np.where(x["night_return"] >= 0, 1, -1)
    return x.dropna(subset=FEATURES + ["gap_return", "close_return"]).reset_index(drop=True)


def fit(frame, target, lam):
    raw = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    med = raw.median().fillna(0)
    mean = raw.fillna(med).mean()
    std = raw.fillna(med).std().replace(0, 1).fillna(1)
    matrix = ((raw.fillna(med) - mean) / std).to_numpy(float)
    matrix = np.column_stack([np.ones(len(matrix)), matrix])
    y = frame[target].to_numpy(float)
    penalty = np.eye(matrix.shape[1]) * lam
    penalty[0, 0] = 0
    weights = np.linalg.pinv(matrix.T @ matrix + penalty) @ matrix.T @ y
    return weights, med, mean, std


def predict(model, frame):
    weights, med, mean, std = model
    raw = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0)
    matrix = ((raw - mean) / std).to_numpy(float)
    score = np.column_stack([np.ones(len(matrix)), matrix]) @ weights
    return np.where(score >= 0, 1, -1), np.abs(score)


def select(history, target):
    train, valid = history.iloc[:-VALID], history.iloc[-VALID:]
    actual = valid[target].to_numpy(int)
    best = None
    for lam in LAMBDAS:
        model = fit(train, target, lam)
        pred, confidence = predict(model, valid)
        train_conf = predict(model, train)[1]
        for quantile in QUANTILES:
            threshold = float(np.quantile(train_conf, quantile))
            use = confidence >= threshold
            cases = int(use.sum())
            if cases < 15 or cases / len(valid) < .10:
                continue
            accuracy = float((pred[use] == actual[use]).mean())
            rank = (accuracy, cases, -lam, -quantile)
            if best is None or rank > best[0]:
                best = (rank, lam, quantile, accuracy, cases)
    return best


def evaluate(frame, target):
    records, windows, tested = [], [], 0
    start = 0
    while start + HISTORY + TEST <= len(frame):
        history = frame.iloc[start:start + HISTORY]
        test = frame.iloc[start + HISTORY:start + HISTORY + TEST]
        tested += len(test)
        chosen = select(history, target)
        if chosen:
            _, lam, quantile, validation_accuracy, validation_cases = chosen
            model = fit(history, target, lam)
            pred, confidence = predict(model, test)
            threshold = float(np.quantile(predict(model, history)[1], quantile))
            use = confidence >= threshold
            for pos in np.flatnonzero(use):
                row = test.iloc[pos]
                records.append({
                    "date": str(row["date"].date()), "vix_date": str(row["vix_date"].date()),
                    "prediction": int(pred[pos]), "target": int(row[target]),
                    "hit": int(pred[pos] == row[target]), "confidence": float(confidence[pos]),
                    "vix_close": float(row["vix_close"]), "lambda": lam,
                    "quantile": quantile, "threshold": threshold,
                    "validation_accuracy": validation_accuracy, "validation_cases": validation_cases,
                    "baselines": {
                        "night_spread": int(row["night_spread_baseline"]),
                        "night_return": int(row["night_return_baseline"]),
                        "always_up": 1,
                    },
                })
            hits = int((pred[use] == test.loc[use, target].to_numpy()).sum())
            cases = int(use.sum())
            windows.append({
                "test_start": str(test["date"].min().date()), "test_end": str(test["date"].max().date()),
                "lambda": lam, "quantile": quantile, "validation_accuracy": validation_accuracy,
                "validation_cases": validation_cases, "test_cases": cases,
                "test_accuracy": hits / cases if cases else 0,
            })
        start += TEST

    cases = len(records)
    hits = sum(row["hit"] for row in records)
    accuracy = hits / cases if cases else 0
    coverage = cases / tested if tested else 0
    baselines = {}
    for name in ["night_spread", "night_return", "always_up"]:
        bh = sum(row["baselines"][name] == row["target"] for row in records)
        baselines[name] = {"hits": bh, "accuracy": bh / cases if cases else 0}
    strongest_name = max(baselines, key=lambda name: baselines[name]["accuracy"])
    strongest = baselines[strongest_name]
    model_only = sum(row["hit"] and row["baselines"][strongest_name] != row["target"] for row in records)
    baseline_only = sum(not row["hit"] and row["baselines"][strongest_name] == row["target"] for row in records)
    p_value = mcnemar(model_only, baseline_only)
    material = [row for row in windows if row["test_cases"] >= 10]
    minimum_block = min((row["test_accuracy"] for row in material), default=0)
    passed = (
        cases >= 100 and coverage >= .10 and accuracy >= .90 and wilson(hits, cases) >= .80
        and accuracy > strongest["accuracy"] and model_only > baseline_only and p_value < .05
        and minimum_block >= .80
    )
    return {
        "target": target, "tested_rows": tested, "cases": cases, "hits": hits,
        "accuracy": accuracy, "coverage": coverage, "wilson_95_lower": wilson(hits, cases),
        "baselines": baselines, "strongest_baseline": strongest_name,
        "strongest_baseline_accuracy": strongest["accuracy"], "model_only": model_only,
        "baseline_only": baseline_only, "mcnemar_exact_p": p_value,
        "minimum_material_window_accuracy": minimum_block, "passed": passed,
        "windows": windows, "records": records,
    }


def main():
    frame = prepare()
    results = [evaluate(frame, "gap_target"), evaluate(frame, "close_target")]
    payload = {
        "status": "registered short-history latent-state experiment; not production",
        "hypothesis": "The latest completed Taiwan VIX close identifies a latent risk state that changes how the following Taiwan night signal transfers into the cash market.",
        "causal_rule": "For target date D, VIX date must be strictly earlier than D; same-day VIX close is forbidden.",
        "source": "Official TAIFEX rolling three-year daily Taiwan VIX endpoint.",
        "method": "378-day rolling history; its final 126 days select ridge penalty and abstention; freeze for the following 126-day OOS block. Seven fixed low-degree night/VIX features. Exact-date simple-baseline audit.",
        "gate": ">=90% accuracy, >=100 cases, >=10% coverage, Wilson lower >=80%, significant superiority over strongest identical-date baseline, every material block >=80%.",
        "history": HISTORY, "validation": VALID, "test": TEST, "features": FEATURES,
        "aligned_rows": len(frame), "date_start": str(frame.date.min().date()),
        "date_end": str(frame.date.max().date()), "results": results,
        "passing": [row["target"] for row in results if row["passed"]],
    }
    out = ROOT / "reports"
    (out / "taifex_vix_latent_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# TAIFEX VIX latent-state research", "", payload["status"], "",
        payload["hypothesis"], "", payload["causal_rule"], "", payload["method"], "",
        payload["gate"], "",
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
        lines += ["", f"## {row['target']} windows", "", "| Test period | Lambda | Quantile | Validation | OOS cases/accuracy |", "| --- | ---: | ---: | ---: | ---: |"]
        for window in row["windows"]:
            lines.append(
                f"| {window['test_start']} to {window['test_end']} | {window['lambda']} | {window['quantile']:.0%} | "
                f"{window['validation_cases']}/{window['validation_accuracy']:.2%} | "
                f"{window['test_cases']}/{window['test_accuracy']:.2%} |"
            )
    lines += ["", "Every selected OOS prediction and its frozen configuration is retained in the JSON companion file."]
    (out / "taifex_vix_latent_state.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "aligned_rows": len(frame), "date_start": payload["date_start"], "date_end": payload["date_end"],
        "passing": payload["passing"],
        "results": [{k: v for k, v in row.items() if k not in ["records", "windows", "baselines"]} for row in results],
    }, indent=2))


if __name__ == "__main__":
    main()
