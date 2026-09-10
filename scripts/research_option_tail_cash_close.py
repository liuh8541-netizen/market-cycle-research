"""Locked nested OOS ablation of TXO night-tail pricing for TWII cash close."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HISTORY, VALID, TEST = 756, 126, 126
LAMBDAS = [0.1, 1.0, 10.0, 100.0]
QUANTILES = [0.0, 0.25, 0.5, 0.65, 0.75, 0.85]
CONTROL = [
    "night_spread", "night_return", "night_range", "night_volume_z",
    "prior_cash_1d", "prior_cash_5d", "prior_cash_20d", "prior_cash_vol20",
]
TAIL = [
    "tail_skew_1pct_z", "tail_skew_2pct_z", "otm_pcr_volume_z",
    "tail_skew_change_1d", "tail_skew_change_5d",
    "night_x_tail1", "night_x_tail2",
]
VARIANTS = {"night_control": CONTROL, "night_plus_option_tail": CONTROL + TAIL}


def causal_z(s, window=252, minimum=80):
    s = pd.to_numeric(s, errors="coerce")
    mean = s.rolling(window, min_periods=minimum).mean().shift(1)
    std = s.rolling(window, min_periods=minimum).std().shift(1).replace(0, np.nan)
    return (s - mean) / std


def prepare():
    cash = pd.read_csv(ROOT / "data/processed/twii_daily.csv", usecols=["date", "close"])
    cash["date"] = pd.to_datetime(cash["date"])
    cash = cash.sort_values("date").drop_duplicates("date", keep="last")
    cash["cash_return"] = pd.to_numeric(cash["close"], errors="coerce").pct_change()
    cash["prior_cash_1d"] = cash["cash_return"].shift(1)
    cash["prior_cash_5d"] = cash["cash_return"].shift(1).rolling(5).sum()
    cash["prior_cash_20d"] = cash["cash_return"].shift(1).rolling(20).sum()
    cash["prior_cash_vol20"] = cash["cash_return"].shift(1).rolling(20).std()

    night = pd.read_csv(ROOT / "data/processed/taiwan_futures_night.csv")
    night["date"] = pd.to_datetime(night["signal_date"])
    night = night.rename(columns={
        "tx_night_spread_per": "night_spread", "tx_night_return": "night_return",
        "tx_night_range": "night_range", "tx_night_volume": "night_volume",
    })
    night["night_volume_z"] = causal_z(night["night_volume"])

    tail = pd.read_csv(ROOT / "data/processed/factors/option_tail_night.csv")
    tail["date"] = pd.to_datetime(tail["signal_date"])
    for col in ["tail_skew_1pct", "tail_skew_2pct", "otm_put_call_volume"]:
        tail[col] = pd.to_numeric(tail[col], errors="coerce")
    tail = tail.sort_values("date").drop_duplicates("date", keep="last")
    tail["tail_skew_1pct_z"] = causal_z(tail["tail_skew_1pct"])
    tail["tail_skew_2pct_z"] = causal_z(tail["tail_skew_2pct"])
    tail["otm_pcr_volume_z"] = causal_z(np.log(tail["otm_put_call_volume"].clip(lower=1e-6)))
    tail["tail_skew_change_1d"] = tail["tail_skew_2pct"].diff()
    tail["tail_skew_change_5d"] = tail["tail_skew_2pct"].diff(5)

    daily = ["date", "night_spread", "night_return", "night_range", "night_volume_z"]
    frame = cash.merge(night[daily], on="date", how="inner").merge(
        tail[["date"] + [c for c in TAIL if not c.startswith("night_x_")]], on="date", how="inner"
    )
    frame["night_x_tail1"] = frame["night_spread"] * frame["tail_skew_1pct_z"]
    frame["night_x_tail2"] = frame["night_spread"] * frame["tail_skew_2pct_z"]
    frame["target"] = np.where(frame["cash_return"] >= 0, 1, -1)
    frame["night_spread_baseline"] = np.where(frame["night_spread"] >= 0, 1, -1)
    frame["night_return_baseline"] = np.where(frame["night_return"] >= 0, 1, -1)
    return frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["cash_return", "night_spread", "night_return"]
    ).sort_values("date").reset_index(drop=True)


def fit(frame, features, lam):
    raw = frame[features]
    med = raw.median().fillna(0)
    filled = raw.fillna(med)
    mean, std = filled.mean(), filled.std().replace(0, 1).fillna(1)
    x = np.column_stack([np.ones(len(filled)), ((filled-mean)/std).to_numpy(float)])
    y = frame["target"].to_numpy(float)
    penalty = np.eye(x.shape[1])*lam
    penalty[0, 0] = 0
    return np.linalg.pinv(x.T@x+penalty)@x.T@y, med, mean, std


def predict(model, frame, features):
    weights, med, mean, std = model
    raw = frame[features].fillna(med).fillna(0)
    x = np.column_stack([np.ones(len(raw)), ((raw-mean)/std).to_numpy(float)])
    score = x@weights
    return np.where(score >= 0, 1, -1), np.abs(score)


def choose(history, features):
    train, valid = history.iloc[:-VALID], history.iloc[-VALID:]
    actual = valid["target"].to_numpy(int)
    best = None
    for lam in LAMBDAS:
        model = fit(train, features, lam)
        pred, conf = predict(model, valid, features)
        train_conf = predict(model, train, features)[1]
        for q in QUANTILES:
            cut = float(np.quantile(train_conf, q))
            use = conf >= cut
            cases = int(use.sum())
            if cases < 15 or use.mean() < .10:
                continue
            acc = float((pred[use] == actual[use]).mean())
            rank = (acc >= .90, acc, cases, -lam, -q)
            if best is None or rank > best[0]:
                best = (rank, lam, q, acc, cases)
    return best


def wilson(h, n):
    if not n:
        return 0
    z, p = 1.959963984540054, h/n
    den = 1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/den


def mcnemar(a, b):
    n = a+b
    if not n:
        return 1
    return min(1, 2*sum(math.comb(n,k) for k in range(min(a,b)+1))/2**n)


def evaluate(frame, features):
    records, windows, tested = [], [], 0
    for start in range(0, len(frame)-HISTORY-TEST+1, TEST):
        history = frame.iloc[start:start+HISTORY]
        test = frame.iloc[start+HISTORY:start+HISTORY+TEST]
        tested += len(test)
        cfg = choose(history, features)
        if not cfg:
            continue
        _, lam, q, va, vn = cfg
        model = fit(history, features, lam)
        pred, conf = predict(model, test, features)
        cut = float(np.quantile(predict(model, history, features)[1], q))
        use = conf >= cut
        for pos in np.flatnonzero(use):
            row = test.iloc[pos]
            records.append({
                "date": str(row.date.date()), "prediction": int(pred[pos]),
                "target": int(row.target), "hit": int(pred[pos] == row.target),
                "night_spread": int(row.night_spread_baseline),
                "night_return": int(row.night_return_baseline),
                "confidence": float(conf[pos]), "lambda": lam, "quantile": q,
            })
        cases = int(use.sum())
        windows.append({
            "test_start": str(test.date.min().date()), "test_end": str(test.date.max().date()),
            "cases": cases, "accuracy": float((pred[use] == test.loc[use,"target"]).mean()) if cases else 0,
            "validation_accuracy": va, "validation_cases": vn,
        })
    n, hits = len(records), sum(r["hit"] for r in records)
    baselines = {}
    for name in ["night_spread", "night_return"]:
        bh = sum(r[name] == r["target"] for r in records)
        baselines[name] = {"hits": bh, "accuracy": bh/n if n else 0}
    strongest = max(baselines, key=lambda k: baselines[k]["accuracy"])
    mo = sum(r["hit"] and r[strongest] != r["target"] for r in records)
    bo = sum(not r["hit"] and r[strongest] == r["target"] for r in records)
    acc = hits/n if n else 0
    material = [w for w in windows if w["cases"] >= 10]
    result = {
        "tested_rows": tested, "cases": n, "hits": hits, "accuracy": acc,
        "coverage": n/tested if tested else 0, "wilson_95_lower": wilson(hits,n),
        "baselines": baselines, "strongest_baseline": strongest,
        "strongest_baseline_accuracy": baselines[strongest]["accuracy"],
        "model_only": mo, "baseline_only": bo, "mcnemar_exact_p": mcnemar(mo,bo),
        "minimum_material_window_accuracy": min((w["accuracy"] for w in material), default=0),
        "windows": windows, "records": records,
    }
    result["passed"] = (
        n >= 100 and result["coverage"] >= .10 and acc >= .90
        and result["wilson_95_lower"] >= .80
        and acc > result["strongest_baseline_accuracy"]
        and mo > bo and result["mcnemar_exact_p"] < .05
        and result["minimum_material_window_accuracy"] >= .80
    )
    return result


def paired(left, right):
    a, b = ({r["date"]:r for r in x["records"]} for x in [left,right])
    dates = sorted(a.keys() & b.keys())
    lo = sum(a[d]["hit"] and not b[d]["hit"] for d in dates)
    ro = sum(b[d]["hit"] and not a[d]["hit"] for d in dates)
    return {
        "identical_dates": len(dates),
        "control_accuracy": sum(a[d]["hit"] for d in dates)/len(dates) if dates else 0,
        "tail_accuracy": sum(b[d]["hit"] for d in dates)/len(dates) if dates else 0,
        "control_only": lo, "tail_only": ro, "mcnemar_exact_p": mcnemar(lo,ro),
    }


def main():
    frame = prepare()
    results = {name:evaluate(frame,features) for name,features in VARIANTS.items()}
    pair = paired(results["night_control"],results["night_plus_option_tail"])
    increment = pair["tail_only"] > pair["control_only"] and pair["mcnemar_exact_p"] < .05
    results["night_plus_option_tail"]["passed"] &= increment
    payload = {
        "experiment_id": "option_tail_cash_close_v1",
        "status": "strict historical OOS; not production unless every gate passes",
        "target": "same-day TWII cash close versus prior completed cash close",
        "decision_time": "after completed TX/TXO night sessions and before TWII cash open",
        "hypothesis": "Night option tail pricing adds cash-close direction beyond TX night price and lagged cash state.",
        "method": "630/126/126 nested ridge/abstention exact-date ablation.",
        "gate": ">=90% accuracy, >=100 cases, >=10% coverage, Wilson >=80%, min block >=80%, significant superiority over strongest night baseline and control.",
        "aligned_rows": len(frame), "variants": results, "tail_increment": pair,
        "tail_increment_passed": increment,
        "passing": [n for n,r in results.items() if r["passed"]],
    }
    out = ROOT/"reports"
    (out/"option_tail_cash_close.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines = ["# Option-tail cash-close research","",payload["status"],"",payload["hypothesis"],"",
             "| Variant | Cases | Accuracy | Coverage | Wilson lower | Strongest baseline | Model/base only | Min block | Passed |",
             "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |"]
    for name,r in results.items():
        lines.append(f"| {name} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | "
                     f"{r['wilson_95_lower']:.2%} | {r['strongest_baseline']} {r['strongest_baseline_accuracy']:.2%} | "
                     f"{r['model_only']}/{r['baseline_only']} (p={r['mcnemar_exact_p']:.4g}) | "
                     f"{r['minimum_material_window_accuracy']:.2%} | {r['passed']} |")
    lines += ["",f"Common control/tail: {pair['control_accuracy']:.2%}/{pair['tail_accuracy']:.2%}; "
              f"exclusive wins {pair['control_only']}/{pair['tail_only']}; p={pair['mcnemar_exact_p']:.4g}.",
              f"Independent tail increment passed: {increment}."]
    (out/"option_tail_cash_close.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"aligned_rows":len(frame),"variants":{n:{k:v for k,v in r.items() if k not in ('records','windows','baselines')} for n,r in results.items()},"tail_increment":pair},indent=2))


if __name__ == "__main__":
    main()
