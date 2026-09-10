import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from market_lifecycle.factor_features import add_factor_features
from research_causal_phase import add_phase_features, wilson_lower


HORIZONS = [20, 60, 120]
TRAIN_DAYS = 756
TEST_DAYS = 126


def trailing_z(series, window=252):
    mean = series.rolling(window, min_periods=80).mean().shift(1)
    std = series.rolling(window, min_periods=80).std().shift(1).replace(0, np.nan)
    return (series - mean) / std


def prepare():
    price = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
    price["date"] = pd.to_datetime(price["date"])
    x = add_factor_features(price, str(ROOT / "data" / "processed" / "factors"))
    external = pd.read_csv(ROOT / "data" / "processed" / "external_markets.csv", usecols=["date", "vix_close"])
    external["date"] = pd.to_datetime(external["date"])
    x = pd.merge_asof(x.sort_values("date"), external.sort_values("date"), on="date", direction="backward", tolerance=pd.Timedelta("5D"))
    x = add_phase_features(x, fast=20, slow=120, bins=12)
    x["ret20"] = x["close"].pct_change(20)
    for col in ["inst_net_20d", "margin_change_20d", "futures_inst_net_20d", "option_put_call_proxy", "vix_close"]:
        if col not in x:
            x[col] = np.nan
        x[col + "_causal_z"] = trailing_z(pd.to_numeric(x[col], errors="coerce"))
    cause = pd.Series("BALANCED", index=x.index, dtype="object")
    cause[x["inst_net_20d_causal_z"] <= -1.0] = "CAPITAL_OUTFLOW"
    cause[(x["margin_change_20d_causal_z"] >= 1.0) & (x["ret20"] < 0)] = "LEVERAGE_SQUEEZE"
    cause[(x["futures_inst_net_20d_causal_z"] <= -1.0) & (x["option_put_call_proxy_causal_z"] >= .5)] = "DERIVATIVE_HEDGE"
    cause[x["vix_close_causal_z"] >= 1.5] = "GLOBAL_FEAR"
    x["cause"] = cause
    x["clinical_state"] = x["state"] + "|" + x["cause"]
    return x


def state_map(train):
    mapping = {}
    for state, group in train.dropna(subset=["target"]).groupby("clinical_state"):
        if len(group) < 20:
            continue
        counts = group["target"].value_counts()
        if counts.max() / len(group) >= .60:
            mapping[state] = int(counts.idxmax())
    return mapping


def evaluate(base, horizon):
    x = base.copy()
    future = x["close"].shift(-horizon) / x["close"] - 1
    x["target"] = np.where(future > 0, 1, -1).astype(float)
    x.loc[future.isna(), "target"] = np.nan
    rows = []
    tested = 0
    start = 0
    while start + TRAIN_DAYS + TEST_DAYS + horizon <= len(x):
        train = x.iloc[start:start + TRAIN_DAYS]
        test = x.iloc[start + TRAIN_DAYS:start + TRAIN_DAYS + TEST_DAYS].copy()
        mapping = state_map(train)
        test["prediction"] = test["clinical_state"].map(mapping)
        usable = test["prediction"].notna() & test["target"].notna()
        for r in test.loc[usable].itertuples():
            rows.append({"cause": r.cause, "hit": int(r.prediction == r.target)})
        tested += len(test)
        start += TEST_DAYS
    pred = pd.DataFrame(rows)
    cases = len(pred)
    hits = int(pred["hit"].sum()) if cases else 0
    actual = x.iloc[TRAIN_DAYS:TRAIN_DAYS + tested]["target"].dropna()
    up = float((actual == 1).mean()) if len(actual) else .5
    baseline = max(up, 1-up)
    by_cause = {}
    if cases:
        for cause, group in pred.groupby("cause"):
            h = int(group["hit"].sum()); n = len(group)
            by_cause[cause] = {"cases": n, "accuracy": h/n, "wilson_95_lower": wilson_lower(h,n)}
    accuracy = hits/cases if cases else 0
    coverage = cases/tested if tested else 0
    lower = wilson_lower(hits,cases)
    return {"horizon_days":horizon,"cases":cases,"hits":hits,"accuracy":accuracy,"coverage":coverage,"baseline":baseline,"edge":accuracy-baseline,"wilson_95_lower":lower,"passed":accuracy>=.9 and cases>=100 and coverage>=.1 and lower>=.8 and accuracy>baseline,"by_cause":by_cause}


def main():
    data = prepare()
    results = [evaluate(data,h) for h in HORIZONS]
    payload={"hypothesis":"FinMind positioning and VIX identify the cause of an abnormal market pulse before trusting cycle phase.","method":"Causal rolling z-scores; fixed cause hierarchy; 3-year train/6-month test walk-forward.","results":results,"passing":[r["horizon_days"] for r in results if r["passed"]]}
    out=ROOT/"reports"
    (out/"pulse_cause_research.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# Pulse cause research","",payload["method"],"","| Horizon | Cases | Accuracy | Coverage | Baseline | Edge | Wilson lower | Passed |","| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results: lines.append(f"| {r['horizon_days']} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['baseline']:.2%} | {r['edge']:.2%} | {r['wilson_95_lower']:.2%} | {r['passed']} |")
    for r in results:
        lines += ["",f"## {r['horizon_days']} days by cause"]+[f"- {k}: {v['accuracy']:.2%} ({v['cases']} cases; lower {v['wilson_95_lower']:.2%})" for k,v in r['by_cause'].items()]
    (out/"pulse_cause_research.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
