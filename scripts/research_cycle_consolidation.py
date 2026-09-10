import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_selective_signals import wilson_lower


HORIZONS = [60, 120]
MARKETS = {"TWII": None, "SP500": "sp500_close", "NASDAQ": "nasdaq_close", "SOX": "sox_close", "DOW": "dow_close"}


def make_price(external, column):
    frame = external[["date", column]].dropna().rename(columns={column: "close"}).copy()
    return frame.reset_index(drop=True)


def event_frame(price):
    x = price.copy().sort_values("date").reset_index(drop=True)
    close = x["close"]
    x["ret20"] = close.pct_change(20)
    x["ret240"] = close.pct_change(240)
    x["ma200"] = close.rolling(200).mean()
    x["ma_slope"] = x["ma200"].pct_change(40)
    x["range20"] = close.rolling(20).max() / close.rolling(20).min() - 1
    x["consolidating"] = x["ret20"].abs().le(0.035) & x["range20"].le(0.08)
    x["macro"] = np.where((x["ret240"] > 0) & (x["ma_slope"] > 0), 1,
                          np.where((x["ret240"] < 0) & (x["ma_slope"] < 0), -1, 0))
    # Exit is known at the close: yesterday was consolidation, today breaks in macro direction.
    aligned_break = ((x["macro"] == 1) & (x["ret20"] > 0.035)) | ((x["macro"] == -1) & (x["ret20"] < -0.035))
    x["event"] = x["consolidating"].shift(1).fillna(False) & (~x["consolidating"]) & aligned_break
    # Prevent overlapping observations from counting the same episode repeatedly.
    keep = []
    last = -10_000
    for idx in x.index[x["event"]]:
        if idx - last >= 20:
            keep.append(idx)
            last = idx
    x["event"] = False
    x.loc[keep, "event"] = True
    return x


def evaluate(name, price, horizon):
    x = event_frame(price)
    x["future_return"] = x["close"].shift(-horizon) / x["close"] - 1
    sample = x.loc[x["event"] & x["future_return"].notna()].copy()
    sample["hit"] = np.sign(sample["future_return"]) == sample["macro"]
    cases = len(sample)
    hits = int(sample["hit"].sum())
    valid = x["future_return"].notna()
    up_baseline = float((x.loc[valid, "future_return"] > 0).mean())
    majority_baseline = max(up_baseline, 1 - up_baseline)
    return {
        "market": name,
        "horizon_days": horizon,
        "cases": cases,
        "hits": hits,
        "accuracy": hits / cases if cases else 0.0,
        "wilson_95_lower": wilson_lower(hits, cases),
        "majority_baseline": majority_baseline,
        "edge": (hits / cases - majority_baseline) if cases else None,
        "bull_cases": int((sample["macro"] == 1).sum()),
        "bear_cases": int((sample["macro"] == -1).sum()),
    }


def main():
    external = pd.read_csv(ROOT / "data" / "processed" / "external_markets.csv")
    external["date"] = pd.to_datetime(external["date"])
    twii = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv", usecols=["date", "close"])
    twii["date"] = pd.to_datetime(twii["date"])
    frames = {"TWII": twii}
    frames.update({name: make_price(external, col) for name, col in MARKETS.items() if col})
    results = [evaluate(name, frame, horizon) for horizon in HORIZONS for name, frame in frames.items()]
    payload = {
        "hypothesis": "A consolidation is a transient state; after an aligned breakout, price resumes the pre-existing 240-day macro cycle.",
        "method": "Fixed thresholds, non-overlapping events, forward 60/120 trading-day direction, majority baseline comparison.",
        "results": results,
    }
    payload["passing"] = [r for r in results if r["accuracy"] >= .9 and r["cases"] >= 100 and r["wilson_95_lower"] >= .8 and r["edge"] > 0]
    out = ROOT / "reports"
    (out / "cycle_consolidation_research.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Cycle consolidation research", "", payload["hypothesis"], "", "| Market | Horizon | Cases | Bull/Bear | Accuracy | Baseline | Edge | Wilson lower |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        lines.append(f"| {r['market']} | {r['horizon_days']} | {r['cases']} | {r['bull_cases']}/{r['bear_cases']} | {r['accuracy']:.2%} | {r['majority_baseline']:.2%} | {r['edge']:.2%} | {r['wilson_95_lower']:.2%} |")
    (out / "cycle_consolidation_research.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
