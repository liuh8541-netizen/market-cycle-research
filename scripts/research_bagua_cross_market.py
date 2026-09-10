import json
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from research_bagua_cycle import build_multiframe_states
from research_selective_signals import wilson_lower


HORIZON = 120
SUPPORTIVE_MONTHLY = {"XUN", "LI", "KUN", "DUI", "QIAN"}
ENTRY_WEEKLY = {"XUN", "LI"}
MARKETS = {
    "SP500": "sp500_close",
    "NASDAQ": "nasdaq_close",
    "SOX": "sox_close",
    "DOW": "dow_close",
}


def close_only_frame(source, column):
    frame = source[["date", column]].dropna().rename(columns={column: "close"}).copy()
    frame["open"] = frame["close"]
    frame["high"] = frame["close"]
    frame["low"] = frame["close"]
    return frame[["date", "open", "high", "low", "close"]]


def evaluate_market(name, price):
    states = build_multiframe_states(price)
    states["prior_weekly"] = states["weekly_state"].shift()
    changed = states["weekly_state"].ne(states["prior_weekly"])
    rule = (
        changed
        & states["weekly_state"].isin(ENTRY_WEEKLY)
        & states["monthly_state"].isin(SUPPORTIVE_MONTHLY)
    )
    states["future_return"] = states["close"].shift(-HORIZON) / states["close"] - 1
    sample = states.loc[rule].dropna(subset=["future_return"]).copy()
    sample["hit"] = sample["future_return"] > 0
    cases = len(sample)
    hits = int(sample["hit"].sum())
    return {
        "market": name,
        "cases": cases,
        "hits": hits,
        "accuracy": hits / cases if cases else 0.0,
        "wilson_95_lower": wilson_lower(hits, cases),
        "median_return": float(sample["future_return"].median()) if cases else None,
        "events": [
            {
                "date": str(row.date.date()),
                "weekly_state": row.weekly_state,
                "monthly_state": row.monthly_state,
                "future_return": float(row.future_return),
                "hit": bool(row.hit),
            }
            for row in sample.itertuples()
        ],
    }


def summarize(results):
    cases = sum(item["cases"] for item in results)
    hits = sum(item["hits"] for item in results)
    accuracy = hits / cases if cases else 0.0
    return {
        "cases": cases,
        "hits": hits,
        "accuracy": accuracy,
        "wilson_95_lower": wilson_lower(hits, cases),
        "passed": accuracy >= 0.90 and cases >= 100 and wilson_lower(hits, cases) >= 0.80,
        "warning": "Markets are correlated; pooled cases are not fully independent.",
    }


def render(payload):
    lines = [
        "# Bagua cycle cross-market validation",
        "",
        "Fixed rule: after a weekly transition into XUN/LI, while the monthly state is supportive, predict a positive 120-trading-day return.",
        "",
        "These markets were not used to discover this specific hierarchical rule.",
        "",
        "| Market | Cases | Hits | Accuracy | Wilson 95% lower | Median return |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["markets"]:
        lines.append(
            f"| {item['market']} | {item['cases']} | {item['hits']} | {item['accuracy']:.2%} | "
            f"{item['wilson_95_lower']:.2%} | {item['median_return']:.2%} |"
        )
    pooled = payload["pooled"]
    lines.extend([
        "",
        f"Pooled: {pooled['hits']}/{pooled['cases']} = {pooled['accuracy']:.2%}; "
        f"Wilson lower {pooled['wilson_95_lower']:.2%}; passed={pooled['passed']}.",
        "",
        "Caution: the pooled count is descriptive because equity indexes are correlated.",
    ])
    return "\n".join(lines) + "\n"


def main():
    external = pd.read_csv(ROOT / "data" / "processed" / "external_markets.csv")
    external["date"] = pd.to_datetime(external["date"])
    results = [evaluate_market(name, close_only_frame(external, column)) for name, column in MARKETS.items()]
    payload = {
        "status": "independent_validation",
        "primary_horizon_days": HORIZON,
        "entry_weekly_states": sorted(ENTRY_WEEKLY),
        "supportive_monthly_states": sorted(SUPPORTIVE_MONTHLY),
        "markets": results,
        "pooled": summarize(results),
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "bagua_cross_market_validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "bagua_cross_market_validation.md").write_text(render(payload), encoding="utf-8")
    print(json.dumps({"markets": [{k: v for k, v in x.items() if k != "events"} for x in results], "pooled": payload["pooled"]}, indent=2))


if __name__ == "__main__":
    main()
