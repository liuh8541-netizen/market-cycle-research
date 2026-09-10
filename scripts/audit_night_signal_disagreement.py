"""Feasibility audit for choosing between night spread and night return signs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    cash = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv", usecols=["date", "open", "close"])
    cash["date"] = pd.to_datetime(cash["date"])
    cash = cash.sort_values("date").drop_duplicates("date", keep="last")
    cash["gap_return"] = pd.to_numeric(cash["open"], errors="coerce") / pd.to_numeric(cash["close"], errors="coerce").shift() - 1
    night = pd.read_csv(ROOT / "data" / "processed" / "taiwan_futures_night.csv")
    night["date"] = pd.to_datetime(night["signal_date"])
    frame = cash.merge(night, on="date", how="inner").dropna(
        subset=["gap_return", "tx_night_spread_per", "tx_night_return"]
    ).sort_values("date")
    frame["spread"] = np.where(frame["tx_night_spread_per"] >= 0, 1, -1)
    frame["return"] = np.where(frame["tx_night_return"] >= 0, 1, -1)
    frame["target"] = np.where(frame["gap_return"] >= 0, 1, -1)
    disagreement = frame.loc[frame["spread"] != frame["return"]].copy()
    by_year = []
    for year, group in disagreement.groupby(disagreement["date"].dt.year):
        by_year.append({
            "year": int(year), "cases": len(group),
            "spread_hits": int((group["spread"] == group["target"]).sum()),
            "return_hits": int((group["return"] == group["target"]).sum()),
        })
    payload = {
        "question": "Can a separate model choose between night spread and night return only when their signs disagree?",
        "aligned_days": len(frame), "disagreement_cases": len(disagreement),
        "disagreement_rate": len(disagreement) / len(frame),
        "spread_hits": int((disagreement["spread"] == disagreement["target"]).sum()),
        "return_hits": int((disagreement["return"] == disagreement["target"]).sum()),
        "date_start": str(disagreement["date"].min().date()),
        "date_end": str(disagreement["date"].max().date()),
        "by_year": by_year,
        "decision": "not_testable_at_required_power",
        "reason": "Only 111 total disagreement events exist. Any honest training period leaves fewer than the required 100 OOS cases, and historical wins are nearly tied 54/57.",
    }
    out = ROOT / "reports"
    (out / "night_signal_disagreement_feasibility.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Night signal disagreement feasibility", "", payload["question"], "",
        f"Aligned days: {payload['aligned_days']}. Sign disagreements: {payload['disagreement_cases']} "
        f"({payload['disagreement_rate']:.2%}), from {payload['date_start']} to {payload['date_end']}.",
        f"On disagreements, spread won {payload['spread_hits']} and return won {payload['return_hits']}.",
        "", "Decision: not testable at the registered sample requirement.",
        payload["reason"], "", "| Year | Cases | Spread hits | Return hits |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in by_year:
        lines.append(f"| {row['year']} | {row['cases']} | {row['spread_hits']} | {row['return_hits']} |")
    (out / "night_signal_disagreement_feasibility.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "by_year"}, indent=2))


if __name__ == "__main__":
    main()
