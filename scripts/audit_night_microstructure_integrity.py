"""Audit reconstructed tick sessions against canonical FinMind night OHLC."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BARS = ROOT / "data" / "processed" / "factors" / "futures_tick_15m.csv"
NIGHT = ROOT / "data" / "processed" / "taiwan_futures_night.csv"
REPORT_JSON = ROOT / "reports" / "night_microstructure_integrity_audit.json"
REPORT_MD = ROOT / "reports" / "night_microstructure_integrity_audit.md"


def main():
    bars = pd.read_csv(BARS)
    bars["bar_start"] = pd.to_datetime(bars["bar_start"])
    bars["contract_date"] = bars["contract_date"].astype(str)
    duplicate_rows = int(bars.duplicated(["contract_date", "bar_start"]).sum())
    bars = bars.drop_duplicates(["contract_date", "bar_start"], keep="last")
    night = pd.read_csv(NIGHT)
    night["signal_date"] = pd.to_datetime(night["signal_date"])
    night["contract_date"] = night["contract_date"].astype(str)
    cash_dates = pd.DatetimeIndex(
        pd.to_datetime(pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv", usecols=["date"])["date"])
        .dropna().drop_duplicates().sort_values()
    )
    rows = []
    for row in night.itertuples():
        prior = cash_dates[cash_dates < row.signal_date]
        if not len(prior):
            continue
        start = prior[-1] + pd.Timedelta(hours=15)
        end = prior[-1] + pd.Timedelta(days=1, hours=5)
        session = bars[
            bars["contract_date"].eq(row.contract_date)
            & bars["bar_start"].ge(start) & bars["bar_start"].lt(end)
        ].sort_values("bar_start")
        complete = (
            len(session) >= 12
            and session["bar_start"].iloc[0] <= start + pd.Timedelta(minutes=30)
            and session["bar_start"].iloc[-1] >= end - pd.Timedelta(minutes=30)
        )
        if not complete:
            continue
        tick_volume = float(session["volume"].sum())
        daily_volume = float(row.tx_night_volume)
        rows.append({
            "signal_date": str(row.signal_date.date()), "contract_date": row.contract_date,
            "session_start": start.isoformat(), "session_end": end.isoformat(),
            "bars": len(session),
            "open_diff": float(session["open"].iloc[0] - row.tx_night_open),
            "high_diff": float(session["high"].max() - row.tx_night_high),
            "low_diff": float(session["low"].min() - row.tx_night_low),
            "close_diff": float(session["close"].iloc[-1] - row.tx_night_close),
            "tick_to_daily_volume_ratio": tick_volume / daily_volume if daily_volume else None,
            "causal_violation": bool(end > row.signal_date + pd.Timedelta(hours=8, minutes=45)),
        })
    frame = pd.DataFrame(rows)
    ohlc_columns = ["open_diff", "high_diff", "low_diff", "close_diff"]
    exact = int(frame[ohlc_columns].eq(0).all(axis=1).sum()) if len(frame) else 0
    within_one_tick = int(frame[ohlc_columns].abs().le(1.0).all(axis=1).sum()) if len(frame) else 0
    ratios = frame["tick_to_daily_volume_ratio"].dropna() if len(frame) else pd.Series(dtype=float)
    payload = {
        "complete_sessions": len(frame), "exact_ohlc_sessions": exact,
        "ohlc_mismatches": len(frame) - exact,
        "ohlc_within_one_tick_sessions": within_one_tick,
        "ohlc_mismatches_beyond_one_tick": len(frame) - within_one_tick,
        "ohlc_tolerance_policy": "TAIFEX TX minimum price tick is 1 index point. Last-tick and daily-close conventions may differ by one tick; more than one point is rejected.",
        "duplicate_contract_bar_rows": duplicate_rows,
        "causal_violations": int(frame["causal_violation"].sum()) if len(frame) else 0,
        "first_signal_date": frame["signal_date"].min() if len(frame) else None,
        "last_signal_date": frame["signal_date"].max() if len(frame) else None,
        "tick_to_daily_volume_ratio": {
            "minimum": float(ratios.min()) if len(ratios) else None,
            "median": float(ratios.median()) if len(ratios) else None,
            "maximum": float(ratios.max()) if len(ratios) else None,
        },
        "volume_policy": "Tick and daily volume have different aggregation bases. Use tick volume only for within-session shares and signed ratios; retain canonical daily volume for level features.",
        "passed": bool(len(frame) and within_one_tick == len(frame) and duplicate_rows == 0 and not frame["causal_violation"].any()),
        "records": rows,
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ratio = payload["tick_to_daily_volume_ratio"]
    lines = [
        "# Night microstructure integrity audit", "",
        f"Complete sessions: {len(frame)}; exact tick-to-daily OHLC: {exact}/{len(frame)}; "
        f"within one TX tick: {within_one_tick}/{len(frame)}; beyond-one-tick mismatches: {len(frame)-within_one_tick}; "
        f"causal violations: {payload['causal_violations']}; "
        f"duplicate contract-bars: {duplicate_rows}.",
        f"Tick/daily volume ratio: min {ratio['minimum']}, median {ratio['median']}, max {ratio['maximum']}.",
        payload["ohlc_tolerance_policy"], payload["volume_policy"], "", f"Passed: {payload['passed']}.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
