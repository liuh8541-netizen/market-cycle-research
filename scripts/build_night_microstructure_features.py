"""Build one causal microstructure row per completed TX night session."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BARS = ROOT / "data" / "processed" / "factors" / "futures_tick_15m.csv"
NIGHT = ROOT / "data" / "processed" / "taiwan_futures_night.csv"
OUTPUT = ROOT / "data" / "processed" / "factors" / "night_microstructure.csv"
AUDIT = ROOT / "reports" / "night_microstructure_build_audit.json"


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator and np.isfinite(denominator) else np.nan


def within_ohlc_tolerance(features, canonical, tolerance=1.0):
    differences = [
        features["_path_open"] - canonical.tx_night_open,
        features["_path_high"] - canonical.tx_night_high,
        features["_path_low"] - canonical.tx_night_low,
        features["_path_close"] - canonical.tx_night_close,
    ]
    return max(abs(value) for value in differences) <= tolerance


def session_features(signal_date, prior_cash_date, contract_date, frame):
    start = prior_cash_date + pd.Timedelta(hours=15)
    end = prior_cash_date + pd.Timedelta(days=1, hours=5)
    session = frame[
        frame["contract_date"].eq(str(contract_date))
        & frame["bar_start"].ge(start) & frame["bar_start"].lt(end)
    ].sort_values("bar_start")
    complete_start = len(session) and session["bar_start"].iloc[0] <= start + pd.Timedelta(minutes=30)
    complete_end = len(session) and session["bar_start"].iloc[-1] >= end - pd.Timedelta(minutes=30)
    if len(session) < 12 or not complete_start or not complete_end:
        return None
    prices = np.r_[session["open"].iloc[0], session["close"].to_numpy(float)]
    returns = np.diff(np.log(prices))
    first = session[session["bar_start"] < start + pd.Timedelta(hours=1)]
    last = session[session["bar_start"] >= end - pd.Timedelta(hours=1)]
    full_return = session["close"].iloc[-1] / session["open"].iloc[0] - 1
    first_return = first["close"].iloc[-1] / first["open"].iloc[0] - 1 if len(first) else np.nan
    last_return = last["close"].iloc[-1] / last["open"].iloc[0] - 1 if len(last) else np.nan
    cumulative = np.cumprod(1 + np.r_[0, np.exp(returns) - 1])
    running_max = np.maximum.accumulate(cumulative)
    running_min = np.minimum.accumulate(cumulative)
    volume = float(session["volume"].sum())
    return {
        "signal_date": signal_date.strftime("%Y-%m-%d"), "contract_date": str(contract_date),
        "_path_open": float(session["open"].iloc[0]),
        "_path_high": float(session["high"].max()),
        "_path_low": float(session["low"].min()),
        "_path_close": float(session["close"].iloc[-1]),
        "bar_count": len(session), "tick_count": int(session["ticks"].sum()),
        "night_path_return": full_return, "first_hour_return": first_return,
        "last_hour_return": last_return, "return_acceleration": last_return - first_return,
        "realized_volatility": float(np.sqrt(np.square(returns).sum())),
        "trend_efficiency": safe_ratio(abs(full_return), float(np.abs(np.exp(returns) - 1).sum())),
        "up_bar_share": float((returns > 0).mean()),
        "close_location": safe_ratio(
            session["close"].iloc[-1] - session["low"].min(),
            session["high"].max() - session["low"].min(),
        ),
        "max_drawdown": float(np.min(cumulative / running_max - 1)),
        "max_runup": float(np.max(cumulative / running_min - 1)),
        "first_hour_volume_share": safe_ratio(float(first["volume"].sum()), volume),
        "last_hour_volume_share": safe_ratio(float(last["volume"].sum()), volume),
        "signed_volume_ratio": safe_ratio(float(session["signed_volume"].sum()), volume),
        "session_volume": volume,
    }


def main():
    if not BARS.exists():
        raise SystemExit(f"Tick bars not found: {BARS}")
    bars = pd.read_csv(BARS)
    bars["bar_start"] = pd.to_datetime(bars["bar_start"])
    bars["contract_date"] = bars["contract_date"].astype(str)
    bars = bars.drop_duplicates(["contract_date", "bar_start"], keep="last").sort_values("bar_start")
    night = pd.read_csv(NIGHT, usecols=[
        "signal_date", "contract_date", "tx_night_open", "tx_night_high",
        "tx_night_low", "tx_night_close",
    ])
    night["signal_date"] = pd.to_datetime(night["signal_date"])
    night["contract_date"] = night["contract_date"].astype(str)
    cash = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv", usecols=["date"])
    cash_dates = pd.DatetimeIndex(pd.to_datetime(cash["date"]).dropna().drop_duplicates().sort_values())
    rows = []
    excluded_ohlc_beyond_one_tick = 0
    for row in night.itertuples():
        prior = cash_dates[cash_dates < row.signal_date]
        if not len(prior):
            continue
        features = session_features(row.signal_date, prior[-1], row.contract_date, bars)
        if features:
            if not within_ohlc_tolerance(features, row):
                excluded_ohlc_beyond_one_tick += 1
                continue
            for key in ["_path_open", "_path_high", "_path_low", "_path_close"]:
                features.pop(key)
            rows.append(features)
    output = pd.DataFrame(rows).sort_values("signal_date") if rows else pd.DataFrame()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False, encoding="utf-8")
    audit = {
        "bar_rows": len(bars), "feature_rows": len(output),
        "first_signal_date": output["signal_date"].min() if len(output) else None,
        "last_signal_date": output["signal_date"].max() if len(output) else None,
        "minimum_bars_required": 12,
        "excluded_ohlc_beyond_one_tick": excluded_ohlc_beyond_one_tick,
        "completeness_rule": "first bar by P 15:30 and last bar at or after P+1 04:30",
        "causal_window": "prior TWII cash trading date P 15:00 inclusive through calendar P+1 05:00 exclusive",
        "duplicate_contract_bar_rows_after_dedup": int(bars.duplicated(["contract_date", "bar_start"]).sum()),
    }
    AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
