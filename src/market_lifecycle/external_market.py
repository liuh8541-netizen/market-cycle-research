from __future__ import annotations

import json
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd


TAIPEI = ZoneInfo("Asia/Taipei")
FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"

EXTERNAL_SYMBOLS = {
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "sox": "^SOX",
    "dow": "^DJI",
    "vix": "^VIX",
    "tsm_adr": "TSM",
    "micron": "MU",
    # Direct overseas Taiwan-market price discovery and its currency bridge.
    "ewt": "EWT",
    "usd_twd": "TWD=X",
    "samsung": "005930.KS",
    "sk_hynix": "000660.KS",
    # Completed U.S. Treasury yield-index observations. These are retained as
    # contextual rate-pressure diagnostics until a point-in-time intraday
    # history passes the registered validation gate.
    "treasury_5y": "^FVX",
    "treasury_10y": "^TNX",
    "treasury_30y": "^TYX",
}


def update_external_markets(output_path: Path, raw_dir: Path) -> dict:
    result = {
        "attempted": True,
        "success": False,
        "source": "Yahoo Finance external markets",
        "symbols": EXTERNAL_SYMBOLS,
        "latest_date": None,
        "row_count": 0,
        "message": "",
    }
    frames = []
    errors = []
    excluded_live_symbols = []

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for name, symbol in EXTERNAL_SYMBOLS.items():
        try:
            raw = fetch_yahoo_chart(symbol)
            (raw_dir / f"yahoo_{name}_chart.json").write_text(
                json.dumps(raw, ensure_ascii=False),
                encoding="utf-8",
            )
            frame = chart_to_close_frame(raw, name)
            frame, live_excluded = exclude_live_external_session(frame, name, datetime.now(timezone.utc))
            if live_excluded:
                excluded_live_symbols.append(name)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if not frames:
        result["message"] = "External update failed; no usable symbol data. " + "; ".join(errors)
        return result

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)
    for name in EXTERNAL_SYMBOLS:
        close_col = f"{name}_close"
        if close_col in merged:
            # The outer calendar contains dates from several countries. Compute
            # each return only across that symbol's own non-null observations;
            # otherwise a holiday row makes the next genuine return disappear.
            valid = merged[close_col].notna()
            merged[f"{name}_return_1d"] = pd.NA
            merged.loc[valid, f"{name}_return_1d"] = (
                pd.to_numeric(merged.loc[valid, close_col], errors="coerce").pct_change(fill_method=None)
            )

    merged.to_csv(output_path, index=False, encoding="utf-8")
    result.update(
        {
            "success": True,
            "latest_date": str(merged.iloc[-1]["date"]),
            "row_count": int(len(merged)),
            "message": "External market data updated." + (f" Partial errors: {'; '.join(errors)}" if errors else ""),
            "excluded_live_symbols": excluded_live_symbols,
        }
    )
    if excluded_live_symbols:
        result["message"] += " Incomplete live rows excluded for: " + ", ".join(excluded_live_symbols) + "."
    return result


def exclude_live_external_session(frame: pd.DataFrame, name: str, now_utc: datetime) -> tuple[pd.DataFrame, bool]:
    """Remove a symbol's current, not-yet-final daily observation."""
    if frame.empty or "date" not in frame:
        return frame, False
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.date
    us_symbols = {
        "sp500", "nasdaq", "sox", "dow", "vix", "tsm_adr", "micron", "ewt",
        "treasury_5y", "treasury_10y", "treasury_30y",
    }
    korea_symbols = {"samsung", "sk_hynix"}
    if name in us_symbols:
        local = now_utc.astimezone(ZoneInfo("America/New_York"))
        unfinished = dates.eq(local.date()) & (local.time() < time(16, 15))
    elif name in korea_symbols:
        local = now_utc.astimezone(ZoneInfo("Asia/Seoul"))
        unfinished = dates.eq(local.date()) & (local.time() < time(16, 0))
    elif name == "usd_twd":
        # Yahoo's current FX daily candle is continuously revised. Admit it only
        # on a later Taiwan calendar day, matching the research's strict D-1 rule.
        local = now_utc.astimezone(TAIPEI)
        unfinished = dates >= local.date()
    else:
        return frame, False
    return frame.loc[~unfinished].copy(), bool(unfinished.any())


def update_taiwan_futures_night(output_path: Path, start_date: str, end_date: str, token: str = "") -> dict:
    result = {
        "attempted": True,
        "success": False,
        "source": "FinMind TaiwanFuturesDaily TX after_market",
        "latest_date": None,
        "row_count": 0,
        "message": "",
    }
    try:
        query_end = pd.to_datetime(end_date).date()
        while query_end.weekday() >= 5:
            query_end += pd.Timedelta(days=1)
        existing = pd.DataFrame()
        fetch_start_date = start_date
        if output_path.exists():
            existing = pd.read_csv(output_path)
            if not existing.empty and "night_date" in existing:
                latest = pd.to_datetime(existing["night_date"]).max()
                fetch_start_date = (latest - pd.Timedelta(days=5)).date().isoformat()

        params = {
            "dataset": "TaiwanFuturesDaily",
            "data_id": "TX",
            "start_date": fetch_start_date,
            "end_date": query_end.isoformat(),
        }
        if token:
            params["token"] = token
        payload = fetch_finmind(params)
        rows = payload.get("data") or []
        if not rows:
            raise ValueError("No Taiwan futures rows returned.")

        frame = pd.DataFrame(rows)
        frame = frame[frame.get("trading_session", "") == "after_market"].copy()
        if frame.empty:
            raise ValueError("No after_market rows returned.")

        for col in ["open", "max", "min", "close", "spread_per", "volume"]:
            if col in frame:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values(["date", "volume"]).groupby("date").tail(1).copy()
        # FinMind labels TX after-market data with the regular-session date it informs.
        # Use the same date as the pre-market signal date.
        frame["signal_date"] = frame["date"]
        frame["tx_night_return"] = frame["close"] / frame["open"] - 1
        frame["tx_night_range"] = frame["max"] / frame["min"] - 1
        frame["tx_night_spread_per"] = frame["spread_per"] / 100
        output = frame[
            [
                "date",
                "signal_date",
                "contract_date",
                "open",
                "max",
                "min",
                "close",
                "volume",
                "tx_night_return",
                "tx_night_range",
                "tx_night_spread_per",
            ]
        ].rename(
            columns={
                "date": "night_date",
                "open": "tx_night_open",
                "max": "tx_night_high",
                "min": "tx_night_low",
                "close": "tx_night_close",
                "volume": "tx_night_volume",
            }
        )
        output["night_date"] = output["night_date"].dt.date.astype(str)
        output["signal_date"] = output["signal_date"].dt.date.astype(str)
        if not existing.empty:
            output = pd.concat([existing, output], ignore_index=True)
            output = output.drop_duplicates(subset=["night_date"], keep="last")
            output = output.sort_values("night_date").reset_index(drop=True)

        # FinMind TaiwanFuturesDaily already attributes after_market to trading day D:
        # prior business day 15:00 through D 05:00. Normalize every retained legacy
        # row, not only newly downloaded rows, so old next-calendar-day mappings
        # cannot survive an incremental merge.
        output["signal_date"] = pd.to_datetime(output["night_date"]).dt.date.astype(str)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8")
        result.update(
            {
                "success": True,
                "latest_date": str(output.iloc[-1]["night_date"]),
                "row_count": int(len(output)),
                "message": "Taiwan futures night-session data updated.",
            }
        )
    except Exception as exc:
        result["message"] = f"Night futures update failed; using local CSV if available. Reason: {exc}"
    return result


def add_external_market_features(
    price: pd.DataFrame,
    external_path: Path | None,
    night_futures_path: Path | None = None,
) -> pd.DataFrame:
    data = price.copy()
    data["date"] = pd.to_datetime(data["date"])

    data = add_intraday_truth_features(data)
    if external_path is None or not external_path.exists():
        data["external_score"] = 0
        data["external_pressure_label"] = "unknown"
        return add_taiwan_night_futures_features(data, night_futures_path)

    external = pd.read_csv(external_path)
    if external.empty or "date" not in external:
        data["external_score"] = 0
        data["external_pressure_label"] = "unknown"
        return add_taiwan_night_futures_features(data, night_futures_path)

    external["external_date"] = pd.to_datetime(external["date"])
    external = external.drop(columns=["date"]).sort_values("external_date")
    data = pd.merge_asof(
        data.sort_values("date"),
        external,
        left_on="date",
        right_on="external_date",
        direction="backward",
    )
    data["external_score"] = data.apply(external_pressure_score, axis=1)
    data["external_pressure_label"] = data["external_score"].apply(external_pressure_label)
    return add_taiwan_night_futures_features(data, night_futures_path)


def add_taiwan_night_futures_features(data: pd.DataFrame, night_futures_path: Path | None) -> pd.DataFrame:
    output = data.copy()
    if night_futures_path is None or not night_futures_path.exists():
        output["night_futures_score"] = 0
        output["night_futures_label"] = "unknown"
        return output

    futures = pd.read_csv(night_futures_path)
    if futures.empty or "signal_date" not in futures:
        output["night_futures_score"] = 0
        output["night_futures_label"] = "unknown"
        return output

    futures["signal_date"] = pd.to_datetime(futures["signal_date"])
    futures = futures.sort_values("signal_date")
    output = pd.merge_asof(
        output.sort_values("date"),
        futures,
        left_on="date",
        right_on="signal_date",
        direction="backward",
        tolerance=pd.Timedelta(days=3),
    )
    output["tx_night_gap_vs_spot"] = output["tx_night_close"] / output["close"].shift(1) - 1
    output["night_futures_raw_score"] = output.apply(night_futures_score, axis=1)
    # Formal walk-forward tests found no directional edge versus the cash
    # close or opening gap. Preserve raw observations, but do not add them to
    # lifecycle direction until a future validation gate passes.
    output["night_futures_score"] = 0
    output["night_futures_label"] = output["night_futures_score"].apply(night_futures_label)
    return output


def add_intraday_truth_features(data: pd.DataFrame) -> pd.DataFrame:
    output = data.sort_values("date").reset_index(drop=True).copy()
    prev_close = output["close"].shift(1)
    daily_range = (output["high"] - output["low"]).replace(0, pd.NA)

    output["open_gap_pct"] = output["open"] / prev_close - 1
    output["intraday_low_pct"] = output["low"] / output["open"] - 1
    output["intraday_high_pct"] = output["high"] / output["open"] - 1
    output["close_return_pct"] = output["close"] / prev_close - 1
    output["close_recovery_ratio"] = ((output["close"] - output["low"]) / daily_range).fillna(0)
    output["panic_reversal_score"] = output.apply(panic_reversal_score, axis=1)
    output["intraday_truth_label"] = output.apply(intraday_truth_label, axis=1)
    return output


def panic_reversal_score(row) -> int:
    low_pct = row.get("intraday_low_pct")
    recovery = row.get("close_recovery_ratio")
    close_ret = row.get("close_return_pct")
    gap = row.get("open_gap_pct")
    if pd.isna(low_pct) or pd.isna(recovery) or pd.isna(close_ret):
        return 0
    if low_pct <= -0.02 and recovery >= 0.70 and close_ret >= -0.005:
        return 2
    if low_pct <= -0.015 and recovery >= 0.60:
        return 1
    if gap <= -0.015 and recovery < 0.35 and close_ret < -0.01:
        return -2
    if low_pct <= -0.02 and recovery < 0.35:
        return -1
    return 0


def intraday_truth_label(row) -> str:
    score = row.get("panic_reversal_score", 0)
    low_pct = row.get("intraday_low_pct")
    recovery = row.get("close_recovery_ratio")
    if score >= 2:
        return "panic_washout_strong_recovery"
    if score == 1:
        return "panic_washout_recovery"
    if score <= -2:
        return "gap_down_failed_recovery"
    if score == -1:
        return "panic_no_recovery"
    if pd.notna(low_pct) and low_pct <= -0.015:
        return "intraday_panic"
    if pd.notna(recovery) and recovery >= 0.70:
        return "strong_close"
    return "normal"


def external_pressure_score(row) -> int:
    score = 0
    weights = {
        "nasdaq_return_1d": (-0.018, 0.012, 1),
        "sox_return_1d": (-0.025, 0.018, 2),
        "sp500_return_1d": (-0.014, 0.010, 1),
        "tsm_adr_return_1d": (-0.025, 0.018, 2),
        "vix_return_1d": (0.10, -0.06, 1),
    }
    for col, (bearish, bullish, weight) in weights.items():
        value = row.get(col)
        if pd.isna(value):
            continue
        if col == "vix_return_1d":
            if value >= bearish:
                score -= weight
            elif value <= bullish:
                score += weight
        else:
            if value <= bearish:
                score -= weight
            elif value >= bullish:
                score += weight
    return int(max(-4, min(4, score)))


def external_pressure_label(score: int) -> str:
    if score <= -3:
        return "heavy_external_pressure"
    if score < 0:
        return "external_pressure"
    if score >= 3:
        return "strong_external_tailwind"
    if score > 0:
        return "external_tailwind"
    return "neutral"


def night_futures_score(row) -> int:
    night_return = row.get("tx_night_return")
    gap_vs_spot = row.get("tx_night_gap_vs_spot")
    score = 0
    for value in [night_return, gap_vs_spot]:
        if pd.isna(value):
            continue
        if value <= -0.015:
            score -= 2
        elif value <= -0.007:
            score -= 1
        elif value >= 0.015:
            score += 2
        elif value >= 0.007:
            score += 1
    return int(max(-3, min(3, score)))


def night_futures_label(score: int) -> str:
    if score <= -2:
        return "night_futures_heavy_pressure"
    if score < 0:
        return "night_futures_pressure"
    if score >= 2:
        return "night_futures_strong_tailwind"
    if score > 0:
        return "night_futures_tailwind"
    return "neutral"


def fetch_finmind(params: dict) -> dict:
    url = FINMIND_API_URL + "?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "market-lifecycle-research/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") not in [200, "200", None]:
        raise RuntimeError(f"FinMind failed: {payload}")
    return payload


def fetch_yahoo_chart(symbol: str) -> dict:
    period1 = int(datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp())
    period2 = int((datetime.now(timezone.utc) + timedelta(days=2)).timestamp())
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(symbol)}?period1={period1}&period2={period2}&interval=1d&events=history"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def chart_to_close_frame(raw: dict, name: str) -> pd.DataFrame:
    chart = raw.get("chart", {})
    if chart.get("error"):
        raise ValueError(chart["error"])
    results = chart.get("result") or []
    if not results:
        raise ValueError("Yahoo response has no result.")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_data = (result.get("indicators", {}).get("quote") or [{}])[0]
    closes = quote_data.get("close") or []

    rows = []
    for index, ts in enumerate(timestamps):
        close = value_at(closes, index)
        if close is None:
            continue
        rows.append(
            {
                "date": datetime.fromtimestamp(ts, TAIPEI).date().isoformat(),
                f"{name}_close": close,
            }
        )
    return pd.DataFrame(rows)


def value_at(values: list | None, index: int):
    if not values or index >= len(values):
        return None
    value = values[index]
    if value is None:
        return None
    return float(value)
