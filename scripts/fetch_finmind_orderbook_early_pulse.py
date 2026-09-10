"""Incrementally compress 5-second Taiwan market order/trade totals through 09:15."""

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
API = "https://api.finmindtrade.com/api/v4/data"
DATASET = "TaiwanStockStatisticsOfOrderBookAndTrade"
OUT = ROOT / "data/processed/factors/orderbook_early_pulse.csv"
MANIFEST = ROOT / "data/processed/factors/orderbook_early_pulse_fetched_dates.csv"
LOCK = ROOT / "data/processed/factors/.orderbook_early_pulse_fetch.lock"
TAIPEI = timezone(timedelta(hours=8))
CUTOFF = "09:15:00"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=str(date.today()))
    parser.add_argument("--request-pause", type=float, default=0.25)
    return parser.parse_args()


def fetch(day, token):
    params = {"dataset": DATASET, "start_date": day, "token": token}
    request = Request(
        API + "?" + urlencode(params),
        headers={"User-Agent": "market-lifecycle-research/1.0"},
    )
    error = None
    for attempt in range(5):
        try:
            with urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") not in (200, "200", None):
                raise RuntimeError(str(payload))
            return payload.get("data", [])
        except (TimeoutError, OSError) as exc:
            error = exc
            if attempt < 4:
                time.sleep(3 * (attempt + 1))
    raise error


def safe_imbalance(buy, sell):
    total = buy + sell
    return (buy - sell) / total if total > 0 else np.nan


def safe_log_ratio(left, right):
    return float(np.log((left + 1.0) / (right + 1.0)))


def summarize(rows, requested_day):
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    required = {
        "date", "Time", "TotalBuyOrder", "TotalBuyVolume", "TotalSellOrder",
        "TotalSellVolume", "TotalDealOrder", "TotalDealVolume", "TotalDealMoney",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Order-book schema changed; missing {sorted(required-set(frame.columns))}")
    frame = frame[frame["date"].astype(str) == requested_day].copy()
    frame["timestamp"] = pd.to_datetime(
        frame["date"].astype(str) + " " + frame["Time"].astype(str), errors="coerce"
    )
    numeric = sorted(required - {"date", "Time"})
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["timestamp"] + numeric).sort_values("timestamp")
    early = frame[frame["Time"].astype(str) <= CUTOFF]
    if len(early) < 120:
        return None
    first, last = early.iloc[0], early.iloc[-1]
    at_0905 = early[early["Time"].astype(str) <= "09:05:00"].iloc[-1]
    buy_size = last.TotalBuyVolume / last.TotalBuyOrder if last.TotalBuyOrder > 0 else np.nan
    sell_size = last.TotalSellVolume / last.TotalSellOrder if last.TotalSellOrder > 0 else np.nan
    def delta(column, left, right):
        return float(right[column] - left[column])
    return {
        "date": requested_day,
        "order_count_imbalance": safe_imbalance(last.TotalBuyOrder, last.TotalSellOrder),
        "order_volume_imbalance": safe_imbalance(last.TotalBuyVolume, last.TotalSellVolume),
        "order_size_log_ratio": safe_log_ratio(buy_size, sell_size),
        "order_count_imbalance_change": safe_imbalance(
            last.TotalBuyOrder, last.TotalSellOrder
        ) - safe_imbalance(first.TotalBuyOrder, first.TotalSellOrder),
        "order_volume_imbalance_change": safe_imbalance(
            last.TotalBuyVolume, last.TotalSellVolume
        ) - safe_imbalance(first.TotalBuyVolume, first.TotalSellVolume),
        "buy_order_growth": delta("TotalBuyOrder", first, last),
        "sell_order_growth": delta("TotalSellOrder", first, last),
        "buy_volume_growth": delta("TotalBuyVolume", first, last),
        "sell_volume_growth": delta("TotalSellVolume", first, last),
        "late_buy_order_growth": delta("TotalBuyOrder", at_0905, last),
        "late_sell_order_growth": delta("TotalSellOrder", at_0905, last),
        "late_buy_volume_growth": delta("TotalBuyVolume", at_0905, last),
        "late_sell_volume_growth": delta("TotalSellVolume", at_0905, last),
        "deal_order_15m": float(last.TotalDealOrder),
        "deal_volume_15m": float(last.TotalDealVolume),
        "deal_money_15m": float(last.TotalDealMoney),
        "observations": int(len(early)),
        "first_timestamp": str(first.timestamp),
        "cutoff_timestamp": str(last.timestamp),
    }


def merge_csv(path, new, key):
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([old, new], ignore_index=True) if not old.empty else new.copy()
    if not combined.empty:
        combined = combined.sort_values(key).drop_duplicates(key, keep="last")
        temporary = path.with_suffix(path.suffix + ".tmp")
        combined.to_csv(temporary, index=False)
        os.replace(temporary, path)
    return combined


def main():
    args = arguments()
    token = os.environ.get("FINMIND_TOKEN", "").strip()
    secret = ROOT / "config/.secrets/finmind_token.txt"
    if not token and secret.exists():
        token = secret.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("FinMind token not available")
    if LOCK.exists():
        raise RuntimeError(f"Order-book fetch lock exists: {LOCK}")
    LOCK.write_text(str(os.getpid()), encoding="ascii")
    try:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        manifest = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame(
            columns=["calendar_date", "status", "feature_rows", "source_rows", "fetched_at"]
        )
        done = set(manifest.loc[manifest.status == "complete", "calendar_date"].astype(str))
        cursor, finish = date.fromisoformat(args.start_date), date.fromisoformat(args.end_date)
        while cursor <= finish:
            day = str(cursor)
            if day in done:
                cursor += timedelta(days=1)
                continue
            weekend = cursor.weekday() >= 5
            print(
                f"orderbook_early_pulse: {'skipping weekend' if weekend else 'fetching'} {day}...",
                flush=True,
            )
            rows = [] if weekend else fetch(day, token)
            feature = summarize(rows, day)
            if feature:
                merge_csv(OUT, pd.DataFrame([feature]), "date")
            update = pd.DataFrame([{
                "calendar_date": day,
                "status": "complete",
                "feature_rows": int(feature is not None),
                "source_rows": len(rows),
                "fetched_at": datetime.now(TAIPEI).isoformat(),
            }])
            manifest = merge_csv(MANIFEST, update, "calendar_date")
            done.add(day)
            print(
                f"orderbook_early_pulse: {len(rows)} source rows -> {int(feature is not None)} daily row.",
                flush=True,
            )
            cursor += timedelta(days=1)
            time.sleep(args.request_pause)
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
