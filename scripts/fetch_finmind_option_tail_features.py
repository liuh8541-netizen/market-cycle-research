"""Incrementally compress FinMind TXO strike rows into causal night-tail features."""

import argparse
import json
import math
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
OUT = ROOT / "data/processed/factors/option_tail_night.csv"
MANIFEST = ROOT / "data/processed/factors/option_tail_fetched_dates.csv"
LOCK = ROOT / "data/processed/factors/.option_tail_fetch.lock"
TAIPEI = timezone(timedelta(hours=8))


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2020-01-01")
    p.add_argument("--end-date", default=str(date.today()))
    p.add_argument("--chunk-days", type=int, default=7)
    p.add_argument("--request-pause", type=float, default=1.2)
    return p.parse_args()


def fetch(start, end, token):
    params = {
        "dataset": "TaiwanOptionDaily", "data_id": "TXO",
        "start_date": start, "end_date": end, "token": token,
    }
    req = Request(API + "?" + urlencode(params), headers={"User-Agent": "market-lifecycle-research/1.0"})
    error = None
    for attempt in range(3):
        try:
            with urlopen(req, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") not in (200, "200", None):
                raise RuntimeError(str(payload))
            return payload.get("data", [])
        except (TimeoutError, OSError) as exc:
            error = exc
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
    raise error


def positive_price(frame):
    settlement = pd.to_numeric(frame["settlement_price"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    return settlement.where(settlement > 0, close.where(close > 0))


def nearest_value(table, side, target):
    eligible = table[(table["call_put"] == side) & table["price"].notna()]
    if eligible.empty:
        return np.nan
    row = eligible.iloc[(eligible["strike_price"] - target).abs().argmin()]
    return float(row["price"])


def summarize_contract(group):
    data = group.copy()
    data["strike_price"] = pd.to_numeric(data["strike_price"], errors="coerce")
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce").fillna(0)
    data["open_interest"] = pd.to_numeric(data["open_interest"], errors="coerce").fillna(0)
    data["price"] = positive_price(data)
    data["call_put"] = data["call_put"].astype(str).str.lower().map(
        lambda x: "put" if x in ("put", "p", "賣權") else "call"
    )
    data = data.dropna(subset=["strike_price"])
    pivot = data.pivot_table(index="strike_price", columns="call_put", values="price", aggfunc="last")
    if not {"call", "put"}.issubset(pivot.columns):
        return None
    both = pivot.dropna(subset=["call", "put"])
    both = both[(both["call"] > 0) & (both["put"] > 0)]
    if len(both) < 5:
        return None
    atm = float((both["call"] - both["put"]).abs().idxmin())
    p1, c1 = nearest_value(data, "put", atm * 0.99), nearest_value(data, "call", atm * 1.01)
    p2, c2 = nearest_value(data, "put", atm * 0.98), nearest_value(data, "call", atm * 1.02)
    if not all(np.isfinite(x) and x > 0 for x in [p1, c1, p2, c2]):
        return None
    put_otm = float(data.loc[(data.call_put == "put") & (data.strike_price < atm), "volume"].sum())
    call_otm = float(data.loc[(data.call_put == "call") & (data.strike_price > atm), "volume"].sum())
    put_oi = float(data.loc[(data.call_put == "put") & (data.strike_price < atm), "open_interest"].sum())
    call_oi = float(data.loc[(data.call_put == "call") & (data.strike_price > atm), "open_interest"].sum())
    return {
        "atm_strike": atm,
        "tail_skew_1pct": math.log(p1 / c1),
        "tail_skew_2pct": math.log(p2 / c2),
        "otm_put_call_volume": put_otm / call_otm if call_otm > 0 else np.nan,
        "otm_put_call_oi": put_oi / call_oi if call_oi > 0 else np.nan,
        "contract_volume": float(data.volume.sum()),
        "strike_rows": int(len(data)),
    }


def summarize(rows):
    if not rows:
        return pd.DataFrame()
    data = pd.DataFrame(rows)
    required = {
        "date", "contract_date", "strike_price", "call_put", "close",
        "settlement_price", "volume", "open_interest", "trading_session",
    }
    if not required.issubset(data.columns):
        raise RuntimeError(f"Option schema changed; missing {sorted(required - set(data.columns))}")
    session = data.trading_session.astype(str).str.lower()
    data = data[session.isin(["after_market", "afterhours", "夜盤"])]
    output = []
    for day, daily in data.groupby("date"):
        candidates = []
        for contract, group in daily.groupby("contract_date"):
            summary = summarize_contract(group)
            if summary:
                summary["contract_date"] = str(contract)
                candidates.append(summary)
        if not candidates:
            continue
        # The most actively traded valid expiry is a point-in-time liquidity
        # rule and avoids hard-coding monthly/weekly expiry naming conventions.
        best = max(candidates, key=lambda x: x["contract_volume"])
        best["signal_date"] = str(day)
        best["valid_contracts"] = len(candidates)
        output.append(best)
    return pd.DataFrame(output)


def merge_csv(path, new, key):
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([old, new], ignore_index=True) if not old.empty else new.copy()
    if not combined.empty:
        combined = combined.sort_values(key).drop_duplicates(key, keep="last")
        combined.to_csv(path, index=False)
    return combined


def main():
    args = arguments()
    token = os.environ.get("FINMIND_TOKEN", "").strip()
    if not token:
        secret = ROOT / "config/.secrets/finmind_token.txt"
        token = secret.read_text(encoding="utf-8").strip() if secret.exists() else ""
    if not token:
        raise RuntimeError("FinMind token not available")
    if LOCK.exists():
        raise RuntimeError(f"Option-tail fetch lock exists: {LOCK}")
    LOCK.write_text(str(os.getpid()), encoding="ascii")
    try:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        manifest = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame(
            columns=["calendar_date", "status", "feature_rows", "fetched_at"]
        )
        done = set(manifest.loc[manifest.status == "complete", "calendar_date"].astype(str))
        cursor, finish = date.fromisoformat(args.start_date), date.fromisoformat(args.end_date)
        while cursor <= finish:
            end = min(finish, cursor + timedelta(days=args.chunk_days - 1))
            dates = [cursor + timedelta(days=i) for i in range((end - cursor).days + 1)]
            missing = [d for d in dates if str(d) not in done]
            if not missing:
                cursor = end + timedelta(days=1)
                continue
            start_s, end_s = str(min(missing)), str(max(missing))
            print(f"option_tail: fetching {start_s} to {end_s}...", flush=True)
            rows = fetch(start_s, end_s, token)
            features = summarize(rows)
            merge_csv(OUT, features, "signal_date")
            available = set(features.signal_date.astype(str)) if not features.empty else set()
            stamp = datetime.now(TAIPEI).isoformat()
            updates = pd.DataFrame([{
                "calendar_date": str(d), "status": "complete",
                "feature_rows": int(str(d) in available), "fetched_at": stamp,
            } for d in missing])
            manifest = merge_csv(MANIFEST, updates, "calendar_date")
            done.update(str(d) for d in missing)
            print(f"option_tail: {len(rows)} source rows -> {len(features)} daily rows.", flush=True)
            cursor = end + timedelta(days=1)
            time.sleep(args.request_pause)
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
