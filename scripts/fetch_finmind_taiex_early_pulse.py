"""Incrementally compress FinMind 5-second TAIEX into fixed 09:15 pulse features."""

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
OUT = ROOT / "data/processed/factors/taiex_early_pulse.csv"
MANIFEST = ROOT / "data/processed/factors/taiex_early_pulse_fetched_dates.csv"
LOCK = ROOT / "data/processed/factors/.taiex_early_pulse_fetch.lock"
TAIPEI = timezone(timedelta(hours=8))
CUTOFF = "09:15:00"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=str(date.today()))
    parser.add_argument("--request-pause", type=float, default=0.25)
    return parser.parse_args()


def fetch(day, token):
    params = {
        "dataset": "TaiwanVariousIndicators5Seconds",
        "start_date": day,
        "token": token,
    }
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


def summarize(rows, requested_day):
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    if not {"date", "TAIEX"}.issubset(frame.columns):
        raise RuntimeError(f"TAIEX 5-second schema changed: {sorted(frame.columns)}")
    frame["timestamp"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["TAIEX"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "price"]).sort_values("timestamp")
    frame = frame[frame.timestamp.dt.strftime("%Y-%m-%d") == requested_day]
    early = frame[frame.timestamp.dt.strftime("%H:%M:%S") <= CUTOFF]
    if len(early) < 120:
        return None
    price = early.price.to_numpy(float)
    logp = np.log(price)
    returns = np.diff(logp)
    elapsed = (early.timestamp - early.timestamp.iloc[0]).dt.total_seconds().to_numpy(float)
    slope = float(np.polyfit(elapsed, logp, 1)[0] * 900) if elapsed[-1] > 0 else 0.0
    path_length = float(np.abs(returns).sum())
    displacement = float(logp[-1] - logp[0])
    first5 = early[early.timestamp <= early.timestamp.iloc[0] + pd.Timedelta(minutes=5)]
    last5 = early[early.timestamp >= early.timestamp.iloc[-1] - pd.Timedelta(minutes=5)]
    return {
        "date": requested_day,
        "open_0900": float(price[0]),
        "price_0915": float(price[-1]),
        "early_return": math.expm1(displacement),
        "first5_return": float(first5.price.iloc[-1] / first5.price.iloc[0] - 1),
        "last5_return": float(last5.price.iloc[-1] / last5.price.iloc[0] - 1),
        "return_acceleration": float(
            last5.price.iloc[-1] / last5.price.iloc[0]
            - first5.price.iloc[-1] / first5.price.iloc[0]
        ),
        "early_range": float(price.max() / price.min() - 1),
        "realized_volatility": float(np.sqrt(np.square(returns).sum())),
        "trend_efficiency": abs(displacement) / path_length if path_length > 0 else 0.0,
        "close_location": float(
            (price[-1] - price.min()) / (price.max() - price.min())
        ) if price.max() > price.min() else 0.5,
        "linear_slope_15m": slope,
        "observations": int(len(early)),
        "first_timestamp": str(early.timestamp.iloc[0]),
        "cutoff_timestamp": str(early.timestamp.iloc[-1]),
    }


def merge_csv(path, new, key):
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([old, new], ignore_index=True) if not old.empty else new.copy()
    if not combined.empty:
        combined = combined.sort_values(key).drop_duplicates(key, keep="last")
        # A cloud-synced reader must never observe a partially rewritten CSV.
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
        raise RuntimeError(f"TAIEX early-pulse fetch lock exists: {LOCK}")
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
                f"taiex_early_pulse: {'skipping weekend' if weekend else 'fetching'} {day}...",
                flush=True,
            )
            rows = [] if weekend else fetch(day, token)
            feature = summarize(rows, day)
            if feature:
                merge_csv(OUT, pd.DataFrame([feature]), "date")
            stamp = datetime.now(TAIPEI).isoformat()
            update = pd.DataFrame([{
                "calendar_date": day,
                "status": "complete",
                "feature_rows": int(feature is not None),
                "source_rows": len(rows),
                "fetched_at": stamp,
            }])
            manifest = merge_csv(MANIFEST, update, "calendar_date")
            done.add(day)
            print(f"taiex_early_pulse: {len(rows)} source rows -> {int(feature is not None)} daily row.", flush=True)
            cursor += timedelta(days=1)
            time.sleep(args.request_pause)
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
