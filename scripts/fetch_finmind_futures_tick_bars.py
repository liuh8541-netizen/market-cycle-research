"""Incrementally cache FinMind TX ticks as compact 15-minute bars."""

import argparse
import ctypes
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.finmindtrade.com/api/v4/data"
OUTPUT = ROOT / "data" / "processed" / "factors" / "futures_tick_15m.csv"
MANIFEST = ROOT / "data" / "processed" / "factors" / "futures_tick_fetched_dates.csv"
LOCK = ROOT / "data" / "processed" / "factors" / ".futures_tick_fetch.lock"


def fetch_day(day, token):
    params = {
        "dataset": "TaiwanFuturesTick", "data_id": "TX",
        "start_date": day.isoformat(), "token": token,
    }
    request = Request(API_URL + "?" + urlencode(params), headers={"User-Agent": "market-research/1.0"})
    last_error = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
            if body.get("status") != 200:
                raise RuntimeError(body.get("msg", "FinMind request failed"))
            return body.get("data") or []
        except (TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise last_error


def aggregate(rows):
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    required = {"date", "contract_date", "price", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Unexpected TaiwanFuturesTick schema: {sorted(frame.columns)}")
    frame["timestamp"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0)
    frame["contract_date"] = frame["contract_date"].astype(str)
    frame = frame[
        frame["timestamp"].notna() & frame["price"].notna()
        & ~frame["contract_date"].str.contains("/", regex=False)
    ].sort_values(["contract_date", "timestamp"])
    if frame.empty:
        return pd.DataFrame()
    frame["bar_start"] = frame["timestamp"].dt.floor("15min")
    delta = frame.groupby("contract_date")["price"].diff()
    frame["signed_volume"] = np.sign(delta.fillna(0)) * frame["volume"]
    bars = frame.groupby(["contract_date", "bar_start"], as_index=False).agg(
        open=("price", "first"), high=("price", "max"), low=("price", "min"),
        close=("price", "last"), volume=("volume", "sum"), ticks=("price", "size"),
        signed_volume=("signed_volume", "sum"),
    )
    bars["calendar_date"] = bars["bar_start"].dt.strftime("%Y-%m-%d")
    return bars[[
        "calendar_date", "contract_date", "bar_start", "open", "high", "low",
        "close", "volume", "ticks", "signed_volume",
    ]]


def load_done():
    if not MANIFEST.exists():
        return set()
    manifest = pd.read_csv(MANIFEST)
    return set(manifest.loc[manifest["status"].eq("complete"), "calendar_date"].astype(str))


def process_exists(pid):
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, int(pid)
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def append_frame(frame, path):
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")


def atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def replace_calendar_day(frame, path, day):
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not old.empty and "calendar_date" in old:
        old = old.loc[old["calendar_date"].astype(str).ne(day.isoformat())]
    combined = pd.concat([old, frame], ignore_index=True, sort=False)
    if not combined.empty:
        combined = combined.sort_values(["calendar_date", "contract_date", "bar_start"])
    atomic_csv(combined, path)


def upsert_manifest(day, row_count, bar_count, status):
    row = pd.DataFrame([{
        "calendar_date": day.isoformat(), "status": status,
        "tick_rows": row_count, "bar_rows": bar_count,
        "fetched_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
    }])
    old = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame()
    if not old.empty and "calendar_date" in old:
        old = old.loc[old["calendar_date"].astype(str).ne(day.isoformat())]
    combined = pd.concat([old, row], ignore_index=True, sort=False)
    atomic_csv(combined.sort_values("calendar_date"), MANIFEST)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=(date.today() - timedelta(days=7)).isoformat())
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--token", default=os.environ.get("FINMIND_TOKEN", ""))
    parser.add_argument("--request-interval", type=float, default=6.1, help="Seconds; 6.1 respects 600 requests/hour.")
    parser.add_argument("--max-days", type=int, default=0, help="Optional limit for a test or staged backfill.")
    parser.add_argument(
        "--refresh", action="store_true",
        help="Refetch and replace the requested calendar dates even if cached.",
    )
    args = parser.parse_args()
    if not args.token:
        raise SystemExit("FINMIND_TOKEN is required.")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            prior_pid = int(LOCK.read_text(encoding="ascii").strip())
        except ValueError:
            prior_pid = 0
        if prior_pid and process_exists(prior_pid):
            print("Another TX tick-bar update is already running; this invocation is skipped.")
            return
        else:
            LOCK.unlink(missing_ok=True)
            descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        start, end = pd.Timestamp(args.start_date).date(), pd.Timestamp(args.end_date).date()
        done = load_done()
        days = []
        cursor = start
        while cursor <= end:
            # TAIFEX has no Sunday session. Saturdays are retained because they
            # contain the 00:00-05:00 continuation of Friday's night session.
            if cursor.weekday() != 6 and (args.refresh or cursor.isoformat() not in done):
                days.append(cursor)
            cursor += timedelta(days=1)
        if args.max_days > 0:
            days = days[:args.max_days]
        print(f"TX tick-bar update: {len(days)} missing calendar dates ({start} to {end}).")
        for index, day in enumerate(days):
            rows = fetch_day(day, args.token)
            bars = aggregate(rows)
            replace_calendar_day(bars, OUTPUT, day)
            taipei_today = pd.Timestamp.now(tz="Asia/Taipei").date()
            status = "complete" if day < taipei_today else "partial"
            upsert_manifest(day, len(rows), len(bars), status)
            print(
                f"{day}: {len(rows)} ticks -> {len(bars)} bars cached "
                f"({status})."
            )
            if index + 1 < len(days) and args.request_interval > 0:
                time.sleep(args.request_interval)
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
