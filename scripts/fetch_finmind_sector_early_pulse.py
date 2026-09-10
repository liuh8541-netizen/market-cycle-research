"""Incrementally compress selected 5-second Taiwan sector indices through 09:15."""

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
DATASET = "TaiwanStockEvery5SecondsIndex"
DATA_IDS = ["Electronic", "FinancialInsurance", "TPExIndex"]
OUT = ROOT / "data/processed/factors/sector_early_pulse.csv"
MANIFEST = ROOT / "data/processed/factors/sector_early_pulse_fetched_dates.csv"
LOCK = ROOT / "data/processed/factors/.sector_early_pulse_fetch.lock"
TAIPEI = timezone(timedelta(hours=8))
CUTOFF = "09:15:00"
SERIES = {
    ("twse", "Electronic"): "twse_electronic",
    ("tpex", "Electronic"): "tpex_electronic",
    ("twse", "FinancialInsurance"): "twse_finance",
    ("tpex", "TPExIndex"): "tpex_index",
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=str(date.today()))
    parser.add_argument("--request-pause", type=float, default=0.15)
    return parser.parse_args()


def fetch(day, data_id, token):
    params = {
        "dataset": DATASET, "start_date": day, "data_id": data_id, "token": token,
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
    required = {"date", "time", "kind", "stock_id", "price"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Sector-index schema changed; missing {sorted(required-set(frame.columns))}")
    frame = frame[frame.date.astype(str) == requested_day].copy()
    frame["timestamp"] = pd.to_datetime(
        frame.date.astype(str) + " " + frame.time.astype(str), errors="coerce"
    )
    frame["price"] = pd.to_numeric(frame.price, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "price"])
    frame = frame[frame.time.astype(str) <= CUTOFF]
    output = {"date": requested_day}
    returns, last_returns = [], []
    for key, prefix in SERIES.items():
        series = frame[
            (frame.kind.astype(str) == key[0]) & (frame.stock_id.astype(str) == key[1])
        ].sort_values("timestamp")
        if len(series) < 120:
            return None
        first, last = float(series.price.iloc[0]), float(series.price.iloc[-1])
        last5 = series[
            series.timestamp >= series.timestamp.iloc[-1] - pd.Timedelta(minutes=5)
        ]
        full_return = last / first - 1
        last_return = float(last5.price.iloc[-1] / last5.price.iloc[0] - 1)
        output[prefix + "_return"] = full_return
        output[prefix + "_last5_return"] = last_return
        output[prefix + "_observations"] = int(len(series))
        returns.append(full_return)
        last_returns.append(last_return)
    output["sector_mean_return"] = float(np.mean(returns))
    output["sector_dispersion"] = float(np.std(returns))
    output["sector_breadth"] = float(np.mean(np.sign(returns)))
    output["sector_last5_breadth"] = float(np.mean(np.sign(last_returns)))
    output["electronic_finance_rotation"] = (
        0.5 * (output["twse_electronic_return"] + output["tpex_electronic_return"])
        - output["twse_finance_return"]
    )
    output["tpex_twse_electronic_rotation"] = (
        output["tpex_electronic_return"] - output["twse_electronic_return"]
    )
    output["first_timestamp"] = requested_day + " 09:00:00"
    output["cutoff_timestamp"] = requested_day + " 09:15:00"
    return output


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
        raise RuntimeError(f"Sector fetch lock exists: {LOCK}")
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
                f"sector_early_pulse: {'skipping weekend' if weekend else 'fetching'} {day}...",
                flush=True,
            )
            rows = []
            if not weekend:
                for data_id in DATA_IDS:
                    rows.extend(fetch(day, data_id, token))
                    time.sleep(args.request_pause)
            feature = summarize(rows, day)
            if feature:
                merge_csv(OUT, pd.DataFrame([feature]), "date")
            update = pd.DataFrame([{
                "calendar_date": day, "status": "complete",
                "feature_rows": int(feature is not None), "source_rows": len(rows),
                "fetched_at": datetime.now(TAIPEI).isoformat(),
            }])
            manifest = merge_csv(MANIFEST, update, "calendar_date")
            done.add(day)
            print(
                f"sector_early_pulse: {len(rows)} source rows -> {int(feature is not None)} daily row.",
                flush=True,
            )
            cursor += timedelta(days=1)
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
