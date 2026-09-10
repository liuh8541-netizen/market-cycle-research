"""Incrementally cache point-in-time all-market PER/PBR/yield breadth."""

import argparse
import concurrent.futures
import json
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
API = "https://api.finmindtrade.com/api/v4/data"
DATASET = "TaiwanStockPER"
OUTPUT = ROOT / "data/processed/factors/market_valuation_breadth.csv"
MANIFEST = ROOT / "data/processed/factors/market_valuation_fetched_dates.csv"
LOCK = ROOT / "data/processed/factors/.market_valuation_fetch.lock"
TRADING_DATES = ROOT / "data/processed/factors/cross_sectional_breadth_fetched_dates.csv"
COMMON_STOCK = re.compile(r"^[1-9][0-9]{3}$")


def request_rows(day, token):
    params = {
        "dataset": DATASET, "start_date": day, "end_date": day, "token": token
    }
    request = Request(
        API + "?" + urlencode(params),
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "market-lifecycle-research/1.0",
        },
    )
    with urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") not in (200, "200", None):
        raise RuntimeError(f"{day}: {payload.get('status')} {payload.get('msg')}")
    return payload.get("data", [])


def fetch(day, token):
    transient = 0
    while True:
        try:
            return day, request_rows(day, token), ""
        except HTTPError as exc:
            if exc.code == 402:
                print(f"{day}: hourly quota reached; waiting 30 seconds...", flush=True)
                time.sleep(30)
                continue
            error = exc
        except (TimeoutError, OSError, RuntimeError) as exc:
            error = exc
        transient += 1
        if transient >= 5:
            return day, [], repr(error)
        time.sleep(3 * transient)


def robust_log_stats(values):
    values = pd.to_numeric(values, errors="coerce")
    valid = values.gt(0) & np.isfinite(values)
    logs = np.log(values[valid])
    return valid, (
        float(logs.median()) if len(logs) else np.nan
    ), (
        float(logs.quantile(0.75) - logs.quantile(0.25)) if len(logs) else np.nan
    )


def aggregate(day, rows):
    data = pd.DataFrame(rows)
    if data.empty or "stock_id" not in data:
        return {"date": day, "valuation_stock_count": 0}
    ids = data["stock_id"].astype(str).str.strip()
    data = data.loc[ids.map(lambda value: bool(COMMON_STOCK.fullmatch(value)))].copy()
    per = pd.to_numeric(data["PER"], errors="coerce")
    pbr = pd.to_numeric(data["PBR"], errors="coerce")
    yield_ = pd.to_numeric(data["dividend_yield"], errors="coerce")
    per_valid, per_log_median, per_log_iqr = robust_log_stats(per)
    pbr_valid, pbr_log_median, pbr_log_iqr = robust_log_stats(pbr)
    yield_valid = yield_.ge(0) & np.isfinite(yield_)
    count = len(data)
    return {
        "date": day,
        "valuation_stock_count": int(data["stock_id"].nunique()),
        "per_valid_fraction": float(per_valid.sum() / count) if count else np.nan,
        "per_log_median": per_log_median,
        "per_log_iqr": per_log_iqr,
        "per_below_10_fraction": float((per_valid & per.lt(10)).sum() / per_valid.sum())
        if per_valid.any() else np.nan,
        "per_above_30_fraction": float((per_valid & per.gt(30)).sum() / per_valid.sum())
        if per_valid.any() else np.nan,
        "pbr_valid_fraction": float(pbr_valid.sum() / count) if count else np.nan,
        "pbr_log_median": pbr_log_median,
        "pbr_log_iqr": pbr_log_iqr,
        "pbr_below_1_fraction": float((pbr_valid & pbr.lt(1)).sum() / pbr_valid.sum())
        if pbr_valid.any() else np.nan,
        "yield_valid_fraction": float(yield_valid.sum() / count) if count else np.nan,
        "yield_median": float(yield_[yield_valid].median()) if yield_valid.any() else np.nan,
        "yield_above_4_fraction": float(
            (yield_valid & yield_.ge(4)).sum() / yield_valid.sum()
        ) if yield_valid.any() else np.nan,
    }


def atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2005-10-01")
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--checkpoint", type=int, default=25)
    args = parser.parse_args()
    token = os.environ.get("FINMIND_TOKEN", "").strip()
    if not token:
        secret = ROOT / "config/.secrets/finmind_token.txt"
        token = secret.read_text(encoding="utf-8").strip() if secret.exists() else ""
    if not token:
        raise RuntimeError("FINMIND_TOKEN is not configured")
    source = pd.read_csv(TRADING_DATES)
    source["price_source_rows"] = pd.to_numeric(
        source["price_source_rows"], errors="coerce"
    ).fillna(0)
    dates = sorted(source.loc[
        source["price_source_rows"].gt(0)
        & source["date"].between(args.start_date, args.end_date), "date"
    ].drop_duplicates().astype(str))
    old_features = pd.read_csv(OUTPUT) if OUTPUT.exists() else pd.DataFrame()
    old_manifest = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame()
    completed = set(
        old_manifest.loc[old_manifest["status"].eq("success"), "date"].astype(str)
    ) if not old_manifest.empty else set()
    pending = [day for day in dates if day not in completed]
    LOCK.write_text(json.dumps({"pending": len(pending), "total": len(dates)}), encoding="utf-8")
    features = [] if old_features.empty else old_features.to_dict("records")
    manifest = [] if old_manifest.empty else old_manifest.to_dict("records")
    done = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch, day, token) for day in pending]
            for future in concurrent.futures.as_completed(futures):
                day, rows, error = future.result()
                if error:
                    manifest.append({"date": day, "status": "failed", "source_rows": 0, "error": error})
                    print(f"{day}: failed {error}", flush=True)
                else:
                    features.append(aggregate(day, rows))
                    manifest.append({"date": day, "status": "success", "source_rows": len(rows), "error": ""})
                    print(f"{day}: {len(rows)} rows", flush=True)
                done += 1
                if done % args.checkpoint == 0:
                    atomic_csv(pd.DataFrame(features).sort_values("date").drop_duplicates("date", keep="last"), OUTPUT)
                    atomic_csv(pd.DataFrame(manifest).sort_values("date").drop_duplicates("date", keep="last"), MANIFEST)
        atomic_csv(pd.DataFrame(features).sort_values("date").drop_duplicates("date", keep="last"), OUTPUT)
        atomic_csv(pd.DataFrame(manifest).sort_values("date").drop_duplicates("date", keep="last"), MANIFEST)
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
