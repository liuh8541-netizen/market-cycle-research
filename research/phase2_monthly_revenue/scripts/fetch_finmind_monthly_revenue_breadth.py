"""Incrementally cache point-in-time all-market monthly-revenue breadth.

This downloader never reads a forecast target.  FinMind's monthly source date
is conservatively delayed by 14 calendar days before it can enter a market
decision, because old rows do not provide a usable create_time.
"""

from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "research/phase2_monthly_revenue"
API = "https://api.finmindtrade.com/api/v4/data"
DATASET = "TaiwanStockMonthRevenue"
OUTPUT = WORKSPACE / "data/monthly_revenue_breadth.csv"
RAW = WORKSPACE / "data/monthly_revenue_common_stock.csv"
MANIFEST = WORKSPACE / "data/monthly_revenue_fetched_months.csv"
LOCK = WORKSPACE / "data/.monthly_revenue_fetch.lock"
COMMON_STOCK = re.compile(r"^[1-9][0-9]{3}$")
AVAILABILITY_LAG_DAYS = 14


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def request_rows(day: str, token: str) -> list[dict]:
    params = {
        "dataset": DATASET,
        "start_date": day,
        "token": token,
    }
    request = Request(
        API + "?" + urlencode(params),
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "market-lifecycle-research/1.0",
        },
    )
    with urlopen(request, timeout=240) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") not in (200, "200", None):
        raise RuntimeError(f"{day}: {payload.get('status')} {payload.get('msg')}")
    return payload.get("data", [])


def fetch(day: str, token: str) -> tuple[str, list[dict], str]:
    failures = 0
    while True:
        try:
            rows = request_rows(day, token)
            if not rows:
                raise RuntimeError("API returned zero rows")
            return day, rows, ""
        except HTTPError as exc:
            if exc.code == 402:
                print("Hourly FinMind quota reached; waiting 30 seconds...", flush=True)
                time.sleep(30)
                continue
            error: Exception = exc
        except (TimeoutError, OSError, RuntimeError) as exc:
            error = exc
        failures += 1
        if failures >= 5:
            return day, [], repr(error)
        time.sleep(5 * failures)


def normalized(rows: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    required = {
        "date",
        "stock_id",
        "revenue",
        "revenue_month",
        "revenue_year",
    }
    if data.empty or not required.issubset(data.columns):
        return pd.DataFrame()
    data["stock_id"] = data["stock_id"].astype(str).str.strip()
    data = data.loc[
        data["stock_id"].map(lambda value: bool(COMMON_STOCK.fullmatch(value)))
    ].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["revenue"] = pd.to_numeric(data["revenue"], errors="coerce")
    data["revenue_month"] = pd.to_numeric(
        data["revenue_month"], errors="coerce"
    ).astype("Int64")
    data["revenue_year"] = pd.to_numeric(
        data["revenue_year"], errors="coerce"
    ).astype("Int64")
    data = data.loc[
        data["date"].notna()
        & data["revenue"].gt(0)
        & np.isfinite(data["revenue"])
        & data["revenue_month"].between(1, 12)
        & data["revenue_year"].notna()
    ].copy()
    return (
        data.groupby(
            ["date", "stock_id", "revenue_year", "revenue_month"],
            as_index=False,
        )["revenue"]
        .sum()
        .sort_values(["stock_id", "revenue_year", "revenue_month"])
    )


def aggregate(rows: list[dict], target_start: pd.Timestamp, target_end: pd.Timestamp) -> pd.DataFrame:
    data = normalized(rows)
    if data.empty:
        return pd.DataFrame()
    previous = data[
        ["stock_id", "revenue_year", "revenue_month", "revenue"]
    ].copy()
    previous["revenue_year"] = previous["revenue_year"] + 1
    previous = previous.rename(columns={"revenue": "revenue_previous_year"})
    data = data.merge(
        previous,
        on=["stock_id", "revenue_year", "revenue_month"],
        how="left",
    )
    data["yoy"] = data["revenue"] / data["revenue_previous_year"] - 1
    data = data.loc[data["date"].between(target_start, target_end)].copy()
    output: list[dict] = []
    for source_date, group in data.groupby("date", sort=True):
        revenue = group.groupby("stock_id")["revenue"].sum().sort_values(ascending=False)
        shares = revenue / revenue.sum()
        logs = np.log(revenue)
        yoy = pd.to_numeric(group["yoy"], errors="coerce")
        yoy = yoy[np.isfinite(yoy) & yoy.between(-0.95, 10)]
        # A source month without any prior-year comparison cannot support the
        # predeclared breadth features.  Exclude it before any target is read.
        if not len(yoy):
            continue
        output.append(
            {
                "source_date": source_date.date().isoformat(),
                "available_date": (
                    source_date + pd.Timedelta(days=AVAILABILITY_LAG_DAYS)
                ).date().isoformat(),
                "revenue_stock_count": int(len(revenue)),
                "revenue_total": float(revenue.sum()),
                "revenue_log_median": float(logs.median()),
                "revenue_log_iqr": float(logs.quantile(0.75) - logs.quantile(0.25)),
                "revenue_top10_share": float(shares.head(10).sum()),
                "revenue_top50_share": float(shares.head(50).sum()),
                "revenue_hhi": float((shares**2).sum()),
                "revenue_yoy_valid_count": int(len(yoy)),
                "revenue_yoy_median": float(yoy.median()) if len(yoy) else np.nan,
                "revenue_yoy_iqr": (
                    float(yoy.quantile(0.75) - yoy.quantile(0.25))
                    if len(yoy)
                    else np.nan
                ),
                "revenue_yoy_positive_fraction": (
                    float(yoy.gt(0).mean()) if len(yoy) else np.nan
                ),
                "revenue_yoy_above20_fraction": (
                    float(yoy.gt(0.20).mean()) if len(yoy) else np.nan
                ),
                "revenue_yoy_below_minus20_fraction": (
                    float(yoy.lt(-0.20).mean()) if len(yoy) else np.nan
                ),
            }
        )
    return pd.DataFrame(output)


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    first = start.to_period("M").to_timestamp()
    last = end.to_period("M").to_timestamp()
    return [value.date().isoformat() for value in pd.date_range(first, last, freq="MS")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2002-02-01")
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--checkpoint", type=int, default=12)
    args = parser.parse_args()
    token = os.environ.get("FINMIND_TOKEN", "").strip()
    if not token:
        secret = ROOT / "config/.secrets/finmind_token.txt"
        token = secret.read_text(encoding="utf-8").strip() if secret.exists() else ""
    if not token:
        raise RuntimeError("FINMIND_TOKEN is not configured")

    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    old_raw = pd.read_csv(RAW) if RAW.exists() else pd.DataFrame()
    old_manifest = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame()
    completed = (
        set(old_manifest.loc[old_manifest["status"].eq("success"), "date"].astype(str))
        if not old_manifest.empty
        else set()
    )
    dates = month_starts(start, end)
    pending = [day for day in dates if day not in completed]
    LOCK.write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "pending": len(pending),
                "total": len(dates),
            }
        ),
        encoding="utf-8",
    )
    raw_records = [] if old_raw.empty else old_raw.to_dict("records")
    manifest = [] if old_manifest.empty else old_manifest.to_dict("records")
    done = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch, day, token) for day in pending]
            for future in concurrent.futures.as_completed(futures):
                day, rows, error = future.result()
                if error:
                    manifest.append(
                        {"date": day, "status": "failed", "source_rows": 0, "error": error}
                    )
                    print(f"{day}: failed {error}", flush=True)
                else:
                    clean = normalized(rows)
                    if clean.empty:
                        manifest.append(
                            {
                                "date": day,
                                "status": "failed",
                                "source_rows": len(rows),
                                "error": "no usable common-stock rows",
                            }
                        )
                        print(f"{day}: no usable common-stock rows", flush=True)
                    else:
                        raw_records.extend(clean.to_dict("records"))
                        manifest.append(
                            {
                                "date": day,
                                "status": "success",
                                "source_rows": len(rows),
                                "error": "",
                            }
                        )
                        print(f"{day}: {len(rows)} rows, {len(clean)} stored", flush=True)
                done += 1
                if done % args.checkpoint == 0:
                    raw_frame = pd.DataFrame(raw_records)
                    if not raw_frame.empty:
                        atomic_csv(
                            raw_frame.sort_values(["date", "stock_id"]).drop_duplicates(
                                ["date", "stock_id", "revenue_year", "revenue_month"],
                                keep="last",
                            ),
                            RAW,
                        )
                    atomic_csv(
                        pd.DataFrame(manifest)
                        .sort_values("date")
                        .drop_duplicates("date", keep="last"),
                        MANIFEST,
                    )
        raw_frame = pd.DataFrame(raw_records)
        if raw_frame.empty:
            raise RuntimeError("No monthly revenue rows were stored")
        raw_frame = raw_frame.sort_values(["date", "stock_id"]).drop_duplicates(
            ["date", "stock_id", "revenue_year", "revenue_month"], keep="last"
        )
        atomic_csv(raw_frame, RAW)
        atomic_csv(
            pd.DataFrame(manifest).sort_values("date").drop_duplicates("date", keep="last"),
            MANIFEST,
        )
        failed = pd.DataFrame(manifest)
        failed = failed.loc[failed["status"].ne("success")]
        if not failed.empty:
            raise RuntimeError(f"{len(failed)} monthly source dates failed; resume required")
        features = aggregate(
            raw_frame.to_dict("records"),
            pd.Timestamp("2003-01-01"),
            end,
        )
        if features.empty:
            raise RuntimeError("Monthly revenue feature aggregation returned no rows")
        atomic_csv(features, OUTPUT)
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
