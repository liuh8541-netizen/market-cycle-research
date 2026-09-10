"""Checkpointed unlabeled backfill of all-market credit short inventory."""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "research/phase2_credit_short_inventory"
API = "https://api.finmindtrade.com/api/v4/data"
DATASET = "TaiwanDailyShortSaleBalances"
OUTPUT = WORKSPACE / "data/credit_short_inventory.csv"
MANIFEST = WORKSPACE / "data/credit_short_inventory_fetched_dates.csv"
LOCK = WORKSPACE / "data/.credit_short_inventory_fetch.lock"
TRADING_DATES = (
    ROOT / "data/processed/factors/cross_sectional_breadth_fetched_dates.csv"
)
SECRET = ROOT / "config/.secrets/finmind_token.txt"
COMMON_STOCK = re.compile(r"^[1-9][0-9]{3}$")
MIN_COMMON_STOCKS = 50

MARGIN_PREVIOUS = "MarginShortSalesPreviousDayBalance"
MARGIN_SELL = "MarginShortSalesShortSales"
MARGIN_COVER = "MarginShortSalesShortCovering"
MARGIN_REDEEM = "MarginShortSalesStockRedemption"
MARGIN_CURRENT = "MarginShortSalesCurrentDayBalance"
MARGIN_QUOTA = "MarginShortSalesQuota"
SBL_PREVIOUS = "SBLShortSalesPreviousDayBalance"
SBL_SELL = "SBLShortSalesShortSales"
SBL_RETURN = "SBLShortSalesReturns"
SBL_ADJUST = "SBLShortSalesAdjustments"
SBL_CURRENT = "SBLShortSalesCurrentDayBalance"
SBL_QUOTA = "SBLShortSalesQuota"
SBL_COVER = "SBLShortSalesShortCovering"
VALUE_COLUMNS = [
    MARGIN_PREVIOUS,
    MARGIN_SELL,
    MARGIN_COVER,
    MARGIN_REDEEM,
    MARGIN_CURRENT,
    MARGIN_QUOTA,
    SBL_PREVIOUS,
    SBL_SELL,
    SBL_RETURN,
    SBL_ADJUST,
    SBL_CURRENT,
    SBL_QUOTA,
    SBL_COVER,
]


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_lock(total: int, pending: int) -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            existing = json.loads(LOCK.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
        pid = int(existing.get("pid", 0) or 0)
        if process_alive(pid):
            raise RuntimeError(f"Backfill already active under PID {pid}")
        stale = LOCK.with_name(
            LOCK.name + ".stale." + datetime.now().strftime("%Y%m%dT%H%M%S")
        )
        LOCK.replace(stale)
    payload = {
        "pid": os.getpid(),
        "dataset": DATASET,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "total_dates": total,
        "pending_at_start": pending,
        "completed_this_run": 0,
    }
    LOCK.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_lock(completed: int, pending: int) -> None:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    payload["completed_this_run"] = completed
    payload["remaining_estimate"] = max(0, pending - completed)
    payload["checkpoint_at"] = datetime.now(timezone.utc).isoformat()
    temporary = LOCK.with_suffix(LOCK.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(LOCK)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def request_rows(day: str, token: str) -> list[dict]:
    params = {
        "dataset": DATASET,
        "start_date": day,
        "end_date": day,
        "token": token,
    }
    request = Request(
        API + "?" + urlencode(params),
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "market-lifecycle-isolated-research/1.0",
        },
    )
    with urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") not in (200, "200", None):
        raise RuntimeError(
            f"{day}: API status {payload.get('status')} {payload.get('msg', '')}"
        )
    return payload.get("data", [])


def fetch(day: str, token: str) -> tuple[str, list[dict], str]:
    transient = 0
    quota_waits = 0
    while True:
        try:
            return day, request_rows(day, token), ""
        except HTTPError as exc:
            if exc.code == 402:
                quota_waits += 1
                if quota_waits > 65:
                    return day, [], "hourly quota did not reset within 65 minutes"
                time.sleep(60)
                continue
            error = exc
        except (TimeoutError, OSError, RuntimeError) as exc:
            error = exc
        transient += 1
        if transient >= 5:
            return day, [], repr(error)
        time.sleep(3 * transient)


def safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return 0.0
    return numerator / denominator if denominator > 0 else 0.0


def aggregate(day: str, rows: list[dict]) -> tuple[dict | None, str, int]:
    data = pd.DataFrame(rows)
    if data.empty or "stock_id" not in data:
        return None, "excluded_no_common_rows", 0
    missing = sorted(set(VALUE_COLUMNS) - set(data.columns))
    if missing:
        raise RuntimeError(f"{day}: missing API columns {missing}")
    ids = data["stock_id"].astype(str).str.strip()
    data = data.loc[
        ids.map(lambda value: bool(COMMON_STOCK.fullmatch(value)))
    ].copy()
    if data.empty:
        return None, "excluded_no_common_rows", 0
    data["stock_id"] = data["stock_id"].astype(str).str.strip()
    before = len(data)
    data = data.drop_duplicates("stock_id", keep="last")
    duplicate_rows = before - len(data)
    for column in VALUE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    balance_columns = [MARGIN_PREVIOUS, MARGIN_CURRENT, SBL_PREVIOUS, SBL_CURRENT]
    valid_balance = data[balance_columns].notna().any(axis=1)
    data = data.loc[valid_balance].copy()
    if len(data) < MIN_COMMON_STOCKS:
        return None, "excluded_low_common_stock_coverage", len(data)
    for column in VALUE_COLUMNS:
        data[column] = data[column].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for column in balance_columns:
        data[column] = data[column].clip(lower=0.0)

    margin_previous = float(data[MARGIN_PREVIOUS].sum())
    margin_current = float(data[MARGIN_CURRENT].sum())
    sbl_previous = float(data[SBL_PREVIOUS].sum())
    sbl_current = float(data[SBL_CURRENT].sum())
    combined = data[MARGIN_CURRENT] + data[SBL_CURRENT]
    combined_total = float(combined.sum())
    positive = combined[combined > 0].sort_values(ascending=False)
    shares = positive / combined_total if combined_total > 0 else positive
    logs = np.log1p(positive)
    margin_increase = data[MARGIN_CURRENT] > data[MARGIN_PREVIOUS]
    sbl_increase = data[SBL_CURRENT] > data[SBL_PREVIOUS]
    margin_sell = float(data[MARGIN_SELL].clip(lower=0).sum())
    margin_cover = float(data[MARGIN_COVER].clip(lower=0).sum())
    margin_redeem = float(data[MARGIN_REDEEM].clip(lower=0).sum())
    sbl_sell = float(data[SBL_SELL].clip(lower=0).sum())
    sbl_return = float(data[SBL_RETURN].clip(lower=0).sum())
    sbl_cover = float(data[SBL_COVER].clip(lower=0).sum())
    margin_quota = data[MARGIN_QUOTA].where(data[MARGIN_QUOTA] > 0, 0.0)
    sbl_quota = data[SBL_QUOTA].where(data[SBL_QUOTA] > 0, 0.0)
    result = {
        "date": day,
        "credit_short_stock_count": int(len(data)),
        "source_duplicate_stock_rows": int(duplicate_rows),
        "positive_combined_balance_count": int((combined > 0).sum()),
        "positive_combined_balance_fraction": float((combined > 0).mean()),
        "margin_previous_balance_total": margin_previous,
        "margin_current_balance_total": margin_current,
        "sbl_previous_balance_total": sbl_previous,
        "sbl_current_balance_total": sbl_current,
        "combined_current_balance_total": combined_total,
        "margin_balance_share": safe_ratio(margin_current, combined_total),
        "margin_new_short_total": margin_sell,
        "margin_cover_total": margin_cover,
        "margin_redemption_total": margin_redeem,
        "sbl_new_short_total": sbl_sell,
        "sbl_returns_total": sbl_return,
        "sbl_adjustments_total": float(data[SBL_ADJUST].sum()),
        "sbl_short_covering_total": sbl_cover,
        "combined_balance_change_total": (
            margin_current + sbl_current - margin_previous - sbl_previous
        ),
        "margin_balance_increase_fraction": float(margin_increase.mean()),
        "sbl_balance_increase_fraction": float(sbl_increase.mean()),
        "margin_new_short_to_previous": safe_ratio(margin_sell, margin_previous),
        "margin_cover_to_previous": safe_ratio(
            margin_cover + margin_redeem, margin_previous
        ),
        "sbl_new_short_to_previous": safe_ratio(sbl_sell, sbl_previous),
        "sbl_return_to_previous": safe_ratio(
            sbl_return + sbl_cover, sbl_previous
        ),
        "combined_top10_share": float(shares.head(10).sum()),
        "combined_top50_share": float(shares.head(50).sum()),
        "combined_hhi": float((shares**2).sum()),
        "combined_log_median": float(logs.median()) if len(logs) else 0.0,
        "combined_log_iqr": (
            float(logs.quantile(0.75) - logs.quantile(0.25))
            if len(logs)
            else 0.0
        ),
        "positive_margin_quota_count": int((margin_quota > 0).sum()),
        "positive_sbl_quota_count": int((sbl_quota > 0).sum()),
        "positive_margin_quota_total": float(margin_quota.sum()),
        "positive_sbl_quota_total": float(sbl_quota.sum()),
    }
    if not all(
        isinstance(value, str) or math.isfinite(float(value))
        for value in result.values()
    ):
        raise RuntimeError(f"{day}: aggregate contains non-finite values")
    return result, "success", len(data)


def trading_dates(start_date: str, end_date: str) -> list[str]:
    source = pd.read_csv(TRADING_DATES)
    source["price_source_rows"] = pd.to_numeric(
        source["price_source_rows"], errors="coerce"
    ).fillna(0)
    return sorted(
        source.loc[
            source["price_source_rows"].gt(0)
            & source["date"].astype(str).between(start_date, end_date),
            "date",
        ]
        .drop_duplicates()
        .astype(str)
    )


def save(
    features: list[dict],
    manifest: list[dict],
    completed_this_run: int,
    pending_at_start: int,
) -> None:
    feature_frame = pd.DataFrame(features)
    if not feature_frame.empty:
        feature_frame = (
            feature_frame.sort_values("date")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True)
        )
        atomic_csv(feature_frame, OUTPUT)
    manifest_frame = (
        pd.DataFrame(manifest)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    atomic_csv(manifest_frame, MANIFEST)
    update_lock(completed_this_run, pending_at_start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2005-07-01")
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--checkpoint", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=60)
    args = parser.parse_args()
    token = os.environ.get("FINMIND_TOKEN", "").strip()
    if not token and SECRET.exists():
        token = SECRET.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("FinMind Token is not configured")
    dates = trading_dates(args.start_date, args.end_date)
    old_features = pd.read_csv(OUTPUT) if OUTPUT.exists() else pd.DataFrame()
    old_manifest = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame()
    completed = (
        set(
            old_manifest.loc[
                old_manifest["status"].isin(
                    [
                        "success",
                        "excluded_no_common_rows",
                        "excluded_low_common_stock_coverage",
                    ]
                ),
                "date",
            ].astype(str)
        )
        if not old_manifest.empty
        else set()
    )
    pending = [day for day in dates if day not in completed]
    acquire_lock(len(dates), len(pending))
    features = [] if old_features.empty else old_features.to_dict("records")
    manifest = [] if old_manifest.empty else old_manifest.to_dict("records")
    done = 0
    try:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, args.workers)
            ) as pool:
                futures = [pool.submit(fetch, day, token) for day in batch]
                for future in concurrent.futures.as_completed(futures):
                    day, rows, error = future.result()
                    if error:
                        manifest.append(
                            {
                                "date": day,
                                "status": "failed",
                                "source_rows": 0,
                                "common_stock_rows": 0,
                                "error": error,
                            }
                        )
                        print(f"{day}: failed {error}", flush=True)
                    else:
                        try:
                            feature, status, common_rows = aggregate(day, rows)
                            if feature is not None:
                                features.append(feature)
                            manifest.append(
                                {
                                    "date": day,
                                    "status": status,
                                    "source_rows": len(rows),
                                    "common_stock_rows": common_rows,
                                    "error": "",
                                }
                            )
                            print(
                                f"{day}: {len(rows)} source rows, "
                                f"{common_rows} common stocks, {status}",
                                flush=True,
                            )
                        except Exception as exc:
                            manifest.append(
                                {
                                    "date": day,
                                    "status": "failed",
                                    "source_rows": len(rows),
                                    "common_stock_rows": 0,
                                    "error": repr(exc),
                                }
                            )
                            print(f"{day}: aggregate failed {exc!r}", flush=True)
                    done += 1
                    if done % args.checkpoint == 0:
                        save(features, manifest, done, len(pending))
            save(features, manifest, done, len(pending))
    finally:
        if LOCK.exists():
            LOCK.unlink()
    print(
        json.dumps(
            {
                "expected_dates": len(dates),
                "pending_at_start": len(pending),
                "processed_this_run": done,
                "feature_file": str(OUTPUT.relative_to(ROOT)),
                "manifest_file": str(MANIFEST.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

