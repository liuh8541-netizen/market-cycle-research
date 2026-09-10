"""Checkpointed unlabeled backfill of cross-sectional market path geometry."""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import json
import math
import os
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "research/phase3_market_absorption_geometry"
API = "https://api.finmindtrade.com/api/v4/data"
DATASET = "TaiwanStockPrice"
OUTPUT = WORKSPACE / "data/market_absorption_geometry.csv"
MANIFEST = WORKSPACE / "data/market_absorption_geometry_fetched_dates.csv"
LOCK = WORKSPACE / "data/.market_absorption_geometry_fetch.lock"
TWII = ROOT / "data/processed/twii_daily.csv"
SECRET = ROOT / "config/.secrets/finmind_token.txt"
COMMON_STOCK = re.compile(r"^[1-9][0-9]{3}$")
MIN_COMMON_STOCKS = 100
FINMIND_ACCOUNT_HOURLY_LIMIT = 6000
DEFAULT_RESEARCH_HOURLY_BUDGET = 4500


class RollingHourlyBudget:
    """Limit this research process while reserving quota for other programs."""

    def __init__(self, maximum_calls: int) -> None:
        if not 1 <= maximum_calls < FINMIND_ACCOUNT_HOURLY_LIMIT:
            raise ValueError(
                "hourly call budget must be between 1 and "
                f"{FINMIND_ACCOUNT_HOURLY_LIMIT - 1}"
            )
        self.maximum_calls = maximum_calls
        self._calls: deque[float] = deque()
        self._condition = threading.Condition()

    def acquire(self) -> None:
        while True:
            with self._condition:
                now = time.monotonic()
                cutoff = now - 3600.0
                while self._calls and self._calls[0] <= cutoff:
                    self._calls.popleft()
                if len(self._calls) < self.maximum_calls:
                    self._calls.append(now)
                    return
                wait_seconds = max(0.05, self._calls[0] + 3600.0 - now)
            time.sleep(min(wait_seconds, 60.0))


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


def acquire_lock(total: int, pending: int, hourly_call_budget: int) -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            old = json.loads(LOCK.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            old = {}
        pid = int(old.get("pid", 0) or 0)
        if process_alive(pid):
            raise RuntimeError(f"Backfill already active under PID {pid}")
        stale = LOCK.with_name(
            LOCK.name + ".stale." + datetime.now().strftime("%Y%m%dT%H%M%S")
        )
        LOCK.replace(stale)
    LOCK.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "dataset": DATASET,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "total_dates": total,
                "pending_at_start": pending,
                "completed_this_run": 0,
                "research_hourly_call_budget": hourly_call_budget,
                "reserved_calls_for_other_programs": (
                    FINMIND_ACCOUNT_HOURLY_LIMIT - hourly_call_budget
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


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
            "User-Agent": "market-lifecycle-unknown-factor-research/1.0",
        },
    )
    with urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") not in (200, "200", None):
        raise RuntimeError(
            f"{day}: API status {payload.get('status')} {payload.get('msg', '')}"
        )
    return payload.get("data", [])


def fetch(
    day: str,
    token: str,
    hourly_budget: RollingHourlyBudget,
) -> tuple[str, list[dict], str]:
    transient = 0
    quota_waits = 0
    while True:
        try:
            hourly_budget.acquire()
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


def pick(data: pd.DataFrame, *names: str) -> pd.Series:
    lower = {str(column).lower(): column for column in data.columns}
    for name in names:
        if name.lower() in lower:
            return pd.to_numeric(data[lower[name.lower()]], errors="coerce")
    return pd.Series(np.nan, index=data.index, dtype=float)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & np.isfinite(values) & weights.gt(0) & np.isfinite(weights)
    total = float(weights.where(valid, 0.0).sum())
    if total <= 0:
        return 0.0
    return float((values.where(valid, 0.0) * weights.where(valid, 0.0)).sum() / total)


def weighted_fraction(condition: pd.Series, valid: pd.Series, weights: pd.Series) -> float:
    usable = valid & weights.gt(0) & np.isfinite(weights)
    total = float(weights.where(usable, 0.0).sum())
    if total <= 0:
        return 0.0
    return float(weights.where(usable & condition, 0.0).sum() / total)


def fraction(condition: pd.Series, valid: pd.Series) -> float:
    count = int(valid.sum())
    return float((condition & valid).sum() / count) if count else 0.0


def normalized_entropy(up: float, down: float, flat: float) -> float:
    values = np.asarray([up, down, flat], dtype=float)
    values = values[values > 0]
    if not len(values):
        return 0.0
    return float(-(values * np.log(values)).sum() / math.log(3.0))


def aggregate(day: str, rows: list[dict]) -> tuple[dict | None, str, int]:
    data = pd.DataFrame(rows)
    if data.empty or "stock_id" not in data:
        return None, "excluded_no_common_rows", 0
    ids = data["stock_id"].astype(str).str.strip()
    data = data.loc[
        ids.map(lambda value: bool(COMMON_STOCK.fullmatch(value)))
    ].copy()
    if data.empty:
        return None, "excluded_no_common_rows", 0
    data["stock_id"] = data["stock_id"].astype(str).str.strip()
    before = len(data)
    data = data.drop_duplicates("stock_id", keep="last")
    duplicates = before - len(data)
    close = pick(data, "close")
    high = pick(data, "max", "high")
    low = pick(data, "min", "low")
    spread = pick(data, "spread")
    money = pick(data, "Trading_money", "trading_money").fillna(0).clip(lower=0)
    prior = close - spread
    valid = (
        close.gt(0)
        & prior.gt(0)
        & high.notna()
        & low.notna()
        & high.ge(low)
        & high.ge(close)
        & low.le(close)
        & np.isfinite(close)
        & np.isfinite(prior)
        & np.isfinite(high)
        & np.isfinite(low)
    )
    if int(valid.sum()) < MIN_COMMON_STOCKS:
        return None, "excluded_low_common_stock_coverage", int(valid.sum())
    day_range = (high - low) / prior
    geometry = valid & day_range.gt(0) & np.isfinite(day_range)
    close_location = ((close - low) / (high - low).replace(0, np.nan)).clip(0, 1)
    daily_return = spread / prior
    signed_efficiency = daily_return / day_range.replace(0, np.nan)
    path_efficiency = signed_efficiency.abs()
    upper_excursion = ((high - prior) / prior).clip(lower=0)
    lower_excursion = ((prior - low) / prior).clip(lower=0)
    up = fraction(daily_return > 0, valid)
    down = fraction(daily_return < 0, valid)
    flat = fraction(daily_return == 0, valid)
    result = {
        "date": day,
        "geometry_stock_count": int(valid.sum()),
        "geometry_range_stock_count": int(geometry.sum()),
        "source_duplicate_stock_rows": int(duplicates),
        "zero_range_fraction": fraction(day_range == 0, valid),
        "return_up_fraction": up,
        "return_down_fraction": down,
        "return_flat_fraction": flat,
        "return_sign_entropy": normalized_entropy(up, down, flat),
        "return_direction_synchronization": abs(up - down),
        "close_location_mean": float(close_location.where(geometry).mean()),
        "close_location_median": float(close_location.where(geometry).median()),
        "close_location_dispersion": float(
            close_location.where(geometry).std(ddof=0)
        ),
        "close_location_money_weighted": weighted_mean(
            close_location.where(geometry), money
        ),
        "low_close_fraction": fraction(close_location <= 0.2, geometry),
        "high_close_fraction": fraction(close_location >= 0.8, geometry),
        "low_close_money_fraction": weighted_fraction(
            close_location <= 0.2, geometry, money
        ),
        "high_close_money_fraction": weighted_fraction(
            close_location >= 0.8, geometry, money
        ),
        "down_but_recovered_fraction": fraction(
            (daily_return < 0) & (close_location >= 0.5), geometry
        ),
        "up_but_faded_fraction": fraction(
            (daily_return > 0) & (close_location <= 0.5), geometry
        ),
        "down_but_recovered_money_fraction": weighted_fraction(
            (daily_return < 0) & (close_location >= 0.5), geometry, money
        ),
        "up_but_faded_money_fraction": weighted_fraction(
            (daily_return > 0) & (close_location <= 0.5), geometry, money
        ),
        "path_efficiency_median": float(path_efficiency.where(geometry).median()),
        "path_efficiency_q75": float(
            path_efficiency.where(geometry).quantile(0.75)
        ),
        "one_way_path_fraction": fraction(path_efficiency >= 0.8, geometry),
        "churn_path_fraction": fraction(path_efficiency <= 0.2, geometry),
        "signed_efficiency_median": float(
            signed_efficiency.where(geometry).median()
        ),
        "signed_efficiency_money_weighted": weighted_mean(
            signed_efficiency.where(geometry), money
        ),
        "range_median": float(day_range.where(geometry).median()),
        "range_dispersion": float(day_range.where(geometry).std(ddof=0)),
        "range_q90": float(day_range.where(geometry).quantile(0.90)),
        "upper_excursion_q90": float(
            upper_excursion.where(geometry).quantile(0.90)
        ),
        "lower_excursion_q90": float(
            lower_excursion.where(geometry).quantile(0.90)
        ),
        "excursion_tail_asymmetry": float(
            upper_excursion.where(geometry).quantile(0.90)
            - lower_excursion.where(geometry).quantile(0.90)
        ),
        "negative_return_money_fraction": weighted_fraction(
            daily_return < 0, valid, money
        ),
        "positive_return_money_fraction": weighted_fraction(
            daily_return > 0, valid, money
        ),
    }
    if not all(
        isinstance(value, str) or math.isfinite(float(value))
        for value in result.values()
    ):
        raise RuntimeError(f"{day}: aggregate contains non-finite values")
    return result, "success", int(valid.sum())


def expected_dates(start_date: str, end_date: str) -> list[str]:
    data = pd.read_csv(TWII, usecols=["date"])
    dates = pd.to_datetime(data["date"], errors="coerce").dropna()
    return (
        dates.loc[dates.between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
        .drop_duplicates()
        .sort_values()
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )


def save(
    features: list[dict],
    manifest: list[dict],
    completed_this_run: int,
    pending_at_start: int,
) -> None:
    feature_frame = pd.DataFrame(features)
    if not feature_frame.empty:
        atomic_csv(
            feature_frame.sort_values("date")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True),
            OUTPUT,
        )
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
    parser.add_argument("--start-date", default="1997-07-02")
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--checkpoint", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument(
        "--hourly-call-budget",
        type=int,
        default=DEFAULT_RESEARCH_HOURLY_BUDGET,
        help=(
            "Maximum FinMind requests made by this research process in any "
            "rolling 60-minute window. The default reserves 1,500 of the "
            "6,000 account calls for other programs."
        ),
    )
    args = parser.parse_args()
    hourly_budget = RollingHourlyBudget(args.hourly_call_budget)
    token = os.environ.get("FINMIND_TOKEN", "").strip()
    if not token and SECRET.exists():
        token = SECRET.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("FinMind Token is not configured")
    dates = expected_dates(args.start_date, args.end_date)
    old_features = pd.read_csv(OUTPUT) if OUTPUT.exists() else pd.DataFrame()
    old_manifest = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame()
    terminal_statuses = {
        "success",
        "excluded_no_common_rows",
        "excluded_low_common_stock_coverage",
    }
    completed = (
        set(
            old_manifest.loc[
                old_manifest["status"].isin(terminal_statuses), "date"
            ].astype(str)
        )
        if not old_manifest.empty
        else set()
    )
    pending = [day for day in dates if day not in completed]
    acquire_lock(len(dates), len(pending), args.hourly_call_budget)
    features = [] if old_features.empty else old_features.to_dict("records")
    manifest = [] if old_manifest.empty else old_manifest.to_dict("records")
    done = 0
    try:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, args.workers)
            ) as pool:
                futures = [
                    pool.submit(fetch, day, token, hourly_budget) for day in batch
                ]
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
