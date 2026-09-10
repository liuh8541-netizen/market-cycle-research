"""Incrementally cache all-market loan-collateral balance structure."""

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
DATASET = "TaiwanStockLoanCollateralBalance"
OUTPUT = ROOT / "data/processed/factors/loan_collateral_structure.csv"
MANIFEST = ROOT / "data/processed/factors/loan_collateral_fetched_dates.csv"
LOCK = ROOT / "data/processed/factors/.loan_collateral_fetch.lock"
TRADING_DATES = ROOT / "data/processed/factors/cross_sectional_breadth_fetched_dates.csv"
COMMON_STOCK = re.compile(r"^[1-9][0-9]{3}$")
CATEGORIES = [
    "Margin",
    "SecuritiesFirmLoan",
    "UnrestrictedLoan",
    "SecuritiesFinanceSecuredLoan",
    "SettlementMargin",
]


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


def numeric(data, column):
    return pd.to_numeric(data.get(column), errors="coerce").fillna(0).clip(lower=0)


def aggregate(day, rows):
    data = pd.DataFrame(rows)
    if data.empty or "stock_id" not in data:
        return {"date": day, "loan_collateral_stock_count": 0}
    ids = data["stock_id"].astype(str).str.strip()
    data = data.loc[
        ids.map(lambda value: bool(COMMON_STOCK.fullmatch(value)))
    ].copy()
    current = {}
    previous = {}
    quota = {}
    for category in CATEGORIES:
        current[category] = numeric(data, f"{category}CurrentDayBalance")
        previous[category] = numeric(data, f"{category}PreviousDayBalance")
        quota[category] = numeric(data, f"{category}NextDayQuota")
    other_categories = CATEGORIES[1:]
    other_current_by_stock = sum(current[name] for name in other_categories)
    other_previous_by_stock = sum(previous[name] for name in other_categories)
    other_quota_by_stock = sum(quota[name] for name in other_categories)
    other_total = float(other_current_by_stock.sum())
    margin_total = float(current["Margin"].sum())
    total_credit = margin_total + other_total
    other_weights = (
        other_current_by_stock / other_total
        if other_total > 0 else other_current_by_stock * np.nan
    )
    category_totals = {
        name: float(current[name].sum()) for name in other_categories
    }
    return {
        "date": day,
        "loan_collateral_stock_count": int(len(data)),
        "loan_margin_balance": margin_total,
        "loan_margin_change": float(
            current["Margin"].sum() - previous["Margin"].sum()
        ),
        "loan_margin_utilization": float(
            current["Margin"].sum()
            / (current["Margin"].sum() + quota["Margin"].sum())
        ) if current["Margin"].sum() + quota["Margin"].sum() > 0 else 0.0,
        "loan_other_balance": other_total,
        "loan_other_change": float(
            other_current_by_stock.sum() - other_previous_by_stock.sum()
        ),
        "loan_other_active_fraction": float(other_current_by_stock.gt(0).mean()),
        "loan_other_utilization": float(
            other_current_by_stock.sum()
            / (other_current_by_stock.sum() + other_quota_by_stock.sum())
        ) if other_current_by_stock.sum() + other_quota_by_stock.sum() > 0 else 0.0,
        "loan_other_top10_share": float(other_weights.nlargest(10).sum())
        if other_total > 0 else 0.0,
        "loan_other_hhi": float((other_weights ** 2).sum())
        if other_total > 0 else 0.0,
        "loan_other_share_of_total_credit": float(other_total / total_credit)
        if total_credit > 0 else 0.0,
        "loan_securities_firm_share": float(
            category_totals["SecuritiesFirmLoan"] / other_total
        ) if other_total > 0 else 0.0,
        "loan_unrestricted_share": float(
            category_totals["UnrestrictedLoan"] / other_total
        ) if other_total > 0 else 0.0,
        "loan_secured_finance_share": float(
            category_totals["SecuritiesFinanceSecuredLoan"] / other_total
        ) if other_total > 0 else 0.0,
        "loan_settlement_share": float(
            category_totals["SettlementMargin"] / other_total
        ) if other_total > 0 else 0.0,
    }


def atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2006-10-02")
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
