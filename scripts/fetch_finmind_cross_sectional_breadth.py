"""Build point-in-time Taiwan stock price/institutional breadth from FinMind.

The universe is determined independently on every source date.  This avoids
using today's constituent list to filter historical observations.  Raw
security rows are reduced to one daily feature row and are not persisted.
"""

import argparse
import concurrent.futures
import json
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
API = "https://api.finmindtrade.com/api/v4/data"
PRICE_DATASET = "TaiwanStockPrice"
INST_DATASET = "TaiwanStockInstitutionalInvestorsBuySellWide"
OUTPUT = ROOT / "data/processed/factors/cross_sectional_breadth.csv"
MANIFEST = ROOT / "data/processed/factors/cross_sectional_breadth_fetched_dates.csv"
LOCK = ROOT / "data/processed/factors/.cross_sectional_breadth_fetch.lock"
TWII = ROOT / "data/processed/twii_daily.csv"
COMMON_STOCK = re.compile(r"^[1-9][0-9]{3}$")


def request_rows(dataset, start_date, end_date, token):
    params = {
        "dataset": dataset,
        "start_date": start_date,
        "end_date": end_date,
        "token": token,
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
        raise RuntimeError(f"{dataset}: {payload.get('status')} {payload.get('msg')}")
    return payload.get("data", [])


def fetch_with_retry(dataset, start_date, end_date, token):
    error = None
    transient_attempt = 0
    while True:
        try:
            return request_rows(dataset, start_date, end_date, token)
        except HTTPError as exc:
            error = exc
            if exc.code == 402:
                print(
                    f"{dataset}: hourly quota reached; waiting 30 seconds...",
                    flush=True,
                )
                time.sleep(30)
                continue
            transient_attempt += 1
        except (TimeoutError, OSError, RuntimeError) as exc:
            error = exc
            transient_attempt += 1
        if transient_attempt >= 4:
            raise error
        time.sleep(3 * transient_attempt)


def point_in_time_common_stock(rows):
    data = pd.DataFrame(rows)
    if data.empty or "stock_id" not in data:
        return data
    ids = data["stock_id"].astype(str).str.strip()
    return data.loc[ids.map(lambda value: bool(COMMON_STOCK.fullmatch(value)))].copy()


def safe_fraction(condition, valid):
    valid_count = int(valid.sum())
    return float((condition & valid).sum() / valid_count) if valid_count else np.nan


def top_share(values, count=10):
    values = pd.to_numeric(values, errors="coerce").fillna(0).clip(lower=0)
    total = float(values.sum())
    return float(values.nlargest(count).sum() / total) if total > 0 else np.nan


def aggregate_price(rows):
    data = point_in_time_common_stock(rows)
    if data.empty:
        return {}
    for column in ["close", "spread", "Trading_Volume", "Trading_money"]:
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    prior_close = data["close"] - data["spread"]
    ret = data["spread"] / prior_close.replace(0, np.nan)
    valid = ret.notna() & np.isfinite(ret) & prior_close.gt(0) & data["close"].gt(0)
    volume = data["Trading_Volume"].where(valid).fillna(0).clip(lower=0)
    money = data["Trading_money"].where(valid).fillna(0).clip(lower=0)
    usable_return = ret.where(valid)
    volume_total = float(volume.sum())
    output = {
        "price_stock_count": int(valid.sum()),
        "price_advancing_fraction": safe_fraction(ret > 0, valid),
        "price_declining_fraction": safe_fraction(ret < 0, valid),
        "price_unchanged_fraction": safe_fraction(ret == 0, valid),
        "price_equal_weight_return": float(usable_return.mean()),
        "price_median_return": float(usable_return.median()),
        "price_return_dispersion": float(usable_return.std(ddof=0)),
        "price_upper_tail_fraction": safe_fraction(ret >= 0.02, valid),
        "price_lower_tail_fraction": safe_fraction(ret <= -0.02, valid),
        "price_up_volume_fraction": (
            float(volume.where(ret > 0, 0).sum() / volume_total)
            if volume_total > 0 else np.nan
        ),
        "price_volume_weighted_return": (
            float((usable_return.fillna(0) * volume).sum() / volume_total)
            if volume_total > 0 else np.nan
        ),
        "price_money_top10_share": top_share(money),
        "price_median_log_money": float(np.log1p(money.where(valid)).median()),
    }
    output["price_advance_decline_breadth"] = (
        output["price_advancing_fraction"] - output["price_declining_fraction"]
    )
    return output


def numeric(data, column):
    if column not in data:
        return pd.Series(0.0, index=data.index)
    return pd.to_numeric(data[column], errors="coerce").fillna(0.0)


def investor_pair(data, prefix):
    return numeric(data, prefix + "_buy"), numeric(data, prefix + "_sell")


def aggregate_institutional(rows):
    data = point_in_time_common_stock(rows)
    if data.empty:
        return {}
    foreign_buy, foreign_sell = investor_pair(data, "Foreign_Investor")
    trust_buy, trust_sell = investor_pair(data, "Investment_Trust")
    dealer_buy = sum(
        (numeric(data, name + "_buy") for name in
         ["Dealer", "Dealer_self", "Dealer_Hedging", "Foreign_Dealer_Self"]),
        start=pd.Series(0.0, index=data.index),
    )
    dealer_sell = sum(
        (numeric(data, name + "_sell") for name in
         ["Dealer", "Dealer_self", "Dealer_Hedging", "Foreign_Dealer_Self"]),
        start=pd.Series(0.0, index=data.index),
    )
    pairs = {
        "foreign": (foreign_buy, foreign_sell),
        "trust": (trust_buy, trust_sell),
        "dealer": (dealer_buy, dealer_sell),
    }
    output = {}
    combined_net = pd.Series(0.0, index=data.index)
    combined_gross = pd.Series(0.0, index=data.index)
    for label, (buy, sell) in pairs.items():
        net = buy - sell
        gross = buy + sell
        active = gross > 0
        intensity = net / gross.replace(0, np.nan)
        output[f"inst_{label}_active_count"] = int(active.sum())
        output[f"inst_{label}_positive_fraction"] = safe_fraction(net > 0, active)
        output[f"inst_{label}_negative_fraction"] = safe_fraction(net < 0, active)
        output[f"inst_{label}_median_intensity"] = float(intensity.where(active).median())
        output[f"inst_{label}_aggregate_intensity"] = (
            float(net.sum() / gross.sum()) if gross.sum() > 0 else np.nan
        )
        combined_net += net
        combined_gross += gross
    active = combined_gross > 0
    combined_intensity = combined_net / combined_gross.replace(0, np.nan)
    output.update({
        "inst_stock_count": int(active.sum()),
        "inst_combined_positive_fraction": safe_fraction(combined_net > 0, active),
        "inst_combined_negative_fraction": safe_fraction(combined_net < 0, active),
        "inst_combined_median_intensity": float(combined_intensity.where(active).median()),
        "inst_combined_aggregate_intensity": (
            float(combined_net.sum() / combined_gross.sum())
            if combined_gross.sum() > 0 else np.nan
        ),
        "inst_absolute_net_top10_share": top_share(combined_net.abs()),
    })
    both = ((foreign_buy + foreign_sell) > 0) & ((trust_buy + trust_sell) > 0)
    output["inst_foreign_trust_disagreement"] = safe_fraction(
        np.sign(foreign_buy - foreign_sell) != np.sign(trust_buy - trust_sell),
        both,
    )
    output["inst_combined_breadth"] = (
        output["inst_combined_positive_fraction"]
        - output["inst_combined_negative_fraction"]
    )
    return output


def atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def existing(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def merge_rows(path, rows, key):
    old = existing(path)
    fresh = pd.DataFrame(rows)
    combined = pd.concat([old, fresh], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(key, keep="last").sort_values(key)
    atomic_csv(combined, path)
    return combined


def expected_dates(start_date, end_date):
    twii = pd.read_csv(TWII, usecols=["date"])
    dates = pd.to_datetime(twii["date"], errors="coerce").dropna()
    mask = dates.between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    return [value.date().isoformat() for value in dates.loc[mask].drop_duplicates()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2005-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--token", default=os.environ.get("FINMIND_TOKEN", ""))
    args = parser.parse_args()
    if not args.token:
        token_file = ROOT / "config/.secrets/finmind_token.txt"
        if token_file.exists():
            args.token = token_file.read_text(encoding="utf-8").strip()
    if not args.token:
        raise SystemExit("FINMIND_TOKEN is required.")

    dates = expected_dates(args.start_date, args.end_date)
    manifest = existing(MANIFEST)
    terminal_statuses = {
        "complete",
        "excluded_low_institutional_coverage",
        "excluded_no_market_source",
    }
    terminal = set(
        manifest.loc[
            manifest.get("status", pd.Series(dtype=str)).isin(terminal_statuses),
            "date",
        ].astype(str)
    ) if not manifest.empty else set()
    pending = [value for value in dates if value not in terminal]
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(os.getpid()), encoding="ascii")
    try:
        for number, day in enumerate(pending, start=1):
            print(
                f"cross_sectional_breadth: fetching {day} "
                f"({number}/{len(pending)})...",
                flush=True,
            )
            # In all-market mode FinMind returns only start_date even when an
            # end_date is supplied.  The two independent sources are therefore
            # requested concurrently for exactly one trading date.
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                price_future = pool.submit(
                    fetch_with_retry, PRICE_DATASET, day, day, args.token
                )
                inst_future = pool.submit(
                    fetch_with_retry, INST_DATASET, day, day, args.token
                )
                price_rows = price_future.result()
                inst_rows = inst_future.result()
            price = aggregate_price(price_rows)
            institutional = aggregate_institutional(inst_rows)
            price_good = price.get("price_stock_count", 0) >= 100
            institutional_good = institutional.get("inst_stock_count", 0) >= 50
            no_market_source = not price_rows and not inst_rows
            good = price_good and institutional_good
            if good:
                merge_rows(
                    OUTPUT,
                    [{"date": day, **price, **institutional}],
                    "date",
                )
            manifest = merge_rows(
                MANIFEST,
                [{
                    "date": day,
                    "status": (
                        "complete" if good else
                        "excluded_no_market_source"
                        if no_market_source else
                        "excluded_low_institutional_coverage"
                        if price_good else "incomplete_price_source"
                    ),
                    "feature_rows": int(good),
                    "price_source_rows": len(price_rows),
                    "institutional_source_rows": len(inst_rows),
                    "fetched_at": pd.Timestamp.now(
                        tz="Asia/Taipei"
                    ).isoformat(),
                }],
                "date",
            )
            if no_market_source:
                print(
                    f"cross_sectional_breadth: {day} excluded; "
                    "both market sources returned zero rows.",
                    flush=True,
                )
                continue
            if not price_good:
                raise RuntimeError(
                    f"incomplete price source date {day}: "
                    f"price={len(price_rows)}, institutional={len(inst_rows)}"
                )
            if not institutional_good:
                print(
                    f"cross_sectional_breadth: {day} excluded; "
                    f"institutional active stocks="
                    f"{institutional.get('inst_stock_count', 0)}.",
                    flush=True,
                )
                continue
            print(
                f"cross_sectional_breadth: {day} complete; "
                f"{len(manifest)}/{len(dates)} manifest rows."
                ,
                flush=True,
            )
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
