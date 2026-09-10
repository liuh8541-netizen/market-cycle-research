import argparse
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


API_URL = "https://api.finmindtrade.com/api/v4/data"
ROOT = Path(__file__).resolve().parents[1]


def load_finmind_token() -> str:
    token = os.environ.get("FINMIND_TOKEN", "").strip()
    if token:
        return token
    token_file = ROOT / "config" / ".secrets" / "finmind_token.txt"
    if token_file.exists():
        return token_file.read_text(encoding="utf-8").strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch configured FinMind factor datasets.")
    parser.add_argument("--config", default="config/finmind_factor_datasets.json")
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--token", default=load_finmind_token())
    parser.add_argument("--only", help="Fetch only one configured dataset name.")
    parser.add_argument("--full", action="store_true", help="Ignore the latest stored date and rebuild from the configured start date.")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("FINMIND_TOKEN is required. Set env var or pass --token.")

    datasets = json.loads(Path(args.config).read_text(encoding="utf-8"))
    failures = []
    for item in datasets:
        if args.only and item["name"] != args.only:
            continue
        output = Path(item["output"])
        initial_start = item.get("start_date", args.start_date)
        if "start_date" not in item:
            initial_start = args.start_date
        fetch_start = initial_start if args.full else incremental_start_date(output, initial_start)
        try:
            downloaded, merged = update_dataset_in_chunks(
                item,
                output,
                fetch_start,
                args.end_date,
                args.token,
            )
            print(f"{item['name']}: {downloaded} downloaded, {len(merged)} stored -> {output}")
        except Exception as exc:
            failures.append(f"{item['name']}: {exc}")
            print(f"WARNING {item['name']}: update skipped after retries: {exc}")

    if failures:
        print("FinMind update completed with partial failures:")
        for failure in failures:
            print("- " + failure)


def incremental_start_date(output: Path, fallback: str, overlap_days: int = 7) -> str:
    if not output.exists():
        return fallback
    existing = pd.read_csv(output, usecols=["date"])
    if existing.empty:
        return fallback
    latest = pd.to_datetime(existing["date"], errors="coerce").max()
    if pd.isna(latest):
        return fallback
    return (latest.date() - timedelta(days=overlap_days)).isoformat()


def replace_date_window(output: Path, rows: list[dict], fetch_start: str) -> pd.DataFrame:
    fresh = pd.DataFrame(rows)
    if not output.exists():
        return fresh
    existing = pd.read_csv(output)
    if existing.empty or "date" not in existing:
        return fresh
    cutoff = pd.to_datetime(fetch_start)
    keep = existing[pd.to_datetime(existing["date"], errors="coerce") < cutoff]
    if fresh.empty:
        return existing
    columns = list(dict.fromkeys(list(existing.columns) + list(fresh.columns)))
    return pd.concat([keep.reindex(columns=columns), fresh.reindex(columns=columns)], ignore_index=True)


def fetch_dataset_batched(
    dataset: str,
    start_date: str,
    end_date: str,
    token: str,
    data_id: str,
    chunk_days: int,
    display_name: str,
) -> list[dict]:
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    rows = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        print(f"{display_name}: fetching {cursor} to {chunk_end}...")
        rows.extend(fetch_with_retry(dataset, cursor.isoformat(), chunk_end.isoformat(), token, data_id))
        cursor = chunk_end + timedelta(days=1)
    return rows


def update_dataset_in_chunks(
    item: dict,
    output: Path,
    start_date: str,
    end_date: str,
    token: str,
) -> tuple[int, pd.DataFrame]:
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    chunk_days = int(item.get("chunk_days", 366))
    data_id = item.get("data_id", "")
    downloaded = 0
    cursor = start
    output.parent.mkdir(parents=True, exist_ok=True)

    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        print(f"{item['name']}: fetching {cursor} to {chunk_end}...")
        raw_rows = fetch_with_retry(
            item["dataset"],
            cursor.isoformat(),
            chunk_end.isoformat(),
            token,
            data_id,
        )
        downloaded += len(raw_rows)
        normalized = normalize_dataset_rows(item["name"], raw_rows)
        if normalized:
            merged = replace_date_window(output, normalized, cursor.isoformat())
            merged.to_csv(output, index=False, encoding="utf-8")
        cursor = chunk_end + timedelta(days=1)

    merged = pd.read_csv(output) if output.exists() else pd.DataFrame()
    return downloaded, merged


def fetch_with_retry(dataset: str, start_date: str, end_date: str, token: str, data_id: str) -> list[dict]:
    last_error = None
    for attempt in range(3):
        try:
            return fetch_dataset(dataset, start_date, end_date, token, data_id)
        except (TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise last_error


def normalize_dataset_rows(name: str, rows: list[dict]) -> list[dict]:
    if name != "option_daily" or not rows:
        return rows
    data = pd.DataFrame(rows)
    if not {"date", "call_put", "volume"}.issubset(data.columns):
        return rows
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce").fillna(0)
    if "open_interest" in data:
        data["open_interest"] = pd.to_numeric(data["open_interest"], errors="coerce").fillna(0)
    call_put = data["call_put"].astype(str).str.lower()
    data["side"] = call_put.map(lambda value: "put" if value in ["put", "p", "賣權"] else "call")
    values = ["volume"] + (["open_interest"] if "open_interest" in data else [])
    grouped = data.groupby(["date", "side"])[values].sum().unstack(fill_value=0)
    grouped.columns = [f"{side}_{metric}" for metric, side in grouped.columns]
    return grouped.reset_index().to_dict("records")


def fetch_dataset(dataset: str, start_date: str, end_date: str, token: str, data_id: str = "") -> list[dict]:
    params = {
        "dataset": dataset,
        "start_date": start_date,
        "end_date": end_date,
        "token": token,
    }
    if data_id:
        params["data_id"] = data_id
    url = API_URL + "?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "market-lifecycle-research/1.0"})
    with urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("status") not in [200, "200", None]:
        raise RuntimeError(f"FinMind failed for {dataset}: {payload}")
    return payload.get("data", [])


if __name__ == "__main__":
    main()
