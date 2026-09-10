"""Fetch the free rolling three-year daily Taiwan VIX series from TAIFEX."""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.taifex.com.tw/indes/index.aspx/GetStockDayPrices"
OUTPUT = ROOT / "data" / "processed" / "factors" / "taifex_vix_daily.csv"


def resolve_effective_end_date(requested_end: date, today: date | None = None) -> date:
    """Clamp TAIFEX daily-history queries to the latest completed calendar day."""
    current_date = today or date.today()
    return min(requested_end, current_date - timedelta(days=1))


def decode_response(envelope: object) -> dict:
    """Decode the ASP.NET response envelope and preserve TAIFEX error text."""
    if not isinstance(envelope, dict) or "d" not in envelope:
        raise RuntimeError("TAIFEX returned an unexpected response envelope")
    payload = envelope["d"]
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"TAIFEX API error: {payload}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("TAIFEX returned an unexpected Taiwan VIX payload")
    return payload


def fetch(start_date: date, end_date: date) -> pd.DataFrame:
    payload = json.dumps({
        "syid": "TAIWANVIX",
        "flag": "MS",
        "startDate": start_date.strftime("%Y/%m/%d"),
        "endDate": end_date.strftime("%Y/%m/%d"),
    }).encode("utf-8")
    request = Request(
        URL,
        data=payload,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "Mozilla/5.0 market-research/1.0",
        },
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        envelope = json.loads(response.read().decode("utf-8"))
    body = decode_response(envelope)
    rows = body.get("TrendData", [])
    frame = pd.DataFrame(rows).rename(columns={"Time": "date", "Price": "vix_close"})
    if frame.empty:
        raise RuntimeError("TAIFEX returned no Taiwan VIX observations")
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="coerce")
    frame["vix_close"] = pd.to_numeric(frame["vix_close"], errors="coerce")
    frame = frame.dropna().drop_duplicates("date", keep="last").sort_values("date")
    frame["source"] = "TAIFEX"
    frame["availability"] = "after_cash_close"
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-date", default=date.today().isoformat())
    args = parser.parse_args()
    requested_end = pd.Timestamp(args.end_date).date()
    end_date = resolve_effective_end_date(requested_end)
    start_date = end_date - timedelta(days=3 * 365)
    frame = fetch(start_date, end_date)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(
        f"TAIFEX Taiwan VIX: {len(frame)} daily closes, "
        f"{frame.date.min().date()} to {frame.date.max().date()} "
        f"(requested {requested_end}, effective {end_date}) -> {OUTPUT}"
    )


if __name__ == "__main__":
    main()
