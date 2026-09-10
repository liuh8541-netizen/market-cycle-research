"""Build the detailed TWII market clinical ledger and reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backtest_psychology_state import replay_psychology_history
from market_lifecycle.clinical_ledger import (
    build_complete_ledger,
    flatten_records,
    render_ledger_report,
    summarize_ledger,
)
from market_lifecycle.clinical_enrichment import (
    build_intraday_course_15m,
    current_auxiliary_snapshot,
    current_cash_technical_snapshot,
    enrich_auxiliary_vitals,
    enrich_forward_outcomes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build complete outcome-aware market clinical records.")
    parser.add_argument("--cash", default="data/processed/twii_daily.csv")
    parser.add_argument("--night", default="data/processed/taiwan_futures_night.csv")
    parser.add_argument("--external", default="data/processed/external_markets.csv")
    parser.add_argument("--current", default="reports/today_market_forecast.json")
    parser.add_argument("--factor-dir", default="data/processed/factors")
    parser.add_argument("--jsonl-output", default="research/market_health_knowledge/complete_market_records.jsonl")
    parser.add_argument("--csv-output", default="reports/market_clinical_ledger.csv")
    parser.add_argument("--json-output", default="reports/market_clinical_ledger.json")
    parser.add_argument("--md-output", default="reports/market_clinical_ledger.md")
    parser.add_argument("--intraday-output", default="reports/market_intraday_course_15m.csv")
    args = parser.parse_args()

    cash = pd.read_csv(rooted(args.cash))
    replay = replay_psychology_history(
        cash, pd.read_csv(rooted(args.night)), pd.read_csv(rooted(args.external))
    )
    replay = enrich_forward_outcomes(replay, cash)
    factor_dir = rooted(args.factor_dir)
    breadth = read_optional(factor_dir / "cross_sectional_breadth.csv")
    futures_daily = read_optional(factor_dir / "futures_daily.csv")
    futures_institutional = read_optional(factor_dir / "futures_institutional.csv")
    night_microstructure = read_optional(factor_dir / "night_microstructure.csv")
    intraday_detail, intraday_summary = build_intraday_course_15m(
        read_optional(factor_dir / "futures_tick_15m.csv")
    )
    replay = enrich_auxiliary_vitals(
        replay, breadth, futures_daily, futures_institutional, night_microstructure
    )
    if not intraday_summary.empty:
        replay = replay.merge(intraday_summary, on="signal_date", how="left", validate="one_to_one")
    current_path = rooted(args.current)
    current = json.loads(current_path.read_text(encoding="utf-8")) if current_path.exists() else None
    if current:
        decision_date = str(current.get("input", {}).get("date"))
        current["_clinical_aux"] = {
            **current_auxiliary_snapshot(
                decision_date, breadth, futures_daily, futures_institutional, night_microstructure
            ),
            **current_cash_technical_snapshot(cash, decision_date),
        }
    records = build_complete_ledger(replay, current)
    flat = flatten_records(records)
    summary = summarize_ledger(records)

    destinations = [rooted(args.jsonl_output), rooted(args.csv_output), rooted(args.json_output), rooted(args.md_output)]
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
    destinations[0].write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in records) + "\n",
        encoding="utf-8",
    )
    flat.to_csv(destinations[1], index=False, encoding="utf-8-sig")
    destinations[2].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    destinations[3].write_text(render_ledger_report(summary), encoding="utf-8")
    intraday_output = rooted(args.intraday_output)
    intraday_output.parent.mkdir(parents=True, exist_ok=True)
    intraday_detail.to_csv(intraday_output, index=False, encoding="utf-8-sig")
    print(
        f"records={summary['records']} resolved={summary['resolved_records']} "
        f"pending={summary['pending_records']} leakage_audit={summary['leakage_audit']['passed']}"
    )
    print(f"Report: {destinations[3]}")


def rooted(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


if __name__ == "__main__":
    main()
