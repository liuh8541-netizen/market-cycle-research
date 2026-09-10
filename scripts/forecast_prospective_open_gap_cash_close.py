"""Time-locked prospective logger for open_gap_acceptance_q85_v1.

The logger may create a prediction only during the Taiwan cash session.  It
never backfills a prediction after 13:30 and resolves outcomes no earlier than
the following calendar day, preventing a live/closing price from leaking into
the immutable forecast record.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TAIPEI = timezone(timedelta(hours=8))
MANIFEST = ROOT / "config" / "prospective_open_gap_cash_close_candidate.json"
DEFAULT_LOG = ROOT / "reports" / "prospective_open_gap_cash_close_log.json"
DEFAULT_STATUS = ROOT / "reports" / "prospective_open_gap_cash_close_status.md"


def atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", help="Testing override, ISO local datetime with or without offset.")
    parser.add_argument("--observed-date", help="Testing/manual observed opening date YYYY-MM-DD.")
    parser.add_argument("--observed-open", type=float, help="Testing/manual observed official opening value.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--no-fetch", action="store_true")
    return parser.parse_args()


def local_now(value=None):
    if not value:
        return datetime.now(TAIPEI)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=TAIPEI)
    return parsed.astimezone(TAIPEI)


def phase(now):
    if now.weekday() >= 5:
        return "non_trading_day"
    if now.time() < time(9, 0):
        return "preopen"
    if now.time() < time(13, 30):
        return "capture_window"
    return "postclose_no_backfill"


def load_cash():
    frame = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv", usecols=["date", "open", "close"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def completed_gap_history(cash, before_date):
    work = cash.loc[cash["date"].dt.date < before_date].copy()
    work["prior_close"] = work["close"].shift(1)
    work["gap_return"] = work["open"] / work["prior_close"] - 1
    return work.dropna(subset=["gap_return", "close"])


def fetch_observed_open(day):
    # Importing does not run the daily report; the raw Yahoo response is read
    # without writing an intraday candle into canonical twii_daily.csv.
    sys.path.insert(0, str(ROOT / "scripts"))
    from predict_market import fetch_yahoo_chart, yahoo_chart_to_rows

    rows = yahoo_chart_to_rows(fetch_yahoo_chart("^TWII"))
    match = [row for row in rows if row.get("date") == day.isoformat() and row.get("open") is not None]
    return float(match[-1]["open"]) if match else None


def night_baseline(day):
    path = ROOT / "data" / "processed" / "taiwan_futures_night.csv"
    frame = pd.read_csv(path, usecols=["signal_date", "tx_night_spread_per"])
    row = frame.loc[frame["signal_date"].astype(str) == day.isoformat()]
    if row.empty:
        return None
    value = float(pd.to_numeric(row.iloc[-1]["tx_night_spread_per"], errors="coerce"))
    return 1 if value >= 0 else -1


def wilson(hits, cases):
    if not cases:
        return 0.0
    z = 1.959963984540054
    p = hits / cases
    den = 1 + z*z/cases
    return (p + z*z/(2*cases) - z*math.sqrt((p*(1-p)+z*z/(4*cases))/cases))/den


def mcnemar(a_only, b_only):
    n = a_only + b_only
    if not n:
        return 1.0
    low = min(a_only, b_only)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(low + 1)) / (2**n))


def reconcile(records, cash, today):
    outcomes = cash.loc[cash["date"].dt.date < today].copy()
    outcomes["prior_close"] = outcomes["close"].shift(1)
    outcomes["cash_return"] = outcomes["close"] / outcomes["prior_close"] - 1
    by_date = outcomes.set_index(outcomes["date"].dt.date.astype(str))
    for record in records:
        day = record["signal_date"]
        if day not in by_date.index or pd.isna(by_date.loc[day, "cash_return"]):
            continue
        value = float(by_date.loc[day, "cash_return"])
        actual = 1 if value >= 0 else -1
        # Only outcome fields are mutable; forecast, threshold and actionability
        # remain exactly as captured during the session.
        record["outcome"] = "up" if actual > 0 else "down"
        record["actual_cash_return"] = value
        record["hit"] = bool(record["prediction_value"] == actual) if record["actionable"] else None
        baseline = record.get("night_spread_baseline_value")
        record["night_spread_baseline_hit"] = bool(baseline == actual) if record["actionable"] and baseline is not None else None


def metrics(records, manifest):
    resolved = [row for row in records if row.get("outcome") is not None]
    actionable = [row for row in resolved if row.get("actionable")]
    cases = len(actionable)
    hits = sum(bool(row.get("hit")) for row in actionable)
    paired = [row for row in actionable if row.get("night_spread_baseline_hit") is not None]
    night_hits = sum(bool(row.get("night_spread_baseline_hit")) for row in paired)
    model_only = sum(bool(row.get("hit")) and not bool(row.get("night_spread_baseline_hit")) for row in paired)
    night_only = sum(not bool(row.get("hit")) and bool(row.get("night_spread_baseline_hit")) for row in paired)
    coverage = cases / len(resolved) if resolved else 0.0
    accuracy = hits / cases if cases else 0.0
    lower = wilson(hits, cases)
    p_value = mcnemar(model_only, night_only)
    gate = manifest["promotion_gate"]
    passed = (
        cases >= gate["prospective_resolved_actionable_cases"]
        and accuracy >= gate["accuracy"] and coverage >= gate["coverage"]
        and lower >= gate["wilson_95_lower"] and model_only > night_only and p_value < .05
    )
    return {
        "records": len(records), "resolved": len(resolved), "cases": cases,
        "hits": hits, "accuracy": accuracy, "coverage": coverage,
        "wilson_95_lower": lower, "paired_cases": len(paired), "night_hits": night_hits,
        "night_accuracy": night_hits / len(paired) if paired else 0.0,
        "model_only": model_only, "night_only": night_only,
        "mcnemar_exact_p": p_value, "forward_gate_passed": passed,
    }


def write_status(path, manifest, now, current_phase, threshold, observed_open, created, stats):
    lines = [
        "# Prospective opening-gap cash-close candidate status", "",
        f"Candidate: {manifest['candidate_id']}",
        f"Run time (Asia/Taipei): {now.isoformat()}",
        f"Phase: {current_phase}",
        f"Today's locked Q85 trigger: {threshold:.4%}" if threshold is not None else "Today's locked Q85 trigger: unavailable",
        f"Observed open: {observed_open:.4f}" if observed_open is not None else "Observed open: not captured",
        f"New immutable forecast written: {created}", "",
        f"Prospective records/resolved/actionable: {stats['records']}/{stats['resolved']}/{stats['cases']}",
        f"Model: {stats['hits']}/{stats['cases']} = {stats['accuracy']:.2%}",
        f"Coverage: {stats['coverage']:.2%}",
        f"Wilson 95% lower: {stats['wilson_95_lower']:.2%}",
        f"Night-spread paired baseline: {stats['night_hits']}/{stats['paired_cases']} = {stats['night_accuracy']:.2%}",
        f"Model-only/night-only: {stats['model_only']}/{stats['night_only']}; p={stats['mcnemar_exact_p']:.4g}",
        f"Forward gate passed: {stats['forward_gate_passed']}", "",
        "A forecast can be created only from 09:00 through 13:29 Taiwan time. Post-close backfilling is forbidden.",
        manifest["warning"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    now = local_now(args.now)
    today = now.date()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = json.loads(args.log.read_text(encoding="utf-8")) if args.log.exists() else []
    cash = load_cash()
    reconcile(records, cash, today)

    history = completed_gap_history(cash, today)
    history_days = int(manifest["parameters_locked"]["history_days"])
    quantile = float(manifest["parameters_locked"]["quantile"])
    threshold = float(history.tail(history_days)["gap_return"].abs().quantile(quantile)) if len(history) >= history_days else None
    current_phase = phase(now)
    observed_open = args.observed_open
    observed_day = pd.Timestamp(args.observed_date).date() if args.observed_date else today
    created = False
    known = {row["signal_date"] for row in records}

    if current_phase == "capture_window" and observed_day == today and today.isoformat() not in known and threshold is not None:
        if observed_open is None and not args.no_fetch:
            observed_open = fetch_observed_open(today)
        prior_rows = cash.loc[cash["date"].dt.date < today].dropna(subset=["close"])
        if observed_open is not None and not prior_rows.empty:
            prior_close = float(prior_rows.iloc[-1]["close"])
            gap_return = observed_open / prior_close - 1
            prediction = 1 if gap_return >= 0 else -1
            records.append({
                "candidate_id": manifest["candidate_id"], "created_from_locked_algorithm": True,
                "created_at_taipei": now.isoformat(), "signal_date": today.isoformat(),
                "decision_phase": current_phase, "observed_open": float(observed_open),
                "prior_close": prior_close, "gap_return": gap_return,
                "prediction": "up" if prediction > 0 else "down", "prediction_value": prediction,
                "threshold": threshold, "actionable": bool(abs(gap_return) >= threshold),
                "night_spread_baseline_value": night_baseline(today), "outcome": None,
            })
            created = True

    atomic_write_json(args.log, records)
    stats = metrics(records, manifest)
    write_status(args.status, manifest, now, current_phase, threshold, observed_open, created, stats)
    print(
        f"Open-gap Q85: phase={current_phase}; created={created}; "
        f"records={stats['records']}; actionable/resolved={stats['cases']}/{stats['resolved']}"
    )


if __name__ == "__main__":
    main()
