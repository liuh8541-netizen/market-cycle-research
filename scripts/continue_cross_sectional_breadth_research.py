"""Wait for breadth backfill, audit immutable inputs, then run formal test once."""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data/processed/factors/.cross_sectional_breadth_fetch.lock"
MANIFEST = ROOT / "data/processed/factors/cross_sectional_breadth_fetched_dates.csv"
FEATURES = ROOT / "data/processed/factors/cross_sectional_breadth.csv"
CONFIG = ROOT / "config/locked_cross_sectional_breadth_cycle_v1.json"
INTEGRITY = ROOT / "reports/cross_sectional_breadth_integrity.json"
RESULT = ROOT / "reports/cross_sectional_breadth_cycle.json"
FORMAL_LOG = ROOT / "reports/cross_sectional_breadth_formal_run.log"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def audit():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = pd.read_csv(MANIFEST)
    features = pd.read_csv(FEATURES)
    twii = pd.read_csv(ROOT / "data/processed/twii_daily.csv", usecols=["date"])
    expected = pd.to_datetime(twii["date"], errors="coerce").dropna()
    expected = expected[
        expected.between(pd.Timestamp("2005-01-01"), pd.Timestamp("2026-07-23"))
    ].dt.strftime("%Y-%m-%d")
    expected_set = set(expected)
    manifest_dates = set(manifest["date"].astype(str))
    feature_dates = set(features["date"].astype(str))
    complete_dates = set(
        manifest.loc[manifest["status"].eq("complete"), "date"].astype(str)
    )
    terminal_statuses = {
        "complete",
        "excluded_low_institutional_coverage",
        "excluded_no_market_source",
    }
    numeric = features.select_dtypes(include=[np.number])
    registered = config["implementation"]
    hashes = {
        relative: sha256(ROOT / relative) == expected_hash
        for relative, expected_hash in registered.items()
    }
    checks = {
        "expected_trading_dates": len(expected_set),
        "manifest_rows": len(manifest),
        "manifest_unique_dates": manifest["date"].nunique(),
        "missing_manifest_dates": sorted(expected_set - manifest_dates),
        "extra_manifest_dates": sorted(manifest_dates - expected_set),
        "all_status_terminal": bool(manifest["status"].isin(terminal_statuses).all()),
        "excluded_low_institutional_dates": sorted(
            manifest.loc[
                manifest["status"].eq("excluded_low_institutional_coverage"),
                "date",
            ].astype(str)
        ),
        "excluded_no_market_source_dates": sorted(
            manifest.loc[
                manifest["status"].eq("excluded_no_market_source"),
                "date",
            ].astype(str)
        ),
        "feature_rows": len(features),
        "feature_unique_dates": features["date"].nunique(),
        "duplicate_feature_dates": int(features["date"].duplicated().sum()),
        "manifest_feature_mismatch": sorted(complete_dates ^ feature_dates),
        "feature_date_coverage": len(feature_dates) / len(expected_set),
        "all_numeric_finite": bool(np.isfinite(numeric.to_numpy()).all()),
        "minimum_price_stock_count": int(features["price_stock_count"].min()),
        "minimum_institutional_stock_count": int(features["inst_stock_count"].min()),
        "all_minimum_stock_counts_met": bool(
            features["price_stock_count"].ge(100).all()
            and features["inst_stock_count"].ge(50).all()
        ),
        "hashes_match": hashes,
    }
    passed = (
        checks["manifest_rows"] == len(expected_set)
        and checks["manifest_unique_dates"] == len(expected_set)
        and not checks["missing_manifest_dates"]
        and not checks["extra_manifest_dates"]
        and checks["all_status_terminal"]
        and checks["feature_rows"] == len(complete_dates)
        and checks["feature_unique_dates"] == len(complete_dates)
        and checks["duplicate_feature_dates"] == 0
        and not checks["manifest_feature_mismatch"]
        and checks["feature_date_coverage"] >= 0.90
        and checks["all_numeric_finite"]
        and checks["all_minimum_stock_counts_met"]
        and all(hashes.values())
    )
    payload = {"passed": passed, "checks": checks}
    INTEGRITY.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return passed


def main():
    # The watcher can start just before the fetch process creates its lock.
    time.sleep(5)
    while LOCK.exists():
        time.sleep(30)
    if RESULT.exists():
        raise SystemExit("Formal result already exists; refusing a second run.")
    if not audit():
        raise SystemExit("Cross-sectional breadth integrity audit failed.")
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts/research_cross_sectional_breadth_cycle.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    FORMAL_LOG.write_text(
        process.stdout + ("\nSTDERR\n" + process.stderr if process.stderr else ""),
        encoding="utf-8",
    )
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
