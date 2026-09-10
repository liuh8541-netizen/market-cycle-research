"""Wait for sector backfill, audit it, then run the locked experiment once."""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data/processed/factors/.sector_early_pulse_fetch.lock"
MANIFEST = ROOT / "data/processed/factors/sector_early_pulse_fetched_dates.csv"
FEATURES = ROOT / "data/processed/factors/sector_early_pulse.csv"
CONFIG = ROOT / "config/locked_sector_early_pulse_close_v1.json"
AUDIT = ROOT / "reports/sector_early_pulse_integrity.json"
RESULT = ROOT / "reports/sector_early_pulse_close.json"
RUN_LOG = ROOT / "reports/sector_early_pulse_formal_run.log"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def audit():
    manifest = pd.read_csv(MANIFEST)
    features = pd.read_csv(FEATURES)
    expected = set(pd.date_range("2020-01-01", "2026-07-23").strftime("%Y-%m-%d"))
    actual = set(manifest.calendar_date.astype(str))
    feature_dates = set(features.date.astype(str))
    claimed = set(
        manifest.loc[
            pd.to_numeric(manifest.feature_rows, errors="coerce") == 1,
            "calendar_date",
        ].astype(str)
    )
    excluded = {"date", "first_timestamp", "cutoff_timestamp"}
    numeric = features[[column for column in features if column not in excluded]].apply(
        pd.to_numeric, errors="coerce"
    )
    observation_columns = [
        column for column in numeric if column.endswith("_observations")
    ]
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    hashes = {
        name: digest(ROOT / name) == expected_hash.upper()
        for name, expected_hash in config["implementation"].items()
    }
    checks = {
        "manifest_rows": len(manifest),
        "manifest_unique_dates": int(manifest.calendar_date.nunique()),
        "missing_calendar_dates": sorted(expected - actual),
        "extra_calendar_dates": sorted(actual - expected),
        "all_status_complete": bool((manifest.status == "complete").all()),
        "feature_rows": len(features),
        "feature_unique_dates": int(features.date.nunique()),
        "duplicate_feature_dates": int(features.date.duplicated().sum()),
        "manifest_feature_mismatch": sorted(feature_dates ^ claimed),
        "all_numeric_finite": bool(np.isfinite(numeric.to_numpy(float)).all()),
        "observation_columns": observation_columns,
        "all_observations_181": bool((numeric[observation_columns] == 181).all().all()),
        "all_start_0900": bool(features.first_timestamp.str.endswith("09:00:00").all()),
        "all_cutoff_0915": bool(features.cutoff_timestamp.str.endswith("09:15:00").all()),
        "hashes_match": hashes,
    }
    passed = (
        checks["manifest_rows"] == 2396
        and checks["manifest_unique_dates"] == 2396
        and not checks["missing_calendar_dates"]
        and not checks["extra_calendar_dates"]
        and checks["all_status_complete"]
        and checks["feature_rows"] == checks["feature_unique_dates"]
        and checks["duplicate_feature_dates"] == 0
        and not checks["manifest_feature_mismatch"]
        and checks["all_numeric_finite"]
        and len(observation_columns) == 4
        and checks["all_observations_181"]
        and checks["all_start_0900"]
        and checks["all_cutoff_0915"]
        and all(hashes.values())
    )
    AUDIT.write_text(
        json.dumps({"passed": passed, "checks": checks}, indent=2), encoding="utf-8"
    )
    return passed


def main():
    while LOCK.exists():
        time.sleep(15)
    if not MANIFEST.exists() or not FEATURES.exists():
        raise RuntimeError("Backfill ended without manifest or feature data")
    if not audit():
        raise RuntimeError(f"Sector integrity audit failed; see {AUDIT}")
    if RESULT.exists():
        raise RuntimeError("Formal result already exists; refusing to execute twice")
    process = subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts/research_sector_early_pulse_close.py")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    RUN_LOG.write_text(
        process.stdout + ("\nSTDERR\n" + process.stderr if process.stderr else ""),
        encoding="utf-8",
    )
    if process.returncode:
        raise RuntimeError(f"Formal experiment failed with exit code {process.returncode}")


if __name__ == "__main__":
    main()
