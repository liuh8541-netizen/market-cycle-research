import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import fetch_finmind_futures_tick_bars as tick_cache


class FuturesTickCacheTest(unittest.TestCase):
    def test_refresh_replaces_calendar_day_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bars.csv"
            old = pd.DataFrame([
                {"calendar_date": "2026-08-18", "contract_date": "202608", "bar_start": "2026-08-18 00:00:00", "close": 1},
                {"calendar_date": "2026-08-19", "contract_date": "202608", "bar_start": "2026-08-19 00:00:00", "close": 2},
            ])
            old.to_csv(path, index=False)
            fresh = pd.DataFrame([
                {"calendar_date": "2026-08-18", "contract_date": "202608", "bar_start": "2026-08-18 00:00:00", "close": 9},
                {"calendar_date": "2026-08-18", "contract_date": "202608", "bar_start": "2026-08-18 00:15:00", "close": 10},
            ])
            tick_cache.replace_calendar_day(fresh, path, date(2026, 8, 18))
            result = pd.read_csv(path)
            target = result[result["calendar_date"].eq("2026-08-18")]
            self.assertEqual(len(target), 2)
            self.assertEqual(target["close"].tolist(), [9, 10])
            self.assertEqual(len(result[result["calendar_date"].eq("2026-08-19")]), 1)

    def test_manifest_upsert_promotes_partial_to_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            with patch.object(tick_cache, "MANIFEST", manifest):
                tick_cache.upsert_manifest(date(2026, 8, 19), 10, 2, "partial")
                tick_cache.upsert_manifest(date(2026, 8, 19), 100, 20, "complete")
            result = pd.read_csv(manifest)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["status"], "complete")
            self.assertEqual(int(result.iloc[0]["bar_rows"]), 20)


if __name__ == "__main__":
    unittest.main()
