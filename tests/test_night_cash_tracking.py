import unittest

import pandas as pd

from market_lifecycle.night_cash_tracking import (
    build_night_cash_impact,
    summarize_night_cash_impact,
)


class NightCashTrackingTest(unittest.TestCase):
    def setUp(self):
        self.cash = pd.DataFrame(
            [
                {"date": "2026-08-05", "open": 99, "high": 101, "low": 98, "close": 100, "volume": 10},
                {"date": "2026-08-06", "open": 99, "high": 100, "low": 97, "close": 98, "volume": 0},
                {"date": "2026-08-07", "open": 99, "high": 103, "low": 99, "close": 102, "volume": 12},
                {"date": "2026-08-10", "open": 103, "high": 104, "low": 98, "close": 99, "volume": 15},
            ]
        )
        self.night = pd.DataFrame(
            [
                self.night_row("2026-08-06", -0.01),
                self.night_row("2026-08-07", -0.005),
                self.night_row("2026-08-10", 0.006),
            ]
        )

    @staticmethod
    def night_row(date, ret):
        return {
            "night_date": date, "signal_date": date, "contract_date": "202608",
            "tx_night_open": 100, "tx_night_high": 101, "tx_night_low": 98,
            "tx_night_close": 99, "tx_night_volume": 1000, "tx_night_return": ret,
            "tx_night_range": 0.03, "tx_night_spread_per": ret,
        }

    def test_tracks_night_to_gap_and_close_separately(self):
        result = build_night_cash_impact(self.night, self.cash)
        first = result.iloc[0]
        second = result.iloc[1]
        self.assertTrue(first["night_gap_aligned"])
        self.assertTrue(first["night_close_aligned"])
        self.assertEqual(first["sync_status"], "full_sync")
        self.assertEqual(first["short_squeeze_status"], "low_reclaim_candidate")
        self.assertFalse(first["cash_volume_available"])
        self.assertFalse(second["night_gap_aligned"])
        self.assertFalse(second["night_close_aligned"])
        self.assertFalse(second["gap_reversed_by_close"])
        self.assertEqual(second["sync_status"], "full_divergence")
        third = result.iloc[2]
        self.assertTrue(third["night_gap_aligned"])
        self.assertFalse(third["night_close_aligned"])
        self.assertTrue(third["gap_reversed_by_close"])
        self.assertEqual(third["sync_status"], "open_only")
        self.assertEqual(third["reason_code"], "cash_session_reversal")

    def test_summary_uses_latest_window_only(self):
        result = build_night_cash_impact(self.night, self.cash)
        summary = summarize_night_cash_impact(result, windows=(1, 2))
        self.assertEqual(summary["latest_date"], "2026-08-10")
        self.assertEqual(summary["rolling"]["1"]["night_to_gap_alignment"], 1.0)
        self.assertEqual(summary["rolling"]["2"]["night_to_gap_alignment"], 0.5)


if __name__ == "__main__":
    unittest.main()
