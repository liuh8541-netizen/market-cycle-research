import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_night_microstructure_features import session_features, within_ohlc_tolerance


def synthetic_bars(start, periods=56, contract="202601"):
    times = pd.date_range(start, periods=periods, freq="15min")
    price = 20000 + np.arange(periods, dtype=float)
    return pd.DataFrame({
        "contract_date": contract,
        "bar_start": times,
        "open": price, "high": price + 2, "low": price - 2, "close": price + 1,
        "volume": 100.0, "ticks": 10, "signed_volume": 5.0,
    })


class NightMicrostructurePipelineTest(unittest.TestCase):
    def test_monday_signal_uses_friday_night_window(self):
        friday = pd.Timestamp("2026-01-09")
        monday = pd.Timestamp("2026-01-12")
        bars = synthetic_bars(friday + pd.Timedelta(hours=15))
        result = session_features(monday, friday, "202601", bars)
        self.assertIsNotNone(result)
        self.assertEqual(result["signal_date"], "2026-01-12")
        self.assertEqual(result["bar_count"], 56)

    def test_calendar_sunday_window_is_not_required(self):
        friday = pd.Timestamp("2026-01-09")
        monday = pd.Timestamp("2026-01-12")
        bars = synthetic_bars(friday + pd.Timedelta(hours=15))
        wrong_prior = monday - pd.Timedelta(days=1)
        self.assertIsNone(session_features(monday, wrong_prior, "202601", bars))

    def test_incomplete_evening_only_session_is_rejected(self):
        prior = pd.Timestamp("2026-01-13")
        signal = pd.Timestamp("2026-01-14")
        bars = synthetic_bars(prior + pd.Timedelta(hours=15), periods=36)
        self.assertIsNone(session_features(signal, prior, "202601", bars))

    def test_incomplete_early_only_session_is_rejected(self):
        prior = pd.Timestamp("2026-01-13")
        signal = pd.Timestamp("2026-01-14")
        bars = synthetic_bars(prior + pd.Timedelta(days=1), periods=20)
        self.assertIsNone(session_features(signal, prior, "202601", bars))

    def test_wrong_contract_is_rejected(self):
        prior = pd.Timestamp("2026-01-13")
        signal = pd.Timestamp("2026-01-14")
        bars = synthetic_bars(prior + pd.Timedelta(hours=15), contract="202602")
        self.assertIsNone(session_features(signal, prior, "202601", bars))

    def test_one_tick_close_difference_is_allowed(self):
        features = {
            "_path_open": 100.0, "_path_high": 110.0,
            "_path_low": 90.0, "_path_close": 101.0,
        }
        canonical = type("Canonical", (), {
            "tx_night_open": 100.0, "tx_night_high": 110.0,
            "tx_night_low": 90.0, "tx_night_close": 100.0,
        })()
        self.assertTrue(within_ohlc_tolerance(features, canonical))

    def test_six_point_close_difference_is_rejected(self):
        features = {
            "_path_open": 100.0, "_path_high": 110.0,
            "_path_low": 90.0, "_path_close": 106.0,
        }
        canonical = type("Canonical", (), {
            "tx_night_open": 100.0, "tx_night_high": 110.0,
            "tx_night_low": 90.0, "tx_night_close": 100.0,
        })()
        self.assertFalse(within_ohlc_tolerance(features, canonical))


if __name__ == "__main__":
    unittest.main()
