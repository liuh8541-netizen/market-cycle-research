import unittest

import pandas as pd

from backtest_psychology_state import (
    classify_realized_path,
    exact_two_sided_binomial_p,
    replay_psychology_history,
)


class PsychologyBacktestTest(unittest.TestCase):
    def test_replay_uses_strictly_prior_external_row(self):
        cash = pd.DataFrame(
            [
                {"date": "2024-01-02", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 1},
                {"date": "2024-01-03", "open": 102, "high": 104, "low": 101, "close": 103, "volume": 1},
            ]
        )
        night = pd.DataFrame(
            [{
                "night_date": "2024-01-03", "signal_date": "2024-01-03", "contract_date": 202401,
                "tx_night_open": 101, "tx_night_high": 103, "tx_night_low": 100,
                "tx_night_close": 102, "tx_night_volume": 100, "tx_night_return": 0.009,
                "tx_night_range": 0.03, "tx_night_spread_per": 0.01,
            }]
        )
        external = pd.DataFrame(
            [
                {"date": "2024-01-02", "nasdaq_return_1d": 0, "sox_return_1d": 0,
                 "sp500_return_1d": 0, "tsm_adr_return_1d": 0, "vix_return_1d": 0},
                {"date": "2024-01-03", "nasdaq_return_1d": -0.10, "sox_return_1d": -0.10,
                 "sp500_return_1d": -0.10, "tsm_adr_return_1d": -0.10, "vix_return_1d": 0.30},
            ]
        )

        replay = replay_psychology_history(cash, night, external)

        self.assertEqual(len(replay), 1)
        self.assertEqual(replay.iloc[0]["external_date"], "2024-01-02")
        self.assertEqual(replay.iloc[0]["external_score"], 0)

    def test_bearish_top_path_requires_close_below_night_low(self):
        recovered = classify_realized_path("bearish", 100, 98, 99, 94, 97, 100, 101, 95, 98)
        extended = classify_realized_path("bearish", 100, 98, 99, 94, 94.5, 100, 101, 95, 98)

        self.assertEqual(recovered, "unresolved")
        self.assertEqual(extended, "bearish_continuation")

    def test_exact_mcnemar_p_is_one_when_no_disagreement(self):
        self.assertEqual(exact_two_sided_binomial_p(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
