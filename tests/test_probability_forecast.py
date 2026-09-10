import unittest

from market_lifecycle.probability_forecast import reconcile_one_day_forecast


class OneDayForecastReconciliationTest(unittest.TestCase):
    def base_forecast(self):
        return {
            "forecasts": [
                {
                    "horizon_days": 1,
                    "predicted_direction": "up",
                    "probability_up": 0.44,
                    "probability_down": 0.33,
                    "probability_sideways": 0.23,
                    "confidence": 0.44,
                    "match_level": "risk",
                },
                {
                    "horizon_days": 5,
                    "predicted_direction": "up",
                    "probability_up": 0.60,
                    "probability_down": 0.30,
                    "probability_sideways": 0.10,
                    "confidence": 0.60,
                    "match_level": "risk",
                },
            ]
        }

    def test_strong_psychology_score_replaces_only_one_day_distribution(self):
        result = reconcile_one_day_forecast(
            self.base_forecast(),
            {"direction": "bearish", "calibrated_direction_score": -3},
        )
        one_day, five_day = result["forecasts"]
        self.assertEqual(one_day["predicted_direction"], "down")
        self.assertAlmostEqual(one_day["probability_down"], 0.632432)
        self.assertTrue(one_day["fixed_factor_review"]["applied"])
        self.assertEqual(
            one_day["fixed_factor_review"]["historical_distribution"]["predicted_direction"],
            "up",
        )
        self.assertEqual(five_day["predicted_direction"], "up")
        self.assertNotIn("fixed_factor_review", five_day)

    def test_weak_score_keeps_distribution_but_adds_tail_watch(self):
        result = reconcile_one_day_forecast(
            self.base_forecast(),
            {
                "direction": "bearish",
                "calibrated_direction_score": -1,
                "night_diagnostics": {
                    "close_position": 0.12,
                    "recovery_from_low": 0.25,
                },
            },
        )
        one_day = result["forecasts"][0]
        self.assertEqual(one_day["predicted_direction"], "up")
        self.assertFalse(one_day["fixed_factor_review"]["applied"])
        self.assertTrue(one_day["fixed_factor_review"]["weak_bearish_tail_watch"])


if __name__ == "__main__":
    unittest.main()
