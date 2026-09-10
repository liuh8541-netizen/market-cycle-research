import unittest

import pandas as pd

from market_lifecycle.path_risk import (
    classify_direction_strength,
    classify_session_path,
    build_path_risk_assessment,
    historical_close_range,
    historical_path_distribution,
)


class PathRiskTest(unittest.TestCase):
    def test_borderline_return_is_preserved_without_rewriting_formal_threshold(self):
        result = classify_direction_strength(0.004794048, "bullish")

        self.assertEqual(result["label"], "borderline_bullish")
        self.assertTrue(result["borderline_confirmation"])
        self.assertFalse(result["formal_material"])
        self.assertEqual(result["formal_threshold"], 0.005)

    def test_august_20_course_is_deep_selloff_recovered(self):
        result = classify_session_path(
            44719.3515625, 44942.01171875, 45160.05078125,
            44446.359375, 44933.73828125,
        )

        self.assertEqual(result["label"], "deep_selloff_recovered")
        self.assertLess(result["adverse_from_open"], -0.01)
        self.assertGreater(result["recovery_ratio"], 0.68)

    def test_close_range_uses_return_quantiles(self):
        result = historical_close_range(pd.Series([-0.02, -0.01, 0, 0.01, 0.02]), 100)

        self.assertEqual(result["sample_count"], 5)
        self.assertLess(result["central_50_range"][0], 100)
        self.assertGreater(result["central_50_range"][1], 100)

    def test_path_distribution_records_reversal_tail(self):
        pool = pd.DataFrame([
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 101, "high": 102, "low": 99, "close": 101.5},
            {"open": 102, "high": 103, "low": 101, "close": 102.5},
        ])

        result = historical_path_distribution(pool)

        self.assertEqual(result["sample_count"], 2)
        self.assertIn("rates", result)

    def test_assessment_marks_settlement_offset_and_missing_structure(self):
        cash = pd.DataFrame([
            {"date": "2024-01-16", "open": 99, "high": 101, "low": 98, "close": 100},
            {"date": "2024-01-17", "open": 100, "high": 102, "low": 99, "close": 101},
        ])

        result = build_path_risk_assessment(cash, "2024-01-18")

        self.assertEqual(result["settlement_context"]["trading_session_offset"], 1)
        self.assertEqual(result["market_structure"]["breadth"]["status"], "missing")
        self.assertEqual(result["direction_confidence_authority"], "reduced")


if __name__ == "__main__":
    unittest.main()
