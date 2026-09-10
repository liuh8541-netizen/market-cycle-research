import unittest

import pandas as pd

from market_lifecycle.clinical_enrichment import (
    enrich_auxiliary_vitals,
    enrich_forward_outcomes,
)


class ClinicalEnrichmentTest(unittest.TestCase):
    def test_breadth_merge_is_strictly_prior(self):
        replay = pd.DataFrame([{"signal_date": "2024-01-03"}])
        breadth = pd.DataFrame([
            {"date": "2024-01-02", "price_advance_decline_breadth": -0.2},
            {"date": "2024-01-03", "price_advance_decline_breadth": 0.9},
        ])

        enriched = enrich_auxiliary_vitals(replay, breadth=breadth)

        self.assertEqual(str(enriched.iloc[0]["breadth_date"].date()), "2024-01-02")
        self.assertEqual(enriched.iloc[0]["price_advance_decline_breadth"], -0.2)

    def test_forward_outcomes_are_kept_outside_diagnosis_window(self):
        replay = pd.DataFrame([{"signal_date": "2024-01-02"}])
        cash = pd.DataFrame([
            {"date": "2024-01-02", "open": 100, "high": 101, "low": 99, "close": 100},
            {"date": "2024-01-03", "open": 100, "high": 103, "low": 98, "close": 102},
            {"date": "2024-01-04", "open": 102, "high": 105, "low": 101, "close": 104},
            {"date": "2024-01-05", "open": 104, "high": 106, "low": 103, "close": 105},
        ])

        enriched = enrich_forward_outcomes(replay, cash)

        self.assertAlmostEqual(enriched.iloc[0]["forward_1d_return"], 0.02)
        self.assertAlmostEqual(enriched.iloc[0]["forward_3d_return"], 0.05)
        self.assertAlmostEqual(enriched.iloc[0]["forward_3d_max_drawdown"], -0.02)


if __name__ == "__main__":
    unittest.main()
