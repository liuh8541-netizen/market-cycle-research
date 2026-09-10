import unittest

from research_preopen_futures_cash_open import classify_path, wilson_interval


class PreopenFuturesCashOpenTest(unittest.TestCase):
    def test_classifies_confirmation_and_divergence_paths(self):
        self.assertEqual(classify_path(0.004, 0.002), "gap_up_momentum_up")
        self.assertEqual(classify_path(0.004, -0.002), "gap_up_fading")
        self.assertEqual(classify_path(-0.004, -0.002), "gap_down_momentum_down")
        self.assertEqual(classify_path(-0.004, 0.002), "gap_down_reclaim")

    def test_neutral_path_waits_when_both_inputs_are_small(self):
        self.assertEqual(classify_path(0.001, 0.001), "neutral_wait")

    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(60, 100)
        self.assertLess(low, 0.6)
        self.assertGreater(high, 0.6)


if __name__ == "__main__":
    unittest.main()
