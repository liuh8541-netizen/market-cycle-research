import unittest

import numpy as np
import pandas as pd

from predict_market import (
    classify_market_state_gua,
    classify_price_location_gua,
)


def frame_with_last_position(position: float, length: int = 120) -> pd.DataFrame:
    close = np.linspace(20.0, 80.0, length)
    close[0] = 0.0
    close[1] = 100.0
    close[-1] = position * 100.0
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=length, freq="B"),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
        }
    )


class BaguaRoleClassificationTest(unittest.TestCase):
    def test_price_gua_uses_fixed_eight_equal_bands(self):
        self.assertEqual(classify_price_location_gua(frame_with_last_position(0.81), 120)["code"], "DUI")
        self.assertEqual(classify_price_location_gua(frame_with_last_position(0.60), 120)["code"], "LI")

    def test_price_gua_does_not_claim_market_state(self):
        result = classify_price_location_gua(frame_with_last_position(0.81), 120)
        self.assertNotIn("recovery", result)
        self.assertNotIn("formal_direction_signal", result)

    def test_crash_then_material_recovery_is_zhen_activation(self):
        frame = frame_with_last_position(0.90)
        frame.loc[:, "close"] = 95.0
        frame.loc[:, "open"] = 95.0
        frame.loc[:, "high"] = 95.0
        frame.loc[:, "low"] = 95.0
        frame.loc[105, ["open", "high", "low", "close"]] = 100.0
        frame.loc[110, ["open", "high", "low", "close"]] = 80.0
        for index, value in enumerate([82.0, 84.0, 86.0, 88.0, 90.0, 91.0, 92.0, 93.0, 94.0], start=111):
            frame.loc[index, ["open", "high", "low", "close"]] = value
        result = classify_market_state_gua(frame)
        self.assertEqual(result["code"], "ZHEN")
        self.assertEqual(result["status"], "repair_activation")
        self.assertIsNone(result["formal_direction_signal"])

    def test_crash_without_leaving_bottom_remains_gen_watch(self):
        frame = frame_with_last_position(0.82)
        frame.loc[:, ["open", "high", "low", "close"]] = 100.0
        frame.loc[115, ["open", "high", "low", "close"]] = 80.0
        frame.loc[116:, ["open", "high", "low", "close"]] = [80.2, 80.5, 80.8, 81.0]
        result = classify_market_state_gua(frame)
        self.assertEqual(result["code"], "GEN")
        self.assertEqual(result["status"], "bottoming_watch")


if __name__ == "__main__":
    unittest.main()
