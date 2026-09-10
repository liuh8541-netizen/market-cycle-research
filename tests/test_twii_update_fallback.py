import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import predict_market


class TwiiUpdateFallbackTest(unittest.TestCase):
    def test_premarket_repairs_previous_completed_weekday(self):
        now = datetime(2026, 8, 20, 6, 30, tzinfo=ZoneInfo("Asia/Taipei"))
        self.assertEqual(
            predict_market.completed_twii_target(now).isoformat(), "2026-08-19"
        )

    def test_post_close_repairs_current_session(self):
        now = datetime(2026, 8, 20, 14, 1, tzinfo=ZoneInfo("Asia/Taipei"))
        self.assertEqual(
            predict_market.completed_twii_target(now).isoformat(), "2026-08-20"
        )

    def test_monday_premarket_repairs_friday(self):
        now = datetime(2026, 8, 24, 6, 30, tzinfo=ZoneInfo("Asia/Taipei"))
        self.assertEqual(
            predict_market.completed_twii_target(now).isoformat(), "2026-08-21"
        )


if __name__ == "__main__":
    unittest.main()
