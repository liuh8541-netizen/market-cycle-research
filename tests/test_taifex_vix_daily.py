import unittest
from datetime import date

import fetch_taifex_vix_daily as taifex_vix


class TaifexVixDailyTest(unittest.TestCase):
    def test_today_is_clamped_to_yesterday(self):
        result = taifex_vix.resolve_effective_end_date(
            date(2026, 8, 20), today=date(2026, 8, 20)
        )
        self.assertEqual(result, date(2026, 8, 19))

    def test_historical_end_date_is_preserved(self):
        result = taifex_vix.resolve_effective_end_date(
            date(2026, 8, 18), today=date(2026, 8, 20)
        )
        self.assertEqual(result, date(2026, 8, 18))

    def test_taifex_plain_text_error_is_preserved(self):
        with self.assertRaisesRegex(RuntimeError, "結束日期不能大於或等於今天"):
            taifex_vix.decode_response({"d": "結束日期不能大於或等於今天！"})

    def test_valid_nested_payload_is_decoded(self):
        result = taifex_vix.decode_response({"d": '{"TrendData": [{"Time": "20260819"}]}'})
        self.assertEqual(result["TrendData"][0]["Time"], "20260819")


if __name__ == "__main__":
    unittest.main()
