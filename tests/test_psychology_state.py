import unittest

from market_lifecycle.psychology_state import build_psychology_state


def base_payload():
    return {
        "premarket": {
            "external_score": 0,
            "night_path": {"status": "neutral", "low": 100, "close": 101, "open": 102},
        },
        "intraday_tactical_monitor": {"code": "premarket_only", "live_usable": False},
        "candlestick_pattern": {"type": "normal"},
        "washout_pattern": {"type": "none"},
        "integrated_summary": {"bias": "mixed"},
    }


class PsychologyStateTest(unittest.TestCase):
    def test_material_bearish_night_and_external_pressure_create_urgent_following(self):
        payload = base_payload()
        payload["premarket"].update(
            {
                "external_score": -4,
                "sox_return_1d": -0.0498,
                "tsm_adr_return_1d": -0.0407,
                "night_path": {
                    "status": "bearish_extension_close_near_low",
                    "low": 44424,
                    "close": 44527,
                    "open": 45137,
                },
            }
        )
        payload["candlestick_pattern"] = {"type": "long_bear"}
        payload["washout_pattern"] = {"type": "failed_washout"}
        result = build_psychology_state(payload)
        self.assertEqual(result["state"], "urgent_following")
        self.assertEqual(result["direction"], "bearish")
        self.assertGreaterEqual(result["bearish_urgency_score"], 7)
        self.assertTrue(result["exogenous_reset_watch"])
        self.assertEqual(result["next_paths"][0]["code"], "bearish_continuation")
        self.assertIsNone(result["formal_direction_signal"])
        self.assertIsNone(result["validated_probability"])

    def test_cash_reclaim_moves_path_to_exhaustion_not_formal_reversal(self):
        payload = base_payload()
        payload["premarket"]["night_path"] = {
            "status": "bearish_extension_close_near_low", "low": 100, "close": 101, "open": 103
        }
        payload["intraday_tactical_monitor"] = {
            "code": "night_down_cash_reclaim_candidate", "live_usable": True
        }
        result = build_psychology_state(payload)
        self.assertEqual(result["state"], "exhaustion")
        self.assertFalse(result["position_confirmation"])
        self.assertIsNone(result["formal_direction_signal"])

    def test_neutral_inputs_remain_doubt(self):
        result = build_psychology_state(base_payload())
        self.assertEqual(result["state"], "doubt")
        self.assertEqual(result["direction"], "mixed")
        self.assertFalse(result["exogenous_reset_watch"])

    def test_resolved_cases_reorder_paths_without_changing_conditions(self):
        payload = base_payload()
        payload["premarket"]["night_path"] = {
            "status": "bearish_extension_close_near_low", "low": 95, "close": 96, "open": 100
        }
        stats = [{
            "psychology_state": "confirmation", "direction": "bearish", "cases": 100,
            "path_distribution": [
                {"path": "short_covering_rebound", "rate": 0.55},
                {"path": "bearish_continuation", "rate": 0.30},
                {"path": "path_reset_reversal", "rate": 0.15},
            ],
        }]

        result = build_psychology_state(payload, stats)

        self.assertEqual(result["next_paths"][0]["code"], "short_covering_rebound")
        self.assertEqual(result["next_paths"][0]["historical_rate"], 0.55)
        self.assertEqual(result["next_paths"][0]["ranking_source"], "resolved_clinical_cases")

    def test_two_fresh_position_signals_confirm_crowding(self):
        payload = base_payload()
        payload["premarket"]["night_path"] = {
            "status": "bearish_extension_close_near_low", "low": 95, "close": 96, "open": 100
        }
        payload["psychology_position_context"] = {
            "breadth_age_days": 20,
            "foreign_futures_net_oi": -50000, "institutional_position_age_days": 1,
            "tx_open_interest_change_1d": 1000, "open_interest_age_days": 1,
        }

        result = build_psychology_state(payload)

        self.assertTrue(result["position_confirmation"])
        self.assertEqual(result["crowding_status"], "confirmed_by_positioning")


if __name__ == "__main__":
    unittest.main()
