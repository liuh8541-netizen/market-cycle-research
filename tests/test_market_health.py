import unittest

from market_lifecycle.market_health import build_market_health_assessment


def payload(**overrides):
    value = {
        "integrated_summary": {
            "bias": "slightly_bearish",
            "confidence": "低(low，多空接近或資料不足)",
            "bearish_evidence": ["價格跌破防守區。"],
            "bullish_evidence": [],
            "neutral_evidence": [],
            "confirmation": ["站回確認價。"],
            "invalidation": ["續破今日低點。"],
        },
        "forecast": {"lifecycle_stage": "topping", "risk_regime": "neutral"},
        "logic_consistency": {"contradiction_count": 0},
        "data_freshness": {"overall_status": "current"},
        "premarket": {"is_premarket": False},
        "candlestick_pattern": {"type": "normal"},
        "washout_pattern": {"type": "no_washout"},
        "crash_monitor": {
            "risk_value": 50,
            "health_score": 50,
            "alert_code": "observe",
        },
        "intraday_tactical_monitor": {"code": "no_live_data"},
        "technical_phase": {"alignment": {"status": "needs_confirmation"}},
        "bagua_lifecycle": {"roles": {"state_gua": {"code": "KUN"}}},
        "production_policy": {
            "main_multi_day_direction": {"enabled": False, "formal_signal": None}
        },
    }
    value.update(overrides)
    return value


class MarketHealthAssessmentTest(unittest.TestCase):
    def test_unvalidated_direction_never_becomes_formal_prescription(self):
        result = build_market_health_assessment(payload())
        self.assertIsNone(result["prescription"]["formal_direction_signal"])
        self.assertTrue(result["safety_checks"]["unvalidated_direction_is_null"])

    def test_low_confidence_caps_prescription_at_light(self):
        result = build_market_health_assessment(payload())
        self.assertEqual(result["prescription"]["action"], "reduce_risk")
        self.assertEqual(result["prescription"]["intensity"], "light")

    def test_incomplete_data_blocks_treatment(self):
        data = payload(
            data_freshness={"overall_status": "stale"},
            premarket={"is_premarket": True, "night_session_complete": False},
        )
        result = build_market_health_assessment(data)
        self.assertEqual(result["diagnosis"]["status"], "insufficient_data")
        self.assertEqual(result["prescription"]["action"], "observe_and_recheck")
        self.assertEqual(result["prescription"]["intensity"], "none")

    def test_conflicting_evidence_requires_differential_diagnosis(self):
        data = payload(logic_consistency={"contradiction_count": 1})
        result = build_market_health_assessment(data)
        self.assertEqual(result["diagnosis"]["status"], "differential_required")
        self.assertEqual(result["prescription"]["action"], "observe_and_recheck")

    def test_bullish_research_state_cannot_authorize_reentry(self):
        summary = payload()["integrated_summary"]
        summary.update({"bias": "bullish", "confidence": "高(high，多數訊號同向)"})
        result = build_market_health_assessment(payload(integrated_summary=summary))
        self.assertEqual(
            result["prescription"]["action"],
            "observe_repair_without_directional_authority",
        )
        self.assertEqual(result["prescription"]["intensity"], "none")

    def test_prescription_never_allows_leverage_or_averaging_down(self):
        result = build_market_health_assessment(payload())
        self.assertFalse(result["prescription"]["may_increase_leverage"])
        self.assertFalse(result["prescription"]["may_average_down"])

    def test_health_value_marks_controlled_healthy_pulse(self):
        summary = payload()["integrated_summary"]
        summary.update({"bias": "slightly_bullish", "bearish_evidence": []})
        result = build_market_health_assessment(
            payload(
                integrated_summary=summary,
                crash_monitor={
                    "risk_value": 25,
                    "health_score": 75,
                    "alert_code": "observe",
                },
            )
        )
        self.assertEqual(result["health_value"]["code"], "healthy_pulse")
        self.assertTrue(result["health_value"]["controllable_risk"])

    def test_health_value_marks_structural_damage_when_risk_exceeds_limit(self):
        result = build_market_health_assessment(
            payload(
                crash_monitor={
                    "risk_value": 80,
                    "health_score": 20,
                    "alert_code": "crash_warning",
                }
            )
        )
        self.assertEqual(result["health_value"]["code"], "structural_damage")
        self.assertFalse(result["health_value"]["controllable_risk"])

    def test_partial_non_core_data_does_not_blank_health_value(self):
        result = build_market_health_assessment(
            payload(
                data_freshness={"overall_status": "partial"},
                crash_monitor={
                    "risk_value": 25,
                    "health_score": 75,
                    "alert_code": "observe",
                },
            )
        )
        self.assertEqual(result["health_value"]["code"], "controlled_volatility")
        self.assertTrue(result["health_value"]["controllable_risk"])


if __name__ == "__main__":
    unittest.main()
