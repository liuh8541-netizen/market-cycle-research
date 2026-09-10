import unittest

from market_lifecycle.self_repair import (
    build_self_repair_assessment,
    validate_repair_proposal,
)


def payload():
    return {
        "input": {"date": "2026-07-25"},
        "index_check": {"signal_date": "2026-07-24"},
        "data_freshness": {"overall_status": "current"},
        "logic_consistency": {"contradiction_count": 0},
        "market_health": {
            "diagnosis": {"primary": "deterioration_watch"},
            "prescription": {"action": "reduce_risk", "intensity": "light"},
        },
        "self_review": {"matured_checks": {"rows": []}},
        "production_policy": {
            "main_multi_day_direction": {"enabled": False, "formal_signal": None}
        },
        "forecast": {"formal_direction_signal": None},
    }


class SelfRepairTest(unittest.TestCase):
    def test_clean_state_only_continues_monitoring(self):
        result = build_self_repair_assessment(payload())
        self.assertEqual(result["status"], "monitoring")
        self.assertEqual(result["automatic_actions"][0]["action"], "continue_monitoring")

    def test_missed_case_is_preserved_and_never_rewritten(self):
        data = payload()
        data["self_review"]["matured_checks"]["rows"] = [
            {
                "signal_date": "2026-07-23",
                "target_date": "2026-07-24",
                "horizon_days": 1,
                "is_hit": False,
            }
        ]
        result = build_self_repair_assessment(data)
        incident = result["incidents"][0]
        self.assertEqual(incident["code"], "outcome_mismatch")
        self.assertFalse(incident["historical_record_mutation_allowed"])
        self.assertEqual(result["automatic_actions"][0]["action"], "append_case_review")

    def test_stale_data_forces_recheck(self):
        data = payload()
        data["data_freshness"]["overall_status"] = "stale"
        result = build_self_repair_assessment(data)
        self.assertEqual(result["incidents"][0]["code"], "data_quality_error")
        self.assertEqual(result["automatic_actions"][0]["action"], "force_observe_and_recheck")

    def test_scope_violation_triggers_protective_block(self):
        data = payload()
        data["forecast"]["formal_direction_signal"] = [{"direction": "up"}]
        result = build_self_repair_assessment(data)
        self.assertEqual(result["status"], "protective_block")
        self.assertEqual(result["automatic_actions"][0]["action"], "block_formal_direction_output")

    def test_protected_changes_cannot_be_automatically_promoted(self):
        for change_type in [
            "model_parameter_change",
            "threshold_change",
            "scope_change",
            "historical_record_change",
            "locked_experiment_rerun",
        ]:
            result = validate_repair_proposal({"change_type": change_type})
            self.assertFalse(result["allowed_automatically"])
            self.assertTrue(result["requires_independent_validation"])

    def test_evidence_fingerprint_is_stable(self):
        first = build_self_repair_assessment(payload())["evidence_sha256"]
        second = build_self_repair_assessment(payload())["evidence_sha256"]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
