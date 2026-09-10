import tempfile
import unittest
from pathlib import Path

from market_lifecycle.clinical_knowledge import (
    append_immutable_case,
    search_similar_cases,
    select_prescription,
    select_reference_profile,
)


class ClinicalKnowledgeTest(unittest.TestCase):
    def test_case_append_is_immutable_and_idempotent(self):
        case = {
            "case_id": "case-1",
            "scope": {"market": "TWII", "timeframe": "daily", "target": "health"},
            "symptoms": [],
            "outcome": {"status": "pending"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            self.assertTrue(append_immutable_case(path, case))
            self.assertFalse(append_immutable_case(path, {**case, "changed": True}))
            self.assertNotIn("changed", path.read_text(encoding="utf-8"))

    def test_profile_requires_exact_scope(self):
        profiles = [
            {
                "profile_id": "daily",
                "scope": {
                    "market": "TWII",
                    "timeframe": "daily",
                    "target": "lifecycle_health",
                    "lifecycle_stage": ["topping"],
                },
                "calibration_status": "validated",
            }
        ]
        query = {
            "market": "TWII",
            "timeframe": "15m",
            "target": "lifecycle_health",
            "lifecycle_stage": "topping",
        }
        result = select_reference_profile(query, profiles)
        self.assertEqual(result["status"], "unsupported_scope")
        self.assertFalse(result["usable"])

    def test_pending_profile_cannot_be_used_formally(self):
        profile = {
            "profile_id": "pending",
            "scope": {
                "market": "TWII",
                "timeframe": "daily",
                "target": "lifecycle_health",
                "lifecycle_stage": ["topping"],
            },
            "calibration_status": "pending_independent_validation",
        }
        query = {
            "market": "TWII",
            "timeframe": "daily",
            "target": "lifecycle_health",
            "lifecycle_stage": "topping",
        }
        result = select_reference_profile(query, [profile])
        self.assertEqual(result["status"], "pending_validation")
        self.assertFalse(result["usable"])

    def test_similar_case_search_never_crosses_market_or_target(self):
        query = {
            "case_id": "query",
            "scope": {
                "market": "TWII",
                "timeframe": "daily",
                "target": "lifecycle_health",
                "lifecycle_stage": "topping",
            },
            "symptoms": [{"code": "support_break"}],
        }
        cases = [
            {
                "case_id": "same",
                "scope": query["scope"],
                "symptoms": [{"code": "support_break"}],
                "outcome": {"status": "resolved"},
            },
            {
                "case_id": "wrong-market",
                "scope": {**query["scope"], "market": "SPX"},
                "symptoms": [{"code": "support_break"}],
                "outcome": {"status": "resolved"},
            },
        ]
        result = search_similar_cases(query, cases)
        self.assertEqual([item["case_id"] for item in result], ["same"])

    def test_contraindication_blocks_prescription(self):
        prescriptions = [
            {
                "prescription_id": "risk-light",
                "scope": {
                    "market": "TWII",
                    "timeframe": "daily",
                    "diagnosis": ["deterioration_watch"],
                    "confidence": ["low"],
                },
                "action": "reduce_risk",
                "intensity": "light",
                "contraindications": ["incomplete_or_stale_data"],
            }
        ]
        result = select_prescription(
            {"primary": "deterioration_watch", "confidence": "low"},
            [{"code": "incomplete_or_stale_data"}],
            prescriptions,
        )
        self.assertEqual(result["status"], "no_safe_prescription")
        self.assertEqual(result["action"], "observe_and_recheck")


if __name__ == "__main__":
    unittest.main()
