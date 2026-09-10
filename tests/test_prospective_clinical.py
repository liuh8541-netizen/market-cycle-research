import tempfile
import unittest
from pathlib import Path

from market_lifecycle.prospective_clinical import load_jsonl, update_prospective_registry


def case(date, status):
    return {
        "case_id": f"case-{date}-{status}", "evidence_sha256": "ABC",
        "identity": {"decision_date": date},
        "diagnosis": {"psychology_state": "urgent_following", "direction": "bearish"},
        "differential_diagnosis": [], "prognosis_from_prior_cases": {}, "prescription": {},
        "outcome": {"status": status, "direction_hit": True, "top_path_hit": False,
                    "material_direction_confirmation": True},
        "follow_up": {},
    }


class ProspectiveClinicalTest(unittest.TestCase):
    def test_prediction_is_idempotent_and_outcome_is_appended_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.jsonl"
            outcomes = Path(directory) / "outcomes.jsonl"
            first = update_prospective_registry([case("2024-01-03", "pending")], predictions, outcomes)
            second = update_prospective_registry([case("2024-01-03", "pending")], predictions, outcomes)
            settled = update_prospective_registry([case("2024-01-03", "resolved")], predictions, outcomes)

            self.assertTrue(first["prediction_appended"])
            self.assertFalse(second["prediction_appended"])
            self.assertEqual(settled["settled_count"], 1)
            self.assertEqual(len(load_jsonl(predictions)), 1)
            self.assertEqual(len(load_jsonl(outcomes)), 1)
            self.assertTrue(settled["prediction_chain_valid"])
            self.assertTrue(settled["outcome_chain_valid"])


if __name__ == "__main__":
    unittest.main()
