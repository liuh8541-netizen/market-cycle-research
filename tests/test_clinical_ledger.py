import unittest
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_lifecycle.clinical_ledger import (
    build_complete_ledger, flatten_records, positioning_diagnosis_from_row, summarize_ledger,
)


def replay_row(date, prior_date, direction_hit, return_value):
    return {
        "signal_date": date, "prior_cash_date": prior_date, "external_date": prior_date,
        "state": "urgent_following", "state_label": "急迫追隨", "direction": "bearish",
        "bearish_urgency_score": 6, "bullish_urgency_score": 0, "net_urgency_score": 6,
        "night_status": "bearish_extension_close_near_low", "night_open": 100,
        "night_high": 101, "night_low": 95, "night_close": 96, "night_return": -0.04,
        "external_score": -3, "prior_cash_close": 100, "prior_candle_type": "long_bear",
        "prior_washout_type": "failed_washout", "crowding_status": "unconfirmed",
        "position_confirmation": False, "exogenous_reset_watch": False,
        "psychology_evidence_json": "[]", "expected_paths_json": "[]",
        "actual_direction": "bearish" if direction_hit else "bullish",
        "realized_path": "bearish_continuation" if direction_hit else "path_reset_reversal",
        "path_resolved": True, "direction_hit": direction_hit, "gap_direction_hit": direction_hit,
        "top_path_hit": direction_hit, "night_baseline_hit": direction_hit,
        "gap_return": -0.01, "cash_close_return": return_value, "intraday_return": 0.002,
        "cash_open": 98, "cash_high": 99, "cash_low": 94, "cash_close": 95,
    }


class ClinicalLedgerTest(unittest.TestCase):
    def test_prognosis_uses_only_strictly_prior_cases(self):
        replay = pd.DataFrame([
            replay_row("2024-01-03", "2024-01-02", True, -0.02),
            replay_row("2024-01-04", "2024-01-03", False, 0.01),
        ])

        records = build_complete_ledger(replay)

        self.assertEqual(records[0]["prognosis_from_prior_cases"]["prior_cases"], 0)
        self.assertEqual(records[1]["prognosis_from_prior_cases"]["prior_cases"], 1)
        self.assertEqual(records[1]["prognosis_from_prior_cases"]["direction_hit_rate"], 1.0)

    def test_record_has_full_clinical_chain_and_flat_index(self):
        records = build_complete_ledger(pd.DataFrame([
            replay_row("2024-01-03", "2024-01-02", True, -0.02),
        ]))
        summary = summarize_ledger(records)
        flat = flatten_records(records)

        self.assertTrue(summary["leakage_audit"]["passed"])
        self.assertEqual([step["role"] for step in records[0]["causal_chain"]], [
            "trigger", "price_memory", "crowd_interpretation", "positioning_confirmation", "possible_transitions",
        ])
        self.assertIn("causal_chain_json", flat.columns)
        self.assertEqual(summary["behavior_statistics"][0]["cases"], 1)

    def test_positioning_requires_two_fresh_confirmations(self):
        row = pd.Series({
            "price_advance_decline_breadth": -0.6, "breadth_age_days": 1,
            "tx_open_interest_change_1d": 1000, "open_interest_age_days": 1,
            "foreign_futures_net_oi": -50000, "institutional_position_age_days": 1,
            "net_urgency_score": -6,
        })

        diagnosis = positioning_diagnosis_from_row(row, "bearish")

        self.assertTrue(diagnosis["confirmed"])
        self.assertEqual(diagnosis["state"], "capitulation_or_euphoria_candidate")

    def test_borderline_course_is_recorded_without_changing_legacy_material_target(self):
        row = replay_row("2024-01-03", "2024-01-02", True, 0.00479)
        row.update({"direction": "bullish", "actual_direction": "bullish", "prior_cash_close": 100,
                    "cash_open": 100.5, "cash_high": 101, "cash_low": 99, "cash_close": 100.479})
        record = build_complete_ledger(pd.DataFrame([row]))[0]

        self.assertFalse(record["outcome"]["material_direction_confirmation"])
        self.assertTrue(record["outcome"]["borderline_direction_confirmation"])
        self.assertEqual(record["outcome"]["direction_strength"]["label"], "borderline_bullish")

    def test_pending_case_records_treatment_course_for_next_illness(self):
        current_payload = {
            "input": {"date": "2024-01-04"},
            "premarket": {
                "spot_latest_date": "2024-01-03",
                "external_date": "2024-01-03",
                "night_signal_date": "2024-01-04",
                "night_path": {"status": "bearish_but_recovered"},
                "tx_night_low": 95,
                "tx_night_close": 97,
                "external_score": -1,
            },
            "psychology_state": {"state": "doubt", "direction": "bearish", "evidence": []},
            "candlestick_pattern": {"type": "doji", "label": "十字線"},
            "washout_pattern": {"type": "none", "label": "非洗盤"},
            "market_health": {"symptoms": [], "diagnosis": {}, "prescription": {}},
            "technical_phase": {"levels": {"close": 100, "ma20": 96}, "status_code": "repair_watch"},
            "capital_flow": {"has_factor_values": False},
            "data_freshness": {"overall_status": "current"},
            "practical_cause_arbitration": {
                "framework": "practical_cause_arbitration_v1",
                "internal_structure_score": 8,
                "external_trigger_score": 5,
                "treatment_phase": 3,
                "treatment_stage": "發作治療",
                "treatment_policy": "加密核對",
                "medicine_type": "壓力測試藥",
                "medicine_effect": "測承接",
                "fact_changing_medicine": {
                    "label": "發病藥主導",
                    "effect": "即效因子偏向降低承接與風險胃納。",
                    "rule": "病灶是累積背景；即效藥是一發生就改變資金風險判斷的因素。",
                },
                "fact_changing_medicines": ["走弱即效藥: 夜盤大跌"],
                "immediate_medicine_bias": "weakening",
                "immediate_medicine_score": -2,
                "causality_rule": "先有因，再有跡象，再由即效藥/觸發按鈕產生果；最後用日盤與收盤驗證，寫入病歷。",
                "causality_pipeline": [{"stage": "因"}, {"stage": "觸發按鈕"}, {"stage": "果"}],
                "correct_medicine": "待日盤驗證",
                "after_effect_risk": "中",
                "gene_trigger_model": "病灶基因與觸發條件",
                "internal_causes": ["高檔估值"],
                "external_triggers": ["夜盤偏空"],
                "confirmation": ["日盤驗證"],
                "exclusion": ["夜盤不得單獨定論"],
            },
        }
        records = build_complete_ledger(pd.DataFrame([replay_row("2024-01-03", "2024-01-02", True, -0.02)]), current_payload)
        record = records[-1]
        treatment = record["treatment_course"]
        flat = flatten_records(records)

        self.assertTrue(treatment["available"])
        self.assertEqual(treatment["treatment"]["phase"], 3)
        self.assertEqual(treatment["treatment"]["medicine_type"], "壓力測試藥")
        self.assertEqual(treatment["fact_changing_medicine"]["bias"], "weakening")
        self.assertIn("先有因", treatment["causality_rule"])
        self.assertEqual(treatment["causality_pipeline"][0]["stage"], "因")
        self.assertIn("下期同類病灶", treatment["next_episode_prior"])
        self.assertIn("treatment_phase", flat.columns)
        self.assertIn("causality_pipeline_json", flat.columns)
        self.assertEqual(flat.iloc[-1]["after_effect_risk"], "中")


if __name__ == "__main__":
    unittest.main()
