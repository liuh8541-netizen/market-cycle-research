import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from predict_market import (
    BAGUA_SEQUENCE,
    analyze_bagua_lifecycle,
    analyze_bottom_event_reference,
    analyze_close_cause_attribution,
    analyze_day_night_variance_pattern,
    analyze_endogenous_regulation_pulse,
    analyze_intraday_tactical_monitor,
    analyze_human_behavior_market_pattern,
    analyze_night_close_psychology,
    analyze_psychological_warfare_pattern,
    analyze_monthly_cycle_monitor,
    analyze_night_path,
    analyze_market_mode_switch,
    analyze_practical_cause_arbitration,
    analyze_programmed_pressure_pattern,
    analyze_candlestick_pattern,
    analyze_crash_monitor,
    analyze_external_event_reset,
    analyze_fundamental_constitution,
    analyze_master_arbitration,
    analyze_peak_to_valley_warning,
    build_dialogue_core_rules,
    build_equation_reasoning_audit,
    analyze_route_reference,
    analyze_sector_pressure_observation,
    analyze_situation_psychology_context,
    analyze_technical_phase,
    analyze_tradeable_rally_segments,
    audit_data_freshness,
    build_error_review,
    build_night_basis_audit,
    build_breath_monitor_model,
    build_market_date_audit,
    build_peak_to_valley_warning_backtest,
    build_production_policy,
    build_weather_satellite_forecast_model,
    bagua_trade_annotation,
    compact_forecast_record,
    classify_bagua_row,
    classify_candlestick,
    classify_drawdown_guardrail,
    expected_night_signal_date,
    market_mode_score_range,
    market_scenario,
    render_breath_monitor,
    render_error_review,
    render_programmed_pressure_pattern,
    strong_target_definition,
    status,
    us_market_holiday_name,
)
from market_lifecycle.features import add_features


class ProductionValidationPolicyTest(unittest.TestCase):
    def setUp(self):
        self.validation = json.loads(
            (ROOT / "config" / "model_validation_status.json").read_text(encoding="utf-8")
        )

    def test_failed_main_gate_disables_multi_day_signal(self):
        policy = build_production_policy(self.validation)
        self.assertFalse(policy["main_multi_day_direction"]["enabled"])
        self.assertIsNone(policy["main_multi_day_direction"]["formal_signal"])

    def test_only_scope_limited_submodels_are_validated(self):
        policy = build_production_policy(self.validation)
        self.assertEqual(
            set(policy["validated_submodels"]),
            {"canonical_night_spread_gap_v1", "daily_night_gap_amplitude_v1"},
        )
        self.assertIn("cash open", policy["validated_submodels"]["canonical_night_spread_gap_v1"]["scope"])
        self.assertIn("opening gap", policy["validated_submodels"]["daily_night_gap_amplitude_v1"]["scope"])

    def test_closeout_is_explicitly_a_failed_target_not_a_pass(self):
        self.assertTrue(self.validation["research_completed"])
        self.assertFalse(self.validation["passed"])
        self.assertEqual(
            self.validation["research_outcome"],
            "main_multi_day_direction_target_not_met",
        )

    def test_unfinished_pre_5am_night_session_is_not_marked_late(self):
        now = datetime(2026, 7, 24, 4, 30, tzinfo=ZoneInfo("Asia/Taipei"))
        self.assertEqual(
            expected_night_signal_date("2026-07-24", now).isoformat(),
            "2026-07-23",
        )

    def test_completed_night_session_uses_same_signal_date(self):
        now = datetime(2026, 7, 24, 5, 1, tzinfo=ZoneInfo("Asia/Taipei"))
        self.assertEqual(
            expected_night_signal_date("2026-07-24", now).isoformat(),
            "2026-07-24",
        )

    def test_weekend_night_session_uses_next_weekday_signal(self):
        now = datetime(2026, 8, 15, 7, 0, tzinfo=ZoneInfo("Asia/Taipei"))
        self.assertEqual(
            expected_night_signal_date("2026-08-15", now).isoformat(),
            "2026-08-17",
        )

    def test_deep_post_peak_decline_maps_to_kan_not_zhen(self):
        row = {
            "close": 41603.36,
            "bagua_pos": 0.62,
            "bagua_ret_short": -0.075,
            "bagua_ret_mid": 0.069,
            "bagua_ret_long": 0.31,
            "bagua_drawdown": -0.129,
            "bagua_ma_short": 44954.60,
            "bagua_ma_mid": 44081.10,
        }
        self.assertEqual(classify_bagua_row(row), "KAN")

    def test_low_base_upward_breakout_maps_to_zhen(self):
        row = {
            "close": 100.0,
            "bagua_pos": 0.40,
            "bagua_ret_short": 0.04,
            "bagua_ret_mid": -0.01,
            "bagua_ret_long": -0.10,
            "bagua_drawdown": -0.12,
            "bagua_ma_short": 98.0,
            "bagua_ma_mid": 105.0,
        }
        self.assertEqual(classify_bagua_row(row), "ZHEN")

    def test_bagua_sequence_starts_from_zhen_and_ends_at_gen(self):
        self.assertEqual(
            BAGUA_SEQUENCE,
            ["ZHEN", "XUN", "LI", "KUN", "DUI", "QIAN", "KAN", "GEN"],
        )

    def test_bagua_trade_annotations_compare_buy_and_sell_behavior(self):
        for code in BAGUA_SEQUENCE:
            note = bagua_trade_annotation(code)
            self.assertIn("買方", f"買方 {note['buy_behavior']}")
            self.assertTrue(note["sell_behavior"])
            self.assertIn("不是投資命令", note["guardrail"])
        self.assertEqual(bagua_trade_annotation("ZHEN")["label"], "買點觀察")
        self.assertEqual(bagua_trade_annotation("KAN")["label"], "空手防守")

    def test_technical_phase_outputs_fixed_formula_steps(self):
        data = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
        data["date"] = pd.to_datetime(data["date"])
        data = data[data["date"] <= pd.Timestamp("2026-07-28")].copy()
        data = add_features(data)
        candle = analyze_candlestick_pattern(data, "2026-07-28")
        bagua = analyze_bagua_lifecycle(data, "2026-07-28", candle, {"type": "failed_washout"})
        cycle = analyze_tradeable_rally_segments(data, "2026-07-28")
        technical = analyze_technical_phase(data, "2026-07-28", bagua, candle, cycle)
        self.assertTrue(technical["enabled"])
        self.assertEqual(len(technical["fixed_formula"]), 6)
        self.assertEqual(technical["fixed_formula"][0]["step"], "1. 主波段")
        self.assertEqual(technical["equation_answer"]["framework"], "technical_equation_answer_v1")
        self.assertIn("答案", technical["equation_answer"]["answer"])
        self.assertTrue(technical["equation_answer"]["best_formula"])
        self.assertEqual(technical["equation_answer"]["research_status"], "experimental_hypothesis")

    def test_route_reference_uses_historical_pattern_samples(self):
        data = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
        data["date"] = pd.to_datetime(data["date"])
        data = data[data["date"] <= pd.Timestamp("2026-07-28")].copy()
        data = add_features(data)
        candle = analyze_candlestick_pattern(data, "2026-07-28")
        bagua = analyze_bagua_lifecycle(data, "2026-07-28", candle, {"type": "failed_washout"})
        cycle = analyze_tradeable_rally_segments(data, "2026-07-28")
        technical = analyze_technical_phase(data, "2026-07-28", bagua, candle, cycle)
        route = analyze_route_reference(data, "2026-07-28", bagua, technical)
        self.assertTrue(route["enabled"])
        self.assertIn("current_path", route)
        self.assertIn("sample_count", route)
        self.assertGreaterEqual(route["sample_count"], 0)

    def test_bottom_event_reference_uses_historical_same_event_samples(self):
        data = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
        data["date"] = pd.to_datetime(data["date"])
        data = data[data["date"] <= pd.Timestamp("2026-07-28")].copy()
        data = add_features(data)
        candle = analyze_candlestick_pattern(data, "2026-07-28")
        bagua = analyze_bagua_lifecycle(data, "2026-07-28", candle, {"type": "failed_washout"})
        reference = analyze_bottom_event_reference(data, "2026-07-28", bagua)
        self.assertTrue(reference["enabled"])
        self.assertIn("current_event", reference)
        self.assertIn("sample_count", reference)
        self.assertIn("event_rates", reference["stats"])
        self.assertIn("不使用事後低點回填", reference["method"])

    def test_peak_to_valley_warning_detects_june_peak_candidate(self):
        data = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
        data["date"] = pd.to_datetime(data["date"])
        data = data[data["date"] <= pd.Timestamp("2026-06-23")].copy()
        data = add_features(data)
        warning = analyze_peak_to_valley_warning(data, "2026-06-23")
        self.assertTrue(warning["enabled"])
        self.assertEqual(warning["status_code"], "peak_candidate")
        self.assertEqual(warning["current_event"]["event_code"], "peak_candidate")
        self.assertTrue(warning["current_event"]["conditions"]["new_60d_high"])
        self.assertTrue(warning["current_event"]["conditions"]["close_in_low_quarter"])

    def test_peak_to_valley_backtest_outputs_auditable_statistics(self):
        data = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
        data["date"] = pd.to_datetime(data["date"])
        data = data[data["date"] <= pd.Timestamp("2026-08-03")].copy()
        data = add_features(data)
        backtest = build_peak_to_valley_warning_backtest(data, "2026-08-03")
        self.assertTrue(backtest["enabled"])
        self.assertGreater(backtest["event_count"], 0)
        self.assertIn("不回填當日判斷", backtest["method"])
        self.assertIn("by_horizon", backtest["stats"])

    def test_crash_monitor_outputs_bottom_and_warning_levels(self):
        data = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
        data["date"] = pd.to_datetime(data["date"])
        data = data[data["date"] <= pd.Timestamp("2026-07-28")].copy()
        data = add_features(data)
        candle = analyze_candlestick_pattern(data, "2026-07-28")
        bagua = analyze_bagua_lifecycle(data, "2026-07-28", candle, {"type": "failed_washout"})
        cycle = analyze_tradeable_rally_segments(data, "2026-07-28")
        technical = analyze_technical_phase(data, "2026-07-28", bagua, candle, cycle)
        monitor = analyze_crash_monitor(
            data,
            "2026-07-28",
            technical,
            bagua,
            {"is_premarket": False},
            {"has_factor_values": False},
        )
        self.assertTrue(monitor["enabled"])
        self.assertIn("alert_level", monitor)
        self.assertIn("early_warnings", monitor)
        self.assertIn("risk_value", monitor)
        self.assertIn("health_score", monitor)
        self.assertEqual(len(monitor["risk_components"]), 6)
        self.assertLess(monitor["levels"]["crash_warning"], monitor["peak"])
        self.assertGreater(monitor["levels"]["normal_bottom_high"], monitor["levels"]["normal_bottom_low"])

    def test_market_date_audit_separates_intraday_snapshot_from_official_close(self):
        audit = build_market_date_audit(
            "2026-08-11",
            {"signal_date": "2026-08-10", "data_close": 44928.76},
            {
                "enabled": True,
                "date": "2026-08-11",
                "price": 44848.82,
                "low": 44652.06,
                "source": "Yahoo ^TWII live chart",
            },
            {"is_premarket": False},
        )
        self.assertEqual(audit["mode"], "intraday_snapshot")
        self.assertTrue(audit["must_not_call_signal_date_today"])
        self.assertFalse(audit["official_daily_complete"])
        self.assertEqual(audit["official_signal_date"], "2026-08-10")
        self.assertEqual(audit["live_snapshot_date"], "2026-08-11")
        self.assertIn("不得稱為今日收盤", audit["status"])

    def test_market_date_audit_allows_today_close_only_when_signal_date_matches(self):
        audit = build_market_date_audit(
            "2026-08-11",
            {"signal_date": "2026-08-11", "data_close": 44800.0},
            {"enabled": False, "date": None},
            {"is_premarket": False},
        )
        self.assertEqual(audit["mode"], "official_daily")
        self.assertFalse(audit["must_not_call_signal_date_today"])
        self.assertTrue(audit["official_daily_complete"])

    def test_market_date_audit_marks_weekend_as_non_trading_day(self):
        audit = build_market_date_audit(
            "2026-08-15",
            {"signal_date": "2026-08-14", "data_close": 45811.01},
            {"enabled": False, "date": None},
            {"is_premarket": False, "is_non_trading_day": True},
        )
        self.assertEqual(audit["mode"], "non_trading_day")
        self.assertIn("非交易日", audit["label"])

    def test_external_event_reset_detects_bearish_global_reset(self):
        reset = analyze_external_event_reset(
            {
                "nasdaq_return_1d": -0.025,
                "sp500_return_1d": -0.016,
                "sox_return_1d": -0.035,
                "tsm_adr_return_1d": -0.030,
                "vix_return_1d": 0.12,
                "tx_night_spread_per": -0.015,
            }
        )
        self.assertTrue(reset["reset_active"])
        self.assertEqual(reset["code"], "bearish_external_reset")
        self.assertEqual(reset["causal_priority"], "external_reset")
        self.assertLess(reset["risk_adjustment"], 0)

    def test_external_event_reset_keeps_internal_chain_when_quiet(self):
        reset = analyze_external_event_reset(
            {
                "nasdaq_return_1d": 0.002,
                "sp500_return_1d": -0.001,
                "sox_return_1d": 0.003,
                "tsm_adr_return_1d": 0.001,
                "vix_return_1d": -0.005,
                "tx_night_spread_per": 0.001,
            }
        )
        self.assertFalse(reset["reset_active"])
        self.assertEqual(reset["code"], "internal_cause_primary")
        self.assertEqual(reset["causal_priority"], "internal_primary")

    def test_external_event_reset_uses_global_news_risk(self):
        reset = analyze_external_event_reset(
            {
                "nasdaq_return_1d": 0.0,
                "sp500_return_1d": 0.0,
                "sox_return_1d": 0.0,
                "vix_return_1d": 0.0,
                "tx_night_spread_per": 0.0,
            },
            {
                "status": "connected",
                "risk_score": 6,
                "tailwind_score": 0,
                "net_risk_score": 6,
                "summary": "國際重大財經政治消息偏風險。",
            },
        )
        self.assertTrue(reset["reset_active"])
        self.assertEqual(reset["causal_priority"], "external_reset")
        self.assertEqual(reset["news_feed_status"], "connected")
        self.assertIn("國際重大財經政治消息偏風險", " ".join(reset["reasons"]))

    def test_night_basis_audit_keeps_three_baselines_separate(self):
        row = pd.Series(
            {
                "night_date": "2026-09-02",
                "signal_date": "2026-09-02",
                "tx_night_open": 47201,
                "tx_night_close": 46679,
                "tx_night_return": -0.011059,
                "tx_night_spread_per": -0.004,
            }
        )
        audit = build_night_basis_audit(row, 45978, 46679 / 45978 - 1)
        self.assertTrue(audit["available"])
        self.assertAlmostEqual(audit["open_close_return"], -0.011059)
        self.assertAlmostEqual(audit["vs_previous_night_close_return"], 46679 / 45978 - 1)
        self.assertTrue(audit["basis_conflict"])
        self.assertIn("不得混", audit["guardrail"])

    def test_fundamental_constitution_scores_body_separately_from_timing(self):
        constitution = analyze_fundamental_constitution(
            {
                "premarket": {
                    "nasdaq_return_1d": -0.02,
                    "sox_return_1d": -0.03,
                    "tsm_adr_return_1d": -0.025,
                    "vix_return_1d": 0.10,
                    "usd_twd_return_1d": 0.005,
                },
                "global_news_risk": {"available": True, "net_risk_score": 6},
                "technical_phase": {"signals": {"above_ma20": True}},
                "bagua_lifecycle": {"primary": {"code": "QIAN"}},
                "capital_flow": {"has_factor_values": True},
                "memory_industry_risk": {"points": 0},
            }
        )
        self.assertLess(constitution["score"], 62)
        self.assertIn("國際重大財經政治消息偏風險", constitution["pressures"])
        self.assertIn("基本面只回答市場身體", constitution["guardrail"])

    def test_practical_cause_arbitration_treats_news_as_trigger_when_high_level_disease_exists(self):
        practical = analyze_practical_cause_arbitration(
            {
                "technical_phase": {
                    "levels": {
                        "close": 47183,
                        "ma20": 45960,
                        "swing_high": 48218,
                        "drawdown_from_swing_high": -0.021,
                        "range_20d": 0.076,
                    }
                },
                "tradeable_cycle": {"current_position": {"label": "末升段/高檔加速"}},
                "bagua_lifecycle": {"roles": {"background_gua": {"code": "QIAN"}}},
                "sector_pressure_observation": {"risk_score": 3},
                "capital_flow": {"score": 0},
                "programmed_pressure_pattern": {"current_score": 2},
                "candlestick_pattern": {"close_position": 0.25},
                "premarket": {
                    "nasdaq_return_1d": -0.0064,
                    "sox_return_1d": 0.0037,
                    "tsm_adr_return_1d": -0.0083,
                    "vix_return_1d": 0.047,
                    "treasury_10y_return_1d": 0.0064,
                    "tx_night_spread_per": -0.0041,
                },
                "global_news_risk": {"net_risk_score": 1},
                "external_event_reset_monitor": {"reset_active": False, "direction": "neutral"},
            }
        )
        self.assertEqual(practical["code"], "internal_disease_external_trigger")
        self.assertEqual(practical["practical_primary"], "internal_structure")
        self.assertIn("病灶基因", practical["gene_trigger_model"])
        self.assertIn("觸發條件", practical["gene_trigger_model"])
        self.assertGreater(practical["internal_structure_score"], practical["external_trigger_score"])
        self.assertEqual(practical["immediate_medicine_bias"], "neutral")
        self.assertIn("先有因", practical["causality_rule"])
        self.assertEqual([step["stage"] for step in practical["causality_pipeline"]], ["因", "跡象", "觸發按鈕", "果", "病歷"])
        self.assertIn("dialogue_core_rules_v1", practical["dialogue_core_rules"]["framework"])
        self.assertIn("單日變化", practical["dialogue_focus"])

    def test_practical_cause_arbitration_allows_external_reset_to_temporarily_lead(self):
        practical = analyze_practical_cause_arbitration(
            {
                "technical_phase": {"levels": {"close": 45000, "ma20": 46000, "drawdown_from_swing_high": -0.08}},
                "tradeable_cycle": {"current_position": {"label": "回測段"}},
                "bagua_lifecycle": {"roles": {"background_gua": {"code": "KAN"}}},
                "sector_pressure_observation": {"risk_score": 1},
                "capital_flow": {"score": -1},
                "programmed_pressure_pattern": {"current_score": 0},
                "candlestick_pattern": {"close_position": 0.5},
                "premarket": {
                    "nasdaq_return_1d": -0.025,
                    "sox_return_1d": -0.03,
                    "tsm_adr_return_1d": -0.025,
                    "vix_return_1d": 0.12,
                    "treasury_10y_return_1d": 0.012,
                    "tx_night_spread_per": -0.011,
                },
                "global_news_risk": {"net_risk_score": 5},
                "external_event_reset_monitor": {"reset_active": True, "direction": "bearish"},
            }
        )
        self.assertEqual(practical["code"], "external_trigger_temporary_primary")
        self.assertEqual(practical["practical_primary"], "external_reset")
        self.assertEqual(practical["immediate_medicine_bias"], "weakening")
        self.assertLess(practical["immediate_medicine_score"], 0)

    def test_dialogue_core_rules_filter_chronic_causes_from_daily_triggers(self):
        rules = build_dialogue_core_rules(
            {
                "technical_phase": {
                    "levels": {
                        "close": 47183,
                        "ma20": 45960,
                        "drawdown_from_swing_high": -0.021,
                        "range_20d": 0.076,
                    }
                },
                "bagua_lifecycle": {"roles": {"background_gua": {"code": "QIAN"}}},
                "sector_pressure_observation": {"risk_score": 3},
                "capital_flow": {"score": -2},
                "programmed_pressure_pattern": {"current_score": 2},
                "premarket": {
                    "tx_night_spread_per": -0.0041,
                    "nasdaq_return_1d": -0.0064,
                    "sox_return_1d": -0.012,
                    "vix_return_1d": 0.047,
                    "treasury_10y_return_1d": 0.0064,
                },
                "global_news_risk": {"net_risk_score": 2},
                "external_event_reset_monitor": {"reset_active": False},
                "day_night_variance_pattern": {"label": "待日盤驗證"},
                "market_heart_rhythm": {"label": "生命徵象混合"},
            }
        )
        self.assertEqual(rules["framework"], "dialogue_core_rules_v1")
        self.assertEqual(rules["state"], "disease_with_trigger")
        self.assertTrue(rules["chronic_conditions"])
        self.assertTrue(rules["trigger_conditions"])
        self.assertIn("觸發鈕", rules["focus"])
        self.assertIn("報告只輸出仲裁結果", "；".join(rules["model_rules"]))

    def test_equation_reasoning_audit_compares_formula_with_reasoning(self):
        audit = build_equation_reasoning_audit(
            {
                "technical_phase": {
                    "equation_answer": {
                        "answer": "答案偏1：修復/續攻候選",
                        "branch": "1",
                        "best_formula": "均線方程式",
                        "best_formula_reason": "收盤站回20日線。",
                    }
                },
                "weather_satellite_forecast": {
                    "zero_one_tilt": {"label": "偏1：修復候選待確認", "branch": "1"}
                },
                "practical_cause_arbitration": {
                    "practical_primary": "internal_structure",
                    "label": "內部主病灶，外部觸發",
                },
                "master_arbitration": {},
                "day_night_variance_pattern": {"relation_code": "pending_day_validation", "label": "待日盤驗證"},
                "market_protection_layers": {"label": "保護層有效", "failed_layers": []},
            }
        )
        self.assertEqual(audit["framework"], "equation_reasoning_audit_v1")
        self.assertEqual(audit["formula_branch"], "1")
        self.assertEqual(audit["reasoning_branch"], "1")
        self.assertIn("一致", audit["label"])
        self.assertIn("推理式程序", audit["rule"])
        self.assertIn("高等數學", audit["math_policy"])
        self.assertTrue(audit["candidate_variables"])
        self.assertEqual(audit["research_status"], "experimental_hypothesis")
        self.assertIn("marginal_label", audit)
        self.assertGreater(audit["push_1_score"], 0)
        self.assertTrue(audit["swing_factors"])
        self.assertIn("不可測", audit["black_box_unpredictable"]["label"])
        self.assertIn("不可假裝能預測", audit["black_box_unpredictable"]["model_policy"])

    def test_compact_forecast_record_persists_treatment_tracking_for_next_episode(self):
        payload = {
            "input": {"date": "2026-09-10"},
            "index_check": {"signal_date": "2026-09-09", "data_close": 47183.36},
            "premarket": {"pressure": "中性", "total_score": 0},
            "cause_analysis": {"scenario": {"name": "一般震盪盤"}, "truth_text": "待驗證"},
            "forecast": {
                "lifecycle_stage": "main_bull",
                "risk_regime": "risk_on",
                "validation_status": "exploratory_not_production",
                "forecasts": [
                    {
                        "horizon_days": 1,
                        "base_trade_date": "2026-09-09",
                        "target_trade_date": "2026-09-10",
                        "target_date_estimated": False,
                        "predicted_direction": "sideways",
                        "confidence": 0.45,
                        "probability_up": 0.37,
                        "probability_down": 0.17,
                        "probability_sideways": 0.45,
                        "fixed_factor_review": None,
                    }
                ],
            },
            "production_policy": {"main_multi_day_direction": {"enabled": False}},
            "bagua_lifecycle": {},
            "tradeable_cycle": {},
            "candlestick_pattern": {},
            "washout_pattern": {},
            "practical_cause_arbitration": {
                "framework": "practical_cause_arbitration_v1",
                "code": "internal_disease_external_trigger",
                "label": "內部主病灶，外部觸發",
                "practical_primary": "internal_structure",
                "internal_structure_score": 8,
                "external_trigger_score": 5,
                "combined_severity": 13,
                "treatment_phase": 3,
                "treatment_stage": "發作治療",
                "medicine_type": "壓力測試藥",
                "fact_changing_medicines": ["走弱即效藥: 夜盤大跌"],
                "immediate_medicine_bias": "weakening",
                "immediate_medicine_score": -2,
                "causality_rule": "先有因，再有跡象，再由即效藥/觸發按鈕產生果；最後用日盤與收盤驗證，寫入病歷。",
                "causality_pipeline": [{"stage": "因"}, {"stage": "果"}],
                "correct_medicine": "待日盤驗證",
                "after_effect_risk": "中",
                "internal_causes": ["高檔估值", "前高套牢"],
                "external_triggers": ["美股轉弱", "夜盤偏空"],
            },
        }
        record = compact_forecast_record(payload)
        treatment = record["treatment_tracking"]
        self.assertEqual(treatment["treatment_phase"], 3)
        self.assertEqual(treatment["medicine_type"], "壓力測試藥")
        self.assertEqual(treatment["after_effect_risk"], "中")
        self.assertEqual(treatment["immediate_medicine_bias"], "weakening")
        self.assertIn("先有因", treatment["causality_rule"])
        self.assertEqual(treatment["causality_pipeline"][0]["stage"], "因")
        self.assertIn("下期同類病灶", treatment["next_episode_prior"])

    def test_master_arbitration_prioritizes_external_shock_over_high_health(self):
        arbitration = analyze_master_arbitration(
            {
                "market_health": {"score": 88, "risk_score": 12},
                "fundamental_constitution": {"score": 68},
                "external_event_reset_monitor": {
                    "code": "bearish_external_reset",
                    "direction": "bearish",
                    "reset_active": True,
                },
                "direction_reliability_policy": {"direction_enabled": False},
                "sector_pressure_observation": {"risk_score": 1},
                "psychological_warfare_pattern": {"score": 3},
                "premarket": {"night_basis_audit": {"basis_conflict": False}},
                "crash_monitor": {},
            }
        )
        self.assertEqual(arbitration["code"], "healthy_body_external_shock")
        self.assertEqual(arbitration["dominant_layer"], "external_reset")
        self.assertTrue(any("健康分數偏高" in item for item in arbitration["conflicts"]))

    def test_monthly_cycle_monitor_flags_weak_range_with_normalized_shock(self):
        data = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
        data["date"] = pd.to_datetime(data["date"])
        data = data[data["date"] <= pd.Timestamp("2026-08-24")].copy()
        cycle = analyze_monthly_cycle_monitor(
            data,
            "2026-08-24",
            {"current_score": 6},
            {"code": "bearish_external_reset"},
        )
        self.assertTrue(cycle["available"])
        self.assertEqual(cycle["code"], "weak_range_with_normalized_shock")
        self.assertIn("突變常態化", cycle["stage"])
        self.assertGreaterEqual(cycle["week_of_month"], 4)
        self.assertIn("站回45,000", cycle["next_validation"][0])

    def test_data_freshness_uses_previous_weekday_for_weekend_spot_data(self):
        data = pd.DataFrame(
            [
                {"date": "2026-08-14", "open": 45500, "high": 46200, "low": 45300, "close": 46000, "volume": 1},
            ]
        )
        data["date"] = pd.to_datetime(data["date"])
        freshness = audit_data_freshness(
            "2026-08-16",
            data,
            ROOT / "data" / "processed" / "external_markets.csv",
            ROOT / "data" / "processed" / "taiwan_futures_night.csv",
            {"has_factor_values": False, "data_status": "測試不接入"},
        )
        spot = next(item for item in freshness["sources"] if item["name"] == "台股現貨")
        self.assertEqual(spot["expected_date"], "2026-08-14")
        self.assertEqual(spot["status"], "current")

    def test_deep_decline_from_new_high_is_not_normal_gen(self):
        levels = {
            "normal_gen_low": 47000,
            "normal_gen_high": 48000,
            "healthy_pullback_low": 45000,
            "major_correction_line": 42500,
            "normal_bottom_low": 41250,
            "normal_bottom_high": 41750,
            "crash_warning": 40000,
        }
        deep = classify_drawdown_guardrail(50000, 41500, 41500, levels)
        self.assertEqual(deep["stage_code"], "major_correction_policy_watch")
        self.assertFalse(deep["normal_gen_allowed"])
        self.assertIn("重大修正", deep["stage"])

        normal = classify_drawdown_guardrail(50000, 47500, 47500, levels)
        self.assertEqual(normal["stage_code"], "normal_high_gen_zone")
        self.assertTrue(normal["normal_gen_allowed"])

    def test_market_mode_switch_surfaces_early_crisis_over_health_wording(self):
        payload = {
            "crash_monitor": {
                "risk_value": 50,
                "alert_code": "watch",
                "alert_level": "預警",
                "drawdown_guardrail": {
                    "stage_code": "healthy_pullback",
                    "stage": "健康修正區",
                },
            },
            "premarket": {
                "is_premarket": True,
                "total_score": -4,
                "tx_night_spread_per": -0.016,
                "sox_return_1d": -0.03,
                "vix_return_1d": 0.09,
            },
            "bagua_lifecycle": {
                "roles": {
                    "state_gua": {"code": "ZHEN"},
                    "background_gua": {"code": "QIAN"},
                },
                "daily": {"code": "KAN"},
            },
            "memory_industry_risk": {"level": "watch"},
            "capital_flow": {"has_factor_values": False},
            "cause_analysis": {},
        }
        mode = analyze_market_mode_switch(payload)
        self.assertIn(mode["mode"], {"early_crisis_watch", "tail_risk_defense"})
        self.assertTrue(mode["crisis_early"])
        self.assertIn("危機", mode["headline"] + mode["action"])

    def test_market_mode_score_range_and_strong_target_definition_are_explicit(self):
        self.assertIn("0-1分", market_mode_score_range(1))
        self.assertIn("低風險觸發", market_mode_score_range(1))
        definition = strong_target_definition()
        self.assertIn("站上5日與20日線", definition)
        self.assertIn("相對大盤", definition)
        self.assertIn("不列強勢", definition)

    def test_close_cause_attribution_flags_tailwind_internal_selloff(self):
        result = analyze_close_cause_attribution(
            {
                "input": {"index": 45857.66},
                "premarket": {
                    "external_score": 1,
                    "nasdaq_return_1d": 0.0045,
                    "sox_return_1d": 0.0044,
                    "vix_return_1d": -0.069,
                },
                "external_event_reset_monitor": {"code": "external_conflict_watch"},
                "intraday_tactical_monitor": {
                    "live_price": 45857.66,
                    "live_open": 46325.48,
                    "live_high": 46517.45,
                    "live_low": 45839.36,
                    "last_close": 46164.72,
                    "vs_prev_close": -0.00665,
                    "close_position": 0.027,
                },
                "sector_pressure_observation": {
                    "risk_score": 3,
                    "weak_sectors": ["中型股", "電子零組件"],
                },
                "programmed_pressure_pattern": {"current_score": 6},
                "market_mode_switch": {"mode": "high_level_rotation"},
                "candlestick_pattern": {"type": "gap_up_close_near_low"},
            }
        )
        self.assertEqual(result["code"], "external_tailwind_internal_selloff")
        self.assertTrue(result["features"]["close_below_46000"])
        self.assertTrue(result["features"]["held_45700"])
        self.assertGreater(result["risk_adjustment"], 0)
        self.assertIn("不得產生買賣命令", result["guardrail"])

    def test_sector_pressure_observation_flags_index_up_breadth_divergence(self):
        data = pd.DataFrame(
            [
                {"date": pd.Timestamp("2026-08-27"), "close": 45975.22},
            ]
        )
        path = ROOT / "reports" / "_tmp_manual_sector_observations.csv"
        path.write_text(
            "\n".join(
                [
                    "date,sector,status,severity,note,source",
                    "2026-08-28,電子科技,weak,warning,大盤漲但電子弱,test",
                    "2026-08-28,軍工/國防自主,weak,warning,大盤漲但軍工弱,test",
                ]
            ),
            encoding="utf-8",
        )
        try:
            result = analyze_sector_pressure_observation(
                "2026-08-28",
                "2026-08-27",
                46331.45,
                data,
                path,
            )
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(result["code"], "index_up_sector_pressure")
        self.assertIn("電子科技", result["weak_sectors"])
        self.assertIn("軍工/國防自主", result["weak_sectors"])

    def test_endogenous_regulation_pulse_detects_quiet_external_reclaim(self):
        data = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-08-01") + pd.Timedelta(days=i),
                    "open": 100 + i,
                    "high": 102 + i,
                    "low": 99 + i,
                    "close": 101 + i,
                    "volume": 1000,
                }
                for i in range(24)
            ]
            + [
                {
                    "date": pd.Timestamp("2026-08-25"),
                    "open": 125,
                    "high": 128,
                    "low": 123,
                    "close": 127,
                    "volume": 1200,
                    "sp500_return_1d": 0.002,
                    "nasdaq_return_1d": -0.001,
                    "sox_return_1d": 0.003,
                    "tsm_adr_return_1d": 0.001,
                }
            ]
        )
        data = add_features(data)
        result = analyze_endogenous_regulation_pulse(
            data,
            "2026-08-25",
            {"code": "internal_cause_primary"},
            {"relation_code": "night_down_cash_reversal", "relation_label": "夜跌日收回變異"},
            {"current_score": 2},
            {"risk_score": 1},
        )
        self.assertEqual(result["code"], "endogenous_regulation_active")
        self.assertTrue(result["external_quiet"])
        self.assertTrue(result["close_above_ma20"])
        self.assertIn("自我調節", result["guardrail"])

    def test_breath_monitor_marks_crisis_mode_as_abnormal_breathing(self):
        payload = {
            "input": {"date": "2026-08-11"},
            "index_check": {"signal_date": "2026-08-11"},
            "market_mode_switch": {
                "headline": "危機初期跡象",
                "action": "先提高監控頻率。",
                "risk_score": 6,
                "crisis_early": True,
                "mode": "early_crisis_watch",
            },
            "crash_monitor": {
                "risk_value": 55,
                "health_score": 45,
                "effective_close": 45120.72,
                "effective_low": 44652.06,
                "drawdown_guardrail": {
                    "stage_code": "healthy_pullback",
                    "stage": "健康修正區",
                    "summary": "仍未跌入重大修正，但節奏偏急。",
                    "drawdown": -0.064,
                },
                "levels": {
                    "normal_gen_low": 45326,
                    "normal_gen_high": 46290,
                    "major_correction_line": 40986,
                    "crash_warning": 38575,
                },
            },
            "premarket": {
                "tx_night_spread_per": -0.016,
                "tx_night_close": 44500,
                "tx_night_low": 44200,
                "sox_return_1d": -0.025,
                "vix_return_1d": 0.08,
            },
            "technical_phase": {"levels": {"ma5": 45000, "ma20": 44000}},
            "memory_industry_risk": {"level": "watch", "summary": "外部市場同步承壓"},
            "capital_flow": {"has_factor_values": False, "data_status": "未接入"},
            "market_date_audit": {
                "mode": "official_daily",
                "label": "日期稽核通過",
                "status": "官方日線完成",
                "official_signal_date": "2026-08-11",
            },
        }
        model = build_breath_monitor_model(payload)
        self.assertIn(model["breath_code"], {"warning", "critical"})
        self.assertIn("異常", model["breath_state"])
        self.assertTrue(any(item["kind"] == "總控" and item["severity"] == "warning" for item in model["events"]))
        self.assertIn("危機初期", " ".join(model["rules"]))

    def test_breath_monitor_html_contains_scrolling_stream_and_abnormal_filter(self):
        payload = {
            "input": {"date": "2026-08-11"},
            "index_check": {"signal_date": "2026-08-11"},
            "market_mode_switch": {"headline": "正常監控", "risk_score": 1, "crisis_early": False},
            "crash_monitor": {
                "risk_value": 10,
                "effective_close": 45120.72,
                "effective_low": 44652.06,
                "drawdown_guardrail": {},
                "levels": {},
            },
            "premarket": {},
            "technical_phase": {"levels": {}},
            "memory_industry_risk": {},
            "capital_flow": {"has_factor_values": True},
            "market_date_audit": {"mode": "official_daily"},
        }
        html = render_breath_monitor(payload)
        self.assertIn("台股呼吸脈動滾動監控", html)
        self.assertIn("eventStream", html)
        self.assertIn("只看異常", html)
        self.assertIn("data-filter=\"abnormal\"", html)

    def test_intraday_tactical_monitor_marks_weekend_as_observation_only(self):
        data = pd.DataFrame(
            [
                {"date": "2026-08-13", "open": 45000, "high": 45600, "low": 44800, "close": 45400},
                {"date": "2026-08-14", "open": 45500, "high": 46200, "low": 45300, "close": 46000},
            ]
        )
        data["date"] = pd.to_datetime(data["date"])
        data = add_features(data)
        monitor = analyze_intraday_tactical_monitor(
            "2026-08-15",
            data,
            "2026-08-14",
            {
                "is_non_trading_day": True,
                "night_signal_date": "2026-08-17",
                "tx_night_spread_per": -0.0019,
                "tx_night_close": 45727,
                "tx_night_low": 45551,
            },
            {"enabled": False, "source": "none", "message": "週末不抓盤中快照。"},
        )
        self.assertEqual(monitor["code"], "non_trading_observation")
        self.assertFalse(monitor["live_usable"])
        self.assertIn("不能做即時盤中判斷", monitor["summary"])

    def test_intraday_tactical_monitor_flags_night_down_cash_reclaim_candidate(self):
        data = pd.DataFrame(
            [
                {"date": "2026-08-13", "open": 45000, "high": 45600, "low": 44800, "close": 45400},
                {"date": "2026-08-14", "open": 45500, "high": 46200, "low": 45300, "close": 46000},
            ]
        )
        data["date"] = pd.to_datetime(data["date"])
        data = add_features(data)
        monitor = analyze_intraday_tactical_monitor(
            "2026-08-17",
            data,
            "2026-08-14",
            {
                "is_premarket": True,
                "night_signal_date": "2026-08-17",
                "tx_night_spread_per": -0.006,
                "tx_night_close": 45700,
                "tx_night_low": 45450,
            },
            {
                "enabled": True,
                "source": "test",
                "date": "2026-08-17",
                "price": 46120,
                "open": 45650,
                "high": 46250,
                "low": 45520,
                "message": "測試盤中快照",
            },
        )
        self.assertEqual(monitor["code"], "night_down_cash_reclaim_candidate")
        self.assertTrue(monitor["live_usable"])
        self.assertTrue(monitor["reclaimed_last_close"])
        self.assertFalse(monitor["broke_last_low"])

    def test_human_behavior_pattern_reads_night_down_cash_reclaim_as_accumulation(self):
        data = pd.DataFrame(
            [
                {"date": "2026-08-13", "open": 45000, "high": 45600, "low": 44800, "close": 45400, "volume": 1},
                {"date": "2026-08-14", "open": 45500, "high": 46200, "low": 45300, "close": 46000, "volume": 2},
            ]
        )
        data["date"] = pd.to_datetime(data["date"])
        data = add_features(data)
        premarket = {
            "night_path": {"status": "bearish_but_recovered"},
            "tx_night_spread_per": -0.006,
            "tx_night_close": 45700,
            "tx_night_low": 45450,
        }
        intraday = {"code": "night_down_cash_reclaim_candidate"}
        pattern = analyze_human_behavior_market_pattern(
            data,
            "2026-08-14",
            premarket,
            intraday,
            {"type": "long_bull"},
            {"type": "no_washout"},
            {"roles": {"state_gua": {"code": "ZHEN"}}},
            {"label": "修復觀察"},
            {"stage": "盤整收斂"},
            {"current_score": 0},
        )
        self.assertEqual(pattern["tactic_candidate"], "誘空吸籌")
        self.assertIn(pattern["direction_bias"], {"watch", "constructive"})
        self.assertIn("不產生買賣命令", pattern["guardrail"])

    def test_day_night_variance_flags_night_up_cash_failed(self):
        data = pd.DataFrame(
            [
                {"date": "2026-08-26", "open": 45800, "high": 46200, "low": 45600, "close": 46000, "volume": 1},
                {"date": "2026-08-27", "open": 46300, "high": 46400, "low": 45700, "close": 45900, "volume": 2},
            ]
        )
        data["date"] = pd.to_datetime(data["date"])
        data = add_features(data)
        pattern = analyze_day_night_variance_pattern(
            data,
            "2026-08-27",
            "2026-08-27",
            {
                "tx_night_spread_per": 0.007,
                "tx_night_close": 46388,
                "tx_night_high": 46450,
                "tx_night_low": 46036,
                "night_path": {"status": "bullish_but_unconfirmed"},
            },
            {"code": "cash_failed_after_night_up"},
            {"tactic_candidate": "高檔換手"},
            {"reset_active": False},
            {"roles": {"background_gua": {"code": "DUI"}, "state_gua": {"code": "ZHEN"}}},
        )
        self.assertEqual(pattern["relation_code"], "night_up_cash_failed")
        self.assertLess(pattern["score"], 0)
        self.assertTrue(any("獲利了結" in item or "高檔" in item for item in pattern["cause_candidates"]))
        self.assertTrue(any("假突破" in item or "高檔" in item for item in pattern["hidden_cause_candidates"]))
        self.assertIn("不產生買賣命令", pattern["guardrail"])

    def test_day_night_variance_marks_premarket_as_pending(self):
        data = pd.DataFrame(
            [
                {"date": "2026-08-26", "open": 45500, "high": 46100, "low": 45400, "close": 45900, "volume": 1},
                {"date": "2026-08-27", "open": 46000, "high": 46150, "low": 45800, "close": 45975, "volume": 2},
            ]
        )
        data["date"] = pd.to_datetime(data["date"])
        data = add_features(data)
        pattern = analyze_day_night_variance_pattern(
            data,
            "2026-08-28",
            "2026-08-27",
            {
                "tx_night_spread_per": 0.007,
                "tx_night_close": 46388,
                "tx_night_low": 46036,
                "night_path": {"status": "bullish_but_unconfirmed"},
            },
            {"code": "premarket_only"},
            {"tactic_candidate": "高檔換手"},
            {"reset_active": True, "label": "外部利多重置", "code": "bullish_external_reset"},
            {"roles": {"background_gua": {"code": "DUI"}, "state_gua": {"code": "ZHEN"}}},
        )
        self.assertEqual(pattern["relation_code"], "pending_day_validation")
        self.assertEqual(pattern["day_validation_status"], "pending_cash_session")
        self.assertIn("等待日盤驗真假", pattern["summary"])

    def test_weather_satellite_forecast_waits_for_cash_ground_truth_and_repairs_errors(self):
        satellite = build_weather_satellite_forecast_model(
            {
                "integrated_summary": {"bias": "bullish", "confidence": "medium"},
                "premarket": {"is_premarket": True, "pressure": "偏多", "tx_night_spread_per": 0.007},
                "intraday_tactical_monitor": {
                    "label": "盤前劇本待驗證",
                    "defense_levels": [45900, 46050],
                    "reclaim_levels": [46150, 46400],
                },
                "day_night_variance_pattern": {
                    "relation_code": "pending_day_validation",
                    "relation_label": "夜盤已出、日盤待驗",
                    "variance_label": "日盤尚未完成，變異待驗",
                    "next_watch": "等待日盤驗真假。",
                },
                "market_health": {"health_value": {"label": "健康脈動", "controllable_risk": True}},
                "crash_monitor": {"alert_code": "none", "health_score": 88, "risk_value": 12, "confirmation": {"summary": "未成立"}},
                "market_mode_switch": {"crisis_early": False, "headline": "高檔換手"},
                "bagua_lifecycle": {"roles": {"summary": "背景卦乾；狀態卦震。"}},
                "external_event_reset_monitor": {"label": "外部利多重置"},
                "capital_flow": {"has_factor_values": False},
                "production_policy": {"main_multi_day_direction": {"enabled": False}},
            }
        )
        self.assertEqual(satellite["next_step_code"], "radar_pending_ground_truth")
        self.assertIn("地面驗證", satellite["headline"])
        self.assertIn("不得回填舊預測", satellite["error_repair_policy"])
        self.assertTrue(any("正式訊號仍停用" in item for item in satellite["confidence_caps"]))
        self.assertTrue(any("假突破" in item for item in satellite["latent_disease_triggers"]["watch_triggers"]))
        root = satellite["root_cause_decomposition"]
        self.assertEqual(root["headline"], "逐層鑑別診斷")
        self.assertIn("表層症狀", root["method"])
        self.assertTrue(any(item["exclude_condition"] for item in root["items"]))

    def test_weather_satellite_triggers_latent_disease_from_external_pressure(self):
        satellite = build_weather_satellite_forecast_model(
            {
                "integrated_summary": {"bias": "bearish", "confidence": "medium"},
                "premarket": {
                    "is_premarket": True,
                    "pressure": "偏空",
                    "sox_return_1d": -0.031,
                    "tsm_adr_return_1d": -0.018,
                    "nasdaq_return_1d": -0.017,
                    "vix_return_1d": 0.11,
                    "tx_night_spread_per": -0.008,
                    "tx_night_low": 45200,
                },
                "intraday_tactical_monitor": {"label": "盤前劇本待驗證", "defense_levels": [45200], "reclaim_levels": [46000]},
                "day_night_variance_pattern": {
                    "relation_code": "pending_day_validation",
                    "relation_label": "夜盤已出、日盤待驗",
                    "variance_label": "日盤尚未完成，變異待驗",
                    "next_watch": "等待日盤驗真假。",
                    "features": {},
                },
                "market_health": {"health_value": {"label": "風險升溫", "controllable_risk": False}},
                "crash_monitor": {"alert_code": "none", "health_score": 42, "risk_value": 58, "confirmation": {"summary": "未成立"}},
                "market_mode_switch": {"crisis_early": True, "headline": "危機初期跡象"},
                "bagua_lifecycle": {"roles": {"summary": "背景卦乾；狀態卦坎。"}},
                "external_event_reset_monitor": {"reset_active": True, "label": "外部利空重置"},
                "capital_flow": {"has_factor_values": False},
                "production_policy": {"main_multi_day_direction": {"enabled": False}},
            }
        )
        active = " ".join(satellite["latent_disease_triggers"]["active_triggers"])
        self.assertIn("費半急跌", active)
        self.assertIn("VIX急升", active)
        self.assertIn("夜盤明顯偏空", active)

    def test_intraday_tactical_monitor_flags_night_low_break_reclaim_washout(self):
        data = pd.DataFrame(
            [
                {"date": "2026-08-21", "open": 44923.34, "high": 45254.84, "low": 44583.87, "close": 45224.29},
                {"date": "2026-08-24", "open": 45240.30, "high": 45362.30, "low": 44761.88, "close": 44762.32},
            ]
        )
        data["date"] = pd.to_datetime(data["date"])
        data = add_features(data)
        monitor = analyze_intraday_tactical_monitor(
            "2026-08-25",
            data,
            "2026-08-24",
            {
                "is_premarket": True,
                "night_signal_date": "2026-08-25",
                "tx_night_spread_per": -0.0046,
                "tx_night_close": 44532,
                "tx_night_low": 44270,
            },
            {
                "enabled": True,
                "source": "test official close",
                "date": "2026-08-25",
                "price": 45169.46,
                "open": 44728.36,
                "high": 45169.46,
                "low": 44210.31,
                "message": "測試今日完整日線",
            },
        )
        self.assertEqual(monitor["code"], "night_down_break_night_low_reclaim_washout")
        self.assertTrue(monitor["broke_night_low"])
        self.assertTrue(monitor["reclaimed_night_close"])
        self.assertTrue(monitor["reclaimed_psych_45000"])
        self.assertGreaterEqual(monitor["close_position"], 0.70)

    def test_intraday_tactical_monitor_uses_strict_prior_day_for_same_day_official_row(self):
        data = pd.DataFrame(
            [
                {"date": "2026-08-17", "open": 45800, "high": 46200, "low": 45700, "close": 45857},
                {"date": "2026-08-18", "open": 45922, "high": 46064, "low": 45225, "close": 45309},
            ]
        )
        data["date"] = pd.to_datetime(data["date"])
        data = add_features(data)
        monitor = analyze_intraday_tactical_monitor(
            "2026-08-18", data, "2026-08-18",
            {"night_signal_date": "2026-08-18", "tx_night_spread_per": -0.0012,
             "tx_night_close": 45811, "tx_night_low": 45747},
            {"enabled": True, "source": "test", "date": "2026-08-18", "price": 45309,
             "open": 45922, "high": 46064, "low": 45225, "message": "close snapshot"},
        )
        self.assertEqual(monitor["last_low"], 45700)
        self.assertEqual(monitor["last_close"], 45857)
        self.assertTrue(monitor["broke_last_low"])
        self.assertFalse(monitor["reclaimed_last_close"])
        self.assertEqual(monitor["code"], "night_down_cash_break_confirming")

    def test_intraday_tactical_monitor_break_answers_are_not_inverted(self):
        data = pd.DataFrame(
            [
                {"date": "2026-09-08", "open": 47335.24, "high": 47578.24, "low": 47023.73, "close": 47105.78},
            ]
        )
        data["date"] = pd.to_datetime(data["date"])
        data = add_features(data)
        monitor = analyze_intraday_tactical_monitor(
            "2026-09-09",
            data,
            "2026-09-08",
            {
                "night_signal_date": "2026-09-09",
                "tx_night_spread_per": 0.0021,
                "tx_night_close": 47055,
                "tx_night_low": 46561,
            },
            {
                "enabled": True,
                "source": "test official close",
                "date": "2026-09-09",
                "price": 47183.36,
                "open": 47142.74,
                "high": 47548.26,
                "low": 47060.78,
                "message": "official close",
            },
        )
        answers = {item["name"]: item.get("answer") for item in monitor["checks"]}
        self.assertFalse(monitor["broke_last_low"])
        self.assertFalse(monitor["broke_night_low"])
        self.assertEqual(answers["是否破昨低"], "否")
        self.assertEqual(answers["是否破夜盤低"], "否")
        self.assertEqual(answers["是否站回昨收"], "是")

    def test_night_path_marks_bearish_close_near_low_without_cash_close_signal(self):
        path = analyze_night_path(
            pd.Series({"tx_night_open": 45137, "tx_night_high": 45208, "tx_night_low": 44424,
                       "tx_night_close": 44527, "tx_night_volume": 28126,
                       "tx_night_return": -0.0135144, "tx_night_range": 0.0176481,
                       "tx_night_spread_per": -0.0124}),
            45308.68,
        )
        self.assertEqual(path["status"], "bearish_extension_close_near_low")
        self.assertLess(path["close_position"], 0.25)
        self.assertTrue(path["material_pressure"])
        self.assertIsNone(path["formal_cash_close_signal"])

    def test_night_close_psychology_explains_bearish_extension(self):
        path = analyze_night_path(
            pd.Series({"tx_night_open": 45137, "tx_night_high": 45208, "tx_night_low": 44424,
                       "tx_night_close": 44527, "tx_night_volume": 28126,
                       "tx_night_return": -0.0135144, "tx_night_range": 0.0176481,
                       "tx_night_spread_per": -0.0124}),
            45308.68,
        )
        psychology = analyze_night_close_psychology(path, {}, {"is_conflict": False}, -2)
        self.assertEqual(psychology["label"], "空方心理延伸")
        self.assertEqual(psychology["bias"], "bearish_psychology")
        self.assertIn("追跌壓測", psychology["tactic"])
        self.assertTrue(psychology["next_validation"])
        self.assertIn("不產生買賣命令", psychology["guardrail"])

    def test_night_close_psychology_marks_recovered_down_night_as_pending_bear_trap(self):
        path = analyze_night_path(
            pd.Series({"tx_night_open": 47000, "tx_night_high": 47020, "tx_night_low": 46600,
                       "tx_night_close": 46900, "tx_night_volume": 10000,
                       "tx_night_return": -0.0021, "tx_night_range": 0.0090,
                       "tx_night_spread_per": -0.0040}),
            47105,
        )
        psychology = analyze_night_close_psychology(path, {}, {"is_conflict": True}, 1)
        self.assertEqual(psychology["label"], "誘空回補待驗")
        self.assertEqual(psychology["bias"], "mixed_psychology")
        self.assertIn("壓低測承接", psychology["tactic"])
        self.assertTrue(any("收回" in item for item in psychology["evidence"]))

    def test_situation_psychology_context_flags_us_labor_day_repricing(self):
        context = analyze_situation_psychology_context(
            "2026-09-08",
            {"sox_return_1d": 0.0, "tsm_adr_return_1d": 0.0, "tx_night_spread_per": 0.0},
            {
                "available": True,
                "status": "connected",
                "risk_score": 4,
                "tailwind_score": 3,
                "pending_events": [{"title": "CPI", "risk_score": 3}],
                "events": [],
            },
        )
        self.assertEqual(us_market_holiday_name(pd.Timestamp("2026-09-07").date()), "美國勞工節休市")
        self.assertIn("us_market_holiday_repricing", context["triggers"])
        self.assertIn("pending_macro_event_gate", context["triggers"])
        self.assertIn("不產生投資命令", context["guardrail"])

    def test_programmed_pressure_pattern_detects_high_level_pressure_fingerprint(self):
        data = pd.DataFrame(
            [
                {"date": "2026-08-21", "open": 44923, "high": 45255, "low": 44584, "close": 45224, "volume": 1},
                {"date": "2026-08-24", "open": 45240, "high": 45362, "low": 44762, "close": 44762, "volume": 1},
            ]
        )
        data["date"] = pd.to_datetime(data["date"])
        night_path = ROOT / "data" / "processed" / "taiwan_futures_night.csv"
        pattern = analyze_programmed_pressure_pattern(
            add_features(data),
            "2026-08-24",
            44762.32,
            night_path,
            ROOT / "data" / "processed" / "factors",
            start_date="2026-08-21",
        )
        self.assertTrue(pattern["available"])
        self.assertGreaterEqual(pattern["current_score"], 4)
        self.assertIn(pattern["code"], {"single_day_pressure", "recent_pressure_cluster", "programmed_pressure_sequence"})
        self.assertIn("公開資料只能辨識行為指紋", pattern["guardrail"])
        report = render_programmed_pressure_pattern(pattern)
        self.assertIn("連續壓低換手行為指紋研究", report)
        self.assertIn("次日驗證", report)

    def test_psychological_warfare_outputs_cause_effect_chain(self):
        pattern = analyze_psychological_warfare_pattern(
            {
                "roles": {
                    "background_gua": {"gua": "乾", "code": "QIAN"},
                    "state_gua": {"gua": "震", "code": "ZHEN"},
                },
                "primary": {"gua": "震", "code": "ZHEN"},
            },
            {"tactic_candidate": "誘空吸籌", "crowd_state": "恐慌轉承接"},
            {"relation_code": "night_down_cash_reversal"},
            {},
            {"code": "regulation_reclaim_pulse"},
            {"risk_value": 20, "health_score": 80},
            {"label": "修復觀察"},
        )
        chain = pattern["cause_effect_chain"]
        self.assertEqual(len(chain), 4)
        self.assertIn("前因", chain[0])
        self.assertIn("心理", chain[1])
        self.assertIn("機制", chain[2])
        self.assertIn("驗證", chain[3])
        self.assertIn("action_policy", pattern)
        self.assertGreaterEqual(len(pattern["semantic_quantification"]), 4)
        self.assertIn("text", pattern["semantic_quantification"][0])
        self.assertIn("score", pattern["semantic_quantification"][0])
        self.assertIn("meaning", pattern["semantic_quantification"][0])
        self.assertIn("不得宣稱單一主體操控", pattern["guardrail"])

    def test_crash_monitor_uses_intraday_break_for_fail_safe_alert(self):
        data = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
        data["date"] = pd.to_datetime(data["date"])
        data = data[data["date"] <= pd.Timestamp("2026-07-28")].copy()
        data = add_features(data)
        candle = analyze_candlestick_pattern(data, "2026-07-28")
        bagua = analyze_bagua_lifecycle(data, "2026-07-28", candle, {"type": "failed_washout"})
        cycle = analyze_tradeable_rally_segments(data, "2026-07-28")
        technical = analyze_technical_phase(data, "2026-07-28", bagua, candle, cycle)
        monitor = analyze_crash_monitor(
            data,
            "2026-07-28",
            technical,
            bagua,
            {"is_premarket": False},
            {"has_factor_values": False},
            {"enabled": True, "source": "test", "date": "2026-07-29", "price": 39700.0, "low": 39700.0},
        )
        self.assertEqual(monitor["status_code"], "bottom_failed_watch")
        self.assertLess(monitor["effective_low"], monitor["levels"]["normal_bottom_low"])
        self.assertEqual(monitor["confirmation"]["stage_code"], "normal_bottom_intraday_fail")
        self.assertEqual(monitor["capital_policy"]["level"], "高")
        self.assertEqual(
            monitor["levels"]["anchor_rule"],
            "合理底數以最近一年波段高點鎖定；未創新高前不因每日新低下修。",
        )

    def test_crash_monitor_flags_intraday_crash_line_break_separately(self):
        data = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
        data["date"] = pd.to_datetime(data["date"])
        data = data[data["date"] <= pd.Timestamp("2026-07-28")].copy()
        data = add_features(data)
        candle = analyze_candlestick_pattern(data, "2026-07-28")
        bagua = analyze_bagua_lifecycle(data, "2026-07-28", candle, {"type": "failed_washout"})
        cycle = analyze_tradeable_rally_segments(data, "2026-07-28")
        technical = analyze_technical_phase(data, "2026-07-28", bagua, candle, cycle)
        monitor = analyze_crash_monitor(
            data,
            "2026-07-28",
            technical,
            bagua,
            {"is_premarket": False},
            {"has_factor_values": False},
            {"enabled": True, "source": "test", "date": "2026-07-29", "price": 38400.0, "low": 38400.0},
        )
        self.assertEqual(monitor["status_code"], "crash_warning")
        self.assertEqual(monitor["confirmation"]["stage_code"], "crash_intraday_break")
        self.assertEqual(monitor["capital_policy"]["level"], "極高")

    def test_false_crash_filter_marks_intraday_normal_break_reclaim_as_candidate(self):
        data = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
        data["date"] = pd.to_datetime(data["date"])
        data = data[data["date"] <= pd.Timestamp("2026-07-28")].copy()
        data = add_features(data)
        candle = analyze_candlestick_pattern(data, "2026-07-28")
        bagua = analyze_bagua_lifecycle(data, "2026-07-28", candle, {"type": "failed_washout"})
        cycle = analyze_tradeable_rally_segments(data, "2026-07-28")
        technical = analyze_technical_phase(data, "2026-07-28", bagua, candle, cycle)
        monitor = analyze_crash_monitor(
            data,
            "2026-07-28",
            technical,
            bagua,
            {"is_premarket": False},
            {"has_factor_values": False},
            {"enabled": True, "source": "test", "date": "2026-07-29", "price": 39950.0, "low": 39600.0},
        )
        self.assertEqual(monitor["confirmation"]["stage_code"], "normal_bottom_intraday_fail")
        self.assertTrue(monitor["false_crash_filter"]["candidate"])
        self.assertEqual(monitor["false_crash_filter"]["label"], "假跌破候選")

    def test_false_crash_filter_downgrades_intraday_crash_break_without_close_confirmation(self):
        data = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
        data["date"] = pd.to_datetime(data["date"])
        data = data[data["date"] <= pd.Timestamp("2026-07-28")].copy()
        data = add_features(data)
        candle = analyze_candlestick_pattern(data, "2026-07-28")
        bagua = analyze_bagua_lifecycle(data, "2026-07-28", candle, {"type": "failed_washout"})
        cycle = analyze_tradeable_rally_segments(data, "2026-07-28")
        technical = analyze_technical_phase(data, "2026-07-28", bagua, candle, cycle)
        monitor = analyze_crash_monitor(
            data,
            "2026-07-28",
            technical,
            bagua,
            {"is_premarket": False},
            {"has_factor_values": False},
            {"enabled": True, "source": "test", "date": "2026-07-29", "price": 38900.0, "low": 38400.0},
        )
        self.assertEqual(monitor["confirmation"]["stage_code"], "crash_intraday_break")
        self.assertTrue(monitor["false_crash_filter"]["candidate"])
        self.assertEqual(monitor["false_crash_filter"]["label"], "假崩盤候選")
        self.assertEqual(monitor["alert_code"], "intraday_unconfirmed")

    def test_skipped_online_update_is_not_reported_as_failure(self):
        self.assertEqual(
            status({"attempted": False, "success": False}),
            "略過（使用既有快取）",
        )
        self.assertEqual(
            status({"attempted": True, "success": False}),
            "失敗",
        )

    def test_error_review_classifies_misses_without_auto_tuning(self):
        review = build_error_review(
            {
                "history_file": "reports/forecast_history.json",
                "matured_checks": {
                    "available": True,
                    "hit_rate": 0.4,
                    "rows": [
                        {
                            "forecast_date": "2026-07-28",
                            "signal_date": "2026-07-28",
                            "target_date": "2026-07-29",
                            "horizon_days": 1,
                            "predicted_direction": "up",
                            "actual_direction": "down",
                            "actual_return": -0.02,
                            "is_hit": False,
                            "bagua_tracking": {"primary_code": "KAN", "primary_gua": "坎"},
                            "candlestick_tracking": {"type": "long_bear", "label": "長黑 K"},
                            "washout_tracking": {"type": "failed_washout", "label": "失敗洗盤"},
                            "tradeable_cycle_tracking": {"position_code": "post_rally_pullback"},
                        }
                    ],
                },
            }
        )
        self.assertEqual(review["miss_count"], 1)
        self.assertFalse(review["governance"]["automatic_parameter_tuning"])
        self.assertEqual(review["cases"][0]["primary_error_code"], "bagua_phase_mismatch")
        validation = review["next_day_validation"]
        self.assertTrue(validation["enabled"])
        self.assertIn("昨日誤差類型", validation["headline"])
        self.assertTrue(validation["tasks"])
        self.assertFalse(review["governance"]["automatic_parameter_tuning"])

    def test_error_review_report_renders_next_day_validation_tasks(self):
        review = build_error_review(
            {
                "history_file": "reports/forecast_history.json",
                "matured_checks": {
                    "available": True,
                    "hit_rate": 0.4,
                    "rows": [
                        {
                            "forecast_date": "2026-08-11",
                            "signal_date": "2026-08-10",
                            "target_date": "2026-08-11",
                            "horizon_days": 1,
                            "predicted_direction": "up",
                            "actual_direction": "sideways",
                            "actual_return": 0.0043,
                            "is_hit": False,
                            "bagua_tracking": {"primary_code": "QIAN", "primary_gua": "乾"},
                            "candlestick_tracking": {"type": "normal", "label": "一般 K 線"},
                            "washout_tracking": {"type": "none", "label": "非洗盤"},
                            "tradeable_cycle_tracking": {"position_code": "rally"},
                        }
                    ],
                },
            }
        )
        report = render_error_review(review)
        self.assertIn("次日強制驗證", report)
        self.assertIn("盤整誤判驗證", report)
        self.assertIn("不自動調參", report)

    def test_gap_up_close_near_low_is_not_classified_as_normal(self):
        row = pd.Series(
            {
                "k_body_pct": -0.00635,
                "k_abs_body_pct": 0.00635,
                "k_upper_shadow_pct": 0.00648,
                "k_lower_shadow_pct": 0.00028,
                "k_body_to_range": 0.4845,
                "k_close_position": 0.021,
                "open_gap_pct": 0.00179,
                "close_return_pct": -0.00457,
            }
        )
        pattern = classify_candlestick(row)
        self.assertEqual(pattern["type"], "gap_up_close_near_low")

    def test_gap_up_close_near_low_scenario_flags_cash_session_reversal(self):
        row = pd.Series(
            {
                "open_gap_pct": 0.00179,
                "close_return_pct": -0.00457,
                "intraday_low_pct": -0.00663,
                "close_recovery_ratio": 0.021,
                "intraday_high_pct": 0.00828,
                "lifecycle_stage": "early_bull",
            }
        )
        scenario = market_scenario(row, "neutral", 0, 0, 0)
        self.assertIn("現貨反證", scenario["name"])


if __name__ == "__main__":
    unittest.main()
