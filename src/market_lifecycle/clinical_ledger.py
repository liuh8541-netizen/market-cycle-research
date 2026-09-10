"""Detailed, outcome-aware market clinical ledger.

Each record separates pre-open evidence, descriptive diagnosis, prior-case
prognosis, safety-scoped prescription, and the later observed outcome.  A
historical case's prognosis is calculated only from strictly earlier resolved
cases in the same psychology-state/direction cohort.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from market_lifecycle.path_risk import classify_direction_strength, classify_session_path


TAIPEI = ZoneInfo("Asia/Taipei")
SCHEMA_VERSION = "2.1"


def build_complete_ledger(replay: pd.DataFrame, current_payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    resolved_history: list[dict[str, Any]] = []
    ordered = replay.sort_values("signal_date").reset_index(drop=True)
    for _, row in ordered.iterrows():
        record = build_resolved_record(row, resolved_history)
        records.append(record)
        resolved_history.append(record)

    if current_payload:
        current_date = str(current_payload.get("input", {}).get("date") or "")
        if current_date and not any(item["identity"]["decision_date"] == current_date for item in records):
            records.append(build_pending_record(current_payload, resolved_history))
    return records


def build_resolved_record(row: pd.Series, prior_records: list[dict[str, Any]]) -> dict[str, Any]:
    decision_date = str(row["signal_date"])
    state = str(row["state"])
    direction = str(row["direction"])
    regime = market_regime_from_row(row)
    phenotype = phenotype_from_row(row, regime)
    prognosis = prior_case_prognosis(prior_records, state, direction, phenotype, regime["code"])
    evidence = json_list(row.get("psychology_evidence_json"))
    expected_paths = json_list(row.get("expected_paths_json"))
    record = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "market": "TWII",
            "timeframe": "daily",
            "decision_date": decision_date,
            "record_phase": "resolved_after_close",
            "prior_cash_date": str(row.get("prior_cash_date")),
            "external_cutoff_date": str(row.get("external_date")),
            "night_signal_date": decision_date,
        },
        "data_governance": {
            "preopen_evidence_only": True,
            "strict_prior_cash": str(row.get("prior_cash_date")) < decision_date,
            "strict_prior_external": str(row.get("external_date")) < decision_date,
            "night_session_same_signal_date": True,
            "outcome_separated_from_diagnosis": True,
            "data_completeness": completeness(row),
            "unavailable_domains": unavailable_domains(row),
        },
        "chief_complaint": chief_complaint(state, direction, row.get("night_status")),
        "vital_signs": vitals_from_row(row),
        "symptoms": symptoms_from_row(row, evidence),
        "causal_chain": causal_chain_from_row(row, expected_paths),
        "diagnosis": {
            **diagnosis_from_row(row), "market_regime": regime,
            "positioning_diagnosis": positioning_diagnosis_from_row(row, direction),
            "behavior_phenotype": phenotype,
        },
        "differential_diagnosis": expected_paths,
        "prognosis_from_prior_cases": prognosis,
        "prescription": safe_prescription(direction, state, row),
        "contraindications": contraindications(row),
        "outcome": outcome_from_row(row),
        "follow_up": follow_up_from_row(row),
    }
    record["case_id"] = case_id(record)
    record["evidence_sha256"] = fingerprint(preoutcome_evidence(record))
    return record


def build_pending_record(payload: dict[str, Any], prior_records: list[dict[str, Any]]) -> dict[str, Any]:
    pre = payload.get("premarket", {})
    psych = payload.get("psychology_state", {})
    practical = payload.get("practical_cause_arbitration", {})
    candle = payload.get("candlestick_pattern", {})
    washout = payload.get("washout_pattern", {})
    health = payload.get("market_health", {})
    technical = payload.get("technical_phase", {})
    capital = payload.get("capital_flow", {})
    auxiliary = payload.get("_clinical_aux", {})
    decision_date = str(payload.get("input", {}).get("date"))
    state = str(psych.get("state", "doubt"))
    direction = str(psych.get("direction", "mixed"))
    night = pre.get("night_path", {})
    synthetic = pd.Series({
        "signal_date": decision_date,
        "prior_cash_date": pre.get("spot_latest_date"),
        "external_date": pre.get("external_date"),
        "state": state,
        "state_label": psych.get("state_label"),
        "direction": direction,
        "bearish_urgency_score": psych.get("bearish_urgency_score"),
        "bullish_urgency_score": psych.get("bullish_urgency_score"),
        "net_urgency_score": psych.get("net_urgency_score"),
        "calibrated_direction_score": psych.get("calibrated_direction_score"),
        "direction_weight_version": psych.get("direction_weight_version"),
        "night_status": night.get("status"),
        "night_close_position": night.get("close_position"),
        "night_recovery_from_low": night.get("recovery_from_low"),
        "night_close_vs_prior_cash": night.get("close_vs_prior_cash"),
        "night_material_pressure": night.get("material_pressure"),
        "prior_candle_type": candle.get("type"),
        "prior_candle_label": candle.get("label"),
        "prior_washout_type": washout.get("type"),
        "prior_washout_label": washout.get("label"),
        "external_score": pre.get("external_score"),
        "exogenous_reset_watch": psych.get("exogenous_reset_watch"),
        "crowding_status": psych.get("crowding_status"),
        "position_confirmation": psych.get("position_confirmation"),
        "night_open": pre.get("tx_night_open"), "night_high": pre.get("tx_night_high"),
        "night_low": pre.get("tx_night_low"), "night_close": pre.get("tx_night_close"),
        "night_return": pre.get("tx_night_return"), "night_range": night.get("range"),
        "night_spread_per": pre.get("tx_night_spread_per"), "night_volume": night.get("volume"),
        "prior_cash_open": technical.get("levels", {}).get("open"),
        "prior_cash_high": technical.get("levels", {}).get("high"),
        "prior_cash_low": technical.get("levels", {}).get("low"),
        "prior_cash_close": pre.get("spot_latest_close"),
        "nasdaq_return_1d": pre.get("nasdaq_return_1d"), "sox_return_1d": pre.get("sox_return_1d"),
        "sp500_return_1d": pre.get("sp500_return_1d"), "tsm_adr_return_1d": pre.get("tsm_adr_return_1d"),
        "vix_return_1d": pre.get("vix_return_1d"),
        "treasury_5y_close": pre.get("treasury_5y_close"), "treasury_10y_close": pre.get("treasury_10y_close"),
        "treasury_30y_close": pre.get("treasury_30y_close"),
        "treasury_5y_return_1d": pre.get("treasury_5y_return_1d"),
        "treasury_10y_return_1d": pre.get("treasury_10y_return_1d"),
        "treasury_30y_return_1d": pre.get("treasury_30y_return_1d"),
        "prior_ma5": technical.get("levels", {}).get("ma5"),
        "prior_ma10": technical.get("levels", {}).get("ma10"),
        "prior_ma20": technical.get("levels", {}).get("ma20"),
        "prior_ma60": technical.get("levels", {}).get("ma60"),
        "prior_close_vs_ma5": ratio_value(pre.get("spot_latest_close"), technical.get("levels", {}).get("ma5")),
        "prior_close_vs_ma20": ratio_value(pre.get("spot_latest_close"), technical.get("levels", {}).get("ma20")),
        "prior_close_vs_ma60": ratio_value(pre.get("spot_latest_close"), technical.get("levels", {}).get("ma60")),
        "prior_drawdown_60": technical.get("levels", {}).get("drawdown_from_swing_high"),
        "prior_range_20": technical.get("levels", {}).get("range_20d"),
        "capital_flow_available": capital.get("has_factor_values"),
        "lifecycle_diagnosis_available": bool(health.get("diagnosis")),
        **auxiliary,
    })
    symptoms = symptoms_from_row(synthetic, psych.get("evidence", []))
    symptoms.extend(health.get("symptoms", []))
    symptoms.extend(treatment_symptoms(practical))
    regime = market_regime_from_row(synthetic)
    phenotype = phenotype_from_row(synthetic, regime)
    record = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "market": "TWII", "timeframe": "daily", "decision_date": decision_date,
            "record_phase": "pending_cash_outcome", "prior_cash_date": pre.get("spot_latest_date"),
            "external_cutoff_date": pre.get("external_date"), "night_signal_date": pre.get("night_signal_date"),
        },
        "data_governance": {
            "preopen_evidence_only": True,
            "strict_prior_cash": str(pre.get("spot_latest_date")) < decision_date,
            "strict_prior_external": str(pre.get("external_date")) < decision_date,
            "night_session_same_signal_date": str(pre.get("night_signal_date")) == decision_date,
            "outcome_separated_from_diagnosis": True,
            "data_completeness": completeness(synthetic),
            "unavailable_domains": unavailable_domains(synthetic),
            "freshness_status": payload.get("data_freshness", {}).get("overall_status"),
        },
        "chief_complaint": chief_complaint(state, direction, night.get("status")),
        "vital_signs": vitals_from_row(synthetic),
        "symptoms": unique_symptoms(symptoms),
        "causal_chain": causal_chain_from_row(synthetic, psych.get("next_paths", [])),
        "diagnosis": {
            **diagnosis_from_row(synthetic),
            "market_regime": regime,
            "positioning_diagnosis": positioning_diagnosis_from_row(synthetic, direction),
            "behavior_phenotype": phenotype,
            "market_health": health.get("diagnosis", {}),
            "technical_phase": {
                "status_code": technical.get("status_code"), "label": technical.get("label"),
                "bias": technical.get("bias"), "levels": technical.get("levels", {}),
            },
            "capital_flow": {
                "data_status": capital.get("data_status"), "chip_score": capital.get("chip_score"),
                "derivative_score": capital.get("derivative_score"), "non_price_score": capital.get("non_price_score"),
            },
        },
        "differential_diagnosis": psych.get("next_paths", []),
        "prognosis_from_prior_cases": prior_case_prognosis(
            prior_records, state, direction, phenotype, regime["code"]
        ),
        "treatment_course": treatment_course_from_payload(practical),
        "prescription": {
            **safe_prescription(direction, state, synthetic),
            "market_health_prescription": health.get("prescription", {}),
        },
        "contraindications": contraindications(synthetic),
        "outcome": {"status": "pending", "reason": "台股現貨尚無正式收盤資料。"},
        "follow_up": {
            "status": "await_cash_session",
            "required_observations": ["正式開盤", "是否跌破夜盤低點", "是否收回夜盤收盤", "正式收盤與成交量"],
        },
    }
    record["case_id"] = case_id(record)
    record["evidence_sha256"] = fingerprint(preoutcome_evidence(record))
    return record


def treatment_course_from_payload(practical: dict[str, Any]) -> dict[str, Any]:
    if not practical:
        return {
            "available": False,
            "reason": "尚未接入實務主因仲裁。",
        }
    return {
        "available": True,
        "framework": practical.get("framework"),
        "disease_gene": {
            "score": practical.get("internal_structure_score"),
            "causes": practical.get("internal_causes", []),
            "model": practical.get("gene_trigger_model"),
        },
        "trigger_condition": {
            "score": practical.get("external_trigger_score"),
            "triggers": practical.get("external_triggers", []),
        },
        "treatment": {
            "phase": practical.get("treatment_phase"),
            "stage": practical.get("treatment_stage"),
            "policy": practical.get("treatment_policy"),
            "medicine_type": practical.get("medicine_type"),
            "medicine_effect": practical.get("medicine_effect"),
            "correct_medicine": practical.get("correct_medicine"),
            "after_effect_risk": practical.get("after_effect_risk"),
        },
        "fact_changing_medicine": {
            "bias": practical.get("immediate_medicine_bias"),
            "score": practical.get("immediate_medicine_score"),
            "label": practical.get("fact_changing_medicine", {}).get("label"),
            "effect": practical.get("fact_changing_medicine", {}).get("effect"),
            "medicines": practical.get("fact_changing_medicines", []),
            "rule": practical.get("fact_changing_medicine", {}).get("rule"),
        },
        "causality_pipeline": practical.get("causality_pipeline", []),
        "causality_rule": practical.get("causality_rule"),
        "next_episode_prior": next_episode_prior_from_treatment(practical),
        "confirmation": practical.get("confirmation", []),
        "exclusion": practical.get("exclusion", []),
        "guardrail": "療程記憶只作下期病灶先驗與監控權重，不產生投資命令。",
    }


def treatment_symptoms(practical: dict[str, Any]) -> list[dict[str, Any]]:
    if not practical:
        return []
    return [
        symptom("disease_gene_score", practical.get("internal_structure_score"), "treatment_course", "watch"),
        symptom("trigger_condition_score", practical.get("external_trigger_score"), "treatment_course", "watch"),
        symptom("treatment_stage", practical.get("treatment_stage"), "treatment_course", "watch"),
        symptom("after_effect_risk", practical.get("after_effect_risk"), "treatment_course", "watch"),
        symptom("immediate_medicine_score", practical.get("immediate_medicine_score"), "treatment_course", "watch"),
    ]


def next_episode_prior_from_treatment(practical: dict[str, Any]) -> str:
    phase = int(number(practical.get("treatment_phase")) or 0)
    after_effect = practical.get("after_effect_risk")
    correct = practical.get("correct_medicine")
    if phase >= 4 or after_effect == "高":
        return "下期同類病灶先以急症復發觀察，需先核對破線與連續收不回。"
    if phase >= 3 or after_effect == "中":
        return "下期同類病灶先檢查是否殘留陰霾；若同類觸發再現，提早升級為發作觀察。"
    if correct == "偏對症":
        return "上期治療偏對症；下期同類觸發先看是否被日盤快速吸收。"
    return "累積為一般病歷先驗；下期仍需由現貨、量能、族群與收盤驗證。"


def prior_case_prognosis(
    records: list[dict[str, Any]], state: str, direction: str,
    phenotype: str | None = None, regime_code: str | None = None,
    min_fine_cases: int = 30,
) -> dict[str, Any]:
    broad = [
        item for item in records
        if item.get("diagnosis", {}).get("psychology_state") == state
        and item.get("diagnosis", {}).get("direction") == direction
        and item.get("outcome", {}).get("status") == "resolved"
    ]
    fine = [
        item for item in broad
        if phenotype and item.get("diagnosis", {}).get("behavior_phenotype") == phenotype
    ]
    regime = [
        item for item in broad
        if regime_code and item.get("diagnosis", {}).get("market_regime", {}).get("code") == regime_code
    ]
    if len(fine) >= min_fine_cases:
        cohort, selected_tier = fine, "fine_phenotype"
    elif len(regime) >= min_fine_cases:
        cohort, selected_tier = regime, "market_regime_backoff"
    else:
        cohort, selected_tier = broad, "state_direction_backoff"
    if not broad:
        return {
            "status": "insufficient_prior_cases", "cohort": f"{state}|{direction}",
            "behavior_phenotype": phenotype, "exact_phenotype_prior_cases": 0,
            "market_regime": regime_code, "regime_prior_cases": 0,
            "selected_tier": "none", "minimum_fine_cases": min_fine_cases,
            "prior_cases": 0, "direction_hit_rate": None, "material_move_rate": None,
            "average_cash_close_return": None, "empirical_path_ranking": [],
            "fine_phenotype_statistics": cohort_statistics([]),
            "regime_statistics": cohort_statistics([]),
            "broad_cohort_statistics": cohort_statistics([]),
            "hierarchical_bayesian_estimate": hierarchical_bayesian_estimate([], []),
        }
    selected = cohort_statistics(cohort)
    return {
        "status": "descriptive_reference" if len(cohort) >= 30 else "small_sample",
        "cohort": f"{state}|{direction}",
        "behavior_phenotype": phenotype,
        "exact_phenotype_prior_cases": len(fine),
        "market_regime": regime_code,
        "regime_prior_cases": len(regime),
        "selected_tier": selected_tier,
        "minimum_fine_cases": min_fine_cases,
        "prior_cases": selected["cases"],
        "direction_hit_rate": selected["direction_hit_rate"],
        "material_move_rate": selected["material_move_rate"],
        "average_cash_close_return": selected["average_cash_close_return"],
        "empirical_path_ranking": selected["path_ranking"],
        "fine_phenotype_statistics": cohort_statistics(fine),
        "regime_statistics": cohort_statistics(regime),
        "broad_cohort_statistics": cohort_statistics(broad),
        "hierarchical_bayesian_estimate": hierarchical_bayesian_estimate(fine, broad),
        "guardrail": "僅使用決策日前同狀態、同方向已結案病例；屬描述性常模，不是正式交易機率。",
    }


def cohort_statistics(cohort: list[dict[str, Any]]) -> dict[str, Any]:
    if not cohort:
        return {
            "cases": 0, "direction_hit_rate": None, "material_move_rate": None,
            "average_cash_close_return": None, "path_ranking": [],
        }
    outcomes = [item["outcome"] for item in cohort]
    path_counts: dict[str, int] = {}
    for outcome in outcomes:
        path = str(outcome.get("realized_path", "unresolved"))
        path_counts[path] = path_counts.get(path, 0) + 1
    ranked = sorted(path_counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "cases": len(cohort),
        "direction_hit_rate": mean(outcome.get("direction_hit") for outcome in outcomes),
        "material_move_rate": mean(outcome.get("material_direction_confirmation") for outcome in outcomes),
        "average_cash_close_return": mean(outcome.get("cash_close_return") for outcome in outcomes),
        "path_ranking": [
            {"rank": index + 1, "path": path, "cases": count, "rate": count / len(cohort)}
            for index, (path, count) in enumerate(ranked)
        ],
    }


def phenotype_from_row(row: pd.Series, regime: dict[str, Any] | None = None) -> str:
    external_score = number(row.get("external_score"))
    if external_score is None:
        external_band = "external_unknown"
    elif external_score <= -3:
        external_band = "external_strong_bear"
    elif external_score < 0:
        external_band = "external_bear"
    elif external_score >= 3:
        external_band = "external_strong_bull"
    elif external_score > 0:
        external_band = "external_bull"
    else:
        external_band = "external_neutral"
    parts = [
        str(row.get("state") or "state_unknown"), str(row.get("direction") or "direction_unknown"),
        str(row.get("night_status") or "night_unknown"), external_band,
        str(row.get("prior_candle_type") or "candle_unknown"),
        str(row.get("prior_washout_type") or "washout_unknown"),
        "shock" if bool_value(row.get("exogenous_reset_watch")) else "no_shock",
        (regime or market_regime_from_row(row)).get("code", "regime_unknown"),
    ]
    return "|".join(parts)


def market_regime_from_row(row: pd.Series) -> dict[str, Any]:
    vs20 = number(row.get("prior_close_vs_ma20"))
    vs60 = number(row.get("prior_close_vs_ma60"))
    volatility = number(row.get("prior_realized_volatility_20"))
    drawdown = number(row.get("prior_drawdown_60"))
    breadth = number(row.get("price_advance_decline_breadth"))
    if vs20 is not None and vs60 is not None and vs20 > 0 and vs60 > 0:
        trend = "uptrend"
    elif vs20 is not None and vs60 is not None and vs20 < 0 and vs60 < 0:
        trend = "downtrend"
    else:
        trend = "mixed_trend"
    if volatility is None:
        volatility_regime = "vol_unknown"
    elif volatility >= 0.018:
        volatility_regime = "high_vol"
    elif volatility <= 0.008:
        volatility_regime = "low_vol"
    else:
        volatility_regime = "normal_vol"
    if drawdown is None:
        location = "location_unknown"
    elif drawdown <= -0.15:
        location = "deep_drawdown"
    elif drawdown >= -0.05:
        location = "near_high"
    else:
        location = "mid_drawdown"
    if breadth is None:
        breadth_regime = "breadth_unknown"
    elif breadth <= -0.30:
        breadth_regime = "weak_breadth"
    elif breadth >= 0.30:
        breadth_regime = "strong_breadth"
    else:
        breadth_regime = "mixed_breadth"
    return {
        "code": f"{trend}|{volatility_regime}|{location}|{breadth_regime}",
        "trend": trend, "volatility": volatility_regime,
        "location": location, "breadth": breadth_regime,
    }


def positioning_diagnosis_from_row(row: pd.Series, direction: str) -> dict[str, Any]:
    breadth = number(row.get("price_advance_decline_breadth"))
    breadth_age = number(row.get("breadth_age_days"))
    oi_change = number(row.get("tx_open_interest_change_1d"))
    oi_age = number(row.get("open_interest_age_days"))
    foreign_net = number(row.get("foreign_futures_net_oi"))
    institutional_age = number(row.get("institutional_position_age_days"))
    urgency = abs(number(row.get("net_urgency_score")) or 0)
    breadth_fresh = breadth is not None and breadth_age is not None and breadth_age <= 5
    oi_fresh = oi_change is not None and oi_age is not None and oi_age <= 3
    institutional_fresh = foreign_net is not None and institutional_age is not None and institutional_age <= 3
    breadth_aligned = bool(
        breadth_fresh and (
            (direction == "bearish" and breadth <= -0.30)
            or (direction == "bullish" and breadth >= 0.30)
        )
    )
    foreign_aligned = bool(
        institutional_fresh and (
            (direction == "bearish" and foreign_net < 0)
            or (direction == "bullish" and foreign_net > 0)
        )
    )
    oi_expanding = bool(oi_fresh and oi_change > 0)
    confirmation_count = sum([breadth_aligned, foreign_aligned, oi_expanding])
    confirmed = confirmation_count >= 2
    extreme_breadth = bool(
        breadth_fresh and (
            (direction == "bearish" and breadth <= -0.55)
            or (direction == "bullish" and breadth >= 0.55)
        )
    )
    if confirmed and extreme_breadth and urgency >= 5:
        state = "capitulation_or_euphoria_candidate"
    elif confirmed and urgency >= 5:
        state = "crowding_confirmed"
    elif confirmed:
        state = "position_alignment_confirmed"
    else:
        state = "unconfirmed"
    return {
        "state": state, "confirmed": confirmed, "confirmation_count": confirmation_count,
        "breadth_aligned": breadth_aligned, "foreign_position_aligned": foreign_aligned,
        "open_interest_expanding": oi_expanding, "extreme_breadth": extreme_breadth,
        "freshness": {
            "breadth": breadth_fresh, "open_interest": oi_fresh,
            "institutional_position": institutional_fresh,
        },
        "formal_direction_signal": None,
        "guardrail": "至少兩項新鮮部位/廣度證據才確認；極端宣洩仍只列候選。",
    }


def hierarchical_bayesian_estimate(
    fine: list[dict[str, Any]], broad: list[dict[str, Any]], prior_strength: float = 20.0,
) -> dict[str, Any]:
    broad_stats = cohort_statistics(broad)
    fine_stats = cohort_statistics(fine)
    if not broad:
        return {"status": "unavailable", "prior_strength": prior_strength}
    n = len(fine)
    broad_direction = broad_stats["direction_hit_rate"] or 0.5
    broad_material = broad_stats["material_move_rate"] or 0.0
    fine_direction_hits = sum(bool(item["outcome"].get("direction_hit")) for item in fine)
    fine_material_hits = sum(bool(item["outcome"].get("material_direction_confirmation")) for item in fine)
    path_names = sorted({
        str(item["outcome"].get("realized_path", "unresolved")) for item in broad
    })
    broad_path = {
        item["path"]: item["rate"] for item in broad_stats.get("path_ranking", [])
    }
    fine_path_counts = {
        path: sum(str(item["outcome"].get("realized_path", "unresolved")) == path for item in fine)
        for path in path_names
    }
    denominator = n + prior_strength
    path_posterior = sorted(
        [
            {
                "path": path,
                "posterior_rate": (fine_path_counts[path] + prior_strength * broad_path.get(path, 0)) / denominator,
            }
            for path in path_names
        ],
        key=lambda item: (-item["posterior_rate"], item["path"]),
    )
    return {
        "status": "available", "fine_cases": n, "broad_cases": len(broad),
        "prior_strength": prior_strength,
        "direction_hit_rate": (fine_direction_hits + prior_strength * broad_direction) / denominator,
        "material_move_rate": (fine_material_hits + prior_strength * broad_material) / denominator,
        "path_posterior": [{"rank": i + 1, **item} for i, item in enumerate(path_posterior)],
        "guardrail": "細分病例以大病例群作先驗收縮，避免小樣本極端比例直接主導。",
    }


def vitals_from_row(row: pd.Series) -> dict[str, Any]:
    treasury_5y = number(row.get("treasury_5y_close"))
    treasury_10y = number(row.get("treasury_10y_close"))
    treasury_30y = number(row.get("treasury_30y_close"))
    return {
        "night_session": select(row, [
            "night_open", "night_high", "night_low", "night_close", "night_volume",
            "night_return", "night_range", "night_spread_per", "night_close_position",
            "night_recovery_from_low", "night_close_vs_prior_cash",
        ]),
        "external_markets": select(row, [
            "external_score", "nasdaq_return_1d", "sox_return_1d", "sp500_return_1d",
            "tsm_adr_return_1d", "vix_return_1d", "usd_twd_return_1d",
        ]),
        "treasury_yields_context_only": select(row, [
            "treasury_5y_close", "treasury_10y_close", "treasury_30y_close",
            "treasury_5y_return_1d", "treasury_10y_return_1d", "treasury_30y_return_1d",
        ]) | {
            "treasury_10y_minus_5y": subtract(treasury_10y, treasury_5y),
            "treasury_30y_minus_10y": subtract(treasury_30y, treasury_10y),
        },
        "prior_cash_session": select(row, [
            "prior_cash_open", "prior_cash_high", "prior_cash_low", "prior_cash_close",
            "prior_cash_volume", "prior_open_gap_pct", "prior_intraday_low_pct",
            "prior_close_return_pct", "prior_close_recovery_ratio",
        ]),
        "technical_context": select(row, [
            "prior_ma5", "prior_ma10", "prior_ma20", "prior_ma60",
            "prior_close_vs_ma5", "prior_close_vs_ma20", "prior_close_vs_ma60",
            "prior_drawdown_60", "prior_realized_volatility_20", "prior_range_20",
            "prior_volume_ratio_20",
        ]),
        "market_breadth": select(row, [
            "breadth_date", "breadth_age_days", "price_advancing_fraction",
            "price_declining_fraction", "price_advance_decline_breadth",
            "price_up_volume_fraction", "price_return_dispersion",
            "inst_foreign_positive_fraction", "inst_foreign_negative_fraction",
            "inst_combined_breadth",
        ]),
        "futures_positioning": select(row, [
            "open_interest_date", "open_interest_age_days", "tx_total_open_interest",
            "tx_open_interest_change_1d", "tx_open_interest_change_5d",
            "institutional_position_date", "institutional_position_age_days",
            "foreign_futures_net_oi", "trust_futures_net_oi", "dealer_futures_net_oi",
            "institutional_futures_net_oi",
        ]),
        "night_microstructure": select(row, [
            "micro_bar_count", "micro_tick_count", "micro_first_hour_return",
            "micro_last_hour_return", "micro_return_acceleration",
            "micro_realized_volatility", "micro_trend_efficiency", "micro_up_bar_share",
            "micro_close_location", "micro_max_drawdown", "micro_max_runup",
            "micro_first_hour_volume_share", "micro_last_hour_volume_share",
            "micro_signed_volume_ratio", "micro_session_volume",
        ]),
        "cash_intraday_course_15m": select(row, [
            "cash_15m_bar_count", "cash_15m_return", "cash_15m_max_drawdown",
            "cash_15m_low_time", "cash_15m_high_time", "cash_15m_signed_volume_ratio",
        ]),
    }


def causal_chain_from_row(row: pd.Series, expected_paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step": 1, "role": "trigger", "status": "observed",
            "evidence": {"night_status": json_safe(row.get("night_status")), "external_score": json_safe(row.get("external_score"))},
        },
        {
            "step": 2, "role": "price_memory", "status": "observed",
            "evidence": {"prior_candle": json_safe(row.get("prior_candle_type")), "prior_washout": json_safe(row.get("prior_washout_type"))},
        },
        {
            "step": 3, "role": "crowd_interpretation", "status": "inferred_descriptive",
            "evidence": {"state": json_safe(row.get("state")), "direction": json_safe(row.get("direction")), "net_urgency": json_safe(row.get("net_urgency_score"))},
        },
        {
            "step": 4, "role": "positioning_confirmation", "status": "confirmed" if bool_value(row.get("position_confirmation")) else "unconfirmed",
            "evidence": {"crowding_status": json_safe(row.get("crowding_status"))},
        },
        {
            "step": 5, "role": "possible_transitions", "status": "conditional",
            "evidence": expected_paths,
        },
    ]


def symptoms_from_row(row: pd.Series, evidence: list[Any]) -> list[dict[str, Any]]:
    symptoms = [
        symptom("night_path", row.get("night_status"), "night", severity_from_night(row.get("night_status"))),
        symptom("external_pressure", row.get("external_score"), "external", severity_from_score(row.get("external_score"))),
        symptom("prior_candle", row.get("prior_candle_type"), "price_memory", "watch"),
        symptom("prior_washout", row.get("prior_washout_type"), "recovery", "watch"),
        symptom("crowding", row.get("crowding_status"), "positioning", "unconfirmed"),
    ]
    if bool_value(row.get("exogenous_reset_watch")):
        symptoms.append(symptom("exogenous_shock", True, "event", "high"))
    for index, item in enumerate(evidence):
        symptoms.append({"code": f"psychology_evidence_{index + 1}", "category": "psychology", "severity": "evidence", "evidence": str(item)})
    return unique_symptoms(symptoms)


def diagnosis_from_row(row: pd.Series) -> dict[str, Any]:
    return {
        "framework": "crowd_psychology_path_v1",
        "psychology_state": row.get("state"), "state_label": row.get("state_label"),
        "direction": row.get("direction"),
        "bearish_urgency_score": number(row.get("bearish_urgency_score")),
        "bullish_urgency_score": number(row.get("bullish_urgency_score")),
        "net_urgency_score": number(row.get("net_urgency_score")),
        "calibrated_direction_score": number(row.get("calibrated_direction_score")),
        "direction_weight_version": row.get("direction_weight_version"),
        "crowding_status": row.get("crowding_status"),
        "position_confirmation": bool_value(row.get("position_confirmation")),
        "exogenous_reset_watch": bool_value(row.get("exogenous_reset_watch")),
        "formal_direction_signal": None,
        "certainty": "descriptive_not_formal",
    }


def safe_prescription(direction: str, state: str, row: pd.Series) -> dict[str, Any]:
    levels = {
        "night_low": number(row.get("night_low")), "night_close": number(row.get("night_close")),
        "night_open": number(row.get("night_open")), "night_high": number(row.get("night_high")),
    }
    if state == "urgent_following" and direction == "bearish":
        action = "defend_and_observe_short_covering_before_continuation"
        checkpoints = ["先核對開盤缺口", "跌破夜盤低點後能否收回", "是否站回夜盤收盤", "是否站回夜盤開盤"]
    elif state == "urgent_following" and direction == "bullish":
        action = "avoid_blind_chasing_and_observe_profit_taking"
        checkpoints = ["先核對開盤缺口", "能否站穩夜盤高位", "是否跌回夜盤收盤", "是否跌破夜盤低點"]
    elif state == "confirmation":
        action = "observe_direction_with_close_confirmation"
        checkpoints = ["開盤是否同向", "夜盤極值是否被突破", "正式收盤是否同向"]
    else:
        action = "observe_and_wait_for_range_confirmation"
        checkpoints = ["等待夜盤高低區間突破", "只用正式收盤更新診斷"]
    return {
        "action": action, "intensity": "observation_only", "levels": levels,
        "monitoring_checkpoints": checkpoints, "formal_direction_signal": None,
        "may_increase_leverage": False, "may_average_down": False,
        "requires_price_confirmation": True,
        "rationale": "病例預後只能調整監控順序，不直接授權交易。",
    }


def contraindications(row: pd.Series) -> list[dict[str, str]]:
    output = [
        {"code": "no_formal_psychology_signal", "reason": "心理模型尚未通過前瞻方向驗證。"},
        {"code": "no_blind_gap_chasing", "reason": "開盤缺口命中不等於日內一路延伸。"},
    ]
    if not bool_value(row.get("position_confirmation")):
        output.append({"code": "position_unconfirmed", "reason": "缺少同步廣度與未平倉量確認，不得宣稱擁擠或投降。"})
    return output


def outcome_from_row(row: pd.Series) -> dict[str, Any]:
    direction = str(row.get("direction"))
    cash_return = number(row.get("cash_close_return"))
    material = bool(
        cash_return is not None and (
            (direction == "bearish" and cash_return <= -0.005)
            or (direction == "bullish" and cash_return >= 0.005)
        )
    )
    strength = classify_direction_strength(cash_return, direction)
    session_path = classify_session_path(
        row.get("prior_cash_close"), row.get("cash_open"), row.get("cash_high"),
        row.get("cash_low"), row.get("cash_close"),
    )
    return {
        "status": "resolved", "actual_direction": row.get("actual_direction"),
        "realized_path": row.get("realized_path"), "path_resolved": bool_value(row.get("path_resolved")),
        "direction_hit": bool_value(row.get("direction_hit")),
        "gap_direction_hit": bool_value(row.get("gap_direction_hit")),
        "top_path_hit": bool_value(row.get("top_path_hit")),
        "night_baseline_hit": bool_value(row.get("night_baseline_hit")),
        "material_direction_confirmation": material,
        "direction_strength": strength,
        "borderline_direction_confirmation": strength["borderline_confirmation"],
        "session_path": session_path,
        "gap_return": number(row.get("gap_return")), "cash_close_return": cash_return,
        "intraday_return": number(row.get("intraday_return")),
        "cash_ohlc": select(row, ["cash_open", "cash_high", "cash_low", "cash_close"]),
        "multi_day": {
            f"{horizon}d": select(row, [
                f"forward_{horizon}d_return", f"forward_{horizon}d_max_drawdown",
                f"forward_{horizon}d_max_runup",
            ])
            for horizon in [1, 3, 5, 20]
        },
    }


def follow_up_from_row(row: pd.Series) -> dict[str, Any]:
    if not bool_value(row.get("direction_hit")):
        category = "diagnosis_direction_miss"
        lesson = "方向診斷失敗，應檢查外部加權是否覆蓋了較強的夜盤訊號。"
    elif not bool_value(row.get("top_path_hit")):
        category = "course_path_miss"
        course = classify_session_path(
            row.get("prior_cash_close"), row.get("cash_open"), row.get("cash_high"),
            row.get("cash_low"), row.get("cash_close"),
        )
        if course.get("label") == "deep_selloff_recovered":
            lesson = "方向相符但先深殺後收復；下次須同時列出深跌回收路徑與最大不利幅度。"
        else:
            lesson = "大方向相符但病程路徑不同，應依同型病例重新排序路徑。"
    else:
        category = "diagnosis_and_course_match"
        lesson = "方向與首選病程均相符，保留作為相似病例參考。"
    return {"status": "reviewed", "error_category": category, "lesson": lesson}


def flatten_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in records:
        identity = item["identity"]
        diagnosis = item["diagnosis"]
        prognosis = item["prognosis_from_prior_cases"]
        outcome = item["outcome"]
        treatment = item.get("treatment_course", {})
        treatment_detail = treatment.get("treatment", {})
        disease_gene = treatment.get("disease_gene", {})
        trigger_condition = treatment.get("trigger_condition", {})
        night = item["vital_signs"]["night_session"]
        external = item["vital_signs"]["external_markets"]
        treasury = item["vital_signs"]["treasury_yields_context_only"]
        prior = item["vital_signs"]["prior_cash_session"]
        technical = item["vital_signs"]["technical_context"]
        breadth = item["vital_signs"]["market_breadth"]
        positioning = item["vital_signs"]["futures_positioning"]
        micro = item["vital_signs"]["night_microstructure"]
        intraday = item["vital_signs"]["cash_intraday_course_15m"]
        cash_ohlc = outcome.get("cash_ohlc", {})
        direction_strength = outcome.get("direction_strength", {})
        session_path = outcome.get("session_path", {})
        rows.append({
            "case_id": item["case_id"], "schema_version": item["schema_version"],
            **identity, "data_completeness": item["data_governance"].get("data_completeness"),
            "chief_complaint": item["chief_complaint"],
            **{f"night_{key.removeprefix('night_')}": value for key, value in night.items()},
            **external, **treasury, **prior, **technical, **breadth, **positioning,
            **micro, **intraday,
            "prior_candle_type": symptom_value(item, "prior_candle"),
            "prior_washout_type": symptom_value(item, "prior_washout"),
            "psychology_state": diagnosis.get("psychology_state"), "state_label": diagnosis.get("state_label"),
            "direction": diagnosis.get("direction"), "bearish_urgency_score": diagnosis.get("bearish_urgency_score"),
            "bullish_urgency_score": diagnosis.get("bullish_urgency_score"), "net_urgency_score": diagnosis.get("net_urgency_score"),
            "calibrated_direction_score": diagnosis.get("calibrated_direction_score"),
            "direction_weight_version": diagnosis.get("direction_weight_version"),
            "crowding_status": diagnosis.get("crowding_status"), "position_confirmation": diagnosis.get("position_confirmation"),
            "exogenous_reset_watch": diagnosis.get("exogenous_reset_watch"),
            "prior_cohort_cases": prognosis.get("prior_cases"),
            "behavior_phenotype": diagnosis.get("behavior_phenotype"),
            "exact_phenotype_prior_cases": prognosis.get("exact_phenotype_prior_cases"),
            "prognosis_selected_tier": prognosis.get("selected_tier"),
            "fine_phenotype_direction_hit_rate": prognosis.get("fine_phenotype_statistics", {}).get("direction_hit_rate"),
            "fine_phenotype_material_move_rate": prognosis.get("fine_phenotype_statistics", {}).get("material_move_rate"),
            "fine_phenotype_average_cash_close_return": prognosis.get("fine_phenotype_statistics", {}).get("average_cash_close_return"),
            "regime_prior_cases": prognosis.get("regime_prior_cases"),
            "bayesian_direction_hit_rate": prognosis.get("hierarchical_bayesian_estimate", {}).get("direction_hit_rate"),
            "bayesian_material_move_rate": prognosis.get("hierarchical_bayesian_estimate", {}).get("material_move_rate"),
            "prior_direction_hit_rate": prognosis.get("direction_hit_rate"),
            "prior_material_move_rate": prognosis.get("material_move_rate"),
            "prior_average_cash_close_return": prognosis.get("average_cash_close_return"),
            "empirical_path_ranking_json": json.dumps(prognosis.get("empirical_path_ranking", []), ensure_ascii=False),
            "prescription_action": item["prescription"].get("action"),
            "prescription_intensity": item["prescription"].get("intensity"),
            "treatment_available": treatment.get("available"),
            "disease_gene_score": disease_gene.get("score"),
            "trigger_condition_score": trigger_condition.get("score"),
            "treatment_phase": treatment_detail.get("phase"),
            "treatment_stage": treatment_detail.get("stage"),
            "medicine_type": treatment_detail.get("medicine_type"),
            "correct_medicine": treatment_detail.get("correct_medicine"),
            "after_effect_risk": treatment_detail.get("after_effect_risk"),
            "immediate_medicine_bias": treatment.get("fact_changing_medicine", {}).get("bias"),
            "immediate_medicine_score": treatment.get("fact_changing_medicine", {}).get("score"),
            "fact_changing_medicines_json": json.dumps(treatment.get("fact_changing_medicine", {}).get("medicines", []), ensure_ascii=False),
            "causality_rule": treatment.get("causality_rule"),
            "causality_pipeline_json": json.dumps(treatment.get("causality_pipeline", []), ensure_ascii=False),
            "next_episode_prior": treatment.get("next_episode_prior"),
            "outcome_status": outcome.get("status"), "actual_direction": outcome.get("actual_direction"),
            "realized_path": outcome.get("realized_path"), "direction_hit": outcome.get("direction_hit"),
            "gap_direction_hit": outcome.get("gap_direction_hit"), "top_path_hit": outcome.get("top_path_hit"),
            "material_direction_confirmation": outcome.get("material_direction_confirmation"),
            "borderline_direction_confirmation": outcome.get("borderline_direction_confirmation"),
            "direction_strength_label": direction_strength.get("label"),
            "session_path_label": session_path.get("label"),
            "session_adverse_from_open": session_path.get("adverse_from_open"),
            "session_close_location": session_path.get("close_location"),
            "session_recovery_ratio": session_path.get("recovery_ratio"),
            "gap_return": outcome.get("gap_return"), "cash_close_return": outcome.get("cash_close_return"),
            "intraday_return": outcome.get("intraday_return"),
            **cash_ohlc,
            **{
                f"forward_{horizon}d_{metric}": outcome.get("multi_day", {}).get(f"{horizon}d", {}).get(f"forward_{horizon}d_{metric}")
                for horizon in [1, 3, 5, 20]
                for metric in ["return", "max_drawdown", "max_runup"]
            },
            "follow_up_category": item["follow_up"].get("error_category"),
            "symptoms_json": json.dumps(item["symptoms"], ensure_ascii=False),
            "causal_chain_json": json.dumps(item["causal_chain"], ensure_ascii=False),
            "differential_diagnosis_json": json.dumps(item["differential_diagnosis"], ensure_ascii=False),
            "contraindications_json": json.dumps(item["contraindications"], ensure_ascii=False),
            "evidence_sha256": item["evidence_sha256"],
        })
    return pd.DataFrame(rows)


def summarize_ledger(records: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = [item for item in records if item["outcome"].get("status") == "resolved"]
    pending = [item for item in records if item["outcome"].get("status") == "pending"]
    leakage_failures = [
        item["case_id"] for item in records
        if not item["data_governance"].get("strict_prior_cash")
        or not item["data_governance"].get("strict_prior_external")
    ]
    return {
        "framework": "complete_market_clinical_ledger_v2",
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "records": len(records), "resolved_records": len(resolved), "pending_records": len(pending),
        "first_date": records[0]["identity"]["decision_date"] if records else None,
        "latest_date": records[-1]["identity"]["decision_date"] if records else None,
        "leakage_audit": {"passed": not leakage_failures, "failure_count": len(leakage_failures), "case_ids": leakage_failures[:20]},
        "sections_per_record": [
            "identity", "data_governance", "chief_complaint", "vital_signs", "symptoms",
            "causal_chain", "diagnosis", "differential_diagnosis", "prognosis_from_prior_cases",
            "treatment_course", "prescription", "contraindications", "outcome", "follow_up",
        ],
        "behavior_statistics": behavior_statistics(resolved),
        "multi_day_behavior_statistics": multi_day_behavior_statistics(resolved),
        "current_case": records[-1] if records else None,
    }


def behavior_statistics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cohorts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        diagnosis = record.get("diagnosis", {})
        key = (str(diagnosis.get("psychology_state")), str(diagnosis.get("direction")))
        cohorts.setdefault(key, []).append(record)
    output = []
    for (state, direction), cohort in sorted(cohorts.items()):
        outcomes = [item["outcome"] for item in cohort]
        actionable = direction in {"bearish", "bullish"}
        path_counts: dict[str, int] = {}
        for outcome in outcomes:
            path = str(outcome.get("realized_path", "unresolved"))
            path_counts[path] = path_counts.get(path, 0) + 1
        output.append({
            "psychology_state": state, "direction": direction, "cases": len(cohort),
            "direction_hit_rate": mean(item.get("direction_hit") for item in outcomes) if actionable else None,
            "gap_direction_hit_rate": mean(item.get("gap_direction_hit") for item in outcomes) if actionable else None,
            "material_move_rate": mean(item.get("material_direction_confirmation") for item in outcomes) if actionable else None,
            "average_cash_close_return": mean(item.get("cash_close_return") for item in outcomes),
            "path_distribution": [
                {"path": path, "cases": count, "rate": count / len(cohort)}
                for path, count in sorted(path_counts.items(), key=lambda item: (-item[1], item[0]))
            ],
        })
    return output


def multi_day_behavior_statistics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    cohorts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        diagnosis = record.get("diagnosis", {})
        key = (str(diagnosis.get("psychology_state")), str(diagnosis.get("direction")))
        cohorts.setdefault(key, []).append(record)
    for (state, direction), cohort in sorted(cohorts.items()):
        if direction not in {"bearish", "bullish"}:
            continue
        for horizon in [1, 3, 5, 20]:
            key = f"forward_{horizon}d_return"
            values = [
                item.get("outcome", {}).get("multi_day", {}).get(f"{horizon}d", {}).get(key)
                for item in cohort
            ]
            values = [float(value) for value in values if present(value)]
            if not values:
                continue
            hits = [value < 0 if direction == "bearish" else value > 0 for value in values]
            output.append({
                "psychology_state": state, "direction": direction,
                "horizon_days": horizon, "cases": len(values),
                "direction_hit_rate": sum(hits) / len(hits),
                "average_return": sum(values) / len(values),
            })
    return output


def render_ledger_report(summary: dict[str, Any]) -> str:
    current = summary.get("current_case") or {}
    identity = current.get("identity", {})
    diagnosis = current.get("diagnosis", {})
    prognosis = current.get("prognosis_from_prior_cases", {})
    prescription = current.get("prescription", {})
    treatment_course = current.get("treatment_course", {})
    treatment = treatment_course.get("treatment", {})
    disease_gene = treatment_course.get("disease_gene", {})
    trigger_condition = treatment_course.get("trigger_condition", {})
    outcome = current.get("outcome", {})
    fine = prognosis.get("fine_phenotype_statistics", {})
    lines = [
        "# 台股完整市場病歷表", "",
        f"病歷期間：{summary.get('first_date')} 至 {summary.get('latest_date')}。",
        f"總病例 {summary.get('records', 0)} 件；已結案 {summary.get('resolved_records', 0)} 件；待追蹤 {summary.get('pending_records', 0)} 件。",
        f"時間洩漏稽核：{'通過' if summary.get('leakage_audit', {}).get('passed') else '失敗'}。", "",
        "## 最新病例", "",
        f"- 病歷號：{current.get('case_id', 'NA')}",
        f"- 決策日：{identity.get('decision_date')}；階段：{identity.get('record_phase')}",
        f"- 主訴：{current.get('chief_complaint')}",
        f"- 心理診斷：{diagnosis.get('state_label')} / {diagnosis.get('direction')}；急迫淨分 {diagnosis.get('net_urgency_score')}",
        f"- 細脈象完全相同病例：{prognosis.get('exact_phenotype_prior_cases', 0)} 件；採用層級：{prognosis.get('selected_tier')}",
        f"- 細脈象探索值：方向命中 {pct(fine.get('direction_hit_rate'))}；顯著幅度 {pct(fine.get('material_move_rate'))}；平均報酬 {pct(fine.get('average_cash_close_return'))}（小樣本不直接採用）",
        f"- 採用病例群：{prognosis.get('prior_cases', 0)} 件；方向命中 {pct(prognosis.get('direction_hit_rate'))}；顯著幅度 {pct(prognosis.get('material_move_rate'))}",
        f"- 療程：第{treatment.get('phase', 'NA')}期 {treatment.get('stage', 'NA')}；藥性 {treatment.get('medicine_type', 'NA')}；對症 {treatment.get('correct_medicine', 'NA')}；陰霾 {treatment.get('after_effect_risk', 'NA')}",
        f"- 即效藥：{treatment_course.get('fact_changing_medicine', {}).get('label', 'NA')}；偏向 {treatment_course.get('fact_changing_medicine', {}).get('bias', 'NA')}；分數 {treatment_course.get('fact_changing_medicine', {}).get('score', 'NA')}",
        f"- 因果程序：{treatment_course.get('causality_rule', 'NA')}",
        f"- 病灶/觸發：{disease_gene.get('score', 'NA')} / {trigger_condition.get('score', 'NA')}；下期先驗：{treatment_course.get('next_episode_prior', 'NA')}",
        f"- 處方：{prescription.get('action')}（{prescription.get('intensity')}）",
        f"- 轉歸：{outcome.get('status')} — {outcome.get('reason', outcome.get('realized_path', ''))}", "",
        "### 同型病例病程排序", "",
        "| 順位 | 病程 | 件數 | 比率 |", "| ---: | --- | ---: | ---: |",
    ]
    ranking = prognosis.get("empirical_path_ranking", [])
    if ranking:
        for item in ranking:
            lines.append(f"| {item['rank']} | {item['path']} | {item['cases']} | {pct(item['rate'])} |")
    else:
        lines.append("| - | 既往病例不足 | 0 | NA |")
    lines.extend([
        "", "## 行為統計總表", "",
        "| 心理狀態 | 方向 | 病例 | 收盤方向命中 | 缺口命中 | 顯著幅度 | 平均收盤報酬 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in summary.get("behavior_statistics", []):
        lines.append(
            f"| {item['psychology_state']} | {item['direction']} | {item['cases']} | "
            f"{pct(item['direction_hit_rate'])} | {pct(item['gap_direction_hit_rate'])} | "
            f"{pct(item['material_move_rate'])} | {pct(item['average_cash_close_return'])} |"
        )
    current_state = diagnosis.get("psychology_state")
    current_direction = diagnosis.get("direction")
    current_multi = [
        item for item in summary.get("multi_day_behavior_statistics", [])
        if item["psychology_state"] == current_state and item["direction"] == current_direction
    ]
    lines.extend(["", "### 最新診斷的多日轉歸", "", "| 期間 | 病例 | 同方向 | 平均報酬 |", "| --- | ---: | ---: | ---: |"])
    for item in current_multi:
        lines.append(
            f"| {item['horizon_days']}日 | {item['cases']} | {pct(item['direction_hit_rate'])} | {pct(item['average_return'])} |"
        )
    lines.extend([
        "", "## 每筆病歷內容", "",
        "1. 身分與資料截止時間。",
        "2. 資料治理與未來資訊洩漏檢查。",
        "3. 主訴、生命徵象及結構化症狀。",
        "4. 心理狀態診斷、方向與急迫分數。",
        "5. 三條鑑別病程與失效條件。",
        "6. 嚴格使用決策日前同型病例形成的預後。",
        "7. 觀察型處方、監控價位與禁忌。",
        "8. 正式收盤後回填的轉歸與誤診檢討。", "",
        "> 病歷表是研究與監控工具，不是個人化投資處方；未通過前瞻驗證的心理方向不升格為正式交易訊號。", "",
    ])
    return "\n".join(lines)


def completeness(row: pd.Series) -> str:
    required = ["night_return", "prior_cash_close", "external_score", "state", "direction"]
    return "complete_core" if all(present(row.get(key)) for key in required) else "partial"


def unavailable_domains(row: pd.Series) -> list[str]:
    output = []
    if not present(row.get("price_advance_decline_breadth")):
        output.append("market_breadth")
    elif number(row.get("breadth_age_days")) is not None and number(row.get("breadth_age_days")) > 7:
        output.append("market_breadth_stale")
    if not present(row.get("tx_total_open_interest")):
        output.append("futures_open_interest")
    if not present(row.get("foreign_futures_net_oi")):
        output.append("institutional_futures_position")
    if not present(row.get("micro_bar_count")):
        output.append("night_microstructure")
    if not present(row.get("cash_15m_bar_count")):
        output.append("cash_intraday_15m_course")
    if not present(row.get("treasury_10y_close")):
        output.append("treasury_yields")
    if not present(row.get("prior_cash_volume")):
        output.append("cash_volume")
    if not bool_value(row.get("capital_flow_available")) and not present(row.get("foreign_futures_net_oi")):
        output.append("institutional_and_derivatives_history")
    if not bool_value(row.get("lifecycle_diagnosis_available")):
        output.append("historical_lifecycle_health_diagnosis")
    return output


def chief_complaint(state: Any, direction: Any, night_status: Any) -> str:
    return f"盤前心理狀態 {state}、方向 {direction}；夜盤病徵 {night_status}。"


def symptom(code: str, value: Any, category: str, severity: str) -> dict[str, Any]:
    return {"code": code, "category": category, "severity": severity, "value": json_safe(value)}


def unique_symptoms(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output, seen = [], set()
    for item in items:
        marker = (str(item.get("code")), str(item.get("category")), str(item.get("evidence", item.get("value"))))
        if marker not in seen:
            seen.add(marker)
            output.append(item)
    return output


def severity_from_night(status: Any) -> str:
    return "high" if "extension" in str(status) else ("watch" if status else "unavailable")


def severity_from_score(value: Any) -> str:
    score = number(value)
    return "high" if score is not None and abs(score) >= 3 else "watch"


def preoutcome_evidence(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in {"case_id", "outcome", "follow_up", "evidence_sha256"}}


def case_id(record: dict[str, Any]) -> str:
    date = record["identity"]["decision_date"]
    return f"TWII-{date}-CLIN-{fingerprint(preoutcome_evidence(record))[:12]}"


def fingerprint(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest().upper()


def select(row: pd.Series, names: list[str]) -> dict[str, Any]:
    return {name: json_safe(row.get(name)) for name in names}


def symptom_value(record: dict[str, Any], code: str):
    match = next((item for item in record.get("symptoms", []) if item.get("code") == code), {})
    return match.get("value")


def json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not present(value):
        return []
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def mean(values) -> float | None:
    cleaned = [float(value) for value in values if present(value)]
    return sum(cleaned) / len(cleaned) if cleaned else None


def present(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not pd.isna(value)
    except (TypeError, ValueError):
        return True


def number(value: Any) -> float | None:
    return float(value) if present(value) else None


def ratio_value(numerator: Any, denominator: Any) -> float | None:
    top, bottom = number(numerator), number(denominator)
    return top / bottom - 1 if top is not None and bottom not in {None, 0} else None


def subtract(first: float | None, second: float | None) -> float | None:
    return first - second if first is not None and second is not None else None


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value) if present(value) else False


def json_safe(value: Any):
    if not present(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def pct(value: Any) -> str:
    return "NA" if not present(value) else f"{float(value):.2%}"
