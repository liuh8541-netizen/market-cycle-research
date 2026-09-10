"""Safety-first market health assessment.

This module treats lifecycle signals as clinical evidence, not prophecy. It
separates observations, differential diagnosis, and permitted risk actions so
an unvalidated or incomplete input cannot silently become a trading command.
"""

from __future__ import annotations

from typing import Any


CONFIDENCE_RANK = {"low": 0, "medium_low": 1, "medium": 2, "high": 3}


def build_market_health_assessment(payload: dict[str, Any]) -> dict[str, Any]:
    symptoms = _collect_symptoms(payload)
    diagnosis = _diagnose(payload, symptoms)
    prescription = _prescribe(payload, diagnosis)
    health_value = _judge_health_value(payload, symptoms, diagnosis)
    return {
        "framework": "preventive_market_medicine_v1",
        "principle": "first_do_no_harm",
        "health_value": health_value,
        "symptoms": symptoms,
        "diagnosis": diagnosis,
        "prescription": prescription,
        "safety_checks": _safety_checks(payload, diagnosis, prescription),
    }


def _collect_symptoms(payload: dict[str, Any]) -> list[dict[str, str]]:
    symptoms: list[dict[str, str]] = []
    integrated = payload.get("integrated_summary", {})
    freshness = payload.get("data_freshness", {})
    candle = payload.get("candlestick_pattern", {})
    washout = payload.get("washout_pattern", {})
    premarket = payload.get("premarket", {})

    _add_evidence(symptoms, integrated.get("bearish_evidence"), "deterioration", "warning")
    _add_evidence(symptoms, integrated.get("bullish_evidence"), "repair", "supportive")
    _add_evidence(symptoms, integrated.get("neutral_evidence"), "uncertain", "watch")

    if freshness.get("overall_status") != "current":
        symptoms.append(
            {
                "code": "incomplete_or_stale_data",
                "category": "data_quality",
                "severity": "high",
                "evidence": f"資料時效狀態為 {freshness.get('overall_status', 'unknown')}。",
            }
        )
    if premarket.get("is_premarket") and not premarket.get("night_session_complete", False):
        symptoms.append(
            {
                "code": "night_session_incomplete",
                "category": "data_quality",
                "severity": "high",
                "evidence": "05:00前夜盤尚未完成，只能形成暫定情境。",
            }
        )
    if candle.get("type") in {"long_bear", "gap_up_failed"}:
        symptoms.append(
            {
                "code": "bearish_price_damage",
                "category": "price",
                "severity": "high",
                "evidence": candle.get("plain_summary", candle.get("label", "價格結構轉弱")),
            }
        )
    if washout.get("type") == "failed_washout":
        symptoms.append(
            {
                "code": "failed_recovery",
                "category": "price",
                "severity": "high",
                "evidence": washout.get("plain_summary", "下殺後未能有效拉回。"),
            }
        )
    return _deduplicate(symptoms)


def _judge_health_value(
    payload: dict[str, Any],
    symptoms: list[dict[str, str]],
    diagnosis: dict[str, Any],
) -> dict[str, Any]:
    crash = payload.get("crash_monitor", {})
    candle = payload.get("candlestick_pattern", {})
    washout = payload.get("washout_pattern", {})
    intraday = payload.get("intraday_tactical_monitor", {})
    technical = payload.get("technical_phase", {})
    bagua = payload.get("bagua_lifecycle", {})
    behavior = payload.get("human_behavior_market_pattern", {})

    risk_value = _num(crash.get("risk_value"))
    health_score = _num(crash.get("health_score"))
    alert_code = str(crash.get("alert_code") or "")
    candle_type = candle.get("type")
    washout_type = washout.get("type")
    intraday_code = intraday.get("code")
    state_code = bagua.get("roles", {}).get("state_gua", {}).get("code")
    technical_status = technical.get("alignment", {}).get("status")
    behavior_bias = behavior.get("direction_bias")
    behavior_label = behavior.get("label")

    evidence: list[str] = []
    if health_score is not None:
        evidence.append(f"健康指數 {health_score:.1f}/100。")
    if risk_value is not None:
        evidence.append(f"風險值 {risk_value:.1f}/100。")
    if state_code:
        evidence.append(f"狀態卦 {state_code}。")
    if intraday_code:
        evidence.append(f"盤中型態 {intraday_code}。")
    if behavior_label:
        evidence.append(f"人類行為模式 {behavior_label}。")

    freshness_status = str(payload.get("data_freshness", {}).get("overall_status") or "").lower()
    incomplete_night = any(item.get("code") == "night_session_incomplete" for item in symptoms)
    core_data_blocked = freshness_status in {"missing", "stale", "late"} or incomplete_night
    data_quality_limited = diagnosis.get("status") == "insufficient_data" and not core_data_blocked
    bearish_damage = candle_type in {"long_bear", "gap_up_failed"} or washout_type == "failed_washout"
    repair_behavior = behavior_bias == "constructive" or intraday_code in {
        "night_down_cash_reclaim",
        "night_down_break_night_low_reclaim_washout",
    } or washout_type in {"strong_washout", "normal_washout", "gap_down_recovery"}
    structural_alert = alert_code in {"crash_warning", "confirmed_warning"} or (
        risk_value is not None and risk_value >= 75
    ) or (health_score is not None and health_score < 30)
    heating_alert = behavior_bias == "defensive" or alert_code in {"bottom_failed_watch", "risk_warning"} or (
        risk_value is not None and risk_value >= 60
    ) or bearish_damage

    if core_data_blocked:
        code = "insufficient_data"
        label = "資料不足"
        value = 0
        headline = "資料不足，不能判斷健康脈動。"
        meaning = "資料時效或關鍵資料缺漏時，只能觀察，不應把模型輸出當成健康結論。"
    elif structural_alert:
        code = "structural_damage"
        label = "結構破壞"
        value = 20
        headline = "市場脈動已超出可控風險，需啟動崩盤核對。"
        meaning = "跌破重要防線且未收回、健康指數過低或風險值過高時，不再視為正常呼吸。"
    elif heating_alert:
        code = "risk_heating"
        label = "風險升溫"
        value = 45
        headline = "市場仍可觀察，但脈動開始變急促。"
        meaning = "價格或K線已有傷害，需看隔日能否守低、收復關鍵線與恢復承接。"
    elif (health_score is not None and health_score >= 65) and (risk_value is not None and risk_value <= 35):
        code = "healthy_pulse"
        label = "健康脈動"
        value = 85
        headline = "漲跌仍屬正常呼吸，風險目前在可控範圍。"
        meaning = "有漲有跌、有換手，但未破壞高低點、均線與承接結構。"
    elif repair_behavior or (health_score is not None and health_score >= 45) or technical_status == "aligned":
        code = "controlled_volatility"
        label = "可控震盪"
        value = 65
        headline = "市場有波動，但暫未失控，重點看防守線是否續守。"
        meaning = "洗盤、回測或換手若能收回關鍵價，仍屬可控風險內的波動。"
    else:
        code = "watch"
        label = "觀察脈動"
        value = 55
        headline = "訊號未惡化到失控，但健康證據仍需補強。"
        meaning = "方向與承接尚未完全一致，應等待下一個交易日驗證。"

    if data_quality_limited and code != "insufficient_data":
        evidence.append("部分非核心資料缺漏，健康價值保留判斷但需降權。")
        value = max(0, value - 10)
        if code == "healthy_pulse":
            code = "controlled_volatility"
            label = "可控震盪"
            headline = "核心價格脈動仍可控，但資料缺口使健康判斷降權。"

    return {
        "code": code,
        "label": label,
        "value": value,
        "headline": headline,
        "meaning": meaning,
        "controllable_risk": code in {"healthy_pulse", "controlled_volatility", "watch"},
        "rule": "健康價值判斷看風險是否可控，不以單日漲跌當唯一依據。",
        "evidence": evidence,
        "guardrail": "健康價值判斷只做風控分層，不產生買賣命令。",
    }


def _add_evidence(
    output: list[dict[str, str]],
    evidence: list[str] | None,
    category: str,
    severity: str,
) -> None:
    for index, text in enumerate(evidence or []):
        output.append(
            {
                "code": f"{category}_{index + 1}",
                "category": category,
                "severity": severity,
                "evidence": str(text),
            }
        )


def _diagnose(payload: dict[str, Any], symptoms: list[dict[str, str]]) -> dict[str, Any]:
    integrated = payload.get("integrated_summary", {})
    forecast = payload.get("forecast", {})
    consistency = payload.get("logic_consistency", {})
    bias = integrated.get("bias", "mixed")
    confidence = _confidence_code(integrated.get("confidence"))
    alternatives = _alternatives(payload, bias)

    data_blocked = any(item["category"] == "data_quality" for item in symptoms)
    contradictory = bool(consistency.get("contradiction_count", 0))
    if data_blocked:
        status = "insufficient_data"
        primary = "observation_required"
        confidence = "low"
    elif contradictory:
        status = "differential_required"
        primary = "conflicting_market_state"
        confidence = "low"
    else:
        status = "provisional"
        primary = {
            "bearish": "deteriorating",
            "slightly_bearish": "deterioration_watch",
            "bullish": "recovering",
            "slightly_bullish": "repair_watch",
        }.get(bias, "indeterminate")

    return {
        "status": status,
        "primary": primary,
        "lifecycle_stage": forecast.get("lifecycle_stage", "unknown"),
        "risk_regime": forecast.get("risk_regime", "neutral"),
        "confidence": confidence,
        "alternatives": alternatives,
        "evidence_count": len(symptoms),
        "confirmation_conditions": integrated.get("confirmation", []),
        "invalidation_conditions": integrated.get("invalidation", []),
        "is_formal_direction_diagnosis": False,
        "reason": (
            "廣義方向模型未通過正式驗證；診斷只描述狀態與風險。"
        ),
    }


def _alternatives(payload: dict[str, Any], bias: str) -> list[dict[str, str]]:
    washout_type = payload.get("washout_pattern", {}).get("type")
    alternatives: list[dict[str, str]] = []
    if bias in {"bearish", "slightly_bearish"}:
        alternatives.extend(
            [
                {"diagnosis": "temporary_external_shock", "discriminator": "外部壓力解除且迅速收復失守價。"},
                {"diagnosis": "washout_then_repair", "discriminator": "守住低點、收盤拉回且廣度改善。"},
            ]
        )
    elif bias in {"bullish", "slightly_bullish"}:
        alternatives.extend(
            [
                {"diagnosis": "short_covering_only", "discriminator": "反彈缺乏成交與廣度擴散。"},
                {"diagnosis": "failed_breakout", "discriminator": "突破後迅速跌回確認價下方。"},
            ]
        )
    else:
        alternatives.extend(
            [
                {"diagnosis": "range_consolidation", "discriminator": "價格持續位於確認價與失效價之間。"},
                {"diagnosis": "transition_state", "discriminator": "多週期與資金訊號開始共同轉向。"},
            ]
        )
    if washout_type in {"strong_washout", "normal_washout", "gap_down_recovery"}:
        alternatives.append(
            {"diagnosis": "unconfirmed_recovery", "discriminator": "隔日守低並站回確認價後才成立。"}
        )
    return alternatives


def _prescribe(payload: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
    policy = payload.get("production_policy", {})
    multi_day_enabled = bool(policy.get("main_multi_day_direction", {}).get("enabled"))
    confidence = diagnosis["confidence"]
    status = diagnosis["status"]
    primary = diagnosis["primary"]

    if status in {"insufficient_data", "differential_required"}:
        action = "observe_and_recheck"
        intensity = "none"
    elif primary in {"deteriorating", "deterioration_watch"}:
        action = "reduce_risk"
        intensity = "moderate" if CONFIDENCE_RANK[confidence] >= 2 else "light"
    elif primary in {"recovering", "repair_watch"}:
        action = "staged_reentry_only_after_confirmation"
        intensity = "light"
    else:
        action = "hold_risk_constant"
        intensity = "none"

    if not multi_day_enabled and action == "staged_reentry_only_after_confirmation":
        action = "observe_repair_without_directional_authority"
        intensity = "none"

    return {
        "action": action,
        "intensity": intensity,
        "formal_direction_signal": None,
        "may_increase_leverage": False,
        "may_average_down": False,
        "requires_confirmation_before_risk_increase": True,
        "stop_conditions": diagnosis.get("invalidation_conditions", []),
        "recovery_conditions": diagnosis.get("confirmation_conditions", []),
        "rationale": "處方強度不得高於診斷信心，未驗證方向不得轉成正式交易命令。",
    }


def _safety_checks(
    payload: dict[str, Any],
    diagnosis: dict[str, Any],
    prescription: dict[str, Any],
) -> dict[str, bool]:
    policy = payload.get("production_policy", {})
    direction_enabled = bool(policy.get("main_multi_day_direction", {}).get("enabled"))
    confidence_rank = CONFIDENCE_RANK[diagnosis["confidence"]]
    return {
        "unvalidated_direction_is_null": (
            direction_enabled or prescription["formal_direction_signal"] is None
        ),
        "low_confidence_not_aggressive": (
            confidence_rank >= 2 or prescription["intensity"] in {"none", "light"}
        ),
        "incomplete_data_blocks_treatment": (
            diagnosis["status"] != "insufficient_data"
            or prescription["action"] == "observe_and_recheck"
        ),
        "risk_increase_requires_confirmation": bool(
            prescription["requires_confirmation_before_risk_increase"]
        ),
        "leverage_increase_blocked": not prescription["may_increase_leverage"],
        "averaging_down_blocked": not prescription["may_average_down"],
    }


def _confidence_code(value: Any) -> str:
    text = str(value or "").lower()
    if text.startswith("高") or text.startswith("high"):
        return "high"
    if text.startswith("中低") or text.startswith("medium_low"):
        return "medium_low"
    if text.startswith("中") or text.startswith("medium"):
        return "medium"
    return "low"


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _deduplicate(items: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    seen = set()
    for item in items:
        key = (item["category"], item["evidence"])
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output
