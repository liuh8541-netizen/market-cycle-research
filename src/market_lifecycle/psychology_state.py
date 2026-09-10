"""Causal, safety-scoped market psychology state machine.

The state machine classifies observable crowd behaviour.  It does not emit a
formal cash-close or multi-day direction and it never treats a narrative label
as a validated probability.
"""

from __future__ import annotations

from typing import Any


STATE_ORDER = [
    "doubt", "confirmation", "urgent_following", "crowding_watch",
    "capitulation", "exhaustion", "reversal_candidate",
]

# Selected on the 2017-2022 calibration segment, then checked on the untouched
# 2023+ time segment.  Washout remains an urgency/exhaustion symptom but its
# directional coefficient is zero because it reduced cash-close accuracy.
DIRECTION_WEIGHTS = {
    "night": 1,
    "external": 1,
    "candle": 1,
    "washout": 0,
    "live_intraday": 2,
}


def build_psychology_state(
    payload: dict[str, Any],
    behavior_statistics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    premarket = payload.get("premarket", {})
    night = premarket.get("night_path") or {}
    intraday = payload.get("intraday_tactical_monitor", {})
    candle = payload.get("candlestick_pattern", {})
    washout = payload.get("washout_pattern", {})
    integrated = payload.get("integrated_summary", {})

    bear = 0
    bull = 0
    direction_score = 0
    evidence: list[str] = []

    if night.get("status") == "bearish_extension_close_near_low":
        bear += 3
        direction_score -= 3 * DIRECTION_WEIGHTS["night"]
        evidence.append("夜盤跌幅擴大且收近低點，追跌急迫性升高。")
    elif night.get("status") == "bullish_extension_close_near_high":
        bull += 3
        direction_score += 3 * DIRECTION_WEIGHTS["night"]
        evidence.append("夜盤漲幅擴大且收近高點，追漲急迫性升高。")
    elif night.get("status") == "bearish_but_recovered":
        bear += 1
        direction_score -= DIRECTION_WEIGHTS["night"]
        evidence.append("夜盤偏空但有收回，方向仍在確認階段。")
    elif night.get("status") == "bullish_but_unconfirmed":
        bull += 1
        direction_score += DIRECTION_WEIGHTS["night"]
        evidence.append("夜盤偏多但未形成強確認。")

    external_score = int(premarket.get("external_score") or 0)
    if external_score <= -2:
        bear += 2
        direction_score -= DIRECTION_WEIGHTS["external"]
        evidence.append("海外風險資產同向轉弱，群眾負面確認增加。")
    elif external_score >= 2:
        bull += 2
        direction_score += DIRECTION_WEIGHTS["external"]
        evidence.append("海外風險資產同向轉強，群眾正面確認增加。")

    if candle.get("type") in {"long_bear", "gap_up_failed"}:
        bear += 1
        direction_score -= DIRECTION_WEIGHTS["candle"]
        evidence.append("前一現貨K線偏弱，追跌具有價格記憶。")
    elif candle.get("type") in {"long_bull", "gap_down_reversal"}:
        bull += 1
        direction_score += DIRECTION_WEIGHTS["candle"]
        evidence.append("前一現貨K線偏強，追漲具有價格記憶。")

    if washout.get("type") == "failed_washout":
        bear += 1
        direction_score -= DIRECTION_WEIGHTS["washout"]
        evidence.append("下殺後未有效收回，賣壓尚未證明耗盡。")
    elif washout.get("type") in {"strong_washout", "normal_washout", "gap_down_recovery"}:
        bull += 1
        direction_score += DIRECTION_WEIGHTS["washout"]
        evidence.append("下殺後出現收回，開始具備宣洩或反轉候選條件。")

    code = intraday.get("code")
    if code in {"night_down_cash_break_confirming", "intraday_key_low_break"}:
        bear += 2
        direction_score -= DIRECTION_WEIGHTS["live_intraday"]
        evidence.append("現貨破低確認夜盤壓力，心理路徑向追跌延伸。")
    elif code in {"night_down_cash_reclaim_candidate", "intraday_reclaim_strengthening"}:
        bull += 2
        direction_score += DIRECTION_WEIGHTS["live_intraday"]
        evidence.append("現貨收回夜盤壓力，追空開始面臨反證。")
    elif code == "night_up_cash_failure":
        bear += 2
        direction_score -= DIRECTION_WEIGHTS["live_intraday"]
        evidence.append("夜盤偏多遭現貨否定，追漲路徑失敗。")

    direction = "bearish" if direction_score < 0 else ("bullish" if direction_score > 0 else "mixed")
    urgency = abs(bear - bull)
    has_live = bool(intraday.get("live_usable"))
    recovery = code in {"night_down_cash_reclaim_candidate", "intraday_reclaim_strengthening"}
    failed_recovery = washout.get("type") == "failed_washout" or code == "night_down_cash_break_confirming"

    if recovery and direction != "bearish":
        state = "reversal_candidate"
    elif recovery:
        state = "exhaustion"
    elif failed_recovery and urgency >= 5:
        state = "urgent_following"
    elif urgency >= 5:
        state = "urgent_following"
    elif urgency >= 3:
        state = "confirmation"
    else:
        state = "doubt"

    # Crowding requires at least two fresh, direction-aligned position/breadth
    # observations.  Price alone never confirms it.
    positioning = _position_confirmation(payload.get("psychology_position_context", {}), direction)
    position_confirmation = positioning["confirmed"]
    crowding_status = positioning["status"]
    if position_confirmation:
        evidence.append(positioning["evidence"])
    elif state == "urgent_following":
        crowding_status = "watch_requires_open_interest_and_breadth"

    shock_reasons = []
    if _le(premarket.get("sox_return_1d"), -0.04):
        shock_reasons.append("費半單日跌幅達4%")
    if _le(premarket.get("tsm_adr_return_1d"), -0.03):
        shock_reasons.append("台積電ADR單日跌幅達3%")
    if _ge(premarket.get("vix_return_1d"), 0.10):
        shock_reasons.append("VIX單日上升達10%")
    shock = bool(shock_reasons)

    next_paths = _rank_paths_from_history(
        _next_paths(direction, state, night, has_live),
        behavior_statistics or [], state, direction,
    )
    return {
        "framework": "crowd_psychology_path_v1",
        "state": state,
        "state_label": _state_label(state),
        "direction": direction,
        "bearish_urgency_score": bear,
        "bullish_urgency_score": bull,
        "net_urgency_score": bear - bull,
        "calibrated_direction_score": direction_score,
        "direction_weight_version": "psychology_direction_weights_v2",
        "direction_weights": DIRECTION_WEIGHTS,
        "night_diagnostics": {
            "status": night.get("status"),
            "close_position": night.get("close_position"),
            "recovery_from_low": night.get("recovery_from_low"),
            "material_pressure": bool(night.get("material_pressure")),
        },
        "evidence": evidence,
        "crowding_status": crowding_status,
        "position_confirmation": position_confirmation,
        "positioning_evidence": positioning,
        "exogenous_reset_watch": shock,
        "exogenous_reset_reasons": shock_reasons,
        "next_paths": next_paths,
        "formal_direction_signal": None,
        "validated_probability": None,
        "guardrail": "心理狀態只描述條件路徑；未經前瞻驗證不得轉為正式收盤或多日方向。",
    }


def _rank_paths_from_history(
    paths: list[dict[str, Any]], statistics: list[dict[str, Any]],
    state: str, direction: str,
) -> list[dict[str, Any]]:
    cohort = next(
        (
            item for item in statistics
            if item.get("psychology_state") == state and item.get("direction") == direction
        ),
        None,
    )
    if not cohort or int(cohort.get("cases") or 0) < 30:
        return paths
    rates = {
        str(item.get("path")): float(item.get("rate") or 0)
        for item in cohort.get("path_distribution", [])
    }
    indexed = list(enumerate(paths))
    ranked = sorted(indexed, key=lambda pair: (-rates.get(pair[1].get("code"), -1), pair[0]))
    output = []
    for rank, (_, path) in enumerate(ranked, start=1):
        output.append({
            **path,
            "rank": rank,
            "historical_cases": int(cohort.get("cases") or 0),
            "historical_rate": rates.get(path.get("code")),
            "ranking_source": "resolved_clinical_cases",
        })
    return output


def _position_confirmation(context: dict[str, Any], direction: str) -> dict[str, Any]:
    breadth = _number(context.get("price_advance_decline_breadth"))
    breadth_age = _number(context.get("breadth_age_days"))
    oi_change = _number(context.get("tx_open_interest_change_1d"))
    oi_age = _number(context.get("open_interest_age_days"))
    foreign_net = _number(context.get("foreign_futures_net_oi"))
    foreign_age = _number(context.get("institutional_position_age_days"))
    breadth_aligned = bool(
        breadth is not None and breadth_age is not None and breadth_age <= 5
        and ((direction == "bearish" and breadth <= -0.30) or (direction == "bullish" and breadth >= 0.30))
    )
    foreign_aligned = bool(
        foreign_net is not None and foreign_age is not None and foreign_age <= 3
        and ((direction == "bearish" and foreign_net < 0) or (direction == "bullish" and foreign_net > 0))
    )
    oi_expanding = bool(oi_change is not None and oi_age is not None and oi_age <= 3 and oi_change > 0)
    count = sum([breadth_aligned, foreign_aligned, oi_expanding])
    confirmed = count >= 2
    return {
        "confirmed": confirmed, "confirmation_count": count,
        "breadth_aligned": breadth_aligned, "foreign_position_aligned": foreign_aligned,
        "open_interest_expanding": oi_expanding,
        "status": "confirmed_by_positioning" if confirmed else "unconfirmed",
        "evidence": "廣度、外資期貨部位與未平倉量中至少兩項同向，擁擠條件獲得確認。",
    }


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _next_paths(direction: str, state: str, night: dict, has_live: bool) -> list[dict[str, Any]]:
    low = night.get("low")
    close = night.get("close")
    open_ = night.get("open")
    if direction == "bearish":
        return [
            {"rank": 1, "code": "bearish_continuation", "label": "恐慌／追跌延續",
             "condition": f"現貨跌破夜盤低點 {low:.0f} 且無法快速收回。" if low else "現貨破低且無法快速收回。",
             "effect": "提高收黑與日內續跌風險。"},
            {"rank": 2, "code": "short_covering_rebound", "label": "空單回補但尚未翻多",
             "condition": f"開低後站回夜盤收盤 {close:.0f}，但未站回夜盤開盤 {open_:.0f}。" if close and open_ else "開低後收回部分跌幅。",
             "effect": "降低日內續跌風險，但不等於收盤翻紅。"},
            {"rank": 3, "code": "path_reset_reversal", "label": "反向事件重置",
             "condition": f"站回夜盤開盤 {open_:.0f} 並有現貨廣度與量能確認。" if open_ else "站回夜盤開盤並有廣度確認。",
             "effect": "原追跌路徑失效，轉入反轉候選。"},
        ]
    if direction == "bullish":
        return [
            {"rank": 1, "code": "bullish_continuation", "label": "追漲延續", "condition": "現貨站穩夜盤高位且廣度擴散。", "effect": "提高開高延續風險。"},
            {"rank": 2, "code": "profit_taking", "label": "獲利了結", "condition": "開高後跌回夜盤收盤下方。", "effect": "追價急迫性降級。"},
            {"rank": 3, "code": "bull_trap_reset", "label": "追漲失敗重置", "condition": "跌破夜盤低點且無法收回。", "effect": "原多方路徑失效。"},
        ]
    return [
        {"rank": 1, "code": "range_confirmation", "label": "等待突破確認", "condition": "突破夜盤高低區間。", "effect": "由懷疑階段轉入方向確認。"}
    ]


def _state_label(state: str) -> str:
    return {
        "doubt": "懷疑／尚未形成共識", "confirmation": "方向確認",
        "urgent_following": "急迫追隨", "crowding_watch": "擁擠觀察",
        "capitulation": "恐慌／亢奮宣洩", "exhaustion": "動能衰竭",
        "reversal_candidate": "反轉候選",
    }.get(state, state)


def _le(value: Any, threshold: float) -> bool:
    try:
        return value is not None and float(value) <= threshold
    except (TypeError, ValueError):
        return False


def _ge(value: Any, threshold: float) -> bool:
    try:
        return value is not None and float(value) >= threshold
    except (TypeError, ValueError):
        return False
