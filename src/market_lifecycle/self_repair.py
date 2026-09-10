"""Auditable self-detection and repair governance.

Safe repairs may block output, lower authority, or request rechecks. Model
parameters, thresholds, scopes, and historical records are never changed here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


PROTECTED_CHANGE_TYPES = {
    "model_parameter_change",
    "threshold_change",
    "scope_change",
    "historical_record_change",
    "locked_experiment_rerun",
}


def build_self_repair_assessment(payload: dict[str, Any]) -> dict[str, Any]:
    evidence = _evidence_snapshot(payload)
    incidents = _detect_incidents(payload)
    automatic_actions = _safe_automatic_actions(payload, incidents)
    candidates = _repair_candidates(incidents)
    return {
        "framework": "auditable_self_repair_v1",
        "evidence_sha256": _fingerprint(evidence),
        "evidence": evidence,
        "incidents": incidents,
        "automatic_actions": automatic_actions,
        "repair_candidates": candidates,
        "governance": {
            "original_records_immutable": True,
            "automatic_model_tuning": False,
            "automatic_threshold_tuning": False,
            "automatic_scope_expansion": False,
            "independent_validation_required": True,
            "protected_change_types": sorted(PROTECTED_CHANGE_TYPES),
        },
        "status": _status(incidents, candidates),
    }


def validate_repair_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    change_type = proposal.get("change_type")
    blocked = change_type in PROTECTED_CHANGE_TYPES
    return {
        "allowed_automatically": not blocked,
        "requires_independent_validation": blocked,
        "reason": (
            "此變更可能造成事後調參或改寫適用範圍，只能建立候選版本並獨立驗證。"
            if blocked
            else "此變更只限制輸出或要求複查，可安全自動執行。"
        ),
    }


def _evidence_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    review = payload.get("self_review", {})
    return {
        "input_date": payload.get("input", {}).get("date"),
        "signal_date": payload.get("index_check", {}).get("signal_date"),
        "freshness": payload.get("data_freshness", {}).get("overall_status"),
        "logic_contradictions": payload.get("logic_consistency", {}).get(
            "contradiction_count", 0
        ),
        "health_diagnosis": payload.get("market_health", {}).get("diagnosis", {}),
        "health_prescription": payload.get("market_health", {}).get(
            "prescription", {}
        ),
        "matured_checks": review.get("matured_checks", {}).get("rows", []),
        "production_direction_enabled": payload.get("production_policy", {})
        .get("main_multi_day_direction", {})
        .get("enabled", False),
    }


def _detect_incidents(payload: dict[str, Any]) -> list[dict[str, Any]]:
    incidents: list[dict[str, Any]] = []
    freshness = payload.get("data_freshness", {}).get("overall_status")
    contradictions = int(
        payload.get("logic_consistency", {}).get("contradiction_count", 0)
    )
    policy = payload.get("production_policy", {}).get(
        "main_multi_day_direction", {}
    )
    forecast = payload.get("forecast", {})
    matured = (
        payload.get("self_review", {}).get("matured_checks", {}).get("rows", [])
    )

    if freshness != "current":
        incidents.append(
            _incident(
                "data_quality_error",
                "high",
                f"資料時效狀態為 {freshness or 'unknown'}。",
                "data_pipeline",
            )
        )
    if contradictions:
        incidents.append(
            _incident(
                "diagnostic_conflict",
                "high",
                f"偵測到 {contradictions} 項硬矛盾。",
                "diagnosis",
            )
        )
    if not policy.get("enabled") and forecast.get("formal_direction_signal") is not None:
        incidents.append(
            _incident(
                "scope_authority_error",
                "critical",
                "未驗證多日方向出現非空正式訊號。",
                "output_policy",
            )
        )

    missed = [row for row in matured if row.get("is_hit") is False]
    if missed:
        by_horizon: dict[str, int] = {}
        for row in missed:
            horizon = str(row.get("horizon_days", "unknown"))
            by_horizon[horizon] = by_horizon.get(horizon, 0) + 1
        incidents.append(
            {
                **_incident(
                    "outcome_mismatch",
                    "warning",
                    f"到期病例有 {len(missed)} 件與原探索性方向不符。",
                    "diagnosis",
                ),
                "details": {
                    "missed_count": len(missed),
                    "by_horizon": by_horizon,
                    "record_keys": [
                        {
                            "signal_date": row.get("signal_date"),
                            "target_date": row.get("target_date"),
                            "horizon_days": row.get("horizon_days"),
                        }
                        for row in missed
                    ],
                },
            }
        )
    return incidents


def _incident(code: str, severity: str, evidence: str, owner: str) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "evidence": evidence,
        "owner": owner,
        "historical_record_mutation_allowed": False,
    }


def _safe_automatic_actions(
    payload: dict[str, Any], incidents: list[dict[str, Any]]
) -> list[dict[str, str]]:
    codes = {item["code"] for item in incidents}
    actions: list[dict[str, str]] = []
    if "scope_authority_error" in codes:
        actions.append(
            {
                "action": "block_formal_direction_output",
                "effect": "將未驗證正式方向輸出強制視為null。",
            }
        )
    if {"data_quality_error", "diagnostic_conflict"} & codes:
        actions.append(
            {
                "action": "force_observe_and_recheck",
                "effect": "停止方向處置，等待完整資料或矛盾解除。",
            }
        )
    if "outcome_mismatch" in codes:
        actions.append(
            {
                "action": "append_case_review",
                "effect": "保存錯誤病例並建立根因分析候選，不修改舊預判。",
            }
        )
    if not actions:
        actions.append(
            {
                "action": "continue_monitoring",
                "effect": "未發現需立即修護的錯誤，持續累積不可回寫病例。",
            }
        )
    return actions


def _repair_candidates(incidents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for incident in incidents:
        code = incident["code"]
        if code == "data_quality_error":
            candidates.append(
                _candidate(
                    code,
                    "data_availability_guard",
                    "safe_guard_change",
                    "加強available_at、is_complete與來源日期檢查。",
                )
            )
        elif code == "diagnostic_conflict":
            candidates.append(
                _candidate(
                    code,
                    "differential_diagnosis_router",
                    "safe_guard_change",
                    "矛盾時拒絕單一診斷並要求鑑別流程。",
                )
            )
        elif code == "scope_authority_error":
            candidates.append(
                _candidate(
                    code,
                    "target_scope_router",
                    "scope_change",
                    "檢查問題目標與模型適用範圍，越權時拒絕輸出。",
                )
            )
        elif code == "outcome_mismatch":
            candidates.append(
                _candidate(
                    code,
                    "mismatch_root_cause_study",
                    "model_parameter_change",
                    "依題型、病因、段位、嚴重度與處方分類錯誤，只建立候選假設。",
                )
            )
    return candidates


def _candidate(
    incident_code: str,
    candidate_id: str,
    change_type: str,
    description: str,
) -> dict[str, Any]:
    proposal = {"change_type": change_type}
    validation = validate_repair_proposal(proposal)
    return {
        "incident_code": incident_code,
        "candidate_id": candidate_id,
        "change_type": change_type,
        "description": description,
        "automatic_promotion_allowed": validation["allowed_automatically"],
        "independent_validation_required": validation[
            "requires_independent_validation"
        ],
    }


def _status(
    incidents: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> str:
    if any(item["severity"] == "critical" for item in incidents):
        return "protective_block"
    if incidents:
        return "review_required"
    if candidates:
        return "candidate_pending"
    return "monitoring"


def _fingerprint(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()

