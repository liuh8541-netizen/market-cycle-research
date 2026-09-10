"""Persistent pathology, pharmacology, and scoped reference knowledge."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def update_clinical_knowledge(
    payload: dict[str, Any],
    knowledge_dir: str | Path,
) -> dict[str, Any]:
    root = Path(knowledge_dir)
    root.mkdir(parents=True, exist_ok=True)
    case_path = root / "pathology_cases.jsonl"
    profiles = _load_json(root / "reference_profiles.json", [])
    prescriptions = _load_json(root / "prescriptions.json", [])
    existing_cases = load_jsonl(case_path)
    case = build_pathology_case(payload)
    query = case["scope"]

    profile_result = select_reference_profile(query, profiles)
    similar = search_similar_cases(case, existing_cases)
    prescription_result = select_prescription(
        payload.get("market_health", {}).get("diagnosis", {}),
        payload.get("market_health", {}).get("symptoms", []),
        prescriptions,
    )
    appended = append_immutable_case(case_path, case)
    return {
        "framework": "market_clinical_knowledge_v1",
        "case_id": case["case_id"],
        "case_appended": appended,
        "pathology_case_count": len(existing_cases) + int(appended),
        "reference_profile": profile_result,
        "prescription_route": prescription_result,
        "similar_cases": similar,
        "knowledge_dir": str(root),
    }


def build_pathology_case(payload: dict[str, Any]) -> dict[str, Any]:
    health = payload.get("market_health", {})
    diagnosis = health.get("diagnosis", {})
    prescription = health.get("prescription", {})
    symptoms = health.get("symptoms", [])
    input_date = payload.get("input", {}).get("date")
    signal_date = payload.get("index_check", {}).get("signal_date")
    scope = {
        "market": "TWII",
        "timeframe": "daily",
        "target": "lifecycle_health",
        "lifecycle_stage": diagnosis.get("lifecycle_stage", "unknown"),
        "decision_date": input_date,
    }
    evidence = {
        "scope": scope,
        "signal_date": signal_date,
        "symptom_codes": sorted(
            {str(item.get("code")) for item in symptoms if item.get("code")}
        ),
        "diagnosis": diagnosis,
        "prescription": prescription,
        "freshness": payload.get("data_freshness", {}).get("overall_status"),
        "self_repair_status": payload.get("self_repair", {}).get("status"),
    }
    evidence_sha = _fingerprint(evidence)
    return {
        "case_id": f"TWII-{signal_date or input_date}-{evidence_sha[:12]}",
        "schema_version": "1.0",
        "scope": scope,
        "signal_date": signal_date,
        "symptoms": symptoms,
        "diagnosis": diagnosis,
        "prescription": prescription,
        "outcome": {"status": "pending"},
        "evidence_sha256": evidence_sha,
        "immutable": True,
    }


def append_immutable_case(path: str | Path, case: dict[str, Any]) -> bool:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(destination)
    if any(item.get("case_id") == case.get("case_id") for item in existing):
        return False
    lines = [
        json.dumps(item, ensure_ascii=False, sort_keys=True) for item in existing
    ]
    lines.append(json.dumps(case, ensure_ascii=False, sort_keys=True))
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return True


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    output = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            output.append(json.loads(line))
    return output


def select_reference_profile(
    query: dict[str, Any],
    profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    matches = [
        profile
        for profile in profiles
        if _scope_matches(query, profile.get("scope", {}))
    ]
    if len(matches) != 1:
        return {
            "status": "unsupported_scope" if not matches else "ambiguous_scope",
            "profile_id": None,
            "usable": False,
            "reason": (
                "找不到完全符合市場、頻率、目標與段位的常模。"
                if not matches
                else "多份常模同時符合，必須先排除範圍重疊。"
            ),
        }
    profile = matches[0]
    calibrated = profile.get("calibration_status") == "validated"
    return {
        "status": "selected" if calibrated else "pending_validation",
        "profile_id": profile.get("profile_id"),
        "usable": calibrated,
        "version": profile.get("version"),
        "reason": (
            "常模範圍完全符合且已驗證。"
            if calibrated
            else "範圍符合，但常數尚未經獨立病例驗證，只可觀察。"
        ),
    }


def select_prescription(
    diagnosis: dict[str, Any],
    symptoms: list[dict[str, Any]],
    prescriptions: list[dict[str, Any]],
) -> dict[str, Any]:
    symptom_codes = {item.get("code") for item in symptoms}
    query = {
        "diagnosis": diagnosis.get("primary"),
        "confidence": diagnosis.get("confidence"),
        "market": "TWII",
        "timeframe": "daily",
    }
    candidates = []
    contraindicated = []
    for item in prescriptions:
        if not _scope_matches(query, item.get("scope", {})):
            continue
        forbidden = set(item.get("contraindications", []))
        if forbidden & symptom_codes:
            contraindicated.append(item.get("prescription_id"))
            continue
        candidates.append(item)
    if len(candidates) != 1:
        return {
            "status": "no_safe_prescription" if not candidates else "ambiguous_prescription",
            "prescription_id": None,
            "action": "observe_and_recheck",
            "contraindicated": contraindicated,
            "reason": (
                "沒有完全符合診斷、信心及範圍且無禁忌的處方。"
                if not candidates
                else "多張處方同時符合，禁止自動開藥。"
            ),
        }
    item = candidates[0]
    return {
        "status": "selected",
        "prescription_id": item.get("prescription_id"),
        "action": item.get("action"),
        "intensity": item.get("intensity"),
        "contraindicated": contraindicated,
        "reason": "診斷、信心、市場、頻率及禁忌均符合。",
    }


def search_similar_cases(
    query_case: dict[str, Any],
    cases: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    query_scope = query_case.get("scope", {})
    query_symptoms = _symptom_codes(query_case)
    scored = []
    for case in cases:
        scope = case.get("scope", {})
        if any(
            scope.get(key) != query_scope.get(key)
            for key in ("market", "timeframe", "target")
        ):
            continue
        symptoms = _symptom_codes(case)
        union = query_symptoms | symptoms
        similarity = len(query_symptoms & symptoms) / len(union) if union else 1.0
        if scope.get("lifecycle_stage") == query_scope.get("lifecycle_stage"):
            similarity = min(1.0, similarity + 0.15)
        scored.append(
            {
                "case_id": case.get("case_id"),
                "similarity": round(similarity, 6),
                "lifecycle_stage": scope.get("lifecycle_stage"),
                "outcome_status": case.get("outcome", {}).get("status"),
            }
        )
    return sorted(
        scored, key=lambda item: (-item["similarity"], str(item["case_id"]))
    )[:limit]


def _scope_matches(query: dict[str, Any], scope: dict[str, Any]) -> bool:
    required = ("market", "timeframe")
    if any(query.get(key) != scope.get(key) for key in required):
        return False
    for key, expected in scope.items():
        if key in required:
            continue
        actual = query.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _symptom_codes(case: dict[str, Any]) -> set[str]:
    return {
        str(item.get("code"))
        for item in case.get("symptoms", [])
        if item.get("code")
    }


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()

