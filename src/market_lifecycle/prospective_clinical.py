"""Append-only prospective clinical prediction and outcome registry."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


TAIPEI = ZoneInfo("Asia/Taipei")


def update_prospective_registry(
    records: list[dict[str, Any]], prediction_path: str | Path, outcome_path: str | Path,
) -> dict[str, Any]:
    predictions = load_jsonl(prediction_path)
    outcomes = load_jsonl(outcome_path)
    prediction_dates = {item.get("decision_date") for item in predictions}
    outcome_dates = {item.get("decision_date") for item in outcomes}
    by_date = {item.get("identity", {}).get("decision_date"): item for item in records}
    latest = records[-1] if records else None
    prediction_appended = False
    outcome_appended = 0

    if latest and latest.get("outcome", {}).get("status") == "pending":
        decision_date = latest["identity"]["decision_date"]
        if decision_date not in prediction_dates:
            entry = prediction_entry(latest, predictions[-1].get("entry_hash") if predictions else None)
            append_jsonl(prediction_path, entry)
            predictions.append(entry)
            prediction_dates.add(decision_date)
            prediction_appended = True

    for prediction in predictions:
        decision_date = prediction.get("decision_date")
        resolved = by_date.get(decision_date)
        if decision_date in outcome_dates or not resolved or resolved.get("outcome", {}).get("status") != "resolved":
            continue
        entry = outcome_entry(
            prediction, resolved, outcomes[-1].get("entry_hash") if outcomes else None
        )
        append_jsonl(outcome_path, entry)
        outcomes.append(entry)
        outcome_dates.add(decision_date)
        outcome_appended += 1

    return summarize_registry(predictions, outcomes, prediction_appended, outcome_appended)


def prediction_entry(case: dict[str, Any], previous_hash: str | None) -> dict[str, Any]:
    diagnosis = case.get("diagnosis", {})
    entry = {
        "registry_version": "prospective_market_clinical_v1",
        "entry_type": "locked_prediction", "locked_at": now(),
        "decision_date": case.get("identity", {}).get("decision_date"),
        "case_id": case.get("case_id"), "evidence_sha256": case.get("evidence_sha256"),
        "previous_hash": previous_hash,
        "psychology_state": diagnosis.get("psychology_state"),
        "direction": diagnosis.get("direction"),
        "direction_weight_version": diagnosis.get("direction_weight_version"),
        "calibrated_direction_score": diagnosis.get("calibrated_direction_score"),
        "behavior_phenotype": diagnosis.get("behavior_phenotype"),
        "market_regime": diagnosis.get("market_regime"),
        "differential_diagnosis": case.get("differential_diagnosis", []),
        "prognosis": case.get("prognosis_from_prior_cases", {}),
        "prescription": case.get("prescription", {}),
        "formal_direction_signal": None,
        "immutable": True,
    }
    entry["entry_hash"] = fingerprint(entry)
    return entry


def outcome_entry(
    prediction: dict[str, Any], resolved_case: dict[str, Any], previous_hash: str | None,
) -> dict[str, Any]:
    entry = {
        "registry_version": "prospective_market_clinical_v1",
        "entry_type": "settled_outcome", "settled_at": now(),
        "decision_date": prediction.get("decision_date"),
        "prediction_entry_hash": prediction.get("entry_hash"),
        "prediction_case_id": prediction.get("case_id"),
        "resolved_case_id": resolved_case.get("case_id"),
        "previous_hash": previous_hash,
        "outcome": resolved_case.get("outcome", {}),
        "follow_up": resolved_case.get("follow_up", {}),
        "immutable": True,
    }
    entry["entry_hash"] = fingerprint(entry)
    return entry


def summarize_registry(
    predictions: list[dict[str, Any]], outcomes: list[dict[str, Any]],
    prediction_appended: bool = False, outcome_appended: int = 0,
) -> dict[str, Any]:
    settled = [item.get("outcome", {}) for item in outcomes]
    actionable = [item for item in settled if item.get("direction_hit") is not None]
    return {
        "framework": "prospective_market_clinical_v1",
        "prediction_count": len(predictions), "settled_count": len(outcomes),
        "pending_count": max(0, len(predictions) - len(outcomes)),
        "prediction_appended": prediction_appended, "outcomes_appended": outcome_appended,
        "direction_accuracy": mean_bool(actionable, "direction_hit"),
        "top_path_accuracy": mean_bool(actionable, "top_path_hit"),
        "material_move_rate": mean_bool(actionable, "material_direction_confirmation"),
        "prediction_chain_valid": validate_chain(predictions),
        "outcome_chain_valid": validate_chain(outcomes),
        "promotion_gate": {
            "minimum_settled_cases": 100, "minimum_accuracy": 0.80,
            "eligible": len(actionable) >= 100 and (mean_bool(actionable, "direction_hit") or 0) >= 0.80,
        },
        "guardrail": "預測與結果分開追加；既有紀錄不回寫。達門檻仍需獨立審核，不能自動升格。",
    }


def validate_chain(entries: list[dict[str, Any]]) -> bool:
    previous = None
    for entry in entries:
        if entry.get("previous_hash") != previous:
            return False
        expected = fingerprint({key: value for key, value in entry.items() if key != "entry_hash"})
        if entry.get("entry_hash") != expected:
            return False
        previous = entry.get("entry_hash")
    return True


def append_jsonl(path: str | Path, entry: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_registry(summary: dict[str, Any]) -> str:
    return "\n".join([
        "# 前瞻市場病例登錄", "",
        f"- 已鎖定預測：{summary['prediction_count']}件",
        f"- 已結案：{summary['settled_count']}件；待追蹤：{summary['pending_count']}件",
        f"- 收盤方向命中：{pct(summary.get('direction_accuracy'))}",
        f"- 首選病程命中：{pct(summary.get('top_path_accuracy'))}",
        f"- 顯著同向幅度：{pct(summary.get('material_move_rate'))}",
        f"- 預測雜湊鏈：{'通過' if summary['prediction_chain_valid'] else '失敗'}",
        f"- 結果雜湊鏈：{'通過' if summary['outcome_chain_valid'] else '失敗'}", "",
        f"> {summary['guardrail']}", "",
    ])


def fingerprint(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def mean_bool(rows: list[dict[str, Any]], key: str):
    values = [bool(item.get(key)) for item in rows if item.get(key) is not None]
    return sum(values) / len(values) if values else None


def now() -> str:
    return datetime.now(TAIPEI).isoformat(timespec="seconds")


def pct(value) -> str:
    return "NA" if value is None else f"{float(value):.2%}"
