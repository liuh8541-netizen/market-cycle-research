"""Write an auditable analysis of the latest resolved one-day forecast miss."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_lifecycle.probability_forecast import ONE_DAY_PSYCHOLOGY_OVERLAY


ERROR_REVIEW = ROOT / "reports" / "error_review.json"
PSYCHOLOGY_REPLAY = ROOT / "reports" / "psychology_state_backtest.csv"
OUTPUT_JSON = ROOT / "reports" / "previous_day_forecast_error_analysis.json"
OUTPUT_MD = ROOT / "reports" / "previous_day_forecast_error_analysis.md"
PATHOLOGY_LEDGER_JSONL = ROOT / "reports" / "forecast_error_pathology_ledger.jsonl"
PATHOLOGY_LEDGER_MD = ROOT / "reports" / "forecast_error_pathology_ledger.md"


def latest_one_day_miss() -> dict:
    review = json.loads(ERROR_REVIEW.read_text(encoding="utf-8"))
    cases = [item for item in review.get("cases", []) if item.get("horizon_days") == 1]
    cases = [
        item for item in cases
        if str(item.get("forecast_date") or "") <= str(item.get("target_date") or "")
    ]
    if not cases:
        return {}
    return sorted(cases, key=lambda item: (item.get("target_date", ""), item.get("forecast_date", "")))[-1]


def psychology_row(target_date: str) -> dict:
    replay = pd.read_csv(PSYCHOLOGY_REPLAY)
    row = replay.loc[replay["signal_date"].astype(str).eq(target_date)]
    return row.iloc[-1].to_dict() if len(row) else {}


def finite(value):
    return None if value is None or pd.isna(value) else value


def pathology_name(code: str | None) -> str:
    mapping = {
        "bearish_reversal_absorption_mismatch": "偏空壓力被現貨強承接反轉",
        "premarket_gap_mismatch": "盤前缺口外推成收盤方向",
        "neutral_threshold_mismatch": "盤整區誤判成方向盤",
        "bagua_phase_mismatch": "卦位轉換過早",
        "high_level_reversal_mismatch": "高檔轉弱低估",
        "kline_confirmation_mismatch": "K線確認不足",
        "washout_mismatch": "洗盤真假判讀錯配",
        "cycle_position_mismatch": "波段位置錯配",
        "unclassified_market_noise": "未分類市場雜訊",
    }
    return mapping.get(code or "", "未分類預測病灶")


def build_pathology_case(miss: dict, psychology: dict, payload: dict) -> dict:
    code = miss.get("primary_error_code")
    context = miss.get("context", {})
    reasons = miss.get("reasons", [])
    core_factors = []
    if miss.get("predicted_direction") == "down" and miss.get("actual_direction") == "up":
        core_factors.extend([
            "偏空預測方向與實際強漲相反。",
            "夜盤接近0%時不可寫成偏空；盤前或外部壓力不能直接外推成日盤收盤方向。",
            "日盤現貨承接、權值股改價或空方回補可能主導收盤。",
        ])
    if context.get("washout"):
        core_factors.append(f"前一基準日洗盤標籤為 {context.get('washout')}，需驗證是否其實是清洗後反攻。")
    if context.get("kline"):
        core_factors.append(f"前一基準日K線為 {context.get('kline')}，需檢查K線是否被誤讀。")

    evidence = {
        "miss": miss,
        "psychology": {
            "state": psychology.get("state"),
            "direction": psychology.get("direction"),
            "calibrated_direction_score": finite(psychology.get("calibrated_direction_score")),
            "night_close_position": finite(psychology.get("night_close_position")),
            "night_recovery_from_low": finite(psychology.get("night_recovery_from_low")),
            "prior_candle_type": psychology.get("prior_candle_type"),
        },
        "error_reasons": reasons,
        "fixed_adjustment": payload.get("fixed_adjustment", {}),
    }
    digest = hashlib.sha256(
        json.dumps(evidence, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12].upper()
    return {
        "case_id": f"ERR-TWII-{miss.get('target_date')}-{code}-{digest}",
        "schema_version": "forecast_error_pathology_v1",
        "market": "TWII",
        "timeframe": "daily",
        "forecast_date": miss.get("forecast_date"),
        "signal_date": miss.get("signal_date"),
        "target_date": miss.get("target_date"),
        "horizon_days": miss.get("horizon_days"),
        "pathology_code": code,
        "pathology_name": pathology_name(code),
        "severity": error_severity(miss),
        "predicted_direction": miss.get("predicted_direction"),
        "actual_direction": miss.get("actual_direction"),
        "actual_return": miss.get("actual_return"),
        "root_cause_summary": root_cause_summary(miss, context),
        "core_factors": core_factors,
        "candidate_adjustments": [
            item.get("candidate_adjustment")
            for item in reasons
            if item.get("candidate_adjustment")
        ],
        "next_validation": next_validation_items(code),
        "evidence": evidence,
        "governance": {
            "historical_prediction_rewritten": False,
            "automatic_parameter_tuning": False,
            "independent_validation_required": True,
            "use": "病例留底、同類錯誤追蹤、降權與候選規則前瞻驗證。",
        },
        "immutable": True,
    }


def error_severity(miss: dict) -> str:
    actual_return = abs(float(miss.get("actual_return") or 0))
    if miss.get("predicted_direction") != miss.get("actual_direction") and actual_return >= 0.015:
        return "high"
    if actual_return >= 0.008:
        return "medium"
    return "low"


def root_cause_summary(miss: dict, context: dict) -> str:
    code = miss.get("primary_error_code")
    if code == "bearish_reversal_absorption_mismatch":
        return (
            "模型把盤前/外部偏空或舊偏空假設視為收盤方向，但夜盤接近0%並未給明確方向，"
            "實際日盤由現貨承接、空方回補與權值股改價主導強收；此病灶核心是日盤自主改價確認不足。"
        )
    if code == "premarket_gap_mismatch":
        return "模型把盤前缺口訊號外推到收盤方向，未充分等待日盤現貨驗證。"
    if code == "neutral_threshold_mismatch":
        return "模型把接近盤整門檻的波動硬判成方向，需降低方向語氣。"
    return f"目前歸因仍需累積病例；前一狀態為 {context.get('scenario', 'NA')}。"


def next_validation_items(code: str | None) -> list[str]:
    if code == "bearish_reversal_absorption_mismatch":
        return [
            "隔日檢查是否續站前一日強收區，若續站，偏空吸收反轉成立。",
            "隔日檢查是否跌回反轉日低點，若跌破且收不回，判定為假反轉。",
            "後續同類病例需統計夜盤近0但日盤強收後的1/3/5日方向。",
        ]
    if code == "premarket_gap_mismatch":
        return [
            "隔日只用夜盤判斷開盤壓力，不用夜盤單獨判斷收盤。",
            "核對開盤缺口是否被日盤現貨完全反向改價。",
        ]
    return ["下一交易日核對同類錯誤是否重複，未滿樣本前不自動新增正式規則。"]


def append_jsonl_immutable(path: Path, case: dict) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        existing = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if any(item.get("case_id") == case.get("case_id") for item in existing):
        return False
    lines = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in existing]
    lines.append(json.dumps(case, ensure_ascii=False, sort_keys=True))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return True


def render_pathology_ledger() -> None:
    if not PATHOLOGY_LEDGER_JSONL.exists():
        PATHOLOGY_LEDGER_MD.write_text("# 預測失準病歷表\n\n尚無病例。\n", encoding="utf-8")
        return
    cases = [
        json.loads(line)
        for line in PATHOLOGY_LEDGER_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups: dict[str, int] = {}
    for case in cases:
        key = case.get("pathology_name", "未分類")
        groups[key] = groups.get(key, 0) + 1
    lines = [
        "# 預測失準病歷表",
        "",
        f"- 病例數: {len(cases)}",
        "- 治理: 不回寫舊預測；只做病因歸納、降權、候選規則與前瞻驗證。",
        "",
        "## 病灶統計",
        "",
        "| 病灶 | 次數 |",
        "| --- | ---: |",
    ]
    for name, count in sorted(groups.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {name} | {count} |")
    lines.extend([
        "",
        "## 最近病例",
        "",
        "| 到期日 | 預測 | 實際 | 報酬 | 病灶 | 嚴重度 | 根因摘要 |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ])
    for case in cases[-20:]:
        lines.append(
            f"| {case.get('target_date')} | {case.get('predicted_direction')} | "
            f"{case.get('actual_direction')} | {float(case.get('actual_return') or 0):.2%} | "
            f"{case.get('pathology_name')} | {case.get('severity')} | "
            f"{case.get('root_cause_summary')} |"
        )
    PATHOLOGY_LEDGER_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    miss = latest_one_day_miss()
    if not miss:
        OUTPUT_JSON.write_text(json.dumps({
            "framework": "previous_day_fixed_factor_review_v1",
            "available": False,
            "summary": "沒有已到期的一日失準病例。",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        OUTPUT_MD.write_text("# 前一日預測誤差與固定因數修正\n\n沒有已到期的一日失準病例。\n", encoding="utf-8")
        render_pathology_ledger()
        print("No resolved one-day miss is available.")
        return
    psychology = psychology_row(miss.get("signal_date") or miss["target_date"])
    score = float(psychology.get("calibrated_direction_score") or 0)
    close_position = float(psychology.get("night_close_position") or 0)
    recovery = float(psychology.get("night_recovery_from_low") or 0)
    overlay_applies = abs(score) >= ONE_DAY_PSYCHOLOGY_OVERLAY["minimum_absolute_score"]
    tail_watch = bool(
        psychology.get("direction") == "bearish"
        and close_position <= 0.25 and recovery <= 0.35
    )
    payload = {
        "framework": "previous_day_fixed_factor_review_v1",
        "case": miss,
        "error_factors": [
            {
                "code": "coarse_generic_similarity",
                "finding": "舊一日模型只以生命週期、風險狀態與總分區間投票，未納入當晚心理方向。",
                "adjustable": True,
            },
            {
                "code": "cross_model_conflict_not_reconciled",
                "finding": f"歷史分布判 {miss['predicted_direction']}，心理模型判 {psychology.get('direction')}；舊程式沒有固定仲裁規則。",
                "adjustable": True,
            },
            {
                "code": "weak_bearish_tail_omitted",
                "finding": f"夜盤收盤位置 {close_position:.3f}、低點回收 {recovery:.3f}，形成弱分數但明顯的下行尾部警示。",
                "adjustable": True,
            },
            {
                "code": "unpredictable_residual",
                "finding": (
                    f"心理固定分數為 {score:.0f}；即使達到覆寫門檻，"
                    "仍不得因單一錯例回寫歷史或任意調門檻，避免過度擬合。"
                ),
                "adjustable": False,
            },
        ],
        "psychology_evidence": {
            "state": psychology.get("state"),
            "direction": psychology.get("direction"),
            "calibrated_direction_score": score,
            "night_close_position": close_position,
            "night_recovery_from_low": recovery,
            "prior_candle_type": psychology.get("prior_candle_type"),
        },
        "fixed_adjustment": {
            "version": ONE_DAY_PSYCHOLOGY_OVERLAY["version"],
            "minimum_absolute_score": ONE_DAY_PSYCHOLOGY_OVERLAY["minimum_absolute_score"],
            "overlay_would_have_applied_to_this_case": overlay_applies,
            "weak_bearish_tail_watch_would_have_applied": tail_watch,
            "retrospective_prediction_rewritten": False,
            "validation": ONE_DAY_PSYCHOLOGY_OVERLAY["validation"],
        },
        "governance": "保留原錯誤；只修改未來固定仲裁與警示規則。",
    }
    pathology_case = build_pathology_case(miss, psychology, payload)
    payload["pathology_case"] = pathology_case
    payload["pathology_case_appended"] = append_jsonl_immutable(PATHOLOGY_LEDGER_JSONL, pathology_case)
    render_pathology_ledger()
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = payload["fixed_adjustment"]["validation"]
    lines = [
        "# 前一日預測誤差與固定因數修正", "",
        f"- 預測日／到期日：{miss['forecast_date']}／{miss['target_date']}",
        f"- 原預測：{miss['predicted_direction']}；實際：{miss['actual_direction']}；報酬：{miss['actual_return']:.2%}",
        f"- 心理模型：{psychology.get('state')} / {psychology.get('direction')}；固定方向分數 {score:.0f}",
        f"- 夜盤收盤位置：{close_position:.3f}；低點回收：{recovery:.3f}", "",
        "## 誤差因素", "",
    ]
    for item in payload["error_factors"]:
        lines.append(f"- {item['finding']}")
    lines.extend([
        "", "## 已採固定修正", "",
        f"- 一日心理覆寫門檻固定為絕對分數 ≥ {ONE_DAY_PSYCHOLOGY_OVERLAY['minimum_absolute_score']}。",
        "- 門檻內改用 2017–2022 校準期的心理條件分布；5、20、60 日完全不覆寫。",
        "- 未達門檻但夜盤收近低點、回收不足時，只增加下行尾部警示。",
        f"- 2023+ 留出期：原一日準確率 {validation['baseline_2023_plus_accuracy']:.2%}，修正後 {validation['reconciled_2023_plus_accuracy']:.2%}。",
        f"- 2023+ 宏觀召回率：{validation['baseline_2023_plus_macro_recall']:.2%} → {validation['reconciled_2023_plus_macro_recall']:.2%}。",
        "- 原預測不回寫，避免用結果污染病例。", "",
        "## 失準病歷歸檔", "",
        f"- 病例ID：{pathology_case['case_id']}",
        f"- 病灶：{pathology_case['pathology_name']}（{pathology_case['pathology_code']}）",
        f"- 嚴重度：{pathology_case['severity']}",
        f"- 根因摘要：{pathology_case['root_cause_summary']}",
        f"- 是否新增入病歷表：{'是' if payload['pathology_case_appended'] else '否，已存在'}",
        "- 病歷表：reports/forecast_error_pathology_ledger.md / reports/forecast_error_pathology_ledger.jsonl",
        "",
    ])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload["fixed_adjustment"], ensure_ascii=False, indent=2))
    print(f"Report: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
