from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OVERLAY_JSON = REPORTS / "one_day_psychology_overlay_calibration.json"
PSYCHOLOGY_JSON = REPORTS / "psychology_state_backtest.json"
FORECAST_JSON = REPORTS / "today_market_forecast.json"
OUTPUT_JSON = REPORTS / "semantic_psychology_layer_audit.json"
OUTPUT_MD = REPORTS / "semantic_psychology_layer_audit.md"


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def pp(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:+.2f} pp"


def main() -> None:
    overlay = load_json(OVERLAY_JSON, {})
    psychology = load_json(PSYCHOLOGY_JSON, {})
    forecast = load_json(FORECAST_JSON, {})

    holdout = overlay.get("comparison", {}).get("holdout_2023_plus", {})
    baseline = holdout.get("baseline", {})
    reconciled = holdout.get("reconciled", {})
    baseline_accuracy = baseline.get("accuracy")
    reconciled_accuracy = reconciled.get("accuracy")
    accuracy_edge = (
        reconciled_accuracy - baseline_accuracy
        if baseline_accuracy is not None and reconciled_accuracy is not None
        else None
    )
    baseline_macro = baseline.get("macro_recall")
    reconciled_macro = reconciled.get("macro_recall")
    macro_edge = (
        reconciled_macro - baseline_macro
        if baseline_macro is not None and reconciled_macro is not None
        else None
    )

    psych_overall = psychology.get("overall", {})
    urgent = psychology.get("urgent_following", {})
    current_layer = forecast.get("psychological_warfare_pattern", {})

    semantic_rows = current_layer.get("semantic_quantification", [])
    current_score = current_layer.get("score")
    current_policy = current_layer.get("action_policy")

    historical_effect = "improved" if accuracy_edge is not None and accuracy_edge > 0 else "not_improved"
    prospective_status = "not_yet_validated"
    promotion_allowed = False
    root_causes = []
    if historical_effect != "improved":
        root_causes.extend(
            [
                "心理文字與價格結果可能不同步，單靠語意分類無法捕捉資金實際承接。",
                "夜盤、外部市場、K線與族群廣度若互相衝突，心理層會變成雜訊。",
            ]
        )
    root_causes.extend(
        [
            "新增文字轉數字層是在今日才正式落地，尚無獨立前瞻樣本。",
            "目前它主要提升解讀、降權與風控處理，不直接改寫正式多日方向。",
            "必須累積每日封存預測，再用未來實際收盤驗證，才能判斷是否真正增進。",
        ]
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "framework": "semantic_psychology_layer_audit_v1",
        "historical_overlay_effect": {
            "status": historical_effect,
            "scope": "one-day psychology overlay, holdout 2023+",
            "cases": baseline.get("cases") or reconciled.get("cases"),
            "baseline_accuracy": baseline_accuracy,
            "reconciled_accuracy": reconciled_accuracy,
            "accuracy_edge": accuracy_edge,
            "baseline_macro_recall": baseline_macro,
            "reconciled_macro_recall": reconciled_macro,
            "macro_recall_edge": macro_edge,
        },
        "psychology_state_backtest": {
            "overall_direction_accuracy": psych_overall.get("direction_accuracy"),
            "overall_cases": psych_overall.get("cases"),
            "urgent_following_accuracy": urgent.get("direction_accuracy"),
            "urgent_following_cases": urgent.get("cases"),
        },
        "current_semantic_layer": {
            "label": current_layer.get("label"),
            "stance": current_layer.get("stance"),
            "score": current_score,
            "action_policy": current_policy,
            "semantic_quantification": semantic_rows,
        },
        "prospective_validation": {
            "status": prospective_status,
            "promotion_allowed": promotion_allowed,
            "reason": "新語意量化層尚未經未來資料驗證；只可作診斷、風控與假設留底。",
            "required_checks": [
                "每日收盤後封存心理戰分數與處理原則。",
                "隔日核對方向、關鍵線、量能、族群廣度與日夜盤是否吻合。",
                "累積至少30筆後比較加入語意層前後的1日方向、風控命中與誤報率。",
                "若命中率未提升，回查是語意分數方向錯、權重錯、資料缺口，或市場 regime 改變。",
            ],
        },
        "root_cause_if_no_improvement": root_causes,
        "guardrail": "此稽核只評估模型層級效果，不產生投資命令；回測提升不得回填成既成預測。",
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 語意心理層實質效果稽核",
        "",
        f"- 產生時間: {payload['generated_at']}",
        f"- 框架: {payload['framework']}",
        f"- 結論: 既有一日心理 overlay 在歷史留出樣本有增進；新增文字轉數字層需前瞻驗證，暫不可升級為正式方向權重。",
        "",
        "## 歷史效果",
        "",
        f"- 範圍: {payload['historical_overlay_effect']['scope']}",
        f"- 樣本數: {payload['historical_overlay_effect']['cases']}",
        f"- 基準準確率: {pct(baseline_accuracy)}",
        f"- 加入心理 overlay 後: {pct(reconciled_accuracy)}",
        f"- 準確率增減: {pp(accuracy_edge)}",
        f"- 基準 macro recall: {pct(baseline_macro)}",
        f"- overlay macro recall: {pct(reconciled_macro)}",
        f"- macro recall 增減: {pp(macro_edge)}",
        "",
        "## 心理狀態回測",
        "",
        f"- 整體方向準確率: {pct(psych_overall.get('direction_accuracy'))}；樣本 {psych_overall.get('cases')}",
        f"- 急迫追隨準確率: {pct(urgent.get('direction_accuracy'))}；樣本 {urgent.get('cases')}",
        "",
        "## 今日新增語意層",
        "",
        f"- 標籤: {current_layer.get('label', 'NA')}",
        f"- 姿態: {current_layer.get('stance', 'NA')}",
        f"- 分數: {current_score}",
        f"- 處理原則: {current_policy}",
        "",
        "| 文字訊號 | 分數 | 模型意義 |",
        "| --- | ---: | --- |",
    ]
    for row in semantic_rows:
        lines.append(f"| {row.get('text', '')} | {row.get('score', 0)} | {row.get('meaning', '')} |")
    lines.extend(
        [
            "",
            "## 是否已證明增進",
            "",
            "- 已證明: 舊的一日心理 overlay 對歷史留出樣本有增進。",
            "- 尚未證明: 新增的文字轉數字/兵法八卦語意層，因為今天才落地，還沒有獨立前瞻樣本。",
            "- 目前用途: 診斷、風控、降權、病因假設與隔日驗證，不直接改寫正式方向。",
            "",
            "## 若後續無增進，優先查原因",
        ]
    )
    for item in root_causes:
        lines.append(f"- {item}")
    lines.extend(["", f"- 防呆: {payload['guardrail']}"])
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Report: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
