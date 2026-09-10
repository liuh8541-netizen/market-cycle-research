from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from predict_market import analyze_external_event_reset


REPORTS = ROOT / "reports"
FORECAST_JSON = REPORTS / "today_market_forecast.json"
OUTPUT_JSON = REPORTS / "global_news_risk_impact_audit.json"
OUTPUT_MD = REPORTS / "global_news_risk_impact_audit.md"


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def effect_label(before: dict, after: dict) -> str:
    if not before.get("reset_active") and after.get("reset_active"):
        return "新聞風險使外部事件重置由未觸發升級為觸發"
    if before.get("reset_active") and after.get("reset_active"):
        delta = int(after.get("reset_score", 0)) - int(before.get("reset_score", 0))
        if delta > 0:
            return "新聞風險強化既有外部利空重置"
        if delta < 0:
            return "新聞風險抵銷部分既有外部壓力"
        return "新聞風險未改變外部重置分數"
    if before.get("reset_active") and not after.get("reset_active"):
        return "新聞利多使外部重置降級"
    return "新聞風險未使外部事件重置升級"


def main() -> None:
    forecast = load_json(FORECAST_JSON, {})
    premarket = forecast.get("premarket", {})
    news = forecast.get("global_news_risk", {})
    after = forecast.get("external_event_reset_monitor", {})
    before = analyze_external_event_reset(
        premarket,
        {
            "status": "counterfactual_disabled",
            "risk_score": 0,
            "tailwind_score": 0,
            "net_risk_score": 0,
            "summary": "反事實測試：不納入國際新聞風險。",
        },
    )

    score_delta = int(after.get("reset_score", 0)) - int(before.get("reset_score", 0))
    risk_delta = int(after.get("risk_adjustment", 0)) - int(before.get("risk_adjustment", 0))
    added_reasons = [
        item for item in after.get("reasons", [])
        if item not in before.get("reasons", [])
    ]
    changed_fields = []
    for key in ("code", "label", "direction", "causal_priority", "reset_active"):
        if before.get(key) != after.get(key):
            changed_fields.append(
                {"field": key, "without_news": before.get(key), "with_news": after.get(key)}
            )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "framework": "global_news_risk_impact_audit_v1",
        "forecast_date": forecast.get("input", {}).get("date"),
        "latest_spot_date": forecast.get("index_check", {}).get("signal_date"),
        "news_status": news.get("status"),
        "news_summary": news.get("summary"),
        "effect": effect_label(before, after),
        "without_news": before,
        "with_news": after,
        "score_delta": score_delta,
        "risk_adjustment_delta": risk_delta,
        "changed_fields": changed_fields,
        "added_reasons": added_reasons,
        "concrete_impact": [
            f"外部重置分數變化: {before.get('reset_score', 0)} -> {after.get('reset_score', 0)}，差異 {score_delta:+d}。",
            f"因果優先權: {before.get('causal_priority')} -> {after.get('causal_priority')}。",
            f"風控修正: {before.get('risk_adjustment', 0)} -> {after.get('risk_adjustment', 0)}，差異 {risk_delta:+d}。",
            "若新聞風險觸發外部重置，內部日夜盤/八卦劇本降權，隔日必須先看現貨是否消化外部利空。",
        ],
        "guardrail": "此稽核只量化新聞規則對模型風險判斷的影響，不產生投資命令；需用日盤收盤驗證新聞壓力是否被市場吸收。",
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 國際新聞風險規則實質影響稽核",
        "",
        f"- 產生時間: {payload['generated_at']}",
        f"- 預測日: {payload.get('forecast_date')}",
        f"- 台股最後有效日: {payload.get('latest_spot_date')}",
        f"- 新聞狀態: {payload.get('news_status')}",
        f"- 結論: {payload['effect']}",
        "",
        "## 反事實比較",
        "",
        "| 項目 | 不納入新聞 | 納入新聞 |",
        "| --- | --- | --- |",
        f"| 外部重置 | {before.get('label')} | {after.get('label')} |",
        f"| 是否觸發 | {before.get('reset_active')} | {after.get('reset_active')} |",
        f"| 方向 | {before.get('direction')} | {after.get('direction')} |",
        f"| 分數 | {before.get('reset_score', 0)} | {after.get('reset_score', 0)} |",
        f"| 因果優先權 | {before.get('causal_priority')} | {after.get('causal_priority')} |",
        f"| 風控修正 | {before.get('risk_adjustment', 0)} | {after.get('risk_adjustment', 0)} |",
        "",
        "## 具體影響",
    ]
    for item in payload["concrete_impact"]:
        lines.append(f"- {item}")
    lines.extend(["", "## 新增觸發原因"])
    if added_reasons:
        for item in added_reasons:
            lines.append(f"- {item}")
    else:
        lines.append("- 無新增原因。")
    lines.extend(
        [
            "",
            "## 新聞摘要",
            f"- {news.get('summary', 'NA')}",
            "",
            f"- 防呆: {payload['guardrail']}",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Report: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
