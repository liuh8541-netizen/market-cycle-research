"""Reproducible closeout audit for the fixed market-cycle research program."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "research_closeout_manifest.json"
OUT_JSON = ROOT / "reports" / "final_research_conclusion.json"
OUT_MD = ROOT / "reports" / "final_research_conclusion.md"


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest().upper()


def variant(
    experiment: str,
    report: str,
    variant_name: str,
    scope: str,
    target_type: str = "direction",
) -> dict[str, Any]:
    payload = load_json(report)
    result = payload["variants"][variant_name]
    return {
        "experiment": experiment,
        "scope": scope,
        "target_type": target_type,
        "accuracy": result["accuracy"],
        "cases": result["cases"],
        "coverage": result["coverage"],
        "wilson_95_lower": result["wilson_95_lower"],
        "strongest_baseline_accuracy": result.get("strongest_baseline_accuracy"),
        "minimum_material_window_accuracy": result.get("minimum_material_window_accuracy"),
        "passed": bool(result.get("passed", False)),
        "evidence": report,
    }


def build_results() -> list[dict[str, Any]]:
    status = load_json("config/model_validation_status.json")
    three_layer = load_json("reports/three_layer_purged_research.json")
    main = next(item for item in three_layer["results"] if item["horizon_days"] == 120)
    night = status["validated_submodels"]["canonical_night_spread_gap_v1"]
    amplitude = status["validated_submodels"]["daily_night_gap_amplitude_v1"]

    results = [
        {
            "experiment": "three_layer_purged_ridge_120d",
            "scope": "120-trading-day TWII direction",
            "target_type": "direction",
            "accuracy": main["accuracy"],
            "cases": main["cases"],
            "coverage": main["coverage"],
            "wilson_95_lower": main["wilson_95_lower"],
            "strongest_baseline_accuracy": main["baseline"],
            "minimum_material_window_accuracy": None,
            "passed": bool(main["passed"]),
            "evidence": "reports/three_layer_purged_research.json",
        },
        {
            "experiment": "canonical_night_spread_gap_v1",
            "scope": night["scope"],
            "target_type": "direction",
            "accuracy": night["accuracy"],
            "cases": night["cases"],
            "coverage": night["coverage"],
            "wilson_95_lower": night["wilson_95_lower"],
            "strongest_baseline_accuracy": night["external_majority_accuracy_on_identical_dates"],
            "minimum_material_window_accuracy": None,
            "passed": True,
            "evidence": night["report"],
        },
        {
            "experiment": "daily_night_gap_amplitude_v1",
            "scope": amplitude["scope"],
            "target_type": "amplitude_risk",
            "accuracy": amplitude["accuracy"],
            "cases": amplitude["cases"],
            "coverage": amplitude["coverage"],
            "wilson_95_lower": amplitude["wilson_95_lower"],
            "strongest_baseline_accuracy": amplitude["absolute_night_spread_baseline_accuracy"],
            "minimum_material_window_accuracy": amplitude["minimum_material_window_accuracy"],
            "passed": True,
            "evidence": amplitude["report"],
        },
        variant(
            "option_tail_cash_close_v1",
            "reports/option_tail_cash_close.json",
            "night_plus_option_tail",
            "same-day cash close versus prior cash close",
        ),
        variant(
            "taiex_early_pulse_close_v1",
            "reports/taiex_early_pulse_close.json",
            "open_plus_early_pulse",
            "09:15-to-close direction",
        ),
        variant(
            "orderbook_early_pulse_close_v1",
            "reports/orderbook_early_pulse_close.json",
            "price_plus_orderbook",
            "09:15-to-close direction",
        ),
        variant(
            "orderbook_remaining_risk_v1",
            "reports/orderbook_remaining_risk.json",
            "price_plus_orderbook_risk",
            "09:15-to-close material-move risk",
            "amplitude_risk",
        ),
        variant(
            "sector_early_pulse_close_v1",
            "reports/sector_early_pulse_close.json",
            "plus_sector_breadth",
            "09:15-to-close direction",
        ),
        variant(
            "cross_sectional_breadth_cycle_v1",
            "reports/cross_sectional_breadth_cycle.json",
            "plus_price_and_institutional_breadth",
            "20-trading-day TWII direction",
        ),
        variant(
            "securities_lending_pressure_v1",
            "reports/securities_lending_pressure.json",
            "plus_securities_lending",
            "20-trading-day TWII direction",
        ),
        variant(
            "market_valuation_breadth_v2",
            "reports/market_valuation_breadth_v2.json",
            "plus_market_valuation_breadth",
            "20-trading-day TWII direction",
        ),
        variant(
            "market_cap_structure_v1",
            "reports/market_cap_structure.json",
            "plus_market_cap_structure",
            "20-trading-day TWII direction",
        ),
        variant(
            "foreign_shareholding_structure_v1",
            "reports/foreign_shareholding_structure.json",
            "plus_foreign_shareholding",
            "20-trading-day TWII direction",
        ),
        variant(
            "loan_collateral_structure_v2",
            "reports/loan_collateral_structure_v2.json",
            "plus_nontraditional_credit",
            "20-trading-day TWII direction",
        ),
    ]

    breadth = load_json("reports/breadth_transition_episode.json")["variants"]["institutional_override"]
    results.append(
        {
            "experiment": "breadth_transition_episode_v1",
            "scope": "20-trading-day breadth-transition direction",
            "target_type": "direction",
            "accuracy": breadth["accuracy"],
            "cases": breadth["cases"],
            "coverage": breadth["coverage"],
            "wilson_95_lower": breadth["wilson_95_lower"],
            "strongest_baseline_accuracy": None,
            "minimum_material_window_accuracy": breadth["minimum_material_block_accuracy"],
            "passed": bool(breadth["passed"]),
            "evidence": "reports/breadth_transition_episode.json",
        }
    )
    return results


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2%}"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    missing = [path for path in manifest["required_evidence"] if not (ROOT / path).exists()]
    parse_errors: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for path in manifest["required_evidence"]:
        if path in missing:
            continue
        hashes[path] = sha256(path)
        if path.endswith(".json"):
            try:
                load_json(path)
            except (OSError, ValueError) as exc:
                parse_errors[path] = str(exc)

    results = build_results() if not missing and not parse_errors else []
    passed = [item for item in results if item["passed"]]
    failed = [item for item in results if not item["passed"]]
    directional = [item for item in results if item["target_type"] == "direction"]
    main_direction = next(item for item in results if item["experiment"] == "three_layer_purged_ridge_120d")
    target = manifest["target"]["accuracy"]
    saturation = load_json("reports/direction_search_saturation_audit.json")

    complete = not missing and not parse_errors
    payload = {
        "closeout_id": manifest["closeout_id"],
        "completed_at": "2026-07-24",
        "audit_complete": complete,
        "missing_evidence": missing,
        "parse_errors": parse_errors,
        "target": manifest["target"],
        "evidence_sha256": hashes,
        "experiments_summarized": len(results),
        "passing_experiments": len(passed),
        "failed_experiments": len(failed),
        "results": results,
        "conclusion": {
            "main_multi_day_direction_target_met": False,
            "best_main_direction_accuracy": main_direction["accuracy"],
            "main_direction_target_gap_percentage_points": (target - main_direction["accuracy"]) * 100,
            "main_direction_baseline_gap_percentage_points": (
                main_direction["accuracy"] - main_direction["strongest_baseline_accuracy"]
            )
            * 100,
            "validated_scopes": [item["experiment"] for item in passed],
            "production_multi_day_direction_enabled": False,
            "retrospective_direction_search_closed": True,
            "saturation_decision": saturation["decision"],
            "scientific_result": (
                "The 90% target is rejected for broad TWII multi-day direction under the "
                "evaluated point-in-time sources and locked validation protocol. Two narrow "
                "pre-open targets pass and must remain scope-limited."
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 大盤週期研究結案報告",
        "",
        "## 結論",
        "",
        "- 研究狀態：歷史回溯研究已完成；前瞻監測保留為上線後追蹤，不算未完成實驗。",
        "- 原始目標：正式樣本外命中率至少90%、至少100件、覆蓋率至少10%、Wilson下限至少80%，且優於最強簡單基準。",
        f"- 大盤多日方向最佳正式結果：{pct(main_direction['accuracy'])}（{main_direction['cases']}件），距90%尚差 {(target-main_direction['accuracy'])*100:.2f} 個百分點。",
        f"- 同一結果的最強基準：{pct(main_direction['strongest_baseline_accuracy'])}；模型反而落後 {(main_direction['strongest_baseline_accuracy']-main_direction['accuracy'])*100:.2f} 個百分點。",
        "- 事實判定：在本次已評估的價格、外部市場、法人、期貨、選擇權、廣度、市值、估值、借券、外資持股與信用結構資料下，無法支持「大盤多日方向可達90%」的研究假設。",
        "- 可保留成果：開盤缺口方向與開盤缺口振幅風險通過各自的限定範圍；不得延伸解讀為當日收盤或多日趨勢。",
        "",
        "## 目標與事實差距",
        "",
        "| 正式研究 | 預測範圍 | 命中率 | 件數 | 覆蓋率 | Wilson下限 | 最強基準 | 結果 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item['experiment']} | {item['scope']} | {pct(item['accuracy'])} | "
            f"{item['cases']} | {pct(item['coverage'])} | {pct(item['wilson_95_lower'])} | "
            f"{pct(item['strongest_baseline_accuracy'])} | {'通過' if item['passed'] else '失敗'} |"
        )
    lines.extend(
        [
            "",
            "## 程式落地政策",
            "",
            "1. 關閉大盤1、5、20、60、120日方向的正式交易訊號；歷史相似機率只能標示為探索統計。",
            "2. FinMind多日方向係數維持0；原始籌碼、衍生品與結構資料只保留為描述及未來新資料研究。",
            "3. 正式保留 `canonical_night_spread_gap_v1`，用途僅為同日現貨開盤相對前收的缺口方向。",
            "4. 正式保留 `daily_night_gap_amplitude_v1`，用途僅為開盤缺口是否達到重大振幅；不可輸出漲跌方向。",
            "5. 任何新模型都必須使用新資訊來源或真正前瞻資料，預先鎖定後再驗證；不得再切割已檢視歷史資料尋找90%子群。",
            "",
            "## 完整性稽核",
            "",
            f"- 必要證據：{len(manifest['required_evidence'])}份；缺少：{len(missing)}；JSON解析失敗：{len(parse_errors)}。",
            f"- 彙整正式結果：{len(results)}項；通過：{len(passed)}；失敗：{len(failed)}。",
            f"- 結案稽核：{'通過' if complete else '失敗'}。",
            "",
            "本報告的「研究完成」表示固定歷史研究計畫已得到可重現結論，不表示大盤多日方向達成90%。",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not complete:
        raise SystemExit("Closeout audit failed; see final_research_conclusion.json")
    print(f"Closeout audit complete: {len(results)} results, {len(passed)} passed, {len(failed)} failed.")
    print(f"Report: {OUT_MD}")


if __name__ == "__main__":
    main()
