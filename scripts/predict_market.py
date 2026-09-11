import argparse
from html import escape
import json
import os
import ssl
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_lifecycle.external_market import (
    add_external_market_features,
    update_external_markets,
    update_taiwan_futures_night,
)
from market_lifecycle.clinical_knowledge import update_clinical_knowledge
from market_lifecycle.factor_features import add_factor_features
from market_lifecycle.features import add_features
from market_lifecycle.lifecycle import score_lifecycle
from market_lifecycle.industry_risk import add_memory_industry_risk
from market_lifecycle.market_health import build_market_health_assessment
from market_lifecycle.night_cash_tracking import write_night_cash_tracking
from market_lifecycle.probability_forecast import (
    forecast_from_history,
    reconcile_one_day_forecast,
    simulate_forecasts,
)
from market_lifecycle.psychology_state import build_psychology_state
from market_lifecycle.clinical_enrichment import current_auxiliary_snapshot
from market_lifecycle.self_repair import build_self_repair_assessment
from market_lifecycle.path_risk import build_path_risk_assessment


TAIPEI = ZoneInfo("Asia/Taipei")
YAHOO_SYMBOL = "^TWII"
FORECAST_HISTORY = ROOT / "reports" / "forecast_history.json"
ERROR_REVIEW_JSON = ROOT / "reports" / "error_review.json"
ERROR_REVIEW_MD = ROOT / "reports" / "error_review.md"
PEAK_WARNING_JSON = ROOT / "reports" / "peak_to_valley_warning_backtest.json"
PEAK_WARNING_MD = ROOT / "reports" / "peak_to_valley_warning_backtest.md"
BREATH_MONITOR_HTML = ROOT / "reports" / "market_breath_monitor.html"
PROGRAMMED_PRESSURE_JSON = ROOT / "reports" / "programmed_pressure_pattern.json"
PROGRAMMED_PRESSURE_MD = ROOT / "reports" / "programmed_pressure_pattern.md"
MANUAL_SECTOR_OBSERVATIONS = ROOT / "config" / "manual_sector_observations.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Update market data and forecast Taiwan index movement.")
    parser.add_argument("--date", help="Forecast date. If omitted, use today's Taipei date.")
    parser.add_argument("--index", type=float, help="Optional market index close value.")
    parser.add_argument("--input", default="data/processed/twii_daily.csv", help="OHLCV CSV input.")
    parser.add_argument("--external-input", default="data/processed/external_markets.csv", help="External market CSV.")
    parser.add_argument("--night-futures-input", default="data/processed/taiwan_futures_night.csv", help="Taiwan TX night futures CSV.")
    parser.add_argument("--global-news-risk", default="reports/daily_global_news_risk.json", help="Daily browsed global finance/political news risk JSON.")
    parser.add_argument("--factor-dir", help="Optional non-price factor directory.")
    parser.add_argument("--memory-events", default="config/memory_risk_events.csv", help="Curated memory-industry event CSV.")
    parser.add_argument("--output", default="reports/today_market_forecast.md", help="Markdown forecast output.")
    parser.add_argument("--json-output", default="reports/today_market_forecast.json", help="JSON forecast output.")
    parser.add_argument("--simulate-step-days", type=int, default=5, help="Historical simulation sampling step.")
    parser.add_argument("--no-update", action="store_true", help="Skip online data update.")
    args = parser.parse_args()

    forecast_date = args.date or today_taipei()
    input_path = rooted_path(args.input)
    external_path = rooted_path(args.external_input)
    night_futures_path = rooted_path(args.night_futures_input)
    global_news_risk_path = rooted_path(args.global_news_risk)
    finmind_token = load_finmind_token()

    data_update = skipped_update()
    external_update = skipped_update()
    night_futures_update = skipped_update()
    if not args.no_update:
        data_update = update_twii_daily(input_path, token=finmind_token)
        external_update = update_external_markets(external_path, ROOT / "data" / "raw")
        night_futures_update = update_taiwan_futures_night(
            night_futures_path,
            start_date="2000-01-01",
            end_date=forecast_date,
            token=finmind_token,
        )

    night_cash_tracking = write_night_cash_tracking(
        night_futures_path,
        input_path,
        ROOT / "data" / "processed" / "night_cash_impact_log.csv",
        ROOT / "reports" / "night_cash_impact_tracking.json",
        ROOT / "reports" / "night_cash_impact_tracking.md",
    )

    price = pd.read_csv(input_path)
    price = add_external_market_features(price, external_path, night_futures_path)
    features = add_factor_features(add_features(price), args.factor_dir)
    scored = add_memory_industry_risk(score_lifecycle(features), rooted_path(args.memory_events))
    scored = scored.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    scored["date"] = pd.to_datetime(scored["date"])

    index_value = args.index if args.index is not None else lookup_close(scored, forecast_date)
    live_monitor = fetch_live_monitor_snapshot(forecast_date, manual_index=args.index)
    forecast = forecast_from_history(scored, forecast_date)
    validation = load_model_validation_status()
    forecast["validation_status"] = (
        "validated" if validation.get("passed") else "exploratory_not_production"
    )
    forecast["formal_direction_signal"] = (
        forecast["forecasts"] if validation.get("passed") else None
    )
    simulation = simulate_forecasts(scored, step_days=args.simulate_step_days)
    index_check = check_input_index(scored, forecast_date, index_value, provided=args.index is not None)
    add_forecast_target_dates(forecast, scored, index_check["signal_date"])
    cause = analyze_cause(scored, index_check["signal_date"])
    premarket = analyze_premarket(forecast_date, scored, external_path, night_futures_path)
    global_news_risk = load_global_news_risk(global_news_risk_path, forecast_date)
    situation_psychology_context = analyze_situation_psychology_context(
        forecast_date,
        premarket,
        global_news_risk,
    )
    external_event_reset = analyze_external_event_reset(premarket, global_news_risk)
    intraday_tactical = analyze_intraday_tactical_monitor(
        forecast_date,
        scored,
        index_check["signal_date"],
        premarket,
        live_monitor,
    )
    washout = analyze_washout_pattern(scored, index_check["signal_date"])
    candlestick = analyze_candlestick_pattern(scored, index_check["signal_date"])
    bagua_lifecycle = analyze_bagua_lifecycle(scored, index_check["signal_date"], candlestick, washout)
    tradeable_cycle = analyze_tradeable_rally_segments(scored, index_check["signal_date"])
    technical_phase = analyze_technical_phase(
        scored,
        index_check["signal_date"],
        bagua_lifecycle,
        candlestick,
        tradeable_cycle,
    )
    peak_to_valley_warning = analyze_peak_to_valley_warning(scored, index_check["signal_date"])
    peak_to_valley_backtest = build_peak_to_valley_warning_backtest(scored, index_check["signal_date"])
    route_reference = analyze_route_reference(
        scored,
        index_check["signal_date"],
        bagua_lifecycle,
        technical_phase,
    )
    bottom_event_reference = analyze_bottom_event_reference(
        scored,
        index_check["signal_date"],
        bagua_lifecycle,
    )
    capital_flow = capital_flow_snapshot(scored, index_check["signal_date"], args.factor_dir)
    date_audit = build_market_date_audit(forecast_date, index_check, live_monitor, premarket)
    crash_monitor = analyze_crash_monitor(
        scored,
        index_check["signal_date"],
        technical_phase,
        bagua_lifecycle,
        premarket,
        capital_flow,
        live_monitor,
    )
    crash_monitor = reconcile_market_state_sop(crash_monitor, bagua_lifecycle)
    freshness = audit_data_freshness(
        forecast_date,
        scored,
        external_path,
        night_futures_path,
        capital_flow,
    )
    self_review = build_self_review(
        scored,
        forecast_date,
        index_check["signal_date"],
        forecast,
        premarket,
        bagua_lifecycle,
        tradeable_cycle,
        candlestick,
        washout,
    )
    error_review = build_error_review(self_review)
    factor_root = rooted_path(args.factor_dir) if args.factor_dir else None
    breadth_for_path = read_optional_factor(factor_root, "cross_sectional_breadth.csv")
    futures_for_path = read_optional_factor(factor_root, "futures_daily.csv")
    path_risk_adjustment = build_path_risk_assessment(
        scored, forecast_date, breadth=breadth_for_path, futures_daily=futures_for_path
    )
    programmed_pressure_pattern = analyze_programmed_pressure_pattern(
        scored,
        forecast_date,
        index_value if args.index is not None else None,
        night_futures_path,
        factor_root,
    )
    monthly_cycle_monitor = analyze_monthly_cycle_monitor(
        scored,
        index_check["signal_date"],
        programmed_pressure_pattern,
        external_event_reset,
    )
    human_behavior_pattern = analyze_human_behavior_market_pattern(
        scored,
        index_check["signal_date"],
        premarket,
        intraday_tactical,
        candlestick,
        washout,
        bagua_lifecycle,
        technical_phase,
        monthly_cycle_monitor,
        programmed_pressure_pattern,
    )
    day_night_variance_pattern = analyze_day_night_variance_pattern(
        scored,
        forecast_date,
        index_check["signal_date"],
        premarket,
        intraday_tactical,
        human_behavior_pattern,
        external_event_reset,
        bagua_lifecycle,
    )
    sector_pressure_observation = analyze_sector_pressure_observation(
        forecast_date,
        index_check["signal_date"],
        index_value,
        scored,
        MANUAL_SECTOR_OBSERVATIONS,
    )
    endogenous_regulation_pulse = analyze_endogenous_regulation_pulse(
        scored,
        index_check["signal_date"],
        external_event_reset,
        day_night_variance_pattern,
        programmed_pressure_pattern,
        sector_pressure_observation,
    )
    market_heart_rhythm = analyze_market_heart_rhythm(scored, index_check["signal_date"])
    market_stethoscope = analyze_market_stethoscope(
        scored,
        index_check["signal_date"],
        factor_root,
        market_heart_rhythm,
        global_news_risk,
    )
    psychological_warfare_pattern = analyze_psychological_warfare_pattern(
        bagua_lifecycle,
        human_behavior_pattern,
        day_night_variance_pattern,
        intraday_tactical,
        endogenous_regulation_pulse,
        crash_monitor,
        technical_phase,
    )

    payload = {
        "input": {
            "date": forecast_date,
            "index": index_value,
            "index_provided_by_user": args.index is not None,
        },
        "data_update": data_update,
        "external_update": external_update,
        "night_futures_update": night_futures_update,
        "night_cash_tracking": night_cash_tracking,
        "index_check": index_check,
        "market_date_audit": date_audit,
        "premarket": premarket,
        "global_news_risk": global_news_risk,
        "situation_psychology_context": situation_psychology_context,
        "external_event_reset_monitor": external_event_reset,
        "intraday_tactical_monitor": intraday_tactical,
        "self_review": self_review,
        "error_review": error_review,
        "path_risk_adjustment": path_risk_adjustment,
        "programmed_pressure_pattern": programmed_pressure_pattern,
        "monthly_cycle_monitor": monthly_cycle_monitor,
        "human_behavior_market_pattern": human_behavior_pattern,
        "day_night_variance_pattern": day_night_variance_pattern,
        "sector_pressure_observation": sector_pressure_observation,
        "endogenous_regulation_pulse": endogenous_regulation_pulse,
        "market_heart_rhythm": market_heart_rhythm,
        "market_stethoscope": market_stethoscope,
        "psychological_warfare_pattern": psychological_warfare_pattern,
        "cause_analysis": cause,
        "washout_pattern": washout,
        "candlestick_pattern": candlestick,
        "bagua_lifecycle": bagua_lifecycle,
        "tradeable_cycle": tradeable_cycle,
        "technical_phase": technical_phase,
        "peak_to_valley_warning": peak_to_valley_warning,
        "peak_to_valley_backtest": peak_to_valley_backtest,
        "route_reference": route_reference,
        "bottom_event_reference": bottom_event_reference,
        "crash_monitor": crash_monitor,
        "live_monitor": live_monitor,
        "capital_flow": capital_flow,
        "data_freshness": freshness,
        "forecast": forecast,
        "model_validation": validation,
        "production_policy": build_production_policy(validation),
        "model_reliability_audit": load_model_reliability_audit(),
        "historical_self_simulation": simulation,
        "memory_industry_risk": memory_risk_snapshot(scored, index_check["signal_date"]),
    }
    payload["direction_reliability_policy"] = build_direction_reliability_policy(
        payload["model_reliability_audit"],
        payload["model_validation"],
    )
    payload["logic_consistency"] = analyze_logic_consistency(payload)
    payload["integrated_summary"] = analyze_integrated_summary(payload)
    payload["psychology_position_context"] = load_psychology_position_context(
        forecast_date, args.factor_dir
    )
    payload["psychology_state"] = build_psychology_state(payload, load_behavior_statistics())
    payload["forecast"] = reconcile_one_day_forecast(
        payload["forecast"], payload["psychology_state"]
    )
    # Re-run summaries after the one-day reconciliation so the report cannot
    # retain a stale generic direction that conflicts with the frozen overlay.
    payload["logic_consistency"] = analyze_logic_consistency(payload)
    payload["integrated_summary"] = analyze_integrated_summary(payload)
    payload["market_mode_switch"] = analyze_market_mode_switch(payload)
    payload["close_cause_attribution"] = analyze_close_cause_attribution(payload)
    payload["market_health"] = build_market_health_assessment(payload)
    payload["fundamental_constitution"] = analyze_fundamental_constitution(payload)
    payload["dialogue_core_rules"] = build_dialogue_core_rules(payload)
    payload["practical_cause_arbitration"] = analyze_practical_cause_arbitration(payload)
    payload["market_protection_layers"] = analyze_market_protection_layers(payload)
    payload["crisis_opportunity_interface"] = analyze_crisis_opportunity_interface(payload)
    payload["cross_market_entanglement"] = analyze_cross_market_entanglement(payload)
    payload["master_arbitration"] = analyze_master_arbitration(payload)
    payload["weather_satellite_forecast"] = build_weather_satellite_forecast_model(payload)
    payload["equation_reasoning_audit"] = build_equation_reasoning_audit(payload)
    payload["self_repair"] = build_self_repair_assessment(payload)
    payload["clinical_knowledge"] = update_clinical_knowledge(
        payload, ROOT / "research" / "market_health_knowledge"
    )

    output = rooted_path(args.output)
    json_output = rooted_path(args.json_output)
    detail_output = output.with_name(output.stem + "_detail" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_brief_forecast(payload), encoding="utf-8")
    detail_output.write_text(render_forecast(payload), encoding="utf-8")
    json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ERROR_REVIEW_JSON.write_text(json.dumps(error_review, ensure_ascii=False, indent=2), encoding="utf-8")
    ERROR_REVIEW_MD.write_text(render_error_review(error_review), encoding="utf-8")
    PEAK_WARNING_JSON.write_text(json.dumps(peak_to_valley_backtest, ensure_ascii=False, indent=2), encoding="utf-8")
    PEAK_WARNING_MD.write_text(render_peak_to_valley_backtest(peak_to_valley_backtest), encoding="utf-8")
    PROGRAMMED_PRESSURE_JSON.write_text(
        json.dumps(programmed_pressure_pattern, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    PROGRAMMED_PRESSURE_MD.write_text(
        render_programmed_pressure_pattern(programmed_pressure_pattern),
        encoding="utf-8",
    )
    BREATH_MONITOR_HTML.write_text(render_breath_monitor(payload), encoding="utf-8")
    upsert_forecast_history(FORECAST_HISTORY, payload)

    print(render_console_summary(payload))
    print(f"\nReport: {output}")
    print(f"Detail: {detail_output}")
    print(f"JSON: {json_output}")
    print(f"BreathMonitor: {BREATH_MONITOR_HTML}")


def load_behavior_statistics() -> list[dict]:
    path = ROOT / "reports" / "market_clinical_ledger.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("behavior_statistics", [])
    except (OSError, ValueError, TypeError):
        return []


def analyze_sector_pressure_observation(
    forecast_date: str,
    signal_date: str,
    index_value: float | None,
    scored: pd.DataFrame,
    observation_path: Path = MANUAL_SECTOR_OBSERVATIONS,
) -> dict:
    factor_root = ROOT / "data" / "processed" / "factors"
    breadth_snapshot = latest_factor_snapshot(factor_root / "cross_sectional_breadth.csv", forecast_date)
    sector_snapshot = latest_factor_snapshot(factor_root / "sector_early_pulse.csv", forecast_date)
    base = {
        "enabled": bool(breadth_snapshot or sector_snapshot),
        "date": forecast_date,
        "label": "無人工族群觀察",
        "code": "no_manual_sector_observation",
        "risk_score": 0,
        "summary": "尚無電子、軍工或其他族群的人工觀察留底。",
        "observations": [],
        "weak_sectors": [],
        "strong_sectors": [],
        "breadth_snapshot": breadth_snapshot or {},
        "sector_snapshot": sector_snapshot or {},
        "cause_candidates": [],
        "hidden_cause_candidates": [],
        "next_validation": [],
        "guardrail": "族群壓力觀察是人工症狀留底；需由正式廣度、成交量、法人與隔日走勢驗證，不證明單一主體操控，也不產生買賣命令。",
    }
    auto_risk, auto_weak, auto_strong, auto_causes = sector_auto_breadth_signals(
        breadth_snapshot, sector_snapshot
    )
    base["risk_score"] = auto_risk
    base["weak_sectors"] = auto_weak
    base["strong_sectors"] = auto_strong
    if auto_risk:
        base.update(
            {
                "enabled": True,
                "label": "族群廣度自動警示",
                "code": "auto_breadth_pressure",
                "summary": "正式廣度或早盤族群資料顯示市場內部有分化壓力。",
                "cause_candidates": auto_causes,
                "hidden_cause_candidates": [
                    "指數可能由少數權值支撐，但多數股票已先行換手或退潮。",
                    "若隔日廣度未收復，代表洗盤轉為內部結構弱化。",
                ],
                "next_validation": [
                    "隔日上漲家數、等權報酬與成交量上漲占比是否同步修復。",
                    "電子、金融、櫃買與加權是否同向，避免只靠少數權值撐盤。",
                ],
            }
        )
    if not observation_path.exists():
        return base

    try:
        observations = pd.read_csv(observation_path)
    except Exception as exc:
        base.update(
            {
                "label": "族群觀察讀取失敗",
                "code": "sector_observation_read_error",
                "summary": f"人工族群觀察檔讀取失敗：{exc}",
            }
        )
        return base

    if observations.empty or "date" not in observations:
        return base

    observations["date"] = pd.to_datetime(observations["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    today_rows = observations[observations["date"] == forecast_date].copy()
    if today_rows.empty:
        return base

    records = today_rows.fillna("").to_dict("records")
    weak = [
        str(row.get("sector", "")).strip()
        for row in records
        if str(row.get("status", "")).strip().lower() in {"weak", "pressure", "selloff", "down"}
    ]
    strong = [
        str(row.get("sector", "")).strip()
        for row in records
        if str(row.get("status", "")).strip().lower() in {"strong", "up", "support"}
    ]

    index_return = None
    if index_value is not None and not scored.empty:
        prior_rows = scored[scored["date"].dt.strftime("%Y-%m-%d") <= signal_date]
        if not prior_rows.empty:
            prior_close = float(prior_rows.iloc[-1]["close"])
            if prior_close:
                index_return = index_value / prior_close - 1

    weak = unique_text(weak + auto_weak)
    strong = unique_text(strong + auto_strong)
    risk_score = min(5, len([item for item in weak if item]) + auto_risk)
    if index_return is not None and index_return > 0 and len(weak) >= 2:
        code = "index_up_sector_pressure"
        label = "指數上漲但族群被壓"
        summary = "大盤表面上攻，但電子、軍工等人氣族群同步承壓，屬於指數強、內部廣度分化。"
        cause_candidates = unique_text(auto_causes + [
            "權值股撐住指數，但中小型與題材股先行獲利了結。",
            "高檔換手期資金從漲多族群撤出，改測市場承接力。",
            "題材股前期漲幅較大，遇到關鍵關卡前容易被調節。",
        ])
        hidden_causes = [
            "若指數續漲但弱族群擴大，代表上攻健康度不足。",
            "若隔日弱族群快速收回，較像洗盤換手；若續破短均，較像退潮。",
        ]
    elif weak:
        code = "sector_pressure_observed"
        label = "族群壓力出現"
        summary = f"觀察到 {len(weak)} 個族群承壓，需要與正式廣度資料交叉驗證。"
        cause_candidates = unique_text(auto_causes + [
            "族群輪動或獲利了結。",
            "關鍵壓力區前資金降低題材股曝險。",
        ])
        hidden_causes = ["需確認是否由單日噪音擴大成連續弱化。"]
    else:
        code = "sector_observation_neutral"
        label = "族群觀察中性"
        summary = "今日人工族群觀察未形成明確壓力。"
        cause_candidates = []
        hidden_causes = []

    return {
        "enabled": True,
        "date": forecast_date,
        "label": label,
        "code": code,
        "risk_score": risk_score,
        "index_return": index_return,
        "summary": summary,
        "observations": records,
        "weak_sectors": weak,
        "strong_sectors": strong,
        "breadth_snapshot": breadth_snapshot or {},
        "sector_snapshot": sector_snapshot or {},
        "cause_candidates": cause_candidates,
        "hidden_cause_candidates": hidden_causes,
        "next_validation": [
            "隔日若指數站穩46,400且弱族群同步收復，分化降級為健康換手。",
            "隔日若指數上攻但弱族群續破短線支撐，上攻健康度降級。",
            "若弱族群擴大到電子、軍工、金融與傳產同步，升級為廣度惡化警戒。",
        ],
        "guardrail": base["guardrail"],
    }


def latest_factor_snapshot(path: Path, forecast_date: str) -> dict:
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return {}
    if frame.empty or "date" not in frame:
        return {}
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    target = pd.to_datetime(forecast_date)
    eligible = frame[frame["date"] <= target].dropna(subset=["date"])
    if eligible.empty:
        return {}
    row = eligible.sort_values("date").iloc[-1]
    snapshot = {}
    for key, value in row.to_dict().items():
        if key == "date":
            snapshot[key] = none_or_str(value)
            snapshot["_source_date"] = none_or_str(value)
            continue
        if key in {"signal_date", "calendar_date"}:
            snapshot[key] = none_or_str(value)
            snapshot["_source_date"] = none_or_str(value)
            continue
        try:
            snapshot[key] = safe_float(value)
        except (TypeError, ValueError):
            snapshot[key] = none_or_str(value)
    return snapshot


def sector_auto_breadth_signals(breadth: dict, sector: dict) -> tuple[int, list[str], list[str], list[str]]:
    risk = 0
    weak: list[str] = []
    strong: list[str] = []
    causes: list[str] = []
    advance = safe_float(breadth.get("price_advancing_fraction"))
    decline = safe_float(breadth.get("price_declining_fraction"))
    equal_weight = safe_float(breadth.get("price_equal_weight_return"))
    up_volume = safe_float(breadth.get("price_up_volume_fraction"))
    ad_breadth = safe_float(breadth.get("price_advance_decline_breadth"))
    if advance is not None and advance < 0.42:
        risk += 1
        weak.append("全市場上漲家數不足")
        causes.append("上漲家數低於42%，代表指數若上漲也可能是集中撐盤。")
    if decline is not None and decline > 0.55:
        risk += 1
        weak.append("全市場下跌家數偏多")
        causes.append("下跌家數過半偏高，市場耐心正在被測試。")
    if equal_weight is not None and equal_weight < -0.003:
        risk += 1
        weak.append("等權報酬偏弱")
        causes.append("等權報酬轉弱，代表非權值股承壓。")
    if up_volume is not None and up_volume < 0.45:
        risk += 1
        weak.append("上漲成交量占比不足")
        causes.append("上漲成交量占比不足，買盤擴散性偏弱。")
    if ad_breadth is not None and ad_breadth > 0.15:
        strong.append("市場廣度偏強")
    electronic = safe_float(sector.get("twse_electronic_return"))
    tpex_electronic = safe_float(sector.get("tpex_twse_electronic_rotation"))
    finance = safe_float(sector.get("twse_finance_return"))
    sector_breadth = safe_float(sector.get("sector_breadth"))
    if electronic is not None and electronic < -0.003:
        risk += 1
        weak.append("上市電子")
        causes.append("上市電子早盤偏弱，AI/半導體主軸需日盤收復驗證。")
    if tpex_electronic is not None and tpex_electronic < -0.004:
        risk += 1
        weak.append("櫃買電子相對弱")
        causes.append("櫃買電子弱於上市電子，題材股換手壓力偏高。")
    if finance is not None and finance > 0.003:
        strong.append("金融")
    if sector_breadth is not None and sector_breadth < 0:
        risk += 1
        weak.append("早盤族群廣度")
    return min(5, risk), unique_text(weak), unique_text(strong), unique_text(causes)


def load_psychology_position_context(forecast_date: str, factor_dir: str | None) -> dict:
    if not factor_dir:
        return {}
    root = rooted_path(factor_dir)
    try:
        breadth = pd.read_csv(root / "cross_sectional_breadth.csv") if (root / "cross_sectional_breadth.csv").exists() else pd.DataFrame()
        futures = pd.read_csv(root / "futures_daily.csv") if (root / "futures_daily.csv").exists() else pd.DataFrame()
        institutional = pd.read_csv(root / "futures_institutional.csv") if (root / "futures_institutional.csv").exists() else pd.DataFrame()
        return current_auxiliary_snapshot(
            forecast_date, breadth=breadth, futures_daily=futures,
            futures_institutional=institutional,
        )
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def rooted_path(path_text: str) -> Path:
    path = Path(path_text)
    return ROOT / path if not path.is_absolute() else path


def read_optional_factor(root: Path | None, filename: str) -> pd.DataFrame:
    if root is None:
        return pd.DataFrame()
    path = root / filename
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def load_global_news_risk(path: Path, forecast_date: str) -> dict:
    if not path.exists():
        return {
            "available": False,
            "status": "missing",
            "date": None,
            "risk_score": 0,
            "tailwind_score": 0,
            "net_risk_score": 0,
            "events": [],
            "sources": [],
            "summary": "每日國際重大財經政治消息尚未產出；外部事件重置只能使用市場價格代理。",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "status": "invalid",
            "date": None,
            "risk_score": 0,
            "tailwind_score": 0,
            "net_risk_score": 0,
            "events": [],
            "sources": [],
            "summary": f"每日國際新聞風險檔讀取失敗: {exc}",
        }
    risk = int(safe_float(payload.get("risk_score")) or 0)
    tailwind = int(safe_float(payload.get("tailwind_score")) or 0)
    events = payload.get("events") if isinstance(payload.get("events"), list) else []
    sources = payload.get("sources") if isinstance(payload.get("sources"), list) else []
    pending_events = payload.get("pending_events") if isinstance(payload.get("pending_events"), list) else []
    model_integration = payload.get("model_integration") if isinstance(payload.get("model_integration"), dict) else {}
    stale = bool(payload.get("date") and str(payload.get("date")) < str(forecast_date))
    return {
        "available": True,
        "status": "stale" if stale else "connected",
        "date": payload.get("date"),
        "risk_score": risk,
        "tailwind_score": tailwind,
        "net_risk_score": risk - tailwind,
        "events": events,
        "pending_events": pending_events,
        "model_integration": model_integration,
        "sources": sources,
        "summary": payload.get("summary") or "已接入每日國際重大財經政治消息風險檔。",
        "guardrail": payload.get("guardrail")
        or "新聞只作外部風險與基本面變動評估，需由價格、量能、匯率、利率與日盤確認。",
    }


def today_taipei() -> str:
    return datetime.now(TAIPEI).date().isoformat()


def audit_data_freshness(
    forecast_date: str,
    scored: pd.DataFrame,
    external_path: Path,
    night_futures_path: Path,
    capital_flow: dict,
) -> dict:
    target = pd.to_datetime(forecast_date).date()
    now = datetime.now(TAIPEI)
    if target.weekday() >= 5:
        spot_expected = previous_weekday(target)
    elif target == now.date() and now.hour < 14:
        spot_expected = previous_weekday(target)
    else:
        spot_expected = target
    external_expected = previous_weekday(target)
    night_expected = expected_night_signal_date(target, now)
    spot_latest = scored[scored["date"] <= pd.Timestamp(target)]["date"].max().date()

    sources = [
        freshness_item("台股現貨", spot_latest, spot_expected, "FinMind Sponsor TAIEX / TWSE official（日線優先）；Yahoo ^TWII 僅作長歷史與即時快照備援"),
    ]
    latest_spot_row = scored[scored["date"] == pd.Timestamp(spot_latest)].iloc[-1]
    latest_volume = latest_spot_row.get("volume")
    sources.append(
        {
            "name": "台股成交量",
            "latest_date": spot_latest.isoformat() if pd.notna(latest_volume) and latest_volume > 0 else None,
            "expected_date": spot_expected.isoformat(),
            "lag_days": 0 if pd.notna(latest_volume) and latest_volume > 0 else None,
            "status": "current" if pd.notna(latest_volume) and latest_volume > 0 else "unavailable",
            "source": "FinMind Sponsor TAIEX / TWSE official（末筆成交量為 0 時不視為有效）",
        }
    )
    if external_path.exists():
        external = pd.read_csv(external_path)
        external["date"] = pd.to_datetime(external["date"])
        for label, column in [
            ("S&P 500", "sp500_close"),
            ("Nasdaq", "nasdaq_close"),
            ("費半", "sox_close"),
            ("VIX", "vix_close"),
            ("TSM ADR", "tsm_adr_close"),
            ("美光", "micron_close"),
            ("三星", "samsung_close"),
            ("SK海力士", "sk_hynix_close"),
        ]:
            valid = external.loc[external.get(column, pd.Series(index=external.index, dtype=float)).notna(), "date"]
            latest = valid.max().date() if not valid.empty else None
            sources.append(freshness_item(label, latest, external_expected, "Yahoo Finance"))
        # These research-only transmission inputs are admissible only when the
        # source date is strictly earlier than the Taiwan forecast date. This
        # explicitly excludes Yahoo's current intraday FX candle.
        for label, column in [("EWT台灣ETF", "ewt_close"), ("美元兌台幣", "usd_twd_close")]:
            values = external.get(column, pd.Series(index=external.index, dtype=float))
            valid = external.loc[values.notna() & (external["date"].dt.date < target), "date"]
            latest = valid.max().date() if not valid.empty else None
            sources.append(freshness_item(label, latest, external_expected, "Yahoo Finance（研究只使用嚴格早於台股日期的完整日資料）"))
    else:
        sources.append(freshness_item("外部市場", None, external_expected, "檔案不存在"))

    night_latest = None
    if night_futures_path.exists():
        night = pd.read_csv(night_futures_path)
        if not night.empty and "signal_date" in night:
            night_latest = pd.to_datetime(night["signal_date"]).max().date()
    sources.append(
        freshness_item(
            "台指期夜盤",
            night_latest,
            night_expected,
            "FinMind TX after_market（台北時間05:00前，當日夜盤尚未完成）",
        )
    )

    factor_status = "current" if capital_flow.get("has_factor_values") else "unavailable"
    sources.append(
        {
            "name": "法人／融資／期權",
            "latest_date": capital_flow.get("signal_date") if capital_flow.get("has_factor_values") else None,
            "expected_date": spot_latest.isoformat(),
            "lag_days": None,
            "status": factor_status,
            "source": capital_flow.get("data_status"),
        }
    )
    usable = [item for item in sources if item["status"] != "unavailable"]
    stale = [item["name"] for item in usable if item["status"] == "stale"]
    unavailable = [item["name"] for item in sources if item["status"] == "unavailable"]
    return {
        "checked_at": now.isoformat(timespec="seconds"),
        "forecast_date": target.isoformat(),
        "overall_status": "stale" if stale else ("partial" if unavailable else "current"),
        "stale_sources": stale,
        "unavailable_sources": unavailable,
        "sources": sources,
        "note": "每個來源都取截至預測時間的最後有效資料；日期不必相同。狀態只檢查是否落後於該市場最近應完成的交易時段，遇休市則以來源實際交易日為準。",
    }


def freshness_item(name: str, latest, expected, source: str) -> dict:
    latest_date = pd.to_datetime(latest).date() if latest is not None else None
    expected_date = pd.to_datetime(expected).date()
    if latest_date is None:
        status_text = "unavailable"
        lag = None
    else:
        lag = (expected_date - latest_date).days
        status_text = "current" if lag <= 0 else "stale"
    return {
        "name": name,
        "latest_date": latest_date.isoformat() if latest_date else None,
        "expected_date": expected_date.isoformat(),
        "lag_days": lag,
        "status": status_text,
        "source": source,
    }


def previous_weekday(value):
    date = pd.Timestamp(value) - pd.Timedelta(days=1)
    while date.weekday() >= 5:
        date -= pd.Timedelta(days=1)
    return date.date()


def next_weekday(value):
    date = pd.Timestamp(value)
    while date.weekday() >= 5:
        date += pd.Timedelta(days=1)
    return date.date()


def expected_night_signal_date(target, now: datetime):
    """Return the latest night-session signal date that should be complete."""
    target_date = pd.Timestamp(target).date()
    if target_date.weekday() >= 5:
        return next_weekday(target_date)
    if target_date == now.date() and now.time() < time(5, 0):
        return previous_weekday(target_date)
    return target_date


def skipped_update() -> dict:
    return {"attempted": False, "success": False, "message": "Skipped by --no-update."}


def load_finmind_token() -> str:
    token = os.environ.get("FINMIND_TOKEN", "").strip()
    if token:
        return token
    token_file = ROOT / "config" / ".secrets" / "finmind_token.txt"
    if token_file.exists():
        return token_file.read_text(encoding="utf-8").strip()
    return ""


def update_twii_daily(output_path: Path, token: str = "") -> dict:
    result = {
        "attempted": True,
        "success": False,
        "source": "Yahoo Finance ^TWII chart + FinMind Sponsor TAIEX + TWSE official fallback",
        "latest_date": None,
        "row_count": 0,
        "message": "",
    }
    try:
        existing = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
        raw = fetch_yahoo_chart(YAHOO_SYMBOL)
        rows, fallback_used = yahoo_chart_to_rows(raw, return_metadata=True)
        if not rows:
            raise ValueError("Downloaded data is empty.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path = ROOT / "data" / "raw" / "yahoo_twii_chart.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

        frame = pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date")
        now_taipei = datetime.now(TAIPEI)
        frame, fresh_current_excluded = exclude_incomplete_current_session(frame, now_taipei)
        existing, existing_current_excluded = exclude_incomplete_current_session(existing, now_taipei)
        current_session_excluded = fresh_current_excluded or existing_current_excluded
        # Yahoo can temporarily omit the latest completed candle from a later
        # response. Merge by date so a successful refresh can never make the
        # local history go backward.
        if not existing.empty and "date" in existing:
            frame = pd.concat([existing, frame], ignore_index=True)
            frame = frame.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        # TWSE's official endpoints only contain completed sessions. During a
        # premarket run, repair the previous completed row instead of skipping
        # the fallback until 14:00 and leaving yesterday's Yahoo volume at zero.
        completed_target = completed_twii_target(now_taipei)
        finmind_used = False
        finmind_error = ""
        if token:
            try:
                finmind_start = completed_target - timedelta(days=10)
                finmind_rows = fetch_finmind_taiex_daily(
                    finmind_start.isoformat(),
                    completed_target.isoformat(),
                    token,
                )
                if finmind_rows:
                    finmind_frame = pd.DataFrame(finmind_rows)
                    frame = pd.concat([frame, finmind_frame], ignore_index=True)
                    frame = frame.drop_duplicates(subset=["date"], keep="last").sort_values("date")
                    finmind_used = True
            except Exception as exc:
                finmind_error = str(exc)
        latest_date = pd.to_datetime(frame["date"], errors="coerce").max()
        latest_date = latest_date.date() if pd.notna(latest_date) else None
        twse_used = False
        if latest_date is None or latest_date < completed_target:
            twse_rows = fetch_twse_taiex_month(completed_target)
            if twse_rows:
                twse_frame = pd.DataFrame(twse_rows)
                frame = pd.concat([frame, twse_frame], ignore_index=True)
                frame = frame.drop_duplicates(subset=["date"], keep="last").sort_values("date")
                twse_used = True
        latest_date = pd.to_datetime(frame["date"], errors="coerce").max()
        latest_date = latest_date.date() if pd.notna(latest_date) else None
        if latest_date is not None and latest_date <= completed_target:
            frame, twse_volume_used = patch_twse_market_volume(frame, latest_date)
        else:
            twse_volume_used = False
        frame.to_csv(output_path, index=False, encoding="utf-8")
        result.update(
            {
                "success": True,
                "source": (
                    "Yahoo Finance ^TWII chart"
                    + (" + FinMind Sponsor TaiwanStockPrice TAIEX" if finmind_used else "")
                    + (" + TWSE official fallback" if (twse_used or twse_volume_used) else "")
                ),
                "latest_date": str(frame.iloc[-1]["date"]),
                "row_count": int(len(frame)),
                "message": "Data updated." + (" Latest close recovered from Yahoo regularMarketPrice." if fallback_used else ""),
                "latest_close_fallback_used": fallback_used,
                "incomplete_current_session_excluded": current_session_excluded,
                "finmind_sponsor_taiex_used": finmind_used,
                "finmind_sponsor_taiex_error": finmind_error,
                "twse_official_fallback_used": twse_used,
                "twse_volume_fallback_used": twse_volume_used,
            }
        )
        if finmind_used:
            result["message"] += " Latest completed TAIEX candle reconciled from FinMind Sponsor."
        elif token and finmind_error:
            result["message"] += f" FinMind Sponsor TAIEX unavailable; fallback retained. Reason: {finmind_error}"
        if twse_used:
            result["message"] += " Latest completed TAIEX candle recovered from TWSE official index history."
        if twse_volume_used:
            result["message"] += " Latest market volume recovered from TWSE market statistics."
        if current_session_excluded:
            result["message"] += " Incomplete current-session candle was excluded before the 14:00 finalization boundary."
    except Exception as exc:
        result["message"] = f"Update failed; using local CSV. Reason: {exc}"
    return result


def exclude_incomplete_current_session(frame: pd.DataFrame, now_taipei: datetime) -> tuple[pd.DataFrame, bool]:
    """Keep the canonical daily file free of a still-open Taiwan session.

    Yahoo's 1d chart exposes today's live OHLC during cash trading.  That row is
    useful to the separate at-open logger but is not a completed daily candle
    and must never enter model history or the main daily report before 14:00.
    """
    if frame.empty or "date" not in frame or now_taipei.time() >= time(14, 0):
        return frame, False
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.date
    current = dates == now_taipei.date()
    return frame.loc[~current].copy(), bool(current.any())


def completed_twii_target(now_taipei: datetime):
    """Return the latest calendar target whose cash session should be complete."""
    if now_taipei.time() >= time(14, 0):
        return now_taipei.date()
    return previous_weekday(now_taipei.date())


def fetch_yahoo_chart(symbol: str) -> dict:
    # Yahoo's TWII series starts before 2000. Request the full available history
    # so independent cycle episodes are not discarded by an arbitrary cutoff.
    period1 = int(datetime(1990, 1, 1, tzinfo=timezone.utc).timestamp())
    period2 = int((datetime.now(timezone.utc) + timedelta(days=2)).timestamp())
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(symbol)}?period1={period1}&period2={period2}&interval=1d&events=history"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_twse_taiex_month(target_date) -> list[dict]:
    """Fetch completed TAIEX daily candles from TWSE for the target month."""
    target = pd.Timestamp(target_date).date()
    url = (
        "https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST"
        f"?date={target.strftime('%Y%m%d')}&response=json"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    # Some Windows Python certificate bundles reject TWSE's certificate chain.
    # This fallback only reads public index history, so use an unverified
    # context instead of failing the whole post-close refresh.
    context = ssl._create_unverified_context()
    with urlopen(request, timeout=30, context=context) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("stat") != "OK":
        return []

    rows = []
    for item in payload.get("data") or []:
        if len(item) < 5:
            continue
        row_date = parse_twse_roc_date(item[0])
        if row_date is None:
            continue
        rows.append(
            {
                "date": row_date.isoformat(),
                "open": parse_twse_number(item[1]),
                "high": parse_twse_number(item[2]),
                "low": parse_twse_number(item[3]),
                "close": parse_twse_number(item[4]),
                "adj_close": parse_twse_number(item[4]),
                "volume": 0,
            }
        )
    return rows


def fetch_finmind_taiex_daily(start_date: str, end_date: str, token: str) -> list[dict]:
    url = (
        "https://api.finmindtrade.com/api/v4/data?"
        f"dataset=TaiwanStockPrice&data_id=TAIEX&start_date={start_date}&end_date={end_date}"
        f"&token={quote(token)}"
    )
    request = Request(url, headers={"User-Agent": "market-lifecycle-research/1.0"})
    with urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") not in [200, "200", None]:
        raise RuntimeError(f"FinMind TAIEX failed: {payload.get('msg') or payload.get('status')}")
    rows = []
    for item in payload.get("data") or []:
        if str(item.get("stock_id")) != "TAIEX":
            continue
        close = safe_float(item.get("close"))
        if close is None:
            continue
        rows.append(
            {
                "date": str(item.get("date")),
                "open": safe_float(item.get("open")),
                "high": safe_float(item.get("max")),
                "low": safe_float(item.get("min")),
                "close": close,
                "adj_close": close,
                "volume": safe_float(item.get("Trading_Volume")),
            }
        )
    return rows


def patch_twse_market_volume(frame: pd.DataFrame, target_date) -> tuple[pd.DataFrame, bool]:
    if frame.empty or "date" not in frame or "volume" not in frame:
        return frame, False
    target = pd.Timestamp(target_date).date()
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.date
    matched = dates == target
    if not matched.any():
        return frame, False
    current_volume = pd.to_numeric(frame.loc[matched, "volume"], errors="coerce").fillna(0)
    if (current_volume > 0).all():
        return frame, False
    volume = fetch_twse_market_volume_thousand(target)
    if volume is None:
        return frame, False
    patched = frame.copy()
    patched.loc[matched, "volume"] = volume
    return patched, True


def fetch_twse_market_volume_thousand(target_date):
    target = pd.Timestamp(target_date).date()
    url = (
        "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
        f"?date={target.strftime('%Y%m%d')}&type=MS&response=json"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    context = ssl._create_unverified_context()
    with urlopen(request, timeout=30, context=context) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("stat") != "OK":
        return None
    for table in payload.get("tables") or []:
        title = str(table.get("title") or "")
        if "大盤統計資訊" not in title:
            continue
        for row in table.get("data") or []:
            if len(row) >= 3 and str(row[0]).startswith("證券合計"):
                shares = parse_twse_number(row[2])
                return shares / 1000 if shares is not None else None
    return None


def parse_twse_roc_date(value: str):
    parts = str(value).split("/")
    if len(parts) != 3:
        return None
    year = int(parts[0]) + 1911
    return pd.Timestamp(year=year, month=int(parts[1]), day=int(parts[2])).date()


def parse_twse_number(value: str):
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    return float(text) if text else None


def yahoo_chart_to_rows(raw: dict, return_metadata: bool = False):
    chart = raw.get("chart", {})
    if chart.get("error"):
        raise ValueError(chart["error"])
    results = chart.get("result") or []
    if not results:
        raise ValueError("Yahoo response has no result.")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_data = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjclose_data = (result.get("indicators", {}).get("adjclose") or [{}])[0]
    adjclose = adjclose_data.get("adjclose") or quote_data.get("close") or []

    meta = result.get("meta", {})
    regular_market_time = meta.get("regularMarketTime")
    regular_market_price = meta.get("regularMarketPrice")
    regular_market_date = (
        datetime.fromtimestamp(regular_market_time, TAIPEI).date()
        if regular_market_time else None
    )
    fallback_used = False
    rows = []
    for i, ts in enumerate(timestamps):
        close = value_at(quote_data.get("close"), i)
        row_date = datetime.fromtimestamp(ts, TAIPEI).date()
        # Yahoo occasionally publishes the completed TWII candle with OHLC but
        # leaves its close null. Yahoo may also append a future placeholder
        # timestamp, so recover the row matching regularMarketDate rather than
        # assuming it is the final array element.
        if (
            close is None
            and regular_market_date == row_date
            and regular_market_price is not None
        ):
            close = float(regular_market_price)
            fallback_used = True
        if close is None:
            continue
        rows.append(
            {
                "date": row_date.isoformat(),
                "open": value_at(quote_data.get("open"), i),
                "high": value_at(quote_data.get("high"), i),
                "low": value_at(quote_data.get("low"), i),
                "close": close,
                "adj_close": value_at(adjclose, i) or close,
                "volume": value_at(quote_data.get("volume"), i) or 0,
            }
        )
    return (rows, fallback_used) if return_metadata else rows


def fetch_live_monitor_snapshot(forecast_date: str, manual_index: float | None = None) -> dict:
    target = pd.to_datetime(forecast_date).date()
    snapshot = {
        "enabled": False,
        "source": "none",
        "date": None,
        "price": safe_float(manual_index),
        "open": None,
        "high": None,
        "low": safe_float(manual_index),
        "message": "",
    }
    if manual_index is not None:
        snapshot.update(
            {
                "enabled": True,
                "source": "manual --index",
                "date": target.isoformat(),
                "message": "使用者輸入即時指數，風控監控優先採用。",
            }
        )
        return snapshot
    if target != datetime.now(TAIPEI).date():
        snapshot["message"] = "非今日報告，不抓盤中快照。"
        return snapshot
    try:
        raw = fetch_yahoo_chart(YAHOO_SYMBOL)
        chart = raw.get("chart", {})
        result = (chart.get("result") or [{}])[0]
        meta = result.get("meta", {})
        rows = yahoo_chart_to_rows(raw)
        row = None
        for item in reversed(rows):
            if pd.to_datetime(item["date"]).date() == target:
                row = item
                break
        regular_time = meta.get("regularMarketTime")
        regular_date = datetime.fromtimestamp(regular_time, TAIPEI).date() if regular_time else None
        if row is None and regular_date == target:
            row = {
                "date": target.isoformat(),
                "open": meta.get("regularMarketDayOpen"),
                "high": meta.get("regularMarketDayHigh"),
                "low": meta.get("regularMarketDayLow"),
                "close": meta.get("regularMarketPrice"),
            }
        if row is None:
            snapshot["message"] = "Yahoo 尚未提供今日盤中列。"
            return snapshot
        snapshot.update(
            {
                "enabled": True,
                "source": "Yahoo ^TWII live chart",
                "date": row["date"],
                "price": safe_float(row.get("close")),
                "open": safe_float(row.get("open")),
                "high": safe_float(row.get("high")),
                "low": safe_float(row.get("low")),
                "message": "已納入盤中風控快照；此資料只用於風險預警，不寫入日線模型歷史。",
            }
        )
    except Exception as exc:
        snapshot["message"] = f"盤中快照抓取失敗: {exc}"
    return snapshot


def value_at(values: list | None, index: int):
    if not values or index >= len(values):
        return None
    value = values[index]
    if value is None:
        return None
    return float(value)


def lookup_close(scored: pd.DataFrame, date: str) -> float:
    target = pd.to_datetime(date)
    eligible = scored[scored["date"] <= target]
    if eligible.empty:
        raise ValueError("No market data is available on or before " + date)
    return float(eligible.iloc[-1]["close"])


def check_input_index(scored: pd.DataFrame, date: str, index_value: float, provided: bool) -> dict:
    target = pd.to_datetime(date)
    eligible = scored[scored["date"] <= target]
    if eligible.empty:
        raise ValueError("No market data is available on or before " + date)
    row = eligible.iloc[-1]
    data_close = float(row["close"])
    diff = index_value - data_close
    diff_pct = diff / data_close if data_close else 0.0
    return {
        "signal_date": str(row["date"].date()),
        "provided": bool(provided),
        "data_close": data_close,
        "difference": diff,
        "difference_pct": diff_pct,
        "is_consistent": abs(diff_pct) <= 0.0001 if not provided else abs(diff_pct) <= 0.01,
    }


def build_market_date_audit(
    forecast_date: str,
    index_check: dict,
    live_monitor: dict | None,
    premarket: dict | None,
) -> dict:
    forecast_day = pd.to_datetime(forecast_date).date()
    signal_day = pd.to_datetime(index_check.get("signal_date")).date()
    live_monitor = live_monitor or {}
    premarket = premarket or {}
    live_day = None
    if live_monitor.get("date"):
        live_day = pd.to_datetime(live_monitor.get("date")).date()
    has_today_live = bool(live_monitor.get("enabled")) and live_day == forecast_day
    official_daily_complete = signal_day == forecast_day
    is_premarket = bool(premarket.get("is_premarket"))
    is_non_trading_day = bool(premarket.get("is_non_trading_day"))

    if official_daily_complete:
        mode = "official_daily"
        label = "正式日線"
        status_text = "可用正式日線資料描述當日收盤。"
    elif is_non_trading_day:
        mode = "non_trading_day"
        label = "非交易日觀察"
        status_text = "今日不是台股現貨交易日；只能用最後有效日線與外部市場作下個交易日觀察。"
    elif has_today_live:
        mode = "intraday_snapshot"
        label = "今日盤中快照"
        status_text = "今日尚無正式日線，只能描述盤中快照，不得稱為今日收盤。"
    elif is_premarket:
        mode = "premarket"
        label = "盤前模式"
        status_text = "台股現貨尚未有今日正式資料，只能比對夜盤與外部市場。"
    else:
        mode = "stale_daily"
        label = "最後有效日線"
        status_text = "正式日線落後報告日期，禁止直接推論今日漲跌。"

    guardrails = [
        f"報告日期 {forecast_day.isoformat()} 與正式日線基準日 {signal_day.isoformat()} 必須分開標示。",
        "只有 signal_date 等於報告日期時，才能使用『今日收盤』描述日線資料。",
        "今日盤中資料只能標為即時快照；不得與前一交易日正式收盤混寫。",
    ]
    if is_non_trading_day:
        guardrails.append("非交易日不得寫成盤前開盤預測；只能標示為週末/休市觀察。")
    if has_today_live:
        guardrails.append(
            f"今日盤中快照 {live_day.isoformat()}：價格 {num(live_monitor.get('price'))}，低點 {num(live_monitor.get('low'))}。"
        )

    return {
        "mode": mode,
        "label": label,
        "status": status_text,
        "forecast_date": forecast_day.isoformat(),
        "official_signal_date": signal_day.isoformat(),
        "official_daily_complete": official_daily_complete,
        "has_today_live_snapshot": has_today_live,
        "live_snapshot_date": live_day.isoformat() if live_day else None,
        "live_price": safe_float(live_monitor.get("price")) if has_today_live else None,
        "live_low": safe_float(live_monitor.get("low")) if has_today_live else None,
        "must_not_call_signal_date_today": not official_daily_complete,
        "guardrails": guardrails,
    }


def add_forecast_target_dates(forecast: dict, scored: pd.DataFrame, signal_date: str) -> None:
    trading_dates = list(scored.sort_values("date")["date"].dt.date)
    base_date = pd.to_datetime(signal_date).date()
    if base_date in trading_dates:
        base_index = trading_dates.index(base_date)
    else:
        base_index = max(i for i, date in enumerate(trading_dates) if date <= base_date)

    last_known_date = trading_dates[-1]
    for item in forecast.get("forecasts", []):
        horizon = int(item["horizon_days"])
        target_index = base_index + horizon
        if target_index < len(trading_dates):
            target_date = trading_dates[target_index]
            estimated = False
        else:
            target_date = estimate_future_trading_date(last_known_date, target_index - len(trading_dates) + 1)
            estimated = True
        item["base_trade_date"] = str(base_date)
        item["target_trade_date"] = str(target_date)
        item["target_date_estimated"] = estimated
        item["date_note"] = (
            f"以 {base_date} 為第0個交易日，第{horizon}個交易日約為 {target_date}"
            + ("，因資料尚未包含未來交易日，日期為估算。" if estimated else "。")
        )
    forecast["base_trade_date"] = str(base_date)


def estimate_future_trading_date(last_known_date, trading_days_ahead: int):
    date = pd.to_datetime(last_known_date).date()
    remaining = trading_days_ahead
    while remaining > 0:
        date = date + timedelta(days=1)
        if date.weekday() < 5:
            remaining -= 1
    return date


def build_self_review(
    scored: pd.DataFrame,
    forecast_date: str,
    signal_date: str,
    forecast: dict,
    premarket: dict,
    bagua: dict,
    tradeable_cycle: dict,
    candle: dict,
    washout: dict,
) -> dict:
    history = load_forecast_history(FORECAST_HISTORY)
    previous = latest_prior_record(history, forecast_date)
    drift = compare_with_previous_forecast(previous, forecast)
    tracking = compare_daily_tracking(previous, bagua, tradeable_cycle, candle, washout)
    matured = evaluate_matured_predictions(history, scored, signal_date)
    correction = correction_suggestion(drift, matured, premarket)
    return {
        "enabled": True,
        "history_file": str(FORECAST_HISTORY),
        "previous_forecast_date": previous.get("input_date") if previous else None,
        "prediction_drift": drift,
        "daily_tracking": tracking,
        "matured_checks": matured,
        "correction_suggestion": correction,
        "plain_summary": self_review_summary(drift, matured, correction, tracking),
    }


def load_forecast_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        # Historical forecast rows produced from a same-day Yahoo live candle
        # before 14:00 are invalid completed-day records. Keep them out of drift
        # and matured-score calculations; a migration preserves them separately.
        return [item for item in data if not is_intraday_forecast_history_record(item)]
    except Exception:
        return []


def is_intraday_forecast_history_record(item: dict) -> bool:
    try:
        run_at = pd.Timestamp(item.get("run_at"))
        if run_at.tzinfo is None:
            run_at = run_at.tz_localize(TAIPEI)
        else:
            run_at = run_at.tz_convert(TAIPEI)
        signal_date = pd.Timestamp(item.get("signal_date")).date()
        return signal_date == run_at.date() and run_at.time() < time(14, 0)
    except Exception:
        return False


def upsert_forecast_history(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = load_forecast_history(path)
    compact = compact_forecast_record(payload)
    key = (compact["input_date"], compact["signal_date"])
    kept = [item for item in records if (item.get("input_date"), item.get("signal_date")) != key]
    kept.append(compact)
    kept = sorted(kept, key=lambda item: (item.get("input_date") or "", item.get("signal_date") or ""))[-260:]
    path.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_forecast_record(payload: dict) -> dict:
    forecast = payload["forecast"]
    return {
        "run_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "input_date": payload["input"]["date"],
        "signal_date": payload["index_check"]["signal_date"],
        "close": payload["index_check"]["data_close"],
        "premarket_pressure": payload["premarket"]["pressure"],
        "premarket_score": payload["premarket"]["total_score"],
        "scenario": payload["cause_analysis"]["scenario"]["name"],
        "truth_text": payload["cause_analysis"]["truth_text"],
        "lifecycle_stage": forecast["lifecycle_stage"],
        "risk_regime": forecast["risk_regime"],
        "forecast_status": forecast.get("validation_status", "exploratory_not_production"),
        "formal_multi_day_signal_enabled": bool(
            payload.get("production_policy", {})
            .get("main_multi_day_direction", {})
            .get("enabled", False)
        ),
        "bagua_tracking": compact_bagua_tracking(payload.get("bagua_lifecycle", {})),
        "tradeable_cycle_tracking": compact_tradeable_cycle_tracking(payload.get("tradeable_cycle", {})),
        "candlestick_tracking": compact_candlestick_tracking(payload.get("candlestick_pattern", {})),
        "washout_tracking": compact_washout_tracking(payload.get("washout_pattern", {})),
        "treatment_tracking": compact_treatment_tracking(payload.get("practical_cause_arbitration", {})),
        "forecasts": [
            {
                "horizon_days": item["horizon_days"],
                "base_trade_date": item.get("base_trade_date"),
                "target_trade_date": item.get("target_trade_date"),
                "target_date_estimated": item.get("target_date_estimated"),
                "predicted_direction": item["predicted_direction"],
                "confidence": item["confidence"],
                "probability_up": item["probability_up"],
                "probability_down": item["probability_down"],
                "probability_sideways": item["probability_sideways"],
                "fixed_factor_review": item.get("fixed_factor_review"),
            }
            for item in forecast["forecasts"]
        ],
    }


def compact_treatment_tracking(practical: dict) -> dict:
    """Persist disease gene / trigger / treatment response context for future episodes."""
    if not practical:
        return {}
    internal_score = safe_int(practical.get("internal_structure_score"), 0)
    trigger_score = safe_int(practical.get("external_trigger_score"), 0)
    phase = safe_int(practical.get("treatment_phase"), 0)
    after_effect = practical.get("after_effect_risk")
    if phase >= 3 or after_effect in {"中", "高"}:
        next_watch = "下期同類病灶出現時，先核對上次是否留下陰霾與是否更快發作。"
    elif internal_score >= 5 and trigger_score < 3:
        next_watch = "病灶基因仍在但觸發不足，下期重點看外部開關是否補上。"
    else:
        next_watch = "維持一般觀察，累積樣本後再判斷是否形成慣性病程。"
    return {
        "framework": practical.get("framework"),
        "code": practical.get("code"),
        "label": practical.get("label"),
        "practical_primary": practical.get("practical_primary"),
        "internal_structure_score": internal_score,
        "external_trigger_score": trigger_score,
        "combined_severity": safe_int(practical.get("combined_severity"), 0),
        "treatment_phase": phase,
        "treatment_stage": practical.get("treatment_stage"),
        "medicine_type": practical.get("medicine_type"),
        "correct_medicine": practical.get("correct_medicine"),
        "after_effect_risk": after_effect,
        "fact_changing_medicines": top_items(practical.get("fact_changing_medicines", []), 6),
        "immediate_medicine_bias": practical.get("immediate_medicine_bias"),
        "immediate_medicine_score": safe_int(practical.get("immediate_medicine_score"), 0),
        "causality_rule": practical.get("causality_rule"),
        "causality_pipeline": practical.get("causality_pipeline", []),
        "next_episode_prior": next_watch,
        "internal_causes": top_items(practical.get("internal_causes", []), 4),
        "external_triggers": top_items(practical.get("external_triggers", []), 4),
    }


def compact_bagua_tracking(bagua: dict) -> dict:
    primary = bagua.get("primary", {})
    monthly = bagua.get("monthly", {})
    weekly = bagua.get("weekly", {})
    daily = bagua.get("daily", {})
    resonance = bagua.get("resonance", {})
    kline = bagua.get("kline_confirmation", {})
    roles = bagua.get("roles", {})
    background = roles.get("background_gua", {})
    state = roles.get("state_gua", {})
    prices = roles.get("price_gua", {}).get("windows", [])
    return {
        "primary_gua": primary.get("gua"),
        "primary_code": primary.get("code"),
        "primary_label": primary.get("label"),
        "phase_progress": primary.get("phase_progress"),
        "next_gua": primary.get("next", {}).get("gua"),
        "fallback_gua": primary.get("fallback", {}).get("gua"),
        "monthly_gua": monthly.get("gua"),
        "weekly_gua": weekly.get("gua"),
        "daily_gua": daily.get("gua"),
        "resonance_label": resonance.get("label"),
        "kline_confirmation": kline.get("label"),
        "background_gua": background.get("gua"),
        "state_gua": state.get("gua"),
        "state_status": state.get("status"),
        "price_gua_120": next((item.get("gua") for item in prices if item.get("window") == 120), None),
        "price_gua_60": next((item.get("gua") for item in prices if item.get("window") == 60), None),
        "price_gua_20": next((item.get("gua") for item in prices if item.get("window") == 20), None),
    }


def compact_tradeable_cycle_tracking(cycle: dict) -> dict:
    position = cycle.get("current_position", {})
    stats = cycle.get("stats", {})
    frequency = stats.get("frequency_state", {})
    return {
        "position_code": position.get("code"),
        "position_label": position.get("label"),
        "current_year_count": stats.get("current_year_count"),
        "rolling_252d_count": stats.get("rolling_252d_count"),
        "average_per_year": stats.get("average_per_year"),
        "frequency_label": frequency.get("label"),
        "frequency_level": frequency.get("level"),
    }


def compact_candlestick_tracking(candle: dict) -> dict:
    return {
        "type": candle.get("type"),
        "label": candle.get("label"),
        "summary": candle.get("plain_summary"),
    }


def compact_washout_tracking(washout: dict) -> dict:
    return {
        "type": washout.get("type"),
        "label": washout.get("label"),
        "summary": washout.get("plain_summary"),
    }


def latest_prior_record(history: list[dict], forecast_date: str) -> dict | None:
    prior = [item for item in history if str(item.get("input_date", "")) <= forecast_date]
    if not prior:
        return None
    return sorted(prior, key=lambda item: (item.get("input_date") or "", item.get("run_at") or ""))[-1]


def compare_with_previous_forecast(previous: dict | None, current_forecast: dict) -> dict:
    if not previous:
        return {"available": False, "summary": "沒有前次探索性分布紀錄可比較。", "rows": []}
    prior_by_horizon = {int(item["horizon_days"]): item for item in previous.get("forecasts", [])}
    rows = []
    changed = 0
    for current in current_forecast.get("forecasts", []):
        horizon = int(current["horizon_days"])
        prior = prior_by_horizon.get(horizon)
        if not prior:
            continue
        direction_changed = prior["predicted_direction"] != current["predicted_direction"]
        if direction_changed:
            changed += 1
        rows.append(
            {
                "horizon_days": horizon,
                "previous_direction": prior["predicted_direction"],
                "current_direction": current["predicted_direction"],
                "direction_changed": direction_changed,
                "previous_confidence": float(prior.get("confidence", 0)),
                "current_confidence": float(current.get("confidence", 0)),
                "confidence_delta": float(current.get("confidence", 0)) - float(prior.get("confidence", 0)),
            }
        )
    summary = (
        "前次與本次探索性歷史多數方向大致一致。"
        if changed == 0
        else f"有 {changed} 個期間的探索性歷史多數方向改變；這是分布變化，不是正式訊號轉向。"
    )
    return {"available": True, "summary": summary, "rows": rows}


def compare_daily_tracking(
    previous: dict | None,
    bagua: dict,
    tradeable_cycle: dict,
    candle: dict,
    washout: dict,
) -> dict:
    current = {
        "bagua": compact_bagua_tracking(bagua),
        "tradeable_cycle": compact_tradeable_cycle_tracking(tradeable_cycle),
        "candlestick": compact_candlestick_tracking(candle),
        "washout": compact_washout_tracking(washout),
    }
    if not previous:
        return {
            "available": False,
            "summary": "尚無前次週期追蹤紀錄，本次起開始累積八卦、波段、K線與洗盤追蹤。",
            "current": current,
            "rows": [],
            "alerts": daily_tracking_alerts(current),
        }

    previous_tracking = {
        "bagua": previous.get("bagua_tracking", {}),
        "tradeable_cycle": previous.get("tradeable_cycle_tracking", {}),
        "candlestick": previous.get("candlestick_tracking", {}),
        "washout": previous.get("washout_tracking", {}),
    }
    rows = [
        tracking_row("背景卦", previous_tracking["bagua"].get("background_gua"), current["bagua"].get("background_gua")),
        tracking_row("狀態卦", previous_tracking["bagua"].get("state_gua"), current["bagua"].get("state_gua")),
        tracking_row("價位卦120/60/20", format_price_gua_stack(previous_tracking["bagua"]), format_price_gua_stack(current["bagua"])),
        tracking_row("舊版主卦", previous_tracking["bagua"].get("primary_gua"), current["bagua"].get("primary_gua")),
        tracking_row("月/週/日", format_bagua_stack(previous_tracking["bagua"]), format_bagua_stack(current["bagua"])),
        tracking_row("共振", previous_tracking["bagua"].get("resonance_label"), current["bagua"].get("resonance_label")),
        tracking_row("波段段位", previous_tracking["tradeable_cycle"].get("position_label"), current["tradeable_cycle"].get("position_label")),
        tracking_row("波段頻率", previous_tracking["tradeable_cycle"].get("frequency_label"), current["tradeable_cycle"].get("frequency_label")),
        tracking_row("K線", previous_tracking["candlestick"].get("label"), current["candlestick"].get("label")),
        tracking_row("洗盤", previous_tracking["washout"].get("label"), current["washout"].get("label")),
    ]
    changed = [row for row in rows if row["changed"]]
    alerts = daily_tracking_alerts(current)
    if changed:
        summary = f"週期追蹤有 {len(changed)} 項變化，需檢查是否為轉卦、波段失守或K線確認。"
    else:
        summary = "週期追蹤與前次大致一致，持續觀察確認條件。"
    if alerts:
        summary += " " + " ".join(alerts)
    return {
        "available": True,
        "summary": summary,
        "previous": previous_tracking,
        "current": current,
        "rows": rows,
        "alerts": alerts,
    }


def tracking_row(name: str, previous, current) -> dict:
    return {
        "name": name,
        "previous": previous or "NA",
        "current": current or "NA",
        "changed": (previous or "NA") != (current or "NA"),
    }


def format_bagua_stack(tracking: dict) -> str:
    values = [tracking.get("monthly_gua"), tracking.get("weekly_gua"), tracking.get("daily_gua")]
    return "/".join(value or "NA" for value in values)


def format_price_gua_stack(tracking: dict) -> str:
    values = [
        tracking.get("price_gua_120"),
        tracking.get("price_gua_60"),
        tracking.get("price_gua_20"),
    ]
    return "/".join(value or "NA" for value in values)


def daily_tracking_alerts(current: dict) -> list[str]:
    alerts = []
    bagua = current.get("bagua", {})
    cycle = current.get("tradeable_cycle", {})
    candle = current.get("candlestick", {})
    washout = current.get("washout", {})
    if bagua.get("resonance_label") == "高位入坎":
        alerts.append("高位入坎仍需追蹤，隔日若無法修復，週線高檔承載可能降級。")
    if cycle.get("position_code") == "post_rally_pullback":
        alerts.append("年度波段已進入回落找底，應追蹤是否出現新低點或洗盤成功。")
    if candle.get("type") in ["long_bear", "gap_up_failed"] and washout.get("type") == "failed_washout":
        alerts.append("K線與洗盤同步偏弱，需追蹤是否跌破長黑K低點。")
    if cycle.get("frequency_level") in ["high", "extreme_high"]:
        alerts.append("今年屬高頻波段環境，每段延續時間可能縮短，追價需更保守。")
    return alerts


def evaluate_matured_predictions(history: list[dict], scored: pd.DataFrame, current_signal_date: str) -> dict:
    if not history:
        return {"available": False, "summary": "沒有探索性歷史分布紀錄可核對。", "rows": []}
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date_key"] = data["date"].dt.date.astype(str)
    current_index = int(data[data["date"] <= pd.to_datetime(current_signal_date)].index[-1])
    rows = []
    for record in history[-80:]:
        signal_date = record.get("signal_date")
        match = data.index[data["date_key"] == signal_date].tolist()
        if not match:
            continue
        start_index = match[-1]
        start_close = float(data.loc[start_index, "close"])
        for item in record.get("forecasts", []):
            horizon = int(item["horizon_days"])
            target_index = start_index + horizon
            if target_index > current_index or target_index >= len(data):
                continue
            target_date = str(data.loc[target_index, "date"].date())
            forecast_date = str(record.get("input_date") or "")
            if forecast_date and forecast_date > target_date:
                continue
            if is_after_close_same_day_replay(record.get("run_at"), forecast_date, target_date):
                continue
            actual_close = float(data.loc[target_index, "close"])
            actual_return = actual_close / start_close - 1
            actual_direction = direction_from_return(actual_return)
            predicted = item["predicted_direction"]
            rows.append(
                {
                    "forecast_date": record.get("input_date"),
                    "signal_date": signal_date,
                    "target_date": target_date,
                    "horizon_days": horizon,
                    "predicted_direction": predicted,
                    "actual_direction": actual_direction,
                    "actual_return": actual_return,
                    "is_hit": predicted == actual_direction,
                    "forecast_status": record.get("forecast_status"),
                    "premarket_pressure": record.get("premarket_pressure"),
                    "premarket_score": record.get("premarket_score"),
                    "scenario": record.get("scenario"),
                    "lifecycle_stage": record.get("lifecycle_stage"),
                    "risk_regime": record.get("risk_regime"),
                    "bagua_tracking": record.get("bagua_tracking", {}),
                    "tradeable_cycle_tracking": record.get("tradeable_cycle_tracking", {}),
                    "candlestick_tracking": record.get("candlestick_tracking", {}),
                    "washout_tracking": record.get("washout_tracking", {}),
                }
            )
    if not rows:
        return {"available": False, "summary": "尚無到期探索性分布可核對。", "rows": []}
    recent = rows[-30:]
    hit_rate = sum(1 for row in recent if row["is_hit"]) / len(recent)
    misses = [row for row in recent if not row["is_hit"]]
    summary = (
        f"最近 {len(recent)} 筆探索性歷史多數方向核對率 {hit_rate:.1%}；"
        "此短期追蹤不取代正式walk-forward結案結果。"
    )
    return {"available": True, "summary": summary, "hit_rate": hit_rate, "rows": recent, "miss_count": len(misses)}


def is_after_close_same_day_replay(run_at: str | None, forecast_date: str, target_date: str) -> bool:
    if not run_at or forecast_date != target_date:
        return False
    try:
        run_time = datetime.fromisoformat(str(run_at)).astimezone(TAIPEI).time()
    except (TypeError, ValueError):
        return False
    return run_time >= time(13, 35)


def build_error_review(self_review: dict) -> dict:
    matured = self_review.get("matured_checks", {})
    rows = matured.get("rows", [])
    misses = [row for row in rows if row.get("is_hit") is False]
    cases = [classify_error_case(row) for row in misses]
    groups = summarize_error_groups(cases)
    next_day_validation = build_next_day_error_validation(cases, groups, matured)
    return {
        "enabled": True,
        "framework": "auditable_error_review_v1",
        "source": self_review.get("history_file"),
        "available": bool(matured.get("available")),
        "hit_rate": matured.get("hit_rate"),
        "total_checked_recent": len(rows),
        "miss_count": len(misses),
        "cases": cases,
        "groups": groups,
        "next_day_validation": next_day_validation,
        "governance": {
            "historical_records_immutable": True,
            "automatic_parameter_tuning": False,
            "automatic_threshold_tuning": False,
            "automatic_promotion_allowed": False,
            "independent_validation_required": True,
        },
        "summary": error_review_summary(matured, groups, len(misses)),
    }


def build_next_day_error_validation(cases: list[dict], groups: list[dict], matured: dict) -> dict:
    if not matured.get("available"):
        return {
            "enabled": False,
            "headline": "尚無到期預測可驗證",
            "summary": "沒有到期紀錄，暫不建立次日強制驗證項目。",
            "tasks": [],
        }
    if not cases:
        return {
            "enabled": False,
            "headline": "昨日預測未出現失準病例",
            "summary": "近期到期預測沒有失準，次日只維持一般監控。",
            "tasks": [],
        }

    latest_case = sorted(
        cases,
        key=lambda item: (
            str(item.get("target_date") or ""),
            safe_int(item.get("horizon_days"), 0),
            str(item.get("forecast_date") or ""),
        ),
        reverse=True,
    )[0]
    top_group = groups[0] if groups else {
        "code": latest_case.get("primary_error_code"),
        "label": latest_case.get("primary_error_label"),
        "count": 1,
    }
    latest_code = latest_case.get("primary_error_code")
    latest_label = latest_case.get("primary_error_label")
    tasks = validation_tasks_for_error_code(latest_code, latest_case)
    return {
        "enabled": True,
        "headline": f"昨日誤差類型/最新：{latest_label or '未分類'}",
        "summary": (
            f"最近失準到期日 {latest_case.get('target_date')}；"
            f"預測 {direction_text(latest_case.get('predicted_direction'))}，"
            f"實際 {direction_text(latest_case.get('actual_direction'))}，"
            f"報酬 {pct(latest_case.get('actual_return'))}。今日必須驗證同類錯誤是否延續。"
        ),
        "latest_case": latest_case,
        "latest_error_code": latest_code,
        "latest_error_label": latest_label,
        "top_error_code": top_group.get("code"),
        "top_error_label": top_group.get("label"),
        "top_error_count": top_group.get("count"),
        "tasks": tasks,
        "governance": "只作次日驗證與降權提醒，不自動調參、不自動改歷史。",
    }


def validation_tasks_for_error_code(code: str | None, latest_case: dict) -> list[dict]:
    task_map = {
        "neutral_threshold_mismatch": [
            ("盤整誤判驗證", "檢查今日實際波動是否仍落在盤整閾值附近，避免把低波動硬判成方向。"),
            ("突破確認", "若要恢復方向判讀，需看是否有效突破前一日高低區間，而不是只看小漲小跌。"),
            ("夜盤牽日盤核對", "核對夜盤方向是否只影響開盤，收盤是否被現貨承接或壓回。"),
        ],
        "premarket_gap_mismatch": [
            ("盤前缺口驗證", "檢查夜盤與外部市場是否只造成開盤缺口，不能直接外推成整日方向。"),
            ("日盤反證", "若日盤收盤反向收回，隔日需降低夜盤單因子權重的解讀強度。"),
        ],
        "bearish_reversal_absorption_mismatch": [
            ("偏空假設反轉驗證", "檢查偏空預測後，日盤是否開低不破、站回夜盤開盤、收近日高或放量收復。"),
            ("夜盤中性防呆", "若夜盤接近0%，不得把夜盤寫成偏空；日盤方向需交給現貨承接、權值股輪動與開盤後改價驗證。"),
            ("病灶留底", "把此類病例歸入偏空假設被日盤強承接推翻，而不是未分類雜訊。"),
        ],
        "bagua_phase_mismatch": [
            ("坎轉艮確認", "確認是否不再破低並站回短均，否則不得提早判成震啟動。"),
            ("卦位防呆", "急跌與破位仍歸坎，只有止跌後修復才可進入艮或震。"),
        ],
        "high_level_reversal_mismatch": [
            ("高檔轉弱驗證", "檢查高檔卦位是否出現長上影、爆量不漲、開高走低或跌破短均。"),
            ("入坎條件", "若跌破重大修正或連續收不回，需把高檔換手升級為危機初期。"),
        ],
        "kline_confirmation_mismatch": [
            ("K線確認", "若前日K線偏弱，今日必須先看是否收復弱K高點或至少守住低點。"),
            ("方向降權", "未確認前，歷史相似多方只能留底，不能作頭條方向。"),
        ],
        "washout_mismatch": [
            ("洗盤真假驗證", "檢查是否快速站回失守線；若隔日續破低點，洗盤成功判讀失效。"),
            ("承接確認", "需看量價是否出現低檔承接，而非只因盤中拉回就判定落底。"),
        ],
        "cycle_position_mismatch": [
            ("波段位置驗證", "回落找底階段不得只因反彈就判成新升段，需確認高低點墊高。"),
            ("週期續航", "日、週、月三層若不同步，短線反彈需標為修復，不標為完整起漲。"),
        ],
        "unclassified_market_noise": [
            ("事件雜訊留底", "檢查是否有外部事件、政策、匯率、期貨或權值股單日擾動。"),
            ("樣本等待", "未能歸因時只保留病例，不新增規則，避免過度擬合。"),
        ],
    }
    rows = task_map.get(code) or task_map["unclassified_market_noise"]
    return [
        {
            "name": name,
            "rule": rule,
            "source_error_code": code or "unknown",
            "source_target_date": latest_case.get("target_date"),
        }
        for name, rule in rows
    ]


def classify_error_case(row: dict) -> dict:
    predicted = row.get("predicted_direction")
    actual = row.get("actual_direction")
    actual_return = safe_float(row.get("actual_return")) or 0
    horizon = safe_int(row.get("horizon_days"), 0)
    bagua = row.get("bagua_tracking", {})
    candle = row.get("candlestick_tracking", {})
    washout = row.get("washout_tracking", {})
    cycle = row.get("tradeable_cycle_tracking", {})
    premarket_pressure = row.get("premarket_pressure")
    reasons = []

    if horizon <= 3 and predicted == "down" and actual == "up" and actual_return >= 0.01:
        reasons.append(error_reason(
            "bearish_reversal_absorption_mismatch",
            "偏空預測被日盤現貨強承接反轉。",
            "將盤前/外部偏空或舊偏空假設降為壓力測試；若夜盤近0且日盤強收，改列日盤自主改價病例。",
        ))
    if abs(actual_return) <= 0.012 and predicted != "sideways":
        reasons.append(error_reason("neutral_threshold_mismatch", "實際報酬接近盤整，但探索性分布給出方向。", "檢查盤整閾值與震盪行情分類。"))
    if horizon <= 3 and premarket_pressure in ["偏多", "偏空"] and predicted != actual:
        reasons.append(error_reason("premarket_gap_mismatch", "短天期結果可能受盤前/夜盤缺口主導。", "僅檢查開盤缺口模型，不擴大到收盤或多日方向。"))
    if bagua.get("primary_code") == "KAN" and predicted == "up" and actual == "down":
        reasons.append(error_reason("bagua_phase_mismatch", "主卦仍在坎，反彈預測過早。", "檢查坎轉艮、艮轉震確認條件是否不足。"))
    if bagua.get("primary_code") in ["KUN", "DUI", "QIAN"] and actual == "down":
        reasons.append(error_reason("high_level_reversal_mismatch", "高檔卦位後續轉弱，歷史相似分布可能低估轉折。", "檢查高檔入坎權重與失效條件。"))
    if candle.get("type") in ["long_bear", "gap_up_failed"] and predicted == "up" and actual != "up":
        reasons.append(error_reason("kline_confirmation_mismatch", "K線偏弱但探索性方向偏多。", "檢查K線確認是否應降權歷史相似多方。"))
    if washout.get("type") == "failed_washout" and predicted == "up" and actual != "up":
        reasons.append(error_reason("washout_mismatch", "洗盤失敗後仍預測上行。", "檢查失敗洗盤對短中期方向的抑制。"))
    if cycle.get("position_code") == "post_rally_pullback" and predicted == "up" and actual == "down":
        reasons.append(error_reason("cycle_position_mismatch", "波段仍在回落找底，反彈預測過早。", "檢查波段位置與預測天期是否錯配。"))
    if not reasons:
        reasons.append(error_reason("unclassified_market_noise", "未能由現有標籤清楚歸因，可能是事件雜訊或特徵不足。", "只留存病例，等待更多樣本後再研究。"))

    primary = reasons[0]
    return {
        "forecast_date": row.get("forecast_date"),
        "signal_date": row.get("signal_date"),
        "target_date": row.get("target_date"),
        "horizon_days": horizon,
        "predicted_direction": predicted,
        "actual_direction": actual,
        "actual_return": actual_return,
        "primary_error_code": primary["code"],
        "primary_error_label": primary["label"],
        "reasons": reasons,
        "context": {
            "premarket_pressure": premarket_pressure,
            "scenario": row.get("scenario"),
            "primary_gua": bagua.get("primary_gua"),
            "primary_code": bagua.get("primary_code"),
            "resonance_label": bagua.get("resonance_label"),
            "kline": candle.get("label"),
            "washout": washout.get("label"),
            "cycle_position": cycle.get("position_label"),
        },
    }


def error_reason(code: str, label: str, candidate: str) -> dict:
    return {
        "code": code,
        "label": label,
        "candidate_adjustment": candidate,
        "automatic_change_allowed": False,
    }


def summarize_error_groups(cases: list[dict]) -> list[dict]:
    groups = {}
    for case in cases:
        code = case.get("primary_error_code", "unknown")
        label = case.get("primary_error_label", code)
        item = groups.setdefault(code, {"code": code, "label": label, "count": 0, "horizons": {}})
        item["count"] += 1
        horizon = str(case.get("horizon_days", "unknown"))
        item["horizons"][horizon] = item["horizons"].get(horizon, 0) + 1
    return sorted(groups.values(), key=lambda item: item["count"], reverse=True)


def error_review_summary(matured: dict, groups: list[dict], miss_count: int) -> str:
    if not matured.get("available"):
        return "尚無到期預測可建立誤差分類。"
    if miss_count == 0:
        return "近期到期預測沒有失準病例，暫不建立修正候選。"
    top = groups[0] if groups else {"label": "未分類", "count": miss_count}
    return (
        f"近期到期預測失準 {miss_count} 筆，最大類別為 {top['label']}（{top['count']} 筆）。"
        " 本報告只建立候選假設，不自動調整權重或門檻。"
    )


def correction_suggestion(drift: dict, matured: dict, premarket: dict) -> dict:
    actions = []
    if premarket.get("is_non_trading_day"):
        actions.append("目前屬於非交易日觀察；外部市場只作下個交易日情境，不可寫成今日開盤預測。")
    elif premarket.get("is_premarket"):
        actions.append("目前屬於盤前模式；外部市場只作情境觀察，正式方向僅能使用已驗證的開盤缺口子模型。")
    if drift.get("available") and any(row.get("direction_changed") for row in drift.get("rows", [])):
        actions.append("探索性歷史分布與前次不同，僅記錄狀態變化，不據此調整正式模型係數。")
    if matured.get("available") and matured.get("hit_rate", 1) < 0.5:
        actions.append("近期探索性核對率低於50%；多日正式訊號維持停用，不以短期結果重新調參。")
    if not actions:
        actions.append("目前沒有需要立即修正的重大偏差，持續累積紀錄。")
    return {"actions": actions, "mode": "只記錄探索性偏差；正式係數由鎖定驗證政策控制"}


def self_review_summary(drift: dict, matured: dict, correction: dict, tracking: dict | None = None) -> str:
    parts = [drift.get("summary", "")]
    if tracking:
        parts.append(tracking.get("summary", ""))
    parts.append(matured.get("summary", ""))
    first_action = correction.get("actions", [""])[0]
    if first_action:
        parts.append(first_action)
    return " ".join(part for part in parts if part)


def direction_from_return(value: float, neutral_threshold: float = 0.005) -> str:
    if value > neutral_threshold:
        return "up"
    if value < -neutral_threshold:
        return "down"
    return "sideways"


def analyze_premarket(
    forecast_date: str,
    scored: pd.DataFrame,
    external_path: Path,
    night_futures_path: Path,
) -> dict:
    now = datetime.now(TAIPEI)
    forecast_day = pd.to_datetime(forecast_date).date()
    night_session_complete = (
        forecast_day < now.date()
        or (forecast_day == now.date() and now.time() >= time(5, 0))
    )
    spot_latest_row = scored.sort_values("date").iloc[-1]
    spot_latest = str(spot_latest_row["date"].date())
    is_non_trading_day = forecast_day.weekday() >= 5
    is_premarket = pd.to_datetime(forecast_date) > pd.to_datetime(spot_latest) and not is_non_trading_day

    night_lookup_date = next_weekday(forecast_day).isoformat() if is_non_trading_day else forecast_date
    external_row = latest_external_row(external_path, forecast_date)
    night_row = latest_night_row(night_futures_path, night_lookup_date)
    external_score = score_external_row(external_row)
    external_context = analyze_external_conflict(external_row)
    night_score = score_night_row(night_row)
    total_score = external_score + night_score

    last_close = safe_float(spot_latest_row.get("close"))
    night_low = safe_float(night_row.get("tx_night_low")) if night_row is not None else None
    night_high = safe_float(night_row.get("tx_night_high")) if night_row is not None else None
    night_close = safe_float(night_row.get("tx_night_close")) if night_row is not None else None
    night_open = safe_float(night_row.get("tx_night_open")) if night_row is not None else None
    previous_night_close = previous_night_session_close(night_futures_path, night_row)
    night_vs_previous_close = (
        night_close / previous_night_close - 1
        if night_close is not None and previous_night_close not in (None, 0)
        else None
    )
    night_path = analyze_night_path(night_row, last_close)
    night_basis_audit = build_night_basis_audit(night_row, previous_night_close, night_vs_previous_close)
    night_psychology = analyze_night_close_psychology(
        night_path,
        night_basis_audit,
        external_context,
        external_score,
    )

    if total_score <= -5:
        pressure = "強偏空"
        scenario_name = "盤前外部與台指期夜盤同步偏空"
        summary = "美股科技與半導體偏弱，台指期夜盤也明顯轉弱，今天台股開盤壓力偏大。"
    elif total_score <= -2:
        pressure = "偏空"
        scenario_name = "盤前壓力偏空，等待現貨承接確認"
        summary = "外部市場或台指期夜盤偏弱，今天台股容易先承壓，重點看開盤後是否有低接。"
    elif total_score >= 4:
        pressure = "偏多"
        scenario_name = "盤前外部與台指期夜盤同步偏多"
        summary = "美股與台指期夜盤偏強，今天台股開盤氣氛較正面。"
    elif total_score > 0:
        pressure = "小偏多"
        scenario_name = "盤前略偏多，但仍需現貨確認"
        summary = "盤前訊號略偏正面，但仍要看開盤量價是否延續。"
    else:
        pressure = "中性"
        if external_context["is_conflict"]:
            scenario_name = "外部訊號互相衝突，防範盤中急殺急拉"
            summary = external_context["summary"] + " 方向總分雖接近零，但不能解讀成市場平靜。"
        else:
            scenario_name = "盤前訊號中性，等待開盤確認"
            summary = "盤前沒有明顯單邊訊號，今天要看開盤後高低點是否被突破。"
    if is_non_trading_day:
        scenario_name = "非交易日觀察，等待下個交易日確認"
        summary = "今天不是台股現貨交易日；外部市場與夜盤只作下個交易日情境觀察，不能稱為今日開盤預測。"

    defense_levels = []
    if night_low is not None:
        defense_levels.append(round_to_base(night_low, 50))
    if last_close is not None:
        defense_levels.append(round_to_base(last_close * 0.985, 100))
    defense_levels = unique_numbers(defense_levels)

    resistance_levels = []
    if night_close is not None:
        resistance_levels.append(round_to_base(night_close, 50))
    if night_open is not None:
        resistance_levels.append(round_to_base(night_open, 50))
    resistance_levels = unique_numbers(resistance_levels)

    if defense_levels:
        watch = f"先看 {format_levels(defense_levels)} 是否守住；跌破代表賣壓擴大。"
    else:
        watch = "先看開盤低點是否被跌破；跌破代表賣壓擴大。"
    if resistance_levels:
        watch += f" 若站回 {format_levels(resistance_levels)}，代表承接力轉強。"
    if is_non_trading_day:
        watch = f"下個交易日再看 {format_levels(defense_levels) or '前一交易日低點'} 是否守住"
        if resistance_levels:
            watch += f"；若站回 {format_levels(resistance_levels)}，代表承接力轉強。"
        else:
            watch += "。"

    locked_gap = locked_night_gap_forecast(forecast_date)
    locked_amplitude = locked_gap_amplitude_forecast(forecast_date)
    return {
        "is_premarket": bool(is_premarket),
        "is_non_trading_day": bool(is_non_trading_day),
        "session_mode": "non_trading_day" if is_non_trading_day else ("premarket" if is_premarket else "official_daily"),
        "night_session_complete": bool(night_session_complete),
        "forecast_date": forecast_date,
        "spot_latest_date": spot_latest,
        "spot_latest_close": last_close,
        "external_date": none_or_str(external_row.get("date")) if external_row is not None else None,
        "night_date": none_or_str(night_row.get("night_date")) if night_row is not None else None,
        "night_signal_date": none_or_str(night_row.get("signal_date")) if night_row is not None else None,
        "night_lookup_date": night_lookup_date,
        "external_score": int(external_score),
        "external_context": external_context,
        "night_futures_score": int(night_score),
        "total_score": int(total_score),
        "pressure": pressure,
        "scenario": scenario_name,
        "summary": summary,
        "watch": watch,
        "defense_levels": defense_levels,
        "resistance_levels": resistance_levels,
        "nasdaq_return_1d": safe_float(external_row.get("nasdaq_return_1d")) if external_row is not None else None,
        "sox_return_1d": safe_float(external_row.get("sox_return_1d")) if external_row is not None else None,
        "sp500_return_1d": safe_float(external_row.get("sp500_return_1d")) if external_row is not None else None,
        "tsm_adr_return_1d": safe_float(external_row.get("tsm_adr_return_1d")) if external_row is not None else None,
        "micron_return_1d": safe_float(external_row.get("micron_return_1d")) if external_row is not None else None,
        "ewt_return_1d": safe_float(external_row.get("ewt_return_1d")) if external_row is not None else None,
        "samsung_return_1d": safe_float(external_row.get("samsung_return_1d")) if external_row is not None else None,
        "sk_hynix_return_1d": safe_float(external_row.get("sk_hynix_return_1d")) if external_row is not None else None,
        "usd_twd_return_1d": safe_float(external_row.get("usd_twd_return_1d")) if external_row is not None else None,
        "vix_return_1d": safe_float(external_row.get("vix_return_1d")) if external_row is not None else None,
        "treasury_5y_close": safe_float(external_row.get("treasury_5y_close")) if external_row is not None else None,
        "treasury_10y_close": safe_float(external_row.get("treasury_10y_close")) if external_row is not None else None,
        "treasury_30y_close": safe_float(external_row.get("treasury_30y_close")) if external_row is not None else None,
        "treasury_5y_return_1d": safe_float(external_row.get("treasury_5y_return_1d")) if external_row is not None else None,
        "treasury_10y_return_1d": safe_float(external_row.get("treasury_10y_return_1d")) if external_row is not None else None,
        "treasury_30y_return_1d": safe_float(external_row.get("treasury_30y_return_1d")) if external_row is not None else None,
        "treasury_context_only": True,
        "tx_night_return": safe_float(night_row.get("tx_night_return")) if night_row is not None else None,
        "tx_night_spread_per": safe_float(night_row.get("tx_night_spread_per")) if night_row is not None else None,
        "tx_night_open": night_open,
        "tx_night_high": night_high,
        "tx_night_low": night_low,
        "tx_night_close": night_close,
        "previous_tx_night_close": previous_night_close,
        "tx_night_vs_previous_close": night_vs_previous_close,
        "night_basis_audit": night_basis_audit,
        "night_path": night_path,
        "night_psychology": night_psychology,
        "validated_night_gap": locked_gap,
        "validated_gap_amplitude": locked_amplitude,
    }


def locked_night_gap_forecast(forecast_date: str) -> dict:
    path = ROOT / "reports" / "locked_night_spread_gap_forecast_log.json"
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        target = str(pd.to_datetime(forecast_date).date())
        return next((row for row in reversed(rows) if row.get("signal_date") == target), {})
    except (OSError, ValueError, TypeError):
        return {}


def locked_gap_amplitude_forecast(forecast_date: str) -> dict:
    path = ROOT / "reports" / "locked_gap_amplitude_forecast_log.json"
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        target = str(pd.to_datetime(forecast_date).date())
        return next((row for row in reversed(rows) if row.get("signal_date") == target), {})
    except (OSError, ValueError, TypeError):
        return {}


def latest_external_row(path: Path, forecast_date: str):
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or "date" not in frame:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    eligible = frame[frame["date"] <= pd.to_datetime(forecast_date)]
    if eligible.empty:
        return None
    return eligible.sort_values("date").iloc[-1]


def latest_night_row(path: Path, forecast_date: str):
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    for col in ["signal_date", "night_date"]:
        if col in frame:
            frame[col] = pd.to_datetime(frame[col])
    target = pd.to_datetime(forecast_date)
    if "signal_date" in frame:
        exact = frame[frame["signal_date"] == target]
        if not exact.empty:
            return exact.sort_values("night_date").iloc[-1]
        eligible = frame[frame["signal_date"] <= target]
        if not eligible.empty:
            return eligible.sort_values("signal_date").iloc[-1]
    if "night_date" in frame:
        eligible = frame[frame["night_date"] <= target]
        if not eligible.empty:
            return eligible.sort_values("night_date").iloc[-1]
    return None


def previous_night_session_close(path: Path, current_row) -> float | None:
    if current_row is None or not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    if frame.empty or "tx_night_close" not in frame:
        return None
    for col in ["signal_date", "night_date"]:
        if col in frame:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    current_signal = pd.to_datetime(current_row.get("signal_date"), errors="coerce")
    current_night = pd.to_datetime(current_row.get("night_date"), errors="coerce")
    if "signal_date" in frame and pd.notna(current_signal):
        prior = frame[frame["signal_date"] < current_signal]
        sort_col = "signal_date"
    elif "night_date" in frame and pd.notna(current_night):
        prior = frame[frame["night_date"] < current_night]
        sort_col = "night_date"
    else:
        return None
    if prior.empty:
        return None
    value = safe_float(prior.sort_values(sort_col).iloc[-1].get("tx_night_close"))
    return value


def build_night_basis_audit(row, previous_close: float | None, vs_previous_close: float | None) -> dict:
    if row is None:
        return {
            "available": False,
            "summary": "夜盤資料不足，無法核對漲跌基準。",
            "guardrail": "不可用單一夜盤百分比推論日盤收盤。",
        }
    official_spread = safe_float(row.get("tx_night_spread_per"))
    open_close_return = safe_float(row.get("tx_night_return"))
    open_ = safe_float(row.get("tx_night_open"))
    close = safe_float(row.get("tx_night_close"))
    basis_conflict = (
        official_spread is not None
        and open_close_return is not None
        and abs(official_spread - open_close_return) >= 0.003
    )
    if basis_conflict:
        summary = "夜盤官方漲跌與開收報酬差異明顯；方向只能先看路徑，不能混用基準。"
    elif official_spread is None and open_close_return is None:
        summary = "夜盤缺少官方漲跌與開收報酬，僅能保留價格觀察。"
    else:
        summary = "夜盤基準差異未達重大衝突；仍須分開標示官方漲跌、開收與前夜收盤比較。"
    return {
        "available": True,
        "night_date": none_or_str(row.get("night_date")),
        "signal_date": none_or_str(row.get("signal_date")),
        "official_spread_per": official_spread,
        "official_basis": "交易所/資料源參考價或前結算基準；不是固定等於夜盤開盤價。",
        "open_close_return": open_close_return,
        "open_close_basis": "夜盤開盤到夜盤收盤的實際路徑報酬。",
        "previous_night_close": previous_close,
        "vs_previous_night_close_return": vs_previous_close,
        "previous_close_basis": "本夜盤收盤相對前一有效夜盤收盤；用來看夜盤自身連續性。",
        "open": open_,
        "close": close,
        "basis_conflict": bool(basis_conflict),
        "summary": summary,
        "guardrail": "三基準不得混為同一個漲跌；官方漲跌偏開盤缺口，開收報酬偏夜盤路徑，前夜比較偏連續趨勢。",
    }


def analyze_night_path(row, prior_cash_close: float | None = None) -> dict:
    """Describe a completed night session without promoting it to a cash-close signal."""
    if row is None:
        return {"available": False, "status": "missing", "formal_cash_close_signal": None}
    open_ = safe_float(row.get("tx_night_open"))
    high = safe_float(row.get("tx_night_high"))
    low = safe_float(row.get("tx_night_low"))
    close = safe_float(row.get("tx_night_close"))
    volume = safe_float(row.get("tx_night_volume"))
    night_return = safe_float(row.get("tx_night_return"))
    spread = safe_float(row.get("tx_night_spread_per"))
    session_range = safe_float(row.get("tx_night_range"))
    close_position = None
    if close is not None and high is not None and low is not None and high > low:
        close_position = max(0.0, min(1.0, (close - low) / (high - low)))
    recovery_from_low = None
    if close is not None and low not in (None, 0) and open_ is not None and open_ > low:
        recovery_from_low = max(0.0, min(1.0, (close - low) / (open_ - low)))
    vs_cash_close = (close / prior_cash_close - 1) if close is not None and prior_cash_close else None

    bearish_extension = bool(
        night_return is not None and night_return <= -0.008
        and close_position is not None and close_position <= 0.25
    )
    bullish_extension = bool(
        night_return is not None and night_return >= 0.008
        and close_position is not None and close_position >= 0.75
    )
    material_pressure = bool(spread is not None and abs(spread) >= 0.008)
    if bearish_extension:
        status, label = "bearish_extension_close_near_low", "空方擴張、收近低點"
        summary = "夜盤跌幅明顯且收在區間低檔，反彈不足；提高次日開低與波動風險。"
    elif bullish_extension:
        status, label = "bullish_extension_close_near_high", "多方擴張、收近高點"
        summary = "夜盤漲幅明顯且收在區間高檔；提高次日開高與波動風險。"
    elif night_return is not None and night_return < 0:
        status, label = "bearish_but_recovered", "夜盤偏空但有收回"
        summary = "夜盤收跌但不是極弱收盤，需由次日現貨確認是否延續。"
    elif night_return is not None and night_return > 0:
        status, label = "bullish_but_unconfirmed", "夜盤偏多但待確認"
        summary = "夜盤收漲但尚不能延伸為次日日盤收盤方向。"
    else:
        status, label = "neutral", "夜盤中性"
        summary = "夜盤缺乏明顯單邊路徑。"
    return {
        "available": True,
        "status": status,
        "label": label,
        "summary": summary,
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
        "return": night_return, "spread_per": spread, "range": session_range,
        "close_position": close_position,
        "recovery_from_low": recovery_from_low,
        "close_vs_prior_cash": vs_cash_close,
        "material_pressure": material_pressure,
        "formal_cash_close_signal": None,
        "guardrail": "夜盤路徑只提升開盤與波動風險，不得直接宣稱次日日盤收盤方向。",
    }


def analyze_night_close_psychology(
    night_path: dict,
    night_basis_audit: dict | None = None,
    external_context: dict | None = None,
    external_score: int | None = None,
) -> dict:
    """Explain the completed night session with psychology and tactics, not only technical levels."""
    if not night_path or not night_path.get("available"):
        return {
            "available": False,
            "label": "夜盤心理資料不足",
            "summary": "缺少有效夜盤路徑，無法判斷夜盤收盤心理。",
            "guardrail": "夜盤心理層只作隔日日盤驗證假設，不產生買賣命令。",
        }
    status_code = night_path.get("status")
    close_position = safe_float(night_path.get("close_position"))
    recovery = safe_float(night_path.get("recovery_from_low"))
    night_return = safe_float(night_path.get("return"))
    spread = safe_float(night_path.get("spread_per"))
    session_range = safe_float(night_path.get("range"))
    external_score = safe_int(external_score, 0)
    basis_conflict = bool((night_basis_audit or {}).get("basis_conflict"))
    external_conflict = bool((external_context or {}).get("is_conflict"))

    score = 0
    evidence: list[str] = []
    validation: list[str] = []

    def add(points: int, text: str) -> None:
        nonlocal score
        score += points
        evidence.append(text)

    if status_code == "bearish_extension_close_near_low":
        add(-3, "夜盤跌幅擴大且收近低點，追跌與停損心理佔上風。")
        label = "空方心理延伸"
        tactic = "追跌壓測"
        dominant_emotion = "恐慌/避險"
        validation.extend([
            "日盤是否跌破夜盤低點且30～60分鐘收不回。",
            "權值股是否同步轉弱，族群弱勢是否擴散。",
        ])
    elif status_code == "bullish_extension_close_near_high":
        add(3, "夜盤漲幅擴大且收近高點，追價與空單回補心理升高。")
        label = "多方心理延伸"
        tactic = "追價/軋空"
        dominant_emotion = "貪婪/回補急迫"
        validation.extend([
            "日盤是否站穩夜盤高位且量能跟上。",
            "開高後是否跌回夜盤收盤下方，避免誘多。",
        ])
    elif status_code == "bearish_but_recovered":
        add(-1, "夜盤收跌但低點後有收回，空方有壓力但尚未取得完整心理控制。")
        label = "誘空回補待驗"
        tactic = "壓低測承接"
        dominant_emotion = "懷疑/防守"
        validation.extend([
            "日盤若守住夜盤低點並站回夜盤收盤，偏向誘空或空單回補。",
            "日盤若跌破夜盤低點且收不回，壓力測試轉為空方延伸。",
        ])
    elif status_code == "bullish_but_unconfirmed":
        add(1, "夜盤收漲但未達強延伸，追價心理存在但仍需現貨確認。")
        label = "追價待驗"
        tactic = "開高測追價"
        dominant_emotion = "樂觀/觀望"
        validation.extend([
            "日盤是否開高續強，或開高後被獲利了結壓回。",
            "量能與族群廣度是否支撐夜盤樂觀。",
        ])
    else:
        label = "夜盤心理中性"
        tactic = "等待日盤定價"
        dominant_emotion = "觀望"
        evidence.append("夜盤沒有明顯單邊心理，日盤現貨才是主判決。")
        validation.append("等待開盤後高低點突破與收盤位置確認。")

    if recovery is not None and recovery >= 0.5 and night_return is not None and night_return < 0:
        add(1, "低點後收回超過半段，代表低檔有承接或空單先行回補。")
    elif recovery is not None and recovery < 0.25 and night_return is not None and night_return < 0:
        add(-1, "低點後收回不足，代表賣壓尚未宣洩完成。")

    if close_position is not None and close_position <= 0.2:
        add(-1, "收盤位置落在夜盤區間低檔，隔日日盤容易先承壓。")
    elif close_position is not None and close_position >= 0.8:
        add(1, "收盤位置落在夜盤區間高檔，隔日日盤開局心理較穩。")

    if session_range is not None and session_range >= 0.012:
        evidence.append("夜盤振幅偏大，代表不是平靜定價，而是主動測壓或洗籌。")
    if basis_conflict:
        evidence.append("夜盤漲跌基準有衝突，心理判讀以高低收路徑為主，百分比降權。")
    if external_conflict:
        evidence.append("外部市場訊號分歧，夜盤容易變成預期差測試，不宜單線外推。")
    if external_score <= -2:
        add(-1, "外部市場偏空，夜盤壓力較可能被日盤開局放大。")
    elif external_score >= 2:
        add(1, "外部市場偏多，夜盤壓力較容易被日盤承接吸收。")

    if score >= 3:
        bias = "bullish_psychology"
        action = "以開高續強與廣度擴散驗證，防範開高誘多。"
    elif score <= -3:
        bias = "bearish_psychology"
        action = "以夜盤低點、前一日低點與開盤後30～60分鐘是否收回驗證。"
    else:
        bias = "mixed_psychology"
        action = "維持待驗，讓日盤現貨承接、量能與族群廣度決定升降級。"

    return {
        "available": True,
        "framework": "night_close_psychology_v1",
        "label": label,
        "tactic": tactic,
        "dominant_emotion": dominant_emotion,
        "score": int(max(-5, min(5, score))),
        "bias": bias,
        "summary": f"夜盤心理為{label}，戰術候選是{tactic}；{action}",
        "evidence": top_items(evidence, 7),
        "next_validation": validation,
        "guardrail": "夜盤心理層只作隔日日盤驗證假設；必須由現貨開盤、量能、權值與收盤確認，不產生買賣命令。",
    }


def analyze_intraday_tactical_monitor(
    forecast_date: str,
    scored: pd.DataFrame,
    signal_date: str,
    premarket: dict,
    live_monitor: dict | None,
) -> dict:
    live_monitor = live_monitor or {}
    forecast_day = pd.to_datetime(forecast_date).date()
    signal_day = pd.to_datetime(signal_date).date()
    live_day = None
    if live_monitor.get("date"):
        live_day = pd.to_datetime(live_monitor.get("date")).date()
    live_usable = bool(live_monitor.get("enabled")) and live_day == forecast_day

    eligible = scored[scored["date"] <= pd.to_datetime(signal_date)].sort_values("date")
    # When an official same-day candle and a same-day live snapshot coexist,
    # the tactical comparison must still use the strictly prior cash session.
    # Otherwise today's low/close are incorrectly labelled as yesterday's.
    reference = eligible
    if live_usable:
        strictly_prior = eligible[eligible["date"].dt.date < live_day]
        if not strictly_prior.empty:
            reference = strictly_prior
    row = reference.iloc[-1] if not reference.empty else pd.Series(dtype="float64")
    prior = reference.iloc[-2] if len(reference) >= 2 else pd.Series(dtype="float64")

    last_close = safe_float(row.get("close"))
    last_low = safe_float(row.get("low"))
    last_high = safe_float(row.get("high"))
    prior_close = safe_float(prior.get("close")) if not prior.empty else None
    ma5 = safe_float(row.get("ma_5"))
    ma20 = safe_float(row.get("ma_20"))
    night_spread = safe_float(premarket.get("tx_night_spread_per"))
    night_close = safe_float(premarket.get("tx_night_close"))
    night_low = safe_float(premarket.get("tx_night_low"))

    reference_close = last_close
    defense_candidates = [
        round_to_base(value, 50)
        for value in [last_low, night_low, ma20]
        if value is not None and pd.notna(value)
    ]
    reclaim_candidates = [
        round_to_base(value, 50)
        for value in [last_close, ma5, night_close, last_high]
        if value is not None and pd.notna(value)
    ]
    if reference_close is not None and pd.notna(reference_close):
        defense_candidates = [level for level in defense_candidates if level <= reference_close]
        reclaim_candidates = [level for level in reclaim_candidates if level >= reference_close]
    defense_levels = sorted(unique_numbers(defense_candidates), reverse=True)
    reclaim_levels = sorted(unique_numbers(reclaim_candidates))

    base = {
        "enabled": True,
        "forecast_date": forecast_day.isoformat(),
        "official_signal_date": signal_day.isoformat(),
        "live_usable": live_usable,
        "live_snapshot_date": live_day.isoformat() if live_day else None,
        "source": live_monitor.get("source", "none"),
        "source_message": live_monitor.get("message", ""),
        "night_signal_date": premarket.get("night_signal_date") or premarket.get("night_date"),
        "night_spread_per": night_spread,
        "night_close": night_close,
        "night_low": night_low,
        "night_path": premarket.get("night_path", {}),
        "last_close": last_close,
        "last_low": last_low,
        "last_high": last_high,
        "prior_close": prior_close,
        "ma5": ma5,
        "ma20": ma20,
        "defense_levels": defense_levels,
        "reclaim_levels": reclaim_levels,
        "checks": [],
    }

    def finalize(code: str, label: str, summary: str, action: str, status: str = "watch") -> dict:
        base.update(
            {
                "code": code,
                "label": label,
                "summary": summary,
                "action": action,
                "status": status,
            }
        )
        return base

    if premarket.get("is_non_trading_day"):
        return finalize(
            "non_trading_observation",
            "非交易日觀察",
            "今天沒有台股現貨日盤，不能做即時盤中判斷；只能把夜盤與外部市場列為下個交易日劇本。",
            "保留防守線與轉強線，等下個交易日日盤是否開低收回或破低不回。",
            "standby",
        )

    if not live_usable:
        code = "premarket_only" if premarket.get("is_premarket") else "no_live_snapshot"
        label = "盤前劇本待驗證" if premarket.get("is_premarket") else "無即時盤中資料"
        summary = (
            "目前只有盤前/外部資料，尚未取得今日現貨開高低與即時價，不能判定日盤是否收回。"
            if premarket.get("is_premarket")
            else "目前沒有今日盤中快照，日盤戰術雷達只能停在最後正式日線。"
        )
        return finalize(
            code,
            label,
            summary,
            "等現貨日盤開盤後，依序核對開盤、盤中低點、是否站回昨收與收盤位置。",
            "standby",
        )

    live_price = safe_float(live_monitor.get("price"))
    live_open = safe_float(live_monitor.get("open"))
    live_high = safe_float(live_monitor.get("high"))
    live_low = safe_float(live_monitor.get("low"))
    close_position = None
    if live_price is not None and live_high is not None and live_low is not None and live_high > live_low:
        close_position = max(0.0, min(1.0, (live_price - live_low) / (live_high - live_low)))
    vs_prev_close = (live_price / last_close - 1) if live_price is not None and last_close else None
    vs_open = (live_price / live_open - 1) if live_price is not None and live_open else None
    broke_last_low = bool(live_low is not None and last_low is not None and live_low < last_low)
    broke_night_low = bool(live_low is not None and night_low is not None and live_low < night_low)
    reclaimed_last_close = bool(live_price is not None and last_close is not None and live_price >= last_close)
    reclaimed_night_close = bool(live_price is not None and night_close is not None and live_price >= night_close)
    reclaimed_open = bool(live_price is not None and live_open is not None and live_price >= live_open)
    reclaimed_psych_45000 = bool(live_price is not None and live_price >= 45000)

    base.update(
        {
            "live_price": live_price,
            "live_open": live_open,
            "live_high": live_high,
            "live_low": live_low,
            "close_position": close_position,
            "vs_prev_close": vs_prev_close,
            "vs_open": vs_open,
            "broke_last_low": broke_last_low,
            "broke_night_low": broke_night_low,
            "reclaimed_last_close": reclaimed_last_close,
            "reclaimed_night_close": reclaimed_night_close,
            "reclaimed_open": reclaimed_open,
            "reclaimed_psych_45000": reclaimed_psych_45000,
            "checks": [
                {
                    "name": "夜盤方向",
                    "passed": night_spread is not None,
                    "value": pct(night_spread),
                    "meaning": "夜盤只代表盤前預期差，必須由日盤現貨驗證。",
                },
                {
                    "name": "是否破昨低",
                    "answer": "是" if broke_last_low else "否",
                    "passed": not broke_last_low,
                    "value": f"{num(live_low)} / 昨低 {num(last_low)}",
                    "meaning": "未破昨低偏承接；破昨低後要看能否快速收回。",
                },
                {
                    "name": "是否破夜盤低",
                    "answer": "是" if broke_night_low else "否",
                    "passed": not broke_night_low,
                    "value": f"{num(live_low)} / 夜低 {num(night_low)}",
                    "meaning": "破夜盤低點後若快速收回，常是誘空/洗籌；破後收不回才是續跌確認。",
                },
                {
                    "name": "是否站回昨收",
                    "answer": "是" if reclaimed_last_close else "否",
                    "passed": reclaimed_last_close,
                    "value": f"{num(live_price)} / 昨收 {num(last_close)}",
                    "meaning": "站回昨收才算現貨有明確反證力。",
                },
                {
                    "name": "盤中收回位置",
                    "answer": "是" if close_position is not None and close_position >= 0.60 else "否",
                    "passed": close_position is not None and close_position >= 0.60,
                    "value": pct(close_position),
                    "meaning": "越接近日內高位，越像下殺後有承接。",
                },
            ],
        }
    )

    if night_spread is not None and night_spread < 0 and broke_night_low and (
        reclaimed_night_close or reclaimed_psych_45000 or reclaimed_last_close
    ) and close_position is not None and close_position >= 0.70:
        return finalize(
            "night_down_break_night_low_reclaim_washout",
            "夜跌破夜低後日收回洗籌",
            "夜盤偏空後日盤再刺破夜盤低點，但未形成續殺，反而收回夜盤收盤/心理關卡並收在日內高位；屬較強誘空/洗籌候選。",
            "隔日優先驗證是否守住今日低點與45,000；若再破低收不回，洗籌候選失效。",
            "constructive",
        )
    if night_spread is not None and night_spread < 0 and not broke_last_low and (
        reclaimed_last_close or (close_position is not None and close_position >= 0.60)
    ):
        return finalize(
            "night_down_cash_reclaim_candidate",
            "夜跌日收回洗籌候選",
            "夜盤偏弱但現貨日盤未破昨低且有收回，符合洗籌/誘空候選；仍需收盤與隔日不破低確認。",
            "觀察是否守住防守線並站穩轉強線；沒有收盤確認前，不升級為完成打底。",
            "constructive",
        )
    if night_spread is not None and night_spread < 0 and broke_last_low and not reclaimed_last_close:
        return finalize(
            "night_down_cash_break_confirming",
            "夜跌日盤破位確認",
            "夜盤跌後日盤也破昨低且未站回昨收，代表夜盤壓力被現貨確認，不能視為單純洗盤。",
            "提高風控警戒，等收盤是否收回；若連續收不回再升級風險。",
            "warning",
        )
    if night_spread is not None and night_spread > 0 and live_open is not None and not reclaimed_open and (
        close_position is not None and close_position <= 0.25
    ):
        return finalize(
            "night_up_cash_failure",
            "夜漲日弱現貨反證",
            "夜盤偏強但日盤開後守不住，且即時價靠近日內低位，偏高檔換手或追價退潮。",
            "盯日內低點與昨收，若收盤仍弱，隔日列為賣壓延續驗證。",
            "warning",
        )
    if broke_last_low:
        return finalize(
            "intraday_key_low_break",
            "盤中破昨低待收回",
            "日盤已刺破昨低，但是否是假跌破要看能否站回昨收與收盤位置。",
            "先列防守警戒；只要收不回昨收，就不得判定洗盤完成。",
            "warning",
        )
    if reclaimed_open and close_position is not None and close_position >= 0.60:
        return finalize(
            "intraday_reclaim_strengthening",
            "日盤收回轉強",
            "日盤站回開盤且位置偏高，短線承接改善；仍以正式收盤核對為準。",
            "觀察轉強線是否站穩，隔日不破低才算延續。",
            "constructive",
        )
    return finalize(
        "intraday_neutral_watch",
        "盤中中性觀察",
        "盤中尚未同時出現破位或收回的強訊號，維持中性監控。",
        "等收盤K線確認，不把盤中噪音當成結論。",
        "neutral",
    )


def score_external_row(row) -> int:
    if row is None:
        return 0
    score = 0
    weights = {
        "nasdaq_return_1d": (-0.018, 0.012, 1),
        "sox_return_1d": (-0.025, 0.018, 2),
        "sp500_return_1d": (-0.014, 0.010, 1),
        "tsm_adr_return_1d": (-0.025, 0.018, 2),
        "vix_return_1d": (0.10, -0.06, 1),
    }
    for col, (bearish, bullish, weight) in weights.items():
        value = row.get(col)
        if pd.isna(value):
            continue
        if col == "vix_return_1d":
            if value >= bearish:
                score -= weight
            elif value <= bullish:
                score += weight
        else:
            if value <= bearish:
                score -= weight
            elif value >= bullish:
                score += weight
    return int(max(-4, min(4, score)))


def analyze_external_conflict(row) -> dict:
    """Describe opposing external forces without changing the validated score.

    A zero net score can mean either a genuinely quiet session or large signals
    cancelling each other.  The latter carries path/volatility risk and must not
    be shown as ordinary neutrality.
    """
    if row is None:
        return {
            "regime": "missing",
            "label": "外部資料不足",
            "summary": "外部市場資料不足，不能判定平靜或分歧。",
            "dispersion": 0.0,
            "is_conflict": False,
            "active_signals": 0,
        }

    values = {
        "Nasdaq": safe_float(row.get("nasdaq_return_1d")),
        "費半": safe_float(row.get("sox_return_1d")),
        "S&P500": safe_float(row.get("sp500_return_1d")),
        "台積電ADR": safe_float(row.get("tsm_adr_return_1d")),
        "VIX反向": None,
    }
    vix = safe_float(row.get("vix_return_1d"))
    if vix is not None:
        values["VIX反向"] = -vix
    valid = [value for value in values.values() if value is not None]
    if not valid:
        return {
            "regime": "missing",
            "label": "外部資料不足",
            "summary": "外部市場資料不足，不能判定平靜或分歧。",
            "dispersion": 0.0,
            "is_conflict": False,
            "active_signals": 0,
        }

    dispersion = float(max(valid) - min(valid))
    positive = sum(value >= 0.005 for value in valid)
    negative = sum(value <= -0.005 for value in valid)
    active = positive + negative
    is_conflict = positive > 0 and negative > 0 and dispersion >= 0.015
    broad_risk = sum(
        value is not None and value <= -0.005
        for value in (values["Nasdaq"], values["S&P500"], values["VIX反向"])
    )
    taiwan_tech = sum(
        value is not None and value >= 0.005
        for value in (values["費半"], values["台積電ADR"])
    )

    if is_conflict and broad_risk >= 2 and taiwan_tech >= 1:
        regime = "taiwan_tech_offset"
        label = "全球風險偏弱、台灣科技相對抗跌"
        summary = "美國大盤與風險情緒偏弱，但半導體或台積電相對強，容易形成權值股撐盤、個股分化與大幅震盪。"
    elif is_conflict:
        regime = "mixed_conflict"
        label = "外部多空高度分歧"
        summary = "外部多空訊號同時活躍，淨分數接近零仍有顯著路徑風險。"
    elif active <= 1 and max(abs(value) for value in valid) < 0.01:
        regime = "calm_neutral"
        label = "外部市場相對平靜"
        summary = "主要外部市場變動有限，接近真正的中性環境。"
    elif positive > negative:
        regime = "broad_tailwind"
        label = "外部環境偏正面"
        summary = "外部正向訊號較多，但仍須由台股現貨量價確認。"
    elif negative > positive:
        regime = "broad_risk_pressure"
        label = "外部風險壓力偏高"
        summary = "外部負向訊號較多，須提高開盤與盤中風險警戒。"
    else:
        regime = "neutral"
        label = "外部方向不明"
        summary = "外部訊號不足以形成一致方向。"
    return {
        "regime": regime,
        "label": label,
        "summary": summary,
        "dispersion": round(dispersion, 6),
        "is_conflict": bool(is_conflict),
        "active_signals": int(active),
    }


def analyze_external_event_reset(premarket: dict, global_news_risk: dict | None = None) -> dict:
    """Detect when overseas markets may reset Taiwan's internal cause chain.

    This monitor is intentionally a risk router, not a trading signal.  Taiwan's
    own day/night cause chain remains primary unless global or regional pressure
    becomes large enough to invalidate the local script.
    """
    premarket = premarket or {}
    global_news_risk = global_news_risk or {}
    signals = {
        "Nasdaq": safe_float(premarket.get("nasdaq_return_1d")),
        "S&P500": safe_float(premarket.get("sp500_return_1d")),
        "SOX": safe_float(premarket.get("sox_return_1d")),
        "TSM ADR": safe_float(premarket.get("tsm_adr_return_1d")),
        "Micron": safe_float(premarket.get("micron_return_1d")),
        "EWT": safe_float(premarket.get("ewt_return_1d")),
        "Samsung": safe_float(premarket.get("samsung_return_1d")),
        "SK Hynix": safe_float(premarket.get("sk_hynix_return_1d")),
        "USD/TWD": safe_float(premarket.get("usd_twd_return_1d")),
        "VIX": safe_float(premarket.get("vix_return_1d")),
        "TX night": safe_float(premarket.get("tx_night_spread_per")),
    }
    valid = {name: value for name, value in signals.items() if value is not None}
    if not valid:
        return {
            "enabled": True,
            "available": False,
            "code": "missing",
            "label": "外部重置資料不足",
            "summary": "外部市場資料不足，台股內部因果鏈維持主判斷。",
            "reset_active": False,
            "direction": "none",
            "reset_score": 0,
            "risk_adjustment": 0,
            "causal_priority": "internal_primary",
            "reasons": [],
            "watch": "補齊美股、日韓/半導體、匯率與台指期夜盤資料後再判斷。",
            "news_feed_status": global_news_risk.get("status", "missing"),
            "news_rule": global_news_risk.get("summary", "每日國際重大財經政治消息尚未產出。"),
            "news_risk": global_news_risk,
            "guardrail": "外部重置只作風控與劇本降權，不直接產生買賣命令。",
        }

    reasons: list[str] = []
    bullish_reasons: list[str] = []
    bearish_score = 0
    bullish_score = 0

    def add_bear(condition: bool, points: int, text: str) -> None:
        nonlocal bearish_score
        if condition:
            bearish_score += points
            reasons.append(text)

    def add_bull(condition: bool, points: int, text: str) -> None:
        nonlocal bullish_score
        if condition:
            bullish_score += points
            bullish_reasons.append(text)

    add_bear((signals["Nasdaq"] or 0) <= -0.018, 2, "Nasdaq急跌")
    add_bear((signals["S&P500"] or 0) <= -0.014, 1, "S&P500同步轉弱")
    add_bear((signals["SOX"] or 0) <= -0.025, 2, "費半急跌")
    add_bear((signals["TSM ADR"] or 0) <= -0.025, 2, "TSM ADR急跌")
    add_bear((signals["Micron"] or 0) <= -0.035, 1, "美光急跌")
    add_bear((signals["EWT"] or 0) <= -0.020, 1, "EWT台灣ETF明顯轉弱")
    add_bear((signals["Samsung"] or 0) <= -0.025, 1, "三星轉弱")
    add_bear((signals["SK Hynix"] or 0) <= -0.035, 1, "SK海力士急跌")
    add_bear((signals["VIX"] or 0) >= 0.08, 2, "VIX急升")
    add_bear((signals["USD/TWD"] or 0) >= 0.004, 1, "美元台幣急升，外資匯率壓力增加")
    add_bear((signals["TX night"] or 0) <= -0.012, 2, "台指期夜盤明顯偏空")

    add_bull((signals["Nasdaq"] or 0) >= 0.012, 1, "Nasdaq轉強")
    add_bull((signals["S&P500"] or 0) >= 0.010, 1, "S&P500轉強")
    add_bull((signals["SOX"] or 0) >= 0.018, 2, "費半轉強")
    add_bull((signals["TSM ADR"] or 0) >= 0.018, 2, "TSM ADR轉強")
    add_bull((signals["Samsung"] or 0) >= 0.018, 1, "三星轉強")
    add_bull((signals["SK Hynix"] or 0) >= 0.025, 1, "SK海力士轉強")
    add_bull((signals["VIX"] or 0) <= -0.06, 1, "VIX回落")
    add_bull((signals["TX night"] or 0) >= 0.010, 2, "台指期夜盤明顯偏多")

    news_net = int(safe_float(global_news_risk.get("net_risk_score")) or 0)
    if news_net >= 3:
        add_bear(True, min(3, max(1, news_net // 2)), f"國際重大財經政治消息偏風險：{global_news_risk.get('summary', '')}")
    elif news_net <= -3:
        add_bull(True, min(3, max(1, abs(news_net) // 2)), f"國際重大財經政治消息偏利多：{global_news_risk.get('summary', '')}")

    pending_events = global_news_risk.get("pending_events")
    if not isinstance(pending_events, list):
        pending_events = []
    pending_event_risk = 0
    pending_event_reasons: list[str] = []
    for event in pending_events:
        if not isinstance(event, dict):
            continue
        status = str(event.get("status") or "pending").lower()
        if status not in {"pending", "scheduled", "watch"}:
            continue
        importance = int(safe_float(event.get("importance")) or 0)
        risk = int(safe_float(event.get("risk_score")) or 0)
        event_risk = max(importance, risk)
        if event_risk <= 0:
            continue
        pending_event_risk += min(3, event_risk)
        title = event.get("title") or event.get("name") or "未公布重大事件"
        release_time = event.get("release_time_taipei") or event.get("release_time") or "時間待確認"
        pending_event_reasons.append(f"未公布後因事件：{title}（{release_time}）")

    net = bearish_score - bullish_score
    reset_active = abs(net) >= 3 or bearish_score >= 4 or bullish_score >= 4
    if reset_active and net >= 3:
        code = "bearish_external_reset"
        label = "外部利空重置"
        direction = "bearish"
        risk_adjustment = -2
        summary = "外部市場同步偏空，台股內部原劇本需降權，風控優先。"
        watch = "若台指期夜盤與日盤現貨都跌破前低且收不回，外部重置升級為主控風險。"
    elif reset_active and net <= -3:
        code = "bullish_external_reset"
        label = "外部利多重置"
        direction = "bullish"
        risk_adjustment = 1
        summary = "外部市場同步偏多，台股偏弱劇本可降權，但仍需現貨站回壓力確認。"
        watch = "若日盤站回昨收與短均，外部利多才算落到台股現貨。"
    elif bearish_score and bullish_score:
        code = "external_conflict_watch"
        label = "外部分歧觀察"
        direction = "mixed"
        risk_adjustment = -1 if bearish_score > bullish_score else 0
        summary = "外部多空訊號互相抵銷，容易放大日夜盤測試與假突破。"
        watch = "先看台指期夜盤與日盤現貨是否同向確認，不提前放大單邊結論。"
    elif pending_event_risk >= 3:
        code = "pending_macro_event_watch"
        label = "重大後因待公布"
        direction = "event_pending"
        risk_adjustment = -1
        summary = "重大總經事件尚未公布，不能提前定多空，但需降低單邊預測信心並設情境閘門。"
        watch = "事件公布後先看美債殖利率、美元、Nasdaq/費半、台指期夜盤是否同向重定價，再決定是否改寫下個日盤劇本。"
    else:
        code = "internal_cause_primary"
        label = "內部因果優先"
        direction = "none"
        risk_adjustment = 0
        summary = "外部市場未達重置門檻，台股內部日夜盤因果鏈維持主判斷。"
        watch = "持續用關鍵價位、量能、法人與夜日盤驗證原劇本。"

    return {
        "enabled": True,
        "available": True,
        "code": code,
        "label": label,
        "summary": summary,
        "reset_active": bool(reset_active),
        "direction": direction,
        "reset_score": int(net),
        "bearish_score": int(bearish_score),
        "bullish_score": int(bullish_score),
        "pending_event_risk_score": int(pending_event_risk),
        "risk_adjustment": int(risk_adjustment),
        "causal_priority": "external_reset" if reset_active else "internal_primary",
        "reasons": unique_text(reasons + bullish_reasons + pending_event_reasons),
        "bearish_reasons": unique_text(reasons),
        "bullish_reasons": unique_text(bullish_reasons),
        "pending_event_reasons": unique_text(pending_event_reasons),
        "pending_events": pending_events,
        "watch": watch,
        "signals": valid,
        "news_feed_status": global_news_risk.get("status", "missing"),
        "news_rule": global_news_risk.get("summary", "每日國際重大財經政治消息尚未產出；重大事件需人工或新聞模組輸入後，才可升級為新聞型重置。"),
        "news_risk": global_news_risk,
        "guardrail": "外部重置只作風控與劇本降權，不直接產生買賣命令。",
    }


def analyze_situation_psychology_context(
    forecast_date: str,
    premarket: dict,
    global_news_risk: dict | None = None,
) -> dict:
    """Score news, calendar and life-rhythm psychology that pure technical rules miss."""
    global_news_risk = global_news_risk or {}
    target = pd.to_datetime(forecast_date).date()
    prior = previous_weekday(target)
    events = global_news_risk.get("events") if isinstance(global_news_risk.get("events"), list) else []
    pending_events = (
        global_news_risk.get("pending_events")
        if isinstance(global_news_risk.get("pending_events"), list)
        else []
    )
    model_integration = (
        global_news_risk.get("model_integration")
        if isinstance(global_news_risk.get("model_integration"), dict)
        else {}
    )

    score = 0
    evidence: list[str] = []
    triggers: list[str] = []
    validation: list[str] = []

    def add(points: int, trigger: str, text: str) -> None:
        nonlocal score
        score += points
        triggers.append(trigger)
        evidence.append(text)

    us_holiday = us_market_holiday_name(prior)
    if us_holiday:
        add(
            -1,
            "us_market_holiday_repricing",
            f"前一美股基準日接近{us_holiday}，主市場訊號不足，台股容易先觀望，復市後由夜盤重新定價。",
        )
        validation.append("夜盤復市後的方向不可直接當答案，需看日盤現貨是否承接。")

    pending_count = len(pending_events)
    if pending_count:
        risk_sum = sum(int(safe_float(item.get("risk_score")) or 0) for item in pending_events if isinstance(item, dict))
        add(
            -min(3, max(1, risk_sum // 2 or 1)),
            "pending_macro_event_gate",
            f"仍有 {pending_count} 項待公布/待驗證事件，資金可能先縮手等待結果。",
        )
        validation.append("重大數據公布後，核對美股、費半、TSM ADR與台指期是否同向重定價。")

    net_news = int(safe_float(global_news_risk.get("net_risk_score")) or 0)
    if net_news >= 3:
        add(-min(3, net_news), "news_risk_pressure", "國際新聞淨風險偏高，容易提高避險與高檔退潮心理。")
    elif net_news <= -3:
        add(min(3, abs(net_news)), "news_tailwind", "國際新聞淨利多偏高，容易支撐風險偏好。")

    event_titles = " ".join(str(item.get("title", "")) for item in events if isinstance(item, dict)).lower()
    if any(word in event_titles for word in ["oil", "middle east", "geopolitical", "tension", "tariff", "fed", "inflation"]):
        add(-1, "headline_sensitivity", "新聞標題含油價、地緣、利率或通膨敏感字，盤中容易被情緒放大。")

    nasdaq = safe_float(premarket.get("nasdaq_return_1d"))
    sox = safe_float(premarket.get("sox_return_1d"))
    tsm_adr = safe_float(premarket.get("tsm_adr_return_1d"))
    tx_night = safe_float(premarket.get("tx_night_spread_per"))
    if (sox is not None and sox >= 0.01) or (tsm_adr is not None and tsm_adr >= 0.015):
        add(2, "us_tech_chain_anchor", "費半或TSM ADR轉強，台股科技鏈主錨提供心理支撐。")
    if nasdaq is not None and nasdaq < 0 and sox is not None and sox > 0:
        add(0, "us_market_mixed_signal", "Nasdaq與費半方向分歧，代表主市場不是全面同向，日盤需看權值股驗證。")
    if tx_night is not None and abs(tx_night) < 0.003:
        evidence.append("台指期夜盤接近中性，代表方向仍交給日盤現貨與開盤後承接決定。")

    if model_integration.get("expected_psychology"):
        evidence.append(f"新聞模組預期心理: {model_integration.get('expected_psychology')}")
    if not validation:
        validation.append("以開盤後30～60分鐘、量能、族群廣度與收盤位置驗證情境心理。")

    if score >= 3:
        label = "情境心理偏多"
        posture = "外部與生活節奏支持風險偏好，但仍需日盤驗證。"
    elif score <= -3:
        label = "情境心理偏防守"
        posture = "新聞/等待/制度節奏提高觀望或避險心理，技術多方需降權。"
    else:
        label = "情境心理混合"
        posture = "情境因素多空交錯，不能只用教科書技術線判方向。"

    return {
        "available": True,
        "framework": "situation_psychology_context_v1",
        "label": label,
        "score": int(max(-6, min(6, score))),
        "posture": posture,
        "triggers": unique_text(triggers),
        "evidence": top_items(evidence, 8),
        "next_validation": unique_text(validation),
        "news_status": global_news_risk.get("status", "missing"),
        "pending_event_count": pending_count,
        "guardrail": "新聞面與生活心理面只作情境濾鏡和風控降權；必須由台指期夜盤、日盤現貨、量能與族群驗證，不產生投資命令。",
    }


def us_market_holiday_name(day) -> str | None:
    """Return common US market holiday names needed for market-rhythm context."""
    d = pd.to_datetime(day).date()
    if d.month == 1 and d.day == 1:
        return "美國元旦休市"
    if d.month == 7 and d.day == 4:
        return "美國獨立日休市"
    if d.month == 12 and d.day == 25:
        return "美國聖誕節休市"
    if d.month == 9 and d.weekday() == 0 and 1 <= d.day <= 7:
        return "美國勞工節休市"
    if d.month == 11 and d.weekday() == 3 and 22 <= d.day <= 28:
        return "美國感恩節休市"
    return None


def analyze_fundamental_constitution(payload: dict) -> dict:
    """Score the market's slower-moving body before tactical timing overrides it."""
    premarket = payload.get("premarket", {})
    news = payload.get("global_news_risk", {})
    bagua = payload.get("bagua_lifecycle", {})
    technical = payload.get("technical_phase", {})
    capital = payload.get("capital_flow", {})
    memory = payload.get("memory_industry_risk", {})
    score = 70
    confidence = 70
    drivers: list[str] = []
    pressures: list[str] = []
    data_gaps: list[str] = []

    def add_driver(condition: bool, points: int, text: str) -> None:
        nonlocal score
        if condition:
            score += points
            drivers.append(text)

    def add_pressure(condition: bool, points: int, text: str) -> None:
        nonlocal score
        if condition:
            score -= points
            pressures.append(text)

    nasdaq = safe_float(premarket.get("nasdaq_return_1d"))
    sox = safe_float(premarket.get("sox_return_1d"))
    tsm_adr = safe_float(premarket.get("tsm_adr_return_1d"))
    ewt = safe_float(premarket.get("ewt_return_1d"))
    samsung = safe_float(premarket.get("samsung_return_1d"))
    hynix = safe_float(premarket.get("sk_hynix_return_1d"))
    vix = safe_float(premarket.get("vix_return_1d"))
    usd_twd = safe_float(premarket.get("usd_twd_return_1d"))
    treasury_10y = safe_float(premarket.get("treasury_10y_return_1d"))
    news_net = int(safe_float(news.get("net_risk_score")) or 0)
    technical_signals = technical.get("signals", {})

    add_driver((nasdaq or 0) >= 0.01, 3, "Nasdaq維持科技風險偏好")
    add_driver((sox or 0) >= 0.015, 5, "費半轉強，AI/半導體主軸支撐")
    add_driver((tsm_adr or 0) >= 0.015, 4, "TSM ADR轉強，權值核心支撐")
    add_driver((samsung or 0) >= 0.012 or (hynix or 0) >= 0.018, 3, "韓系半導體轉強，產業鏈共振")
    add_driver((ewt or 0) >= 0.01, 3, "台灣ETF偏強，外資對台股體質仍支持")
    add_driver(technical_signals.get("above_ma20") or technical_signals.get("above_ma60"), 4, "指數仍在主要均線結構上方")
    add_driver(bagua.get("primary", {}).get("code") in {"ZHEN", "XUN", "LI", "QIAN"}, 3, "主週期仍屬進攻或極盛段")

    add_pressure((nasdaq or 0) <= -0.015, 4, "Nasdaq下跌，科技估值承壓")
    add_pressure((sox or 0) <= -0.02, 6, "費半下跌，半導體主軸受壓")
    add_pressure((tsm_adr or 0) <= -0.02, 5, "TSM ADR下跌，台股權值核心受壓")
    add_pressure((ewt or 0) <= -0.015, 4, "台灣ETF轉弱，外資對台股折價")
    add_pressure((vix or 0) >= 0.08, 5, "VIX升高，市場避險情緒升溫")
    add_pressure((usd_twd or 0) >= 0.004, 3, "美元台幣上升，匯率壓力增加")
    add_pressure((treasury_10y or 0) >= 0.01, 3, "美債殖利率上行，股市估值壓力增加")
    add_pressure(news_net >= 4, min(8, news_net), "國際重大財經政治消息偏風險")
    add_pressure((safe_float(memory.get("points")) or 0) >= 4, 4, "記憶體/半導體尾部風險升高")

    if not news.get("available"):
        confidence -= 12
        data_gaps.append("缺每日國際重大財經政治新聞風險檔")
    if not capital or capital.get("status") in {"missing", "stale"}:
        confidence -= 10
        data_gaps.append("法人/期貨籌碼資料不足或過期")
    if all(value is None for value in [nasdaq, sox, tsm_adr, ewt, vix, usd_twd]):
        confidence -= 18
        data_gaps.append("外部市場價格代理不足")

    score = int(round(clamp(score / 100) * 100))
    confidence = int(round(clamp(confidence / 100) * 100))
    if score >= 78:
        label = "體質強健"
        code = "strong_constitution"
    elif score >= 62:
        label = "體質尚穩"
        code = "stable_constitution"
    elif score >= 46:
        label = "體質轉弱觀察"
        code = "weakening_constitution"
    else:
        label = "體質弱化"
        code = "fragile_constitution"
    return {
        "framework": "fundamental_constitution_v1",
        "code": code,
        "label": label,
        "score": score,
        "confidence": confidence,
        "drivers": unique_text(drivers),
        "pressures": unique_text(pressures),
        "data_gaps": unique_text(data_gaps),
        "summary": f"基本面命格判斷為{label}；分數 {score}/100，可信度 {confidence}/100。",
        "interpretation": "命格代表大方向體質，不能取代日盤驗證；外部重置或族群廣度惡化時，短線戰術必須降權。",
        "guardrail": "基本面只回答市場身體是否耐震，不回答今日一定漲跌，也不得成為投資命令。",
    }


def classify_fact_changing_medicines(payload: dict) -> dict:
    """Find immediate catalysts that can change market facts, not just explain background."""
    premarket = payload.get("premarket", {})
    external = payload.get("external_event_reset_monitor", {})
    crash = payload.get("crash_monitor", {})
    health = payload.get("market_health", {})
    technical = payload.get("technical_phase", {})
    levels = technical.get("levels", {})
    breadth = payload.get("market_breadth", {})

    medicines: list[str] = []
    weakening_score = 0
    repair_score = 0
    distortion_score = 0

    def weaken(points: int, text: str) -> None:
        nonlocal weakening_score
        weakening_score += points
        medicines.append(f"走弱即效藥: {text}")

    def repair(points: int, text: str) -> None:
        nonlocal repair_score
        repair_score += points
        medicines.append(f"修復即效藥: {text}")

    def distort(points: int, text: str) -> None:
        nonlocal distortion_score
        distortion_score += points
        medicines.append(f"失真即效藥: {text}")

    nasdaq = safe_float(premarket.get("nasdaq_return_1d"))
    sox = safe_float(premarket.get("sox_return_1d"))
    tsm_adr = safe_float(premarket.get("tsm_adr_return_1d"))
    vix = safe_float(premarket.get("vix_return_1d"))
    night = safe_float(premarket.get("tx_night_spread_per"))
    treasury_10y = safe_float(premarket.get("treasury_10y_return_1d"))
    usd_twd = safe_float(premarket.get("usd_twd_return_1d"))
    close = safe_float(levels.get("close"))
    ma20 = safe_float(levels.get("ma20"))
    health_score = safe_int(health.get("score") or crash.get("health_score"), 50)
    risk_score = safe_int(health.get("risk_score") or crash.get("risk_score"), 50)
    advancing = safe_float(breadth.get("price_advancing_fraction"))
    up_volume = safe_float(breadth.get("price_up_volume_fraction"))

    if external.get("reset_active") and external.get("direction") == "bearish":
        weaken(4, "外部事件重置偏空，足以先改變風控優先序。")
    if external.get("reset_active") and external.get("direction") == "bullish":
        repair(4, "外部事件重置偏多，足以先改變風險胃納。")
    if nasdaq is not None and nasdaq <= -0.015:
        weaken(3, "Nasdaq急跌，科技估值與風險偏好立即降溫。")
    elif nasdaq is not None and nasdaq >= 0.015:
        repair(2, "Nasdaq強彈，科技風險胃納立即修復。")
    if sox is not None and sox <= -0.018:
        weaken(3, "費半急跌，半導體權值定價立即承壓。")
    elif sox is not None and sox >= 0.018:
        repair(3, "費半急彈，半導體主軸立即獲得修復藥。")
    if tsm_adr is not None and tsm_adr <= -0.012:
        weaken(2, "TSM ADR明顯轉弱，台股權值開盤定價容易被改寫。")
    elif tsm_adr is not None and tsm_adr >= 0.012:
        repair(2, "TSM ADR明顯轉強，台股權值承接預期立即提高。")
    if vix is not None and vix >= 0.08:
        weaken(3, "VIX急升，避險程式會先降低風險部位。")
    if treasury_10y is not None and treasury_10y >= 0.008:
        weaken(2, "美債殖利率急升，估值折現率立即被重新定價。")
    elif treasury_10y is not None and treasury_10y <= -0.008:
        repair(2, "美債殖利率急降，估值壓力立即舒緩。")
    if usd_twd is not None and usd_twd >= 0.006:
        weaken(2, "台幣急貶，外資撤出與匯損疑慮立即升高。")
    elif usd_twd is not None and usd_twd <= -0.006:
        repair(2, "台幣轉強，外資風險胃納與承接意願改善。")
    if night is not None and night <= -0.008:
        weaken(2, "台指夜盤大幅偏空，日盤開局容易被迫重新定價。")
    elif night is not None and night >= 0.008:
        repair(2, "台指夜盤大幅偏多，日盤開局容易先修復風險偏好。")
    if close is not None and ma20 is not None and close < ma20:
        weaken(2, "指數跌破20日線，技術防線由背景壓力轉成即刻事實。")
    if health_score < 30 or risk_score > 75:
        weaken(4, "健康/風險紅線觸發，模型需先切到急症風控。")
    if advancing is not None and advancing < 0.35 and up_volume is not None and up_volume < 0.45:
        weaken(2, "下跌家數與量能同步惡化，現貨廣度直接確認壓力。")
    elif advancing is not None and advancing > 0.58 and up_volume is not None and up_volume > 0.58:
        repair(2, "上漲家數與量能同步擴散，現貨廣度直接確認修復。")

    if abs(weakening_score - repair_score) <= 1 and (weakening_score or repair_score):
        distort(1, "多空即效藥互相抵銷，容易形成假突破、假跌破或盤整失真。")

    net_score = repair_score - weakening_score
    if net_score >= 3:
        bias = "repair"
        label = "修復藥主導"
        effect = "即效因子偏向提高承接與風險胃納。"
    elif net_score <= -3:
        bias = "weakening"
        label = "發病藥主導"
        effect = "即效因子偏向降低承接與風險胃納。"
    elif distortion_score:
        bias = "distortion"
        label = "藥性互抵/失真"
        effect = "多空藥性互抵，日盤更需要用開高低收與族群廣度驗證。"
    else:
        bias = "neutral"
        label = "無強即效藥"
        effect = "目前沒有足以單獨改變市場事實的即效因子。"

    return {
        "framework": "fact_changing_medicine_v1",
        "label": label,
        "bias": bias,
        "net_score": net_score,
        "weakening_score": weakening_score,
        "repair_score": repair_score,
        "distortion_score": distortion_score,
        "medicines": unique_text(medicines),
        "effect": effect,
        "rule": "病灶是累積背景；即效藥是一發生就改變資金風險判斷的因素；藥效必須由日盤現貨驗證。",
    }


def build_dialogue_core_rules(payload: dict) -> dict:
    """Convert repeated research dialogue into reusable model rules."""
    technical = payload.get("technical_phase", {})
    levels = technical.get("levels", {})
    bagua = payload.get("bagua_lifecycle", {})
    sector = payload.get("sector_pressure_observation", {})
    capital = payload.get("capital_flow", {})
    programmed = payload.get("programmed_pressure_pattern", {})
    premarket = payload.get("premarket", {})
    news = payload.get("global_news_risk", {})
    external = payload.get("external_event_reset_monitor", {})
    day_night = payload.get("day_night_variance_pattern", {})
    heart = payload.get("market_heart_rhythm", {})

    chronic_conditions: list[str] = []
    trigger_conditions: list[str] = []
    transmission_checks: list[str] = []
    model_rules: list[str] = []

    close = safe_float(levels.get("close"))
    ma20 = safe_float(levels.get("ma20"))
    drawdown = safe_float(levels.get("drawdown_from_swing_high"))
    range_20 = safe_float(levels.get("range_20d"))
    sector_risk = safe_int(sector.get("risk_score"), 0)
    capital_score = safe_int(capital.get("score"), 0)
    pressure_score = safe_int(programmed.get("current_score"), 0)
    night = safe_float(premarket.get("tx_night_spread_per"))
    nasdaq = safe_float(premarket.get("nasdaq_return_1d"))
    sox = safe_float(premarket.get("sox_return_1d"))
    vix = safe_float(premarket.get("vix_return_1d"))
    treasury_10y = safe_float(premarket.get("treasury_10y_return_1d"))
    usd_twd = safe_float(premarket.get("usd_twd_return_1d"))
    news_net = safe_int(news.get("net_risk_score"), 0)

    if close is not None and ma20 is not None and close >= ma20:
        chronic_conditions.append("高檔仍在短中期均線上，代表不是低位恐慌，而是高檔壓力測試。")
    if drawdown is not None and drawdown >= -0.05:
        chronic_conditions.append("距離波段高點仍近，前高套牢、獲利了結與追價退潮屬既有病灶。")
    if range_20 is not None and range_20 >= 0.06:
        chronic_conditions.append("近20日振幅偏大，表示換手、停損與壓力測試已累積。")
    if bagua.get("roles", {}).get("background_gua", {}).get("code") in {"QIAN", "LI", "DUI"}:
        chronic_conditions.append("背景卦在高位或極盛段，市場對利率、估值與權值過熱更敏感。")
    if sector_risk >= 3:
        chronic_conditions.append("族群分化偏高，指數表面強弱可能掩蓋內部退潮。")
    if capital_score <= -2:
        chronic_conditions.append("法人/期貨籌碼偏防守，代表倉位並非完全無壓。")
    if pressure_score >= 2:
        chronic_conditions.append("近期壓低行為群聚，單日下跌不可只用新聞解釋。")

    if night is not None and abs(night) >= 0.003:
        trigger_conditions.append(f"台指夜盤 {pct(night)}，期貨端先做預期重定價。")
    if nasdaq is not None and abs(nasdaq) >= 0.006:
        trigger_conditions.append(f"Nasdaq {pct(nasdaq)}，科技風險偏好被重新定價。")
    if sox is not None and abs(sox) >= 0.01:
        trigger_conditions.append(f"費半 {pct(sox)}，半導體權值預期被觸發。")
    if vix is not None and abs(vix) >= 0.04:
        trigger_conditions.append(f"VIX {pct(vix)}，避險程式可能調整倉位。")
    if treasury_10y is not None and abs(treasury_10y) >= 0.004:
        trigger_conditions.append(f"美債10年殖利率 {pct(treasury_10y)}，估值折現率被觸發重估。")
    if usd_twd is not None and abs(usd_twd) >= 0.004:
        trigger_conditions.append(f"美元/台幣 {pct(usd_twd)}，外資匯率風險被觸發重估。")
    if news_net >= 2:
        trigger_conditions.append("國際重大財經政治新聞偏風險，作為觸發鈕而非單獨主因。")
    if external.get("reset_active"):
        trigger_conditions.append(f"外部事件重置啟動，方向 {external.get('direction', 'unknown')}。")

    transmission_checks.extend(
        [
            "夜盤只當前哨，日盤開盤、30～60分鐘收回、低點與收盤才是裁判。",
            "若夜盤方向傳到日盤收盤，代表觸發被現貨承認；若日盤收回，代表只是避險或洗盤。",
            "一天大變不是慢性病灶消失或生成，而是倉位、停損、避險與預期差重新定價。",
        ]
    )
    model_rules.extend(
        [
            "先定病灶，再找觸發，再看日盤承認，最後寫入病歷。",
            "新聞、油價、通膨、利率只能當觸發鈕；除非連續收盤與族群廣度確認，否則不得升級為主因。",
            "0/1分流：1=觸發被吸收並修復；0=觸發被現貨確認並延伸成破線/回測。",
            "報告只輸出仲裁結果；細部計算留在資料庫與 detail。",
        ]
    )

    chronic_present = bool(chronic_conditions)
    trigger_present = bool(trigger_conditions)
    if chronic_present and trigger_present:
        focus = "慢性病灶存在，單日變化主看觸發鈕與倉位重定價是否被日盤承認。"
        state = "disease_with_trigger"
    elif chronic_present:
        focus = "慢性病灶存在，但今日缺少強觸發；以關鍵線與收盤位置等待發病或修復。"
        state = "disease_without_trigger"
    elif trigger_present:
        focus = "觸發鈕存在，但慢性病灶不重；需防止把短線事件誤判成主趨勢。"
        state = "trigger_without_disease"
    else:
        focus = "病灶與觸發都不強，維持心律與日夜盤傳導觀察。"
        state = "no_strong_core"

    return {
        "framework": "dialogue_core_rules_v1",
        "state": state,
        "focus": focus,
        "chronic_conditions": unique_text(chronic_conditions),
        "trigger_conditions": unique_text(trigger_conditions),
        "transmission_checks": transmission_checks,
        "model_rules": model_rules,
        "day_night_state": day_night.get("label", "資料不足"),
        "heart_state": heart.get("label", "資料不足"),
        "report_policy": "細部分析進資料庫與detail；主報告只輸出0/1、關鍵數字、核心病因、觸發與下一驗證。",
        "guardrail": "對談規則是可驗證框架，不是投資命令，也不證明單一主體操控。",
    }


def analyze_practical_cause_arbitration(payload: dict) -> dict:
    """Separate persistent internal causes from irregular external triggers."""
    technical = payload.get("technical_phase", {})
    levels = technical.get("levels", {})
    tradeable = payload.get("tradeable_cycle", {})
    bagua = payload.get("bagua_lifecycle", {})
    sector = payload.get("sector_pressure_observation", {})
    capital = payload.get("capital_flow", {})
    premarket = payload.get("premarket", {})
    news = payload.get("global_news_risk", {})
    external = payload.get("external_event_reset_monitor", {})
    candle = payload.get("candlestick_pattern", {})
    programmed = payload.get("programmed_pressure_pattern", {})
    dialogue = payload.get("dialogue_core_rules") or build_dialogue_core_rules(payload)

    internal_score = 0
    trigger_score = 0
    internal_causes: list[str] = []
    external_triggers: list[str] = []
    confirmations: list[str] = []
    exclusions: list[str] = []

    def add_internal(points: int, text: str) -> None:
        nonlocal internal_score
        internal_score += points
        internal_causes.append(text)

    def add_trigger(points: int, text: str) -> None:
        nonlocal trigger_score
        trigger_score += points
        external_triggers.append(text)

    close = safe_float(levels.get("close"))
    swing_high = safe_float(levels.get("swing_high"))
    ma20 = safe_float(levels.get("ma20"))
    drawdown = safe_float(levels.get("drawdown_from_swing_high"))
    range_20 = safe_float(levels.get("range_20d"))
    close_position = safe_float(candle.get("close_position"))
    sector_risk = safe_int(sector.get("risk_score"), 0)
    capital_score = safe_int(capital.get("score"), 0)
    pressure_score = safe_int(programmed.get("current_score"), 0)

    if close is not None and ma20 is not None and close >= ma20:
        add_internal(1, "指數仍在20日線上方，屬高檔結構而非低位恐慌。")
    if drawdown is not None and drawdown >= -0.04:
        add_internal(2, "距離近期波段高點不遠，高檔估值、前高套牢與獲利了結壓力仍在。")
    if range_20 is not None and range_20 >= 0.06:
        add_internal(1, "近20日波動區間偏大，代表高檔換手與測壓頻繁。")
    current_position = tradeable.get("current_position", {})
    if current_position.get("code") in {"late_rally", "mature_rally"} or "末升" in current_position.get("label", ""):
        add_internal(2, "年度波段燈塔接近末升/高檔加速，追價與反轉敏感度提高。")
    if bagua.get("roles", {}).get("background_gua", {}).get("code") in {"QIAN", "LI", "DUI"}:
        add_internal(1, "背景卦位處高位或極盛段，市場容易把既有壓力放大檢驗。")
    if sector_risk >= 3:
        add_internal(2, "族群分化偏高，指數上漲可能掩蓋內部換手壓力。")
    if capital_score <= -2:
        add_internal(1, "法人/期貨籌碼偏防守，代表高檔承接仍需驗證。")
    if pressure_score >= 2:
        add_internal(1, "近期壓低群聚，顯示高檔壓力不是單日雜訊。")
    if close_position is not None and close_position <= 0.35:
        add_internal(1, "前一日收盤位置偏低，尾盤承接仍未完全證明。")
    if dialogue.get("state") in {"disease_with_trigger", "disease_without_trigger"}:
        add_internal(1, f"對談核心規則: {dialogue.get('focus')}")

    nasdaq = safe_float(premarket.get("nasdaq_return_1d"))
    sox = safe_float(premarket.get("sox_return_1d"))
    tsm_adr = safe_float(premarket.get("tsm_adr_return_1d"))
    vix = safe_float(premarket.get("vix_return_1d"))
    night = safe_float(premarket.get("tx_night_spread_per"))
    treasury_10y = safe_float(premarket.get("treasury_10y_return_1d"))
    news_net = safe_int(news.get("net_risk_score"), 0)

    if news_net >= 2:
        add_trigger(min(3, news_net), "國際新聞風險偏高，容易成為啟動高檔病灶的開關。")
    if nasdaq is not None and nasdaq <= -0.006:
        add_trigger(1, "Nasdaq轉弱，科技風險偏好先降溫。")
    if sox is not None and sox <= -0.012:
        add_trigger(2, "費半轉弱，半導體權值承接需重新驗證。")
    if tsm_adr is not None and tsm_adr <= -0.008:
        add_trigger(1, "TSM ADR轉弱，台股權值定價先承壓。")
    if vix is not None and vix >= 0.04:
        add_trigger(1, "VIX上升，避險程式容易先用期貨降風險。")
    if treasury_10y is not None and treasury_10y >= 0.004:
        add_trigger(1, "美債殖利率上行，估值壓力被重新定價。")
    if night is not None and night <= -0.003:
        add_trigger(1, "台指期夜盤偏空，表示外部風險先在期貨端試壓。")
    if external.get("reset_active") and external.get("direction") == "bearish":
        add_trigger(3, "外部利空重置已啟動，短線可暫時接管風控優先權。")
    if dialogue.get("state") in {"disease_with_trigger", "trigger_without_disease"}:
        add_trigger(1, "對談核心規則: 單日劇變優先檢查觸發鈕、倉位重定價與日盤是否承認。")

    if internal_score >= 5 and trigger_score >= 2:
        code = "internal_disease_external_trigger"
        label = "內部主病灶，外部觸發"
        practical_primary = "internal_structure"
        summary = "高檔結構壓力早已存在，今日外部因素較像啟動開關，而非突然生成的新主因。"
    elif trigger_score >= 5 and external.get("reset_active"):
        code = "external_trigger_temporary_primary"
        label = "外部重置暫時主導"
        practical_primary = "external_reset"
        summary = "外部衝擊已達重置門檻，短線先由外部風控接管，但仍需日盤現貨確認是否改變主趨勢。"
    elif internal_score >= 5:
        code = "internal_structure_primary"
        label = "內部結構主導"
        practical_primary = "internal_structure"
        summary = "盤勢主因仍是高檔估值、套牢、換手與族群分化，外部消息只作輔助觀察。"
    elif trigger_score >= 3:
        code = "trigger_watch_without_internal_confirmation"
        label = "外部觸發待驗"
        practical_primary = "trigger_watch"
        summary = "外部訊號偏防守，但內部病灶分數不足，需等日盤現貨確認是否傳導。"
    else:
        code = "no_clear_practical_primary"
        label = "主因未明"
        practical_primary = "cash_validation"
        summary = "目前主因與觸發器都未形成明確優勢，等待日盤現貨、量能與族群廣度確認。"

    combined_severity = internal_score + trigger_score
    if practical_primary == "external_reset" and trigger_score >= 6:
        treatment_stage = "急症風控"
        treatment_policy = "立即提高破線、收不回與連續惡化核對；所有樂觀敘事降權。"
        treatment_phase = 4
    elif combined_severity >= 9:
        treatment_stage = "發作治療"
        treatment_policy = "病灶與觸發已共振，盤中需加密核對權值、族群廣度、量能與收盤位置。"
        treatment_phase = 3
    elif combined_severity >= 6:
        treatment_stage = "早期治療"
        treatment_policy = "早期處理最有效；先降低方向信心、提高防守線與觸發條件監控，等待日盤確認。"
        treatment_phase = 2
    else:
        treatment_stage = "潛伏觀察"
        treatment_policy = "病灶或觸發尚未共振，維持觀察，不把單一消息放大成危機。"
        treatment_phase = 1

    if trigger_score >= 6 and external.get("reset_active"):
        medicine_type = "重藥急救"
        medicine_effect = "外部衝擊夠強，先以風控處理，等現貨確認是否止血。"
    elif trigger_score >= 3 and internal_score >= 5:
        medicine_type = "壓力測試藥"
        medicine_effect = "用外部壓力測試高檔承接，若日盤收回，代表治療有效；若收不回，病灶擴大。"
    elif trigger_score >= 3:
        medicine_type = "外部試藥"
        medicine_effect = "外部訊號偏強，但內部病灶不重，先觀察是否傳導到現貨。"
    elif internal_score >= 5:
        medicine_type = "內部調理"
        medicine_effect = "主要靠換手、量能與族群輪動自行消化病灶。"
    else:
        medicine_type = "觀察處方"
        medicine_effect = "暫無明確藥性，維持例行監測。"

    if treatment_phase >= 3 and trigger_score >= 3:
        correct_medicine = "待日盤驗證"
        after_effect_risk = "中"
    elif treatment_phase <= 2 and internal_score >= trigger_score:
        correct_medicine = "偏對症"
        after_effect_risk = "低至中"
    else:
        correct_medicine = "待確認"
        after_effect_risk = "低"

    confirmations.extend(
        [
            "若日盤跌破夜盤低點或昨低且30～60分鐘收不回，代表外部觸發已傳導成現貨轉弱。",
            "若日盤開低後站回夜盤收盤、昨收與短均，代表外部因素多屬避險觸發，不是主趨勢轉弱。",
            "若權值股與金融同步轉弱且下跌家數擴大，內部病灶升級。",
            "若族群廣度擴散且成交量為換手承接，內部病灶降級。",
        ]
    )
    exclusions.extend(
        [
            "單一新聞標題不得直接當作下跌主因，必須有夜盤、日盤或族群驗證。",
            "夜盤偏空不得直接視為現貨轉弱，必須由日盤承認。",
            "基本面尚穩時，外部觸發只先改變節奏，不自動改變主波段。",
        ]
    )
    fact_medicine = classify_fact_changing_medicines(payload)
    causality_pipeline = [
        {
            "stage": "因",
            "meaning": "長期病灶基因先存在，不一定當天發作。",
            "evidence": top_items(unique_text(internal_causes), 4),
        },
        {
            "stage": "跡象",
            "meaning": "病灶透過高檔換手、壓低群聚、尾盤承接不足或族群分化露出症狀。",
            "evidence": top_items(unique_text(internal_causes), 4),
        },
        {
            "stage": "觸發按鈕",
            "meaning": "即效藥或外部觸發出現，資金開始重新定價風險。",
            "evidence": top_items(fact_medicine.get("medicines", []) or external_triggers, 4),
        },
        {
            "stage": "果",
            "meaning": "日盤現貨用開高低收、權值、廣度與量能決定是壓力被確認，還是反向修復啟動。",
            "evidence": [
                "跌破夜盤低點或昨低且收不回 = 壓力被確認。",
                "站回夜盤收盤、昨收或短均 = 反向修復啟動。",
            ],
        },
        {
            "stage": "病歷",
            "meaning": "收盤後把判斷、錯因、藥效與陰霾留底，作為下次同型病例先驗。",
            "evidence": ["不得用事後結果回填舊預測，只能作下一次權重與警戒修正。"],
        },
    ]

    return {
        "framework": "practical_cause_arbitration_v1",
        "code": code,
        "label": label,
        "practical_primary": practical_primary,
        "internal_structure_score": internal_score,
        "external_trigger_score": trigger_score,
        "internal_causes": unique_text(internal_causes),
        "external_triggers": unique_text(external_triggers),
        "combined_severity": combined_severity,
        "treatment_stage": treatment_stage,
        "treatment_phase": treatment_phase,
        "treatment_policy": treatment_policy,
        "medicine_type": medicine_type,
        "medicine_effect": medicine_effect,
        "fact_changing_medicine": fact_medicine,
        "fact_changing_medicines": fact_medicine.get("medicines", []),
        "dialogue_core_rules": dialogue,
        "dialogue_rules_applied": top_items(dialogue.get("model_rules", []), 4),
        "dialogue_focus": dialogue.get("focus"),
        "immediate_medicine_bias": fact_medicine.get("bias"),
        "immediate_medicine_score": fact_medicine.get("net_score"),
        "causality_pipeline": causality_pipeline,
        "causality_rule": "先有因，再有跡象，再由即效藥/觸發按鈕產生果；最後用日盤與收盤驗證，寫入病歷。",
        "correct_medicine": correct_medicine,
        "after_effect_risk": after_effect_risk,
        "summary": summary,
        "confirmation": confirmations,
        "exclusion": exclusions,
        "one_zero_rule": "1=外部觸發被日盤吸收，主趨勢維持；0=外部觸發被日盤確認，進入回測或風險升級。",
        "gene_trigger_model": "病灶基因=高檔估值、前高套牢、融資追價、權值過熱、族群分化等長期內部條件；觸發條件=美股、油價、利率、新聞、夜盤、開盤缺口等不定期開關。",
        "model_effect": "原因歸因先建立長期存在的病灶基因，再判斷今日外部事件是否觸發發作；避免每天用不同新聞重寫主因。",
        "guardrail": "實務主因仲裁只做風險與病因排序，不證明單一主體操控，也不產生投資命令。",
    }


def analyze_market_protection_layers(payload: dict) -> dict:
    """Diagnose whether the market's self-protection layers are still working."""
    crash = payload.get("crash_monitor", {})
    health = payload.get("market_health", {})
    health_value = health.get("health_value", {})
    diagnosis = health.get("diagnosis", {})
    intraday = payload.get("intraday_tactical_monitor", {})
    stethoscope = payload.get("market_stethoscope", {})
    practical = payload.get("practical_cause_arbitration", {})
    fundamental = payload.get("fundamental_constitution", {})
    external = payload.get("external_event_reset_monitor", {})
    day_night = payload.get("day_night_variance_pattern", {})
    cross_market = payload.get("cross_market_entanglement", {})
    close_cause = payload.get("close_cause_attribution", {})

    risk_value = safe_float(crash.get("risk_value")) or 0.0
    health_score = safe_float(crash.get("health_score")) or 0.0
    close = safe_float(crash.get("close"))
    low = safe_float(crash.get("low") or crash.get("effective_low"))
    levels = crash.get("levels", {})
    normal_gen_low = safe_float(levels.get("normal_gen_low"))
    normal_gen_high = safe_float(levels.get("normal_gen_high"))
    normal_bottom_low = safe_float(levels.get("normal_bottom_low"))
    crash_warning = safe_float(levels.get("crash_warning"))
    fundamental_score = safe_float(fundamental.get("score"))
    stethoscope_score = safe_int(stethoscope.get("score"), 0)
    practical_phase = safe_int(practical.get("treatment_phase"), 0)
    external_score = safe_int(external.get("reset_score"), 0)
    cross_score = safe_int(cross_market.get("score"), 0)

    layers: list[dict] = []

    def add_layer(name: str, state: str, score: int, evidence: str, failure: str) -> None:
        labels = {
            "active": "啟動/有效",
            "watch": "觀察",
            "warning": "偏弱",
            "failed": "失效警戒",
        }
        layers.append(
            {
                "name": name,
                "state": state,
                "label": labels.get(state, state),
                "score": score,
                "evidence": evidence,
                "failure_condition": failure,
            }
        )

    if crash.get("alert_code") in {"red", "crash_warning"} or risk_value >= 75:
        add_layer(
            "價格線防呆",
            "failed",
            -3,
            "風險紅線或崩盤前奏條件已接近成立，價格防線失靈風險升高。",
            "跌破合理底或崩盤線且收盤、連續收盤都收不回。",
        )
    elif low is not None and normal_gen_low is not None and low < normal_gen_low and close is not None and close >= normal_gen_low:
        add_layer(
            "價格線防呆",
            "active",
            2,
            "盤中刺破高檔正常艮區下緣後收回，代表防線有被測試並暫時守住。",
            "隔日再破今日低點且收不回。",
        )
    elif close is not None and normal_gen_low is not None and normal_gen_high is not None and normal_gen_low <= close <= normal_gen_high:
        add_layer(
            "價格線防呆",
            "active",
            2,
            "收盤仍在高檔正常艮區，屬臨界壓測但未跌入重大修正區。",
            "收盤跌出高檔正常艮區，且無法快速站回。",
        )
    elif close is not None and normal_bottom_low is not None and close > normal_bottom_low:
        add_layer(
            "價格線防呆",
            "watch",
            0,
            "尚未接近合理底失守，價格防線仍需用下一交易日確認。",
            "跌破合理底部區下緣且收不回。",
        )
    else:
        add_layer(
            "價格線防呆",
            "warning",
            -1,
            "價格資料不足或已偏離正常防守區，需保守看待。",
            "資料補齊後若仍破線，升級為失效警戒。",
        )

    if health_score >= 65 and risk_value < 50 and not health_value.get("controllable_risk", True):
        add_layer(
            "資金承接防呆",
            "watch",
            0,
            "健康分數仍高但健康價值判斷已轉風險升溫，代表有承接但不可放鬆。",
            "承接量縮、族群廣度惡化、期現同步破位。",
        )
    elif health_score >= 65 and risk_value < 45:
        add_layer(
            "資金承接防呆",
            "active",
            2,
            "健康分數高且風險值未升高，市場仍有自我承接能力。",
            "健康指數跌破30或風險值升破75。",
        )
    elif risk_value >= 60:
        add_layer(
            "資金承接防呆",
            "warning",
            -2,
            "風險值升高，承接防呆可能轉為被動止血。",
            "放量收低且隔日續破。",
        )
    else:
        add_layer(
            "資金承接防呆",
            "watch",
            0,
            "承接力尚需由量能、族群廣度與期現差補證。",
            "反彈無量、下跌放量且收近低。",
        )

    if stethoscope_score <= -5 or diagnosis.get("primary") == "deteriorating":
        add_layer(
            "心理嚇阻防呆",
            "warning",
            -2,
            "生命徵象偏混合或轉弱，恐慌與壓低測底正在提高。",
            "跌破關鍵線後不再快速收回，恐慌由試探轉成連鎖。",
        )
    elif day_night.get("score", 0) >= 0:
        add_layer(
            "心理嚇阻防呆",
            "active",
            1,
            "日夜盤沒有同步惡化，壓測仍可能被收盤裁判吸收。",
            "夜盤偏空被日盤現貨確認且連續收不回。",
        )
    else:
        add_layer(
            "心理嚇阻防呆",
            "watch",
            0,
            "心理層仍在多空拉鋸，需看空方壓線是否反被回補牽制。",
            "空方壓線成功且多方不再守線。",
        )

    if external.get("reset_active") and external.get("direction") == "bearish" and cross_score <= -4:
        add_layer(
            "跨盤/制度風控防呆",
            "warning",
            -2,
            "外部重置與跨盤偏空接近共振，制度與保證金風控容易放大波動。",
            "美股、台指夜盤、日盤現貨同步破位。",
        )
    elif practical_phase >= 4:
        add_layer(
            "跨盤/制度風控防呆",
            "failed",
            -3,
            "模型已進入急症風控，制度性賣壓或連鎖保證金壓力需優先處理。",
            "急症風控後仍連續破線收不回。",
        )
    else:
        add_layer(
            "跨盤/制度風控防呆",
            "watch",
            0,
            "目前仍是外部壓力與內部承接的拉鋸，尚未見制度層失序證據。",
            "期現同步失衡、波動擴大、成交流動性明顯惡化。",
        )

    if close is not None and crash_warning is not None and close <= crash_warning:
        add_layer(
            "政策底線防呆",
            "failed",
            -3,
            "收盤接近或跌破崩盤前奏線，才進入政策底線觀察。",
            "政策或市場穩定工具出現後仍無法止穩。",
        )
    elif risk_value >= 75:
        add_layer(
            "政策底線防呆",
            "warning",
            -2,
            "風險值已達高警戒，政策底線需列入觀察但不可假設必然出手。",
            "風險值高檔且連續收不回關鍵線。",
        )
    else:
        add_layer(
            "政策底線防呆",
            "watch",
            0,
            "目前離崩盤前奏線仍遠，未達政策級保護機制啟動條件。",
            "跌破38,575或合理底失守後連續收不回。",
        )

    if fundamental_score is not None and fundamental_score >= 60:
        add_layer(
            "基本面體質防呆",
            "active",
            2,
            "基本面命格仍屬體質尚穩，表示市場不是單靠情緒在硬撐。",
            "AI/半導體、匯率、資金環境或獲利預期同步轉弱。",
        )
    elif fundamental_score is not None and fundamental_score < 45:
        add_layer(
            "基本面體質防呆",
            "warning",
            -2,
            "基本面命格偏弱，價格防線一旦破裂容易延長修正。",
            "基本面壓力與價格破線同時成立。",
        )
    else:
        add_layer(
            "基本面體質防呆",
            "watch",
            0,
            "基本面資料不足或未形成明確支撐，需要外部市場與族群資料補證。",
            "資料補齊後若命格分數續降，保護層降級。",
        )

    score = sum(safe_int(layer.get("score"), 0) for layer in layers)
    failed = [layer for layer in layers if layer["state"] == "failed"]
    warnings = [layer for layer in layers if layer["state"] == "warning"]
    active = [layer for layer in layers if layer["state"] == "active"]

    if failed or score <= -5:
        label = "保護層失靈警戒"
        summary = "市場自我防呆已有失靈風險；需優先核對破線、收盤破與連續收不回。"
        posture = "crisis_watch"
    elif len(warnings) >= 2 or score <= -1:
        label = "保護層偏弱拉鋸"
        summary = "防呆層仍在，但價格、心理或跨盤壓力正在測試承接極限。"
        posture = "defensive_watch"
    elif len(active) >= 3 and score >= 4:
        label = "保護層有效"
        summary = "價格、資金或基本面防線仍有效；目前偏合理壓測，不是崩盤確認。"
        posture = "reasonable_volatility"
    else:
        label = "保護層觀察中"
        summary = "多數保護層尚未失靈，但仍需下一交易日與收盤確認。"
        posture = "watch"

    if close is not None and normal_gen_low is not None and close >= normal_gen_low and risk_value < 50:
        current_judgment = "尚屬高檔合理壓測，但已啟動臨界防守觀察。"
    elif warnings and not failed:
        current_judgment = "不是崩盤確認，是防呆層偏弱的拉鋸狀態。"
    else:
        current_judgment = "需等盤中低點、收盤與連續性確認後再定性。"

    return {
        "framework": "market_protection_layers_v1",
        "label": label,
        "score": score,
        "posture": posture,
        "summary": summary,
        "current_judgment": current_judgment,
        "active_layers": [layer["name"] for layer in active],
        "warning_layers": [layer["name"] for layer in warnings],
        "failed_layers": [layer["name"] for layer in failed],
        "layers": layers,
        "crisis_conditions": [
            "價格防線跌破且收盤收不回。",
            "期現同步破位，逆價差或避險壓力擴大。",
            "權值股、族群廣度、成交量同時惡化。",
            "外部事件重置連續發作且日盤無法吸收。",
            "政策或制度層被迫接手後仍無法止穩。",
        ],
        "duration_rule": "影響期限 = 病灶深度 + 觸發強度 - 日盤吸收力 + 復發次數。",
        "guardrail": "市場防呆只判斷保護層是否失靈；不證明單一主體操控，不產生投資命令。",
        "source_links": {
            "practical_cause": practical.get("framework"),
            "stethoscope": stethoscope.get("framework"),
            "crash_monitor": crash.get("enabled"),
            "close_cause": close_cause.get("framework"),
        },
    }


def analyze_crisis_opportunity_interface(payload: dict) -> dict:
    """Locate the boundary where crisis risk can turn into repair evidence."""
    crash = payload.get("crash_monitor", {})
    protection = payload.get("market_protection_layers", {})
    intraday = payload.get("intraday_tactical_monitor", {})
    practical = payload.get("practical_cause_arbitration", {})
    health = payload.get("market_health", {})
    health_value = health.get("health_value", {})
    stethoscope = payload.get("market_stethoscope", {})
    day_night = payload.get("day_night_variance_pattern", {})
    technical = payload.get("technical_phase", {})
    peak_warning = payload.get("peak_to_valley_warning", {})
    washout = payload.get("washout_pattern", {})
    candle = payload.get("candlestick_pattern", {})

    levels = crash.get("levels", {})
    close = safe_float(crash.get("close"))
    low = safe_float(crash.get("low") or crash.get("effective_low"))
    risk_value = safe_float(crash.get("risk_value")) or 0.0
    health_score = safe_float(crash.get("health_score")) or 0.0
    normal_gen_low = safe_float(levels.get("normal_gen_low"))
    normal_gen_high = safe_float(levels.get("normal_gen_high"))
    normal_bottom_low = safe_float(levels.get("normal_bottom_low"))
    crash_warning = safe_float(levels.get("crash_warning"))
    ma20 = safe_float(technical.get("levels", {}).get("ma20"))
    defense_levels = [x for x in intraday.get("defense_levels", []) if x is not None]
    reclaim_levels = [x for x in intraday.get("reclaim_levels", []) if x is not None]

    crisis_edge: list[str] = []
    opportunity_edge: list[str] = []
    user_actions: list[str] = []
    watch_points: list[str] = []

    if normal_gen_low is not None:
        crisis_edge.append(f"高檔正常艮區下緣 {num(normal_gen_low)}：跌破收不回，代表合理呼吸轉臨界壓測。")
        opportunity_edge.append(f"守住或站回 {num(normal_gen_low)}：代表價格線防呆尚未失靈。")
    if normal_gen_high is not None:
        opportunity_edge.append(f"站回高檔正常艮區上緣 {num(normal_gen_high)}：代表防守轉修復的第一層證據。")
    if ma20 is not None:
        opportunity_edge.append(f"收復20日線 {num(ma20)}：技術修復可信度提高。")
    if normal_bottom_low is not None:
        crisis_edge.append(f"合理底下緣 {num(normal_bottom_low)}：跌破且收不回，進入重大破底核對。")
    if crash_warning is not None:
        crisis_edge.append(f"崩盤前奏線 {num(crash_warning)}：跌破後需啟動盤中破、收盤破、連續收不回三重核對。")
    if defense_levels:
        watch_points.append("防守觀察: " + " / ".join(num(x) for x in defense_levels[:4]))
    if reclaim_levels:
        watch_points.append("轉強觀察: " + " / ".join(num(x) for x in reclaim_levels[:4]))

    peak_valley_signals: list[str] = []
    peak_label = peak_warning.get("label") or peak_warning.get("status") or "NA"
    if peak_label != "NA":
        peak_valley_signals.append(f"峰轉谷預警: {peak_label}。")
    if candle.get("label"):
        peak_valley_signals.append(f"K線轉折: {candle.get('label')}。")
    if washout.get("label"):
        peak_valley_signals.append(f"洗盤/谷底驗證: {washout.get('label')}。")
    if protection.get("warning_layers"):
        peak_valley_signals.append("保護層偏弱，峰轉谷候選需提高權重。")
    if protection.get("active_layers") and not protection.get("failed_layers"):
        peak_valley_signals.append("價格或基本面防呆仍有效，谷轉峰候選保留。")

    if risk_value >= 75 or protection.get("failed_layers"):
        interface_state = "危機端"
        label = "危機端待急症核對"
        summary = "保護層已有失效或風險紅線壓力；此時轉機必須先由收盤收回與連續修復證明。"
    elif protection.get("warning_layers") and close is not None and normal_gen_low is not None and close >= normal_gen_low:
        interface_state = "臨界線"
        label = "危機/轉機交界"
        summary = "市場正在臨界壓測：危機尚未確認，轉機也尚未完全成立，關鍵在守線與收復。"
    elif health_score >= 65 and risk_value < 50:
        interface_state = "轉機端"
        label = "合理壓測轉修復候選"
        summary = "健康分數與價格防線仍支撐市場，轉機候選存在，但需補收盤與族群擴散確認。"
    else:
        interface_state = "觀察線"
        label = "界面觀察"
        summary = "資料尚未形成單邊答案，先以關鍵線與保護層變化判斷。"

    if close is not None and low is not None:
        watch_points.append(f"今日收盤/低點: {num(close)} / {num(low)}")
    watch_points.append(f"健康/風險: {num(health_score)} / {num(risk_value)}")
    watch_points.append(f"防呆狀態: {protection.get('label', 'NA')}，失效層 {len(protection.get('failed_layers', []))}，偏弱層 {len(protection.get('warning_layers', []))}")
    watch_points.append(f"日夜盤: {day_night.get('label', 'NA')}；實務主因: {practical.get('label', 'NA')}")

    user_actions.extend(
        [
            "先標線：把防守線、轉強線、合理底與崩盤線列成固定觀測，不隨情緒移動。",
            "等確認：盤中刺破不等於崩盤，收盤與連續收不回才升級。",
            "分層處理：價格線守住看轉機，價格線破且族群/期現同步惡化看危機。",
            "降雜訊：單一新聞只當觸發，不直接當主因；回到日盤現貨驗證。",
            "寫病歷：每天把預判、實際、錯因、保護層變化留底，隔日驗證同類錯誤是否延續。",
        ]
    )
    if not health_value.get("controllable_risk", True):
        user_actions.append("可控風險已轉弱時，不把反彈直接視為安全，先看是否站回轉強線。")

    return {
        "framework": "crisis_opportunity_interface_v1",
        "label": label,
        "interface_state": interface_state,
        "summary": summary,
        "crisis_edge": crisis_edge,
        "opportunity_edge": opportunity_edge,
        "watch_points": watch_points,
        "peak_valley_signals": unique_text(peak_valley_signals),
        "what_i_can_do": user_actions,
        "zero_one_rule": "0=危機被確認，保護層失效並進入三重核對；1=轉機被確認，守線後站回轉強線並由量能/族群補證。",
        "guardrail": "本段回答危機與轉機的判斷界面及風控作業，不產生買賣命令。",
    }


def analyze_market_heart_rhythm(scored: pd.DataFrame, signal_date: str) -> dict:
    """Diagnose whether daily up/down closes are a healthy heartbeat or disorder."""
    if scored.empty or "date" not in scored or "close" not in scored:
        return {
            "available": False,
            "label": "心律資料不足",
            "summary": "缺少正式收盤資料，不能判斷每日心跳頻率。",
        }
    df = scored.copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = pd.to_datetime(signal_date)
    df = df[df["date"] <= cutoff].sort_values("date").copy()
    if len(df) < 25:
        return {
            "available": False,
            "label": "心律樣本不足",
            "summary": "正式收盤樣本不足，暫不判斷心律。",
        }

    df["ret"] = df["close"].pct_change()
    df["sign"] = df["ret"].apply(lambda x: "up" if x > 0 else ("down" if x < 0 else "flat"))
    df["absret"] = df["ret"].abs()
    if {"high", "low"}.issubset(df.columns):
        df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    else:
        df["range_pct"] = pd.NA
    recent = df.dropna(subset=["ret"]).tail(20).copy()
    medium = df.dropna(subset=["ret"]).tail(60).copy()

    def switch_rate(x: pd.DataFrame) -> float:
        signs = x["sign"].replace("flat", pd.NA).dropna()
        if len(signs) <= 1:
            return 0.0
        return float((signs != signs.shift(1)).iloc[1:].mean())

    def max_streak(signs: pd.Series, target: str) -> int:
        best = 0
        current = 0
        for value in signs:
            if value == target:
                current += 1
                best = max(best, current)
            elif value != "flat":
                current = 0
        return best

    recent_switch = switch_rate(recent)
    medium_switch = switch_rate(medium)
    avg_abs = float(recent["absret"].mean())
    avg_range = float(recent["range_pct"].dropna().mean()) if recent["range_pct"].notna().any() else None
    up_days = int((recent["sign"] == "up").sum())
    down_days = int((recent["sign"] == "down").sum())
    max_down = max_streak(recent["sign"], "down")
    max_up = max_streak(recent["sign"], "up")
    after_down = recent[recent["sign"].shift(1) == "down"]
    down_repair_rate = float((after_down["sign"] == "up").mean()) if len(after_down) else None
    cumulative = float(recent["close"].iloc[-1] / recent["close"].iloc[0] - 1)

    score = 50
    reasons: list[str] = []
    if 0.35 <= recent_switch <= 0.70:
        score += 12
        reasons.append("正負切換落在健康換手區間，像規律心跳。")
    elif recent_switch > 0.70:
        score -= 14
        reasons.append("正負切換過快，容易形成心律不整與短線失真。")
    else:
        score -= 8
        reasons.append("正負切換偏低，需防單邊慣性過強。")
    if avg_abs <= 0.012:
        score += 10
        reasons.append("平均日振幅可控，未見急促失序。")
    elif avg_abs <= 0.020:
        score -= 3
        reasons.append("平均日振幅偏大，仍屬高檔敏感心跳。")
    else:
        score -= 16
        reasons.append("平均日振幅過大，心跳偏急促。")
    if max_down >= 4:
        score -= 22
        reasons.append("連跌過長，需防正常呼吸轉病態。")
    elif max_down >= 3:
        score -= 10
        reasons.append("出現三連跌壓力，心律需提高警戒。")
    if down_repair_rate is not None and down_repair_rate >= 0.60:
        score += 12
        reasons.append("跌後修復率高，代表免疫/承接仍有效。")
    elif down_repair_rate is not None and down_repair_rate < 0.40:
        score -= 12
        reasons.append("跌後修復率低，代表承接心跳轉弱。")
    if cumulative > 0:
        score += 5
        reasons.append("近期累積報酬仍為正，主趨勢心跳尚未轉空。")
    else:
        score -= 5
        reasons.append("近期累積報酬轉負，需防主趨勢降溫。")
    score = max(0, min(100, round(score)))

    clinical_state = "normal"
    clinical_label = "正常心跳"
    clinical_meaning = "正負交錯與振幅仍在可控範圍，屬市場正常生命脈動。"
    if max_down >= 4 or (avg_abs > 0.020 and cumulative < 0) or (
        down_repair_rate is not None and down_repair_rate < 0.35 and cumulative < 0
    ):
        clinical_state = "sick"
        clinical_label = "生病心跳"
        clinical_meaning = "連跌、振幅或修復力惡化，正常呼吸可能轉為病態。"
    elif recent_switch > 0.75 and avg_abs > 0.012:
        clinical_state = "arrhythmia"
        clinical_label = "心律不整"
        clinical_meaning = "正負切換過快且振幅偏大，代表多空互相干擾，短線容易失真。"
    elif cumulative < -0.030 or (down_days >= 12 and (down_repair_rate is None or down_repair_rate < 0.50)):
        clinical_state = "depressed"
        clinical_label = "沮喪心跳"
        clinical_meaning = "跌多漲少或跌後修復偏弱，承接信心下降。"
    elif cumulative > 0.040 and up_days >= 13 and max_down <= 2:
        clinical_state = "excited"
        clinical_label = "興奮心跳"
        clinical_meaning = "上漲日偏多且累積漲幅較大，趨勢強但追價敏感度升高。"
    elif avg_abs <= 0.008 and 0.35 <= recent_switch <= 0.70:
        clinical_state = "expectant"
        clinical_label = "期待心跳"
        clinical_meaning = "振幅收斂且正負切換規律，像等待觸發的蓄勢盤。"

    if score >= 72:
        code = "healthy_rotation_heartbeat"
        label = f"{clinical_label}/健康換手"
        direction_bias = "uptrend_with_wash"
        summary = "正負交錯屬多頭換手心跳，跌後修復仍是主觀察。"
    elif score >= 55:
        code = "sensitive_but_controlled"
        label = f"{clinical_label}/敏感可控"
        direction_bias = "range_with_uptrend_bias"
        summary = "心跳偏敏感但未失序，趨勢要看關鍵線收復。"
    elif max_down >= 3 or score < 40:
        code = "arrhythmia_pressure"
        label = f"{clinical_label}/壓力升高"
        direction_bias = "risk_watch"
        summary = "正負或振幅結構轉亂，需防測壓變成壓力確認。"
    else:
        code = "unclear_rhythm"
        label = f"{clinical_label}/待觀察"
        direction_bias = "neutral"
        summary = "心跳沒有明確健康或惡化優勢，等待下一收盤確認。"

    return {
        "available": True,
        "framework": "market_heart_rhythm_v1",
        "signal_date": df["date"].iloc[-1].strftime("%Y-%m-%d"),
        "code": code,
        "label": label,
        "clinical_state": clinical_state,
        "clinical_label": clinical_label,
        "clinical_meaning": clinical_meaning,
        "score": score,
        "direction_bias": direction_bias,
        "summary": summary,
        "recent_days": int(len(recent)),
        "up_days": up_days,
        "down_days": down_days,
        "switch_rate": recent_switch,
        "medium_switch_rate": medium_switch,
        "avg_abs_return": avg_abs,
        "avg_range": avg_range,
        "max_down_streak": max_down,
        "max_up_streak": max_up,
        "down_repair_rate": down_repair_rate,
        "cumulative_return": cumulative,
        "reasons": unique_text(reasons),
        "rule": "心跳看每日收盤正負與振幅是否規律；跌後能修復是健康心跳，連跌收不回是異常心律。",
        "guardrail": "心律只判斷市場節奏健康度，不單獨產生買賣命令。",
    }


def analyze_market_stethoscope(
    scored: pd.DataFrame,
    signal_date: str,
    factor_root: Path | None,
    heart: dict,
    news: dict,
) -> dict:
    """Combine vital signs into one clinical-style market diagnosis."""
    factor_root = factor_root or (ROOT / "data" / "processed" / "factors")
    signal_ts = pd.to_datetime(signal_date)
    close = None
    if not scored.empty and "date" in scored and "close" in scored:
        history = scored.copy()
        history["date"] = pd.to_datetime(history["date"])
        eligible = history[history["date"] <= signal_ts].sort_values("date")
        if not eligible.empty:
            close = safe_float(eligible.iloc[-1].get("close"))

    early = latest_factor_snapshot(factor_root / "taiex_early_pulse.csv", signal_date)
    breadth = latest_factor_snapshot(factor_root / "cross_sectional_breadth.csv", signal_date)
    futures = latest_factor_snapshot(factor_root / "futures_daily.csv", signal_date)
    options = latest_factor_snapshot(factor_root / "option_daily.csv", signal_date)
    cap = latest_factor_snapshot(factor_root / "market_cap_structure.csv", signal_date)

    diagnostics: list[dict] = []
    missing: list[str] = []
    total_score = 0

    def add(name: str, label: str, score: int, summary: str, severity: str = "normal") -> None:
        nonlocal total_score
        total_score += score
        diagnostics.append(
            {
                "name": name,
                "label": label,
                "score": score,
                "summary": summary,
                "severity": severity,
            }
        )

    def fresh(snapshot: dict, name: str, max_age_days: int = 3) -> bool:
        source = snapshot.get("_source_date") or snapshot.get("date") or snapshot.get("signal_date") or snapshot.get("calendar_date")
        if not snapshot or not source:
            missing.append(name)
            return False
        age = (signal_ts - pd.to_datetime(source)).days
        if age > max_age_days:
            missing.append(f"{name}過期：{source}")
            return False
        return True

    heart_score = safe_int(heart.get("score"), 50)
    if heart_score >= 72:
        add("心跳", heart.get("clinical_label", "健康心跳"), 2, heart.get("summary", ""), "normal")
    elif heart_score >= 55:
        add("心跳", heart.get("clinical_label", "敏感心跳"), 0, heart.get("summary", ""), "watch")
    else:
        add("心跳", heart.get("clinical_label", "心律偏弱"), -2, heart.get("summary", ""), "warning")

    early_return = safe_float(early.get("early_return"))
    early_eff = safe_float(early.get("trend_efficiency"))
    if fresh(early, "早盤15分鐘心電圖"):
        if early_return is not None and early_return >= 0.004 and (early_eff is None or early_eff >= 0.12):
            add("心電圖", "早盤脈衝偏多", 2, "早盤15分鐘價格脈衝偏強，代表開局買盤較主動。", "normal")
        elif early_return is not None and early_return <= -0.004 and (early_eff is None or early_eff >= 0.12):
            add("心電圖", "早盤脈衝偏空", -2, "早盤15分鐘價格脈衝偏弱，代表開局賣壓較主動。", "warning")
        else:
            add("心電圖", "早盤脈衝中性", 0, "早盤15分鐘未形成強單邊，仍需收盤驗證。", "watch")

    advance = safe_float(breadth.get("price_advancing_fraction"))
    up_volume = safe_float(breadth.get("price_up_volume_fraction"))
    eq_ret = safe_float(breadth.get("price_equal_weight_return"))
    if fresh(breadth, "族群廣度血氧"):
        if advance is not None and up_volume is not None and advance >= 0.55 and up_volume >= 0.55:
            add("血氧", "族群供氧良好", 2, "上漲家數與上漲成交量同步擴散，指數健康度較高。", "normal")
        elif advance is not None and up_volume is not None and (advance < 0.42 or up_volume < 0.45):
            add("血氧", "族群供氧不足", -2, "上漲家數或上漲量能不足，指數可能靠少數權值撐住。", "warning")
        elif eq_ret is not None and eq_ret < -0.003:
            add("血氧", "等權轉弱", -1, "非權值股承壓，需防假健康。", "watch")
        else:
            add("血氧", "族群供氧普通", 0, "族群廣度未給出強烈多空訊號。", "watch")

    oi = safe_float(futures.get("open_interest"))
    fut_close = safe_float(futures.get("close"))
    spread_per = safe_float(futures.get("spread_per"))
    if fresh(futures, "期貨未平倉/期現差"):
        basis = (fut_close / close - 1) if fut_close is not None and close else None
        if basis is not None and basis >= 0.004:
            add("期貨脈壓", "期貨領先偏多", 1, "期貨相對現貨有溢價，資金預期偏向支撐。", "normal")
        elif basis is not None and basis <= -0.004:
            add("期貨脈壓", "期貨領先偏空", -1, "期貨相對現貨折價，避險或壓低預期較重。", "watch")
        else:
            add("期貨脈壓", "期現差中性", 0, "期貨與現貨未明顯偏離。", "watch")

    call_oi = safe_float(options.get("call_open_interest"))
    put_oi = safe_float(options.get("put_open_interest"))
    if fresh(options, "選擇權血壓") and call_oi and put_oi:
        pc_oi = put_oi / call_oi
        if pc_oi >= 1.45:
            add("血壓", "避險血壓偏高", -1, f"Put/Call OI 約 {pc_oi:.2f}，避險需求較高。", "watch")
        elif pc_oi <= 0.85:
            add("血壓", "追價血壓偏熱", -1, f"Put/Call OI 約 {pc_oi:.2f}，需防樂觀過熱。", "watch")
        else:
            add("血壓", "選擇權血壓正常", 1, f"Put/Call OI 約 {pc_oi:.2f}，避險與追價相對平衡。", "normal")

    top1 = safe_float(cap.get("market_cap_top1_share"))
    top10 = safe_float(cap.get("market_cap_top10_share"))
    if fresh(cap, "權值集中度", max_age_days=45):
        if top1 is not None and top1 >= 0.28:
            add("權值集中", "單一權值依賴偏高", -2, "指數過度依賴最大權值股，需防一股轉弱拖累全盤。", "warning")
        elif top10 is not None and top10 >= 0.55:
            add("權值集中", "權值集中偏高", -1, "前十大市值集中度偏高，指數健康需看權值同步承接。", "watch")
        else:
            add("權值集中", "權值集中可控", 1, "市值集中未達過熱警戒。", "normal")

    net_news = safe_int(news.get("net_risk_score"), 0)
    if net_news >= 3:
        add("事件日曆", "外部觸發偏空", -2, "重大新聞/政策/利率事件偏空，會按下走弱觸發鈕。", "warning")
    elif net_news <= -2:
        add("事件日曆", "外部觸發偏多", 2, "重大外部事件偏多，會按下修復觸發鈕。", "normal")
    elif news.get("available"):
        add("事件日曆", "外部觸發混合", 0, "新聞面多空互抵，不能單獨定方向。", "watch")
    else:
        missing.append("國際事件日曆")

    if total_score >= 5:
        label = "生命徵象偏健康"
        result_bias = "repair_or_uptrend"
        summary = "多數生命徵象支持承接，跌後修復機率較高。"
    elif total_score <= -5:
        label = "生命徵象轉弱"
        result_bias = "pressure_confirm_watch"
        summary = "多項生命徵象同步轉弱，需防測壓變成壓力確認。"
    else:
        label = "生命徵象混合"
        result_bias = "wait_for_cash_confirmation"
        summary = "生命徵象多空交錯，結果仍需日盤關鍵線與收盤確認。"

    return {
        "available": True,
        "framework": "market_stethoscope_v1",
        "signal_date": signal_date,
        "label": label,
        "score": total_score,
        "result_bias": result_bias,
        "summary": summary,
        "diagnostics": diagnostics,
        "missing": missing,
        "rule": "心跳看節奏，心電圖看早盤脈衝，血氧看族群廣度，血壓看選擇權避險，期貨脈壓看期現差，事件日曆看觸發鈕。",
        "guardrail": "聽診器只整合生命徵象，提高風險診斷品質；缺資料時降權，不產生投資命令。",
    }


def analyze_cross_market_entanglement(payload: dict) -> dict:
    """Classify US/US futures/TX night/cash TWII cross-market confirmation."""
    premarket = payload.get("premarket", {})
    intraday = payload.get("intraday_tactical_monitor", {})
    practical = payload.get("practical_cause_arbitration", {})
    external = payload.get("external_event_reset_monitor", {})
    score = 0
    evidence: list[str] = []
    missing: list[str] = []

    def direction(value, threshold=0.002):
        v = safe_float(value)
        if v is None:
            return "missing"
        if v >= threshold:
            return "up"
        if v <= -threshold:
            return "down"
        return "flat"

    nasdaq_dir = direction(premarket.get("nasdaq_return_1d"), 0.003)
    sox_dir = direction(premarket.get("sox_return_1d"), 0.004)
    tsm_dir = direction(premarket.get("tsm_adr_return_1d"), 0.004)
    night_dir = direction(premarket.get("tx_night_spread_per"), 0.003)
    cash_code = " ".join(
        str(intraday.get(key) or "")
        for key in ("code", "label", "summary", "headline")
    ).strip()
    cash_dir = "missing"
    if cash_code:
        text = str(cash_code)
        if any(k in text for k in ["破位", "偏空", "跌", "壓力"]):
            cash_dir = "down"
        elif any(k in text for k in ["修復", "轉強", "站回", "偏多"]):
            cash_dir = "up"
        elif "待" in text:
            cash_dir = "pending"
        else:
            cash_dir = "flat"
    else:
        missing.append("台股日盤即時/收盤確認")

    risk_dirs = [nasdaq_dir, sox_dir, tsm_dir]
    if "missing" in risk_dirs:
        missing.append("部分美股/ADR資料")
    us_down = sum(1 for x in risk_dirs if x == "down")
    us_up = sum(1 for x in risk_dirs if x == "up")
    if us_down >= 2:
        score -= 2
        evidence.append("美股科技鏈偏空，美盤對台股風險胃納形成壓力。")
    elif us_up >= 2:
        score += 2
        evidence.append("美股科技鏈偏多，美盤對台股風險胃納形成支撐。")
    else:
        evidence.append("美股科技鏈分歧，外部主市場未給單邊命令。")

    if night_dir == "down":
        score -= 1
        evidence.append("台指夜盤先行偏空，台盤開局容易被迫測壓。")
    elif night_dir == "up":
        score += 1
        evidence.append("台指夜盤先行偏多，台盤開局偏向修復。")
    else:
        evidence.append("台指夜盤接近中性，日盤自主判斷權提高。")

    if cash_dir == "down":
        score -= 2
        evidence.append("台股日盤承認壓力，夜盤或外部訊號被現貨確認。")
    elif cash_dir == "up":
        score += 2
        evidence.append("台股日盤反向修復，夜盤或外部壓力被現貨吸收。")
    elif cash_dir == "pending":
        evidence.append("台股日盤尚待驗證，不能把夜盤直接當結果。")

    if external.get("reset_active"):
        score += -2 if external.get("direction") == "bearish" else 2
        evidence.append("外部事件重置啟動，跨盤優先權提高。")

    if cash_dir == "up" and (night_dir == "down" or us_down >= 2):
        code = "cash_repair_against_external"
        label = "台盤反向修復"
        summary = "美盤/夜盤壓力被台股現貨吸收，日盤取得裁判權。"
    elif cash_dir == "down" and (night_dir == "down" or us_down >= 2):
        code = "cross_market_bearish_confirmation"
        label = "跨盤偏空確認"
        summary = "美盤、夜盤與台盤同向偏弱，壓力被確認。"
    elif cash_dir == "pending":
        code = "cash_validation_pending"
        label = "日盤待裁判"
        summary = "美盤與夜盤只提供前哨，正式結果仍需台股現貨確認。"
    elif us_up >= 2 and night_dir == "down":
        code = "night_decouples_from_us"
        label = "夜盤背離美盤"
        summary = "美盤偏多但夜盤偏空，需查避險、結算或籌碼壓力。"
    elif us_down >= 2 and night_dir == "up":
        code = "night_repair_against_us"
        label = "夜盤反向修復"
        summary = "美盤偏空但夜盤偏多，台股可能先押日盤修復。"
    else:
        code = "cross_market_mixed"
        label = "跨盤混合糾結"
        summary = "美指、美盤、台指與台盤未形成同向共振，需降方向信心。"

    return {
        "available": True,
        "framework": "cross_market_entanglement_v1",
        "code": code,
        "label": label,
        "score": score,
        "summary": summary,
        "directions": {
            "nasdaq": nasdaq_dir,
            "sox": sox_dir,
            "tsm_adr": tsm_dir,
            "tx_night": night_dir,
            "twii_cash": cash_dir,
        },
        "evidence": unique_text(evidence),
        "missing": unique_text(missing),
        "model_link": practical.get("causality_rule", "先由跨盤前哨形成觸發，再由台股日盤確認果。"),
        "rule": "美盤是外部主市場，台指夜盤是前哨，台股日盤是現貨裁判；同向才提高信心，背離就降權等待。",
        "guardrail": "跨盤糾結只判斷訊號主從與確認程度，不指認操盤者，也不產生投資命令。",
    }


def analyze_master_arbitration(payload: dict) -> dict:
    """Resolve conflicts among constitution, health, external shock and tactical layers."""
    health = payload.get("market_health", {})
    constitution = payload.get("fundamental_constitution", {})
    practical = payload.get("practical_cause_arbitration", {})
    external = payload.get("external_event_reset_monitor", {})
    reliability = payload.get("direction_reliability_policy", {})
    sector = payload.get("sector_pressure_observation", {})
    psychological = payload.get("psychological_warfare_pattern", {})
    situation = payload.get("situation_psychology_context", {})
    premarket = payload.get("premarket", {})
    crash = payload.get("crash_monitor", {})

    health_score = safe_int(health.get("score") or crash.get("health_score"), 50)
    risk_score = safe_int(health.get("risk_score") or crash.get("risk_score"), 50)
    constitution_score = safe_int(constitution.get("score"), 50)
    external_code = external.get("code", "unknown")
    external_bearish = external.get("direction") == "bearish" and external.get("reset_active")
    external_bullish = external.get("direction") == "bullish" and external.get("reset_active")
    sector_risk = safe_int(sector.get("risk_score"), 0)
    reliability_enabled = bool(
        reliability.get("main_multi_day_direction", {}).get("enabled")
        if "main_multi_day_direction" in reliability
        else reliability.get("direction_enabled")
    )

    conflicts: list[str] = []
    validations: list[str] = []
    if health_score >= 70 and external_bearish:
        conflicts.append("健康分數偏高，但外部事件已轉空，短線須先防外感衝擊。")
    if constitution_score >= 62 and external_bearish:
        conflicts.append("基本面命格尚穩，但國際消息/夜盤壓力可暫時壓低估值。")
    if sector_risk >= 3:
        conflicts.append("族群分化壓力偏高，指數上漲不能自動視為全面健康。")
    situation_score = safe_int(situation.get("score"), 0)
    if situation_score <= -3:
        conflicts.append("新聞面/生活心理面偏防守，技術多方訊號需等日盤驗證。")
    elif situation_score >= 3:
        conflicts.append("新聞面/生活心理面偏有利，但仍不得取代現貨收盤確認。")
    if not reliability_enabled:
        conflicts.append("多日方向正式模型停用，方向預判只能作研究與風控參考。")
    if practical.get("code") == "internal_disease_external_trigger":
        conflicts.append("實務歸因顯示內部高檔病灶才是主因，外部新聞只是今日觸發開關。")
    basis = premarket.get("night_basis_audit", {})
    if basis.get("basis_conflict"):
        conflicts.append("夜盤官方漲跌與開收報酬差異明顯，需分基準解讀。")

    if risk_score >= 75 or crash.get("alert_code") in {"crash_warning", "bottom_failed_watch"}:
        code = "systemic_risk_priority"
        headline = "風控優先"
        dominant_layer = "crash_risk"
        risk_posture = "紅色警戒核對"
        decision = "先核對盤中破、收盤破、連續收不回；未排除前，所有進攻劇本降權。"
    elif external_bearish and health_score >= 65:
        code = "healthy_body_external_shock"
        headline = "體質尚穩但外部風寒"
        dominant_layer = "external_reset"
        risk_posture = "防守觀察"
        decision = "外部壓力優先於短線樂觀；等日盤量價收回、族群廣度未惡化後才恢復進攻解讀。"
    elif external_bullish and constitution_score >= 60:
        code = "external_tailwind_requires_cash_confirm"
        headline = "外部順風但需現貨確認"
        dominant_layer = "cash_validation"
        risk_posture = "有利但不追認"
        decision = "外部利多只給開盤與情緒加分，必須由日盤站回關鍵防線與量價確認。"
    elif sector_risk >= 3:
        code = "sector_breadth_warning"
        headline = "指數與族群分化"
        dominant_layer = "breadth"
        risk_posture = "降低追價"
        decision = "權值撐盤不能掩蓋族群退潮；觀察弱族群是否隔日收復。"
    elif practical.get("practical_primary") == "internal_structure" and practical.get("internal_structure_score", 0) >= 5:
        code = "practical_internal_structure_priority"
        headline = "內部病灶優先"
        dominant_layer = "internal_structure"
        risk_posture = "實務高檔測壓"
        decision = "先把高檔估值、前高套牢、追價籌碼、權值過熱與換手壓力列為主因；外部消息只作觸發器驗證。"
    elif health_score >= 70 and constitution_score >= 62:
        code = "constitution_supports_tactical_monitor"
        headline = "體質支撐但仍逐日驗證"
        dominant_layer = "constitution_health"
        risk_posture = "可控觀察"
        decision = "維持主波段觀察，以夜日盤與收盤位置逐日校正，不升級成投資命令。"
    else:
        code = "mixed_wait_for_confirmation"
        headline = "訊號混合等待確認"
        dominant_layer = "cash_close"
        risk_posture = "中性偏審慎"
        decision = "等下一個日盤收盤、量能與族群廣度給出確認，避免用單一訊號硬判方向。"

    validations.extend(
        [
            "日盤是否守住前一交易日低點與夜盤低點。",
            "收盤是否站回夜盤收盤/開盤與關鍵整數關卡。",
            "電子、半導體、軍工與金融是否同步擴散或輪動修復。",
            "外部新聞風險是否被台指夜盤與日盤現貨吸收。",
        ]
    )
    validations.extend(top_items(situation.get("next_validation", []), 2))
    return {
        "framework": "master_arbitration_v1",
        "code": code,
        "headline": headline,
        "dominant_layer": dominant_layer,
        "risk_posture": risk_posture,
        "decision": decision,
        "health_score": health_score,
        "risk_score": risk_score,
        "constitution_score": constitution_score,
        "external_reset_code": external_code,
        "sector_risk_score": sector_risk,
        "situation_score": situation_score,
        "psychological_score": psychological.get("score"),
        "conflicts": unique_text(conflicts),
        "validation": validations,
        "summary": f"{headline}：{decision}",
        "guardrail": "總仲裁只決定訊號優先序與風控語氣；不得輸出買賣命令、槓桿建議或回測回填式結論。",
    }


def score_night_row(row) -> int:
    if row is None:
        return 0
    # The canonical night spread is validated for the same-day opening gap,
    # not for cash-close or multi-day direction. Keep the broad forecast
    # coefficient at zero and report the scoped gap signal separately.
    return 0


def analyze_cause(scored: pd.DataFrame, signal_date: str) -> dict:
    row = scored[scored["date"] == pd.to_datetime(signal_date)].iloc[-1]
    external_score = int(row.get("external_score", 0))
    night_score = int(row.get("night_futures_score", 0))
    panic_score = int(row.get("panic_reversal_score", 0))
    close_ret = safe_float(row.get("close_return_pct"))
    recovery = safe_float(row.get("close_recovery_ratio"))
    strong_cash_close = (
        (close_ret is not None and close_ret >= 0.003)
        and (recovery is not None and recovery >= 0.65)
    )
    truth = "neutral"
    if (external_score < 0 or night_score < 0) and (panic_score > 0 or strong_cash_close):
        truth = "external_shock_absorbed"
    elif (external_score < 0 or night_score < 0) and panic_score <= 0:
        truth = "external_pressure_not_absorbed"
    elif panic_score >= 2:
        truth = "domestic_buying_absorbed_panic"
    scenario_data = market_scenario(row, truth, external_score, night_score, panic_score)
    return {
        "signal_date": signal_date,
        "truth_label": truth,
        "truth_text": truth_text(truth),
        "plain_summary": plain_summary(truth, external_score, night_score, panic_score),
        "scenario": scenario_data,
        "external_score": external_score,
        "external_pressure_label": row.get("external_pressure_label", "unknown"),
        "external_pressure_text": external_pressure_text(row.get("external_pressure_label", "unknown")),
        "night_futures_score": night_score,
        "night_futures_label": row.get("night_futures_label", "unknown"),
        "night_futures_text": night_futures_text(row.get("night_futures_label", "unknown")),
        "intraday_truth_label": row.get("intraday_truth_label", "normal"),
        "intraday_truth_text": intraday_truth_text(row.get("intraday_truth_label", "normal")),
        "panic_reversal_score": panic_score,
        "open_gap_pct": safe_float(row.get("open_gap_pct")),
        "intraday_low_pct": safe_float(row.get("intraday_low_pct")),
        "close_return_pct": safe_float(row.get("close_return_pct")),
        "close_recovery_ratio": safe_float(row.get("close_recovery_ratio")),
        "nasdaq_return_1d": safe_float(row.get("nasdaq_return_1d")),
        "sox_return_1d": safe_float(row.get("sox_return_1d")),
        "sp500_return_1d": safe_float(row.get("sp500_return_1d")),
        "tsm_adr_return_1d": safe_float(row.get("tsm_adr_return_1d")),
        "vix_return_1d": safe_float(row.get("vix_return_1d")),
        "tx_night_return": safe_float(row.get("tx_night_return")),
        "tx_night_gap_vs_spot": safe_float(row.get("tx_night_gap_vs_spot")),
        "reasons": cause_reasons(external_score, night_score, panic_score),
    }


def memory_risk_snapshot(scored: pd.DataFrame, signal_date: str) -> dict:
    eligible = scored[scored["date"] <= pd.to_datetime(signal_date)]
    row = eligible.iloc[-1]
    return {
        "points": int(row.get("memory_risk_points", 0)),
        "score": int(row.get("memory_risk_score", 0)),
        "level": row.get("memory_risk_level", "low"),
        "market_confirmed": bool(row.get("memory_market_confirmed", False)),
        "reasons": row.get("memory_risk_reasons", ""),
        "event_severity": int(row.get("memory_event_severity", 0)),
        "event_titles": row.get("memory_event_titles", ""),
        "basket_return_1d": safe_float(row.get("memory_basket_return_1d")),
        "basket_return_5d": safe_float(row.get("memory_basket_return_5d")),
        "constituent_count": int(row.get("memory_constituent_count", 0)),
        "relative_sox_5d": safe_float(row.get("memory_relative_sox_5d")),
        "base_risk_regime": row.get("base_risk_regime", row.get("risk_regime")),
        "adjusted_risk_regime": row.get("risk_regime"),
        "base_confidence": safe_float(row.get("base_confidence")),
        "adjusted_confidence": safe_float(row.get("confidence")),
    }


def capital_flow_snapshot(scored: pd.DataFrame, signal_date: str, factor_dir: str | None) -> dict:
    eligible = scored[scored["date"] <= pd.to_datetime(signal_date)].sort_values("date")
    row = eligible.iloc[-1]
    concrete_columns = [
        "inst_net_5d",
        "inst_net_20d",
        "margin_change_20d",
        "short_change_20d",
        "margin_short_ratio",
        "futures_inst_net_5d",
        "futures_inst_net_20d",
        "option_put_call_20d_z",
        "option_vix_20d_z",
    ]
    has_values = any(col in row.index and pd.notna(row.get(col)) for col in concrete_columns)
    factor_path = rooted_path(factor_dir) if factor_dir else None
    if not factor_dir:
        status_text = "未接入"
        plain = "法人籌碼、融資融券、期貨與選擇權資料尚未由正式資料源自動接入，本次綜合判斷會降低籌碼權重。"
    elif not factor_path.exists():
        status_text = "路徑不存在"
        plain = "指定的籌碼資料夾不存在，本次無法使用法人與衍生性商品資料。"
    elif not has_values:
        status_text = "已接入但當日無有效值"
        plain = "已有籌碼資料夾，但目前使用交易日沒有可用數值；可能是資料尚未更新或欄位格式未對上。"
    else:
        status_text = "已接入"
        plain = capital_flow_plain_text(row)

    chip_score = safe_int(row.get("chip_score"), 0) if "chip_score" in row.index else 0
    derivative_score = safe_int(row.get("derivative_score"), 0) if "derivative_score" in row.index else 0
    non_price_score = safe_int(row.get("non_price_score"), chip_score + derivative_score) if "non_price_score" in row.index else chip_score + derivative_score
    heuristic_chip_score = safe_int(row.get("heuristic_chip_score"), 0) if "heuristic_chip_score" in row.index else 0
    heuristic_derivative_score = safe_int(row.get("heuristic_derivative_score"), 0) if "heuristic_derivative_score" in row.index else 0
    return {
        "signal_date": str(row["date"].date()),
        "factor_dir": str(factor_path) if factor_path else None,
        "data_status": status_text,
        "has_factor_values": has_values,
        "plain_summary": plain,
        "chip_score": chip_score,
        "derivative_score": derivative_score,
        "non_price_score": non_price_score,
        "heuristic_chip_score": heuristic_chip_score,
        "heuristic_derivative_score": heuristic_derivative_score,
        "directional_validation": "failed_multi_day_gate_coefficients_zero",
        "institutional": {
            "net_5d": safe_float(row.get("inst_net_5d")) if "inst_net_5d" in row.index else None,
            "net_20d": safe_float(row.get("inst_net_20d")) if "inst_net_20d" in row.index else None,
        },
        "margin": {
            "margin_change_20d": safe_float(row.get("margin_change_20d")) if "margin_change_20d" in row.index else None,
            "short_change_20d": safe_float(row.get("short_change_20d")) if "short_change_20d" in row.index else None,
            "margin_short_ratio": safe_float(row.get("margin_short_ratio")) if "margin_short_ratio" in row.index else None,
        },
        "derivatives": {
            "futures_inst_net_5d": safe_float(row.get("futures_inst_net_5d")) if "futures_inst_net_5d" in row.index else None,
            "futures_inst_net_20d": safe_float(row.get("futures_inst_net_20d")) if "futures_inst_net_20d" in row.index else None,
            "option_put_call_20d_z": safe_float(row.get("option_put_call_20d_z")) if "option_put_call_20d_z" in row.index else None,
            "option_vix_20d_z": safe_float(row.get("option_vix_20d_z")) if "option_vix_20d_z" in row.index else None,
        },
    }


def capital_flow_plain_text(row) -> str:
    chip_score = safe_int(row.get("chip_score"), 0)
    derivative_score = safe_int(row.get("derivative_score"), 0)
    non_price_score = safe_int(row.get("non_price_score"), chip_score + derivative_score)
    if non_price_score >= 2:
        return "法人籌碼與衍生性商品整體偏多，代表資金面有支撐。"
    if non_price_score <= -2:
        return "法人籌碼與衍生性商品整體偏空，代表資金面仍有賣壓或避險需求。"
    if chip_score > 0 and derivative_score < 0:
        return "法人現貨偏多但期權偏空，代表現貨買盤與避險部位不同步，需要觀察是否收斂。"
    if chip_score < 0 and derivative_score > 0:
        return "法人現貨偏空但期權偏多，可能有空單回補或避險降溫，但尚未形成全面偏多。"
    return "法人籌碼與期權訊號接近中性，暫時不能單獨作為方向依據。"


def analyze_programmed_pressure_pattern(
    scored: pd.DataFrame,
    forecast_date: str,
    manual_close: float | None,
    night_futures_path: Path,
    factor_root: Path | None,
    start_date: str = "2026-08-01",
) -> dict:
    base = {
        "enabled": True,
        "name": "連續壓低換手行為指紋",
        "start_date": start_date,
        "forecast_date": forecast_date,
        "guardrail": "公開資料只能辨識行為指紋與機率，不能指認單一資金或證明操盤。",
    }
    frame = scored.copy()
    if frame.empty or "date" not in frame:
        return {**base, "available": False, "label": "資料不足", "summary": "缺少日線資料，無法建立行為序列。"}
    frame["date"] = pd.to_datetime(frame["date"])
    price_cols = ["date", "open", "high", "low", "close", "volume"]
    frame = frame[[col for col in price_cols if col in frame.columns]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    target = pd.to_datetime(forecast_date)
    if manual_close is not None:
        existing = frame[frame["date"] == target]
        if existing.empty:
            reference = frame[frame["date"] < target].sort_values("date").tail(1)
            prev_close = safe_float(reference.iloc[-1].get("close")) if not reference.empty else manual_close
            manual_row = {
                "date": target,
                "open": manual_close,
                "high": manual_close,
                "low": manual_close,
                "close": manual_close,
                "volume": None,
            }
            frame = pd.concat([frame, pd.DataFrame([manual_row])], ignore_index=True)
        else:
            idx = existing.index[-1]
            frame.loc[idx, "close"] = manual_close
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    frame["prev_close"] = frame["close"].shift(1)
    frame["prev_low"] = frame["low"].shift(1)
    frame["ret"] = frame["close"] / frame["prev_close"] - 1
    frame["gap"] = frame["open"] / frame["prev_close"] - 1
    frame["intraday"] = frame["close"] / frame["open"] - 1
    spread = (frame["high"] - frame["low"]).replace(0, pd.NA)
    frame["close_position"] = (frame["close"] - frame["low"]) / spread
    frame["break_prev_low"] = frame["low"] < frame["prev_low"]
    frame["reclaim_prev_close"] = frame["close"] >= frame["prev_close"]

    night = pd.DataFrame()
    if night_futures_path.exists():
        try:
            night = pd.read_csv(night_futures_path)
            if "signal_date" in night:
                night["date"] = pd.to_datetime(night["signal_date"])
                night = night[["date", "tx_night_spread_per", "tx_night_close", "tx_night_low"]]
        except (OSError, ValueError, pd.errors.ParserError):
            night = pd.DataFrame()
    if not night.empty:
        frame = frame.merge(night, on="date", how="left")
    else:
        frame["tx_night_spread_per"] = pd.NA

    breadth = read_optional_factor(factor_root, "cross_sectional_breadth.csv")
    if not breadth.empty and "date" in breadth:
        breadth["date"] = pd.to_datetime(breadth["date"])
        keep = [
            "date",
            "price_advancing_fraction",
            "price_declining_fraction",
            "price_advance_decline_breadth",
            "inst_combined_breadth",
        ]
        frame = frame.merge(breadth[[c for c in keep if c in breadth.columns]], on="date", how="left")
    if "price_declining_fraction" not in frame:
        frame["price_declining_fraction"] = pd.NA

    institutional = read_optional_factor(factor_root, "institutional_total.csv")
    if not institutional.empty and {"date", "name", "buy", "sell"}.issubset(institutional.columns):
        institutional["date"] = pd.to_datetime(institutional["date"])
        institutional["net_billion"] = (
            pd.to_numeric(institutional["buy"], errors="coerce")
            - pd.to_numeric(institutional["sell"], errors="coerce")
        ) / 1_000_000_000
        inst_pivot = institutional.pivot_table(index="date", columns="name", values="net_billion", aggfunc="sum").reset_index()
        frame = frame.merge(inst_pivot, on="date", how="left")

    margin = read_optional_factor(factor_root, "margin_total.csv")
    if not margin.empty and {"date", "name", "TodayBalance", "YesBalance"}.issubset(margin.columns):
        margin["date"] = pd.to_datetime(margin["date"])
        margin = margin[margin["name"].eq("MarginPurchase")].copy()
        margin["margin_balance_change"] = (
            pd.to_numeric(margin["TodayBalance"], errors="coerce")
            - pd.to_numeric(margin["YesBalance"], errors="coerce")
        )
        frame = frame.merge(margin[["date", "margin_balance_change"]], on="date", how="left")
    if "margin_balance_change" not in frame:
        frame["margin_balance_change"] = pd.NA

    frame = frame[frame["date"] >= pd.to_datetime(start_date)].copy()
    if frame.empty:
        return {**base, "available": False, "label": "資料不足", "summary": "指定期間內沒有資料。"}

    rows = []
    for _, row in frame.iterrows():
        score = 0
        reasons = []
        def add(condition: bool, reason: str):
            nonlocal score
            if condition:
                score += 1
                reasons.append(reason)

        ret = safe_float(row.get("ret"))
        close_pos = safe_float(row.get("close_position"))
        gap = safe_float(row.get("gap"))
        intraday = safe_float(row.get("intraday"))
        declining = safe_float(row.get("price_declining_fraction"))
        night_spread = safe_float(row.get("tx_night_spread_per"))
        foreign_net = safe_float(row.get("Foreign_Investor"))
        dealer_hedge = safe_float(row.get("Dealer_Hedging"))
        margin_change = safe_float(row.get("margin_balance_change"))

        add(ret is not None and ret < 0, "收黑")
        add(close_pos is not None and close_pos < 0.35, "收低位")
        add(gap is not None and gap > 0 and intraday is not None and intraday < 0, "開高壓回")
        add(bool(row.get("break_prev_low")), "破前低")
        add(declining is not None and declining > 0.55, "廣度賣壓")
        add(night_spread is not None and night_spread < 0, "夜盤先壓")
        add(foreign_net is not None and foreign_net < 0, "外資現貨賣")
        add(dealer_hedge is not None and dealer_hedge < -5, "自營避險賣")
        add(margin_change is not None and margin_change > 15000, "融資硬接")

        if score >= 6:
            label = "強連續壓低"
        elif score >= 4:
            label = "壓低換手"
        elif score >= 2:
            label = "一般測壓"
        else:
            label = "無明顯壓低"
        rows.append(
            {
                "date": str(row["date"].date()),
                "close": safe_float(row.get("close")),
                "return": ret,
                "night_spread": night_spread,
                "close_position": close_pos,
                "break_prev_low": bool(row.get("break_prev_low")),
                "reclaim_prev_close": bool(row.get("reclaim_prev_close")),
                "foreign_net_billion": foreign_net,
                "trust_net_billion": safe_float(row.get("Investment_Trust")),
                "dealer_hedge_net_billion": dealer_hedge,
                "margin_balance_change": margin_change,
                "score": score,
                "label": label,
                "reasons": reasons,
            }
        )
    current = rows[-1]
    recent = rows[-5:]
    recent_scores = [item["score"] for item in recent]
    hot_dates = [item for item in rows if item["score"] >= 4]
    consecutive_pressure = 0
    for item in reversed(rows):
        if item["score"] >= 4:
            consecutive_pressure += 1
        else:
            break

    current_score = current["score"]
    if consecutive_pressure >= 2 and current_score >= 4:
        code, label = "programmed_pressure_sequence", "連續壓低換手"
        summary = "公開資料行為指紋顯示壓低訊號連續出現，需把隔日破低收不回列為優先風控驗證。"
        prediction_adjustment = -2
    elif current_score >= 4:
        code, label = "single_day_pressure", "單日壓低測壓"
        summary = "今日符合壓低換手條件，但尚未形成連續惡化；隔日需看是否破低或收回昨收。"
        prediction_adjustment = -1
    elif sum(score >= 4 for score in recent_scores) >= 2:
        code, label = "recent_pressure_cluster", "近期壓低群聚"
        summary = "近五日多次出現壓低換手，代表盤勢不是隨機震盪，需追蹤行為序列。"
        prediction_adjustment = -1
    else:
        code, label = "no_active_pressure_sequence", "未見連續壓低"
        summary = "目前沒有足夠證據顯示連續壓低行為正在擴大。"
        prediction_adjustment = 0

    next_validation = [
        "若跌破前一日低點且收不回，壓低換手升級為回測/出貨風險。",
        "若站回前一日收盤，壓低測試降級，偏洗盤或換手完成。",
        "若法人現貨賣、期貨加空、廣度偏空同日共振，隔日風控權重提高。",
    ]
    return {
        **base,
        "available": True,
        "code": code,
        "label": label,
        "summary": summary,
        "current_score": current_score,
        "score_range": "0-1無明顯壓低；2-3一般測壓；4-5壓低換手；6以上強連續壓低。",
        "prediction_adjustment": prediction_adjustment,
        "recent_average_score": sum(recent_scores) / len(recent_scores),
        "recent_pressure_days": sum(score >= 4 for score in recent_scores),
        "consecutive_pressure_days": consecutive_pressure,
        "pressure_dates": [item["date"] for item in hot_dates],
        "current": current,
        "recent_rows": recent,
        "next_validation": next_validation,
        "rule": "夜盤定情緒，日盤驗真假；開高壓回、收低位、破前低、廣度賣壓、法人/避險賣壓共振時，判為壓低換手行為指紋。",
    }


def analyze_endogenous_regulation_pulse(
    scored: pd.DataFrame,
    signal_date: str,
    external_reset: dict | None = None,
    day_night_variance: dict | None = None,
    programmed_pressure: dict | None = None,
    sector_pressure: dict | None = None,
) -> dict:
    """Detect self-regulating market rhythm when external shock is not dominant."""
    base = {
        "enabled": True,
        "name": "市場內生調節脈動",
        "guardrail": "只能辨識市場自我調節的統計指紋；不指認單一操盤者，也不產生買賣命令。",
        "principle": "在無重大外部新聞或外部市場重置時，指數與績優股仍可能透過拉回、收復、換手、低點墊高來調節成本與風險。",
    }
    if scored.empty or "date" not in scored:
        return {**base, "available": False, "label": "資料不足", "summary": "缺少日線資料，無法判斷內生調節。"}

    data = scored.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["date"] <= pd.to_datetime(signal_date)].sort_values("date").reset_index(drop=True)
    if len(data) < 25:
        return {**base, "available": False, "label": "資料不足", "summary": "日線樣本不足，先保留觀察。"}

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ma_5",
        "ma_20",
        "sp500_return_1d",
        "nasdaq_return_1d",
        "sox_return_1d",
        "tsm_adr_return_1d",
        "ewt_return_1d",
        "usd_twd_return_1d",
    ]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    row = data.iloc[-1]
    prev = data.iloc[-2]
    recent = data.tail(20)
    recent5 = data.tail(5)
    close = safe_float(row.get("close"))
    open_ = safe_float(row.get("open"))
    high = safe_float(row.get("high"))
    low = safe_float(row.get("low"))
    prev_close = safe_float(prev.get("close"))
    ma5 = safe_float(row.get("ma_5"))
    ma20 = safe_float(row.get("ma_20"))
    recent20_low = safe_float(recent["low"].min()) if "low" in recent else None
    recent5_low = safe_float(recent5["low"].min()) if "low" in recent5 else None
    recent20_high = safe_float(recent["high"].max()) if "high" in recent else None
    day_range = high - low if high is not None and low is not None else None
    close_position = (
        max(0.0, min(1.0, (close - low) / day_range))
        if close is not None and low is not None and day_range not in [None, 0]
        else None
    )
    intraday_low_pct = low / prev_close - 1 if low is not None and prev_close not in [None, 0] else None
    close_return = close / prev_close - 1 if close is not None and prev_close not in [None, 0] else None
    recovery_from_low = close_position
    low_base_higher = (
        recent5_low is not None
        and recent20_low is not None
        and recent5_low > recent20_low
    )
    close_above_ma20 = close is not None and ma20 is not None and close >= ma20
    close_above_ma5 = close is not None and ma5 is not None and close >= ma5
    close_near_20d_high = (
        close is not None
        and recent20_high not in [None, 0]
        and close >= recent20_high * 0.97
    )

    external_cols = [
        "sp500_return_1d",
        "nasdaq_return_1d",
        "sox_return_1d",
        "tsm_adr_return_1d",
        "ewt_return_1d",
        "usd_twd_return_1d",
    ]
    external_values = [abs(safe_float(row.get(col)) or 0.0) for col in external_cols if col in data.columns]
    external_abs_max = max(external_values) if external_values else None
    reset_code = (external_reset or {}).get("code")
    external_quiet = (external_abs_max is None or external_abs_max < 0.015) and reset_code != "bearish_external_reset"

    score = 0
    evidence: list[str] = []
    missing: list[str] = []

    def add(points: int, text: str) -> None:
        nonlocal score
        score += points
        evidence.append(text)

    if external_quiet:
        add(2, f"外部市場未達重大重置，最大外部波動 {pct(external_abs_max)}。")
    else:
        add(-2, f"外部市場可能重置，最大外部波動 {pct(external_abs_max)}。")

    if intraday_low_pct is not None and intraday_low_pct <= -0.003 and recovery_from_low is not None and recovery_from_low >= 0.45:
        add(2, f"日內先下探 {pct(intraday_low_pct)} 後收復至區間 {pct(recovery_from_low)}。")
    elif recovery_from_low is not None and recovery_from_low <= 0.25:
        add(-2, f"收盤留在區間低位 {pct(recovery_from_low)}，調節未收復。")

    if close_above_ma20:
        add(1, "收盤仍站在20日線上，屬可控調節候選。")
    else:
        add(-1, "收盤未站回20日線，調節品質降級。")

    if low_base_higher:
        add(1, "近5日低點仍高於近20日低點，低點墊高未破壞。")
    else:
        add(-1, "近5日低點未高於近20日低點，需防測底延長。")

    relation_code = (day_night_variance or {}).get("relation_code")
    if relation_code in {"night_down_cash_reversal", "night_down_cash_down_validation"}:
        add(1, f"日夜盤關係顯示壓力被日盤驗證或收復：{(day_night_variance or {}).get('relation_label')}。")
    elif relation_code == "night_up_cash_failed":
        add(-1, "夜強日弱，較像高檔調節或假突破。")

    pressure_score = safe_int((programmed_pressure or {}).get("current_score"), 0)
    if pressure_score >= 6:
        add(-2, f"壓低分數 {pressure_score} 偏高，可能由調節轉為壓力測試。")
    elif pressure_score >= 4:
        add(-1, f"壓低分數 {pressure_score}，仍需確認是否只是換手。")
    else:
        evidence.append(f"壓低分數 {pressure_score}，未達強連續壓低。")

    sector_score = safe_int((sector_pressure or {}).get("risk_score"), 0)
    if sector_score >= 4:
        add(-1, "績優/強勢族群分化壓力偏高，內生調節需看族群是否回穩。")
    elif sector_score > 0:
        evidence.append("有局部族群分化，但尚未蓋過大盤收復訊號。")

    if close_above_ma5 and close_near_20d_high:
        add(1, "收盤站回短均且接近20日高位，調節後仍保留攻擊能力。")
    if close_return is not None:
        evidence.append(f"正式日線漲跌 {pct(close_return)}。")
    if not external_values:
        missing.append("外部市場欄位不足，外部安靜條件只能降權判斷。")

    score = int(max(-5, min(8, score)))
    if score >= 5:
        code = "endogenous_regulation_active"
        label = "內生調節脈動成立"
        summary = "外部衝擊未主導，日盤有下探/收復/低點墊高跡象，較像市場自行調節成本與籌碼。"
        model_effect = "提高洗盤/換手解釋權重；維持關鍵低點與收盤確認。"
    elif score >= 2:
        code = "endogenous_regulation_watch"
        label = "內生調節觀察"
        summary = "部分調節指紋存在，但仍需隔日確認族群與低點是否守住。"
        model_effect = "保留為路徑解釋，不提高方向信心。"
    elif score <= -2:
        code = "regulation_failed_pressure_test"
        label = "調節失敗/壓力測試"
        summary = "內生調節條件不足，較偏壓力測試或外部重置；需提高風控核對。"
        model_effect = "降低洗盤解釋權重，提高破低收不回監控。"
    else:
        code = "no_clear_endogenous_pulse"
        label = "無明確內生脈動"
        summary = "尚未形成可辨識的自動調節節奏。"
        model_effect = "維持中性觀察。"

    return {
        **base,
        "available": True,
        "date": str(pd.to_datetime(row.get("date")).date()),
        "code": code,
        "label": label,
        "score": score,
        "score_range": "-5～1調節不足；2～4觀察；5～8內生調節脈動較明顯。",
        "summary": summary,
        "model_effect": model_effect,
        "external_quiet": external_quiet,
        "external_abs_max": external_abs_max,
        "close_return": close_return,
        "intraday_low_pct": intraday_low_pct,
        "recovery_from_low": recovery_from_low,
        "close_above_ma20": close_above_ma20,
        "low_base_higher": low_base_higher,
        "pressure_score": pressure_score,
        "sector_pressure_score": sector_score,
        "evidence": top_items(evidence, 8),
        "missing_or_limited_data": missing,
        "next_validation": [
            "隔日若守住今日低點並站回前收，內生調節脈動升級為換手成功。",
            "隔日若跌破今日低點且收不回，降級為壓力測試失敗。",
            "若外部市場出現重大新聞或急跌，因果優先權改由外部重置接管。",
        ],
    }


def analyze_monthly_cycle_monitor(
    scored: pd.DataFrame,
    signal_date: str,
    programmed_pressure: dict | None = None,
    external_reset: dict | None = None,
) -> dict:
    base = {
        "enabled": True,
        "name": "月內/季度週期階段監控",
        "guardrail": "週期階段只描述市場位置與風控重點，不直接產生買賣命令。",
    }
    if scored.empty or "date" not in scored or "close" not in scored:
        return {**base, "available": False, "stage": "資料不足", "summary": "缺少日線資料。"}
    data = scored.copy()
    data["date"] = pd.to_datetime(data["date"])
    for column in ["open", "high", "low", "close", "volume"]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[data["date"] <= pd.to_datetime(signal_date)].dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    if data.empty:
        return {**base, "available": False, "stage": "資料不足", "summary": "指定日期以前沒有可用日線。"}

    row = data.iloc[-1]
    month_key = row["date"].to_period("M")
    quarter_key = row["date"].to_period("Q")
    month = data[data["date"].dt.to_period("M") == month_key].copy()
    quarter = data[data["date"].dt.to_period("Q") == quarter_key].copy()
    month_trade_day = int(len(month))
    week_of_month = int(min(5, ((month_trade_day - 1) // 5) + 1))
    close = safe_float(row.get("close"))
    month_open = safe_float(month.iloc[0].get("open")) or safe_float(month.iloc[0].get("close"))
    month_high = safe_float(month["high"].max()) if "high" in month else safe_float(month["close"].max())
    month_low = safe_float(month["low"].min()) if "low" in month else safe_float(month["close"].min())
    month_range = month_high - month_low if month_high is not None and month_low is not None else None
    month_position = (
        max(0.0, min(1.0, (close - month_low) / month_range))
        if close is not None and month_range not in [None, 0]
        else None
    )
    month_return = close / month_open - 1 if close is not None and month_open not in [None, 0] else None
    recent5 = data.tail(5)
    recent5_return = (
        safe_float(recent5.iloc[-1].get("close")) / safe_float(recent5.iloc[0].get("close")) - 1
        if len(recent5) >= 2 and safe_float(recent5.iloc[0].get("close")) not in [None, 0]
        else None
    )
    recent_low_break = bool(
        len(data) >= 6
        and safe_float(row.get("low")) is not None
        and safe_float(row.get("low")) < safe_float(data.iloc[-6:-1]["low"].min())
    )

    pressure_score = safe_int((programmed_pressure or {}).get("current_score"), 0)
    external_code = (external_reset or {}).get("code", "")
    external_bearish = external_code == "bearish_external_reset"
    external_conflict = external_code == "external_conflict_watch"

    if external_bearish and pressure_score >= 4:
        code = "weak_range_with_normalized_shock"
        stage = "偏弱盤整／突變常態化"
        summary = "月內已過主攻窗口，外部利空與壓低換手共振；目前屬偏弱盤整，等待站回壓力或續測下方。"
        action = "先看是否站回45,000/前收，未站回前不把反彈當主攻。"
    elif week_of_month <= 2 and month_return is not None and month_return > 0.025 and month_position is not None and month_position >= 0.65:
        code = "main_attack_window"
        stage = "主攻週"
        summary = "月初至月中主窗口正在發動，價位靠近月內高位。"
        action = "追蹤量能與族群擴散，避免主攻後段過熱追價。"
    elif month_position is not None and month_position >= 0.75 and recent5_return is not None and recent5_return <= 0:
        code = "peak_rotation"
        stage = "高峰換手期"
        summary = "月內高位附近轉為震盪或收斂，獲利了結與換手壓力升高。"
        action = "檢查開高壓回、長上影與量增不漲；高檔利潤啟動保護。"
    elif pressure_score >= 4 or recent_low_break:
        code = "testing_bottom"
        stage = "測底期"
        summary = "壓低測試或近期低點被測，重點是跌破後能否收回。"
        action = "看前低、昨收與短均是否收回；連續收不回則升級風控。"
    elif month_position is not None and month_position <= 0.35 and recent5_return is not None and recent5_return > 0:
        code = "restart_candidate"
        stage = "再啟動候選"
        summary = "月內低位後開始回升，但尚需站回壓力確認。"
        action = "確認高低點墊高與成交量配合。"
    elif week_of_month <= 2:
        code = "setup_or_test"
        stage = "鋪陳／測試週"
        summary = "月初主窗口仍在形成，需觀察測上壓或測下支撐哪邊成功。"
        action = "等待夜盤、日盤、量能與法人同向確認。"
    elif external_conflict:
        code = "range_with_shock_watch"
        stage = "盤整突變觀察"
        summary = "月內後段方向未明，外部分歧可能放大日夜盤突變。"
        action = "不追單一路徑，依關鍵線收回或失守判斷。"
    else:
        code = "range_consolidation"
        stage = "盤整收斂"
        summary = "月內主窗口不明顯，偏向區間整理與等待下一次測試。"
        action = "降低每日方向權重，等待下一個主窗口。"

    quarter_open = safe_float(quarter.iloc[0].get("open")) or safe_float(quarter.iloc[0].get("close"))
    quarter_return = close / quarter_open - 1 if close is not None and quarter_open not in [None, 0] else None
    if quarter_return is not None and quarter_return >= 0.10:
        quarter_stage = "季度主升後段"
    elif quarter_return is not None and quarter_return <= -0.08:
        quarter_stage = "季度修正段"
    elif quarter_return is not None and quarter_return > 0:
        quarter_stage = "季度修復/上行段"
    else:
        quarter_stage = "季度盤整/防守段"

    return {
        **base,
        "available": True,
        "date": str(row["date"].date()),
        "code": code,
        "stage": stage,
        "summary": summary,
        "action": action,
        "week_of_month": week_of_month,
        "month_trade_day": month_trade_day,
        "month": str(month_key),
        "month_return": safe_float(month_return),
        "month_position": safe_float(month_position),
        "month_high": month_high,
        "month_low": month_low,
        "recent5_return": safe_float(recent5_return),
        "recent_low_break": recent_low_break,
        "pressure_score": pressure_score,
        "external_reset_code": external_code or "none",
        "quarter": str(quarter_key),
        "quarter_return": safe_float(quarter_return),
        "quarter_stage": quarter_stage,
        "cycle_order": "鋪陳 → 測試 → 主攻 → 高峰 → 換手 → 測底 → 再啟動",
        "next_validation": [
            "站回45,000與前一日收盤，才把偏弱測底降級為修復。",
            "跌破前低且收不回，測底升級為回測/風控。",
            "若外部重置連續兩日偏空，月內階段維持偏弱，不提前判主攻。",
        ],
    }


def analyze_human_behavior_market_pattern(
    scored: pd.DataFrame,
    signal_date: str,
    premarket: dict,
    intraday: dict,
    candle: dict,
    washout: dict,
    bagua: dict,
    technical: dict,
    monthly_cycle: dict,
    programmed_pressure: dict,
) -> dict:
    data = scored[scored["date"] <= pd.to_datetime(signal_date)].sort_values("date")
    if data.empty:
        return {
            "enabled": False,
            "label": "資料不足",
            "summary": "缺少日線資料，無法判斷群眾行為模式。",
            "guardrail": "人類行為模式只作統計觀察，不證明單一主體操控，也不產生買賣命令。",
        }
    row = data.iloc[-1]

    close = safe_float(row.get("close"))
    open_ = safe_float(row.get("open"))
    high = safe_float(row.get("high"))
    low = safe_float(row.get("low"))
    volume_z = safe_float(row.get("volume_z"))
    close_position = None
    if close is not None and high is not None and low is not None and high > low:
        close_position = max(0.0, min(1.0, (close - low) / (high - low)))
    intraday_return = (close / open_ - 1) if close is not None and open_ else None
    night_status = premarket.get("night_path", {}).get("status")
    intraday_code = intraday.get("code")
    candle_type = candle.get("type")
    washout_type = washout.get("type")
    state_code = bagua.get("roles", {}).get("state_gua", {}).get("code")
    technical_label = technical.get("label")
    cycle_stage = monthly_cycle.get("stage")
    pressure_score = safe_int(programmed_pressure.get("current_score"), 0)

    evidence: list[str] = []
    score = 0

    def add(points: int, text: str) -> None:
        nonlocal score
        score += points
        evidence.append(text)

    if night_status == "bearish_but_recovered" and intraday_code in {
        "night_down_cash_reclaim_candidate",
        "night_down_break_night_low_reclaim_washout",
    }:
        add(2, "夜盤偏空但日盤收回，符合誘空後承接的行為指紋。")
        crowd_state = "試探"
        tactic = "誘空吸籌"
    elif night_status == "bullish_but_unconfirmed" and candle_type == "gap_up_failed":
        add(-2, "夜盤偏多但日盤開高失敗，符合誘多或高檔出貨候選。")
        crowd_state = "貪婪退潮"
        tactic = "誘多出貨"
    elif candle_type in {"long_bear", "gap_up_failed"} or washout_type == "failed_washout":
        add(-2, "K線或洗盤結構失敗，群眾恐慌與賣壓擴大。")
        crowd_state = "恐慌"
        tactic = "壓低測底"
    elif candle_type == "long_bull" and close_position is not None and close_position >= 0.65:
        add(2, "收盤靠近日內高位且K線偏強，代表追價與承接同步改善。")
        crowd_state = "貪婪轉強"
        tactic = "軋空推進"
    elif state_code in {"KUN", "DUI", "QIAN"} or cycle_stage in {"高峰期", "換手期", "盤整收斂"}:
        add(0, "高檔或月內整理階段，群眾多空意見分歧。")
        crowd_state = "猶豫"
        tactic = "高檔換手"
    else:
        add(0, "未見強烈單邊行為，維持一般市場呼吸觀察。")
        crowd_state = "觀察"
        tactic = "自然換手"

    if pressure_score >= 4:
        add(-1, f"壓低分數 {pressure_score}，賣方測市場耐心的指紋升高。")
    elif pressure_score <= 1:
        evidence.append(f"壓低分數 {pressure_score}，未見連續壓低擴大。")

    if volume_z is not None and volume_z >= 1.5 and close_position is not None and close_position < 0.45:
        add(-1, "量能放大但收盤位置不佳，需防換手轉出貨。")
    elif volume_z is not None and volume_z >= 0.5 and close_position is not None and close_position >= 0.60:
        add(1, "量能放大且收盤位置偏高，承接力較健康。")

    if intraday_return is not None:
        evidence.append(f"日內漲跌 {pct(intraday_return)}，收盤區間位置 {pct(close_position)}。")
    if technical_label:
        evidence.append(f"技術段位 {technical_label}。")

    strategy_driver = {
        "direction_driver": "大資金與程式路徑",
        "amplitude_driver": "散戶情緒與短線部位",
        "principle": "方向看大資金，震幅看散戶。",
        "interpretation": (
            "外資、法人、大戶、造市、避險、ETF與程式資金較常決定主劇本；"
            "散戶、當沖、融資、停損、追高與恐慌反應較常決定主劇本被放大或干擾的程度。"
        ),
        "guardrail": "此分層只作群體行為假設，不證明單一主體操控，也不產生買賣命令。",
    }
    evidence.append(strategy_driver["principle"])

    if score >= 3:
        label = "承接推進"
        direction_bias = "constructive"
        next_watch = "觀察隔日是否守住今日低點並站上壓力線；若續攻量價健康，行為路徑延續。"
    elif score <= -3:
        label = "恐慌惡化"
        direction_bias = "defensive"
        next_watch = "觀察是否跌破合理底或關鍵低點且收不回；若連續成立，升級風控。"
    elif score < 0:
        label = "壓力測試"
        direction_bias = "cautious"
        next_watch = "觀察賣壓是否只是測試耐心；收回關鍵價才可降級風險。"
    else:
        label = "可控試探"
        direction_bias = "watch"
        next_watch = "觀察夜盤與隔日日盤是否繼續維持可控震盪。"

    return {
        "enabled": True,
        "date": str(pd.to_datetime(row.get("date")).date()),
        "label": label,
        "score": int(max(-5, min(5, score))),
        "crowd_state": crowd_state,
        "tactic_candidate": tactic,
        "direction_bias": direction_bias,
        "controllable_risk": direction_bias in {"constructive", "watch"},
        "summary": f"群眾狀態偏{crowd_state}，戰術候選為{tactic}；{next_watch}",
        "next_watch": next_watch,
        "strategy_driver": strategy_driver,
        "evidence": top_items(evidence, 6),
        "features": {
            "night_status": night_status,
            "intraday_code": intraday_code,
            "candle_type": candle_type,
            "washout_type": washout_type,
            "bagua_state_code": state_code,
            "monthly_cycle_stage": cycle_stage,
            "programmed_pressure_score": pressure_score,
            "close_position": close_position,
            "intraday_return": intraday_return,
            "volume_z": volume_z,
        },
        "guardrail": "人類行為模式只作統計觀察，不證明單一主體操控，也不產生買賣命令。",
    }


def analyze_day_night_variance_pattern(
    scored: pd.DataFrame,
    forecast_date: str,
    signal_date: str,
    premarket: dict,
    intraday: dict,
    human_behavior: dict,
    external_reset: dict,
    bagua: dict,
) -> dict:
    """Audit why the night futures path and cash session did or may diverge."""
    data = scored[scored["date"] <= pd.to_datetime(signal_date)].sort_values("date")
    if data.empty:
        return {
            "enabled": False,
            "label": "資料不足",
            "summary": "缺少日線資料，無法建立日夜盤變異診斷。",
            "guardrail": "日夜盤變異只記錄可驗證足跡；原因為候選，不證明單一主體操控。",
        }

    row = data.iloc[-1]
    prior = data.iloc[-2] if len(data) >= 2 else pd.Series(dtype="float64")
    cash_date = pd.to_datetime(row.get("date")).date()
    forecast_day = pd.to_datetime(forecast_date).date()
    close = safe_float(row.get("close"))
    open_ = safe_float(row.get("open"))
    high = safe_float(row.get("high"))
    low = safe_float(row.get("low"))
    prior_close = safe_float(prior.get("close")) if not prior.empty else None
    night_return = safe_float(premarket.get("tx_night_spread_per"))
    night_close = safe_float(premarket.get("tx_night_close"))
    night_high = safe_float(premarket.get("tx_night_high"))
    night_low = safe_float(premarket.get("tx_night_low"))
    night_path = premarket.get("night_path", {}) or {}
    night_status = night_path.get("status")

    cash_return = (close / prior_close - 1) if close is not None and prior_close else None
    gap_return = (open_ / prior_close - 1) if open_ is not None and prior_close else None
    cash_range_pct = ((high - low) / prior_close) if None not in (high, low, prior_close) else None
    cash_close_position = None
    if close is not None and high is not None and low is not None and high > low:
        cash_close_position = max(0.0, min(1.0, (close - low) / (high - low)))
    night_range_pct = safe_float(night_path.get("range"))
    if night_range_pct is None and None not in (night_high, night_low, night_close) and night_close:
        night_range_pct = (night_high - night_low) / night_close

    day_validation_status = "completed" if cash_date == forecast_day else "pending_cash_session"
    evidence: list[str] = []
    causes: list[str] = []
    hidden_causes: list[str] = []
    score = 0

    def add_cause(text: str) -> None:
        if text not in causes:
            causes.append(text)

    def add_hidden_cause(text: str) -> None:
        if text not in hidden_causes:
            hidden_causes.append(text)

    if night_return is None:
        relation_code = "missing_night"
        relation_label = "夜盤資料不足"
        add_cause("缺少夜盤資料，不能判斷夜日落差。")
    elif cash_return is None:
        relation_code = "missing_cash_reference"
        relation_label = "日盤參考不足"
        add_cause("缺少前一日收盤參考，不能計算日盤驗證。")
    elif day_validation_status != "completed":
        relation_code = "pending_day_validation"
        relation_label = "夜盤已出、日盤待驗"
        evidence.append("正式日線尚未更新到報告日；目前只能把夜盤視為預期差，等待日盤驗真假。")
        if night_return >= 0.004:
            score += 1
        elif night_return <= -0.004:
            score -= 1
    elif night_return >= 0.004 and cash_return > 0:
        relation_code = "night_up_cash_up_validation"
        relation_label = "夜漲日漲確認"
        score += 2
        evidence.append("夜盤偏多且日盤收漲，預期差被現貨確認。")
    elif night_return >= 0.004 and cash_return <= 0:
        relation_code = "night_up_cash_failed"
        relation_label = "夜漲日弱反證"
        score -= 2
        evidence.append("夜盤偏多但日盤未收漲，代表現貨承接不足或高檔換手壓力。")
    elif night_return <= -0.004 and cash_return < 0:
        relation_code = "night_down_cash_down_validation"
        relation_label = "夜跌日跌確認"
        score -= 2
        evidence.append("夜盤偏空且日盤收跌，夜盤壓力被現貨確認。")
    elif night_return <= -0.004 and cash_return >= 0:
        relation_code = "night_down_cash_reversal"
        relation_label = "夜跌日收回變異"
        score += 2
        evidence.append("夜盤偏空但日盤收回，符合誘空後承接或低檔換手候選。")
    else:
        relation_code = "weak_night_signal"
        relation_label = "夜盤訊號較弱"
        evidence.append("夜盤幅度未達明顯門檻，日盤仍以自身開高低收驗證。")

    completed_cash_session = day_validation_status == "completed"
    variance_gap = (
        abs(cash_return - night_return)
        if completed_cash_session and cash_return is not None and night_return is not None
        else None
    )
    if not completed_cash_session:
        variance_level = "pending"
        variance_label = "日盤尚未完成，變異待驗"
    elif variance_gap is None:
        variance_level = "unknown"
        variance_label = "變異幅度不足以計算"
    elif variance_gap >= 0.010:
        variance_level = "large"
        variance_label = "大型日夜變異"
        add_cause("夜盤預期與日盤現貨落差大，需列入隔日誤差檢討。")
    elif variance_gap >= 0.005:
        variance_level = "medium"
        variance_label = "中度日夜變異"
        add_cause("夜盤與日盤有明顯落差，需檢查開盤缺口與日內收復。")
    else:
        variance_level = "normal"
        variance_label = "一般日夜落差"

    cash_scope = "日盤" if completed_cash_session else "上一有效日現貨背景"
    if completed_cash_session and abs(gap_return or 0) >= 0.004:
        add_cause("開盤缺口先消化夜盤預期，日內走勢需另行驗證。")
        add_hidden_cause("開盤缺口病因：隔夜預期已提前反映，日盤真正重點轉為承接力而非開盤方向。")
    if cash_range_pct is not None and cash_range_pct >= 0.012:
        add_cause(f"{cash_scope}高低差擴大，代表洗盤、停損觸發或換手加劇。")
        add_hidden_cause(f"{cash_scope}波動病因：高低差擴大，代表多空正在測試停損線、追價意願與承接深度。")
    if cash_range_pct is not None and cash_close_position is not None and cash_range_pct >= 0.010 and cash_close_position >= 0.65:
        add_cause(f"{cash_scope}高低差擴大但收高，偏向下探後收復。")
        add_hidden_cause(f"{cash_scope}收復病因：低檔有承接，但仍需隔日確認不是短線回補。")
    if cash_close_position is not None and cash_close_position <= 0.30:
        add_cause(f"{cash_scope}收盤靠近日低，賣壓未完全解除。")
        add_hidden_cause(f"{cash_scope}賣壓病因：收盤靠近日低，代表尾盤資金尚未願意明確承接。")
    if completed_cash_session and night_return is not None and cash_return is not None and night_return > 0 and cash_return < night_return - 0.005:
        add_cause("夜盤先強但日盤削弱，需防獲利了結或誘多失敗。")
        add_hidden_cause("假突破病因：夜盤拉高後現貨無法跟上，可能是高檔獲利了結、追價力不足或誘多失敗。")
    if completed_cash_session and night_return is not None and cash_return is not None and night_return < 0 and cash_return > night_return + 0.005:
        add_cause("夜盤先弱但日盤改善，需觀察是否為恐慌測試後承接。")
        add_hidden_cause("誘空病因：夜盤恐慌未被日盤確認，可能是低檔換手或空方回補。")
    if external_reset.get("reset_active"):
        add_cause(f"外部事件重置：{external_reset.get('label', '資料不足')}。")
        add_hidden_cause(f"外部病因：{external_reset.get('label', '資料不足')}使原本內部週期暫時降權，需看現貨是否重新定價。")
    if human_behavior.get("tactic_candidate") in {"高檔換手", "誘空吸籌", "誘多出貨", "壓低測底"}:
        add_cause(f"人類行為候選：{human_behavior.get('tactic_candidate')}。")
        add_hidden_cause(f"行為病因：{human_behavior.get('tactic_candidate')}代表市場正在重新分配籌碼，不能只看指數紅黑。")
    background_code = bagua.get("roles", {}).get("background_gua", {}).get("code")
    state_code = bagua.get("roles", {}).get("state_gua", {}).get("code")
    if background_code in {"DUI", "QIAN"} or state_code in {"KUN", "DUI", "QIAN"}:
        add_cause("高檔背景容易放大預期差，需防換手與誘多/誘空來回測試。")
        add_hidden_cause("高檔病因：位置越高，越容易出現換手、假突破、延長賽與隔日反覆驗證。")

    if not evidence:
        evidence.append("日夜盤差異已列入監控，但尚未形成強烈單邊證據。")
    cash_label = "日盤" if completed_cash_session else "上一有效日日盤"
    evidence.append(
        f"夜盤 {pct(night_return)}；{cash_label} {pct(cash_return)}；開盤缺口 {pct(gap_return)}；"
        f"日內高低差 {pct(cash_range_pct)}；收盤區間位置 {pct(cash_close_position)}。"
    )

    if relation_code == "pending_day_validation":
        label = "待日盤驗證"
        next_watch = "等待日盤驗真假；先看開盤是否一次反映夜盤，再看低點是否守住與收盤是否站回關鍵線。"
    elif relation_code in {"night_up_cash_failed", "night_down_cash_down_validation"}:
        label = "變異偏防守"
        next_watch = "隔日若續破今日低點或收不回，將把變異升級為風險惡化。"
    elif relation_code in {"night_down_cash_reversal", "night_up_cash_up_validation"}:
        label = "變異可控"
        next_watch = "隔日若守住今日低點並站回壓力線，變異可視為健康換手候選。"
    else:
        label = "一般觀察"
        next_watch = "等待更明確的日盤高低收與量能確認。"

    return {
        "enabled": True,
        "date": cash_date.isoformat(),
        "forecast_date": forecast_day.isoformat(),
        "day_validation_status": day_validation_status,
        "label": label,
        "relation_code": relation_code,
        "relation_label": relation_label,
        "variance_level": variance_level,
        "variance_label": variance_label,
        "score": int(max(-5, min(5, score))),
        "summary": f"{relation_label}；{variance_label}。{next_watch}",
        "next_watch": next_watch,
        "cause_candidates": top_items(causes, 7),
        "hidden_cause_candidates": top_items(hidden_causes, 7),
        "evidence": top_items(evidence, 6),
        "features": {
            "night_status": night_status,
            "night_return": night_return,
            "night_close": night_close,
            "night_high": night_high,
            "night_low": night_low,
            "night_range_pct": night_range_pct,
            "cash_return": cash_return,
            "cash_open": open_,
            "cash_high": high,
            "cash_low": low,
            "cash_close": close,
            "cash_range_pct": cash_range_pct,
            "cash_close_position": cash_close_position,
            "gap_return": gap_return,
            "variance_gap": variance_gap,
            "intraday_code": intraday.get("code"),
            "human_tactic_candidate": human_behavior.get("tactic_candidate"),
            "external_reset_code": external_reset.get("code"),
            "background_gua_code": background_code,
            "state_gua_code": state_code,
        },
        "guardrail": "日夜盤變異是智慧判斷的資料根基；原因候選需由量能、廣度、籌碼與隔日走勢驗證，不證明單一主體操控，也不產生買賣命令。",
    }


def analyze_psychological_warfare_pattern(
    bagua: dict,
    human_behavior: dict,
    day_night: dict,
    intraday: dict,
    endogenous_pulse: dict,
    crash: dict,
    technical: dict,
) -> dict:
    """Translate human psychology, Bagua phase, and Sun Tzu tactics into auditable market hypotheses."""
    tactic = human_behavior.get("tactic_candidate", "自然換手")
    crowd = human_behavior.get("crowd_state", "觀察")
    relation = day_night.get("relation_code")
    state_gua = bagua.get("roles", {}).get("state_gua", {})
    background_gua = bagua.get("roles", {}).get("background_gua", {})
    state_code = state_gua.get("code") or bagua.get("primary", {}).get("code")
    risk_value = safe_float(crash.get("risk_value"))
    health_score = safe_float(crash.get("health_score"))
    technical_label = technical.get("label", "資料不足")

    evidence: list[str] = []
    principles: list[str] = []
    score = 0

    def add(points: int, principle: str, text: str) -> None:
        nonlocal score
        score += points
        principles.append(principle)
        evidence.append(text)

    if relation == "night_down_cash_reversal":
        add(2, "以正合，以奇勝", "夜盤弱是奇，日盤收回是正；偏向誘空後承接或空方回補。")
    elif relation == "night_up_cash_failed":
        add(-2, "兵者詭道", "夜盤強但日盤失敗，偏向誘多、獲利了結或高檔換手反證。")
    elif relation == "pending_day_validation":
        add(0, "知可戰不可戰", "日盤尚未完成，夜盤只能當前哨，不可提前宣稱勝負。")
    elif relation == "night_down_cash_down_validation":
        add(-2, "虛實相應", "夜盤弱被日盤確認，空方壓力從前哨進入現貨戰場。")

    if tactic == "誘空吸籌":
        add(2, "利而誘之", "市場利用恐懼測出浮籌，若收回關鍵線，代表承接力勝出。")
    elif tactic == "誘多出貨":
        add(-2, "能而示之不能/近而示遠", "表面偏強但現貨追價失敗，需防高檔派發。")
    elif tactic == "壓低測底":
        add(-1, "避實擊虛", "賣方測支撐與耐心，真正答案在低點是否越墊越高。")
    elif tactic == "軋空推進":
        add(2, "勢如彍弩，節如發機", "承接與追價同步時，短線容易由空方回補推進。")
    elif tactic == "高檔換手":
        add(0, "兵貴勝，不貴久", "高檔整理可接受，但久攻不下或量價轉弱要降級。")

    if state_code == "ZHEN":
        add(1, "動如雷震", "狀態卦在震，代表發動或再啟動，需用回測不破驗證。")
    elif state_code in {"LI", "KUN", "DUI", "QIAN"}:
        add(-1, "先為不可勝", "高位卦代表先控風險，再承認續攻。")
    elif state_code in {"KAN", "GEN"}:
        add(0, "陷之死地而後生", "低位或止跌卦要等不再破低，不能只靠反彈判起漲。")

    if endogenous_pulse.get("code") == "regulation_reclaim_pulse":
        add(1, "治眾如治寡，分數是也", "內生調節脈動成立，表示市場可能正在有秩序地換手。")
    elif endogenous_pulse.get("code") == "regulation_failed_pressure_test":
        add(-1, "久則鈍兵挫銳", "調節失敗，代表沖灌久了反而消耗承接。")

    if risk_value is not None and risk_value >= 55:
        add(-2, "先求不敗", f"風險值 {num(risk_value)} 偏高，心理戰判讀必須讓位給風控。")
    elif health_score is not None and health_score >= 70:
        add(1, "勝可知，不可為", f"健康指數 {num(health_score)} 尚可，但仍須等對手露出可勝之形。")

    if score >= 4:
        label = "攻勢心理戰佔優"
        stance = "可控偏攻"
        next_read = "若隔日守住強收區並站回壓力線，代表虛實測試後多方取得節奏。"
        action_policy = "偏攻觀察；只承認守線後續攻，不追認未驗證突破。"
    elif score <= -3:
        label = "防守心理戰升級"
        stance = "先求不敗"
        next_read = "若跌破關鍵防線且收不回，代表空方測壓成功，風控優先。"
        action_policy = "風控優先；等待破線收回或風險降溫，不用心理敘事抵消破位。"
    else:
        label = "虛實交錯待驗"
        stance = "等地面戰確認"
        next_read = "目前真偽未分，需等日盤開高低收、量能與族群廣度確認。"
        action_policy = "中性觀察；用開盤後30～60分鐘與收盤位置決定升降級。"

    cause_effect_chain = [
        f"前因: 日夜盤關係 {relation or '資料不足'}，背景卦 {background_gua.get('gua') or 'NA'}，狀態卦 {state_gua.get('gua') or 'NA'}。",
        f"心理: 群眾狀態 {crowd}，戰術候選 {tactic}；數據只當症狀，需解讀恐懼、貪婪、承接與換手意圖。",
        (
            f"機制: 以 {technical_label} 搭配兵法虛實判讀，確認這是測壓、誘敵、換手、軋空或真破位；"
            "主劇本看大資金與程式路徑，震幅變數看散戶情緒、當沖、融資、停損與追高恐慌是否被觸發。"
        ),
        f"驗證: {next_read}",
    ]
    semantic_quantification = [
        {
            "text": f"日夜盤關係:{relation or '資料不足'}",
            "score": 2 if relation == "night_down_cash_reversal" else -2 if relation in {"night_up_cash_failed", "night_down_cash_down_validation"} else 0,
            "meaning": "夜盤是前哨，日盤是驗證；同向確認才提高方向權重，反向則判虛實。",
        },
        {
            "text": f"人性戰術:{tactic}",
            "score": 2 if tactic in {"誘空吸籌", "軋空推進"} else -2 if tactic == "誘多出貨" else -1 if tactic == "壓低測底" else 0,
            "meaning": "把恐慌、貪婪、換手、測底等文字轉為可比較分數。",
        },
        {
            "text": f"八卦狀態:{state_gua.get('gua') or bagua.get('primary', {}).get('gua') or 'NA'}",
            "score": 1 if state_code == "ZHEN" else -1 if state_code in {"LI", "KUN", "DUI", "QIAN"} else 0,
            "meaning": "週期位置決定攻守節奏；高位先控風險，震位看回測不破。",
        },
        {
            "text": f"風控讀數:{num(risk_value) if risk_value is not None else 'NA'}",
            "score": -2 if risk_value is not None and risk_value >= 55 else 1 if health_score is not None and health_score >= 70 else 0,
            "meaning": "哲學判讀必須服從風險儀表；風險升高時心理戰降權。",
        },
        {
            "text": "資金分層:大資金方向/散戶震幅",
            "score": 0,
            "meaning": "主方向優先觀察法人、期貨、權值與程式路徑；盤中震幅觀察散戶、當沖、融資、停損與追高恐慌是否放大劇本。",
        },
    ]
    intent_trace = [
        {
            "data": "台指夜盤",
            "intent": "前哨試探與預期差",
            "sunzi": "虛實",
            "validation": "日盤開盤缺口、30～60分鐘是否收回夜盤低點/收盤價。",
        },
        {
            "data": "日盤收盤位置",
            "intent": "現貨裁判與真實承接",
            "sunzi": "軍形",
            "validation": "收盤站回昨收、短均與關鍵整數關卡才承認修復；收不回則承認壓力。",
        },
        {
            "data": "成交量與族群廣度",
            "intent": "換手是否健康、是否只是權值硬撐",
            "sunzi": "兵勢",
            "validation": "量能放大但廣度惡化偏派發；量縮守線或廣度擴散偏健康換手。",
        },
        {
            "data": "期現差與法人期貨",
            "intent": "避險、套利、壓力測試或空單回補",
            "sunzi": "奇正",
            "validation": "期貨領先現貨且現貨不跟跌，偏試壓失敗；期現同破才升級偏空。",
        },
        {
            "data": "新聞與生活心理面",
            "intent": "觸發按鈕，不是單獨主因",
            "sunzi": "用間/九變",
            "validation": "必須被夜盤、日盤、量能或族群同步確認，否則只作降權濾鏡。",
        },
    ]
    intent_summary = (
        "資料是意圖的表現：夜盤看試探，日盤看承接，量能看換手，期現差看避險與套利，"
        "新聞看觸發按鈕；全部必須回到收盤與後續病歷驗證。"
    )

    return {
        "enabled": True,
        "framework": "human_psychology_bagua_sunzi_v1",
        "label": label,
        "stance": stance,
        "score": int(max(-8, min(8, score))),
        "crowd_state": crowd,
        "tactic_candidate": tactic,
        "bagua_state": state_gua.get("gua") or bagua.get("primary", {}).get("gua"),
        "bagua_background": background_gua.get("gua"),
        "technical_phase": technical_label,
        "sunzi_principles": top_items(principles, 5),
        "evidence": top_items(evidence, 8),
        "cause_effect_chain": cause_effect_chain,
        "semantic_quantification": semantic_quantification,
        "intent_summary": intent_summary,
        "intent_trace": intent_trace,
        "action_policy": action_policy,
        "truth_test": [
            "夜盤只看前哨與預期差，日盤收盤才是現貨戰場真相。",
            "若價格、量能、廣度與關鍵線不同向，兵法敘事必須降權。",
            "同類病例需放入病歷表，等前瞻樣本累積後才可升級規則。",
        ],
        "next_validation": next_read,
        "guardrail": "兵法與八卦是人性與週期的分析語言，必須被市場數據驗證；不得宣稱單一主體操控，也不得產生投資命令。",
    }


def analyze_integrated_summary(payload: dict) -> dict:
    score = 0
    bullish: list[str] = []
    bearish: list[str] = []
    neutral: list[str] = []
    missing: list[str] = []

    forecast = payload.get("forecast", {})
    cause = payload.get("cause_analysis", {})
    premarket = payload.get("premarket", {})
    memory = payload.get("memory_industry_risk", {})
    bagua = payload.get("bagua_lifecycle", {})
    candle = payload.get("candlestick_pattern", {})
    washout = payload.get("washout_pattern", {})
    cycle = payload.get("tradeable_cycle", {})
    capital = payload.get("capital_flow", {})
    consistency = payload.get("logic_consistency", {})
    programmed_pressure = payload.get("programmed_pressure_pattern", {})
    external_reset = payload.get("external_event_reset_monitor", {})
    human_behavior = payload.get("human_behavior_market_pattern", {})
    psychological_warfare = payload.get("psychological_warfare_pattern", {})
    day_night_variance = payload.get("day_night_variance_pattern", {})
    endogenous_pulse = payload.get("endogenous_regulation_pulse", {})
    psychological_warfare = payload.get("psychological_warfare_pattern", {})

    risk_score = score_risk_regime(forecast.get("risk_regime"))
    score += risk_score
    append_evidence(risk_score, bullish, bearish, neutral, f"大盤風險狀態為 {regime_text(forecast.get('risk_regime', 'neutral'))}。")

    external_score = safe_int(cause.get("external_score"), 0)
    night_score = safe_int(cause.get("night_futures_score"), 0)
    panic_score = safe_int(cause.get("panic_reversal_score"), 0)
    score += external_score + night_score + panic_score
    append_evidence(external_score, bullish, bearish, neutral, f"外部市場: {cause.get('external_pressure_text', '資料不足')}。")
    append_evidence(night_score, bullish, bearish, neutral, f"台指期夜盤: {cause.get('night_futures_text', '資料不足')}。")
    append_evidence(panic_score, bullish, bearish, neutral, f"日盤承接: {cause.get('intraday_truth_text', '資料不足')}。")

    premarket_score = safe_int(premarket.get("total_score"), 0) if premarket.get("is_premarket") else 0
    score += premarket_score
    if premarket.get("is_premarket"):
        append_evidence(premarket_score, bullish, bearish, neutral, f"盤前模式: {premarket.get('summary', '資料不足')}")

    bagua_score = score_bagua_summary(bagua)
    score += bagua_score
    append_evidence(
        bagua_score,
        bullish,
        bearish,
        neutral,
        f"八卦生命週期: {bagua.get('resonance', {}).get('label', '資料不足')}，{bagua.get('resonance', {}).get('plain_summary', '')}",
    )

    cycle_score = score_tradeable_cycle_summary(cycle)
    score += cycle_score
    append_evidence(
        cycle_score,
        bullish,
        bearish,
        neutral,
        f"年度波段燈塔: {cycle.get('current_position', {}).get('label', '資料不足')}，{cycle.get('current_position', {}).get('plain_summary', '')}",
    )

    candle_score = score_candlestick_summary(candle)
    score += candle_score
    append_evidence(candle_score, bullish, bearish, neutral, f"K線: {candle.get('label', '資料不足')}，{candle.get('plain_summary', '')}")

    washout_score = score_washout_summary(washout)
    score += washout_score
    append_evidence(washout_score, bullish, bearish, neutral, f"洗盤偵測: {washout.get('label', '資料不足')}，{washout.get('plain_summary', '')}")

    memory_score = score_memory_summary(memory)
    score += memory_score
    append_evidence(memory_score, bullish, bearish, neutral, f"產業尾部風險: {memory_risk_text(memory.get('level', 'low'))}，{memory.get('reasons', '')}")

    capital_score = safe_int(capital.get("non_price_score"), 0)
    if capital.get("has_factor_values"):
        score += capital_score
        append_evidence(capital_score, bullish, bearish, neutral, f"法人籌碼與期權: {capital.get('plain_summary', '')}")
    else:
        missing.append(capital.get("plain_summary", "法人籌碼與期權資料不足。"))

    pressure_adjustment = safe_int(programmed_pressure.get("prediction_adjustment"), 0)
    if programmed_pressure.get("available") and pressure_adjustment:
        score += pressure_adjustment
        bearish.append(
            f"行為模式: {programmed_pressure.get('label')}，壓低分數 "
            f"{programmed_pressure.get('current_score')}；{programmed_pressure.get('summary')}"
        )
    elif programmed_pressure.get("available"):
        neutral.append(
            f"行為模式: {programmed_pressure.get('label')}，壓低分數 "
            f"{programmed_pressure.get('current_score')}。"
        )

    if external_reset.get("available"):
        reset_adjustment = safe_int(external_reset.get("risk_adjustment"), 0)
        score += reset_adjustment
        text = (
            f"外部重置: {external_reset.get('label')}，分數 {external_reset.get('reset_score')}；"
            f"{external_reset.get('summary')}"
        )
        if reset_adjustment < 0:
            bearish.append(text)
        elif reset_adjustment > 0:
            bullish.append(text)
        else:
            neutral.append(text)
    elif external_reset.get("enabled"):
        missing.append(external_reset.get("summary", "外部重置資料不足。"))

    if human_behavior.get("enabled"):
        behavior_score = safe_int(human_behavior.get("score"), 0)
        behavior_adjustment = 1 if behavior_score >= 3 else -1 if behavior_score <= -3 else 0
        score += behavior_adjustment
        text = (
            f"人類行為模式: {human_behavior.get('label')}，群眾狀態"
            f"{human_behavior.get('crowd_state')}，戰術候選{human_behavior.get('tactic_candidate')}。"
        )
        if behavior_adjustment > 0:
            bullish.append(text)
        elif behavior_adjustment < 0:
            bearish.append(text)
        else:
            neutral.append(text)

    if day_night_variance.get("enabled"):
        variance_score = safe_int(day_night_variance.get("score"), 0)
        variance_adjustment = 1 if variance_score >= 2 else -1 if variance_score <= -2 else 0
        score += variance_adjustment
        text = (
            f"日夜盤變異: {day_night_variance.get('relation_label')}，"
            f"{day_night_variance.get('variance_label')}；{day_night_variance.get('summary')}"
        )
        if day_night_variance.get("day_validation_status") == "pending_cash_session":
            neutral.append(text)
        elif variance_adjustment > 0:
            bullish.append(text)
        elif variance_adjustment < 0:
            bearish.append(text)
        else:
            neutral.append(text)

    if endogenous_pulse.get("enabled"):
        pulse_score = safe_int(endogenous_pulse.get("score"), 0)
        pulse_adjustment = 1 if pulse_score >= 5 else -1 if pulse_score <= -2 else 0
        score += pulse_adjustment
        text = (
            f"內生調節脈動: {endogenous_pulse.get('label')}，分數 "
            f"{endogenous_pulse.get('score')}；{endogenous_pulse.get('summary')}"
        )
        if pulse_adjustment > 0:
            bullish.append(text)
        elif pulse_adjustment < 0:
            bearish.append(text)
        else:
            neutral.append(text)

    if psychological_warfare.get("enabled"):
        warfare_score = safe_int(psychological_warfare.get("score"), 0)
        warfare_adjustment = 1 if warfare_score >= 4 else -1 if warfare_score <= -3 else 0
        score += warfare_adjustment
        text = (
            f"心理戰框架: {psychological_warfare.get('label')}，"
            f"{psychological_warfare.get('stance')}；戰術候選{psychological_warfare.get('tactic_candidate')}。"
        )
        if warfare_adjustment > 0:
            bullish.append(text)
        elif warfare_adjustment < 0:
            bearish.append(text)
        else:
            neutral.append(text)

    if consistency.get("contradiction_count", 0):
        score -= 3
        bearish.append("判斷邏輯出現硬矛盾，需降低預測可信度。")
    elif consistency.get("needs_explanation_count", 0):
        neutral.append("部分訊號屬於不同時間級別，已用總結層分開解讀。")

    score = int(max(-12, min(12, score)))
    bias = integrated_bias(score)
    confidence = integrated_confidence(score, missing, consistency)
    confidence = apply_direction_reliability_confidence_cap(
        confidence,
        payload.get("direction_reliability_policy", {}),
    )
    conclusion = integrated_plain_conclusion(bias, score, bullish, bearish, missing)
    return {
        "score": score,
        "bias": bias,
        "bias_text": integrated_bias_text(bias),
        "confidence": confidence,
        "plain_summary": conclusion,
        "bullish_evidence": top_items(bullish, 5),
        "bearish_evidence": top_items(bearish, 5),
        "neutral_evidence": top_items(neutral, 4),
        "missing_or_limited_data": top_items(missing, 4),
        "confirmation": integrated_confirmation_rules(payload, bias),
        "invalidation": integrated_invalidation_rules(payload, bias),
    }


def build_weather_satellite_forecast_model(payload: dict) -> dict:
    """Weather-style market nowcast: observe, diagnose, forecast, verify."""
    integrated = payload.get("integrated_summary", {})
    premarket = payload.get("premarket", {})
    intraday = payload.get("intraday_tactical_monitor", {})
    day_night = payload.get("day_night_variance_pattern", {})
    health = payload.get("market_health", {})
    health_value = health.get("health_value", {})
    crash = payload.get("crash_monitor", {})
    mode = payload.get("market_mode_switch", {})
    bagua = payload.get("bagua_lifecycle", {})
    external_reset = payload.get("external_event_reset_monitor", {})
    endogenous_pulse = payload.get("endogenous_regulation_pulse", {})
    reliability = payload.get("direction_reliability_policy", {})

    storm_cells: list[str] = []
    route_checks: list[str] = []
    confidence_caps: list[str] = []
    activation_triggers: list[str] = []

    def add_unique(items: list[str], text: str) -> None:
        if text and text not in items:
            items.append(text)

    health_score = safe_float(crash.get("health_score"))
    risk_value = safe_float(crash.get("risk_value"))
    relation_code = day_night.get("relation_code")
    alert_code = crash.get("alert_code")

    if alert_code in {"confirmed_warning", "early_alert", "intraday_unconfirmed"}:
        add_unique(storm_cells, f"崩盤監控 {crash.get('alert_level', '資料不足')}。")
    if risk_value is not None and risk_value >= 55:
        add_unique(storm_cells, f"風險值 {num(risk_value)} 偏高。")
    if health_score is not None and health_score < 45:
        add_unique(storm_cells, f"健康指數 {num(health_score)} 轉弱。")
    if relation_code in {"night_up_cash_failed", "night_down_cash_down_validation"}:
        add_unique(storm_cells, f"日夜盤雷達偏防守：{day_night.get('relation_label')}。")
    if mode.get("crisis_early"):
        add_unique(storm_cells, f"模式切換：{mode.get('headline')}。")
    if endogenous_pulse.get("code") == "regulation_failed_pressure_test":
        add_unique(storm_cells, f"內生調節失敗：{endogenous_pulse.get('summary')}")

    if premarket.get("is_premarket"):
        add_unique(confidence_caps, "盤前只有夜盤與外部雲圖，需等日盤地面站驗證。")
    if not payload.get("capital_flow", {}).get("has_factor_values"):
        add_unique(confidence_caps, "法人、融資、期權資料缺口使方向信心上限降低。")
    if payload.get("production_policy", {}).get("main_multi_day_direction", {}).get("enabled") is False:
        add_unique(confidence_caps, "大盤多日方向正式訊號仍停用，只能輸出研究性路徑。")
    if reliability.get("level") in {"severe_degrade", "degrade"}:
        add_unique(confidence_caps, f"{reliability.get('headline')}：{reliability.get('summary')}")

    defense_levels = intraday.get("defense_levels", [])
    reclaim_levels = intraday.get("reclaim_levels", [])
    if defense_levels:
        add_unique(route_checks, f"地面防守站：{format_levels(defense_levels)}。")
    if reclaim_levels:
        add_unique(route_checks, f"轉強雷達站：{format_levels(reclaim_levels)}。")
    if day_night.get("next_watch"):
        add_unique(route_checks, day_night.get("next_watch"))
    if crash.get("confirmation", {}).get("summary"):
        add_unique(route_checks, f"風暴三重核對：{crash.get('confirmation', {}).get('summary')}")
    if endogenous_pulse.get("available"):
        add_unique(route_checks, f"內生調節：{endogenous_pulse.get('label')}，{endogenous_pulse.get('model_effect')}")

    trigger_model = build_latent_disease_activation_triggers(
        premarket, intraday, day_night, crash, external_reset
    )
    activation_triggers = trigger_model["active_triggers"]
    night_trend = build_night_trend_summary(premarket, intraday, day_night)

    night_spread = safe_float(night_trend.get("official_spread_per"))
    if storm_cells and any("崩盤" in item or "危機" in item for item in storm_cells):
        next_step_code = "storm_defense"
        next_step = "先防守風暴雲系；盤中破線、收盤破線、連續收不回需逐級升警。"
    elif relation_code == "pending_day_validation" and night_spread is not None and night_spread >= 0.008:
        next_step_code = "night_bullish_pending_cash"
        next_step = "夜盤前哨偏多，下個日盤開局偏修復；重點看能否站回轉強線並守住夜盤低點。"
    elif relation_code == "pending_day_validation" and night_spread is not None and night_spread <= -0.008:
        next_step_code = "night_bearish_pending_cash"
        next_step = "夜盤前哨偏空，下個日盤開局偏測壓；重點看是否跌破夜盤低點後收不回。"
    elif relation_code == "pending_day_validation" and night_spread is not None and abs(night_spread) < 0.003:
        next_step_code = "night_neutral_cash_decides"
        next_step = "夜盤接近無方向，下個日盤由現貨開盤、權值承接與量能決定主軸。"
    elif relation_code == "pending_day_validation":
        next_step_code = "radar_pending_ground_truth"
        next_step = "夜盤已有小幅試探，下個日盤依開盤缺口、低點防守與收盤位置確認。"
    elif relation_code == "night_up_cash_failed":
        next_step_code = "false_breakout_overtime"
        next_step = "夜盤先強但現貨反證，下一步以假突破延長賽處理。"
    elif relation_code == "night_down_cash_reversal":
        next_step_code = "washout_reclaim_watch"
        next_step = "夜盤先弱但日盤收回，下一步觀察低點是否墊高與承接是否延續。"
    elif integrated.get("bias") == "bullish" and health_value.get("controllable_risk") is not False:
        next_step_code = "controlled_advance"
        next_step = "偏多氣流仍在，但只在防守線有效且轉強線站回時承認續攻。"
    elif integrated.get("bias") == "bearish":
        next_step_code = "weak_pressure_system"
        next_step = "偏空氣壓增強，下一步先看是否跌破有效低點與合理底防線。"
    else:
        next_step_code = "range_observation"
        next_step = "多空氣流未定，先用高低點突破與收盤位置判斷下一步。"
    forecast_layers = [
        f"夜盤前哨: {night_trend.get('label', '資料不足')}",
        f"日盤開局: {next_step}",
        "收盤定案: 站回轉強線偏修復，跌破防守線收不回偏風險升級。",
    ]
    zero_one_tilt = build_zero_one_directional_tilt(
        integrated,
        night_trend,
        external_reset,
        health_value,
        intraday,
        day_night,
    )
    confidence_scope = "信心低不是不預測；意思是夜盤路徑已可預判開局，但日盤收盤方向仍需現貨驗證。"
    if premarket.get("is_non_trading_day"):
        confidence_scope = "非交易日低信心不是不預測；夜盤與外部市場已形成下個交易日劇本，但台股現貨尚未開盤，收盤結論不能先定案。"
    elif relation_code == "pending_day_validation":
        confidence_scope = "盤前低信心不是不預測；夜盤趨勢可先判開局，但不得直接等同日盤收盤方向。"
    root_cause = build_root_cause_decomposition(
        next_step_code=next_step_code,
        day_night=day_night,
        trigger_model=trigger_model,
        intraday=intraday,
        crash=crash,
        mode=mode,
    )

    return {
        "enabled": True,
        "framework": "market_weather_satellite_v1",
        "headline": weather_satellite_headline(next_step_code),
        "next_step_code": next_step_code,
        "next_step": next_step,
        "bias": integrated.get("bias"),
        "confidence": integrated.get("confidence"),
        "confidence_scope": confidence_scope,
        "forecast_layers": forecast_layers,
        "zero_one_tilt": zero_one_tilt,
        "market_weather": {
            "external_pressure": premarket.get("pressure", "資料不足"),
            "night_radar": day_night.get("relation_label", "資料不足"),
            "cash_ground_truth": intraday.get("label", "資料不足"),
            "volatility_cloud": day_night.get("variance_label", "資料不足"),
            "health_value": health_value.get("label", "資料不足"),
            "bagua_terrain": bagua.get("roles", {}).get("summary", "資料不足"),
            "external_reset": external_reset.get("label", "資料不足"),
        },
        "night_trend": night_trend,
        "storm_cells": top_items(storm_cells, 5),
        "route_checks": top_items(route_checks, 6),
        "latent_disease_triggers": trigger_model,
        "activation_triggers": top_items(activation_triggers, 8),
        "root_cause_decomposition": root_cause,
        "confidence_caps": top_items(confidence_caps, 5),
        "direction_reliability": reliability,
        "principle": "像氣象衛星一樣先觀測多層資料，再給路徑機率與警戒線；每次收盤後用實際結果校正。",
        "error_repair_policy": "若預測錯誤，先分類錯因與資料時點，再找合理病因、降權或修正程式，隔日續驗；不得回填舊預測。",
        "guardrail": "這是風險預報與路徑監控，不是投資命令；準確度必須靠前瞻驗證累積，不能事後回填。",
    }


def build_latent_disease_activation_triggers(
    premarket: dict,
    intraday: dict,
    day_night: dict,
    crash: dict,
    external_reset: dict,
) -> dict:
    """External conditions that can activate latent market diseases."""
    active: list[str] = []
    watch: list[str] = []

    def add(bucket: list[str], text: str) -> None:
        if text and text not in bucket:
            bucket.append(text)

    nasdaq = safe_float(premarket.get("nasdaq_return_1d"))
    sox = safe_float(premarket.get("sox_return_1d"))
    adr = safe_float(premarket.get("tsm_adr_return_1d"))
    vix = safe_float(premarket.get("vix_return_1d"))
    night = safe_float(premarket.get("tx_night_spread_per"))
    night_low = safe_float(premarket.get("tx_night_low"))
    risk_value = safe_float(crash.get("risk_value"))
    features = day_night.get("features", {}) or {}
    gap = safe_float(features.get("gap_return"))
    cash_range = safe_float(features.get("cash_range_pct"))
    cash_close_position = safe_float(features.get("cash_close_position"))

    if sox is not None and sox <= -0.02:
        add(active, "費半急跌：半導體權值股易成為壓力源。")
    elif sox is not None and sox >= 0.02:
        add(watch, "費半強彈：有利攻擊，但若日盤收不住，反而是假突破測試。")
    if adr is not None and adr <= -0.015:
        add(active, "台積電ADR轉弱：權值股定價壓力升高。")
    elif adr is not None and adr >= 0.015:
        add(watch, "台積電ADR偏強：支撐日盤開盤氣氛，仍需現貨確認。")
    if nasdaq is not None and nasdaq <= -0.015:
        add(active, "Nasdaq轉弱：科技風險偏好下降。")
    if vix is not None and vix >= 0.08:
        add(active, "VIX急升：避險情緒可能觸發潛伏賣壓。")
    elif vix is not None and vix <= -0.04:
        add(watch, "VIX回落：恐慌降溫，有助可控續攻。")
    if night is not None and night >= 0.006:
        add(watch, "夜盤明顯偏多：若日盤不能收穩，假突破病灶會被觸發。")
    elif night is not None and night <= -0.006:
        add(active, "夜盤明顯偏空：若日盤低點收不回，風險病灶會發作。")
    if night_low is not None:
        add(watch, f"夜盤低點 {num(night_low)}：日盤跌破後能否收回，是誘空或轉弱分界。")
    if gap is not None and abs(gap) >= 0.004:
        add(active, "開盤缺口擴大：表示外在預期已引爆日內重新定價。")
    if cash_range is not None and cash_range >= 0.012:
        add(active, "日內高低差擴大：表示洗盤、停損與換手壓力已被觸發。")
    if cash_close_position is not None and cash_close_position <= 0.30:
        add(active, "收盤靠近日低：尾盤承接不足，潛伏賣壓偏發作。")
    elif cash_close_position is not None and cash_close_position >= 0.65:
        add(watch, "收盤靠近日高：承接改善，但隔日仍要確認不是短線回補。")
    if risk_value is not None and risk_value >= 55:
        add(active, "風險值升高：潛伏病灶更容易由消息或破線觸發。")
    if external_reset.get("reset_active"):
        add(active, f"外部事件重置：{external_reset.get('label', '資料不足')}。")

    return {
        "active_triggers": top_items(active, 8),
        "watch_triggers": top_items(watch, 8),
        "trigger_rule": "病灶不等於發病；必須由外部市場、夜盤幅度、開盤缺口、日內高低差、收盤位置與風險值共同觸發。",
        "guardrail": "觸發因子只提高或降低路徑機率，不單獨構成投資命令。",
    }


def build_night_trend_summary(premarket: dict, intraday: dict, day_night: dict) -> dict:
    """Summarize TX night-session trend for the weather satellite headline."""
    path = premarket.get("night_path") or intraday.get("night_path") or {}
    basis = premarket.get("night_basis_audit") or {}
    spread = safe_float(premarket.get("tx_night_spread_per") or path.get("spread_per"))
    open_close = safe_float(premarket.get("tx_night_return") or path.get("return"))
    previous_close_return = safe_float(
        premarket.get("tx_night_vs_previous_close") or basis.get("vs_previous_night_close_return")
    )
    night_close = safe_float(premarket.get("tx_night_close") or path.get("close"))
    night_low = safe_float(premarket.get("tx_night_low") or path.get("low"))
    night_high = safe_float(premarket.get("tx_night_high") or path.get("high"))
    close_position = safe_float(path.get("close_position"))
    relation = day_night.get("relation_label", "待日盤驗證")

    if spread is None and open_close is None and night_close is None:
        return {
            "available": False,
            "label": "夜盤資料不足",
            "summary": "目前沒有可用台指夜盤路徑，氣象衛星只能用外部市場與日盤資料。",
            "validation": "補齊夜盤收盤、低點、開收報酬後再判斷。",
        }

    if spread is not None and spread >= 0.008:
        label = "夜盤偏多待驗"
        summary = "夜盤明顯偏多，但只代表前哨預期轉強；仍需日盤現貨承認。"
    elif spread is not None and spread <= -0.008:
        label = "夜盤偏空待驗"
        summary = "夜盤明顯偏空，代表前哨壓力升高；需看日盤是否放大或收回。"
    elif spread is not None and abs(spread) < 0.003:
        label = "夜盤無方向"
        summary = "夜盤接近0%，代表前哨不表態，日盤現貨將取得主導權。"
    else:
        label = "夜盤小幅試探"
        summary = "夜盤只小幅試探，不足以直接改寫日盤方向。"

    path_label = path.get("label") or label
    path_summary = path.get("summary") or ""
    if close_position is not None and close_position >= 0.75:
        path_shape = "收在夜盤區間高檔，前哨心理偏穩。"
    elif close_position is not None and close_position <= 0.25:
        path_shape = "收在夜盤區間低檔，前哨壓力偏重。"
    else:
        path_shape = "收在夜盤區間中段，仍屬拉鋸。"

    return {
        "available": True,
        "label": label,
        "path_label": path_label,
        "summary": summary,
        "path_summary": path_summary,
        "official_spread_per": spread,
        "open_close_return": open_close,
        "previous_night_close_return": previous_close_return,
        "close": night_close,
        "low": night_low,
        "high": night_high,
        "close_position": close_position,
        "path_shape": path_shape,
        "relation": relation,
        "validation": "夜盤只作前哨雷達；日盤需用開盤是否一次反映、30～60分鐘是否收回、低點是否守住、收盤是否站回關鍵線驗證。",
        "guardrail": "夜盤趨勢不得直接宣稱日盤收盤方向；接近0%必須標為無方向或待驗。",
    }


def build_zero_one_directional_tilt(
    integrated: dict,
    night_trend: dict,
    external_reset: dict,
    health_value: dict,
    intraday: dict,
    day_night: dict,
) -> dict:
    """Choose which 0/1 branch currently has better evidence without making a trade command."""
    score = 0
    evidence: list[str] = []
    counter: list[str] = []

    def add(points: int, text: str) -> None:
        nonlocal score
        score += points
        evidence.append(text)

    night_spread = safe_float(night_trend.get("official_spread_per"))
    night_close_position = safe_float(night_trend.get("close_position"))
    if night_spread is not None and night_spread >= 0.008:
        add(2, f"夜盤官方 {pct(night_spread)}，偏修復。")
    elif night_spread is not None and night_spread <= -0.008:
        add(-2, f"夜盤官方 {pct(night_spread)}，偏測壓。")
    elif night_spread is not None and abs(night_spread) < 0.003:
        evidence.append("夜盤近0%，日盤權重提高。")

    if night_close_position is not None and night_close_position >= 0.75:
        add(1, "夜盤收高檔，追價心理仍在。")
    elif night_close_position is not None and night_close_position <= 0.25:
        add(-1, "夜盤收低檔，避險壓力仍在。")

    if integrated.get("bias") == "bullish":
        add(1, "綜合訊號偏多。")
    elif integrated.get("bias") == "bearish":
        add(-1, "綜合訊號偏空。")

    if external_reset.get("reset_active") and external_reset.get("direction") == "bearish":
        add(-2, "外部利空重置啟動，偏向0/測壓。")
    elif external_reset.get("direction") == "mixed":
        evidence.append("外部多空混合，單邊確定性下降。")

    if health_value.get("controllable_risk") is False:
        add(-1, "健康值轉謹慎，不能完全放鬆。")
    elif health_value.get("value") and safe_float(health_value.get("value")) >= 60:
        add(1, "健康價值仍支撐可控波動。")

    relation_code = day_night.get("relation_code")
    if relation_code == "night_down_cash_down_validation":
        add(-2, "夜跌日跌已被現貨確認。")
    elif relation_code == "night_down_cash_reversal":
        add(2, "夜跌被日盤收回，偏向轉機。")
    elif relation_code == "pending_day_validation":
        evidence.append("日盤未完成，仍需現貨確認。")

    defense = format_levels(intraday.get("defense_levels", [])) or "NA"
    reclaim = format_levels(intraday.get("reclaim_levels", [])) or "NA"
    counter.extend(
        [
            f"跌破防守 {defense} 且30～60分鐘收不回，轉0。",
            f"站回轉強 {reclaim} 且量能/族群不背離，確認1。",
            "開高走低且收近低，夜盤偏多降級為假突破。",
        ]
    )

    if score >= 4:
        branch = "1"
        label = "偏1：修復候選較強"
        summary = "目前事實與歷史經驗明顯偏向修復劇本，但仍需日盤收盤確認。"
    elif score >= 2:
        branch = "1"
        label = "偏1：修復候選待確認"
        summary = "目前事實與歷史經驗偏向修復劇本，但優勢尚未壓倒性，需由日盤站回轉強線確認。"
    elif score <= -4:
        branch = "0"
        label = "偏0：測壓/風險候選較強"
        summary = "目前事實與歷史經驗明顯偏向測壓劇本，需優先核對防守線是否失守。"
    elif score <= -2:
        branch = "0"
        label = "偏0：測壓/風險候選待確認"
        summary = "目前事實與歷史經驗偏向測壓劇本，但優勢尚未壓倒性，需由日盤跌破防守線確認。"
    else:
        branch = "0/1拉鋸"
        label = "0/1未分勝負"
        summary = "目前證據未形成壓倒性方向，先依關鍵線分流。"

    return {
        "framework": "zero_one_directional_tilt_v1",
        "branch": branch,
        "label": label,
        "score": score,
        "summary": summary,
        "evidence": unique_text(evidence),
        "counter_conditions": counter,
        "rule": "先列0/1雙劇本，再用事實與歷史經驗判斷目前偏哪一方；偏向不是投資命令，必須由日盤收盤驗證。",
    }


def build_root_cause_decomposition(
    next_step_code: str,
    day_night: dict,
    trigger_model: dict,
    intraday: dict,
    crash: dict,
    mode: dict,
) -> dict:
    """Break surface labels into auditable root-cause candidates."""
    active = trigger_model.get("active_triggers", [])
    watch = trigger_model.get("watch_triggers", [])
    relation = day_night.get("relation_label", "資料不足")
    variance = day_night.get("variance_label", "資料不足")
    defense = format_levels(intraday.get("defense_levels", [])) or "NA"
    reclaim = format_levels(intraday.get("reclaim_levels", [])) or "NA"

    def cause_item(surface: str, mechanism: str, roots: list[str], confirm: str, exclude: str) -> dict:
        return {
            "surface_symptom": surface,
            "mechanism": mechanism,
            "root_cause_candidates": roots,
            "confirm_condition": confirm,
            "exclude_condition": exclude,
        }

    items: list[dict] = []
    if next_step_code == "false_breakout_overtime":
        items.append(cause_item(
            "假突破延長賽",
            "夜盤或開盤先推升，但現貨收盤沒有完成站穩。",
            [
                "外部利多已在夜盤提前反映，日盤追價資金不足。",
                "高檔獲利了結壓力壓過新增買盤。",
                "權值股拉指數但市場廣度不足。",
            ],
            f"收盤未站回 {reclaim}，或收盤靠近日低、隔日再破低。",
            f"回測不破 {defense} 且收盤重新站回 {reclaim}。",
        ))
    elif next_step_code == "controlled_advance":
        items.append(cause_item(
            "可控續攻",
            "外部氣壓與夜盤偏多，日盤若守線並站回，代表承接有效。",
            [
                "恐慌降溫後資金回補。",
                "低點墊高造成空手資金被迫提高成本。",
                "權值股與族群同步性改善。",
            ],
            f"守住 {defense} 且收盤站回 {reclaim}，隔日低點不破。",
            f"站上後收不穩，或爆量不漲、收盤靠近日低。",
        ))
    elif next_step_code in {"storm_defense", "weak_pressure_system"}:
        items.append(cause_item(
            "風險升級",
            "外部利空、夜盤壓力或日盤破線共同觸發潛伏賣壓。",
            [
                "避險情緒升高造成期貨先行壓低。",
                "關鍵防線跌破後停損與風控賣壓連鎖觸發。",
                "健康值下降且風險值升高，市場呼吸由波動轉病態。",
            ],
            f"跌破 {defense} 且收不回，並符合崩盤三重核對：{crash.get('confirmation', {}).get('summary', 'NA')}",
            "盤中跌破後快速收回，且隔日未再破低。",
        ))
    elif next_step_code == "radar_pending_ground_truth":
        items.append(cause_item(
            "盤前待驗",
            "夜盤與外部市場只提供雲圖，日盤現貨還沒給地面真相。",
            [
                f"夜日關係仍是 {relation}。",
                f"變異狀態仍是 {variance}。",
                "真正病因要等開盤缺口、日內高低差與收盤位置決定。",
            ],
            f"日盤完成後，以守 {defense}、站回 {reclaim}、收盤位置三項確認。",
            "日盤資料未完成前，排除任何收盤方向定論。",
        ))
    else:
        items.append(cause_item(
            "區間觀察",
            "多空尚未形成足夠觸發，先以高低點與收盤位置分流。",
            [
                "可能只是正常市場呼吸。",
                "也可能是高檔換手的前置整理。",
                "若外部觸發增加，才會轉成明確病灶。",
            ],
            f"突破 {reclaim} 或跌破 {defense} 並由收盤確認。",
            "高低點未突破且風險值未升高，維持觀察。",
        ))

    if active:
        items.append(cause_item(
            "已觸發外因",
            "外在條件已使某些潛伏病灶進入觀察或發作區。",
            active,
            "觸發因子延續到收盤，且與價格路徑同向。",
            "觸發因子消退，且日盤收回關鍵線。",
        ))
    if watch:
        items.append(cause_item(
            "待觸發外因",
            "目前只是在觀察名單，尚未足以單獨下結論。",
            watch,
            "觀察因子與日盤破線/站回同時成立。",
            "觀察因子沒有延續，或與日盤收盤方向相反。",
        ))

    return {
        "enabled": True,
        "method": "表層症狀 → 中層機制 → 根因候選 → 確認/排除條件",
        "headline": "逐層鑑別診斷",
        "items": items,
        "guardrail": "根因候選必須被後續資料確認或排除；不能用單一敘事直接取代實際市場證據。",
    }


def weather_satellite_headline(code: str) -> str:
    mapping = {
        "storm_defense": "風暴防守優先",
        "radar_pending_ground_truth": "盤前雲圖待地面驗證",
        "night_bullish_pending_cash": "夜盤偏多，日盤待驗",
        "night_bearish_pending_cash": "夜盤偏空，日盤待驗",
        "night_neutral_cash_decides": "夜盤無方向，日盤主導",
        "false_breakout_overtime": "假突破延長賽",
        "washout_reclaim_watch": "下探收復觀察",
        "controlled_advance": "可控偏多續攻",
        "weak_pressure_system": "弱勢氣壓增強",
        "range_observation": "區間天氣觀測",
    }
    return mapping.get(code, "市場天氣觀測")


def analyze_market_mode_switch(payload: dict) -> dict:
    """Top-level regime router. This guards against calm wording during early crisis."""
    crash = payload.get("crash_monitor", {})
    premarket = payload.get("premarket", {})
    bagua = payload.get("bagua_lifecycle", {})
    cause = payload.get("cause_analysis", {})
    memory = payload.get("memory_industry_risk", {})
    capital = payload.get("capital_flow", {})
    external_reset = payload.get("external_event_reset_monitor", {})
    drawdown_guardrail = crash.get("drawdown_guardrail", {})

    triggers: list[str] = []
    risk_score = 0

    if drawdown_guardrail.get("stage_code") in {"major_correction_policy_watch", "crash_precursor_deep_kan"}:
        risk_score += 4
        triggers.append(drawdown_guardrail.get("stage", "重大回撤"))
    if crash.get("alert_code") in {"confirmed_warning", "early_alert", "intraday_unconfirmed"}:
        risk_score += 4
        triggers.append(f"崩盤監控{crash.get('alert_level')}")
    elif safe_float(crash.get("risk_value")) is not None and safe_float(crash.get("risk_value")) >= 45:
        risk_score += 2
        triggers.append("風險值進入預警區")

    if premarket.get("is_premarket") and safe_int(premarket.get("total_score"), 0) <= -3:
        risk_score += 2
        triggers.append("夜盤/外部盤前偏空")
    if safe_float(premarket.get("tx_night_spread_per")) is not None and safe_float(premarket.get("tx_night_spread_per")) <= -0.015:
        risk_score += 2
        triggers.append("台指期夜盤明顯下跌")
    if safe_float(premarket.get("sox_return_1d")) is not None and safe_float(premarket.get("sox_return_1d")) <= -0.025:
        risk_score += 1
        triggers.append("費半急跌")
    if safe_float(premarket.get("vix_return_1d")) is not None and safe_float(premarket.get("vix_return_1d")) >= 0.08:
        risk_score += 1
        triggers.append("VIX急升")
    if external_reset.get("code") == "bearish_external_reset":
        risk_score += 3
        triggers.append("外部利空重置")
    elif external_reset.get("code") == "external_conflict_watch":
        risk_score += 1
        triggers.append("外部分歧重置觀察")
    elif external_reset.get("code") == "bullish_external_reset":
        triggers.append("外部利多重置")

    state_code = bagua.get("roles", {}).get("state_gua", {}).get("code")
    background_code = bagua.get("roles", {}).get("background_gua", {}).get("code")
    daily_code = bagua.get("daily", {}).get("code")
    if background_code in {"DUI", "QIAN"} and daily_code == "KAN":
        risk_score += 2
        triggers.append("高檔背景日線入坎")
    elif background_code in {"DUI", "QIAN"} and state_code in {"ZHEN", "KUN"}:
        triggers.append("高檔修復/換手")

    if memory.get("level") in {"high", "critical"}:
        risk_score += 2
        triggers.append("產業風險升高")
    elif memory.get("level") == "watch":
        risk_score += 1
        triggers.append("產業風險注意")

    if not capital.get("has_factor_values"):
        risk_score += 1
        triggers.append("法人/期權資料缺口")
    elif safe_int(capital.get("non_price_score"), 0) < 0:
        risk_score += 1
        triggers.append("法人或衍生品偏空")

    if risk_score >= 7:
        mode = "tail_risk_defense"
        headline = "危機升級：災變/尾端風險防守"
        action = "提高監控頻率；盤中破線先防守，收盤再確認。"
        crisis_early = True
    elif risk_score >= 4:
        mode = "early_crisis_watch"
        headline = "危機初期跡象"
        action = "不得用健康修復文字覆蓋風險；先看夜盤、外部市場、前低與收盤確認。"
        crisis_early = True
    elif background_code in {"DUI", "QIAN"} and state_code in {"ZHEN", "KUN", "DUI"}:
        mode = "high_level_rotation"
        headline = "高檔換手"
        action = "以夜盤牽日盤、現貨承接、45,000附近站穩度作主控。"
        crisis_early = False
    elif state_code in {"GEN", "ZHEN", "XUN"}:
        mode = "repair_confirmation"
        headline = "修復確認"
        action = "確認高低點墊高與均線收復，不把反彈直接當長多。"
        crisis_early = False
    else:
        mode = "normal_cycle"
        headline = "正常週期"
        action = "依日週月卦與技術波段正常追蹤。"
        crisis_early = False

    return {
        "mode": mode,
        "headline": headline,
        "risk_score": risk_score,
        "score_range": market_mode_score_range(risk_score),
        "crisis_early": crisis_early,
        "action": action,
        "triggers": unique_text(triggers),
        "rule": "總控模式優先於健康分數；危機初期組合成立時，報告頭條必須先提示風險，不能寫成太平盛世。",
    }


def analyze_close_cause_attribution(payload: dict) -> dict:
    """Explain the latest cash-session move with testable causes.

    This layer is deliberately diagnostic: it can lower/raise next-day watch
    posture, but it must not become an investment command or rewrite history.
    """
    premarket = payload.get("premarket", {})
    intraday = payload.get("intraday_tactical_monitor", {})
    sector = payload.get("sector_pressure_observation", {})
    external_reset = payload.get("external_event_reset_monitor", {})
    mode = payload.get("market_mode_switch", {})
    programmed = payload.get("programmed_pressure_pattern", {})
    candle = payload.get("candlestick_pattern", {})

    close_price = safe_float(intraday.get("live_price")) or safe_float(payload.get("input", {}).get("index"))
    open_price = safe_float(intraday.get("live_open"))
    high_price = safe_float(intraday.get("live_high"))
    low_price = safe_float(intraday.get("live_low"))
    prior_close = safe_float(intraday.get("last_close"))
    cash_return = safe_float(intraday.get("vs_prev_close"))
    if cash_return is None and close_price is not None and prior_close:
        cash_return = close_price / prior_close - 1
    gap_return = (open_price / prior_close - 1) if open_price is not None and prior_close else None
    intraday_return = (close_price / open_price - 1) if close_price is not None and open_price else None
    high_to_close = (close_price / high_price - 1) if close_price is not None and high_price else None
    close_position = safe_float(intraday.get("close_position"))
    if close_position is None and close_price is not None and high_price is not None and low_price is not None and high_price > low_price:
        close_position = max(0.0, min(1.0, (close_price - low_price) / (high_price - low_price)))

    nasdaq = safe_float(premarket.get("nasdaq_return_1d"))
    sox = safe_float(premarket.get("sox_return_1d"))
    sp500 = safe_float(premarket.get("sp500_return_1d"))
    vix = safe_float(premarket.get("vix_return_1d"))
    external_score = safe_int(premarket.get("external_score"), 0)
    sector_risk = safe_int(sector.get("risk_score"), 0)
    pressure_score = safe_int(programmed.get("current_score"), 0)

    external_tailwind = (
        external_reset.get("code") == "bullish_external_reset"
        or external_score >= 1
        or bool(
            (nasdaq is not None and nasdaq > 0)
            and (sox is not None and sox > 0)
            and (vix is None or vix < 0)
        )
    )
    external_bearish = external_reset.get("code") == "bearish_external_reset" or external_score <= -2
    cash_down = cash_return is not None and cash_return <= -0.003
    cash_up = cash_return is not None and cash_return >= 0.003
    gap_up_close_down = bool(
        gap_return is not None
        and gap_return >= 0.001
        and intraday_return is not None
        and intraday_return <= -0.004
        and (close_position is None or close_position <= 0.35)
    )
    close_below_46000 = close_price is not None and close_price < 46000
    held_45700 = low_price is not None and low_price >= 45700
    broke_45700 = low_price is not None and low_price < 45700
    close_near_low = close_position is not None and close_position <= 0.25

    evidence: list[str] = []
    cause_candidates: list[str] = []
    next_validation: list[str] = []
    risk_adjustment = 0

    def add_evidence(text: str) -> None:
        if text:
            evidence.append(text)

    add_evidence(f"日盤報酬 {pct(cash_return)}；開盤缺口 {pct(gap_return)}；開收 {pct(intraday_return)}。")
    if high_to_close is not None:
        add_evidence(f"高點至收盤回落 {pct(high_to_close)}，收盤位置 {pct(close_position)}。")
    if external_tailwind:
        add_evidence("外部市場偏順風或風險下降，台股若走弱需優先檢查內部換手。")
    if external_bearish:
        add_evidence("外部市場偏利空，需檢查是否為外部傳導。")
    if sector_risk >= 2:
        add_evidence(f"族群分化分數 {sector_risk}，弱勢族群：{'、'.join(sector.get('weak_sectors', [])[:5]) or 'NA'}。")
    if close_below_46000:
        add_evidence("收盤跌破 46,000 灘頭堡主防線。")
    if held_45700:
        add_evidence("盤中低點仍守在 45,700 之上，尚未確認主防線下緣失守。")

    if external_tailwind and cash_down and gap_up_close_down:
        code = "external_tailwind_internal_selloff"
        label = "外部順風但內部開高走低"
        headline = "外部不是主跌因；主因轉向高檔換手、族群分化與壓力測試。"
        cause_candidates.extend(
            [
                "外盤利多被早盤開高一次反映，追價買盤不足後轉為獲利了結。",
                "高檔關卡前先測上方賣壓，站不穩後引發短線浮額出場。",
                "權值或防禦族群可能撐住指數，但中小型與題材電子先行退潮。",
                "收破46,000代表前沿陣地退守，但未破45,700表示尚未形成崩壞確認。",
            ]
        )
        next_validation.extend(
            [
                "隔日先看45,700是否有效跌破；跌破後快速收回則列假跌破/清洗完成候選。",
                "隔日若收不回46,000，退守警戒延續；若站回46,200/46,350，空方壓測失敗。",
                "補法人、期貨未平倉與族群廣度，確認是內部換手還是出貨擴散。",
            ]
        )
        risk_adjustment = 2
    elif external_bearish and cash_down:
        code = "external_bearish_transmission"
        label = "外部利空傳導"
        headline = "外部風險與日盤下跌同向，內部劇本需暫時降權。"
        cause_candidates.extend(
            [
                "美股、半導體、VIX、利率或國際政治風險壓低風險胃納。",
                "若台股跌破關鍵防線且族群同步弱，外部衝擊可能轉成內部賣壓循環。",
            ]
        )
        next_validation.extend(
            [
                "隔日看外部風險是否緩和後台股能否收回前一日跌幅。",
                "若外部續弱且台股破低不回，升級為外部重置主導。",
            ]
        )
        risk_adjustment = 3
    elif cash_down and close_below_46000 and held_45700:
        code = "key_line_pressure_test"
        label = "46,000失守但45,700未破"
        headline = "短線轉弱，但仍屬防線壓力測試，等待隔日確認。"
        cause_candidates.extend(
            [
                "46,000整數關卡失守觸發短線防守，但45,700仍有承接。",
                "市場正在測試多方是否願意在主防線下緣接回。",
            ]
        )
        next_validation.extend(
            [
                "隔日守45,700並收回46,000，歸類為有效壓洗。",
                "隔日跌破45,700且收不回，歸類為退守45,500。",
            ]
        )
        risk_adjustment = 1
    elif cash_up and external_tailwind:
        code = "external_tailwind_confirmed"
        label = "外部順風被日盤確認"
        headline = "外部利多與日盤現貨同向，維持可控修復觀察。"
        cause_candidates.append("外盤、夜盤與日盤現貨方向同向，市場承接較健康。")
        next_validation.append("隔日看上漲是否由族群廣度擴散支持，而不是只靠少數權值。")
        risk_adjustment = -1
    else:
        code = "mixed_cause_watch"
        label = "漲跌原因混合待驗"
        headline = "外部、夜盤、日盤或族群訊號不足以給單一原因。"
        cause_candidates.extend(
            [
                "若日夜盤與外部訊號不同向，需等收盤、法人與族群廣度補證。",
                "短線方向不可由單一指標外推。",
            ]
        )
        next_validation.append("補齊法人、期貨、族群廣度與隔日收盤位置後再升降級。")

    if sector_risk >= 2 and code != "external_tailwind_internal_selloff":
        cause_candidates.append("族群分化已達觀察門檻，指數紅黑不可單獨代表市場健康。")
        risk_adjustment = max(risk_adjustment, 1)
    if pressure_score >= 5:
        cause_candidates.append("近期壓低測壓行為指紋偏明顯，需追蹤是否連續破低。")
    if close_near_low and cash_down:
        cause_candidates.append("收盤靠近日低，代表尾盤承接不足，隔日容易續測。")

    return {
        "enabled": close_price is not None or bool(evidence),
        "framework": "close_cause_attribution_v1",
        "code": code,
        "label": label,
        "headline": headline,
        "risk_adjustment": risk_adjustment,
        "model_effect": (
            "隔日方向信心降權，優先看關鍵線驗證。"
            if risk_adjustment > 0
            else "維持可控觀察。"
        ),
        "features": {
            "close": close_price,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "prior_close": prior_close,
            "cash_return": cash_return,
            "gap_return": gap_return,
            "intraday_return": intraday_return,
            "high_to_close": high_to_close,
            "close_position": close_position,
            "external_tailwind": external_tailwind,
            "external_bearish": external_bearish,
            "gap_up_close_down": gap_up_close_down,
            "close_below_46000": close_below_46000,
            "held_45700": held_45700,
            "broke_45700": broke_45700,
            "sector_risk_score": sector_risk,
            "market_mode": mode.get("mode"),
            "candlestick_type": candle.get("type"),
        },
        "evidence": unique_text(evidence),
        "cause_candidates": unique_text(cause_candidates),
        "next_validation": unique_text(next_validation),
        "guardrail": "漲跌原因歸因只作可驗證病因候選與模型降權；不得指認單一操盤者，不得產生買賣命令，不得事後回填舊預測。",
    }


def market_mode_score_range(score) -> str:
    value = safe_int(score, 0)
    if value >= 7:
        return "7分以上：紅色防守，尾端/災變風險；盤中破線先防守，收盤再確認。"
    if value >= 4:
        return "4-6分：危機初期，不能用健康修復文字覆蓋風險。"
    if value >= 2:
        return "2-3分：高敏感換手，方向未壞但需確認承接。"
    return "0-1分：低風險觸發，屬正常換手/輕微干擾；仍需看關鍵價位是否守住。"


def strong_target_definition() -> str:
    return (
        "強勢標的定義：相對大盤抗跌或領漲，收盤站上5日與20日線，回檔不破前低，"
        "量能溫和放大或縮量守穩，且所屬族群強於大盤；若只是開高急拉但收不住，不列強勢。"
    )


def score_risk_regime(value: str | None) -> int:
    return {"risk_on": 2, "neutral": 0, "risk_off": -2}.get(value or "neutral", 0)


def score_bagua_summary(bagua: dict) -> int:
    resonance = bagua.get("resonance", {}).get("label")
    primary = bagua.get("primary", {}).get("code")
    if resonance in ["高位入坎", "高位轉弱"]:
        return -2
    if resonance in ["多週期同向轉強", "低位轉強"]:
        return 2
    if primary in ["KAN"]:
        return -1
    if primary in ["GEN", "ZHEN", "XUN", "LI"]:
        return 1
    if primary in ["DUI", "QIAN"]:
        return -1
    return 0


def score_tradeable_cycle_summary(cycle: dict) -> int:
    code = cycle.get("current_position", {}).get("code")
    frequency = cycle.get("stats", {}).get("frequency_state", {}).get("level")
    base = {
        "base_building": -1,
        "repair_watch": 1,
        "early_rally": 2,
        "main_rally": 2,
        "late_rally": -1,
        "pullback_watch": -1,
        "post_rally_pullback": -2,
    }.get(code, 0)
    if frequency in ["high", "extreme_high"] and base <= 0:
        base -= 1
    return max(-3, min(3, base))


def score_candlestick_summary(candle: dict) -> int:
    return {
        "gap_down_reversal": 2,
        "long_lower_shadow": 2,
        "long_bull": 2,
        "gap_up_failed": -2,
        "long_upper_shadow": -1,
        "long_bear": -2,
        "doji": 0,
        "normal": 0,
    }.get(candle.get("type"), 0)


def score_washout_summary(washout: dict) -> int:
    return {
        "strong_washout": 2,
        "normal_washout": 1,
        "gap_down_recovery": 1,
        "failed_washout": -2,
        "no_washout": 0,
    }.get(washout.get("type"), 0)


def score_memory_summary(memory: dict) -> int:
    return {"low": 0, "watch": -1, "high": -2, "critical": -3}.get(memory.get("level"), 0)


def append_evidence(score: int, bullish: list[str], bearish: list[str], neutral: list[str], text: str) -> None:
    if score > 0:
        bullish.append(text)
    elif score < 0:
        bearish.append(text)
    else:
        neutral.append(text)


def integrated_bias(score: int) -> str:
    if score >= 5:
        return "bullish"
    if score >= 2:
        return "slightly_bullish"
    if score <= -5:
        return "bearish"
    if score <= -2:
        return "slightly_bearish"
    return "mixed"


def integrated_bias_text(value: str) -> str:
    return {
        "bullish": "偏多(bullish，綜合訊號支持上攻)",
        "slightly_bullish": "震盪偏多(slightly_bullish，有利多但仍需確認)",
        "mixed": "多空混合(mixed，方向需等待確認)",
        "slightly_bearish": "震盪偏空(slightly_bearish，短線偏防守)",
        "bearish": "偏空(bearish，綜合訊號偏向修正或防守)",
    }.get(value, value)


def integrated_confidence(score: int, missing: list[str], consistency: dict) -> str:
    strength = abs(score)
    if consistency.get("contradiction_count", 0):
        return "低(low，邏輯仍需修正)"
    if missing and strength >= 8:
        return "中(medium，方向訊號同向但重要籌碼資料缺漏)"
    if missing:
        return "中低(medium_low，重要籌碼資料缺漏)"
    if strength >= 8:
        return "高(high，多數訊號同向)"
    if strength >= 4:
        return "中(medium，訊號有方向但仍需確認)"
    return "低(low，多空接近或資料不足)"


def apply_direction_reliability_confidence_cap(confidence: str, policy: dict) -> str:
    level = policy.get("level")
    if level == "severe_degrade":
        return "低(low，短線方向可靠度退化，風控升權)"
    if level in {"degrade", "reduced", "unknown"} and confidence.startswith("高"):
        return "中(medium，方向可靠度降權後上限)"
    if level in {"degrade", "reduced", "unknown"} and confidence.startswith("中"):
        return "中低(medium_low，方向可靠度降權)"
    return confidence


def integrated_plain_conclusion(bias: str, score: int, bullish: list[str], bearish: list[str], missing: list[str]) -> str:
    if bias == "bullish":
        return "綜合所有訊號後，目前偏多，若外部市場與台指期夜盤不轉弱，後續較容易延續上攻或震盪墊高。"
    if bias == "slightly_bullish":
        return "綜合判斷為震盪偏多，但仍要等現貨突破確認；若追價失敗，容易回到區間震盪。"
    if bias == "bearish":
        return "綜合所有訊號後，目前偏空，代表賣壓、風險或高位轉弱因素較集中，操作上應先防守。"
    if bias == "slightly_bearish":
        return "綜合判斷為震盪偏空，短線不宜只看反彈，需等夜盤、外部市場與K線確認轉強。"
    if missing and not bullish and not bearish:
        return "目前關鍵資料不足，綜合判斷只能視為等待確認，不能把單一價格訊號當成完整預測。"
    return "多空訊號混合，代表市場正在換手或等待新事件確認；此時重點是確認價與失效價，而不是硬猜方向。"


def integrated_confirmation_rules(payload: dict, bias: str) -> list[str]:
    candle = payload.get("candlestick_pattern", {}).get("confirmation", {})
    washout = payload.get("washout_pattern", {}).get("invalidation", {})
    premarket = payload.get("premarket", {})
    rules = []
    if candle.get("confirm_rule"):
        rules.append(candle["confirm_rule"])
    if bias in ["bearish", "slightly_bearish"]:
        rules.append("若隔日不再跌破今日低點，且台指期夜盤轉強，偏空判斷才可降級。")
    elif bias in ["bullish", "slightly_bullish"]:
        rules.append("若隔日站穩今日高點或關鍵轉強價，偏多判斷才算延續。")
    else:
        rules.append("需等待隔日突破今日高點或跌破今日低點，才有較明確方向。")
    if premarket.get("is_premarket") and premarket.get("resistance_levels"):
        rules.append(f"盤前轉強觀察: {format_levels(premarket['resistance_levels'])}")
    if washout.get("rule") and "偏空" in integrated_bias_text(bias):
        rules.append(f"洗盤反證: {washout['rule']}")
    return unique_text(rules)


def integrated_invalidation_rules(payload: dict, bias: str) -> list[str]:
    candle = payload.get("candlestick_pattern", {}).get("confirmation", {})
    washout = payload.get("washout_pattern", {}).get("invalidation", {})
    premarket = payload.get("premarket", {})
    rules = []
    if candle.get("fail_rule"):
        rules.append(candle["fail_rule"])
    if washout.get("rule"):
        rules.append(washout["rule"])
    if bias in ["bullish", "slightly_bullish"]:
        rules.append("若外部市場轉弱、台指期夜盤續跌，且現貨跌破今日低點，偏多判斷失效。")
    elif bias in ["bearish", "slightly_bearish"]:
        rules.append("若指數站回長黑K中段以上並放量，偏空判斷需降級。")
    else:
        rules.append("若連續兩日同方向突破，盤整判斷失效，需改採趨勢判斷。")
    if premarket.get("is_premarket") and premarket.get("defense_levels"):
        rules.append(f"盤前防守觀察: {format_levels(premarket['defense_levels'])}")
    return unique_text(rules)


def top_items(values: list[str], limit: int) -> list[str]:
    return unique_text([value for value in values if value])[:limit]


def unique_text(values: list[str]) -> list[str]:
    output = []
    for value in values:
        if value and value not in output:
            output.append(value)
    return output


def safe_int(value, default: int = 0) -> int:
    if pd.isna(value):
        return default
    return int(value)


def analyze_washout_pattern(scored: pd.DataFrame, signal_date: str) -> dict:
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    for horizon in [1, 5, 10, 20, 60]:
        data[f"fwd_{horizon}d"] = data["close"].shift(-horizon) / data["close"] - 1

    row = data[data["date"] <= pd.to_datetime(signal_date)].iloc[-1]
    pattern = classify_washout(row)
    stats = washout_historical_stats(data, pattern["type"])
    invalidation = washout_invalidation(row, pattern)
    event_trigger = washout_event_trigger(row)
    return {
        "signal_date": str(row["date"].date()),
        "type": pattern["type"],
        "label": pattern["label"],
        "plain_summary": pattern["summary"],
        "is_washout": pattern["type"] in ["strong_washout", "normal_washout", "gap_down_recovery"],
        "event_trigger": event_trigger,
        "invalidation": invalidation,
        "today_metrics": {
            "open_gap_pct": safe_float(row.get("open_gap_pct")),
            "intraday_low_pct": safe_float(row.get("intraday_low_pct")),
            "close_return_pct": safe_float(row.get("close_return_pct")),
            "close_recovery_ratio": safe_float(row.get("close_recovery_ratio")),
            "external_score": int(row.get("external_score", 0)),
            "night_futures_score": int(row.get("night_futures_score", 0)),
        },
        "historical_stats": stats,
    }


def analyze_candlestick_pattern(scored: pd.DataFrame, signal_date: str) -> dict:
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    if "open_gap_pct" not in data:
        data["open_gap_pct"] = data["open"] / data["close"].shift(1) - 1
    if "close_return_pct" not in data:
        data["close_return_pct"] = data["close"] / data["close"].shift(1) - 1
    add_candlestick_metrics(data)
    for horizon in [1, 5, 10, 20, 60]:
        data[f"fwd_{horizon}d"] = data["close"].shift(-horizon) / data["close"] - 1
    row = data[data["date"] <= pd.to_datetime(signal_date)].iloc[-1]
    pattern = classify_candlestick(row)
    stats = candlestick_historical_stats(data, pattern["type"])
    return {
        "signal_date": str(row["date"].date()),
        "type": pattern["type"],
        "label": pattern["label"],
        "plain_summary": pattern["summary"],
        "position_text": candlestick_position_text(row),
        "confirmation": candlestick_confirmation(row, pattern),
        "today_metrics": {
            "body_pct": safe_float(row.get("k_body_pct")),
            "upper_shadow_pct": safe_float(row.get("k_upper_shadow_pct")),
            "lower_shadow_pct": safe_float(row.get("k_lower_shadow_pct")),
            "body_to_range": safe_float(row.get("k_body_to_range")),
            "close_position": safe_float(row.get("k_close_position")),
            "open_gap_pct": safe_float(row.get("open_gap_pct")),
            "close_return_pct": safe_float(row.get("close_return_pct")),
        },
        "historical_stats": stats,
    }


def add_candlestick_metrics(data: pd.DataFrame) -> None:
    high_low = (data["high"] - data["low"]).replace(0, pd.NA)
    body = data["close"] - data["open"]
    upper_shadow = data["high"] - data[["open", "close"]].max(axis=1)
    lower_shadow = data[["open", "close"]].min(axis=1) - data["low"]
    data["k_body_pct"] = body / data["open"]
    data["k_abs_body_pct"] = body.abs() / data["open"]
    data["k_upper_shadow_pct"] = upper_shadow / data["open"]
    data["k_lower_shadow_pct"] = lower_shadow / data["open"]
    data["k_body_to_range"] = body.abs() / high_low
    data["k_close_position"] = (data["close"] - data["low"]) / high_low


def classify_candlestick(row) -> dict:
    body = row.get("k_body_pct")
    abs_body = row.get("k_abs_body_pct")
    upper = row.get("k_upper_shadow_pct")
    lower = row.get("k_lower_shadow_pct")
    body_to_range = row.get("k_body_to_range")
    close_pos = row.get("k_close_position")
    open_gap = row.get("open_gap_pct")
    close_ret = row.get("close_return_pct")
    open_gap = 0 if pd.isna(open_gap) else open_gap
    close_ret = 0 if pd.isna(close_ret) else close_ret

    if pd.isna(body) or pd.isna(abs_body):
        return candle_pattern("unknown", "資料不足", "今日 K 線資料不足，無法判斷。")
    if (
        open_gap >= 0.001
        and close_ret <= -0.003
        and close_pos <= 0.25
        and body <= -0.004
        and upper >= 0.004
    ):
        return candle_pattern(
            "gap_up_close_near_low",
            "開高壓回收近低",
            "早盤承接夜盤或外部偏多，但日盤追價失敗，收盤貼近低點，代表現貨賣壓或獲利了結主導。",
        )
    if open_gap >= 0.008 and close_ret <= -0.003:
        return candle_pattern("gap_up_failed", "開高走低", "早盤偏多但收盤轉弱，容易代表追價失敗或上方賣壓。")
    if open_gap <= -0.008 and close_ret >= 0.003:
        return candle_pattern("gap_down_reversal", "開低走高", "早盤受壓但收盤轉強，代表低檔承接明確。")
    if lower >= 0.015 and close_pos >= 0.65:
        return candle_pattern("long_lower_shadow", "長下影線", "盤中曾急跌但收盤拉回，代表低檔有買盤承接。")
    if upper >= 0.015 and close_pos <= 0.45:
        return candle_pattern("long_upper_shadow", "長上影線", "盤中曾急拉但收盤壓回，代表上方賣壓或追價退潮。")
    if body >= 0.012 and body_to_range >= 0.55:
        return candle_pattern("long_bull", "長紅 K", "買盤主導，收盤明顯高於開盤，短線氣勢偏多。")
    if body <= -0.012 and body_to_range >= 0.55:
        return candle_pattern("long_bear", "長黑 K", "賣壓主導，收盤明顯低於開盤，短線氣勢偏空。")
    if abs_body <= 0.0025 and body_to_range <= 0.25:
        return candle_pattern("doji", "十字線", "開收接近，代表多空猶豫，常需要隔日確認方向。")
    return candle_pattern("normal", "一般 K 線", "今日 K 線沒有明顯極端型態，需搭配趨勢、夜盤與外部市場判斷。")


def candle_pattern(kind: str, label: str, summary: str) -> dict:
    return {"type": kind, "label": label, "summary": summary}


def candlestick_position_text(row) -> str:
    drawdown = row.get("drawdown")
    if pd.isna(drawdown):
        return "位置資料不足"
    if drawdown > -0.05:
        return "高檔區或接近高點"
    if drawdown < -0.15:
        return "低檔區或深度回檔"
    return "中段整理區"


def candlestick_confirmation(row, pattern: dict) -> dict:
    high = safe_float(row.get("high"))
    low = safe_float(row.get("low"))
    close = safe_float(row.get("close"))
    kind = pattern["type"]
    if kind in ["long_lower_shadow", "gap_down_reversal"]:
        return {
            "confirm_rule": "隔日守住今日低點，並站回今日收盤，代表承接延續。",
            "fail_rule": "隔日跌破今日低點，代表承接失敗。",
            "confirm_level": close,
            "fail_level": low,
        }
    if kind in ["long_upper_shadow", "gap_up_failed"]:
        return {
            "confirm_rule": "隔日突破今日高點，代表賣壓被吸收。",
            "fail_rule": "隔日跌破今日收盤，代表追價失敗延續。",
            "confirm_level": high,
            "fail_level": close,
        }
    if kind == "long_bull":
        return {
            "confirm_rule": "隔日守住今日收盤或續創高，代表買盤延續。",
            "fail_rule": "隔日跌回今日 K 線中段，代表強勢降溫。",
            "confirm_level": high,
            "fail_level": (high + low) / 2 if high is not None and low is not None else None,
        }
    if kind == "long_bear":
        return {
            "confirm_rule": "隔日站回今日 K 線中段，代表賣壓緩和。",
            "fail_rule": "隔日續破今日低點，代表賣壓延續。",
            "confirm_level": (high + low) / 2 if high is not None and low is not None else None,
            "fail_level": low,
        }
    return {
        "confirm_rule": "隔日突破今日高低點，才有較明確方向。",
        "fail_rule": "若仍在今日區間內，代表盤整延續。",
        "confirm_level": high,
        "fail_level": low,
    }


def candlestick_historical_stats(data: pd.DataFrame, kind: str) -> dict:
    prepared = data[data["date"] >= pd.to_datetime("2005-01-01")].copy()
    prepared = prepared.dropna(subset=["fwd_5d", "fwd_20d"])
    masks = {
        "gap_up_failed": (prepared["open_gap_pct"] >= 0.008) & (prepared["close_return_pct"] <= -0.003),
        "gap_down_reversal": (prepared["open_gap_pct"] <= -0.008) & (prepared["close_return_pct"] >= 0.003),
        "long_lower_shadow": (prepared["k_lower_shadow_pct"] >= 0.015) & (prepared["k_close_position"] >= 0.65),
        "long_upper_shadow": (prepared["k_upper_shadow_pct"] >= 0.015) & (prepared["k_close_position"] <= 0.45),
        "long_bull": (prepared["k_body_pct"] >= 0.012) & (prepared["k_body_to_range"] >= 0.55),
        "long_bear": (prepared["k_body_pct"] <= -0.012) & (prepared["k_body_to_range"] >= 0.55),
        "doji": (prepared["k_abs_body_pct"] <= 0.0025) & (prepared["k_body_to_range"] <= 0.25),
        "normal": pd.Series(True, index=prepared.index),
    }
    mask = masks.get(kind)
    if mask is None:
        return {"sample_count": 0, "message": "此 K 線型態無專屬歷史統計。", "by_horizon": []}
    sample = prepared[mask].copy()
    rows = []
    for horizon in [1, 5, 10, 20, 60]:
        col = f"fwd_{horizon}d"
        values = sample[col].dropna()
        if values.empty:
            continue
        rows.append(
            {
                "horizon_days": horizon,
                "avg_return": float(values.mean()),
                "up_rate": float((values > 0).mean()),
                "down_rate": float((values < -0.005).mean()),
            }
        )
    return {
        "sample_count": int(len(sample)),
        "sample_ratio": float(len(sample) / len(prepared)) if len(prepared) else 0,
        "by_horizon": rows,
    }


def analyze_technical_phase(
    scored: pd.DataFrame,
    signal_date: str,
    bagua: dict,
    candle: dict,
    cycle: dict,
) -> dict:
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["date"] <= pd.to_datetime(signal_date)].dropna(subset=["close"]).reset_index(drop=True)
    if len(data) < 60:
        return {"enabled": False, "message": "資料不足，無法計算技術波段檢討。"}
    for column in ["open", "high", "low", "close"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for window in [5, 10, 20, 60]:
        data[f"ma{window}"] = data["close"].rolling(window).mean()
    row = data.iloc[-1]
    recent = data.tail(10)
    swing = data.tail(60)
    swing_high_row = swing.loc[swing["high"].idxmax()]
    swing_low_after_high = data[data["date"] >= swing_high_row["date"]].loc[
        data[data["date"] >= swing_high_row["date"]]["low"].idxmin()
    ]
    close = safe_float(row.get("close"))
    high = safe_float(row.get("high"))
    low = safe_float(row.get("low"))
    open_ = safe_float(row.get("open"))
    prev_close = safe_float(data.iloc[-2].get("close")) if len(data) >= 2 else None
    ma = {f"ma{window}": safe_float(row.get(f"ma{window}")) for window in [5, 10, 20, 60]}
    below_ma = [name for name, value in ma.items() if close is not None and value is not None and close < value]
    prior_recent = data.iloc[:-1].tail(10)
    broke_recent_low = (
        low is not None
        and not prior_recent.empty
        and low < safe_float(prior_recent["low"].min())
    )
    broke_recent_close = (
        close is not None
        and not prior_recent.empty
        and close < safe_float(prior_recent["close"].min())
    )
    drawdown_from_swing_high = (
        close / safe_float(swing_high_row.get("high")) - 1
        if close is not None and safe_float(swing_high_row.get("high")) not in [None, 0]
        else None
    )
    day_range_pct = high / low - 1 if high not in [None, 0] and low not in [None, 0] else None
    day_range_vs_prev_close = high / prev_close - low / prev_close if prev_close not in [None, 0] and high is not None and low is not None else None
    five_day_range = data.tail(5)
    twenty_day_range = data.tail(20)
    range_5d = (
        safe_float(five_day_range["high"].max()) / safe_float(five_day_range["low"].min()) - 1
        if not five_day_range.empty and safe_float(five_day_range["low"].min()) not in [None, 0]
        else None
    )
    range_20d = (
        safe_float(twenty_day_range["high"].max()) / safe_float(twenty_day_range["low"].min()) - 1
        if not twenty_day_range.empty and safe_float(twenty_day_range["low"].min()) not in [None, 0]
        else None
    )
    if len(below_ma) == 4 and (broke_recent_low or broke_recent_close):
        label = "下行波段確認"
        status_code = "downtrend_confirmed"
        bias = "bearish"
        summary = "收盤跌破5/10/20/60日均線，且跌破近期低點或低收盤，成熟技術分析支持下行波段尚未止住。"
    elif len(below_ma) >= 3:
        label = "下行波段"
        status_code = "downtrend"
        bias = "bearish"
        summary = "多條均線失守，技術面偏空；若未收復短均，反彈先視為弱修復。"
    elif close is not None and ma["ma20"] is not None and close >= ma["ma20"]:
        label = "修復觀察"
        status_code = "repair_watch"
        bias = "neutral"
        summary = "收盤重新站回月線附近，技術面開始修復，但仍需確認高低點結構。"
    else:
        label = "震盪未明"
        status_code = "range_unclear"
        bias = "neutral"
        summary = "技術面尚未形成單邊確認，需用均線與前低追蹤。"

    monthly = bagua.get("monthly", {})
    weekly = bagua.get("weekly", {})
    daily = bagua.get("daily", {})
    forecast_alignment = technical_forecast_alignment(
        bias,
        bagua,
        candle,
        cycle,
    )
    return {
        "enabled": True,
        "signal_date": str(row["date"].date()),
        "label": label,
        "status_code": status_code,
        "bias": bias,
        "summary": summary,
        "mature_technical_basis": [
            "均線結構: 5/10/20/60日均線用來判斷短中期趨勢是否同步。",
            "高低點結構: 跌破近期低點或低收盤代表下行波段延續。",
            "K線結構: 長黑且收在低位，代表賣壓尚未被吸收。",
            "回撤結構: 從波段高點回撤幅度用來判斷是否仍在高峰後陷落段。",
        ],
        "levels": {
            "close": close,
            "open": open_,
            "high": high,
            "low": low,
            **ma,
            "recent_low": safe_float(prior_recent["low"].min()) if not prior_recent.empty else None,
            "recent_low_close": safe_float(prior_recent["close"].min()) if not prior_recent.empty else None,
            "swing_high_date": str(pd.to_datetime(swing_high_row["date"]).date()),
            "swing_high": safe_float(swing_high_row.get("high")),
            "low_after_swing_high_date": str(pd.to_datetime(swing_low_after_high["date"]).date()),
            "low_after_swing_high": safe_float(swing_low_after_high.get("low")),
            "drawdown_from_swing_high": safe_float(drawdown_from_swing_high),
            "day_range_pct": safe_float(day_range_pct),
            "day_range_vs_prev_close": safe_float(day_range_vs_prev_close),
            "range_5d": safe_float(range_5d),
            "range_20d": safe_float(range_20d),
        },
        "signals": {
            "below_ma_count": len(below_ma),
            "below_mas": below_ma,
            "broke_recent_low": bool(broke_recent_low),
            "broke_recent_close": bool(broke_recent_close),
            "candlestick": candle.get("label"),
            "cycle_position": cycle.get("current_position", {}).get("label"),
        },
        "bagua_timeframe_roles": {
            "monthly": {
                "gua": monthly.get("gua"),
                "label": monthly.get("label"),
                "role": "整波段背景，用來判斷大方向仍在高位承載、極盛、入坎或築底。",
            },
            "weekly": {
                "gua": weekly.get("gua"),
                "label": weekly.get("label"),
                "role": "月卦波段中的鐘擺與轉折，用來判斷加速下滑、修復或止跌雛形。",
            },
            "daily": {
                "gua": daily.get("gua"),
                "label": daily.get("label"),
                "role": "短線確認，用來驗證週卦轉折是否已落到日K。",
            },
        },
        "fixed_formula": build_fixed_kline_formula(
            label,
            monthly,
            weekly,
            daily,
            cycle,
            candle,
            close,
            low,
            swing_high_row,
            drawdown_from_swing_high,
            below_ma,
            broke_recent_low,
            broke_recent_close,
            day_range_pct,
            range_5d,
            range_20d,
        ),
        "equation_answer": build_technical_equation_answer(
            label,
            bias,
            monthly,
            weekly,
            daily,
            cycle,
            candle,
            close,
            ma,
            below_ma,
            broke_recent_low,
            broke_recent_close,
            drawdown_from_swing_high,
            day_range_pct,
            range_20d,
        ),
        "alignment": forecast_alignment,
    }


def build_fixed_kline_formula(
    label: str,
    monthly: dict,
    weekly: dict,
    daily: dict,
    cycle: dict,
    candle: dict,
    close,
    low,
    swing_high_row,
    drawdown_from_swing_high,
    below_ma: list[str],
    broke_recent_low: bool,
    broke_recent_close: bool,
    day_range_pct,
    range_5d,
    range_20d,
) -> list[dict]:
    cycle_position = cycle.get("current_position", {})
    return [
        {
            "step": "1. 主波段",
            "result": (
                f"{swing_high_row['date'].date()} 高點 {num(swing_high_row.get('high'))} "
                f"後回撤 {pct(drawdown_from_swing_high)}，月卦 {monthly.get('gua')}/{monthly.get('label')}"
            ),
        },
        {
            "step": "2. 中波段",
            "result": (
                f"週卦 {weekly.get('gua')}/{weekly.get('label')}，"
                f"波段段位 {cycle_position.get('label')}，{cycle_position.get('plain_summary')}"
            ),
        },
        {
            "step": "3. 日K",
            "result": (
                f"日卦 {daily.get('gua')}/{daily.get('label')}，K線 {candle.get('label')}，"
                f"{candle.get('plain_summary')}"
            ),
        },
        {
            "step": "4. 均線",
            "result": f"跌破 {len(below_ma)} 條均線 ({', '.join(below_ma) if below_ma else '無'})。",
        },
        {
            "step": "5. 震幅",
            "result": (
                f"單日震幅 {pct(day_range_pct)}，5日震幅 {pct(range_5d)}，20日震幅 {pct(range_20d)}。"
            ),
        },
        {
            "step": "6. 最後位置",
            "result": (
                f"{label}；破近期低點 {'是' if broke_recent_low else '否'}，"
                f"破近期低收盤 {'是' if broke_recent_close else '否'}，目前低點 {num(low)}、收盤 {num(close)}。"
            ),
        },
    ]


def build_technical_equation_answer(
    technical_label: str,
    technical_bias: str,
    monthly: dict,
    weekly: dict,
    daily: dict,
    cycle: dict,
    candle: dict,
    close,
    ma: dict,
    below_ma: list[str],
    broke_recent_low: bool,
    broke_recent_close: bool,
    drawdown_from_swing_high,
    day_range_pct,
    range_20d,
) -> dict:
    """Summarize which technical equation currently gives the clearest answer."""
    equations: list[dict] = []

    def add(name: str, score: int, answer: str, reason: str) -> None:
        equations.append({"name": name, "score": score, "answer": answer, "reason": reason})

    high_stage = {monthly.get("code"), weekly.get("code")} & {"QIAN", "LI", "DUI", "KUN"}
    daily_code = daily.get("code")
    if high_stage and daily_code in {"ZHEN", "DUI"}:
        add(
            "卦位方程式",
            1,
            "高位修復候選",
            f"月週位於高檔/極盛，日線 {daily.get('gua')}/{daily.get('label')}，代表可修復但追價風險仍在。",
        )
    elif daily_code in {"KAN", "GEN"}:
        add(
            "卦位方程式",
            -2,
            "回測/築底候選",
            f"日線 {daily.get('gua')}/{daily.get('label')}，短線仍需先確認止跌。",
        )
    else:
        add(
            "卦位方程式",
            0,
            "卦位中性待驗",
            f"月週日 {monthly.get('gua')}/{weekly.get('gua')}/{daily.get('gua')} 未形成強單邊答案。",
        )

    ma20 = safe_float(ma.get("ma20"))
    close_value = safe_float(close)
    if len(below_ma) >= 3:
        add("均線方程式", -3, "轉弱優先", f"收盤跌破 {len(below_ma)} 條均線，趨勢防線偏弱。")
    elif close_value is not None and ma20 is not None and close_value >= ma20:
        add("均線方程式", 2, "修復成立候選", "收盤站回20日線，技術修復得到均線支持。")
    else:
        add("均線方程式", 0, "均線未定", "均線沒有形成強多或強空答案。")

    candle_type = candle.get("type")
    close_position = safe_float(candle.get("close_position"))
    if candle_type in {"long_bear", "gap_up_failed"} or (close_position is not None and close_position <= 0.25):
        add("K線方程式", -2, "賣壓未解", f"K線 {candle.get('label')}，收盤位置偏低，需防續壓。")
    elif candle_type in {"long_bull", "gap_down_reversal", "long_lower_shadow"} or (
        close_position is not None and close_position >= 0.65
    ):
        add("K線方程式", 2, "承接修復", f"K線 {candle.get('label')}，收盤位置偏高，承接力較佳。")
    else:
        add("K線方程式", 0, "K線待驗", f"K線 {candle.get('label')} 未形成強確認。")

    if broke_recent_low and broke_recent_close:
        add("破線方程式", -3, "破底轉弱", "近期低點與低收盤同步失守，0劇本權重提高。")
    elif broke_recent_low and not broke_recent_close:
        add("破線方程式", 1, "刺破收回", "盤中破近期低點但收盤未破低收盤，屬轉機候選。")
    else:
        add("破線方程式", 1, "未破底", "近期低點/低收盤尚未同步失守，防線仍有效。")

    if range_20d is not None and range_20d >= 0.08:
        add("波動方程式", -1, "高波動降信心", f"20日震幅 {pct(range_20d)}，代表換手與洗盤頻繁，方向信心需降權。")
    elif day_range_pct is not None and day_range_pct <= 0.008:
        add("波動方程式", 0, "窄幅待變", f"單日震幅 {pct(day_range_pct)}，尚未表態。")
    else:
        add("波動方程式", 0, "波動中性", f"單日震幅 {pct(day_range_pct)}，20日震幅 {pct(range_20d)}。")

    total_score = sum(item["score"] for item in equations)
    winning = max(equations, key=lambda item: abs(item["score"])) if equations else {}

    if total_score >= 3:
        answer = "答案偏1：修復/續攻候選"
        branch = "1"
        summary = "多公式合成後偏向修復，但仍須日盤收盤與量能確認。"
    elif total_score <= -3:
        answer = "答案偏0：回測/轉弱候選"
        branch = "0"
        summary = "多公式合成後偏向回測或轉弱，需優先核對防守線。"
    else:
        answer = "答案未定：0/1拉鋸"
        branch = "0/1"
        summary = "各方程式互相抵銷，目前只能用關鍵線與下一根K線分流。"

    return {
        "framework": "technical_equation_answer_v1",
        "answer": answer,
        "branch": branch,
        "score": total_score,
        "best_formula": winning.get("name", "NA"),
        "best_formula_answer": winning.get("answer", "NA"),
        "best_formula_reason": winning.get("reason", "NA"),
        "summary": summary,
        "technical_label": technical_label,
        "technical_bias": technical_bias,
        "equations": equations,
        "rule": "計算式必須保留，但主報告以多公式仲裁後的答案為主；哪個公式最貼近收盤事實，後續由病歷驗證提高或降低權重。",
        "guardrail": "方程式答案只作技術研判與風控分流，不產生投資命令。",
    }


def build_equation_reasoning_audit(payload: dict) -> dict:
    """Compare formula output with reasoning layers before trusting the answer."""
    technical_answer = payload.get("technical_phase", {}).get("equation_answer", {})
    zero_one = payload.get("weather_satellite_forecast", {}).get("zero_one_tilt", {})
    practical = payload.get("practical_cause_arbitration", {})
    arbitration = payload.get("master_arbitration", {})
    day_night = payload.get("day_night_variance_pattern", {})
    protection = payload.get("market_protection_layers", {})

    formula_branch = str(technical_answer.get("branch", "NA"))
    reasoning_branch = str(zero_one.get("branch", "NA"))
    agreements: list[str] = []
    conflicts: list[str] = []
    checks: list[str] = []
    error_sources: list[str] = []
    missing_variables: list[str] = []
    candidate_variables: list[str] = []

    if formula_branch in {"0", "1"} and reasoning_branch in {"0", "1"}:
        if formula_branch == reasoning_branch:
            agreements.append(f"技術方程式與0/1推理同向，皆偏 {formula_branch}。")
        else:
            conflicts.append(f"技術方程式偏 {formula_branch}，0/1推理偏 {reasoning_branch}，需等待日盤裁判。")
            error_sources.append("技術公式與推理分支不同，可能是均線/卦位落後、夜盤前哨過度反應或日盤尚未裁判。")
    else:
        conflicts.append("技術或0/1其中一方尚未給出明確分支，不能提高信心。")
        error_sources.append("公式或推理缺少明確分支，誤差來源可能是資料不足或多空接近。")

    if practical.get("practical_primary") in {"internal_structure", "external_reset"}:
        agreements.append(f"實務病因已有主控層: {practical.get('label', 'NA')}。")
    else:
        conflicts.append("實務病因未給明確主控，公式答案需降權。")
        missing_variables.append("主控病因不明: 需補法人、融資、期貨未平倉、選擇權壓力與族群廣度。")

    if day_night.get("relation_code") in {"night_down_cash_down_validation", "night_down_cash_reversal"}:
        agreements.append(f"日夜盤傳導已有收盤驗證: {day_night.get('label', 'NA')}。")
    else:
        checks.append("日夜盤仍待日盤收盤驗證，公式答案只能列為候選。")
        missing_variables.append("日夜盤傳導未完成: 需等日盤開盤、30～60分鐘、低點與收盤位置。")

    if protection.get("failed_layers"):
        conflicts.append("市場保護層出現失效，任何修復公式需降權。")
        error_sources.append("保護層失效會讓一般技術公式失真，需切換到風控模型。")
    elif protection.get("label"):
        agreements.append(f"保護層判讀: {protection.get('label')}。")

    candidate_variables.extend(
        [
            "期現差收斂/擴大: 判斷期貨領先是否被現貨承認。",
            "台指期未平倉與大額交易人淨部位: 判斷避險、放空或回補壓力。",
            "選擇權Put/Call、最大痛點與Gamma壓力: 判斷關鍵價位附近是否有被動避險流。",
            "上市/櫃買電子、金融、半導體等族群廣度: 判斷指數是否只靠權值撐盤。",
            "成交金額與上漲成交量占比: 判斷修復是換手承接還是量縮反彈。",
            "13:15後尾盤期現差與收盤位置: 判斷隔夜倉位是否提前表態。",
            "國際事件日曆: CPI、FOMC、非農、結算日、假期與財報周。",
        ]
    )

    checks.extend(
        [
            "下一交易日先看公式答案是否被開盤方向承認。",
            "再看30～60分鐘是否收回或跌破關鍵線。",
            "最後用收盤位置、量能與族群廣度判定公式勝負。",
        ]
    )

    agreement_score = len(agreements) - len(conflicts)
    if agreement_score >= 2:
        label = "公式與推理大致一致"
        trust = "medium"
        conclusion = "可採用公式答案作為主劇本候選，但仍需收盤驗證。"
    elif agreement_score <= -1:
        label = "公式與推理衝突"
        trust = "low"
        conclusion = "公式答案不得單獨採用，需等待日盤裁判或降權。"
    else:
        label = "公式與推理待驗"
        trust = "low_to_medium"
        conclusion = "公式與推理尚未形成足夠共識，只能作0/1分流參考。"

    return {
        "framework": "equation_reasoning_audit_v1",
        "label": label,
        "trust": trust,
        "agreement_score": agreement_score,
        "formula_answer": technical_answer.get("answer", "NA"),
        "formula_branch": formula_branch,
        "reasoning_answer": zero_one.get("label", "NA"),
        "reasoning_branch": reasoning_branch,
        "best_formula": technical_answer.get("best_formula", "NA"),
        "best_formula_reason": technical_answer.get("best_formula_reason", "NA"),
        "conclusion": conclusion,
        "agreements": agreements,
        "conflicts": conflicts,
        "error_sources": unique_text(error_sources),
        "missing_variables": unique_text(missing_variables),
        "candidate_variables": candidate_variables,
        "math_policy": "先補變數與誤差病歷，再做權重/貝葉斯/狀態轉移；高等數學不得用來掩蓋缺資料。",
        "next_checks": checks,
        "rule": "先用複雜方程式算出答案，再用推理式程序核對0/1、病因、日夜盤與保護層；一致才提高可信度，衝突就降權待驗。",
        "guardrail": "公式推理對照只決定研究信心，不產生投資命令。",
    }


def technical_forecast_alignment(bias: str, bagua: dict, candle: dict, cycle: dict) -> dict:
    primary = bagua.get("primary", {})
    weekly = bagua.get("weekly", {})
    daily = bagua.get("daily", {})
    cycle_code = cycle.get("current_position", {}).get("code")
    bearish_bagua = primary.get("code") == "KAN" or weekly.get("code") == "KAN" or daily.get("code") == "KAN"
    weak_cycle = cycle_code in ["post_rally_pullback", "pullback_watch"]
    weak_candle = candle.get("type") in ["long_bear", "gap_up_failed"]
    if bias == "bearish" and bearish_bagua and weak_cycle and weak_candle:
        return {
            "status": "一致",
            "summary": "技術分析、坎卦、回落波段與長黑K同向，支持防守預測。",
        }
    if bias == "bearish" and (bearish_bagua or weak_cycle or weak_candle):
        return {
            "status": "大致一致",
            "summary": "技術分析偏空，且部分卦位、波段或K線同步轉弱。",
        }
    if bias != "bearish" and bearish_bagua:
        return {
            "status": "矛盾",
            "summary": "卦位已入坎，但成熟技術分析未同步轉弱，需檢查資料或分類條件。",
        }
    return {
        "status": "需後續確認",
        "summary": "成熟技術分析與卦位沒有形成強烈同向，需等待下一根K線確認。",
    }


def analyze_peak_to_valley_warning(scored: pd.DataFrame, signal_date: str) -> dict:
    data = prepare_peak_warning_frame(scored, signal_date)
    if len(data) < 80:
        return {"enabled": False, "message": "資料不足，無法判斷峰轉谷早期預警。"}
    idx = len(data) - 1
    current = peak_warning_event_at(data, idx)
    recent_events = [
        peak_warning_event_at(data, pos)
        for pos in range(max(60, idx - 5), idx + 1)
    ]
    recent_events = [item for item in recent_events if item["event_code"] != "none"]
    last_event = recent_events[-1] if recent_events else current
    peak_date = last_event.get("date")
    peak_close = last_event.get("close")
    latest = data.iloc[-1]
    close = safe_float(latest.get("close"))
    ma5 = safe_float(latest.get("ma5"))
    low = safe_float(latest.get("low"))
    drawdown_from_signal = close / peak_close - 1 if peak_close not in [None, 0] and close is not None else None
    break_ma5 = close is not None and ma5 is not None and close < ma5
    break_signal_low = low is not None and last_event.get("low") is not None and low < last_event.get("low")

    if current["event_code"] == "peak_candidate":
        level = "早期預警"
        status_code = "peak_candidate"
        headline = "高峰轉折候選"
        summary = "創高後收在低位，代表高檔賣壓出現；此時只能標示候選，不能直接確認已做頭。"
    elif recent_events and (break_ma5 or break_signal_low or (drawdown_from_signal is not None and drawdown_from_signal <= -0.035)):
        level = "轉弱確認"
        status_code = "valley_turn_confirming"
        headline = "峰轉谷確認中"
        summary = "近期高峰候選後，價格跌破短均或候選低點，峰轉谷風險已從候選升級為確認中。"
    elif recent_events:
        level = "追蹤"
        status_code = "candidate_watch"
        headline = "高峰候選追蹤"
        summary = "近期出現過高峰轉折候選，但尚未跌破確認條件。"
    else:
        level = "無"
        status_code = "none"
        headline = "無高峰轉折預警"
        summary = "目前沒有創高收低的峰轉谷早期訊號。"

    return {
        "enabled": True,
        "signal_date": str(pd.to_datetime(latest["date"]).date()),
        "level": level,
        "status_code": status_code,
        "headline": headline,
        "summary": summary,
        "current_event": current,
        "last_event": last_event,
        "drawdown_from_signal_close": safe_float(drawdown_from_signal),
        "confirmation_checks": [
            {"name": "創高收低候選", "passed": bool(current["event_code"] == "peak_candidate" or recent_events), "basis": "60日新高附近，收盤落在當日低位。"},
            {"name": "跌破5日線", "passed": bool(break_ma5), "basis": "高峰候選後跌破短均，代表短線動能轉弱。"},
            {"name": "跌破候選日低點", "passed": bool(break_signal_low), "basis": "跌破候選日低點，代表創高失敗被價格確認。"},
            {"name": "候選後回撤逾3.5%", "passed": bool(drawdown_from_signal is not None and drawdown_from_signal <= -0.035), "basis": "候選後快速回撤，峰轉谷機率升高。"},
        ],
        "rules": [
            "創高收低只算高峰候選，不直接判定頂部完成。",
            "候選後跌破5日線、候選低點或快速回撤，才升級為峰轉谷確認中。",
            "若候選後站回高點並續創高，預警失效。",
        ],
    }


def prepare_peak_warning_frame(scored: pd.DataFrame, signal_date: str) -> pd.DataFrame:
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["date"] <= pd.to_datetime(signal_date)].dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    for column in ["open", "high", "low", "close"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["ma5"] = data["close"].rolling(5).mean()
    data["prior_60_high"] = data["high"].shift(1).rolling(60).max()
    data["day_range_pct"] = data["high"] / data["low"] - 1
    data["close_position"] = (data["close"] - data["low"]) / (data["high"] - data["low"])
    data["close_position"] = data["close_position"].replace([float("inf"), -float("inf")], None)
    return data


def peak_warning_event_at(data: pd.DataFrame, idx: int) -> dict:
    row = data.iloc[idx]
    high = safe_float(row.get("high"))
    low = safe_float(row.get("low"))
    close = safe_float(row.get("close"))
    open_ = safe_float(row.get("open"))
    prior_high = safe_float(row.get("prior_60_high"))
    close_position = safe_float(row.get("close_position"))
    day_range = safe_float(row.get("day_range_pct"))
    new_high = high is not None and prior_high is not None and high >= prior_high
    weak_close = close_position is not None and close_position <= 0.25
    wide_range = day_range is not None and day_range >= 0.015
    bearish_body = close is not None and open_ is not None and close < open_
    event_code = "peak_candidate" if new_high and weak_close and wide_range else "none"
    label = "創高收低高峰候選" if event_code == "peak_candidate" else "無"
    score = sum([bool(new_high), bool(weak_close), bool(wide_range), bool(bearish_body)])
    return {
        "date": str(pd.to_datetime(row["date"]).date()),
        "event_code": event_code,
        "label": label,
        "score": score,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "prior_60_high": prior_high,
        "close_position": close_position,
        "day_range_pct": day_range,
        "conditions": {
            "new_60d_high": bool(new_high),
            "close_in_low_quarter": bool(weak_close),
            "range_at_least_1_5pct": bool(wide_range),
            "bearish_body": bool(bearish_body),
        },
    }


def build_peak_to_valley_warning_backtest(scored: pd.DataFrame, signal_date: str) -> dict:
    data = prepare_peak_warning_frame(scored, signal_date)
    max_forward = 20
    events = []
    if len(data) < 100:
        return {"enabled": False, "message": "資料不足，無法回測峰轉谷預警。"}
    for idx in range(60, len(data) - max_forward):
        event = peak_warning_event_at(data, idx)
        if event["event_code"] != "peak_candidate":
            continue
        events.append(build_peak_warning_sample(data, idx, event))
    stats = peak_warning_backtest_stats(events)
    return {
        "enabled": True,
        "framework": "peak_to_valley_early_warning_v1",
        "signal_date": str(data.iloc[-1]["date"].date()),
        "method": "只使用當日以前可見資料偵測60日新高後收低；往後5/10/20日只用於事後驗證，不回填當日判斷。",
        "rules": {
            "candidate": "當日創60日新高、日內震幅至少1.5%、收盤落在全日低位25%以內。",
            "confirmation": "候選後跌破5日線、跌破候選低點或快速回撤，才升級峰轉谷確認。",
        },
        "event_count": len(events),
        "stats": stats,
        "recent_events": events[-8:],
        "summary": peak_warning_backtest_summary(stats, len(events)),
    }


def build_peak_warning_sample(data: pd.DataFrame, idx: int, event: dict) -> dict:
    close = event.get("close")
    low = event.get("low")
    rows = data.iloc[idx + 1 : idx + 21].copy()
    sample = {
        "date": event.get("date"),
        "close": close,
        "high": event.get("high"),
        "low": low,
        "close_position": event.get("close_position"),
        "day_range_pct": event.get("day_range_pct"),
        "forward": {},
    }
    for horizon in [5, 10, 20]:
        window = rows.head(horizon)
        if window.empty or close in [None, 0]:
            continue
        min_low = safe_float(window["low"].min())
        min_close = safe_float(window["close"].min())
        max_high = safe_float(window["high"].max())
        sample["forward"][f"{horizon}d"] = {
            "max_low_drawdown": safe_float(min_low / close - 1) if min_low is not None else None,
            "max_close_drawdown": safe_float(min_close / close - 1) if min_close is not None else None,
            "max_high_return": safe_float(max_high / close - 1) if max_high is not None else None,
            "break_event_low": bool(min_low is not None and low is not None and min_low < low),
        }
    return sample


def peak_warning_backtest_stats(events: list[dict]) -> dict:
    rows = []
    for horizon in ["5d", "10d", "20d"]:
        values = [event.get("forward", {}).get(horizon, {}) for event in events if event.get("forward", {}).get(horizon)]
        if not values:
            continue
        low_dd = [item.get("max_low_drawdown") for item in values if item.get("max_low_drawdown") is not None]
        close_dd = [item.get("max_close_drawdown") for item in values if item.get("max_close_drawdown") is not None]
        rows.append(
            {
                "horizon": horizon,
                "sample_count": len(values),
                "avg_max_low_drawdown": safe_float(sum(low_dd) / len(low_dd)) if low_dd else None,
                "avg_max_close_drawdown": safe_float(sum(close_dd) / len(close_dd)) if close_dd else None,
                "down_3pct_rate": safe_float(sum(1 for value in low_dd if value <= -0.03) / len(low_dd)) if low_dd else None,
                "down_5pct_rate": safe_float(sum(1 for value in low_dd if value <= -0.05) / len(low_dd)) if low_dd else None,
                "break_event_low_rate": safe_float(sum(1 for item in values if item.get("break_event_low")) / len(values)),
            }
        )
    return {"by_horizon": rows}


def peak_warning_backtest_summary(stats: dict, event_count: int) -> str:
    if event_count == 0:
        return "歷史上未找到足夠峰轉谷候選事件。"
    row10 = next((row for row in stats.get("by_horizon", []) if row.get("horizon") == "10d"), None)
    if not row10:
        return f"共找到 {event_count} 筆峰轉谷候選事件，樣本仍需累積。"
    return (
        f"共找到 {event_count} 筆峰轉谷候選事件；10日內平均最大低點回撤 "
        f"{pct(row10.get('avg_max_low_drawdown'))}，跌逾3%比例 {pct(row10.get('down_3pct_rate'))}，"
        f"跌破候選日低點比例 {pct(row10.get('break_event_low_rate'))}。"
    )


def analyze_route_reference(
    scored: pd.DataFrame,
    signal_date: str,
    bagua: dict,
    technical: dict,
    window: int = 12,
) -> dict:
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["date"] <= pd.to_datetime(signal_date)].dropna(subset=["close"]).reset_index(drop=True)
    if len(data) < 260:
        return {"enabled": False, "message": "歷史資料不足，無法建立模式路線參考。"}

    try:
        routes = build_route_reference_frame(data)
    except Exception as exc:
        return {"enabled": False, "message": f"模式路線建立失敗: {exc}"}
    if len(routes) < window + 80:
        return {"enabled": False, "message": "可比對路線不足。"}

    current_idx = len(routes) - 1
    current_override = {
        "monthly_code": bagua.get("monthly", {}).get("code"),
        "weekly_code": bagua.get("weekly", {}).get("code"),
        "daily_code": bagua.get("daily", {}).get("code"),
    }
    for column, value in current_override.items():
        if value:
            routes.loc[current_idx, column] = value
    routes.loc[current_idx, "route_state"] = route_state(
        routes.loc[current_idx, "monthly_code"],
        routes.loc[current_idx, "weekly_code"],
        routes.loc[current_idx, "daily_code"],
    )

    current_window = routes.iloc[current_idx - window + 1 : current_idx + 1].copy()
    matches = []
    max_forward = 60
    last_candidate = len(routes) - max_forward - 1
    for end_idx in range(window - 1, last_candidate + 1):
        if end_idx >= current_idx - 60:
            continue
        candidate = routes.iloc[end_idx - window + 1 : end_idx + 1]
        score = route_similarity(current_window, candidate)
        if score < 0.55:
            continue
        end_close = safe_float(routes.loc[end_idx, "close"])
        if end_close in [None, 0]:
            continue
        matches.append(
            {
                "end_idx": end_idx,
                "start_date": str(routes.loc[end_idx - window + 1, "date"].date()),
                "end_date": str(routes.loc[end_idx, "date"].date()),
                "similarity": score,
                "state_path": compressed_route(candidate["route_state"].tolist()),
                "end_close": end_close,
                "returns": {
                    f"{horizon}d": safe_float(routes.loc[end_idx + horizon, "close"] / end_close - 1)
                    for horizon in [1, 5, 20, 60]
                    if end_idx + horizon < len(routes)
                },
            }
        )

    matches = sorted(matches, key=lambda item: item["similarity"], reverse=True)[:20]
    stats = route_follow_through_stats(matches)
    current_path = compressed_route(current_window["route_state"].tolist())
    current_state = routes.loc[current_idx, "route_state"]
    return {
        "enabled": True,
        "signal_date": str(routes.loc[current_idx, "date"].date()),
        "method": (
            "以固定順序震→巽→離→坤→兌→乾→坎→艮，將最近12個交易日的月/週/日卦壓成路線，"
            "再找歷史相似路線；這是路線參考，不是把單一年份硬套到現在。"
        ),
        "current_state": current_state,
        "current_state_label": route_state_label(current_state),
        "current_path": current_path,
        "technical_anchor": technical.get("label"),
        "sample_count": len(matches),
        "stats": stats,
        "matches": matches[:5],
        "plain_summary": route_reference_summary(current_state, stats, len(matches)),
    }


def build_route_reference_frame(data: pd.DataFrame) -> pd.DataFrame:
    base = data[["date", "open", "high", "low", "close"]].copy()
    daily = prepare_bagua_features(base, 120, 20, 60)[["date"]].copy()
    daily_source = prepare_bagua_features(base, 120, 20, 60)
    daily["daily_code"] = daily_source.apply(classify_bagua_row, axis=1)

    weekly_source = prepare_bagua_features(resample_market_frame(base, "W-FRI"), 26, 4, 13)
    weekly = weekly_source[["date"]].copy()
    weekly["weekly_code"] = weekly_source.apply(classify_bagua_row, axis=1)

    monthly_source = prepare_bagua_features(resample_market_frame(base, "ME"), 24, 3, 8)
    monthly = monthly_source[["date"]].copy()
    monthly["monthly_code"] = monthly_source.apply(classify_bagua_row, axis=1)

    routes = pd.merge_asof(daily.sort_values("date"), weekly.sort_values("date"), on="date")
    routes = pd.merge_asof(routes.sort_values("date"), monthly.sort_values("date"), on="date")
    routes = routes.merge(base[["date", "close"]], on="date", how="left")
    routes = routes.dropna(subset=["daily_code", "weekly_code", "monthly_code", "close"]).reset_index(drop=True)
    routes["route_state"] = routes.apply(
        lambda row: route_state(row["monthly_code"], row["weekly_code"], row["daily_code"]),
        axis=1,
    )
    return routes


def route_state(monthly_code: str, weekly_code: str, daily_code: str) -> str:
    high_codes = {"LI", "KUN", "DUI", "QIAN"}
    bottom_codes = {"GEN", "ZHEN"}
    repair_codes = {"ZHEN", "XUN", "LI"}
    if monthly_code in high_codes and weekly_code == "KAN" and daily_code == "KAN":
        return "M_HIGH_WD_KAN"
    if monthly_code in high_codes and weekly_code in high_codes and daily_code == "KAN":
        return "M_W_HIGH_D_KAN"
    if monthly_code in high_codes and weekly_code in high_codes and daily_code in high_codes:
        return "ALL_HIGH"
    if monthly_code == "KAN" and (weekly_code in bottom_codes or daily_code in bottom_codes):
        return "M_KAN_WD_BOTTOM_START"
    if monthly_code in {"KAN", "GEN"} and weekly_code in repair_codes and daily_code in repair_codes:
        return "REPAIR_UP_ALIGNED"
    return "OTHER"


def route_similarity(current: pd.DataFrame, candidate: pd.DataFrame) -> float:
    if len(current) != len(candidate) or current.empty:
        return 0
    state_score = (current["route_state"].to_numpy() == candidate["route_state"].to_numpy()).mean()
    code_scores = []
    for column in ["monthly_code", "weekly_code", "daily_code"]:
        code_scores.append((current[column].to_numpy() == candidate[column].to_numpy()).mean())
    return float(state_score * 0.7 + (sum(code_scores) / len(code_scores)) * 0.3)


def compressed_route(states: list[str]) -> str:
    output = []
    for state in states:
        label = route_state_label(state)
        if not output or output[-1] != label:
            output.append(label)
    return " -> ".join(output)


def route_state_label(state: str) -> str:
    labels = {
        "ALL_HIGH": "高檔同向",
        "M_W_HIGH_D_KAN": "高檔日入坎",
        "M_HIGH_WD_KAN": "高檔週日入坎",
        "M_KAN_WD_BOTTOM_START": "坎後止跌啟動",
        "REPAIR_UP_ALIGNED": "修復同向",
        "OTHER": "混合段",
    }
    return labels.get(state, state)


def route_follow_through_stats(matches: list[dict]) -> dict:
    rows = []
    for horizon in ["1d", "5d", "20d", "60d"]:
        values = [
            item["returns"].get(horizon)
            for item in matches
            if item.get("returns", {}).get(horizon) is not None
        ]
        if not values:
            continue
        rows.append(
            {
                "horizon": horizon,
                "sample_count": len(values),
                "avg_return": safe_float(sum(values) / len(values)),
                "up_rate": safe_float(sum(1 for value in values if value > 0) / len(values)),
                "down_5pct_rate": safe_float(sum(1 for value in values if value <= -0.05) / len(values)),
            }
        )
    return {"by_horizon": rows}


def route_reference_summary(current_state: str, stats: dict, sample_count: int) -> str:
    if sample_count == 0:
        return "目前路線沒有足夠相似歷史樣本；只能回到技術位階與K線確認。"
    row20 = next((row for row in stats.get("by_horizon", []) if row["horizon"] == "20d"), None)
    if current_state == "M_HIGH_WD_KAN":
        base = "現在屬於高檔後週日同入坎，歷史上常有短反彈，但還不是艮底確認。"
    elif current_state == "M_KAN_WD_BOTTOM_START":
        base = "現在接近坎後止跌啟動，才比較像艮轉震的底部路線。"
    else:
        base = f"現在路線為{route_state_label(current_state)}。"
    if row20:
        return f"{base} 相似路線20日後平均 {pct(row20.get('avg_return'))}、上漲率 {pct(row20.get('up_rate'))}。"
    return base


def analyze_bottom_event_reference(scored: pd.DataFrame, signal_date: str, bagua: dict) -> dict:
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["date"] <= pd.to_datetime(signal_date)].dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if len(data) < 300:
        return {"enabled": False, "message": "歷史資料不足，無法統計測底事件。"}

    current = data.iloc[-1]
    current_event = classify_bottom_test_event(data, len(data) - 1, bagua)
    events = []
    max_forward = 20
    for idx in range(252, len(data) - max_forward):
        if idx >= len(data) - 60:
            continue
        event = classify_bottom_test_event(data, idx)
        if event["type"] == "none":
            continue
        events.append(build_bottom_event_sample(data, idx, event))

    same_type = [event for event in events if event["type"] == current_event["type"]]
    same_family = [event for event in events if event["family"] == current_event["family"]]
    selected = same_type if len(same_type) >= 20 else same_family
    stats = bottom_event_follow_through_stats(selected)
    return {
        "enabled": True,
        "signal_date": str(pd.to_datetime(current["date"]).date()),
        "method": (
            "用每個歷史日期當下可見的近一年高點鎖定合理底部區；只統計當日之後的1/3/5/10/20日，"
            "不使用事後低點回填。"
        ),
        "current_event": current_event,
        "sample_scope": "same_type" if len(same_type) >= 20 else "same_family",
        "sample_count": len(selected),
        "stats": stats,
        "matches": selected[-5:],
        "plain_summary": bottom_event_summary(current_event, stats, len(selected)),
    }


def classify_bottom_test_event(data: pd.DataFrame, idx: int, bagua: dict | None = None) -> dict:
    row = data.iloc[idx]
    history = data.iloc[max(0, idx - 251) : idx + 1]
    peak = safe_float(pd.to_numeric(history["high"], errors="coerce").max())
    low = safe_float(row.get("low"))
    close = safe_float(row.get("close"))
    normal_low = peak * 0.825 if peak else None
    normal_high = peak * 0.835 if peak else None
    crash_line = peak * 0.80 if peak else None
    if not all(value is not None for value in [low, close, normal_low, normal_high, crash_line]):
        return {"type": "none", "family": "none", "label": "資料不足"}

    primary = bagua.get("primary", {}).get("code") if bagua else None
    weekly = bagua.get("weekly", {}).get("code") if bagua else None
    daily = bagua.get("daily", {}).get("code") if bagua else None
    in_kan = any(code == "KAN" for code in [primary, weekly, daily] if code)
    if low < crash_line and close >= crash_line:
        event_type = "crash_line_intraday_reclaim"
        label = "崩盤線盤中刺破收回"
        family = "false_crash_watch"
    elif close < crash_line:
        event_type = "crash_line_close_break"
        label = "崩盤線收盤跌破"
        family = "confirmed_break"
    elif low < normal_low and close >= normal_low:
        event_type = "normal_bottom_intraday_reclaim"
        label = "合理底盤中刺破收回"
        family = "bottom_test_reclaim"
    elif close < normal_low:
        event_type = "normal_bottom_close_break"
        label = "合理底收盤跌破"
        family = "confirmed_break"
    elif normal_low <= close <= normal_high or normal_low <= low <= normal_high:
        event_type = "normal_bottom_zone_test"
        label = "合理底區測試"
        family = "bottom_test"
    else:
        event_type = "none"
        label = "非測底事件"
        family = "none"

    if event_type == "none":
        return {"type": "none", "family": "none", "label": label}
    return {
        "type": event_type,
        "family": family,
        "label": label,
        "in_kan": in_kan,
        "date": str(pd.to_datetime(row["date"]).date()),
        "close": close,
        "low": low,
        "anchor_peak": peak,
        "normal_bottom_low": round(normal_low, 2),
        "normal_bottom_high": round(normal_high, 2),
        "crash_warning": round(crash_line, 2),
    }


def build_bottom_event_sample(data: pd.DataFrame, idx: int, event: dict) -> dict:
    close = safe_float(data.loc[idx, "close"])
    low = safe_float(data.loc[idx, "low"])
    normal_low = event.get("normal_bottom_low")
    crash_line = event.get("crash_warning")
    returns = {}
    for horizon in [1, 3, 5, 10, 20]:
        if idx + horizon < len(data) and close not in [None, 0]:
            returns[f"{horizon}d"] = safe_float(data.loc[idx + horizon, "close"] / close - 1)
    next20 = data.iloc[idx + 1 : idx + 21].copy()
    next_lows = pd.to_numeric(next20["low"], errors="coerce") if not next20.empty else pd.Series(dtype=float)
    next_closes = pd.to_numeric(next20["close"], errors="coerce") if not next20.empty else pd.Series(dtype=float)
    return {
        "date": event.get("date"),
        "type": event.get("type"),
        "family": event.get("family"),
        "label": event.get("label"),
        "close": close,
        "low": low,
        "normal_bottom_low": normal_low,
        "crash_warning": crash_line,
        "returns": returns,
        "rebreak_normal_5d": bool(len(next_lows.head(5)) and (next_lows.head(5) < normal_low).any()) if normal_low else None,
        "reclaim_normal_3d": bool(len(next_closes.head(3)) and (next_closes.head(3) >= normal_low).any()) if normal_low else None,
        "break_crash_20d": bool(len(next_lows) and (next_lows < crash_line).any()) if crash_line else None,
        "gen_bottom_confirm_5d": bool(len(next_lows.head(5)) >= 3 and next_lows.head(3).min() >= low and next_closes.head(5).max() > close),
    }


def bottom_event_follow_through_stats(events: list[dict]) -> dict:
    rows = []
    for horizon in ["1d", "3d", "5d", "10d", "20d"]:
        values = [
            item["returns"].get(horizon)
            for item in events
            if item.get("returns", {}).get(horizon) is not None
        ]
        if not values:
            continue
        rows.append(
            {
                "horizon": horizon,
                "sample_count": len(values),
                "avg_return": safe_float(sum(values) / len(values)),
                "up_rate": safe_float(sum(1 for value in values if value > 0) / len(values)),
                "down_3pct_rate": safe_float(sum(1 for value in values if value <= -0.03) / len(values)),
            }
        )
    binary = {}
    for key in ["rebreak_normal_5d", "reclaim_normal_3d", "break_crash_20d", "gen_bottom_confirm_5d"]:
        values = [item.get(key) for item in events if item.get(key) is not None]
        binary[key] = {
            "sample_count": len(values),
            "rate": safe_float(sum(1 for value in values if value) / len(values)) if values else None,
        }
    return {"by_horizon": rows, "event_rates": binary}


def bottom_event_summary(current_event: dict, stats: dict, sample_count: int) -> str:
    if current_event.get("type") == "none":
        return "今天不是明確測底事件，歷史測底樣本不作主判斷。"
    if sample_count == 0:
        return "目前測底型態缺少足夠歷史同型樣本，仍以收盤確認與風控線為主。"
    row5 = next((row for row in stats.get("by_horizon", []) if row["horizon"] == "5d"), None)
    rates = stats.get("event_rates", {})
    reclaim = rates.get("reclaim_normal_3d", {}).get("rate")
    rebreak = rates.get("rebreak_normal_5d", {}).get("rate")
    crash = rates.get("break_crash_20d", {}).get("rate")
    text = f"目前事件為{current_event.get('label')}，歷史同型/同族樣本 {sample_count} 筆。"
    if row5:
        text += f" 5日後平均 {pct(row5.get('avg_return'))}，上漲率 {pct(row5.get('up_rate'))}。"
    if reclaim is not None:
        text += f" 3日內站回合理底率 {pct(reclaim)}。"
    if rebreak is not None:
        text += f" 5日內再破合理底率 {pct(rebreak)}。"
    if crash is not None:
        text += f" 20日內觸及崩盤線率 {pct(crash)}。"
    return text


def analyze_crash_monitor(
    scored: pd.DataFrame,
    signal_date: str,
    technical: dict,
    bagua: dict,
    premarket: dict,
    capital: dict,
    live_monitor: dict | None = None,
) -> dict:
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["date"] <= pd.to_datetime(signal_date)].dropna(subset=["close"]).reset_index(drop=True)
    if len(data) < 60:
        return {"enabled": False, "message": "資料不足，無法監控合理底部與崩盤前奏。"}

    for column in ["open", "high", "low", "close"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    row = data.iloc[-1]
    recent_year = data.tail(252)
    peak_row = recent_year.loc[recent_year["high"].idxmax()]
    peak = safe_float(peak_row.get("high"))
    close = safe_float(row.get("close"))
    low = safe_float(row.get("low"))
    live_monitor = live_monitor or {}
    live_price = safe_float(live_monitor.get("price")) if live_monitor.get("enabled") else None
    live_low = safe_float(live_monitor.get("low")) if live_monitor.get("enabled") else None
    effective_close = min([value for value in [close, live_price] if value is not None], default=close)
    effective_low = min([value for value in [low, live_low, live_price] if value is not None], default=low)
    drawdown = effective_close / peak - 1 if peak not in [None, 0] and effective_close is not None else None
    levels = {
        "normal_gen_low": round(peak * 0.94, 2) if peak else None,
        "normal_gen_high": round(peak * 0.96, 2) if peak else None,
        "healthy_pullback_low": round(peak * 0.90, 2) if peak else None,
        "major_correction_line": round(peak * 0.85, 2) if peak else None,
        "normal_bottom_low": round(peak * 0.825, 2) if peak else None,
        "normal_bottom_high": round(peak * 0.835, 2) if peak else None,
        "normal_bottom_center": round(peak * 0.83, 2) if peak else None,
        "early_warning": round(peak * 0.84, 2) if peak else None,
        "crash_warning": round(peak * 0.80, 2) if peak else None,
        "deep_kan_low": round(peak * 0.75, 2) if peak else None,
        "deep_kan_high": round(peak * 0.77, 2) if peak else None,
        "anchor_date": str(pd.to_datetime(peak_row["date"]).date()) if peak else None,
        "anchor_peak": peak,
        "anchor_rule": "合理底數以最近一年波段高點鎖定；未創新高前不因每日新低下修。",
    }
    drawdown_guardrail = classify_drawdown_guardrail(peak, effective_close, effective_low, levels)

    signals = technical.get("signals", {})
    primary_code = bagua.get("primary", {}).get("code")
    weekly_code = bagua.get("weekly", {}).get("code")
    daily_code = bagua.get("daily", {}).get("code")
    risk_points = 0
    triggers = []
    early_warnings = []
    if signals.get("below_ma_count", 0) >= 4:
        risk_points += 2
        triggers.append("跌破5/10/20/60日均線")
        early_warnings.append("均線系統全面轉弱")
    if signals.get("broke_recent_low"):
        risk_points += 2
        triggers.append("跌破近期低點")
        early_warnings.append("連續破低風險升高")
    if primary_code == "KAN" or weekly_code == "KAN" or daily_code == "KAN":
        risk_points += 2
        triggers.append("卦位入坎")
        early_warnings.append("波段位置已入坎")
    if live_monitor.get("enabled") and live_price is not None and close is not None and live_price < close:
        risk_points += 2
        triggers.append("盤中價低於最後日線收盤")
        early_warnings.append("盤中惡化已超過日線資料")
    if effective_low is not None and levels["early_warning"] is not None and effective_low < levels["early_warning"]:
        risk_points += 1
        triggers.append("接近合理底部區")
        early_warnings.append("已接近合理底部區，需防第一底失守")
    if effective_low is not None and levels["normal_bottom_low"] is not None and effective_low < levels["normal_bottom_low"]:
        risk_points += 2
        triggers.append("跌破合理底部區下緣")
        early_warnings.append("第一合理底已失守")
    if effective_low is not None and levels["crash_warning"] is not None and effective_low < levels["crash_warning"]:
        risk_points += 3
        triggers.append("跌破20%修正崩盤警戒線")
    if premarket.get("is_premarket") and premarket.get("total_score", 0) < 0:
        risk_points += 1
        triggers.append("盤前外部市場或夜盤仍偏空")
        early_warnings.append("外部壓力未解除")
    if capital.get("has_factor_values") and capital.get("non_price_score", 0) < 0:
        risk_points += 1
        triggers.append("法人籌碼或期權偏空")
        early_warnings.append("籌碼或避險部位偏空")
    elif not capital.get("has_factor_values"):
        triggers.append("法人籌碼期權未接入，籌碼監控需降權")
        early_warnings.append("籌碼資料缺口，需降權但不可忽略")

    if effective_close is None:
        state = "資料不足"
        status_code = "unknown"
        summary = "缺少收盤資料，無法判斷。"
    elif effective_low is not None and effective_low < levels["crash_warning"]:
        state = "崩盤前奏警戒"
        status_code = "crash_warning"
        summary = "已跌破20%修正警戒，不能再視為一般修正，需用崩盤風險規則監控。"
    elif effective_low is not None and effective_low < levels["normal_bottom_low"]:
        state = "合理底失守，崩盤前奏觀察"
        status_code = "bottom_failed_watch"
        summary = "第一合理底部已失守，但尚未有效跌破20%修正線；需看是否快速拉回。"
    elif effective_close >= levels["normal_bottom_low"]:
        state = "合理底部區測試"
        status_code = "normal_bottom_test"
        summary = "仍在合理修正底部區附近，重點看是否不再破低並收復前低。"
    else:
        state = "崩盤前奏警戒"
        status_code = "crash_warning"
        summary = "已跌破20%修正警戒，不能再視為一般修正，需用崩盤風險規則監控。"

    confirmation = crash_confirmation_audit(data, levels, live_monitor, effective_low)
    false_crash = false_crash_filter(data, levels, live_monitor, effective_low, confirmation)
    capital_policy = capital_preservation_policy(status_code, confirmation, effective_low, levels)

    risk_model = market_risk_value(
        drawdown,
        technical,
        bagua,
        premarket,
        capital,
        status_code=status_code,
    )
    risk_value = risk_model["risk_value"]
    health_score = risk_model["health_score"]

    if false_crash.get("candidate") and confirmation.get("stage_code") == "crash_intraday_break":
        alert_level = "極高警戒"
        alert_code = "intraday_unconfirmed"
        alert_summary = "盤中已刺破崩盤線，但尚未收盤確認；先控風險，不能直接定案為崩盤。"
    elif status_code == "crash_warning" or risk_value >= 75:
        alert_level = "確認警戒"
        alert_code = "confirmed_warning"
        alert_summary = "已達高風險警戒，需以崩盤前奏規則控管，不再用一般回檔心態處理。"
    elif status_code == "bottom_failed_watch" or risk_value >= 60:
        alert_level = "提前警戒"
        alert_code = "early_alert"
        alert_summary = "尚未確認崩盤，但已有多項早期風險條件同時出現，需提前控管。"
    elif risk_value >= 45:
        alert_level = "預警"
        alert_code = "watch"
        alert_summary = "已有部分風險條件轉弱，需提高監控頻率。"
    else:
        alert_level = "觀察"
        alert_code = "observe"
        alert_summary = "尚未出現足夠的早期警戒條件。"

    return {
        "enabled": True,
        "signal_date": str(row["date"].date()),
        "state": state,
        "status_code": status_code,
        "summary": summary,
        "alert_level": alert_level,
        "alert_code": alert_code,
        "alert_summary": alert_summary,
        "risk_value": risk_value,
        "health_score": health_score,
        "risk_method": risk_model["method"],
        "risk_components": risk_model["components"],
        "risk_points": risk_points,
        "confirmation": confirmation,
        "false_crash_filter": false_crash,
        "capital_policy": capital_policy,
        "peak_date": str(pd.to_datetime(peak_row["date"]).date()),
        "peak": peak,
        "close": close,
        "low": low,
        "effective_close": effective_close,
        "effective_low": effective_low,
        "live_monitor": live_monitor,
        "drawdown_from_peak": safe_float(drawdown),
        "drawdown_guardrail": drawdown_guardrail,
        "levels": levels,
        "triggers": triggers,
        "early_warnings": unique_text(early_warnings),
        "monitor_rules": [
            "預警只代表風險升高，不代表猜測一定崩盤。",
            "合理底數是固定風控線，不是每天跟著新低下移。",
            "艮代表跌勢停止，不代表必須深跌到合理底；高檔正常艮與深坎後艮必須分開。",
            "從高點回落15%以上先列重大修正，接近20%列崩盤前奏，不得用一般盤整或正常艮包裝。",
            "政策護盤只列可能觀察，不可假設政府一定進場或一定解除風險。",
            "提前警戒代表要先降風險、等確認，不等崩盤線跌破才反應。",
            "崩盤結論必須經過盤中、收盤、連續收盤三重核對。",
            "合理底部區守住，才談艮底觀察。",
            "跌破合理底部區但快速收回，視為假跌破或洗盤觀察。",
            "盤中破崩盤線但收盤未確認且快速收回，只能列為假崩盤候選，不可直接定案。",
            "有效跌破20%修正線，升級為崩盤前奏警戒。",
            "崩盤前奏需同時看外部市場、夜盤、法人籌碼、期貨避險與是否連續破低。",
        ],
    }


def classify_drawdown_guardrail(peak, effective_close, effective_low, levels: dict) -> dict:
    peak_value = safe_float(peak)
    close_value = safe_float(effective_close)
    low_value = safe_float(effective_low)
    reference = min([value for value in [close_value, low_value] if value is not None], default=close_value)
    drawdown = reference / peak_value - 1 if peak_value not in [None, 0] and reference is not None else None

    if drawdown is None:
        stage = "資料不足"
        stage_code = "unknown"
        summary = "缺少高點或有效價格，無法分層判斷。"
        normal_gen_allowed = False
    elif drawdown >= -0.06:
        stage = "高檔正常止跌區"
        stage_code = "normal_high_gen_zone"
        summary = "若停止破低並重新站穩，可視為高檔正常艮候選；不是深跌築底。"
        normal_gen_allowed = True
    elif drawdown >= -0.10:
        stage = "健康修正區"
        stage_code = "healthy_pullback"
        summary = "屬正常回檔到中度修正，需看均線與前低是否守住。"
        normal_gen_allowed = False
    elif drawdown >= -0.15:
        stage = "中度修正警戒"
        stage_code = "medium_correction_watch"
        summary = "修正已加深，不能只用高檔盤整解釋。"
        normal_gen_allowed = False
    elif drawdown >= -0.20:
        stage = "重大修正／政策觀察區"
        stage_code = "major_correction_policy_watch"
        summary = "從高點回落15%以上，屬重大修正；若伴隨市場失序或信心危機，可能進入政策安定觀察，但不能假設必然護盤。"
        normal_gen_allowed = False
    else:
        stage = "崩盤前奏／深坎區"
        stage_code = "crash_precursor_deep_kan"
        summary = "回落接近或超過20%，不再視為一般盤整；需用崩盤三重核對與資金保全規則處理。"
        normal_gen_allowed = False

    return {
        "stage": stage,
        "stage_code": stage_code,
        "drawdown": safe_float(drawdown),
        "reference_price": reference,
        "normal_gen_allowed": normal_gen_allowed,
        "policy_support_assumption": "不可假設必然進場；只在國內外重大事件、國際資金大幅移動並顯著影響信心、致市場有失序或損及國家安定之虞時列為政策觀察。",
        "summary": summary,
        "levels": {
            "normal_high_gen_zone": f"{num(levels.get('normal_gen_low'))}～{num(levels.get('normal_gen_high'))}",
            "healthy_pullback_floor": num(levels.get("healthy_pullback_low")),
            "major_correction_line": num(levels.get("major_correction_line")),
            "deep_gen_candidate": f"{num(levels.get('normal_bottom_low'))}～{num(levels.get('normal_bottom_high'))}",
            "crash_warning": num(levels.get("crash_warning")),
        },
    }


def reconcile_market_state_sop(crash: dict, bagua: dict) -> dict:
    """Make lifecycle state the single source for diagnosis text and SOP routing."""
    if not crash.get("enabled"):
        return crash

    result = dict(crash)
    state_gua = bagua.get("roles", {}).get("state_gua", {})
    state_code = state_gua.get("code")
    health_score = safe_float(result.get("health_score"))
    alert_code = result.get("alert_code")
    emergency = alert_code in {"confirmed_warning", "early_alert", "intraday_unconfirmed"}

    mappings = {
        "KAN": {
            "state": "坎危機防守",
            "status_code": "kan_crisis_defense",
            "summary": "市場仍在急跌危機路徑，先確認合理底與崩盤線是否守住。",
            "sop": "危機防守",
            "action": "降低曝險，等待止跌與收盤確認",
            "bottom_role": "當前測試目標",
        },
        "GEN": {
            "state": "艮止跌觀察",
            "status_code": "gen_bottoming_watch",
            "summary": "跌勢暫停但尚未明顯脫離低點，觀察不再破低與短期均線收復。",
            "sop": "止跌確認",
            "action": "維持小部位觀察，等待修復啟動",
            "bottom_role": "止跌確認基準",
        },
        "ZHEN": {
            "state": "震啟動修復",
            "status_code": "zhen_repair_activation",
            "summary": "已明顯脫離近期低點並進入修復啟動；尚未確認完整新升段。",
            "sop": "修復確認",
            "action": "可追蹤強勢標的，但不盲目追高；確認高低點墊高與均線收復",
            "bottom_role": "風控失效防線",
        },
        "XUN": {
            "state": "巽趨勢延伸",
            "status_code": "xun_trend_extension",
            "summary": "修復結構持續延伸，重點轉為趨勢穩定度與回檔承接。",
            "sop": "趨勢跟隨",
            "action": "順勢管理部位，跌破結構支撐時降風險",
            "bottom_role": "遠端風控防線",
        },
        "LI": {
            "state": "離加速確認",
            "status_code": "li_acceleration",
            "summary": "價格進入加速或擴張階段，動能增強但追價風險同步上升。",
            "sop": "動能控管",
            "action": "持強汰弱並收緊移動風控，避免高位過度追價",
            "bottom_role": "遠端風控防線",
        },
    }

    mapping = mappings.get(state_code)
    if mapping and not emergency:
        result.update({key: mapping[key] for key in ("state", "status_code", "summary")})
        result["capital_policy"] = {
            **result.get("capital_policy", {}),
            "level": "中" if state_code in {"GEN", "ZHEN"} else "低",
            "action": mapping["action"],
            "summary": f"目前依{mapping['state']}執行「{mapping['sop']}」SOP；合理底定位為{mapping['bottom_role']}。",
        }

    if emergency:
        health_label = market_health_label(health_score, alert_code)
        sop_name = "危機風控"
        sop_action = result.get("capital_policy", {}).get("action")
        bottom_role = "即時風控線"
    elif mapping:
        if health_score is None:
            health_label = "資料不足"
        elif health_score >= 81:
            health_label = "健康修復"
        elif health_score >= 66:
            health_label = "修復中"
        elif health_score >= 46:
            health_label = "轉弱觀察"
        else:
            health_label = "高風險"
        sop_name = mapping["sop"]
        sop_action = mapping["action"]
        bottom_role = mapping["bottom_role"]
    else:
        health_label = market_health_label(health_score, alert_code)
        sop_name = "資料確認"
        sop_action = "維持既有風控，等待狀態卦確認"
        bottom_role = "風控參考線"

    result["health_label"] = health_label
    result["state_sop"] = {
        "source": "bagua_lifecycle.roles.state_gua",
        "state_code": state_code,
        "state_gua": state_gua.get("gua"),
        "state_status": state_gua.get("status"),
        "sop": sop_name,
        "action": sop_action,
        "reasonable_bottom_role": bottom_role,
        "background_gua": bagua.get("roles", {}).get("background_gua", {}).get("gua"),
        "rule": "狀態卦決定當前診斷與SOP；背景卦保留長週期風險；合理底依階段改變用途。",
    }
    return result


def market_risk_value(
    drawdown,
    technical: dict,
    bagua: dict,
    premarket: dict,
    capital: dict,
    status_code: str | None,
) -> dict:
    signals = technical.get("signals", {})
    levels = technical.get("levels", {})
    below_ma_count = safe_int(signals.get("below_ma_count"), 0)
    day_range = safe_float(levels.get("day_range_pct")) or 0
    range_20d = safe_float(levels.get("range_20d")) or 0
    drawdown_abs = abs(safe_float(drawdown) or 0)

    drawdown_score = clamp((drawdown_abs - 0.05) / 0.20) * 100
    trend_score = clamp((below_ma_count / 4) * 0.70 + (0.30 if signals.get("broke_recent_low") else 0)) * 100
    volatility_score = (
        clamp((day_range - 0.01) / 0.04) * 45
        + clamp((range_20d - 0.05) / 0.15) * 55
    )

    primary_code = bagua.get("primary", {}).get("code")
    weekly_code = bagua.get("weekly", {}).get("code")
    daily_code = bagua.get("daily", {}).get("code")
    kan_count = sum(1 for code in [primary_code, weekly_code, daily_code] if code == "KAN")
    structure_score = clamp(kan_count / 3) * 60
    if status_code == "bottom_failed_watch":
        structure_score += 20
    elif status_code == "crash_warning":
        structure_score += 40
    structure_score = clamp(structure_score / 100) * 100

    external_score = 25
    if premarket.get("is_premarket"):
        total_score = safe_int(premarket.get("total_score"), 0)
        external_score = 50 + clamp(abs(total_score) / 5) * 50 if total_score < 0 else 35
        sox = safe_float(premarket.get("sox_return_1d")) or 0
        tsm = safe_float(premarket.get("tsm_adr_return_1d")) or 0
        if sox <= -0.03 or tsm <= -0.015:
            external_score = min(100, external_score + 15)

    if not capital.get("has_factor_values"):
        capital_score = 55
    else:
        non_price = safe_int(capital.get("non_price_score"), 0)
        if non_price < 0:
            capital_score = 75
        elif non_price > 0:
            capital_score = 30
        else:
            capital_score = 50

    components = [
        risk_component("跌幅深度", 0.25, drawdown_score, "高點回撤越接近20%～25%，系統性風險越高。"),
        risk_component("趨勢破壞", 0.25, trend_score, "跌破多條均線與近期低點，代表技術停損鏈啟動。"),
        risk_component("波動擴大", 0.15, volatility_score, "單日與20日震幅越大，代表流動性與情緒不穩。"),
        risk_component("波段結構", 0.15, structure_score, "月週日卦位入坎越一致，代表波段風險越集中。"),
        risk_component("外部壓力", 0.10, external_score, "美股、半導體與台指期夜盤偏空會提高開盤與連鎖賣壓。"),
        risk_component("籌碼可信度", 0.10, capital_score, "籌碼未接入或偏空時，風險值需保守上修。"),
    ]
    risk_value = round(sum(item["weighted_score"] for item in components), 1)
    health_score = round(100 - risk_value, 1)
    return {
        "method": "風險值 = 跌幅深度25% + 趨勢破壞25% + 波動擴大15% + 波段結構15% + 外部壓力10% + 籌碼可信度10%。",
        "risk_value": risk_value,
        "health_score": health_score,
        "components": components,
    }


def risk_component(name: str, weight: float, score, basis: str) -> dict:
    score_value = round(clamp((safe_float(score) or 0) / 100) * 100, 1)
    return {
        "name": name,
        "weight": weight,
        "score": score_value,
        "score_range": risk_score_range(score_value),
        "weighted_score": round(score_value * weight, 1),
        "basis": basis,
    }


def risk_score_range(score) -> str:
    score_value = safe_float(score)
    if score_value is None:
        return "NA"
    if score_value < 25:
        return "0～24 低風險"
    if score_value < 50:
        return "25～49 觀察"
    if score_value < 75:
        return "50～74 警戒"
    return "75～100 高風險"


def crash_confirmation_audit(
    data: pd.DataFrame,
    levels: dict,
    live_monitor: dict,
    effective_low,
) -> dict:
    normal_low = levels.get("normal_bottom_low")
    crash_line = levels.get("crash_warning")
    closes = pd.to_numeric(data["close"], errors="coerce").dropna()
    latest_close = safe_float(closes.iloc[-1]) if not closes.empty else None
    recent_closes = closes.tail(3).tolist()
    live_break_normal = effective_low is not None and normal_low is not None and effective_low < normal_low
    live_break_crash = effective_low is not None and crash_line is not None and effective_low < crash_line
    close_break_normal = latest_close is not None and normal_low is not None and latest_close < normal_low
    close_break_crash = latest_close is not None and crash_line is not None and latest_close < crash_line
    consecutive_crash_closes = (
        len(recent_closes) >= 2
        and crash_line is not None
        and all(value < crash_line for value in recent_closes[-2:])
    )
    consecutive_normal_fail = (
        len(recent_closes) >= 2
        and normal_low is not None
        and all(value < normal_low for value in recent_closes[-2:])
    )

    checks = [
        {
            "name": "盤中破合理底",
            "passed": bool(live_break_normal),
            "level": normal_low,
            "basis": "盤中或即時低點跌破合理底部下緣，代表洗底區失守，需要提前警戒。",
        },
        {
            "name": "收盤破合理底",
            "passed": bool(close_break_normal),
            "level": normal_low,
            "basis": "完成日K收在合理底部下方，代表不是單純盤中刺破。",
        },
        {
            "name": "盤中破崩盤線",
            "passed": bool(live_break_crash),
            "level": crash_line,
            "basis": "盤中跌破20%修正線，進入崩盤前奏警戒。",
        },
        {
            "name": "收盤破崩盤線",
            "passed": bool(close_break_crash),
            "level": crash_line,
            "basis": "完成日K收在20%修正線下，崩盤風險大幅提高。",
        },
        {
            "name": "連續收不回崩盤線",
            "passed": bool(consecutive_crash_closes),
            "level": crash_line,
            "basis": "連續兩日收不回崩盤線，才接近崩盤確認。",
        },
    ]
    if consecutive_crash_closes:
        stage = "崩盤風險確認升高"
        stage_code = "crash_risk_confirming"
        summary = "已連續收不回崩盤線，資金風控必須以保命優先。"
    elif close_break_crash:
        stage = "崩盤前奏收盤確認"
        stage_code = "crash_close_break"
        summary = "收盤已跌破崩盤線，需要隔日是否收回作最後核對。"
    elif live_break_crash:
        stage = "崩盤前奏盤中警戒"
        stage_code = "crash_intraday_break"
        summary = "盤中跌破崩盤線，需等待收盤確認，但風控必須先亮紅燈。"
    elif close_break_normal or consecutive_normal_fail:
        stage = "合理底收盤失守"
        stage_code = "normal_bottom_close_fail"
        summary = "合理底不只是盤中跌破，收盤也未站回，洗底完成機率下降。"
    elif live_break_normal:
        stage = "合理底盤中失守"
        stage_code = "normal_bottom_intraday_fail"
        summary = "盤中已破合理底，需看收盤能否拉回。"
    else:
        stage = "尚未失守合理底"
        stage_code = "no_confirmed_break"
        summary = "尚未出現合理底失守確認。"
    return {
        "stage": stage,
        "stage_code": stage_code,
        "summary": summary,
        "latest_close": latest_close,
        "effective_low": safe_float(effective_low),
        "live_source": live_monitor.get("source", "none"),
        "checks": checks,
    }


def false_crash_filter(
    data: pd.DataFrame,
    levels: dict,
    live_monitor: dict,
    effective_low,
    confirmation: dict,
) -> dict:
    """Separate fail-safe intraday warnings from confirmed crash evidence."""
    normal_low = levels.get("normal_bottom_low")
    crash_line = levels.get("crash_warning")
    closes = pd.to_numeric(data["close"], errors="coerce").dropna()
    latest_close = safe_float(closes.iloc[-1]) if not closes.empty else None
    recent_closes = closes.tail(3).tolist()
    live_enabled = bool(live_monitor.get("enabled"))
    live_price = safe_float(live_monitor.get("price")) if live_enabled else None
    stage_code = confirmation.get("stage_code")

    intraday_only_normal = (
        stage_code == "normal_bottom_intraday_fail"
        and latest_close is not None
        and normal_low is not None
        and latest_close >= normal_low
    )
    intraday_only_crash = (
        stage_code == "crash_intraday_break"
        and latest_close is not None
        and crash_line is not None
        and latest_close >= crash_line
    )
    reclaimed_normal = live_price is not None and normal_low is not None and live_price >= normal_low
    reclaimed_crash = live_price is not None and crash_line is not None and live_price >= crash_line
    two_close_confirmed = (
        len(recent_closes) >= 2
        and crash_line is not None
        and all(value < crash_line for value in recent_closes[-2:])
    )
    close_confirmed = stage_code in ["normal_bottom_close_fail", "crash_close_break", "crash_risk_confirming"]

    checks = [
        {
            "name": "只盤中跌破，尚未收盤確認",
            "passed": bool(intraday_only_normal or intraday_only_crash),
            "basis": "假崩盤最常見型態是盤中殺破關鍵線，但日K沒有收在關鍵線下方。",
        },
        {
            "name": "即時價已站回關鍵線",
            "passed": bool(reclaimed_crash or reclaimed_normal),
            "basis": "破線後快速站回，代表可能是停損掃單或流動性瞬間失衡。",
        },
        {
            "name": "尚未連續兩日收不回崩盤線",
            "passed": bool(not two_close_confirmed),
            "basis": "崩盤確認必須看連續性，不能只用單一盤中低點定案。",
        },
        {
            "name": "尚未形成收盤破線",
            "passed": bool(not close_confirmed),
            "basis": "收盤破線比盤中刺破更重要；未收盤確認前只能列警戒。",
        },
    ]

    score = sum(1 for item in checks if item["passed"])
    if stage_code in ["crash_close_break", "crash_risk_confirming"]:
        label = "非假崩盤，已進入收盤確認"
        candidate = False
        action = "依崩盤風控處理，不用假跌破邏輯放鬆。"
    elif stage_code == "crash_intraday_break" and score >= 3:
        label = "假崩盤候選"
        candidate = True
        action = "先降風險但等收盤核對；若收回崩盤線，不升級為正式崩盤。"
    elif stage_code == "normal_bottom_intraday_fail" and score >= 3:
        label = "假跌破候選"
        candidate = True
        action = "停止攤平並等收盤；若收回合理底，改列洗盤觀察。"
    else:
        label = "尚無假崩盤證據"
        candidate = False
        action = "維持原警戒，不因假跌破假設降低風控。"

    return {
        "label": label,
        "candidate": candidate,
        "score": score,
        "action": action,
        "latest_close": latest_close,
        "live_price": live_price,
        "effective_low": safe_float(effective_low),
        "checks": checks,
        "rule": "假崩盤需同時具備盤中破線、收盤未確認、快速站回或沒有連續收不回；否則維持崩盤警戒。",
    }


def capital_preservation_policy(status_code: str, confirmation: dict, effective_low, levels: dict) -> dict:
    stage_code = confirmation.get("stage_code")
    crash_line = levels.get("crash_warning")
    normal_low = levels.get("normal_bottom_low")
    if stage_code in ["crash_risk_confirming", "crash_close_break"]:
        level = "最高"
        action = "保命優先"
        summary = "這不是加碼攤平環境；即使帳面虧損很大，也應依預設風控降低曝險或避險，避免本金繼續暴露在系統性風險。"
    elif stage_code == "crash_intraday_break":
        level = "極高"
        action = "先降風險，等收盤核對"
        summary = "盤中破崩盤線不可等閒視之，但仍需收盤確認；風控上應先降低部位風險，不把希望當策略。"
    elif status_code == "bottom_failed_watch" or stage_code in ["normal_bottom_close_fail", "normal_bottom_intraday_fail"]:
        level = "高"
        action = "停止攤平，降低風險"
        summary = "合理底已失守，資金處理應先停止加碼與攤平，等待收回合理底或跌破崩盤線的下一步確認。"
    else:
        level = "中"
        action = "觀察但不放大部位"
        summary = "尚未確認崩盤，但仍在坎中，應避免放大曝險。"
    return {
        "level": level,
        "action": action,
        "summary": summary,
        "not_investment_command": True,
        "guardrails": [
            "不是因為已腰斬就必須無條件砍出，而是依崩盤線與收盤確認執行預設風控。",
            "若跌破崩盤線且收不回，資金保全優先於攤平與凹單。",
            f"合理底下緣 {num(normal_low)} 未收回前，不視為洗底完成。",
            f"崩盤線 {num(crash_line)} 有效跌破後，不再用一般修正邏輯處理。",
        ],
    }


BAGUA_SEQUENCE = [
    "ZHEN",
    "XUN",
    "LI",
    "KUN",
    "DUI",
    "QIAN",
    "KAN",
    "GEN",
]

BAGUA_INFO = {
    "KAN": {
        "gua": "坎",
        "label": "深跌危機",
        "plain": "水深風險，市場承壓最重，重點是先活下來、等待止跌。",
    },
    "GEN": {
        "gua": "艮",
        "label": "築底止步",
        "plain": "跌勢開始停住，但還沒真正發動，重點是看低點是否守住。",
    },
    "ZHEN": {
        "gua": "震",
        "label": "起漲發動",
        "plain": "止跌後重新發動，代表行情從靜止轉為啟動；急跌不歸震，急跌歸坎。",
    },
    "XUN": {
        "gua": "巽",
        "label": "修復推進",
        "plain": "趨勢開始修補，資金慢慢回來，但還不是最強主升。",
    },
    "LI": {
        "gua": "離",
        "label": "主升明亮",
        "plain": "趨勢被市場看見，行情較明確，屬於可順勢觀察的階段。",
    },
    "KUN": {
        "gua": "坤",
        "label": "高檔承載",
        "plain": "漲多後進入承載與換手，仍可能撐住，但市場負重變大。",
    },
    "DUI": {
        "gua": "兌",
        "label": "亢奮收穫",
        "plain": "市場樂觀、獲利感強，但追價與反轉風險升高。",
    },
    "QIAN": {
        "gua": "乾",
        "label": "極盛轉折",
        "plain": "趨勢極盛，容易盛極而衰，需防由強轉弱後進入坎。",
    },
}


BAGUA_TRADE_ANNOTATIONS = {
    "ZHEN": {
        "label": "買點觀察",
        "bias": "buy_watch",
        "action": "試單建倉候選，只能在突破點站穩、量價改善時成立。",
        "buy_behavior": "小部位試單、觀察突破後回測不破。",
        "sell_behavior": "不急著賣；若跌回發動點，先退回觀察。",
        "condition": "止跌後轉強、低點不再破、短線重新站回關鍵均線。",
        "risk": "跌回發動點或量縮假突破時，震啟動失效。",
    },
    "XUN": {
        "label": "順勢持有",
        "bias": "hold_or_add_watch",
        "action": "順勢抱牢候選，是否加碼需看資金比例與族群擴散。",
        "buy_behavior": "回測守線可評估比例加碼，不追急拉。",
        "sell_behavior": "趨勢未破前以持有為主；跌破短均才降碼。",
        "condition": "高低點墊高、主流族群擴散、回檔守住短中均。",
        "risk": "跌破短均且無法收回時，順勢推進降級。",
    },
    "LI": {
        "label": "賣點保護",
        "bias": "take_profit_watch",
        "action": "分批獲利了結候選，重點是保護已出現的利潤。",
        "buy_behavior": "只等健康回檔，不在狂熱急拉處追價。",
        "sell_behavior": "分批實現利潤，保護已達10%至15%之類的合理成果。",
        "condition": "主升明確但開始過熱、漲幅集中、追價速度過快。",
        "risk": "爆量不漲、長上影或開高走低時，容易轉入坤高檔承載。",
    },
    "KUN": {
        "label": "減碼防守",
        "bias": "reduce_watch",
        "action": "高檔換手階段，優先保護本金與檢查承接力。",
        "buy_behavior": "只買回測有承接的強勢股，不買弱彈。",
        "sell_behavior": "逢高調節、汰弱留強，確認是否只是換手不是出貨。",
        "condition": "漲多後橫盤、成交放大、指數撐住但個股輪流震盪。",
        "risk": "承載失敗會轉成兌誘多或乾轉折壓力。",
    },
    "DUI": {
        "label": "禁追警戒",
        "bias": "avoid_chase",
        "action": "反彈誘惑高，只看確認，不盲目追高。",
        "buy_behavior": "不追高；等突破壓力後回測站穩才重新評估。",
        "sell_behavior": "反彈遇壓先保護部位，防止漂亮反彈變誘多。",
        "condition": "市場樂觀、反彈漂亮，但尚未證明能突破並守住壓力。",
        "risk": "反彈無量、爆量不漲或隔日不續攻時，容易成為誘多。",
    },
    "QIAN": {
        "label": "風控賣點",
        "bias": "defensive_sell_watch",
        "action": "極盛轉折警戒，降低曝險候選，避免由強轉弱還誤判安全。",
        "buy_behavior": "停止新增風險部位，除非重新站穩並解除轉弱訊號。",
        "sell_behavior": "破線、失速、權值轉弱時提高防守，必要時降低曝險。",
        "condition": "高檔加速後出現失速、破線、或權值股轉弱。",
        "risk": "若防守線失守且收不回，容易轉入坎深跌。",
    },
    "KAN": {
        "label": "空手防守",
        "bias": "cash_defense",
        "action": "保留現金與停止攤平候選，等待止跌三重確認。",
        "buy_behavior": "不攤平；只記錄候選名單，等止跌確認。",
        "sell_behavior": "若崩盤三重核對成立，優先風控與保留現金。",
        "condition": "主跌、恐慌、合理底失守或低點連續下移。",
        "risk": "未確認止跌就搶反彈，容易陷入二次破底。",
    },
    "GEN": {
        "label": "買前篩選",
        "bias": "screening_watch",
        "action": "篩選新底型與強勢標的候選，等待艮轉震確認。",
        "buy_behavior": "建立觀察清單，小量測試需等低點不破與站回短均。",
        "sell_behavior": "弱勢反彈仍先整理，不把止跌誤認為主升。",
        "condition": "低點停止下移、量縮止跌、跌深股與強勢股開始分化。",
        "risk": "只有不跌還不夠，未站回關鍵線前仍可能再探底。",
    },
}


def analyze_bagua_lifecycle(scored: pd.DataFrame, signal_date: str, candle: dict, washout: dict) -> dict:
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["date"] <= pd.to_datetime(signal_date)].dropna(subset=["close"]).reset_index(drop=True)
    if len(data) < 260:
        return {"enabled": False, "message": "資料不足，無法計算八卦生命週期。"}

    daily = classify_bagua_timeframe(data, "日線", "D", window=120, short_window=20, mid_window=60)
    weekly_data = resample_market_frame(data, "W-FRI")
    weekly = classify_bagua_timeframe(weekly_data, "週線", "W", window=104, short_window=8, mid_window=26)
    monthly_data = resample_market_frame(data, "ME")
    monthly = classify_bagua_timeframe(monthly_data, "月線", "M", window=60, short_window=3, mid_window=12)
    primary = select_bagua_primary(monthly, weekly, daily)
    resonance = bagua_resonance(monthly, weekly, daily)
    kline_check = bagua_kline_confirmation(primary, daily, candle, washout)
    historical = bagua_historical_stats(data, daily.get("code"))
    bottom_watch = bagua_bottom_watch(data)
    role_classification = classify_bagua_roles(data, monthly, weekly, daily)

    return {
        "enabled": True,
        "signal_date": signal_date,
        "sequence": [bagua_brief(code) for code in BAGUA_SEQUENCE],
        "primary_timeframe": primary.get("timeframe"),
        "primary": primary,
        "monthly": monthly,
        "weekly": weekly,
        "daily": daily,
        "resonance": resonance,
        "kline_confirmation": kline_check,
        "historical_stats": historical,
        "bottom_watch": bottom_watch,
        "roles": role_classification,
        "plain_summary": bagua_plain_summary(primary, resonance, kline_check),
    }


def classify_bagua_roles(
    data: pd.DataFrame,
    monthly: dict,
    weekly: dict,
    daily: dict,
) -> dict:
    """Separate background, price location, and market path state.

    These are different questions and must not be collapsed into one gua.
    Price gua uses only the close's location in a frozen lookback range. State
    gua uses the ordered peak-to-trough-to-recovery path.
    """
    price_windows = [
        classify_price_location_gua(data, 120),
        classify_price_location_gua(data, 60),
        classify_price_location_gua(data, 20),
    ]
    state = classify_market_state_gua(data)
    background = {
        "role": "background_gua",
        "source": "monthly_lifecycle",
        "gua": monthly.get("gua"),
        "code": monthly.get("code"),
        "label": monthly.get("label"),
        "progress": monthly.get("phase_progress"),
        "trade_annotation": bagua_trade_annotation(monthly.get("code")),
        "meaning": "長週期位於哪個風險階段，不直接回答短線正在發生什麼。",
    }
    return {
        "classification_contract": "background_price_state_v1",
        "background_gua": background,
        "price_gua": {
            "role": "price_location_gua",
            "method": "close location in eight equal bands of each lookback close range",
            "windows": price_windows,
            "meaning": "只回答目前價格位於各自區間哪裡，不描述修復或惡化路徑。",
        },
        "state_gua": state,
        "legacy_timeframes": {
            "monthly": {"gua": monthly.get("gua"), "label": monthly.get("label")},
            "weekly": {"gua": weekly.get("gua"), "label": weekly.get("label")},
            "daily": {"gua": daily.get("gua"), "label": daily.get("label")},
        },
        "single_gua_prohibited": True,
        "summary": (
            f"背景卦{background.get('gua') or 'NA'}；"
            f"價位卦120/60/20日為"
            f"{'/'.join(item.get('gua', 'NA') for item in price_windows)}；"
            f"狀態卦{state.get('gua', 'NA')}/{state.get('label', '資料不足')}。"
        ),
    }


def classify_price_location_gua(data: pd.DataFrame, window: int) -> dict:
    frame = data.dropna(subset=["close"]).tail(window)
    if len(frame) < window:
        return {"window": window, "status": "insufficient_history", "gua": None}
    low = float(frame["close"].min())
    high = float(frame["close"].max())
    close = float(frame.iloc[-1]["close"])
    position = 0.5 if high == low else (close - low) / (high - low)
    band = min(7, max(0, int(position * 8)))
    price_sequence = ["KAN", "GEN", "ZHEN", "XUN", "LI", "KUN", "DUI", "QIAN"]
    code = price_sequence[band]
    return {
        "window": window,
        "status": "classified",
        "position": float(position),
        "band": band + 1,
        "code": code,
        "gua": BAGUA_INFO[code]["gua"],
        "label": BAGUA_INFO[code]["label"],
        "trade_annotation": bagua_trade_annotation(code),
        "range_low": low,
        "range_high": high,
        "close": close,
    }


def classify_market_state_gua(data: pd.DataFrame, window: int = 120) -> dict:
    frame = data.dropna(subset=["close", "low"]).tail(window).reset_index(drop=True)
    if len(frame) < 20:
        return {"role": "state_gua", "status": "insufficient_history", "gua": None}
    peak_index = int(frame["close"].idxmax())
    after_peak = frame.iloc[peak_index:]
    trough_index = int(after_peak["close"].idxmin())
    peak = float(frame.loc[peak_index, "close"])
    trough = float(frame.loc[trough_index, "close"])
    close = float(frame.iloc[-1]["close"])
    drawdown = trough / peak - 1 if peak else 0.0
    recovery = close / trough - 1 if trough else 0.0
    days_after_trough = len(frame) - 1 - trough_index

    if drawdown <= -0.08:
        if recovery >= 0.06 and days_after_trough >= 2:
            code = "ZHEN"
            status = "repair_activation"
            reason = "急跌後已明顯脫離低點，進入震啟動修復，但尚非完整新升段。"
        elif days_after_trough >= 2 and recovery >= 0:
            code = "GEN"
            status = "bottoming_watch"
            reason = "急跌後停止破低，但尚未明顯脫離底部，屬艮止跌觀察。"
        else:
            code = "KAN"
            status = "crisis_continuation"
            reason = "高點後急跌且尚未形成持續修復，仍屬坎危機。"
    else:
        code = classify_price_location_gua(frame, min(window, len(frame))).get("code") or "KAN"
        status = "no_recent_crisis_path"
        reason = "近期沒有符合急跌後修復的路徑，狀態暫依現有結構觀察。"
    return {
        "role": "state_gua",
        "status": status,
        "code": code,
        "gua": BAGUA_INFO[code]["gua"],
        "label": BAGUA_INFO[code]["label"],
        "trade_annotation": bagua_trade_annotation(code),
        "peak_date": str(pd.to_datetime(frame.loc[peak_index, "date"]).date()),
        "peak_close": peak,
        "trough_date": str(pd.to_datetime(frame.loc[trough_index, "date"]).date()),
        "trough_close": trough,
        "drawdown": float(drawdown),
        "recovery_from_trough": float(recovery),
        "days_after_trough": int(days_after_trough),
        "reason": reason,
        "confirmation": "站穩修復區、廣度與量價改善後，才可由震啟動升級為完整推進。",
        "invalidation": "重新跌近或跌破前低，震啟動失效並退回坎。",
        "formal_direction_signal": None,
    }


def resample_market_frame(data: pd.DataFrame, rule: str) -> pd.DataFrame:
    frame = data.copy()
    frame = frame.set_index("date")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in frame.columns:
        agg["volume"] = "sum"
    output = frame.resample(rule).agg(agg).dropna(subset=["close"]).reset_index()
    return output


def classify_bagua_timeframe(data: pd.DataFrame, label: str, code_label: str, window: int, short_window: int, mid_window: int) -> dict:
    prepared = prepare_bagua_features(data, window, short_window, mid_window)
    if prepared.empty:
        return {"enabled": False, "timeframe": label, "message": "資料不足"}
    row = prepared.iloc[-1]
    code = classify_bagua_row(row)
    progress = bagua_phase_progress(code, row)
    return {
        "enabled": True,
        "timeframe": label,
        "timeframe_code": code_label,
        "date": str(row["date"].date()),
        "code": code,
        "gua": BAGUA_INFO[code]["gua"],
        "label": BAGUA_INFO[code]["label"],
        "plain": BAGUA_INFO[code]["plain"],
        "trade_annotation": bagua_trade_annotation(code),
        "sequence_index": BAGUA_SEQUENCE.index(code) + 1,
        "phase_progress": progress,
        "next": bagua_brief(next_bagua(code)),
        "fallback": bagua_brief(previous_bagua(code)),
        "metrics": {
            "close": safe_float(row.get("close")),
            "range_position": safe_float(row.get("bagua_pos")),
            "short_return": safe_float(row.get("bagua_ret_short")),
            "mid_return": safe_float(row.get("bagua_ret_mid")),
            "long_return": safe_float(row.get("bagua_ret_long")),
            "drawdown": safe_float(row.get("bagua_drawdown")),
            "ma_short": safe_float(row.get("bagua_ma_short")),
            "ma_mid": safe_float(row.get("bagua_ma_mid")),
        },
        "confirm_condition": bagua_confirm_condition(code),
        "fail_condition": bagua_fail_condition(code),
    }


def select_bagua_primary(monthly: dict, weekly: dict, daily: dict) -> dict:
    if daily.get("code") == "KAN" and weekly.get("code") == "KAN":
        return weekly
    if daily.get("code") == "KAN" and monthly.get("code") in ["KUN", "DUI", "QIAN"]:
        return daily
    return monthly if monthly.get("enabled") else weekly if weekly.get("enabled") else daily


def prepare_bagua_features(data: pd.DataFrame, window: int, short_window: int, mid_window: int) -> pd.DataFrame:
    prepared = data[["date", "open", "high", "low", "close"]].copy()
    prepared["bagua_high"] = prepared["close"].rolling(window).max()
    prepared["bagua_low"] = prepared["close"].rolling(window).min()
    range_width = (prepared["bagua_high"] - prepared["bagua_low"]).replace(0, pd.NA)
    prepared["bagua_pos"] = (prepared["close"] - prepared["bagua_low"]) / range_width
    prepared["bagua_ret_short"] = prepared["close"] / prepared["close"].shift(short_window) - 1
    prepared["bagua_ret_mid"] = prepared["close"] / prepared["close"].shift(mid_window) - 1
    prepared["bagua_ret_long"] = prepared["close"] / prepared["close"].shift(window) - 1
    prepared["bagua_drawdown"] = prepared["close"] / prepared["bagua_high"] - 1
    prepared["bagua_ma_short"] = prepared["close"].rolling(short_window).mean()
    prepared["bagua_ma_mid"] = prepared["close"].rolling(mid_window).mean()
    return prepared.dropna(subset=["bagua_pos", "bagua_ret_short", "bagua_ret_mid", "bagua_ret_long"])


def classify_bagua_row(row) -> str:
    close = row.get("close")
    pos = row.get("bagua_pos")
    ret_short = row.get("bagua_ret_short")
    ret_mid = row.get("bagua_ret_mid")
    ret_long = row.get("bagua_ret_long")
    drawdown = row.get("bagua_drawdown")
    ma_short = row.get("bagua_ma_short")
    ma_mid = row.get("bagua_ma_mid")
    above_short = close >= ma_short
    above_mid = close >= ma_mid

    if drawdown <= -0.30 and ret_mid < -0.05:
        return "KAN"
    if ret_short <= -0.06 and drawdown <= -0.08:
        return "KAN"
    if pos <= 0.28 and ret_short >= -0.03:
        return "GEN"
    if pos <= 0.42 and ret_short > 0.03:
        return "ZHEN"
    if pos <= 0.60 and ret_mid > 0 and above_short:
        return "XUN"
    if pos <= 0.78 and ret_mid > 0.04 and above_mid:
        return "LI"
    if pos <= 0.88 and ret_mid >= -0.02:
        return "KUN"
    if pos <= 0.95 and ret_long > 0.12:
        return "DUI"
    return "QIAN"


def bagua_phase_progress(code: str, row) -> float:
    pos = safe_float(row.get("bagua_pos")) or 0
    ret_short = safe_float(row.get("bagua_ret_short")) or 0
    drawdown = safe_float(row.get("bagua_drawdown")) or 0
    if code == "KAN":
        return clamp((ret_short + 0.18) / 0.18)
    if code == "GEN":
        return clamp(pos / 0.28)
    if code == "ZHEN":
        return clamp(abs(ret_short) / 0.10)
    if code == "XUN":
        return clamp((pos - 0.30) / 0.30)
    if code == "LI":
        return clamp((pos - 0.55) / 0.25)
    if code == "KUN":
        return clamp((pos - 0.70) / 0.20)
    if code == "DUI":
        return clamp((pos - 0.82) / 0.15)
    if code == "QIAN":
        return clamp((pos - 0.88) / 0.12 + abs(drawdown) / 0.10)
    return clamp(pos)


def next_bagua(code: str) -> str:
    index = BAGUA_SEQUENCE.index(code)
    return BAGUA_SEQUENCE[(index + 1) % len(BAGUA_SEQUENCE)]


def previous_bagua(code: str) -> str:
    index = BAGUA_SEQUENCE.index(code)
    return BAGUA_SEQUENCE[(index - 1) % len(BAGUA_SEQUENCE)]


def bagua_trade_annotation(code: str | None) -> dict:
    if code not in BAGUA_INFO or code not in BAGUA_TRADE_ANNOTATIONS:
        return {
            "code": code,
            "gua": None,
            "label": "資料不足",
            "bias": "unknown",
            "action": "資料不足，不能產生買賣標註。",
            "buy_behavior": "資料不足。",
            "sell_behavior": "資料不足。",
            "condition": "等待資料補齊。",
            "risk": "資料不足時禁止強行解讀。",
            "guardrail": "買賣標註只作風控與行為對比，不是投資命令。",
        }
    info = BAGUA_INFO[code]
    trade = BAGUA_TRADE_ANNOTATIONS[code].copy()
    trade.update(
        {
            "code": code,
            "gua": info["gua"],
            "stage_label": info["label"],
            "guardrail": "買賣標註只作風控與行為對比，不是投資命令。",
        }
    )
    return trade


def bagua_brief(code: str) -> dict:
    info = BAGUA_INFO[code]
    return {
        "code": code,
        "gua": info["gua"],
        "label": info["label"],
        "trade_annotation": bagua_trade_annotation(code),
    }


def bagua_resonance(monthly: dict, weekly: dict, daily: dict) -> dict:
    active = [item for item in [monthly, weekly, daily] if item.get("enabled")]
    if not active:
        return {"label": "資料不足", "plain_summary": "多週期資料不足，無法判斷共振。", "score": 0}
    indexes = [item["sequence_index"] for item in active]
    spread = max(indexes) - min(indexes)
    gua_text = " / ".join(f"{item['timeframe']}{item['gua']}" for item in active)
    month_code = monthly.get("code")
    day_code = daily.get("code")
    if month_code in ["LI", "KUN", "DUI", "QIAN"] and day_code == "KAN":
        return {
            "label": "高位入坎",
            "plain_summary": f"{gua_text}。大週期仍在高位，但日線已入坎下滑，代表高檔承載失敗或轉弱警戒。",
            "score": -1,
        }
    if spread <= 1:
        return {
            "label": "多週期同向",
            "plain_summary": f"{gua_text}。月、週、日位置接近，訊號較一致。",
            "score": 2,
        }
    if spread <= 3:
        return {
            "label": "多週期部分共振",
            "plain_summary": f"{gua_text}。大方向有參考價值，但短線仍需確認。",
            "score": 1,
        }
    return {
        "label": "多週期分歧",
        "plain_summary": f"{gua_text}。月、週、日差距大，代表市場處於換檔或混沌期。",
        "score": -1,
    }


def bagua_kline_confirmation(primary: dict, daily: dict, candle: dict, washout: dict) -> dict:
    candle_type = candle.get("type")
    washout_type = washout.get("type")
    primary_code = primary.get("code")
    daily_code = daily.get("code")
    if candle_type in ["long_bear", "gap_up_failed"] and washout_type == "failed_washout":
        return {
            "label": "K線支持轉弱",
            "plain_summary": "長黑K或開高走低，加上失敗洗盤，支持坎卦陷落或高位轉弱警戒。",
            "alignment": "bearish_confirm",
        }
    if candle_type in ["long_lower_shadow", "gap_down_reversal"] and washout_type in ["strong_washout", "normal_washout", "gap_down_recovery"]:
        return {
            "label": "K線支持止跌",
            "plain_summary": "下影線或開低走高，加上洗盤拉回，支持艮卦止跌或震卦起漲觀察。",
            "alignment": "bottoming_confirm",
        }
    if primary_code in ["LI", "KUN", "DUI", "QIAN"] and daily_code == "KAN":
        return {
            "label": "大週期高位，日線入坎",
            "plain_summary": "大週期仍在高位，但日線已入坎下滑，需看隔日是否止跌，否則高位轉弱風險升高。",
            "alignment": "mixed_warning",
        }
    return {
        "label": "K線尚未給出強確認",
        "plain_summary": "目前K線與卦位沒有形成強烈同向訊號，仍需下一交易日確認。",
        "alignment": "neutral",
    }


def bagua_historical_stats(data: pd.DataFrame, code: str | None) -> dict:
    if not code:
        return {"sample_count": 0, "message": "無卦位資料"}
    prepared = prepare_bagua_features(data, 120, 20, 60)
    if prepared.empty:
        return {"sample_count": 0, "message": "資料不足"}
    prepared["bagua_code"] = prepared.apply(classify_bagua_row, axis=1)
    for horizon in [20, 60]:
        prepared[f"fwd_{horizon}d"] = prepared["close"].shift(-horizon) / prepared["close"] - 1
    sample = prepared[(prepared["bagua_code"] == code) & prepared["fwd_20d"].notna()]
    rows = []
    for horizon in [20, 60]:
        valid = sample.dropna(subset=[f"fwd_{horizon}d"])
        rows.append(
            {
                "horizon_days": horizon,
                "sample_count": int(len(valid)),
                "avg_return": safe_float(valid[f"fwd_{horizon}d"].mean()) if len(valid) else None,
                "up_rate": safe_float((valid[f"fwd_{horizon}d"] > 0).mean()) if len(valid) else None,
                "down_5pct_rate": safe_float((valid[f"fwd_{horizon}d"] <= -0.05).mean()) if len(valid) else None,
            }
        )
    return {"sample_count": int(len(sample)), "current_code": code, "by_horizon": rows}


def bagua_bottom_watch(data: pd.DataFrame) -> dict:
    if len(data) < 20:
        return {"enabled": False, "message": "資料不足，無法判斷止跌確認。"}
    frame = data.sort_values("date").reset_index(drop=True).copy()
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    latest = frame.iloc[-1]
    prior = frame.iloc[:-1]
    recent = prior.tail(10)
    if recent.empty:
        return {"enabled": False, "message": "資料不足，無法判斷止跌確認。"}

    prior_close_floor = safe_float(recent["close"].min())
    prior_low_floor = safe_float(recent["low"].min())
    latest_low = safe_float(latest.get("low"))
    latest_close = safe_float(latest.get("close"))
    latest_open = safe_float(latest.get("open"))
    ma5 = safe_float(frame["close"].rolling(5).mean().iloc[-1])
    broke_prior_low = (
        latest_low is not None
        and prior_low_floor is not None
        and latest_low < prior_low_floor
    )
    broke_prior_close = (
        latest_close is not None
        and prior_close_floor is not None
        and latest_close < prior_close_floor
    )
    if broke_prior_low or broke_prior_close:
        status = "kan_not_bottomed"
        label = "坎中，未到艮"
        summary = "最新K線跌破近期低點或低收盤，代表仍在陷落段；至少要先停止破低，才有止跌觀察資格。"
    elif latest_close is not None and prior_close_floor is not None and latest_close >= prior_close_floor:
        status = "bottom_watch"
        label = "止跌觀察"
        summary = "沒有續破近期低點，且收回前一個低收盤，開始有艮卦止住的雛形。"
    else:
        status = "unconfirmed"
        label = "尚未確認"
        summary = "尚未續破，但也沒有足夠收復，仍需下一交易日確認。"
    return {
        "enabled": True,
        "status": status,
        "label": label,
        "summary": summary,
        "date": str(pd.to_datetime(latest.get("date")).date()),
        "latest_low_hold": latest_low,
        "first_reclaim_close": prior_close_floor,
        "strong_reclaim_open": latest_open,
        "ma5_reclaim": ma5,
        "rule": "1天不破低只算反彈；2天不破低才進入止跌觀察；3天不破低且收復前低或短期均線，才比較像艮卦止跌。",
    }


def bagua_confirm_condition(code: str) -> str:
    return {
        "KAN": "至少1天不再破低只算反彈；連續2天不破低才進入止跌觀察；連續3天不破低且收復前低或短期均線，才有機會轉艮。",
        "GEN": "守住前低，並突破短期高點，才有機會轉震。",
        "ZHEN": "低位止穩後向上發動，並站穩突破點，才算震卦有效。",
        "XUN": "維持短期均線之上，回檔不破前低，修復才算延續。",
        "LI": "沿均線上攻且回檔不破，主升才算穩定。",
        "KUN": "高檔換手後仍守住中期均線，才代表承載成功。",
        "DUI": "高檔亢奮後仍能量價健康，否則容易轉乾或回落。",
        "QIAN": "極盛後若仍能守住高檔支撐，才可延長強勢；跌破則防回坎。",
    }.get(code, "等待下一交易日確認。")


def bagua_fail_condition(code: str) -> str:
    return {
        "KAN": "續破新低且反彈無力，代表危機延續。",
        "GEN": "跌破築底低點，會退回坎卦危機。",
        "ZHEN": "起漲突破失敗並跌回發動起點，代表震卦失效。",
        "XUN": "跌回修復起點，代表修復失敗。",
        "LI": "跌破主升支撐與中期均線，主升要降級。",
        "KUN": "高檔承載失敗，容易轉為入坎下滑或高檔回落。",
        "DUI": "爆量不漲、長上影或開高走低，代表亢奮退潮。",
        "QIAN": "跌破高檔防守線，盛極轉衰，容易進入坎卦循環。",
    }.get(code, "若跌破關鍵支撐，原判斷降級。")


def bagua_plain_summary(primary: dict, resonance: dict, kline_check: dict) -> str:
    if not primary.get("enabled"):
        return "八卦資料不足。"
    return (
        f"主卦以{primary['timeframe']}判斷為{primary['gua']}卦/{primary['label']}。"
        f"{resonance.get('plain_summary', '')} "
        f"{kline_check.get('plain_summary', '')}"
    )


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def analyze_tradeable_rally_segments(scored: pd.DataFrame, signal_date: str) -> dict:
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data[(data["date"] >= pd.to_datetime("2005-01-01")) & (data["date"] <= pd.to_datetime(signal_date))]
    data = data.dropna(subset=["close"]).reset_index(drop=True)
    if data.empty:
        return {
            "enabled": False,
            "message": "資料不足，無法計算年度可交易波段。",
        }

    min_gain = 0.06
    pullback = 0.04
    max_days = 60
    segments, active, candidate = detect_tradeable_rally_segments(data, min_gain, pullback, max_days)
    closed_segments = [segment for segment in segments if not segment.get("active")]
    signal_year = int(pd.to_datetime(signal_date).year)
    years = list(range(int(data["date"].dt.year.min()), signal_year + 1))
    annual_counts = tradeable_segment_annual_counts(closed_segments, years)
    average_per_year = sum(annual_counts.values()) / len(annual_counts) if annual_counts else 0
    median_per_year = float(pd.Series(list(annual_counts.values())).median()) if annual_counts else 0
    current_year_count = int(annual_counts.get(signal_year, 0))
    rolling_252d_count = count_segments_since(closed_segments, data.iloc[max(0, len(data) - 252)]["date"])
    frequency_state = tradeable_frequency_state(current_year_count, rolling_252d_count, average_per_year)
    current_position = classify_tradeable_cycle_position(data, active, candidate, closed_segments, min_gain, pullback)
    recent_segments = sorted(closed_segments, key=lambda item: item["peak_date"])[-5:]

    return {
        "enabled": True,
        "signal_date": str(data.iloc[-1]["date"].date()),
        "definition": {
            "min_gain": min_gain,
            "pullback": pullback,
            "max_trading_days": max_days,
            "plain_text": "從低點起算上漲至少6%才算可交易漲幅區段；從高點回落4%或觀察滿60個交易日，視為該段結束或進入防守。",
        },
        "stats": {
            "history_start": str(data.iloc[0]["date"].date()),
            "history_end": str(data.iloc[-1]["date"].date()),
            "total_segments": len(closed_segments),
            "average_per_year": average_per_year,
            "median_per_year": median_per_year,
            "current_year": signal_year,
            "current_year_count": current_year_count,
            "rolling_252d_count": rolling_252d_count,
            "frequency_state": frequency_state,
            "annual_counts_recent": [
                {"year": int(year), "count": int(annual_counts.get(year, 0))}
                for year in years[-10:]
            ],
        },
        "current_position": current_position,
        "active_segment": active,
        "candidate_segment": candidate,
        "recent_segments": recent_segments,
    }


def detect_tradeable_rally_segments(data: pd.DataFrame, min_gain: float, pullback: float, max_days: int) -> tuple[list[dict], dict | None, dict]:
    segments = []
    state = "seeking"
    trough_index = 0
    trough_close = float(data.iloc[0]["close"])
    peak_index = 0
    peak_close = trough_close
    confirm_index = None

    for index in range(1, len(data)):
        close = float(data.iloc[index]["close"])
        if state == "seeking":
            if close < trough_close:
                trough_index = index
                trough_close = close
            gain = close / trough_close - 1 if trough_close else 0
            if gain >= min_gain:
                state = "rally"
                confirm_index = index
                peak_index = index
                peak_close = close
            continue

        if close > peak_close:
            peak_index = index
            peak_close = close

        drawdown = close / peak_close - 1 if peak_close else 0
        duration = index - trough_index
        if drawdown <= -pullback or duration >= max_days:
            segments.append(
                build_tradeable_segment(
                    data,
                    trough_index,
                    confirm_index,
                    peak_index,
                    index,
                    active=False,
                    end_reason="pullback" if drawdown <= -pullback else "max_days",
                )
            )
            state = "seeking"
            trough_index = index
            trough_close = close
            peak_index = index
            peak_close = close
            confirm_index = None

    candidate = build_candidate_segment(data, trough_index)
    active = None
    if state == "rally":
        active = build_tradeable_segment(
            data,
            trough_index,
            confirm_index,
            peak_index,
            len(data) - 1,
            active=True,
            end_reason="active",
        )
    return segments, active, candidate


def build_tradeable_segment(
    data: pd.DataFrame,
    trough_index: int,
    confirm_index: int | None,
    peak_index: int,
    end_index: int,
    active: bool,
    end_reason: str,
) -> dict:
    trough = data.iloc[trough_index]
    peak = data.iloc[peak_index]
    end = data.iloc[end_index]
    confirm = data.iloc[confirm_index] if confirm_index is not None else peak
    trough_close = float(trough["close"])
    peak_close = float(peak["close"])
    end_close = float(end["close"])
    return {
        "trough_date": str(trough["date"].date()),
        "confirm_date": str(confirm["date"].date()),
        "peak_date": str(peak["date"].date()),
        "end_date": str(end["date"].date()),
        "trough_close": trough_close,
        "confirm_close": float(confirm["close"]),
        "peak_close": peak_close,
        "end_close": end_close,
        "gain": peak_close / trough_close - 1 if trough_close else None,
        "end_return": end_close / trough_close - 1 if trough_close else None,
        "drawdown_from_peak": end_close / peak_close - 1 if peak_close else None,
        "duration_trading_days": int(end_index - trough_index),
        "active": active,
        "end_reason": end_reason,
    }


def build_candidate_segment(data: pd.DataFrame, trough_index: int) -> dict:
    trough = data.iloc[trough_index]
    current = data.iloc[-1]
    trough_close = float(trough["close"])
    current_close = float(current["close"])
    return {
        "trough_date": str(trough["date"].date()),
        "trough_close": trough_close,
        "current_close": current_close,
        "gain_from_trough": current_close / trough_close - 1 if trough_close else None,
        "duration_trading_days": int(len(data) - 1 - trough_index),
    }


def tradeable_segment_annual_counts(segments: list[dict], years: list[int]) -> dict[int, int]:
    counts = {int(year): 0 for year in years}
    for segment in segments:
        year = int(pd.to_datetime(segment["peak_date"]).year)
        if year in counts:
            counts[year] += 1
    return counts


def count_segments_since(segments: list[dict], start_date) -> int:
    start = pd.to_datetime(start_date)
    return sum(pd.to_datetime(segment["peak_date"]) >= start for segment in segments)


def tradeable_frequency_state(current_year_count: int, rolling_252d_count: int, average_per_year: float) -> dict:
    if average_per_year <= 0:
        return {
            "level": "unknown",
            "label": "資料不足",
            "ratio_current_year": None,
            "ratio_rolling_252d": None,
            "plain_summary": "歷史樣本不足，無法判斷今年波段頻率是否異常。",
            "risk_hint": "先累積資料，再比較年度節奏。",
        }

    ratio_current = current_year_count / average_per_year
    ratio_rolling = rolling_252d_count / average_per_year
    max_ratio = max(ratio_current, ratio_rolling)
    if max_ratio >= 1.75:
        level = "extreme_high"
        label = "極端高頻波段"
        summary = "近一年可交易波段次數遠高於歷史平均，代表市場節奏很快，容易急漲急跌。"
        risk = "追高風險明顯提高，應更重視失效防守價與分段停利。"
    elif max_ratio >= 1.35:
        level = "high"
        label = "高頻波段"
        summary = "近一年波段次數明顯高於歷史平均，代表機會較多，但每段持續時間可能縮短。"
        risk = "不宜把每次反彈都當長多，應確認段位後再進攻。"
    elif max_ratio >= 1.10:
        level = "above_average"
        label = "略高於常態"
        summary = "今年波段次數已高於歷史平均，市場比一般年份更活躍。"
        risk = "仍有機會，但追價前要確認不是末升段或回落段。"
    elif max_ratio <= 0.70:
        level = "quiet"
        label = "低頻波段"
        summary = "今年波段次數低於歷史平均，市場可能偏盤整或趨勢尚未展開。"
        risk = "不要因為久盤就硬做，等突破確認較穩。"
    else:
        level = "normal"
        label = "接近常態"
        summary = "今年波段次數接近歷史平均，市場節奏沒有明顯異常。"
        risk = "依照段位與確認價判斷，不需要額外放大風險假設。"

    return {
        "level": level,
        "label": label,
        "ratio_current_year": ratio_current,
        "ratio_rolling_252d": ratio_rolling,
        "plain_summary": summary,
        "risk_hint": risk,
    }


def classify_tradeable_cycle_position(
    data: pd.DataFrame,
    active: dict | None,
    candidate: dict,
    closed_segments: list[dict],
    min_gain: float,
    pullback: float,
) -> dict:
    current = data.iloc[-1]
    close = float(current["close"])
    recent_low_20 = float(data.tail(20)["close"].min())
    recent_high_20 = float(data.tail(20)["close"].max())
    latest_closed = closed_segments[-1] if closed_segments else None

    if active:
        gain = active.get("end_return")
        drawdown = active.get("drawdown_from_peak")
        duration = active.get("duration_trading_days", 0)
        if drawdown is not None and drawdown <= -(pullback * 0.75):
            label = "回落警戒段"
            code = "pullback_watch"
            summary = "已經有一段可交易漲幅，但從高點回落接近失守線，應把重點放在保護獲利。"
        elif gain is not None and gain >= 0.18 and duration >= 20:
            label = "末升段/高檔加速"
            code = "late_rally"
            summary = "漲幅已大且時間拉長，仍可能續強，但追價風險提高。"
        elif gain is not None and gain >= 0.10:
            label = "主升段"
            code = "main_rally"
            summary = "可交易漲幅已成立且仍未明顯失守，屬於順勢段。"
        else:
            label = "起漲確認段"
            code = "early_rally"
            summary = "剛脫離低點並達到可交易漲幅門檻，仍需觀察是否站穩。"
        return {
            "code": code,
            "label": label,
            "plain_summary": summary,
            "gain_from_segment_low": gain,
            "drawdown_from_segment_high": drawdown,
            "days_in_segment": duration,
            "confirm_level": active.get("peak_close"),
            "fail_level": active.get("peak_close") * (1 - pullback) if active.get("peak_close") is not None else None,
            "lighthouse": tradeable_lighthouse_text(code),
        }

    candidate_gain = candidate.get("gain_from_trough") or 0
    if candidate_gain >= min_gain * 0.65:
        code = "repair_watch"
        label = "修復段/起漲觀察"
        summary = "低點後已有修復，但尚未形成完整可交易漲幅段，需突破確認。"
    elif latest_closed and close / latest_closed["peak_close"] - 1 <= -pullback:
        code = "post_rally_pullback"
        label = "回落段/重新找底"
        summary = "上一段漲幅已失守，目前比較像回落後重新找支撐。"
    elif close <= recent_low_20 * 1.02:
        code = "bottoming_watch"
        label = "築底段/等待洗盤確認"
        summary = "位置接近短期低點，還沒有足夠證據證明新一段漲幅已啟動。"
    elif close >= recent_high_20 * 0.98:
        code = "breakout_watch"
        label = "突破觀察段"
        summary = "已接近短期高點，若補量突破，可能進入下一段可交易漲幅。"
    else:
        code = "range_wait"
        label = "整理段/等待方向"
        summary = "目前在短期區間中間，方向感不夠，較適合等確認訊號。"

    return {
        "code": code,
        "label": label,
        "plain_summary": summary,
        "gain_from_segment_low": candidate_gain,
        "drawdown_from_segment_high": close / recent_high_20 - 1 if recent_high_20 else None,
        "days_in_segment": candidate.get("duration_trading_days"),
        "confirm_level": candidate["trough_close"] * (1 + min_gain),
        "fail_level": recent_low_20,
        "lighthouse": tradeable_lighthouse_text(code),
    }


def tradeable_lighthouse_text(code: str) -> str:
    return {
        "bottoming_watch": "燈塔: 先找止跌與洗盤成功，不急著預設大漲。",
        "repair_watch": "燈塔: 觀察是否突破6%門檻，突破才算新一段可交易漲幅成立。",
        "breakout_watch": "燈塔: 若突破短期高點並站穩，可視為下一段漲幅啟動；若突破失敗要防假突破。",
        "early_rally": "燈塔: 起漲剛成立，重點是守住回落失效線。",
        "main_rally": "燈塔: 順勢段仍在，重點是不要被短線震盪洗出，但也要守紀律。",
        "late_rally": "燈塔: 末升段要防急拉後回落，追價要保守。",
        "pullback_watch": "燈塔: 已接近波段失守，重點從進攻改為防守。",
        "post_rally_pullback": "燈塔: 上一段已結束，等待新的低點、洗盤或修復訊號。",
        "range_wait": "燈塔: 茫海中先不要硬猜方向，等突破或跌破再判斷。",
    }.get(code, "燈塔: 先看位置，再看確認條件。")


def classify_washout(row) -> dict:
    low_pct = row.get("intraday_low_pct")
    recovery = row.get("close_recovery_ratio")
    close_ret = row.get("close_return_pct")
    open_gap = row.get("open_gap_pct")
    if pd.isna(low_pct) or pd.isna(recovery) or pd.isna(close_ret):
        return washout_pattern("unknown", "資料不足", "今日資料不足，無法判斷是否為洗盤。")

    if low_pct <= -0.015 and recovery >= 0.65 and close_ret >= -0.006:
        return washout_pattern("strong_washout", "強洗盤", "盤中急殺後收盤明顯拉回，低檔承接強，歷史上較容易進入修復或反彈。")
    if low_pct <= -0.012 and recovery >= 0.55:
        return washout_pattern("normal_washout", "一般洗盤", "盤中有明顯下殺且收盤拉回，但強度未達強洗盤，仍需隔日確認。")
    if low_pct <= -0.015 and recovery < 0.40:
        return washout_pattern("failed_washout", "失敗洗盤/真賣壓", "盤中急殺後沒有有效拉回，較像賣壓延續，不宜過早判定洗盤成功。")
    if open_gap <= -0.008 and recovery >= 0.65:
        return washout_pattern("gap_down_recovery", "跳空開低後拉回", "開盤受外部事件壓低後拉回，但歷史上不一定等於強洗盤。")
    return washout_pattern("none", "非洗盤", "今日沒有明顯洗盤結構，應以一般趨勢與盤前訊號判斷。")


def washout_pattern(kind: str, label: str, summary: str) -> dict:
    return {"type": kind, "label": label, "summary": summary}


def washout_historical_stats(data: pd.DataFrame, kind: str) -> dict:
    prepared = data[data["date"] >= pd.to_datetime("2005-01-01")].copy()
    prepared = prepared.dropna(subset=["fwd_5d", "fwd_20d"])
    masks = {
        "strong_washout": (
            (prepared["intraday_low_pct"] <= -0.015)
            & (prepared["close_recovery_ratio"] >= 0.65)
            & (prepared["close_return_pct"] >= -0.006)
        ),
        "normal_washout": (
            (prepared["intraday_low_pct"] <= -0.012)
            & (prepared["close_recovery_ratio"] >= 0.55)
        ),
        "failed_washout": (
            (prepared["intraday_low_pct"] <= -0.015)
            & (prepared["close_recovery_ratio"] < 0.40)
        ),
        "gap_down_recovery": (
            (prepared["open_gap_pct"] <= -0.008)
            & (prepared["close_recovery_ratio"] >= 0.65)
        ),
    }
    mask = masks.get(kind)
    if mask is None:
        return {"sample_count": 0, "message": "此類型無專屬歷史統計。", "by_horizon": []}
    sample = prepared[mask].copy()
    rows = []
    for horizon in [1, 5, 10, 20, 60]:
        col = f"fwd_{horizon}d"
        values = sample[col].dropna()
        if values.empty:
            continue
        rows.append(
            {
                "horizon_days": horizon,
                "avg_return": float(values.mean()),
                "up_rate": float((values > 0).mean()),
                "down_rate": float((values < -0.005).mean()),
            }
        )
    return {
        "sample_count": int(len(sample)),
        "sample_ratio": float(len(sample) / len(prepared)) if len(prepared) else 0,
        "by_horizon": rows,
    }


def washout_invalidation(row, pattern: dict) -> dict:
    low = safe_float(row.get("low"))
    close = safe_float(row.get("close"))
    if pattern["type"] not in ["strong_washout", "normal_washout", "gap_down_recovery"]:
        return {
            "active": False,
            "rule": "非洗盤或失敗洗盤，不啟用洗盤成功條件。",
            "key_level": low,
        }
    return {
        "active": True,
        "rule": "隔日若跌破今日低點，洗盤成功判斷降級；若守住低點並站回收盤，承接力確認。",
        "key_level": low,
        "confirmation_level": close,
    }


def washout_event_trigger(row) -> dict:
    external_score = int(row.get("external_score", 0))
    night_score = int(row.get("night_futures_score", 0))
    triggers = []
    if external_score < 0:
        triggers.append("外部市場偏弱")
    if night_score < 0:
        triggers.append("台指期夜盤偏弱")
    if safe_float(row.get("open_gap_pct")) is not None and row.get("open_gap_pct") <= -0.008:
        triggers.append("開盤跳空下跌")
    if not triggers:
        triggers.append("未偵測到明顯外部觸發")
    return {
        "likely_triggered_by_event": any(t != "未偵測到明顯外部觸發" for t in triggers),
        "triggers": triggers,
        "plain_text": "、".join(triggers),
    }


def cause_reasons(external_score: int, night_score: int, panic_score: int) -> list[str]:
    reasons = []
    if external_score < 0:
        reasons.append("外部市場偏弱，對台股形成壓力。")
    elif external_score > 0:
        reasons.append("外部市場偏強，對台股有支撐。")
    else:
        reasons.append("外部市場訊號中性或資料不足。")
    if night_score < 0:
        reasons.append("台指期夜盤偏弱，現貨開盤容易承壓。")
    elif night_score > 0:
        reasons.append("台指期夜盤偏強，對現貨開盤有支撐。")
    else:
        reasons.append("台指期夜盤訊號中性或資料不足。")
    if panic_score > 0:
        reasons.append("盤中有急殺後拉回，代表低檔承接存在。")
    elif panic_score < 0:
        reasons.append("盤中急跌後未能有效拉回，短線風險偏高。")
    else:
        reasons.append("盤中沒有明顯急殺拉回型態。")
    return reasons


def market_scenario(row, truth: str, external_score: int, night_score: int, panic_score: int) -> dict:
    open_gap = row.get("open_gap_pct")
    close_ret = row.get("close_return_pct")
    low_pct = row.get("intraday_low_pct")
    recovery = row.get("close_recovery_ratio")
    high_pct = row.get("intraday_high_pct")
    stage = row.get("lifecycle_stage", "unknown")

    if truth == "external_shock_absorbed":
        return scenario("外部利空開低，盤中承接拉回", "開盤前有壓力，但現貨盤中有人承接。", "隔日若夜盤轉強、開盤不再破低，容易走反彈或震盪偏多。", "這種盤常不是立即大多頭，要確認恐慌是否解除。")
    if truth == "external_pressure_not_absorbed":
        return scenario("外部壓力未解除，現貨承接不足", "開盤前壓力存在，但盤中沒有明顯收回跌勢。", "觀察夜盤是否續弱、隔日是否跌破今日低點。", "不要只看收盤跌幅小，因為內部承接力可能還不夠。")
    if ge(open_gap, 0.001) and le(close_ret, -0.003) and le(recovery, 0.25):
        return scenario(
            "夜盤偏強但日盤收弱，現貨反證",
            "早盤有偏多預期，但日盤收在低位，代表現貨追價被賣壓或獲利了結壓回。",
            "隔日先看是否站回今日開盤價；若無法站回，夜盤牽引力下降，盤勢容易回到高檔換手或回測。",
            "此型態不能寫成單純健康修復，需提高對追價失敗與隔日續弱的監控。",
        )
    if ge(open_gap, 0.008) and le(close_ret, -0.003):
        return scenario("開高走低，誘多失敗", "早盤偏多但收盤轉弱，代表追價買盤被賣壓壓回。", "若隔日無法站回今日開盤價，容易轉為震盪或回測。", "高檔出現時要提防獲利了結。")
    if le(open_gap, -0.008) and ge(close_ret, 0.003):
        return scenario("開低走高，低接成功", "早盤恐慌開低，但買盤把指數拉回。", "隔日若能守住今日中位區，反彈延續機率較高。", "若隔日又跌破今日低點，代表只是短線反抽。")
    if le(low_pct, -0.02) and ge(recovery, 0.70):
        return scenario("盤中急殺洗盤，尾盤強拉", "盤中曾恐慌下殺，但收盤收在相對高位。", "觀察隔日是否不破低；若確認，短線反彈力量增加。", "若隔日再破低，今日拉回只是暫時止血。")
    if le(low_pct, -0.02) and not ge(recovery, 0.45):
        return scenario("盤中急殺無力拉回", "盤中賣壓很重，收盤沒有有效收復。", "隔日若續破低點，容易進入連續修正。", "不宜過早判斷已落底。")
    if ge(high_pct, 0.012) and le(close_ret, 0):
        return scenario("盤中急拉失敗，追價退潮", "盤中曾往上急拉，但收盤沒有守住。", "隔日若不能突破今日高點，容易回到震盪或轉弱。", "常見於反彈中繼或高檔震盪。")
    if stage in ["topping", "main_bull"] and le(close_ret, -0.01) and le(recovery, 0.35):
        return scenario("高檔轉弱，賣壓擴大", "高檔區弱收，代表籌碼開始鬆動。", "觀察是否跌破短期均線與夜盤是否續弱。", "高檔弱收比低檔弱收更需要防守。")
    if stage in ["bottoming", "early_bear", "main_bear"] and ge(close_ret, 0.01) and ge(recovery, 0.65):
        return scenario("低檔強彈，恐慌後換手", "弱勢或低檔區強收，代表空方回補或買盤進場。", "隔日若量價延續，可能形成反彈段。", "低檔強彈要等第二天確認。")
    if between(close_ret, -0.003, 0.003) and ge(recovery, 0.65):
        return scenario("收盤小變動，但低檔有承接", "表面漲跌不大，但盤中低點被拉回。", "後續看夜盤與隔日開盤是否延續承接力。", "不能只用收盤小漲小跌判斷沒有故事。")
    return scenario("一般震盪盤，等待確認", "目前沒有偵測到強烈單一劇本。", "觀察開盤位置、今日高低點是否被突破。", "沒有明確劇本時要降低信心。")


def scenario(name: str, meaning: str, watch: str, risk: str) -> dict:
    return {"name": name, "meaning": meaning, "watch": watch, "risk": risk}


def truth_text(label: str) -> str:
    return {
        "external_shock_absorbed": "外部衝擊被盤中買盤吸收",
        "external_pressure_not_absorbed": "外部壓力仍在，尚未完全解除",
        "domestic_buying_absorbed_panic": "台股自身買盤承接恐慌賣壓",
        "neutral": "沒有明顯單一主因",
    }.get(label, label)


def external_pressure_text(label: str) -> str:
    return {
        "heavy_external_pressure": "外部市場明顯偏空，容易造成開盤壓力",
        "external_pressure": "外部市場偏弱，對台股有壓力",
        "strong_external_tailwind": "外部市場明顯偏多，對台股有支撐",
        "external_tailwind": "外部市場偏強，對台股有支撐",
        "neutral": "外部市場中性",
        "unknown": "外部市場資料不足",
    }.get(label, label)


def night_futures_text(label: str) -> str:
    return {
        "night_futures_heavy_pressure": "台指期夜盤明顯偏空，現貨開盤前已有壓力",
        "night_futures_pressure": "台指期夜盤偏弱，現貨開盤容易承壓",
        "night_futures_strong_tailwind": "台指期夜盤明顯偏多，現貨開盤前氣氛偏強",
        "night_futures_tailwind": "台指期夜盤偏強，對現貨開盤有支撐",
        "neutral": "台指期夜盤中性",
        "unknown": "台指期夜盤資料不足",
    }.get(label, label)


def intraday_truth_text(label: str) -> str:
    return {
        "panic_washout_strong_recovery": "盤中急殺後強力拉回，恐慌賣壓被明顯承接",
        "panic_washout_recovery": "盤中急殺後拉回，承接力存在",
        "gap_down_failed_recovery": "跳空下跌後沒有有效拉回，短線風險偏高",
        "panic_no_recovery": "盤中急跌後未能拉回，賣壓仍重",
        "intraday_panic": "盤中曾出現恐慌下殺",
        "strong_close": "收盤位置偏強，代表低檔有買盤承接",
        "normal": "盤中型態普通，沒有明顯極端訊號",
    }.get(label, label)


def plain_summary(truth: str, external_score: int, night_score: int, panic_score: int) -> str:
    if truth == "external_shock_absorbed":
        return "開盤前或外部市場有壓力，但盤中買盤把恐慌賣壓接住。"
    if truth == "external_pressure_not_absorbed":
        return "外部或夜盤壓力尚未解除，短線仍需防守。"
    if truth == "domestic_buying_absorbed_panic":
        return "台股自身盤中承接明確，但仍要看隔日是否延續。"
    if external_score > 0 or night_score > 0:
        return "外部與夜盤對台股有支撐，若現貨不轉弱，後續較偏震盪偏多。"
    return "目前沒有單一強烈主因，應等待夜盤與現貨方向確認。"


def analyze_logic_consistency(payload: dict) -> dict:
    rows = []

    def add(item: str, status_text: str, explanation: str) -> None:
        rows.append({"item": item, "status": status_text, "explanation": explanation})

    bagua = payload.get("bagua_lifecycle", {})
    primary = bagua.get("primary", {})
    daily = bagua.get("daily", {})
    resonance = bagua.get("resonance", {})
    candle = payload.get("candlestick_pattern", {})
    washout = payload.get("washout_pattern", {})
    cycle = payload.get("tradeable_cycle", {})
    technical = payload.get("technical_phase", {})
    cycle_position = cycle.get("current_position", {})
    frequency = cycle.get("stats", {}).get("frequency_state", {})
    forecast = payload.get("forecast", {})
    cause = payload.get("cause_analysis", {})
    premarket = payload.get("premarket", {})
    memory = payload.get("memory_industry_risk", {})

    primary_code = primary.get("code")
    daily_code = daily.get("code")
    candle_type = candle.get("type")
    washout_type = washout.get("type")
    cycle_code = cycle_position.get("code")

    if primary_code in ["DUI", "QIAN", "KUN", "LI"] and daily_code == "KAN":
        add(
            "宏觀卦位 vs 日線卦位",
            "一致但需說明",
            "大週期仍在高位，日線已入坎，這是高位承載失敗或轉弱警戒，不是低檔築底訊號。",
        )
    elif primary_code and daily_code:
        add("宏觀卦位 vs 日線卦位", "一致", "月線主卦與日線卦位沒有明顯衝突。")
    else:
        add("宏觀卦位 vs 日線卦位", "資料不足", "八卦多週期資料不足。")

    if resonance.get("label") == "高位入坎" and candle_type in ["long_bear", "gap_up_failed"] and washout_type == "failed_washout":
        add(
            "八卦共振 vs K線洗盤",
            "一致",
            "高位入坎、長黑K、失敗洗盤三者同向，皆指向高檔轉弱或承接不足。",
        )
    elif candle_type in ["long_lower_shadow", "gap_down_reversal"] and washout_type in ["strong_washout", "normal_washout", "gap_down_recovery"]:
        add(
            "八卦共振 vs K線洗盤",
            "一致",
            "K線與洗盤同時偏向承接，若卦位在艮或震，代表止跌/起漲邏輯相互呼應。",
        )
    else:
        add(
            "八卦共振 vs K線洗盤",
            "需後續確認",
            "K線或洗盤沒有形成強烈同向確認，需用下一交易日確認價與失效價追蹤。",
        )

    if technical.get("enabled"):
        alignment = technical.get("alignment", {})
        add(
            "成熟技術分析 vs 卦位預測",
            alignment.get("status", "需後續確認"),
            alignment.get("summary", ""),
        )
        if technical.get("bias") == "bearish" and daily_code == "KAN":
            add(
                "技術波段 vs 日線卦",
                "一致",
                "均線、前低與K線結構偏空，日線入坎，不應再用震卦解釋急跌。",
            )
        else:
            add(
                "技術波段 vs 日線卦",
                "需後續確認",
                "技術波段與日線卦尚未完全同向，需下一根K線確認。",
            )

    if cycle_code == "post_rally_pullback" and primary_code in ["DUI", "QIAN", "KUN", "LI"]:
        add(
            "年度波段 vs 宏觀卦位",
            "一致",
            "宏觀仍在高位，但年度波段已回落找底，符合高位波段結束後等待新支撐的邏輯。",
        )
    elif cycle_code in ["early_rally", "main_rally"] and primary_code in ["GEN", "ZHEN", "XUN", "LI"]:
        add(
            "年度波段 vs 宏觀卦位",
            "一致",
            "波段轉強與宏觀修復/主升方向相互呼應。",
        )
    elif cycle_code:
        add("年度波段 vs 宏觀卦位", "需說明", "波段段位與宏觀卦位不是同一時間級別，需以月線主卦與日線確認分開解讀。")
    else:
        add("年度波段 vs 宏觀卦位", "資料不足", "年度波段資料不足。")

    short_forecasts = {int(item["horizon_days"]): item for item in forecast.get("forecasts", []) if int(item["horizon_days"]) in [1, 5]}
    long_forecasts = {int(item["horizon_days"]): item for item in forecast.get("forecasts", []) if int(item["horizon_days"]) in [20, 60]}
    short_down = any(item.get("predicted_direction") == "down" for item in short_forecasts.values())
    long_up = any(item.get("predicted_direction") == "up" for item in long_forecasts.values())
    if short_down and long_up:
        add(
            "短線預測 vs 中長線預測",
            "一致但需說明",
            "1日/5日偏弱、20日/60日偏強，代表短線入坎後可能先測底，再看中期修復，不是同一期間互相矛盾。",
        )
    elif short_down and cycle_code in ["post_rally_pullback", "pullback_watch"]:
        add("短線預測 vs 波段段位", "一致", "短線偏弱與回落/防守段位一致。")
    elif long_up and primary_code in ["GEN", "ZHEN", "XUN", "LI", "KUN", "DUI"]:
        add("中長線預測 vs 宏觀卦位", "一致", "中長期偏修復或偏多與非深跌危機卦位相容。")
    else:
        add("預測期間結構", "需後續確認", "各期間預測沒有明顯矛盾，但信心仍需由到期驗證累積。")

    if premarket.get("is_premarket") and premarket.get("total_score", 0) > 0 and cause.get("panic_reversal_score", 0) < 0:
        add(
            "盤前訊號 vs 日盤結果",
            "一致但需說明",
            "盤前偏多只代表開盤前條件，日盤長黑與承接不足代表現貨未吸收賣壓，兩者時間點不同。",
        )
    elif premarket.get("is_premarket"):
        add("盤前訊號 vs 日盤結果", "一致", "盤前資料僅作開盤壓力參考，日盤仍以現貨K線與洗盤結果確認。")
    else:
        add("盤前訊號 vs 日盤結果", "不適用", "目前不是盤前模式。")

    if memory.get("level") in ["watch", "high", "critical"] and forecast.get("risk_regime") == "risk_off":
        add("產業風險 vs 風險狀態", "一致", "記憶體產業風險與 risk_off 同向，支持防守解讀。")
    elif memory.get("level") in ["watch", "high", "critical"]:
        add("產業風險 vs 風險狀態", "需說明", "產業風險升高但總體風險狀態未完全轉空，應作為降信心與警戒因子。")
    else:
        add("產業風險 vs 風險狀態", "一致", "產業風險未形成主要矛盾。")

    if frequency.get("level") in ["high", "extreme_high"] and cycle_code in ["post_rally_pullback", "pullback_watch", "late_rally"]:
        add(
            "高頻波段 vs 操作節奏",
            "一致",
            "今年波段頻率偏高，若又處於回落或末升，應縮短持有假設並提高防守權重。",
        )
    elif frequency.get("level") in ["high", "extreme_high"]:
        add("高頻波段 vs 操作節奏", "需說明", "波段頻率偏高，代表機會多但持續性較短，需搭配段位決定進退。")
    else:
        add("高頻波段 vs 操作節奏", "一致", "波段頻率未顯著偏離歷史常態。")

    contradiction_count = sum(1 for row in rows if row["status"] == "矛盾")
    explain_count = sum(1 for row in rows if "需" in row["status"])
    if contradiction_count:
        summary = f"發現 {contradiction_count} 項硬矛盾，需修正公式或文字解讀。"
        level = "conflict"
    elif explain_count:
        summary = f"未發現硬矛盾，但有 {explain_count} 項屬於不同時間級別或期間差異，已補充說明。"
        level = "needs_explanation"
    else:
        summary = "各層判斷前後一致，沒有明顯矛盾。"
        level = "consistent"
    return {
        "level": level,
        "summary": summary,
        "contradiction_count": contradiction_count,
        "needs_explanation_count": explain_count,
        "rows": rows,
    }


def load_model_validation_status() -> dict:
    fallback = {"passed": False, "required_accuracy": .9, "required_cases": 100, "required_coverage": .1, "required_wilson_lower": .8}
    path = ROOT / "config" / "model_validation_status.json"
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def build_production_policy(validation: dict) -> dict:
    validated = validation.get("validated_submodels", {})
    return {
        "research_completed": bool(validation.get("research_completed", False)),
        "main_multi_day_direction": {
            "enabled": bool(validation.get("production_multi_day_direction_enabled", False)),
            "status": validation.get("research_outcome", "not_validated"),
            "formal_signal": None,
            "reason": validation.get("message", ""),
        },
        "validated_submodels": {
            name: {
                "scope": item.get("scope"),
                "accuracy": item.get("accuracy"),
                "cases": item.get("cases"),
                "coverage": item.get("coverage"),
                "warning": item.get("warning"),
            }
            for name, item in validated.items()
        },
        "scope_rule": (
            "Validated opening-gap models may not alter cash-close or multi-day direction. "
            "Exploratory lifecycle probabilities are descriptive historical frequencies only."
        ),
        "closeout_report": validation.get("closeout_report"),
    }


def load_model_reliability_audit() -> dict:
    path = ROOT / "reports" / "model_reliability_trend_audit.json"
    if not path.exists():
        return {"available": False, "summary": "尚未產生模型可靠度稽核。"}
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"available": False, "summary": "模型可靠度稽核讀取失敗。"}
    audit["available"] = True
    return audit


def build_direction_reliability_policy(audit: dict, validation: dict) -> dict:
    recent = audit.get("recent_windows", {}) if audit.get("available") else {}
    recent_10 = safe_float((recent.get("10") or {}).get("hit_rate"))
    recent_30 = safe_float((recent.get("30") or {}).get("hit_rate"))
    overall = safe_float((audit.get("overall") or {}).get("hit_rate"))
    one_day_late = safe_float(
        ((audit.get("chronological_thirds_1d") or {}).get("late") or {}).get("hit_rate")
    )
    production_enabled = bool(validation.get("production_multi_day_direction_enabled", False))

    if not audit.get("available"):
        level = "unknown"
        headline = "模型可靠度：尚未稽核"
        summary = "尚無可靠度趨勢檔，方向預測只能維持研究性觀察。"
        direction_weight = "low"
    elif production_enabled and recent_30 is not None and recent_30 >= 0.60:
        level = "watch"
        headline = "模型可靠度：方向可觀察"
        summary = "近期方向命中率達觀察門檻，但仍需每日收盤驗證。"
        direction_weight = "medium"
    elif recent_10 is not None and recent_10 < 0.30:
        level = "severe_degrade"
        headline = "模型可靠度：短線方向強制降權"
        summary = "最近10筆方向命中率低於30%，短線方向不得主導結論，風控與確認線升為主軸。"
        direction_weight = "very_low"
    elif recent_30 is not None and recent_30 < 0.55:
        level = "degrade"
        headline = "模型可靠度：方向預測降權"
        summary = "最近30筆方向命中率未達55%，短線方向只能當假設，不能當正式信號。"
        direction_weight = "low"
    else:
        level = "reduced"
        headline = "模型可靠度：研究性降權"
        summary = "多日方向模型未通過正式生產閘門，方向判讀保持降權。"
        direction_weight = "low"

    return {
        "enabled": True,
        "level": level,
        "headline": headline,
        "summary": summary,
        "direction_weight": direction_weight,
        "risk_weight": "high",
        "validated_submodel_weight": "high",
        "overall_hit_rate": overall,
        "recent_10_hit_rate": recent_10,
        "recent_30_hit_rate": recent_30,
        "one_day_late_hit_rate": one_day_late,
        "production_direction_enabled": production_enabled,
        "rules": [
            "未通過生產閘門前，多日方向預測只作研究假設。",
            "夜盤已驗證用途限於日盤開盤缺口與幅度風險，不可直接改寫收盤方向。",
            "近期命中率低於55%時，主報告必須改以風控、健康指數、確認線與失效線為核心。",
            "每日收盤後重新核對前一日預測，錯誤先分類，再決定降權、補資料或候選規則前瞻驗證。",
        ],
        "report": "reports/model_reliability_trend_audit.md",
        "guardrail": "可靠度政策只管理模型發言權，不產生投資命令。",
    }


def render_error_review(review: dict) -> str:
    lines = [
        "# 預測誤差閉迴路檢討",
        "",
        f"- 框架: {review.get('framework', 'NA')}",
        f"- 來源: {review.get('source', 'NA')}",
        f"- 到期核對筆數: {review.get('total_checked_recent', 0)}",
        f"- 失準筆數: {review.get('miss_count', 0)}",
        f"- 近期核對率: {pct(review.get('hit_rate'))}",
        f"- 摘要: {review.get('summary', '')}",
        "",
        "## 治理規則",
        "",
    ]
    governance = review.get("governance", {})
    lines.extend(
        [
            f"- 歷史紀錄不可改寫: {'是' if governance.get('historical_records_immutable') else '否'}",
            f"- 自動調參: {'是' if governance.get('automatic_parameter_tuning') else '否'}",
            f"- 自動改門檻: {'是' if governance.get('automatic_threshold_tuning') else '否'}",
            f"- 候選規則需獨立驗證: {'是' if governance.get('independent_validation_required') else '否'}",
        ]
    )
    lines.extend(render_next_day_validation_lines(review.get("next_day_validation", {})))
    lines.extend(
        [
            "",
            "## 失準分類",
            "",
            "| 類別 | 筆數 | 天期分布 |",
            "| --- | ---: | --- |",
        ]
    )
    for group in review.get("groups", []):
        horizons = "、".join(f"{key}日:{value}" for key, value in sorted(group.get("horizons", {}).items()))
        lines.append(f"| {group.get('label')} | {group.get('count')} | {horizons} |")
    if not review.get("groups"):
        lines.append("| 無 | 0 | NA |")

    lines.extend(
        [
            "",
            "## 最近失準病例",
            "",
            "| 預測日 | 基準日 | 到期日 | 天期 | 預測 | 實際 | 報酬 | 主因 | 候選修正 |",
            "| --- | --- | --- | ---: | --- | --- | ---: | --- | --- |",
        ]
    )
    for case in review.get("cases", [])[-12:]:
        reason = (case.get("reasons") or [{}])[0]
        lines.append(
            f"| {case.get('forecast_date')} | {case.get('signal_date')} | {case.get('target_date')} | "
            f"{case.get('horizon_days')} | {direction_text(case.get('predicted_direction'))} | "
            f"{direction_text(case.get('actual_direction'))} | {pct(case.get('actual_return'))} | "
            f"{case.get('primary_error_label')} | {reason.get('candidate_adjustment', '')} |"
        )
    return "\n".join(lines) + "\n"


def render_next_day_validation_lines(validation: dict, heading: str = "## 次日強制驗證") -> list[str]:
    if not validation:
        return []
    lines = [
        "",
        heading,
        "",
        f"- 頭條: {validation.get('headline', '尚無到期預測可驗證')}",
        f"- 摘要: {validation.get('summary', '')}",
        f"- 治理: {validation.get('governance', '不自動調參；只做核對。')}",
    ]
    tasks = validation.get("tasks", [])
    if tasks:
        lines.extend(["", "| 今日必查項目 | 驗證規則 |", "| --- | --- |"])
        for task in tasks:
            lines.append(f"| {task.get('name')} | {task.get('rule')} |")
    return lines


def render_peak_to_valley_backtest(backtest: dict) -> str:
    lines = [
        "# 峰轉谷早期預警歷史回測",
        "",
        f"- 框架: {backtest.get('framework', 'NA')}",
        f"- 基準日: {backtest.get('signal_date', 'NA')}",
        f"- 方法: {backtest.get('method', '')}",
        f"- 候選規則: {backtest.get('rules', {}).get('candidate', '')}",
        f"- 確認規則: {backtest.get('rules', {}).get('confirmation', '')}",
        f"- 事件數: {backtest.get('event_count', 0)}",
        f"- 摘要: {backtest.get('summary', '')}",
        "",
        "| 期間 | 樣本 | 平均最大低點回撤 | 平均最大收盤回撤 | 跌逾3%率 | 跌逾5%率 | 跌破候選低點率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in backtest.get("stats", {}).get("by_horizon", []):
        lines.append(
            f"| {row.get('horizon')} | {row.get('sample_count')} | {pct(row.get('avg_max_low_drawdown'))} | "
            f"{pct(row.get('avg_max_close_drawdown'))} | {pct(row.get('down_3pct_rate'))} | "
            f"{pct(row.get('down_5pct_rate'))} | {pct(row.get('break_event_low_rate'))} |"
        )
    lines.extend(
        [
            "",
            "## 最近事件",
            "",
            "| 日期 | 收盤 | 高點 | 低點 | 收盤位置 | 震幅 | 5日最大低點回撤 | 10日最大低點回撤 | 20日最大低點回撤 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for event in backtest.get("recent_events", []):
        fwd = event.get("forward", {})
        lines.append(
            f"| {event.get('date')} | {num(event.get('close'))} | {num(event.get('high'))} | {num(event.get('low'))} | "
            f"{pct(event.get('close_position'))} | {pct(event.get('day_range_pct'))} | "
            f"{pct(fwd.get('5d', {}).get('max_low_drawdown'))} | "
            f"{pct(fwd.get('10d', {}).get('max_low_drawdown'))} | "
            f"{pct(fwd.get('20d', {}).get('max_low_drawdown'))} |"
        )
    return "\n".join(lines) + "\n"


def build_breath_monitor_model(payload: dict) -> dict:
    crash = payload.get("crash_monitor", {})
    mode = payload.get("market_mode_switch", {})
    premarket = payload.get("premarket", {})
    date_audit = payload.get("market_date_audit", {})
    drawdown = crash.get("drawdown_guardrail", {})
    levels = crash.get("levels", {})
    technical = payload.get("technical_phase", {})
    tech_levels = technical.get("levels", {})
    memory = payload.get("memory_industry_risk", {})
    capital = payload.get("capital_flow", {})
    heart = payload.get("market_heart_rhythm", {})
    risk_value = safe_float(crash.get("risk_value")) or 0
    mode_score = safe_int(mode.get("risk_score"), 0)
    night_move = abs(safe_float(premarket.get("tx_night_spread_per")) or 0)
    sox_move = abs(safe_float(premarket.get("sox_return_1d")) or 0)
    vix_move = abs(safe_float(premarket.get("vix_return_1d")) or 0)
    external_pressure = min(24, round((night_move * 500) + (sox_move * 260) + (vix_move * 90), 1))
    breath_pressure = round(min(100, risk_value + mode_score * 5 + external_pressure), 1)
    if breath_pressure >= 75:
        breath_state = "異常呼吸／危機防守"
        breath_code = "critical"
    elif breath_pressure >= 50:
        breath_state = "急促呼吸／提前警戒"
        breath_code = "warning"
    elif breath_pressure >= 25:
        breath_state = "偏快呼吸／高敏感"
        breath_code = "watch"
    else:
        breath_state = "正常呼吸／規律脈動"
        breath_code = "normal"

    def event(kind: str, title: str, detail: str, severity: str = "normal", value: str = "") -> dict:
        return {
            "kind": kind,
            "title": title,
            "detail": detail,
            "severity": severity,
            "value": value,
        }

    events = [
        event(
            "總控",
            mode.get("headline", "資料不足"),
            mode.get("action", ""),
            "warning" if mode.get("crisis_early") else ("watch" if mode.get("mode") == "high_level_rotation" else "normal"),
            f"分數 {mode.get('risk_score', 'NA')}",
        ),
        event(
            "呼吸",
            breath_state,
            "正常波動可視為生命脈動；分數升高代表節奏變急、變亂，需提高監控頻率。",
            breath_code,
            f"{num(breath_pressure)}/100",
        ),
        event(
            "心律",
            heart.get("label", "心律資料不足"),
            heart.get("summary", "每日收盤正負與振幅資料不足，暫不判斷心跳頻率。"),
            "normal" if safe_int(heart.get("score"), 0) >= 72 else ("watch" if safe_int(heart.get("score"), 0) >= 55 else "warning"),
            f"{heart.get('score', 'NA')}/100",
        ),
        event(
            "日盤",
            "現貨收盤與低點",
            f"收盤 {num(crash.get('effective_close'))}，低點 {num(crash.get('effective_low'))}；收盤/低點用日期稽核後資料。",
            "normal" if breath_pressure < 50 else "watch",
            f"{num(crash.get('effective_close'))}",
        ),
        event(
            "夜盤",
            "夜盤牽動觀察",
            f"台指期夜盤 {pct(premarket.get('tx_night_spread_per'))}，收 {num(premarket.get('tx_night_close'))}，低 {num(premarket.get('tx_night_low'))}。",
            "warning" if abs(safe_float(premarket.get("tx_night_spread_per")) or 0) >= 0.015 else "watch",
            pct(premarket.get("tx_night_spread_per")),
        ),
        event(
            "回撤",
            drawdown.get("stage", "資料不足"),
            drawdown.get("summary", "資料不足"),
            "warning" if drawdown.get("stage_code") in {"major_correction_policy_watch", "crash_precursor_deep_kan"} else "watch",
            pct(drawdown.get("drawdown")),
        ),
        event(
            "風控線",
            "正常艮、重大修正、崩盤線",
            f"高檔正常艮 {num(levels.get('normal_gen_low'))}～{num(levels.get('normal_gen_high'))}；重大修正 {num(levels.get('major_correction_line'))}；崩盤前奏 {num(levels.get('crash_warning'))}。",
            "watch",
        ),
        event(
            "產業",
            f"記憶體風險 {memory.get('level', 'NA')}",
            memory.get("summary", memory.get("plain_summary", "資料不足")),
            "warning" if memory.get("level") in {"high", "critical"} else ("watch" if memory.get("level") == "watch" else "normal"),
            f"{memory.get('score', 'NA')}",
        ),
        event(
            "籌碼",
            capital.get("data_status", "資料不足"),
            capital.get("plain_summary", "法人籌碼、融資融券、期貨與選擇權若缺漏，總控模式需保守。"),
            "watch" if not capital.get("has_factor_values") else "normal",
        ),
        event(
            "稽核",
            date_audit.get("label", "日期稽核"),
            date_audit.get("status", "資料不足"),
            "normal" if date_audit.get("mode") == "official_daily" else "watch",
            date_audit.get("official_signal_date", ""),
        ),
    ]
    return {
        "date": payload.get("input", {}).get("date"),
        "signal_date": payload.get("index_check", {}).get("signal_date"),
        "breath_pressure": breath_pressure,
        "breath_state": breath_state,
        "breath_code": breath_code,
        "market_mode": mode.get("headline", "資料不足"),
        "heart_rhythm": heart,
        "mode_score": mode_score,
        "health_score": crash.get("health_score"),
        "risk_value": crash.get("risk_value"),
        "close": crash.get("effective_close"),
        "low": crash.get("effective_low"),
        "ma5": tech_levels.get("ma5"),
        "ma20": tech_levels.get("ma20"),
        "events": events,
        "rules": [
            "總控模式優先於健康分數。",
            "危機初期組合成立時，先提示風險，不寫成太平盛世。",
            "正常異動是規律呼吸；異常異動要看節奏是否急促、失序、連續惡化。",
            "艮是跌勢停止，不是深跌本身。",
        ],
    }


def render_breath_monitor(payload: dict) -> str:
    model = build_breath_monitor_model(payload)
    events = model["events"]

    def cls(severity: str) -> str:
        return {
            "critical": "sev-critical",
            "warning": "sev-warning",
            "watch": "sev-watch",
            "normal": "sev-normal",
        }.get(severity, "sev-normal")

    event_rows = "\n".join(
        f"""
        <article class="breath-event {cls(item.get('severity'))}" data-severity="{escape(item.get('severity', 'normal'))}">
          <div class="event-kind">{escape(item.get('kind', ''))}</div>
          <div class="event-main">
            <div class="event-title">{escape(item.get('title', ''))}</div>
            <div class="event-detail">{escape(item.get('detail', ''))}</div>
          </div>
          <div class="event-value">{escape(str(item.get('value') or ''))}</div>
        </article>
        """
        for item in events
    )
    rules = "\n".join(f"<li>{escape(rule)}</li>" for rule in model["rules"])
    pressure = safe_float(model.get("breath_pressure")) or 0
    html = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>台股呼吸脈動滾動監控</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: light-dark(#f7f8fb, #0d1117);
      --fg: light-dark(#172033, #e6edf3);
      --muted: light-dark(#5e6a7d, #9aa7b8);
      --line: light-dark(#d8deea, #263244);
      --panel: light-dark(#ffffff, #151b24);
      --normal: #2f8f5b;
      --watch: #9b7b18;
      --warning: #bd5b17;
      --critical: #b4232a;
      --accent: light-dark(#2b5fb8, #7aa2ff);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft JhengHei", "Noto Sans TC", system-ui, sans-serif;
      background: var(--bg);
      color: var(--fg);
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 18px;
    }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: color-mix(in srgb, var(--bg) 92%, transparent);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--line);
      padding: 14px 0;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      font-weight: 600;
      letter-spacing: 0;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 86px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .metric strong {{
      font-size: 22px;
      font-weight: 600;
    }}
    .breath-bar {{
      height: 16px;
      border: 1px solid var(--line);
      border-radius: 999px;
      overflow: hidden;
      background: color-mix(in srgb, var(--line) 35%, transparent);
      margin: 16px 0 10px;
    }}
    .breath-fill {{
      width: {pressure:.1f}%;
      height: 100%;
      background: linear-gradient(90deg, var(--normal), var(--watch), var(--warning), var(--critical));
    }}
    .pulse {{
      width: 100%;
      height: 86px;
      margin: 8px 0 14px;
    }}
    .pulse path {{
      fill: none;
      stroke: var(--accent);
      stroke-width: 3;
    }}
    .monitor-layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 300px;
      gap: 16px;
      align-items: start;
      margin-top: 16px;
    }}
    .stream {{
      max-height: 580px;
      overflow-y: auto;
      padding-right: 8px;
    }}
    .breath-event {{
      display: grid;
      grid-template-columns: 82px minmax(0, 1fr) 92px;
      gap: 12px;
      align-items: center;
      background: var(--panel);
      border: 1px solid var(--line);
      border-left-width: 6px;
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 10px;
    }}
    .sev-normal {{ border-left-color: var(--normal); }}
    .sev-watch {{ border-left-color: var(--watch); }}
    .sev-warning {{ border-left-color: var(--warning); }}
    .sev-critical {{ border-left-color: var(--critical); }}
    .event-kind, .event-value {{
      color: var(--muted);
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }}
    .event-value {{ text-align: right; }}
    .event-title {{
      font-weight: 600;
      margin-bottom: 4px;
    }}
    .event-detail {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }}
    .side {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .side h2 {{
      font-size: 17px;
      margin: 0 0 10px;
    }}
    .side ul {{
      margin: 0;
      padding-left: 20px;
      color: var(--muted);
      line-height: 1.65;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    button {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--fg);
      border-radius: 6px;
      padding: 8px 10px;
      cursor: pointer;
    }}
    button[aria-pressed="true"] {{
      border-color: var(--accent);
      color: var(--accent);
    }}
    @media (max-width: 760px) {{
      main {{ padding: 12px; }}
      .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .monitor-layout {{ grid-template-columns: 1fr; }}
      .breath-event {{ grid-template-columns: 64px minmax(0, 1fr); }}
      .event-value {{ grid-column: 2; text-align: left; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="topbar">
      <h1>台股呼吸脈動滾動監控</h1>
      <div>資料日 {escape(str(model.get('signal_date')))}｜報告日 {escape(str(model.get('date')))}｜{escape(str(model.get('breath_state')))}</div>
      <div class="breath-bar" aria-label="呼吸壓力值 {num(model.get('breath_pressure'))} 分"><div class="breath-fill"></div></div>
      <svg class="pulse" viewBox="0 0 720 86" role="img" aria-label="市場呼吸波形">
        <path d="M0,46 C40,46 44,46 58,46 C70,46 76,18 88,18 C100,18 106,68 120,68 C132,68 138,46 152,46 C210,46 226,46 250,46 C264,46 270,28 282,28 C296,28 304,58 318,58 C332,58 340,46 356,46 C430,46 450,46 472,46 C486,46 494,12 508,12 C524,12 532,74 548,74 C564,74 572,46 590,46 C648,46 676,46 720,46" />
      </svg>
      <div class="summary">
        <div class="metric"><span>呼吸壓力</span><strong>{num(model.get('breath_pressure'))}/100</strong></div>
        <div class="metric"><span>市場模式</span><strong>{escape(str(model.get('market_mode')))}</strong></div>
        <div class="metric"><span>健康 / 風險</span><strong>{num(model.get('health_score'))} / {num(model.get('risk_value'))}</strong></div>
        <div class="metric"><span>收盤 / 低點</span><strong>{num(model.get('close'))} / {num(model.get('low'))}</strong></div>
      </div>
      <div class="controls">
        <button type="button" data-filter="all" aria-pressed="true">全部</button>
        <button type="button" data-filter="abnormal">只看異常</button>
      </div>
    </section>
    <section class="monitor-layout">
      <div class="stream" id="eventStream" aria-label="滾動式監控事件">
        {event_rows}
      </div>
      <aside class="side">
        <h2>介面判讀規則</h2>
        <ul>{rules}</ul>
      </aside>
    </section>
  </main>
  <script>
    const buttons = document.querySelectorAll('[data-filter]');
    const events = document.querySelectorAll('.breath-event');
    buttons.forEach((button) => {{
      button.addEventListener('click', () => {{
        buttons.forEach((item) => item.setAttribute('aria-pressed', 'false'));
        button.setAttribute('aria-pressed', 'true');
        const filter = button.dataset.filter;
        events.forEach((item) => {{
          const abnormal = item.dataset.severity !== 'normal';
          item.style.display = filter === 'all' || abnormal ? '' : 'none';
        }});
      }});
    }});
  </script>
</body>
</html>
"""
    return html


def _top_text(items, limit: int = 3, default: str = "NA") -> str:
    values = [str(item) for item in (items or []) if str(item).strip()]
    return "；".join(values[:limit]) if values else default


def _compact_vitals(stethoscope: dict, limit: int = 5) -> str:
    vitals = []
    for item in stethoscope.get("diagnostics", [])[:limit]:
        name = item.get("name")
        label = item.get("label")
        if name or label:
            vitals.append(f"{name or 'NA'}/{label or 'NA'}")
    return "；".join(vitals) if vitals else "NA"


def _signed_points(value) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "NA"
    sign = "+" if numeric > 0 else ""
    return f"{sign}{numeric:,.2f}"


def _market_dashboard_lines(payload: dict, night_trend: dict, intraday: dict) -> list[str]:
    check = payload["index_check"]
    premarket = payload.get("premarket", {})
    latest_tracking = payload.get("night_cash_tracking", {}).get("latest", {})
    cash_close = safe_float(check.get("data_close"))
    prior_close = safe_float(latest_tracking.get("prior_close"))
    cash_change = cash_close - prior_close if cash_close is not None and prior_close is not None else None
    cash_return = safe_float(latest_tracking.get("cash_close_return"))
    cash_high = safe_float(latest_tracking.get("high"))
    cash_low = safe_float(latest_tracking.get("low"))
    cash_volume = safe_float(latest_tracking.get("volume"))
    cash_volume_text = f"{cash_volume / 100000000:,.0f}億股" if cash_volume is not None else "NA"

    night_open = safe_float(night_trend.get("open") or premarket.get("tx_night_open"))
    night_high = safe_float(night_trend.get("high") or premarket.get("tx_night_high"))
    night_low = safe_float(night_trend.get("low") or premarket.get("tx_night_low"))
    night_close = safe_float(night_trend.get("close") or premarket.get("tx_night_close"))
    night_volume = safe_float(premarket.get("night_path", {}).get("volume"))
    night_volume_text = f"{night_volume:,.0f}口" if night_volume is not None else "NA"

    basis = night_close - cash_close if night_close is not None and cash_close is not None else None
    basis_pct = basis / cash_close if basis is not None and cash_close else None

    return [
        "# 市場儀表板",
        f"- 台指期夜盤: {pct(night_trend.get('official_spread_per'))}，收 {num(night_close)}，低 {num(night_low)}，高 {num(night_high)}，開 {num(night_open)}，量 {night_volume_text}",
        f"- 加權指數: 收 {num(cash_close)}，漲跌 {_signed_points(cash_change)}（{pct(cash_return)}），低 {num(cash_low)}，高 {num(cash_high)}，成交股數 {cash_volume_text}",
        f"- 期現差: {num(basis)}（{pct(basis_pct)}）；夜盤收盤相對加權收盤",
        f"- 關鍵線: 防守 {format_levels(intraday.get('defense_levels', [])) or 'NA'}；轉強 {format_levels(intraday.get('reclaim_levels', [])) or 'NA'}",
        "",
    ]


def _night_cash_relation_lines(payload: dict, night_trend: dict) -> list[str]:
    premarket = payload.get("premarket", {})
    basis_audit = premarket.get("night_basis_audit", {})
    latest_tracking = payload.get("night_cash_tracking", {}).get("latest", {})

    official = safe_float(night_trend.get("official_spread_per") or basis_audit.get("official_spread_per"))
    open_close = safe_float(night_trend.get("open_close_return") or basis_audit.get("open_close_return"))
    previous_night = safe_float(
        night_trend.get("previous_night_close_return") or basis_audit.get("vs_previous_night_close_return")
    )
    official_vs_path = official - open_close if official is not None and open_close is not None else None
    previous_vs_official = previous_night - official if previous_night is not None and official is not None else None

    relation_lines = [
        f"- 夜盤基準差: 官方 {pct(official)}；開收 {pct(open_close)}；前夜 {pct(previous_night)}；官方-開收 {pct(official_vs_path)}；前夜-官方 {pct(previous_vs_official)}",
        f"- 基準說明: {basis_audit.get('summary', '三基準分開看：官方漲跌看報價基準，開收看夜盤路徑，前夜比較看夜盤連續性。')}",
    ]

    if latest_tracking:
        relation_lines.append(
            "- 最近夜日傳導: "
            f"{latest_tracking.get('signal_date', 'NA')}；夜盤 {pct(latest_tracking.get('tx_night_spread_per'))}；"
            f"日盤開盤 {pct(latest_tracking.get('gap_return'))}；盤中 {pct(latest_tracking.get('intraday_return'))}；"
            f"收盤 {pct(latest_tracking.get('cash_close_return'))}；狀態 {latest_tracking.get('sync_status', 'NA')}；"
            f"{latest_tracking.get('reason_summary', '')}"
        )
        relation_lines.append(
            "- 日盤影響判讀: "
            f"夜盤到開盤 {'同步' if latest_tracking.get('night_gap_aligned') else '不同步'}；"
            f"夜盤到收盤 {'同步' if latest_tracking.get('night_close_aligned') else '不同步'}；"
            f"開盤後 {'延續' if latest_tracking.get('gap_intraday_continuation') else '反向修正'}；"
            f"{latest_tracking.get('short_squeeze_summary', '')}"
        )
    else:
        relation_lines.append("- 最近夜日傳導: 無已完成對照資料，僅能等待日盤驗證。")

    return relation_lines


def render_compact_brief_forecast(payload: dict) -> str:
    check = payload["index_check"]
    date_audit = payload.get("market_date_audit", {})
    freshness = payload["data_freshness"]
    intraday = payload.get("intraday_tactical_monitor", {})
    satellite = payload.get("weather_satellite_forecast", {})
    zero_one_tilt = satellite.get("zero_one_tilt", {})
    night_trend = satellite.get("night_trend", {})
    arbitration = payload.get("master_arbitration", {})
    practical = payload.get("practical_cause_arbitration", {})
    stethoscope = payload.get("market_stethoscope", {})
    protection = payload.get("market_protection_layers", {})
    interface = payload.get("crisis_opportunity_interface", {})
    cross_market = payload.get("cross_market_entanglement", {})
    fundamental = payload.get("fundamental_constitution", {})
    reliability = payload.get("direction_reliability_policy", {})
    bagua = payload["bagua_lifecycle"]
    technical = payload["technical_phase"]
    technical_equation = technical.get("equation_answer", {})
    crash = payload["crash_monitor"]
    market_health = payload.get("market_health", {})
    health_value = market_health.get("health_value", {})
    close_cause = payload.get("close_cause_attribution", {})
    capital = payload.get("capital_flow", {})
    psychological_warfare = payload.get("psychological_warfare_pattern", {})
    day_night_variance = payload.get("day_night_variance_pattern", {})
    equation_reasoning = payload.get("equation_reasoning_audit", {})
    next_day_validation = payload.get("error_review", {}).get("next_day_validation", {})
    summary = payload["integrated_summary"]
    policy = payload["production_policy"]

    primary = bagua.get("primary", {})
    monthly = bagua.get("monthly", {})
    weekly = bagua.get("weekly", {})
    daily = bagua.get("daily", {})
    health_score = crash.get("health_score", market_health_score(crash))
    risk_value = crash.get("risk_value")
    health_label = crash.get("health_label") or market_health_label(health_score, crash.get("alert_code"))
    confirmation = crash.get("confirmation", {})
    false_crash = crash.get("false_crash_filter", {})
    state_sop = crash.get("state_sop", {})
    root_cause_items = satellite.get("root_cause_decomposition", {}).get("items", [])
    primary_root_cause = root_cause_items[0] if root_cause_items else {}
    state_trade = bagua.get("roles", {}).get("state_gua", {}).get("trade_annotation") or bagua_trade_annotation(primary.get("code"))

    defense_levels = format_levels(intraday.get("defense_levels", [])) or "NA"
    reclaim_levels = format_levels(intraday.get("reclaim_levels", [])) or "NA"
    capital_text = (
        capital.get("summary")
        or capital.get("headline")
        or ("資料已接入" if capital.get("has_factor_values") else "資料缺口，僅作保守觀察")
    )

    lines = [
        f"# 台股大盤重點報告（{payload['input']['date']}）",
        "",
        f"# 0/1走向重點：{zero_one_tilt.get('label', '資料不足')}",
        f"**{zero_one_tilt.get('summary', summary.get('bias_text', '資料不足'))}**",
        f"- 偏向分數: {zero_one_tilt.get('score', 'NA')}；分支 {zero_one_tilt.get('branch', 'NA')}",
        f"- 有利證據: {_top_text(zero_one_tilt.get('evidence'), 3)}",
        f"- 反證條件: {_top_text(zero_one_tilt.get('counter_conditions'), 3)}",
        f"- 關鍵線: 防守 {defense_levels}；轉強 {reclaim_levels}",
        "",
        *_market_dashboard_lines(payload, night_trend, intraday),
        "# 核心判斷",
        f"- 下一步: {satellite.get('headline', '資料不足')}｜{satellite.get('next_step', '')}",
        f"- 總仲裁: {arbitration.get('headline', '資料不足')}｜{arbitration.get('risk_posture', 'NA')}｜主控 {arbitration.get('dominant_layer', 'NA')}",
        f"- 防呆保護: {protection.get('label', '資料不足')}｜分數 {protection.get('score', 'NA')}｜失效層 {_top_text(protection.get('failed_layers'), 3, '尚未見失效')}",
        f"- 公式/推理對照: {equation_reasoning.get('label', '資料不足')}｜信任度 {equation_reasoning.get('trust', 'NA')}｜公式 {equation_reasoning.get('formula_answer', 'NA')}｜推理 {equation_reasoning.get('reasoning_answer', 'NA')}｜結論 {equation_reasoning.get('conclusion', '')}",
        f"- 危機/轉機: {interface.get('label', '資料不足')}｜{interface.get('interface_state', 'NA')}｜峰谷 {_top_text(interface.get('peak_valley_signals'), 2)}",
        f"- 健康/風險: 健康 {num(health_score)} / 100（{health_label}）；風險 {num(risk_value)} / 100；健康價值 {health_value.get('label', '資料不足')} / {health_value.get('value', 'NA')}",
        f"- 崩盤核對: {confirmation.get('stage', '資料不足')}｜假崩盤 {false_crash.get('label', '資料不足')}｜SOP {state_sop.get('state_gua', 'NA')}/{state_sop.get('sop', '資料不足')}",
        "",
        "# 證據分層",
        f"- 夜盤: {night_trend.get('label', '資料不足')}｜官方 {pct(night_trend.get('official_spread_per'))}；開收 {pct(night_trend.get('open_close_return'))}；前夜 {pct(night_trend.get('previous_night_close_return'))}；收 {num(night_trend.get('close'))}；低 {num(night_trend.get('low'))}；{night_trend.get('path_shape', '')}",
        *_night_cash_relation_lines(payload, night_trend),
        f"- 日盤戰術: {intraday.get('label', '資料不足')}｜{intraday.get('summary', '')}｜動作 {intraday.get('action', 'NA')}",
        f"- 外部/跨盤: {cross_market.get('label', '資料不足')}｜{cross_market.get('summary', '')}｜{_top_text(cross_market.get('evidence'), 3)}",
        f"- 基本面命格: {fundamental.get('label', '資料不足')}｜{fundamental.get('score', 'NA')}/100｜支撐 {_top_text(fundamental.get('drivers'), 2)}｜壓力 {_top_text(fundamental.get('pressures'), 2)}",
        f"- 技術/卦位答案: {technical_equation.get('answer', '資料不足')}｜分數 {technical_equation.get('score', 'NA')}｜最可信公式 {technical_equation.get('best_formula', 'NA')} / {technical_equation.get('best_formula_answer', 'NA')}｜算式: 主卦 {primary.get('gua')} / {primary.get('label')}；月週日 {monthly.get('gua')} / {weekly.get('gua')} / {daily.get('gua')}；技術段位 {technical.get('label', '資料不足')}",
        f"- 公式推理驗證: 同向 {_top_text(equation_reasoning.get('agreements'), 2)}｜衝突 {_top_text(equation_reasoning.get('conflicts'), 2, '無重大衝突')}｜下一步 {_top_text(equation_reasoning.get('next_checks'), 2)}",
        f"- 公式誤差追因: {_top_text(equation_reasoning.get('error_sources'), 2, '暫無重大公式衝突')}｜待補變數 {_top_text(equation_reasoning.get('missing_variables'), 2, '目前無立即缺口')}｜數學策略 {equation_reasoning.get('math_policy', '先補變數再升級數學')}",
        f"- 買賣行為對比: {state_trade.get('gua', 'NA')}/{state_trade.get('label', '資料不足')}；買方 {state_trade.get('buy_behavior', '')}；賣方 {state_trade.get('sell_behavior', '')}",
        f"- 籌碼/量能: {capital_text}",
        f"- 市場心跳: {stethoscope.get('label', '資料不足')}｜分數 {stethoscope.get('score', 'NA')}｜{_compact_vitals(stethoscope)}",
        f"- 人性/兵法: {psychological_warfare.get('label', '資料不足')}｜戰術 {psychological_warfare.get('tactic_candidate', 'NA')}｜{_top_text(psychological_warfare.get('cause_effect_chain'), 2)}",
        f"- 日夜盤變異: {day_night_variance.get('label', '資料不足')}｜{day_night_variance.get('relation_label', 'unknown')}｜{_top_text(day_night_variance.get('cause_candidates'), 2)}",
        f"- 實務病因: {practical.get('label', '資料不足')}｜{practical.get('dialogue_focus', '慢性病灶不會一天消失；單日劇變要看觸發鈕與倉位重定價')}｜即效藥 {practical.get('fact_changing_medicine', {}).get('label', 'NA')}｜主因 {_top_text(practical.get('internal_causes'), 3)}｜觸發 {_top_text(practical.get('external_triggers'), 3)}",
        f"- 漲跌歸因: {close_cause.get('label', '資料不足')}｜{close_cause.get('headline', '')}｜{_top_text(close_cause.get('cause_candidates'), 3)}",
        "",
        "# 驗證與留底",
        f"- 下個驗證: {_top_text(satellite.get('route_checks') or interface.get('watch_points'), 4)}",
        f"- 誤差檢討: {next_day_validation.get('headline', '尚無到期預測可驗證')}｜{next_day_validation.get('governance', '只做核對，不自動升級為投資命令')}",
        f"- 模型可靠度: {reliability.get('headline', '資料不足')}｜整體 {pct(reliability.get('overall_hit_rate'))}；近10筆 {pct(reliability.get('recent_10_hit_rate'))}；近30筆 {pct(reliability.get('recent_30_hit_rate'))}",
        f"- 資料時點: 報告日 {payload['input']['date']}；分析基準日 {check['signal_date']}；資料狀態 {freshness_status_text(freshness['overall_status'])}；日期稽核 {date_audit.get('label', '資料不足')}",
        f"- 正式方向訊號: {'啟用' if policy['main_multi_day_direction']['enabled'] else '停用；以下為研究判讀'}",
        "- 細節留底: `reports/today_market_forecast_detail.md`；完整資料: `reports/today_market_forecast.json`",
        "- 防呆: 本報告只做風險預警、0/1劇本與驗證重點，不產生投資命令。",
        "",
        "## 一句話",
        f"**{summary.get('bias_text', '資料不足')}。{summary.get('plain_summary', '')}**",
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_brief_forecast(payload: dict) -> str:
    return render_compact_brief_forecast(payload)


def render_verbose_brief_forecast_legacy(payload: dict) -> str:
    check = payload["index_check"]
    date_audit = payload.get("market_date_audit", {})
    mode_switch = payload.get("market_mode_switch", {})
    freshness = payload["data_freshness"]
    premarket = payload["premarket"]
    intraday = payload.get("intraday_tactical_monitor", {})
    human_behavior = payload.get("human_behavior_market_pattern", {})
    psychological_warfare = payload.get("psychological_warfare_pattern", {})
    night_psychology = premarket.get("night_psychology", {})
    day_night_variance = payload.get("day_night_variance_pattern", {})
    sector_pressure = payload.get("sector_pressure_observation", {})
    endogenous_pulse = payload.get("endogenous_regulation_pulse", {})
    heart_rhythm = payload.get("market_heart_rhythm", {})
    close_cause = payload.get("close_cause_attribution", {})
    programmed_pressure = payload.get("programmed_pressure_pattern", {})
    external_reset = payload.get("external_event_reset_monitor", {})
    situation = payload.get("situation_psychology_context", {})
    fundamental = payload.get("fundamental_constitution", {})
    arbitration = payload.get("master_arbitration", {})
    practical = payload.get("practical_cause_arbitration", {})
    stethoscope = payload.get("market_stethoscope", {})
    protection = payload.get("market_protection_layers", {})
    interface = payload.get("crisis_opportunity_interface", {})
    cross_market = payload.get("cross_market_entanglement", {})
    monthly_cycle = payload.get("monthly_cycle_monitor", {})
    summary = payload["integrated_summary"]
    satellite = payload.get("weather_satellite_forecast", {})
    bagua = payload["bagua_lifecycle"]
    technical = payload["technical_phase"]
    peak_warning = payload.get("peak_to_valley_warning", {})
    peak_backtest = payload.get("peak_to_valley_backtest", {})
    route = payload["route_reference"]
    bottom_history = payload.get("bottom_event_reference", {})
    crash = payload["crash_monitor"]
    market_health = payload.get("market_health", {})
    health_value = market_health.get("health_value", {})
    candle = payload["candlestick_pattern"]
    washout = payload["washout_pattern"]
    forecast = payload["forecast"]
    capital = payload["capital_flow"]
    policy = payload["production_policy"]
    reliability = payload.get("direction_reliability_policy", {})
    primary = bagua.get("primary", {})
    monthly = bagua.get("monthly", {})
    weekly = bagua.get("weekly", {})
    daily = bagua.get("daily", {})
    levels = technical.get("levels", {})
    bottom = bagua.get("bottom_watch", {})
    crash_levels = crash.get("levels", {})
    drawdown_guardrail = crash.get("drawdown_guardrail", {})
    health_score = crash.get("health_score", market_health_score(crash))
    risk_value = crash.get("risk_value")
    health_label = crash.get("health_label") or market_health_label(health_score, crash.get("alert_code"))
    confirmation = crash.get("confirmation", {})
    false_crash = crash.get("false_crash_filter", {})
    capital_policy = crash.get("capital_policy", {})
    state_sop = crash.get("state_sop", {})
    next_day_validation = payload.get("error_review", {}).get("next_day_validation", {})
    psychology = payload.get("psychology_state", {})
    state_trade = bagua.get("roles", {}).get("state_gua", {}).get("trade_annotation") or bagua_trade_annotation(primary.get("code"))
    root_cause_items = satellite.get("root_cause_decomposition", {}).get("items", [])
    primary_root_cause = root_cause_items[0] if root_cause_items else {}
    night_trend = satellite.get("night_trend", {})
    zero_one_tilt = satellite.get("zero_one_tilt", {})
    tilt_evidence = zero_one_tilt.get("evidence", [])[:3]
    tilt_counter = zero_one_tilt.get("counter_conditions", [])[:3]

    lines = [
        f"# 台股大盤重點報告（{payload['input']['date']}）",
        "",
        f"# 0/1走向重點：{zero_one_tilt.get('label', '資料不足')}",
        f"**{zero_one_tilt.get('summary', '')}**",
        f"- 偏向分數: {zero_one_tilt.get('score', 'NA')}；分支 {zero_one_tilt.get('branch', 'NA')}",
        f"- 有利證據: {'；'.join(tilt_evidence) or 'NA'}",
        f"- 反證條件: {'；'.join(tilt_counter) or 'NA'}",
        f"- 關鍵線: 防守 {format_levels(intraday.get('defense_levels', [])) or 'NA'}；轉強 {format_levels(intraday.get('reclaim_levels', [])) or 'NA'}",
        "",
        f"# 氣象衛星式下一步預判：{satellite.get('headline', '資料不足')}",
        f"**{satellite.get('next_step', '')}**",
        f"- 預報框架: {satellite.get('framework', 'NA')}；信心 {satellite.get('confidence', '資料不足')}",
        f"- 信心範圍: {satellite.get('confidence_scope', '信心指整體日盤方向，不等於夜盤趨勢。')}",
        f"- 預測三層: {'；'.join(satellite.get('forecast_layers', [])[:3]) or 'NA'}",
        f"- 夜盤趨勢: {night_trend.get('label', '資料不足')}；官方 {pct(night_trend.get('official_spread_per'))}；開收 {pct(night_trend.get('open_close_return'))}；前夜比較 {pct(night_trend.get('previous_night_close_return'))}；收 {num(night_trend.get('close'))}；低 {num(night_trend.get('low'))}；{night_trend.get('path_shape', '')}",
        f"- 觀測站: {'；'.join(satellite.get('route_checks', [])[:4]) or 'NA'}",
        f"- 風暴雲系: {'；'.join(satellite.get('storm_cells', [])[:3]) or '目前未見重大風暴雲系'}",
        f"- 發病觸發因子: {'；'.join(satellite.get('activation_triggers', [])[:4]) or '目前未觸發，列入觀察'}",
        f"- 主病灶分解: {primary_root_cause.get('surface_symptom', 'NA')}｜{primary_root_cause.get('mechanism', '')}",
        f"- 根因候選: {'；'.join(primary_root_cause.get('root_cause_candidates', [])[:3]) or 'NA'}",
        f"- 自主修正: {satellite.get('error_repair_policy', '錯誤需分類、找因、降權或修正，隔日續驗。')}",
        "",
        f"# 總仲裁：{arbitration.get('headline', '資料不足')}｜{arbitration.get('risk_posture', 'NA')}",
        f"**{arbitration.get('decision', '')}**",
        f"- 主控層: {arbitration.get('dominant_layer', 'NA')}；健康 {arbitration.get('health_score', 'NA')}；風險 {arbitration.get('risk_score', 'NA')}；命格 {arbitration.get('constitution_score', 'NA')}",
        f"- 衝突點: {'；'.join(arbitration.get('conflicts', [])[:4]) or '目前未見重大分層衝突'}",
        f"- 驗證點: {'；'.join(arbitration.get('validation', [])[:4]) or 'NA'}",
        f"- 防呆: {arbitration.get('guardrail', '總仲裁只決定訊號優先序與風控語氣。')}",
        "",
        f"# 實務主因仲裁：{practical.get('label', '資料不足')}｜內部 {practical.get('internal_structure_score', 'NA')} / 外部 {practical.get('external_trigger_score', 'NA')}",
        f"**{practical.get('summary', '')}**",
        f"- 治療階段: 第{practical.get('treatment_phase', 'NA')}期 / {practical.get('treatment_stage', 'NA')}；{practical.get('treatment_policy', '')}",
        f"- 藥效觀測: {practical.get('medicine_type', 'NA')}；對症 {practical.get('correct_medicine', 'NA')}；陰霾風險 {practical.get('after_effect_risk', 'NA')}",
        f"- 即效藥: {practical.get('fact_changing_medicine', {}).get('label', 'NA')}；分數 {practical.get('immediate_medicine_score', 'NA')}；{practical.get('fact_changing_medicine', {}).get('effect', '')}",
        f"- 即效因子: {'；'.join(practical.get('fact_changing_medicines', [])[:4]) or 'NA'}",
        f"- 因果程序: {practical.get('causality_rule', 'NA')}",
        f"- 主因候選: {'；'.join(practical.get('internal_causes', [])[:4]) or 'NA'}",
        f"- 今日觸發器: {'；'.join(practical.get('external_triggers', [])[:4]) or 'NA'}",
        f"- 病灶基因模型: {practical.get('gene_trigger_model', 'NA')}",
        f"- 0/1規則: {practical.get('one_zero_rule', 'NA')}",
        f"- 模型效果: {practical.get('model_effect', 'NA')}",
        f"- 防呆: {practical.get('guardrail', '只作病因排序，不產生買賣命令。')}",
        "",
        f"# 市場聽診器：{stethoscope.get('label', '資料不足')}｜分數 {stethoscope.get('score', 'NA')}",
        f"**{stethoscope.get('summary', '')}**",
        f"- 生命徵象: {'；'.join([d.get('name', '') + '/' + d.get('label', '') for d in stethoscope.get('diagnostics', [])[:6]]) or 'NA'}",
        f"- 缺口: {'；'.join(stethoscope.get('missing', [])) or '無重大缺口'}",
        f"- 規則: {stethoscope.get('rule', 'NA')}",
        "",
        f"# 市場防呆保護層：{protection.get('label', '資料不足')}｜分數 {protection.get('score', 'NA')}",
        f"**{protection.get('current_judgment', protection.get('summary', ''))}**",
        f"- 啟動層: {'；'.join(protection.get('active_layers', [])[:4]) or '無明確啟動'}",
        f"- 偏弱層: {'；'.join(protection.get('warning_layers', [])[:4]) or '無明顯偏弱'}",
        f"- 失效層: {'；'.join(protection.get('failed_layers', [])[:4]) or '尚未見失效'}",
        f"- 危機條件: {'；'.join(protection.get('crisis_conditions', [])[:3]) or 'NA'}",
        f"- 防呆: {protection.get('guardrail', '只判斷保護層是否失靈，不產生投資命令。')}",
        "",
        f"# 危機與轉機界面：{interface.get('label', '資料不足')}｜{interface.get('interface_state', 'NA')}",
        f"**{interface.get('summary', '')}**",
        f"- 危機邊界: {'；'.join(interface.get('crisis_edge', [])[:3]) or 'NA'}",
        f"- 轉機邊界: {'；'.join(interface.get('opportunity_edge', [])[:3]) or 'NA'}",
        f"- 峰谷信號: {'；'.join(interface.get('peak_valley_signals', [])[:4]) or 'NA'}",
        f"- 觀測點: {'；'.join(interface.get('watch_points', [])[:4]) or 'NA'}",
        f"- 我能做什麼: {'；'.join(interface.get('what_i_can_do', [])[:3]) or 'NA'}",
        f"- 0/1規則: {interface.get('zero_one_rule', 'NA')}",
        "",
        f"# 跨盤糾結：{cross_market.get('label', '資料不足')}｜分數 {cross_market.get('score', 'NA')}",
        f"**{cross_market.get('summary', '')}**",
        f"- 方向: 美Nasdaq {cross_market.get('directions', {}).get('nasdaq', 'NA')} / 費半 {cross_market.get('directions', {}).get('sox', 'NA')} / ADR {cross_market.get('directions', {}).get('tsm_adr', 'NA')} / 台指夜盤 {cross_market.get('directions', {}).get('tx_night', 'NA')} / 台股日盤 {cross_market.get('directions', {}).get('twii_cash', 'NA')}",
        f"- 判斷: {'；'.join(cross_market.get('evidence', [])[:4]) or 'NA'}",
        f"- 規則: {cross_market.get('rule', 'NA')}",
        "",
        f"# 基本面命格：{fundamental.get('label', '資料不足')}｜{fundamental.get('score', 'NA')}/100",
        f"**{fundamental.get('summary', '')}**",
        f"- 支撐因素: {'；'.join(fundamental.get('drivers', [])[:4]) or 'NA'}",
        f"- 壓力因素: {'；'.join(fundamental.get('pressures', [])[:4]) or 'NA'}",
        f"- 資料缺口: {'；'.join(fundamental.get('data_gaps', [])[:3]) or '無重大缺口'}",
        f"- 防呆: {fundamental.get('guardrail', '基本面不等於今日漲跌。')}",
        "",
        f"# {reliability.get('headline', '模型可靠度：資料不足')}",
        f"**{reliability.get('summary', '')}**",
        f"- 整體命中率: {pct(reliability.get('overall_hit_rate'))}；最近10筆 {pct(reliability.get('recent_10_hit_rate'))}；最近30筆 {pct(reliability.get('recent_30_hit_rate'))}",
        f"- 權重配置: 方向 {reliability.get('direction_weight', 'low')}；風控 {reliability.get('risk_weight', 'high')}；已驗證子模組 {reliability.get('validated_submodel_weight', 'high')}",
        f"- 規則: {'；'.join(reliability.get('rules', [])[:2]) or '方向預測維持研究性觀察。'}",
        f"- 留底: `{reliability.get('report', 'reports/model_reliability_trend_audit.md')}`",
        "",
        f"# 市場模式：{mode_switch.get('headline', '資料不足')}｜分數 {mode_switch.get('risk_score', 'NA')}",
        f"**{mode_switch.get('action', '')}**",
        f"- 分數等級: {mode_switch.get('score_range', market_mode_score_range(mode_switch.get('risk_score')))}",
        f"- 模式依據: {'；'.join(mode_switch.get('triggers', [])) or '無'}",
        f"- 模式規則: {mode_switch.get('rule', '')}",
        "",
        f"# 市場心律：{heart_rhythm.get('label', '資料不足')}｜分數 {heart_rhythm.get('score', 'NA')}",
        f"**{heart_rhythm.get('summary', '')}**",
        f"- 臨床分型: {heart_rhythm.get('clinical_label', 'NA')}；{heart_rhythm.get('clinical_meaning', '')}",
        f"- 正負節奏: 近{heart_rhythm.get('recent_days', 'NA')}日，上漲 {heart_rhythm.get('up_days', 'NA')} / 下跌 {heart_rhythm.get('down_days', 'NA')}；切換率 {pct(heart_rhythm.get('switch_rate'))}",
        f"- 修復能力: 跌後修復率 {pct(heart_rhythm.get('down_repair_rate'))}；最大連跌 {heart_rhythm.get('max_down_streak', 'NA')}；平均日振幅 {pct(heart_rhythm.get('avg_abs_return'))}",
        f"- 心律規則: {heart_rhythm.get('rule', 'NA')}",
        "",
        f"# 漲跌原因歸因：{close_cause.get('label', '資料不足')}｜修正 {close_cause.get('risk_adjustment', 'NA')}",
        f"**{close_cause.get('headline', '')}**",
        f"- 證據: {'；'.join(close_cause.get('evidence', [])[:3]) or 'NA'}",
        f"- 原因候選: {'；'.join(close_cause.get('cause_candidates', [])[:4]) or 'NA'}",
        f"- 模型效果: {close_cause.get('model_effect', 'NA')}",
        f"- 次日驗證: {'；'.join(close_cause.get('next_validation', [])[:3]) or 'NA'}",
        f"- 防呆: {close_cause.get('guardrail', '只作原因歸因，不產生買賣命令。')}",
        "",
        f"# 族群分化壓力：{sector_pressure.get('label', '無人工族群觀察')}｜分數 {sector_pressure.get('risk_score', 'NA')}",
        f"**{sector_pressure.get('summary', '')}**",
        f"- 弱勢族群: {'、'.join(sector_pressure.get('weak_sectors', [])) or '無'}",
        f"- 原因候選: {'；'.join(sector_pressure.get('cause_candidates', [])[:3]) or 'NA'}",
        f"- 次日驗證: {'；'.join(sector_pressure.get('next_validation', [])[:2]) or 'NA'}",
        f"- 防呆: {sector_pressure.get('guardrail', '族群壓力觀察只作風控留底，不產生買賣命令。')}",
        "",
        f"# 市場內生調節脈動：{endogenous_pulse.get('label', '資料不足')}｜分數 {endogenous_pulse.get('score', 'NA')}",
        f"**{endogenous_pulse.get('summary', '')}**",
        f"- 分數等級: {endogenous_pulse.get('score_range', 'NA')}",
        f"- 外部安靜: {'是' if endogenous_pulse.get('external_quiet') else '否'}；最大外部波動 {pct(endogenous_pulse.get('external_abs_max'))}",
        f"- 模型效果: {endogenous_pulse.get('model_effect', '')}",
        f"- 次日驗證: {'；'.join(endogenous_pulse.get('next_validation', [])[:2]) or 'NA'}",
        f"- 防呆: {endogenous_pulse.get('guardrail', '只作統計觀察，不產生買賣命令。')}",
        "",
        f"# 次日誤差驗證：{next_day_validation.get('headline', '尚無到期預測可驗證')}",
        f"**{next_day_validation.get('summary', '')}**",
        f"- 驗證治理: {next_day_validation.get('governance', '不自動調參；只做核對。')}",
        *[
            f"- 今日必查: {task.get('name')}｜{task.get('rule')}"
            for task in next_day_validation.get("tasks", [])[:3]
        ],
        "",
        f"# 盤中戰術雷達：{intraday.get('label', '資料不足')}",
        f"**{intraday.get('summary', '')}**",
        f"- 即時資料: {'有' if intraday.get('live_usable') else '無'}；來源 {intraday.get('source', 'none')}",
        f"- 戰術動作: {intraday.get('action', '')}",
        f"- 防守觀察: {format_levels(intraday.get('defense_levels', [])) or 'NA'}",
        f"- 轉強觀察: {format_levels(intraday.get('reclaim_levels', [])) or 'NA'}",
        "",
        f"# 夜盤收盤心理：{night_psychology.get('label', '資料不足')}｜分數 {night_psychology.get('score', 'NA')}",
        f"**{night_psychology.get('summary', '')}**",
        f"- 主導情緒: {night_psychology.get('dominant_emotion', 'NA')}；戰術候選 {night_psychology.get('tactic', 'NA')}",
        f"- 心理證據: {'；'.join(night_psychology.get('evidence', [])[:4]) or 'NA'}",
        f"- 日盤驗證: {'；'.join(night_psychology.get('next_validation', [])[:3]) or 'NA'}",
        f"- 防呆: {night_psychology.get('guardrail', '夜盤心理層只作驗證假設，不產生買賣命令。')}",
        "",
        f"# 日夜盤變異分析：{day_night_variance.get('label', '資料不足')}｜{day_night_variance.get('relation_label', 'unknown')}",
        f"**{day_night_variance.get('summary', '')}**",
        f"- 變異幅度: {day_night_variance.get('variance_label', '資料不足')}；分數 {day_night_variance.get('score', 'NA')}",
        f"- 高低差原因候選: {'；'.join(day_night_variance.get('cause_candidates', [])[:4]) or 'NA'}",
        f"- 隱藏病因候選: {'；'.join(day_night_variance.get('hidden_cause_candidates', [])[:4]) or 'NA'}",
        f"- 防呆: {day_night_variance.get('guardrail', '日夜盤變異只作統計觀察，不產生買賣命令。')}",
        "",
        f"# 行為模式偵測：{programmed_pressure.get('label', '資料不足')}｜壓低分數 {programmed_pressure.get('current_score', 'NA')}",
        f"**{programmed_pressure.get('summary', '')}**",
        f"- 壓低日期: {', '.join(programmed_pressure.get('pressure_dates', [])) or '無'}",
        f"- 次日驗證: {'；'.join(programmed_pressure.get('next_validation', [])[:2]) or 'NA'}",
        "",
        f"# 外部事件重置：{external_reset.get('label', '資料不足')}｜分數 {external_reset.get('reset_score', 'NA')}",
        f"**{external_reset.get('summary', '')}**",
        f"- 是否重置: {'是' if external_reset.get('reset_active') else '否'}；因果優先權 {external_reset.get('causal_priority', 'internal_primary')}",
        f"- 觸發原因: {'、'.join(external_reset.get('reasons', [])) or '無明確觸發'}",
        f"- 觀察: {external_reset.get('watch', '')}",
        "",
        f"# 新聞面與生活心理：{situation.get('label', '資料不足')}｜分數 {situation.get('score', 'NA')}",
        f"**{situation.get('posture', '')}**",
        f"- 觸發因子: {'；'.join(situation.get('triggers', [])[:4]) or '無明確情境觸發'}",
        f"- 情境證據: {'；'.join(situation.get('evidence', [])[:4]) or 'NA'}",
        f"- 日盤驗證: {'；'.join(situation.get('next_validation', [])[:3]) or 'NA'}",
        f"- 防呆: {situation.get('guardrail', '情境心理只作濾鏡，不產生買賣命令。')}",
        "",
        f"# 月內週期階段：{monthly_cycle.get('stage', '資料不足')}｜第{monthly_cycle.get('week_of_month', 'NA')}週",
        f"**{monthly_cycle.get('summary', '')}**",
        f"- 週期順序: {monthly_cycle.get('cycle_order', '')}",
        f"- 月內位置: {pct(monthly_cycle.get('month_position'))}；月報酬 {pct(monthly_cycle.get('month_return'))}；近5日 {pct(monthly_cycle.get('recent5_return'))}",
        f"- 季度背景: {monthly_cycle.get('quarter_stage', '資料不足')}；{monthly_cycle.get('quarter')} {pct(monthly_cycle.get('quarter_return'))}",
        f"- 監控動作: {monthly_cycle.get('action', '')}",
        "",
        f"# 人類行為模式：{human_behavior.get('label', '資料不足')}｜{human_behavior.get('crowd_state', 'unknown')}",
        f"**{human_behavior.get('summary', '')}**",
        f"- 戰術候選: {human_behavior.get('tactic_candidate', '資料不足')}；分數 {human_behavior.get('score', 'NA')}",
        f"- 可控風險: {'是' if human_behavior.get('controllable_risk') else '否'}",
        f"- 防呆: {human_behavior.get('guardrail', '人類行為模式只作統計觀察，不產生買賣命令。')}",
        "",
        f"# 心理戰與八卦兵法：{psychological_warfare.get('label', '資料不足')}｜{psychological_warfare.get('stance', 'unknown')}",
        f"**戰術候選 {psychological_warfare.get('tactic_candidate', 'NA')}；分數 {psychological_warfare.get('score', 'NA')}。{psychological_warfare.get('next_validation', '')}**",
        f"- 資料意圖: {psychological_warfare.get('intent_summary', '資料是意圖表現，仍需收盤驗證。')}",
        f"- 因果鏈: {'；'.join(psychological_warfare.get('cause_effect_chain', [])[:3]) or 'NA'}",
        f"- 文字轉數字處理: {psychological_warfare.get('action_policy', 'NA')}",
        f"- 兵法原理: {'；'.join(psychological_warfare.get('sunzi_principles', [])[:4]) or 'NA'}",
        f"- 虛實驗證: {'；'.join(psychological_warfare.get('truth_test', [])[:2]) or 'NA'}",
        f"- 防呆: {psychological_warfare.get('guardrail', '兵法與八卦只作分析語言，不產生投資命令。')}",
        "",
        f"# 群眾心理路徑：{psychology.get('state_label', '資料不足')}｜{psychology.get('direction', 'unknown')}",
        f"**淨急迫分數 {psychology.get('net_urgency_score', 'NA')}；{psychology.get('guardrail', '')}**",
        f"- 外生事件重置觀察: {'是' if psychology.get('exogenous_reset_watch') else '否'}；{'、'.join(psychology.get('exogenous_reset_reasons', [])) or '無明確觸發'}",
        *[
            f"- 路徑{item.get('rank')}: {item.get('label')}｜{item.get('condition')}｜{item.get('effect')}"
            for item in psychology.get("next_paths", [])[:3]
        ],
        "",
        f"# 峰轉谷早期預警：{peak_warning.get('level', '資料不足')}｜{peak_warning.get('headline', '資料不足')}",
        f"**{peak_warning.get('summary', '')}**",
        "",
        f"# 大盤健康指數：{num(health_score)}/100｜{health_label}",
        f"# 大盤風險值：{num(risk_value)}/100｜{crash.get('alert_level', '資料不足')}",
        f"# 健康價值判斷：{health_value.get('label', '資料不足')}｜{health_value.get('value', 'NA')}/100",
        f"**{health_value.get('headline', '')}**",
        f"- 判斷規則: {health_value.get('rule', '健康價值判斷看風險是否可控，不以單日漲跌當唯一依據。')}",
        f"- 風險是否可控: {'是' if health_value.get('controllable_risk') else '否'}",
        "",
        f"## 預警頭條：{crash.get('alert_level', '資料不足')}｜{crash.get('state', '資料不足')}",
        "",
        f"**{crash.get('alert_summary', '')} {crash.get('summary', '')}**",
        "",
        f"## 狀態SOP：{state_sop.get('state_gua', 'NA')}｜{state_sop.get('sop', '資料不足')}",
        "",
        f"**{state_sop.get('action', '')}**",
        f"- 強勢標的定義: {strong_target_definition()}",
        f"- 合理底用途: {state_sop.get('reasonable_bottom_role', '風控參考線')}",
        f"- 長週期背景: {state_sop.get('background_gua', 'NA')}（不與當前狀態混用）",
        "",
        f"## 崩盤三重核對：{confirmation.get('stage', '資料不足')}",
        "",
        f"**{confirmation.get('summary', '')}**",
        "",
        f"## 假崩盤過濾：{false_crash.get('label', '資料不足')}",
        "",
        f"**{false_crash.get('action', '')}**",
        "",
        f"- 峰轉谷依據: {peak_backtest.get('summary', '資料不足')}",
        f"- 峰轉谷留底: `reports/peak_to_valley_warning_backtest.md` / `reports/peak_to_valley_warning_backtest.json`",
        "",
        f"- 底數基準: {crash_levels.get('anchor_date', 'NA')} 波段高點 {num(crash_levels.get('anchor_peak'))}",
        f"- 底數規則: {crash_levels.get('anchor_rule', '')}",
        f"- 高檔正常艮區: {num(crash_levels.get('normal_gen_low'))}～{num(crash_levels.get('normal_gen_high'))}",
        f"- 重大修正警戒線: {num(crash_levels.get('major_correction_line'))}",
        f"- 合理底部區: {num(crash_levels.get('normal_bottom_low'))}～{num(crash_levels.get('normal_bottom_high'))}",
        f"- 崩盤前奏警戒線: {num(crash_levels.get('crash_warning'))}",
        f"- 深坎防守區: {num(crash_levels.get('deep_kan_low'))}～{num(crash_levels.get('deep_kan_high'))}",
        f"- 風控有效低點: {num(crash.get('effective_low'))}（{crash.get('live_monitor', {}).get('source', '日線')}）",
        "",
        f"- 報告日期: {payload['input']['date']}",
        f"- 報告模式: {date_audit.get('label', '資料不足')}｜{date_audit.get('status', '')}",
        f"- 分析基準日: {check['signal_date']}",
        f"- 日期稽核: 正式日線 {date_audit.get('official_signal_date', check['signal_date'])}；今日盤中快照 {date_audit.get('live_snapshot_date') or 'NA'}",
        f"- 資料狀態: {freshness_status_text(freshness['overall_status'])}",
        f"- 正式方向訊號: {'啟用' if policy['main_multi_day_direction']['enabled'] else '停用；以下為研究判讀'}",
        "",
        "## 一句話",
        "",
        f"**{summary.get('bias_text', '資料不足')}。{summary.get('plain_summary', '')}**",
        "",
        "## 目前位置",
        "",
        "- 八卦順序: 震 → 巽 → 離 → 坤 → 兌 → 乾 → 坎 → 艮 → 震",
        f"- 主卦: {primary.get('gua')} / {primary.get('label')} ({primary.get('code')})",
        f"- 月/週/日: {monthly.get('gua')} / {weekly.get('gua')} / {daily.get('gua')}",
        f"- 狀態卦買賣標註: {state_trade.get('gua', 'NA')}/{state_trade.get('label', '資料不足')}；買方: {state_trade.get('buy_behavior', '')}；賣方: {state_trade.get('sell_behavior', '')}",
        f"- 標註防呆: {state_trade.get('guardrail', '買賣標註只作風控與行為對比，不是投資命令。')}",
        f"- 路線: {route.get('current_state_label', '資料不足')}；{route.get('plain_summary', '')}",
        f"- 峰轉谷預警: {peak_warning.get('level', '資料不足')}；{peak_warning.get('summary', '')}",
        f"- 測底歷史: {bottom_history.get('plain_summary', '資料不足')}",
        f"- 技術段位: {technical.get('label', '資料不足')}；{technical.get('summary', '')}",
        f"- 回撤分層: {drawdown_guardrail.get('stage', '資料不足')}；{drawdown_guardrail.get('summary', '')}",
        f"- K線與洗盤: {candle.get('label')}；{washout.get('label')}。",
        f"- 風險預警: {crash.get('alert_level', '資料不足')}；{crash.get('alert_summary', '')}",
            f"- 崩盤監控: {crash.get('state', '資料不足')}；{crash.get('summary', '')}",
            f"- 假崩盤過濾: {false_crash.get('label', '資料不足')}；{false_crash.get('rule', '')}",
            "",
            "## 關鍵價位",
        "",
        f"- 收盤/低點: {num(levels.get('close'))} / {num(levels.get('low'))}",
        f"- 近期低點/低收盤: {num(levels.get('recent_low'))} / {num(levels.get('recent_low_close'))}",
        f"- 均線: 5日 {num(levels.get('ma5'))}、10日 {num(levels.get('ma10'))}、20日 {num(levels.get('ma20'))}、60日 {num(levels.get('ma60'))}",
        f"- 波段高點回撤: {levels.get('swing_high_date')} 高點 {num(levels.get('swing_high'))} 至今 {pct(levels.get('drawdown_from_swing_high'))}",
    ]
    if bottom.get("enabled"):
        lines.extend(
            [
                f"- 不破觀察: {num(bottom.get('latest_low_hold'))}",
                f"- 先收復: {num(bottom.get('first_reclaim_close'))}",
                f"- 強確認: {num(bottom.get('strong_reclaim_open'))} 或 5日線 {num(bottom.get('ma5_reclaim'))}",
            ]
        )
    if crash.get("enabled"):
        lines.extend(
            [
                f"- 底數基準: {crash_levels.get('anchor_date', 'NA')} 波段高點 {num(crash_levels.get('anchor_peak'))}",
                f"- 底數規則: {crash_levels.get('anchor_rule', '')}",
                f"- 高檔正常艮區: {num(crash_levels.get('normal_gen_low'))}～{num(crash_levels.get('normal_gen_high'))}",
                f"- 健康修正下緣: {num(crash_levels.get('healthy_pullback_low'))}",
                f"- 重大修正警戒線: {num(crash_levels.get('major_correction_line'))}",
                f"- 合理底部區: {num(crash_levels.get('normal_bottom_low'))}～{num(crash_levels.get('normal_bottom_high'))}",
                f"- 崩盤前奏警戒線: {num(crash_levels.get('crash_warning'))}",
                f"- 深坎防守區: {num(crash_levels.get('deep_kan_low'))}～{num(crash_levels.get('deep_kan_high'))}",
            ]
        )
    if premarket.get("is_premarket"):
        night_basis = premarket.get("night_basis_audit", {})
        lines.extend(
            [
                "",
                "## 今日盤前",
                "",
                f"- 壓力: {premarket.get('pressure')} / {premarket.get('scenario')}",
                f"- 觀察: {premarket.get('watch')}",
                f"- 美股/半導體: Nasdaq {pct(premarket.get('nasdaq_return_1d'))}、費半 {pct(premarket.get('sox_return_1d'))}、TSM ADR {pct(premarket.get('tsm_adr_return_1d'))}",
                f"- 台指期夜盤: {pct(premarket.get('tx_night_spread_per'))}，收 {num(premarket.get('tx_night_close'))}，低 {num(premarket.get('tx_night_low'))}",
                f"- 夜盤三基準: 官方 {pct(night_basis.get('official_spread_per'))}；開收 {pct(night_basis.get('open_close_return'))}；較前夜收盤 {pct(night_basis.get('vs_previous_night_close_return'))}",
                f"- 三基準防呆: {night_basis.get('summary', '三基準不得混用。')}",
            ]
        )
    lines.extend(render_intraday_tactical_lines(intraday))
    lines.extend(
        [
            "",
            "## 日期稽核防錯",
            "",
            f"- 模式: {date_audit.get('label', '資料不足')} / {date_audit.get('mode', 'NA')}",
            f"- 報告日期: {date_audit.get('forecast_date', payload['input']['date'])}",
            f"- 正式日線基準日: {date_audit.get('official_signal_date', check['signal_date'])}",
            f"- 今日盤中快照: {date_audit.get('live_snapshot_date') or 'NA'} / 價格 {num(date_audit.get('live_price'))} / 低點 {num(date_audit.get('live_low'))}",
            f"- 防錯結論: {date_audit.get('status', '資料不足')}",
            "",
            "## 風險監控",
            "",
            "- 原則: 不猜底、不猜崩盤；只監控早期條件是否同時惡化。",
            f"- 峰轉谷早期預警: {peak_warning.get('level', '資料不足')}｜{peak_warning.get('headline', '')}",
            f"- 預警等級: {crash.get('alert_level', '資料不足')}",
            f"- 目前狀態: {crash.get('state', '資料不足')}",
            f"- 回撤分層: {drawdown_guardrail.get('stage', '資料不足')}｜{drawdown_guardrail.get('summary', '')}",
            f"- 精算風險值: {num(crash.get('risk_value'))}/100",
            f"- 大盤健康指數: {num(crash.get('health_score'))}/100",
            f"- 風控有效價/低點: {num(crash.get('effective_close'))} / {num(crash.get('effective_low'))}",
            f"- 盤中監控來源: {crash.get('live_monitor', {}).get('source', 'none')}；{crash.get('live_monitor', {}).get('message', '')}",
            f"- 計算方法: {crash.get('risk_method', '')}",
            f"- 觸發條件: {'；'.join(crash.get('triggers', [])) or '無'}",
            f"- 早期預警: {'；'.join(crash.get('early_warnings', [])) or '無'}",
            f"- 假崩盤防呆: {false_crash.get('label', '資料不足')}；{false_crash.get('action', '')}",
            "",
            f"## 資金風控政策：{capital_policy.get('level', '資料不足')}｜{capital_policy.get('action', '')}",
            "",
            f"**{capital_policy.get('summary', '')}**",
            "",
            "| 核對項目 | 是否觸發 | 關鍵線 | 依據 |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for item in confirmation.get("checks", []):
        lines.append(
            f"| {item.get('name')} | {'是' if item.get('passed') else '否'} | "
            f"{num(item.get('level'))} | {item.get('basis')} |"
        )
    if capital_policy.get("guardrails"):
        lines.extend(["", "資金守則:"])
        for item in capital_policy.get("guardrails", []):
            lines.append(f"- {item}")
    if false_crash.get("checks"):
        lines.extend(
            [
                "",
                "| 假崩盤防呆 | 是否成立 | 依據 |",
                "| --- | --- | --- |",
            ]
        )
        for item in false_crash.get("checks", []):
            lines.append(f"| {item.get('name')} | {'是' if item.get('passed') else '否'} | {item.get('basis')} |")
    lines.extend(
        [
            "",
            "分項風險級距: 0～24 低風險；25～49 觀察；50～74 警戒；75～100 高風險。",
            "",
            "| 風險分項 | 權重 | 分項風險 | 分數範圍 | 加權分數 | 依據 |",
            "| --- | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for item in crash.get("risk_components", []):
        lines.append(
            f"| {item.get('name')} | {pct(item.get('weight'))} | {num(item.get('score'))} | "
            f"{item.get('score_range', risk_score_range(item.get('score')))} | "
            f"{num(item.get('weighted_score'))} | {item.get('basis')} |"
        )
    lines.extend(
        [
            "",
            "## 後續確認",
            "",
            f"- 坎延續: 跌破 {num(levels.get('low'))} 或反彈無力再破低。",
            f"- 止跌觀察: 至少 2 天不破低，並先收復 {num(levels.get('recent_low_close'))}。",
            "- 艮底確認: 3 天不破低且站回短均線，才可開始看艮。",
            "- 震啟動: 艮底後再向上突破，才談震；急跌不歸震。",
            "",
            "## 測底歷史樣本",
            "",
            f"- 方法: {bottom_history.get('method', '資料不足')}",
            f"- 目前事件: {bottom_history.get('current_event', {}).get('label', '資料不足')}",
            f"- 樣本範圍: {bottom_history.get('sample_scope', '資料不足')} / 樣本 {bottom_history.get('sample_count', 0)}",
            f"- 摘要: {bottom_history.get('plain_summary', '資料不足')}",
            "",
            "| 期間 | 樣本 | 平均報酬 | 上漲率 | 下跌逾3%率 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in bottom_history.get("stats", {}).get("by_horizon", []):
        lines.append(
            f"| {item.get('horizon')} | {item.get('sample_count')} | {pct(item.get('avg_return'))} | "
            f"{pct(item.get('up_rate'))} | {pct(item.get('down_3pct_rate'))} |"
        )
    rates = bottom_history.get("stats", {}).get("event_rates", {})
    if rates:
        lines.extend(
            [
                "",
                "| 歷史事件率 | 樣本 | 機率 |",
                "| --- | ---: | ---: |",
                f"| 3日內站回合理底 | {rates.get('reclaim_normal_3d', {}).get('sample_count')} | {pct(rates.get('reclaim_normal_3d', {}).get('rate'))} |",
                f"| 5日內再破合理底 | {rates.get('rebreak_normal_5d', {}).get('sample_count')} | {pct(rates.get('rebreak_normal_5d', {}).get('rate'))} |",
                f"| 20日內觸及崩盤線 | {rates.get('break_crash_20d', {}).get('sample_count')} | {pct(rates.get('break_crash_20d', {}).get('rate'))} |",
                f"| 5日內艮底雛形 | {rates.get('gen_bottom_confirm_5d', {}).get('sample_count')} | {pct(rates.get('gen_bottom_confirm_5d', {}).get('rate'))} |",
            ]
        )
    lines.extend(
        [
            "## 研究分布",
            "",
        ]
    )
    for row in forecast.get("forecasts", []):
        lines.append(
            f"- {row['horizon_days']}日: 歷史多數 {direction_text(row['predicted_direction'])}，"
            f"上漲 {pct(row.get('probability_up'))} / 下跌 {pct(row.get('probability_down'))}，樣本 {row.get('case_count')}"
        )
        if row.get("horizon_days") == 1:
            close_range = row.get("close_range_forecast", {})
            central = close_range.get("central_50_range", [])
            outer = close_range.get("outer_80_range", [])
            paths = row.get("intraday_path_distribution", {})
            if len(central) == 2 and len(outer) == 2:
                lines.append(
                    f"  - 收盤分位區間: 中央50% {num(central[0])}～{num(central[1])}；"
                    f"外圍80% {num(outer[0])}～{num(outer[1])}（探索性，非保證區間）"
                )
            if paths.get("sample_count"):
                lines.append(
                    f"  - 盤中深殺後收復病例率: {pct(paths.get('deep_selloff_recovered_rate'))}；"
                    f"跌破前低後收復率: {pct(paths.get('prior_low_breach_then_recover_rate'))}"
                )
    lines.extend(render_path_risk_lines(payload.get("path_risk_adjustment", {})))
    lines.extend(
        [
            "",
            "## 留底",
            "",
            f"- 法人籌碼期權: {capital.get('data_status', '資料不足')}。",
            "- 完整細節: `reports/today_market_forecast_detail.md`",
            "- 完整資料: `reports/today_market_forecast.json`",
            "- 誤差閉迴路: `reports/error_review.md` / `reports/error_review.json`",
            "- 峰轉谷回測: `reports/peak_to_valley_warning_backtest.md` / `reports/peak_to_valley_warning_backtest.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def market_health_score(crash: dict) -> int:
    risk_points = safe_int(crash.get("risk_points"), 0)
    score = 100 - risk_points * 8
    alert_code = crash.get("alert_code")
    if alert_code == "confirmed_warning":
        score = min(score, 30)
    elif alert_code == "early_alert":
        score = min(score, 45)
    elif alert_code == "watch":
        score = min(score, 65)
    return max(0, min(100, int(score)))


def market_health_label(score: int, alert_code: str | None = None) -> str:
    if alert_code == "confirmed_warning" or score <= 30:
        return "重病警戒"
    if alert_code == "early_alert" or score <= 45:
        return "高風險"
    if alert_code == "watch" or score <= 65:
        return "轉弱預警"
    if score <= 80:
        return "普通偏弱"
    return "健康"


def render_forecast(payload: dict) -> str:
    forecast = payload["forecast"]
    check = payload["index_check"]
    update = payload["data_update"]
    external_update = payload["external_update"]
    night_futures_update = payload["night_futures_update"]
    premarket = payload["premarket"]
    intraday = payload.get("intraday_tactical_monitor", {})
    cause = payload["cause_analysis"]
    simulation = payload["historical_self_simulation"]
    memory = payload["memory_industry_risk"]
    freshness = payload["data_freshness"]
    validation = payload["model_validation"]
    policy = payload["production_policy"]

    lines = [
        "# 今日市場研究與風險報告",
        "",
        f"- 預測基準日期: {payload['input']['date']}",
        f"- 台股分析基準日（最後有效資料）: {check['signal_date']}",
        f"- 台股資料更新: {status(update)} ({update.get('latest_date') or '未更新'})",
        f"- 外部市場更新: {status(external_update)} ({external_update.get('latest_date') or '未更新'})",
        f"- 台指期夜盤更新: {status(night_futures_update)} ({night_futures_update.get('latest_date') or '未更新'})",
        f"- 時效稽核: {freshness_status_text(freshness['overall_status'])}（{freshness['checked_at']}）",
        f"- 大盤多日正式方向訊號: {'啟用' if policy['main_multi_day_direction']['enabled'] else '停用'}",
        "",
        "## 最後有效資料逐項確認",
        "",
        "| 資料 | 最後有效日期 | 本次應有基準 | 差距天數 | 狀態 | 來源 |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    validation_lines = [
        "",
        "## \u6b63\u5f0f\u6a21\u578b\u9a57\u8b49\u72c0\u614b",
        "",
        f"- \u5be6\u7528\u9580\u6abb\u662f\u5426\u901a\u904e: {'\u662f' if validation.get('passed') else '\u5426'}",
        f"- \u76ee\u6a19: \u547d\u4e2d\u7387 {validation.get('required_accuracy', .9):.0%}\u3001\u6a23\u672c\u81f3\u5c11 {validation.get('required_cases', 100)}\u3001\u8986\u84cb\u7387 {validation.get('required_coverage', .1):.0%}\u300195% \u4fe1\u8cf4\u4e0b\u9650 {validation.get('required_wilson_lower', .8):.0%}\u3001\u4e14\u512a\u65bc\u57fa\u6e96",
        f"- \u6b63\u5f0f\u7d50\u8ad6: {'\u5df2\u9054\u5be6\u7528\u9580\u6abb' if validation.get('passed') else '\u5c1a\u672a\u9054\u5be6\u7528\u9580\u6abb\uff1b\u4ee5\u4e0b\u65b9\u5411\u8207\u6a5f\u7387\u50c5\u4f9b\u7814\u7a76\uff0c\u4e0d\u4ee3\u8868\u5df2\u9a57\u8b49\u7684\u9ad8\u4fe1\u5fc3'}",
        f"- \u7814\u7a76\u7d50\u6848: {'\u5df2\u5b8c\u6210' if validation.get('research_completed') else '\u5c1a\u672a\u5b8c\u6210'}\uff1b{research_outcome_text(validation.get('research_outcome'))}",
        f"- \u7d50\u6848\u5831\u544a: {validation.get('closeout_report', 'NA')}",
        "",
        "### \u901a\u904e\u9a57\u8b49\u7684\u9650\u5b9a\u6a21\u7d44",
        "",
    ]
    for name, item in policy["validated_submodels"].items():
        validation_lines.append(
            f"- {name}: {item['scope']}\uff1b\u6b77\u53f2\u9a57\u8b49 {item['accuracy']:.2%} / {item['cases']} \u4ef6\u3002"
        )
    validation_lines.append(
        "- \u7bc4\u570d\u9650\u5236: \u958b\u76e4\u7f3a\u53e3\u6210\u679c\u4e0d\u5f97\u5ef6\u4f38\u6210\u7576\u65e5\u6536\u76e4\u6216\u591a\u65e5\u5927\u76e4\u65b9\u5411\u3002"
    )
    lines[8:8] = validation_lines
    for item in freshness["sources"]:
        lag = "NA" if item["lag_days"] is None else str(item["lag_days"])
        lines.append(
            f"| {item['name']} | {item['latest_date'] or 'NA'} | {item['expected_date']} | "
            f"{lag} | {freshness_status_text(item['status'])} | {item['source']} |"
        )
    lines.extend(["", f"- 稽核說明: {freshness['note']}"])
    if premarket["is_premarket"] or premarket.get("is_non_trading_day"):
        lines.extend(render_premarket_lines(premarket))
    lines.extend(render_external_event_reset_lines(payload.get("external_event_reset_monitor", {})))
    lines.extend(render_situation_psychology_context_lines(payload.get("situation_psychology_context", {})))
    lines.extend(render_fundamental_constitution_lines(payload.get("fundamental_constitution", {})))
    lines.extend(render_practical_cause_arbitration_lines(payload.get("practical_cause_arbitration", {})))
    lines.extend(render_market_stethoscope_lines(payload.get("market_stethoscope", {})))
    lines.extend(render_market_protection_layers_lines(payload.get("market_protection_layers", {})))
    lines.extend(render_crisis_opportunity_interface_lines(payload.get("crisis_opportunity_interface", {})))
    lines.extend(render_cross_market_entanglement_lines(payload.get("cross_market_entanglement", {})))
    lines.extend(render_master_arbitration_lines(payload.get("master_arbitration", {})))
    lines.extend(render_monthly_cycle_monitor_lines(payload.get("monthly_cycle_monitor", {})))
    lines.extend(render_intraday_tactical_lines(payload.get("intraday_tactical_monitor", {})))
    lines.extend(render_day_night_variance_pattern_lines(payload.get("day_night_variance_pattern", {})))
    lines.extend(render_close_cause_attribution_lines(payload.get("close_cause_attribution", {})))
    lines.extend(render_sector_pressure_observation_lines(payload.get("sector_pressure_observation", {})))
    lines.extend(render_market_heart_rhythm_lines(payload.get("market_heart_rhythm", {})))
    lines.extend(render_endogenous_regulation_pulse_lines(payload.get("endogenous_regulation_pulse", {})))
    lines.extend(render_human_behavior_market_pattern_lines(payload.get("human_behavior_market_pattern", {})))
    lines.extend(render_psychological_warfare_pattern_lines(payload.get("psychological_warfare_pattern", {})))
    lines.extend(render_programmed_pressure_lines(payload.get("programmed_pressure_pattern", {})))
    lines.extend(render_path_risk_lines(payload.get("path_risk_adjustment", {})))
    lines.extend(render_weather_satellite_forecast_lines(payload.get("weather_satellite_forecast", {})))
    lines.extend(render_self_review_lines(payload["self_review"]))
    lines.extend(render_direction_reliability_policy_lines(payload.get("direction_reliability_policy", {})))
    lines.extend(render_error_review_lines(payload["error_review"]))
    lines.extend(render_integrated_summary_lines(payload["integrated_summary"]))
    lines.extend(render_psychology_state_lines(payload.get("psychology_state", {})))
    lines.extend(render_market_health_lines(payload["market_health"]))
    lines.extend(render_self_repair_lines(payload["self_repair"]))
    lines.extend(render_clinical_knowledge_lines(payload["clinical_knowledge"]))
    lines.extend(render_logic_consistency_lines(payload["logic_consistency"]))
    lines.extend(render_peak_to_valley_warning_lines(payload["peak_to_valley_warning"], payload["peak_to_valley_backtest"]))
    lines.extend(render_technical_phase_lines(payload["technical_phase"]))
    lines.extend(render_route_reference_lines(payload["route_reference"]))
    lines.extend(render_capital_flow_lines(payload["capital_flow"]))
    lines.extend(render_bagua_lifecycle_lines(payload["bagua_lifecycle"]))
    lines.extend(render_tradeable_cycle_lines(payload["tradeable_cycle"]))
    lines.extend(render_candlestick_lines(payload["candlestick_pattern"]))
    lines.extend(render_washout_lines(payload["washout_pattern"]))
    lines.extend(
        [
            "",
            "## 日盤狀態",
            "",
            f"- 今日劇本: {cause['scenario']['name']}",
            f"- 劇本含義: {cause['scenario']['meaning']}",
            f"- 後續觀察: {cause['scenario']['watch']}",
            f"- 真相判斷: {cause['truth_text']} ({cause['truth_label']})",
            f"- 白話結論: {cause['plain_summary']}",
            f"- 外部壓力: {cause['external_pressure_text']} / 分數 {cause['external_score']}",
            f"- 台指期夜盤: {cause['night_futures_text']} / 分數 {cause['night_futures_score']}",
            f"- 盤中型態: {cause['intraday_truth_text']} / 分數 {cause['panic_reversal_score']}",
            f"- 生命周期階段: {stage_text(forecast['lifecycle_stage'])}",
            f"- 風險狀態: {regime_text(forecast['risk_regime'])}",
            "",
            "## 記憶體產業尾部風險",
            "",
            f"- 風險等級: {memory_risk_text(memory['level'])} / {memory['points']} 分",
            f"- 市場價格確認: {'是' if memory['market_confirmed'] else '否'}",
            f"- 風險原因: {memory['reasons']}",
            f"- 記憶體三大廠近1日/5日: {pct(memory['basket_return_1d'])} / {pct(memory['basket_return_5d'])}",
            f"- 當日報價涵蓋: {memory['constituent_count']} / 3 家",
            f"- 相對費半近5日: {pct(memory['relative_sox_5d'])}",
            f"- 人工事件: {memory['event_titles'] or '無已確認事件'}",
            f"- 覆蓋前後風險狀態: {memory['base_risk_regime']} → {memory['adjusted_risk_regime']}",
            "",
            "## 探索性歷史相似分布（非正式方向訊號）",
            "",
            "- 驗證狀態: 未通過多日方向門檻；下表只是歷史相似狀態的結果分布，不可解讀為已驗證機率或交易勝率。",
            f"- 分布基準交易日: {forecast.get('base_trade_date', check['signal_date'])}，此日視為第0個交易日。",
            "- 1日、5日、20日、60日都指「交易日」，不是自然日；遇週末或休市會順延。",
            "",
            "| 期間 | 對照交易日 | 歷史多數方向 | 上漲占比 | 下跌占比 | 盤整占比 | 最大占比 | 相似樣本 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in forecast["forecasts"]:
        lines.append(
            f"| {item['horizon_days']}日 | {item.get('target_trade_date', 'NA')} | {direction_text(item['predicted_direction'])} | "
            f"{item['probability_up']:.2%} | {item['probability_down']:.2%} | "
            f"{item['probability_sideways']:.2%} | {item['confidence']:.2%} | {item['case_count']} |"
        )
    one_day = next((item for item in forecast["forecasts"] if item.get("horizon_days") == 1), {})
    close_range = one_day.get("close_range_forecast", {})
    central = close_range.get("central_50_range", [])
    outer = close_range.get("outer_80_range", [])
    paths = one_day.get("intraday_path_distribution", {})
    if len(central) == 2 and len(outer) == 2:
        lines.extend([
            "",
            "### 次日收盤區間與病程風險（新增）",
            "",
            f"- 中央50%收盤區間: {num(central[0])}～{num(central[1])}",
            f"- 外圍80%收盤區間: {num(outer[0])}～{num(outer[1])}",
            f"- 深殺後收復病例率: {pct(paths.get('deep_selloff_recovered_rate'))}",
            f"- 跌破前低後收復病例率: {pct(paths.get('prior_low_breach_then_recover_rate'))}",
            "- 治理: 僅使用嚴格早於預測日的相似病例；區間與路徑率均為探索性，不改寫正式方向訊號。",
        ])
    lines.extend(
        [
            "",
            "## 探索性方法歷史回放（不等於正式驗證）",
            "",
            "- 此回放含重疊與反覆觀察用途；正式結論以結案報告的purged、非重疊及強基準稽核為準。",
            "",
            "| 期間 | 樣本數 | 歷史核對率 | 高占比核對率 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for item in simulation["by_horizon"]:
        lines.append(f"| {item['horizon_days']}日 | {item['sample_count']} | {item['hit_rate']:.2%} | {item['confident_hit_rate']:.2%} |")
    summary = simulation["summary"]
    lines.extend(
        [
            "",
            "## 解讀",
            "",
            f"- 探索性平均歷史核對率: {summary.get('avg_hit_rate', 0):.2%}",
            f"- 探索性最高期間: {format_best(summary.get('best_horizon'))}",
            "- 正式多日方向結論: 66.18%，未達90%且低於68.42%基準，因此維持停用。",
            "- 若進入盤前模式，日盤劇本需等台股現貨更新後再確認。",
        ]
    )
    lines.extend(glossary_lines())
    return "\n".join(lines) + "\n"


def render_path_risk_lines(assessment: dict) -> list[str]:
    if not assessment:
        return []
    course = assessment.get("prior_session_path", {})
    strength = assessment.get("prior_return_strength", {})
    settlement = assessment.get("settlement_context", {})
    structure = assessment.get("market_structure", {})
    breadth = structure.get("breadth", {})
    basis = structure.get("futures_basis", {})
    lines = [
        "", "## 次日路徑與結構風險（新增）", "",
        f"- 前一日病程: {course.get('label', 'unknown')}；開盤後最大不利 {pct(course.get('adverse_from_open'))}；收復比例 {pct(course.get('recovery_ratio'))}",
        f"- 收盤方向強度: {strength.get('label', 'unknown')}；正式0.5%確認 {'是' if strength.get('formal_material') else '否'}；臨界0.4%確認 {'是' if strength.get('borderline_confirmation') else '否'}",
        f"- 結算位置: {settlement.get('label', 'unknown')}；歷史 {settlement.get('historical_cases', 0)} 件；上漲率 {pct(settlement.get('historical_up_rate'))}；不得直接覆寫方向",
        f"- 市場廣度/集中度: {breadth.get('status', 'missing')}；{breadth.get('reason', '')}",
        f"- 期現貨基差: {basis.get('status', 'missing')}；{basis.get('reason', '')}",
        f"- 方向信心權限: {assessment.get('direction_confidence_authority', 'reduced')}；缺少確認 {', '.join(assessment.get('missing_confirmations', [])) or '無'}",
    ]
    for warning in assessment.get("warnings", []):
        lines.append(f"- 路徑警示: {warning}")
    return lines


def render_programmed_pressure_lines(pattern: dict) -> list[str]:
    if not pattern:
        return []
    lines = [
        "",
        f"## 行為模式偵測：{pattern.get('label', '資料不足')}",
        "",
        f"- 結論: {pattern.get('summary', '')}",
        f"- 壓低分數: {pattern.get('current_score', 'NA')}；{pattern.get('score_range', '')}",
        f"- 近期平均分數: {pattern.get('recent_average_score', 'NA'):.2f}" if isinstance(pattern.get("recent_average_score"), (int, float)) else f"- 近期平均分數: {pattern.get('recent_average_score', 'NA')}",
        f"- 近五日壓低日數: {pattern.get('recent_pressure_days', 0)}；連續壓低日數: {pattern.get('consecutive_pressure_days', 0)}",
        f"- 壓低日期: {', '.join(pattern.get('pressure_dates', [])) or '無'}",
        f"- 預測修正: {pattern.get('prediction_adjustment', 0)}；規則: {pattern.get('rule', '')}",
        f"- 防呆: {pattern.get('guardrail', '')}",
        "",
        "| 日期 | 收盤 | 漲跌 | 夜盤 | 分數 | 型態 | 原因 |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in pattern.get("recent_rows", []):
        lines.append(
            f"| {row.get('date')} | {num(row.get('close'))} | {pct(row.get('return'))} | "
            f"{pct(row.get('night_spread'))} | {row.get('score')} | {row.get('label')} | "
            f"{'、'.join(row.get('reasons', [])) or '無'} |"
        )
    if pattern.get("next_validation"):
        lines.extend(["", "次日驗證:"])
        for item in pattern.get("next_validation", []):
            lines.append(f"- {item}")
    return lines


def render_weather_satellite_forecast_lines(satellite: dict) -> list[str]:
    if not satellite:
        return []
    weather = satellite.get("market_weather", {})
    night_trend = satellite.get("night_trend", {})
    lines = [
        "",
        f"## 氣象衛星式下一步預判：{satellite.get('headline', '資料不足')}",
        "",
        f"- 下一步: {satellite.get('next_step', '')}",
        f"- 框架: {satellite.get('framework', 'NA')}",
        f"- 綜合方向/信心: {satellite.get('bias', 'unknown')} / {satellite.get('confidence', 'unknown')}",
        f"- 信心範圍: {satellite.get('confidence_scope', '')}",
        f"- 預測三層: {'；'.join(satellite.get('forecast_layers', [])) or 'NA'}",
        f"- 外部氣壓: {weather.get('external_pressure', '資料不足')}",
        f"- 夜盤雷達: {weather.get('night_radar', '資料不足')}",
        f"- 日盤地面站: {weather.get('cash_ground_truth', '資料不足')}",
        f"- 波動雲系: {weather.get('volatility_cloud', '資料不足')}",
        f"- 健康值: {weather.get('health_value', '資料不足')}",
        f"- 八卦地形: {weather.get('bagua_terrain', '資料不足')}",
        f"- 外部重置: {weather.get('external_reset', '資料不足')}",
        f"- 原理: {satellite.get('principle', '')}",
        f"- 自主修正: {satellite.get('error_repair_policy', '')}",
        f"- 防呆: {satellite.get('guardrail', '')}",
    ]
    if night_trend:
        lines.extend(
            [
                "",
                f"### 夜盤趨勢：{night_trend.get('label', '資料不足')}",
                f"- 路徑判讀: {night_trend.get('path_label', 'NA')}；{night_trend.get('path_summary', '')}",
                f"- 三基準: 官方 {pct(night_trend.get('official_spread_per'))}；開收 {pct(night_trend.get('open_close_return'))}；前夜比較 {pct(night_trend.get('previous_night_close_return'))}",
                f"- 高低收: 高 {num(night_trend.get('high'))}；低 {num(night_trend.get('low'))}；收 {num(night_trend.get('close'))}",
                f"- 收盤位置: {pct(night_trend.get('close_position'))}；{night_trend.get('path_shape', '')}",
                f"- 日盤驗證: {night_trend.get('validation', '')}",
                f"- 防呆: {night_trend.get('guardrail', '')}",
            ]
        )
    zero_one_tilt = satellite.get("zero_one_tilt", {})
    if zero_one_tilt:
        lines.extend(
            [
                "",
                f"### 0/1偏向仲裁：{zero_one_tilt.get('label', '資料不足')}",
                f"- 分數: {zero_one_tilt.get('score', 'NA')}；分支: {zero_one_tilt.get('branch', 'NA')}",
                f"- 結論: {zero_one_tilt.get('summary', '')}",
                f"- 規則: {zero_one_tilt.get('rule', '')}",
            ]
        )
        if zero_one_tilt.get("evidence"):
            lines.append("- 偏向理由: " + "；".join(zero_one_tilt.get("evidence", [])))
        if zero_one_tilt.get("counter_conditions"):
            lines.extend(["", "#### 反證條件"])
            for item in zero_one_tilt.get("counter_conditions", []):
                lines.append(f"- {item}")
    if satellite.get("route_checks"):
        lines.extend(["", "### 下一步觀測站"])
        for item in satellite["route_checks"]:
            lines.append(f"- {item}")
    if satellite.get("storm_cells"):
        lines.extend(["", "### 風暴雲系"])
        for item in satellite["storm_cells"]:
            lines.append(f"- {item}")
    triggers = satellite.get("latent_disease_triggers", {})
    if triggers:
        lines.extend([
            "",
            "### 潛伏病灶發病觸發因子",
            f"- 規則: {triggers.get('trigger_rule', '')}",
            f"- 防呆: {triggers.get('guardrail', '')}",
        ])
        for item in triggers.get("active_triggers", []):
            lines.append(f"- 已觸發: {item}")
        for item in triggers.get("watch_triggers", []):
            lines.append(f"- 觀察中: {item}")
    root_cause = satellite.get("root_cause_decomposition", {})
    if root_cause:
        lines.extend([
            "",
            f"### {root_cause.get('headline', '病因分解')}",
            f"- 方法: {root_cause.get('method', '')}",
            f"- 防呆: {root_cause.get('guardrail', '')}",
        ])
        for item in root_cause.get("items", []):
            lines.extend([
                "",
                f"- 表層症狀: {item.get('surface_symptom', '資料不足')}",
                f"- 中層機制: {item.get('mechanism', '')}",
                f"- 根因候選: {'；'.join(item.get('root_cause_candidates', [])) or 'NA'}",
                f"- 確認條件: {item.get('confirm_condition', '')}",
                f"- 排除條件: {item.get('exclude_condition', '')}",
            ])
    if satellite.get("confidence_caps"):
        lines.extend(["", "### 信心上限"])
        for item in satellite["confidence_caps"]:
            lines.append(f"- {item}")
    return lines


def render_human_behavior_market_pattern_lines(pattern: dict) -> list[str]:
    if not pattern:
        return []
    lines = [
        "",
        f"## 人類行為模式：{pattern.get('label', '資料不足')}",
        "",
        f"- 群眾狀態: {pattern.get('crowd_state', '資料不足')}",
        f"- 戰術候選: {pattern.get('tactic_candidate', '資料不足')}",
        f"- 行為分數: {pattern.get('score', 'NA')} / -5～5",
        f"- 可控風險: {'是' if pattern.get('controllable_risk') else '否'}",
        f"- 結論: {pattern.get('summary', '')}",
        f"- 次日觀察: {pattern.get('next_watch', '')}",
        f"- 防呆: {pattern.get('guardrail', '人類行為模式只作統計觀察，不證明單一主體操控，也不產生買賣命令。')}",
    ]
    driver = pattern.get("strategy_driver") or {}
    if driver:
        lines.extend([
            f"- 主劇本: {driver.get('direction_driver', 'NA')}",
            f"- 震幅變數: {driver.get('amplitude_driver', 'NA')}",
            f"- 分層原則: {driver.get('principle', '')}",
        ])
    if pattern.get("evidence"):
        lines.extend(["", "### 行為證據"])
        for item in pattern["evidence"]:
            lines.append(f"- {item}")
    return lines


def render_psychological_warfare_pattern_lines(pattern: dict) -> list[str]:
    if not pattern:
        return []
    lines = [
        "",
        f"## 心理戰與八卦兵法：{pattern.get('label', '資料不足')}",
        "",
        f"- 框架: {pattern.get('framework', '')}",
        f"- 姿態: {pattern.get('stance', '資料不足')}",
        f"- 分數: {pattern.get('score', 'NA')} / -8～8",
        f"- 群眾狀態: {pattern.get('crowd_state', '資料不足')}",
        f"- 戰術候選: {pattern.get('tactic_candidate', '資料不足')}",
        f"- 背景卦/狀態卦: {pattern.get('bagua_background', 'NA')} / {pattern.get('bagua_state', 'NA')}",
        f"- 技術段位: {pattern.get('technical_phase', '資料不足')}",
        f"- 文字轉數字處理: {pattern.get('action_policy', '資料不足')}",
        f"- 次日驗證: {pattern.get('next_validation', '')}",
        f"- 防呆: {pattern.get('guardrail', '')}",
    ]
    if pattern.get("sunzi_principles"):
        lines.extend(["", "### 對應兵法"])
        for item in pattern["sunzi_principles"]:
            lines.append(f"- {item}")
    if pattern.get("cause_effect_chain"):
        lines.extend(["", "### 前因後果鏈"])
        for item in pattern["cause_effect_chain"]:
            lines.append(f"- {item}")
    if pattern.get("semantic_quantification"):
        lines.extend(["", "### 文字轉數字"])
        lines.append("| 文字訊號 | 分數 | 模型意義 |")
        lines.append("| --- | ---: | --- |")
        for item in pattern["semantic_quantification"]:
            lines.append(f"| {item.get('text', '')} | {item.get('score', 0)} | {item.get('meaning', '')} |")
    if pattern.get("intent_trace"):
        lines.extend(["", "### 資料意圖判讀"])
        if pattern.get("intent_summary"):
            lines.append(f"- 核心: {pattern.get('intent_summary')}")
        lines.append("| 資料 | 意圖表現 | 兵法對照 | 驗證方式 |")
        lines.append("| --- | --- | --- | --- |")
        for item in pattern["intent_trace"]:
            lines.append(
                f"| {item.get('data', '')} | {item.get('intent', '')} | "
                f"{item.get('sunzi', '')} | {item.get('validation', '')} |"
            )
    if pattern.get("evidence"):
        lines.extend(["", "### 心理戰證據"])
        for item in pattern["evidence"]:
            lines.append(f"- {item}")
    if pattern.get("truth_test"):
        lines.extend(["", "### 真偽驗證"])
        for item in pattern["truth_test"]:
            lines.append(f"- {item}")
    return lines


def render_sector_pressure_observation_lines(pattern: dict) -> list[str]:
    if not pattern:
        return []
    lines = [
        "",
        f"## 族群分化壓力：{pattern.get('label', '資料不足')}",
        "",
        f"- 日期: {pattern.get('date', 'NA')}",
        f"- 分數: {pattern.get('risk_score', 'NA')} / 0～5",
        f"- 指數相對前一正式日: {pct(pattern.get('index_return'))}",
        f"- 弱勢族群: {'、'.join(pattern.get('weak_sectors', [])) or '無'}",
        f"- 強勢族群: {'、'.join(pattern.get('strong_sectors', [])) or '無'}",
        f"- 結論: {pattern.get('summary', '')}",
        f"- 防呆: {pattern.get('guardrail', '族群壓力觀察只作風控留底，不產生買賣命令。')}",
    ]
    observations = pattern.get("observations", [])
    if observations:
        lines.extend([
            "",
            "| 族群 | 狀態 | 嚴重度 | 觀察 | 來源 |",
            "| --- | --- | --- | --- | --- |",
        ])
        for row in observations:
            lines.append(
                f"| {row.get('sector', '')} | {row.get('status', '')} | "
                f"{row.get('severity', '')} | {row.get('note', '')} | {row.get('source', '')} |"
            )
    if pattern.get("cause_candidates"):
        lines.extend(["", "### 原因候選"])
        for item in pattern["cause_candidates"]:
            lines.append(f"- {item}")
    if pattern.get("hidden_cause_candidates"):
        lines.extend(["", "### 隱藏病因候選"])
        for item in pattern["hidden_cause_candidates"]:
            lines.append(f"- {item}")
    if pattern.get("next_validation"):
        lines.extend(["", "### 次日驗證"])
        for item in pattern["next_validation"]:
            lines.append(f"- {item}")
    return lines


def render_close_cause_attribution_lines(pattern: dict) -> list[str]:
    if not pattern:
        return []
    features = pattern.get("features", {})
    lines = [
        "",
        f"## 漲跌原因歸因：{pattern.get('label', '資料不足')}",
        "",
        f"- 框架: {pattern.get('framework', 'NA')}",
        f"- 結論: {pattern.get('headline', '')}",
        f"- 風控修正: {pattern.get('risk_adjustment', 'NA')}",
        f"- 模型效果: {pattern.get('model_effect', '')}",
        f"- 防呆: {pattern.get('guardrail', '漲跌原因歸因只作可驗證病因候選，不產生買賣命令。')}",
        "",
        "| 特徵 | 數值 |",
        "| --- | ---: |",
        f"| 收盤 | {num(features.get('close'))} |",
        f"| 開盤 | {num(features.get('open'))} |",
        f"| 最高 | {num(features.get('high'))} |",
        f"| 最低 | {num(features.get('low'))} |",
        f"| 日盤報酬 | {pct(features.get('cash_return'))} |",
        f"| 開盤缺口 | {pct(features.get('gap_return'))} |",
        f"| 開收到收盤 | {pct(features.get('intraday_return'))} |",
        f"| 高點至收盤 | {pct(features.get('high_to_close'))} |",
        f"| 收盤位置 | {pct(features.get('close_position'))} |",
        f"| 族群壓力分數 | {features.get('sector_risk_score', 'NA')} |",
    ]
    if pattern.get("evidence"):
        lines.extend(["", "### 歸因證據"])
        for item in pattern["evidence"]:
            lines.append(f"- {item}")
    if pattern.get("cause_candidates"):
        lines.extend(["", "### 原因候選"])
        for item in pattern["cause_candidates"]:
            lines.append(f"- {item}")
    if pattern.get("next_validation"):
        lines.extend(["", "### 次日驗證"])
        for item in pattern["next_validation"]:
            lines.append(f"- {item}")
    return lines


def render_endogenous_regulation_pulse_lines(pulse: dict) -> list[str]:
    if not pulse:
        return []
    lines = [
        "",
        f"## 市場內生調節脈動：{pulse.get('label', '資料不足')}",
        "",
        f"- 原理: {pulse.get('principle', '')}",
        f"- 分數: {pulse.get('score', 'NA')}；{pulse.get('score_range', '')}",
        f"- 結論: {pulse.get('summary', '')}",
        f"- 模型效果: {pulse.get('model_effect', '')}",
        f"- 外部安靜條件: {'是' if pulse.get('external_quiet') else '否'}；最大外部波動 {pct(pulse.get('external_abs_max'))}",
        f"- 日內下探: {pct(pulse.get('intraday_low_pct'))}；低點收復 {pct(pulse.get('recovery_from_low'))}",
        f"- 20日線上: {'是' if pulse.get('close_above_ma20') else '否'}；低點墊高: {'是' if pulse.get('low_base_higher') else '否'}",
        f"- 壓低分數: {pulse.get('pressure_score', 'NA')}；族群壓力分數: {pulse.get('sector_pressure_score', 'NA')}",
        f"- 防呆: {pulse.get('guardrail', '只作統計觀察，不產生買賣命令。')}",
    ]
    if pulse.get("evidence"):
        lines.extend(["", "### 調節證據"])
        for item in pulse["evidence"]:
            lines.append(f"- {item}")
    if pulse.get("next_validation"):
        lines.extend(["", "### 次日驗證"])
        for item in pulse["next_validation"]:
            lines.append(f"- {item}")
    if pulse.get("missing_or_limited_data"):
        lines.extend(["", "### 資料限制"])
        for item in pulse["missing_or_limited_data"]:
            lines.append(f"- {item}")
    return lines


def render_market_heart_rhythm_lines(heart: dict) -> list[str]:
    if not heart:
        return []
    lines = [
        "",
        f"## 市場心律監測：{heart.get('label', '資料不足')}",
        "",
        f"- 框架: {heart.get('framework', 'market_heart_rhythm_v1')}",
        f"- 資料日: {heart.get('signal_date', 'NA')}",
        f"- 心律分數: {heart.get('score', 'NA')}/100",
        f"- 臨床分型: {heart.get('clinical_label', 'NA')}；{heart.get('clinical_meaning', '')}",
        f"- 趨勢偏向: {heart.get('direction_bias', 'NA')}",
        f"- 近{heart.get('recent_days', 'NA')}日正負: 上漲 {heart.get('up_days', 'NA')} / 下跌 {heart.get('down_days', 'NA')}",
        f"- 正負切換率: {pct(heart.get('switch_rate'))}；60日切換率 {pct(heart.get('medium_switch_rate'))}",
        f"- 平均日振幅: {pct(heart.get('avg_abs_return'))}；平均高低振幅 {pct(heart.get('avg_range'))}",
        f"- 最大連漲/連跌: {heart.get('max_up_streak', 'NA')} / {heart.get('max_down_streak', 'NA')}",
        f"- 跌後修復率: {pct(heart.get('down_repair_rate'))}；近20日累積 {pct(heart.get('cumulative_return'))}",
        f"- 結論: {heart.get('summary', '')}",
        f"- 規則: {heart.get('rule', '')}",
        f"- 防呆: {heart.get('guardrail', '')}",
    ]
    if heart.get("reasons"):
        lines.extend(["", "### 心律判斷原因"])
        for item in heart.get("reasons", []):
            lines.append(f"- {item}")
    return lines


def render_day_night_variance_pattern_lines(pattern: dict) -> list[str]:
    if not pattern:
        return []
    features = pattern.get("features", {})
    lines = [
        "",
        f"## 日夜盤變異分析：{pattern.get('label', '資料不足')}",
        "",
        f"- 關係: {pattern.get('relation_label', '資料不足')}（{pattern.get('relation_code', 'unknown')}）",
        f"- 變異幅度: {pattern.get('variance_label', '資料不足')}（{pattern.get('variance_level', 'unknown')}）",
        f"- 日盤驗證狀態: {pattern.get('day_validation_status', 'unknown')}",
        f"- 分數: {pattern.get('score', 'NA')} / -5～5",
        f"- 結論: {pattern.get('summary', '')}",
        f"- 次日觀察: {pattern.get('next_watch', '')}",
        f"- 夜盤/日盤: {pct(features.get('night_return'))} / {pct(features.get('cash_return'))}",
        f"- 開盤缺口/日內高低差/收盤位置: {pct(features.get('gap_return'))} / {pct(features.get('cash_range_pct'))} / {pct(features.get('cash_close_position'))}",
        f"- 防呆: {pattern.get('guardrail', '日夜盤變異只作統計觀察，不產生買賣命令。')}",
    ]
    if pattern.get("cause_candidates"):
        lines.extend(["", "### 高低差原因候選"])
        for item in pattern["cause_candidates"]:
            lines.append(f"- {item}")
    if pattern.get("hidden_cause_candidates"):
        lines.extend(["", "### 隱藏病因候選"])
        for item in pattern["hidden_cause_candidates"]:
            lines.append(f"- {item}")
    if pattern.get("evidence"):
        lines.extend(["", "### 變異證據"])
        for item in pattern["evidence"]:
            lines.append(f"- {item}")
    return lines


def render_programmed_pressure_pattern(pattern: dict) -> str:
    lines = [
        "# 連續壓低換手行為指紋研究",
        "",
        f"- 期間: {pattern.get('start_date', 'NA')} 至 {pattern.get('forecast_date', 'NA')}",
        f"- 狀態: {pattern.get('label', '資料不足')}",
        f"- 摘要: {pattern.get('summary', '')}",
    ]
    lines.extend(render_programmed_pressure_lines(pattern))
    return "\n".join(lines) + "\n"


def render_external_event_reset_lines(reset: dict) -> list[str]:
    if not reset:
        return []
    lines = [
        "",
        f"## 外部事件重置監控：{reset.get('label', '資料不足')}",
        "",
        f"- 結論: {reset.get('summary', '')}",
        f"- 是否重置: {'是' if reset.get('reset_active') else '否'}；方向: {reset.get('direction', 'none')}；分數: {reset.get('reset_score', 0)}",
        f"- 因果優先權: {reset.get('causal_priority', 'internal_primary')}",
        f"- 風控修正: {reset.get('risk_adjustment', 0)}",
        f"- 觀察: {reset.get('watch', '')}",
        f"- 突發新聞來源: {reset.get('news_feed_status', 'unknown')}；{reset.get('news_rule', '')}",
        f"- 防呆: {reset.get('guardrail', '')}",
    ]
    if reset.get("reasons"):
        lines.append(f"- 觸發原因: {'、'.join(reset.get('reasons', []))}")
    news = reset.get("news_risk") or {}
    if news:
        lines.extend(
            [
                "",
                "### 國際重大消息風險",
                f"- 日期: {news.get('date') or 'NA'}；狀態: {news.get('status', 'unknown')}",
                f"- 新聞風險分數: {news.get('risk_score', 0)}；利多分數: {news.get('tailwind_score', 0)}；淨風險: {news.get('net_risk_score', 0)}",
                f"- 摘要: {news.get('summary', '')}",
            ]
        )
        for item in news.get("events", [])[:6]:
            lines.append(
                f"- 事件: {item.get('title', item.get('topic', 'NA'))}｜影響 {item.get('risk_impact', item.get('impact', 'NA'))}"
            )
        if news.get("sources"):
            lines.append(f"- 來源: {'；'.join(str(src) for src in news.get('sources', [])[:6])}")
    signals = reset.get("signals") or {}
    if signals:
        lines.extend(["", "| 外部項目 | 變動 |", "| --- | ---: |"])
        for name, value in signals.items():
            lines.append(f"| {name} | {pct(value)} |")
    return lines


def render_situation_psychology_context_lines(context: dict) -> list[str]:
    if not context:
        return []
    lines = [
        "",
        f"## 新聞面與生活心理：{context.get('label', '資料不足')}",
        "",
        f"- 框架: {context.get('framework', 'NA')}",
        f"- 分數: {context.get('score', 'NA')} / -6～6",
        f"- 姿態: {context.get('posture', '')}",
        f"- 新聞來源狀態: {context.get('news_status', 'unknown')}；待公布/待驗證事件 {context.get('pending_event_count', 0)} 項",
        f"- 觸發因子: {'；'.join(context.get('triggers', [])) or '無明確情境觸發'}",
        f"- 防呆: {context.get('guardrail', '情境心理只作濾鏡，不產生買賣命令。')}",
    ]
    if context.get("evidence"):
        lines.extend(["", "### 情境證據"])
        for item in context.get("evidence", []):
            lines.append(f"- {item}")
    if context.get("next_validation"):
        lines.extend(["", "### 驗證條件"])
        for item in context.get("next_validation", []):
            lines.append(f"- {item}")
    return lines


def render_fundamental_constitution_lines(constitution: dict) -> list[str]:
    if not constitution:
        return []
    lines = [
        "",
        f"## 基本面命格：{constitution.get('label', '資料不足')}",
        "",
        f"- 框架: {constitution.get('framework', 'NA')}",
        f"- 分數/可信度: {constitution.get('score', 'NA')}/100 / {constitution.get('confidence', 'NA')}/100",
        f"- 結論: {constitution.get('summary', '')}",
        f"- 解讀: {constitution.get('interpretation', '')}",
        f"- 防呆: {constitution.get('guardrail', '')}",
    ]
    if constitution.get("drivers"):
        lines.extend(["", "### 支撐因素"])
        for item in constitution.get("drivers", []):
            lines.append(f"- {item}")
    if constitution.get("pressures"):
        lines.extend(["", "### 壓力因素"])
        for item in constitution.get("pressures", []):
            lines.append(f"- {item}")
    if constitution.get("data_gaps"):
        lines.extend(["", "### 資料缺口"])
        for item in constitution.get("data_gaps", []):
            lines.append(f"- {item}")
    return lines


def render_practical_cause_arbitration_lines(practical: dict) -> list[str]:
    if not practical:
        return []
    lines = [
        "",
        f"## 實務主因仲裁：{practical.get('label', '資料不足')}",
        "",
        f"- 框架: {practical.get('framework', 'NA')}",
        f"- 主控判斷: {practical.get('practical_primary', 'NA')}",
        f"- 內部病灶基因分數: {practical.get('internal_structure_score', 'NA')}",
        f"- 外部觸發條件分數: {practical.get('external_trigger_score', 'NA')}",
        f"- 共振嚴重度: {practical.get('combined_severity', 'NA')}；治療階段 第{practical.get('treatment_phase', 'NA')}期 / {practical.get('treatment_stage', 'NA')}",
        f"- 治療原則: {practical.get('treatment_policy', '')}",
        f"- 藥性/藥效: {practical.get('medicine_type', 'NA')}；{practical.get('medicine_effect', '')}",
        f"- 對症/陰霾: {practical.get('correct_medicine', 'NA')} / {practical.get('after_effect_risk', 'NA')}",
        f"- 即效藥/事實改變因: {practical.get('fact_changing_medicine', {}).get('label', 'NA')}；偏向 {practical.get('immediate_medicine_bias', 'NA')}；淨分 {practical.get('immediate_medicine_score', 'NA')}",
        f"- 因果程序: {practical.get('causality_rule', '')}",
        f"- 結論: {practical.get('summary', '')}",
        f"- 病灶基因模型: {practical.get('gene_trigger_model', '')}",
        f"- 0/1規則: {practical.get('one_zero_rule', '')}",
        f"- 模型效果: {practical.get('model_effect', '')}",
        f"- 防呆: {practical.get('guardrail', '')}",
    ]
    if practical.get("internal_causes"):
        lines.extend(["", "### 病灶基因"])
        for item in practical.get("internal_causes", []):
            lines.append(f"- {item}")
    if practical.get("external_triggers"):
        lines.extend(["", "### 今日觸發條件"])
        for item in practical.get("external_triggers", []):
            lines.append(f"- {item}")
    if practical.get("fact_changing_medicines"):
        lines.extend(["", "### 即效藥：一發生就改變市場事實的因子"])
        for item in practical.get("fact_changing_medicines", []):
            lines.append(f"- {item}")
        lines.append(f"- 規則: {practical.get('fact_changing_medicine', {}).get('rule', '')}")
    if practical.get("causality_pipeline"):
        lines.extend(["", "### 因果程序"])
        for step in practical.get("causality_pipeline", []):
            evidence = "；".join(step.get("evidence", [])[:3])
            lines.append(f"- {step.get('stage')}: {step.get('meaning')} {evidence}")
    if practical.get("confirmation"):
        lines.extend(["", "### 發作確認條件"])
        for item in practical.get("confirmation", []):
            lines.append(f"- {item}")
    if practical.get("exclusion"):
        lines.extend(["", "### 排除條件"])
        for item in practical.get("exclusion", []):
            lines.append(f"- {item}")
    return lines


def render_market_stethoscope_lines(stethoscope: dict) -> list[str]:
    if not stethoscope:
        return []
    lines = [
        "",
        f"## 市場聽診器：{stethoscope.get('label', '資料不足')}",
        "",
        f"- 框架: {stethoscope.get('framework', 'market_stethoscope_v1')}",
        f"- 資料日: {stethoscope.get('signal_date', 'NA')}",
        f"- 總分: {stethoscope.get('score', 'NA')}",
        f"- 結果偏向: {stethoscope.get('result_bias', 'NA')}",
        f"- 結論: {stethoscope.get('summary', '')}",
        f"- 規則: {stethoscope.get('rule', '')}",
        f"- 防呆: {stethoscope.get('guardrail', '')}",
    ]
    if stethoscope.get("diagnostics"):
        lines.extend(["", "### 生命徵象"])
        for item in stethoscope.get("diagnostics", []):
            lines.append(
                f"- {item.get('name')}: {item.get('label')}；分數 {item.get('score')}；{item.get('summary')}"
            )
    if stethoscope.get("missing"):
        lines.extend(["", "### 待補資料"])
        for item in stethoscope.get("missing", []):
            lines.append(f"- {item}")
    return lines


def render_market_protection_layers_lines(protection: dict) -> list[str]:
    if not protection:
        return []
    lines = [
        "",
        f"## 市場防呆保護層：{protection.get('label', '資料不足')}",
        "",
        f"- 框架: {protection.get('framework', 'market_protection_layers_v1')}",
        f"- 總分: {protection.get('score', 'NA')}",
        f"- 風控姿態: {protection.get('posture', 'NA')}",
        f"- 目前判斷: {protection.get('current_judgment', protection.get('summary', ''))}",
        f"- 結論: {protection.get('summary', '')}",
        f"- 啟動層: {'；'.join(protection.get('active_layers', [])) or '無'}",
        f"- 偏弱層: {'；'.join(protection.get('warning_layers', [])) or '無'}",
        f"- 失效層: {'；'.join(protection.get('failed_layers', [])) or '無'}",
        f"- 期限規則: {protection.get('duration_rule', '')}",
        f"- 防呆: {protection.get('guardrail', '')}",
    ]
    if protection.get("layers"):
        lines.extend(
            [
                "",
                "| 保護層 | 狀態 | 分數 | 證據 | 失效條件 |",
                "| --- | --- | ---: | --- | --- |",
            ]
        )
        for layer in protection.get("layers", []):
            lines.append(
                f"| {layer.get('name')} | {layer.get('label')} | {layer.get('score')} | "
                f"{layer.get('evidence')} | {layer.get('failure_condition')} |"
            )
    if protection.get("crisis_conditions"):
        lines.extend(["", "### 危機升級條件"])
        for item in protection.get("crisis_conditions", []):
            lines.append(f"- {item}")
    return lines


def render_crisis_opportunity_interface_lines(interface: dict) -> list[str]:
    if not interface:
        return []
    lines = [
        "",
        f"## 危機與轉機界面：{interface.get('label', '資料不足')}",
        "",
        f"- 框架: {interface.get('framework', 'crisis_opportunity_interface_v1')}",
        f"- 界面狀態: {interface.get('interface_state', 'NA')}",
        f"- 結論: {interface.get('summary', '')}",
        f"- 0/1規則: {interface.get('zero_one_rule', '')}",
        f"- 防呆: {interface.get('guardrail', '')}",
    ]
    for title, key in [
        ("危機邊界", "crisis_edge"),
        ("轉機邊界", "opportunity_edge"),
        ("峰谷信號", "peak_valley_signals"),
        ("觀測點", "watch_points"),
        ("我能做什麼", "what_i_can_do"),
    ]:
        values = interface.get(key, [])
        if values:
            lines.extend(["", f"### {title}"])
            for item in values:
                lines.append(f"- {item}")
    return lines


def render_cross_market_entanglement_lines(cross_market: dict) -> list[str]:
    if not cross_market:
        return []
    directions = cross_market.get("directions", {})
    lines = [
        "",
        f"## 跨盤糾結：{cross_market.get('label', '資料不足')}",
        "",
        f"- 框架: {cross_market.get('framework', 'cross_market_entanglement_v1')}",
        f"- 分數: {cross_market.get('score', 'NA')}",
        f"- 結論: {cross_market.get('summary', '')}",
        f"- 方向鏈: Nasdaq {directions.get('nasdaq', 'NA')} / SOX {directions.get('sox', 'NA')} / TSM ADR {directions.get('tsm_adr', 'NA')} / 台指夜盤 {directions.get('tx_night', 'NA')} / 台股日盤 {directions.get('twii_cash', 'NA')}",
        f"- 因果連結: {cross_market.get('model_link', '')}",
        f"- 規則: {cross_market.get('rule', '')}",
        f"- 防呆: {cross_market.get('guardrail', '')}",
    ]
    if cross_market.get("evidence"):
        lines.extend(["", "### 跨盤證據"])
        for item in cross_market.get("evidence", []):
            lines.append(f"- {item}")
    if cross_market.get("missing"):
        lines.extend(["", "### 待補資料"])
        for item in cross_market.get("missing", []):
            lines.append(f"- {item}")
    return lines


def render_master_arbitration_lines(arbitration: dict) -> list[str]:
    if not arbitration:
        return []
    lines = [
        "",
        f"## 總仲裁：{arbitration.get('headline', '資料不足')}",
        "",
        f"- 框架: {arbitration.get('framework', 'NA')}",
        f"- 主控層: {arbitration.get('dominant_layer', 'NA')}",
        f"- 風險姿態: {arbitration.get('risk_posture', 'NA')}",
        f"- 決策語句: {arbitration.get('decision', '')}",
        f"- 健康/風險/命格: {arbitration.get('health_score', 'NA')} / {arbitration.get('risk_score', 'NA')} / {arbitration.get('constitution_score', 'NA')}",
        f"- 外部重置: {arbitration.get('external_reset_code', 'NA')}；族群壓力 {arbitration.get('sector_risk_score', 'NA')}；心理戰分數 {arbitration.get('psychological_score', 'NA')}",
        f"- 防呆: {arbitration.get('guardrail', '')}",
    ]
    if arbitration.get("conflicts"):
        lines.extend(["", "### 分層衝突"])
        for item in arbitration.get("conflicts", []):
            lines.append(f"- {item}")
    if arbitration.get("validation"):
        lines.extend(["", "### 後續驗證"])
        for item in arbitration.get("validation", []):
            lines.append(f"- {item}")
    return lines


def render_monthly_cycle_monitor_lines(cycle: dict) -> list[str]:
    if not cycle:
        return []
    lines = [
        "",
        f"## 月內/季度週期階段監控：{cycle.get('stage', '資料不足')}",
        "",
        f"- 結論: {cycle.get('summary', '')}",
        f"- 日期: {cycle.get('date', 'NA')}；月份 {cycle.get('month', 'NA')} 第 {cycle.get('week_of_month', 'NA')} 週 / 第 {cycle.get('month_trade_day', 'NA')} 個交易日",
        f"- 月內高低: {num(cycle.get('month_low'))}～{num(cycle.get('month_high'))}；目前月內位置 {pct(cycle.get('month_position'))}",
        f"- 月報酬: {pct(cycle.get('month_return'))}；近5日: {pct(cycle.get('recent5_return'))}",
        f"- 季度背景: {cycle.get('quarter_stage', '資料不足')}；{cycle.get('quarter', 'NA')} {pct(cycle.get('quarter_return'))}",
        f"- 壓低分數: {cycle.get('pressure_score', 0)}；外部重置: {cycle.get('external_reset_code', 'none')}",
        f"- 週期順序: {cycle.get('cycle_order', '')}",
        f"- 監控動作: {cycle.get('action', '')}",
        f"- 防呆: {cycle.get('guardrail', '')}",
    ]
    if cycle.get("next_validation"):
        lines.extend(["", "後續驗證:"])
        for item in cycle.get("next_validation", []):
            lines.append(f"- {item}")
    return lines


def render_premarket_lines(premarket: dict) -> list[str]:
    is_non_trading_day = bool(premarket.get("is_non_trading_day"))
    title = "非交易日觀察" if is_non_trading_day else "盤前模式"
    pressure_label = "下個交易日情境" if is_non_trading_day else "盤前壓力"
    scenario_label = "情境劇本" if is_non_trading_day else "盤前劇本"
    summary_label = "觀察白話" if is_non_trading_day else "盤前白話"
    watch_label = "下個交易日觀察" if is_non_trading_day else "盤前觀察"
    mode_text = (
        f"報告日 {premarket['forecast_date']} 為非交易日，以台股最後有效交易日 {premarket['spot_latest_date']} 為基準，觀察下個交易日風險。"
        if is_non_trading_day
        else f"以台股最後有效交易日 {premarket['spot_latest_date']} 為基準，和 {premarket['forecast_date']} 可取得的最新外部市場及台指期夜盤資料比對分析。"
    )
    lines = [
        "",
        f"## {title}",
        "",
        f"- 模式說明: {mode_text}",
        f"- {pressure_label}: {premarket['pressure']} / 總分 {premarket['total_score']}",
        f"- {scenario_label}: {premarket['scenario']}",
        f"- {summary_label}: {premarket['summary']}",
        f"- {watch_label}: {premarket['watch']}",
        f"- 美股: Nasdaq(美國科技股指標) {pct(premarket['nasdaq_return_1d'])}, 費半(美國半導體指數) {pct(premarket['sox_return_1d'])}, S&P500(美國大盤) {pct(premarket['sp500_return_1d'])}, TSM ADR(台積電美國掛牌股) {pct(premarket['tsm_adr_return_1d'])}, VIX(恐慌指數) {pct(premarket['vix_return_1d'])}",
        f"- 台指期夜盤({'最近有效夜盤' if is_non_trading_day else '台股開盤前的期貨交易'}): 日期 {premarket.get('night_date') or 'NA'}, 收盤 {num(premarket['tx_night_close'])}, 漲跌 {pct(premarket['tx_night_spread_per'])}, 開收 {pct(premarket['tx_night_return'])}",
        f"- 美債殖利率指標(完成日線、描述用): 5年 {num(premarket.get('treasury_5y_close'))} ({pct(premarket.get('treasury_5y_return_1d'))}), 10年 {num(premarket.get('treasury_10y_close'))} ({pct(premarket.get('treasury_10y_return_1d'))}), 30年 {num(premarket.get('treasury_30y_close'))} ({pct(premarket.get('treasury_30y_return_1d'))})；尚未取得逐時前瞻驗證，不進正式方向分數。",
        f"- 關鍵防守: {format_levels(premarket['defense_levels']) or 'NA'}",
        f"- 轉強觀察: {format_levels(premarket['resistance_levels']) or 'NA'}",
    ]
    external_context = premarket.get("external_context") or {}
    if external_context:
        lines.insert(
            8,
            f"- 外部結構: {external_context.get('label', 'NA')}；{external_context.get('summary', '')} 分歧幅度 {pct(external_context.get('dispersion'))}。",
        )
    path = premarket.get("night_path") or {}
    basis = premarket.get("night_basis_audit") or {}
    if basis.get("available"):
        lines.extend([
            "",
            "### 台指期夜盤三基準防呆",
            "",
            f"- 官方漲跌: {pct(basis.get('official_spread_per'))}；{basis.get('official_basis', '')}",
            f"- 開收報酬: {pct(basis.get('open_close_return'))}；{basis.get('open_close_basis', '')}",
            f"- 相對前夜收盤: {pct(basis.get('vs_previous_night_close_return'))}；前夜收盤 {num(basis.get('previous_night_close'))}",
            f"- 基準衝突: {'是' if basis.get('basis_conflict') else '否'}；{basis.get('summary', '')}",
            f"- 防呆: {basis.get('guardrail', '')}",
        ])
    if path.get("available"):
        lines.extend([
            f"- 夜盤路徑: {path.get('label')}；開高低收 {num(path.get('open'))} / {num(path.get('high'))} / {num(path.get('low'))} / {num(path.get('close'))}。",
            f"- 夜盤品質: 振幅 {pct(path.get('range'))}、收盤位置 {pct(path.get('close_position'))}、低點後收回 {pct(path.get('recovery_from_low'))}、相對前一現貨收盤 {pct(path.get('close_vs_prior_cash'))}。",
            f"- 夜盤解讀: {path.get('summary')} {path.get('guardrail')}",
        ])
    psychology = premarket.get("night_psychology") or {}
    if psychology.get("available"):
        lines.extend([
            "",
            "### 夜盤收盤心理層",
            "",
            f"- 框架: {psychology.get('framework', '')}",
            f"- 心理結論: {psychology.get('label', '資料不足')}；分數 {psychology.get('score', 'NA')} / -5～5",
            f"- 主導情緒: {psychology.get('dominant_emotion', 'NA')}",
            f"- 戰術候選: {psychology.get('tactic', 'NA')}",
            f"- 總結: {psychology.get('summary', '')}",
            f"- 防呆: {psychology.get('guardrail', '')}",
        ])
        if psychology.get("evidence"):
            lines.extend(["", "#### 心理證據"])
            for item in psychology.get("evidence", []):
                lines.append(f"- {item}")
        if psychology.get("next_validation"):
            lines.extend(["", "#### 日盤驗證條件"])
            for item in psychology.get("next_validation", []):
                lines.append(f"- {item}")
    gap = premarket.get("validated_night_gap") or {}
    if gap:
        direction = "向上跳空" if gap.get("prediction_value") == 1 else "向下跳空"
        status = "有效訊號" if gap.get("actionable") else "未達出手門檻"
        lines.extend([
            "",
            "### 已驗證的開盤缺口子模型",
            "",
            f"- 今日判斷: {direction}；{status}。",
            f"- 夜盤價差幅度 / 今日門檻: {gap.get('confidence_score', 0):.2%} / {gap.get('threshold', 0):.2%}。",
            "- 歷史嚴格樣本外: 335/344 = 97.38%，覆蓋24.82%，Wilson 95%下限95.10%。",
            "- 限定用途: 只預測現貨開盤相對前一日收盤的方向，不代表當日收盤或中長期趨勢。",
        ])
    amplitude = premarket.get("validated_gap_amplitude") or {}
    if amplitude:
        risk = "大幅跳空風險" if amplitude.get("prediction_value") == 1 else "一般跳空幅度"
        status = "有效訊號" if amplitude.get("actionable") else "信心未達門檻"
        lines.extend([
            "",
            "### 已驗證的開盤幅度風險子模型",
            "",
            f"- 今日幅度判斷: {risk}；{status}。",
            f"- 大幅跳空定義: 絕對缺口至少 {amplitude.get('event_threshold', 0):.2%}。",
            f"- 模型信心 / 今日門檻: {amplitude.get('confidence_score', 0):.4f} / {amplitude.get('confidence_threshold', 0):.4f}。",
            "- 歷史嚴格樣本外: 240/259 = 92.66%，覆蓋34.26%，Wilson 95%下限88.83%；顯著優於單純絕對夜盤價差基準（p=.00781）。",
            "- 限定用途: 只判斷開盤缺口幅度風險；方向仍由夜盤價差子模型獨立判斷。",
        ])
    return lines


def render_intraday_tactical_lines(intraday: dict) -> list[str]:
    if not intraday:
        return []
    lines = [
        "",
        f"## 盤中戰術雷達：{intraday.get('label', '資料不足')}",
        "",
        f"- 雷達結論: {intraday.get('summary', '')}",
        f"- 風控動作: {intraday.get('action', '')}",
        f"- 資料狀態: {'已納入今日盤中快照' if intraday.get('live_usable') else '沒有今日盤中快照'}；來源 {intraday.get('source', 'none')}；{intraday.get('source_message', '')}",
        f"- 正式日線基準: {intraday.get('official_signal_date', 'NA')}；夜盤訊號日: {intraday.get('night_signal_date') or 'NA'}",
        f"- 夜盤: {pct(intraday.get('night_spread_per'))}，收 {num(intraday.get('night_close'))}，低 {num(intraday.get('night_low'))}",
        f"- 防守觀察: {format_levels(intraday.get('defense_levels', [])) or 'NA'}",
        f"- 轉強觀察: {format_levels(intraday.get('reclaim_levels', [])) or 'NA'}",
    ]
    if intraday.get("live_usable"):
        lines.extend(
            [
                f"- 盤中價/開高低: {num(intraday.get('live_price'))} / {num(intraday.get('live_open'))} / {num(intraday.get('live_high'))} / {num(intraday.get('live_low'))}",
                f"- 相對昨收/開盤: {pct(intraday.get('vs_prev_close'))} / {pct(intraday.get('vs_open'))}",
                f"- 日內位置: {pct(intraday.get('close_position'))}",
                f"- 夜低刺破/夜收收回: {'是' if intraday.get('broke_night_low') else '否'} / {'是' if intraday.get('reclaimed_night_close') else '否'}",
                "",
                "| 檢查 | 答案 | 檢核通過 | 數值 | 意義 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for item in intraday.get("checks", []):
            answer = item.get("answer")
            if answer is None:
                answer = "是" if item.get("passed") else "否"
            lines.append(
                f"| {item.get('name')} | {answer} | {'是' if item.get('passed') else '否'} | "
                f"{item.get('value', 'NA')} | {item.get('meaning', '')} |"
            )
    return lines


def render_self_review_lines(review: dict) -> list[str]:
    lines = [
        "",
        "## 自主檢討與修正迴路",
        "",
        f"- 檢討摘要: {review.get('plain_summary', '')}",
        f"- 前次紀錄日期: {review.get('previous_forecast_date') or '尚無'}",
        f"- 探索性分布差異: {review.get('prediction_drift', {}).get('summary', '')}",
        f"- 週期追蹤: {review.get('daily_tracking', {}).get('summary', '')}",
        f"- 到期探索性核對: {review.get('matured_checks', {}).get('summary', '')}",
        f"- 修正模式: {review.get('correction_suggestion', {}).get('mode', '')}",
    ]
    for action in review.get("correction_suggestion", {}).get("actions", []):
        lines.append(f"- 修正建議: {action}")

    drift_rows = review.get("prediction_drift", {}).get("rows", [])
    if drift_rows:
        lines.extend(
            [
                "",
                "| 期間 | 前次歷史多數 | 本次歷史多數 | 是否變化 | 最大占比變化 |",
                "| --- | --- | --- | --- | ---: |",
            ]
        )
        for row in drift_rows:
            lines.append(
                f"| {row['horizon_days']}日 | {direction_text(row['previous_direction'])} | "
                f"{direction_text(row['current_direction'])} | {'是' if row['direction_changed'] else '否'} | "
                f"{row['confidence_delta']:.2%} |"
            )

    tracking_rows = review.get("daily_tracking", {}).get("rows", [])
    if tracking_rows:
        lines.extend(
            [
                "",
                "| 追蹤項目 | 前次 | 本次 | 是否變化 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in tracking_rows:
            lines.append(
                f"| {row['name']} | {row['previous']} | {row['current']} | "
                f"{'是' if row['changed'] else '否'} |"
            )
    for alert in review.get("daily_tracking", {}).get("alerts", []):
        lines.append(f"- 追蹤警示: {alert}")

    matured_rows = review.get("matured_checks", {}).get("rows", [])
    if matured_rows:
        lines.extend(
            [
                "",
                "| 核對日 | 期間 | 原歷史多數 | 實際 | 實際漲跌 | 相符 |",
                "| --- | ---: | --- | --- | ---: | --- |",
            ]
        )
        for row in matured_rows[-8:]:
            lines.append(
                f"| {row['target_date']} | {row['horizon_days']} | "
                f"{direction_text(row['predicted_direction'])} | {direction_text(row['actual_direction'])} | "
                f"{row['actual_return']:.2%} | {'是' if row['is_hit'] else '否'} |"
            )
    return lines


def render_error_review_lines(review: dict) -> list[str]:
    lines = [
        "",
        "## 預測誤差閉迴路",
        "",
        f"- 摘要: {review.get('summary', '')}",
        f"- 到期核對筆數: {review.get('total_checked_recent', 0)}",
        f"- 失準筆數: {review.get('miss_count', 0)}",
        f"- 近期核對率: {pct(review.get('hit_rate'))}",
        "- 治理: 只建立候選假設；不自動調參、不改門檻、不改歷史紀錄。",
        "- 留底: `reports/error_review.md` / `reports/error_review.json`",
    ]
    lines.extend(render_next_day_validation_lines(review.get("next_day_validation", {}), heading="### 次日強制驗證"))
    groups = review.get("groups", [])
    if groups:
        lines.extend(["", "| 失準類別 | 筆數 |", "| --- | ---: |"])
        for group in groups:
            lines.append(f"| {group.get('label')} | {group.get('count')} |")
    return lines


def render_direction_reliability_policy_lines(policy: dict) -> list[str]:
    if not policy:
        return []
    lines = [
        "",
        "## 模型可靠度與權重政策",
        "",
        f"- 狀態: {policy.get('headline', '資料不足')}",
        f"- 結論: {policy.get('summary', '')}",
        f"- 整體命中率: {pct(policy.get('overall_hit_rate'))}",
        f"- 最近10筆: {pct(policy.get('recent_10_hit_rate'))}",
        f"- 最近30筆: {pct(policy.get('recent_30_hit_rate'))}",
        f"- 近期1日預測後段: {pct(policy.get('one_day_late_hit_rate'))}",
        f"- 權重: 方向={policy.get('direction_weight', 'low')}；風控={policy.get('risk_weight', 'high')}；已驗證子模組={policy.get('validated_submodel_weight', 'high')}",
        f"- 正式多日方向: {'啟用' if policy.get('production_direction_enabled') else '停用'}",
        f"- 留底: `{policy.get('report', 'reports/model_reliability_trend_audit.md')}`",
        f"- 防呆: {policy.get('guardrail', '')}",
        "",
        "### 權重規則",
    ]
    lines.extend([f"- {item}" for item in policy.get("rules", [])] or ["- 方向預測維持研究性觀察。"])
    return lines


def render_integrated_summary_lines(summary: dict) -> list[str]:
    lines = [
        "",
        "## 綜合總結報告",
        "",
        "- 定位: 描述目前風險與確認條件，不是已驗證的大盤漲跌訊號。",
        f"- 綜合偏向: {summary.get('bias_text', '資料不足')}",
        f"- 綜合分數: {summary.get('score', 0)} 分",
        f"- 信心等級: {summary.get('confidence', '資料不足')}",
        f"- 總結白話: {summary.get('plain_summary', '')}",
    ]
    if summary.get("bullish_evidence"):
        lines.extend(["", "### 支持上漲或修復的因素"])
        for item in summary["bullish_evidence"]:
            lines.append(f"- {item}")
    if summary.get("bearish_evidence"):
        lines.extend(["", "### 支持下跌或防守的因素"])
        for item in summary["bearish_evidence"]:
            lines.append(f"- {item}")
    if summary.get("neutral_evidence"):
        lines.extend(["", "### 需要等待確認的因素"])
        for item in summary["neutral_evidence"]:
            lines.append(f"- {item}")
    if summary.get("missing_or_limited_data"):
        lines.extend(["", "### 資料限制"])
        for item in summary["missing_or_limited_data"]:
            lines.append(f"- {item}")
    if summary.get("confirmation"):
        lines.extend(["", "### 後續確認條件"])
        for item in summary["confirmation"]:
            lines.append(f"- {item}")
    if summary.get("invalidation"):
        lines.extend(["", "### 失效或翻案條件"])
        for item in summary["invalidation"]:
            lines.append(f"- {item}")
    return lines


def render_psychology_state_lines(psychology: dict) -> list[str]:
    if not psychology:
        return []
    lines = [
        "",
        "## 群眾心理條件路徑",
        "",
        f"- 目前狀態: {psychology.get('state_label', '資料不足')} ({psychology.get('state', 'unknown')})",
        f"- 心理方向: {psychology.get('direction', 'unknown')}",
        f"- 急迫分數: 空方 {psychology.get('bearish_urgency_score', 0)} / 多方 {psychology.get('bullish_urgency_score', 0)} / 淨值 {psychology.get('net_urgency_score', 0)}",
        f"- 擁擠確認: {psychology.get('crowding_status', 'unknown')}；未平倉與廣度確認 {'已有' if psychology.get('position_confirmation') else '尚無'}",
        f"- 外生事件重置觀察: {'是' if psychology.get('exogenous_reset_watch') else '否'}；{'、'.join(psychology.get('exogenous_reset_reasons', [])) or '無明確觸發'}",
        f"- 使用限制: {psychology.get('guardrail', '')}",
        "",
        "### 當前證據",
        "",
    ]
    lines.extend([f"- {item}" for item in psychology.get("evidence", [])] or ["- 資料不足。"])
    lines.extend(["", "### 下一步條件路徑", ""])
    for item in psychology.get("next_paths", []):
        lines.append(
            f"- {item.get('rank')}. {item.get('label')}：{item.get('condition')} {item.get('effect')}"
        )
    return lines


def render_market_health_lines(health: dict) -> list[str]:
    diagnosis = health.get("diagnosis", {})
    prescription = health.get("prescription", {})
    health_value = health.get("health_value", {})
    lines = [
        "",
        "## 市場健康診療",
        "",
        "- 原則: 先避免傷害；本節是狀態診斷與風險處置，不是方向預言。",
        f"- 健康價值判斷: {health_value.get('label', '資料不足')} / {health_value.get('value', 'NA')}/100",
        f"- 健康白話: {health_value.get('headline', '')}",
        f"- 可控風險: {'是' if health_value.get('controllable_risk') else '否'}",
        f"- 判斷規則: {health_value.get('rule', '健康價值判斷看風險是否可控，不以單日漲跌當唯一依據。')}",
        f"- 防呆: {health_value.get('guardrail', '健康價值判斷只做風控分層，不產生買賣命令。')}",
        f"- 診斷狀態: {diagnosis.get('status', 'unknown')}",
        f"- 主要診斷: {diagnosis.get('primary', 'unknown')}",
        f"- 生命週期: {diagnosis.get('lifecycle_stage', 'unknown')}",
        f"- 診斷信心: {diagnosis.get('confidence', 'low')}",
        f"- 處置: {prescription.get('action', 'observe_and_recheck')}",
        f"- 處置強度: {prescription.get('intensity', 'none')}",
        "- 正式方向處方: 無；未驗證方向不得轉成交易命令。",
        f"- 安全限制: 禁止提高槓桿；禁止攤平；增加風險前必須確認。",
    ]
    if health_value.get("evidence"):
        lines.extend(["", "### 健康價值依據"])
        for item in health_value["evidence"]:
            lines.append(f"- {item}")
    symptoms = health.get("symptoms", [])
    if symptoms:
        lines.extend(["", "### 生命徵象與病兆"])
        for item in symptoms:
            lines.append(
                f"- [{item.get('severity', 'watch')}] {item.get('evidence', '')}"
            )
    alternatives = diagnosis.get("alternatives", [])
    if alternatives:
        lines.extend(["", "### 鑑別診斷"])
        for item in alternatives:
            lines.append(
                f"- {item.get('diagnosis')}: {item.get('discriminator')}"
            )
    if prescription.get("stop_conditions"):
        lines.extend(["", "### 停藥或翻案條件"])
        for item in prescription["stop_conditions"]:
            lines.append(f"- {item}")
    if prescription.get("recovery_conditions"):
        lines.extend(["", "### 康復確認條件"])
        for item in prescription["recovery_conditions"]:
            lines.append(f"- {item}")
    return lines


def render_self_repair_lines(repair: dict) -> list[str]:
    lines = [
        "",
        "## 自我偵測與修護",
        "",
        f"- 狀態: {repair.get('status', 'unknown')}",
        f"- 證據指紋: `{repair.get('evidence_sha256', 'NA')}`",
        "- 原始病歷: 永久保留，不回寫舊預判。",
        "- 自動修護邊界: 只允許阻擋輸出、降低權限、要求複查與追加病例。",
        "- 受保護變更: 模型係數、門檻、適用範圍及locked experiment不得自動修改。",
    ]
    incidents = repair.get("incidents", [])
    if incidents:
        lines.extend(["", "### 偵測到的錯誤"])
        for item in incidents:
            lines.append(
                f"- [{item.get('severity')}] {item.get('code')}: {item.get('evidence')}"
            )
    else:
        lines.extend(["", "- 本次未偵測到需立即修護的錯誤。"])
    actions = repair.get("automatic_actions", [])
    if actions:
        lines.extend(["", "### 已執行或要求的安全動作"])
        for item in actions:
            lines.append(f"- {item.get('action')}: {item.get('effect')}")
    candidates = repair.get("repair_candidates", [])
    if candidates:
        lines.extend(["", "### 候選修復"])
        for item in candidates:
            mode = (
                "需獨立驗證"
                if item.get("independent_validation_required")
                else "可安全自動套用"
            )
            lines.append(
                f"- {item.get('candidate_id')} ({mode}): {item.get('description')}"
            )
    return lines


def render_clinical_knowledge_lines(knowledge: dict) -> list[str]:
    profile = knowledge.get("reference_profile", {})
    route = knowledge.get("prescription_route", {})
    lines = [
        "",
        "## 病理與藥理知識庫",
        "",
        f"- 病例編號: `{knowledge.get('case_id', 'NA')}`",
        f"- 病例保存: {'新增不可回寫病例' if knowledge.get('case_appended') else '既有病例，不重複寫入'}",
        f"- 病例總數: {knowledge.get('pathology_case_count', 0)}",
        f"- 常模選擇: {profile.get('status', 'unsupported_scope')} / {profile.get('profile_id') or '無'}",
        f"- 常模可正式使用: {'是' if profile.get('usable') else '否'}",
        f"- 常模說明: {profile.get('reason', '')}",
        f"- 處方路由: {route.get('status', 'no_safe_prescription')} / {route.get('prescription_id') or '無'}",
        f"- 處方動作: {route.get('action', 'observe_and_recheck')} / {route.get('intensity', 'none')}",
        f"- 處方說明: {route.get('reason', '')}",
    ]
    similar = knowledge.get("similar_cases", [])
    if similar:
        lines.extend(["", "### 相似病例"])
        for item in similar:
            lines.append(
                f"- {item.get('case_id')}: 相似度 {item.get('similarity', 0):.2%}，"
                f"段位 {item.get('lifecycle_stage')}，結果 {item.get('outcome_status')}"
            )
    else:
        lines.extend(["", "- 尚無相同市場、頻率及目標的歷史病例可比對。"])
    return lines


def render_capital_flow_lines(capital: dict) -> list[str]:
    inst = capital.get("institutional", {})
    margin = capital.get("margin", {})
    derivatives = capital.get("derivatives", {})
    return [
        "",
        "## 法人籌碼與期權資金",
        "",
        f"- 資料狀態: {capital.get('data_status', '資料不足')}",
        f"- 白話結論: {capital.get('plain_summary', '')}",
        f"- 正式方向分數: 籌碼 {capital.get('chip_score', 0)}；期權 {capital.get('derivative_score', 0)}；合計 {capital.get('non_price_score', 0)}",
        f"- 未驗證啟發式觀察值: 籌碼 {capital.get('heuristic_chip_score', 0)}；期權 {capital.get('heuristic_derivative_score', 0)}（只顯示，不計入正式方向）",
        "- 驗證狀態: FinMind多日因子未通過purged樣本外門檻，正式係數維持0。",
        f"- 法人買賣超代理值: 5日 {num(inst.get('net_5d'))} / 20日 {num(inst.get('net_20d'))}",
        f"- 融資20日變化: {pct(margin.get('margin_change_20d'))}；融券20日變化: {pct(margin.get('short_change_20d'))}；資券比: {format_ratio(margin.get('margin_short_ratio'))}",
        f"- 期貨法人淨部位代理值: 5日 {num(derivatives.get('futures_inst_net_5d'))} / 20日 {num(derivatives.get('futures_inst_net_20d'))}",
        f"- 選擇權賣買權偏離: {format_ratio(derivatives.get('option_put_call_20d_z'))}；選擇權VIX偏離: {format_ratio(derivatives.get('option_vix_20d_z'))}",
    ]


def render_candlestick_lines(candle: dict) -> list[str]:
    stats = candle.get("historical_stats", {})
    metrics = candle.get("today_metrics", {})
    confirmation = candle.get("confirmation", {})
    lines = [
        "",
        "## K 線劇本分析",
        "",
        f"- K 線型態: {candle['label']} ({candle['type']})",
        f"- 白話說明: {candle['plain_summary']}",
        f"- 所在位置: {candle['position_text']}",
        f"- 實體幅度: {pct(metrics.get('body_pct'))}",
        f"- 上影線: {pct(metrics.get('upper_shadow_pct'))}",
        f"- 下影線: {pct(metrics.get('lower_shadow_pct'))}",
        f"- 收盤位置: {pct(metrics.get('close_position'))}",
        f"- 確認條件: {confirmation.get('confirm_rule', '')}",
        f"- 失效條件: {confirmation.get('fail_rule', '')}",
    ]
    if confirmation.get("confirm_level") is not None:
        lines.append(f"- 確認價位: {num(confirmation.get('confirm_level'))}")
    if confirmation.get("fail_level") is not None:
        lines.append(f"- 失效價位: {num(confirmation.get('fail_level'))}")
    lines.append(f"- 歷史樣本數: {stats.get('sample_count', 0)}")
    if stats.get("by_horizon"):
        lines.extend(
            [
                "",
                "| 後續期間 | 平均漲跌 | 上漲率 | 明顯下跌率 |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in stats["by_horizon"]:
            lines.append(
                f"| {row['horizon_days']}日 | {row['avg_return']:.2%} | "
                f"{row['up_rate']:.1%} | {row['down_rate']:.1%} |"
            )
    else:
        lines.append(f"- 歷史統計: {stats.get('message', '無專屬統計')}")
    return lines


def render_bagua_lifecycle_lines(bagua: dict) -> list[str]:
    if not bagua.get("enabled"):
        return [
            "",
            "## 八卦輪迴生命週期",
            "",
            f"- 狀態: {bagua.get('message', '資料不足')}",
        ]

    primary = bagua.get("primary", {})
    resonance = bagua.get("resonance", {})
    kline = bagua.get("kline_confirmation", {})
    stats = bagua.get("historical_stats", {})
    bottom_watch = bagua.get("bottom_watch", {})
    roles = bagua.get("roles", {})
    background = roles.get("background_gua", {})
    price_windows = roles.get("price_gua", {}).get("windows", [])
    state = roles.get("state_gua", {})
    background_trade = background.get("trade_annotation") or bagua_trade_annotation(background.get("code"))
    state_trade = state.get("trade_annotation") or bagua_trade_annotation(state.get("code"))
    primary_trade = primary.get("trade_annotation") or bagua_trade_annotation(primary.get("code"))
    lines = [
        "",
        "## 八卦輪迴生命週期",
        "",
        "- 固定原則: 背景卦、價位卦、狀態卦回答不同問題，不得強迫合併成單一卦。",
        f"- 背景卦: {background.get('gua', 'NA')} / {background.get('label', '資料不足')}，{background.get('meaning', '')}",
        f"- 狀態卦: {state.get('gua', 'NA')} / {state.get('label', '資料不足')}，{state.get('reason', '')}",
        f"- 狀態買賣標註: {state_trade.get('label', '資料不足')}；買方: {state_trade.get('buy_behavior', '')}；賣方: {state_trade.get('sell_behavior', '')}",
        f"- 背景買賣標註: {background_trade.get('label', '資料不足')}；{background_trade.get('action', '')}",
        f"- 標註防呆: {state_trade.get('guardrail', '買賣標註只作風控與行為對比，不是投資命令。')}",
        f"- 狀態路徑: 高點 {num(state.get('peak_close'))} → 低點 {num(state.get('trough_close'))}；"
        f"跌幅 {pct(state.get('drawdown'))}，低點後修復 {pct(state.get('recovery_from_trough'))}。",
        f"- 狀態確認: {state.get('confirmation', '')}",
        f"- 狀態失效: {state.get('invalidation', '')}",
        "",
        "### 價位卦（八等分固定尺規）",
        "",
        "| 回看區間 | 區間位置 | 價位卦 | 適用範圍 |",
        "| ---: | ---: | --- | --- |",
    ]
    for item in price_windows:
        lines.append(
            f"| {item.get('window')}日 | {pct(item.get('position'))} | "
            f"{item.get('gua', 'NA')}/{item.get('label', '資料不足')} | 只描述價格所在區間 |"
        )
    lines.extend([
        "",
        "### 各卦位買賣行為對比",
        "",
        "| 卦位 | 週期階段 | 買方行為 | 賣方行為 | 防呆條件 |",
        "| --- | --- | --- | --- | --- |",
    ])
    for item in bagua.get("sequence", []):
        trade = item.get("trade_annotation", {})
        lines.append(
            f"| {item.get('gua', 'NA')} | {item.get('label', '資料不足')} / {trade.get('label', '資料不足')} | "
            f"{trade.get('buy_behavior', '')} | {trade.get('sell_behavior', '')} | {trade.get('condition', '')} |"
        )
    lines.extend([
        "",
        "### 舊版多時間尺度卦位（保留供相容與核對）",
        "",
        "- 羅盤順序: 震 → 巽 → 離 → 坤 → 兌 → 乾 → 坎 → 艮 → 震",
        f"- 主卦時間級別: {bagua.get('primary_timeframe')}",
        f"- 目前主卦: {primary.get('gua')} / {primary.get('label')} ({primary.get('code')})",
        f"- 卦內進度: {pct(primary.get('phase_progress'))}",
        f"- 主卦白話: {primary.get('plain')}",
        f"- 下一卦: {primary.get('next', {}).get('gua')} / {primary.get('next', {}).get('label')}",
        f"- 退回卦: {primary.get('fallback', {}).get('gua')} / {primary.get('fallback', {}).get('label')}",
        f"- 轉卦確認: {primary.get('confirm_condition')}",
        f"- 失效退回: {primary.get('fail_condition')}",
        f"- 主卦買賣標註: {primary_trade.get('label', '資料不足')}；買方: {primary_trade.get('buy_behavior', '')}；賣方: {primary_trade.get('sell_behavior', '')}",
        f"- 多週期共振: {resonance.get('label')} / {resonance.get('plain_summary')}",
        f"- K線驗證: {kline.get('label')} / {kline.get('plain_summary')}",
        f"- 總結: {bagua.get('plain_summary')}",
    ])
    if bottom_watch.get("enabled"):
        lines.extend(
            [
                f"- 止跌檢討: {bottom_watch.get('label')} / {bottom_watch.get('summary')}",
                f"- 止跌時間規則: {bottom_watch.get('rule')}",
                (
                    f"- 止跌點位規則: 不破 {num(bottom_watch.get('latest_low_hold'))}；"
                    f"先收復 {num(bottom_watch.get('first_reclaim_close'))}；"
                    f"強確認看 {num(bottom_watch.get('strong_reclaim_open'))} 或 5日線 {num(bottom_watch.get('ma5_reclaim'))}"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "| 時間級別 | 卦位 | 狀態 | 卦內進度 | 區間位置 | 短期漲跌 | 中期漲跌 | 長期漲跌 | 高點回落 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in [bagua.get("monthly", {}), bagua.get("weekly", {}), bagua.get("daily", {})]:
        if not item.get("enabled"):
            continue
        metrics = item.get("metrics", {})
        lines.append(
            f"| {item.get('timeframe')} | {item.get('gua')} | {item.get('label')} | "
            f"{pct(item.get('phase_progress'))} | {pct(metrics.get('range_position'))} | "
            f"{pct(metrics.get('short_return'))} | {pct(metrics.get('mid_return'))} | "
            f"{pct(metrics.get('long_return'))} | {pct(metrics.get('drawdown'))} |"
        )

    if stats.get("by_horizon"):
        lines.extend(
            [
                "",
                f"- 日線同卦歷史樣本數: {stats.get('sample_count', 0)}",
                "",
                "| 日線同卦後續 | 樣本數 | 平均漲跌 | 上漲率 | 大跌率(<=-5%) |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in stats["by_horizon"]:
            lines.append(
                f"| {row['horizon_days']}日 | {row['sample_count']} | {pct(row.get('avg_return'))} | "
                f"{pct(row.get('up_rate'))} | {pct(row.get('down_5pct_rate'))} |"
            )
    return lines


def render_logic_consistency_lines(consistency: dict) -> list[str]:
    lines = [
        "",
        "## 判斷邏輯一致性檢查",
        "",
        f"- 總結: {consistency.get('summary', '')}",
        f"- 檢查等級: {consistency.get('level', '')}",
        f"- 硬矛盾數: {consistency.get('contradiction_count', 0)}",
        f"- 需補充說明數: {consistency.get('needs_explanation_count', 0)}",
        "",
        "| 檢查項目 | 狀態 | 說明 |",
        "| --- | --- | --- |",
    ]
    for row in consistency.get("rows", []):
        lines.append(f"| {row['item']} | {row['status']} | {row['explanation']} |")
    return lines


def render_peak_to_valley_warning_lines(warning: dict, backtest: dict) -> list[str]:
    lines = [
        "",
        "## 峰轉谷早期預警",
        "",
        f"- 預警等級: {warning.get('level', '資料不足')}",
        f"- 狀態: {warning.get('headline', '資料不足')}",
        f"- 摘要: {warning.get('summary', '')}",
        f"- 回測摘要: {backtest.get('summary', '')}",
        "- 留底: `reports/peak_to_valley_warning_backtest.md` / `reports/peak_to_valley_warning_backtest.json`",
        "",
        "| 確認項目 | 是否成立 | 依據 |",
        "| --- | --- | --- |",
    ]
    for item in warning.get("confirmation_checks", []):
        lines.append(f"| {item.get('name')} | {'是' if item.get('passed') else '否'} | {item.get('basis')} |")
    if warning.get("rules"):
        lines.extend(["", "規則:"])
        for rule in warning.get("rules", []):
            lines.append(f"- {rule}")
    return lines


def render_technical_phase_lines(technical: dict) -> list[str]:
    if not technical.get("enabled"):
        return [
            "",
            "## 大盤技術波段K線檢討",
            "",
            f"- 狀態: {technical.get('message', '資料不足')}",
        ]
    levels = technical.get("levels", {})
    signals = technical.get("signals", {})
    roles = technical.get("bagua_timeframe_roles", {})
    alignment = technical.get("alignment", {})
    lines = [
        "",
        "## 大盤技術波段K線檢討",
        "",
        f"- 技術結論: {technical.get('label')} / {technical.get('summary')}",
        f"- 與卦位預測一致性: {alignment.get('status')} / {alignment.get('summary')}",
        f"- 收盤/低點: {num(levels.get('close'))} / {num(levels.get('low'))}",
        f"- 均線: 5日 {num(levels.get('ma5'))}、10日 {num(levels.get('ma10'))}、20日 {num(levels.get('ma20'))}、60日 {num(levels.get('ma60'))}",
        f"- 近期低點/低收盤: {num(levels.get('recent_low'))} / {num(levels.get('recent_low_close'))}",
        f"- 波段高點回撤: {levels.get('swing_high_date')} 高點 {num(levels.get('swing_high'))} 至今 {pct(levels.get('drawdown_from_swing_high'))}",
        f"- 技術訊號: 跌破均線 {signals.get('below_ma_count', 0)} 條；破近期低點 {'是' if signals.get('broke_recent_low') else '否'}；破近期低收盤 {'是' if signals.get('broke_recent_close') else '否'}；K線 {signals.get('candlestick')}",
        "",
        "| 時間級別 | 卦位 | 角色 |",
        "| --- | --- | --- |",
    ]
    for key in ["monthly", "weekly", "daily"]:
        item = roles.get(key, {})
        lines.append(f"| {key} | {item.get('gua')} / {item.get('label')} | {item.get('role')} |")
    lines.extend(["", "成熟技術分析基礎:"])
    for item in technical.get("mature_technical_basis", []):
        lines.append(f"- {item}")
    if technical.get("fixed_formula"):
        lines.extend(
            [
                "",
                "固定式波段分析:",
                "",
                "| 步驟 | 結果 |",
                "| --- | --- |",
            ]
        )
        for row in technical.get("fixed_formula", []):
            lines.append(f"| {row.get('step')} | {row.get('result')} |")
    return lines


def render_route_reference_lines(route: dict) -> list[str]:
    if not route.get("enabled"):
        return [
            "",
            "## 歷史模式路線參考",
            "",
            f"- 狀態: {route.get('message', '資料不足')}",
        ]
    lines = [
        "",
        "## 歷史模式路線參考",
        "",
        f"- 方法: {route.get('method')}",
        f"- 目前路線狀態: {route.get('current_state_label')} ({route.get('current_state')})",
        f"- 目前壓縮路線: {route.get('current_path')}",
        f"- 技術錨點: {route.get('technical_anchor')}",
        f"- 白話結論: {route.get('plain_summary')}",
        f"- 相似歷史樣本數: {route.get('sample_count', 0)}",
    ]
    stats = route.get("stats", {}).get("by_horizon", [])
    if stats:
        lines.extend(
            [
                "",
                "| 相似路線後續 | 樣本數 | 平均漲跌 | 上漲率 | 大跌率(<=-5%) |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in stats:
            lines.append(
                f"| {row.get('horizon')} | {row.get('sample_count')} | {pct(row.get('avg_return'))} | "
                f"{pct(row.get('up_rate'))} | {pct(row.get('down_5pct_rate'))} |"
            )
    matches = route.get("matches", [])
    if matches:
        lines.extend(
            [
                "",
                "| 最相似路線 | 相似度 | 路線 | 後1日 | 後5日 | 後20日 | 後60日 |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in matches:
            returns = item.get("returns", {})
            lines.append(
                f"| {item.get('start_date')}~{item.get('end_date')} | {pct(item.get('similarity'))} | "
                f"{item.get('state_path')} | {pct(returns.get('1d'))} | {pct(returns.get('5d'))} | "
                f"{pct(returns.get('20d'))} | {pct(returns.get('60d'))} |"
            )
    return lines


def render_tradeable_cycle_lines(cycle: dict) -> list[str]:
    if not cycle.get("enabled"):
        return [
            "",
            "## 年度交易波段燈塔",
            "",
            f"- 狀態: {cycle.get('message', '資料不足')}",
        ]

    stats = cycle.get("stats", {})
    position = cycle.get("current_position", {})
    definition = cycle.get("definition", {})
    frequency = stats.get("frequency_state", {})
    lines = [
        "",
        "## 年度交易波段燈塔",
        "",
        f"- 段位判斷: {position.get('label')} ({position.get('code')})",
        f"- 白話說明: {position.get('plain_summary')}",
        f"- 燈塔方向: {position.get('lighthouse')}",
        f"- 今年已完成可交易漲幅段: {stats.get('current_year_count', 0)} 段",
        f"- 近252個交易日已完成可交易漲幅段: {stats.get('rolling_252d_count', 0)} 段",
        f"- 歷史平均每年: {stats.get('average_per_year', 0):.2f} 段",
        f"- 歷史中位數每年: {stats.get('median_per_year', 0):.1f} 段",
        f"- 頻率異常判斷: {frequency.get('label', '資料不足')} ({frequency.get('level', 'unknown')})",
        f"- 異常白話說明: {frequency.get('plain_summary', '')}",
        f"- 風險提醒: {frequency.get('risk_hint', '')}",
        f"- 今年/歷史平均倍數: {format_ratio(frequency.get('ratio_current_year'))}",
        f"- 近一年/歷史平均倍數: {format_ratio(frequency.get('ratio_rolling_252d'))}",
        f"- 目前自波段低點漲幅: {pct(position.get('gain_from_segment_low'))}",
        f"- 目前自波段高點回落: {pct(position.get('drawdown_from_segment_high'))}",
        f"- 段內交易日: {position.get('days_in_segment')}",
        f"- 突破確認價: {num(position.get('confirm_level'))}",
        f"- 失效防守價: {num(position.get('fail_level'))}",
        f"- 判斷定義: {definition.get('plain_text')}",
    ]
    recent = cycle.get("recent_segments", [])
    if recent:
        lines.extend(
            [
                "",
                "| 最近完成波段 | 低點日 | 高點日 | 結束日 | 最大漲幅 | 高點後回落 | 天數 |",
                "| ---: | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for idx, segment in enumerate(recent, start=1):
            lines.append(
                f"| {idx} | {segment['trough_date']} | {segment['peak_date']} | {segment['end_date']} | "
                f"{pct(segment.get('gain'))} | {pct(segment.get('drawdown_from_peak'))} | "
                f"{segment.get('duration_trading_days')} |"
            )

    annual = stats.get("annual_counts_recent", [])
    if annual:
        lines.extend(
            [
                "",
                "| 最近年度 | 可交易漲幅段數 |",
                "| ---: | ---: |",
            ]
        )
        for row in annual:
            lines.append(f"| {row['year']} | {row['count']} |")
    return lines


def render_washout_lines(washout: dict) -> list[str]:
    invalidation = washout.get("invalidation", {})
    stats = washout.get("historical_stats", {})
    metrics = washout.get("today_metrics", {})
    lines = [
        "",
        "## 洗盤規律偵測",
        "",
        f"- 洗盤判斷: {washout['label']} ({washout['type']})",
        f"- 白話說明: {washout['plain_summary']}",
        f"- 事件觸發: {washout['event_trigger']['plain_text']}",
        f"- 盤中最低跌幅: {pct(metrics.get('intraday_low_pct'))}",
        f"- 收盤拉回比例: {pct(metrics.get('close_recovery_ratio'))}",
        f"- 收盤漲跌: {pct(metrics.get('close_return_pct'))}",
        f"- 失效條件: {invalidation.get('rule', '')}",
    ]
    if invalidation.get("key_level") is not None:
        lines.append(f"- 關鍵低點: {num(invalidation.get('key_level'))}")
    if invalidation.get("confirmation_level") is not None:
        lines.append(f"- 轉強確認: 站回 {num(invalidation.get('confirmation_level'))}")
    lines.append(f"- 歷史樣本數: {stats.get('sample_count', 0)}")
    if stats.get("by_horizon"):
        lines.extend(
            [
                "",
                "| 後續期間 | 平均漲跌 | 上漲率 | 明顯下跌率 |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in stats["by_horizon"]:
            lines.append(
                f"| {row['horizon_days']}日 | {row['avg_return']:.2%} | "
                f"{row['up_rate']:.1%} | {row['down_rate']:.1%} |"
            )
    else:
        lines.append(f"- 歷史統計: {stats.get('message', '無專屬統計')}")
    return lines


def render_console_summary(payload: dict) -> str:
    forecast = payload["forecast"]
    check = payload["index_check"]
    update = payload["data_update"]
    external_update = payload["external_update"]
    night_update = payload["night_futures_update"]
    premarket = payload["premarket"]
    cause = payload["cause_analysis"]
    memory = payload["memory_industry_risk"]
    freshness = payload["data_freshness"]
    review = payload["self_review"]
    washout = payload["washout_pattern"]
    candle = payload["candlestick_pattern"]
    bagua = payload["bagua_lifecycle"]
    cycle = payload["tradeable_cycle"]
    technical = payload["technical_phase"]
    route = payload["route_reference"]
    consistency = payload["logic_consistency"]
    integrated = payload["integrated_summary"]
    capital = payload["capital_flow"]
    intraday = payload.get("intraday_tactical_monitor", {})
    human_behavior = payload.get("human_behavior_market_pattern", {})
    day_night_variance = payload.get("day_night_variance_pattern", {})
    programmed_pressure = payload.get("programmed_pressure_pattern", {})
    external_reset = payload.get("external_event_reset_monitor", {})
    practical = payload.get("practical_cause_arbitration", {})
    monthly_cycle = payload.get("monthly_cycle_monitor", {})
    validation = payload["model_validation"]
    policy = payload["production_policy"]
    state_trade = bagua.get("roles", {}).get("state_gua", {}).get("trade_annotation") or bagua_trade_annotation(
        bagua.get("primary", {}).get("code")
    )
    lines = [
        "今日市場研究與風險報告完成",
        f"預測基準日期: {payload['input']['date']}",
        f"台股分析基準日（最後有效資料）: {check['signal_date']}",
        f"台股資料更新: {status(update)} ({update.get('latest_date') or '無最新日期'})",
        f"外部市場更新: {status(external_update)} ({external_update.get('latest_date') or '無最新日期'})",
        f"台指期夜盤更新: {status(night_update)} ({night_update.get('latest_date') or '無最新日期'})",
        f"最後有效資料稽核: {freshness_status_text(freshness['overall_status'])}",
        f"研究結案: {'已完成' if validation.get('research_completed') else '尚未完成'} / {research_outcome_text(validation.get('research_outcome'))}",
        f"大盤多日正式方向訊號: {'啟用' if policy['main_multi_day_direction']['enabled'] else '停用（歷史相似機率僅供探索）'}",
    ]
    for item in freshness["sources"]:
        lines.append(
            f"- {item['name']}: {item['latest_date'] or 'NA'} / 預期 {item['expected_date']} / "
            f"{freshness_status_text(item['status'])}"
        )
    if premarket["is_premarket"]:
        lines.extend(
            [
                f"今日比較模式: 台股最後有效日 {premarket['spot_latest_date']}，比對 {premarket['forecast_date']} 最新外部市場與夜盤訊號",
                f"盤前壓力: {premarket['pressure']} / {premarket['scenario']}",
                f"盤前白話: {premarket['summary']}",
                f"盤前觀察: {premarket['watch']}",
                f"美股: Nasdaq(科技股) {pct(premarket['nasdaq_return_1d'])}, 費半(半導體) {pct(premarket['sox_return_1d'])}, TSM ADR(台積電美股) {pct(premarket['tsm_adr_return_1d'])}, VIX(恐慌指數) {pct(premarket['vix_return_1d'])}",
                f"台指期夜盤(台股盤前期貨): {pct(premarket['tx_night_spread_per'])}, 收 {num(premarket['tx_night_close'])}, 低 {num(premarket['tx_night_low'])}",
            ]
        )
    elif premarket.get("is_non_trading_day"):
        lines.extend(
            [
                f"非交易日觀察: 台股最後有效日 {premarket['spot_latest_date']}，報告日 {premarket['forecast_date']} 為休市/週末。",
                f"下個交易日情境: {premarket['pressure']} / {premarket['scenario']}",
                f"觀察說明: {premarket['summary']}",
                f"下個交易日觀察: {premarket['watch']}",
                f"美股: Nasdaq(科技股) {pct(premarket['nasdaq_return_1d'])}, 費半(半導體) {pct(premarket['sox_return_1d'])}, TSM ADR(台積電美股) {pct(premarket['tsm_adr_return_1d'])}, VIX(恐慌指數) {pct(premarket['vix_return_1d'])}",
                f"台指期夜盤(最近有效夜盤): {pct(premarket['tx_night_spread_per'])}, 收 {num(premarket['tx_night_close'])}, 低 {num(premarket['tx_night_low'])}",
            ]
        )
    lines.extend(
        [
            f"自主檢討: {review.get('plain_summary', '')}",
            f"前次探索紀錄: {review.get('previous_forecast_date') or '尚無'}",
            f"探索分布差異: {review.get('prediction_drift', {}).get('summary', '')}",
            f"週期追蹤: {review.get('daily_tracking', {}).get('summary', '')}",
            f"到期探索核對: {review.get('matured_checks', {}).get('summary', '')}",
            f"邏輯一致性: {consistency.get('summary', '')}",
            f"綜合總結: {integrated.get('bias_text', '資料不足')} / 分數 {integrated.get('score', 0)} / 信心 {integrated.get('confidence', '資料不足')}",
            f"總結白話: {integrated.get('plain_summary', '')}",
            f"法人籌碼期權: {capital.get('data_status', '資料不足')} / {capital.get('plain_summary', '')}",
            f"盤中戰術雷達: {intraday.get('label', '資料不足')} / {intraday.get('summary', '')}",
            f"盤中防守/轉強: {format_levels(intraday.get('defense_levels', [])) or 'NA'} / {format_levels(intraday.get('reclaim_levels', [])) or 'NA'}",
            f"日夜盤變異: {day_night_variance.get('label', '資料不足')} / {day_night_variance.get('relation_label', 'NA')} / {day_night_variance.get('summary', '')}",
            f"人類行為模式: {human_behavior.get('label', '資料不足')} / 群眾{human_behavior.get('crowd_state', 'NA')} / 戰術候選{human_behavior.get('tactic_candidate', 'NA')}",
            f"行為模式偵測: {programmed_pressure.get('label', '資料不足')} / 壓低分數 {programmed_pressure.get('current_score', 'NA')} / {programmed_pressure.get('summary', '')}",
            f"外部事件重置: {external_reset.get('label', '資料不足')} / 分數 {external_reset.get('reset_score', 'NA')} / {external_reset.get('summary', '')}",
            f"實務主因仲裁: {practical.get('label', '資料不足')} / 內部病灶 {practical.get('internal_structure_score', 'NA')} / 外部觸發 {practical.get('external_trigger_score', 'NA')} / 第{practical.get('treatment_phase', 'NA')}期{practical.get('treatment_stage', 'NA')} / 藥效{practical.get('correct_medicine', 'NA')} / {practical.get('summary', '')}",
            f"月內週期階段: {monthly_cycle.get('stage', '資料不足')} / 第{monthly_cycle.get('week_of_month', 'NA')}週 / {monthly_cycle.get('summary', '')}",
            f"八卦背景卦: {bagua.get('roles', {}).get('background_gua', {}).get('gua', 'NA')}/{bagua.get('roles', {}).get('background_gua', {}).get('label', '資料不足')}",
            f"八卦狀態卦: {bagua.get('roles', {}).get('state_gua', {}).get('gua', 'NA')}/{bagua.get('roles', {}).get('state_gua', {}).get('label', '資料不足')}，{bagua.get('roles', {}).get('state_gua', {}).get('reason', '')}",
            f"八卦狀態買賣標註: {state_trade.get('gua', 'NA')}/{state_trade.get('label', '資料不足')} / 買方 {state_trade.get('buy_behavior', '')} / 賣方 {state_trade.get('sell_behavior', '')}",
            f"八卦分類摘要: {bagua.get('roles', {}).get('summary', '資料不足')}",
            f"舊版主卦路線（僅相容）: 下一卦 {bagua.get('primary', {}).get('next', {}).get('gua', 'NA')}/{bagua.get('primary', {}).get('next', {}).get('label', 'NA')}，退回 {bagua.get('primary', {}).get('fallback', {}).get('gua', 'NA')}/{bagua.get('primary', {}).get('fallback', {}).get('label', 'NA')}",
            f"八卦共振: {bagua.get('resonance', {}).get('label', '資料不足')} / {bagua.get('resonance', {}).get('plain_summary', '')}",
            f"八卦K線驗證: {bagua.get('kline_confirmation', {}).get('label', '資料不足')} / {bagua.get('kline_confirmation', {}).get('plain_summary', '')}",
            f"技術波段K線檢討: {technical.get('label', '資料不足')} / {technical.get('alignment', {}).get('status', '資料不足')} / {technical.get('summary', '')}",
            f"歷史模式路線: {route.get('current_state_label', '資料不足')} / 樣本 {route.get('sample_count', 0)} / {route.get('plain_summary', '')}",
            f"交易波段燈塔: {cycle.get('current_position', {}).get('label', '資料不足')} / {cycle.get('current_position', {}).get('plain_summary', '')}",
            f"年度波段次數: 今年 {cycle.get('stats', {}).get('current_year_count', 0)} 段，近252交易日 {cycle.get('stats', {}).get('rolling_252d_count', 0)} 段，歷史平均 {cycle.get('stats', {}).get('average_per_year', 0):.2f} 段/年",
            f"波段頻率異常: {cycle.get('stats', {}).get('frequency_state', {}).get('label', '資料不足')} / {cycle.get('stats', {}).get('frequency_state', {}).get('plain_summary', '')}",
            f"異常風險提醒: {cycle.get('stats', {}).get('frequency_state', {}).get('risk_hint', '')}",
            f"K線分析: {candle['label']} / {candle['plain_summary']}",
            f"K線確認: {candle['confirmation'].get('confirm_rule', '')}",
            f"洗盤偵測: {washout['label']} / {washout['plain_summary']}",
            f"洗盤失效條件: {washout['invalidation'].get('rule', '')}",
        ]
    )
    lines.extend(
        [
            f"今日劇本: {cause['scenario']['name']}",
            f"真相分析: {cause['truth_text']} / {cause['intraday_truth_text']}",
            f"白話評估: {cause['plain_summary']}",
            f"後續觀察: {cause['scenario']['watch']}",
            f"指數檢查: {'一致' if check['is_consistent'] else '不一致'}",
            f"當日狀態: {stage_text(forecast['lifecycle_stage'])} / {regime_text(forecast['risk_regime'])}",
            f"記憶體產業風險: {memory_risk_text(memory['level'])} / {memory['points']} 分 / {memory['reasons']}",
            f"探索性歷史分布基準日: {forecast.get('base_trade_date', check['signal_date'])}，此日為第0個交易日",
            "探索性歷史相似分布（非正式方向訊號）:",
        ]
    )
    for item in forecast["forecasts"]:
        lines.append(
            f"- {item['horizon_days']}日(對照 {item.get('target_trade_date', 'NA')}): 歷史多數 {direction_text(item['predicted_direction'])} "
            f"(上漲占比 {item['probability_up']:.1%}, 下跌占比 {item['probability_down']:.1%}, 盤整占比 {item['probability_sideways']:.1%})"
        )
    lines.extend(
        [
            "名詞說明: 費半=美國半導體指數，ADR=在美國交易的外國公司股票，VIX=市場恐慌指數。",
            "名詞說明: risk_on=市場偏願意承擔風險，sideways=盤整，表示方向不明顯。",
        ]
    )
    return "\n".join(lines)


def status(update: dict) -> str:
    if update.get("success"):
        return "成功"
    if not update.get("attempted"):
        return "略過（使用既有快取）"
    return "失敗"


def research_outcome_text(value: str | None) -> str:
    return {
        "main_multi_day_direction_target_not_met": "大盤多日方向未達90%目標，正式訊號停用",
        "not_validated": "尚未通過正式驗證",
    }.get(value or "not_validated", value or "尚未通過正式驗證")


def freshness_status_text(value: str) -> str:
    return {
        "current": "最新",
        "partial": "部分資料不可用",
        "stale": "有資料過期",
        "unavailable": "不可用",
    }.get(value, value)


def safe_float(value):
    if pd.isna(value):
        return None
    return float(value)


def none_or_str(value) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return str(value)


def le(value, threshold) -> bool:
    return pd.notna(value) and value <= threshold


def ge(value, threshold) -> bool:
    return pd.notna(value) and value >= threshold


def between(value, low, high) -> bool:
    return pd.notna(value) and low <= value <= high


def pct(value) -> str:
    return "NA" if value is None else f"{value:.2%}"


def format_ratio(value) -> str:
    return "NA" if value is None else f"{value:.2f}x"


def num(value) -> str:
    return "NA" if value is None else f"{value:,.0f}"


def round_to_base(value: float, base: int) -> int:
    return int(round(value / base) * base)


def unique_numbers(values: list[int]) -> list[int]:
    output = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def format_levels(values: list[int]) -> str:
    return " / ".join(f"{value:,}" for value in values)


def direction_text(value: str) -> str:
    return {
        "up": "上漲(up)",
        "down": "下跌(down)",
        "sideways": "盤整(sideways，方向不明顯)",
        "unknown": "未知(unknown)",
    }.get(value, value)


def stage_text(value: str) -> str:
    return {
        "bottoming": "築底(bottoming，低檔整理、可能醞釀止跌)",
        "early_bull": "早期多頭(early_bull，偏多剛開始或轉強初期)",
        "main_bull": "主升段(main_bull，多頭趨勢較明確)",
        "topping": "高檔整理(topping，漲多後進入壓力區)",
        "early_bear": "早期空頭(early_bear，轉弱初期)",
        "main_bear": "主跌段(main_bear，空頭趨勢較明確)",
    }.get(value, value)


def regime_text(value: str) -> str:
    return {
        "risk_on": "風險偏多(risk_on，資金較願意買股票等風險資產)",
        "risk_off": "風險偏空(risk_off，資金偏保守、容易賣股票避險)",
        "neutral": "中性(neutral，多空沒有明顯一方主導)",
    }.get(value, value)


def memory_risk_text(value: str) -> str:
    return {
        "low": "低(low)",
        "watch": "注意(watch)",
        "high": "高(high)",
        "critical": "嚴重(critical)",
    }.get(value, value)


def glossary_lines() -> list[str]:
    return [
        "",
        "## 名詞白話說明",
        "",
        "- 盤前模式: 台股現貨尚未有今天完整資料時，先用美股收盤與台指期夜盤判斷今天開盤壓力。",
        "- 現貨: 一般股票市場日盤交易，例如台股加權指數當天的開高低收。",
        "- 台指期夜盤: 台灣加權指數期貨的夜間交易，常會先反映美股與國際消息，影響隔天台股開盤。",
        "- Nasdaq: 美國科技股指數，科技股與 AI 股的重要風向。",
        "- 費半/SOX: 費城半導體指數，用來觀察美國半導體族群強弱。",
        "- TSM ADR: 台積電在美國交易的股票，常提前反映海外投資人對台積電的看法。",
        "- VIX: 恐慌指數，越高代表市場避險情緒越強。",
        "- 盤前壓力: 開盤前綜合美股、半導體、VIX、台指期夜盤後得到的偏多或偏空壓力。",
        "- 綜合總結報告: 把價格、K線、洗盤、外部市場、夜盤、法人籌碼、期貨選擇權與週期位置合併後，給出最後偏向與風險條件。",
        "- 法人籌碼: 外資、投信、自營商等大型資金的買賣狀況，用來觀察主力資金是偏買還是偏賣。",
        "- 外資期貨空單: 外資在台指期建立的偏空部位；若淨空單下降，常代表避險或套利壓力減少，但不等於一定上漲。",
        "- 空單回補: 原本放空的人買回部位，可能推升短線反彈，但若沒有現貨買盤接力，反彈仍可能失敗。",
        "- 融資融券: 融資代表散戶借錢買股票，融券代表借股票放空；融資太熱常增加回檔風險，融券減少可能代表空方退場。",
        "- 借券賣出: 借股票賣出，常被用於避險或放空；金額偏高時表示市場仍有賣壓或套利部位。",
        "- 去槓桿: 投資人或機構降低借錢、期貨、選擇權等槓桿部位，常造成短線急跌或急拉。",
        "- 選擇權賣買權比: 用賣權和買權的相對強弱看避險情緒；賣權偏熱通常代表市場較緊張。",
        "- 選擇權VIX: 從選擇權價格反映的波動風險，偏高時代表市場預期震盪加大。",
        "- 關鍵防守: 若跌破這些位置，表示賣壓可能擴大；若守住，代表低檔有承接。",
        "- 轉強觀察: 若站回這些位置，代表買盤承接力增強。",
        "- risk_on: 風險偏多，市場較願意買股票等風險資產。",
        "- risk_off: 風險偏空，市場偏保守，容易賣股票或避險。",
        "- sideways: 盤整，表示方向不明顯，不是明確上漲也不是明確下跌。",
        "- 相似樣本: 歷史上和今天狀態相近的案例數，用來估算未來機率。",
        "- 信心: 最高機率方向的比例；越高代表歷史相似案例越集中在同一方向。",
        "- 判斷邏輯一致性: 檢查月線、週線、日線、K線、洗盤、波段與預測期間是否互相打架。",
        "- 硬矛盾: 同一時間級別、同一期間出現互相否定的判斷，需修正公式或文字。",
        "- 需補充說明: 看似衝突但其實是不同時間級別或不同期間，例如短線下跌但中期修復。",
        "- 八卦羅盤: 報告、卦位序號、下一卦與退回卦，統一依震、巽、離、坤、兌、乾、坎、艮的後天八卦順序呈現。",
        "- 主卦: 以月線為優先的大方向卦位，用來判斷大盤目前在整體生命週期的位置。",
        "- 卦內進度: 目前在該卦位內走到前段、中段或後段；越接近100%，越接近轉往下一卦或出現變化。",
        "- 下一卦: 若目前狀態順利延續，理論上最可能前進到的下一個生命週期階段。",
        "- 退回卦: 若目前狀態失敗或轉弱，可能退回的前一個生命週期階段。",
        "- 多週期共振: 月線、週線、日線是否指向相近卦位；越一致，訊號越穩，差距越大，越代表混沌或換檔。",
        "- 週期追蹤: 每日把八卦、波段、K線與洗盤狀態和前次紀錄比較，用來判斷是否轉卦、轉弱或修復。",
        "- 可交易漲幅段: 從低點起算，後續漲幅達到可操作空間的波段；本程式先用6%作為基本門檻。",
        "- 年度交易波段燈塔: 統計一年內可交易漲幅段出現幾次，並判斷目前位於築底、起漲、主升、末升或回落。",
        "- 波段頻率異常: 把今年或近一年波段次數拿去和歷史平均相比；若明顯偏高，代表市場節奏比平常更快。",
        "- 高頻波段: 可交易漲幅段出現得比歷史平均密集，表示機會多但也更容易急漲急跌。",
        "- 段位: 目前在整個波段中的位置，例如剛起漲、主升中、漲多末段、回落找底。",
        "- 突破確認價: 站上後較能確認新一段漲幅成立或延續的位置。",
        "- 失效防守價: 跌破後原本偏多或洗盤成功的判斷要降級的位置。",
        "- 洗盤: 盤中急跌製造恐慌，但若收盤明顯拉回，代表低檔可能有人承接。",
        "- 強洗盤: 急殺後拉回很明顯，歷史上後續修復機率較高。",
        "- 失敗洗盤: 盤中急殺後沒有拉回，較像真賣壓，不宜太早判斷止跌。",
        "- 失效條件: 原本的洗盤判斷被推翻的條件，例如隔日跌破洗盤日低點。",
        "- 事件觸發: 可能引發洗盤的表面理由，例如美股下跌、台指期夜盤轉弱或開盤跳空。",
        "- K 線: 用一天的開盤、最高、最低、收盤，描述當天多空攻防結果。",
        "- K 線實體: 收盤和開盤之間的距離；實體越大，代表當天方向越明顯。",
        "- 上影線: 盤中曾經上攻但被壓回的部分；上影越長，代表上方賣壓越明顯。",
        "- 下影線: 盤中曾經下殺但被拉回的部分；下影越長，代表低檔承接越明顯。",
        "- 收盤位置: 收盤落在當日高低區間的位置；越接近高點代表尾盤越強，越接近低點代表尾盤越弱。",
        "- K 線確認條件: 隔日需要突破或守住的價位，用來確認今天 K 線訊號是否有效。",
    ]


def format_best(best: dict | None) -> str:
    if not best:
        return "無"
    return f"{best['horizon_days']}日 ({best['hit_rate']:.2%})"


if __name__ == "__main__":
    main()
