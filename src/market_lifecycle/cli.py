import argparse
import json
from pathlib import Path

import pandas as pd

from market_lifecycle.backtest import backtest_regime
from market_lifecycle.causal_database import build_causal_database, summarize_causal_edges
from market_lifecycle.causal_edge_refinement import refine_causal_edge
from market_lifecycle.context_analysis import analyze_market_context
from market_lifecycle.edge_research import find_high_edge_conditions, walk_forward_edge_validation
from market_lifecycle.failure_analysis import (
    analyze_failure_factors,
    apply_rules_one_by_one,
    neutralize_failure_factors,
    walk_forward_failure_filter,
)
from market_lifecycle.factor_features import add_factor_features
from market_lifecycle.features import add_features
from market_lifecycle.lifecycle import score_lifecycle
from market_lifecycle.industry_risk import add_memory_industry_risk
from market_lifecycle.point_validation import validate_prediction_on_date
from market_lifecycle.point_validation import validate_prediction_batch
from market_lifecycle.probability_forecast import forecast_from_history, simulate_forecasts
from market_lifecycle.reporting import (
    write_failure_analysis_report,
    write_edge_research_report,
    write_market_context_report,
    write_markdown_report,
    write_point_validation_report,
    write_probability_forecast_report,
    write_scenario_research_report,
    write_success_evaluation_report,
    write_causal_database_report,
    write_causal_edge_refinement_report,
)
from market_lifecycle.scenario_research import research_market_scenarios, walk_forward_market_scenarios
from market_lifecycle.success_evaluation import (
    evaluate_prediction_success,
    walk_forward_success_evaluation,
)
from market_lifecycle.validation import walk_forward_validate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run market lifecycle prediction.")
    parser.add_argument("--input", required=True, help="CSV with date, open, high, low, close, volume.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--report", required=True, help="Output JSON report path.")
    parser.add_argument("--markdown-report", help="Optional Markdown report path.")
    parser.add_argument("--validate-date", help="Validate the signal on or before this date.")
    parser.add_argument("--validate-horizon-days", type=int, default=5, help="Forward days for point validation.")
    parser.add_argument("--validation-report", help="Optional single-date validation Markdown path.")
    parser.add_argument("--failure-report", help="Optional failure factor Markdown report path.")
    parser.add_argument("--corrected-output", help="Optional CSV path after neutralizing failure factors.")
    parser.add_argument("--factor-dir", help="Optional directory with non-price factor CSV files.")
    parser.add_argument("--memory-events", help="Optional curated memory-industry event CSV.")
    parser.add_argument("--context-date", help="Analyze before/after trend around this date.")
    parser.add_argument("--context-index", type=float, help="Optional user-provided index value for context analysis.")
    parser.add_argument("--lookback-days", type=int, default=20, help="Prior trading days for context analysis.")
    parser.add_argument("--forward-days", type=int, default=20, help="Future trading days for context analysis.")
    parser.add_argument("--context-report", help="Optional before/after context Markdown report path.")
    parser.add_argument("--success-report", help="Optional multi-horizon success-rate Markdown report path.")
    parser.add_argument("--success-json", help="Optional multi-horizon success-rate JSON report path.")
    parser.add_argument("--forecast-date", help="Create weather-like probability forecast for this date.")
    parser.add_argument("--forecast-report", help="Optional probability forecast Markdown report path.")
    parser.add_argument("--forecast-json", help="Optional probability forecast JSON report path.")
    parser.add_argument("--simulate-forecast", action="store_true", help="Run historical self-simulation for probability forecasts.")
    parser.add_argument("--simulate-step-days", type=int, default=5, help="Sampling step for forecast self-simulation.")
    parser.add_argument("--edge-report", help="Optional high-edge condition Markdown report path.")
    parser.add_argument("--edge-json", help="Optional high-edge condition JSON report path.")
    parser.add_argument("--edge-min-hit-rate", type=float, default=0.90, help="Minimum hit rate for edge search.")
    parser.add_argument("--edge-min-samples", type=int, default=120, help="Minimum samples for edge search.")
    parser.add_argument("--scenario-report", help="Optional general market-scenario Markdown report path.")
    parser.add_argument("--scenario-json", help="Optional general market-scenario JSON report path.")
    parser.add_argument("--scenario-min-hit-rate", type=float, default=0.90, help="Target hit rate for scenario research.")
    parser.add_argument("--scenario-min-samples", type=int, default=80, help="Minimum samples for scenario research.")
    parser.add_argument("--causal-output", help="Optional causal database CSV output path.")
    parser.add_argument("--causal-report", help="Optional causal database Markdown summary path.")
    parser.add_argument("--causal-min-hit-rate", type=float, default=0.60, help="Minimum hit rate for causal edges.")
    parser.add_argument("--causal-min-samples", type=int, default=80, help="Minimum samples for causal edges.")
    parser.add_argument("--refine-causal-signature", help="Causal signature to refine.")
    parser.add_argument("--refine-horizon-days", type=int, default=60, help="Horizon for causal edge refinement.")
    parser.add_argument("--refine-report", help="Optional causal edge refinement Markdown report path.")
    parser.add_argument("--refine-json", help="Optional causal edge refinement JSON report path.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

    features = add_factor_features(add_features(df), args.factor_dir)
    scored = add_memory_industry_risk(score_lifecycle(features), args.memory_events)
    scored = scored.dropna(subset=["close"])
    report = {
        "full_period": backtest_regime(scored),
        "walk_forward": walk_forward_validate(scored),
        "latest_signal": latest_signal(scored),
    }

    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    scored.to_csv(output_path, index=False)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.markdown_report:
        write_markdown_report(report, args.markdown_report)

    if args.validate_date:
        validation = validate_prediction_on_date(
            scored,
            args.validate_date,
            args.validate_horizon_days,
        )
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        if args.validation_report:
            write_point_validation_report(validation, args.validation_report)

    if args.failure_report:
        before = analyze_failure_factors(scored, args.validate_horizon_days, actionable_only=True)
        corrected, rules = neutralize_failure_factors(scored, before)
        after = analyze_failure_factors(corrected, args.validate_horizon_days, actionable_only=True)
        steps = apply_rules_one_by_one(scored, rules, args.validate_horizon_days)
        wf_failure = walk_forward_failure_filter(scored, args.validate_horizon_days)
        batch = validate_prediction_batch(corrected, args.validate_horizon_days)
        write_failure_analysis_report(before, after, rules, args.failure_report, steps, wf_failure)
        if args.corrected_output:
            corrected.to_csv(args.corrected_output, index=False)
        batch_summary = {key: value for key, value in batch.items() if key != "rows"}
        print(json.dumps({"failure_analysis": after, "batch_success": batch_summary, "rules": rules, "walk_forward_failure_filter": wf_failure["summary"]}, ensure_ascii=False, indent=2))

    if args.context_date:
        analysis = analyze_market_context(
            scored,
            args.context_date,
            input_index=args.context_index,
            lookback_days=args.lookback_days,
            forward_days=args.forward_days,
        )
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
        if args.context_report:
            write_market_context_report(analysis, args.context_report)

    if args.success_report or args.success_json:
        success = evaluate_prediction_success(scored)
        wf_success = walk_forward_success_evaluation(scored)
        if args.success_report:
            write_success_evaluation_report(success, wf_success, args.success_report)
        if args.success_json:
            Path(args.success_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.success_json).write_text(
                json.dumps({"full_sample": success, "walk_forward": wf_success}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(json.dumps({"success_summary": success["summary"], "walk_forward_summary": wf_success["summary"]}, ensure_ascii=False, indent=2))

    if args.forecast_report or args.forecast_json or args.forecast_date:
        forecast = forecast_from_history(scored, args.forecast_date)
        simulation = simulate_forecasts(scored, step_days=args.simulate_step_days) if args.simulate_forecast else None
        if args.forecast_report:
            write_probability_forecast_report(forecast, simulation, args.forecast_report)
        if args.forecast_json:
            Path(args.forecast_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.forecast_json).write_text(
                json.dumps({"forecast": forecast, "simulation": simulation}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        output = {"forecast": forecast}
        if simulation:
            output["simulation_summary"] = simulation["summary"]
        print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.edge_report or args.edge_json:
        edges = find_high_edge_conditions(
            scored,
            min_samples=args.edge_min_samples,
            min_hit_rate=args.edge_min_hit_rate,
        )
        wf_edges = walk_forward_edge_validation(
            scored,
            min_samples=max(40, args.edge_min_samples // 2),
            min_hit_rate=args.edge_min_hit_rate,
        )
        if args.edge_report:
            write_edge_research_report(edges, wf_edges, args.edge_report)
        if args.edge_json:
            Path(args.edge_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.edge_json).write_text(
                json.dumps({"full_sample": edges, "walk_forward": wf_edges}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(json.dumps({"edge_conditions": edges, "walk_forward_summary": [item["summary"] for item in wf_edges["horizons"]]}, ensure_ascii=False, indent=2))

    if args.scenario_report or args.scenario_json:
        scenarios = research_market_scenarios(
            scored,
            min_samples=args.scenario_min_samples,
            target_hit_rate=args.scenario_min_hit_rate,
        )
        wf_scenarios = walk_forward_market_scenarios(
            scored,
            min_samples=max(30, args.scenario_min_samples // 2),
            target_hit_rate=args.scenario_min_hit_rate,
        )
        if args.scenario_report:
            write_scenario_research_report(scenarios, wf_scenarios, args.scenario_report)
        if args.scenario_json:
            Path(args.scenario_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.scenario_json).write_text(
                json.dumps({"full_sample": scenarios, "walk_forward": wf_scenarios}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(json.dumps({"scenario_passing_count": scenarios["passing_count"], "walk_forward": wf_scenarios["summary"]}, ensure_ascii=False, indent=2))

    if args.causal_output or args.causal_report:
        causal = build_causal_database(scored)
        summary = summarize_causal_edges(
            causal,
            min_samples=args.causal_min_samples,
            min_hit_rate=args.causal_min_hit_rate,
        )
        if args.causal_output:
            Path(args.causal_output).parent.mkdir(parents=True, exist_ok=True)
            causal.to_csv(args.causal_output, index=False)
        if args.causal_report:
            write_causal_database_report(summary, args.causal_report)
        print(json.dumps({"causal_rows": len(causal), "summary": summary}, ensure_ascii=False, indent=2))

    if args.refine_causal_signature:
        causal = build_causal_database(scored)
        refinement = refine_causal_edge(
            causal,
            args.refine_causal_signature,
            args.refine_horizon_days,
        )
        if args.refine_report:
            write_causal_edge_refinement_report(refinement, args.refine_report)
        if args.refine_json:
            Path(args.refine_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.refine_json).write_text(json.dumps(refinement, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(refinement, ensure_ascii=False, indent=2))


def latest_signal(scored: pd.DataFrame) -> dict:
    latest = scored.sort_values("date").iloc[-1]
    return {
        "date": str(latest["date"].date()),
        "lifecycle_stage": latest["lifecycle_stage"],
        "stage_score": float(latest["stage_score"]),
        "risk_regime": latest["risk_regime"],
        "forward_1w_view": latest["forward_1w_view"],
        "forward_4w_view": latest["forward_4w_view"],
        "confidence": float(latest["confidence"]),
        "memory_risk_points": int(latest.get("memory_risk_points", 0)),
        "memory_risk_score": int(latest.get("memory_risk_score", 0)),
        "memory_risk_level": latest.get("memory_risk_level", "low"),
        "memory_market_confirmed": bool(latest.get("memory_market_confirmed", False)),
        "memory_risk_reasons": latest.get("memory_risk_reasons", ""),
    }


if __name__ == "__main__":
    main()
