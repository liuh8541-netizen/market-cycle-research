"""Leakage-safe retrospective audit for the crowd psychology state machine.

The replay uses only information available before the Taiwan cash open:

* the completed night session carrying the same signal date;
* the strictly prior US-market observation;
* the strictly prior Taiwan cash candle.

This is a retrospective audit, not a prospective validation.  The model does
not emit a formal direction signal; directional accuracy is reported only as a
diagnostic of the descriptive psychology label.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_lifecycle.external_market import add_intraday_truth_features
from market_lifecycle.psychology_state import build_psychology_state

from predict_market import (
    add_candlestick_metrics,
    analyze_night_path,
    classify_candlestick,
    classify_washout,
    score_external_row,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the crowd psychology path model.")
    parser.add_argument("--cash", default="data/processed/twii_daily.csv")
    parser.add_argument("--night", default="data/processed/taiwan_futures_night.csv")
    parser.add_argument("--external", default="data/processed/external_markets.csv")
    parser.add_argument("--csv-output", default="reports/psychology_state_backtest.csv")
    parser.add_argument("--json-output", default="reports/psychology_state_backtest.json")
    parser.add_argument("--md-output", default="reports/psychology_state_backtest.md")
    args = parser.parse_args()

    rows = replay_psychology_history(
        pd.read_csv(rooted(args.cash)),
        pd.read_csv(rooted(args.night)),
        pd.read_csv(rooted(args.external)),
    )
    report = summarize_backtest(rows)

    csv_path = rooted(args.csv_output)
    json_path = rooted(args.json_output)
    md_path = rooted(args.md_output)
    for path in [csv_path, json_path, md_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_report(report), encoding="utf-8")

    overall = report["overall"]
    urgent = report["urgent_following"]
    print(
        f"rows={report['sample']['cases']} "
        f"direction={pct(overall['direction_accuracy'])} "
        f"top_path={pct(overall['top_path_hit_rate'])} "
        f"urgent_direction={pct(urgent['direction_accuracy'])}"
    )
    print(f"Report: {md_path}")


def replay_psychology_history(cash: pd.DataFrame, night: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    cash = prepare_cash(cash)
    night = prepare_dates(night, "signal_date")
    external = prepare_dates(external, "date")

    cash_dates = cash["date"].to_numpy(dtype="datetime64[ns]")
    external_dates = external["date"].to_numpy(dtype="datetime64[ns]")
    cash_by_date = cash.set_index("date", drop=False)
    output: list[dict] = []
    ranking_counts: dict[tuple[str, str], dict[str, int]] = {}

    for _, nrow in night.sort_values("signal_date").drop_duplicates("signal_date", keep="last").iterrows():
        signal_date = pd.Timestamp(nrow["signal_date"])
        if signal_date not in cash_by_date.index:
            continue
        prior_cash_index = int(cash_dates.searchsorted(signal_date.to_datetime64(), side="left")) - 1
        prior_external_index = int(external_dates.searchsorted(signal_date.to_datetime64(), side="left")) - 1
        if prior_cash_index < 0 or prior_external_index < 0:
            continue

        prior = cash.iloc[prior_cash_index]
        outcome = cash_by_date.loc[signal_date]
        if isinstance(outcome, pd.DataFrame):
            outcome = outcome.iloc[-1]
        erow = external.iloc[prior_external_index]
        required = [prior.get("close"), outcome.get("open"), outcome.get("high"), outcome.get("low"), outcome.get("close")]
        if any(pd.isna(value) for value in required):
            continue

        night_path = analyze_night_path(nrow, float(prior["close"]))
        candle = classify_candlestick(prior)
        washout = classify_washout(prior)
        payload = {
            "premarket": {
                "night_path": night_path,
                "external_score": score_external_row(erow),
                "sox_return_1d": number_or_none(erow.get("sox_return_1d")),
                "tsm_adr_return_1d": number_or_none(erow.get("tsm_adr_return_1d")),
                "vix_return_1d": number_or_none(erow.get("vix_return_1d")),
            },
            "intraday_tactical_monitor": {"code": "premarket_only", "live_usable": False},
            "candlestick_pattern": candle,
            "washout_pattern": washout,
            "integrated_summary": {},
        }
        psychology = build_psychology_state(payload, behavior_statistics_from_counts(ranking_counts))
        result = evaluate_day(
            signal_date, prior, outcome, nrow, erow, night_path, candle, washout, psychology
        )
        output.append(result)
        cohort_key = (str(result["state"]), str(result["direction"]))
        path_counts = ranking_counts.setdefault(cohort_key, {})
        realized = str(result["realized_path"])
        path_counts[realized] = path_counts.get(realized, 0) + 1

    return pd.DataFrame(output).sort_values("signal_date").reset_index(drop=True)


def behavior_statistics_from_counts(
    counts: dict[tuple[str, str], dict[str, int]],
) -> list[dict]:
    output = []
    for (state, direction), paths in counts.items():
        cases = sum(paths.values())
        output.append({
            "psychology_state": state,
            "direction": direction,
            "cases": int(cases),
            "path_distribution": [
                {"path": str(path), "cases": int(count), "rate": float(count / cases)}
                for path, count in paths.items()
            ],
        })
    return output


def prepare_cash(frame: pd.DataFrame) -> pd.DataFrame:
    data = prepare_dates(frame, "date")
    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    data = add_intraday_truth_features(data)
    add_candlestick_metrics(data)
    data["cash_return_1d"] = data["close"].pct_change()
    for window in [5, 10, 20, 60]:
        data[f"ma_{window}"] = data["close"].rolling(window).mean()
    data["close_vs_ma5"] = data["close"] / data["ma_5"] - 1
    data["close_vs_ma20"] = data["close"] / data["ma_20"] - 1
    data["close_vs_ma60"] = data["close"] / data["ma_60"] - 1
    data["drawdown_60"] = data["close"] / data["close"].rolling(60).max() - 1
    data["realized_volatility_20"] = data["cash_return_1d"].rolling(20).std()
    data["range_20"] = data["high"].rolling(20).max() / data["low"].rolling(20).min() - 1
    volume_mean = data["volume"].where(data["volume"].gt(0)).rolling(20).mean()
    data["volume_ratio_20"] = data["volume"].where(data["volume"].gt(0)) / volume_mean
    return data.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def prepare_dates(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    data = frame.copy()
    data[column] = pd.to_datetime(data[column], errors="coerce")
    return data.dropna(subset=[column]).sort_values(column).reset_index(drop=True)


def evaluate_day(
    signal_date, prior, cash, night, external, night_path: dict,
    candle: dict, washout: dict, psychology: dict,
) -> dict:
    prior_close = float(prior["close"])
    cash_open = float(cash["open"])
    cash_high = float(cash["high"])
    cash_low = float(cash["low"])
    cash_close = float(cash["close"])
    night_open = float(night["tx_night_open"])
    night_high = float(night["tx_night_high"])
    night_low = float(night["tx_night_low"])
    night_close = float(night["tx_night_close"])
    direction = psychology["direction"]

    gap_return = cash_open / prior_close - 1
    close_return = cash_close / prior_close - 1
    intraday_return = cash_close / cash_open - 1
    actual_direction = sign_label(close_return)
    gap_direction = sign_label(gap_return)
    night_direction = sign_label(float(night["tx_night_return"]))
    actionable = direction in {"bearish", "bullish"}

    realized_path = classify_realized_path(
        direction, prior_close, cash_open, cash_high, cash_low, cash_close,
        night_open, night_high, night_low, night_close,
    )
    top_code = psychology.get("next_paths", [{}])[0].get("code")
    top_hit = bool(top_code == realized_path)
    return {
        "signal_date": str(pd.Timestamp(signal_date).date()),
        "prior_cash_date": str(pd.Timestamp(prior["date"]).date()),
        "external_date": str(pd.Timestamp(external["date"]).date()),
        "state": psychology["state"],
        "state_label": psychology["state_label"],
        "direction": direction,
        "bearish_urgency_score": psychology["bearish_urgency_score"],
        "bullish_urgency_score": psychology["bullish_urgency_score"],
        "net_urgency_score": psychology["net_urgency_score"],
        "calibrated_direction_score": psychology.get("calibrated_direction_score"),
        "direction_weight_version": psychology.get("direction_weight_version"),
        "night_status": night_path.get("status", "neutral"),
        "night_close_position": night_path.get("close_position"),
        "night_recovery_from_low": night_path.get("recovery_from_low"),
        "night_close_vs_prior_cash": night_path.get("close_vs_prior_cash"),
        "night_material_pressure": night_path.get("material_pressure"),
        "prior_candle_type": candle.get("type"),
        "prior_candle_label": candle.get("label"),
        "prior_washout_type": washout.get("type"),
        "prior_washout_label": washout.get("label"),
        "external_score": score_external_row(external),
        "exogenous_reset_watch": psychology["exogenous_reset_watch"],
        "crowding_status": psychology.get("crowding_status"),
        "position_confirmation": psychology.get("position_confirmation"),
        "psychology_evidence_json": json.dumps(psychology.get("evidence", []), ensure_ascii=False),
        "exogenous_reset_reasons_json": json.dumps(psychology.get("exogenous_reset_reasons", []), ensure_ascii=False),
        "expected_paths_json": json.dumps(psychology.get("next_paths", []), ensure_ascii=False),
        "top_path": top_code,
        "realized_path": realized_path,
        "path_resolved": realized_path != "unresolved",
        "top_path_hit": top_hit,
        "actionable_direction": actionable,
        "actual_direction": actual_direction,
        "direction_hit": actionable and direction_matches(direction, actual_direction),
        "gap_direction_hit": actionable and direction_matches(direction, gap_direction),
        "night_baseline_direction": night_direction,
        "night_baseline_hit": direction_matches(night_direction, actual_direction),
        "gap_return": gap_return,
        "cash_close_return": close_return,
        "intraday_return": intraday_return,
        "cash_open": cash_open,
        "cash_high": cash_high,
        "cash_low": cash_low,
        "cash_close": cash_close,
        "night_open": night_open,
        "night_high": night_high,
        "night_low": night_low,
        "night_close": night_close,
        "night_return": float(night["tx_night_return"]),
        "night_range": number_or_none(night.get("tx_night_range")),
        "night_spread_per": number_or_none(night.get("tx_night_spread_per")),
        "night_volume": number_or_none(night.get("tx_night_volume")),
        "prior_cash_open": number_or_none(prior.get("open")),
        "prior_cash_high": number_or_none(prior.get("high")),
        "prior_cash_low": number_or_none(prior.get("low")),
        "prior_cash_close": prior_close,
        "prior_cash_volume": number_or_none(prior.get("volume")),
        "prior_open_gap_pct": number_or_none(prior.get("open_gap_pct")),
        "prior_intraday_low_pct": number_or_none(prior.get("intraday_low_pct")),
        "prior_close_return_pct": number_or_none(prior.get("close_return_pct")),
        "prior_close_recovery_ratio": number_or_none(prior.get("close_recovery_ratio")),
        "prior_ma5": number_or_none(prior.get("ma_5")),
        "prior_ma10": number_or_none(prior.get("ma_10")),
        "prior_ma20": number_or_none(prior.get("ma_20")),
        "prior_ma60": number_or_none(prior.get("ma_60")),
        "prior_close_vs_ma5": number_or_none(prior.get("close_vs_ma5")),
        "prior_close_vs_ma20": number_or_none(prior.get("close_vs_ma20")),
        "prior_close_vs_ma60": number_or_none(prior.get("close_vs_ma60")),
        "prior_drawdown_60": number_or_none(prior.get("drawdown_60")),
        "prior_realized_volatility_20": number_or_none(prior.get("realized_volatility_20")),
        "prior_range_20": number_or_none(prior.get("range_20")),
        "prior_volume_ratio_20": number_or_none(prior.get("volume_ratio_20")),
        "nasdaq_return_1d": number_or_none(external.get("nasdaq_return_1d")),
        "sox_return_1d": number_or_none(external.get("sox_return_1d")),
        "sp500_return_1d": number_or_none(external.get("sp500_return_1d")),
        "tsm_adr_return_1d": number_or_none(external.get("tsm_adr_return_1d")),
        "vix_return_1d": number_or_none(external.get("vix_return_1d")),
        "usd_twd_return_1d": number_or_none(external.get("usd_twd_return_1d")),
        "treasury_5y_close": number_or_none(external.get("treasury_5y_close")),
        "treasury_10y_close": number_or_none(external.get("treasury_10y_close")),
        "treasury_30y_close": number_or_none(external.get("treasury_30y_close")),
        "treasury_5y_return_1d": number_or_none(external.get("treasury_5y_return_1d")),
        "treasury_10y_return_1d": number_or_none(external.get("treasury_10y_return_1d")),
        "treasury_30y_return_1d": number_or_none(external.get("treasury_30y_return_1d")),
    }


def classify_realized_path(direction, prior_close, cash_open, cash_high, cash_low, cash_close,
                           night_open, night_high, night_low, night_close) -> str:
    if direction == "bearish":
        if cash_low < night_low and cash_close < night_low:
            return "bearish_continuation"
        if cash_open < prior_close and night_close <= cash_close < night_open:
            return "short_covering_rebound"
        if cash_close >= night_open:
            return "path_reset_reversal"
        return "unresolved"
    if direction == "bullish":
        if cash_high > night_high and cash_close > night_high:
            return "bullish_continuation"
        if cash_open > prior_close and night_low < cash_close <= night_close:
            return "profit_taking"
        if cash_close <= night_low:
            return "bull_trap_reset"
        return "unresolved"
    if cash_high > night_high or cash_low < night_low:
        return "range_confirmation"
    return "unresolved"


def summarize_backtest(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {"framework": "crowd_psychology_path_v1_backtest", "sample": {"cases": 0}}
    rows = rows.copy()
    rows["year"] = pd.to_datetime(rows["signal_date"]).dt.year
    actionable = rows[rows["actionable_direction"]]
    urgent = rows[rows["state"].eq("urgent_following") & rows["actionable_direction"]]
    first_recent_year = max(int(rows["year"].min()), 2023)
    early = rows[rows["year"] < first_recent_year]
    recent = rows[rows["year"] >= first_recent_year]

    comparison = compare_with_night_baseline(actionable)
    urgent_comparison = compare_with_night_baseline(urgent)
    urgent_bearish = urgent[urgent["direction"].eq("bearish")]
    urgent_bullish = urgent[urgent["direction"].eq("bullish")]
    majority_rate = max(
        float(actionable["actual_direction"].eq("bullish").mean()),
        float(actionable["actual_direction"].eq("bearish").mean()),
    ) if len(actionable) else None
    return {
        "framework": "crowd_psychology_path_v1_backtest",
        "generated_from": "completed local history",
        "method": {
            "timing": "same-date completed night session + strictly prior external row + strictly prior Taiwan cash candle",
            "direction_success": "bearish/bullish psychology direction equals cash close direction versus prior cash close",
            "top_path_success": "bearish: cash breaks and closes below night low; bullish: cash breaks and closes above night high",
            "path_coverage": "actual cash close satisfies one of the predeclared price-only route bands; breadth confirmation is not inferred",
            "warning": "Retrospective diagnostic only. Rules were not frozen before this history occurred, so this is not genuine prospective validation.",
        },
        "sample": {
            "cases": int(len(rows)),
            "first_date": rows.iloc[0]["signal_date"],
            "last_date": rows.iloc[-1]["signal_date"],
            "actionable_cases": int(len(actionable)),
            "actionable_coverage": ratio(len(actionable), len(rows)),
        },
        "overall": metrics(rows),
        "urgent_following": metrics(urgent),
        "urgent_bearish": metrics(urgent_bearish),
        "urgent_bullish": metrics(urgent_bullish),
        "urgent_bearish_path_distribution": path_distribution(urgent_bearish),
        "urgent_bullish_path_distribution": path_distribution(urgent_bullish),
        "early_period": {"label": f"through {first_recent_year - 1}", **metrics(early)},
        "recent_period": {"label": f"{first_recent_year} onward", **metrics(recent)},
        "by_state": grouped_metrics(rows, "state"),
        "by_direction": grouped_metrics(rows, "direction"),
        "by_year": grouped_metrics(rows, "year"),
        "weight_calibration": direction_weight_calibration(rows),
        "baselines": {
            "unconditional_majority_direction_accuracy": majority_rate,
            "night_direction_accuracy_on_same_actionable_cases": comparison["night_accuracy"],
            "psychology_direction_accuracy_on_same_actionable_cases": comparison["psychology_accuracy"],
            "psychology_minus_night_percentage_points": comparison["edge_pp"],
            "mcnemar_exact_p_value": comparison["mcnemar_p_value"],
            "psychology_only_correct": comparison["psychology_only_correct"],
            "night_only_correct": comparison["night_only_correct"],
        },
        "urgent_baseline_comparison": urgent_comparison,
    }


def direction_weight_calibration(rows: pd.DataFrame) -> dict:
    work = rows.copy()
    work["year"] = pd.to_datetime(work["signal_date"]).dt.year
    configurations = {
        "legacy_v1": {"night": 1, "external": 2, "candle": 1, "washout": 1},
        "selected_v2": {"night": 1, "external": 1, "candle": 1, "washout": 0},
        "night_only": {"night": 1, "external": 0, "candle": 0, "washout": 0},
    }
    output = {}
    for name, weights in configurations.items():
        score = work.apply(lambda row: retrospective_direction_score(row, weights), axis=1)
        predicted = score.map(lambda value: "bullish" if value > 0 else ("bearish" if value < 0 else "mixed"))
        actionable = predicted.ne("mixed")
        periods = {}
        for period, mask in {
            "calibration_2017_2022": work["year"].le(2022),
            "temporal_holdout_2023_plus": work["year"].ge(2023),
            "all_history": pd.Series(True, index=work.index),
        }.items():
            use = mask & actionable
            periods[period] = {
                "cases": int(use.sum()),
                "coverage": float(use.sum() / mask.sum()) if mask.sum() else None,
                "accuracy": float(predicted[use].eq(work.loc[use, "actual_direction"]).mean()) if use.any() else None,
            }
        output[name] = {"weights": weights, "periods": periods}
    selected_test = output["selected_v2"]["periods"]["temporal_holdout_2023_plus"]["accuracy"]
    legacy_test = output["legacy_v1"]["periods"]["temporal_holdout_2023_plus"]["accuracy"]
    return {
        "selection_rule": "choose on 2017-2022 calibration accuracy; washout direction weight removed; verify on 2023+ temporal holdout",
        "selected": "selected_v2", "configurations": output,
        "holdout_improvement_vs_legacy_pp": (selected_test - legacy_test) * 100,
        "guardrail": "時間切分屬回溯性時間外檢查；仍需模型凍結後的真正前瞻病例。",
    }


def retrospective_direction_score(row: pd.Series, weights: dict) -> float:
    night = {
        "bearish_extension_close_near_low": -3,
        "bullish_extension_close_near_high": 3,
        "bearish_but_recovered": -1,
        "bullish_but_unconfirmed": 1,
    }.get(row.get("night_status"), 0)
    external = -1 if row.get("external_score", 0) <= -2 else (1 if row.get("external_score", 0) >= 2 else 0)
    candle = -1 if row.get("prior_candle_type") in {"long_bear", "gap_up_failed"} else (
        1 if row.get("prior_candle_type") in {"long_bull", "gap_down_reversal"} else 0
    )
    washout = -1 if row.get("prior_washout_type") == "failed_washout" else (
        1 if row.get("prior_washout_type") in {"strong_washout", "normal_washout", "gap_down_recovery"} else 0
    )
    return (
        night * weights["night"] + external * weights["external"]
        + candle * weights["candle"] + washout * weights["washout"]
    )


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "cases": 0, "actionable_cases": 0, "direction_accuracy": None,
            "direction_wilson_95": [None, None], "gap_direction_accuracy": None,
            "material_direction_confirmation_rate": None,
            "top_path_hit_rate": None, "path_coverage": None,
        }
    actionable = frame[frame["actionable_direction"]]
    hits = int(actionable["direction_hit"].sum())
    return {
        "cases": int(len(frame)),
        "actionable_cases": int(len(actionable)),
        "direction_accuracy": ratio(hits, len(actionable)),
        "direction_wilson_95": list(wilson(hits, len(actionable))),
        "gap_direction_accuracy": mean_bool(actionable, "gap_direction_hit"),
        "material_direction_confirmation_rate": material_direction_rate(actionable),
        "top_path_hit_rate": mean_bool(actionable, "top_path_hit"),
        "path_coverage": mean_bool(actionable, "path_resolved"),
        "average_cash_close_return": float(frame["cash_close_return"].mean()),
    }


def grouped_metrics(frame: pd.DataFrame, column: str) -> list[dict]:
    output = []
    for key, group in frame.groupby(column, dropna=False, sort=True):
        output.append({"group": str(key), **metrics(group)})
    return output


def path_distribution(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    counts = frame["realized_path"].value_counts()
    return [
        {"path": str(path), "cases": int(count), "rate": float(count / len(frame))}
        for path, count in counts.items()
    ]


def material_direction_rate(frame: pd.DataFrame, threshold: float = 0.005):
    if frame.empty:
        return None
    confirmed = (
        (frame["direction"].eq("bearish") & frame["cash_close_return"].le(-threshold))
        | (frame["direction"].eq("bullish") & frame["cash_close_return"].ge(threshold))
    )
    return float(confirmed.mean())


def compare_with_night_baseline(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"psychology_accuracy": None, "night_accuracy": None, "edge_pp": None,
                "psychology_only_correct": 0, "night_only_correct": 0, "mcnemar_p_value": None}
    psych = frame["direction_hit"].astype(bool)
    night = frame["night_baseline_hit"].astype(bool)
    psych_only = int((psych & ~night).sum())
    night_only = int((~psych & night).sum())
    return {
        "psychology_accuracy": float(psych.mean()),
        "night_accuracy": float(night.mean()),
        "edge_pp": float((psych.mean() - night.mean()) * 100),
        "psychology_only_correct": psych_only,
        "night_only_correct": night_only,
        "mcnemar_p_value": exact_two_sided_binomial_p(psych_only, night_only),
    }


def render_report(report: dict) -> str:
    sample = report["sample"]
    overall = report["overall"]
    urgent = report["urgent_following"]
    urgent_bearish = report["urgent_bearish"]
    urgent_bullish = report["urgent_bullish"]
    urgent_bear_paths = {item["path"]: item for item in report["urgent_bearish_path_distribution"]}
    baseline = report["baselines"]
    calibration = report["weight_calibration"]
    urgent_baseline = report["urgent_baseline_comparison"]
    ci = overall["direction_wilson_95"]
    lines = [
        "# 群眾心理路徑模型歷史回測",
        "",
        f"資料期間：{sample['first_date']} 至 {sample['last_date']}，共 {sample['cases']} 個夜盤—日盤配對。",
        "",
        "## 結論",
        "",
        f"- 心理方向對日盤收盤方向命中率：{pct(overall['direction_accuracy'])}（{overall['actionable_cases']} 件；95% Wilson 區間 {pct(ci[0])}–{pct(ci[1])}）。",
        f"- 心理方向對開盤缺口方向命中率：{pct(overall['gap_direction_accuracy'])}。",
        f"- 同方向且收盤幅度至少 0.5%：{pct(overall['material_direction_confirmation_rate'])}。",
        f"- 嚴格首選路徑命中率：{pct(overall['top_path_hit_rate'])}。",
        f"- 三條價格路徑涵蓋率：{pct(overall['path_coverage'])}；此值不是方向成功率。",
        f"- 急迫追隨狀態方向命中率：{pct(urgent['direction_accuracy'])}（{urgent['actionable_cases']} 件）；其夜盤方向基準同為 {pct(urgent_baseline['night_accuracy'])}。",
        f"- 急迫偏空：{pct(urgent_bearish['direction_accuracy'])}（{urgent_bearish['actionable_cases']} 件）；急迫偏多：{pct(urgent_bullish['direction_accuracy'])}（{urgent_bullish['actionable_cases']} 件）。",
        f"- 急迫偏空且收跌至少 0.5%：{pct(urgent_bearish['material_direction_confirmation_rate'])}；平均日盤收盤報酬 {pct(urgent_bearish['average_cash_close_return'])}。",
        f"- 急迫偏空實際路徑：嚴格續跌 {pct(urgent_bear_paths.get('bearish_continuation', {}).get('rate'))}、空單回補 {pct(urgent_bear_paths.get('short_covering_rebound', {}).get('rate'))}、反向重置 {pct(urgent_bear_paths.get('path_reset_reversal', {}).get('rate'))}、未分類 {pct(urgent_bear_paths.get('unresolved', {}).get('rate'))}。",
        "",
        "## 基準比較",
        "",
        f"- 單用夜盤方向：{pct(baseline['night_direction_accuracy_on_same_actionable_cases'])}。",
        f"- 完整心理模型：{pct(baseline['psychology_direction_accuracy_on_same_actionable_cases'])}。",
        f"- 差異：{baseline['psychology_minus_night_percentage_points']:.2f} 個百分點；McNemar exact p={baseline['mcnemar_exact_p_value']:.4f}。",
        f"- 無條件多數方向基準：{pct(baseline['unconditional_majority_direction_accuracy'])}。",
        "",
        "## 時間穩定性",
        "",
        "| 區間 | 樣本 | 方向命中 | 開盤缺口命中 | 嚴格首選路徑 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in ["early_period", "recent_period"]:
        item = report[key]
        lines.append(
            f"| {item['label']} | {item['actionable_cases']} | {pct(item['direction_accuracy'])} | "
            f"{pct(item['gap_direction_accuracy'])} | {pct(item['top_path_hit_rate'])} |"
        )
    lines.extend(["", "## 各心理狀態", "", "| 狀態 | 樣本 | 方向命中 | 缺口命中 | 首選路徑 |", "| --- | ---: | ---: | ---: | ---: |"])
    for item in report["by_state"]:
        lines.append(
            f"| {item['group']} | {item['actionable_cases']} | {pct(item['direction_accuracy'])} | "
            f"{pct(item['gap_direction_accuracy'])} | {pct(item['top_path_hit_rate'])} |"
        )
    lines.extend(["", "## 心理方向權重時間切分", "", "| 版本 | 校準期命中 | 2023+時間留出命中 | 全歷史命中 |", "| --- | ---: | ---: | ---: |"])
    for name, item in calibration["configurations"].items():
        periods = item["periods"]
        lines.append(
            f"| {name} | {pct(periods['calibration_2017_2022']['accuracy'])} | "
            f"{pct(periods['temporal_holdout_2023_plus']['accuracy'])} | {pct(periods['all_history']['accuracy'])} |"
        )
    lines.extend([
        "",
        "## 驗證限制",
        "",
        f"- {report['method']['warning']}",
        "- 外部資料使用嚴格前一日期，避免把台股收盤後的同日美股資料偷渡進盤前判斷。",
        "- 首選路徑採嚴格收盤確認；盤中短暫跌破或突破後收回，不算首選路徑成功。",
        "- 急迫追隨的高命中率與夜盤基準相同，不能歸功於新增心理加權。",
        "- 病歷庫已接入廣度與未平倉量，但各來源覆蓋期和新鮮度不同；擁擠／宣洩需另按同步樣本驗證。",
        "- 真正可升格的成功率必須來自模型凍結後的新資料前瞻追蹤。",
        "",
    ])
    return "\n".join(lines)


def direction_matches(predicted: str, actual: str) -> bool:
    return predicted in {"bearish", "bullish"} and predicted == actual


def sign_label(value: float) -> str:
    if value > 0:
        return "bullish"
    if value < 0:
        return "bearish"
    return "mixed"


def mean_bool(frame: pd.DataFrame, column: str):
    return None if frame.empty else float(frame[column].astype(bool).mean())


def ratio(numerator: int, denominator: int):
    return None if denominator == 0 else float(numerator / denominator)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def exact_two_sided_binomial_p(first: int, second: int) -> float:
    total = first + second
    if total == 0:
        return 1.0
    cutoff = min(first, second)
    lower = sum(math.comb(total, k) for k in range(cutoff + 1)) / (2 ** total)
    return min(1.0, 2 * lower)


def number_or_none(value):
    return None if value is None or pd.isna(value) else float(value)


def pct(value) -> str:
    return "NA" if value is None or pd.isna(value) else f"{float(value):.2%}"


def rooted(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
