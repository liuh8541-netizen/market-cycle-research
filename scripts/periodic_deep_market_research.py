"""Periodic deep research of repeatable Taiwan market patterns.

This script is intentionally conservative: it records candidate "laws" as
probabilistic, auditable patterns instead of promoting them to trading rules.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_lifecycle.external_market import add_external_market_features
from market_lifecycle.factor_features import add_factor_features
from market_lifecycle.features import add_features


TAIPEI = ZoneInfo("Asia/Taipei")
REPORT_JSON = ROOT / "reports" / "periodic_deep_market_research.json"
REPORT_MD = ROOT / "reports" / "periodic_deep_market_research.md"
DB_PATH = ROOT / "data" / "processed" / "market_research_stats.sqlite"
MIN_CASES = 30
HORIZONS = [1, 3, 5, 10, 20]


@dataclass(frozen=True)
class PatternSpec:
    code: str
    name: str
    hypothesis: str
    target: str
    active_if: str
    risk_if_failed: str


PATTERNS = [
    PatternSpec(
        code="night_down_cash_reclaim",
        name="夜跌日收回洗籌候選",
        hypothesis="夜盤下跌後，日盤若低檔收回，常代表恐慌被現貨承接，偏向洗籌/換手。",
        target="後續上漲或至少不續破",
        active_if="tx_night_spread_per < 0 且日盤收盤位置 >= 60% 且收盤不低於昨收 -0.3%",
        risk_if_failed="若隔日跌破當日低點，洗籌假設降級為夜盤領跌。",
    ),
    PatternSpec(
        code="night_down_cash_break",
        name="夜跌日破位確認",
        hypothesis="夜盤下跌後，日盤收近低且收跌，表示夜盤壓力被現貨確認。",
        target="後續偏弱或回測",
        active_if="tx_night_spread_per < 0 且日盤收盤位置 <= 25% 且收盤跌幅 <= -0.3%",
        risk_if_failed="若隔日快速站回開盤價，破位假設可能是假跌破。",
    ),
    PatternSpec(
        code="night_up_cash_fail",
        name="夜漲日弱現貨反證",
        hypothesis="夜盤上漲但日盤開高壓回，常代表誘多或高檔換手壓力。",
        target="後續盤整或回測",
        active_if="tx_night_spread_per > 0 且日盤收盤位置 <= 25% 且日盤收黑",
        risk_if_failed="若隔日站回高點，現貨反證失效，改看攻擊延續。",
    ),
    PatternSpec(
        code="gap_up_close_near_low",
        name="開高壓回收近低",
        hypothesis="開高後收近低，代表追價失敗；高檔時需提高隔日續弱監控。",
        target="後續盤整或回測",
        active_if="開盤跳空 >= 0.1%、收跌 >= 0.3%、收盤位置 <= 25%",
        risk_if_failed="若隔日紅K反包，代表只是日內換手。",
    ),
    PatternSpec(
        code="bottom_repair_higher_low",
        name="修復段低點墊高",
        hypothesis="回撤未深、站上月線附近且低點墊高，表示震卦修復仍有效。",
        target="後續偏強或區間墊高",
        active_if="收盤高於20日線、近5日低點高於近20日低點、距高點回撤小於10%",
        risk_if_failed="若跌破近5日低點且收不回，修復段降級。",
    ),
    PatternSpec(
        code="endogenous_regulation_pulse",
        name="內生調節脈動",
        hypothesis="無重大外部衝擊時，市場仍會透過拉回、收復、低點墊高與量能換手，自動調節大盤與績優股成本。",
        target="後續偏強或至少不續破",
        active_if="外部市場最大絕對波動 < 1.5%、日內下探後收復 >= 45%、收盤站上20日線、近5日低點高於近20日低點",
        risk_if_failed="若隔日跌破當日低點且收不回，內生調節假設降級為壓力測試失敗。",
    ),
]


def main() -> None:
    data = build_frame()
    prepared = add_pattern_columns(data)
    laws = [evaluate_pattern(prepared, spec) for spec in PATTERNS]
    day_night_variance_cases = build_day_night_variance_cases(prepared)
    payload = {
        "framework": "periodic_deep_market_research_v1",
        "created_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "data_start": str(prepared["date"].min().date()),
        "data_end": str(prepared["date"].max().date()),
        "rows": int(len(prepared)),
        "principle": "只記錄可回測候選規律；不自動升級為交易命令。",
        "requirements": {
            "min_cases": MIN_CASES,
            "horizons": HORIZONS,
            "passed_rule": "樣本數達標，且至少一個週期方向命中率 >= 60%，Wilson 95% 下限 >= 50%。",
        },
        "laws": laws,
        "active_laws": [law for law in laws if law["current"]["active"]],
        "research_queue": research_queue(laws),
        "day_night_variance_cases": day_night_variance_cases,
    }
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(render_report(payload), encoding="utf-8")
    write_database(payload, DB_PATH)
    print(json.dumps({"active": [x["name"] for x in payload["active_laws"]], "report": str(REPORT_MD), "database": str(DB_PATH)}, ensure_ascii=False, indent=2))


def build_frame() -> pd.DataFrame:
    price = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
    price = add_external_market_features(
        price,
        ROOT / "data" / "processed" / "external_markets.csv",
        ROOT / "data" / "processed" / "taiwan_futures_night.csv",
    )
    data = add_factor_features(add_features(price), str(ROOT / "data" / "processed" / "factors"))
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values("date").dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def add_pattern_columns(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    prev_close = output["close"].shift(1)
    daily_range = (output["high"] - output["low"]).replace(0, pd.NA)
    output["open_gap_pct"] = output["open"] / prev_close - 1
    output["close_return_pct"] = output["close"] / prev_close - 1
    output["intraday_body_pct"] = output["close"] / output["open"] - 1
    output["close_position"] = ((output["close"] - output["low"]) / daily_range).fillna(0.5)
    output["rolling_peak_252"] = output["close"].rolling(252, min_periods=60).max()
    output["drawdown_252"] = output["close"] / output["rolling_peak_252"] - 1
    output["low_5"] = output["low"].rolling(5, min_periods=5).min()
    output["low_20"] = output["low"].rolling(20, min_periods=20).min()
    output["above_ma20"] = output["close"] > output["ma_20"]
    external_cols = [
        col
        for col in [
            "sp500_return_1d",
            "nasdaq_return_1d",
            "sox_return_1d",
            "tsm_adr_return_1d",
            "ewt_return_1d",
            "usd_twd_return_1d",
        ]
        if col in output.columns
    ]
    if external_cols:
        output["external_abs_max"] = output[external_cols].apply(pd.to_numeric, errors="coerce").abs().max(axis=1)
    else:
        output["external_abs_max"] = pd.NA
    output["intraday_low_drawdown"] = output["low"] / prev_close - 1
    output["intraday_recovery_ratio"] = (
        (output["close"] - output["low"]) / (output["high"] - output["low"]).replace(0, pd.NA)
    ).fillna(0.5)
    for horizon in HORIZONS:
        output[f"fwd_{horizon}d_return"] = output["close"].shift(-horizon) / output["close"] - 1
        output[f"fwd_{horizon}d_low_dd"] = rolling_forward_low(output["low"], horizon) / output["close"] - 1
    return output


def build_day_night_variance_cases(data: pd.DataFrame) -> list[dict]:
    """Collect the raw footprints behind night/day divergence hypotheses."""
    cases: list[dict] = []
    night = pd.to_numeric(data.get("tx_night_spread_per"), errors="coerce")
    cash = pd.to_numeric(data.get("close_return_pct"), errors="coerce")
    gap = pd.to_numeric(data.get("open_gap_pct"), errors="coerce")
    intraday_range = (data["high"] - data["low"]) / data["close"].shift(1)
    close_position = pd.to_numeric(data.get("close_position"), errors="coerce")
    for index, row in data[night.notna() & cash.notna()].iterrows():
        night_return = safe_float(night.loc[index])
        cash_return = safe_float(cash.loc[index])
        gap_return = safe_float(gap.loc[index])
        range_pct = safe_float(intraday_range.loc[index])
        close_pos = safe_float(close_position.loc[index])
        variance_gap = abs(cash_return - night_return) if night_return is not None and cash_return is not None else None
        relation_code, relation_label = classify_day_night_relation(night_return, cash_return)
        causes = day_night_variance_causes(night_return, cash_return, gap_return, range_pct, close_pos)
        cases.append(
            {
                "date": str(pd.to_datetime(row["date"]).date()),
                "close": safe_float(row.get("close")),
                "night_return": night_return,
                "cash_return": cash_return,
                "gap_return": gap_return,
                "cash_range_pct": range_pct,
                "close_position": close_pos,
                "variance_gap": variance_gap,
                "relation_code": relation_code,
                "relation_label": relation_label,
                "cause_candidates": causes,
            }
        )
    return cases


def classify_day_night_relation(night_return: float | None, cash_return: float | None) -> tuple[str, str]:
    if night_return is None or cash_return is None:
        return "missing", "資料不足"
    if night_return >= 0.004 and cash_return > 0:
        return "night_up_cash_up_validation", "夜漲日漲確認"
    if night_return >= 0.004 and cash_return <= 0:
        return "night_up_cash_failed", "夜漲日弱反證"
    if night_return <= -0.004 and cash_return < 0:
        return "night_down_cash_down_validation", "夜跌日跌確認"
    if night_return <= -0.004 and cash_return >= 0:
        return "night_down_cash_reversal", "夜跌日收回變異"
    return "weak_night_signal", "夜盤訊號較弱"


def day_night_variance_causes(
    night_return: float | None,
    cash_return: float | None,
    gap_return: float | None,
    range_pct: float | None,
    close_position: float | None,
) -> list[str]:
    causes: list[str] = []
    if abs(gap_return or 0) >= 0.004:
        causes.append("開盤缺口消化夜盤預期")
    if range_pct is not None and range_pct >= 0.012:
        causes.append("日內高低差擴大")
    if range_pct is not None and close_position is not None and range_pct >= 0.010 and close_position >= 0.65:
        causes.append("下探後收復")
    if close_position is not None and close_position <= 0.30:
        causes.append("收近低點賣壓未解")
    if night_return is not None and cash_return is not None and night_return > 0 and cash_return < night_return - 0.005:
        causes.append("夜強日弱獲利了結/誘多失敗候選")
    if night_return is not None and cash_return is not None and night_return < 0 and cash_return > night_return + 0.005:
        causes.append("夜弱日強恐慌測試後承接候選")
    return causes or ["一般日夜落差"]


def rolling_forward_low(low: pd.Series, horizon: int) -> pd.Series:
    values = []
    for index in range(len(low)):
        window = low.iloc[index + 1 : index + 1 + horizon]
        values.append(float(window.min()) if not window.empty else math.nan)
    return pd.Series(values, index=low.index)


def evaluate_pattern(data: pd.DataFrame, spec: PatternSpec) -> dict:
    mask = pattern_mask(data, spec.code)
    sample = data[mask].copy()
    horizon_rows = [evaluate_horizon(sample, horizon, spec.target) for horizon in HORIZONS]
    active_row = data.iloc[-1]
    active = bool(pattern_mask(data.tail(1), spec.code).iloc[-1])
    best = max(horizon_rows, key=lambda row: row["edge_score"]) if horizon_rows else None
    passed = bool(best and best["cases"] >= MIN_CASES and best["hit_rate"] >= 0.60 and best["wilson_95_lower"] >= 0.50)
    return {
        "code": spec.code,
        "name": spec.name,
        "hypothesis": spec.hypothesis,
        "target": spec.target,
        "active_if": spec.active_if,
        "risk_if_failed": spec.risk_if_failed,
        "cases": int(len(sample)),
        "sample_start": str(sample["date"].min().date()) if len(sample) else None,
        "sample_end": str(sample["date"].max().date()) if len(sample) else None,
        "passed": passed,
        "best_horizon": best,
        "by_horizon": horizon_rows,
        "current": {
            "date": str(active_row["date"].date()),
            "active": active,
            "close": safe_float(active_row.get("close")),
            "tx_night_spread_per": safe_float(active_row.get("tx_night_spread_per")),
            "close_position": safe_float(active_row.get("close_position")),
            "close_return_pct": safe_float(active_row.get("close_return_pct")),
            "next_check": spec.risk_if_failed if active else "未觸發，僅保留歷史統計。",
        },
    }


def pattern_mask(data: pd.DataFrame, code: str) -> pd.Series:
    night = pd.to_numeric(data.get("tx_night_spread_per"), errors="coerce")
    close_pos = pd.to_numeric(data.get("close_position"), errors="coerce")
    close_ret = pd.to_numeric(data.get("close_return_pct"), errors="coerce")
    body = pd.to_numeric(data.get("intraday_body_pct"), errors="coerce")
    gap = pd.to_numeric(data.get("open_gap_pct"), errors="coerce")
    if code == "night_down_cash_reclaim":
        return night.lt(0) & close_pos.ge(0.60) & close_ret.ge(-0.003)
    if code == "night_down_cash_break":
        return night.lt(0) & close_pos.le(0.25) & close_ret.le(-0.003)
    if code == "night_up_cash_fail":
        return night.gt(0) & close_pos.le(0.25) & body.lt(0)
    if code == "gap_up_close_near_low":
        return gap.ge(0.001) & close_ret.le(-0.003) & close_pos.le(0.25)
    if code == "bottom_repair_higher_low":
        return data.get("above_ma20", False) & data["low_5"].gt(data["low_20"]) & data["drawdown_252"].gt(-0.10)
    if code == "endogenous_regulation_pulse":
        external_abs = pd.to_numeric(data.get("external_abs_max"), errors="coerce")
        recovery = pd.to_numeric(data.get("intraday_recovery_ratio"), errors="coerce")
        low_dd = pd.to_numeric(data.get("intraday_low_drawdown"), errors="coerce")
        return (
            external_abs.lt(0.015)
            & recovery.ge(0.45)
            & low_dd.le(-0.003)
            & data.get("above_ma20", False)
            & data["low_5"].gt(data["low_20"])
        )
    return pd.Series(False, index=data.index)


def evaluate_horizon(sample: pd.DataFrame, horizon: int, target: str) -> dict:
    returns = pd.to_numeric(sample.get(f"fwd_{horizon}d_return"), errors="coerce")
    low_dd = pd.to_numeric(sample.get(f"fwd_{horizon}d_low_dd"), errors="coerce")
    valid = returns.notna()
    returns = returns[valid]
    low_dd = low_dd[valid]
    cases = int(len(returns))
    if cases == 0:
        return empty_horizon(horizon)
    expects_weak = any(word in target for word in ["弱", "回測", "盤整"])
    if expects_weak:
        hits = int((returns <= 0.005).sum())
    else:
        hits = int((returns >= -0.005).sum())
    hit_rate = hits / cases
    baseline = max(float((returns >= -0.005).mean()), float((returns <= 0.005).mean()))
    return {
        "horizon_days": horizon,
        "cases": cases,
        "hits": hits,
        "hit_rate": hit_rate,
        "baseline": baseline,
        "edge_vs_baseline": hit_rate - baseline,
        "wilson_95_lower": wilson_lower(hits, cases),
        "avg_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "avg_max_low_drawdown": float(low_dd.mean()),
        "loss_gt_2pct_rate": float((returns <= -0.02).mean()),
        "edge_score": hit_rate + wilson_lower(hits, cases) + min(cases, 300) / 1000,
    }


def empty_horizon(horizon: int) -> dict:
    return {
        "horizon_days": horizon,
        "cases": 0,
        "hits": 0,
        "hit_rate": 0.0,
        "baseline": 0.0,
        "edge_vs_baseline": 0.0,
        "wilson_95_lower": 0.0,
        "avg_return": None,
        "median_return": None,
        "avg_max_low_drawdown": None,
        "loss_gt_2pct_rate": None,
        "edge_score": 0.0,
    }


def wilson_lower(hits: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = hits / total
    denom = 1 + z * z / total
    center = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (center - margin) / denom)


def research_queue(laws: list[dict]) -> list[dict]:
    queue = []
    for law in laws:
        best = law.get("best_horizon") or {}
        if law["cases"] < MIN_CASES:
            reason = "樣本不足，先累積。"
        elif not law["passed"]:
            reason = "命中率或 Wilson 下限不足，只保留觀察。"
        else:
            reason = "候選規律通過初步門檻，可列入每日監控，但仍非交易命令。"
        queue.append({
            "code": law["code"],
            "name": law["name"],
            "status": "active" if law["current"]["active"] else "inactive",
            "passed": law["passed"],
            "best_horizon": best.get("horizon_days"),
            "reason": reason,
        })
    return queue


def render_report(payload: dict) -> str:
    lines = [
        "# 週期性深度研究：股市規律機率考測",
        "",
        f"- 產生時間: {payload['created_at']}",
        f"- 資料期間: {payload['data_start']} ～ {payload['data_end']}",
        f"- 樣本列數: {payload['rows']}",
        f"- 原則: {payload['principle']}",
        f"- 統計資料庫: `{DB_PATH.relative_to(ROOT)}`",
        "",
        "## 目前觸發",
        "",
    ]
    if payload["active_laws"]:
        for law in payload["active_laws"]:
            current = law["current"]
            best = law.get("best_horizon") or {}
            lines.append(
                f"- {law['name']}: 收盤 {num(current['close'])}，夜盤 {pct(current['tx_night_spread_per'])}，"
                f"最佳週期 {best.get('horizon_days', 'NA')}日，命中率 {pct(best.get('hit_rate'))}，"
                f"Wilson {pct(best.get('wilson_95_lower'))}。"
            )
    else:
        lines.append("- 今日未觸發已登錄候選規律。")
    variance_cases = payload.get("day_night_variance_cases", [])
    relation_counts: dict[str, int] = {}
    for case in variance_cases:
        label = case.get("relation_label", "資料不足")
        relation_counts[label] = relation_counts.get(label, 0) + 1
    lines.extend([
        "",
        "## 日夜盤變異資料庫",
        "",
        f"- 已留底樣本: {len(variance_cases)} 筆",
        f"- 關係分布: {'；'.join(f'{key} {value}筆' for key, value in relation_counts.items()) or 'NA'}",
        "- 用途: 收集夜盤預期、日盤驗證、高低差與原因候選，供後續前瞻驗證，不作投資命令。",
        "",
        "| 日期 | 夜盤 | 日盤 | 缺口 | 高低差 | 收盤位置 | 關係 | 原因候選 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    for case in variance_cases[-15:]:
        lines.append(
            f"| {case.get('date')} | {pct(case.get('night_return'))} | {pct(case.get('cash_return'))} | "
            f"{pct(case.get('gap_return'))} | {pct(case.get('cash_range_pct'))} | "
            f"{pct(case.get('close_position'))} | {case.get('relation_label')} | "
            f"{'、'.join(case.get('cause_candidates', [])) or 'NA'} |"
        )
    lines.extend([
        "",
        "## 候選規律總表",
        "",
        "| 規律 | 樣本 | 通過 | 最佳週期 | 命中率 | Wilson下限 | 平均報酬 | 平均最大低點回撤 | 目前 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for law in payload["laws"]:
        best = law.get("best_horizon") or {}
        lines.append(
            f"| {law['name']} | {law['cases']} | {'是' if law['passed'] else '否'} | "
            f"{best.get('horizon_days', 'NA')} | {pct(best.get('hit_rate'))} | "
            f"{pct(best.get('wilson_95_lower'))} | {pct(best.get('avg_return'))} | "
            f"{pct(best.get('avg_max_low_drawdown'))} | {'觸發' if law['current']['active'] else '未觸發'} |"
        )
    lines.extend([
        "",
        "## 規律定義",
        "",
    ])
    for law in payload["laws"]:
        lines.extend([
            f"### {law['name']}",
            f"- 假設: {law['hypothesis']}",
            f"- 目標: {law['target']}",
            f"- 觸發: {law['active_if']}",
            f"- 失效: {law['risk_if_failed']}",
            "",
        ])
    lines.extend([
        "## 研究佇列",
        "",
        "| 規律 | 狀態 | 通過 | 下一步 |",
        "| --- | --- | --- | --- |",
    ])
    for item in payload["research_queue"]:
        lines.append(
            f"| {item['name']} | {item['status']} | {'是' if item['passed'] else '否'} | {item['reason']} |"
        )
    lines.extend([
        "",
        "## 使用限制",
        "",
        "- 這是規律機率考測，不是投資命令。",
        "- 樣本數、命中率、Wilson 下限三者必須同時看；單一高命中率但樣本太少不採信。",
        "- 每次新增資料後重新跑，若規律衰退，降級留底。",
    ])
    return "\n".join(lines) + "\n"


def write_database(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        create_schema(conn)
        run_id = payload["created_at"]
        conn.execute(
            """
            INSERT OR REPLACE INTO research_runs
            (run_id, created_at, data_start, data_end, rows, framework, principle)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                payload["created_at"],
                payload["data_start"],
                payload["data_end"],
                payload["rows"],
                payload["framework"],
                payload["principle"],
            ),
        )
        for law in payload["laws"]:
            best = law.get("best_horizon") or {}
            current = law.get("current") or {}
            conn.execute(
                """
                INSERT OR REPLACE INTO law_stats
                (run_id, law_code, law_name, hypothesis, target, active_if, risk_if_failed,
                 cases, sample_start, sample_end, passed, active_now, current_date,
                 current_close, current_night_spread, current_close_position,
                 current_close_return, best_horizon_days, best_hit_rate,
                 best_wilson_95_lower, best_avg_return, best_avg_max_low_drawdown)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    law["code"],
                    law["name"],
                    law["hypothesis"],
                    law["target"],
                    law["active_if"],
                    law["risk_if_failed"],
                    law["cases"],
                    law["sample_start"],
                    law["sample_end"],
                    int(bool(law["passed"])),
                    int(bool(current.get("active"))),
                    current.get("date"),
                    current.get("close"),
                    current.get("tx_night_spread_per"),
                    current.get("close_position"),
                    current.get("close_return_pct"),
                    best.get("horizon_days"),
                    best.get("hit_rate"),
                    best.get("wilson_95_lower"),
                    best.get("avg_return"),
                    best.get("avg_max_low_drawdown"),
                ),
            )
            for horizon in law.get("by_horizon", []):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO law_horizon_stats
                    (run_id, law_code, horizon_days, cases, hits, hit_rate, baseline,
                     edge_vs_baseline, wilson_95_lower, avg_return, median_return,
                     avg_max_low_drawdown, loss_gt_2pct_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        law["code"],
                        horizon["horizon_days"],
                        horizon["cases"],
                        horizon["hits"],
                        horizon["hit_rate"],
                        horizon["baseline"],
                        horizon["edge_vs_baseline"],
                        horizon["wilson_95_lower"],
                        horizon["avg_return"],
                        horizon["median_return"],
                        horizon["avg_max_low_drawdown"],
                        horizon["loss_gt_2pct_rate"],
                    ),
                )
        for item in payload.get("research_queue", []):
            conn.execute(
                """
                INSERT OR REPLACE INTO research_queue
                (run_id, law_code, law_name, status, passed, best_horizon_days, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    item["code"],
                    item["name"],
                    item["status"],
                    int(bool(item["passed"])),
                    item.get("best_horizon"),
                    item["reason"],
                ),
            )
        for item in payload.get("day_night_variance_cases", []):
            conn.execute(
                """
                INSERT OR REPLACE INTO day_night_variance_cases
                (run_id, date, close, night_return, cash_return, gap_return,
                 cash_range_pct, close_position, variance_gap, relation_code,
                 relation_label, cause_candidates_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    item["date"],
                    item.get("close"),
                    item.get("night_return"),
                    item.get("cash_return"),
                    item.get("gap_return"),
                    item.get("cash_range_pct"),
                    item.get("close_position"),
                    item.get("variance_gap"),
                    item.get("relation_code"),
                    item.get("relation_label"),
                    json.dumps(item.get("cause_candidates", []), ensure_ascii=False),
                ),
            )
        conn.commit()


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS research_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            data_start TEXT NOT NULL,
            data_end TEXT NOT NULL,
            rows INTEGER NOT NULL,
            framework TEXT NOT NULL,
            principle TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS law_stats (
            run_id TEXT NOT NULL,
            law_code TEXT NOT NULL,
            law_name TEXT NOT NULL,
            hypothesis TEXT NOT NULL,
            target TEXT NOT NULL,
            active_if TEXT NOT NULL,
            risk_if_failed TEXT NOT NULL,
            cases INTEGER NOT NULL,
            sample_start TEXT,
            sample_end TEXT,
            passed INTEGER NOT NULL,
            active_now INTEGER NOT NULL,
            current_date TEXT,
            current_close REAL,
            current_night_spread REAL,
            current_close_position REAL,
            current_close_return REAL,
            best_horizon_days INTEGER,
            best_hit_rate REAL,
            best_wilson_95_lower REAL,
            best_avg_return REAL,
            best_avg_max_low_drawdown REAL,
            PRIMARY KEY (run_id, law_code),
            FOREIGN KEY (run_id) REFERENCES research_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS law_horizon_stats (
            run_id TEXT NOT NULL,
            law_code TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            cases INTEGER NOT NULL,
            hits INTEGER NOT NULL,
            hit_rate REAL NOT NULL,
            baseline REAL NOT NULL,
            edge_vs_baseline REAL NOT NULL,
            wilson_95_lower REAL NOT NULL,
            avg_return REAL,
            median_return REAL,
            avg_max_low_drawdown REAL,
            loss_gt_2pct_rate REAL,
            PRIMARY KEY (run_id, law_code, horizon_days),
            FOREIGN KEY (run_id, law_code) REFERENCES law_stats(run_id, law_code)
        );

        CREATE TABLE IF NOT EXISTS research_queue (
            run_id TEXT NOT NULL,
            law_code TEXT NOT NULL,
            law_name TEXT NOT NULL,
            status TEXT NOT NULL,
            passed INTEGER NOT NULL,
            best_horizon_days INTEGER,
            reason TEXT NOT NULL,
            PRIMARY KEY (run_id, law_code),
            FOREIGN KEY (run_id) REFERENCES research_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS day_night_variance_cases (
            run_id TEXT NOT NULL,
            date TEXT NOT NULL,
            close REAL,
            night_return REAL,
            cash_return REAL,
            gap_return REAL,
            cash_range_pct REAL,
            close_position REAL,
            variance_gap REAL,
            relation_code TEXT NOT NULL,
            relation_label TEXT NOT NULL,
            cause_candidates_json TEXT NOT NULL,
            PRIMARY KEY (run_id, date),
            FOREIGN KEY (run_id) REFERENCES research_runs(run_id)
        );

        CREATE VIEW IF NOT EXISTS latest_law_stats AS
        SELECT s.*
        FROM law_stats s
        JOIN (SELECT max(created_at) AS created_at FROM research_runs) latest
        JOIN research_runs r ON r.run_id = s.run_id AND r.created_at = latest.created_at;

        CREATE VIEW IF NOT EXISTS latest_law_horizon_stats AS
        SELECT h.*
        FROM law_horizon_stats h
        JOIN (SELECT max(created_at) AS created_at FROM research_runs) latest
        JOIN research_runs r ON r.run_id = h.run_id AND r.created_at = latest.created_at;

        CREATE VIEW IF NOT EXISTS latest_day_night_variance_cases AS
        SELECT c.*
        FROM day_night_variance_cases c
        JOIN (SELECT max(created_at) AS created_at FROM research_runs) latest
        JOIN research_runs r ON r.run_id = c.run_id AND r.created_at = latest.created_at;

        CREATE INDEX IF NOT EXISTS idx_law_stats_code ON law_stats(law_code, run_id);
        CREATE INDEX IF NOT EXISTS idx_law_horizon_code ON law_horizon_stats(law_code, horizon_days, run_id);
        CREATE INDEX IF NOT EXISTS idx_day_night_variance_relation ON day_night_variance_cases(relation_code, date);
        """
    )


def safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def pct(value) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.2%}"
    except Exception:
        return "NA"


def num(value) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):,.0f}"
    except Exception:
        return "NA"


if __name__ == "__main__":
    main()
