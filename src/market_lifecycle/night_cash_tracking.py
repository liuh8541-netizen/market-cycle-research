from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_night_cash_impact(night: pd.DataFrame, cash: pd.DataFrame) -> pd.DataFrame:
    night = night.copy()
    cash = cash.copy()
    night["signal_date"] = pd.to_datetime(night["signal_date"], errors="coerce")
    cash["date"] = pd.to_datetime(cash["date"], errors="coerce")
    for column in ["tx_night_open", "tx_night_high", "tx_night_low", "tx_night_close", "tx_night_return", "tx_night_range", "tx_night_spread_per"]:
        night[column] = pd.to_numeric(night.get(column), errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        cash[column] = pd.to_numeric(cash.get(column), errors="coerce")

    cash = cash.sort_values("date").drop_duplicates("date", keep="last")
    cash["prior_close"] = cash["close"].shift(1)
    joined = night.sort_values("signal_date").drop_duplicates("signal_date", keep="last").merge(
        cash, left_on="signal_date", right_on="date", how="inner", validate="one_to_one"
    )
    joined = joined.dropna(subset=["prior_close", "open", "high", "low", "close", "tx_night_return"])
    joined["gap_return"] = joined["open"] / joined["prior_close"] - 1
    joined["cash_close_return"] = joined["close"] / joined["prior_close"] - 1
    joined["intraday_return"] = joined["close"] / joined["open"] - 1
    joined["cash_high_return"] = joined["high"] / joined["prior_close"] - 1
    joined["cash_low_return"] = joined["low"] / joined["prior_close"] - 1
    joined["cash_range"] = joined["high"] / joined["low"] - 1

    sign = lambda values: values.ge(0).map({True: 1, False: -1})
    joined["night_direction"] = sign(joined["tx_night_return"])
    joined["gap_direction"] = sign(joined["gap_return"])
    joined["cash_close_direction"] = sign(joined["cash_close_return"])
    joined["intraday_direction"] = sign(joined["intraday_return"])
    joined["night_gap_aligned"] = joined["night_direction"].eq(joined["gap_direction"])
    joined["night_close_aligned"] = joined["night_direction"].eq(joined["cash_close_direction"])
    joined["gap_intraday_continuation"] = joined["gap_direction"].eq(joined["intraday_direction"])
    joined["gap_reversed_by_close"] = joined["gap_direction"].ne(joined["cash_close_direction"])
    joined["sync_status"] = [
        _sync_status(gap_aligned, close_aligned)
        for gap_aligned, close_aligned in zip(joined["night_gap_aligned"], joined["night_close_aligned"])
    ]
    reason_pairs = [
        _divergence_reason(row)
        for _, row in joined.iterrows()
    ]
    joined["reason_code"] = [item[0] for item in reason_pairs]
    joined["reason_summary"] = [item[1] for item in reason_pairs]
    strategy_pairs = [_short_squeeze_watch(row) for _, row in joined.iterrows()]
    joined["short_squeeze_status"] = [item[0] for item in strategy_pairs]
    joined["short_squeeze_summary"] = [item[1] for item in strategy_pairs]
    joined["night_abs_move"] = joined["tx_night_return"].abs()
    joined["night_strength"] = pd.cut(
        joined["night_abs_move"],
        bins=[-float("inf"), 0.003, 0.007, float("inf")],
        labels=["small", "medium", "large"],
    ).astype(str)
    joined["cash_volume_available"] = joined["volume"].gt(0)
    joined["tracking_version"] = "night_cash_impact_v1"

    ordered = [
        "signal_date", "night_date", "contract_date", "tx_night_open", "tx_night_high",
        "tx_night_low", "tx_night_close", "tx_night_volume", "tx_night_return",
        "tx_night_range", "tx_night_spread_per", "prior_close", "open", "high", "low",
        "close", "volume", "gap_return", "cash_close_return", "intraday_return",
        "cash_high_return", "cash_low_return", "cash_range", "night_direction",
        "gap_direction", "cash_close_direction", "intraday_direction", "night_gap_aligned",
        "night_close_aligned", "gap_intraday_continuation", "gap_reversed_by_close",
        "sync_status", "reason_code", "reason_summary", "night_strength",
        "short_squeeze_status", "short_squeeze_summary", "cash_volume_available",
        "tracking_version",
    ]
    result = joined[ordered].sort_values("signal_date").reset_index(drop=True)
    result["signal_date"] = result["signal_date"].dt.strftime("%Y-%m-%d")
    return result


def summarize_night_cash_impact(frame: pd.DataFrame, windows=(20, 60, 252)) -> dict:
    summaries = {}
    for window in windows:
        sample = frame.tail(window)
        summaries[str(window)] = {
            "cases": int(len(sample)),
            "night_to_gap_alignment": _mean(sample, "night_gap_aligned"),
            "night_to_close_alignment": _mean(sample, "night_close_aligned"),
            "gap_intraday_continuation": _mean(sample, "gap_intraday_continuation"),
            "gap_reversal_by_close": _mean(sample, "gap_reversed_by_close"),
            "average_abs_night_move": _mean_abs(sample, "tx_night_return"),
            "average_abs_gap": _mean_abs(sample, "gap_return"),
            "average_abs_cash_close_move": _mean_abs(sample, "cash_close_return"),
        }
    latest = frame.iloc[-1].to_dict() if len(frame) else {}
    recent = frame.tail(20)
    status_counts = {
        str(key): int(value) for key, value in recent["sync_status"].value_counts().items()
    } if len(recent) else {}
    reason_counts = {
        str(key): int(value) for key, value in recent.loc[
            recent["sync_status"].ne("full_sync"), "reason_summary"
        ].value_counts().items()
    } if len(recent) else {}
    return {
        "tracking_version": "night_cash_impact_v1",
        "scope": "completed TX night session aligned to the same signal-date TAIEX cash session",
        "interpretation_guardrail": "Alignment is descriptive transmission tracking, not proof of causality or a trading signal.",
        "rows": int(len(frame)),
        "first_date": frame.iloc[0]["signal_date"] if len(frame) else None,
        "latest_date": frame.iloc[-1]["signal_date"] if len(frame) else None,
        "latest": _json_safe(latest),
        "recent_20_status_counts": status_counts,
        "recent_20_divergence_reasons": reason_counts,
        "rolling": summaries,
    }


def write_night_cash_tracking(
    night_path: Path, cash_path: Path, csv_path: Path, json_path: Path, md_path: Path
) -> dict:
    frame = build_night_cash_impact(pd.read_csv(night_path), pd.read_csv(cash_path))
    summary = summarize_night_cash_impact(frame)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_night_cash_tracking(summary), encoding="utf-8")
    return summary


def render_night_cash_tracking(summary: dict) -> str:
    latest = summary.get("latest", {})
    lines = [
        "# 夜盤與日盤影響追蹤",
        "",
        f"資料期間：{summary.get('first_date')} 至 {summary.get('latest_date')}，共 {summary.get('rows', 0)} 組配對。",
        "",
        "## 最新一日",
        "",
        f"- 日期：{latest.get('signal_date', 'NA')}",
        f"- 夜盤報酬：{_pct(latest.get('tx_night_return'))}；日盤缺口：{_pct(latest.get('gap_return'))}",
        f"- 日盤收盤報酬：{_pct(latest.get('cash_close_return'))}；開盤至收盤：{_pct(latest.get('intraday_return'))}",
        f"- 夜盤方向傳到開盤：{'是' if latest.get('night_gap_aligned') else '否'}",
        f"- 夜盤方向傳到收盤：{'是' if latest.get('night_close_aligned') else '否'}",
        f"- 缺口被收盤反轉：{'是' if latest.get('gap_reversed_by_close') else '否'}",
        f"- 同步判定：{_status_label(latest.get('sync_status'))}",
        f"- 路徑原因：{latest.get('reason_summary', '資料不足')}",
        f"- 誘空／軋空觀察：{latest.get('short_squeeze_summary', '資料不足')}",
        "",
        "## 滾動統計",
        "",
        "| 期間 | 樣本 | 夜盤→缺口一致 | 夜盤→收盤一致 | 缺口日內延續 | 缺口收盤反轉 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in summary.get("rolling", {}).items():
        lines.append(
            f"| {name}日 | {item['cases']} | {_pct(item['night_to_gap_alignment'])} | "
            f"{_pct(item['night_to_close_alignment'])} | {_pct(item['gap_intraday_continuation'])} | "
            f"{_pct(item['gap_reversal_by_close'])} |"
        )
    lines.extend(["", "## 最近20日不同步原因", ""])
    reasons = summary.get("recent_20_divergence_reasons", {})
    if reasons:
        for reason, count in reasons.items():
            lines.append(f"- {reason}：{count}次")
    else:
        lines.append("- 無不同步紀錄")
    lines.extend(["", f"> {summary.get('interpretation_guardrail')}", ""])
    return "\n".join(lines)


def _mean(frame: pd.DataFrame, column: str):
    value = frame[column].mean() if len(frame) and column in frame else None
    return None if pd.isna(value) else float(value)


def _sync_status(gap_aligned: bool, close_aligned: bool) -> str:
    if gap_aligned and close_aligned:
        return "full_sync"
    if gap_aligned:
        return "open_only"
    if close_aligned:
        return "delayed_sync"
    return "full_divergence"


def _divergence_reason(row: pd.Series) -> tuple[str, str]:
    status = _sync_status(bool(row["night_gap_aligned"]), bool(row["night_close_aligned"]))
    night_move = abs(float(row["tx_night_return"]))
    gap_move = abs(float(row["gap_return"]))
    if status == "full_sync":
        return "night_transmitted", "夜盤方向由開盤延續至收盤"
    if status == "open_only":
        return "cash_session_reversal", "夜盤先影響開盤，但日盤交易時段反向並扭轉收盤"
    if status == "delayed_sync":
        return "delayed_cash_alignment", "開盤先與夜盤不同步，盤中資金再轉回夜盤方向"
    if night_move >= 0.007 and gap_move <= night_move * 0.35:
        return "preopen_repricing_compression", "夜盤波動較大，但開盤前重新定價使影響明顯壓縮"
    return "preopen_opposite_repricing", "日盤開盤前已出現反向重新定價，且收盤未回到夜盤方向"


def _status_label(status) -> str:
    return {
        "full_sync": "完全同步",
        "open_only": "只同步開盤，收盤反轉",
        "delayed_sync": "開盤不同步，收盤延後同步",
        "full_divergence": "全程不同步",
    }.get(status, "資料不足")


def _short_squeeze_watch(row: pd.Series) -> tuple[str, str]:
    if int(row["night_direction"]) >= 0:
        return "not_applicable", "夜盤不是下跌路徑，本項不啟動"
    night_low = float(row["tx_night_low"])
    night_close = float(row["tx_night_close"])
    cash_low = float(row["low"])
    cash_close = float(row["close"])
    if cash_low >= night_low:
        return "no_bear_trap_test", "日盤未跌破夜盤低點，尚未形成誘空測試"
    if cash_close >= night_close:
        return "close_reclaim_confirmed", "跌破夜盤低點後收回夜盤收盤，列為軋空確認候選"
    if cash_close >= night_low:
        return "low_reclaim_candidate", "跌破夜盤低點後收回該低點，列為多方誘空候選"
    return "downside_extension", "跌破夜盤低點且收盤未收回，空方延伸仍有效"


def _mean_abs(frame: pd.DataFrame, column: str):
    value = frame[column].abs().mean() if len(frame) and column in frame else None
    return None if pd.isna(value) else float(value)


def _pct(value) -> str:
    return "NA" if value is None or pd.isna(value) else f"{float(value):.2%}"


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
