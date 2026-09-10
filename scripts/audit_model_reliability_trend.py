from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "processed"
CONFIG = ROOT / "config"
THRESHOLD = 0.005


@dataclass
class Score:
    checked: int
    hit: int
    miss: int
    hit_rate: float | None


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def direction(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value > THRESHOLD:
        return "up"
    if value < -THRESHOLD:
        return "down"
    return "sideways"


def score(rows: list[dict[str, Any]]) -> Score:
    checked = len(rows)
    hit = sum(1 for row in rows if row["hit"])
    miss = checked - hit
    return Score(checked=checked, hit=hit, miss=miss, hit_rate=(hit / checked if checked else None))


def split_thirds(rows: list[dict[str, Any]]) -> dict[str, Score]:
    if not rows:
        return {}
    chunk = max(1, len(rows) // 3)
    return {
        "early": score(rows[:chunk]),
        "middle": score(rows[chunk : 2 * chunk]),
        "late": score(rows[2 * chunk :]),
    }


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def as_dict(item: Score) -> dict[str, Any]:
    return {
        "checked": item.checked,
        "hit": item.hit,
        "miss": item.miss,
        "hit_rate": item.hit_rate,
    }


def is_after_close_same_day_replay(run_at: str | None, forecast_date: str, target_date: str) -> bool:
    if not run_at or forecast_date != target_date:
        return False
    try:
        run_time = datetime.fromisoformat(str(run_at)).time()
    except (TypeError, ValueError):
        return False
    return run_time >= time(13, 35)


def main() -> None:
    history = load_json(REPORTS / "forecast_history.json", [])
    daily = pd.read_csv(DATA / "twii_daily.csv", parse_dates=["date"]).sort_values("date")
    daily = daily.drop_duplicates("date", keep="last").reset_index(drop=True)
    dates = [d.date() for d in daily["date"]]
    close_by_date = dict(zip(dates, daily["close"]))
    date_index = {date: idx for idx, date in enumerate(dates)}
    latest_date = dates[-1]

    evaluated: list[dict[str, Any]] = []
    for record in history:
        signal_date_raw = record.get("signal_trade_date") or record.get("signal_date") or record.get("date")
        if not signal_date_raw:
            continue
        signal_date = datetime.fromisoformat(str(signal_date_raw)[:10]).date()
        if signal_date not in date_index:
            continue
        base_close = close_by_date.get(signal_date)
        for forecast in record.get("forecasts", []):
            horizon = int(forecast.get("horizon") or forecast.get("horizon_days") or 0)
            target_idx = date_index[signal_date] + horizon
            if horizon <= 0 or target_idx >= len(dates) or base_close in (None, 0):
                continue
            target_date = dates[target_idx]
            forecast_date = str(record.get("input_date") or record.get("date") or "")
            if forecast_date and forecast_date > target_date.isoformat():
                continue
            if is_after_close_same_day_replay(record.get("run_at"), forecast_date, target_date.isoformat()):
                continue
            actual_return = (close_by_date[target_date] - base_close) / base_close
            actual_dir = direction(actual_return)
            predicted_dir = str(forecast.get("direction") or forecast.get("predicted_direction") or "unknown")
            evaluated.append(
                {
                    "run_at": record.get("run_at") or record.get("created_at") or "",
                    "signal_date": signal_date.isoformat(),
                    "target_date": target_date.isoformat(),
                    "horizon": horizon,
                    "predicted": predicted_dir,
                    "actual": actual_dir,
                    "actual_return": actual_return,
                    "hit": predicted_dir == actual_dir,
                }
            )

    evaluated.sort(key=lambda row: (row["run_at"], row["signal_date"], row["horizon"]))
    by_horizon = {
        str(horizon): as_dict(score([row for row in evaluated if row["horizon"] == horizon]))
        for horizon in sorted({row["horizon"] for row in evaluated})
    }
    thirds_all = {name: as_dict(item) for name, item in split_thirds(evaluated).items()}
    one_day = [row for row in evaluated if row["horizon"] == 1]
    thirds_1d = {name: as_dict(item) for name, item in split_thirds(one_day).items()}

    recent_windows = {}
    for size in (10, 20, 30):
        recent_windows[str(size)] = as_dict(score(evaluated[-size:])) if evaluated else as_dict(Score(0, 0, 0, None))

    error_review = load_json(REPORTS / "error_review.json", {})
    validation = load_json(CONFIG / "model_validation_status.json", {})
    psychology = load_json(REPORTS / "psychology_state_backtest.json", {})
    night_cash = load_json(REPORTS / "night_cash_impact_tracking.json", {})
    research_stats = load_json(REPORTS / "periodic_deep_market_research.json", {})

    formal = validation.get("research_results", {}).get("best_model", {}) or validation.get("best_formal_result", {})
    validated = validation.get("validated_submodels", {}) or validation.get("validated_models", {})
    psych_overall = psychology.get("overall", {})
    night_rolling = night_cash.get("rolling_alignment", {}) or night_cash.get("rolling", {})
    pulse = research_stats.get("patterns", {}).get("endogenous_regulation_pulse", {})
    if not pulse:
        pulse = next(
            (
                item
                for item in research_stats.get("laws", [])
                if item.get("code") == "endogenous_regulation_pulse"
            ),
            {},
        )
    pulse_best = pulse.get("best_horizon", {}) if pulse else {}

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_market_date": latest_date.isoformat(),
        "threshold": THRESHOLD,
        "forecast_history_records": len(history),
        "evaluated_forecasts": len(evaluated),
        "overall": as_dict(score(evaluated)),
        "by_horizon": by_horizon,
        "chronological_thirds_all": thirds_all,
        "chronological_thirds_1d": thirds_1d,
        "recent_windows": recent_windows,
        "current_error_review": {
            "recent_checked": error_review.get("total_checked_recent"),
            "hit_rate": error_review.get("hit_rate"),
            "miss_count": error_review.get("miss_count"),
        },
        "production_gate": {
            "passed": validation.get("passed"),
            "production_multi_day_direction_enabled": validation.get("production_multi_day_direction_enabled"),
            "best_model_accuracy": formal.get("accuracy"),
            "baseline_accuracy": formal.get("baseline_accuracy") or formal.get("baseline"),
            "best_model_name": formal.get("model") or formal.get("method"),
        },
        "validated_submodels": validated,
        "psychology_backtest": {
            "overall_direction_accuracy": psych_overall.get("direction_accuracy"),
            "urgent_following": psychology.get("states", {}).get("urgent_following", {}),
        },
        "night_cash_alignment": night_rolling,
        "endogenous_regulation_pulse": pulse,
        "last_10_evaluated": evaluated[-10:],
    }
    (REPORTS / "model_reliability_trend_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 模型可靠度趨勢稽核",
        "",
        f"- 產生時間: {payload['generated_at']}",
        f"- 最新大盤資料日: {latest_date.isoformat()}",
        f"- 評估門檻: 漲跌超過 {THRESHOLD * 100:.2f}% 才算方向，上下未超過門檻算盤整",
        f"- 預測紀錄數: {len(history)}",
        f"- 已成熟可驗證預測數: {len(evaluated)}",
        f"- 整體命中率: {payload['overall']['hit']}/{payload['overall']['checked']} = {pct(payload['overall']['hit_rate'])}",
        "",
        "## 逐段可靠度",
    ]
    for name, label in (("early", "前段"), ("middle", "中段"), ("late", "後段")):
        item = thirds_all.get(name, {})
        lines.append(f"- {label}: {item.get('hit', 0)}/{item.get('checked', 0)} = {pct(item.get('hit_rate'))}")
    lines.extend(["", "## 近期香港命中率"])
    for size, item in recent_windows.items():
        lines.append(f"- 最近 {size} 筆: {item['hit']}/{item['checked']} = {pct(item['hit_rate'])}")
    lines.extend(["", "## 1日預測逐段"])
    for name, label in (("early", "前段"), ("middle", "中段"), ("late", "後段")):
        item = thirds_1d.get(name, {})
        lines.append(f"- {label}: {item.get('hit', 0)}/{item.get('checked', 0)} = {pct(item.get('hit_rate'))}")
    lines.extend(
        [
            "",
            "## 正式驗證狀態",
            f"- 多日方向正式上線: {validation.get('production_multi_day_direction_enabled')}",
            f"- 生產閘門通過: {validation.get('passed')}",
            f"- 最佳正式模型: {formal.get('model') or formal.get('method')}，準確率 {pct(formal.get('accuracy'))}，基準 {pct(formal.get('baseline_accuracy') or formal.get('baseline'))}",
            "",
            "## 子模組可靠度",
        ]
    )
    for name, item in validated.items():
        lines.append(f"- {name}: scope={item.get('scope')}，accuracy={pct(item.get('accuracy'))}，cases={item.get('cases')}")
    lines.extend(
        [
            f"- 心理狀態回測整體方向: {pct(psych_overall.get('direction_accuracy'))}",
            f"- 內生調節脈動: hit_rate={pct(pulse_best.get('hit_rate') or pulse.get('best_hit_rate'))}，cases={pulse.get('cases') or pulse.get('samples')}，passed={pulse.get('passed')}",
            f"- 夜盤對日盤開盤20/60/252日對齊: {pct((night_rolling.get('20') or {}).get('night_to_gap_alignment'))} / {pct((night_rolling.get('60') or {}).get('night_to_gap_alignment'))} / {pct((night_rolling.get('252') or {}).get('night_to_gap_alignment'))}",
            f"- 夜盤對日盤收盤20/60/252日對齊: {pct((night_rolling.get('20') or {}).get('night_to_close_alignment'))} / {pct((night_rolling.get('60') or {}).get('night_to_close_alignment'))} / {pct((night_rolling.get('252') or {}).get('night_to_close_alignment'))}",
            "",
            "## 結論",
            "- 診斷與風控可靠度已有提升，因為夜盤/日盤、心理狀態、內生調節、資料時點稽查已拆成可驗證模組。",
            "- 純粹多日方向預測尚未證明逐次穩定提高；正式生產閘門仍未通過，不能升級成投資命令。",
        ]
    )
    (REPORTS / "model_reliability_trend_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
