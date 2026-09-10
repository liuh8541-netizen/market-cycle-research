from copy import deepcopy

import pandas as pd

from market_lifecycle.path_risk import historical_close_range, historical_path_distribution


DEFAULT_HORIZONS = [1, 5, 20, 60]

# Frozen from the 2017-2022 calibration segment.  The 2023+ segment remains
# untouched validation.  This overlay applies only to the next cash session;
# it must never be propagated to 5/20/60-day forecasts.
ONE_DAY_PSYCHOLOGY_OVERLAY = {
    "version": "one_day_psychology_overlay_v1",
    "minimum_absolute_score": 2,
    "calibration_period": "2017-05-16 through 2022-12-30",
    "bearish": {"up": 0.064865, "down": 0.632432, "sideways": 0.302703},
    "bullish": {"up": 0.510511, "down": 0.075075, "sideways": 0.414414},
    "validation": {
        "baseline_2023_plus_accuracy": 0.4108,
        "reconciled_2023_plus_accuracy": 0.5572,
        "baseline_2023_plus_macro_recall": 0.3677,
        "reconciled_2023_plus_macro_recall": 0.5448,
    },
}


def reconcile_one_day_forecast(forecast: dict, psychology_state: dict) -> dict:
    """Reconcile the weak generic one-day vote with a calibrated psychology state.

    The original historical distribution is retained in every case.  Strong
    psychology scores replace only the one-day distribution using frozen
    calibration frequencies.  Weak scores never receive a directional
    override; a low-close/poor-recovery night can only add a downside watch.
    """
    output = deepcopy(forecast)
    direction = psychology_state.get("direction")
    try:
        score = float(psychology_state.get("calibrated_direction_score"))
    except (TypeError, ValueError):
        score = 0.0
    diagnostics = psychology_state.get("night_diagnostics") or {}
    close_position = _safe_float(diagnostics.get("close_position"))
    recovery = _safe_float(diagnostics.get("recovery_from_low"))
    weak_bearish_tail = bool(
        direction == "bearish"
        and close_position is not None and close_position <= 0.25
        and recovery is not None and recovery <= 0.35
    )
    eligible = bool(
        direction in {"bearish", "bullish"}
        and abs(score) >= ONE_DAY_PSYCHOLOGY_OVERLAY["minimum_absolute_score"]
    )

    for item in output.get("forecasts", []):
        if int(item.get("horizon_days") or 0) != 1:
            continue
        original = {
            "predicted_direction": item.get("predicted_direction"),
            "probability_up": item.get("probability_up"),
            "probability_down": item.get("probability_down"),
            "probability_sideways": item.get("probability_sideways"),
            "confidence": item.get("confidence"),
            "match_level": item.get("match_level"),
        }
        review = {
            "version": ONE_DAY_PSYCHOLOGY_OVERLAY["version"],
            "applied": eligible,
            "psychology_direction": direction,
            "calibrated_direction_score": score,
            "minimum_absolute_score": ONE_DAY_PSYCHOLOGY_OVERLAY["minimum_absolute_score"],
            "weak_bearish_tail_watch": weak_bearish_tail,
            "historical_distribution": original,
            "validation": ONE_DAY_PSYCHOLOGY_OVERLAY["validation"],
        }
        if eligible:
            probabilities = ONE_DAY_PSYCHOLOGY_OVERLAY[direction]
            mapped_direction = "down" if direction == "bearish" else "up"
            item.update({
                "predicted_direction": mapped_direction,
                "probability_up": probabilities["up"],
                "probability_down": probabilities["down"],
                "probability_sideways": probabilities["sideways"],
                "confidence": probabilities[mapped_direction],
                "match_level": ONE_DAY_PSYCHOLOGY_OVERLAY["version"],
            })
            review["reason"] = "心理方向分數達校準門檻，使用固定的一日條件分布。"
        elif weak_bearish_tail:
            review["reason"] = "分數未達覆寫門檻；夜盤收近低點且回收不足，只增加下行尾部警示。"
        else:
            review["reason"] = "心理分數未達固定門檻，保留原歷史分布。"
        item["fixed_factor_review"] = review
    return output


def forecast_from_history(
    scored: pd.DataFrame,
    target_date: str | None = None,
    horizons: list[int] | None = None,
    min_cases: int = 30,
) -> dict:
    horizons = horizons or DEFAULT_HORIZONS
    data = _prepare(scored)
    target_index = _target_index(data, target_date)
    target = data.loc[target_index]
    history = data.iloc[:target_index].copy()

    if history.empty:
        raise ValueError("Not enough prior history to forecast.")

    forecast = []
    for horizon in horizons:
        pool, match_level = _similar_history_pool(history, target, horizon, min_cases)
        forecast.append(_forecast_horizon(pool, target, horizon, match_level))

    return {
        "requested_date": target_date,
        "forecast_date": str(target["date"].date()),
        "close": float(target["close"]),
        "lifecycle_stage": target["lifecycle_stage"],
        "risk_regime": target["risk_regime"],
        "stage_score": float(target["stage_score"]),
        "score_bucket": target["score_bucket"],
        "forecasts": forecast,
    }


def simulate_forecasts(
    scored: pd.DataFrame,
    horizons: list[int] | None = None,
    min_cases: int = 30,
    warmup_days: int = 756,
    step_days: int = 5,
) -> dict:
    horizons = horizons or DEFAULT_HORIZONS
    data = _prepare(scored)
    rows = []

    max_horizon = max(horizons)
    for index in range(warmup_days, len(data) - max_horizon, step_days):
        history = data.iloc[:index].copy()
        target = data.loc[index]
        for horizon in horizons:
            pool, match_level = _similar_history_pool(history, target, horizon, min_cases)
            forecast = _forecast_horizon(
                pool, target, horizon, match_level, include_path_diagnostics=False
            )
            actual_return = data.loc[index + horizon, "close"] / target["close"] - 1
            actual_direction = _direction(actual_return)
            rows.append(
                {
                    "date": str(target["date"].date()),
                    "horizon_days": horizon,
                    "predicted_direction": forecast["predicted_direction"],
                    "actual_direction": actual_direction,
                    "is_hit": forecast["predicted_direction"] == actual_direction,
                    "confidence": forecast["confidence"],
                    "case_count": forecast["case_count"],
                    "match_level": match_level,
                    "actual_return": float(actual_return),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"summary": {}, "by_horizon": [], "rows": []}

    by_horizon = []
    for horizon, group in frame.groupby("horizon_days"):
        confident = group[group["confidence"] >= 0.45]
        by_horizon.append(
            {
                "horizon_days": int(horizon),
                "sample_count": int(len(group)),
                "hit_rate": float(group["is_hit"].mean()),
                "avg_confidence": float(group["confidence"].mean()),
                "confident_count": int(len(confident)),
                "confident_hit_rate": float(confident["is_hit"].mean()) if len(confident) else 0.0,
            }
        )

    summary_frame = pd.DataFrame(by_horizon)
    return {
        "summary": {
            "avg_hit_rate": float(summary_frame["hit_rate"].mean()),
            "best_horizon": _best_horizon(summary_frame, "hit_rate"),
            "avg_confident_hit_rate": float(summary_frame["confident_hit_rate"].mean()),
        },
        "by_horizon": by_horizon,
        "rows": rows,
    }


def _forecast_horizon(
    pool: pd.DataFrame, target: pd.Series, horizon: int, match_level: str,
    include_path_diagnostics: bool = True,
) -> dict:
    returns = pool["close"].shift(-horizon) / pool["close"] - 1
    directions = returns.apply(_direction)
    valid = directions[directions != "unknown"]
    counts = valid.value_counts()
    total = int(counts.sum())

    probabilities = {
        "up": float(counts.get("up", 0) / total) if total else 0.0,
        "down": float(counts.get("down", 0) / total) if total else 0.0,
        "sideways": float(counts.get("sideways", 0) / total) if total else 0.0,
    }
    predicted = max(probabilities, key=probabilities.get)

    output = {
        "horizon_days": int(horizon),
        "predicted_direction": predicted,
        "probability_up": probabilities["up"],
        "probability_down": probabilities["down"],
        "probability_sideways": probabilities["sideways"],
        "confidence": probabilities[predicted],
        "case_count": total,
        "match_level": match_level,
    }
    if horizon == 1 and include_path_diagnostics:
        output["close_range_forecast"] = historical_close_range(returns, target.get("close"))
        output["intraday_path_distribution"] = historical_path_distribution(pool)
    return output


def _similar_history_pool(
    history: pd.DataFrame,
    target: pd.Series,
    horizon: int,
    min_cases: int,
) -> tuple[pd.DataFrame, str]:
    usable = history.iloc[: max(0, len(history) - horizon)].copy()
    levels = [
        (
            "stage+risk+score",
            (usable["lifecycle_stage"] == target["lifecycle_stage"])
            & (usable["risk_regime"] == target["risk_regime"])
            & (usable["score_bucket"] == target["score_bucket"]),
        ),
        (
            "risk+score",
            (usable["risk_regime"] == target["risk_regime"])
            & (usable["score_bucket"] == target["score_bucket"]),
        ),
        ("risk", usable["risk_regime"] == target["risk_regime"]),
        ("all_history", pd.Series(True, index=usable.index)),
    ]

    for name, mask in levels:
        pool = usable[mask].copy()
        if len(pool) >= min_cases:
            return pool, name
    return usable, "all_history"


def _prepare(scored: pd.DataFrame) -> pd.DataFrame:
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    data["score_bucket"] = data["stage_score"].apply(_score_bucket)
    return data


def _target_index(data: pd.DataFrame, target_date: str | None) -> int:
    if target_date is None:
        return len(data) - 1
    target = pd.to_datetime(target_date)
    eligible = data[data["date"] <= target]
    if eligible.empty:
        raise ValueError("No market data is available on or before " + target_date)
    return int(eligible.index[-1])


def _score_bucket(value: float) -> str:
    if value >= 5:
        return "strong_positive"
    if value >= 2:
        return "positive"
    if value > -2:
        return "neutral"
    if value > -5:
        return "negative"
    return "strong_negative"


def _direction(value: float, neutral_threshold: float = 0.005) -> str:
    if pd.isna(value):
        return "unknown"
    if value > neutral_threshold:
        return "up"
    if value < -neutral_threshold:
        return "down"
    return "sideways"


def _best_horizon(frame: pd.DataFrame, metric: str) -> dict | None:
    if frame.empty:
        return None
    row = frame.sort_values(metric, ascending=False).iloc[0]
    return {"horizon_days": int(row["horizon_days"]), metric: float(row[metric])}


def _safe_float(value) -> float | None:
    try:
        return None if value is None or pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None
