import pandas as pd


DEFAULT_HORIZONS = [1, 5, 10, 20, 60]


def evaluate_prediction_success(
    scored: pd.DataFrame,
    horizons: list[int] | None = None,
    lookback_days: int = 20,
) -> dict:
    horizons = horizons or DEFAULT_HORIZONS
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])

    horizon_reports = []
    for horizon in horizons:
        horizon_reports.append(_evaluate_horizon(data, horizon, lookback_days))

    frame = pd.DataFrame(horizon_reports)
    return {
        "horizons": horizon_reports,
        "summary": {
            "best_actionable_horizon": _best_horizon(frame, "actionable_hit_rate"),
            "best_overall_horizon": _best_horizon(frame, "overall_hit_rate"),
            "avg_actionable_hit_rate": float(frame["actionable_hit_rate"].mean()),
            "avg_overall_hit_rate": float(frame["overall_hit_rate"].mean()),
            "avg_actionable_coverage": float(frame["actionable_coverage"].mean()),
        },
    }


def walk_forward_success_evaluation(
    scored: pd.DataFrame,
    horizons: list[int] | None = None,
    train_days: int = 756,
    test_days: int = 126,
) -> dict:
    horizons = horizons or DEFAULT_HORIZONS
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    reports = []

    for horizon in horizons:
        windows = []
        start = 0
        while start + train_days + test_days + horizon <= len(data):
            train = data.iloc[start : start + train_days]
            test = data.iloc[start + train_days : start + train_days + test_days]
            metrics = _evaluate_horizon(test, horizon, lookback_days=20)
            windows.append(
                {
                    "train_start_date": str(train["date"].min().date()),
                    "train_end_date": str(train["date"].max().date()),
                    "test_start_date": str(test["date"].min().date()),
                    "test_end_date": str(test["date"].max().date()),
                    **metrics,
                }
            )
            start += test_days

        reports.append(_summarize_windows(horizon, windows))

    frame = pd.DataFrame([item["summary"] for item in reports if item["summary"]])
    return {
        "horizons": reports,
        "summary": {
            "avg_walk_forward_actionable_hit_rate": float(frame["avg_actionable_hit_rate"].mean()) if not frame.empty else 0.0,
            "avg_walk_forward_overall_hit_rate": float(frame["avg_overall_hit_rate"].mean()) if not frame.empty else 0.0,
            "best_walk_forward_horizon": _best_horizon(frame, "avg_actionable_hit_rate") if not frame.empty else None,
        },
    }


def _evaluate_horizon(data: pd.DataFrame, horizon: int, lookback_days: int) -> dict:
    evaluated = data.copy()
    evaluated["future_close"] = evaluated["close"].shift(-horizon)
    evaluated["future_return"] = evaluated["future_close"] / evaluated["close"] - 1
    evaluated["prior_return"] = evaluated["close"] / evaluated["close"].shift(lookback_days) - 1
    evaluated["predicted_direction"] = evaluated["risk_regime"].map(
        {"risk_on": "up", "neutral": "sideways", "risk_off": "down"}
    )
    evaluated["actual_direction"] = evaluated["future_return"].apply(_direction)
    evaluated["prior_trend"] = evaluated["prior_return"].apply(_direction)
    evaluated["turning_point"] = [
        _turning_point(prior, after)
        for prior, after in zip(evaluated["prior_trend"], evaluated["actual_direction"])
    ]

    valid = evaluated["future_return"].notna()
    actionable = valid & (evaluated["predicted_direction"] != "sideways")

    overall_hits = evaluated.loc[valid, "predicted_direction"] == evaluated.loc[valid, "actual_direction"]
    actionable_hits = evaluated.loc[actionable, "predicted_direction"] == evaluated.loc[actionable, "actual_direction"]

    by_regime = _group_hit_rate(evaluated[valid], "risk_regime")
    by_stage = _group_hit_rate(evaluated[valid], "lifecycle_stage")
    by_turning = _group_hit_rate(evaluated[valid], "turning_point")

    return {
        "horizon_days": int(horizon),
        "sample_count": int(valid.sum()),
        "overall_hit_rate": _mean_bool(overall_hits),
        "actionable_count": int(actionable.sum()),
        "actionable_hit_rate": _mean_bool(actionable_hits),
        "actionable_coverage": float(actionable.sum() / valid.sum()) if valid.sum() else 0.0,
        "by_risk_regime": by_regime,
        "by_lifecycle_stage": by_stage,
        "by_turning_point": by_turning,
    }


def _summarize_windows(horizon: int, windows: list[dict]) -> dict:
    if not windows:
        return {"horizon_days": int(horizon), "window_count": 0, "windows": [], "summary": {}}
    frame = pd.DataFrame(windows)
    return {
        "horizon_days": int(horizon),
        "window_count": int(len(windows)),
        "windows": windows,
        "summary": {
            "horizon_days": int(horizon),
            "avg_overall_hit_rate": float(frame["overall_hit_rate"].mean()),
            "avg_actionable_hit_rate": float(frame["actionable_hit_rate"].mean()),
            "avg_actionable_coverage": float(frame["actionable_coverage"].mean()),
            "positive_edge_windows": int((frame["actionable_hit_rate"] > 0.5).sum()),
        },
    }


def _group_hit_rate(data: pd.DataFrame, column: str) -> list[dict]:
    output = []
    for value, group in data.groupby(column, dropna=False):
        hits = group["predicted_direction"] == group["actual_direction"]
        actionable = group["predicted_direction"] != "sideways"
        actionable_hits = group.loc[actionable, "predicted_direction"] == group.loc[actionable, "actual_direction"]
        output.append(
            {
                "value": str(value),
                "sample_count": int(len(group)),
                "overall_hit_rate": _mean_bool(hits),
                "actionable_count": int(actionable.sum()),
                "actionable_hit_rate": _mean_bool(actionable_hits),
            }
        )
    return sorted(output, key=lambda item: item["actionable_hit_rate"], reverse=True)


def _direction(value: float, neutral_threshold: float = 0.005) -> str:
    if pd.isna(value):
        return "unknown"
    if value > neutral_threshold:
        return "up"
    if value < -neutral_threshold:
        return "down"
    return "sideways"


def _turning_point(prior: str, after: str) -> str:
    if prior == "down" and after == "up":
        return "bottom_reversal"
    if prior == "up" and after == "down":
        return "top_reversal"
    if prior == after:
        return "trend_continuation"
    if after == "sideways":
        return "trend_pause"
    if prior == "sideways":
        return "trend_breakout"
    return "mixed"


def _mean_bool(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(values.mean())


def _best_horizon(frame: pd.DataFrame, metric: str) -> dict | None:
    if frame.empty or metric not in frame:
        return None
    row = frame.sort_values(metric, ascending=False).iloc[0]
    return {
        "horizon_days": int(row["horizon_days"]),
        metric: float(row[metric]),
    }

