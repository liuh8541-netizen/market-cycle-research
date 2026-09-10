import itertools

import pandas as pd


DEFAULT_HORIZONS = [1, 5, 20, 60]
DEFAULT_FACTORS = [
    "risk_regime",
    "lifecycle_stage",
    "score_bucket",
    "trend_score",
    "momentum_score",
    "volatility_bucket",
    "drawdown_bucket",
    "prior_trend",
]


def find_high_edge_conditions(
    scored: pd.DataFrame,
    horizons: list[int] | None = None,
    min_samples: int = 120,
    min_hit_rate: float = 0.60,
    max_factors: int = 3,
) -> dict:
    horizons = horizons or DEFAULT_HORIZONS
    data = _prepare(scored)
    reports = []

    for horizon in horizons:
        horizon_data = _add_forward_result(data, horizon)
        valid = horizon_data[horizon_data["actual_direction"] != "unknown"].copy()
        conditions = []

        for factor_count in range(1, max_factors + 1):
            for factors in itertools.combinations(DEFAULT_FACTORS, factor_count):
                conditions.extend(
                    _scan_factor_combo(
                        valid,
                        list(factors),
                        min_samples=min_samples,
                        min_hit_rate=min_hit_rate,
                    )
                )

        conditions = _dedupe_conditions(conditions)
        conditions = sorted(
            conditions,
            key=lambda item: (item["hit_rate"], item["sample_count"]),
            reverse=True,
        )
        reports.append(
            {
                "horizon_days": int(horizon),
                "condition_count": len(conditions),
                "top_conditions": conditions[:30],
            }
        )

    return {"horizons": reports}


def walk_forward_edge_validation(
    scored: pd.DataFrame,
    horizons: list[int] | None = None,
    min_samples: int = 80,
    min_hit_rate: float = 0.60,
    train_days: int = 1008,
    test_days: int = 252,
) -> dict:
    horizons = horizons or DEFAULT_HORIZONS
    data = _prepare(scored)
    reports = []

    for horizon in horizons:
        windows = []
        start = 0
        while start + train_days + test_days + horizon <= len(data):
            train = data.iloc[start : start + train_days].copy()
            test = data.iloc[start + train_days : start + train_days + test_days].copy()

            train_edges = find_high_edge_conditions(
                train,
                horizons=[horizon],
                min_samples=min_samples,
                min_hit_rate=min_hit_rate,
                max_factors=2,
            )["horizons"][0]["top_conditions"]

            selected = train_edges[:5]
            evaluated = [_evaluate_condition(test, condition, horizon) for condition in selected]
            evaluated = [item for item in evaluated if item["sample_count"] > 0]
            combined = _evaluate_combined_conditions(test, selected, horizon)

            windows.append(
                {
                    "test_start_date": str(test["date"].min().date()),
                    "test_end_date": str(test["date"].max().date()),
                    "selected_condition_count": len(selected),
                    "evaluated_conditions": evaluated,
                    "combined": combined,
                }
            )
            start += test_days

        reports.append(_summarize_edge_windows(horizon, windows))

    return {"horizons": reports}


def _scan_factor_combo(
    data: pd.DataFrame,
    factors: list[str],
    min_samples: int,
    min_hit_rate: float,
) -> list[dict]:
    output = []
    grouped = data.groupby(factors, dropna=False)
    for values, group in grouped:
        if len(group) < min_samples:
            continue
        if not isinstance(values, tuple):
            values = (values,)
        for prediction in ["up", "down", "sideways"]:
            hit_rate = float((group["actual_direction"] == prediction).mean())
            if hit_rate < min_hit_rate:
                continue
            output.append(
                {
                    "factors": factors,
                    "values": [str(value) for value in values],
                    "prediction": prediction,
                    "sample_count": int(len(group)),
                    "hit_rate": hit_rate,
                    "coverage": float(len(group) / len(data)) if len(data) else 0.0,
                }
            )
    return output


def _evaluate_condition(test: pd.DataFrame, condition: dict, horizon: int) -> dict:
    data = _add_forward_result(test, horizon)
    mask = _condition_mask(data, condition)
    sample = data[mask & (data["actual_direction"] != "unknown")]
    if sample.empty:
        hit_rate = 0.0
    else:
        hit_rate = float((sample["actual_direction"] == condition["prediction"]).mean())
    return {
        **condition,
        "sample_count": int(len(sample)),
        "hit_rate": hit_rate,
    }


def _evaluate_combined_conditions(test: pd.DataFrame, conditions: list[dict], horizon: int) -> dict:
    data = _add_forward_result(test, horizon)
    predictions = []
    for index, row in data.iterrows():
        prediction = None
        for condition in conditions:
            if _row_matches(row, condition):
                prediction = condition["prediction"]
                break
        if prediction is not None and row["actual_direction"] != "unknown":
            predictions.append(prediction == row["actual_direction"])

    return {
        "sample_count": len(predictions),
        "hit_rate": float(pd.Series(predictions).mean()) if predictions else 0.0,
    }


def _summarize_edge_windows(horizon: int, windows: list[dict]) -> dict:
    combined = [window["combined"] for window in windows if window["combined"]["sample_count"] > 0]
    if not combined:
        return {"horizon_days": int(horizon), "window_count": len(windows), "summary": {}, "windows": windows}
    frame = pd.DataFrame(combined)
    return {
        "horizon_days": int(horizon),
        "window_count": len(windows),
        "summary": {
            "avg_combined_hit_rate": float(frame["hit_rate"].mean()),
            "avg_combined_sample_count": float(frame["sample_count"].mean()),
            "positive_edge_windows": int((frame["hit_rate"] >= 0.60).sum()),
        },
        "windows": windows,
    }


def _condition_mask(data: pd.DataFrame, condition: dict) -> pd.Series:
    mask = pd.Series(True, index=data.index)
    for factor, value in zip(condition["factors"], condition["values"]):
        mask = mask & (data[factor].astype(str) == str(value))
    return mask


def _row_matches(row: pd.Series, condition: dict) -> bool:
    return all(str(row[factor]) == str(value) for factor, value in zip(condition["factors"], condition["values"]))


def _dedupe_conditions(conditions: list[dict]) -> list[dict]:
    seen = set()
    output = []
    for condition in conditions:
        key = (
            tuple(condition["factors"]),
            tuple(condition["values"]),
            condition["prediction"],
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(condition)
    return output


def _prepare(scored: pd.DataFrame) -> pd.DataFrame:
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    data["score_bucket"] = data["stage_score"].apply(_score_bucket)
    data["volatility_bucket"] = data["volatility_20"].apply(_volatility_bucket)
    data["drawdown_bucket"] = data["drawdown"].apply(_drawdown_bucket)
    data["prior_return_20d"] = data["close"].pct_change(20)
    data["prior_trend"] = data["prior_return_20d"].apply(_direction)
    return data


def _add_forward_result(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    output = data.copy()
    output["future_return"] = output["close"].shift(-horizon) / output["close"] - 1
    output["actual_direction"] = output["future_return"].apply(_direction)
    return output


def _score_bucket(value) -> str:
    if pd.isna(value):
        return "unknown"
    if value >= 5:
        return "strong_positive"
    if value >= 2:
        return "positive"
    if value > -2:
        return "neutral"
    if value > -5:
        return "negative"
    return "strong_negative"


def _volatility_bucket(value) -> str:
    if pd.isna(value):
        return "unknown"
    if value < 0.16:
        return "low"
    if value < 0.28:
        return "normal"
    return "high"


def _drawdown_bucket(value) -> str:
    if pd.isna(value):
        return "unknown"
    if value > -0.05:
        return "near_high"
    if value > -0.12:
        return "mild_drawdown"
    if value > -0.25:
        return "deep_drawdown"
    return "crash_drawdown"


def _direction(value: float, neutral_threshold: float = 0.005) -> str:
    if pd.isna(value):
        return "unknown"
    if value > neutral_threshold:
        return "up"
    if value < -neutral_threshold:
        return "down"
    return "sideways"

