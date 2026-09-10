import pandas as pd


def validate_prediction_on_date(
    scored: pd.DataFrame,
    target_date: str,
    horizon_days: int = 5,
) -> dict:
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    target = pd.to_datetime(target_date)

    eligible = data[data["date"] <= target]
    if eligible.empty:
        raise ValueError("No market data is available on or before " + target_date)

    signal_index = eligible.index[-1]
    future_index = signal_index + horizon_days
    if future_index >= len(data):
        raise ValueError("Not enough future data to validate " + target_date)

    signal = data.loc[signal_index]
    future = data.loc[future_index]
    future_return = future["close"] / signal["close"] - 1

    predicted_direction = _predicted_direction(signal["risk_regime"])
    actual_direction = _actual_direction(future_return)

    return {
        "requested_date": target_date,
        "signal_date": str(signal["date"].date()),
        "validation_date": str(future["date"].date()),
        "horizon_days": int(horizon_days),
        "input_close": float(signal["close"]),
        "future_close": float(future["close"]),
        "future_return": float(future_return),
        "lifecycle_stage": signal["lifecycle_stage"],
        "stage_score": float(signal["stage_score"]),
        "risk_regime": signal["risk_regime"],
        "predicted_direction": predicted_direction,
        "actual_direction": actual_direction,
        "is_hit": bool(predicted_direction == actual_direction),
        "confidence": float(signal["confidence"]),
    }


def validate_prediction_batch(
    scored: pd.DataFrame,
    horizon_days: int = 5,
) -> dict:
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    future = data.shift(-horizon_days)
    future_return = future["close"] / data["close"] - 1

    predicted = data["risk_regime"].apply(_predicted_direction)
    actual = future_return.apply(_actual_direction)
    valid = future_return.notna()
    actionable = valid & (predicted != "sideways")

    overall_hits = predicted[valid] == actual[valid]
    actionable_hits = predicted[actionable] == actual[actionable]

    rows = []
    for index in data[valid].index:
        rows.append(
            {
                "signal_date": str(data.loc[index, "date"].date()),
                "validation_date": str(future.loc[index, "date"].date()),
                "horizon_days": int(horizon_days),
                "input_close": float(data.loc[index, "close"]),
                "future_close": float(future.loc[index, "close"]),
                "future_return": float(future_return.loc[index]),
                "risk_regime": data.loc[index, "risk_regime"],
                "predicted_direction": predicted.loc[index],
                "actual_direction": actual.loc[index],
                "is_hit": bool(predicted.loc[index] == actual.loc[index]),
            }
        )

    return {
        "horizon_days": int(horizon_days),
        "sample_count": int(valid.sum()),
        "overall_hit_count": int(overall_hits.sum()),
        "overall_hit_rate": float(overall_hits.mean()) if len(overall_hits) else 0.0,
        "actionable_count": int(actionable.sum()),
        "actionable_hit_count": int(actionable_hits.sum()),
        "actionable_hit_rate": float(actionable_hits.mean()) if len(actionable_hits) else 0.0,
        "actionable_coverage": float(actionable.sum() / valid.sum()) if valid.sum() else 0.0,
        "rows": rows,
    }


def _predicted_direction(risk_regime: str) -> str:
    if risk_regime == "risk_on":
        return "up"
    if risk_regime == "risk_off":
        return "down"
    return "sideways"


def _actual_direction(future_return: float, neutral_threshold: float = 0.005) -> str:
    if future_return > neutral_threshold:
        return "up"
    if future_return < -neutral_threshold:
        return "down"
    return "sideways"
