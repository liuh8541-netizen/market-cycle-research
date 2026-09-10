import pandas as pd


def analyze_market_context(
    scored: pd.DataFrame,
    target_date: str,
    input_index: float | None = None,
    lookback_days: int = 20,
    forward_days: int = 20,
    tolerance_pct: float = 0.01,
) -> dict:
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    target = pd.to_datetime(target_date)

    eligible = data[data["date"] <= target]
    if eligible.empty:
        raise ValueError("No market data is available on or before " + target_date)

    signal_index = int(eligible.index[-1])
    lookback_index = signal_index - lookback_days
    forward_index = signal_index + forward_days

    if lookback_index < 0:
        raise ValueError("Not enough prior data for lookback_days=" + str(lookback_days))
    if forward_index >= len(data):
        raise ValueError("Not enough future data for forward_days=" + str(forward_days))

    before = data.loc[lookback_index]
    signal = data.loc[signal_index]
    after = data.loc[forward_index]

    close = float(signal["close"])
    prior_return = close / float(before["close"]) - 1
    forward_return = float(after["close"]) / close - 1
    predicted_direction = _predicted_direction(signal["risk_regime"])
    actual_direction = _direction(forward_return)

    input_check = _check_input_index(input_index, close, tolerance_pct)

    return {
        "requested_date": target_date,
        "signal_date": str(signal["date"].date()),
        "lookback_start_date": str(before["date"].date()),
        "forward_end_date": str(after["date"].date()),
        "lookback_days": int(lookback_days),
        "forward_days": int(forward_days),
        "input_index": None if input_index is None else float(input_index),
        "data_close": close,
        "input_check": input_check,
        "lookback_start_close": float(before["close"]),
        "forward_end_close": float(after["close"]),
        "prior_return": float(prior_return),
        "forward_return": float(forward_return),
        "prior_trend": _direction(prior_return),
        "forward_trend": actual_direction,
        "turning_point": _turning_point(_direction(prior_return), actual_direction),
        "lifecycle_stage": signal["lifecycle_stage"],
        "stage_score": float(signal["stage_score"]),
        "risk_regime": signal["risk_regime"],
        "predicted_direction": predicted_direction,
        "actual_direction": actual_direction,
        "is_hit": bool(predicted_direction == actual_direction),
        "confidence": float(signal["confidence"]),
    }


def _check_input_index(input_index: float | None, data_close: float, tolerance_pct: float) -> dict:
    if input_index is None:
        return {
            "provided": False,
            "is_consistent": None,
            "difference": None,
            "difference_pct": None,
        }

    difference = float(input_index) - data_close
    difference_pct = difference / data_close if data_close else 0.0
    return {
        "provided": True,
        "is_consistent": abs(difference_pct) <= tolerance_pct,
        "difference": float(difference),
        "difference_pct": float(difference_pct),
    }


def _predicted_direction(risk_regime: str) -> str:
    if risk_regime == "risk_on":
        return "up"
    if risk_regime == "risk_off":
        return "down"
    return "sideways"


def _direction(value: float, neutral_threshold: float = 0.005) -> str:
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

