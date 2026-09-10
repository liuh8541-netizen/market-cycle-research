import pandas as pd


DEFAULT_HORIZONS = [5, 20]


def analyze_failed_rebound(
    scored: pd.DataFrame,
    horizons: list[int] | None = None,
    crash_threshold: float = -0.03,
    rebound_threshold: float = 0.01,
) -> dict:
    horizons = horizons or DEFAULT_HORIZONS
    data = _prepare(scored)
    events = _find_failed_rebound_events(data, crash_threshold, rebound_threshold)

    horizon_reports = []
    for horizon in horizons:
        horizon_reports.append(_evaluate_events(data, events, horizon))

    return {
        "definition": {
            "crash_threshold": crash_threshold,
            "rebound_threshold": rebound_threshold,
            "rules": [
                "第 0 日大跌",
                "第 1 日反彈",
                "反彈日未站回 5 日或 20 日均線",
                "反彈日量能低於大跌日，或反彈日收盤未收復大跌日實體一半",
            ],
        },
        "event_count": int(len(events)),
        "events": events,
        "horizons": horizon_reports,
    }


def _prepare(scored: pd.DataFrame) -> pd.DataFrame:
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    data["return_1d"] = data["close"].pct_change()
    data["body_mid_recovery"] = (
        data["close"] - data["low"].shift(1)
    ) / (data["high"].shift(1) - data["low"].shift(1)).replace(0, pd.NA)
    return data


def _find_failed_rebound_events(
    data: pd.DataFrame,
    crash_threshold: float,
    rebound_threshold: float,
) -> list[dict]:
    events = []

    for crash_index in range(1, len(data) - 1):
        rebound_index = crash_index + 1
        crash = data.loc[crash_index]
        rebound = data.loc[rebound_index]

        is_crash = crash["return_1d"] <= crash_threshold
        is_rebound = rebound["return_1d"] >= rebound_threshold
        if not is_crash or not is_rebound:
            continue

        below_ma5 = pd.notna(rebound.get("ma_20")) and rebound["close"] < rebound["ma_20"]
        below_ma20 = pd.notna(rebound.get("ma_60")) and rebound["close"] < rebound["ma_60"]
        volume_weak = rebound["volume"] < crash["volume"] if "volume" in data.columns else False
        recovery_weak = pd.notna(rebound["body_mid_recovery"]) and rebound["body_mid_recovery"] < 0.5
        structure_not_repaired = below_ma5 or below_ma20
        rebound_quality_weak = volume_weak or recovery_weak

        if not (structure_not_repaired and rebound_quality_weak):
            continue

        events.append(
            {
                "crash_index": int(crash_index),
                "rebound_index": int(rebound_index),
                "crash_date": str(crash["date"].date()),
                "rebound_date": str(rebound["date"].date()),
                "crash_return": float(crash["return_1d"]),
                "rebound_return": float(rebound["return_1d"]),
                "crash_close": float(crash["close"]),
                "rebound_close": float(rebound["close"]),
                "below_ma20": bool(below_ma5),
                "below_ma60": bool(below_ma20),
                "volume_weak": bool(volume_weak),
                "recovery_weak": bool(recovery_weak),
                "risk_regime": rebound["risk_regime"],
                "lifecycle_stage": rebound["lifecycle_stage"],
            }
        )

    return events


def _evaluate_events(data: pd.DataFrame, events: list[dict], horizon: int) -> dict:
    rows = []
    for event in events:
        start = event["rebound_index"]
        end = start + horizon
        if end >= len(data):
            continue
        future_return = data.loc[end, "close"] / data.loc[start, "close"] - 1
        rows.append(
            {
                **event,
                "horizon_days": int(horizon),
                "future_date": str(data.loc[end, "date"].date()),
                "future_return": float(future_return),
                "future_direction": _direction(future_return),
                "is_down": bool(future_return < -0.005),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "horizon_days": int(horizon),
            "sample_count": 0,
            "down_probability": 0.0,
            "up_probability": 0.0,
            "sideways_probability": 0.0,
            "avg_return": 0.0,
            "median_return": 0.0,
            "rows": [],
        }

    counts = frame["future_direction"].value_counts()
    total = len(frame)
    return {
        "horizon_days": int(horizon),
        "sample_count": int(total),
        "down_probability": float(counts.get("down", 0) / total),
        "up_probability": float(counts.get("up", 0) / total),
        "sideways_probability": float(counts.get("sideways", 0) / total),
        "avg_return": float(frame["future_return"].mean()),
        "median_return": float(frame["future_return"].median()),
        "rows": rows,
    }


def _direction(value: float, neutral_threshold: float = 0.005) -> str:
    if value > neutral_threshold:
        return "up"
    if value < -neutral_threshold:
        return "down"
    return "sideways"

