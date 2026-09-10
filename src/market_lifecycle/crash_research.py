import pandas as pd


CRASH_SCENARIOS = [
    {"name": "single_day_drop_2pct", "column": "return_1d", "operator": "<=", "threshold": -0.02},
    {"name": "five_day_drop_5pct", "column": "return_5d", "operator": "<=", "threshold": -0.05},
    {"name": "twenty_day_drop_10pct", "column": "return_20d", "operator": "<=", "threshold": -0.10},
    {"name": "drawdown_10pct", "column": "drawdown", "operator": "<=", "threshold": -0.10},
    {"name": "drawdown_20pct", "column": "drawdown", "operator": "<=", "threshold": -0.20},
]

DEFAULT_HORIZONS = [1, 5, 20, 60]


def analyze_after_crash(
    scored: pd.DataFrame,
    horizons: list[int] | None = None,
) -> dict:
    horizons = horizons or DEFAULT_HORIZONS
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])

    reports = []
    for scenario in CRASH_SCENARIOS:
        mask = _scenario_mask(data, scenario)
        scenario_rows = data[mask].copy()
        horizon_reports = []
        for horizon in horizons:
            horizon_reports.append(_analyze_scenario_horizon(data, mask, horizon))
        reports.append(
            {
                "scenario": scenario["name"],
                "condition": f"{scenario['column']} {scenario['operator']} {scenario['threshold']:.2%}",
                "event_count": int(len(scenario_rows)),
                "first_event": str(scenario_rows["date"].min().date()) if not scenario_rows.empty else None,
                "last_event": str(scenario_rows["date"].max().date()) if not scenario_rows.empty else None,
                "horizons": horizon_reports,
            }
        )

    return {"scenarios": reports}


def forecast_after_crash_today(
    scored: pd.DataFrame,
    target_date: str | None = None,
    horizons: list[int] | None = None,
) -> dict:
    horizons = horizons or DEFAULT_HORIZONS
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    target_index = _target_index(data, target_date)
    target = data.loc[target_index]
    active = []

    for scenario in CRASH_SCENARIOS:
        if _row_matches_scenario(target, scenario):
            mask = _scenario_mask(data.iloc[:target_index], scenario)
            history = data.iloc[:target_index][mask].copy()
            active.append(
                {
                    "scenario": scenario["name"],
                    "condition": f"{scenario['column']} {scenario['operator']} {scenario['threshold']:.2%}",
                    "historical_cases": int(len(history)),
                    "horizons": [
                        _historical_distribution(history, horizon)
                        for horizon in horizons
                    ],
                }
            )

    return {
        "target_date": str(target["date"].date()),
        "close": float(target["close"]),
        "active_crash_scenarios": active,
    }


def _analyze_scenario_horizon(data: pd.DataFrame, mask: pd.Series, horizon: int) -> dict:
    valid = mask & data["close"].shift(-horizon).notna()
    sample = data[valid].copy()
    future_return = data["close"].shift(-horizon) / data["close"] - 1
    directions = future_return[valid].apply(_direction)
    counts = directions.value_counts()
    total = int(counts.sum())
    avg_return = float(future_return[valid].mean()) if total else 0.0
    median_return = float(future_return[valid].median()) if total else 0.0
    return {
        "horizon_days": int(horizon),
        "sample_count": total,
        "probability_up": float(counts.get("up", 0) / total) if total else 0.0,
        "probability_down": float(counts.get("down", 0) / total) if total else 0.0,
        "probability_sideways": float(counts.get("sideways", 0) / total) if total else 0.0,
        "avg_return": avg_return,
        "median_return": median_return,
        "best_direction": str(counts.idxmax()) if total else "none",
    }


def _historical_distribution(history: pd.DataFrame, horizon: int) -> dict:
    if history.empty:
        return {
            "horizon_days": int(horizon),
            "sample_count": 0,
            "probability_up": 0.0,
            "probability_down": 0.0,
            "probability_sideways": 0.0,
            "best_direction": "none",
        }
    future_return = history["close"].shift(-horizon) / history["close"] - 1
    directions = future_return.apply(_direction)
    valid = directions[directions != "unknown"]
    counts = valid.value_counts()
    total = int(counts.sum())
    return {
        "horizon_days": int(horizon),
        "sample_count": total,
        "probability_up": float(counts.get("up", 0) / total) if total else 0.0,
        "probability_down": float(counts.get("down", 0) / total) if total else 0.0,
        "probability_sideways": float(counts.get("sideways", 0) / total) if total else 0.0,
        "best_direction": str(counts.idxmax()) if total else "none",
    }


def _scenario_mask(data: pd.DataFrame, scenario: dict) -> pd.Series:
    if scenario["operator"] == "<=":
        return data[scenario["column"]] <= scenario["threshold"]
    raise ValueError("Unsupported operator: " + scenario["operator"])


def _row_matches_scenario(row: pd.Series, scenario: dict) -> bool:
    value = row.get(scenario["column"])
    if pd.isna(value):
        return False
    if scenario["operator"] == "<=":
        return value <= scenario["threshold"]
    return False


def _target_index(data: pd.DataFrame, target_date: str | None) -> int:
    if target_date is None:
        return len(data) - 1
    target = pd.to_datetime(target_date)
    eligible = data[data["date"] <= target]
    if eligible.empty:
        raise ValueError("No market data is available on or before " + target_date)
    return int(eligible.index[-1])


def _direction(value: float, neutral_threshold: float = 0.005) -> str:
    if pd.isna(value):
        return "unknown"
    if value > neutral_threshold:
        return "up"
    if value < -neutral_threshold:
        return "down"
    return "sideways"

