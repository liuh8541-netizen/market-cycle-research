import pandas as pd


DEFAULT_HORIZONS = [1, 5, 20, 60]


def research_market_scenarios(
    scored: pd.DataFrame,
    horizons: list[int] | None = None,
    min_samples: int = 80,
    target_hit_rate: float = 0.90,
) -> dict:
    horizons = horizons or DEFAULT_HORIZONS
    data = _prepare(scored)
    scenarios = _build_scenarios(data)
    reports = []

    for scenario in scenarios:
        mask = scenario["mask"]
        event_count = int(mask.sum())
        horizon_reports = []
        for horizon in horizons:
            horizon_reports.append(_evaluate_scenario(data, mask, horizon))

        best = _best_horizon(horizon_reports)
        reports.append(
            {
                "scenario": scenario["name"],
                "description": scenario["description"],
                "event_count": event_count,
                "first_event": _first_event(data, mask),
                "last_event": _last_event(data, mask),
                "best_horizon": best,
                "passes_target": bool(
                    best
                    and best["sample_count"] >= min_samples
                    and best["best_probability"] >= target_hit_rate
                ),
                "horizons": horizon_reports,
            }
        )

    reports = sorted(
        reports,
        key=lambda item: (
            item["passes_target"],
            item["best_horizon"]["best_probability"] if item["best_horizon"] else 0,
            item["event_count"],
        ),
        reverse=True,
    )
    return {
        "target_hit_rate": float(target_hit_rate),
        "min_samples": int(min_samples),
        "scenario_count": len(reports),
        "passing_count": int(sum(item["passes_target"] for item in reports)),
        "scenarios": reports,
    }


def walk_forward_market_scenarios(
    scored: pd.DataFrame,
    horizons: list[int] | None = None,
    min_samples: int = 40,
    target_hit_rate: float = 0.90,
    train_days: int = 1008,
    test_days: int = 252,
) -> dict:
    horizons = horizons or DEFAULT_HORIZONS
    data = _prepare(scored)
    windows = []
    start = 0

    while start + train_days + test_days + max(horizons) <= len(data):
        train = data.iloc[start : start + train_days].copy()
        test = data.iloc[start + train_days : start + train_days + test_days].copy()

        train_report = research_market_scenarios(
            train,
            horizons=horizons,
            min_samples=min_samples,
            target_hit_rate=target_hit_rate,
        )
        selected = [
            _select_scenario_rule(item)
            for item in train_report["scenarios"]
            if item["passes_target"]
        ][:10]

        evaluated = [_evaluate_rule_on_test(test, rule) for rule in selected]
        evaluated = [item for item in evaluated if item["sample_count"] > 0]
        windows.append(
            {
                "test_start_date": str(test["date"].min().date()),
                "test_end_date": str(test["date"].max().date()),
                "selected_count": len(selected),
                "evaluated_count": len(evaluated),
                "evaluated": evaluated,
                "combined": _combined_result(evaluated),
            }
        )
        start += test_days

    return _summarize_windows(windows)


def _prepare(scored: pd.DataFrame) -> pd.DataFrame:
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    data["return_1d"] = data["close"].pct_change()
    data["return_5d"] = data["close"].pct_change(5)
    data["return_20d"] = data["close"].pct_change(20)
    data["distance_from_high"] = data["close"] / data["close"].cummax() - 1
    data["distance_to_ma20"] = data["close"] / data["ma_20"] - 1
    data["distance_to_ma60"] = data["close"] / data["ma_60"] - 1
    data["rsi_bucket"] = data["rsi_14"].apply(_rsi_bucket)
    data["volatility_bucket"] = data["volatility_20"].apply(_volatility_bucket)
    data["drawdown_bucket"] = data["drawdown"].apply(_drawdown_bucket)
    return data


def _build_scenarios(data: pd.DataFrame) -> list[dict]:
    return [
        _scenario("single_day_crash", "單日跌幅超過 2%", data["return_1d"] <= -0.02),
        _scenario("single_day_surge", "單日漲幅超過 2%", data["return_1d"] >= 0.02),
        _scenario("five_day_crash", "5 日跌幅超過 5%", data["return_5d"] <= -0.05),
        _scenario("five_day_surge", "5 日漲幅超過 5%", data["return_5d"] >= 0.05),
        _scenario("twenty_day_crash", "20 日跌幅超過 10%", data["return_20d"] <= -0.10),
        _scenario("twenty_day_surge", "20 日漲幅超過 10%", data["return_20d"] >= 0.10),
        _scenario("near_high", "距歷史高點小於 3%", data["distance_from_high"] >= -0.03),
        _scenario("mild_drawdown", "回撤 5% 至 12%", (data["drawdown"] <= -0.05) & (data["drawdown"] > -0.12)),
        _scenario("deep_drawdown", "回撤 12% 至 25%", (data["drawdown"] <= -0.12) & (data["drawdown"] > -0.25)),
        _scenario("crash_drawdown", "回撤超過 25%", data["drawdown"] <= -0.25),
        _scenario("strong_uptrend", "強多頭：risk_on 且分數 >= 5", (data["risk_regime"] == "risk_on") & (data["stage_score"] >= 5)),
        _scenario("weak_uptrend", "偏多：risk_on 但分數 < 5", (data["risk_regime"] == "risk_on") & (data["stage_score"] < 5)),
        _scenario("strong_downtrend", "強空頭：risk_off 且分數 <= -5", (data["risk_regime"] == "risk_off") & (data["stage_score"] <= -5)),
        _scenario("weak_downtrend", "偏空：risk_off 但分數 > -5", (data["risk_regime"] == "risk_off") & (data["stage_score"] > -5)),
        _scenario("neutral_market", "中性風險狀態", data["risk_regime"] == "neutral"),
        _scenario("above_ma20_ma60", "收盤同時站上 20 日與 60 日均線", (data["distance_to_ma20"] > 0) & (data["distance_to_ma60"] > 0)),
        _scenario("below_ma20_ma60", "收盤同時跌破 20 日與 60 日均線", (data["distance_to_ma20"] < 0) & (data["distance_to_ma60"] < 0)),
        _scenario("rsi_overbought", "RSI 過熱", data["rsi_bucket"] == "overbought"),
        _scenario("rsi_oversold", "RSI 超跌", data["rsi_bucket"] == "oversold"),
        _scenario("high_volatility", "高波動", data["volatility_bucket"] == "high"),
        _scenario("low_volatility", "低波動", data["volatility_bucket"] == "low"),
        _scenario("bottoming_stage", "生命周期：築底", data["lifecycle_stage"] == "bottoming"),
        _scenario("topping_stage", "生命周期：高檔震盪", data["lifecycle_stage"] == "topping"),
        _scenario("main_bull_stage", "生命周期：主升", data["lifecycle_stage"] == "main_bull"),
        _scenario("main_bear_stage", "生命周期：主跌", data["lifecycle_stage"] == "main_bear"),
    ]


def _scenario(name: str, description: str, mask: pd.Series) -> dict:
    return {"name": name, "description": description, "mask": mask.fillna(False)}


def _evaluate_scenario(data: pd.DataFrame, mask: pd.Series, horizon: int) -> dict:
    future_return = data["close"].shift(-horizon) / data["close"] - 1
    valid = mask & future_return.notna()
    directions = future_return[valid].apply(_direction)
    counts = directions.value_counts()
    total = int(counts.sum())
    probabilities = {
        "up": float(counts.get("up", 0) / total) if total else 0.0,
        "down": float(counts.get("down", 0) / total) if total else 0.0,
        "sideways": float(counts.get("sideways", 0) / total) if total else 0.0,
    }
    best_direction = max(probabilities, key=probabilities.get)
    returns = future_return[valid]
    return {
        "horizon_days": int(horizon),
        "sample_count": total,
        "best_direction": best_direction,
        "best_probability": probabilities[best_direction],
        "probability_up": probabilities["up"],
        "probability_down": probabilities["down"],
        "probability_sideways": probabilities["sideways"],
        "avg_return": float(returns.mean()) if total else 0.0,
        "median_return": float(returns.median()) if total else 0.0,
    }


def _best_horizon(horizons: list[dict]) -> dict | None:
    if not horizons:
        return None
    return sorted(horizons, key=lambda item: (item["best_probability"], item["sample_count"]), reverse=True)[0]


def _select_scenario_rule(scenario_report: dict) -> dict:
    best = scenario_report["best_horizon"]
    return {
        "scenario": scenario_report["scenario"],
        "description": scenario_report["description"],
        "horizon_days": best["horizon_days"],
        "prediction": best["best_direction"],
    }


def _evaluate_rule_on_test(test: pd.DataFrame, rule: dict) -> dict:
    data = _prepare(test)
    scenarios = {item["name"]: item for item in _build_scenarios(data)}
    if rule["scenario"] not in scenarios:
        return {**rule, "sample_count": 0, "hit_rate": 0.0}
    mask = scenarios[rule["scenario"]]["mask"]
    future_return = data["close"].shift(-rule["horizon_days"]) / data["close"] - 1
    valid = mask & future_return.notna()
    directions = future_return[valid].apply(_direction)
    if directions.empty:
        return {**rule, "sample_count": 0, "hit_rate": 0.0}
    return {
        **rule,
        "sample_count": int(len(directions)),
        "hit_rate": float((directions == rule["prediction"]).mean()),
    }


def _combined_result(evaluated: list[dict]) -> dict:
    if not evaluated:
        return {"sample_count": 0, "avg_hit_rate": 0.0, "passing_count": 0}
    frame = pd.DataFrame(evaluated)
    return {
        "sample_count": int(frame["sample_count"].sum()),
        "avg_hit_rate": float(frame["hit_rate"].mean()),
        "passing_count": int((frame["hit_rate"] >= 0.90).sum()),
    }


def _summarize_windows(windows: list[dict]) -> dict:
    combined = [window["combined"] for window in windows if window["combined"]["sample_count"] > 0]
    if not combined:
        return {"window_count": len(windows), "summary": {}, "windows": windows}
    frame = pd.DataFrame(combined)
    return {
        "window_count": len(windows),
        "summary": {
            "avg_hit_rate": float(frame["avg_hit_rate"].mean()),
            "avg_sample_count": float(frame["sample_count"].mean()),
            "windows_with_90pct_rule": int((frame["passing_count"] > 0).sum()),
        },
        "windows": windows,
    }


def _first_event(data: pd.DataFrame, mask: pd.Series) -> str | None:
    sample = data[mask]
    return str(sample["date"].min().date()) if not sample.empty else None


def _last_event(data: pd.DataFrame, mask: pd.Series) -> str | None:
    sample = data[mask]
    return str(sample["date"].max().date()) if not sample.empty else None


def _rsi_bucket(value) -> str:
    if pd.isna(value):
        return "unknown"
    if value >= 70:
        return "overbought"
    if value <= 30:
        return "oversold"
    return "normal"


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

