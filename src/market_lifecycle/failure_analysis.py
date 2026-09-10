import pandas as pd


def analyze_failure_factors(scored: pd.DataFrame, horizon_days: int = 5, actionable_only: bool = True) -> dict:
    data = _with_validation_columns(scored, horizon_days)
    valid = data[data["future_return"].notna()].copy()
    if actionable_only:
        valid = valid[valid["predicted_direction"] != "sideways"].copy()

    groups = []
    for column in [
        "risk_regime",
        "lifecycle_stage",
        "stage_score_bucket",
        "trend_score",
        "momentum_score",
        "volatility_bucket",
        "drawdown_bucket",
        "price_vs_ma60",
        "price_vs_ma120",
    ]:
        groups.extend(_summarize_group(valid, column))

    groups = sorted(groups, key=lambda item: (item["hit_rate"], -item["sample_count"]))
    return {
        "horizon_days": int(horizon_days),
        "actionable_only": bool(actionable_only),
        "sample_count": int(len(valid)),
        "overall_hit_rate": float(valid["is_hit"].mean()) if len(valid) else 0.0,
        "worst_factors": groups[:20],
        "all_factors": groups,
    }


def neutralize_failure_factors_iteratively(
    scored: pd.DataFrame,
    failure_analysis: dict,
    min_samples: int = 60,
    max_hit_rate: float = 0.45,
    max_rules: int = 5,
) -> tuple[pd.DataFrame, list[dict]]:
    data = scored.copy()
    rules = []
    current = analyze_failure_factors(data, failure_analysis["horizon_days"], actionable_only=True)
    current_hit_rate = current["overall_hit_rate"]

    for factor in failure_analysis["all_factors"]:
        if len(rules) >= max_rules:
            break
        if factor["sample_count"] < min_samples:
            continue
        if factor["hit_rate"] > max_hit_rate:
            continue
        if factor["factor"] in ["risk_regime"]:
            continue

        rule = {
            "factor": factor["factor"],
            "value": factor["value"],
            "hit_rate": factor["hit_rate"],
            "sample_count": factor["sample_count"],
        }
        candidate = _neutralize_by_rule(data, rule)
        candidate_analysis = analyze_failure_factors(
            candidate,
            failure_analysis["horizon_days"],
            actionable_only=True,
        )
        if candidate_analysis["overall_hit_rate"] <= current_hit_rate:
            continue

        data = candidate
        current_hit_rate = candidate_analysis["overall_hit_rate"]
        rules.append(rule)

    return data, rules


def neutralize_failure_factors(
    scored: pd.DataFrame,
    failure_analysis: dict,
    min_samples: int = 60,
    max_hit_rate: float = 0.45,
    max_rules: int = 5,
) -> tuple[pd.DataFrame, list[dict]]:
    return neutralize_failure_factors_iteratively(
        scored,
        failure_analysis,
        min_samples=min_samples,
        max_hit_rate=max_hit_rate,
        max_rules=max_rules,
    )


def _neutralize_by_rule(scored: pd.DataFrame, rule: dict) -> pd.DataFrame:
    data = scored.copy()
    prepared = _add_factor_columns(data)
    mask = prepared[rule["factor"]].astype(str) == str(rule["value"])
    positions = mask.to_numpy()
    data.loc[data.index[positions], "risk_regime"] = "neutral"
    data.loc[data.index[positions], "forward_1w_view"] = "neutral"
    data.loc[data.index[positions], "forward_4w_view"] = "neutral"
    data.loc[data.index[positions], "confidence"] = data.loc[data.index[positions], "confidence"] * 0.5
    return data


def apply_rules_one_by_one(scored: pd.DataFrame, rules: list[dict], horizon_days: int) -> list[dict]:
    data = scored.copy()
    steps = []
    for rule in rules:
        data = _neutralize_by_rule(data, rule)
        analysis = analyze_failure_factors(data, horizon_days, actionable_only=True)
        steps.append(
            {
                "rule": rule,
                "actionable_sample_count": analysis["sample_count"],
                "actionable_hit_rate": analysis["overall_hit_rate"],
            }
        )
    return steps


def apply_rule_list(scored: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    data = scored.copy()
    for rule in rules:
        data = _neutralize_by_rule(data, rule)
    return data


def walk_forward_failure_filter(
    scored: pd.DataFrame,
    horizon_days: int = 20,
    train_days: int = 756,
    test_days: int = 126,
) -> dict:
    data = scored.sort_values("date").reset_index(drop=True)
    windows = []
    start = 0

    while start + train_days + test_days + horizon_days <= len(data):
        train = data.iloc[start : start + train_days].copy()
        test = data.iloc[start + train_days : start + train_days + test_days].copy()

        train_analysis = analyze_failure_factors(train, horizon_days, actionable_only=True)
        _, rules = neutralize_failure_factors(train, train_analysis)
        corrected_test = apply_rule_list(test, rules)

        before = analyze_failure_factors(test, horizon_days, actionable_only=True)
        after = analyze_failure_factors(corrected_test, horizon_days, actionable_only=True)
        windows.append(
            {
                "train_start_date": str(train["date"].min().date()),
                "train_end_date": str(train["date"].max().date()),
                "test_start_date": str(test["date"].min().date()),
                "test_end_date": str(test["date"].max().date()),
                "before_actionable_count": before["sample_count"],
                "before_actionable_hit_rate": before["overall_hit_rate"],
                "after_actionable_count": after["sample_count"],
                "after_actionable_hit_rate": after["overall_hit_rate"],
                "rules": rules,
            }
        )
        start += test_days

    return _summarize_walk_forward_windows(windows)


def _summarize_walk_forward_windows(windows: list[dict]) -> dict:
    if not windows:
        return {"window_count": 0, "windows": [], "summary": {}}

    frame = pd.DataFrame(windows)
    improved = frame["after_actionable_hit_rate"] > frame["before_actionable_hit_rate"]
    return {
        "window_count": int(len(windows)),
        "windows": windows,
        "summary": {
            "avg_before_actionable_hit_rate": float(frame["before_actionable_hit_rate"].mean()),
            "avg_after_actionable_hit_rate": float(frame["after_actionable_hit_rate"].mean()),
            "avg_before_actionable_count": float(frame["before_actionable_count"].mean()),
            "avg_after_actionable_count": float(frame["after_actionable_count"].mean()),
            "improved_windows": int(improved.sum()),
        },
    }


def _with_validation_columns(scored: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    data = _add_factor_columns(scored)
    future = data.shift(-horizon_days)
    data["future_return"] = future["close"] / data["close"] - 1
    data["predicted_direction"] = data["risk_regime"].map(
        {"risk_on": "up", "neutral": "sideways", "risk_off": "down"}
    )
    data["actual_direction"] = data["future_return"].apply(_actual_direction)
    data["is_hit"] = data["predicted_direction"] == data["actual_direction"]
    return data


def _add_factor_columns(scored: pd.DataFrame) -> pd.DataFrame:
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["stage_score_bucket"] = data["stage_score"].apply(_score_bucket)
    data["volatility_bucket"] = data["volatility_20"].apply(_volatility_bucket)
    data["drawdown_bucket"] = data["drawdown"].apply(_drawdown_bucket)
    data["price_vs_ma60"] = _price_vs_ma(data, "ma_60")
    data["price_vs_ma120"] = _price_vs_ma(data, "ma_120")
    return data


def _summarize_group(data: pd.DataFrame, column: str) -> list[dict]:
    output = []
    for value, group in data.groupby(column, dropna=False):
        if len(group) == 0:
            continue
        hit_rate = float(group["is_hit"].mean())
        output.append(
            {
                "factor": column,
                "value": str(value),
                "sample_count": int(len(group)),
                "hit_rate": hit_rate,
                "miss_rate": float(1 - hit_rate),
            }
        )
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


def _price_vs_ma(data: pd.DataFrame, ma_column: str) -> pd.Series:
    return (data["close"] > data[ma_column]).map({True: "above", False: "below"}).fillna("unknown")


def _actual_direction(future_return: float, neutral_threshold: float = 0.005) -> str:
    if pd.isna(future_return):
        return "unknown"
    if future_return > neutral_threshold:
        return "up"
    if future_return < -neutral_threshold:
        return "down"
    return "sideways"
