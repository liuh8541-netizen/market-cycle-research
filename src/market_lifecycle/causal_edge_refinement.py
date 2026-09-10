import pandas as pd


DEFAULT_CANDIDATE_FILTERS = [
    ("rsi_14", "<", 30),
    ("rsi_14", "<", 35),
    ("rsi_14", ">", 45),
    ("volatility_20", ">", 0.28),
    ("volatility_20", ">", 0.35),
    ("drawdown", "<", -0.30),
    ("drawdown", "<", -0.40),
    ("cause_return_1d", "<", -0.03),
    ("cause_return_5d", "<", -0.05),
    ("cause_return_20d", "<", -0.10),
    ("trend_score", "<=", -2),
    ("momentum_score", "<=", -2),
    ("stage_score", "<=", -6),
]


def refine_causal_edge(
    causal: pd.DataFrame,
    signature: str,
    horizon_days: int,
    target_direction: str = "up",
    target_hit_rate: float = 0.90,
    min_samples: int = 50,
) -> dict:
    effect_col = f"effect_direction_{horizon_days}d"
    return_col = f"effect_return_{horizon_days}d"
    sample = causal[
        (causal["causal_signature"] == signature)
        & (causal[effect_col] != "unknown")
    ].copy()

    if sample.empty:
        return {"signature": signature, "horizon_days": horizon_days, "sample_count": 0}

    sample["is_success"] = sample[effect_col] == target_direction
    baseline = _metrics(sample, target_direction, return_col)
    comparisons = _compare_success_failure(sample)
    filters = _test_filters(sample, target_direction, return_col, target_hit_rate, min_samples)

    best = filters[0] if filters else None
    return {
        "signature": signature,
        "horizon_days": int(horizon_days),
        "target_direction": target_direction,
        "baseline": baseline,
        "success_failure_comparison": comparisons,
        "candidate_filters": filters,
        "best_filter": best,
        "passes_target": bool(best and best["hit_rate"] >= target_hit_rate and best["sample_count"] >= min_samples),
    }


def _metrics(sample: pd.DataFrame, target_direction: str, return_col: str) -> dict:
    return {
        "sample_count": int(len(sample)),
        "success_count": int((sample["is_success"]).sum()),
        "fail_count": int((~sample["is_success"]).sum()),
        "hit_rate": float(sample["is_success"].mean()),
        "avg_return": float(sample[return_col].mean()),
        "median_return": float(sample[return_col].median()),
        "target_direction": target_direction,
    }


def _compare_success_failure(sample: pd.DataFrame) -> list[dict]:
    numeric_cols = [
        "rsi_14",
        "volatility_20",
        "drawdown",
        "stage_score",
        "trend_score",
        "momentum_score",
        "cause_return_1d",
        "cause_return_5d",
        "cause_return_20d",
        "volume",
    ]
    rows = []
    success = sample[sample["is_success"]]
    failure = sample[~sample["is_success"]]
    for col in numeric_cols:
        if col not in sample.columns:
            continue
        rows.append(
            {
                "factor": col,
                "success_mean": _safe_mean(success[col]),
                "failure_mean": _safe_mean(failure[col]),
                "difference": _safe_mean(success[col]) - _safe_mean(failure[col]),
            }
        )
    return sorted(rows, key=lambda item: abs(item["difference"]), reverse=True)


def _test_filters(
    sample: pd.DataFrame,
    target_direction: str,
    return_col: str,
    target_hit_rate: float,
    min_samples: int,
) -> list[dict]:
    rows = []
    for factor, operator, threshold in DEFAULT_CANDIDATE_FILTERS:
        if factor not in sample.columns:
            continue
        mask = _filter_mask(sample[factor], operator, threshold)
        kept = sample[~mask].copy()
        removed = sample[mask].copy()
        if kept.empty:
            continue
        metrics = _metrics(kept, target_direction, return_col)
        rows.append(
            {
                "filter": f"exclude {factor} {operator} {threshold}",
                "removed_count": int(len(removed)),
                "removed_fail_rate": float((~removed["is_success"]).mean()) if len(removed) else 0.0,
                **metrics,
                "passes_target": bool(metrics["hit_rate"] >= target_hit_rate and metrics["sample_count"] >= min_samples),
            }
        )

    rows = sorted(
        rows,
        key=lambda item: (item["passes_target"], item["hit_rate"], item["sample_count"]),
        reverse=True,
    )
    return rows


def _filter_mask(series: pd.Series, operator: str, threshold: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if operator == "<":
        return values < threshold
    if operator == "<=":
        return values <= threshold
    if operator == ">":
        return values > threshold
    if operator == ">=":
        return values >= threshold
    raise ValueError("Unsupported operator: " + operator)


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return 0.0
    return float(values.mean())

