"""Leakage-safe close-range and intraday-course diagnostics.

The formal 0.5% direction target is intentionally left unchanged.  A fuzzy
borderline band is recorded beside it so near-threshold sessions are not
misdescribed as completely neutral, while historical validation remains
comparable with prior reports.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


MATERIAL_RETURN = 0.005
BORDERLINE_RETURN = 0.004
DEEP_INTRADAY_SELL_OFF = -0.008
STRONG_RECOVERY_RATIO = 0.65
CENTRAL_RANGE_QUANTILES = (0.23, 0.77)
OUTER_RANGE_QUANTILES = (0.09, 0.91)


def classify_direction_strength(value: Any, expected_direction: str | None = None) -> dict[str, Any]:
    return_value = _number(value)
    if return_value is None:
        return {
            "label": "unknown", "formal_material": False,
            "borderline_confirmation": False, "threshold_version": "fuzzy_boundary_v1",
        }
    magnitude = abs(return_value)
    actual = "bullish" if return_value > 0 else ("bearish" if return_value < 0 else "sideways")
    same_direction = expected_direction in {None, actual}
    if magnitude >= MATERIAL_RETURN:
        strength = "material"
    elif magnitude >= BORDERLINE_RETURN:
        strength = "borderline"
    elif magnitude > 0:
        strength = "weak"
    else:
        strength = "flat"
    return {
        "label": f"{strength}_{actual}",
        "actual_direction": actual,
        "absolute_return": magnitude,
        "same_as_expected": bool(same_direction),
        "formal_material": bool(same_direction and magnitude >= MATERIAL_RETURN),
        "borderline_confirmation": bool(
            same_direction and BORDERLINE_RETURN <= magnitude < MATERIAL_RETURN
        ),
        "formal_threshold": MATERIAL_RETURN,
        "borderline_lower": BORDERLINE_RETURN,
        "threshold_version": "fuzzy_boundary_v1",
    }


def classify_session_path(
    prior_close: Any, session_open: Any, session_high: Any,
    session_low: Any, session_close: Any,
) -> dict[str, Any]:
    prior, opened, high, low, close = map(
        _number, [prior_close, session_open, session_high, session_low, session_close]
    )
    if any(value is None for value in [prior, opened, high, low, close]) or not prior or not opened:
        return {"label": "unknown", "path_version": "daily_course_v1"}
    day_range = high - low
    close_return = close / prior - 1
    open_to_close = close / opened - 1
    adverse_from_open = low / opened - 1
    favorable_from_open = high / opened - 1
    close_location = (close - low) / day_range if day_range > 0 else 0.5
    recovery_ratio = close_location
    deep_recovered = bool(
        adverse_from_open <= DEEP_INTRADAY_SELL_OFF
        and close_return > 0
        and recovery_ratio >= STRONG_RECOVERY_RATIO
    )
    if deep_recovered:
        label = "deep_selloff_recovered"
    elif close_return >= MATERIAL_RETURN and adverse_from_open > -BORDERLINE_RETURN and close_location >= 0.65:
        label = "steady_bullish"
    elif close_return <= -MATERIAL_RETURN and close_location <= 0.35:
        label = "bearish_expansion"
    elif close_return > 0 and open_to_close < 0:
        label = "gap_gain_intraday_fade"
    elif close_return < 0 and open_to_close > 0:
        label = "gap_loss_intraday_recovery"
    else:
        label = "range_rotation"
    return {
        "label": label,
        "path_version": "daily_course_v1",
        "close_return": close_return,
        "open_to_close_return": open_to_close,
        "adverse_from_open": adverse_from_open,
        "favorable_from_open": favorable_from_open,
        "close_location": close_location,
        "recovery_ratio": recovery_ratio,
        "deep_selloff_recovered": deep_recovered,
        "deep_selloff_threshold": DEEP_INTRADAY_SELL_OFF,
        "recovery_threshold": STRONG_RECOVERY_RATIO,
    }


def historical_close_range(returns: pd.Series, anchor_close: Any) -> dict[str, Any]:
    anchor = _number(anchor_close)
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if anchor is None or not anchor or clean.empty:
        return {"status": "insufficient_data", "sample_count": int(len(clean))}
    quantiles = clean.quantile([
        OUTER_RANGE_QUANTILES[0], CENTRAL_RANGE_QUANTILES[0], 0.50,
        CENTRAL_RANGE_QUANTILES[1], OUTER_RANGE_QUANTILES[1],
    ])
    values = {f"p{int(level * 100):02d}": float(value) for level, value in quantiles.items()}
    close_values = {key: anchor * (1 + value) for key, value in values.items()}
    return {
        "status": "exploratory_historical_distribution",
        "sample_count": int(len(clean)),
        "return_quantiles": values,
        "close_quantiles": close_values,
        "central_50_range": [close_values["p23"], close_values["p77"]],
        "outer_80_range": [close_values["p09"], close_values["p91"]],
        "method": "strictly-prior similar-case return quantiles",
        "quantile_calibration": {
            "version": "range_quantiles_2017_2022_v1",
            "period": "2017-01-01 through 2022-12-31",
            "central": list(CENTRAL_RANGE_QUANTILES),
            "outer": list(OUTER_RANGE_QUANTILES),
            "2023_plus_holdout_coverage": {"central": 0.5171, "outer": 0.7740},
        },
        "formal_target": False,
    }


def historical_path_distribution(pool: pd.DataFrame) -> dict[str, Any]:
    required = {"open", "high", "low", "close"}
    if not required.issubset(pool.columns):
        return {"status": "insufficient_data", "sample_count": 0}
    labels: list[str] = []
    deep_recovered = 0
    prior_low_breach_recovered = 0
    for index in range(len(pool) - 1):
        prior = pool.iloc[index]
        future = pool.iloc[index + 1]
        result = classify_session_path(
            prior.get("close"), future.get("open"), future.get("high"),
            future.get("low"), future.get("close"),
        )
        if result["label"] == "unknown":
            continue
        labels.append(result["label"])
        deep_recovered += int(result.get("deep_selloff_recovered", False))
        prior_low = _number(prior.get("low"))
        future_low = _number(future.get("low"))
        future_close = _number(future.get("close"))
        prior_close = _number(prior.get("close"))
        prior_low_breach_recovered += int(
            None not in {prior_low, future_low, future_close, prior_close}
            and future_low < prior_low and future_close > prior_close
        )
    total = len(labels)
    if not total:
        return {"status": "insufficient_data", "sample_count": 0}
    counts = pd.Series(labels).value_counts()
    return {
        "status": "exploratory_historical_distribution",
        "sample_count": total,
        "rates": {str(label): float(count / total) for label, count in counts.items()},
        "deep_selloff_recovered_rate": deep_recovered / total,
        "prior_low_breach_then_recover_rate": prior_low_breach_recovered / total,
        "formal_signal": False,
        "method": "strictly-prior similar-case next-session OHLC paths",
    }


def build_path_risk_assessment(
    cash: pd.DataFrame, target_date: str,
    breadth: pd.DataFrame | None = None, futures_daily: pd.DataFrame | None = None,
) -> dict[str, Any]:
    data = cash.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ["open", "high", "low", "close"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    target = pd.to_datetime(target_date)
    history = data[data["date"] < target].dropna(subset=["date", "close"]).sort_values("date")
    if history.empty:
        return {"status": "insufficient_data", "target_date": target_date}
    latest = history.iloc[-1]
    prior_close = history.iloc[-2]["close"] if len(history) >= 2 else None
    latest_return = latest["close"] / prior_close - 1 if prior_close else None
    course = classify_session_path(
        prior_close, latest.get("open"), latest.get("high"), latest.get("low"), latest.get("close")
    )
    structure = _prior_structure(history, latest, breadth, futures_daily)
    settlement = _settlement_context(data, target)
    warnings = []
    if course.get("deep_selloff_recovered"):
        warnings.append("前一現貨日曾深殺後收復，次日不可只保留平順延續路徑。")
    if structure.get("breadth", {}).get("index_breadth_divergence"):
        warnings.append("指數與市場廣度背離，權值拉升不等於全面轉強。")
    if structure.get("breadth", {}).get("high_liquidity_concentration"):
        warnings.append("成交金額集中度位於歷史高分位，方向信心降級。")
    if structure.get("futures_basis", {}).get("negative_tail"):
        warnings.append("期貨逆價差位於歷史負向尾部，現貨方向尚未獲期貨完整確認。")
    missing = [
        name for name in ["breadth", "futures_basis"]
        if structure.get(name, {}).get("status") != "current"
    ]
    return {
        "framework": "one_day_path_risk_v1",
        "status": "complete" if not missing else "partial",
        "target_date": target_date,
        "prior_cash_date": str(latest["date"].date()),
        "prior_return_strength": classify_direction_strength(latest_return),
        "prior_session_path": course,
        "settlement_context": settlement,
        "market_structure": structure,
        "warnings": warnings,
        "missing_confirmations": missing,
        "direction_confidence_authority": "reduced" if warnings or missing else "standard",
        "governance": "diagnostic/range authority only; does not override validated direction probabilities",
    }


def _prior_structure(
    cash_history: pd.DataFrame, latest: pd.Series,
    breadth: pd.DataFrame | None, futures_daily: pd.DataFrame | None,
) -> dict[str, Any]:
    date = latest["date"]
    return {
        "breadth": _breadth_snapshot(breadth, date, cash_history),
        "futures_basis": _futures_basis_snapshot(futures_daily, date, latest.get("close"), cash_history),
    }


def _breadth_snapshot(frame: pd.DataFrame | None, date: pd.Timestamp, cash_history: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty or "date" not in frame:
        return {"status": "missing", "reason": "cross-sectional breadth file unavailable"}
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    exact = data[data["date"] == date]
    latest_available = data["date"].max()
    if exact.empty:
        return {
            "status": "stale", "required_date": str(date.date()),
            "latest_available_date": str(latest_available.date()) if pd.notna(latest_available) else None,
            "reason": "same-day breadth/concentration not yet available",
        }
    row = exact.iloc[-1]
    concentration = _number(row.get("price_money_top10_share"))
    past = pd.to_numeric(data.loc[data["date"] < date, "price_money_top10_share"], errors="coerce").dropna() \
        if "price_money_top10_share" in data else pd.Series(dtype=float)
    percentile = float((past <= concentration).mean()) if concentration is not None and len(past) else None
    ad_breadth = _number(row.get("price_advance_decline_breadth"))
    prior_return = cash_history.iloc[-1]["close"] / cash_history.iloc[-2]["close"] - 1 if len(cash_history) >= 2 else None
    return {
        "status": "current", "date": str(date.date()),
        "advance_decline_breadth": ad_breadth,
        "advancing_fraction": _number(row.get("price_advancing_fraction")),
        "up_volume_fraction": _number(row.get("price_up_volume_fraction")),
        "money_top10_share": concentration,
        "money_top10_historical_percentile": percentile,
        "high_liquidity_concentration": bool(percentile is not None and percentile >= 0.80),
        "index_breadth_divergence": bool(prior_return is not None and prior_return > 0 and ad_breadth is not None and ad_breadth < 0),
        "proxy_warning": "money concentration is a liquidity proxy, not exact index-point contribution",
    }


def _futures_basis_snapshot(
    frame: pd.DataFrame | None, date: pd.Timestamp, cash_close: Any, cash_history: pd.DataFrame,
) -> dict[str, Any]:
    if frame is None or frame.empty or "date" not in frame:
        return {"status": "missing", "reason": "futures daily file unavailable"}
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if "trading_session" not in data:
        return {"status": "missing", "reason": "day/night session flag unavailable"}
    day = data[
        data["trading_session"].astype(str).eq("position")
        & data.get("futures_id", pd.Series("", index=data.index)).astype(str).eq("TX")
    ].copy()
    day["volume"] = pd.to_numeric(day.get("volume"), errors="coerce")
    day["close"] = pd.to_numeric(day.get("close"), errors="coerce")
    day = day[(day["volume"] > 0) & (day["close"] > 0)]
    exact = day[day["date"] == date].sort_values("volume", ascending=False)
    if exact.empty:
        latest_available = day["date"].max()
        return {
            "status": "stale", "required_date": str(date.date()),
            "latest_available_date": str(latest_available.date()) if pd.notna(latest_available) else None,
            "reason": "same-day TX day-session settlement/position row not yet available",
        }
    row = exact.iloc[0]
    future_close = float(row["close"])
    cash_value = _number(cash_close)
    basis = future_close - cash_value if cash_value is not None else None
    basis_pct = basis / cash_value if basis is not None and cash_value else None
    historical = []
    cash_lookup = cash_history.set_index("date")["close"]
    for futures_date, group in day[day["date"] < date].groupby("date"):
        if futures_date not in cash_lookup.index:
            continue
        main = group.sort_values("volume", ascending=False).iloc[0]
        historical.append(float(main["close"]) / float(cash_lookup.loc[futures_date]) - 1)
    percentile = float((pd.Series(historical) <= basis_pct).mean()) if historical and basis_pct is not None else None
    return {
        "status": "current", "date": str(date.date()),
        "contract": str(row.get("contract_date")), "futures_close": future_close,
        "cash_close": cash_value, "basis_points": basis, "basis_pct": basis_pct,
        "historical_percentile": percentile,
        "negative_tail": bool(percentile is not None and percentile <= 0.10),
    }


def _settlement_context(cash: pd.DataFrame, target: pd.Timestamp) -> dict[str, Any]:
    settlement = _third_wednesday(target.year, target.month)
    if target <= settlement:
        previous_month = target - pd.offsets.MonthBegin(1)
        settlement = _third_wednesday(previous_month.year, previous_month.month)
    dates = sorted(pd.to_datetime(cash["date"], errors="coerce").dropna().unique())
    completed_after = sum(settlement < pd.Timestamp(value) < target for value in dates)
    offset = completed_after + 1
    historical_returns = []
    prepared = cash.copy().sort_values("date").reset_index(drop=True)
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["return"] = pd.to_numeric(prepared["close"], errors="coerce").pct_change()
    for _, row in prepared[prepared["date"] < target].iterrows():
        row_date = row["date"]
        month_settlement = _third_wednesday(row_date.year, row_date.month)
        if row_date <= month_settlement:
            continue
        month_dates = prepared[(prepared["date"] > month_settlement) & (prepared["date"] <= row_date)]["date"]
        if len(month_dates) == offset and pd.notna(row["return"]):
            historical_returns.append(float(row["return"]))
    values = pd.Series(historical_returns, dtype=float)
    return {
        "monthly_settlement_date": str(settlement.date()),
        "trading_session_offset": offset,
        "label": f"monthly_settlement_plus_{offset}",
        "historical_cases": int(len(values)),
        "historical_up_rate": float((values > 0).mean()) if len(values) else None,
        "historical_average_return": float(values.mean()) if len(values) else None,
        "direction_override_allowed": False,
    }


def _third_wednesday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    days_to_wednesday = (2 - first.weekday()) % 7
    return first + pd.Timedelta(days=days_to_wednesday + 14)


def _number(value: Any) -> float | None:
    try:
        return None if value is None or pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None
