"""Outcome and auxiliary-vital enrichment for market clinical records."""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd


FORWARD_HORIZONS = [1, 3, 5, 20]


def enrich_forward_outcomes(replay: pd.DataFrame, cash: pd.DataFrame) -> pd.DataFrame:
    output = replay.copy()
    prices = cash.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    for column in ["open", "high", "low", "close"]:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    lookup = {str(date.date()): index for index, date in enumerate(prices["date"])}
    values = {f"forward_{horizon}d_return": [] for horizon in FORWARD_HORIZONS}
    values.update({f"forward_{horizon}d_max_drawdown": [] for horizon in FORWARD_HORIZONS})
    values.update({f"forward_{horizon}d_max_runup": [] for horizon in FORWARD_HORIZONS})
    for signal_date in output["signal_date"].astype(str):
        index = lookup.get(signal_date)
        base = float(prices.iloc[index]["close"]) if index is not None else None
        for horizon in FORWARD_HORIZONS:
            if index is None or index + horizon >= len(prices) or not base:
                ret = drawdown = runup = None
            else:
                future = prices.iloc[index + 1:index + horizon + 1]
                ret = float(prices.iloc[index + horizon]["close"] / base - 1)
                drawdown = float(future["low"].min() / base - 1)
                runup = float(future["high"].max() / base - 1)
            values[f"forward_{horizon}d_return"].append(ret)
            values[f"forward_{horizon}d_max_drawdown"].append(drawdown)
            values[f"forward_{horizon}d_max_runup"].append(runup)
    for column, column_values in values.items():
        output[column] = column_values
    return output


def enrich_auxiliary_vitals(
    replay: pd.DataFrame,
    breadth: pd.DataFrame | None = None,
    futures_daily: pd.DataFrame | None = None,
    futures_institutional: pd.DataFrame | None = None,
    night_microstructure: pd.DataFrame | None = None,
) -> pd.DataFrame:
    output = replay.copy()
    output["_decision_date"] = pd.to_datetime(output["signal_date"])
    if breadth is not None and not breadth.empty:
        output = strict_prior_merge(output, prepare_breadth(breadth), "breadth_date")
        output["breadth_age_days"] = (output["_decision_date"] - output["breadth_date"]).dt.days
    if futures_daily is not None and not futures_daily.empty:
        output = strict_prior_merge(output, prepare_open_interest(futures_daily), "open_interest_date")
        output["open_interest_age_days"] = (output["_decision_date"] - output["open_interest_date"]).dt.days
    if futures_institutional is not None and not futures_institutional.empty:
        output = strict_prior_merge(output, prepare_institutional_position(futures_institutional), "institutional_position_date")
        output["institutional_position_age_days"] = (
            output["_decision_date"] - output["institutional_position_date"]
        ).dt.days
    if night_microstructure is not None and not night_microstructure.empty:
        micro = night_microstructure.copy()
        micro["_decision_date"] = pd.to_datetime(micro["signal_date"], errors="coerce")
        micro = micro.drop(columns=["signal_date"], errors="ignore").add_prefix("micro_")
        micro = micro.rename(columns={"micro__decision_date": "_decision_date"})
        output = output.merge(micro, on="_decision_date", how="left", validate="many_to_one")
    return output.drop(columns=["_decision_date"], errors="ignore")


def current_auxiliary_snapshot(
    decision_date: str,
    breadth: pd.DataFrame | None = None,
    futures_daily: pd.DataFrame | None = None,
    futures_institutional: pd.DataFrame | None = None,
    night_microstructure: pd.DataFrame | None = None,
) -> dict:
    base = pd.DataFrame([{"signal_date": decision_date}])
    enriched = enrich_auxiliary_vitals(
        base, breadth, futures_daily, futures_institutional, night_microstructure
    )
    snapshot = enriched.iloc[0].drop(labels=["signal_date"], errors="ignore").to_dict()
    output = {}
    for key, value in snapshot.items():
        if isinstance(value, pd.Timestamp):
            output[key] = value.date().isoformat()
        elif value is None or (np.isscalar(value) and pd.isna(value)):
            output[key] = None
        elif isinstance(value, np.generic):
            output[key] = value.item()
        else:
            output[key] = value
    return output


def current_cash_technical_snapshot(cash: pd.DataFrame, decision_date: str) -> dict:
    data = cash.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data[data["date"] < pd.to_datetime(decision_date)].sort_values("date")
    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data.empty:
        return {}
    returns = data["close"].pct_change()
    row = data.iloc[-1]
    output = {
        "prior_cash_open": row.get("open"), "prior_cash_high": row.get("high"),
        "prior_cash_low": row.get("low"), "prior_cash_close": row.get("close"),
        "prior_cash_volume": row.get("volume"),
        "prior_realized_volatility_20": returns.tail(20).std(),
        "prior_range_20": data.tail(20)["high"].max() / data.tail(20)["low"].min() - 1,
        "prior_drawdown_60": row.get("close") / data.tail(60)["close"].max() - 1,
    }
    for window in [5, 10, 20, 60]:
        ma = data["close"].tail(window).mean() if len(data) >= window else None
        output[f"prior_ma{window}"] = ma
        output[f"prior_close_vs_ma{window}"] = row.get("close") / ma - 1 if ma else None
    valid_volume = data["volume"].where(data["volume"].gt(0)).tail(20)
    output["prior_volume_ratio_20"] = (
        row.get("volume") / valid_volume.mean()
        if row.get("volume") and valid_volume.notna().any() else None
    )
    return output


def strict_prior_merge(left: pd.DataFrame, right: pd.DataFrame, source_date_column: str) -> pd.DataFrame:
    prepared = right.copy().sort_values(source_date_column)
    return pd.merge_asof(
        left.sort_values("_decision_date"), prepared,
        left_on="_decision_date", right_on=source_date_column,
        direction="backward", allow_exact_matches=False,
    ).sort_index()


def prepare_breadth(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date", "price_advancing_fraction", "price_declining_fraction",
        "price_advance_decline_breadth", "price_up_volume_fraction",
        "price_return_dispersion", "inst_foreign_positive_fraction",
        "inst_foreign_negative_fraction", "inst_combined_breadth",
    ]
    data = frame[[column for column in columns if column in frame]].copy()
    data["breadth_date"] = pd.to_datetime(data.pop("date"), errors="coerce")
    return data.dropna(subset=["breadth_date"]).drop_duplicates("breadth_date", keep="last")


def prepare_open_interest(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data = data[data.get("futures_id").astype(str).eq("TX")]
    if "trading_session" in data:
        data = data[data["trading_session"].astype(str).eq("position")]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["open_interest"] = pd.to_numeric(data["open_interest"], errors="coerce").fillna(0)
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce").fillna(0)
    grouped = data.groupby("date", as_index=False).agg(
        tx_total_open_interest=("open_interest", "sum"), tx_total_volume=("volume", "sum")
    ).sort_values("date")
    grouped["tx_open_interest_change_1d"] = grouped["tx_total_open_interest"].diff()
    grouped["tx_open_interest_change_5d"] = grouped["tx_total_open_interest"].diff(5)
    return grouped.rename(columns={"date": "open_interest_date"})


def prepare_institutional_position(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data = data[data.get("futures_id").astype(str).eq("TX")]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["net_open_interest"] = (
        pd.to_numeric(data["long_open_interest_balance_volume"], errors="coerce").fillna(0)
        - pd.to_numeric(data["short_open_interest_balance_volume"], errors="coerce").fillna(0)
    )
    pivot = data.pivot_table(index="date", columns="institutional_investors", values="net_open_interest", aggfunc="sum")
    rename = {"外資": "foreign_futures_net_oi", "投信": "trust_futures_net_oi", "自營商": "dealer_futures_net_oi"}
    pivot = pivot.rename(columns=rename).reset_index()
    for column in rename.values():
        if column not in pivot:
            pivot[column] = pd.NA
    pivot["institutional_futures_net_oi"] = pivot[list(rename.values())].sum(axis=1, min_count=1)
    return pivot.rename(columns={"date": "institutional_position_date"})


def build_intraday_course_15m(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame is None or frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    data = frame.copy()
    data["bar_start"] = pd.to_datetime(data["bar_start"], errors="coerce")
    data = data.dropna(subset=["bar_start"])
    data = data[
        data["bar_start"].dt.time.ge(time(8, 45))
        & data["bar_start"].dt.time.le(time(13, 45))
    ]
    for column in ["open", "high", "low", "close", "volume", "signed_volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    volumes = data.groupby([data["bar_start"].dt.date, "contract_date"])["volume"].sum()
    selected = volumes.groupby(level=0).idxmax().tolist() if len(volumes) else []
    allowed = {(date, contract) for date, contract in selected}
    mask = [(stamp.date(), contract) in allowed for stamp, contract in zip(data["bar_start"], data["contract_date"])]
    data = data.loc[mask].sort_values("bar_start").copy()
    detail_rows, summaries = [], []
    for date, group in data.groupby(data["bar_start"].dt.date):
        group = group.sort_values("bar_start").copy()
        first_open = float(group.iloc[0]["open"])
        group["return_from_open"] = group["close"] / first_open - 1
        group["cumulative_high"] = group["high"].cummax()
        group["cumulative_low"] = group["low"].cummin()
        group["drawdown_from_running_high"] = group["close"] / group["cumulative_high"] - 1
        group["recovery_from_running_low"] = group["close"] / group["cumulative_low"] - 1
        group["cumulative_volume"] = group["volume"].cumsum()
        group["cumulative_signed_volume"] = group["signed_volume"].fillna(0).cumsum()
        group["signal_date"] = str(date)
        detail_rows.append(group)
        low_row = group.loc[group["low"].idxmin()]
        high_row = group.loc[group["high"].idxmax()]
        summaries.append({
            "signal_date": str(date), "cash_15m_bar_count": int(len(group)),
            "cash_15m_return": float(group.iloc[-1]["close"] / first_open - 1),
            "cash_15m_max_drawdown": float(group["drawdown_from_running_high"].min()),
            "cash_15m_low_time": str(low_row["bar_start"]), "cash_15m_high_time": str(high_row["bar_start"]),
            "cash_15m_signed_volume_ratio": float(group["signed_volume"].fillna(0).sum() / group["volume"].sum()) if group["volume"].sum() else None,
        })
    detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    return detail, pd.DataFrame(summaries)
