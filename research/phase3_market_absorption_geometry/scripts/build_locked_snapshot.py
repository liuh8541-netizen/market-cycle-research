"""Build the unlabeled point-in-time snapshot for the absorption test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "research/phase3_market_absorption_geometry"
SOURCE = WORKSPACE / "data/market_absorption_geometry.csv"
SNAPSHOT = WORKSPACE / "data/locked_daily_feature_snapshot.csv"
TWII = ROOT / "data/processed/twii_daily.csv"

CONTROL = [
    "ret_5",
    "ret_20",
    "ret_60",
    "ret_120",
    "ret_240",
    "realized_vol_20",
    "vol_ratio_20_120",
    "drawdown_120",
]
RAW_GEOMETRY = [
    "zero_range_fraction",
    "return_up_fraction",
    "return_down_fraction",
    "return_flat_fraction",
    "return_sign_entropy",
    "return_direction_synchronization",
    "close_location_mean",
    "close_location_median",
    "close_location_dispersion",
    "close_location_money_weighted",
    "low_close_fraction",
    "high_close_fraction",
    "low_close_money_fraction",
    "high_close_money_fraction",
    "down_but_recovered_fraction",
    "up_but_faded_fraction",
    "down_but_recovered_money_fraction",
    "up_but_faded_money_fraction",
    "path_efficiency_median",
    "path_efficiency_q75",
    "one_way_path_fraction",
    "churn_path_fraction",
    "signed_efficiency_median",
    "signed_efficiency_money_weighted",
    "range_median",
    "range_dispersion",
    "range_q90",
    "upper_excursion_q90",
    "lower_excursion_q90",
    "excursion_tail_asymmetry",
    "negative_return_money_fraction",
    "positive_return_money_fraction",
]
GEOMETRY = [f"mag_{column}_z" for column in RAW_GEOMETRY]
BASELINES = [
    "momentum_20",
    "reversal_20",
    "trend_120",
    "breadth_sign",
    "close_location_sign",
    "signed_efficiency_sign",
    "absorption_balance_sign",
]


def causal_z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    prior_mean = values.rolling(252, min_periods=126).mean().shift(1)
    prior_std = (
        values.rolling(252, min_periods=126)
        .std(ddof=0)
        .shift(1)
        .replace(0, np.nan)
    )
    return (values - prior_mean) / prior_std


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def sign(values: pd.Series) -> np.ndarray:
    return np.where(pd.to_numeric(values, errors="coerce") >= 0, 1, -1)


def main() -> None:
    cash = pd.read_csv(TWII)
    cash["information_date"] = pd.to_datetime(cash["date"], errors="raise")
    cash = (
        cash.sort_values("information_date")
        .drop_duplicates("information_date", keep="last")
        .reset_index(drop=True)
    )
    cash["close"] = pd.to_numeric(cash["close"], errors="coerce")
    cash = cash.loc[cash["close"].gt(0)].reset_index(drop=True)
    cash["_twii_position"] = np.arange(len(cash))
    cash["decision_date"] = cash["information_date"].shift(-1)
    daily_return = cash["close"].pct_change()
    for horizon in [5, 20, 60, 120, 240]:
        cash[f"ret_{horizon}"] = cash["close"].pct_change(horizon)
    cash["realized_vol_20"] = daily_return.rolling(20).std(ddof=0)
    vol_120 = daily_return.rolling(120).std(ddof=0).replace(0, np.nan)
    cash["vol_ratio_20_120"] = cash["realized_vol_20"] / vol_120
    cash["drawdown_120"] = cash["close"] / cash["close"].rolling(120).max() - 1
    cash["momentum_20"] = sign(cash["ret_20"])
    cash["reversal_20"] = -cash["momentum_20"]
    cash["trend_120"] = sign(cash["ret_120"])

    source = pd.read_csv(SOURCE)
    source["information_date"] = pd.to_datetime(source["date"], errors="raise")
    if source["information_date"].duplicated().any():
        raise RuntimeError("Duplicate source information dates")
    for raw, transformed in zip(RAW_GEOMETRY, GEOMETRY):
        source[transformed] = causal_z(source[raw])
    source["breadth_sign"] = sign(
        source["return_up_fraction"] - source["return_down_fraction"]
    )
    source["close_location_sign"] = sign(
        source["close_location_money_weighted"] - 0.5
    )
    source["signed_efficiency_sign"] = sign(
        source["signed_efficiency_money_weighted"]
    )
    source["absorption_balance_sign"] = sign(
        source["down_but_recovered_money_fraction"]
        - source["up_but_faded_money_fraction"]
    )

    keep_source = ["information_date"] + GEOMETRY + BASELINES[3:]
    frame = cash.merge(
        source[keep_source],
        on="information_date",
        how="inner",
        validate="one_to_one",
    )
    columns = [
        "information_date",
        "decision_date",
        "_twii_position",
        "close",
    ] + CONTROL + GEOMETRY + BASELINES
    frame = frame[columns].replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(
        subset=["decision_date", "close"] + CONTROL + GEOMETRY + BASELINES
    ).reset_index(drop=True)
    if not (frame["decision_date"] > frame["information_date"]).all():
        raise RuntimeError("Completed-session factors were not delayed one session")
    numeric = frame.drop(columns=["information_date", "decision_date"])
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise RuntimeError("Snapshot contains non-finite numeric values")
    forbidden = [
        column
        for column in frame
        if any(
            token in column.lower()
            for token in ["target", "label", "future", "forward", "hit"]
        )
    ]
    if forbidden:
        raise RuntimeError(f"Target-like columns in snapshot: {forbidden}")
    atomic_csv(frame, SNAPSHOT)
    print(
        f"Unlabeled snapshot: {len(frame)} rows; "
        f"information {frame['information_date'].min().date()} to "
        f"{frame['information_date'].max().date()}; "
        f"decisions {frame['decision_date'].min().date()} to "
        f"{frame['decision_date'].max().date()}"
    )


if __name__ == "__main__":
    main()
