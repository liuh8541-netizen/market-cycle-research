"""Build an unlabeled point-in-time snapshot for the final credit-short test."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "research/phase2_credit_short_inventory"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from research_three_layer_purged import build_frame


SOURCE = WORKSPACE / "data/credit_short_inventory.csv"
SNAPSHOT = WORKSPACE / "data/locked_daily_feature_snapshot.csv"
CASH_MANIFEST = (
    ROOT / "data/processed/factors/cross_sectional_breadth_fetched_dates.csv"
)
CONTROL = [
    "ret_5",
    "ret_20",
    "ret_60",
    "ret_120",
    "ret_240",
    "vol_ratio",
    "drawdown_120",
    "pulse_rate",
    "inst_foreign_net_20d_z",
    "inst_trust_net_20d_z",
    "inst_dealer_net_20d_z",
]
CREDIT_FEATURES = [
    "cs_log_combined_balance_z",
    "cs_log_margin_balance_z",
    "cs_log_sbl_balance_z",
    "cs_margin_balance_share_z",
    "cs_combined_balance_change_rate_z",
    "cs_margin_increase_breadth_z",
    "cs_sbl_increase_breadth_z",
    "cs_margin_new_short_rate_z",
    "cs_margin_cover_rate_z",
    "cs_sbl_new_short_rate_z",
    "cs_sbl_return_rate_z",
    "cs_positive_balance_breadth_z",
    "cs_top10_share_z",
    "cs_top50_share_z",
    "cs_hhi_z",
    "cs_log_iqr_z",
]
BASELINES = [
    "momentum_20",
    "reversal_20",
    "trend_120",
    "short_build_bearish",
    "short_build_contrarian",
    "margin_build_bearish",
    "sbl_build_bearish",
]


def causal_z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.rolling(252, min_periods=126).mean().shift(1)
    std = (
        values.rolling(252, min_periods=126)
        .std(ddof=0)
        .shift(1)
        .replace(0, np.nan)
    )
    return (values - mean) / std


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    base, _ = build_frame()
    base["date"] = pd.to_datetime(base["date"], errors="raise")
    cash = pd.read_csv(CASH_MANIFEST)
    cash["date"] = pd.to_datetime(cash["date"], errors="raise")
    cash["price_source_rows"] = pd.to_numeric(
        cash["price_source_rows"], errors="coerce"
    ).fillna(0)
    cash_dates = (
        cash.loc[cash["price_source_rows"].gt(0), "date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    next_cash = dict(zip(cash_dates.iloc[:-1], cash_dates.iloc[1:]))
    base = base.loc[base["date"].isin(set(cash_dates))].sort_values("date").copy()
    base["_twii_position"] = np.arange(len(base))

    source = pd.read_csv(SOURCE)
    source["information_date"] = pd.to_datetime(source["date"], errors="raise")
    source["decision_date"] = source["information_date"].map(next_cash)
    previous_total = (
        pd.to_numeric(source["margin_previous_balance_total"], errors="coerce")
        + pd.to_numeric(source["sbl_previous_balance_total"], errors="coerce")
    )
    transforms = {
        "cs_log_combined_balance_z": np.log1p(
            source["combined_current_balance_total"]
        ),
        "cs_log_margin_balance_z": np.log1p(
            source["margin_current_balance_total"]
        ),
        "cs_log_sbl_balance_z": np.log1p(source["sbl_current_balance_total"]),
        "cs_margin_balance_share_z": source["margin_balance_share"],
        "cs_combined_balance_change_rate_z": (
            source["combined_balance_change_total"]
            / previous_total.replace(0, np.nan)
        ),
        "cs_margin_increase_breadth_z": source[
            "margin_balance_increase_fraction"
        ],
        "cs_sbl_increase_breadth_z": source["sbl_balance_increase_fraction"],
        "cs_margin_new_short_rate_z": source["margin_new_short_to_previous"],
        "cs_margin_cover_rate_z": source["margin_cover_to_previous"],
        "cs_sbl_new_short_rate_z": source["sbl_new_short_to_previous"],
        "cs_sbl_return_rate_z": source["sbl_return_to_previous"],
        "cs_positive_balance_breadth_z": source[
            "positive_combined_balance_fraction"
        ],
        "cs_top10_share_z": source["combined_top10_share"],
        "cs_top50_share_z": source["combined_top50_share"],
        "cs_hhi_z": source["combined_hhi"],
        "cs_log_iqr_z": source["combined_log_iqr"],
    }
    for name, values in transforms.items():
        source[name] = causal_z(values)

    combined_change = pd.to_numeric(
        source["combined_balance_change_total"], errors="coerce"
    )
    margin_change = (
        pd.to_numeric(source["margin_current_balance_total"], errors="coerce")
        - pd.to_numeric(source["margin_previous_balance_total"], errors="coerce")
    )
    sbl_change = (
        pd.to_numeric(source["sbl_current_balance_total"], errors="coerce")
        - pd.to_numeric(source["sbl_previous_balance_total"], errors="coerce")
    )
    source["short_build_bearish"] = np.where(combined_change >= 0, -1, 1)
    source["short_build_contrarian"] = -source["short_build_bearish"]
    source["margin_build_bearish"] = np.where(margin_change >= 0, -1, 1)
    source["sbl_build_bearish"] = np.where(sbl_change >= 0, -1, 1)
    keep_source = [
        "information_date",
        "decision_date",
    ] + CREDIT_FEATURES + BASELINES[3:]
    frame = base.merge(
        source[keep_source],
        left_on="date",
        right_on="information_date",
        how="inner",
        validate="one_to_one",
    )
    frame["momentum_20"] = np.where(frame["ret_20"] >= 0, 1, -1)
    frame["reversal_20"] = -frame["momentum_20"]
    frame["trend_120"] = np.where(frame["ret_120"] >= 0, 1, -1)
    columns = [
        "information_date",
        "decision_date",
        "_twii_position",
        "close",
    ] + CONTROL + CREDIT_FEATURES + BASELINES
    frame = frame[columns].replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(
        subset=["close", "decision_date"] + CONTROL + CREDIT_FEATURES + BASELINES
    )
    if not (frame["decision_date"] > frame["information_date"]).all():
        raise RuntimeError("Credit data was not delayed to the next cash session")
    if frame[CONTROL + CREDIT_FEATURES].isna().any().any():
        raise RuntimeError("Non-finite model feature in snapshot")
    atomic_csv(frame.reset_index(drop=True), SNAPSHOT)
    print(
        f"Unlabeled snapshot: {len(frame)} rows, "
        f"information {frame['information_date'].min().date()} to "
        f"{frame['information_date'].max().date()}, decisions "
        f"{frame['decision_date'].min().date()} to "
        f"{frame['decision_date'].max().date()}"
    )


if __name__ == "__main__":
    main()

