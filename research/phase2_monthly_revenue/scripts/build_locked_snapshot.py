"""Build an unlabeled, point-in-time feature snapshot inside the research branch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "research/phase2_monthly_revenue"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from research_three_layer_purged import build_frame


REVENUE = WORKSPACE / "data/monthly_revenue_breadth.csv"
SNAPSHOT = WORKSPACE / "data/locked_daily_feature_snapshot.csv"
CASH_MANIFEST = ROOT / "data/processed/factors/cross_sectional_breadth_fetched_dates.csv"

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
REVENUE_FEATURES = [
    "rev_stock_count_z",
    "rev_log_total_z",
    "rev_log_median_z",
    "rev_log_iqr_z",
    "rev_top10_share_z",
    "rev_top50_share_z",
    "rev_hhi_z",
    "rev_yoy_coverage_z",
    "rev_yoy_median_z",
    "rev_yoy_iqr_z",
    "rev_yoy_positive_z",
    "rev_yoy_above20_z",
    "rev_yoy_below_minus20_z",
    "rev_total_yoy_z",
    "rev_positive_change3_z",
    "rev_median_change3_z",
]
BASELINES = [
    "momentum_20",
    "reversal_20",
    "trend_120",
    "revenue_breadth_sign",
    "revenue_median_sign",
    "revenue_total_yoy_sign",
]


def causal_monthly_z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.rolling(60, min_periods=36).mean().shift(1)
    std = values.rolling(60, min_periods=36).std(ddof=0).shift(1).replace(0, np.nan)
    return (values - mean) / std


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    base, _ = build_frame()
    base["date"] = pd.to_datetime(base["date"], errors="raise")
    cash = pd.read_csv(CASH_MANIFEST)
    cash_dates = set(
        pd.to_datetime(
            cash.loc[
                pd.to_numeric(cash["price_source_rows"], errors="coerce")
                .fillna(0)
                .gt(0),
                "date",
            ],
            errors="raise",
        )
    )
    base = base.loc[base["date"].isin(cash_dates)].sort_values("date").copy()
    base["_twii_position"] = np.arange(len(base))

    monthly = pd.read_csv(REVENUE)
    monthly["source_date"] = pd.to_datetime(monthly["source_date"], errors="raise")
    monthly["available_date"] = pd.to_datetime(monthly["available_date"], errors="raise")
    transforms = {
        "rev_stock_count_z": monthly["revenue_stock_count"],
        "rev_log_total_z": np.log(monthly["revenue_total"]),
        "rev_log_median_z": monthly["revenue_log_median"],
        "rev_log_iqr_z": monthly["revenue_log_iqr"],
        "rev_top10_share_z": monthly["revenue_top10_share"],
        "rev_top50_share_z": monthly["revenue_top50_share"],
        "rev_hhi_z": monthly["revenue_hhi"],
        "rev_yoy_coverage_z": (
            monthly["revenue_yoy_valid_count"] / monthly["revenue_stock_count"]
        ),
        "rev_yoy_median_z": monthly["revenue_yoy_median"],
        "rev_yoy_iqr_z": monthly["revenue_yoy_iqr"],
        "rev_yoy_positive_z": monthly["revenue_yoy_positive_fraction"],
        "rev_yoy_above20_z": monthly["revenue_yoy_above20_fraction"],
        "rev_yoy_below_minus20_z": monthly["revenue_yoy_below_minus20_fraction"],
        "rev_total_yoy_z": monthly["revenue_total"].pct_change(12),
        "rev_positive_change3_z": monthly["revenue_yoy_positive_fraction"].diff(3),
        "rev_median_change3_z": monthly["revenue_yoy_median"].diff(3),
    }
    for name, values in transforms.items():
        monthly[name] = causal_monthly_z(values)
    total_yoy = monthly["revenue_total"].pct_change(12)
    monthly["revenue_breadth_sign"] = np.where(
        monthly["revenue_yoy_positive_fraction"] >= 0.5, 1, -1
    )
    monthly["revenue_median_sign"] = np.where(
        monthly["revenue_yoy_median"] >= 0, 1, -1
    )
    monthly["revenue_total_yoy_sign"] = np.where(total_yoy >= 0, 1, -1)

    keep_monthly = ["source_date", "available_date"] + REVENUE_FEATURES + BASELINES[3:]
    frame = pd.merge_asof(
        base.sort_values("date"),
        monthly[keep_monthly].sort_values("available_date"),
        left_on="date",
        right_on="available_date",
        direction="backward",
    )
    frame["momentum_20"] = np.where(frame["ret_20"] >= 0, 1, -1)
    frame["reversal_20"] = -frame["momentum_20"]
    frame["trend_120"] = np.where(frame["ret_120"] >= 0, 1, -1)
    columns = [
        "date",
        "_twii_position",
        "close",
        "source_date",
        "available_date",
    ] + CONTROL + REVENUE_FEATURES + BASELINES
    frame = frame[columns].replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["close"] + CONTROL + REVENUE_FEATURES + BASELINES)
    if not (frame["available_date"] <= frame["date"]).all():
        raise RuntimeError("Point-in-time violation in revenue available_date")
    if frame[CONTROL + REVENUE_FEATURES].isna().any().any():
        raise RuntimeError("Non-finite model feature in locked snapshot")
    atomic_csv(frame.reset_index(drop=True), SNAPSHOT)
    print(
        f"Unlabeled snapshot: {len(frame)} rows, "
        f"{frame['date'].min().date()} to {frame['date'].max().date()}"
    )


if __name__ == "__main__":
    main()
