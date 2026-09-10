"""Validate close ranges, fuzzy outcomes, and deep-recovery path labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_lifecycle.path_risk import classify_session_path


def main() -> None:
    data = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ["open", "high", "low", "close"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    data["return"] = data["close"].pct_change()
    data["next_return"] = data["close"].shift(-1) / data["close"] - 1
    data["path"] = [
        "unknown" if index == 0 else classify_session_path(
            data.iloc[index - 1]["close"], row["open"], row["high"], row["low"], row["close"]
        )["label"]
        for index, row in data.iterrows()
    ]
    holdout = data[data["date"] >= "2023-01-01"].copy()
    frozen_quantiles = calibrate_quantiles(data)
    range_rows = expanding_range_backtest(
        data, "2023-01-01", None,
        frozen_quantiles["central_lower"], frozen_quantiles["outer_lower"],
    )
    borderline = holdout[holdout["return"].abs().between(0.004, 0.005, inclusive="left")]
    settlement = settlement_plus_one_rows(data)
    settlement_holdout = settlement[settlement["date"] >= "2023-01-01"]
    payload = {
        "framework": "path_risk_adjustment_validation_v1",
        "holdout_period": "2023-01-01 onward",
        "close_range_holdout": summarize_ranges(range_rows),
        "close_range_calibration": frozen_quantiles,
        "borderline_band": {
            "cases": int(len(borderline)),
            "share": float(len(borderline) / len(holdout)) if len(holdout) else None,
            "average_return": float(borderline["return"].mean()) if len(borderline) else None,
            "next_day_same_sign_rate": same_sign_rate(borderline),
            "policy": "retain formal 0.5% target; add descriptive 0.4%-0.5% borderline label",
        },
        "deep_selloff_recovered": path_stats(holdout, "deep_selloff_recovered"),
        "settlement_plus_one": {
            "all_history": simple_stats(settlement),
            "holdout": simple_stats(settlement_holdout),
            "policy": "context diagnostic only; no probability override without independent directional edge",
        },
        "production_decision": {
            "close_quantile_ranges": "enable_as_exploratory",
            "intraday_path_distribution": "enable_as_exploratory",
            "borderline_outcome_label": "enable_beside_legacy_target",
            "settlement_direction_override": "disabled",
            "reason": "new fields improve path/risk communication without post-hoc rewriting of formal accuracy",
        },
    }
    report_dir = ROOT / "reports"
    (report_dir / "path_risk_adjustment_validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "path_risk_adjustment_validation.md").write_text(render(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def expanding_range_backtest(
    data: pd.DataFrame, start: str, end: str | None,
    central_lower: float, outer_lower: float, lookback: int = 756,
) -> pd.DataFrame:
    rows = []
    date_mask = data["date"] >= start
    if end:
        date_mask &= data["date"] <= end
    for index in data.index[date_mask]:
        history = data.loc[max(1, index - lookback):index - 1, "return"].dropna()
        if len(history) < 252:
            continue
        actual = data.loc[index, "return"]
        quantiles = history.quantile(sorted({
            outer_lower, central_lower, 0.50, 1 - central_lower, 1 - outer_lower,
        }))
        rows.append({
            "date": str(data.loc[index, "date"].date()), "actual": actual,
            "central_hit": bool(quantiles.loc[central_lower] <= actual <= quantiles.loc[1 - central_lower]),
            "outer_hit": bool(quantiles.loc[outer_lower] <= actual <= quantiles.loc[1 - outer_lower]),
            "median_error": abs(actual - quantiles.loc[0.50]),
            "no_change_error": abs(actual),
        })
    return pd.DataFrame(rows)


def calibrate_quantiles(data: pd.DataFrame) -> dict:
    candidates = [value / 100 for value in range(3, 31)]
    cache = {
        value: expanding_range_backtest(data, "2017-01-01", "2022-12-31", value, value)
        for value in candidates
    }
    central = min(candidates, key=lambda value: abs(float(cache[value]["central_hit"].mean()) - 0.50))
    outer = min(candidates, key=lambda value: abs(float(cache[value]["outer_hit"].mean()) - 0.80))
    return {
        "period": "2017-01-01 through 2022-12-31",
        "central_lower": central, "central_upper": 1 - central,
        "outer_lower": outer, "outer_upper": 1 - outer,
        "central_calibration_coverage": float(cache[central]["central_hit"].mean()),
        "outer_calibration_coverage": float(cache[outer]["outer_hit"].mean()),
        "selection_rule": "symmetric quantiles closest to 50%/80% coverage on calibration only",
    }


def summarize_ranges(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"cases": 0}
    return {
        "cases": int(len(frame)),
        "central_50_coverage": float(frame["central_hit"].mean()),
        "outer_80_coverage": float(frame["outer_hit"].mean()),
        "median_return_mae": float(frame["median_error"].mean()),
        "no_change_baseline_mae": float(frame["no_change_error"].mean()),
    }


def settlement_plus_one_rows(data: pd.DataFrame) -> pd.DataFrame:
    mask = []
    for index, row in data.iterrows():
        if index == 0:
            mask.append(False)
            continue
        previous = data.iloc[index - 1]["date"]
        mask.append(previous.weekday() == 2 and 15 <= previous.day <= 21)
    return data[pd.Series(mask, index=data.index)].copy()


def simple_stats(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"cases": 0}
    return {
        "cases": int(len(frame)), "up_rate": float((frame["return"] > 0).mean()),
        "average_return": float(frame["return"].mean()), "median_return": float(frame["return"].median()),
        "deep_selloff_recovered_rate": float((frame["path"] == "deep_selloff_recovered").mean()),
    }


def path_stats(frame: pd.DataFrame, label: str) -> dict:
    selected = frame[frame["path"] == label]
    return {
        "cases": int(len(selected)),
        "rate": float(len(selected) / len(frame)) if len(frame) else None,
        "average_close_return": float(selected["return"].mean()) if len(selected) else None,
        "next_day_up_rate": float((selected["next_return"] > 0).mean()) if len(selected) else None,
        "next_day_average_return": float(selected["next_return"].mean()) if len(selected) else None,
    }


def same_sign_rate(frame: pd.DataFrame) -> float | None:
    valid = frame.dropna(subset=["return", "next_return"])
    if valid.empty:
        return None
    return float(((valid["return"] > 0) == (valid["next_return"] > 0)).mean())


def render(payload: dict) -> str:
    ranges = payload["close_range_holdout"]
    border = payload["borderline_band"]
    deep = payload["deep_selloff_recovered"]
    settle = payload["settlement_plus_one"]["holdout"]
    return "\n".join([
        "# 收盤區間與盤中病程修正驗證", "",
        f"- 2023+區間樣本: {ranges.get('cases', 0)}",
        f"- 中央50%實際覆蓋: {ranges.get('central_50_coverage', 0):.2%}",
        f"- 外圍80%實際覆蓋: {ranges.get('outer_80_coverage', 0):.2%}",
        f"- 歷史中位數MAE: {ranges.get('median_return_mae', 0):.2%}；不變基準MAE: {ranges.get('no_change_baseline_mae', 0):.2%}",
        "", "## 臨界漲跌", "",
        f"- 0.4%～0.5%病例: {border.get('cases', 0)}；占比 {border.get('share', 0):.2%}",
        "- 正式0.5%績效口徑不變，另記臨界確認，避免把接近門檻的正確方向寫成完全中性。",
        "", "## 深殺後收復", "",
        f"- 2023+病例: {deep.get('cases', 0)}；發生率 {deep.get('rate', 0):.2%}",
        f"- 次日上漲率: {deep.get('next_day_up_rate', 0):.2%}；次日平均 {deep.get('next_day_average_return', 0):.2%}",
        "", "## 結算後第一日", "",
        f"- 2023+病例: {settle.get('cases', 0)}；上漲率 {settle.get('up_rate', 0):.2%}；平均 {settle.get('average_return', 0):.2%}",
        "- 只保留情境標籤，不直接覆寫方向機率。", "",
    ])


if __name__ == "__main__":
    main()
