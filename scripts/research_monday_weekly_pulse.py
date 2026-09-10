"""Test whether Monday observations predict the non-overlapping rest of week."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "reports" / "monday_weekly_pulse_research.json"
OUTPUT_MD = ROOT / "reports" / "monday_weekly_pulse_research.md"
FEATURES = [
    "friday_return", "friday_intraday_return", "friday_monday_consensus",
    "monday_reversal_response", "monday_return", "intraday_return", "gap_return", "tx_night_return",
    "price_advance_decline_breadth", "first_hour_return", "first_hour_signed_ratio",
]


def build_long_history_price_samples() -> pd.DataFrame:
    cash = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
    cash["date"] = pd.to_datetime(cash["date"], errors="coerce")
    cash = cash.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    cash["close"] = pd.to_numeric(cash["close"], errors="coerce")
    cash["daily_return"] = cash["close"].pct_change()
    cash["friday_date"] = cash["date"].shift(1)
    cash["friday_return"] = cash["daily_return"].shift(1)
    iso = cash["date"].dt.isocalendar()
    cash["iso_year"], cash["iso_week"] = iso.year.astype(int), iso.week.astype(int)
    weeks = cash.groupby(["iso_year", "iso_week"]).agg(
        week_days=("date", "size"), week_end_date=("date", "max"),
        week_end_close=("close", "last"),
    ).reset_index()
    samples = cash.loc[cash["date"].dt.weekday.eq(0)].merge(
        weeks, on=["iso_year", "iso_week"], how="inner"
    )
    samples = samples.loc[
        samples["friday_date"].dt.weekday.eq(4) & samples["week_days"].ge(3)
        & samples["week_end_date"].gt(samples["date"])
    ].copy()
    samples["rest_week_return"] = samples["week_end_close"] / samples["close"] - 1
    samples["actual_label"] = np.select(
        [samples["rest_week_return"].gt(0.005), samples["rest_week_return"].lt(-0.005)],
        ["up", "down"], default="sideways",
    )
    friday_sign = np.sign(samples["friday_return"])
    monday_sign = np.sign(samples["daily_return"])
    samples["friday_monday_path"] = np.select(
        [
            (friday_sign > 0) & (monday_sign > 0),
            (friday_sign < 0) & (monday_sign < 0),
            (friday_sign < 0) & (monday_sign > 0),
            (friday_sign > 0) & (monday_sign < 0),
        ],
        ["up_continuation", "down_continuation", "reversal_to_up", "reversal_to_down"],
        default="neutral_transition",
    )
    return samples.reset_index(drop=True)


def association_test(frame: pd.DataFrame, permutations: int = 5000) -> dict:
    path_codes, paths = pd.factorize(frame["friday_monday_path"], sort=True)
    outcome_codes, outcomes = pd.factorize(frame["actual_label"], sort=True)
    rows, columns = len(paths), len(outcomes)

    def statistic(labels):
        observed = np.bincount(
            path_codes * columns + labels, minlength=rows * columns
        ).reshape(rows, columns).astype(float)
        expected = observed.sum(axis=1)[:, None] * observed.sum(axis=0)[None, :] / observed.sum()
        return float(np.divide(
            (observed - expected) ** 2, expected,
            out=np.zeros_like(expected), where=expected > 0,
        ).sum())

    observed = statistic(outcome_codes)
    random = np.random.default_rng(20260819)
    exceedances = sum(
        statistic(random.permutation(outcome_codes)) >= observed
        for _ in range(permutations)
    )
    degrees = min(rows - 1, columns - 1)
    return {
        "chi_square": observed,
        "permutation_count": permutations,
        "permutation_p_value": (exceedances + 1) / (permutations + 1),
        "cramers_v": float(np.sqrt(observed / (len(frame) * degrees))) if degrees > 0 else None,
    }


def era_statistics(frame: pd.DataFrame) -> list[dict]:
    periods = [
        ("1997-2007", "1997-01-01", "2008-01-01"),
        ("2008-2014", "2008-01-01", "2015-01-01"),
        ("2015-2022", "2015-01-01", "2023-01-01"),
        ("2023+", "2023-01-01", "2100-01-01"),
    ]
    output = []
    for label, start, end in periods:
        period = frame.loc[frame["date"].ge(start) & frame["date"].lt(end)]
        monday_prediction = np.where(period["daily_return"].gt(0), "up", "down")
        output.append({
            "period": label,
            "cases": int(len(period)),
            "monday_direction_accuracy": float((monday_prediction == period["actual_label"]).mean()),
            "path_statistics": [
                {
                    "path": path, "cases": int(len(group)),
                    "up_rate": float(group["actual_label"].eq("up").mean()),
                    "down_rate": float(group["actual_label"].eq("down").mean()),
                }
                for path, group in period.groupby("friday_monday_path")
            ],
        })
    return output


def build_samples() -> pd.DataFrame:
    cash = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
    cash["date"] = pd.to_datetime(cash["date"], errors="coerce")
    cash = cash.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    for column in ["open", "high", "low", "close", "volume"]:
        cash[column] = pd.to_numeric(cash[column], errors="coerce")
    cash["prior_close"] = cash["close"].shift(1)
    cash["monday_return"] = cash["close"] / cash["prior_close"] - 1
    cash["intraday_return"] = cash["close"] / cash["open"] - 1
    cash["gap_return"] = cash["open"] / cash["prior_close"] - 1
    cash["close_location"] = (
        (cash["close"] - cash["low"]) / (cash["high"] - cash["low"]).replace(0, np.nan)
    )
    cash["friday_date"] = cash["date"].shift(1)
    cash["friday_return"] = cash["monday_return"].shift(1)
    cash["friday_intraday_return"] = cash["intraday_return"].shift(1)
    cash["friday_close_location"] = cash["close_location"].shift(1)
    iso = cash["date"].dt.isocalendar()
    cash["iso_year"], cash["iso_week"] = iso.year.astype(int), iso.week.astype(int)
    weeks = cash.groupby(["iso_year", "iso_week"]).agg(
        week_days=("date", "size"), week_end_date=("date", "max"),
        week_end_close=("close", "last"),
    ).reset_index()
    samples = cash.loc[cash["date"].dt.weekday.eq(0)].merge(
        weeks, on=["iso_year", "iso_week"], how="inner"
    )
    samples = samples.loc[
        samples["week_days"].ge(3) & samples["week_end_date"].gt(samples["date"])
        & samples["friday_date"].dt.weekday.eq(4)
        & samples["date"].ge("2017-01-01")
    ].copy()
    samples["rest_week_return"] = samples["week_end_close"] / samples["close"] - 1
    samples["actual"] = np.select(
        [samples["rest_week_return"].gt(0.005), samples["rest_week_return"].lt(-0.005)],
        [1, -1], default=0,
    )
    friday_sign = np.sign(samples["friday_return"]).astype(int)
    monday_sign = np.sign(samples["monday_return"]).astype(int)
    samples["friday_monday_consensus"] = np.where(
        friday_sign.eq(monday_sign), monday_sign, 0
    )
    samples["monday_reversal_response"] = np.where(
        friday_sign.eq(-monday_sign), monday_sign, 0
    )
    samples["friday_monday_path"] = np.select(
        [
            (friday_sign > 0) & (monday_sign > 0),
            (friday_sign < 0) & (monday_sign < 0),
            (friday_sign < 0) & (monday_sign > 0),
            (friday_sign > 0) & (monday_sign < 0),
        ],
        ["up_continuation", "down_continuation", "reversal_to_up", "reversal_to_down"],
        default="neutral_transition",
    )

    night = pd.read_csv(ROOT / "data" / "processed" / "taiwan_futures_night.csv")
    night["date"] = pd.to_datetime(night["signal_date"], errors="coerce")
    samples = samples.merge(night[["date", "tx_night_return"]], on="date", how="left")
    breadth = pd.read_csv(ROOT / "data" / "processed" / "factors" / "cross_sectional_breadth.csv")
    breadth["date"] = pd.to_datetime(breadth["date"], errors="coerce")
    samples = samples.merge(
        breadth[["date", "price_advance_decline_breadth"]], on="date", how="left"
    )

    course = pd.read_csv(ROOT / "reports" / "market_intraday_course_15m.csv")
    course["date"] = pd.to_datetime(course["signal_date"], errors="coerce")
    course["bar_start"] = pd.to_datetime(course["bar_start"], errors="coerce")
    start, end = pd.Timestamp("08:45").time(), pd.Timestamp("09:45").time()
    course = course.loc[
        course["bar_start"].dt.time.ge(start) & course["bar_start"].dt.time.lt(end)
    ]
    first = course.groupby("date").agg(
        first_open=("open", "first"), first_close=("close", "last"),
        first_volume=("volume", "sum"), first_signed=("signed_volume", "sum"),
    ).reset_index()
    first["first_hour_return"] = first["first_close"] / first["first_open"] - 1
    first["first_hour_signed_ratio"] = first["first_signed"] / first["first_volume"].replace(0, np.nan)
    return samples.merge(
        first[["date", "first_hour_return", "first_hour_signed_ratio"]], on="date", how="left"
    ).sort_values("date").reset_index(drop=True)


def evaluate(prediction: np.ndarray, actual: np.ndarray) -> dict:
    actionable = prediction != 0
    return {
        "cases": int(len(actual)),
        "coverage": float(actionable.mean()),
        "actionable_accuracy": float((prediction[actionable] == actual[actionable]).mean()) if actionable.any() else None,
        "all_case_accuracy": float((prediction == actual).mean()),
        "down_recall": float((prediction[actual == -1] == -1).mean()),
        "up_recall": float((prediction[actual == 1] == 1).mean()),
    }


def main() -> None:
    data = build_samples()
    long_history = build_long_history_price_samples()
    association = association_test(long_history)
    eras = era_statistics(long_history)
    calibration = data["date"].lt("2023-01-01").to_numpy()
    holdout = ~calibration
    votes = {
        feature: np.sign(pd.to_numeric(data[feature], errors="coerce").fillna(0)).astype(int).to_numpy()
        for feature in FEATURES
    }
    univariate = {}
    for feature, prediction in votes.items():
        univariate[feature] = {
            "calibration": evaluate(prediction[calibration], data.loc[calibration, "actual"].to_numpy()),
            "holdout": evaluate(prediction[holdout], data.loc[holdout, "actual"].to_numpy()),
        }

    candidates = []
    for size in range(2, len(FEATURES) + 1):
        for subset in itertools.combinations(FEATURES, size):
            score = sum(votes[feature] for feature in subset)
            for threshold in range(1, size + 1):
                prediction = np.where(score >= threshold, 1, np.where(score <= -threshold, -1, 0))
                cal = evaluate(prediction[calibration], data.loc[calibration, "actual"].to_numpy())
                if cal["coverage"] < 0.35:
                    continue
                objective = np.mean([cal["all_case_accuracy"], cal["down_recall"], cal["up_recall"]])
                candidates.append({
                    "features": list(subset), "threshold": threshold,
                    "calibration_objective": float(objective), "calibration": cal,
                    "holdout": evaluate(prediction[holdout], data.loc[holdout, "actual"].to_numpy()),
                })
    selected = max(candidates, key=lambda item: item["calibration_objective"])
    holdout_actual = data.loc[holdout, "actual"]
    majority_accuracy = float((holdout_actual == holdout_actual.value_counts().index[0]).mean())
    path_statistics = []
    for path, group in data.groupby("friday_monday_path"):
        item = {"path": path}
        for label, mask in [("calibration", group["date"].lt("2023-01-01")), ("holdout", group["date"].ge("2023-01-01"))]:
            part = group.loc[mask]
            item[label] = {
                "cases": int(len(part)),
                "up_rate": float(part["actual"].eq(1).mean()) if len(part) else None,
                "down_rate": float(part["actual"].eq(-1).mean()) if len(part) else None,
                "sideways_rate": float(part["actual"].eq(0).mean()) if len(part) else None,
                "average_rest_week_return": float(part["rest_week_return"].mean()) if len(part) else None,
            }
        path_statistics.append(item)
    payload = {
        "framework": "monday_weekly_pulse_v1",
        "causal_target": "Monday close to the final cash close of the same ISO week",
        "neutral_threshold": 0.005,
        "sample": {
            "cases": int(len(data)), "first_monday": str(data["date"].min().date()),
            "last_complete_monday": str(data["date"].max().date()),
            "calibration_cases": int(calibration.sum()), "holdout_cases": int(holdout.sum()),
        },
        "long_history_price_layer": {
            "cases": int(len(long_history)),
            "first_monday": str(long_history["date"].min().date()),
            "last_complete_monday": str(long_history["date"].max().date()),
            "association": association,
            "by_era": eras,
        },
        "holdout_majority_accuracy": majority_accuracy,
        "univariate": univariate,
        "friday_monday_path_statistics": path_statistics,
        "selected_calibration_rule": selected,
        "formal_direction_factor_eligible": bool(
            selected["holdout"]["all_case_accuracy"] > majority_accuracy
            and selected["holdout"]["actionable_accuracy"] >= 0.55
        ),
        "interpretation": "Friday supplies the prior state and Monday reveals acceptance or rejection, but the tested direction rules do not beat the holdout majority baseline.",
        "guardrail": "Do not convert Monday colour or a single intraday feature into a fixed weekly direction weight.",
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    monday = univariate["monday_return"]
    night = univariate["tx_night_return"]
    lines = [
        "# 週一脈象與本週剩餘趨勢研究", "",
        f"長歷史價格層：{len(long_history)} 個完整週（{long_history['date'].min().date()} 至 {long_history['date'].max().date()}）。",
        f"樣本：{len(data)} 個不重疊完整週；校準期 {calibration.sum()} 週，2023+ 留出期 {holdout.sum()} 週。",
        "目標嚴格定義為週一收盤到該週最後交易日收盤，避免把週一已發生的漲跌偷算進預測成果。", "",
        "## 結論", "",
        f"- 單用週一漲跌：留出期方向準確率 {monday['holdout']['actionable_accuracy']:.2%}。",
        f"- 單用週一夜盤方向：留出期方向準確率 {night['holdout']['actionable_accuracy']:.2%}。",
        f"- 校準期選出的最佳多因子規則：留出期方向準確率 {selected['holdout']['actionable_accuracy']:.2%}。",
        f"- 留出期無條件多數基準：{majority_accuracy:.2%}。",
        f"- 可否升格固定週方向因子：{'是' if payload['formal_direction_factor_eligible'] else '否'}。", "",
        f"- 長歷史路徑關聯：Cramér's V {association['cramers_v']:.3f}；置換檢定 p={association['permutation_p_value']:.3f}，未達顯著。", "",
        "週五是前置病因，週末資訊與夜盤是傳導，週一是市場接受或否定該病因的第一次正式表態。這條鏈適合判定延續、回補、反轉或震盪病程，而不是直接把紅K／黑K當成整週方向。", "",
        "## 週五—週一路徑", "",
    ]
    for item in path_statistics:
        cal, hold = item["calibration"], item["holdout"]
        lines.append(
            f"- {item['path']}：校準 {cal['cases']} 週（上 {cal['up_rate']:.1%}／下 {cal['down_rate']:.1%}）；"
            f"留出 {hold['cases']} 週（上 {hold['up_rate']:.1%}／下 {hold['down_rate']:.1%}）。"
        )
    lines.extend([
        "", 
        "## 應保留的週一細脈象", "",
        "- 週五收盤方向、收盤位置與量能，作為週末前的持倉伏筆。",
        "- 週一是否延續週五方向，或以跳空／日內走勢明確否定。",
        "- 跳空方向與日內方向是否互相否定。",
        "- 收盤位於當日區間的位置，以及低點回收程度。",
        "- 上下跌家數與價格是否背離。",
        "- 夜盤方向是否被現貨開盤後一小時確認。",
        "- 第一小時報酬與主動成交量是否同向。",
        "- 外資期貨部位與未平倉量是否至少兩項共振。", "",
        "> 目前只適合建立「週五伏筆—週一表態」病程分流監測，不可升格為固定週方向訊號。", "",
    ])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "cases": len(data), "monday_holdout_accuracy": monday["holdout"]["actionable_accuracy"],
        "selected_holdout_accuracy": selected["holdout"]["actionable_accuracy"],
        "majority_accuracy": majority_accuracy,
        "eligible": payload["formal_direction_factor_eligible"],
    }, indent=2))
    print(f"Report: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
