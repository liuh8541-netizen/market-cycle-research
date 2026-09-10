"""Research the causal 08:45-09:00 TX signal before the 09:00 cash open."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BARS_PATH = ROOT / "data" / "processed" / "factors" / "futures_tick_15m.csv"
CASH_PATH = ROOT / "data" / "processed" / "twii_daily.csv"
OUTPUT_CSV = ROOT / "reports" / "preopen_futures_cash_open_cases.csv"
OUTPUT_JSON = ROOT / "reports" / "preopen_futures_cash_open_research.json"
OUTPUT_MD = ROOT / "reports" / "preopen_futures_cash_open_research.md"

GAP_THRESHOLD = 0.0025
MOMENTUM_THRESHOLD = 0.0015
CASH_MOVE_THRESHOLD = 0.003
HOLDOUT_START = pd.Timestamp("2023-01-01")

PATH_LABELS = {
    "gap_up_momentum_up": "期貨溢價且15分鐘續強",
    "gap_up_fading": "期貨溢價但15分鐘轉弱",
    "gap_down_momentum_down": "期貨折價且15分鐘續弱",
    "gap_down_reclaim": "期貨折價但15分鐘收斂",
    "gap_up_only": "期貨溢價、15分鐘動能中性",
    "gap_down_only": "期貨折價、15分鐘動能中性",
    "neutral_mark_up": "接近平盤、15分鐘轉強",
    "neutral_mark_down": "接近平盤、15分鐘轉弱",
    "neutral_wait": "接近平盤且動能不足",
}


def classify_path(mark_return: float, momentum_return: float) -> str:
    if mark_return >= GAP_THRESHOLD:
        if momentum_return >= MOMENTUM_THRESHOLD:
            return "gap_up_momentum_up"
        if momentum_return <= -MOMENTUM_THRESHOLD:
            return "gap_up_fading"
        return "gap_up_only"
    if mark_return <= -GAP_THRESHOLD:
        if momentum_return <= -MOMENTUM_THRESHOLD:
            return "gap_down_momentum_down"
        if momentum_return >= MOMENTUM_THRESHOLD:
            return "gap_down_reclaim"
        return "gap_down_only"
    if momentum_return >= MOMENTUM_THRESHOLD:
        return "neutral_mark_up"
    if momentum_return <= -MOMENTUM_THRESHOLD:
        return "neutral_mark_down"
    return "neutral_wait"


def direction_label(value: pd.Series, threshold: float) -> pd.Series:
    return pd.Series(
        np.select([value >= threshold, value <= -threshold], ["up", "down"], default="sideways"),
        index=value.index,
    )


def build_cases() -> pd.DataFrame:
    bars = pd.read_csv(BARS_PATH)
    bars["bar_start"] = pd.to_datetime(bars["bar_start"], errors="coerce")
    bars["date"] = pd.to_datetime(bars["calendar_date"], errors="coerce")
    preopen = bars.loc[bars["bar_start"].dt.time.eq(pd.Timestamp("08:45").time())].copy()
    # Select the most active contract using only information inside the first
    # 15-minute bar. Whole-day volume would leak post-open contract activity.
    preopen = preopen.sort_values(["date", "volume"]).drop_duplicates("date", keep="last")
    for column in ["open", "high", "low", "close", "volume", "signed_volume"]:
        preopen[column] = pd.to_numeric(preopen[column], errors="coerce")

    cash = pd.read_csv(CASH_PATH)
    cash["date"] = pd.to_datetime(cash["date"], errors="coerce")
    cash = cash.sort_values("date").drop_duplicates("date", keep="last")
    for column in ["open", "high", "low", "close"]:
        cash[column] = pd.to_numeric(cash[column], errors="coerce")
    cash["prior_cash_close"] = cash["close"].shift(1)

    selected = preopen[[
        "date", "contract_date", "open", "high", "low", "close", "volume", "signed_volume"
    ]].rename(columns={
        "open": "tx_open_0845", "high": "tx_high_0845_0900",
        "low": "tx_low_0845_0900", "close": "tx_mark_0900",
        "volume": "tx_volume_0845_0900", "signed_volume": "tx_signed_volume_0845_0900",
    })
    selected["contract_date"] = selected["contract_date"].astype(str)
    cases = selected.merge(
        cash[["date", "prior_cash_close", "open", "high", "low", "close"]],
        on="date", how="inner",
    ).rename(columns={
        "open": "cash_open", "high": "cash_high", "low": "cash_low", "close": "cash_close",
    })
    cases = cases.dropna(subset=[
        "prior_cash_close", "cash_open", "cash_high", "cash_low", "cash_close",
        "tx_open_0845", "tx_mark_0900",
    ]).copy()
    cases = cases.loc[
        cases["prior_cash_close"].gt(0) & cases["cash_open"].gt(0)
        & cases["tx_open_0845"].gt(0) & cases["tx_mark_0900"].gt(0)
    ]

    cases["tx_open_gap_0845"] = cases["tx_open_0845"] / cases["prior_cash_close"] - 1
    cases["tx_mark_return_0900"] = cases["tx_mark_0900"] / cases["prior_cash_close"] - 1
    cases["tx_momentum_0845_0900"] = cases["tx_mark_0900"] / cases["tx_open_0845"] - 1
    cases["tx_range_0845_0900"] = cases["tx_high_0845_0900"] / cases["tx_low_0845_0900"] - 1
    cases["tx_close_position"] = (
        (cases["tx_mark_0900"] - cases["tx_low_0845_0900"])
        / (cases["tx_high_0845_0900"] - cases["tx_low_0845_0900"]).replace(0, np.nan)
    )
    cases["tx_signed_ratio"] = (
        cases["tx_signed_volume_0845_0900"] / cases["tx_volume_0845_0900"].replace(0, np.nan)
    )
    cases["cash_gap"] = cases["cash_open"] / cases["prior_cash_close"] - 1
    cases["cash_open_to_close"] = cases["cash_close"] / cases["cash_open"] - 1
    cases["cash_close_return"] = cases["cash_close"] / cases["prior_cash_close"] - 1
    cases["cash_max_runup_from_open"] = cases["cash_high"] / cases["cash_open"] - 1
    cases["cash_max_drawdown_from_open"] = cases["cash_low"] / cases["cash_open"] - 1
    cases["path"] = [
        classify_path(mark, momentum)
        for mark, momentum in zip(cases["tx_mark_return_0900"], cases["tx_momentum_0845_0900"])
    ]
    cases["cash_gap_label"] = direction_label(cases["cash_gap"], GAP_THRESHOLD)
    cases["predicted_gap_label"] = direction_label(cases["tx_mark_return_0900"], GAP_THRESHOLD)
    cases["cash_intraday_label"] = direction_label(cases["cash_open_to_close"], CASH_MOVE_THRESHOLD)
    return cases.sort_values("date").reset_index(drop=True)


def wilson_interval(successes: int, cases: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if cases == 0:
        return None, None
    p = successes / cases
    denominator = 1 + z * z / cases
    center = (p + z * z / (2 * cases)) / denominator
    margin = z * np.sqrt(p * (1 - p) / cases + z * z / (4 * cases * cases)) / denominator
    return float(center - margin), float(center + margin)


def summarize_group(frame: pd.DataFrame) -> dict:
    cases = len(frame)
    up_count = int(frame["cash_open_to_close"].gt(0).sum())
    low, high = wilson_interval(up_count, cases)
    return {
        "cases": int(cases),
        "avg_cash_gap": float(frame["cash_gap"].mean()) if cases else None,
        "avg_cash_open_to_close": float(frame["cash_open_to_close"].mean()) if cases else None,
        "avg_cash_close_return": float(frame["cash_close_return"].mean()) if cases else None,
        "cash_intraday_up_rate": float(up_count / cases) if cases else None,
        "cash_intraday_up_rate_wilson95": [low, high],
        "material_up_rate": float(frame["cash_intraday_label"].eq("up").mean()) if cases else None,
        "material_down_rate": float(frame["cash_intraday_label"].eq("down").mean()) if cases else None,
        "avg_max_runup_from_open": float(frame["cash_max_runup_from_open"].mean()) if cases else None,
        "avg_max_drawdown_from_open": float(frame["cash_max_drawdown_from_open"].mean()) if cases else None,
    }


def path_statistics(cases: pd.DataFrame) -> list[dict]:
    output = []
    for path in PATH_LABELS:
        group = cases.loc[cases["path"].eq(path)]
        calibration = group.loc[group["date"].lt(HOLDOUT_START)]
        holdout = group.loc[group["date"].ge(HOLDOUT_START)]
        output.append({
            "path": path,
            "label": PATH_LABELS[path],
            "calibration": summarize_group(calibration),
            "holdout": summarize_group(holdout),
        })
    return output


def opening_accuracy(frame: pd.DataFrame) -> dict:
    binary = frame.loc[
        frame["tx_mark_return_0900"].abs().gt(0) & frame["cash_gap"].abs().gt(0)
    ]
    return {
        "cases": int(len(frame)),
        "binary_direction_accuracy": float(
            (np.sign(binary["tx_mark_return_0900"]) == np.sign(binary["cash_gap"])).mean()
        ),
        "three_class_accuracy": float(
            frame["predicted_gap_label"].eq(frame["cash_gap_label"]).mean()
        ),
        "mean_absolute_gap_error": float(
            (frame["tx_mark_return_0900"] - frame["cash_gap"]).abs().mean()
        ),
    }


def tactical_label(stats: dict) -> str:
    cases = stats["cases"]
    rate = stats["cash_intraday_up_rate"]
    interval = stats["cash_intraday_up_rate_wilson95"]
    if cases < 30 or rate is None:
        return "樣本不足，只觀察"
    if interval[0] is not None and interval[0] > 0.5:
        return "歷史偏向開盤後續強"
    if interval[1] is not None and interval[1] < 0.5:
        return "歷史偏向開盤後轉弱"
    return "方向未通過95%區間，僅作風控分流"


def pct(value) -> str:
    return "NA" if value is None or pd.isna(value) else f"{value * 100:.2f}%"


def main() -> None:
    cases = build_cases()
    calibration = cases.loc[cases["date"].lt(HOLDOUT_START)]
    holdout = cases.loc[cases["date"].ge(HOLDOUT_START)]
    paths = path_statistics(cases)
    payload = {
        "status": "retrospective_tactical_research_not_formal_signal",
        "data": {
            "first_date": cases["date"].min().date().isoformat(),
            "last_date": cases["date"].max().date().isoformat(),
            "cases": int(len(cases)),
            "calibration_cases": int(len(calibration)),
            "holdout_cases": int(len(holdout)),
            "available_window": "08:45:00-08:59:59 actual TX trades",
            "unavailable_window": "08:30:00-08:44:59 indicative auction/order book",
        },
        "fixed_thresholds": {
            "tx_mark_vs_prior_cash_close": GAP_THRESHOLD,
            "tx_15m_momentum": MOMENTUM_THRESHOLD,
            "cash_open_to_close_material_move": CASH_MOVE_THRESHOLD,
        },
        "opening_gap_accuracy": {
            "calibration": opening_accuracy(calibration),
            "holdout": opening_accuracy(holdout),
        },
        "overall_cash_path": {
            "calibration": summarize_group(calibration),
            "holdout": summarize_group(holdout),
        },
        "path_statistics": paths,
        "guardrails": [
            "08:30-08:45 is order entry/indicative auction and is not represented by trade bars.",
            "The 08:45 bar must be complete before using its close as the 09:00 mark.",
            "The active contract is selected only by 08:45-09:00 volume; whole-day volume is forbidden.",
            "Opening-gap accuracy is not cash-close direction accuracy.",
            "No path becomes a formal trade signal without prospective validation.",
        ],
    }
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    serializable = cases.copy()
    serializable["date"] = serializable["date"].dt.date.astype(str)
    serializable.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 08:45–09:00台指期對09:00現貨開盤戰術研究", "",
        f"資料期間：{payload['data']['first_date']} 至 {payload['data']['last_date']}，共 {len(cases)} 個配對日。", "",
        "## 資料邊界", "",
        "- 08:30–08:45：集合競價收單／試撮，歷史成交檔沒有可回測成交價。",
        "- 08:45–09:00：台指期已正式交易；使用完整第一根15分鐘K，資料截止嚴格早於現貨開盤後行情。",
        "- 本研究預測的是現貨開盤缺口與開盤後病程，不把開盤方向冒充收盤方向。", "",
        "## 開盤缺口驗證", "",
        "| 區間 | 樣本 | 二元方向命中 | 三分類命中 | 平均絕對誤差 |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, key in [("2020–2022校準", "calibration"), ("2023+時間留出", "holdout")]:
        item = payload["opening_gap_accuracy"][key]
        lines.append(
            f"| {label} | {item['cases']} | {pct(item['binary_direction_accuracy'])} | "
            f"{pct(item['three_class_accuracy'])} | {pct(item['mean_absolute_gap_error'])} |"
        )
    lines.extend(["", "## 09:00後路徑統計（2023+時間留出）", "",
        f"未分組基準：開盤後上漲率 {pct(payload['overall_cash_path']['holdout']['cash_intraday_up_rate'])}，"
        f"平均開到收 {pct(payload['overall_cash_path']['holdout']['avg_cash_open_to_close'])}。", "",
        "| 08:45–09:00脈象 | 樣本 | 開盤後上漲率 | 平均開到收 | 顯著上漲 | 顯著下跌 | 戰術結論 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for item in paths:
        stats = item["holdout"]
        lines.append(
            f"| {item['label']} | {stats['cases']} | {pct(stats['cash_intraday_up_rate'])} | "
            f"{pct(stats['avg_cash_open_to_close'])} | {pct(stats['material_up_rate'])} | "
            f"{pct(stats['material_down_rate'])} | {tactical_label(stats)} |"
        )
    lines.extend([
        "", "## 正式開盤戰術", "",
        "1. 08:30–08:45只看試撮穩定度與掛單變化，不把試撮點位寫入歷史勝率。",
        "2. 08:45–08:59記錄期貨開盤、最後價、區間、成交量與主動量；必須等第一根15分鐘K完整。",
        "3. 09:00先用期貨最後價相對昨現貨收盤判斷開盤缺口；這一層只管理開盤滑價。",
        "4. 再用『溢／折價 × 15分鐘動能』選擇上表病程；若95%區間未排除50%，不得直接追多或追空。",
        "5. 現貨開盤後以5–15分鐘是否守住開盤價、期貨08:45高低點作二次確認；失守即撤銷原路徑。",
        "", "## 限制", "",
        "- 回測是歷史診斷，不是已鎖定的前瞻交易模型。",
        "- 期貨主力契約只依08:45–09:00成交量選取；結算換月日仍須另做分層。",
        "- 未取得歷史試撮委託簿，因此無法驗證08:30–08:45假突破率。",
    ])
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"cases={len(cases)} holdout={len(holdout)} "
        f"gap_direction={payload['opening_gap_accuracy']['holdout']['binary_direction_accuracy']:.2%}"
    )
    print(f"Report: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
