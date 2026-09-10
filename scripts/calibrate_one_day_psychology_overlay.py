"""Reproduce the frozen one-day psychology overlay calibration and holdout audit."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_lifecycle.external_market import add_external_market_features
from market_lifecycle.factor_features import add_factor_features
from market_lifecycle.features import add_features
from market_lifecycle.industry_risk import add_memory_industry_risk
from market_lifecycle.lifecycle import score_lifecycle
from market_lifecycle.probability_forecast import ONE_DAY_PSYCHOLOGY_OVERLAY


OUTPUT_JSON = ROOT / "reports" / "one_day_psychology_overlay_calibration.json"
OUTPUT_MD = ROOT / "reports" / "one_day_psychology_overlay_calibration.md"


def scored_market() -> pd.DataFrame:
    price = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv")
    price = add_external_market_features(
        price,
        ROOT / "data" / "processed" / "external_markets.csv",
        ROOT / "data" / "processed" / "taiwan_futures_night.csv",
    )
    data = add_memory_industry_risk(
        score_lifecycle(add_factor_features(add_features(price), ROOT / "data" / "processed" / "factors")),
        ROOT / "config" / "memory_risk_events.csv",
    )
    data = data.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["score_bucket"] = np.select(
        [data["stage_score"].ge(5), data["stage_score"].ge(2), data["stage_score"].gt(-2), data["stage_score"].gt(-5)],
        ["strong_positive", "positive", "neutral", "negative"],
        default="strong_negative",
    )
    return data


def label(value: float) -> str:
    return "up" if value > 0.005 else ("down" if value < -0.005 else "sideways")


def hierarchy(row) -> list[tuple]:
    return [
        (row.lifecycle_stage, row.risk_regime, row.score_bucket),
        (row.risk_regime, row.score_bucket),
        (row.risk_regime,),
        ("all",),
    ]


def generic_replay(data: pd.DataFrame) -> pd.DataFrame:
    keys = [hierarchy(row) for row in data.itertuples()]
    returns = data["close"].shift(-1) / data["close"] - 1
    labels = returns.map(lambda value: label(value) if pd.notna(value) else None).tolist()
    counts: dict[tuple, Counter] = defaultdict(Counter)
    output = []
    for index in range(2, len(data) - 1):
        resolved = index - 2
        for key in keys[resolved]:
            counts[key][labels[resolved]] += 1
        if data.loc[index, "date"] < pd.Timestamp("2017-01-01"):
            continue
        selected = next(
            (counts[key] for key in keys[index] if sum(counts[key].values()) >= 30),
            counts[("all",)],
        )
        predicted = max(["up", "down", "sideways"], key=lambda item: selected[item])
        output.append({
            "signal_date": data.loc[index + 1, "date"],
            "generic_prediction": predicted,
            "actual": labels[index],
        })
    return pd.DataFrame(output)


def metrics(frame: pd.DataFrame, prediction: str) -> dict:
    recalls = {
        direction: float(
            frame.loc[frame["actual"].eq(direction), prediction].eq(direction).mean()
        )
        for direction in ["up", "down", "sideways"]
    }
    return {
        "cases": int(len(frame)),
        "accuracy": float(frame[prediction].eq(frame["actual"]).mean()),
        "macro_recall": float(np.mean(list(recalls.values()))),
        "recall": recalls,
    }


def main() -> None:
    generic = generic_replay(scored_market())
    psychology = pd.read_csv(ROOT / "reports" / "psychology_state_backtest.csv")
    psychology["signal_date"] = pd.to_datetime(psychology["signal_date"], errors="coerce")
    data = generic.merge(psychology, on="signal_date", how="inner", validate="one_to_one")
    data["score"] = pd.to_numeric(data["calibrated_direction_score"], errors="coerce").fillna(0)
    eligible = data["score"].abs().ge(ONE_DAY_PSYCHOLOGY_OVERLAY["minimum_absolute_score"])
    mapped = data["direction"].map({"bullish": "up", "bearish": "down"})
    data["reconciled_prediction"] = np.where(eligible & mapped.notna(), mapped, data["generic_prediction"])
    calibration = data["signal_date"].lt("2023-01-01")
    holdout = ~calibration

    conditional = {}
    for direction in ["bearish", "bullish"]:
        group = data.loc[calibration & eligible & data["direction"].eq(direction)]
        counts = group["actual"].value_counts().reindex(["up", "down", "sideways"], fill_value=0)
        conditional[direction] = {
            "cases": int(len(group)),
            "counts": {key: int(value) for key, value in counts.items()},
            "probabilities": {key: float(value / len(group)) for key, value in counts.items()},
        }
    comparison = {
        "calibration": {
            "baseline": metrics(data.loc[calibration], "generic_prediction"),
            "reconciled": metrics(data.loc[calibration], "reconciled_prediction"),
        },
        "holdout_2023_plus": {
            "baseline": metrics(data.loc[holdout], "generic_prediction"),
            "reconciled": metrics(data.loc[holdout], "reconciled_prediction"),
        },
    }
    drift = {}
    for direction in ["bearish", "bullish"]:
        drift[direction] = {
            key: abs(conditional[direction]["probabilities"][key] - ONE_DAY_PSYCHOLOGY_OVERLAY[direction][key])
            for key in ["up", "down", "sideways"]
        }
    payload = {
        "framework": ONE_DAY_PSYCHOLOGY_OVERLAY["version"],
        "calibration_cutoff": "2023-01-01",
        "neutral_threshold": 0.005,
        "minimum_absolute_score": ONE_DAY_PSYCHOLOGY_OVERLAY["minimum_absolute_score"],
        "conditional_calibration": conditional,
        "comparison": comparison,
        "constant_drift": drift,
        "constants_match_replay": all(value <= 0.000001 for group in drift.values() for value in group.values()),
        "guardrail": "Only the one-day exploratory distribution is reconciled; multi-day and formal signals remain unchanged.",
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    hold = comparison["holdout_2023_plus"]
    lines = [
        "# 一日心理覆寫固定因數校準", "",
        f"- 固定版本：{ONE_DAY_PSYCHOLOGY_OVERLAY['version']}",
        f"- 心理分數門檻：|score| ≥ {ONE_DAY_PSYCHOLOGY_OVERLAY['minimum_absolute_score']}",
        "- 校準期：2017–2022；留出期：2023+。", "",
        "## 固定條件分布", "",
    ]
    for direction, item in conditional.items():
        probabilities = item["probabilities"]
        lines.append(
            f"- {direction}（{item['cases']}件）：上 {probabilities['up']:.2%}、"
            f"下 {probabilities['down']:.2%}、盤整 {probabilities['sideways']:.2%}。"
        )
    lines.extend([
        "", "## 2023+時間留出", "",
        f"- 原模型準確率：{hold['baseline']['accuracy']:.2%}；修正後：{hold['reconciled']['accuracy']:.2%}。",
        f"- 原宏觀召回率：{hold['baseline']['macro_recall']:.2%}；修正後：{hold['reconciled']['macro_recall']:.2%}。",
        f"- 程式固定常數與重算結果一致：{'是' if payload['constants_match_replay'] else '否'}。", "",
        "> 校準報告只支持一日探索性分布，不構成正式交易訊號。", "",
    ])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "cases": len(data), "constants_match": payload["constants_match_replay"],
        "holdout_baseline": hold["baseline"]["accuracy"],
        "holdout_reconciled": hold["reconciled"]["accuracy"],
    }, indent=2))
    print(f"Report: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
