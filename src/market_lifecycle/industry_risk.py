from __future__ import annotations

from pathlib import Path

import pandas as pd


MEMORY_RETURN_COLUMNS = [
    "micron_return_1d",
    "samsung_return_1d",
    "sk_hynix_return_1d",
]


def add_memory_industry_risk(
    scored: pd.DataFrame,
    event_path: str | Path | None = None,
) -> pd.DataFrame:
    """Add a memory-industry tail-risk overlay without rewriting lifecycle scores.

    Market confirmation is derived only from information available on each date.
    Manually curated events can raise alertness, but cannot force risk-off by
    themselves.
    """
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    data = _add_market_signals(data)
    data = _merge_events(data, event_path)
    data["memory_risk_points"] = data.apply(_risk_points, axis=1)
    data["memory_risk_score"] = data["memory_risk_points"].map(_score_from_points)
    data["memory_risk_level"] = data["memory_risk_points"].map(_level_from_points)
    data["memory_market_confirmed"] = data.apply(_market_confirmed, axis=1)
    data["memory_risk_reasons"] = data.apply(_reasons, axis=1)
    data["base_risk_regime"] = data["risk_regime"]
    data["base_confidence"] = data["confidence"]
    data["risk_regime"] = data.apply(_overlay_regime, axis=1)
    data["forward_1w_view"] = data["risk_regime"].map(_market_view)
    data["forward_4w_view"] = data["risk_regime"].map(_market_view)
    penalty = (data["memory_risk_points"] / 200).clip(0, 0.30)
    data["confidence"] = (data["base_confidence"] - penalty).clip(0, 1)
    return data


def _add_market_signals(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    available = [col for col in MEMORY_RETURN_COLUMNS if col in output]
    if available:
        output["memory_constituent_count"] = output[available].notna().sum(axis=1)
        output["memory_basket_return_1d"] = output[available].mean(axis=1, skipna=True)
        output["memory_basket_return_5d"] = (
            (1 + output["memory_basket_return_1d"].fillna(0)).rolling(5).apply(lambda x: x.prod()) - 1
        )
    else:
        output["memory_constituent_count"] = 0
        output["memory_basket_return_1d"] = float("nan")
        output["memory_basket_return_5d"] = float("nan")
    sox = pd.to_numeric(
        output.get("sox_return_1d", pd.Series(float("nan"), index=output.index)),
        errors="coerce",
    )
    output["memory_relative_sox_1d"] = output["memory_basket_return_1d"] - sox
    output["memory_relative_sox_5d"] = output["memory_relative_sox_1d"].rolling(5).sum()
    return output


def _merge_events(data: pd.DataFrame, event_path: str | Path | None) -> pd.DataFrame:
    output = data.copy()
    output["memory_event_severity"] = 0
    output["memory_event_count"] = 0
    output["memory_event_titles"] = ""
    if not event_path or not Path(event_path).exists():
        return output
    events = pd.read_csv(event_path)
    required = {"start_date", "severity", "title"}
    if events.empty or not required.issubset(events.columns):
        return output
    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events.get("end_date", events["start_date"])).fillna(events["start_date"])
    events["severity"] = pd.to_numeric(events["severity"], errors="coerce").fillna(0).clip(0, 5)
    if "confirmed" in events:
        events = events[events["confirmed"].astype(str).str.lower().isin(["1", "true", "yes", "y"])]
    for index, date in output["date"].items():
        active = events[(events["start_date"] <= date) & (events["end_date"] >= date)]
        if active.empty:
            continue
        output.at[index, "memory_event_severity"] = int(active["severity"].max())
        output.at[index, "memory_event_count"] = int(len(active))
        output.at[index, "memory_event_titles"] = "；".join(active["title"].astype(str).tolist())
    return output


def _risk_points(row) -> int:
    points = int(row.get("memory_event_severity", 0)) * 10
    basket_1d = row.get("memory_basket_return_1d")
    basket_5d = row.get("memory_basket_return_5d")
    relative_5d = row.get("memory_relative_sox_5d")
    if pd.notna(basket_1d) and basket_1d <= -0.03:
        points += 20
    elif pd.notna(basket_1d) and basket_1d <= -0.015:
        points += 10
    if pd.notna(basket_5d) and basket_5d <= -0.08:
        points += 25
    elif pd.notna(basket_5d) and basket_5d <= -0.04:
        points += 15
    if pd.notna(relative_5d) and relative_5d <= -0.05:
        points += 20
    elif pd.notna(relative_5d) and relative_5d <= -0.025:
        points += 10
    if row.get("external_score", 0) <= -2:
        points += 10
    return min(points, 100)


def _score_from_points(points: int) -> int:
    if points >= 75:
        return -3
    if points >= 50:
        return -2
    if points >= 25:
        return -1
    return 0


def _level_from_points(points: int) -> str:
    if points >= 75:
        return "critical"
    if points >= 50:
        return "high"
    if points >= 25:
        return "watch"
    return "low"


def _market_confirmed(row) -> bool:
    basket_5d = row.get("memory_basket_return_5d")
    relative_5d = row.get("memory_relative_sox_5d")
    return bool(
        (pd.notna(basket_5d) and basket_5d <= -0.04)
        or (pd.notna(relative_5d) and relative_5d <= -0.025)
    )


def _overlay_regime(row) -> str:
    base = row["base_risk_regime"]
    points = row["memory_risk_points"]
    confirmed = row["memory_market_confirmed"]
    external_negative = row.get("external_score", 0) < 0
    if confirmed and points >= 75 and external_negative:
        return "risk_off"
    if base == "risk_on" and confirmed and points >= 50:
        return "neutral"
    return base


def _reasons(row) -> str:
    reasons = []
    if row.get("memory_event_severity", 0) > 0:
        reasons.append(f"已確認產業事件({int(row['memory_event_severity'])}/5)")
    if pd.notna(row.get("memory_basket_return_5d")) and row["memory_basket_return_5d"] <= -0.04:
        reasons.append("記憶體三大廠近5日轉弱")
    if pd.notna(row.get("memory_relative_sox_5d")) and row["memory_relative_sox_5d"] <= -0.025:
        reasons.append("記憶體股弱於費半")
    if row.get("external_score", 0) <= -2:
        reasons.append("外部市場同步承壓")
    if row.get("memory_constituent_count", 0) == 0:
        reasons.append("當日成分股報價未齊，採近5日訊號")
    return "；".join(reasons) if reasons else "未出現明顯記憶體產業風險"


def _market_view(regime: str) -> str:
    return {"risk_on": "bullish", "risk_off": "bearish"}.get(regime, "neutral")
