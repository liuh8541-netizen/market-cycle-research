import pandas as pd


STAGE_NAMES = {
    1: "bottoming",
    2: "early_bull",
    3: "main_bull",
    4: "topping",
    5: "early_bear",
    6: "main_bear",
}


def score_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["trend_score"] = data.apply(_trend_score, axis=1)
    data["momentum_score"] = data.apply(_momentum_score, axis=1)
    data["volatility_score"] = data.apply(_volatility_score, axis=1)
    data["drawdown_score"] = data.apply(_drawdown_score, axis=1)

    data["stage_score"] = (
        data["trend_score"]
        + data["momentum_score"]
        + data["volatility_score"]
        + data["drawdown_score"]
        + data.get("non_price_score", 0)
        + data.get("external_score", 0)
        + data.get("night_futures_score", 0)
        + data.get("panic_reversal_score", 0)
    )
    data["lifecycle_stage_id"] = data.apply(_stage_id, axis=1)
    data["lifecycle_stage"] = data["lifecycle_stage_id"].map(STAGE_NAMES)
    data["risk_regime"] = data["lifecycle_stage_id"].map(_risk_regime)
    data["forward_1w_view"] = data["risk_regime"].map(_market_view)
    data["forward_4w_view"] = data["risk_regime"].map(_market_view)
    data["confidence"] = (data["stage_score"].abs() / 8).clip(0, 1)
    return data


def _trend_score(row) -> int:
    score = 0
    close = row.get("close")
    for window in [20, 60, 120, 240]:
        ma = row.get(f"ma_{window}")
        slope = row.get(f"ma_{window}_slope")
        if pd.notna(ma) and close > ma:
            score += 1
        if pd.notna(slope) and slope > 0:
            score += 1
    if score >= 6:
        return 2
    if score >= 4:
        return 1
    if score <= 1:
        return -2
    if score <= 3:
        return -1
    return 0


def _momentum_score(row) -> int:
    rsi = row.get("rsi_14")
    ret_20 = row.get("return_20d")
    if pd.isna(rsi) or pd.isna(ret_20):
        return 0
    if rsi >= 60 and ret_20 > 0:
        return 2
    if rsi >= 50 and ret_20 > 0:
        return 1
    if rsi <= 40 and ret_20 < 0:
        return -2
    if rsi < 50 and ret_20 < 0:
        return -1
    return 0


def _volatility_score(row) -> int:
    vol = row.get("volatility_20")
    ret_20 = row.get("return_20d")
    if pd.isna(vol) or pd.isna(ret_20):
        return 0
    if vol < 0.18 and ret_20 > 0:
        return 1
    if vol > 0.30 and ret_20 < 0:
        return -2
    if vol > 0.24:
        return -1
    return 0


def _drawdown_score(row) -> int:
    drawdown = row.get("drawdown")
    if pd.isna(drawdown):
        return 0
    if drawdown > -0.05:
        return 1
    if drawdown < -0.20:
        return -2
    if drawdown < -0.10:
        return -1
    return 0


def _stage_id(row) -> int:
    score = row["stage_score"]
    close = row.get("close")
    ma_60 = row.get("ma_60")
    ma_120 = row.get("ma_120")
    drawdown = row.get("drawdown")

    if pd.isna(ma_60) or pd.isna(ma_120):
        return 1
    if score >= 5 and close > ma_60 > ma_120:
        return 3
    if score >= 2 and close > ma_60:
        return 2
    if score <= -5 and close < ma_60 < ma_120:
        return 6
    if score <= -2 and close < ma_60:
        return 5
    if pd.notna(drawdown) and drawdown > -0.08 and score <= 1:
        return 4
    return 1


def _risk_regime(stage_id: int) -> str:
    if stage_id in [2, 3]:
        return "risk_on"
    if stage_id in [5, 6]:
        return "risk_off"
    return "neutral"


def _market_view(regime: str) -> str:
    if regime == "risk_on":
        return "bullish"
    if regime == "risk_off":
        return "bearish"
    return "neutral"
