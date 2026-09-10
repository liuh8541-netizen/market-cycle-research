import itertools
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RiskPolicy:
    bull_score: int = 2
    bear_score: int = -2
    require_bull_momentum: bool = True
    require_bear_momentum: bool = False


def apply_risk_policy(scored: pd.DataFrame, policy: RiskPolicy) -> pd.DataFrame:
    data = scored.copy()
    regimes = data.apply(lambda row: _risk_regime(row, policy), axis=1)
    data["risk_regime"] = regimes
    data["forward_1w_view"] = regimes.map(_market_view)
    data["forward_4w_view"] = regimes.map(_market_view)
    data["confidence"] = (data["stage_score"].abs() / 8).clip(0, 1)
    return data


def optimize_policy(train_scored: pd.DataFrame, horizon_days: int = 5) -> RiskPolicy:
    candidates = []
    for bull_score, bear_score, require_bull_momentum, require_bear_momentum in itertools.product(
        [1, 2, 3, 4],
        [-1, -2, -3, -4],
        [True, False],
        [True, False],
    ):
        policy = RiskPolicy(
            bull_score=bull_score,
            bear_score=bear_score,
            require_bull_momentum=require_bull_momentum,
            require_bear_momentum=require_bear_momentum,
        )
        evaluated = evaluate_policy(train_scored, policy, horizon_days)
        candidates.append((evaluated["selection_score"], policy))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def evaluate_policy(scored: pd.DataFrame, policy: RiskPolicy, horizon_days: int = 5) -> dict:
    applied = apply_risk_policy(scored, policy)
    future_return = applied["close"].pct_change(horizon_days).shift(-horizon_days)
    prediction = applied["risk_regime"].map({"risk_on": 1, "neutral": 0, "risk_off": -1})
    actual = future_return.apply(lambda value: 1 if value > 0.005 else (-1 if value < -0.005 else 0))

    valid = future_return.notna() & prediction.notna()
    actionable = valid & (prediction != 0)

    overall_hit_rate = _safe_mean(prediction[valid] == actual[valid])
    actionable_hit_rate = _safe_mean(prediction[actionable] == actual[actionable])
    coverage = float(actionable.sum() / valid.sum()) if valid.sum() else 0.0

    selection_score = actionable_hit_rate + 0.20 * coverage
    if coverage < 0.20:
        selection_score -= 0.20

    return {
        "overall_hit_rate": float(overall_hit_rate),
        "actionable_hit_rate": float(actionable_hit_rate),
        "actionable_coverage": coverage,
        "selection_score": float(selection_score),
    }


def _risk_regime(row, policy: RiskPolicy) -> str:
    close = row.get("close")
    ma_20 = row.get("ma_20")
    ma_60 = row.get("ma_60")
    ma_120 = row.get("ma_120")
    stage_score = row.get("stage_score")
    trend_score = row.get("trend_score")
    momentum_score = row.get("momentum_score")
    drawdown = row.get("drawdown")

    if pd.isna(stage_score) or pd.isna(ma_60):
        return "neutral"

    bear_break = (
        pd.notna(ma_120)
        and close < ma_60
        and ma_60 < ma_120
        and trend_score <= -1
    )
    fast_selloff = pd.notna(ma_20) and pd.notna(drawdown) and close < ma_20 and drawdown < -0.08
    bear_momentum_ok = momentum_score <= 0 if policy.require_bear_momentum else True

    if (stage_score <= policy.bear_score and close < ma_60 and bear_momentum_ok) or bear_break or fast_selloff:
        return "risk_off"

    bull_momentum_ok = momentum_score >= 0 if policy.require_bull_momentum else True
    if stage_score >= policy.bull_score and close > ma_60 and bull_momentum_ok:
        return "risk_on"

    return "neutral"


def _market_view(regime: str) -> str:
    if regime == "risk_on":
        return "bullish"
    if regime == "risk_off":
        return "bearish"
    return "neutral"


def _safe_mean(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(values.mean())

