import math
import pandas as pd


def backtest_regime(df: pd.DataFrame) -> dict:
    data = df.copy()
    data["market_return"] = data["close"].pct_change()
    data["position"] = data["risk_regime"].map(
        {"risk_on": 1.0, "neutral": 0.5, "risk_off": 0.0}
    )
    data["strategy_return"] = data["position"].shift(1).fillna(0) * data["market_return"]

    equity = (1 + data["strategy_return"].fillna(0)).cumprod()
    market_equity = (1 + data["market_return"].fillna(0)).cumprod()

    return {
        "rows": int(len(data)),
        "start_date": str(data["date"].min().date()),
        "end_date": str(data["date"].max().date()),
        "strategy_total_return": float(equity.iloc[-1] - 1),
        "market_total_return": float(market_equity.iloc[-1] - 1),
        "strategy_max_drawdown": float(max_drawdown(equity)),
        "market_max_drawdown": float(max_drawdown(market_equity)),
        "strategy_sharpe": float(sharpe_ratio(data["strategy_return"])),
        "direction_accuracy_5d": float(direction_accuracy(data, 5)),
        "risk_off_days": int((data["risk_regime"] == "risk_off").sum()),
    }


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return drawdown.min()


def sharpe_ratio(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.std() == 0 or clean.empty:
        return 0.0
    return (clean.mean() / clean.std()) * math.sqrt(252)


def direction_accuracy(data: pd.DataFrame, horizon: int) -> float:
    future_return = data["close"].pct_change(horizon).shift(-horizon)
    prediction = data["risk_regime"].map(
        {"risk_on": 1, "neutral": 0, "risk_off": -1}
    )
    actual = future_return.apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
    valid = prediction.notna() & actual.notna() & (prediction != 0)
    if valid.sum() == 0:
        return 0.0
    return (prediction[valid] == actual[valid]).mean()

