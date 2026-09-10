import pandas as pd

from market_lifecycle.backtest import backtest_regime


def walk_forward_validate(
    scored: pd.DataFrame,
    train_days: int = 756,
    test_days: int = 126,
) -> dict:
    data = scored.sort_values("date").reset_index(drop=True)
    windows = []
    start = 0

    while start + train_days + test_days <= len(data):
        train = data.iloc[start : start + train_days]
        test = data.iloc[start + train_days : start + train_days + test_days]

        metrics = backtest_regime(test)
        metrics["train_start_date"] = str(train["date"].min().date())
        metrics["train_end_date"] = str(train["date"].max().date())
        metrics["test_start_date"] = str(test["date"].min().date())
        metrics["test_end_date"] = str(test["date"].max().date())
        windows.append(metrics)

        start += test_days

    if not windows:
        return {
            "window_count": 0,
            "windows": [],
            "summary": {},
        }

    frame = pd.DataFrame(windows)
    return {
        "window_count": len(windows),
        "windows": windows,
        "summary": {
            "avg_direction_accuracy_5d": float(frame["direction_accuracy_5d"].mean()),
            "avg_strategy_max_drawdown": float(frame["strategy_max_drawdown"].mean()),
            "worst_strategy_max_drawdown": float(frame["strategy_max_drawdown"].min()),
            "avg_strategy_sharpe": float(frame["strategy_sharpe"].mean()),
            "positive_windows": int((frame["strategy_total_return"] > 0).sum()),
        },
    }

