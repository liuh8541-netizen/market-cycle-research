import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    close = data["close"]
    high = data["high"]
    low = data["low"]

    data["return_1d"] = close.pct_change()
    data["return_5d"] = close.pct_change(5)
    data["return_20d"] = close.pct_change(20)

    for window in [20, 60, 120, 240]:
      data[f"ma_{window}"] = close.rolling(window).mean()
      data[f"ma_{window}_slope"] = data[f"ma_{window}"].pct_change(5)

    data["rsi_14"] = rsi(close, 14)
    data["volatility_20"] = data["return_1d"].rolling(20).std() * (252 ** 0.5)
    data["atr_14"] = atr(high, low, close, 14)
    data["drawdown"] = close / close.cummax() - 1

    return data


def rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()

