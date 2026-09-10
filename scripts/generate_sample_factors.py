import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic factor files for pipeline testing.")
    parser.add_argument("--price-input", default="data/processed/twii_daily.csv")
    parser.add_argument("--output-dir", default="data/processed/factors")
    args = parser.parse_args()

    price = pd.read_csv(args.price_input, parse_dates=["date"]).sort_values("date")
    returns = price["close"].pct_change().fillna(0)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "date": price["date"],
            "foreign_net_buy": returns.rolling(5).sum().fillna(0) * 1_000_000_000,
            "investment_trust_net_buy": returns.rolling(20).sum().fillna(0) * 300_000_000,
            "dealer_net_buy": returns.rolling(10).sum().fillna(0) * 150_000_000,
        }
    ).to_csv(root / "institutional_total.csv", index=False)

    pd.DataFrame(
        {
            "date": price["date"],
            "margin_balance": (1 + price["close"].pct_change(20).fillna(0)).cumprod() * 100_000_000,
            "short_balance": (1 - price["close"].pct_change(20).fillna(0)).cumprod() * 10_000_000,
        }
    ).to_csv(root / "margin_total.csv", index=False)

    pd.DataFrame(
        {
            "date": price["date"],
            "contract": "TX",
            "volume": price["volume"].rolling(5).mean().fillna(price["volume"]),
            "open_interest": price["volume"].rolling(20).mean().fillna(price["volume"]) / 1000,
        }
    ).to_csv(root / "futures_daily.csv", index=False)

    pd.DataFrame(
        {
            "date": price["date"],
            "foreign_net": returns.rolling(5).sum().fillna(0) * 10000,
            "dealer_net": returns.rolling(10).sum().fillna(0) * 3000,
        }
    ).to_csv(root / "futures_institutional.csv", index=False)

    pd.DataFrame(
        {
            "date": price["date"],
            "put_volume": (1 - returns.rolling(5).sum().fillna(0)).clip(lower=0.1) * 100000,
            "call_volume": (1 + returns.rolling(5).sum().fillna(0)).clip(lower=0.1) * 100000,
        }
    ).to_csv(root / "option_daily.csv", index=False)

    pd.DataFrame(
        {
            "date": price["date"],
            "put_net": -returns.rolling(10).sum().fillna(0) * 5000,
            "call_net": returns.rolling(10).sum().fillna(0) * 5000,
        }
    ).to_csv(root / "option_institutional.csv", index=False)

    pd.DataFrame(
        {
            "date": price["date"],
            "vix": 18 + price["close"].pct_change().rolling(20).std().fillna(0) * 1000,
        }
    ).to_csv(root / "option_vix.csv", index=False)


if __name__ == "__main__":
    main()

