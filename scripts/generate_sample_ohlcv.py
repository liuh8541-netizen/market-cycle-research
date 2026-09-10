import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path


def main() -> None:
    random.seed(7)
    output = Path("data/processed/sample_index_daily.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    current = date(2010, 1, 1)
    price = 8000.0
    trading_days = 0

    while trading_days < 3600:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        cycle = math.sin(trading_days / 180.0)
        long_cycle = math.sin(trading_days / 620.0)
        drift = 0.00025 + 0.00085 * cycle + 0.00055 * long_cycle
        shock = random.gauss(0, 0.011 + 0.006 * max(-cycle, 0))
        day_return = drift + shock

        open_price = price * (1 + random.gauss(0, 0.002))
        close = max(1000.0, price * (1 + day_return))
        high = max(open_price, close) * (1 + abs(random.gauss(0, 0.004)))
        low = min(open_price, close) * (1 - abs(random.gauss(0, 0.004)))
        volume = int(2_000_000_000 * (1 + abs(day_return) * 12 + random.random() * 0.4))

        rows.append(
            {
                "date": current.isoformat(),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume,
            }
        )

        price = close
        trading_days += 1
        current += timedelta(days=1)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

