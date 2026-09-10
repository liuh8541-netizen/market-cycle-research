import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_lifecycle.night_cash_tracking import write_night_cash_tracking


def main():
    parser = argparse.ArgumentParser(description="Track completed TX night-session impact on TAIEX cash sessions.")
    parser.add_argument("--night", default="data/processed/taiwan_futures_night.csv")
    parser.add_argument("--cash", default="data/processed/twii_daily.csv")
    args = parser.parse_args()
    summary = write_night_cash_tracking(
        ROOT / args.night,
        ROOT / args.cash,
        ROOT / "data/processed/night_cash_impact_log.csv",
        ROOT / "reports/night_cash_impact_tracking.json",
        ROOT / "reports/night_cash_impact_tracking.md",
    )
    print(f"Night/cash impact tracking updated through {summary['latest_date']} ({summary['rows']} rows).")


if __name__ == "__main__":
    main()
