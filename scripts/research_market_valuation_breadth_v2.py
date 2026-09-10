"""Locked v2 wrapper excluding the unlabeled zero-variance yield-valid field."""

from pathlib import Path

import research_market_valuation_breadth as engine


ROOT = Path(__file__).resolve().parents[1]
engine.CONFIG = ROOT / "config/locked_market_valuation_breadth_v2.json"
engine.OUTPUT_JSON = ROOT / "reports/market_valuation_breadth_v2.json"
engine.OUTPUT_MD = ROOT / "reports/market_valuation_breadth_v2.md"
engine.RAW_VALUATION = [
    column for column in engine.RAW_VALUATION
    if column != "yield_valid_fraction"
]
engine.VALUATION = [
    f"valuation_{column}_z" for column in engine.RAW_VALUATION
]


if __name__ == "__main__":
    engine.main()
