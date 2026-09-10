"""Run the independent valuation audit against the locked v2 artifacts."""

from pathlib import Path

import audit_market_valuation_breadth as audit


ROOT = Path(__file__).resolve().parents[1]
audit.CONFIG = ROOT / "config/locked_market_valuation_breadth_v2.json"
audit.RESULT = ROOT / "reports/market_valuation_breadth_v2.json"
audit.OUTPUT = ROOT / "reports/market_valuation_breadth_v2_integrity.json"


if __name__ == "__main__":
    audit.main()
