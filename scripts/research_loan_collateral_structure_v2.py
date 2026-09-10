"""Locked v2 wrapper retaining only historically continuous credit features."""

from pathlib import Path

import research_loan_collateral_structure as engine


ROOT = Path(__file__).resolve().parents[1]
engine.CONFIG = ROOT / "config/locked_loan_collateral_structure_v2.json"
engine.OUTPUT_JSON = ROOT / "reports/loan_collateral_structure_v2.json"
engine.OUTPUT_MD = ROOT / "reports/loan_collateral_structure_v2.md"
engine.LOAN_FEATURES = [
    feature for feature in engine.LOAN_FEATURES
    if feature not in {
        "lc_unrestricted_share_z",
        "lc_secured_finance_share_z",
        "lc_settlement_share_z",
    }
]


if __name__ == "__main__":
    engine.main()
