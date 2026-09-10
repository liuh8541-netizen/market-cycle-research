"""Run the independent loan-collateral audit against locked v2 artifacts."""

from pathlib import Path

import audit_loan_collateral_structure as audit


ROOT = Path(__file__).resolve().parents[1]
audit.CONFIG = ROOT / "config/locked_loan_collateral_structure_v2.json"
audit.RESULT = ROOT / "reports/loan_collateral_structure_v2.json"
audit.OUTPUT = ROOT / "reports/loan_collateral_structure_v2_integrity.json"


if __name__ == "__main__":
    audit.main()
