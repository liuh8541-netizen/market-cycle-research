"""Independent audit of the locked monthly-revenue formal result."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "research/phase2_monthly_revenue"
CONFIG = WORKSPACE / "config/locked_monthly_revenue_breadth_v1.json"
RESULT = WORKSPACE / "reports/monthly_revenue_breadth_v1.json"
RECEIPT = WORKSPACE / "reports/formal_run_receipt.json"
OUTPUT = WORKSPACE / "reports/monthly_revenue_breadth_v1_integrity.json"
HORIZON = 20


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def wilson(hits: int, cases: int, z: float = 1.959963984540054) -> float:
    if not cases:
        return 0.0
    p = hits / cases
    denominator = 1 + z * z / cases
    center = p + z * z / (2 * cases)
    spread = z * math.sqrt(p * (1 - p) / cases + z * z / (4 * cases * cases))
    return (center - spread) / denominator


def mcnemar(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(left_only, right_only) + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    checks = {
        "receipt_complete": receipt.get("status") == "complete",
        "receipt_result_hash": receipt.get("result_sha256") == sha256(RESULT),
        "result_snapshot_hash": result["snapshot_sha256"]
        == config["hashes"]["research/phase2_monthly_revenue/data/locked_daily_feature_snapshot.csv"],
        "all_locked_hashes": all(
            sha256(ROOT / relative) == digest
            for relative, digest in config["hashes"].items()
        ),
        "aligned_rows_locked": result["aligned_rows"]
        == config["data_semantics"]["aligned_rows"],
    }
    for name, variant in result["variants"].items():
        records = variant["records"]
        positions = [int(record["position"]) for record in records]
        hits = sum(record["prediction"] == record["target"] for record in records)
        checks[f"{name}_unique_dates"] = len(records) == len(
            {record["date"] for record in records}
        )
        checks[f"{name}_spacing"] = all(
            right - left >= HORIZON for left, right in zip(positions, positions[1:])
        )
        checks[f"{name}_cases"] = len(records) == variant["cases"]
        checks[f"{name}_hits"] = hits == variant["hits"]
        checks[f"{name}_accuracy"] = math.isclose(
            hits / len(records) if records else 0.0,
            variant["accuracy"],
            abs_tol=1e-15,
        )
        checks[f"{name}_wilson"] = math.isclose(
            wilson(hits, len(records)), variant["wilson_95_lower"], abs_tol=1e-15
        )
        checks[f"{name}_spacing_count"] = len(positions) == len(set(positions))
        baseline = variant["strongest_baseline"]
        baseline_hits = sum(
            record[baseline] == record["target"] for record in records
        )
        checks[f"{name}_baseline_hits"] = (
            baseline_hits == variant["baselines"][baseline]["hits"]
        )

    left = {
        row["date"]: bool(row["hit"])
        for row in result["variants"]["index_aggregate_control"]["records"]
    }
    right = {
        row["date"]: bool(row["hit"])
        for row in result["variants"]["plus_monthly_revenue_breadth"]["records"]
    }
    common = set(left).intersection(right)
    control_only = sum(left[day] and not right[day] for day in common)
    treatment_only = sum(right[day] and not left[day] for day in common)
    increment = result["revenue_increment"]
    checks["paired_dates"] = len(common) == increment["identical_dates"]
    checks["paired_control_only"] = control_only == increment["control_only"]
    checks["paired_treatment_only"] = treatment_only == increment["treatment_only"]
    checks["paired_p"] = math.isclose(
        mcnemar(control_only, treatment_only),
        increment["mcnemar_exact_p"],
        abs_tol=1e-15,
    )
    checks = {name: bool(value) for name, value in checks.items()}
    payload = {"passed": all(checks.values()), "checks": checks}
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not payload["passed"]:
        raise RuntimeError(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
