"""Independent audit of the locked breadth-transition event study."""

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/locked_breadth_transition_episode_v1.json"
RESULT = ROOT / "reports/breadth_transition_episode.json"
AUDIT = ROOT / "reports/breadth_transition_episode_integrity.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def wilson(hits, cases, z=1.959963984540054):
    p = hits / cases
    d = 1 + z * z / cases
    return (
        p + z * z / (2 * cases)
        - z * math.sqrt(p * (1 - p) / cases + z * z / (4 * cases * cases))
    ) / d


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    checks = {}
    script = ROOT / config["implementation"]["script"]
    checks["implementation_hash"] = (
        sha256(script) == config["implementation"]["sha256"]
        == result["locked_implementation_hash"]
    )
    for name, variant in result["variants"].items():
        records = variant["records"]
        dates = [record["date"] for record in records]
        positions = [record["position"] for record in records]
        hits = sum(
            record["prediction"] == record["target"] for record in records
        )
        checks[f"{name}_unique_dates"] = len(dates) == len(set(dates))
        checks[f"{name}_spacing"] = all(
            b - a >= config["event"]["minimum_origin_spacing"]
            for a, b in zip(positions, positions[1:])
        )
        checks[f"{name}_cases"] = len(records) == variant["cases"]
        checks[f"{name}_hits"] = hits == variant["hits"]
        checks[f"{name}_accuracy"] = math.isclose(
            hits / len(records), variant["accuracy"], abs_tol=1e-15
        )
        checks[f"{name}_wilson"] = math.isclose(
            wilson(hits, len(records)),
            variant["wilson_95_lower"],
            abs_tol=1e-15,
        )
        checks[f"{name}_binary"] = all(
            record["prediction"] in (-1, 1) and record["target"] in (-1, 1)
            for record in records
        )
    control = {
        record["date"]: record["prediction"] == record["target"]
        for record in result["variants"]["price_transition"]["records"]
    }
    treatment = {
        record["date"]: record["prediction"] == record["target"]
        for record in result["variants"]["institutional_override"]["records"]
    }
    increment = result["institutional_increment"]
    checks["identical_control_treatment_dates"] = set(control) == set(treatment)
    checks["treatment_only"] = sum(
        treatment[date] and not control[date] for date in control
    ) == increment["treatment_only"]
    checks["control_only"] = sum(
        control[date] and not treatment[date] for date in control
    ) == increment["control_only"]
    audit = {"passed": all(checks.values()), "checks": checks}
    AUDIT.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not audit["passed"]:
        raise RuntimeError(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
