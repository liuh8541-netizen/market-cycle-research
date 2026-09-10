"""Lock pre-open clinical cases and append settled outcomes without rewriting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_lifecycle.prospective_clinical import (
    load_jsonl, render_registry, update_prospective_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update append-only prospective market clinical registry.")
    parser.add_argument("--records", default="research/market_health_knowledge/complete_market_records.jsonl")
    parser.add_argument("--predictions", default="research/market_health_knowledge/prospective_predictions.jsonl")
    parser.add_argument("--outcomes", default="research/market_health_knowledge/prospective_outcomes.jsonl")
    parser.add_argument("--json-output", default="reports/prospective_clinical_registry.json")
    parser.add_argument("--md-output", default="reports/prospective_clinical_registry.md")
    args = parser.parse_args()
    records = load_jsonl(rooted(args.records))
    summary = update_prospective_registry(records, rooted(args.predictions), rooted(args.outcomes))
    json_output, md_output = rooted(args.json_output), rooted(args.md_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_output.write_text(render_registry(summary), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ["prediction_count", "settled_count", "pending_count", "prediction_chain_valid", "outcome_chain_valid"]}, ensure_ascii=False))


def rooted(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
