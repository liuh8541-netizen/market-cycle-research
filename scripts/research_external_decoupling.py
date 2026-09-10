"""Strict nested walk-forward test of when Taiwan's open decouples from US markets."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_afterhours_institutional as base
import research_afterhours_with_complexion as extended

TRAIN, VALID, TEST = 756, 126, 126
PULSE = ["futures_total", "futures_foreign", "futures_trust", "futures_dealer",
         "option_total", "option_foreign", "option_trust", "option_dealer"]
EXTERNAL_VOTES = ["sp500_return_1d", "nasdaq_return_1d", "sox_return_1d",
                  "dow_return_1d", "tsm_adr_return_1d"]


def prepare():
    x = extended.prepare().sort_values("date").reset_index(drop=True)
    votes = np.sign(x[EXTERNAL_VOTES].fillna(0))
    x["external_vote_sum"] = votes.sum(axis=1)
    x["baseline"] = np.where(x.external_vote_sum >= 0, 1, -1)
    x["external_strength"] = x.external_vote_sum.abs() / len(EXTERNAL_VOTES)
    x["external_mean"] = x[EXTERNAL_VOTES].mean(axis=1)
    x["external_dispersion"] = x[EXTERNAL_VOTES].std(axis=1)
    x["target"] = np.where(x.gap_return >= 0, 1, -1)
    # Current premarket observations standardized only by observations strictly before D.
    for c in PULSE + ["tx_night_return", "tx_night_spread_per"]:
        s = pd.to_numeric(x[c], errors="coerce")
        mean = s.shift(1).rolling(252, min_periods=63).mean()
        std = s.shift(1).rolling(252, min_periods=63).std().replace(0, np.nan)
        x[c + "_z"] = ((s - mean) / std).clip(-8, 8)
    zcols = [c + "_z" for c in PULSE]
    x["pulse_mean_z"] = x[zcols].mean(axis=1)
    x["pulse_external_divergence"] = x.pulse_mean_z * x.baseline
    x["foreign_divergence"] = ((x.futures_foreign_z + x.option_foreign_z) / 2) * x.baseline
    x["option_divergence"] = x.option_total_z * x.baseline
    x["futures_divergence"] = x.futures_total_z * x.baseline
    return x.dropna(subset=["target", "baseline", "external_strength"]).reset_index(drop=True)


RULE_FEATURES = [
    "pulse_external_divergence", "foreign_divergence", "option_divergence",
    "futures_divergence", "tx_night_return_z", "tx_night_spread_per_z",
    "external_dispersion", "vix_return_1d",
]


def apply_rule(frame, feature, cut, side):
    value = pd.to_numeric(frame[feature], errors="coerce")
    override = value <= cut if side == "low" else value >= cut
    pred = frame.baseline.to_numpy(int).copy()
    pred[override.fillna(False).to_numpy()] *= -1
    return pred, override.fillna(False).to_numpy()


def select(train):
    fit, valid = train.iloc[:-VALID], train.iloc[-VALID:]
    best = None
    # Select actionable consensus and one interpretable override rule using validation only.
    for min_strength in [0.2, 0.6, 1.0]:
        eligible = valid.external_strength.to_numpy() >= min_strength
        if eligible.sum() < 13:
            continue
        baseline_acc = float((valid.baseline.to_numpy(int)[eligible] == valid.target.to_numpy(int)[eligible]).mean())
        # No-override is explicitly included and prevents forced complexity.
        candidates = [(None, 0.0, "none")]
        for feature in RULE_FEATURES:
            values = pd.to_numeric(fit[feature], errors="coerce").dropna()
            if len(values) < 100:
                continue
            for q in [.05, .10, .15, .20, .25, .75, .80, .85, .90, .95]:
                cut = float(values.quantile(q))
                candidates.append((feature, cut, "low" if q < .5 else "high"))
        for feature, cut, side in candidates:
            if feature is None:
                pred = valid.baseline.to_numpy(int)
                overrides = np.zeros(len(valid), dtype=bool)
            else:
                pred, overrides = apply_rule(valid, feature, cut, side)
            acc = float((pred[eligible] == valid.target.to_numpy(int)[eligible]).mean())
            override_n = int((overrides & eligible).sum())
            # Paired edge is primary; accuracy and fewer overrides break ties.
            score = (acc - baseline_acc, acc, -override_n)
            if best is None or score > best[0]:
                best = (score, min_strength, feature, cut, side, acc, baseline_acc, int(eligible.sum()), override_n)
    return best


def wilson(h, n, z=1.959963984540054):
    if not n:
        return 0.0
    p = h / n; d = 1 + z*z/n
    return (p + z*z/(2*n) - z*np.sqrt((p*(1-p) + z*z/(4*n))/n)) / d


def exact_mcnemar(a, b):
    n = a + b
    if not n:
        return 1.0
    from math import comb
    tail = sum(comb(n, k) for k in range(0, min(a, b) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    x = prepare(); records = []; windows = []; tested = 0
    start = 0
    while start + TRAIN + TEST <= len(x):
        train = x.iloc[start:start+TRAIN]
        test = x.iloc[start+TRAIN:start+TRAIN+TEST]
        cfg = select(train); tested += len(test)
        if cfg:
            _, strength, feature, cut, side, vacc, vbacc, vn, von = cfg
            eligible = test.external_strength.to_numpy() >= strength
            if feature is None:
                pred = test.baseline.to_numpy(int); override = np.zeros(len(test), dtype=bool)
            else:
                pred, override = apply_rule(test, feature, cut, side)
            for pos in np.flatnonzero(eligible):
                records.append({"date": str(test.iloc[pos].date.date()), "target": int(test.iloc[pos].target),
                                "model": int(pred[pos]), "baseline": int(test.iloc[pos].baseline),
                                "override": bool(override[pos]), "feature": feature})
            use = np.flatnonzero(eligible)
            windows.append({"test_end": str(test.date.max().date()), "cases": int(len(use)),
                            "model_accuracy": float((pred[use] == test.target.to_numpy(int)[use]).mean()) if len(use) else 0,
                            "baseline_accuracy": float((test.baseline.to_numpy(int)[use] == test.target.to_numpy(int)[use]).mean()) if len(use) else 0,
                            "minimum_external_strength": strength, "override_feature": feature,
                            "override_cut": cut, "override_side": side, "overrides": int((override & eligible).sum()),
                            "validation_model_accuracy": vacc, "validation_baseline_accuracy": vbacc,
                            "validation_cases": vn, "validation_overrides": von})
        start += TEST
    n = len(records); mh = sum(r["model"] == r["target"] for r in records); bh = sum(r["baseline"] == r["target"] for r in records)
    mo = sum(r["model"] == r["target"] and r["baseline"] != r["target"] for r in records)
    bo = sum(r["model"] != r["target"] and r["baseline"] == r["target"] for r in records)
    acc = mh/n if n else 0; bacc = bh/n if n else 0; coverage = n/tested if tested else 0; lower = wilson(mh, n)
    passed = n >= 100 and coverage >= .10 and acc >= .90 and lower >= .80 and acc > bacc and exact_mcnemar(mo, bo) < .05
    payload = {"status": "exploratory; historical period inspected, so any passing result requires a newly locked prospective test",
               "hypothesis": "Premarket institutional pulse identifies the rare dates when TWII opening direction decouples from prior US-market complexion.",
               "method": "756/126/126 nested rolling walk-forward. Past-only 252-day standardization. Validation selects external-consensus coverage and one single-feature override rule. Comparison is paired on identical OOS dates.",
               "aligned_days": len(x), "tested_days": tested, "result": {"cases": n, "hits": mh, "accuracy": acc,
               "coverage": coverage, "wilson_95_lower": lower, "baseline_hits": bh, "baseline_accuracy": bacc,
               "edge": acc-bacc, "model_only_correct": mo, "baseline_only_correct": bo,
               "mcnemar_exact_p": exact_mcnemar(mo, bo), "passed": passed}, "windows": windows, "records": records}
    out = ROOT / "reports"
    (out / "external_decoupling_research.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    r = payload["result"]
    lines = ["# External-complexion decoupling research", "", payload["status"], "", payload["hypothesis"], payload["method"], "",
             "| Cases | Accuracy | Coverage | Wilson lower | External baseline | Edge | Model-only / baseline-only | McNemar p | Passed |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
             f"| {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['wilson_95_lower']:.2%} | {r['baseline_accuracy']:.2%} | {r['edge']:.2%} | {r['model_only_correct']} / {r['baseline_only_correct']} | {r['mcnemar_exact_p']:.4f} | {r['passed']} |", "",
             "## Walk-forward windows", "", "| Test end | Cases | Model | Baseline | Override rule | Overrides |", "| --- | ---: | ---: | ---: | --- | ---: |"]
    for w in windows:
        rule = "none" if w["override_feature"] is None else f"{w['override_feature']} {w['override_side']} {w['override_cut']:.4g}"
        lines.append(f"| {w['test_end']} | {w['cases']} | {w['model_accuracy']:.2%} | {w['baseline_accuracy']:.2%} | {rule} | {w['overrides']} |")
    (out / "external_decoupling_research.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"aligned_days": len(x), "tested_days": tested, "result": r}, indent=2))


if __name__ == "__main__":
    main()
