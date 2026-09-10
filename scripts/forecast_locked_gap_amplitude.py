"""Prospective logger for validated daily-night material-gap amplitude risk."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_microstructure_gap_amplitude as study  # noqa: E402
import research_night_microstructure_gap as source  # noqa: E402

LOCK = ROOT / "config/locked_microstructure_gap_amplitude_v1.json"
LOG = ROOT / "reports/locked_gap_amplitude_forecast_log.json"
FEATURES = source.CONTROL_FEATURES


def verify_lock():
    cfg = json.loads(LOCK.read_text(encoding="utf-8"))
    import hashlib
    for name, expected in cfg["implementation"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest().upper()
        if actual != expected.upper():
            raise RuntimeError(f"Locked amplitude dependency changed: {name}")
    return cfg


def latest_feature_rows():
    night = pd.read_csv(ROOT / "data/processed/taiwan_futures_night.csv")
    night["date"] = pd.to_datetime(night["signal_date"])
    micro = pd.read_csv(ROOT / "data/processed/factors/night_microstructure.csv")
    micro["date"] = pd.to_datetime(micro["signal_date"])
    frame = night.merge(micro.drop(columns=["signal_date", "contract_date"]), on="date", how="inner")
    frame["log_night_volume"] = np.log1p(pd.to_numeric(frame["tx_night_volume"], errors="coerce"))
    frame["log_tick_count"] = np.log1p(pd.to_numeric(frame["tick_count"], errors="coerce"))
    return frame.sort_values("date").replace([np.inf, -np.inf], np.nan)


def main():
    lock = verify_lock()
    history = source.prepare().dropna(subset=["gap_return"]).reset_index(drop=True)
    if len(history) < study.HISTORY:
        raise RuntimeError("Insufficient amplitude-model history")
    train = history.tail(study.HISTORY).copy()
    selected = study.choose(train, FEATURES)
    if selected is None:
        raise RuntimeError("Amplitude-model nested selection failed")
    _, lam, quantile, validation_accuracy, validation_cases = selected
    event_cut = float(train["gap_return"].abs().quantile(study.EVENT_QUANTILE))
    target = np.where(train["gap_return"].abs() >= event_cut, 1, -1)
    model = study.fit(train, FEATURES, target, lam)
    confidence_cut = float(np.quantile(study.predict(model, train, FEATURES)[1], quantile))
    spread_cut = float(train["tx_night_spread_per"].abs().quantile(study.EVENT_QUANTILE))

    cash = pd.read_csv(ROOT / "data/processed/twii_daily.csv", usecols=["date", "open", "close"])
    cash["date"] = pd.to_datetime(cash["date"])
    cash = cash.sort_values("date").drop_duplicates("date", keep="last")
    cash["gap_return"] = pd.to_numeric(cash["open"], errors="coerce") / pd.to_numeric(
        cash["close"], errors="coerce"
    ).shift(1) - 1
    outcomes = cash.set_index(cash["date"].dt.date.astype(str))["gap_return"]
    latest_cash = cash["date"].max()

    existing = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else []
    for row in existing:
        day = row["signal_date"]
        if day in outcomes.index and pd.notna(outcomes.loc[day]):
            gap = float(outcomes.loc[day])
            actual = 1 if abs(gap) >= row["event_threshold"] else -1
            row["actual_gap_return"] = gap
            row["outcome"] = "material" if actual == 1 else "normal"
            row["hit"] = bool(row["prediction_value"] == actual) if row["actionable"] else None

    known = {row["signal_date"] for row in existing}
    future = latest_feature_rows()
    future = future[future["date"] > latest_cash]
    created = []
    if not future.empty:
        pred, confidence = study.predict(model, future, FEATURES)
        for pos, (_, row) in enumerate(future.iterrows()):
            day = str(row["date"].date())
            if day in known:
                continue
            value = int(pred[pos])
            record = {
                "experiment_id": lock["experiment_id"], "signal_date": day,
                "prediction": "material_gap" if value == 1 else "normal_gap",
                "prediction_value": value, "confidence_score": float(confidence[pos]),
                "confidence_threshold": confidence_cut,
                "actionable": bool(confidence[pos] >= confidence_cut),
                "event_threshold": event_cut, "night_spread_threshold": spread_cut,
                "night_spread_abs": abs(float(row["tx_night_spread_per"])),
                "lambda": lam, "quantile": quantile,
                "validation_accuracy_at_run": validation_accuracy,
                "validation_cases_at_run": validation_cases, "outcome": None,
            }
            existing.append(record)
            created.append(record)
    LOG.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Locked gap amplitude: {len(created)} new; {len(existing)} total prospective record(s).")


if __name__ == "__main__":
    main()
