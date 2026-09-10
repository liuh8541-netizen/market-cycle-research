import pandas as pd


HORIZONS = [1, 5, 20, 60]


def build_causal_database(scored: pd.DataFrame) -> pd.DataFrame:
    data = scored.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])

    output = data[["date", "open", "high", "low", "close", "volume"]].copy()
    output["lifecycle_stage"] = data["lifecycle_stage"]
    output["risk_regime"] = data["risk_regime"]
    output["stage_score"] = data["stage_score"]
    output["trend_score"] = data["trend_score"]
    output["momentum_score"] = data["momentum_score"]
    output["volatility_score"] = data["volatility_score"]
    output["drawdown_score"] = data["drawdown_score"]
    output["drawdown"] = data["drawdown"]
    output["volatility_20"] = data["volatility_20"]
    output["rsi_14"] = data["rsi_14"]

    for horizon in HORIZONS:
        output[f"cause_return_{horizon}d"] = data["close"].pct_change(horizon)
        output[f"effect_return_{horizon}d"] = data["close"].shift(-horizon) / data["close"] - 1
        output[f"cause_direction_{horizon}d"] = output[f"cause_return_{horizon}d"].apply(_direction)
        output[f"effect_direction_{horizon}d"] = output[f"effect_return_{horizon}d"].apply(_direction)

    output["cause_scenario"] = output.apply(_cause_scenario, axis=1)
    output["market_mode"] = output.apply(_market_mode, axis=1)
    output["event_intensity"] = output.apply(_event_intensity, axis=1)
    output["causal_signature"] = output.apply(_causal_signature, axis=1)
    return output


def summarize_causal_edges(
    causal: pd.DataFrame,
    min_samples: int = 80,
    min_hit_rate: float = 0.60,
) -> dict:
    reports = []
    for horizon in HORIZONS:
        direction_col = f"effect_direction_{horizon}d"
        valid = causal[causal[direction_col] != "unknown"].copy()
        rows = []
        for signature, group in valid.groupby("causal_signature"):
            if len(group) < min_samples:
                continue
            counts = group[direction_col].value_counts()
            total = int(counts.sum())
            best_direction = str(counts.idxmax())
            best_hit_rate = float(counts.max() / total)
            if best_hit_rate < min_hit_rate:
                continue
            rows.append(
                {
                    "causal_signature": signature,
                    "horizon_days": horizon,
                    "sample_count": total,
                    "best_direction": best_direction,
                    "hit_rate": best_hit_rate,
                    "probability_up": float(counts.get("up", 0) / total),
                    "probability_down": float(counts.get("down", 0) / total),
                    "probability_sideways": float(counts.get("sideways", 0) / total),
                    "avg_effect_return": float(group[f"effect_return_{horizon}d"].mean()),
                }
            )
        rows = sorted(rows, key=lambda item: (item["hit_rate"], item["sample_count"]), reverse=True)
        reports.append({"horizon_days": horizon, "edges": rows})

    return {
        "min_samples": int(min_samples),
        "min_hit_rate": float(min_hit_rate),
        "horizons": reports,
    }


def _cause_scenario(row) -> str:
    r1 = row.get("cause_return_1d")
    r5 = row.get("cause_return_5d")
    r20 = row.get("cause_return_20d")
    drawdown = row.get("drawdown")

    if pd.notna(r1) and r1 <= -0.03:
        return "single_day_crash"
    if pd.notna(r1) and r1 >= 0.03:
        return "single_day_surge"
    if pd.notna(r5) and r5 <= -0.05:
        return "five_day_crash"
    if pd.notna(r5) and r5 >= 0.05:
        return "five_day_surge"
    if pd.notna(r20) and r20 <= -0.10:
        return "twenty_day_crash"
    if pd.notna(r20) and r20 >= 0.10:
        return "twenty_day_surge"
    if pd.notna(drawdown) and drawdown <= -0.20:
        return "deep_drawdown"
    if pd.notna(drawdown) and drawdown >= -0.03:
        return "near_high"
    return "normal"


def _market_mode(row) -> str:
    if row["risk_regime"] == "risk_on" and row["stage_score"] >= 5:
        return "strong_risk_on"
    if row["risk_regime"] == "risk_on":
        return "risk_on"
    if row["risk_regime"] == "risk_off" and row["stage_score"] <= -5:
        return "strong_risk_off"
    if row["risk_regime"] == "risk_off":
        return "risk_off"
    return "neutral"


def _event_intensity(row) -> str:
    vol = row.get("volatility_20")
    r1 = abs(row.get("cause_return_1d", 0)) if pd.notna(row.get("cause_return_1d")) else 0
    if r1 >= 0.03 or (pd.notna(vol) and vol >= 0.28):
        return "shock"
    if r1 >= 0.015 or (pd.notna(vol) and vol >= 0.20):
        return "elevated"
    return "normal"


def _causal_signature(row) -> str:
    return "|".join(
        [
            str(row["cause_scenario"]),
            str(row["market_mode"]),
            str(row["event_intensity"]),
            str(row["lifecycle_stage"]),
        ]
    )


def _direction(value: float, neutral_threshold: float = 0.005) -> str:
    if pd.isna(value):
        return "unknown"
    if value > neutral_threshold:
        return "up"
    if value < -neutral_threshold:
        return "down"
    return "sideways"

