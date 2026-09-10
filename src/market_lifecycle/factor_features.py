from pathlib import Path

import pandas as pd


FACTOR_FILES = {
    "institutional_total": "institutional_total.csv",
    "margin_total": "margin_total.csv",
    "futures_daily": "futures_daily.csv",
    "futures_institutional": "futures_institutional.csv",
    "option_daily": "option_daily.csv",
    "option_institutional": "option_institutional.csv",
    "option_vix": "option_vix.csv",
}


def add_factor_features(price_features: pd.DataFrame, factor_dir: str | None) -> pd.DataFrame:
    if not factor_dir:
        return price_features

    base = price_features.copy()
    base["date"] = pd.to_datetime(base["date"])
    root = Path(factor_dir)
    if not root.exists():
        return base

    builders = {
        "institutional_total": build_institutional_features,
        "margin_total": build_margin_features,
        "futures_daily": build_futures_daily_features,
        "futures_institutional": build_futures_institutional_features,
        "option_daily": build_option_daily_features,
        "option_institutional": build_option_institutional_features,
        "option_vix": build_option_vix_features,
    }

    for name, filename in FACTOR_FILES.items():
        path = root / filename
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        if raw.empty or "date" not in raw.columns:
            continue
        features = builders[name](raw)
        if features.empty:
            continue
        base = base.merge(features, on="date", how="left")

    return add_factor_scores(base)


def build_institutional_features(raw: pd.DataFrame) -> pd.DataFrame:
    data = normalize_date(raw)
    if {"buy", "sell"}.issubset(data.columns):
        data["inst_net_proxy"] = pd.to_numeric(data["buy"], errors="coerce") - pd.to_numeric(data["sell"], errors="coerce")
        if "name" in data and data["name"].astype(str).str.lower().eq("total").any():
            total = data[data["name"].astype(str).str.lower().eq("total")]
            grouped = total.groupby("date")["inst_net_proxy"].sum().reset_index()
            names = data["name"].astype(str).str.strip()
            categories = {
                "foreign": names.eq("Foreign_Investor"),
                "trust": names.eq("Investment_Trust"),
                "dealer": names.isin(["Dealer", "Dealer_Hedging", "Dealer_self", "Foreign_Dealer_Self"]),
            }
            for label, mask in categories.items():
                series = data.loc[mask].groupby("date")["inst_net_proxy"].sum()
                grouped[f"inst_{label}_net_proxy"] = grouped["date"].map(series)
                grouped[f"inst_{label}_net_5d"] = grouped[f"inst_{label}_net_proxy"].rolling(5).sum()
                grouped[f"inst_{label}_net_20d"] = grouped[f"inst_{label}_net_proxy"].rolling(20).sum()
        else:
            grouped = data.groupby("date")["inst_net_proxy"].sum().reset_index()
    else:
        numeric = numeric_columns(data, exclude={"date"})
        if not numeric:
            return pd.DataFrame()
        grouped = data.groupby("date")[numeric].sum(numeric_only=True).reset_index()
        grouped["inst_net_proxy"] = grouped[numeric].sum(axis=1)
    grouped["inst_net_5d"] = grouped["inst_net_proxy"].rolling(5).sum()
    grouped["inst_net_20d"] = grouped["inst_net_proxy"].rolling(20).sum()
    # Preserve investor-level features when the source provides them. Earlier
    # code calculated these columns and then silently dropped them here,
    # leaving downstream foreign/trust/dealer model inputs entirely missing.
    ordered = ["date", "inst_net_proxy", "inst_net_5d", "inst_net_20d"]
    split = [
        col for col in grouped.columns
        if col.startswith(("inst_foreign_", "inst_trust_", "inst_dealer_"))
    ]
    return grouped[ordered + split]


def build_margin_features(raw: pd.DataFrame) -> pd.DataFrame:
    data = normalize_date(raw)
    if {"name", "TodayBalance"}.issubset(data.columns):
        data["TodayBalance"] = pd.to_numeric(data["TodayBalance"], errors="coerce")
        pivot = data.pivot_table(index="date", columns="name", values="TodayBalance", aggfunc="last").reset_index()
        margin_col = find_first(pivot, ["MarginPurchase"])
        short_col = find_first(pivot, ["ShortSale"])
        output = pivot[["date"]].copy()
        if margin_col:
            output["margin_balance_proxy"] = pivot[margin_col]
            output["margin_change_20d"] = pivot[margin_col].pct_change(20)
        if short_col:
            output["short_balance_proxy"] = pivot[short_col]
            output["short_change_20d"] = pivot[short_col].pct_change(20)
        if margin_col and short_col:
            output["margin_short_ratio"] = pivot[margin_col] / pivot[short_col].replace(0, pd.NA)
        return output
    numeric = numeric_columns(data, exclude={"date"})
    if not numeric:
        return pd.DataFrame()
    grouped = data.groupby("date")[numeric].sum(numeric_only=True).reset_index()
    margin_col = find_first(grouped, ["margin", "Margin", "融資"])
    short_col = find_first(grouped, ["short", "Short", "融券"])
    output = grouped[["date"]].copy()
    if margin_col:
        output["margin_balance_proxy"] = grouped[margin_col]
        output["margin_change_20d"] = grouped[margin_col].pct_change(20)
    if short_col:
        output["short_balance_proxy"] = grouped[short_col]
        output["short_change_20d"] = grouped[short_col].pct_change(20)
    if margin_col and short_col:
        output["margin_short_ratio"] = grouped[margin_col] / grouped[short_col].replace(0, pd.NA)
    return output


def build_futures_daily_features(raw: pd.DataFrame) -> pd.DataFrame:
    data = normalize_date(raw)
    data = filter_contract(data, ["TX", "TXF", "臺股期貨", "台股期貨"])
    numeric = numeric_columns(data, exclude={"date"})
    if not numeric:
        return pd.DataFrame()
    grouped = data.groupby("date")[numeric].sum(numeric_only=True).reset_index()
    volume_col = find_first(grouped, ["volume", "Volume", "成交"])
    oi_col = find_first(grouped, ["open_interest", "OpenInterest", "未沖銷", "未平倉"])
    output = grouped[["date"]].copy()
    if volume_col:
        output["futures_volume_proxy"] = grouped[volume_col]
        output["futures_volume_20d_z"] = rolling_zscore(grouped[volume_col], 20)
    if oi_col:
        output["futures_oi_proxy"] = grouped[oi_col]
        output["futures_oi_change_20d"] = grouped[oi_col].pct_change(20)
    return output


def build_futures_institutional_features(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "long_open_interest_balance_volume", "short_open_interest_balance_volume"}
    if required.issubset(raw.columns):
        data = normalize_date(raw)
        long_oi = pd.to_numeric(data["long_open_interest_balance_volume"], errors="coerce")
        short_oi = pd.to_numeric(data["short_open_interest_balance_volume"], errors="coerce")
        data["futures_inst_net_proxy"] = long_oi - short_oi
        grouped = data.groupby("date")["futures_inst_net_proxy"].sum().reset_index()
        if "institutional_investors" in data:
            names = data["institutional_investors"].astype(str).str.strip()
            categories = {"foreign": "外資", "trust": "投信", "dealer": "自營商"}
            for label, source_name in categories.items():
                series = data.loc[names.eq(source_name)].groupby("date")["futures_inst_net_proxy"].sum()
                grouped[f"futures_{label}_net_proxy"] = grouped["date"].map(series)
                grouped[f"futures_{label}_net_5d"] = grouped[f"futures_{label}_net_proxy"].rolling(5).sum()
                grouped[f"futures_{label}_net_20d"] = grouped[f"futures_{label}_net_proxy"].rolling(20).sum()
        grouped["futures_inst_net_5d"] = grouped["futures_inst_net_proxy"].rolling(5).sum()
        grouped["futures_inst_net_20d"] = grouped["futures_inst_net_proxy"].rolling(20).sum()
        return grouped
    return build_derivative_institutional_features(raw, "futures_inst")


def build_option_daily_features(raw: pd.DataFrame) -> pd.DataFrame:
    data = normalize_date(raw)
    if {"put_volume", "call_volume"}.issubset(data.columns):
        grouped = data.groupby("date")[[col for col in ["put_volume", "call_volume", "put_open_interest", "call_open_interest"] if col in data]].sum().reset_index()
        output = grouped[["date"]].copy()
        output["option_put_call_proxy"] = grouped["put_volume"] / grouped["call_volume"].replace(0, pd.NA)
        output["option_put_call_20d_z"] = rolling_zscore(output["option_put_call_proxy"], 20)
        if {"put_open_interest", "call_open_interest"}.issubset(grouped.columns):
            output["option_oi_put_call"] = grouped["put_open_interest"] / grouped["call_open_interest"].replace(0, pd.NA)
        return output
    numeric = numeric_columns(data, exclude={"date"})
    if not numeric:
        return pd.DataFrame()
    grouped = data.groupby("date")[numeric].sum(numeric_only=True).reset_index()
    put_col = find_first(grouped, ["put", "Put", "賣權"])
    call_col = find_first(grouped, ["call", "Call", "買權"])
    output = grouped[["date"]].copy()
    if put_col and call_col:
        output["option_put_call_proxy"] = grouped[put_col] / grouped[call_col].replace(0, pd.NA)
        output["option_put_call_20d_z"] = rolling_zscore(output["option_put_call_proxy"], 20)
    return output


def build_option_institutional_features(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "call_put", "long_open_interest_balance_volume", "short_open_interest_balance_volume"}
    if required.issubset(raw.columns):
        data = normalize_date(raw)
        net = pd.to_numeric(data["long_open_interest_balance_volume"], errors="coerce") - pd.to_numeric(data["short_open_interest_balance_volume"], errors="coerce")
        is_put = data["call_put"].astype(str).str.lower().isin(["put", "p", "賣權"])
        data["option_inst_net_proxy"] = net.where(~is_put, -net)
        grouped = data.groupby("date")["option_inst_net_proxy"].sum().reset_index()
        if "institutional_investors" in data:
            names = data["institutional_investors"].astype(str).str.strip()
            categories = {"foreign": "外資", "trust": "投信", "dealer": "自營商"}
            for label, source_name in categories.items():
                series = data.loc[names.eq(source_name)].groupby("date")["option_inst_net_proxy"].sum()
                grouped[f"option_{label}_net_proxy"] = grouped["date"].map(series)
                grouped[f"option_{label}_net_5d"] = grouped[f"option_{label}_net_proxy"].rolling(5).sum()
                grouped[f"option_{label}_net_20d"] = grouped[f"option_{label}_net_proxy"].rolling(20).sum()
        grouped["option_inst_net_5d"] = grouped["option_inst_net_proxy"].rolling(5).sum()
        grouped["option_inst_net_20d"] = grouped["option_inst_net_proxy"].rolling(20).sum()
        return grouped
    return build_derivative_institutional_features(raw, "option_inst")


def build_option_vix_features(raw: pd.DataFrame) -> pd.DataFrame:
    data = normalize_date(raw)
    if "time" in data:
        data = data.sort_values(["date", "time"])
    numeric = numeric_columns(data, exclude={"date"})
    if not numeric:
        return pd.DataFrame()
    value_col = numeric[0]
    output = data.groupby("date")[value_col].last().reset_index()
    output = output.rename(columns={value_col: "option_vix"})
    output["option_vix_20d_change"] = output["option_vix"].pct_change(20)
    output["option_vix_20d_z"] = rolling_zscore(output["option_vix"], 20)
    return output


def build_derivative_institutional_features(raw: pd.DataFrame, prefix: str) -> pd.DataFrame:
    data = normalize_date(raw)
    numeric = numeric_columns(data, exclude={"date"})
    if not numeric:
        return pd.DataFrame()
    grouped = data.groupby("date")[numeric].sum(numeric_only=True).reset_index()
    grouped[f"{prefix}_net_proxy"] = grouped[numeric].sum(axis=1)
    grouped[f"{prefix}_net_5d"] = grouped[f"{prefix}_net_proxy"].rolling(5).sum()
    grouped[f"{prefix}_net_20d"] = grouped[f"{prefix}_net_proxy"].rolling(20).sum()
    return grouped[["date", f"{prefix}_net_proxy", f"{prefix}_net_5d", f"{prefix}_net_20d"]]


def add_factor_scores(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    output["heuristic_chip_score"] = 0
    output["heuristic_derivative_score"] = 0

    if "inst_net_20d" in output:
        output["heuristic_chip_score"] += signal_from_series(output["inst_net_20d"])
    if "margin_change_20d" in output:
        output["heuristic_chip_score"] += signal_from_series(output["margin_change_20d"], high_is_bearish=True)
    if "futures_inst_net_20d" in output:
        output["heuristic_derivative_score"] += signal_from_series(output["futures_inst_net_20d"])
    if "option_put_call_20d_z" in output:
        output["heuristic_derivative_score"] += signal_from_series(output["option_put_call_20d_z"], high_is_bearish=True)
    if "option_inst_net_20d" in output:
        output["heuristic_derivative_score"] += signal_from_series(output["option_inst_net_20d"])
    if "option_vix_20d_z" in output:
        output["heuristic_derivative_score"] += signal_from_series(output["option_vix_20d_z"], high_is_bearish=True)

    # Formal purged polarity audit found no FinMind multi-day directional
    # factor meeting the accuracy/sample/coverage/non-overlap gate. Preserve
    # heuristic diagnostics, but do not let unvalidated fixed signs alter the
    # production lifecycle score.
    output["chip_score"] = 0
    output["derivative_score"] = 0
    output["non_price_score"] = 0
    return output


def normalize_date(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values("date").reset_index(drop=True)


def numeric_columns(data: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols = []
    for col in data.columns:
        if col in exclude:
            continue
        converted = pd.to_numeric(data[col], errors="coerce")
        if converted.notna().any():
            data[col] = converted
            cols.append(col)
    return cols


def find_first(data: pd.DataFrame, patterns: list[str]) -> str | None:
    for pattern in patterns:
        matches = [col for col in data.columns if pattern.lower() in col.lower()]
        if matches:
            return matches[0]
    return None


def filter_contract(data: pd.DataFrame, patterns: list[str]) -> pd.DataFrame:
    text_columns = [col for col in data.columns if data[col].dtype == "object"]
    if not text_columns:
        return data
    mask = pd.Series(False, index=data.index)
    for col in text_columns:
        values = data[col].astype(str)
        for pattern in patterns:
            mask = mask | values.str.contains(pattern, case=False, na=False)
    filtered = data[mask]
    return filtered if not filtered.empty else data


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, pd.NA)


def signal_from_series(series: pd.Series, high_is_bearish: bool = False) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    z = rolling_zscore(series, 60)
    signal = pd.Series(0, index=series.index)
    signal[z > 0.75] = -1 if high_is_bearish else 1
    signal[z < -0.75] = 1 if high_is_bearish else -1
    return signal.fillna(0).astype(int)
