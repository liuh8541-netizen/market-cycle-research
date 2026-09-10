import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FACTORS = DATA / "factors"
REPORT_MD = ROOT / "reports" / "repeated_index_cycle_study.md"
REPORT_JSON = ROOT / "reports" / "repeated_index_cycle_study.json"


def main() -> None:
    price = pd.read_csv(DATA / "twii_daily.csv", parse_dates=["date"]).dropna(subset=["close"])
    price = price.sort_values("date").reset_index(drop=True)
    for window in [5, 10, 20, 60]:
        price[f"ma{window}"] = price["close"].rolling(window).mean()
    price["ret5"] = price["close"] / price["close"].shift(5) - 1
    price["ret20"] = price["close"] / price["close"].shift(20) - 1
    price["fwd5"] = price["close"].shift(-5) / price["close"] - 1
    price["fwd20"] = price["close"].shift(-20) / price["close"] - 1
    price["drawdown_60_high"] = price["close"] / price["high"].rolling(60).max() - 1

    current = price.iloc[-1]
    current_close = float(current["close"])
    low = current_close * 0.99
    high = current_close * 1.01
    same_level = price[
        (price["date"] < current["date"]) & price["close"].between(low, high)
    ].copy()
    recent = same_level[same_level["date"] >= pd.Timestamp("2025-01-01")].copy()
    recent["phase"] = recent.apply(classify_phase, axis=1)
    recent["group"] = (recent["date"].diff().dt.days.fillna(99) > 7).cumsum()

    representatives = []
    for _, group in recent.groupby("group"):
        closest = group.iloc[(group["close"] - current_close).abs().argmin()]
        representatives.append(closest)

    snapshots = [snapshot(row) for row in representatives]
    current_snapshot = snapshot(current)
    current_snapshot["phase"] = "現在：修復回同點"
    snapshots.append(current_snapshot)

    output = {
        "method": {
            "current_date": str(current["date"].date()),
            "current_close": current_close,
            "same_level_range": [low, high],
            "same_level_rule": "收盤落在目前正式日線收盤的正負1%內。",
            "note": "本研究比較同指數點位的內部結構；專案目前沒有完整逐檔OHLC，個股變化以族群廣度、法人、融資、期權與市值結構代理。",
        },
        "same_level_count_all": int(len(same_level)),
        "same_level_count_recent": int(len(recent)),
        "recent_same_level_dates": [
            {
                "date": str(row.date.date()),
                "close": round(float(row.close), 2),
                "phase": row.phase,
                "ret5": safe_round(row.ret5),
                "ret20": safe_round(row.ret20),
                "fwd5": safe_round(row.fwd5),
                "fwd20": safe_round(row.fwd20),
            }
            for row in recent.itertuples()
        ],
        "representative_snapshots": snapshots,
        "comparison": build_comparison(snapshots),
        "improvements": build_improvements(),
    }
    REPORT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(render_report(output), encoding="utf-8")
    print(f"Report: {REPORT_MD}")
    print(f"JSON: {REPORT_JSON}")


def classify_phase(row: pd.Series) -> str:
    if row["ret20"] > 0.03 and row["drawdown_60_high"] > -0.035:
        return "上攻到同點"
    if row["ret20"] < -0.03:
        return "跌回同點"
    if row["drawdown_60_high"] < -0.05 and row["ret5"] > 0:
        return "修復回同點"
    return "高檔換手同點"


def snapshot(row: pd.Series) -> dict:
    date = row["date"]
    out = {
        "date": str(date.date()),
        "close": round(float(row["close"]), 2),
        "phase": row.get("phase", "目前基準"),
        "ret5": safe_round(row.get("ret5")),
        "ret20": safe_round(row.get("ret20")),
        "fwd5": safe_round(row.get("fwd5")),
        "fwd20": safe_round(row.get("fwd20")),
        "drawdown_60_high": safe_round(row.get("drawdown_60_high")),
        "ma5_gap": safe_round(row["close"] / row["ma5"] - 1 if pd.notna(row.get("ma5")) else None),
        "ma20_gap": safe_round(row["close"] / row["ma20"] - 1 if pd.notna(row.get("ma20")) else None),
        "ma60_gap": safe_round(row["close"] / row["ma60"] - 1 if pd.notna(row.get("ma60")) else None),
    }
    out.update(prefix(load_nearest("cross_sectional_breadth.csv", date, [
        "price_advancing_fraction",
        "price_declining_fraction",
        "price_equal_weight_return",
        "price_median_return",
        "price_return_dispersion",
        "price_money_top10_share",
        "inst_combined_positive_fraction",
        "inst_combined_breadth",
    ]), "breadth_"))
    out.update(prefix(load_nearest("sector_early_pulse.csv", date, [
        "twse_electronic_return",
        "twse_electronic_last5_return",
        "twse_finance_return",
        "twse_finance_last5_return",
        "sector_breadth",
        "sector_last5_breadth",
        "electronic_finance_rotation",
        "tpex_twse_electronic_rotation",
    ]), "sector_"))
    out.update(prefix(load_nearest("external_markets.csv", date, [
        "nasdaq_return_1d",
        "sox_return_1d",
        "vix_return_1d",
        "tsm_adr_return_1d",
        "micron_return_1d",
        "ewt_return_1d",
        "usd_twd_return_1d",
    ], DATA), "external_"))
    out.update(institutional_snapshot(date))
    out.update(margin_snapshot(date))
    out.update(futures_institutional_snapshot(date))
    out.update(option_snapshot(date))
    return out


def load_nearest(filename: str, date: pd.Timestamp, columns: list[str], base: Path = FACTORS) -> dict:
    path = base / filename
    if not path.exists():
        return {}
    frame = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    frame = frame[frame["date"] <= date]
    if frame.empty:
        return {}
    row = frame.iloc[-1]
    out = {"source_date": str(row["date"].date())}
    for column in columns:
        if column in row and pd.notna(row[column]):
            out[column] = safe_round(row[column])
    return out


def institutional_snapshot(date: pd.Timestamp) -> dict:
    path = FACTORS / "institutional_total.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame[frame["date"] == date].copy()
    if frame.empty:
        return {}
    frame["net_100m"] = (frame["buy"] - frame["sell"]) / 1e8
    out = {}
    for name, key in [
        ("Foreign_Investor", "inst_foreign_net_100m"),
        ("Investment_Trust", "inst_trust_net_100m"),
        ("Dealer", "inst_dealer_net_100m"),
        ("total", "inst_total_net_100m"),
    ]:
        row = frame[frame["name"] == name]
        if not row.empty:
            out[key] = safe_round(row["net_100m"].sum())
    return out


def margin_snapshot(date: pd.Timestamp) -> dict:
    path = FACTORS / "margin_total.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame[frame["date"] == date].copy()
    out = {}
    for name, key in [
        ("MarginPurchase", "margin_balance"),
        ("ShortSale", "short_balance"),
        ("MarginPurchaseMoney", "margin_money_balance"),
    ]:
        row = frame[frame["name"] == name]
        if not row.empty:
            out[key] = safe_round(row["TodayBalance"].iloc[-1])
    return out


def futures_institutional_snapshot(date: pd.Timestamp) -> dict:
    path = FACTORS / "futures_institutional.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame[(frame["date"] == date) & (frame["futures_id"] == "TX")].copy()
    out = {}
    for name, key in [("外資", "tx_foreign_net_oi"), ("自營商", "tx_dealer_net_oi"), ("投信", "tx_trust_net_oi")]:
        row = frame[frame["institutional_investors"] == name]
        if not row.empty:
            net = row["long_open_interest_balance_volume"].sum() - row["short_open_interest_balance_volume"].sum()
            out[key] = safe_round(net)
    return out


def option_snapshot(date: pd.Timestamp) -> dict:
    path = FACTORS / "option_daily.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame[frame["date"] == date]
    if frame.empty:
        return {}
    row = frame.iloc[-1]
    return {
        "put_call_oi": safe_round(row["put_open_interest"] / row["call_open_interest"] if row["call_open_interest"] else None),
        "put_call_volume": safe_round(row["put_volume"] / row["call_volume"] if row["call_volume"] else None),
    }


def build_comparison(snapshots: list[dict]) -> list[str]:
    current = snapshots[-1]
    prior = snapshots[:-1]
    notes = []
    if prior:
        avg_fwd20 = pd.Series([item.get("fwd20") for item in prior if item.get("fwd20") is not None]).mean()
        notes.append(f"近期同點歷史的20日後平均約 {pct(avg_fwd20)}；但樣本混有上攻、跌回、修復，不能單用點數外推。")
    breadth = current.get("breadth_price_advancing_fraction")
    if breadth is not None:
        notes.append(f"目前同點更需要看個股廣度：上漲家數占比 {pct(breadth)}，若指數漲但廣度不足，代表權值拉抬、個股未同步。")
    rotation = current.get("sector_electronic_finance_rotation")
    if rotation is not None:
        notes.append(f"電子對金融輪動值 {pct(rotation)}；若電子強、金融弱，屬科技權值帶動，不等於全市場普漲。")
    foreign = current.get("inst_foreign_net_100m")
    if foreign is not None:
        notes.append(f"外資現貨淨買賣約 {foreign:.1f} 億元；同點若外資與廣度不同，後續慣性會不同。")
    tx = current.get("tx_foreign_net_oi")
    if tx is not None:
        notes.append(f"外資台指期淨部位 {tx:.0f} 口；同樣指數若期貨避險偏空，代表上方仍有保護性賣壓。")
    notes.append("結論：同樣指數不是同樣市場。前一波是價格重複，現在要用內部結構判斷是健康輪動、權值撐盤，還是修復未完成。")
    return notes


def build_improvements() -> list[str]:
    return [
        "新增同指數輪迴比較：每日找出目前指數正負1%內的歷史同點，分成上攻、跌回、修復三類。",
        "大盤報告不要只寫點位相似，要同步列出廣度、族群輪動、法人現貨、期貨避險與選擇權壓力。",
        "若指數回同點但廣度下降、外資期貨偏空、權值集中度升高，標記為權值撐盤/假修復風險。",
        "若指數回同點且廣度改善、均線收復、外資現貨與期貨不背離，標記為健康修復。",
        "目前缺完整逐檔OHLC，若要真正回答個股強弱，下一步應接入逐檔日線，建立強勢股清單與弱勢股擴散率。",
    ]


def render_report(data: dict) -> str:
    method = data["method"]
    lines = [
        "# 同指數輪迴差異研究",
        "",
        f"- 基準日: {method['current_date']}",
        f"- 基準收盤: {num(method['current_close'])}",
        f"- 同指數區間: {num(method['same_level_range'][0])}～{num(method['same_level_range'][1])}",
        f"- 規則: {method['same_level_rule']}",
        f"- 全歷史同點筆數: {data['same_level_count_all']}",
        f"- 2025以來同點筆數: {data['same_level_count_recent']}",
        f"- 資料限制: {method['note']}",
        "",
        "## 重點結論",
        "",
    ]
    for item in data["comparison"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## 近期同點日期",
        "",
        "| 日期 | 收盤 | 階段 | 前5日 | 前20日 | 後5日 | 後20日 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in data["recent_same_level_dates"]:
        lines.append(
            f"| {row['date']} | {num(row['close'])} | {row['phase']} | {pct(row['ret5'])} | "
            f"{pct(row['ret20'])} | {pct(row['fwd5'])} | {pct(row['fwd20'])} |"
        )
    lines.extend([
        "",
        "## 代表情境比較",
        "",
        "| 日期 | 階段 | 收盤 | 60日高點回撤 | 5日乖離 | 20日乖離 | 廣度 | 電子/金融輪動 | 外資現貨億 | 外資期貨口 | P/C OI |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in data["representative_snapshots"]:
        lines.append(
            f"| {row['date']} | {row['phase']} | {num(row['close'])} | {pct(row.get('drawdown_60_high'))} | "
            f"{pct(row.get('ma5_gap'))} | {pct(row.get('ma20_gap'))} | "
            f"{pct(row.get('breadth_price_advancing_fraction'))} | {pct(row.get('sector_electronic_finance_rotation'))} | "
            f"{num(row.get('inst_foreign_net_100m'))} | {num(row.get('tx_foreign_net_oi'))} | {row.get('put_call_oi', 'NA')} |"
        )
    lines.extend([
        "",
        "## 可改進項目",
        "",
    ])
    for item in data["improvements"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def prefix(values: dict, name: str) -> dict:
    return {f"{name}{key}": value for key, value in values.items()}


def safe_round(value, digits: int = 4):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def pct(value) -> str:
    return "NA" if value is None or pd.isna(value) else f"{value:.2%}"


def num(value) -> str:
    return "NA" if value is None or pd.isna(value) else f"{value:,.0f}"


if __name__ == "__main__":
    main()
