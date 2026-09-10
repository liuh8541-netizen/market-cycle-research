import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
TAIPEI = ZoneInfo("Asia/Taipei")

FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC,%5EIXIC,%5ESOX,TSM,NVDA&region=US&lang=en-US",
    "https://www.investing.com/rss/news_25.rss",
]

RISK_TERMS = {
    "yield": 2,
    "treasury": 2,
    "oil": 1,
    "inflation": 2,
    "tariff": 2,
    "war": 3,
    "sanction": 2,
    "fed": 1,
    "selloff": 2,
    "chip": 1,
    "semiconductor": 1,
    "taiwan": 1,
    "china": 1,
}

TAILWIND_TERMS = {
    "rate cut": 2,
    "rally": 1,
    "ai": 1,
    "earnings beat": 2,
    "semiconductor": 1,
    "chip": 1,
    "stimulus": 2,
}


def fetch(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=12) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_titles(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    items = []
    for item in root.findall(".//item")[:20]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if title:
            items.append({"title": title, "link": link})
    return items


def score_titles(items: list[dict]) -> tuple[int, int, list[dict]]:
    risk = 0
    tailwind = 0
    events = []
    seen = set()
    for item in items:
        title = re.sub(r"\s+", " ", item["title"])
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        title_risk = sum(points for term, points in RISK_TERMS.items() if term in key)
        title_tailwind = sum(points for term, points in TAILWIND_TERMS.items() if term in key)
        if title_risk or title_tailwind:
            risk += min(3, title_risk)
            tailwind += min(2, title_tailwind)
            events.append(
                {
                    "title": title,
                    "risk_impact": "risk" if title_risk > title_tailwind else "tailwind",
                    "risk_score": min(3, title_risk),
                    "tailwind_score": min(2, title_tailwind),
                    "source": item.get("link") or "rss",
                }
            )
    return min(10, risk), min(10, tailwind), events[:8]


def load_existing(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now(TAIPEI).date().isoformat())
    parser.add_argument("--output", default="reports/daily_global_news_risk.json")
    args = parser.parse_args()
    output = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    all_items: list[dict] = []
    sources: list[str] = []
    errors: list[str] = []
    for feed in FEEDS:
        try:
            all_items.extend(parse_titles(fetch(feed)))
            sources.append(feed)
        except Exception as exc:
            errors.append(f"{feed}: {exc}")

    if not all_items:
        existing = load_existing(output)
        existing.update(
            {
                "available": bool(existing),
                "status": "fetch_failed_preserved_existing" if existing else "fetch_failed",
                "fetch_errors": errors,
                "guardrail": "RSS自動抓取失敗時保留既有人工/瀏覽新聞風險檔；不可把缺資料解讀為無風險。",
            }
        )
        output.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"global_news_risk_fetch_failed: {output}")
        return 0

    risk, tailwind, events = score_titles(all_items)
    payload = {
        "available": True,
        "date": args.date,
        "framework": "daily_global_news_risk_rss_v1",
        "risk_score": risk,
        "tailwind_score": tailwind,
        "net_risk_score": risk - tailwind,
        "summary": "RSS自動抓取國際財經標題後形成初步風險分數；仍需Codex每日人工瀏覽重大新聞校正。",
        "events": events,
        "sources": sources,
        "fetch_errors": errors,
        "guardrail": "自動新聞分數只作外部風險初篩；重大事件必須由日盤、夜盤、匯率、利率與正式新聞來源再確認。",
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"global_news_risk_written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
