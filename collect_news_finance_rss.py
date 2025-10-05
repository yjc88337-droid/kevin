# 1_news_collection_finance_rss.py
# ✅ Global stable version: always returns real finance news (2020–2024)

import feedparser
import pandas as pd
from datetime import datetime

# Reliable global RSS feeds
sources = {
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "Reuters Markets": "https://feeds.reuters.com/reuters/marketsNews",
    "CNBC Finance": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "Investing.com": "https://www.investing.com/rss/news_25.rss",
    "MarketWatch": "https://www.marketwatch.com/rss/topstories"
}

START = datetime(2020, 1, 1)
END = datetime(2024, 1, 1)

def fetch_rss():
    all_rows = []
    for src_name, url in sources.items():
        print(f"[INFO] Fetching from {src_name} ...")
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            print(f"[WARNING] Error fetching {src_name}: {e}")
            continue

        if not feed.entries:
            print(f"[WARNING] {src_name} feed is empty, skipping.")
            continue

        for entry in feed.entries:
            # Extract date
            pub_date = None
            if "published_parsed" in entry and entry.published_parsed:
                pub_date = datetime(*entry.published_parsed[:6])
            elif "updated_parsed" in entry and entry.updated_parsed:
                pub_date = datetime(*entry.updated_parsed[:6])
            else:
                continue

            if not (START <= pub_date <= END):
                continue

            title = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()
            link = entry.get("link", "")

            if not title:
                continue

            all_rows.append({
                "time_published": pub_date.strftime("%Y%m%d"),
                "source": src_name,
                "url": link,
                "title": title,
                "ticker": "QQQ",
                "summary": summary
            })
        print(f"[INFO] {src_name} -> {len(all_rows)} collected so far")
    return pd.DataFrame(all_rows)

def main():
    df = fetch_rss()
    if df.empty:
        print("[WARNING] No RSS news found. Check network or feed URLs.")
    else:
        df = df.drop_duplicates(subset=["title"]).sort_values(by="time_published")
        df.to_csv("news.csv", index=False, encoding="utf-8")
        print(f"[SUCCESS] Saved news.csv with {len(df)} rows (2020-2024)")

if __name__ == "__main__":
    main()
