# 1_news_collection_simple.py
# Simple news collection using RSS feeds for current news

import feedparser
import pandas as pd
from datetime import datetime, timedelta

# Use feeds that work for current news
sources = {
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "MarketWatch Top": "http://feeds.marketwatch.com/marketwatch/topstories/",
    "MarketWatch Real-time": "http://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
}

def fetch_current_news(ticker="QQQ", max_items=100):
    """
    Fetch current news from RSS feeds
    Note: RSS feeds only provide recent news (usually last few days/weeks)
    """
    print(f"[INFO] Fetching current news for {ticker}...")
    all_rows = []

    for src_name, url in sources.items():
        print(f"[INFO] Trying {src_name}...")
        try:
            feed = feedparser.parse(url)

            if not feed.entries:
                print(f"[WARNING] {src_name} returned no entries")
                continue

            for entry in feed.entries[:max_items]:  # Limit per source
                # Get publication date
                pub_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    pub_date = datetime(*entry.updated_parsed[:6])
                else:
                    pub_date = datetime.now()

                title = entry.get("title", "").strip()
                if not title:
                    continue

                all_rows.append({
                    "time_published": pub_date.strftime("%Y%m%d"),
                    "source": src_name,
                    "url": entry.get("link", ""),
                    "title": title,
                    "ticker": ticker,
                    "summary": entry.get("summary", "")
                })

            print(f"[INFO] {src_name}: collected {len([r for r in all_rows if r['source']==src_name])} articles")

        except Exception as e:
            print(f"[WARNING] Error with {src_name}: {e}")
            continue

    df = pd.DataFrame(all_rows)

    if df.empty:
        print("[WARNING] No news collected from any source")
        print("[INFO] Creating minimal placeholder data for testing...")
        # Create minimal placeholder for testing
        df = pd.DataFrame([{
            "time_published": datetime.now().strftime("%Y%m%d"),
            "source": "placeholder",
            "url": "",
            "title": "Market update",
            "ticker": ticker,
            "summary": "Financial markets trading activity"
        }])
    else:
        df = df.drop_duplicates(subset=["title"])
        print(f"[SUCCESS] Collected {len(df)} unique news items")

    df.to_csv("news.csv", index=False, encoding="utf-8")
    print(f"[SUCCESS] Saved to news.csv")
    return df

if __name__ == "__main__":
    fetch_current_news("QQQ")
