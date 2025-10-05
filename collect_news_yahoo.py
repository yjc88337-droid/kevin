# 1_news_collection_yahoo.py
# ✅ 稳定版：使用 yfinance 自带的新闻接口，直接抓取 QQQ 新闻

import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_yahoo_news(ticker="QQQ"):
    """
    Fetch news from Yahoo Finance for a given ticker
    Returns: DataFrame with news data
    """
    print(f"[INFO] Fetching Yahoo Finance news for {ticker} ...")
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news  # yfinance 自带新闻属性

        if not news_items:
            print("[WARNING] No news found for this ticker")
            return pd.DataFrame()

        rows = []
        for n in news_items:
            # Convert Unix timestamp to readable format
            timestamp = n.get("providerPublishTime", 0)
            if timestamp:
                date_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d")
            else:
                date_str = ""

            rows.append({
                "time_published": date_str,
                "ticker": ticker,
                "title": n.get("title", ""),
                "summary": n.get("summary", ""),
                "url": n.get("link", ""),
                "source": n.get("publisher", "")
            })

        df = pd.DataFrame(rows)
        # Remove duplicates and empty titles
        df = df[df["title"].str.strip() != ""].drop_duplicates(subset=["url"])

        df.to_csv("news.csv", index=False, encoding="utf-8")
        print(f"[SUCCESS] Saved news.csv with {len(df)} rows")
        return df
    except Exception as e:
        print(f"[ERROR] Error fetching news: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

if __name__ == "__main__":
    fetch_yahoo_news("QQQ")
