# 3_news_data_cleaning.py
import pandas as pd
import os

def clean_news_data():
    """
    Clean and align news data with stock trading days
    """
    print("[INFO] Starting news data cleaning...")

    # Check if required files exist
    if not os.path.exists("news.csv"):
        print("[ERROR] news.csv not found! Please run news collection first.")
        return False

    if not os.path.exists("stock_price.csv"):
        print("[ERROR] stock_price.csv not found! Please run stock data collection first.")
        return False

    try:
        # 读取新闻与股价
        news = pd.read_csv("news.csv")
        stock = pd.read_csv("stock_price.csv")

        print(f"[INFO] Loaded {len(news)} news items")
        print(f"[INFO] Loaded {len(stock)} stock price records")

        # 确保股价数据有正确的列
        if "Date" not in stock.columns:
            stock = pd.read_csv("stock_price.csv", parse_dates=[0])
            stock.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

        # 确保日期列是 datetime 类型
        stock["Date"] = pd.to_datetime(stock["Date"], errors="coerce")
        stock["Date"] = stock["Date"].dt.date

        # 统一新闻时间格式
        if "time_published" in news.columns:
            news["Date"] = pd.to_datetime(
                news["time_published"].astype(str).str[:8],
                format="%Y%m%d",
                errors="coerce"
            ).dt.date
        else:
            print("[ERROR] 'time_published' column not found in news.csv")
            return False

        # 只保留需要的列 & 与交易日对齐
        news = news[["Date", "ticker", "title"]].dropna(subset=["Date", "title"])

        # Filter news to match trading days
        trade_days = set(stock["Date"])
        news_before = len(news)
        news = news[news["Date"].isin(trade_days)].sort_values(["Date"])
        news_after = len(news)

        print(f"[INFO] Filtered news: {news_before} -> {news_after} (aligned with trading days)")

        # 保存清洗后的逐条新闻
        news.to_csv("news_data.csv", index=False)
        print(f"[SUCCESS] Saved news_data.csv: {news.shape[0]} rows, {news.shape[1]} columns")

        # Print date range
        if len(news) > 0:
            print(f"[INFO] News date range: {news['Date'].min()} to {news['Date'].max()}")
        else:
            print("[WARNING] No news data after cleaning!")

        return True

    except Exception as e:
        print(f"[ERROR] Error during cleaning: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    clean_news_data()
