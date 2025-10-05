import pandas as pd
import yfinance as yf
from datetime import datetime

def download_stock_data(ticker="QQQ", start="2015-01-01", end="2024-01-01"):
    """
    Download stock price data from Yahoo Finance and save as CSV

    Args:
        ticker: Stock ticker symbol (default: QQQ)
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
    """
    print(f"[INFO] Downloading {ticker} from {start} to {end} ...")
    try:
        stock_data = yf.download(ticker, start=start, end=end, progress=True)

        if stock_data.empty:
            print("[ERROR] No data downloaded! Please check ticker or date range.")
            return None

        # Reset index to make Date a column
        stock_data.reset_index(inplace=True)

        # Ensure consistent column names
        if 'Adj Close' in stock_data.columns:
            stock_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Save to CSV
        stock_data.to_csv("stock_price.csv", index=False)
        print(f"[SUCCESS] Downloaded {len(stock_data)} rows and saved to stock_price.csv")
        print(f"[INFO] Date range: {stock_data['Date'].min()} to {stock_data['Date'].max()}")
        return stock_data

    except Exception as e:
        print(f"[ERROR] Error downloading stock data: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    download_stock_data("QQQ", "2015-01-01", "2024-01-01")
