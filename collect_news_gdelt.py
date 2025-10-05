# 1_news_collection_gdelt.py
# GDELT Historical News Collection - Automatically handles field variations
import requests, time, pandas as pd
from datetime import datetime, timedelta
from io import StringIO

BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY = '(nasdaq OR "nasdaq 100" OR "qqq" OR "stock market")'

# Configurable date range
START = "2015-01-01"
END   = "2023-12-31"

def month_ranges(start_date, end_date):
    d = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")
    cur = datetime(d.year, d.month, 1)
    out = []
    while cur <= e:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        last_day = (nxt - timedelta(seconds=1))
        if last_day > e:
            last_day = e.replace(hour=23, minute=59, second=59)
        out.append((cur.strftime("%Y%m%d000000"), last_day.strftime("%Y%m%d%H%M%S")))
        cur = nxt
    return out

def fetch_month(sdt, edt):
    params = {
        "query": QUERY,
        "mode": "artlist",
        "maxrecords": 250,
        "format": "CSV",
        "sourcelang": "English",
        "startdatetime": sdt,
        "enddatetime": edt,
    }
    r = requests.get(BASE, params=params, timeout=30)
    if not r.text.strip():
        print(f"[WARNING] Empty for {sdt[:6]}")
        return pd.DataFrame()
    df = pd.read_csv(StringIO(r.text))
    if df.empty:
        print(f"[WARNING] No data for {sdt[:6]}")
        return pd.DataFrame()

    # 自动匹配字段
    cols = [c.lower() for c in df.columns]
    if "documenttitle" in cols:
        df = df.rename(columns={"DocumentTitle": "title"})
    elif "title" not in cols:
        print(f"[WARNING] Skip {sdt[:6]} (no title field)")
        return pd.DataFrame()

    # 时间列
    if "seendate" in cols:
        df["time_published"] = pd.to_datetime(df["Seendate"], errors="coerce").dt.strftime("%Y%m%d")
    elif "date" in cols:
        df["time_published"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y%m%d")
    else:
        df["time_published"] = None

    # 来源列
    if "sourcecommonname" in cols:
        df["source"] = df["SourceCommonName"]
    elif "sourcecollectionidentifier" in cols:
        df["source"] = df["SourceCollectionIdentifier"]
    else:
        df["source"] = "Unknown"

    # 网址列
    if "documentidentifier" in cols:
        df["url"] = df["DocumentIdentifier"]
    elif "url" not in cols:
        df["url"] = None

    df["ticker"] = "QQQ"
    df = df[["time_published", "source", "url", "title", "ticker"]].dropna(subset=["title"])
    return df

def main():
    frames = []
    for sdt, edt in month_ranges(START, END):
        print(f"Fetching {sdt[:6]} ...")
        try:
            df = fetch_month(sdt, edt)
            if not df.empty:
                frames.append(df)
            time.sleep(1.5)
        except Exception as e:
            print(f"[ERROR] Error in {sdt[:6]}: {e}")
            time.sleep(3)
            continue

    if frames:
        out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["url", "title"])
        out.to_csv("news.csv", index=False)
        print(f"[SUCCESS] Saved news.csv with {len(out)} rows ({START} to {END})")
    else:
        print("[ERROR] No news data fetched at all.")

if __name__ == "__main__":
    main()
