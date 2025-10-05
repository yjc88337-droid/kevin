# 4_news_sentiment_analysis.py
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# 读取清洗后的逐条新闻 & 股价（用于对齐）
news = pd.read_csv("news_data.csv", parse_dates=["Date"])
stock = pd.read_csv("stock_price.csv", parse_dates=["Date"])
stock["Date"] = stock["Date"].dt.date

if news.empty:
    # 没有新闻时也给出全 0 的情绪列，保证后续能跑
    print("⚠️ news_data.csv 为空，将输出全 0 情绪。")
    pd.DataFrame({"FinBERT score": [0.0]*len(stock)}).to_csv("sentiment.csv", index=False)
    raise SystemExit(0)

# FinBERT 一次加载，自动用 GPU（若可用）
device = 0 if torch.cuda.is_available() else -1
tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
clf = pipeline("sentiment-analysis", model=mdl, tokenizer=tok, device=device, truncation=True)

# 批量打分（对 title）
titles = news["title"].fillna("").astype(str).tolist()
batch_size = 128
signed_scores = []

for i in tqdm(range(0, len(titles), batch_size), desc="Scoring with FinBERT"):
    batch = titles[i:i+batch_size]
    results = clf(batch)
    for r in results:
        label = r["label"].lower()
        score = r["score"]
        if "pos" in label:
            signed_scores.append(+score)
        elif "neg" in label:
            signed_scores.append(-score)
        else:
            signed_scores.append(0.0)

news["finbert_signed"] = signed_scores

# 按日聚合（平均情绪 & 新闻条数）
daily = (
    news.assign(Date=news["Date"].dt.date if hasattr(news["Date"].dtype, "tz") else news["Date"].dt.date)
        .groupby("Date")
        .agg(finbert_mean=("finbert_signed", "mean"),
             news_count=("finbert_signed", "size"))
        .reset_index()
        .sort_values("Date")
)

# 与股价交易日对齐（缺失填 0）
daily_aligned = stock[["Date"]].merge(daily, on="Date", how="left").fillna({"finbert_mean": 0.0, "news_count": 0})

# 调试文件（含日期和统计）
daily_aligned.to_csv("sentiment_daily_debug.csv", index=False)

# 生成 7_lstm_model_bert.py 需要的格式：只有一列 FinBERT score，长度与股价一致
out = daily_aligned.rename(columns={"finbert_mean": "FinBERT score"})[["FinBERT score"]]
out.to_csv("sentiment.csv", index=False)
print("✅ saved sentiment.csv:", out.shape, "（与 stock_price.csv 行数一致）")
