# 历史新闻数据收集尝试记录

## 尝试时间
2025-10-05

## 目标
收集2015-2023年的历史新闻数据，以匹配股票价格数据的时间范围

## 尝试的数据源

### 1. Yahoo Finance News API ❌
**方法**: 使用 `yfinance` 库的 `.news` 属性
**结果**: 失败
**原因**:
- API返回空数据
- Yahoo Finance已经限制或移除了历史新闻API

**代码测试**:
```python
import yfinance as yf
stock = yf.Ticker("QQQ")
news = stock.news  # 返回 []
```

---

### 2. RSS Feeds (Reuters, CNBC, MarketWatch) ❌
**方法**: 使用 feedparser 解析RSS feeds
**结果**: 部分成功但不满足需求
**问题**:
- RSS feeds只提供最近7-30天的新闻
- 无法获取2015-2023的历史数据
- 获取的数据日期为2024-2025

**获取结果**:
- 成功获取50条新闻
- 但全部是2024年9月-2025年10月的新闻
- 与股票数据时间范围（2015-2023）不匹配

---

### 3. GDELT Project API ❌
**方法**: 使用GDELT v2 DOC API
**结果**: API工作但无法获取历史数据
**详情**:

**测试1 - 无日期参数**:
```python
params = {
    'query': 'QQQ nasdaq',
    'mode': 'artlist',
    'maxrecords': 50,
    'format': 'CSV'
}
```
- ✅ 成功返回50条新闻
- ❌ 但全部是2025年的新闻

**测试2 - 指定历史日期范围**:
```python
params = {
    'query': 'nasdaq',
    'startdatetime': '20231201000000',
    'enddatetime': '20231231235959',
    ...
}
```
- ❌ 返回空数据
- 结论：GDELT API不支持历史查询，或需要付费订阅

---

### 4. AlphaVantage News API ❌
**方法**: 原项目中的脚本
**结果**: API密钥已失效
**操作**: 已删除该脚本

---

## 其他可能的数据源（未尝试）

### 需要付费或API密钥：
1. **NewsAPI.org**
   - 免费版：只能获取最近30天
   - 付费版：可获取历史数据
   - 需要API密钥

2. **Financial Modeling Prep**
   - 提供历史新闻
   - 免费tier有限制
   - 需要API密钥

3. **Polygon.io**
   - 提供股票新闻
   - 需要付费订阅

4. **Bloomberg API** / **Reuters API**
   - 专业级数据
   - 需要企业订阅

### 免费但需要手动：
5. **Kaggle 数据集**
   - 可能有现成的金融新闻数据集
   - 需要手动下载和处理

6. **Web Scraping**
   - 爬取新闻网站归档
   - 可能违反服务条款
   - 技术难度高

---

## 结论

**无法通过免费API获取2015-2023的历史新闻数据**

原因：
1. 大多数免费API只提供最近的新闻（7-30天）
2. 历史新闻数据通常需要付费订阅
3. GDELT虽然是历史数据库，但其API不支持免费的历史查询

---

## 决策：使用备份的sentiment.csv

### 备份数据评估

**文件**: `backup_data/sentiment.csv`

**基本信息**:
- 行数: 503行
- 时间范围: 2020-10-01 至 2022-09-29
- 数据格式: 每天10条新闻标题 + 1个FinBERT情感分数

**数据质量检查**:
- ✅ 新闻标题看起来真实且相关
- ✅ 日期连续，无明显缺失
- ✅ 情感分数范围合理（-0.98 至 +0.95）
- ✅ 包含疫情、选举、市场等重大事件的新闻
- ⚠️ 数据来源不明

**示例新闻**（2020-10-01）:
1. "A standoff over further federal aid and concern over the pandemic's duration..."
2. "Tesla reported record deliveries in the third quarter..."
3. "Lawmakers said they found multiple problems with each of the four giant tech companies..."

**相关性验证**:
- 新闻内容与QQQ（科技股ETF）高度相关
- 包含Apple, Amazon, Google, Facebook等Nasdaq 100成分股的新闻
- 反映了2020-2022年的主要市场事件

---

## 推荐方案

### 方案 A：使用备份数据 + 匹配股票数据（推荐）

**步骤**:
1. 使用 `backup_data/sentiment.csv` (2020-2022数据)
2. 重新下载2020-2022的股票数据以匹配
3. 训练和测试模型

**优点**:
- 数据时间范围匹配
- 立即可用
- 包含真实的历史事件

**缺点**:
- 时间范围较短（2年 vs 原计划的8年）
- 数据来源不完全透明

---

### 方案 B：扩展到当前数据（未来工作）

如果需要更多数据，可以：
1. 保留2020-2022的历史数据
2. 添加2024-2025的当前数据（已收集）
3. 更新股票数据到2025年
4. 创建两个数据集用于对比研究

---

## 下一步操作

根据"找不到就用原来的"原则：

1. ✅ 恢复 `backup_data/sentiment.csv` 作为主数据
2. 🔄 调整股票数据以匹配时间范围（2020-2022）
3. ▶️ 继续pipeline：测试数据清洗和模型训练

---

*记录人: Claude Code*
*日期: 2025-10-05*
