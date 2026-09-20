# 📈 Global Financial Markets Analysis & ML Forecasting

An end-to-end data analysis and machine learning project that explores **25+ years of global financial market data** across 34 assets — stocks, currencies, commodities, and cryptocurrencies — and delivers all findings through an interactive **Streamlit dashboard**.

---

## 📌 Project Highlights

- **190,545 daily records** spanning 2000–2026
- **34 assets** across 4 classes: Stock Indices, Currencies, Commodities, Cryptocurrencies
- **Zero missing values** in the raw dataset
- **6-tab interactive dashboard** with plain-English labels for non-technical users
- **Random Forest ML model** for price forecasting with rolling future predictions
- **Crash analysis**, seasonality heatmaps, drawdown charts, and business insights

---

## 🗂 Project Structure

```
STOCK_ANALYSIS/
│
├── app.py                              # Main Streamlit dashboard (run this)
├── global_financial_markets_2000_Now.csv  # Dataset (190,545 rows)
├── requirements.txt                   # Python dependencies
│
├── project_report.docx                # Full project report with charts
├── README.md                          # This file
│
├── report_images/                     # Auto-generated chart PNGs for the report
│   ├── 01_growth_1000.png
│   ├── 02_annual_returns.png
│   ├── 03_asset_types.png
│   ├── 04_crash_overlay.png
│   ├── 05_drawdown.png
│   ├── 06_heatmap.png
│   ├── 07_volatility.png
│   ├── 08_actual_vs_pred.png
│   ├── 09_forecast.png
│   ├── 10_feature_importance.png
│   └── 11_data_quality.png
│
├── generate_screenshots.py            # Script that produced report_images/
├── test_pipeline.py                   # Data pipeline smoke tests
├── analyze_data.py                    # Quick dataset inspection script
└── check_miss.py                      # Missing value checker
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the dashboard

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## 📊 Dashboard Tabs

| Tab | What it shows |
|-----|---------------|
| 🏠 **Overview** | $1,000 investment growth chart, performance summary table, asset breakdown |
| 📋 **Data Quality** | Zero-missing-value confirmation, price range checks, daily change histogram |
| 📈 **Price History** | Price line + 50/200-day moving averages, year-by-year returns, normalised comparison |
| 🔍 **Market Crashes** | 4 major crash overlays, crash impact table, all-time high drawdown chart |
| 📅 **Patterns & Seasons** | Monthly/quarterly seasonality bars, monthly heatmap, annual volatility |
| 🤖 **Price Forecast** | ML model output: actual vs predicted, rolling forecast, feature importance |

---

## 🗃 Dataset

**File:** `global_financial_markets_2000_Now.csv`

| Column | Description |
|--------|-------------|
| `date` | Trading date (YYYY-MM-DD) |
| `open` | Opening price |
| `high` | Intraday high |
| `low` | Intraday low |
| `close` | Closing price |
| `volume` | Trading volume |
| `symbol` | Ticker symbol (e.g. `^GSPC`, `BTC-USD`) |
| `asset_name` | Human-readable name (e.g. `S&P500`, `Bitcoin`) |
| `asset_type` | `Stock Index` / `Currency` / `Commodity` / `Cryptocurrency` |
| `region` | `Global` for all records |

### Assets Covered

| Class | Count | Examples |
|-------|-------|---------|
| Stock Indices | 10 | S&P 500, NASDAQ, SENSEX, Nikkei 225, DAX, FTSE 100, HSI, Shanghai, KOSPI, NIFTY 50 |
| Currencies | 8 | EUR/USD, GBP/USD, JPY/USD, INR/USD, CHF/USD, AUD/USD, CAD/USD, CNY/USD |
| Commodities | 8 | Gold, Silver, Crude Oil (WTI), Brent Crude, Natural Gas, Copper, Corn, Wheat |
| Cryptocurrencies | 8 | Bitcoin, Ethereum, BNB, Solana, XRP, Cardano, Dogecoin, Litecoin |

---

## 🤖 ML Model Details

The **Price Forecast** tab uses a **Random Forest Regressor** trained on each asset's historical data.

### Features engineered (13 total)

| Feature | Description |
|---------|-------------|
| `close` | Current day's closing price |
| `ma10`, `ma30`, `ma90` | 10 / 30 / 90-day moving averages |
| `vol10` | 10-day rolling return volatility |
| `mom10`, `mom30` | 10 / 30-day price momentum |
| `lag1` – `lag10` | Closing prices from 1, 2, 3, 5, 10 days ago |
| `month` | Calendar month (captures seasonality) |

### Training setup

- **Split:** 80% train / 20% test (chronological, no data leakage)
- **Scaler:** MinMaxScaler applied to all features
- **Estimators:** 300 trees, max depth 10
- **Forecast:** Rolling day-by-day prediction for up to 180 business days forward
- **Confidence band:** ±1.96 × RMSE (95% interval)

### Key metrics (S&P 500 example)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MAPE | ~17% | Average % price error (across a 25-year price range) |
| Direction Accuracy | ~53–55% | % of days the model correctly called up/down |
| Most important feature | `lag1` (yesterday's price) | Price has strong day-to-day continuity |

> **Note:** Direction Accuracy > 50% means the model beats a random coin flip. On price-level prediction across a multi-decade test set, MAPE is more informative than R².

---

## 🔍 Key Findings

### Hidden Trends
- **Crash cycle:** Major crashes occur approximately every 7–10 years, driven by credit over-expansion. Pattern: bubble → trigger → 30–60% drop → 2–5 year recovery.
- **Long-run equity bias:** Every index has recovered to new all-time highs after every crash, with no exceptions in the 25-year dataset.
- **India (SENSEX) acceleration:** Relative strength vs other EMs has accelerated sharply post-2020, signalling a structural bull market.

### Seasonality
- **January Effect:** Historically one of the strongest months across equity indices.
- **Q4 Rally:** October–December is the strongest quarter for US indices on average.
- **Sell in May:** April and May show below-average returns across most global indices.
- **Volatility spikes:** 2008, 2020, and 2022 stand out as extreme-risk years.

### Asset Class Insights
- **NASDAQ** delivered the highest long-run total return among stock indices.
- **Gold** acts as a reliable crisis hedge — it held value during every equity bear market.
- **Bitcoin** has the highest return ceiling but also the deepest drawdowns (70–85%).
- **Emerging market currencies (INR, CNY)** show a persistent long-run depreciation vs USD.

---

## 📦 Requirements

```
streamlit>=1.29.0
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.18.0
scikit-learn>=1.3.0
```

Install with:
```bash
pip install -r requirements.txt
```

To regenerate the report chart images (requires `kaleido>=1`):
```bash
pip install "kaleido>=1"
python generate_screenshots.py
```

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. All analysis, ML predictions, and portfolio recommendations are based on historical statistical patterns and do **not** constitute financial advice. Past performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.

---

## 📄 Report

A full written report with embedded charts is available as [`project_report.docx`](project_report.docx).

It covers:
1. Project Overview
2. Dataset Description
3. Data Collection and Loading
4. Data Quality Check
5. Exploratory Data Analysis
6. Price History and Trend Analysis
7. Market Crashes and Drawdowns
8. Hidden Trends and Seasonality
9. Machine Learning Price Forecast
10. Business Insights and Recommendations
11. Conclusion and Technology Stack
