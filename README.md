# NIFTY 50 Trend Classification using Machine Learning

A complete end-to-end ML pipeline that predicts the 5-day trend
direction of India's NIFTY 50 index using Decision Tree,
Random Forest, and Logistic Regression.

---

## Project Overview

**Problem statement:**  
Given today's NIFTY 50 market indicators, will the index be
higher or lower 5 trading days from now?  
→ Binary classification: **UP (1)** or **DOWN (0)**

**Why 5-day instead of 1-day?**  
Next-day prediction is too noisy — one tweet or FII flow can move
markets randomly. Over 5 trading days, the underlying trend
dominates over daily noise. Our features (RSI, MA_ratio, momentum)
are trend signals and deserve a trend target.

---

## Results

| Algorithm | Accuracy | Down recall | Up recall | Window |
|---|---|---|---|---|
| Decision Tree | 53.8% | 16% | 78% | 2yr (2022-23) |
| Random Forest | 52.8% | 29% | 68% | 10yr (2014-23) |
| **Logistic Regression** | **56.4%** | **60%** | **54%** | 10yr (2014-23) |

All models pass the **honest selection rule**:
- Down recall >= 25% (no majority class trap)
- Up recall >= 40% (genuinely predicts both classes)

---

## Data Source

**Kaggle NIFTY 50 Dataset (1990-2024)**

```
URL: https://www.kaggle.com/datasets/adarshde/nifty-50-data-1990-2024
Rows: 8,325 trading days
Columns: Date, Open, High, Low, Close
```

**Why Kaggle over yfinance?**
- 33 years of data vs ~5 years from yfinance
- Verified to match yfinance exactly (0.00 pts difference)
- More training data = better generalisation for ML

**Download instructions:** Run `get_data.py` for step-by-step guide.

---

## File Structure

```
├── get_data.py         # Download guide + dataset validator
├── eda.py              # 5-check EDA on 33 years of data
├── featureeng.py       # Feature engineering (11 signals)
├── train_dt.py         # Decision Tree + grid search
├── train_rf.py         # Random Forest + honest selection
├── train_lr.py         # Logistic Regression + coefficients
├── predict.py          # 3-model consensus predictor (live)
├── track.py            # Daily prediction tracker
└── trade_signal.py     # Trade levels module (entry/SL/target)
```

---

## Features Engineered (11 signals)

| Feature | What it captures |
|---|---|
| MA_ratio | Scale-free short vs long-term trend |
| Market_regime | Price above/below MA50 |
| RSI | Momentum exhaustion (overbought/oversold) |
| Daily_Return | Today's percentage move |
| Return_3d | 3-day momentum |
| Return_5d | Weekly momentum |
| Volatility_5 | 5-day rolling market nervousness |
| Price_Position | Where close landed in day's range |
| RSI_lag1 | Yesterday's RSI (direction of momentum) |
| Daily_Return_lag1 | Yesterday's return |
| MA_ratio_lag1 | Yesterday's MA ratio |

**Key insight:** RSI_lag1 (yesterday's RSI) was the most
important feature — the *direction* of momentum matters more
than its current level.

---

## Data Cleaning Decisions (EDA-informed)

| What removed | Why | Rows |
|---|---|---|
| Year 1990 | Only 88 days, illiquid pre-modern market | 88 |
| Year 1992 | Harshad Mehta scam, 28 extreme days | 190 |
| GFC Sep2008-Mar2009 | 35 extreme days — noisiest period | 139 |
| COVID Feb-Jun 2020 | Structural break, 12 extreme days | 89 |
| Pre-1995 OHLC rows | Open/High/Low stored as dash | 1,125 |

---

## Live Prediction Workflow

```bash
# Step 1 — Generate features from Kaggle CSV
python featureeng.py

# Step 2 — Get 3-model consensus prediction
python predict.py
# Enter today's Open, High, Low, Close
# DT + RF + LR vote → consensus UP/DOWN + strength

# Step 3 — Log prediction
python track.py → option 1

# Step 4 — Monitor daily (after 3:30 PM IST)
python track.py → option 2
# Automatically fires BUY/SELL/WAIT when signals >= 70%
```

---

## Key ML Concepts Demonstrated

- Correlated features (OHLC 0.99) vs independent engineered signals
- Majority class trap detection and correction
- Time-based train/test split (no data leakage)
- Walk-forward training window comparison (2yr vs 5yr vs 10yr)
- Grid search: depth x min_leaf x class_weight
- Honest model selection with minimum recall thresholds
- Crash period removal (structural breaks)
- Feature scaling pipeline (StandardScaler + LogisticRegression)
- Three-model consensus voting system

---

## Project Journey

| Version | Change | Accuracy |
|---|---|---|
| v1 | Basic DT, raw OHLC | 45.4% (majority trap) |
| v2 | Engineered features | 50.0% (majority trap) |
| v3 | class_weight + grid search | 44.2% (honest) |
| v4 | Lag features + crash removal | 49.6% |
| v5 | Kaggle 33yr + walk-forward | 54.9% |
| Final | 3-model consensus (DT+RF+LR) | **56.4%** |

---

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn reportlab
```

---

## Disclaimer

> This project is for **EDUCATIONAL purposes only**.
> It is **NOT** financial advice.
> Do **NOT** use this for real trading decisions.
> Model accuracy is ~53-56%.
> Always consult a SEBI-registered financial advisor.
