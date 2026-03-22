# NIFTY 50 Trend Classification using Decision Tree

## Project Overview
ML pipeline for classifying NIFTY 50 5-day trend direction
using Decision Tree, Random Forest, and Logistic Regression.

## Results
- Decision Tree  : 53.8% honest accuracy
- Random Forest  : 52.8% honest accuracy  
- Logistic Regr. : 56.4% honest accuracy

## File Structure
dataset.py      → fetch raw data
eda.py          → exploratory data analysis
featureeng.py   → feature engineering (11 signals)
train_dt.py     → Decision Tree (main model)
train_rf.py     → Random Forest (comparison)
train_lr.py     → Logistic Regression (comparison)
predict.py      → 3-model consensus predictor
track.py        → daily prediction tracker
trade_signal.py → trade level generator

## DISCLAIMER
This project is for EDUCATIONAL purposes only.
Not financial advice. Do not use for real trading.
Model accuracy is ~53-56%. Past performance does
not guarantee future results.

## Tech Stack
Python, scikit-learn, pandas, matplotlib, yfinance
```

---

## What to exclude — add a `.gitignore`
```
# Data files (too large + personal)
nifty50_kaggle.csv
nifty50_features.csv
nifty50_raw.csv

# Personal trading logs
predictions_log.csv
last_prediction.csv
signals_log.csv

# Generated plots
*.png

# Python cache
__pycache__/
*.pyc
