# ============================================================
# get_data.py -- How to get the NIFTY 50 Kaggle dataset
# ============================================================
# This project uses the Kaggle NIFTY 50 dataset instead of
# yfinance because:
#   1. Kaggle has 33 years of data (1990-2024)
#   2. yfinance only provides ~5 years reliably for NIFTY
#   3. Kaggle data was verified to match yfinance exactly
#      (0.00 pts difference for overlapping dates)
#
# STEP 1: Download the dataset from Kaggle
# ============================================================
#
# URL: https://www.kaggle.com/datasets/adarshde/nifty-50-data-1990-2024
#
# Steps:
#   1. Create a free Kaggle account at kaggle.com
#   2. Go to the URL above
#   3. Click "Download" button
#   4. Extract the ZIP file
#   5. Rename the CSV to: nifty50_kaggle.csv
#   6. Place it in the same folder as this project
#
# ============================================================
# STEP 2: Run this script to validate the downloaded file
# ============================================================

import pandas as pd
import os
import sys

FILENAME = "nifty50_kaggle.csv"

print("=" * 55)
print("NIFTY 50 Kaggle Dataset Validator")
print("=" * 55)

# -- Check file exists ----------------------------------------
if not os.path.exists(FILENAME):
    print(f"\nFile not found: {FILENAME}")
    print("\nPlease download it from Kaggle:")
    print("https://www.kaggle.com/datasets/adarshde/nifty-50-data-1990-2024")
    print("\nSteps:")
    print("  1. Create a free Kaggle account")
    print("  2. Go to the URL above")
    print("  3. Click Download")
    print("  4. Rename CSV to: nifty50_kaggle.csv")
    print("  5. Place in this project folder")
    print("  6. Run this script again")
    sys.exit(1)

# -- Load and validate ----------------------------------------
print(f"\nFound: {FILENAME}")

df = pd.read_csv(FILENAME, parse_dates=["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

print(f"\nBasic info:")
print(f"  Rows        : {len(df)}")
print(f"  Columns     : {df.columns.tolist()}")
print(f"  Date range  : {df['Date'].min().date()} "
      f"to {df['Date'].max().date()}")

# -- Check required columns -----------------------------------
required = ["Date", "Open", "High", "Low", "Close"]
missing  = [c for c in required if c not in df.columns]

if missing:
    print(f"\nMissing columns: {missing}")
    print("This may be a different version of the dataset.")
    print("Expected columns: Date, Open, High, Low, Close")
    sys.exit(1)

# -- Check date range -----------------------------------------
years = df["Date"].dt.year.nunique()
print(f"\nData quality:")
print(f"  Years covered : {years}")
print(f"  Total rows    : {len(df)}")

if len(df) < 5000:
    print("  WARNING: Fewer rows than expected (>8000).")
    print("  Make sure you downloaded the full dataset.")
else:
    print(f"  Row count OK  : {len(df)} rows (expected ~8,325)")

# -- Check for early dash values ------------------------------
sample = df.head(500)
dash_count = (sample[["Open", "High", "Low"]]
              .astype(str).apply(lambda x: x.str.strip() == "-")
              .sum().sum())
print(f"  Dash values   : {dash_count} found in first 500 rows")
print(f"  (Early years had only Close price -- normal)")

# -- Final verdict --------------------------------------------
print(f"\n{'='*55}")
if len(df) >= 5000 and not missing:
    print("Dataset looks good! Ready to run featureeng.py")
    print("\nNext step: python featureeng.py")
else:
    print("Dataset has issues. Check the warnings above.")
print(f"{'='*55}")