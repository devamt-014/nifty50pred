# ============================================================
# featureeng.py -- v6 (EDA-informed cleaning)
# ============================================================
# Cleaning decisions from EDA:
#   DROP entire years : 1990 (88 days), 1992 (28 extreme days)
#   DROP crash windows: GFC Sep 2008-Mar 2009 (35 extreme days)
#                       COVID Feb 15-Jun 30 2020 (12 extreme days)
#   KEEP              : 1991, 1993-2007, 2010-2019, 2021-2023
# ============================================================

import pandas as pd

# -- 0. Load Kaggle CSV ---------------------------------------
df = pd.read_csv("nifty50_kaggle.csv",
                 parse_dates=["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)
df = df.set_index("Date")
df = df[["Open", "High", "Low", "Close"]].copy()

# Early years (pre-2000) had only Close prices recorded
# Open/High/Low stored as "-" strings -- convert to NaN then drop
for col in ["Open", "High", "Low", "Close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

n_before = len(df)
df.dropna(inplace=True)
n_dropped = n_before - len(df)
print(f"Raw rows loaded : {n_before}")
print(f"Dropped {n_dropped} rows with missing Open/High/Low (early data)")

# -- 1. Drop thin/broken years --------------------------------
# 1990: only 88 days, illiquid pre-modern market
# 1992: Harshad Mehta scam -- 28 extreme days, 19-day gaps
bad_years = [1990, 1992]
df = df[~df.index.year.isin(bad_years)].copy()
print(f"After dropping years {bad_years} : {len(df)} rows")

# -- 2. Drop crash windows ------------------------------------
# GFC: Sep 2008 - Mar 2009 (35 extreme days -- noisiest period)
# COVID: Feb 15 - Jun 30 2020 (structural break)
crash_windows = [
    ("2008-09-01", "2009-03-31", "GFC"),
    ("2020-02-15", "2020-06-30", "COVID"),
]
for start, end, label in crash_windows:
    mask = (df.index >= start) & (df.index <= end)
    n    = mask.sum()
    df   = df[~mask].copy()
    print(f"Removed {n:>4} rows  [{label}: {start} to {end}]")

print(f"\nClean rows remaining : {len(df)}")
print(f"Date range           : {df.index[0].date()} --> "
      f"{df.index[-1].date()}")

# -- 3. MA ratio & regime -------------------------------------
MA5  = df["Close"].rolling(5).mean()
MA20 = df["Close"].rolling(20).mean()
MA50 = df["Close"].rolling(50).mean()
df["MA_ratio"]      = MA5 / MA20
df["Market_regime"] = (df["Close"] > MA50).astype(int)

# -- 4. RSI (14-day) ------------------------------------------
delta      = df["Close"].diff()
gain       = delta.clip(lower=0)
loss       = (-delta).clip(lower=0)
df["RSI"]  = 100 - (100 / (1 + gain.rolling(14).mean()
                               / loss.rolling(14).mean()))

# -- 5. Return features ---------------------------------------
df["Daily_Return"] = df["Close"].pct_change() * 100
df["Return_3d"]    = df["Close"].pct_change(3) * 100
df["Return_5d"]    = df["Close"].pct_change(5) * 100

# -- 6. Volatility --------------------------------------------
df["Volatility_5"] = df["Daily_Return"].rolling(5).std()

# -- 7. Price position ----------------------------------------
df["Price_Position"] = ((df["Close"] - df["Low"])
                        / (df["High"] - df["Low"]))

# Note: No Volume_Ratio -- Kaggle has no Volume column

# -- 8. Lag features ------------------------------------------
df["RSI_lag1"]          = df["RSI"].shift(1)
df["Daily_Return_lag1"] = df["Daily_Return"].shift(1)
df["MA_ratio_lag1"]     = df["MA_ratio"].shift(1)

# -- 9. Target -- 5-day forward direction ---------------------
df["Target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)

# -- 10. Drop NaN from rolling windows ------------------------
df.dropna(inplace=True)

print(f"After dropping NaN  : {len(df)} rows")

# -- 11. Final health check -----------------------------------
counts = df["Target"].value_counts()
pct    = df["Target"].value_counts(normalize=True) * 100
print(f"\nTarget distribution (5-day):")
print(f"  Up (1)   : {counts[1]}  ({pct[1]:.1f}%)")
print(f"  Down (0) : {counts[0]}  ({pct[0]:.1f}%)")

print(f"\nRows available per training window:")
for label, start in [("2 yrs  2022-2023", "2022-01-01"),
                     ("5 yrs  2019-2023", "2019-01-01"),
                     ("10 yrs 2014-2023", "2014-01-01"),
                     ("Full  1991-2023",  "1991-01-01")]:
    n = ((df.index >= start) & (df.index < "2024-01-01")).sum()
    print(f"  {label} : {n} days")

# -- 12. Save -------------------------------------------------
FEATURES = ["MA_ratio", "Market_regime", "RSI",
            "Daily_Return", "Return_3d", "Return_5d",
            "Volatility_5", "Price_Position",
            "RSI_lag1", "Daily_Return_lag1", "MA_ratio_lag1",
            "Target"]

df[FEATURES].to_csv("nifty50_features.csv", date_format="%d-%m-%Y")
print(f"\nSaved nifty50_features.csv  "
      f"({len(FEATURES)-1} features, {len(df)} rows)")