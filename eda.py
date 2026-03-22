# ============================================================
# eda.py -- v2 Kaggle dataset (33 years of NIFTY 50)
# ============================================================
# 5 checks specific to the new longer dataset:
#   1. Date continuity  -- any gaps or missing years?
#   2. Cross-check      -- prices match yfinance for 2019-2024?
#   3. Regime changes   -- how different are the decades?
#   4. Class balance    -- Up/Down ratio consistent across years?
#   5. Extreme events   -- all crashes, not just COVID
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import numpy as np

# -- 0. Load Kaggle CSV ---------------------------------------
df = pd.read_csv("nifty50_kaggle.csv",
                 parse_dates=["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)
df = df.set_index("Date")
df = df[["Open", "High", "Low", "Close"]].dropna()

print("=" * 55)
print("KAGGLE DATASET -- BASIC INFO")
print("=" * 55)
print(f"Total rows    : {len(df)}")
print(f"Date range    : {df.index[0].date()} --> {df.index[-1].date()}")
print(f"Years covered : {df.index[-1].year - df.index[0].year + 1}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
print(f"\nLast 3 rows:\n{df.tail(3)}")
print(f"\nSummary stats:\n{df.describe().round(2)}")

# -- CHECK 1: Date continuity ---------------------------------
print("\n" + "=" * 55)
print("CHECK 1 -- DATE CONTINUITY")
print("=" * 55)

# Count trading days per year
df["Year"] = df.index.year
yearly_counts = df.groupby("Year").size()
print("\nTrading days per year:")
print(yearly_counts.to_string())

# Flag years with suspiciously few days (< 200 = incomplete)
thin_years = yearly_counts[yearly_counts < 200]
if len(thin_years) > 0:
    print(f"\nWARNING -- Thin years (< 200 trading days):")
    print(thin_years)
else:
    print("\nAll years have >= 200 trading days. Good.")

# Check for gaps > 5 calendar days (excluding weekends)
df_sorted   = df.sort_index()
date_diffs  = df_sorted.index.to_series().diff().dt.days.dropna()
big_gaps    = date_diffs[date_diffs > 7]
print(f"\nGaps > 7 calendar days: {len(big_gaps)}")
if len(big_gaps) > 0:
    print("Largest gaps:")
    print(big_gaps.sort_values(ascending=False).head(10))

# -- CHECK 2: Cross-check vs yfinance -------------------------
print("\n" + "=" * 55)
print("CHECK 2 -- CROSS-CHECK vs YFINANCE (2019-2024)")
print("=" * 55)

try:
    yf_df = yf.download("^NSEI", start="2019-01-01",
                        end="2024-12-31", auto_adjust=True)
    yf_df.columns = yf_df.columns.get_level_values(0)
    yf_close = yf_df["Close"]

    # Compare on common dates
    kaggle_2019 = df.loc["2019-01-01":"2024-12-31", "Close"]
    common_dates = kaggle_2019.index.intersection(yf_close.index)

    if len(common_dates) > 0:
        diff = (kaggle_2019[common_dates] - yf_close[common_dates]).abs()
        print(f"Common dates compared : {len(common_dates)}")
        print(f"Max price difference  : {diff.max():.2f} pts")
        print(f"Mean price difference : {diff.mean():.2f} pts")
        print(f"Dates with diff > 50  : {(diff > 50).sum()}")
        if diff.max() < 10:
            print("Prices match closely. Data is consistent.")
        elif diff.max() < 100:
            print("Minor differences -- likely rounding. Acceptable.")
        else:
            print("WARNING -- Large differences. Check data source.")
    else:
        print("No common dates found for comparison.")
except Exception as e:
    print(f"yfinance comparison skipped: {e}")

# -- CHECK 3: Regime changes across decades -------------------
print("\n" + "=" * 55)
print("CHECK 3 -- MARKET REGIMES ACROSS DECADES")
print("=" * 55)

df["Daily_Return"] = df["Close"].pct_change() * 100
decade_stats = df.groupby(df.index.year // 10 * 10)["Daily_Return"].agg(
    ["mean", "std", "min", "max", "count"]).round(3)
decade_stats.index = [f"{y}s" for y in decade_stats.index]
print("\nReturn stats by decade:")
print(decade_stats.to_string())

# -- CHECK 4: Class balance per period ------------------------
print("\n" + "=" * 55)
print("CHECK 4 -- TARGET CLASS BALANCE BY PERIOD")
print("=" * 55)

# Create 5-day target
df["Target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)

# Balance per 5-year period
df["Period"] = pd.cut(df.index.year,
                      bins=[1989, 1995, 2000, 2005, 2010,
                            2015, 2020, 2025],
                      labels=["1990-95", "1996-00", "2001-05",
                               "2006-10", "2011-15", "2016-20",
                               "2021-24"])
balance = df.groupby("Period")["Target"].agg(
    Up=lambda x: (x == 1).sum(),
    Down=lambda x: (x == 0).sum(),
    Up_pct=lambda x: f"{(x == 1).mean()*100:.1f}%"
)
print("\nUp/Down balance per 5-year period:")
print(balance.to_string())

# -- CHECK 5: Extreme events inventory ------------------------
print("\n" + "=" * 55)
print("CHECK 5 -- EXTREME EVENTS (daily move > 4%)")
print("=" * 55)

extreme = df[df["Daily_Return"].abs() > 4][["Close", "Daily_Return"]]
print(f"Total extreme days (>4% move): {len(extreme)}")
print(f"\nTop 15 biggest moves:")
print(extreme.reindex(
    extreme["Daily_Return"].abs().sort_values(
        ascending=False).index).head(15).round(2))

# Cluster extreme days by year
extreme_by_year = extreme.groupby(extreme.index.year).size()
print(f"\nExtreme days per year (top 10 years):")
print(extreme_by_year.sort_values(ascending=False).head(10))

print(f"\nCrash periods to consider removing from training:")
print(f"  COVID  : Feb 15 - Jun 30, 2020 "
      f"({((extreme.index >= '2020-02-15') & (extreme.index <= '2020-06-30')).sum()} days)")
print(f"  GFC    : Sep 01 - Dec 31, 2008 "
      f"({((extreme.index >= '2008-09-01') & (extreme.index <= '2008-12-31')).sum()} days)")
print(f"  Dotcom : Jan 01 - Dec 31, 2001 "
      f"({((extreme.index >= '2001-01-01') & (extreme.index <= '2001-12-31')).sum()} days)")

# -- PLOTS ----------------------------------------------------
fig = plt.figure(figsize=(18, 14))
fig.suptitle("NIFTY 50 Kaggle EDA -- 33 Years of Data",
             fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# Plot 1: Full price history
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df.index, df["Close"], color="#1565C0",
         linewidth=0.8, alpha=0.9)
ax1.fill_between(df.index, df["Close"],
                 df["Close"].min(), alpha=0.06, color="#1565C0")
# Annotate major events
events = {
    "Dotcom\n2001" : ("2001-09-01", 900),
    "GFC\n2008"    : ("2008-10-01", 2500),
    "COVID\n2020"  : ("2020-03-23", 5000),
}
for label, (date, y_pos) in events.items():
    try:
        ax1.annotate(label,
                     xy=(pd.Timestamp(date),
                         df.loc[date:date].iloc[0]["Close"]),
                     xytext=(pd.Timestamp(date), y_pos),
                     arrowprops=dict(arrowstyle="->", color="red"),
                     color="red", fontsize=8, ha="center")
    except Exception:
        pass
ax1.set_title("33 Years of NIFTY 50 -- Full Price History")
ax1.set_ylabel("Index Value")
ax1.grid(alpha=0.3)

# Plot 2: Trading days per year
ax2 = fig.add_subplot(gs[1, 0])
colors_bar = ["#EF5350" if v < 200 else "#42A5F5"
              for v in yearly_counts.values]
yearly_counts.plot(kind="bar", ax=ax2, color=colors_bar, width=0.8)
ax2.axhline(200, color="red", linestyle="--",
            linewidth=1, label="Min threshold (200)")
ax2.set_title("Check 1 -- Trading Days per Year")
ax2.set_ylabel("Days")
ax2.set_xlabel("")
ax2.legend(fontsize=8)
ax2.tick_params(axis="x", rotation=90, labelsize=7)
ax2.grid(axis="y", alpha=0.3)

# Plot 3: Class balance per period
ax3 = fig.add_subplot(gs[1, 1])
up_pcts = df.groupby("Period")["Target"].mean() * 100
colors_period = ["#EF5350" if v < 45 else
                 "#66BB6A" if v > 55 else "#42A5F5"
                 for v in up_pcts.values]
up_pcts.plot(kind="bar", ax=ax3, color=colors_period, width=0.7)
ax3.axhline(50, color="black", linestyle="--",
            linewidth=1, label="50% baseline")
ax3.set_title("Check 4 -- Up% by Period\n(red<45%, blue=balanced, green>55%)")
ax3.set_ylabel("Up days %")
ax3.set_ylim(30, 70)
ax3.set_xlabel("")
ax3.legend(fontsize=8)
ax3.tick_params(axis="x", rotation=30)
ax3.grid(axis="y", alpha=0.3)

# Plot 4: Daily returns distribution
ax4 = fig.add_subplot(gs[2, 0])
returns = df["Daily_Return"].dropna()
ax4.hist(returns, bins=120, color="#7B1FA2",
         edgecolor="none", alpha=0.8)
ax4.axvline(0, color="black", linewidth=1)
ax4.axvline(-4, color="red", linestyle="--",
            linewidth=1, label="-4% threshold")
ax4.axvline(4, color="red", linestyle="--",
            linewidth=1, label="+4% threshold")
ax4.set_title("Daily Returns Distribution (33 years)")
ax4.set_xlabel("Daily Return %")
ax4.set_ylabel("Frequency")
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# Plot 5: Volatility by decade (rolling 30-day std)
ax5 = fig.add_subplot(gs[2, 1])
rolling_vol = returns.rolling(30).std()
ax5.plot(rolling_vol.index, rolling_vol,
         color="#E65100", linewidth=0.7, alpha=0.8)
ax5.set_title("30-day Rolling Volatility (33 years)\n"
              "Spikes = crash periods")
ax5.set_ylabel("Volatility (std %)")
ax5.grid(alpha=0.3)

plt.savefig("nifty50_kaggle_eda.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved nifty50_kaggle_eda.png")

print("\n" + "=" * 55)
print("KEY TAKEAWAYS FOR FEATUREENG + TRAINING")
print("=" * 55)
print("  1. Note any thin years -- consider dropping from training")
print("  2. Check if price mismatch vs yfinance is acceptable")
print("  3. Regime diff across decades --> walk-forward matters more")
print("  4. Class balance varies per period --> use class_weight")
print("  5. Note all crash clusters beyond COVID to decide on removal")