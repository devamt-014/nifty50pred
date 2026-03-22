# ============================================================
# predict.py -- NIFTY 50 Three-Model Consensus Predictor
# ============================================================
# Runs all 3 trained models and shows a consensus vote:
#   Model 1: Decision Tree  (best accuracy on Up days)
#   Model 2: Random Forest  (best at catching Down days)
#   Model 3: Logistic Reg.  (best overall + probability)
#
# Consensus rule:
#   3/3 agree --> STRONG signal
#   2/3 agree --> MODERATE signal
#   1/3 agree --> WEAK -- do not trade
#
# DISCLAIMER: Educational only. ~55% accuracy. Not financial advice.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

FEATURES = ["MA_ratio", "Market_regime", "RSI",
            "Daily_Return", "Return_3d", "Return_5d",
            "Volatility_5", "Price_Position",
            "RSI_lag1", "Daily_Return_lag1", "MA_ratio_lag1"]

# ── Step 1: Train all 3 models ────────────────────────────────
def train_all_models():
    try:
        df = pd.read_csv("nifty50_features.csv",
                         index_col="Date", parse_dates=True,
                         dayfirst=True)
        df.dropna(inplace=True)
    except FileNotFoundError:
        print(f"{RED}Error: nifty50_features.csv not found.{RESET}")
        print("Please run featureeng.py first.")
        exit()

    X = df[FEATURES]
    y = df["Target"]

    # DT -- best window: 2yr (2022-2023)
    X_dt = X[(X.index >= "2022-01-01") & (X.index < "2024-01-01")]
    y_dt = y[(y.index >= "2022-01-01") & (y.index < "2024-01-01")]
    dt   = DecisionTreeClassifier(
               max_depth=3, min_samples_leaf=10,
               class_weight={0: 1.2, 1: 1.0}, random_state=42)
    dt.fit(X_dt, y_dt)

    # RF -- best window: 10yr (2014-2023)
    X_rf = X[(X.index >= "2014-01-01") & (X.index < "2024-01-01")]
    y_rf = y[(y.index >= "2014-01-01") & (y.index < "2024-01-01")]
    rf   = RandomForestClassifier(
               n_estimators=200, max_depth=5,
               class_weight={0: 1.2, 1: 1.0},
               random_state=42, n_jobs=-1)
    rf.fit(X_rf, y_rf)

    # LR -- best window: 10yr (2014-2023), balanced, C=10
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=10.0, class_weight="balanced",
            solver="lbfgs", max_iter=1000, random_state=42))
    ])
    lr_pipe.fit(X_rf, y_rf)   # same 10yr window

    print(f"{GREEN}All 3 models trained:{RESET}")
    print(f"  DT  -- depth=3, 2yr window (2022-2023), "
          f"{len(X_dt)} days")
    print(f"  RF  -- 200 trees, 10yr window (2014-2023), "
          f"{len(X_rf)} days")
    print(f"  LR  -- C=10, balanced, 10yr window (2014-2023), "
          f"{len(X_rf)} days")
    return dt, rf, lr_pipe, df

# ── Step 2: Compute RSI ───────────────────────────────────────
def compute_rsi(closes, period=14):
    closes = np.array(closes, dtype=float)
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    if len(gains) < period:
        return 50.0
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    return round(100 - (100 / (1 + avg_gain / avg_loss)), 2)

# ── Step 3: Get user inputs ───────────────────────────────────
def get_inputs():
    print(f"\n{BOLD}{BLUE}{'='*58}{RESET}")
    print(f"{BOLD}{BLUE}  NIFTY 50 -- Three-Model Consensus Predictor{RESET}")
    print(f"{BOLD}{BLUE}{'='*58}{RESET}")
    print(f"\n  {YELLOW}Enter today's NIFTY 50 data after market close.{RESET}")
    print(f"  {YELLOW}Get values from NSE website or moneycontrol.com{RESET}\n")

    def ask(prompt, example, low=0, high=999999):
        while True:
            try:
                val = float(input(
                    f"  {prompt} {BLUE}(e.g. {example}){RESET}: "))
                if low <= val <= high:
                    return val
                print(f"  {RED}Enter a value between {low} and {high}{RESET}")
            except ValueError:
                print(f"  {RED}Please enter a valid number{RESET}")

    print(f"{BOLD}-- Today's prices --{RESET}")
    open_  = ask("Open  price", "23197.75")
    high   = ask("High  price", "23378.70", low=open_)
    low_   = ask("Low   price", "22930.35", high=high)
    close  = ask("Close price", "23002.15", low=low_, high=high)

    return {"open": open_, "high": high, "low": low_, "close": close}

# ── Step 4: Build features ───────────────────────────────────
def build_features(inputs, df):
    close  = inputs["close"]
    open_  = inputs["open"]
    high   = inputs["high"]
    low    = inputs["low"]

    last_row    = df.dropna().iloc[-1]
    second_last = df.dropna().iloc[-2]

    ma_ratio      = float(last_row["MA_ratio"])
    market_regime = int(last_row["Market_regime"])
    rsi           = float(last_row["RSI"])
    rsi_lag1      = float(second_last["RSI"])
    ma_ratio_lag1 = float(second_last["MA_ratio"])

    daily_return      = round((close - open_) / open_ * 100, 4)
    return_3d         = float(last_row["Return_3d"])
    return_5d         = float(last_row["Return_5d"])
    daily_return_lag1 = float(last_row["Daily_Return"])

    recent_returns = list(df["Daily_Return"].tail(4)) + [daily_return]
    volatility_5   = round(float(np.std(recent_returns)), 4)

    price_position = round(
        (close - low) / (high - low) if high != low else 0.5, 4)

    return {
        "MA_ratio"          : round(ma_ratio, 6),
        "Market_regime"     : market_regime,
        "RSI"               : round(rsi, 4),
        "Daily_Return"      : daily_return,
        "Return_3d"         : round(return_3d, 4),
        "Return_5d"         : round(return_5d, 4),
        "Volatility_5"      : volatility_5,
        "Price_Position"    : price_position,
        "RSI_lag1"          : round(rsi_lag1, 4),
        "Daily_Return_lag1" : round(daily_return_lag1, 4),
        "MA_ratio_lag1"     : round(ma_ratio_lag1, 6),
    }

# ── Step 5: Run all 3 models and show consensus ───────────────
def predict_consensus(dt, rf, lr_pipe, features, inputs):
    X = pd.DataFrame([features])[FEATURES]

    # -- DT prediction ----------------------------------------
    dt_pred    = dt.predict(X)[0]
    dt_proba   = dt.predict_proba(X)[0]
    dt_conf    = max(dt_proba) * 100
    dt_label   = "UP" if dt_pred == 1 else "DOWN"

    # -- RF prediction ----------------------------------------
    rf_pred    = rf.predict(X)[0]
    rf_proba   = rf.predict_proba(X)[0]
    rf_conf    = max(rf_proba) * 100
    rf_label   = "UP" if rf_pred == 1 else "DOWN"

    # -- LR prediction ----------------------------------------
    lr_pred    = lr_pipe.predict(X)[0]
    lr_proba   = lr_pipe.predict_proba(X)[0]
    lr_prob_up = lr_proba[1] * 100   # probability of UP specifically
    lr_label   = "UP" if lr_pred == 1 else "DOWN"

    # -- Consensus --------------------------------------------
    votes    = [dt_pred, rf_pred, lr_pred]
    up_votes = sum(votes)
    dn_votes = 3 - up_votes

    if up_votes == 3:
        consensus     = "UP"
        strength      = "STRONG"
        strength_col  = GREEN
        agreement     = "3/3 models agree"
    elif up_votes == 2:
        consensus     = "UP"
        strength      = "MODERATE"
        strength_col  = YELLOW
        agreement     = "2/3 models agree"
    elif dn_votes == 3:
        consensus     = "DOWN"
        strength      = "STRONG"
        strength_col  = GREEN
        agreement     = "3/3 models agree"
    else:
        consensus     = "DOWN"
        strength      = "MODERATE"
        strength_col  = YELLOW
        agreement     = "2/3 models agree"

    trade_ok = strength in ["STRONG", "MODERATE"]

    # ── Print results ─────────────────────────────────────────
    print(f"\n{BOLD}{'='*58}{RESET}")
    print(f"{BOLD}  FEATURE VALUES USED{RESET}")
    print(f"{'='*58}")
    print(f"  {'Feature':<22} {'Value':>12}")
    print(f"  {'-'*36}")
    for feat, val in features.items():
        star = f" {YELLOW}*{RESET}" \
               if feat in ["RSI", "Volatility_5", "Return_5d"] \
               else ""
        print(f"  {feat:<22} {str(val):>12}{star}")
    print(f"  {YELLOW}* = top 3 most important features{RESET}")

    # Individual model results
    print(f"\n{BOLD}{'='*58}{RESET}")
    print(f"{BOLD}  INDIVIDUAL MODEL PREDICTIONS{RESET}")
    print(f"{'='*58}")

    for label, pred, conf, acc, strength_str, extra in [
        ("Decision Tree",
         dt_label, dt_conf, 53.8,
         "best Up recall (78%)",
         f"confidence: {dt_conf:.1f}%"),

        ("Random Forest",
         rf_label, rf_conf, 52.8,
         "best Down recall (29%)",
         f"confidence: {rf_conf:.1f}%"),

        ("Logistic Regr.",
         lr_label, lr_prob_up, 56.4,
         "best overall (60% Down, 54% Up)",
         f"P(UP) = {lr_prob_up:.1f}%"),
    ]:
        color  = GREEN if pred == "UP" else RED
        icon   = "↑" if pred == "UP" else "↓"
        print(f"\n  {BOLD}{label:<18}{RESET} "
              f"[trained acc: {acc}%]")
        print(f"    Prediction : {color}{BOLD}{pred} {icon}{RESET}")
        print(f"    {extra}")
        print(f"    Strength   : {strength_str}")

    # Consensus result
    cons_color = GREEN if consensus == "UP" else RED
    print(f"\n{BOLD}{strength_col}{'='*58}{RESET}")
    print(f"{BOLD}{strength_col}  CONSENSUS: {consensus} "
          f"({agreement}){RESET}")
    print(f"{BOLD}{strength_col}  Signal strength: {strength}{RESET}")
    print(f"{BOLD}{strength_col}{'='*58}{RESET}")

    if strength == "STRONG":
        print(f"\n  {GREEN}{BOLD}All 3 models agree -- strongest possible signal.{RESET}")
        print(f"  {GREEN}Historical accuracy when all 3 agree is higher{RESET}")
        print(f"  {GREEN}than any single model alone.{RESET}")
        print(f"\n  {GREEN}Proceed to track.py to log and monitor.{RESET}")
    else:
        print(f"\n  {YELLOW}2/3 models agree -- moderate signal.{RESET}")
        disagree = []
        if dt_label != consensus:
            disagree.append(f"Decision Tree says {dt_label}")
        if rf_label != consensus:
            disagree.append(f"Random Forest says {rf_label}")
        if lr_label != consensus:
            disagree.append(f"Logistic Regression says {lr_label}")
        for d in disagree:
            print(f"  {YELLOW}Disagreement: {d}{RESET}")
        print(f"\n  {YELLOW}Proceed to track.py but be cautious.{RESET}")
        print(f"  {YELLOW}Wait for Day 1 signal score >= 70% before trading.{RESET}")

    # Feature signal explanation
    print(f"\n{BOLD}{'='*58}{RESET}")
    print(f"{BOLD}  WHY THE MODELS THINK {consensus}{RESET}")
    print(f"{'='*58}")

    rsi = features["RSI"]
    vol = features["Volatility_5"]
    r5d = features["Return_5d"]
    mar = features["MA_ratio"]
    ppo = features["Price_Position"]

    if rsi < 35:
        print(f"  RSI={rsi:.1f} -- oversold (<35). Bounce expected. {GREEN}+UP{RESET}")
    elif rsi > 65:
        print(f"  RSI={rsi:.1f} -- overbought (>65). Pullback risk. {RED}+DOWN{RESET}")
    else:
        print(f"  RSI={rsi:.1f} -- neutral zone (35-65). No strong signal.")

    if vol < 0.7:
        print(f"  Vol={vol:.2f}  -- calm market. Trend likely continues. {GREEN}+UP{RESET}")
    elif vol > 1.5:
        print(f"  Vol={vol:.2f}  -- high volatility. Uncertain. {RED}+DOWN{RESET}")
    else:
        print(f"  Vol={vol:.2f}  -- normal volatility.")

    if r5d > 2:
        print(f"  5d return={r5d:.2f}% -- strong weekly uptrend. {GREEN}+UP{RESET}")
    elif r5d < -2:
        print(f"  5d return={r5d:.2f}% -- recent weakness. Mean reversion "
              f"possible. {GREEN}+UP{RESET}")
    else:
        print(f"  5d return={r5d:.2f}% -- flat recent trend.")

    if mar > 1.0:
        print(f"  MA_ratio={mar:.3f} -- short-term above long-term. "
              f"{GREEN}+UP{RESET}")
    else:
        print(f"  MA_ratio={mar:.3f} -- short-term below long-term. "
              f"{RED}+DOWN{RESET}")

    if ppo < 0.3:
        print(f"  Price position={ppo:.2f} -- closed near day's low. "
              f"Bears strong today. {RED}+DOWN{RESET}")
    elif ppo > 0.7:
        print(f"  Price position={ppo:.2f} -- closed near day's high. "
              f"Bulls strong today. {GREEN}+UP{RESET}")
    else:
        print(f"  Price position={ppo:.2f} -- closed in middle of range.")

    # LR probability insight
    print(f"\n  {CYAN}LR probability of UP: {lr_prob_up:.1f}%{RESET}")
    if lr_prob_up > 60:
        print(f"  {CYAN}LR is leaning bullish with moderate conviction.{RESET}")
    elif lr_prob_up < 40:
        print(f"  {CYAN}LR is leaning bearish with moderate conviction.{RESET}")
    else:
        print(f"  {CYAN}LR is nearly 50/50 -- market is genuinely uncertain.{RESET}")

    print(f"\n{BOLD}{'='*58}{RESET}")
    print(f"{YELLOW}  DISCLAIMER: Educational only. Models are ~53-56%{RESET}")
    print(f"{YELLOW}  accurate. Always use stop losses. Not financial advice.{RESET}")
    print(f"{BOLD}{'='*58}{RESET}\n")

    return {
        "consensus"  : consensus,
        "strength"   : strength,
        "dt_label"   : dt_label,
        "rf_label"   : rf_label,
        "lr_label"   : lr_label,
        "lr_prob_up" : lr_prob_up,
        "trade_ok"   : trade_ok,
    }

# ── Save to last_prediction.csv for track.py auto-load ────────
def save_last_prediction(inputs, features, result):
    import datetime
    row = {
        "date"          : datetime.date.today().isoformat(),
        "open"          : inputs["open"],
        "high"          : inputs["high"],
        "low"           : inputs["low"],
        "close"         : inputs["close"],
        "rsi"           : features["RSI"],
        "volatility"    : features["Volatility_5"],
        "ma_ratio"      : features["MA_ratio"],
        "price_position": features["Price_Position"],
        "daily_return"  : features["Daily_Return"],
        "prediction"    : result["consensus"],
        "confidence"    : result["lr_prob_up"],
        "dt_vote"       : result["dt_label"],
        "rf_vote"       : result["rf_label"],
        "lr_vote"       : result["lr_label"],
        "strength"      : result["strength"],
        "lr_prob_up"    : result["lr_prob_up"],
    }
    pd.DataFrame([row]).to_csv("last_prediction.csv", index=False)
    print(f"  {GREEN}Saved to last_prediction.csv -- "
          f"track.py will auto-load this!{RESET}")

# ── Main ──────────────────────────────────────────────────────
def main():
    dt, rf, lr_pipe, df = train_all_models()

    while True:
        inputs   = get_inputs()
        features = build_features(inputs, df)
        result   = predict_consensus(dt, rf, lr_pipe, features, inputs)
        save_last_prediction(inputs, features, result)

        print(f"  {BOLD}Next steps:{RESET}")
        print(f"  1. Run track.py --> option 1 to log this prediction")
        print(f"  2. Update daily with track.py --> option 2")
        print(f"  3. Trade signal fires automatically when score >= 70%")

        again = input(f"\n  Predict again? {BLUE}[Y/n]{RESET}: ").strip().lower()
        if again == "n":
            print(f"\n  {GREEN}Goodbye!{RESET}\n")
            break

if __name__ == "__main__":
    main()