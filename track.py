# ============================================================
# track.py -- NIFTY 50 Prediction Tracker + Trade Trigger
# ============================================================
# CORRECT WORKFLOW:
#   Day 0 : python predict.py         --> gets UP/DOWN
#           python track.py --> option 1  --> logs the prediction
#   Day 1 : python track.py --> option 2  --> update Friday data
#           --> if signals strong: trade_signal.py fires automatically
#   Day 2 : python track.py --> option 2  --> update Monday data
#           --> if signals now strong: trade_signal.py fires
#   Day 3+ : same -- track until signal confirms or cancels
#
# trade_signal.py is NEVER run directly -- track.py calls it.
# ============================================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from trade_signal import generate_trade, print_trade

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

LOG_FILE = "predictions_log.csv"

def banner(title, color=BLUE):
    print(f"\n{BOLD}{color}{'='*58}{RESET}")
    print(f"{BOLD}{color}  {title}{RESET}")
    print(f"{BOLD}{color}{'='*58}{RESET}")

def ask_float(prompt, example, low=0, high=999999):
    while True:
        try:
            val = float(input(
                f"  {prompt} {BLUE}(e.g. {example}){RESET}: "))
            if low <= val <= high:
                return val
            print(f"  {RED}Value must be between {low} and {high}{RESET}")
        except ValueError:
            print(f"  {RED}Please enter a valid number{RESET}")

def load_log():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE, parse_dates=["date"])
    cols = ["prediction_id", "date", "day_number",
            "open", "high", "low", "close",
            "rsi", "volatility", "ma_ratio",
            "price_position", "daily_return",
            "prediction", "entry_close",
            "signal_score", "verdict", "note"]
    return pd.DataFrame(columns=cols)

def save_log(df):
    df.to_csv(LOG_FILE, index=False)

def score_signals(close, open_, high, low,
                  rsi, prev_rsi, price_pos,
                  daily_return, entry_close,
                  prediction):
    signals = []

    signals.append({
        "name"   : "Price vs entry",
        "bullish": close > entry_close,
        "value"  : f"{close:.2f} vs {entry_close:.2f}",
        "note"   : ("Above entry -- moving right" if close > entry_close
                    else "Below entry -- moving wrong way"),
    })
    rsi_rising = rsi > prev_rsi if prev_rsi else True
    signals.append({
        "name"   : "RSI direction",
        "bullish": rsi_rising,
        "value"  : f"{rsi:.1f} (was {prev_rsi:.1f})" if prev_rsi else f"{rsi:.1f}",
        "note"   : ("RSI rising -- momentum building"
                    if rsi_rising else "RSI falling -- momentum fading"),
    })
    rsi_ok = (rsi > 30 and prediction == "UP") or \
             (rsi < 70 and prediction == "DOWN")
    signals.append({
        "name"   : "RSI level",
        "bullish": rsi_ok,
        "value"  : f"{rsi:.1f}",
        "note"   : ("RSI in healthy range for this trade"
                    if rsi_ok else "RSI at extreme -- momentum exhausting"),
    })
    strong_close = (price_pos > 0.5 and prediction == "UP") or \
                   (price_pos < 0.5 and prediction == "DOWN")
    signals.append({
        "name"   : "Price position",
        "bullish": strong_close,
        "value"  : f"{price_pos:.2f}",
        "note"   : ("Closed in favourable half of range"
                    if strong_close else "Closed in unfavourable half"),
    })
    pos_return = (daily_return > 0 and prediction == "UP") or \
                 (daily_return < 0 and prediction == "DOWN")
    signals.append({
        "name"   : "Daily return",
        "bullish": pos_return,
        "value"  : f"{daily_return:+.2f}%",
        "note"   : ("Moving in predicted direction"
                    if pos_return else "Moving against prediction"),
    })
    green_candle = close > open_
    candle_ok    = (green_candle and prediction == "UP") or \
                   (not green_candle and prediction == "DOWN")
    signals.append({
        "name"   : "Candle",
        "bullish": candle_ok,
        "value"  : "Green" if green_candle else "Red",
        "note"   : ("Candle confirms prediction"
                    if candle_ok else "Candle contradicts prediction"),
    })
    range_ok = (price_pos > 0.6 and prediction == "UP") or \
               (price_pos < 0.4 and prediction == "DOWN")
    signals.append({
        "name"   : "Range structure",
        "bullish": range_ok,
        "value"  : f"H:{high:.0f} L:{low:.0f}",
        "note"   : ("Strong range structure for trade"
                    if range_ok else "Weak range structure"),
    })

    bullish_count = sum(1 for s in signals if s["bullish"])
    score = bullish_count / len(signals) * 100
    return signals, round(score, 1), len(signals)

# ── Option 1: Log new prediction ─────────────────────────────
def new_prediction(log):
    banner("Log New Prediction (Day 0)")

    from datetime import date

    # Auto-load from last_prediction.csv if available
    auto = {}
    if os.path.exists("last_prediction.csv"):
        try:
            lp   = pd.read_csv("last_prediction.csv")
            auto = lp.iloc[0].to_dict()
            print(f"\n  {GREEN}Found last_prediction.csv from predict.py!{RESET}")
            print(f"  {GREEN}Auto-loading: {auto['prediction']} "
                  f"({auto['confidence']:.1f}% confidence, "
                  f"close={auto['close']}){RESET}")
            use_auto = input(
                f"  Use this data? {BLUE}[Y/n]{RESET}: "
            ).strip().lower()
            if use_auto == "n":
                auto = {}
        except Exception:
            auto = {}

    if not auto:
        print(f"\n  {YELLOW}Enter today's data manually.{RESET}\n")

    today      = date.today().isoformat()
    entry_date = auto.get("date", "")
    if not entry_date:
        entry_date = input(
            f"  Prediction date {BLUE}(Enter = today {today}){RESET}: "
        ).strip() or today

    pred_id = (f"P{len(log['prediction_id'].unique())+1:03d}"
               if len(log) > 0 else "P001")

    if auto:
        open_ = float(auto["open"])
        high  = float(auto["high"])
        low_  = float(auto["low"])
        close = float(auto["close"])
        rsi   = float(auto["rsi"])
        vol   = float(auto["volatility"])
        mar   = float(auto["ma_ratio"])
        ppos  = float(auto["price_position"])
        dret  = float(auto["daily_return"])
        pred  = str(auto["prediction"])
        conf  = float(auto["confidence"])
        print(f"  All values loaded from last_prediction.csv.")
    else:
        open_  = ask_float("Today's Open",         "23197.75")
        high   = ask_float("Today's High",         "23378.70", low=open_)
        low_   = ask_float("Today's Low",          "22930.35", high=high)
        close  = ask_float("Today's Close",        "23002.15", low=low_, high=high)
        rsi    = ask_float("Today's RSI",          "30.7",     low=0, high=100)
        vol    = ask_float("Today's Volatility_5", "0.55",     low=0, high=20)
        mar    = ask_float("Today's MA_ratio",     "0.979",    low=0.5, high=1.5)
        ppos   = (close - low_) / (high - low_) if high != low_ else 0.5
        dret   = round((close - open_) / open_ * 100, 4)
        pred   = ""
        while pred not in ["UP", "DOWN"]:
            pred = input(
                f"\n  Model prediction? {BLUE}[UP/DOWN]{RESET}: "
            ).strip().upper()
        conf = ask_float("Model confidence %", "53.1", low=0, high=100)

    new_row = {
        "prediction_id": pred_id,
        "date"         : entry_date,
        "day_number"   : 0,
        "open"         : open_, "high": high,
        "low"          : low_,  "close": close,
        "rsi"          : rsi,   "volatility": vol,
        "ma_ratio"     : mar,   "price_position": round(ppos, 4),
        "daily_return" : dret,
        "prediction"   : pred,
        "entry_close"  : close,
        "signal_score" : conf,
        "verdict"      : "Logged",
        "note"         : f"Day 0 confidence {conf:.1f}%",
    }
    log = pd.concat([log, pd.DataFrame([new_row])],
                    ignore_index=True)
    save_log(log)

    print(f"\n  {GREEN}Saved {pred_id}: {pred} "
          f"(entry={close}, confidence={conf:.1f}%){RESET}")
    print(f"  {YELLOW}Run track.py option 2 after tomorrow's close.{RESET}")
    return log

# ── Option 2: Daily update + auto trade signal ────────────────
def daily_update(log):
    banner("Daily Update")

    active   = log[log["day_number"] < 5]
    pred_ids = active["prediction_id"].unique()

    if len(pred_ids) == 0:
        print(f"\n  {YELLOW}No active predictions. Start one with option 1.{RESET}")
        return log

    print(f"\n  Active: {', '.join(pred_ids)}")
    pred_id = input(
        f"  Which to update? {BLUE}[{pred_ids[0]}]{RESET}: ").strip()
    if not pred_id:
        pred_id = pred_ids[0]

    p_rows      = log[log["prediction_id"] == pred_id].copy()
    entry_row   = p_rows[p_rows["day_number"] == 0].iloc[0]
    last_row    = p_rows.iloc[-1]
    last_day    = int(p_rows["day_number"].max())
    next_day    = last_day + 1

    if next_day > 5:
        print(f"\n  {GREEN}{pred_id} is already complete.{RESET}")
        return log

    entry_close = float(entry_row["entry_close"])
    prediction  = entry_row["prediction"]
    prev_rsi    = float(last_row["rsi"])

    print(f"\n  {BOLD}Tracking: {prediction} from {entry_close} "
          f"(Day {next_day}/5){RESET}\n")

    from datetime import date
    today = date.today().isoformat()
    edate = input(
        f"  Today's date {BLUE}(Enter={today}){RESET}: ").strip()
    if not edate:
        edate = today

    open_  = ask_float("Today's Open",         "23110.00")
    high   = ask_float("Today's High",         "23345.00", low=open_)
    low_   = ask_float("Today's Low",          "23067.00", high=high)
    close  = ask_float("Today's Close",        "23114.00", low=low_, high=high)
    rsi    = ask_float("Today's RSI",          "31.8",     low=0, high=100)
    vol    = ask_float("Today's Volatility_5", "0.55",     low=0, high=20)
    mar    = ask_float("Today's MA_ratio",     "0.979",    low=0.5, high=1.5)

    ppos   = (close - low_) / (high - low_) if high != low_ else 0.5
    dret   = round((close - open_) / open_ * 100, 4)

    # Score signals
    signals, score, total = score_signals(
        close, open_, high, low_, rsi, prev_rsi,
        ppos, dret, entry_close, prediction)

    # Print signal dashboard
    print(f"\n{BOLD}  SIGNAL DASHBOARD -- Day {next_day}/5{RESET}")
    print(f"  {'Signal':<20} {'Value':>18}  Status")
    print(f"  {'-'*56}")
    for s in signals:
        icon = f"{GREEN}+{RESET}" if s["bullish"] else f"{RED}-{RESET}"
        print(f"  {s['name']:<20} {s['value']:>18}  "
              f"[{icon}] {s['note']}")

    # Verdict
    print(f"\n  Score: {score:.0f}% signals "
          f"in {prediction} direction")

    if score >= 70:
        verdict = f"{GREEN}STRONG{RESET}"
    elif score >= 50:
        verdict = f"{YELLOW}MODERATE{RESET}"
    else:
        verdict = f"{RED}WEAK{RESET}"
    print(f"  Verdict: {verdict}")

    # Price vs entry
    diff     = close - entry_close
    diff_pct = round(diff / entry_close * 100, 2)
    dir_ok   = (close > entry_close and prediction == "UP") or \
               (close < entry_close and prediction == "DOWN")
    print(f"\n  Price vs entry: {close:.2f} vs {entry_close:.2f} "
          f"({diff_pct:+.2f}%)")
    if dir_ok:
        print(f"  {GREEN}Moving in predicted direction{RESET}")
    else:
        print(f"  {RED}Moving against prediction{RESET}")

    # Save row
    new_row = {
        "prediction_id" : pred_id,
        "date"          : edate,
        "day_number"    : next_day,
        "open"          : open_, "high": high,
        "low"           : low_,  "close": close,
        "rsi"           : rsi,   "volatility": vol,
        "ma_ratio"      : mar,   "price_position": round(ppos, 4),
        "daily_return"  : dret,
        "prediction"    : prediction,
        "entry_close"   : entry_close,
        "signal_score"  : score,
        "verdict"       : ("STRONG" if score >= 70
                           else "MODERATE" if score >= 50
                           else "WEAK"),
        "note"          : f"Day {next_day} update",
    }
    log = pd.concat([log, pd.DataFrame([new_row])],
                    ignore_index=True)
    save_log(log)

    # ── AUTO TRIGGER trade_signal ─────────────────────────────
    print(f"\n{BOLD}{'='*58}{RESET}")
    print(f"{BOLD}  TRADE DECISION{RESET}")
    print(f"{'='*58}")

    trade = generate_trade(
        prediction   = prediction,
        signal_score = score,
        day_number   = next_day,
        close        = close,
        high         = high,
        low          = low_,
        rsi          = rsi,
        volatility   = vol,
        ma_ratio     = mar,
        entry_close  = entry_close,
    )
    print_trade(trade, close, entry_close, prediction)

    # ── Day 5 final verdict ───────────────────────────────────
    if next_day == 5:
        print(f"\n{BOLD}{'='*58}{RESET}")
        print(f"{BOLD}  FINAL VERDICT (5 days complete){RESET}")
        print(f"{'='*58}")
        if dir_ok:
            print(f"\n  {GREEN}{BOLD}MODEL WAS CORRECT!{RESET}")
            print(f"  {GREEN}Predicted {prediction}. "
                  f"NIFTY moved {diff_pct:+.2f}% in right direction.{RESET}")
        else:
            print(f"\n  {RED}{BOLD}MODEL WAS WRONG.{RESET}")
            print(f"  {RED}Predicted {prediction} but NIFTY went "
                  f"{'UP' if close > entry_close else 'DOWN'} "
                  f"({diff_pct:+.2f}%).{RESET}")
        print(f"\n  {YELLOW}Run predict.py to start a new prediction.{RESET}")

    return log

# ── Option 3: History ────────────────────────────────────────
def view_history(log):
    banner("Prediction History")
    if len(log) == 0:
        print(f"\n  {YELLOW}No predictions yet.{RESET}")
        return
    for pid in log["prediction_id"].unique():
        rows  = log[log["prediction_id"] == pid]
        entry = rows[rows["day_number"] == 0].iloc[0]
        last  = rows.iloc[-1]
        ec    = float(entry["entry_close"])
        lc    = float(last["close"])
        pct   = round((lc - ec) / ec * 100, 2)
        days  = int(last["day_number"])
        pred  = entry["prediction"]

        if days == 5:
            ok = ((pred == "UP"   and lc > ec) or
                  (pred == "DOWN" and lc < ec))
            res = f"{GREEN}CORRECT{RESET}" if ok else f"{RED}WRONG{RESET}"
        else:
            score = float(last["signal_score"])
            col   = GREEN if score>=70 else YELLOW if score>=50 else RED
            res   = f"{col}Day {days}/5 — {score:.0f}%{RESET}"

        print(f"\n  {BOLD}{pid}{RESET}  |  {pred}  |  "
              f"Entry:{ec}  Current:{lc} ({pct:+.2f}%)  |  {res}")

# ── Option 4: Accuracy ───────────────────────────────────────
def accuracy_summary(log):
    banner("Model Accuracy Tracker")
    done = []
    for pid in log["prediction_id"].unique():
        rows = log[log["prediction_id"] == pid]
        if int(rows["day_number"].max()) < 5:
            continue
        entry = rows[rows["day_number"] == 0].iloc[0]
        final = rows[rows["day_number"] == 5].iloc[0]
        ec    = float(entry["entry_close"])
        fc    = float(final["close"])
        pred  = entry["prediction"]
        ok    = ((pred == "UP" and fc > ec) or
                 (pred == "DOWN" and fc < ec))
        done.append(ok)

    if not done:
        print(f"\n  {YELLOW}No completed predictions yet.{RESET}")
        return

    total   = len(done)
    correct = sum(done)
    acc     = correct / total * 100
    print(f"\n  Completed : {total}")
    print(f"  Correct   : {correct}")
    print(f"  Accuracy  : {acc:.1f}%")
    print(f"  Baseline  : 50.0% (random)")
    if acc > 55:
        print(f"  {GREEN}Model working well on live data!{RESET}")
    elif acc > 50:
        print(f"  {YELLOW}Slight edge — need more samples to confirm.{RESET}")
    else:
        print(f"  {RED}Below 50% — model struggling. Review features.{RESET}")

# ── Main ──────────────────────────────────────────────────────
def main():
    log = load_log()
    banner("NIFTY 50 Prediction Tracker")
    print(f"\n  {YELLOW}Workflow: predict.py → track.py → trade auto-fires{RESET}")

    while True:
        print(f"\n  {BOLD}Menu:{RESET}")
        print(f"  1. Log new prediction  (after predict.py)")
        print(f"  2. Update today's data (run daily after 3:30 PM)")
        print(f"  3. View history")
        print(f"  4. Accuracy summary")
        print(f"  5. Exit")

        c = input(f"\n  Choice {BLUE}[1-5]{RESET}: ").strip()
        if   c == "1": log = new_prediction(log)
        elif c == "2": log = daily_update(log)
        elif c == "3": view_history(log)
        elif c == "4": accuracy_summary(log)
        elif c == "5":
            print(f"\n  {GREEN}Goodbye!{RESET}\n")
            break
        else:
            print(f"  {RED}Enter 1-5{RESET}")

if __name__ == "__main__":
    main()