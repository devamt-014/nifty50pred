# ============================================================
# trade_signal.py -- Called automatically by track.py
# ============================================================
# DO NOT run this directly.
# track.py calls this when tracking signals are strong enough.
# ============================================================

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def generate_trade(prediction, signal_score,
                   day_number, close, high, low,
                   rsi, volatility, ma_ratio,
                   entry_close):
    """
    Called by track.py with accumulated tracking data.
    Returns full trade decision: BUY/SELL/WAIT/CANCEL.
    """
    warnings  = []
    action    = "WAIT"
    day_range = high - low

    # ── Decision logic ────────────────────────────────────────
    # Based on signal score from track.py + day number

    if signal_score >= 70 and day_number <= 3:
        action = "BUY NOW" if prediction == "UP" else "SELL NOW"
    elif signal_score >= 50 and day_number <= 2:
        action = "WAIT — one more day of confirmation"
    elif day_number >= 4:
        action = "TOO LATE — only 1 day left in window"
        warnings.append(
            "Day 4+ of 5-day window. Not enough time for "
            "trade to play out. Skip this one.")
    else:
        action = "CANCEL — signals too weak"
        warnings.append(
            f"Signal score {signal_score:.0f}% is below 50%. "
            "Prediction likely wrong. Do not trade.")

    # ── Extra warnings ────────────────────────────────────────
    if volatility > 2.0:
        warnings.append(
            f"Volatility = {volatility:.2f} is very high. "
            "Stop loss may be hit by noise.")

    if prediction == "UP" and rsi > 70:
        warnings.append(
            f"RSI = {rsi:.1f} is now overbought. "
            "Upside may be limited from here.")

    if prediction == "DOWN" and rsi < 30:
        warnings.append(
            f"RSI = {rsi:.1f} is now oversold. "
            "Downside may be limited from here.")

    price_pos = (close - low) / day_range if day_range > 0 else 0.5
    if prediction == "UP" and price_pos < 0.3:
        warnings.append(
            f"Price closed near day's low (pos={price_pos:.2f}). "
            "Bears still in control today.")

    # ── Entry, SL, Target calculations ───────────────────────
    if prediction == "UP":
        entry   = round(close + day_range * 0.02, 2)
        sl      = round(low   - day_range * 0.05, 2)
        risk    = round(entry - sl, 2)
        t1      = round(entry + risk * 1.5, 2)
        t2      = round(entry + risk * 2.5, 2)
    else:
        entry   = round(close - day_range * 0.02, 2)
        sl      = round(high  + day_range * 0.05, 2)
        risk    = round(sl - entry, 2)
        t1      = round(entry - risk * 1.5, 2)
        t2      = round(entry - risk * 2.5, 2)

    rr1 = round(abs(t1 - entry) / risk, 2) if risk > 0 else 0
    rr2 = round(abs(t2 - entry) / risk, 2) if risk > 0 else 0

    # Current P&L from entry close
    move     = close - entry_close
    move_pct = round(move / entry_close * 100, 2)

    return {
        "action"     : action,
        "prediction" : prediction,
        "signal_score": signal_score,
        "day_number" : day_number,
        "entry"      : entry,
        "sl"         : sl,
        "t1"         : t1,
        "t2"         : t2,
        "risk"       : risk,
        "rr1"        : rr1,
        "rr2"        : rr2,
        "move_pct"   : move_pct,
        "warnings"   : warnings,
        "price_pos"  : round(price_pos, 2),
    }

def print_trade(sig, close, entry_close, prediction):
    is_buy    = "BUY"    in sig["action"]
    is_sell   = "SELL"   in sig["action"]
    is_wait   = "WAIT"   in sig["action"]
    is_cancel = "CANCEL" in sig["action"] or "LATE" in sig["action"]
    color     = (GREEN  if is_buy   else
                 RED    if is_sell  else
                 YELLOW if is_wait  else
                 RED)

    print(f"\n{BOLD}{color}{'='*58}{RESET}")
    print(f"{BOLD}{color}  TRADE DECISION{RESET}")
    print(f"{BOLD}{color}{'='*58}{RESET}")

    # Warnings first
    if sig["warnings"]:
        print(f"\n  {RED}{BOLD}Warnings:{RESET}")
        for w in sig["warnings"]:
            print(f"  {RED}  ! {w}{RESET}")

    print(f"\n  {BOLD}Action   : {color}{sig['action']}{RESET}")
    print(f"  Signals  : {sig['signal_score']:.0f}% in {prediction} direction")
    print(f"  Day      : {sig['day_number']} of 5")
    print(f"  Since prediction: {sig['move_pct']:+.2f}%")

    if is_cancel:
        print(f"\n  {RED}Do not enter this trade.{RESET}")
        print(f"  {YELLOW}Wait for next predict.py signal.{RESET}\n")
        return

    if is_wait:
        print(f"\n  {YELLOW}Signals not strong enough yet.{RESET}")
        print(f"  {YELLOW}Update track.py again tomorrow.{RESET}")
        print(f"  {YELLOW}If tomorrow's score is 70%+, then trade.{RESET}\n")
        return

    # Full trade card for BUY/SELL
    print(f"\n{BOLD}{'='*58}{RESET}")
    print(f"{BOLD}  TRADE LEVELS{RESET}")
    print(f"{'='*58}")
    print(f"""
  Today's close  : {close:.2f}
  Entry close    : {entry_close:.2f}
  Move so far    : {sig['move_pct']:+.2f}%

  ENTRY          : {color}{BOLD}{sig['entry']:.2f}{RESET}
  Enter ONLY if next day's price crosses this level.
  Do NOT enter at market open blindly.

  STOP LOSS      : {RED}{BOLD}{sig['sl']:.2f}{RESET}
  Exit immediately if price hits this. No hesitation.
  Risk           : {sig['risk']:.2f} points

  TARGET 1       : {GREEN}{BOLD}{sig['t1']:.2f}{RESET}  (R:R = 1:{sig['rr1']}) — book 50% here
  TARGET 2       : {GREEN}{BOLD}{sig['t2']:.2f}{RESET}  (R:R = 1:{sig['rr2']}) — trail rest
""")

    print(f"{BOLD}{'='*58}{RESET}")
    print(f"{BOLD}  HOW TO EXECUTE{RESET}")
    print(f"{'='*58}")

    if is_buy:
        print(f"""
  WHERE  : Zerodha Kite / Groww / Upstox / Angel One
  WHAT   : NIFTYBEES ETF (simplest) or NIFTY Index Fund

  STEPS  :
    1. Open broker app after market opens (9:15 AM)
    2. Search "NIFTYBEES"
    3. Wait for price to cross {sig['entry']:.2f}
    4. Place LIMIT BUY at {sig['entry']:.2f}
    5. Immediately set STOP LOSS at {sig['sl']:.2f}
    6. Set target alert at {sig['t1']:.2f}
    7. Book 50% profit at Target 1
    8. Move stop loss to entry price (risk-free trade)
    9. Let rest run to Target 2
""")
    else:
        print(f"""
  WHERE  : Zerodha Kite / Upstox (F&O segment needed)
  WHAT   : NIFTY Futures (sell) or Buy PUT option

  STEPS  :
    1. Open broker app after market opens (9:15 AM)
    2. Go to F&O → NIFTY Futures
    3. Wait for price to BREAK BELOW {sig['entry']:.2f}
    4. Place SELL order at {sig['entry']:.2f}
    5. Set STOP LOSS BUY at {sig['sl']:.2f}
    6. Book 50% profit at {sig['t1']:.2f}
    7. Trail stop loss for remaining position
""")

    print(f"{BOLD}{'='*58}{RESET}")
    print(f"{BOLD}  POSITION SIZING{RESET}")
    print(f"{'='*58}")
    print(f"""
  Rule: Never risk more than 1% of your capital.

  Your capital     Risk 1%    Units to buy
  Rs  50,000    →  Rs  500  → {int(500/sig['risk']) if sig['risk']>0 else 'N/A'} units
  Rs 1,00,000   →  Rs 1,000 → {int(1000/sig['risk']) if sig['risk']>0 else 'N/A'} units
  Rs 5,00,000   →  Rs 5,000 → {int(5000/sig['risk']) if sig['risk']>0 else 'N/A'} units

  Adjust to YOUR actual capital.
""")

    print(f"  {YELLOW}{'='*54}{RESET}")
    print(f"  {YELLOW}DISCLAIMER: Educational only. ~55% model accuracy.{RESET}")
    print(f"  {YELLOW}Always use stop losses. Never trade borrowed money.{RESET}")
    print(f"  {YELLOW}Consult SEBI-registered advisor for real trading.{RESET}")
    print(f"  {YELLOW}{'='*54}{RESET}\n")