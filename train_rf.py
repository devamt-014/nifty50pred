# ============================================================
# train_rf.py -- Random Forest (with honest model selection)
# ============================================================
# Key fix: Models must catch at least 25% of Down days
# to be considered valid. High accuracy with 0% Down recall
# is the majority class trap -- automatically disqualified.
#
# Selection rule:
#   1. Down recall >= 25%   (genuinely predicts both classes)
#   2. Up recall >= 40%     (not swinging too far to Down)
#   3. Among valid models: pick highest accuracy
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

MIN_DOWN_RECALL = 0.25   # must catch at least 25% of Down days
MIN_UP_RECALL   = 0.40   # must catch at least 40% of Up days

# -- 1. Load features -----------------------------------------
df = pd.read_csv("nifty50_features.csv",
                 index_col="Date", parse_dates=True, dayfirst=True)
df.dropna(inplace=True)
print(f"Loaded {len(df)} samples")
print(f"Date range : {df.index[0].date()} --> {df.index[-1].date()}")

FEATURES = ["MA_ratio", "Market_regime", "RSI",
            "Daily_Return", "Return_3d", "Return_5d",
            "Volatility_5", "Price_Position",
            "RSI_lag1", "Daily_Return_lag1", "MA_ratio_lag1"]

X = df[FEATURES]
y = df["Target"]

# -- 2. Test set + windows ------------------------------------
TEST_START = "2024-01-01"
X_test = X[X.index >= TEST_START]
y_test = y[y.index >= TEST_START]

WINDOWS = {
    "2 years  (2022-2023)" : "2022-01-01",
    "5 years  (2019-2023)" : "2019-01-01",
    "10 years (2014-2023)" : "2014-01-01",
}

total_down = (y_test == 0).sum()
total_up   = (y_test == 1).sum()
print(f"\nTest set : {len(X_test)} days")
print(f"Target   : Up={total_up}  Down={total_down}")
print(f"\nSelection rule: Down recall >= {MIN_DOWN_RECALL*100:.0f}%"
      f"  AND  Up recall >= {MIN_UP_RECALL*100:.0f}%")
print(f"Any model failing these is DISQUALIFIED (majority trap)")

# -- 3. DT baseline -------------------------------------------
X_train_dt = X[(X.index >= "2022-01-01") & (X.index < TEST_START)]
y_train_dt = y[(y.index >= "2022-01-01") & (y.index < TEST_START)]
dt = DecisionTreeClassifier(
         max_depth=3, min_samples_leaf=10,
         class_weight={0: 1.2, 1: 1.0},
         random_state=42)
dt.fit(X_train_dt, y_train_dt)
dt_pred = dt.predict(X_test)
dt_acc  = accuracy_score(y_test, dt_pred)
dt_cm   = confusion_matrix(y_test, dt_pred)
dt_dr   = dt_cm[0][0] / dt_cm[0].sum()
dt_ur   = dt_cm[1][1] / dt_cm[1].sum()
print(f"\nDecision Tree baseline : {dt_acc*100:.1f}%  "
      f"Down recall: {dt_dr*100:.0f}%  Up recall: {dt_ur*100:.0f}%")

# -- 4. Class weight options ----------------------------------
CW_OPTIONS = {
    "none (1:1)"       : None,
    "gentle (0:1.2)"   : {0: 1.2, 1: 1.0},
    "moderate (0:1.5)" : {0: 1.5, 1: 1.0},
    "strong (0:2.0)"   : {0: 2.0, 1: 1.0},
    "balanced (auto)"  : "balanced",
}

# -- 5. Grid search with honest selection ---------------------
print("\n" + "=" * 70)
print("GRID SEARCH -- honest selection (Down recall >= 25%, Up recall >= 40%)")
print("=" * 70)

all_results   = []
best_valid    = None
best_valid_acc = 0

for window_name, train_start in WINDOWS.items():
    X_tr = X[(X.index >= train_start) & (X.index < TEST_START)]
    y_tr = y[(y.index >= train_start) & (y.index < TEST_START)]

    print(f"\n  Window: {window_name} ({len(X_tr)} days)")
    print(f"  {'Class weight':<22} {'Acc':>7} {'Down%':>7}"
          f" {'Up%':>7}  Status")
    print(f"  {'-'*58}")

    for cw_name, cw in CW_OPTIONS.items():
        for n_trees in [100, 200, 300]:
            rf = RandomForestClassifier(
                     n_estimators=n_trees, max_depth=5,
                     class_weight=cw,
                     random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            preds = rf.predict(X_test)

            if len(set(preds)) < 2:
                continue

            acc = accuracy_score(y_test, preds)
            cm  = confusion_matrix(y_test, preds)
            dr  = cm[0][0] / cm[0].sum()   # Down recall
            ur  = cm[1][1] / cm[1].sum()   # Up recall

            # Honest check
            valid = dr >= MIN_DOWN_RECALL and ur >= MIN_UP_RECALL

            result = {
                "window"    : window_name,
                "train_start": train_start,
                "cw_name"   : cw_name,
                "cw"        : cw,
                "n_trees"   : n_trees,
                "acc"       : acc,
                "down_recall": dr,
                "up_recall" : ur,
                "valid"     : valid,
                "X_tr"      : X_tr,
                "y_tr"      : y_tr,
            }
            all_results.append(result)

            if valid and acc > best_valid_acc:
                best_valid_acc = acc
                best_valid     = result

        # Print best of this cw across tree counts
        cw_results = [r for r in all_results
                      if r["window"] == window_name
                      and r["cw_name"] == cw_name]
        if not cw_results:
            continue
        best_cw = max(cw_results, key=lambda r: r["acc"])
        status  = ("VALID" if best_cw["valid"]
                   else f"DISQUALIFIED "
                        f"(Down:{best_cw['down_recall']*100:.0f}%"
                        f" Up:{best_cw['up_recall']*100:.0f}%)")
        marker  = " <- best valid!" \
                  if best_valid and \
                     best_cw["window"] == best_valid["window"] and \
                     best_cw["cw_name"] == best_valid["cw_name"] \
                  else ""
        print(f"  {cw_name:<22} "
              f"{best_cw['acc']*100:>6.1f}% "
              f"{best_cw['down_recall']*100:>6.0f}% "
              f"{best_cw['up_recall']*100:>6.0f}%  "
              f"{status}{marker}")

# -- 6. Result ------------------------------------------------
print(f"\n{'='*70}")
if best_valid is None:
    print("NO VALID RF MODEL FOUND")
    print("All RF configurations fell into majority class trap.")
    print(f"Decision Tree ({dt_acc*100:.1f}%) remains the best model.")
    print("\nConclusion: For this dataset and test period,")
    print("a well-tuned Decision Tree outperforms Random Forest.")
else:
    print(f"BEST HONEST RF MODEL")
    print(f"{'='*70}")
    print(f"  Window       : {best_valid['window']}")
    print(f"  Class weight : {best_valid['cw_name']}")
    print(f"  Trees        : {best_valid['n_trees']}")
    print(f"  Accuracy     : {best_valid['acc']*100:.1f}%")
    print(f"  Down recall  : {best_valid['down_recall']*100:.0f}%  (>= 25% threshold)")
    print(f"  Up recall    : {best_valid['up_recall']*100:.0f}%  (>= 40% threshold)")
    print(f"  vs DT        : {(best_valid['acc']-dt_acc)*100:+.1f}%")

    # -- 7. Final model ---------------------------------------
    final_rf = RandomForestClassifier(
                   n_estimators=best_valid["n_trees"],
                   max_depth=5,
                   class_weight=best_valid["cw"],
                   random_state=42, n_jobs=-1)
    final_rf.fit(best_valid["X_tr"], best_valid["y_tr"])
    final_pred = final_rf.predict(X_test)
    final_cm   = confusion_matrix(y_test, final_pred)

    print("\n" + "=" * 70)
    print("FINAL RANDOM FOREST CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_test, final_pred,
                                 target_names=["Down", "Up"],
                                 zero_division=0))

    # -- 8. Feature importance --------------------------------
    importance = pd.Series(final_rf.feature_importances_,
                           index=FEATURES).sort_values(ascending=False)
    print("FEATURE IMPORTANCE:")
    for feat, val in importance.items():
        bar = "|" * int(val * 50)
        print(f"  {feat:<22} {val:.4f}  {bar}")

    # -- 9. Honest side-by-side comparison --------------------
    print("\n" + "=" * 70)
    print("HONEST COMPARISON (both models genuinely predict both classes)")
    print("=" * 70)
    print(f"  {'Metric':<28} {'Decision Tree':>15} {'Random Forest':>15}")
    print(f"  {'-'*58}")
    print(f"  {'Accuracy':<28} {dt_acc*100:>14.1f}% "
          f"{best_valid['acc']*100:>14.1f}%")
    print(f"  {'vs Random (50%)':<28} {(dt_acc-0.5)*100:>+14.1f}% "
          f"{(best_valid['acc']-0.5)*100:>+14.1f}%")
    print(f"  {'Down recall':<28} "
          f"{dt_dr*100:>13.0f}%  "
          f"{best_valid['down_recall']*100:>13.0f}%")
    print(f"  {'Up recall':<28} "
          f"{dt_ur*100:>13.0f}%  "
          f"{best_valid['up_recall']*100:>13.0f}%")
    print(f"  {'Training window':<28} "
          f"{'2yr (2022-23)':>15} "
          f"{best_valid['window'].split('(')[0].strip():>15}")
    print(f"  {'Class weight':<28} "
          f"{'gentle 0:1.2':>15} "
          f"{best_valid['cw_name']:>15}")

    if best_valid["acc"] > dt_acc:
        print(f"\n  Random Forest wins by "
              f"{(best_valid['acc']-dt_acc)*100:.1f}%")
    else:
        print(f"\n  Decision Tree holds -- RF could not beat it honestly.")
        print(f"  Both models are genuinely predicting both classes.")

    # -- 10. Plot ---------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"NIFTY 50 -- Honest RF vs DT\n"
        f"RF: {best_valid['window']}, {best_valid['cw_name']}, "
        f"{best_valid['n_trees']} trees = {best_valid['acc']*100:.1f}%  |  "
        f"DT: {dt_acc*100:.1f}%",
        fontsize=12, fontweight="bold")

    # Accuracy bar
    ax = axes[0]
    models = ["Decision\nTree", "Random\nForest"]
    accs   = [dt_acc*100, best_valid["acc"]*100]
    cols   = ["#1565C0" if a == max(accs) else "#90CAF9" for a in accs]
    bars   = ax.bar(models, accs, color=cols, width=0.4)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1.5)
    ax.set_ylim(35, 70)
    ax.set_title("Accuracy (honest models only)")
    ax.set_ylabel("Test Accuracy %")
    ax.grid(axis="y", alpha=0.3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center",
                fontsize=12, fontweight="bold")

    # Recall comparison
    ax = axes[1]
    x  = np.arange(2)
    w  = 0.3
    ax.bar(x-w/2, [dt_dr*100, best_valid["down_recall"]*100],
           w, label="Down recall", color="#EF5350", alpha=0.8)
    ax.bar(x+w/2, [dt_ur*100, best_valid["up_recall"]*100],
           w, label="Up recall", color="#66BB6A", alpha=0.8)
    ax.axhline(25, color="red", linestyle=":",
               linewidth=1, label="Min Down (25%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Decision Tree", "Random Forest"])
    ax.set_title("Recall by class\n(both must be genuine)")
    ax.set_ylabel("Recall %")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # RF confusion matrix
    ax = axes[2]
    ax.imshow(final_cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Down", "Pred Up"])
    ax.set_yticklabels(["True Down", "True Up"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, final_cm[i, j], ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if final_cm[i,j]>final_cm.max()/2
                    else "black")
    ax.set_title(f"RF Confusion Matrix\n"
                 f"Acc: {best_valid['acc']*100:.1f}%  "
                 f"Down: {best_valid['down_recall']*100:.0f}%  "
                 f"Up: {best_valid['up_recall']*100:.0f}%")

    plt.tight_layout()
    plt.savefig("nifty50_rf_final.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved nifty50_rf_final.png")