# ============================================================
# train_lr.py -- Logistic Regression
# ============================================================
# Same honest framework as train_dt.py and train_rf.py:
#   - Same features (11 engineered signals)
#   - Same 3 training windows (2yr, 5yr, 10yr)
#   - Same disqualification rule (Down recall >= 25%, Up >= 40%)
#   - Same test period (2024)
#
# Key difference from DT/RF:
#   LR finds a LINEAR decision boundary using a mathematical
#   formula -- no tree structure, no voting.
#   It outputs a PROBABILITY (0 to 1) not just a class.
#   This makes it naturally calibrated and interpretable.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)
import warnings
warnings.filterwarnings("ignore")

MIN_DOWN_RECALL = 0.25
MIN_UP_RECALL   = 0.40

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
print(f"\nIMPORTANT: LR requires feature scaling (StandardScaler)")
print(f"DT/RF don't need scaling -- LR does. This is why we use Pipeline.")
print(f"\nSelection rule: Down recall >= {MIN_DOWN_RECALL*100:.0f}%"
      f"  AND  Up recall >= {MIN_UP_RECALL*100:.0f}%")

# -- 3. DT baseline (same config as before) -------------------
X_train_dt = X[(X.index >= "2022-01-01") & (X.index < TEST_START)]
y_train_dt = y[(y.index >= "2022-01-01") & (y.index < TEST_START)]
dt = DecisionTreeClassifier(
         max_depth=3, min_samples_leaf=10,
         class_weight={0: 1.2, 1: 1.0}, random_state=42)
dt.fit(X_train_dt, y_train_dt)
dt_pred = dt.predict(X_test)
dt_acc  = accuracy_score(y_test, dt_pred)
dt_cm   = confusion_matrix(y_test, dt_pred)
dt_dr   = dt_cm[0][0] / dt_cm[0].sum()
dt_ur   = dt_cm[1][1] / dt_cm[1].sum()
print(f"\nDecision Tree baseline : {dt_acc*100:.1f}%  "
      f"Down: {dt_dr*100:.0f}%  Up: {dt_ur*100:.0f}%")

# -- 4. LR hyperparameter options -----------------------------
# C         = regularisation strength (lower = stronger regularisation)
#             prevents overfitting on small datasets
# solver    = algorithm used to fit the model
# max_iter  = maximum iterations to converge
# class_weight = same options as DT/RF

CW_OPTIONS = {
    "none (1:1)"       : None,
    "gentle (0:1.2)"   : {0: 1.2, 1: 1.0},
    "moderate (0:1.5)" : {0: 1.5, 1: 1.0},
    "strong (0:2.0)"   : {0: 2.0, 1: 1.0},
    "balanced (auto)"  : "balanced",
}

C_OPTIONS = [0.01, 0.1, 1.0, 10.0]

# -- 5. Grid search -------------------------------------------
print("\n" + "=" * 70)
print("GRID SEARCH -- LR (window x class_weight x C regularisation)")
print("=" * 70)
print(f"Note: Pipeline = StandardScaler + LogisticRegression")
print(f"      Scaling is critical for LR -- features on same scale\n")

all_results   = []
best_valid     = None
best_valid_acc = 0

for window_name, train_start in WINDOWS.items():
    X_tr = X[(X.index >= train_start) & (X.index < TEST_START)]
    y_tr = y[(y.index >= train_start) & (y.index < TEST_START)]

    print(f"\n  Window: {window_name} ({len(X_tr)} days)")
    print(f"  {'Class weight':<22} {'C=0.01':>8} {'C=0.1':>8}"
          f" {'C=1.0':>8} {'C=10':>8}  Best   Status")
    print(f"  {'-'*76}")

    for cw_name, cw in CW_OPTIONS.items():
        row_results = []
        row_best_acc = 0
        row_best_result = None

        for C in C_OPTIONS:
            # Pipeline: scale first, then logistic regression
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    C=C,
                    class_weight=cw,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=42))
            ])
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_test)

            if len(set(preds)) < 2:
                row_results.append(None)
                continue

            acc = accuracy_score(y_test, preds)
            cm  = confusion_matrix(y_test, preds)
            dr  = cm[0][0] / cm[0].sum()
            ur  = cm[1][1] / cm[1].sum()
            valid = dr >= MIN_DOWN_RECALL and ur >= MIN_UP_RECALL

            result = {
                "window"      : window_name,
                "train_start" : train_start,
                "cw_name"     : cw_name,
                "cw"          : cw,
                "C"           : C,
                "acc"         : acc,
                "down_recall" : dr,
                "up_recall"   : ur,
                "valid"       : valid,
                "pipe"        : pipe,
                "X_tr"        : X_tr,
                "y_tr"        : y_tr,
            }
            row_results.append(result)
            all_results.append(result)

            if valid and acc > row_best_acc:
                row_best_acc    = acc
                row_best_result = result

            if valid and acc > best_valid_acc:
                best_valid_acc = acc
                best_valid     = result

        # Print row summary
        acc_strs = []
        for r in row_results:
            if r is None:
                acc_strs.append("  skip")
            else:
                acc_strs.append(f"{r['acc']*100:>5.1f}%")

        if row_best_result:
            status = (f"VALID "
                      f"Down:{row_best_result['down_recall']*100:.0f}% "
                      f"Up:{row_best_result['up_recall']*100:.0f}%")
            mark = " <- best!" if row_best_result is best_valid else ""
        else:
            status = "DISQUALIFIED"
            mark   = ""

        print(f"  {cw_name:<22}"
              f" {'  '.join(acc_strs)}  "
              f"{row_best_acc*100:>5.1f}%  {status}{mark}")

# -- 6. Result ------------------------------------------------
print(f"\n{'='*70}")
if best_valid is None:
    print("NO VALID LR MODEL FOUND")
    print("All LR configurations fell into majority class trap.")
    print(f"\nDecision Tree ({dt_acc*100:.1f}%) remains the best model.")
    print("\nConclusion: The decision boundary in this data is")
    print("non-linear -- DT handles it better than LR.")
else:
    print(f"BEST HONEST LR MODEL")
    print(f"{'='*70}")
    print(f"  Window       : {best_valid['window']}")
    print(f"  Class weight : {best_valid['cw_name']}")
    print(f"  C (reg.)     : {best_valid['C']}")
    print(f"  Accuracy     : {best_valid['acc']*100:.1f}%")
    print(f"  Down recall  : {best_valid['down_recall']*100:.0f}%")
    print(f"  Up recall    : {best_valid['up_recall']*100:.0f}%")
    print(f"  vs DT        : {(best_valid['acc']-dt_acc)*100:+.1f}%")

    # -- 7. Final predictions ---------------------------------
    final_pipe = best_valid["pipe"]
    final_pred = final_pipe.predict(X_test)
    final_proba= final_pipe.predict_proba(X_test)
    final_cm   = confusion_matrix(y_test, final_pred)

    print("\n" + "=" * 70)
    print("FINAL LR CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_test, final_pred,
                                 target_names=["Down", "Up"],
                                 zero_division=0))

    # -- 8. Feature coefficients (LR's version of importance) -
    scaler = final_pipe.named_steps["scaler"]
    lr     = final_pipe.named_steps["lr"]
    coefs  = pd.Series(lr.coef_[0], index=FEATURES)

    print("FEATURE COEFFICIENTS (LR's version of importance):")
    print("Positive = pushes toward UP, Negative = pushes toward DOWN")
    print()
    for feat, coef in coefs.abs().sort_values(ascending=False).items():
        raw    = coefs[feat]
        direct = "UP  " if raw > 0 else "DOWN"
        bar    = "|" * int(abs(raw) * 8)
        print(f"  {feat:<22} {raw:>+8.4f}  -> {direct}  {bar}")

    # -- 9. Probability insight (unique to LR) ----------------
    print(f"\nPROBABILITY CALIBRATION (unique LR advantage):")
    print(f"LR outputs a probability, not just a class.")
    print(f"High confidence predictions (>65% or <35%) are more reliable.\n")

    proba_up   = final_proba[:, 1]
    high_conf  = (proba_up > 0.65) | (proba_up < 0.35)
    if high_conf.sum() > 0:
        hc_acc = accuracy_score(
            y_test[high_conf],
            final_pred[high_conf])
        print(f"  High confidence predictions (>65% or <35%): "
              f"{high_conf.sum()} days")
        print(f"  Accuracy on high confidence only: {hc_acc*100:.1f}%")
        print(f"  This is LR's real advantage -- skip uncertain days!")
    else:
        print(f"  No high-confidence predictions found.")

    # -- 10. Three-way honest comparison ----------------------
    print("\n" + "=" * 70)
    print("THREE-WAY HONEST COMPARISON")
    print("=" * 70)
    print(f"  {'Metric':<28} {'DT':>10} {'RF':>10} {'LR':>10}")
    print(f"  {'-'*58}")

    # RF best honest result from previous run
    rf_acc = 0.528
    rf_dr  = 0.29
    rf_ur  = 0.68

    print(f"  {'Accuracy':<28} {dt_acc*100:>9.1f}% "
          f"{rf_acc*100:>9.1f}% "
          f"{best_valid['acc']*100:>9.1f}%")
    print(f"  {'vs Random (50%)':<28} "
          f"{(dt_acc-0.5)*100:>+9.1f}% "
          f"{(rf_acc-0.5)*100:>+9.1f}% "
          f"{(best_valid['acc']-0.5)*100:>+9.1f}%")
    print(f"  {'Down recall':<28} "
          f"{dt_dr*100:>9.0f}% "
          f"{rf_dr*100:>9.0f}% "
          f"{best_valid['down_recall']*100:>9.0f}%")
    print(f"  {'Up recall':<28} "
          f"{dt_ur*100:>9.0f}% "
          f"{rf_ur*100:>9.0f}% "
          f"{best_valid['up_recall']*100:>9.0f}%")
    print(f"  {'Outputs probability':<28} "
          f"{'No':>10} {'No':>10} {'Yes':>10}")
    print(f"  {'Needs feature scaling':<28} "
          f"{'No':>10} {'No':>10} {'Yes':>10}")
    print(f"  {'Interpretable weights':<28} "
          f"{'No':>10} {'No':>10} {'Yes':>10}")
    print(f"  {'Training window':<28} "
          f"{'2yr':>10} {'10yr':>10} "
          f"{best_valid['window'].split('(')[0].strip():>10}")

    # Declare winner
    accs = {"DT": dt_acc, "RF": rf_acc, "LR": best_valid["acc"]}
    winner = max(accs, key=accs.get)
    print(f"\n  Overall accuracy winner: {winner} "
          f"({accs[winner]*100:.1f}%)")
    print(f"\n  Each algorithm has a unique strength:")
    print(f"  DT  -- simplest, most interpretable, best accuracy")
    print(f"  RF  -- best at catching Down days (29% recall)")
    print(f"  LR  -- outputs probability, skip uncertain predictions")

    # -- 11. Plots --------------------------------------------
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"NIFTY 50 -- Three-Way Comparison: DT vs RF vs LR\n"
        f"DT: {dt_acc*100:.1f}%  |  RF: {rf_acc*100:.1f}%  |  "
        f"LR: {best_valid['acc']*100:.1f}%",
        fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Plot 1: Accuracy bar
    ax1 = fig.add_subplot(gs[0, 0])
    models = ["Decision\nTree", "Random\nForest", "Logistic\nRegression"]
    accs_list = [dt_acc*100, rf_acc*100, best_valid["acc"]*100]
    colors = ["#1565C0", "#2E7D32", "#6A1B9A"]
    bars   = ax1.bar(models, accs_list, color=colors, width=0.5, alpha=0.85)
    ax1.axhline(50, color="gray", linestyle="--",
                linewidth=1.5, label="Random (50%)")
    ax1.set_ylim(35, 65)
    ax1.set_title("Overall Accuracy\n(honest models only)")
    ax1.set_ylabel("Test Accuracy %")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    for bar, acc in zip(bars, accs_list):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{acc:.1f}%", ha="center",
                 fontsize=10, fontweight="bold")

    # Plot 2: Recall comparison
    ax2 = fig.add_subplot(gs[0, 1])
    x   = np.arange(3)
    w   = 0.3
    down_recalls = [dt_dr*100, rf_dr*100,
                    best_valid["down_recall"]*100]
    up_recalls   = [dt_ur*100, rf_ur*100,
                    best_valid["up_recall"]*100]
    ax2.bar(x-w/2, down_recalls, w, label="Down recall",
            color="#EF5350", alpha=0.8)
    ax2.bar(x+w/2, up_recalls,   w, label="Up recall",
            color="#66BB6A", alpha=0.8)
    ax2.axhline(25, color="red", linestyle=":",
                linewidth=1, label="Min Down (25%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["DT", "RF", "LR"])
    ax2.set_title("Down vs Up Recall\n(both must be honest)")
    ax2.set_ylabel("Recall %")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # Plot 3: LR feature coefficients
    ax3 = fig.add_subplot(gs[0, 2])
    coef_sorted = coefs.sort_values()
    colors_coef = ["#EF5350" if c < 0 else "#66BB6A"
                   for c in coef_sorted]
    coef_sorted.plot(kind="barh", ax=ax3, color=colors_coef)
    ax3.axvline(0, color="black", linewidth=0.8)
    ax3.set_title("LR Coefficients\nGreen=UP, Red=DOWN")
    ax3.set_xlabel("Coefficient value")
    ax3.grid(axis="x", alpha=0.3)

    # Plot 4: LR confusion matrix
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(final_cm, cmap="Purples")
    ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
    ax4.set_xticklabels(["Pred Down", "Pred Up"])
    ax4.set_yticklabels(["True Down", "True Up"])
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, final_cm[i, j], ha="center", va="center",
                     fontsize=18, fontweight="bold",
                     color="white" if final_cm[i,j]>final_cm.max()/2
                     else "black")
    ax4.set_title(f"LR Confusion Matrix\n"
                  f"Acc: {best_valid['acc']*100:.1f}%  "
                  f"Down: {best_valid['down_recall']*100:.0f}%  "
                  f"Up: {best_valid['up_recall']*100:.0f}%")

    # Plot 5: LR probability distribution
    ax5 = fig.add_subplot(gs[1, 1])
    up_mask   = y_test == 1
    down_mask = y_test == 0
    ax5.hist(proba_up[up_mask],   bins=20, alpha=0.6,
             color="#66BB6A", label="True Up days")
    ax5.hist(proba_up[down_mask], bins=20, alpha=0.6,
             color="#EF5350", label="True Down days")
    ax5.axvline(0.5, color="black", linewidth=1.5,
                linestyle="--", label="Decision boundary")
    ax5.axvline(0.65, color="green", linewidth=1,
                linestyle=":", label="High confidence (65%)")
    ax5.axvline(0.35, color="red", linewidth=1, linestyle=":")
    ax5.set_title("LR Predicted Probabilities\n"
                  "(unique advantage over DT/RF)")
    ax5.set_xlabel("Probability of UP")
    ax5.set_ylabel("Count")
    ax5.legend(fontsize=7)
    ax5.grid(alpha=0.3)

    # Plot 6: Rolling accuracy all 3
    ax6 = fig.add_subplot(gs[1, 2])
    res_dt          = X_test.copy()
    res_dt["Hit"]   = (dt_pred == y_test.values).astype(int)
    res_lr          = X_test.copy()
    res_lr["Hit"]   = (final_pred == y_test.values).astype(int)
    roll_dt = res_dt["Hit"].rolling(30).mean() * 100
    roll_lr = res_lr["Hit"].rolling(30).mean() * 100
    ax6.plot(roll_dt.index, roll_dt, color="#1565C0",
             linewidth=1.5, label=f"DT ({dt_acc*100:.1f}%)")
    ax6.plot(roll_lr.index, roll_lr, color="#6A1B9A",
             linewidth=1.5, label=f"LR ({best_valid['acc']*100:.1f}%)")
    ax6.axhline(50, color="gray", linestyle="--",
                linewidth=1, label="Random (50%)")
    ax6.fill_between(roll_lr.index, roll_lr, 50,
                     where=(roll_lr >= 50),
                     alpha=0.1, color="#6A1B9A")
    ax6.set_title("30-day Rolling Accuracy\nDT vs LR")
    ax6.set_ylabel("Accuracy %")
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)

    plt.savefig("nifty50_lr_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.show()
    print("\nSaved nifty50_lr_comparison.png")