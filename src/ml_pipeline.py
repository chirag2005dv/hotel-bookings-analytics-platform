"""
ml_pipeline.py
==============
Phase 5 — Feature Engineering & Machine Learning
Hotel Bookings — Cancellation Prediction

Objective
---------
Binary classification: predict whether a booking will be canceled
(is_canceled = 1) at the time of booking — before check-in occurs.

Models
------
  1. Logistic Regression  — interpretable baseline
  2. Random Forest        — primary model (handles non-linearity, mixed types)
  3. Gradient Boosting    — secondary model (scikit-learn HistGradientBoosting,
                            chosen over GradientBoostingClassifier for speed
                            and native categorical support)

Leakage Controls
----------------
  Excluded: reservation_status, reservation_status_date — these are
            assigned AFTER the outcome is known (post-booking).
  Excluded: flag_adr_outlier, flag_long_lead_time, flag_zero_adr —
            engineered from adr/lead_time which are already included.
  Excluded: revenue_estimate — derived from adr × total_nights (leaks ADR).
  Excluded: arrival_date (raw datetime) — encoded numerically instead.

Outputs
-------
  models/                      — directory for all model artifacts
  models/logistic_regression.pkl
  models/random_forest.pkl
  models/gradient_boosting.pkl
  models/best_model.pkl        — copy of best model by ROC-AUC
  models/feature_names.txt     — ordered feature list used during training
  plots/ML_confusion_matrix.png
  plots/ML_roc_curves.png
  plots/ML_feature_importance.png
  ml_report.txt                — structured evaluation report
"""

import warnings
import textwrap
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent.parent
PLOTS_DIR  = ROOT / "plots"
MODELS_DIR = ROOT / "models"
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

PALETTE = ["#3b82d4", "#e05c5c", "#10b981"]
GRID_COLOR = "#e5e7eb"

# ---------------------------------------------------------------------------
# SECTION 1 — LOAD DATA
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION 1 — LOAD DATA")
print("=" * 65)

df = pd.read_csv(ROOT / "data" / "processed" / "hotel_bookings_cleaned.csv")
print(f"  Loaded: {len(df):,} rows × {df.shape[1]} columns")
print(f"  Class balance — 0: {(df['is_canceled']==0).sum():,}  1: {(df['is_canceled']==1).sum():,}  ({df['is_canceled'].mean():.2%} canceled)")


# ---------------------------------------------------------------------------
# SECTION 2 — FEATURE ENGINEERING & LEAKAGE REMOVAL
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION 2 — FEATURE ENGINEERING & LEAKAGE REMOVAL")
print("=" * 65)

# --- Post-outcome columns (must be excluded — leakage) ---
LEAKAGE_COLS = [
    "reservation_status",         # assigned after outcome
    "reservation_status_date",    # assigned after outcome
    "revenue_estimate",           # adr × total_nights — partial leakage
    "flag_adr_outlier",           # derived from adr (already included)
    "flag_long_lead_time",        # derived from lead_time (already included)
    "flag_zero_adr",              # derived from adr (already included)
    "arrival_date",               # raw datetime; replaced by numeric features below
]

# --- Features already in the dataset usable at booking time ---
NUMERIC_FEATURES = [
    "lead_time",
    "arrival_date_year",
    "arrival_month_num",            # integer month
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "total_nights",                 # engineered: weekend + week nights
    "adults",
    "children",
    "babies",
    "total_guests",                 # engineered: adults + children + babies
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "room_type_match",              # engineered: 1 if reserved == assigned
    "is_high_season",               # engineered: Jun/Jul/Aug flag
]

CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_date_month",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
]

TARGET = "is_canceled"

print(f"  Numeric features   : {len(NUMERIC_FEATURES)}")
print(f"  Categorical features: {len(CATEGORICAL_FEATURES)}")
print(f"  Leakage columns removed: {LEAKAGE_COLS}")

# --- Encode categoricals ---
df_ml = df.copy()

label_encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    df_ml[col + "_enc"] = le.fit_transform(df_ml[col].astype(str))
    label_encoders[col] = le

ENCODED_CAT_FEATURES = [c + "_enc" for c in CATEGORICAL_FEATURES]
ALL_FEATURES = NUMERIC_FEATURES + ENCODED_CAT_FEATURES

X = df_ml[ALL_FEATURES].copy()
y = df_ml[TARGET].copy()

# Fill any remaining NaN in children column (should be 0 from cleaning)
X["children"] = X["children"].fillna(0)

print(f"\n  Final feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"  Target distribution: {y.value_counts().to_dict()}")

# Save feature names
feature_names_path = MODELS_DIR / "feature_names.txt"
feature_names_path.write_text("\n".join(ALL_FEATURES), encoding="utf-8")
print(f"  Feature list saved: {feature_names_path}")


# ---------------------------------------------------------------------------
# SECTION 3 — TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION 3 — TRAIN / TEST SPLIT")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"  Train set : {X_train.shape[0]:,} rows  (cancel rate: {y_train.mean():.2%})")
print(f"  Test set  : {X_test.shape[0]:,} rows  (cancel rate: {y_test.mean():.2%})")


# ---------------------------------------------------------------------------
# SECTION 4 — MODEL DEFINITIONS
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION 4 — MODEL DEFINITIONS")
print("=" * 65)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
            solver="lbfgs",
            C=1.0,
        )),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=42,
        class_weight="balanced",
    ),
}

for name in models:
    print(f"  Registered: {name}")


# ---------------------------------------------------------------------------
# SECTION 5 — TRAIN, EVALUATE, CROSS-VALIDATE
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION 5 — TRAIN & EVALUATE")
print("=" * 65)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n  --- {name} ---")

    # Train
    model.fit(X_train, y_train)
    trained_models[name] = model

    # Predict
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc         = accuracy_score(y_test, y_pred)
    prec        = precision_score(y_test, y_pred, zero_division=0)
    rec         = recall_score(y_test, y_pred, zero_division=0)
    f1          = f1_score(y_test, y_pred, zero_division=0)
    roc_auc     = roc_auc_score(y_test, y_prob)

    # 5-fold CV ROC-AUC on training set
    cv_scores   = cross_val_score(model, X_train, y_train, cv=cv,
                                   scoring="roc_auc", n_jobs=-1)
    cv_mean     = cv_scores.mean()
    cv_std      = cv_scores.std()

    results[name] = {
        "accuracy":   acc,
        "precision":  prec,
        "recall":     rec,
        "f1":         f1,
        "roc_auc":    roc_auc,
        "cv_roc_auc_mean": cv_mean,
        "cv_roc_auc_std":  cv_std,
        "y_pred":     y_pred,
        "y_prob":     y_prob,
    }

    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {roc_auc:.4f}")
    print(f"    CV ROC-AUC: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"\n    Classification Report (test set):")
    print(textwrap.indent(
        classification_report(y_test, y_pred, target_names=["Not Canceled", "Canceled"]),
        "    "
    ))

    # Save model
    model_path = MODELS_DIR / f"{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, model_path)
    print(f"    Saved: {model_path}")


# ---------------------------------------------------------------------------
# SECTION 6 — BEST MODEL & FEATURE IMPORTANCE
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION 6 — BEST MODEL & FEATURE IMPORTANCE")
print("=" * 65)

best_name = max(results, key=lambda n: results[n]["roc_auc"])
best_model = trained_models[best_name]
print(f"  Best model by ROC-AUC: {best_name} ({results[best_name]['roc_auc']:.4f})")

# Save best model separately
best_path = MODELS_DIR / "best_model.pkl"
joblib.dump(best_model, best_path)
print(f"  Best model saved: {best_path}")

# Feature importance (Random Forest or Gradient Boosting)
fi_model_name = "Random Forest" if "Random Forest" in trained_models else best_name
fi_model = trained_models[fi_model_name]

if hasattr(fi_model, "feature_importances_"):
    importances = fi_model.feature_importances_
elif hasattr(fi_model, "named_steps"):
    clf = fi_model.named_steps.get("clf")
    importances = clf.coef_[0] if hasattr(clf, "coef_") else None
else:
    importances = None

if importances is not None:
    fi_df = pd.DataFrame({
        "feature":    ALL_FEATURES,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    fi_df.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    print(f"\n  Top 15 features ({fi_model_name}):")
    print(fi_df.head(15).to_string(index=False))

    # Plot
    top_n = 20
    fi_top = fi_df.head(top_n).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [PALETTE[0]] * len(fi_top)
    ax.barh(fi_top["feature"], fi_top["importance"], color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance (Mean Decrease Impurity)")
    ax.set_title(f"Top {top_n} Feature Importances — {fi_model_name}", fontweight="bold")
    ax.grid(axis="x", color=GRID_COLOR)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ML_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: plots/ML_feature_importance.png")


# ---------------------------------------------------------------------------
# SECTION 7 — VISUALIZATIONS
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION 7 — VISUALIZATIONS")
print("=" * 65)

# --- 7a. ROC Curves ---
fig, ax = plt.subplots(figsize=(7, 6))
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    roc_val      = res["roc_auc"]
    ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_val:.3f})", color=PALETTE[i])
ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#9ca3af", label="Random Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Cancellation Prediction Models", fontweight="bold")
ax.legend(loc="lower right")
ax.grid(color=GRID_COLOR)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "ML_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: plots/ML_roc_curves.png")

# --- 7b. Confusion Matrices (one per model) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Canceled", "Canceled"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nROC-AUC={res['roc_auc']:.3f}  F1={res['f1']:.3f}", fontweight="bold")
fig.suptitle("Confusion Matrices — Test Set", fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "ML_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: plots/ML_confusion_matrices.png")

# --- 7c. Model Comparison Bar Chart ---
metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
model_names  = list(results.keys())
x = np.arange(len(metric_names))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 5))
for i, mname in enumerate(model_names):
    vals = [results[mname][m] for m in metric_names]
    bars = ax.bar(x + i * width, vals, width, label=mname, color=PALETTE[i], edgecolor="white")

ax.set_xticks(x + width)
ax.set_xticklabels([m.replace("_", " ").title() for m in metric_names])
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison — Test Set", fontweight="bold")
ax.legend()
ax.grid(axis="y", color=GRID_COLOR)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "ML_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: plots/ML_model_comparison.png")


# ---------------------------------------------------------------------------
# SECTION 8 — WRITE REPORT
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION 8 — ML REPORT")
print("=" * 65)

report_lines = []
report_lines.append("Hotel Bookings — Phase 5 ML Pipeline Report")
report_lines.append("=" * 65)
report_lines.append(f"Objective   : Binary classification — predict is_canceled")
report_lines.append(f"Dataset     : hotel_bookings_cleaned.csv  ({len(df):,} rows)")
report_lines.append(f"Train/Test  : 80% / 20% (stratified, random_state=42)")
report_lines.append(f"Features    : {len(ALL_FEATURES)} total ({len(NUMERIC_FEATURES)} numeric, {len(ENCODED_CAT_FEATURES)} encoded categorical)")
report_lines.append(f"CV Strategy : StratifiedKFold(n_splits=5)")
report_lines.append("")

report_lines.append("--- LEAKAGE REMOVED ---")
for col in LEAKAGE_COLS:
    report_lines.append(f"  {col}")
report_lines.append("")

report_lines.append("--- MODEL RESULTS (TEST SET) ---")
header = f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9} {'CV AUC':>10}"
report_lines.append(header)
report_lines.append("-" * len(header))
for name, res in results.items():
    row = (
        f"{name:<25} "
        f"{res['accuracy']:>9.4f} "
        f"{res['precision']:>10.4f} "
        f"{res['recall']:>8.4f} "
        f"{res['f1']:>8.4f} "
        f"{res['roc_auc']:>9.4f} "
        f"{res['cv_roc_auc_mean']:>7.4f}±{res['cv_roc_auc_std']:.4f}"
    )
    report_lines.append(row)
report_lines.append("")

report_lines.append(f"--- BEST MODEL: {best_name} (ROC-AUC={results[best_name]['roc_auc']:.4f}) ---")
report_lines.append("")

report_lines.append("--- CLASSIFICATION REPORT (BEST MODEL, TEST SET) ---")
cr = classification_report(
    y_test, results[best_name]["y_pred"],
    target_names=["Not Canceled", "Canceled"]
)
report_lines.append(cr)

if importances is not None:
    report_lines.append("--- TOP 15 FEATURE IMPORTANCES (Random Forest) ---")
    report_lines.append(fi_df.head(15).to_string(index=False))

report_text = "\n".join(report_lines)
print(report_text)

(ROOT / "reports" / "ml_report.txt").write_text(report_text, encoding="utf-8")
print(f"\n  Report saved: reports/ml_report.txt")
print("\nPhase 5 ML Pipeline complete.")
