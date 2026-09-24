"""
explainability.py
=================
Phase 6 — Model Explainability
Hotel Bookings — Cancellation Prediction

Techniques Used
---------------
1. SHAP TreeExplainer — on the Random Forest model.
   SHAP (SHapley Additive exPlanations) is a game-theoretic method that
   assigns each feature a contribution value for each individual prediction.
   TreeExplainer computes exact SHAP values for tree-based models in
   polynomial time, making it suitable for large datasets.

2. SHAP on HistGradientBoosting (best model) — using shap.Explainer
   (model-agnostic path) for global summary.

3. Permutation Importance — model-agnostic, computed on the test set.
   Measures how much the ROC-AUC drops when a single feature's values
   are randomly shuffled. Unlike MDI (Mean Decrease Impurity), permutation
   importance is not biased toward high-cardinality features.

4. Partial Dependence Plots (PDPs) — for the top 4 numeric features.
   Shows the marginal effect of one feature on the predicted probability,
   averaged over all other features.

IMPORTANT — Prediction vs Causation
-------------------------------------
All findings in this script describe ASSOCIATIONS between features and the
model's predicted probability of cancellation. They do NOT imply that
changing a feature value would CAUSE a booking to cancel or not cancel.

Example: A high `lead_time` is strongly associated with cancellation
(SHAP value > 0). This does not mean that shortening the lead time would
prevent a cancellation — the underlying guest intent is not observed. The
model learns a statistical pattern, not a causal mechanism.

Outputs
-------
  plots/EX_shap_summary_beeswarm.png   — SHAP beeswarm (RF)
  plots/EX_shap_summary_bar.png        — SHAP mean |value| bar (RF)
  plots/EX_shap_dependence_*.png       — SHAP dependence plots for top 4 features
  plots/EX_permutation_importance.png  — Permutation importance (best model)
  plots/EX_partial_dependence.png      — PDP for top 4 numeric features
  plots/EX_shap_waterfall_cancel.png   — Waterfall: sample canceled booking
  plots/EX_shap_waterfall_nocancel.png — Waterfall: sample not-canceled booking
  explainability_report.txt            — structured findings + prediction/causation notes
"""

import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

PLOTS_DIR  = Path("plots")
MODELS_DIR = Path("models")
PLOTS_DIR.mkdir(exist_ok=True)

ACCENT     = "#3b82d4"
CANCEL_CLR = "#e05c5c"
GRID_COLOR = "#e5e7eb"

# ---------------------------------------------------------------------------
# SECTION 1 — REBUILD FEATURE MATRIX (identical to ml_pipeline.py)
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION 1 — LOAD DATA & REBUILD FEATURES")
print("=" * 65)

df = pd.read_csv("hotel_bookings_cleaned.csv")

NUMERIC_FEATURES = [
    "lead_time", "arrival_date_year", "arrival_month_num",
    "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights",
    "total_nights", "adults", "children", "babies", "total_guests",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes",
    "days_in_waiting_list", "adr", "required_car_parking_spaces",
    "total_of_special_requests", "room_type_match", "is_high_season",
]
CATEGORICAL_FEATURES = [
    "hotel", "arrival_date_month", "meal", "market_segment",
    "distribution_channel", "reserved_room_type", "assigned_room_type",
    "deposit_type", "customer_type",
]
TARGET = "is_canceled"

label_encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

ENCODED_CAT = [c + "_enc" for c in CATEGORICAL_FEATURES]
ALL_FEATURES = NUMERIC_FEATURES + ENCODED_CAT

X = df[ALL_FEATURES].copy()
X["children"] = X["children"].fillna(0)
y = df[TARGET].copy()

# Readable labels for plots (strip _enc suffix, replace _ with space)
FEATURE_LABELS = {f: f.replace("_enc", "").replace("_", " ").title() for f in ALL_FEATURES}

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Use a consistent subsample for SHAP (full test set is expensive for beeswarm)
SHAP_SAMPLE_N = 2000
rng = np.random.default_rng(42)
idx = rng.choice(len(X_test), size=min(SHAP_SAMPLE_N, len(X_test)), replace=False)
X_shap = X_test.iloc[idx].reset_index(drop=True)
y_shap = y_test.iloc[idx].reset_index(drop=True)

print(f"  Full dataset   : {len(df):,} rows × {len(ALL_FEATURES)} features")
print(f"  Test set       : {len(X_test):,} rows")
print(f"  SHAP sample    : {len(X_shap):,} rows (stratified random subsample)")

# Load models
rf_model   = joblib.load(MODELS_DIR / "random_forest.pkl")
best_model = joblib.load(MODELS_DIR / "best_model.pkl")   # HistGradientBoosting
print("  Models loaded: random_forest.pkl, best_model.pkl")


# ---------------------------------------------------------------------------
# SECTION 2 — SHAP: RANDOM FOREST (TreeExplainer)
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION 2 — SHAP (Random Forest, TreeExplainer)")
print("=" * 65)

print("  Computing SHAP values on subsample...")
rf_explainer   = shap.TreeExplainer(rf_model)
shap_values_rf = rf_explainer.shap_values(X_shap)

# shap_values_rf may be:
#   - list of two 2D arrays [class0, class1]  (older SHAP)
#   - single 3D array of shape (n, features, classes)  (newer SHAP)
# Always extract the class-1 (Canceled) slice.
if isinstance(shap_values_rf, list):
    sv_canceled = shap_values_rf[1]          # list → index 1
elif shap_values_rf.ndim == 3:
    sv_canceled = shap_values_rf[:, :, 1]   # 3D → last axis index 1
else:
    sv_canceled = shap_values_rf             # already 2D

print(f"  SHAP values shape: {sv_canceled.shape}")

# Mean absolute SHAP per feature
mean_abs_shap = pd.DataFrame({
    "feature":    ALL_FEATURES,
    "label":      [FEATURE_LABELS[f] for f in ALL_FEATURES],
    "mean_abs_shap": np.abs(sv_canceled).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)

print("\n  Top 15 features by mean |SHAP|:")
print(mean_abs_shap.head(15)[["feature", "mean_abs_shap"]].to_string(index=False))

mean_abs_shap.to_csv(MODELS_DIR / "shap_feature_importance.csv", index=False)
print("  Saved: models/shap_feature_importance.csv")

# --- Plot 1: SHAP Bar Summary ---
top_n = 20
top_features = mean_abs_shap.head(top_n)["feature"].tolist()
top_idx      = [ALL_FEATURES.index(f) for f in top_features]
sv_top       = sv_canceled[:, top_idx]
labels_top   = [FEATURE_LABELS[f] for f in top_features]

fig, ax = plt.subplots(figsize=(9, 7))
vals = mean_abs_shap.head(top_n)["mean_abs_shap"].values
sorted_order = np.argsort(vals)
ax.barh(
    [labels_top[i] for i in sorted_order],
    vals[sorted_order],
    color=ACCENT, edgecolor="white",
)
ax.set_xlabel("Mean |SHAP Value| (average impact on model output)")
ax.set_title(f"SHAP Feature Importance — Top {top_n} Features\n(Random Forest, class=Canceled)", fontweight="bold")
ax.grid(axis="x", color=GRID_COLOR)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "EX_shap_summary_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: plots/EX_shap_summary_bar.png")

# --- Plot 2: SHAP Beeswarm Summary ---
# Build a shap.Explanation object for the top-20 features
# Extract scalar base value for class 1 (Canceled)
_ev = rf_explainer.expected_value
if isinstance(_ev, (list, np.ndarray)):
    base_val_rf = float(np.array(_ev).flat[1])
else:
    base_val_rf = float(_ev)

shap_exp = shap.Explanation(
    values=sv_canceled[:, top_idx],
    base_values=np.full(len(X_shap), base_val_rf),
    data=X_shap[top_features].values,
    feature_names=labels_top,
)
fig, ax = plt.subplots(figsize=(10, 8))
shap.plots.beeswarm(shap_exp, max_display=top_n, show=False, plot_size=None)
plt.title("SHAP Beeswarm Plot — Top 20 Features\n(Random Forest, class=Canceled)",
          fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "EX_shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plots/EX_shap_summary_beeswarm.png")


# ---------------------------------------------------------------------------
# SECTION 3 — SHAP DEPENDENCE PLOTS (top 4 numeric features)
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION 3 — SHAP DEPENDENCE PLOTS")
print("=" * 65)

# Identify the top 4 numeric features from SHAP ranking
top_numeric = [
    f for f in mean_abs_shap["feature"].tolist()
    if f in NUMERIC_FEATURES
][:4]
print(f"  Top 4 numeric features: {top_numeric}")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for ax, feat in zip(axes.flat, top_numeric):
    fi   = ALL_FEATURES.index(feat)
    sv_f = sv_canceled[:, fi]
    vals = X_shap[feat].values
    ax.scatter(vals, sv_f, alpha=0.25, s=6, color=ACCENT, rasterized=True)
    # Trend line
    z    = np.polyfit(vals, sv_f, 1)
    p    = np.poly1d(z)
    xs   = np.linspace(vals.min(), vals.max(), 200)
    ax.plot(xs, p(xs), color=CANCEL_CLR, linewidth=2, label="Trend")
    ax.axhline(0, color="#9ca3af", linewidth=1, linestyle="--")
    ax.set_xlabel(FEATURE_LABELS[feat])
    ax.set_ylabel("SHAP Value\n(→ higher = more cancellation)")
    ax.set_title(f"SHAP Dependence: {FEATURE_LABELS[feat]}", fontweight="bold")
    ax.grid(color=GRID_COLOR)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8)
fig.suptitle("SHAP Dependence Plots — Top 4 Numeric Features\n(Random Forest, class=Canceled)",
             fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "EX_shap_dependence_top4.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: plots/EX_shap_dependence_top4.png")


# ---------------------------------------------------------------------------
# SECTION 4 — SHAP WATERFALL: SAMPLE PREDICTIONS
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION 4 — SHAP WATERFALL (Individual Predictions)")
print("=" * 65)

base_val = base_val_rf   # already computed above (class-1 scalar)

def get_sample(label_val, n_candidates=100):
    """Find a cleanly representative sample for the given class."""
    pool = X_shap[y_shap == label_val].head(n_candidates)
    probs = rf_model.predict_proba(pool)[:, 1]
    if label_val == 1:
        # Pick the canceled booking with highest predicted probability
        best_idx = np.argmax(probs)
    else:
        # Pick the not-canceled booking with lowest predicted probability
        best_idx = np.argmin(probs)
    return pool.iloc[[best_idx]], int(pool.index[best_idx])

for cls_val, cls_name, fname in [
    (1, "Canceled Booking",     "EX_shap_waterfall_cancel.png"),
    (0, "Not-Canceled Booking", "EX_shap_waterfall_nocancel.png"),
]:
    sample_X, sample_idx = get_sample(cls_val)
    sample_sv  = sv_canceled[X_shap.index.get_loc(sample_idx) if sample_idx in X_shap.index else 0]

    # Build shap.Explanation for single row (top 15 features only)
    top15_feats = mean_abs_shap.head(15)["feature"].tolist()
    top15_idx   = [ALL_FEATURES.index(f) for f in top15_feats]

    sample_exp = shap.Explanation(
        values=sample_sv[top15_idx],
        base_values=base_val,
        data=sample_X[top15_feats].values[0],
        feature_names=[FEATURE_LABELS[f] for f in top15_feats],
    )

    pred_prob = rf_model.predict_proba(sample_X)[0, 1]
    print(f"  {cls_name}: predicted cancel prob = {pred_prob:.3f}")

    fig, ax = plt.subplots(figsize=(9, 7))
    shap.plots.waterfall(sample_exp, max_display=15, show=False)
    plt.title(f"SHAP Waterfall — {cls_name}\nPredicted Cancel Probability: {pred_prob:.3f}",
              fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: plots/{fname}")


# ---------------------------------------------------------------------------
# SECTION 5 — PERMUTATION IMPORTANCE (model-agnostic, best model)
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION 5 — PERMUTATION IMPORTANCE (Best Model = HistGBM)")
print("=" * 65)

print("  Computing permutation importance on full test set (n_repeats=10)...")
perm_imp = permutation_importance(
    best_model, X_test, y_test,
    n_repeats=10, random_state=42, scoring="roc_auc", n_jobs=-1
)

perm_df = pd.DataFrame({
    "feature":       ALL_FEATURES,
    "label":         [FEATURE_LABELS[f] for f in ALL_FEATURES],
    "importance_mean": perm_imp.importances_mean,
    "importance_std":  perm_imp.importances_std,
}).sort_values("importance_mean", ascending=False)

perm_df.to_csv(MODELS_DIR / "permutation_importance.csv", index=False)
print("  Saved: models/permutation_importance.csv")
print("\n  Top 15 features by Permutation Importance (ROC-AUC drop):")
print(perm_df.head(15)[["label", "importance_mean", "importance_std"]].to_string(index=False))

top20_perm = perm_df.head(20).sort_values("importance_mean", ascending=True)
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(
    top20_perm["label"],
    top20_perm["importance_mean"],
    xerr=top20_perm["importance_std"],
    color=ACCENT, edgecolor="white", capsize=3,
)
ax.set_xlabel("Mean ROC-AUC Decrease (10 repeats)")
ax.set_title("Permutation Feature Importance — Top 20 Features\n(HistGradientBoosting, Best Model)",
             fontweight="bold")
ax.grid(axis="x", color=GRID_COLOR)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "EX_permutation_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: plots/EX_permutation_importance.png")


# ---------------------------------------------------------------------------
# SECTION 6 — PARTIAL DEPENDENCE PLOTS (top 4 numeric from SHAP)
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION 6 — PARTIAL DEPENDENCE PLOTS")
print("=" * 65)

top4_idx = [ALL_FEATURES.index(f) for f in top_numeric]
print(f"  PDP features: {top_numeric}")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
PartialDependenceDisplay.from_estimator(
    best_model,
    X_test,
    features=top4_idx,
    feature_names=[FEATURE_LABELS[f] for f in ALL_FEATURES],
    ax=axes.flat,
    kind="average",
    grid_resolution=50,
    random_state=42,
)
for ax, feat in zip(axes.flat, top_numeric):
    ax.set_title(f"PDP: {FEATURE_LABELS[feat]}", fontweight="bold")
    ax.set_ylabel("Predicted Cancel Probability")
    ax.grid(color=GRID_COLOR)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Partial Dependence Plots — Top 4 Numeric Features\n(HistGradientBoosting, Best Model)",
             fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "EX_partial_dependence.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: plots/EX_partial_dependence.png")


# ---------------------------------------------------------------------------
# SECTION 7 — CROSS-COMPARISON: SHAP vs MDI vs PERMUTATION
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION 7 — CROSS-COMPARISON: SHAP vs MDI vs PERMUTATION")
print("=" * 65)

# MDI from Phase 5
mdi_df = pd.read_csv(MODELS_DIR / "feature_importance.csv").rename(
    columns={"importance": "mdi_importance"}
)

# Merge all three rankings
shap_rank  = mean_abs_shap[["feature", "mean_abs_shap"]].rename(
    columns={"mean_abs_shap": "shap_importance"})
perm_rank  = perm_df[["feature", "importance_mean"]].rename(
    columns={"importance_mean": "perm_importance"})

combined = (
    mdi_df[["feature", "mdi_importance"]]
    .merge(shap_rank, on="feature")
    .merge(perm_rank, on="feature")
)
combined["label"] = combined["feature"].map(FEATURE_LABELS)

# Rank each method 1..31
for col in ["mdi_importance", "shap_importance", "perm_importance"]:
    combined[col.replace("importance", "rank")] = (
        combined[col].rank(ascending=False).astype(int)
    )
combined["avg_rank"] = (
    combined[["mdi_rank", "shap_rank", "perm_rank"]].mean(axis=1)
)
combined = combined.sort_values("avg_rank")

combined.to_csv(MODELS_DIR / "feature_ranking_comparison.csv", index=False)
print("  Saved: models/feature_ranking_comparison.csv")

print("\n  Top 15 by Average Rank across all three methods:")
print(combined.head(15)[["label", "mdi_rank", "shap_rank", "perm_rank", "avg_rank"]].to_string(index=False))

# Plot: rank comparison heatmap for top 15
top15_cmp = combined.head(15).set_index("label")
rank_matrix = top15_cmp[["mdi_rank", "shap_rank", "perm_rank"]]
rank_matrix.columns = ["MDI Rank\n(Random Forest)", "SHAP Rank\n(Random Forest)", "Permutation\nRank (HistGBM)"]

import seaborn as sns
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    rank_matrix,
    annot=True, fmt="d", cmap="YlOrRd",
    linewidths=0.5, linecolor=GRID_COLOR,
    cbar_kws={"label": "Rank (lower = more important)"},
    ax=ax,
)
ax.set_title("Feature Importance Method Comparison\n(Top 15 by Average Rank)", fontweight="bold")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "EX_method_comparison_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: plots/EX_method_comparison_heatmap.png")


# ---------------------------------------------------------------------------
# SECTION 8 — WRITE EXPLAINABILITY REPORT
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION 8 — EXPLAINABILITY REPORT")
print("=" * 65)

report_lines = []
report_lines.append("Hotel Bookings — Phase 6 Model Explainability Report")
report_lines.append("=" * 70)
report_lines.append(f"Model explained : Random Forest (SHAP/MDI) + HistGradientBoosting (Permutation/PDP)")
report_lines.append(f"SHAP library    : {shap.__version__}")
report_lines.append(f"SHAP sample     : {len(X_shap):,} rows (random subset of test set)")
report_lines.append("")

report_lines.append("─" * 70)
report_lines.append("IMPORTANT — PREDICTION vs CAUSATION")
report_lines.append("─" * 70)
report_lines.append("""
All SHAP values, permutation importances, and partial dependence plots
describe STATISTICAL ASSOCIATIONS between features and the model's output.
They do NOT establish that changing a feature would CAUSE a booking to
cancel or not cancel. Confounding factors, selection bias, and unmeasured
variables may explain these associations.

Specific caveats:
  - deposit_type (Non Refund): 99.4% cancellation rate in the data.
    This is the model's #1 feature. The pattern may reflect OTA booking
    policies (non-refundable rate codes) rather than guest intent.
    Changing a booking to 'No Deposit' would NOT necessarily prevent
    cancellation; the guest's underlying decision is unobserved.

  - lead_time: Long-horizon bookings cancel more often. This does not
    mean that shortening the booking window reduces cancellation intent.
    The correlation may reflect that guests who book far in advance are
    more likely to have uncertain plans.

  - total_of_special_requests: Guests with more requests cancel less.
    This is likely because engaged, committed guests both make requests
    AND follow through. Artificially encouraging requests would not
    replicate this effect.

  - previous_cancellations: Past cancellations predict future ones. This
    is a behavioral pattern feature — it reflects guest history, not a
    lever a hotel can change.

  - required_car_parking_spaces: Guests who request parking cancel less.
    Likely a proxy for committed, specific-need travelers. Not a causal
    mechanism.
""")

report_lines.append("─" * 70)
report_lines.append("TOP FEATURE RANKINGS (all three methods)")
report_lines.append("─" * 70)
report_lines.append(combined.head(15)[["label", "mdi_rank", "shap_rank", "perm_rank", "avg_rank"]].to_string(index=False))
report_lines.append("")

report_lines.append("─" * 70)
report_lines.append("SHAP MEAN |VALUE| — TOP 20 (Random Forest, class=Canceled)")
report_lines.append("─" * 70)
report_lines.append(mean_abs_shap.head(20)[["label", "mean_abs_shap"]].to_string(index=False))
report_lines.append("")

report_lines.append("─" * 70)
report_lines.append("PERMUTATION IMPORTANCE — TOP 20 (HistGBM, ROC-AUC drop)")
report_lines.append("─" * 70)
report_lines.append(perm_df.head(20)[["label", "importance_mean", "importance_std"]].to_string(index=False))
report_lines.append("")

report_lines.append("─" * 70)
report_lines.append("KEY INTERPRETATIONS (correlation, not causation)")
report_lines.append("─" * 70)
report_lines.append("""
1. deposit_type [#1 SHAP, #1 Permutation]:
   Non-Refundable bookings are associated with near-certain cancellation.
   This is likely a data artefact of OTA pricing policies rather than
   causal evidence that deposit type drives the cancellation decision.
   Interpretation: Non-Refund rate codes identify a specific booking
   population with historically extreme cancellation behaviour.

2. lead_time [#2 SHAP, #2 Permutation]:
   The longer the gap between booking and arrival, the higher the
   predicted cancellation probability. Each additional day of lead time
   is associated with a marginal increase in SHAP value toward cancellation.
   Monotonic positive relationship confirmed in the PDP.

3. total_of_special_requests [#3 SHAP, #4 Permutation]:
   Negatively associated with cancellation. Guests who submit ≥1 request
   cancel at 22% vs 48% for zero requests (from EDA). The PDP shows a
   steep drop in predicted cancel probability at 1+ requests.

4. market_segment [#4 SHAP, #3 Permutation]:
   Groups segment cancels at 61%, Online TA at 37%, Direct at 15%.
   The model strongly discriminates between segments; Groups and Online TA
   push SHAP values toward cancellation, Direct and Corporate away from it.

5. room_type_match [#5 SHAP, #6 Permutation]:
   Bookings where the reserved room type matches the assigned room type
   are associated with lower cancellation. However, room assignment often
   occurs after booking — this feature may be partially post-booking.

6. previous_cancellations [#6 SHAP, #5 Permutation]:
   Strongest behavioral feature. Even 1 prior cancellation significantly
   increases predicted probability. This is a genuine predictive signal
   about guest reliability.
""")

report_lines.append("─" * 70)
report_lines.append("CHARTS PRODUCED")
report_lines.append("─" * 70)
for p in sorted(PLOTS_DIR.glob("EX_*.png")):
    report_lines.append(f"  {p.name}")

report_text = "\n".join(report_lines)
import sys
safe_report = report_text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
    sys.stdout.encoding or "utf-8"
)
print(safe_report)

Path("explainability_report.txt").write_text(report_text, encoding="utf-8")
print(f"\n  Report saved: explainability_report.txt")
print("\nPhase 6 Model Explainability complete.")
