"""
eda_analysis.py
===============
Phase 3 — Exploratory Data Analysis & KPI Analytics
Hotel Bookings Dataset (hotel_bookings_cleaned.csv)

Sections
--------
  A. Load & Sanity Check
  B. Business KPIs
  C. Cancellation Analysis
  D. Revenue & ADR Analysis
  E. Booking Channel & Market Segment Analysis
  F. Lead Time & Booking Behaviour
  G. Seasonality & Time Trends
  H. Guest Profile Analysis
  I. KPI Summary Report

Outputs
-------
  plots/          — all chart images (PNG, 150 dpi)
  kpi_report.txt  — structured KPI summary
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# GLOBAL STYLE
# ---------------------------------------------------------------------------

PALETTE_MAIN   = ["#3b82d4", "#e05c5c"]          # blue=not-canceled, red=canceled
PALETTE_HOTEL  = ["#3b82d4", "#7c5cd8"]          # City vs Resort
PALETTE_SEQ    = "Blues_d"
ACCENT         = "#3b82d4"
GRID_COLOR     = "#e5e7eb"
FIG_DPI        = 150

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         GRID_COLOR,
    "axes.edgecolor":     "#d1d5db",
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

ROOT      = Path(__file__).parent.parent
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

def save(fig: plt.Figure, name: str) -> None:
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def hbar(ax, series, color=ACCENT, annotate=True):
    """Horizontal bar helper — series index=labels, values=counts/rates."""
    bars = ax.barh(series.index, series.values, color=color, edgecolor="white")
    if annotate:
        for bar, val in zip(bars, series.values):
            ax.text(
                val + series.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}" if val >= 1 else f"{val:.1%}",
                va="center", fontsize=9, color="#374151",
            )
    ax.invert_yaxis()


def pct_fmt(x, _):
    return f"{x:.0f}%"


# ---------------------------------------------------------------------------
# A. LOAD & SANITY CHECK
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  LOADING DATA")
print("=" * 65)

df = pd.read_csv(ROOT / "data" / "processed" / "hotel_bookings_cleaned.csv", parse_dates=["arrival_date"])
df["arrival_date_month"] = pd.Categorical(
    df["arrival_date_month"], categories=MONTH_ORDER, ordered=True
)

print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
print(f"  Date range: {df['arrival_date'].min().date()} -> {df['arrival_date'].max().date()}")
print(f"  Hotels: {df['hotel'].value_counts().to_dict()}")


# ---------------------------------------------------------------------------
# B. BUSINESS KPIs
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION B — BUSINESS KPIs")
print("=" * 65)

kpis = {}

# --- B1. Overall Cancellation Rate ---
kpis["cancellation_rate_overall"] = df["is_canceled"].mean()
kpis["total_bookings"]            = len(df)
kpis["total_canceled"]            = int(df["is_canceled"].sum())
kpis["total_checked_out"]         = int((df["reservation_status"] == "Check-Out").sum())
kpis["total_no_show"]             = int((df["reservation_status"] == "No-Show").sum())

# --- B2. ADR (Average Daily Rate) ---
# ADR is computed only on non-canceled, non-zero-ADR bookings (actual stayed)
stayed = df[df["is_canceled"] == 0].copy()
kpis["adr_overall"]       = stayed["adr"].mean()
kpis["adr_city_hotel"]    = stayed[stayed["hotel"] == "City Hotel"]["adr"].mean()
kpis["adr_resort_hotel"]  = stayed[stayed["hotel"] == "Resort Hotel"]["adr"].mean()
kpis["adr_median"]        = stayed["adr"].median()

# --- B3. Revenue ---
kpis["total_revenue_estimate"]   = stayed["revenue_estimate"].sum()
kpis["avg_revenue_per_booking"]  = stayed["revenue_estimate"].mean()

# --- B4. Average Lead Time ---
kpis["avg_lead_time_days"]           = df["lead_time"].mean()
kpis["avg_lead_time_canceled"]       = df[df["is_canceled"] == 1]["lead_time"].mean()
kpis["avg_lead_time_not_canceled"]   = df[df["is_canceled"] == 0]["lead_time"].mean()

# --- B5. Average Length of Stay ---
kpis["avg_length_of_stay_nights"]        = stayed["total_nights"].mean()
kpis["avg_weekend_nights"]               = stayed["stays_in_weekend_nights"].mean()
kpis["avg_week_nights"]                  = stayed["stays_in_week_nights"].mean()

# --- B6. Repeat Guest Rate ---
kpis["repeat_guest_rate"] = df["is_repeated_guest"].mean()

# --- B7. Room Type Match Rate ---
kpis["room_type_match_rate"] = df["room_type_match"].mean()

# --- B8. Special Requests Rate ---
kpis["avg_special_requests"] = df["total_of_special_requests"].mean()

# --- B9. Occupancy proxy: bookings per month (normalized) ---
monthly_bookings = stayed.groupby("arrival_date_month", observed=True).size()
kpis["peak_month"]    = monthly_bookings.idxmax()
kpis["low_month"]     = monthly_bookings.idxmin()

for k, v in kpis.items():
    if isinstance(v, float):
        print(f"  {k:45s}: {v:.4f}")
    else:
        print(f"  {k:45s}: {v}")


# ---------------------------------------------------------------------------
# C. CANCELLATION ANALYSIS
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION C — CANCELLATION ANALYSIS")
print("=" * 65)

# --- C1. Cancellation Rate by Hotel Type ---
cancel_by_hotel = (
    df.groupby("hotel")["is_canceled"]
    .agg(["mean", "sum", "count"])
    .rename(columns={"mean": "rate", "sum": "canceled", "count": "total"})
    .sort_values("rate", ascending=False)
)
cancel_by_hotel["rate_pct"] = cancel_by_hotel["rate"] * 100
print("\nCancellation by Hotel:")
print(cancel_by_hotel)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
# Left: rate
colors = PALETTE_HOTEL[:len(cancel_by_hotel)]
axes[0].bar(cancel_by_hotel.index, cancel_by_hotel["rate_pct"], color=colors, edgecolor="white", width=0.5)
axes[0].set_title("Cancellation Rate by Hotel Type", fontweight="bold")
axes[0].set_ylabel("Cancellation Rate (%)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
for i, (idx, row) in enumerate(cancel_by_hotel.iterrows()):
    axes[0].text(i, row["rate_pct"] + 0.5, f"{row['rate_pct']:.1f}%", ha="center", fontweight="bold")

# Right: volume stacked
not_canceled = cancel_by_hotel["total"] - cancel_by_hotel["canceled"]
x = np.arange(len(cancel_by_hotel))
axes[1].bar(x, not_canceled.values, label="Not Canceled", color="#3b82d4", edgecolor="white")
axes[1].bar(x, cancel_by_hotel["canceled"].values, bottom=not_canceled.values, label="Canceled", color="#e05c5c", edgecolor="white")
axes[1].set_xticks(x); axes[1].set_xticks(x)
axes[1].set_xticklabels(cancel_by_hotel.index)
axes[1].set_title("Booking Volume by Hotel Type", fontweight="bold")
axes[1].set_ylabel("Number of Bookings")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
axes[1].legend()
fig.tight_layout()
save(fig, "C1_cancellation_by_hotel")

# --- C2. Cancellation Rate by Market Segment ---
cancel_by_seg = (
    df.groupby("market_segment")["is_canceled"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "rate", "count": "total"})
    .query("total >= 50")
    .sort_values("rate", ascending=True)
)
cancel_by_seg["rate_pct"] = cancel_by_seg["rate"] * 100
print("\nCancellation by Market Segment:")
print(cancel_by_seg)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(cancel_by_seg.index, cancel_by_seg["rate_pct"], color=ACCENT, edgecolor="white")
for bar, val in zip(bars, cancel_by_seg["rate_pct"]):
    ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9)
ax.set_xlabel("Cancellation Rate (%)")
ax.set_title("Cancellation Rate by Market Segment", fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
fig.tight_layout()
save(fig, "C2_cancellation_by_market_segment")

# --- C3. Cancellation Rate by Deposit Type ---
cancel_by_dep = (
    df.groupby("deposit_type")["is_canceled"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "rate", "count": "total"})
    .sort_values("rate", ascending=False)
)
cancel_by_dep["rate_pct"] = cancel_by_dep["rate"] * 100
print("\nCancellation by Deposit Type:")
print(cancel_by_dep)

fig, ax = plt.subplots(figsize=(7, 4))
colors_dep = [ACCENT if i % 2 == 0 else "#7c5cd8" for i in range(len(cancel_by_dep))]
bars = ax.bar(cancel_by_dep.index, cancel_by_dep["rate_pct"], color=colors_dep, edgecolor="white", width=0.5)
for bar, val in zip(bars, cancel_by_dep["rate_pct"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Cancellation Rate (%)")
ax.set_title("Cancellation Rate by Deposit Type", fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
fig.tight_layout()
save(fig, "C3_cancellation_by_deposit_type")

# --- C4. Cancellation Rate by Lead Time Bucket ---
df["lead_time_bucket"] = pd.cut(
    df["lead_time"],
    bins=[0, 7, 30, 90, 180, 365, 710],
    labels=["0-7d", "8-30d", "31-90d", "91-180d", "181-365d", "365d+"],
    right=True,
)
cancel_by_lt = (
    df.groupby("lead_time_bucket", observed=True)["is_canceled"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "rate", "count": "total"})
)
cancel_by_lt["rate_pct"] = cancel_by_lt["rate"] * 100
print("\nCancellation by Lead Time Bucket:")
print(cancel_by_lt)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(cancel_by_lt.index.astype(str), cancel_by_lt["rate_pct"],
        marker="o", linewidth=2.5, color=ACCENT, markersize=8)
ax.fill_between(range(len(cancel_by_lt)), cancel_by_lt["rate_pct"].values,
                alpha=0.1, color=ACCENT)
for i, (idx, row) in enumerate(cancel_by_lt.iterrows()):
    ax.text(i, row["rate_pct"] + 0.8, f"{row['rate_pct']:.1f}%", ha="center", fontsize=9)
ax.set_xlabel("Lead Time Bucket")
ax.set_ylabel("Cancellation Rate (%)")
ax.set_title("Cancellation Rate by Lead Time Bucket", fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
fig.tight_layout()
save(fig, "C4_cancellation_by_lead_time_bucket")

# --- C5. Cancellation Rate by Customer Type ---
cancel_by_cust = (
    df.groupby("customer_type")["is_canceled"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "rate", "count": "total"})
    .sort_values("rate", ascending=False)
)
cancel_by_cust["rate_pct"] = cancel_by_cust["rate"] * 100


# ---------------------------------------------------------------------------
# D. REVENUE & ADR ANALYSIS
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION D — REVENUE & ADR ANALYSIS")
print("=" * 65)

# --- D1. Monthly ADR Trend (both hotel types) ---
monthly_adr = (
    stayed.groupby(["arrival_date_month", "hotel"], observed=True)["adr"]
    .mean()
    .reset_index()
    .pivot(index="arrival_date_month", columns="hotel", values="adr")
)
print("\nMonthly ADR by Hotel:")
print(monthly_adr.round(2))

fig, ax = plt.subplots(figsize=(11, 5))
for i, hotel in enumerate(monthly_adr.columns):
    ax.plot(monthly_adr.index.astype(str), monthly_adr[hotel],
            marker="o", linewidth=2.5, label=hotel, color=PALETTE_HOTEL[i])
ax.set_title("Monthly Average Daily Rate (ADR) by Hotel Type", fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("ADR (USD)")
ax.tick_params(axis="x", rotation=40)
ax.legend()
fig.tight_layout()
save(fig, "D1_monthly_adr_by_hotel")

# --- D2. ADR Distribution ---
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, hotel in enumerate(df["hotel"].unique()):
    subset = stayed[stayed["hotel"] == hotel]["adr"]
    ax.hist(subset, bins=60, alpha=0.6, label=hotel, color=PALETTE_HOTEL[i], edgecolor="none")
ax.axvline(stayed["adr"].mean(), color="#e05c5c", linestyle="--", linewidth=1.5, label=f"Overall Mean: ${stayed['adr'].mean():.0f}")
ax.set_xlabel("ADR (USD)")
ax.set_ylabel("Count")
ax.set_title("ADR Distribution by Hotel Type", fontweight="bold")
ax.set_xlim(0, 500)
ax.legend()
fig.tight_layout()
save(fig, "D2_adr_distribution")

# --- D3. ADR by Market Segment ---
adr_by_seg = (
    stayed.groupby("market_segment")["adr"]
    .agg(["mean", "median", "count"])
    .query("count >= 50")
    .sort_values("mean", ascending=True)
)
print("\nADR by Market Segment:")
print(adr_by_seg.round(2))

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(adr_by_seg.index, adr_by_seg["mean"], color=ACCENT, edgecolor="white")
for bar, val in zip(bars, adr_by_seg["mean"]):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"${val:.0f}", va="center", fontsize=9)
ax.set_xlabel("Mean ADR (USD)")
ax.set_title("Mean ADR by Market Segment (Checked-Out Bookings)", fontweight="bold")
fig.tight_layout()
save(fig, "D3_adr_by_market_segment")

# --- D4. Revenue Estimate by Month ---
monthly_rev = (
    stayed.groupby("arrival_date_month", observed=True)["revenue_estimate"]
    .sum()
    .reset_index()
)
monthly_rev.columns = ["month", "revenue"]

fig, ax = plt.subplots(figsize=(11, 4.5))
bars = ax.bar(monthly_rev["month"].astype(str), monthly_rev["revenue"] / 1e6,
              color=ACCENT, edgecolor="white")
ax.set_title("Total Revenue Estimate by Month (All Years Combined)", fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue ($ Millions)")
ax.tick_params(axis="x", rotation=40)
for bar, val in zip(bars, monthly_rev["revenue"] / 1e6):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05, f"${val:.1f}M", ha="center", fontsize=8)
fig.tight_layout()
save(fig, "D4_monthly_revenue_estimate")


# ---------------------------------------------------------------------------
# E. BOOKING CHANNEL & MARKET SEGMENT ANALYSIS
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION E — BOOKING CHANNEL & MARKET SEGMENT")
print("=" * 65)

# --- E1. Booking Volume by Market Segment ---
seg_volume = df["market_segment"].value_counts().sort_values(ascending=True)
print("\nBooking Volume by Market Segment:")
print(seg_volume)

fig, ax = plt.subplots(figsize=(9, 5))
colors_seg = [ACCENT] * len(seg_volume)
bars = ax.barh(seg_volume.index, seg_volume.values, color=colors_seg, edgecolor="white")
for bar, val in zip(bars, seg_volume.values):
    ax.text(val + 200, bar.get_y() + bar.get_height() / 2, f"{val:,}", va="center", fontsize=9)
ax.set_xlabel("Number of Bookings")
ax.set_title("Booking Volume by Market Segment", fontweight="bold")
fig.tight_layout()
save(fig, "E1_volume_by_market_segment")

# --- E2. Distribution Channel Mix ---
dist_vol = df["distribution_channel"].value_counts()
print("\nDistribution Channel:")
print(dist_vol)

fig, ax = plt.subplots(figsize=(6, 6))
wedge_colors = ["#3b82d4", "#7c5cd8", "#e05c5c", "#f59e0b", "#10b981"]
wedges, texts, autotexts = ax.pie(
    dist_vol.values,
    labels=dist_vol.index,
    autopct="%1.1f%%",
    colors=wedge_colors[:len(dist_vol)],
    startangle=140,
    wedgeprops=dict(edgecolor="white", linewidth=1.5),
)
for at in autotexts:
    at.set_fontsize(9)
ax.set_title("Distribution Channel Mix", fontweight="bold")
fig.tight_layout()
save(fig, "E2_distribution_channel_mix")

# --- E3. Cancellation rate + ADR side by side by segment ---
seg_combined = (
    df.groupby("market_segment")
    .agg(
        cancel_rate=("is_canceled", "mean"),
        mean_adr=("adr", "mean"),
        total=("is_canceled", "count"),
    )
    .query("total >= 50")
    .sort_values("cancel_rate", ascending=False)
)
seg_combined["cancel_pct"] = seg_combined["cancel_rate"] * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].barh(seg_combined.index, seg_combined["cancel_pct"], color="#e05c5c", edgecolor="white")
axes[0].set_xlabel("Cancellation Rate (%)")
axes[0].set_title("Cancellation Rate by Segment", fontweight="bold")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
axes[0].invert_yaxis()

axes[1].barh(seg_combined.index, seg_combined["mean_adr"], color=ACCENT, edgecolor="white")
axes[1].set_xlabel("Mean ADR (USD)")
axes[1].set_title("Mean ADR by Segment", fontweight="bold")
axes[1].invert_yaxis()
fig.suptitle("Market Segment: Cancellation vs ADR", fontweight="bold", fontsize=13)
fig.tight_layout()
save(fig, "E3_segment_cancel_vs_adr")


# ---------------------------------------------------------------------------
# F. LEAD TIME & BOOKING BEHAVIOUR
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION F — LEAD TIME & BOOKING BEHAVIOUR")
print("=" * 65)

# --- F1. Lead Time Distribution (canceled vs not) ---
fig, ax = plt.subplots(figsize=(10, 4.5))
bins = np.linspace(0, 500, 60)
ax.hist(df[df["is_canceled"] == 0]["lead_time"].clip(upper=500),
        bins=bins, alpha=0.6, label="Not Canceled", color="#3b82d4", edgecolor="none")
ax.hist(df[df["is_canceled"] == 1]["lead_time"].clip(upper=500),
        bins=bins, alpha=0.6, label="Canceled", color="#e05c5c", edgecolor="none")
ax.set_xlabel("Lead Time (days, capped at 500)")
ax.set_ylabel("Count")
ax.set_title("Lead Time Distribution: Canceled vs Not Canceled", fontweight="bold")
ax.legend()
fig.tight_layout()
save(fig, "F1_lead_time_distribution")

# --- F2. Booking Changes vs Cancellation ---
df["changes_bucket"] = pd.cut(df["booking_changes"], bins=[-1, 0, 1, 3, 20],
                               labels=["0 changes", "1 change", "2-3 changes", "4+"])
cancel_by_changes = (
    df.groupby("changes_bucket", observed=True)["is_canceled"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "rate", "count": "total"})
)
cancel_by_changes["rate_pct"] = cancel_by_changes["rate"] * 100
print("\nCancellation by Booking Changes:")
print(cancel_by_changes)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(cancel_by_changes.index.astype(str), cancel_by_changes["rate_pct"],
              color=ACCENT, edgecolor="white", width=0.5)
for bar, val in zip(bars, cancel_by_changes["rate_pct"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}%", ha="center", fontsize=10)
ax.set_ylabel("Cancellation Rate (%)")
ax.set_title("Cancellation Rate by Number of Booking Changes", fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
fig.tight_layout()
save(fig, "F2_cancellation_by_booking_changes")

# --- F3. Special Requests vs Cancellation ---
cancel_by_sr = (
    df.groupby("total_of_special_requests")["is_canceled"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "rate", "count": "total"})
    .query("total >= 30")
)
cancel_by_sr["rate_pct"] = cancel_by_sr["rate"] * 100
print("\nCancellation by Special Requests:")
print(cancel_by_sr)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cancel_by_sr.index.astype(int), cancel_by_sr["rate_pct"],
        marker="o", linewidth=2.5, color=ACCENT, markersize=8)
ax.set_xlabel("Number of Special Requests")
ax.set_ylabel("Cancellation Rate (%)")
ax.set_title("Cancellation Rate by Number of Special Requests", fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
for i, (idx, row) in enumerate(cancel_by_sr.iterrows()):
    ax.text(idx + 0.05, row["rate_pct"] + 1, f"{row['rate_pct']:.1f}%", fontsize=9)
fig.tight_layout()
save(fig, "F3_cancellation_by_special_requests")


# ---------------------------------------------------------------------------
# G. SEASONALITY & TIME TRENDS
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION G — SEASONALITY & TIME TRENDS")
print("=" * 65)

# --- G1. Monthly Bookings Volume & Cancellation Rate ---
monthly = (
    df.groupby("arrival_date_month", observed=True)
    .agg(
        total=("is_canceled", "count"),
        canceled=("is_canceled", "sum"),
        mean_adr=("adr", "mean"),
    )
    .reset_index()
)
monthly["cancel_rate"] = monthly["canceled"] / monthly["total"] * 100
print("\nMonthly Booking + Cancellation:")
print(monthly[["arrival_date_month", "total", "cancel_rate", "mean_adr"]].to_string())

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
bars = ax1.bar(monthly["arrival_date_month"].astype(str), monthly["total"],
               color=ACCENT, alpha=0.75, edgecolor="white", label="Total Bookings")
ax2.plot(monthly["arrival_date_month"].astype(str), monthly["cancel_rate"],
         color="#e05c5c", linewidth=2.5, marker="o", markersize=6, label="Cancel Rate")
ax1.set_xlabel("Month")
ax1.set_ylabel("Total Bookings", color=ACCENT)
ax2.set_ylabel("Cancellation Rate (%)", color="#e05c5c")
ax1.tick_params(axis="x", rotation=40)
ax1.set_title("Monthly Booking Volume & Cancellation Rate", fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
fig.tight_layout()
save(fig, "G1_monthly_bookings_and_cancel_rate")

# --- G2. Yearly Trend ---
yearly = (
    df.groupby("arrival_date_year")
    .agg(
        total=("is_canceled", "count"),
        cancel_rate=("is_canceled", "mean"),
        mean_adr=("adr", "mean"),
        total_revenue=("revenue_estimate", "sum"),
    )
    .reset_index()
)
yearly["cancel_pct"] = yearly["cancel_rate"] * 100
print("\nYearly Trend:")
print(yearly)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, col, title, fmt in zip(
    axes,
    ["total", "cancel_pct", "mean_adr"],
    ["Total Bookings", "Cancellation Rate (%)", "Mean ADR ($)"],
    ["{:.0f}", "{:.1f}%", "${:.0f}"],
):
    ax.bar(yearly["arrival_date_year"].astype(str), yearly[col], color=ACCENT, edgecolor="white", width=0.5)
    ax.set_title(title, fontweight="bold")
    for i, val in enumerate(yearly[col]):
        ax.text(i, val * 1.01, fmt.format(val), ha="center", fontsize=9)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
fig.suptitle("Year-over-Year Trends (2015–2017)", fontweight="bold", fontsize=13)
fig.tight_layout()
save(fig, "G2_yearly_trends")

# --- G3. Heatmap: Cancellation Rate by Hotel × Month ---
heat_data = (
    df.groupby(["hotel", "arrival_date_month"], observed=True)["is_canceled"]
    .mean()
    .unstack(level="arrival_date_month")
    * 100
)
fig, ax = plt.subplots(figsize=(13, 3.5))
sns.heatmap(
    heat_data,
    annot=True, fmt=".1f", cmap="RdYlGn_r",
    linewidths=0.5, linecolor=GRID_COLOR,
    cbar_kws={"label": "Cancellation Rate (%)"},
    ax=ax,
)
ax.set_title("Cancellation Rate Heatmap: Hotel × Month", fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("")
fig.tight_layout()
save(fig, "G3_cancellation_heatmap_hotel_month")


# ---------------------------------------------------------------------------
# H. GUEST PROFILE ANALYSIS
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION H — GUEST PROFILE ANALYSIS")
print("=" * 65)

# --- H1. Top 15 Countries by Bookings ---
country_vol = df["country"].value_counts().head(15)
print("\nTop 15 Countries:")
print(country_vol)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(country_vol.index[::-1], country_vol.values[::-1], color=ACCENT, edgecolor="white")
for bar, val in zip(bars, country_vol.values[::-1]):
    ax.text(val + 200, bar.get_y() + bar.get_height() / 2, f"{val:,}", va="center", fontsize=9)
ax.set_xlabel("Number of Bookings")
ax.set_title("Top 15 Countries by Booking Volume", fontweight="bold")
fig.tight_layout()
save(fig, "H1_top_countries")

# --- H2. Length of Stay Distribution ---
los = stayed["total_nights"].clip(upper=20).value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(los.index, los.values, color=ACCENT, edgecolor="white")
ax.set_xlabel("Total Nights (capped at 20)")
ax.set_ylabel("Number of Bookings")
ax.set_title("Length of Stay Distribution (Checked-Out Bookings)", fontweight="bold")
ax.set_xticks(range(1, 21))
fig.tight_layout()
save(fig, "H2_length_of_stay_distribution")

# --- H3. Meal Plan Distribution ---
meal_vol = df.groupby(["meal", "hotel"]).size().reset_index(name="count")
meal_pivot = meal_vol.pivot(index="meal", columns="hotel", values="count").fillna(0)
print("\nMeal Plan by Hotel:")
print(meal_pivot)

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(meal_pivot))
w = 0.35
for i, hotel in enumerate(meal_pivot.columns):
    bars = ax.bar(x + i * w, meal_pivot[hotel], width=w, label=hotel,
                  color=PALETTE_HOTEL[i], edgecolor="white")
ax.set_xticks(x + w / 2)
ax.set_xticklabels(meal_pivot.index)
ax.set_ylabel("Number of Bookings")
ax.set_title("Meal Plan Selection by Hotel Type", fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.legend()
fig.tight_layout()
save(fig, "H3_meal_plan_by_hotel")

# --- H4. Repeat vs New Guest Cancellation ---
repeat_cancel = (
    df.groupby("is_repeated_guest")["is_canceled"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "rate", "count": "total"})
)
repeat_cancel.index = ["New Guest", "Repeat Guest"]
repeat_cancel["rate_pct"] = repeat_cancel["rate"] * 100
print("\nCancellation: Repeat vs New Guest:")
print(repeat_cancel)

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(repeat_cancel.index, repeat_cancel["rate_pct"],
              color=PALETTE_HOTEL, edgecolor="white", width=0.45)
for bar, val in zip(bars, repeat_cancel["rate_pct"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}%",
            ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("Cancellation Rate (%)")
ax.set_title("Cancellation Rate: New vs Repeat Guests", fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
fig.tight_layout()
save(fig, "H4_new_vs_repeat_guest_cancellation")


# ---------------------------------------------------------------------------
# I. KPI SUMMARY REPORT
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  SECTION I — KPI SUMMARY REPORT")
print("=" * 65)

report_lines = []
report_lines.append("Hotel Bookings — Phase 3 KPI Analytics Report")
report_lines.append("=" * 65)
report_lines.append(f"Dataset: hotel_bookings_cleaned.csv")
report_lines.append(f"Total Bookings Analyzed: {kpis['total_bookings']:,}")
report_lines.append(f"Date Range: {df['arrival_date'].min().date()} to {df['arrival_date'].max().date()}")
report_lines.append("")

report_lines.append("--- CORE KPIs ---")
report_lines.append(f"Overall Cancellation Rate    : {kpis['cancellation_rate_overall']:.2%}")
report_lines.append(f"  - Total Canceled           : {kpis['total_canceled']:,}")
report_lines.append(f"  - Total Checked Out        : {kpis['total_checked_out']:,}")
report_lines.append(f"  - Total No-Show            : {kpis['total_no_show']:,}")
report_lines.append("")
report_lines.append(f"Average Daily Rate (ADR)     : ${kpis['adr_overall']:.2f}")
report_lines.append(f"  - City Hotel ADR           : ${kpis['adr_city_hotel']:.2f}")
report_lines.append(f"  - Resort Hotel ADR         : ${kpis['adr_resort_hotel']:.2f}")
report_lines.append(f"  - ADR Median               : ${kpis['adr_median']:.2f}")
report_lines.append("")
report_lines.append(f"Total Revenue Estimate       : ${kpis['total_revenue_estimate']:,.0f}")
report_lines.append(f"Avg Revenue per Booking      : ${kpis['avg_revenue_per_booking']:.2f}")
report_lines.append("")
report_lines.append(f"Avg Lead Time (All)          : {kpis['avg_lead_time_days']:.1f} days")
report_lines.append(f"Avg Lead Time (Canceled)     : {kpis['avg_lead_time_canceled']:.1f} days")
report_lines.append(f"Avg Lead Time (Not Canceled) : {kpis['avg_lead_time_not_canceled']:.1f} days")
report_lines.append("")
report_lines.append(f"Avg Length of Stay           : {kpis['avg_length_of_stay_nights']:.2f} nights")
report_lines.append(f"  - Weekend Nights           : {kpis['avg_weekend_nights']:.2f}")
report_lines.append(f"  - Week Nights              : {kpis['avg_week_nights']:.2f}")
report_lines.append("")
report_lines.append(f"Repeat Guest Rate            : {kpis['repeat_guest_rate']:.2%}")
report_lines.append(f"Room Type Match Rate         : {kpis['room_type_match_rate']:.2%}")
report_lines.append(f"Avg Special Requests/Booking : {kpis['avg_special_requests']:.2f}")
report_lines.append("")
report_lines.append(f"Peak Month (by volume)       : {kpis['peak_month']}")
report_lines.append(f"Low Season Month             : {kpis['low_month']}")
report_lines.append("")

report_lines.append("--- CANCELLATION BREAKDOWN ---")
report_lines.append("By Hotel Type:")
for idx, row in cancel_by_hotel.iterrows():
    report_lines.append(f"  {idx:20s}: {row['rate_pct']:.1f}%  ({int(row['canceled']):,} / {int(row['total']):,})")
report_lines.append("")
report_lines.append("By Deposit Type:")
for idx, row in cancel_by_dep.iterrows():
    report_lines.append(f"  {str(idx):20s}: {row['rate_pct']:.1f}%  (n={int(row['total']):,})")
report_lines.append("")
report_lines.append("By Lead Time Bucket:")
for idx, row in cancel_by_lt.iterrows():
    report_lines.append(f"  {str(idx):20s}: {row['rate_pct']:.1f}%  (n={int(row['total']):,})")
report_lines.append("")
report_lines.append("By Market Segment:")
for idx, row in cancel_by_seg.sort_values("rate_pct", ascending=False).iterrows():
    report_lines.append(f"  {str(idx):20s}: {row['rate_pct']:.1f}%  (n={int(row['total']):,})")
report_lines.append("")

report_lines.append("--- CHARTS PRODUCED ---")
chart_list = sorted(PLOTS_DIR.glob("*.png"))
for c in chart_list:
    report_lines.append(f"  {c.name}")

report_text = "\n".join(report_lines)
print(report_text)

with open(ROOT / "reports" / "kpi_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"\n  KPI report saved: reports/kpi_report.txt")
print(f"  Charts saved:     {len(chart_list)} files in {PLOTS_DIR}/")
print("\nPhase 3 EDA complete.")
