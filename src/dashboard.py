"""
dashboard.py
============
Phase 7 & 8 — Interactive Dashboard + AI Insights
Hotel Bookings Analytics & ML Platform

Run with:
    streamlit run src/dashboard.py

Tabs
----
  1. Overview        — KPI tiles + hotel/year filter
  2. Cancellation    — Cancellation analysis charts + interactive Plotly
  3. Revenue & ADR   — Revenue trends, ADR by segment, monthly breakdown
  4. Segments        — Market segment deep-dive with filterable table
  5. ML Prediction   — Real-time cancellation probability from trained model
  6. Explainability  — SHAP, permutation importance, PDP charts + caveats
  7. AI Insights     — GPT-powered natural language analysis of validated metrics

All values loaded dynamically from:
  - sql_results/*.csv                    (pre-computed SQL query outputs)
  - data/processed/hotel_bookings_cleaned.csv (raw cleaned data for interactive charts)
  - models/best_model.pkl                (HistGradientBoosting, ROC-AUC 0.9157)
  - models/feature_names.txt
  - plots/                               (static EDA + explainability PNGs)
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import LabelEncoder

from ai_insights import (
    get_overview_insight,
    get_cancellation_insight,
    get_revenue_insight,
    get_ml_insight,
    get_explainability_insight,
    get_custom_insight,
    build_overview_prompt,
    build_cancellation_prompt,
    build_revenue_prompt,
    build_ml_prompt,
    build_explainability_prompt,
    _load_core_kpis,
    _load_cancellation_context,
    _load_revenue_context,
    _load_ml_context,
    _load_explainability_context,
    FALLBACK_MSG,
)

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Hotel Bookings Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

BASE          = Path(__file__).parent.parent
SQL_DIR       = BASE / "sql_results"
PLOTS_DIR     = BASE / "plots"
MODELS_DIR    = BASE / "models"
DATA_FILE     = BASE / "data" / "processed" / "hotel_bookings_cleaned.csv"

# ---------------------------------------------------------------------------
# COLOUR PALETTE
# ---------------------------------------------------------------------------

CLR_BLUE   = "#3b82d4"
CLR_RED    = "#e05c5c"
CLR_PURPLE = "#7c5cd8"
CLR_GREEN  = "#10b981"
CLR_AMBER  = "#f59e0b"
PALETTE    = [CLR_BLUE, CLR_RED, CLR_PURPLE, CLR_GREEN, CLR_AMBER]

MONTH_ORDER = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December",
]

# ---------------------------------------------------------------------------
# DATA LOADERS  (cached so they run once per session)
# ---------------------------------------------------------------------------

@st.cache_data
def load_cleaned_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=["arrival_date"])
    df["arrival_date_month"] = pd.Categorical(
        df["arrival_date_month"], categories=MONTH_ORDER, ordered=True
    )
    return df

@st.cache_data
def load_sql(filename: str) -> pd.DataFrame:
    return pd.read_csv(SQL_DIR / filename)

@st.cache_resource
def load_model():
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    feature_names = (MODELS_DIR / "feature_names.txt").read_text().splitlines()
    return model, feature_names

@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(MODELS_DIR / "shap_feature_importance.csv")

@st.cache_data
def load_perm_importance() -> pd.DataFrame:
    return pd.read_csv(MODELS_DIR / "permutation_importance.csv")

@st.cache_data
def load_ranking_comparison() -> pd.DataFrame:
    return pd.read_csv(MODELS_DIR / "feature_ranking_comparison.csv")

# ---------------------------------------------------------------------------
# CATEGORICAL ENCODING HELPER (mirrors ml_pipeline.py exactly)
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES = [
    "hotel", "arrival_date_month", "meal", "market_segment",
    "distribution_channel", "reserved_room_type", "assigned_room_type",
    "deposit_type", "customer_type",
]

@st.cache_data
def build_label_encoders(df: pd.DataFrame) -> dict:
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders

# ---------------------------------------------------------------------------
# SIDEBAR GLOBAL FILTERS
# ---------------------------------------------------------------------------

def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("🔍 Global Filters")

    hotels = ["All"] + sorted(df["hotel"].unique().tolist())
    hotel_sel = st.sidebar.selectbox("Hotel Type", hotels)

    years = ["All"] + sorted(df["arrival_date_year"].unique().tolist())
    year_sel = st.sidebar.selectbox("Arrival Year", years)

    segments = ["All"] + sorted(df["market_segment"].unique().tolist())
    seg_sel = st.sidebar.multiselect("Market Segment", segments[1:],
                                     default=segments[1:])

    filtered = df.copy()
    if hotel_sel != "All":
        filtered = filtered[filtered["hotel"] == hotel_sel]
    if year_sel != "All":
        filtered = filtered[filtered["arrival_date_year"] == int(year_sel)]
    if seg_sel:
        filtered = filtered[filtered["market_segment"].isin(seg_sel)]

    st.sidebar.markdown("---")
    st.sidebar.metric("Filtered Rows", f"{len(filtered):,}")
    return filtered

# ---------------------------------------------------------------------------
# KPI TILE HELPER
# ---------------------------------------------------------------------------

def kpi_tile(col, label: str, value: str, delta: str = None, help_text: str = None):
    col.metric(label=label, value=value, delta=delta, help=help_text)

# ---------------------------------------------------------------------------
# TAB 1 — OVERVIEW
# ---------------------------------------------------------------------------

def tab_overview(df: pd.DataFrame):
    st.header("📊 Overview KPIs")

    kpi = load_sql("Q01_Core_KPI_Snapshot.csv").iloc[0]

    # Compute filtered KPIs dynamically
    total         = len(df)
    canceled      = int(df["is_canceled"].sum())
    cancel_rate   = df["is_canceled"].mean() * 100
    stayed        = df[df["is_canceled"] == 0]
    avg_adr       = stayed["adr"].mean()
    total_rev     = stayed["revenue_estimate"].sum()
    avg_nights    = stayed["total_nights"].mean()
    repeat_rate   = df["is_repeated_guest"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    kpi_tile(c1, "Total Bookings",          f"{total:,}")
    kpi_tile(c2, "Cancellation Rate",       f"{cancel_rate:.1f}%",
             help_text="% of bookings that were canceled")
    kpi_tile(c3, "Avg Daily Rate (ADR)",    f"${avg_adr:.2f}",
             help_text="Mean ADR for checked-out bookings")
    kpi_tile(c4, "Total Revenue Estimate",  f"${total_rev:,.0f}",
             help_text="ADR × total nights, checked-out only")

    c5, c6, c7, c8 = st.columns(4)
    kpi_tile(c5, "Canceled Bookings",       f"{canceled:,}")
    kpi_tile(c6, "Avg Length of Stay",      f"{avg_nights:.2f} nights")
    kpi_tile(c7, "Repeat Guest Rate",       f"{repeat_rate:.2f}%")
    kpi_tile(c8, "Room Type Match Rate",    f"{df['room_type_match'].mean()*100:.1f}%")

    st.markdown("---")
    st.subheader("Bookings by Hotel Type & Year")

    col_l, col_r = st.columns(2)

    # Hotel split donut
    hotel_counts = df["hotel"].value_counts().reset_index()
    hotel_counts.columns = ["hotel", "count"]
    fig_donut = px.pie(hotel_counts, names="hotel", values="count",
                       hole=0.45, color_discrete_sequence=[CLR_BLUE, CLR_PURPLE],
                       title="Booking Share by Hotel Type")
    fig_donut.update_traces(textinfo="percent+label")
    col_l.plotly_chart(fig_donut, use_container_width=True)

    # Year-over-year bar
    yoy = load_sql("Q10_Year-over-Year_Revenue_and_ADR_Trends.csv")
    fig_yoy = go.Figure()
    fig_yoy.add_bar(x=yoy["arrival_date_year"].astype(str),
                    y=yoy["total_bookings"], name="Total Bookings",
                    marker_color=CLR_BLUE)
    fig_yoy.add_bar(x=yoy["arrival_date_year"].astype(str),
                    y=yoy["stayed"], name="Stayed",
                    marker_color=CLR_GREEN)
    fig_yoy.add_bar(x=yoy["arrival_date_year"].astype(str),
                    y=yoy["canceled"], name="Canceled",
                    marker_color=CLR_RED)
    fig_yoy.update_layout(title="Bookings by Year (Total / Stayed / Canceled)",
                          barmode="group", xaxis_title="Year",
                          yaxis_title="Count", height=380)
    col_r.plotly_chart(fig_yoy, use_container_width=True)

    # YoY table
    st.subheader("Year-over-Year Performance Table")
    yoy_disp = yoy.copy()
    yoy_disp["cancellation_rate_pct"] = yoy_disp["cancellation_rate_pct"].apply(lambda x: f"{x:.1f}%")
    yoy_disp["avg_adr_stayed"]        = yoy_disp["avg_adr_stayed"].apply(lambda x: f"${x:.2f}")
    yoy_disp["total_revenue"]         = yoy_disp["total_revenue"].apply(lambda x: f"${x:,.0f}")
    yoy_disp["avg_revenue_per_booking"] = yoy_disp["avg_revenue_per_booking"].apply(lambda x: f"${x:.2f}")
    st.dataframe(yoy_disp, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 2 — CANCELLATION ANALYSIS
# ---------------------------------------------------------------------------

def tab_cancellation(df: pd.DataFrame):
    st.header("❌ Cancellation Analysis")

    col_l, col_r = st.columns(2)

    # By Hotel
    q02 = load_sql("Q02_Cancellation_Rate_by_Hotel_Type.csv")
    fig_hotel = px.bar(q02, x="hotel", y="cancellation_rate_pct",
                       color="hotel", color_discrete_sequence=[CLR_BLUE, CLR_PURPLE],
                       title="Cancellation Rate by Hotel Type",
                       labels={"cancellation_rate_pct": "Cancellation Rate (%)"},
                       text="cancellation_rate_pct")
    fig_hotel.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_hotel.update_layout(showlegend=False, height=380)
    col_l.plotly_chart(fig_hotel, use_container_width=True)

    # By Deposit Type
    q04 = load_sql("Q04_Cancellation_Rate_by_Deposit_Type.csv")
    fig_dep = px.bar(q04.sort_values("cancellation_rate_pct", ascending=True),
                     x="cancellation_rate_pct", y="deposit_type",
                     orientation="h",
                     color="cancellation_rate_pct",
                     color_continuous_scale=["#3b82d4", "#e05c5c"],
                     title="Cancellation Rate by Deposit Type",
                     labels={"cancellation_rate_pct": "Cancellation Rate (%)"},
                     text="cancellation_rate_pct")
    fig_dep.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_dep.update_layout(height=380, coloraxis_showscale=False)
    col_r.plotly_chart(fig_dep, use_container_width=True)

    st.markdown("---")

    col_l2, col_r2 = st.columns(2)

    # By Lead Time Bucket
    q05 = load_sql("Q05_Cancellation_Rate_by_Lead_Time_Bucket.csv")
    q05["bucket_label"] = q05["lead_time_bucket"].str.replace(r"^\d+_", "", regex=True)
    fig_lt = px.line(q05, x="bucket_label", y="cancellation_rate_pct",
                     markers=True, title="Cancellation Rate by Lead Time Bucket",
                     labels={"cancellation_rate_pct": "Cancellation Rate (%)",
                             "bucket_label": "Lead Time Bucket"},
                     color_discrete_sequence=[CLR_BLUE])
    fig_lt.update_traces(line_width=3, marker_size=10)
    fig_lt.update_layout(height=380)
    col_l2.plotly_chart(fig_lt, use_container_width=True)

    # By Market Segment
    q03 = load_sql("Q03_Cancellation_Rate_by_Market_Segment.csv")
    fig_seg = px.bar(q03.sort_values("cancellation_rate_pct"),
                     x="cancellation_rate_pct", y="market_segment",
                     orientation="h", title="Cancellation Rate by Market Segment",
                     labels={"cancellation_rate_pct": "Cancellation Rate (%)"},
                     color="cancellation_rate_pct",
                     color_continuous_scale=["#3b82d4", "#e05c5c"],
                     text="cancellation_rate_pct")
    fig_seg.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_seg.update_layout(height=380, coloraxis_showscale=False)
    col_r2.plotly_chart(fig_seg, use_container_width=True)

    st.markdown("---")

    # Interactive: Monthly cancellation heatmap from filtered data
    st.subheader("Monthly Cancellation Rate Heatmap (Filtered Data)")
    heat = (
        df.groupby(["hotel", "arrival_date_month"], observed=True)["is_canceled"]
        .mean().mul(100).unstack("arrival_date_month")
    )
    fig_heat = px.imshow(
        heat,
        color_continuous_scale="RdYlGn_r",
        title="Cancellation Rate % by Hotel × Month",
        labels={"color": "Cancel Rate (%)"},
        text_auto=".1f",
        aspect="auto",
    )
    fig_heat.update_layout(height=300)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Cancellation by Customer Type
    st.subheader("Cancellation & ADR by Customer Type")
    q06 = load_sql("Q06_Cancellation_Rate_by_Customer_Type.csv")
    fig_cust = px.scatter(
        q06, x="avg_adr_stayed", y="cancellation_rate_pct",
        size="total_bookings", color="customer_type",
        color_discrete_sequence=PALETTE,
        hover_data=["avg_lead_time_days", "avg_nights_stayed"],
        labels={"avg_adr_stayed": "Avg ADR (Stayed, $)",
                "cancellation_rate_pct": "Cancellation Rate (%)"},
        title="Customer Type: Cancellation Rate vs ADR  (bubble = volume)",
        size_max=50,
    )
    fig_cust.update_layout(height=380)
    st.plotly_chart(fig_cust, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 3 — REVENUE & ADR
# ---------------------------------------------------------------------------

def tab_revenue(df: pd.DataFrame):
    st.header("💰 Revenue & ADR Analysis")

    col_l, col_r = st.columns(2)

    # Monthly ADR by hotel (from filtered data)
    stayed = df[df["is_canceled"] == 0]
    monthly_adr = (
        stayed.groupby(["arrival_date_month", "hotel"], observed=True)["adr"]
        .mean().reset_index()
    )
    fig_adr = px.line(monthly_adr, x="arrival_date_month", y="adr",
                      color="hotel", markers=True,
                      color_discrete_sequence=[CLR_BLUE, CLR_PURPLE],
                      title="Monthly ADR by Hotel Type (Filtered)",
                      labels={"adr": "Avg ADR ($)", "arrival_date_month": "Month"})
    fig_adr.update_traces(line_width=2.5, marker_size=7)
    fig_adr.update_layout(height=380)
    col_l.plotly_chart(fig_adr, use_container_width=True)

    # Monthly revenue estimate
    monthly_rev = (
        stayed.groupby("arrival_date_month", observed=True)["revenue_estimate"]
        .sum().reset_index()
    )
    monthly_rev.columns = ["month", "revenue"]
    fig_rev = px.bar(monthly_rev, x="month", y="revenue",
                     color_discrete_sequence=[CLR_GREEN],
                     title="Monthly Revenue Estimate (Filtered)",
                     labels={"revenue": "Revenue ($)", "month": "Month"})
    fig_rev.update_layout(height=380)
    col_r.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("---")

    # Revenue by market segment
    q07 = load_sql("Q07_Revenue_and_ADR_by_Market_Segment_Checked-Out_Bookings.csv")
    col_l2, col_r2 = st.columns(2)

    fig_rev_seg = px.bar(
        q07.sort_values("total_revenue", ascending=True),
        x="total_revenue", y="market_segment", orientation="h",
        color="total_revenue", color_continuous_scale=["#bfdbfe", CLR_BLUE],
        title="Total Revenue by Market Segment",
        labels={"total_revenue": "Revenue ($)", "market_segment": "Segment"},
        text="revenue_share_pct",
    )
    fig_rev_seg.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_rev_seg.update_layout(height=400, coloraxis_showscale=False)
    col_l2.plotly_chart(fig_rev_seg, use_container_width=True)

    fig_adr_seg = px.bar(
        q07.sort_values("avg_adr", ascending=True),
        x="avg_adr", y="market_segment", orientation="h",
        color="avg_adr", color_continuous_scale=["#bfdbfe", CLR_PURPLE],
        title="Average ADR by Market Segment",
        labels={"avg_adr": "Avg ADR ($)", "market_segment": "Segment"},
        text="avg_adr",
    )
    fig_adr_seg.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
    fig_adr_seg.update_layout(height=400, coloraxis_showscale=False)
    col_r2.plotly_chart(fig_adr_seg, use_container_width=True)

    st.markdown("---")

    # ADR distribution from filtered data
    st.subheader("ADR Distribution (Filtered, Checked-Out Bookings)")
    stayed_filtered = df[df["is_canceled"] == 0]
    fig_hist = px.histogram(
        stayed_filtered, x="adr", color="hotel",
        nbins=80, barmode="overlay",
        color_discrete_sequence=[CLR_BLUE, CLR_PURPLE],
        title="ADR Distribution by Hotel Type",
        labels={"adr": "ADR ($)"},
        range_x=[0, 400],
        opacity=0.7,
    )
    fig_hist.add_vline(x=stayed_filtered["adr"].mean(), line_dash="dash",
                       line_color=CLR_RED,
                       annotation_text=f"Mean ${stayed_filtered['adr'].mean():.0f}")
    fig_hist.update_layout(height=360)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Lead time vs ADR from stayed bookings
    st.subheader("Lead Time Buckets vs ADR & Revenue")
    q17 = load_sql("Q17_Lead_Time_Buckets_vs_Revenue_and_ADR_Stayed_Bookings.csv")
    q17["bucket_label"] = q17["lead_time_bucket"].str.replace(r"^\d+_", "", regex=True)
    fig_lt_adr = go.Figure()
    fig_lt_adr.add_bar(x=q17["bucket_label"], y=q17["avg_adr"],
                       name="Avg ADR ($)", marker_color=CLR_BLUE)
    fig_lt_adr.add_bar(x=q17["bucket_label"], y=q17["avg_revenue_per_booking"],
                       name="Avg Revenue/Booking ($)", marker_color=CLR_GREEN)
    fig_lt_adr.update_layout(barmode="group", title="Lead Time vs ADR & Revenue (Stayed Bookings)",
                              height=360)
    st.plotly_chart(fig_lt_adr, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 4 — SEGMENT DEEP-DIVE
# ---------------------------------------------------------------------------

def tab_segments(df: pd.DataFrame):
    st.header("🔍 Segment Deep-Dive")

    # Segment selector
    seg_choice = st.selectbox(
        "Select a market segment to explore",
        sorted(df["market_segment"].unique().tolist()),
    )
    seg_df   = df[df["market_segment"] == seg_choice]
    seg_stay = seg_df[seg_df["is_canceled"] == 0]

    c1, c2, c3, c4 = st.columns(4)
    kpi_tile(c1, "Bookings",          f"{len(seg_df):,}")
    kpi_tile(c2, "Cancellation Rate", f"{seg_df['is_canceled'].mean()*100:.1f}%")
    kpi_tile(c3, "Avg ADR",           f"${seg_stay['adr'].mean():.2f}" if len(seg_stay) else "N/A")
    kpi_tile(c4, "Avg Lead Time",     f"{seg_df['lead_time'].mean():.0f} days")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    # Monthly volume
    monthly_vol = (
        seg_df.groupby("arrival_date_month", observed=True)
        .agg(total=("is_canceled","count"), canceled=("is_canceled","sum"))
        .reset_index()
    )
    monthly_vol["cancel_rate"] = monthly_vol["canceled"] / monthly_vol["total"] * 100
    fig_mv = go.Figure()
    fig_mv.add_bar(x=monthly_vol["arrival_date_month"].astype(str),
                   y=monthly_vol["total"], name="Total", marker_color=CLR_BLUE)
    fig_mv.add_bar(x=monthly_vol["arrival_date_month"].astype(str),
                   y=monthly_vol["canceled"], name="Canceled", marker_color=CLR_RED)
    fig_mv.update_layout(barmode="group", title=f"{seg_choice} — Monthly Volume",
                          height=360, xaxis_tickangle=40)
    col_l.plotly_chart(fig_mv, use_container_width=True)

    # Lead time distribution
    fig_ld = px.histogram(
        seg_df, x="lead_time", color="is_canceled",
        color_discrete_map={0: CLR_BLUE, 1: CLR_RED},
        nbins=50, barmode="overlay", opacity=0.7,
        title=f"{seg_choice} — Lead Time Distribution",
        labels={"lead_time": "Lead Time (days)", "is_canceled": "Canceled"},
        category_orders={"is_canceled": [0, 1]},
    )
    fig_ld.update_layout(height=360)
    col_r.plotly_chart(fig_ld, use_container_width=True)

    st.markdown("---")

    # Full segment comparison table from SQL
    st.subheader("All Segments — Revenue & ADR Comparison")
    q07 = load_sql("Q07_Revenue_and_ADR_by_Market_Segment_Checked-Out_Bookings.csv")
    q03 = load_sql("Q03_Cancellation_Rate_by_Market_Segment.csv")[
        ["market_segment", "cancellation_rate_pct"]
    ]
    seg_table = q07.merge(q03, on="market_segment", how="left")
    seg_table = seg_table.sort_values("total_revenue", ascending=False)
    seg_table["total_revenue"]   = seg_table["total_revenue"].apply(lambda x: f"${x:,.0f}")
    seg_table["avg_adr"]         = seg_table["avg_adr"].apply(lambda x: f"${x:.2f}")
    seg_table["revenue_share_pct"] = seg_table["revenue_share_pct"].apply(lambda x: f"{x:.1f}%")
    seg_table["cancellation_rate_pct"] = seg_table["cancellation_rate_pct"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(seg_table, use_container_width=True, hide_index=True)

    # Top Countries
    st.subheader("Top 15 Countries")
    q11 = load_sql("Q11_Top_15_Countries_by_Booking_Volume_and_Cancellation_Rate.csv")
    fig_countries = px.bar(
        q11, x="total_bookings", y="country", orientation="h",
        color="cancellation_rate_pct",
        color_continuous_scale=["#bfdbfe", CLR_RED],
        title="Top 15 Countries: Bookings (colour = cancel rate)",
        labels={"total_bookings": "Bookings", "country": "Country"},
    )
    fig_countries.update_layout(height=460)
    st.plotly_chart(fig_countries, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 5 — ML PREDICTION
# ---------------------------------------------------------------------------

def tab_prediction(df: pd.DataFrame):
    st.header("🤖 ML Cancellation Predictor")

    model, feature_names = load_model()
    encoders = build_label_encoders(df)

    st.markdown(
        """
        **Model:** HistGradientBoostingClassifier — ROC-AUC **0.9157** · F1 **0.783** on held-out test set  
        Enter booking details below to predict cancellation probability.
        """
    )

    st.markdown("---")
    st.subheader("Enter Booking Details")

    NUMERIC_FEATURES = [
        "lead_time","arrival_date_year","arrival_month_num",
        "arrival_date_week_number","arrival_date_day_of_month",
        "stays_in_weekend_nights","stays_in_week_nights",
        "total_nights","adults","children","babies","total_guests",
        "is_repeated_guest","previous_cancellations",
        "previous_bookings_not_canceled","booking_changes",
        "days_in_waiting_list","adr","required_car_parking_spaces",
        "total_of_special_requests","room_type_match","is_high_season",
    ]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Stay Details**")
        hotel           = st.selectbox("Hotel Type", sorted(df["hotel"].unique()))
        lead_time       = st.number_input("Lead Time (days)", 0, 740, 30)
        weekend_nights  = st.number_input("Weekend Nights", 0, 20, 1)
        week_nights     = st.number_input("Week Nights", 0, 50, 2)
        total_nights_v  = weekend_nights + week_nights
        arrival_month   = st.selectbox("Arrival Month", MONTH_ORDER)
        arrival_year    = st.selectbox("Arrival Year", [2015, 2016, 2017, 2018], index=2)

    with c2:
        st.markdown("**Guest Profile**")
        adults           = st.number_input("Adults", 1, 10, 2)
        children         = st.number_input("Children", 0, 10, 0)
        babies           = st.number_input("Babies", 0, 5, 0)
        total_guests_v   = adults + children + babies
        is_repeated      = st.selectbox("Repeated Guest?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        prev_cancels     = st.number_input("Previous Cancellations", 0, 26, 0)
        prev_not_cancel  = st.number_input("Prev Bookings Not Canceled", 0, 72, 0)
        special_req      = st.number_input("Special Requests", 0, 5, 0)
        parking          = st.number_input("Parking Spaces Requested", 0, 8, 0)

    with c3:
        st.markdown("**Booking Details**")
        market_seg       = st.selectbox("Market Segment", sorted(df["market_segment"].unique()))
        dist_channel     = st.selectbox("Distribution Channel", sorted(df["distribution_channel"].unique()))
        deposit          = st.selectbox("Deposit Type", sorted(df["deposit_type"].unique()))
        meal             = st.selectbox("Meal Plan", sorted(df["meal"].unique()))
        reserved_room    = st.selectbox("Reserved Room Type", sorted(df["reserved_room_type"].unique()))
        assigned_room    = st.selectbox("Assigned Room Type", sorted(df["assigned_room_type"].unique()))
        customer_type    = st.selectbox("Customer Type", sorted(df["customer_type"].unique()))
        adr_v            = st.number_input("ADR ($)", 0.0, 5400.0, 100.0, step=5.0)
        booking_changes  = st.number_input("Booking Changes", 0, 21, 0)
        waiting_days     = st.number_input("Days in Waiting List", 0, 400, 0)

    # Derived features
    month_map = {m: i+1 for i, m in enumerate(MONTH_ORDER)}
    arrival_month_num = month_map[arrival_month]
    is_high_season_v  = 1 if arrival_month_num in [6, 7, 8] else 0
    room_match_v      = 1 if reserved_room == assigned_room else 0

    import datetime
    try:
        dt = datetime.date(arrival_year, arrival_month_num, 1)
        week_num = dt.isocalendar()[1]
        day_of_month = 1
    except Exception:
        week_num = 1
        day_of_month = 1

    # Encode categoricals using same LabelEncoders as training
    def safe_encode(le: LabelEncoder, val: str) -> int:
        classes = list(le.classes_)
        return le.transform([val])[0] if val in classes else 0

    cat_values = {
        "hotel":                  safe_encode(encoders["hotel"], hotel),
        "arrival_date_month":     safe_encode(encoders["arrival_date_month"], arrival_month),
        "meal":                   safe_encode(encoders["meal"], meal),
        "market_segment":         safe_encode(encoders["market_segment"], market_seg),
        "distribution_channel":   safe_encode(encoders["distribution_channel"], dist_channel),
        "reserved_room_type":     safe_encode(encoders["reserved_room_type"], reserved_room),
        "assigned_room_type":     safe_encode(encoders["assigned_room_type"], assigned_room),
        "deposit_type":           safe_encode(encoders["deposit_type"], deposit),
        "customer_type":          safe_encode(encoders["customer_type"], customer_type),
    }

    num_values = {
        "lead_time":                       lead_time,
        "arrival_date_year":               arrival_year,
        "arrival_month_num":               arrival_month_num,
        "arrival_date_week_number":        week_num,
        "arrival_date_day_of_month":       day_of_month,
        "stays_in_weekend_nights":         weekend_nights,
        "stays_in_week_nights":            week_nights,
        "total_nights":                    total_nights_v,
        "adults":                          adults,
        "children":                        float(children),
        "babies":                          babies,
        "total_guests":                    total_guests_v,
        "is_repeated_guest":               is_repeated,
        "previous_cancellations":          prev_cancels,
        "previous_bookings_not_canceled":  prev_not_cancel,
        "booking_changes":                 booking_changes,
        "days_in_waiting_list":            waiting_days,
        "adr":                             adr_v,
        "required_car_parking_spaces":     parking,
        "total_of_special_requests":       special_req,
        "room_type_match":                 room_match_v,
        "is_high_season":                  is_high_season_v,
    }

    row = {}
    row.update(num_values)
    for col_name, enc_col in [
        ("hotel","hotel_enc"),("arrival_date_month","arrival_date_month_enc"),
        ("meal","meal_enc"),("market_segment","market_segment_enc"),
        ("distribution_channel","distribution_channel_enc"),
        ("reserved_room_type","reserved_room_type_enc"),
        ("assigned_room_type","assigned_room_type_enc"),
        ("deposit_type","deposit_type_enc"),
        ("customer_type","customer_type_enc"),
    ]:
        row[enc_col] = cat_values[col_name]

    X_input = pd.DataFrame([row])[feature_names]

    if st.button("🎯 Predict Cancellation Probability", type="primary"):
        prob = model.predict_proba(X_input)[0, 1]
        pred = "Canceled ❌" if prob >= 0.5 else "Not Canceled ✅"
        risk_color = CLR_RED if prob >= 0.5 else CLR_GREEN

        st.markdown("---")
        c_res1, c_res2, c_res3 = st.columns(3)
        c_res1.metric("Prediction", pred)
        c_res2.metric("Cancel Probability", f"{prob*100:.1f}%")
        c_res3.metric("Confidence",
                      "High" if abs(prob - 0.5) > 0.3 else
                      "Medium" if abs(prob - 0.5) > 0.1 else "Low")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Cancellation Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 30],  "color": "#d1fae5"},
                    {"range": [30, 60], "color": "#fef3c7"},
                    {"range": [60, 100],"color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 3},
                    "thickness": 0.75, "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(height=320)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.info(
            "⚠️ **Prediction vs Causation:** This probability is a statistical "
            "estimate based on patterns in historical data. It does not imply "
            "that any individual input *causes* cancellation. Use as a risk "
            "signal, not a definitive outcome."
        )

    st.markdown("---")
    st.subheader("Model Performance Summary")
    perf_data = {
        "Model":     ["Logistic Regression", "Random Forest", "Gradient Boosting ✓ (Best)"],
        "Accuracy":  [0.7764, 0.8414, 0.8359],
        "Precision": [0.6884, 0.7941, 0.7712],
        "Recall":    [0.7305, 0.7753, 0.7957],
        "F1":        [0.7088, 0.7846, 0.7832],
        "ROC-AUC":   [0.8550, 0.9149, 0.9157],
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 6 — EXPLAINABILITY
# ---------------------------------------------------------------------------

def tab_explainability():
    st.header("🔬 Model Explainability")

    st.info(
        "**Prediction vs Causation — Important Caveat**\n\n"
        "All SHAP values and importance scores describe *statistical associations* "
        "between features and the model's output. They do **not** establish that "
        "changing a feature value would *cause* a booking to cancel. "
        "Confounding factors, selection bias, and unmeasured variables may "
        "explain these associations. See `explainability_report.txt` for full details."
    )

    # Method comparison heatmap
    st.subheader("Feature Importance — Method Comparison (MDI / SHAP / Permutation)")
    rank_df = load_ranking_comparison()

    top15 = rank_df.head(15).copy()
    fig_cmp = px.imshow(
        top15[["mdi_rank", "shap_rank", "perm_rank"]].values,
        x=["MDI Rank\n(RF)", "SHAP Rank\n(RF)", "Permutation\n(HistGBM)"],
        y=top15["label"].tolist(),
        color_continuous_scale="YlOrRd",
        text_auto=True,
        title="Feature Importance Rank Comparison — Top 15 (lower rank = more important)",
        aspect="auto",
    )
    fig_cmp.update_layout(height=460)
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    # SHAP bar from CSV
    shap_df = load_feature_importance().head(20)
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=True)
    fig_shap = px.bar(
        shap_df, x="mean_abs_shap", y="label", orientation="h",
        color="mean_abs_shap", color_continuous_scale=["#bfdbfe", CLR_BLUE],
        title="SHAP Mean |Value| — Top 20 Features\n(Random Forest, class=Canceled)",
        labels={"mean_abs_shap": "Mean |SHAP Value|", "label": "Feature"},
    )
    fig_shap.update_layout(height=520, coloraxis_showscale=False)
    col_l.plotly_chart(fig_shap, use_container_width=True)

    # Permutation importance
    perm_df = load_perm_importance().head(20)
    perm_df = perm_df.sort_values("importance_mean", ascending=True)
    fig_perm = px.bar(
        perm_df, x="importance_mean", y="label", orientation="h",
        error_x="importance_std",
        color="importance_mean", color_continuous_scale=["#bfdbfe", CLR_PURPLE],
        title="Permutation Importance (ROC-AUC Drop) — Top 20\n(HistGradientBoosting)",
        labels={"importance_mean": "Mean ROC-AUC Decrease", "label": "Feature"},
    )
    fig_perm.update_layout(height=520, coloraxis_showscale=False)
    col_r.plotly_chart(fig_perm, use_container_width=True)

    st.markdown("---")
    st.subheader("SHAP Beeswarm — Top 20 Features")
    beeswarm_path = PLOTS_DIR / "EX_shap_summary_beeswarm.png"
    if beeswarm_path.exists():
        st.image(str(beeswarm_path), use_container_width=True)
    else:
        st.warning("Beeswarm plot not found. Run explainability.py first.")

    st.markdown("---")
    col_dep, col_wf = st.columns(2)

    col_dep.subheader("SHAP Dependence — Top 4 Numeric Features")
    dep_path = PLOTS_DIR / "EX_shap_dependence_top4.png"
    if dep_path.exists():
        col_dep.image(str(dep_path), use_container_width=True)

    col_wf.subheader("Partial Dependence Plots")
    pdp_path = PLOTS_DIR / "EX_partial_dependence.png"
    if pdp_path.exists():
        col_wf.image(str(pdp_path), use_container_width=True)

    st.markdown("---")
    st.subheader("Individual Prediction Waterfall Charts")
    c_wf1, c_wf2 = st.columns(2)
    wf_cancel = PLOTS_DIR / "EX_shap_waterfall_cancel.png"
    wf_nocancel = PLOTS_DIR / "EX_shap_waterfall_nocancel.png"
    if wf_cancel.exists():
        c_wf1.markdown("**Representative Canceled Booking** (predicted prob = 1.000)")
        c_wf1.image(str(wf_cancel), use_container_width=True)
    if wf_nocancel.exists():
        c_wf2.markdown("**Representative Not-Canceled Booking** (predicted prob = 0.005)")
        c_wf2.image(str(wf_nocancel), use_container_width=True)

    st.markdown("---")
    st.subheader("Key Feature Interpretations")
    interpretations = {
        "🏦 Deposit Type (#1 across all methods)":
            "Non-Refundable bookings show 99.4% cancellation rate in the data. "
            "This is likely an OTA pricing policy artefact — not evidence that "
            "changing deposit type prevents cancellation.",
        "⏰ Lead Time (#2 across all methods)":
            "Longer lead times are strongly associated with higher cancellation risk "
            "(monotonic positive relationship in PDP). Reflects uncertainty in long-range plans.",
        "✨ Special Requests (#3 across all methods)":
            "Guests with ≥1 special request cancel at 22% vs 48% for zero requests. "
            "Likely reflects committed, engaged guests — not a lever that can be artificially triggered.",
        "🏷️ Market Segment (#4)":
            "Groups cancel at 61.2%, Online TA at 36.9%, Direct at 15.5%. "
            "The model strongly discriminates between segment types.",
        "🛏️ Previous Cancellations (#5–6)":
            "Strongest behavioral signal. Even 1 prior cancellation meaningfully "
            "increases predicted probability. Genuine predictive history, not modifiable.",
        "🚗 Car Parking Requested (#4 Permutation)":
            "Guests who request parking cancel less often. Likely proxy for committed, "
            "specific-need travelers. Not a causal mechanism.",
    }
    for title, text in interpretations.items():
        with st.expander(title):
            st.write(text)


# ---------------------------------------------------------------------------
# TAB 7 — AI INSIGHTS
# ---------------------------------------------------------------------------

def tab_ai_insights():
    st.header("🤖 AI Insights")

    # ── API Key status banner ────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        st.warning(
            "**OPENAI_API_KEY not configured.** "
            "Copy `.env.example` to `.env`, add your key, and restart the app. "
            "All metrics and charts on the other tabs remain fully accurate — "
            "this tab only adds AI-generated natural language explanations."
        )
        st.info(
            "💡 **What this tab does:** It sends *only* the validated metrics "
            "already computed in this project (SQL KPIs, model scores, SHAP rankings) "
            "to the OpenAI API. The AI is instructed to use *only* those numbers — "
            "it cannot invent statistics or add outside knowledge."
        )
        with st.expander("Preview: What data is sent to the AI?"):
            kpis = _load_core_kpis()
            if kpis:
                st.code(build_overview_prompt(kpis), language="text")
            else:
                st.write("Run `sql_analytics.py` to generate KPI data.")
        return

    st.success(f"✅ OpenAI API key loaded. Model: `{os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}`")
    st.info(
        "🔒 **Grounding guarantee:** The AI is given only validated metrics from "
        "this project's SQL results, ML evaluation, and SHAP rankings. It is "
        "explicitly instructed not to add statistics, benchmarks, or external "
        "knowledge beyond what is provided."
    )

    st.markdown("---")

    # ── Section selector ─────────────────────────────────────────────────────
    insight_topic = st.selectbox(
        "Choose an insight topic",
        [
            "📊 Executive Overview (KPIs)",
            "❌ Cancellation Drivers",
            "💰 Revenue & ADR Performance",
            "🤖 ML Model Performance",
            "🔬 Feature Explainability",
            "💬 Custom Question",
        ],
    )

    # Show the data context that will be sent
    with st.expander("🔍 View data being sent to AI (transparency)"):
        if insight_topic == "📊 Executive Overview (KPIs)":
            st.code(build_overview_prompt(_load_core_kpis()), language="text")
        elif insight_topic == "❌ Cancellation Drivers":
            st.code(build_cancellation_prompt(_load_cancellation_context()), language="text")
        elif insight_topic == "💰 Revenue & ADR Performance":
            st.code(build_revenue_prompt(_load_revenue_context()), language="text")
        elif insight_topic == "🤖 ML Model Performance":
            st.code(build_ml_prompt(_load_ml_context()), language="text")
        elif insight_topic == "🔬 Feature Explainability":
            st.code(build_explainability_prompt(_load_explainability_context()), language="text")
        else:
            st.write("Custom prompt will be shown after you enter your question below.")

    st.markdown("---")

    # ── Custom question input ────────────────────────────────────────────────
    custom_question = ""
    custom_context  = {}
    if insight_topic == "💬 Custom Question":
        st.subheader("Custom Question")
        custom_question = st.text_input(
            "Ask a question about the hotel bookings data",
            placeholder="e.g. Which market segment has the best balance of low cancellations and high ADR?",
            max_chars=200,
        )
        # Pre-populate context with core KPIs + segment data
        kpis = _load_core_kpis()
        seg_df = load_sql("Q07_Revenue_and_ADR_by_Market_Segment_Checked-Out_Bookings.csv")
        cancel_seg = load_sql("Q03_Cancellation_Rate_by_Market_Segment.csv")
        if not seg_df.empty and not cancel_seg.empty:
            merged = seg_df.merge(cancel_seg, on="market_segment", how="left")
            for _, row in merged.iterrows():
                k = row["market_segment"]
                custom_context[f"{k} — ADR"]             = f"${row['avg_adr']:.2f}"
                custom_context[f"{k} — cancel rate"]     = f"{row['cancellation_rate_pct']:.1f}%"
                custom_context[f"{k} — revenue share"]   = f"{row['revenue_share_pct']:.1f}%"
        if kpis:
            custom_context["overall cancellation rate"] = f"{kpis['cancellation_rate_pct']:.1f}%"
            custom_context["overall avg ADR"]           = f"${kpis['avg_adr_stayed']:.2f}"

    # ── Generate button ──────────────────────────────────────────────────────
    btn_label = "💡 Generate AI Insight"
    if st.button(btn_label, type="primary"):
        with st.spinner("Calling OpenAI API…"):
            if insight_topic == "📊 Executive Overview (KPIs)":
                result = get_overview_insight()
            elif insight_topic == "❌ Cancellation Drivers":
                result = get_cancellation_insight()
            elif insight_topic == "💰 Revenue & ADR Performance":
                result = get_revenue_insight()
            elif insight_topic == "🤖 ML Model Performance":
                result = get_ml_insight()
            elif insight_topic == "🔬 Feature Explainability":
                result = get_explainability_insight()
            elif insight_topic == "💬 Custom Question":
                if not custom_question.strip():
                    st.error("Please enter a question first.")
                    st.stop()
                result = get_custom_insight(custom_question, custom_context)
            else:
                result = FALLBACK_MSG

        st.markdown("---")
        st.subheader("AI Insight")
        # Detect fallback / error messages (start with ⚠️)
        if result.startswith("⚠️"):
            st.warning(result)
        else:
            st.markdown(result)
            st.caption(
                "⚠️ This insight is generated by an AI based solely on the validated "
                "metrics shown in the 'View data' expander above. It describes "
                "statistical associations — not causal relationships."
            )

    st.markdown("---")
    st.subheader("All Pre-Computed Insights (batch preview)")
    st.write(
        "Use the button above to generate individual insights on demand. "
        "Below are the validated data summaries used as AI inputs — "
        "these are computed entirely from project artifacts with no AI involvement."
    )

    c1, c2 = st.columns(2)
    kpis = _load_core_kpis()
    if kpis:
        with c1.expander("Core KPIs"):
            for k, v in kpis.items():
                st.write(f"**{k}:** {v}")

    shap_df = load_feature_importance().head(10)
    if not shap_df.empty:
        with c2.expander("Top 10 SHAP Features"):
            st.dataframe(shap_df[["label", "mean_abs_shap"]], hide_index=True,
                         use_container_width=True)


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------

def main():
    st.title("🏨 Hotel Bookings Analytics Dashboard")
    st.caption(
        "Phases 7 & 8 — Interactive Analytics + AI Insights | "
        "Data: hotel_bookings_cleaned.csv | Model: HistGradientBoosting (ROC-AUC 0.9157)"
    )

    df = load_cleaned_data()
    filtered_df = sidebar_filters(df)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview",
        "❌ Cancellation",
        "💰 Revenue & ADR",
        "🔍 Segments",
        "🤖 ML Prediction",
        "🔬 Explainability",
        "💡 AI Insights",
    ])

    with tab1:
        tab_overview(filtered_df)
    with tab2:
        tab_cancellation(filtered_df)
    with tab3:
        tab_revenue(filtered_df)
    with tab4:
        tab_segments(filtered_df)
    with tab5:
        tab_prediction(df)
    with tab6:
        tab_explainability()
    with tab7:
        tab_ai_insights()


if __name__ == "__main__":
    main()
