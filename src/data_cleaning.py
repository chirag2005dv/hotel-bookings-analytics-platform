"""
data_cleaning.py
================
Phase 2 — Data Cleaning & Preprocessing Pipeline
Hotel Bookings Dataset (hotel_bookings.csv)

Preprocessing decisions are documented inline and summarised in
reports/PREPROCESSING_DECISIONS.md.

Usage:
    python src/data_cleaning.py

Output:
    data/processed/hotel_bookings_cleaned.csv   — cleaned dataset
    reports/cleaning_report.txt                 — row-level audit log
"""

import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress pandas FutureWarning for replace() downcasting (resolved in pandas 3.x)
pd.set_option("future.no_silent_downcasting", True)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).parent.parent
INPUT_FILE  = ROOT / "data" / "raw" / "hotel_bookings.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "hotel_bookings_cleaned.csv"
REPORT_FILE = ROOT / "reports" / "cleaning_report.txt"

# ADR cap: values above this are treated as extreme outliers.
# The single row with ADR=5400 (canceled, Non Refund) is 53× the mean;
# 5400 is retained as flagged rather than dropped, but values above
# this hard cap are removed (none exist beyond 5400, so this is a safety net).
ADR_UPPER_CAP = 5400.0

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def section(title: str, width: int = 70) -> str:
    bar = "=" * width
    return f"\n{bar}\n  {title}\n{bar}"


def log(report_lines: list, msg: str) -> None:
    safe = msg.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
        sys.stdout.encoding or "utf-8"
    )
    print(safe)
    report_lines.append(msg)


# ---------------------------------------------------------------------------
# STEP 1 — LOAD RAW DATA
# ---------------------------------------------------------------------------

def load_raw(path: str, report: list) -> pd.DataFrame:
    log(report, section("STEP 1 — LOAD RAW DATA"))
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    log(report, f"  Rows loaded : {len(df):,}")
    log(report, f"  Columns     : {df.shape[1]}")
    log(report, f"  File        : {path}")
    return df


# ---------------------------------------------------------------------------
# STEP 2 — SCHEMA VALIDATION (expected columns & raw types)
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "hotel", "is_canceled", "lead_time", "arrival_date_year",
    "arrival_date_month", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies", "meal",
    "country", "market_segment", "distribution_channel",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "reserved_room_type",
    "assigned_room_type", "booking_changes", "deposit_type", "agent",
    "company", "days_in_waiting_list", "customer_type", "adr",
    "required_car_parking_spaces", "total_of_special_requests",
    "reservation_status", "reservation_status_date",
]


def validate_schema(df: pd.DataFrame, report: list) -> pd.DataFrame:
    log(report, section("STEP 2 — SCHEMA VALIDATION"))

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols   = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    if missing_cols:
        log(report, f"  MISSING columns : {missing_cols}")
        sys.exit("Schema error — missing expected columns. Aborting.")
    if extra_cols:
        log(report, f"  EXTRA columns (will be kept) : {extra_cols}")
    else:
        log(report, "  All 32 expected columns present. ✓")

    # Enforce canonical column order
    df = df[EXPECTED_COLUMNS + extra_cols]
    return df


# ---------------------------------------------------------------------------
# STEP 3 — REPLACE "NULL" STRINGS WITH REAL NaN
# ---------------------------------------------------------------------------

def replace_null_strings(df: pd.DataFrame, report: list) -> pd.DataFrame:
    """
    Decision: The CSV stores missing values as the literal string "NULL"
    for `agent` and `company`. Replace with NaN so downstream logic
    treats them as missing.
    """
    log(report, section("STEP 3 — REPLACE 'NULL' STRINGS WITH NaN"))
    # Replace "NULL" strings globally — the CSV uses this as a missing-value
    # marker for agent, company, and country.
    before_null_str = df.isna().sum().sum()
    df = df.replace("NULL", np.nan).infer_objects(copy=False)
    after_null_str = df.isna().sum().sum()
    log(report, f"  'NULL' string → NaN conversions (all columns): {after_null_str - before_null_str:,}")

    # Also replace empty strings with NaN across all columns
    before = df.isna().sum().sum()
    df = df.replace("", np.nan).infer_objects(copy=False)
    after  = df.isna().sum().sum()
    log(report, f"  Empty string → NaN conversions: {after - before:,}")
    return df


# ---------------------------------------------------------------------------
# STEP 4 — CAST DATA TYPES
# ---------------------------------------------------------------------------

INTEGER_COLS = [
    "is_canceled", "lead_time", "arrival_date_year",
    "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "babies", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled",
    "booking_changes", "days_in_waiting_list",
    "required_car_parking_spaces", "total_of_special_requests",
]

FLOAT_COLS = ["adr"]

CATEGORICAL_COLS = [
    "hotel", "arrival_date_month", "meal", "country", "market_segment",
    "distribution_channel", "reserved_room_type", "assigned_room_type",
    "deposit_type", "customer_type", "reservation_status",
]


def cast_dtypes(df: pd.DataFrame, report: list) -> pd.DataFrame:
    """
    Decision: children is cast to float (not int) because it contains
    NA strings that are converted to NaN, and pandas int cannot hold NaN.
    agent and company are kept as string (they are IDs / labels).
    """
    log(report, section("STEP 4 — CAST DATA TYPES"))

    # children: coerce 'NA' and '10' alongside numerics
    df["children"] = pd.to_numeric(df["children"], errors="coerce")
    log(report, "  children  → float64 (NA strings → NaN; '10' kept as numeric)")

    for col in INTEGER_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        log(report, f"  {col}  → Int64")

    for col in FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        log(report, f"  {col}  → float64")

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
        log(report, f"  {col}  → category")

    # Parse reservation_status_date as date
    df["reservation_status_date"] = pd.to_datetime(
        df["reservation_status_date"], errors="coerce"
    )
    log(report, "  reservation_status_date → datetime64")

    return df


# ---------------------------------------------------------------------------
# STEP 5 — HANDLE MISSING VALUES
# ---------------------------------------------------------------------------

def handle_missing(df: pd.DataFrame, report: list) -> pd.DataFrame:
    """
    Decisions:
    - country (488 nulls, 0.41%): fill with 'Unknown' — small fraction,
      dropping rows would lose valid booking data.
    - agent (16,340 nulls, 13.7%): fill with 0 (sentinel for 'no agent') —
      consistent with the dataset convention used in published research.
    - company (112,593 nulls, 94.3%): fill with 0 (sentinel for 'no company')
      — nearly all rows are null; 0 = "direct / no company affiliation".
    - children (4 nulls): fill with 0 — only 4 rows; 0 is the modal value.
    """
    log(report, section("STEP 5 — HANDLE MISSING VALUES"))

    fills = {
        "country":  "Unknown",
        "agent":    "0",
        "company":  "0",
        "children": 0,
    }
    for col, fill_val in fills.items():
        n = df[col].isna().sum()
        # Categorical columns need the new value added to their category list first
        if hasattr(df[col], "cat"):
            if fill_val not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories([fill_val])
        df[col] = df[col].fillna(fill_val)
        log(report, f"  {col}: {n:,} nulls filled with {repr(fill_val)}")

    after_nulls = df.isna().sum()
    remaining   = after_nulls[after_nulls > 0]
    if len(remaining):
        log(report, f"  Remaining nulls:\n{remaining.to_string()}")
    else:
        log(report, "  No remaining nulls after filling. ✓")

    return df


# ---------------------------------------------------------------------------
# STEP 6 — INVALID / ANOMALOUS ROW REMOVAL
# ---------------------------------------------------------------------------

def remove_invalid_rows(df: pd.DataFrame, report: list) -> pd.DataFrame:
    """
    Decisions:
    - Zero-night stays (both stays_in_weekend_nights AND stays_in_week_nights
      = 0): 715 rows. These represent bookings with no stay duration and are
      logically invalid for demand / revenue analysis. Removed.
    - Zero-adult bookings (adults = 0 AND children = 0 AND babies = 0):
      no legitimate reservation can have zero guests. Removed.
    - Rows where ADR is negative: 1 row (ADR = -6.38, Resort Hotel,
      Check-Out). Negative revenue is not meaningful. Removed.
    - ADR > ADR_UPPER_CAP: safety net removal of extreme ADR outliers
      (the ADR=5400 row is a Canceled booking and is retained; values beyond
      the hard cap would be removed if they existed).
    """
    log(report, section("STEP 6 — INVALID / ANOMALOUS ROW REMOVAL"))
    initial = len(df)

    # Zero-night stays
    mask_zero_nights = (df["stays_in_weekend_nights"] == 0) & (df["stays_in_week_nights"] == 0)
    n_zero_nights = mask_zero_nights.sum()
    df = df[~mask_zero_nights].copy()
    log(report, f"  Removed zero-night stays        : {n_zero_nights:,} rows")

    # Zero-guest bookings
    mask_zero_guests = (df["adults"] == 0) & (df["children"].fillna(0) == 0) & (df["babies"] == 0)
    n_zero_guests = mask_zero_guests.sum()
    df = df[~mask_zero_guests].copy()
    log(report, f"  Removed zero-guest bookings     : {n_zero_guests:,} rows")

    # Negative ADR
    mask_neg_adr = df["adr"] < 0
    n_neg_adr = mask_neg_adr.sum()
    df = df[~mask_neg_adr].copy()
    log(report, f"  Removed negative ADR rows       : {n_neg_adr:,} rows")

    # ADR above hard cap (safety net — no rows expected beyond 5400)
    mask_cap_adr = df["adr"] > ADR_UPPER_CAP
    n_cap_adr = mask_cap_adr.sum()
    if n_cap_adr:
        df = df[~mask_cap_adr].copy()
    log(report, f"  Removed ADR > {ADR_UPPER_CAP} rows        : {n_cap_adr:,} rows")

    removed = initial - len(df)
    log(report, f"  Total rows removed              : {removed:,}")
    log(report, f"  Rows remaining                  : {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# STEP 7 — OUTLIER FLAGGING (non-destructive)
# ---------------------------------------------------------------------------

def flag_outliers(df: pd.DataFrame, report: list) -> pd.DataFrame:
    """
    Decision: Rather than removing high-ADR or extreme lead-time rows
    (which may be genuine premium bookings or long-range planners), we
    add binary flag columns so downstream analysis can filter them.
    No rows are dropped here.
    """
    log(report, section("STEP 7 — OUTLIER FLAGGING (non-destructive)"))

    # ADR outlier: > mean + 3*std
    adr_mean = df["adr"].mean()
    adr_std  = df["adr"].std()
    adr_upper = adr_mean + 3 * adr_std
    df["flag_adr_outlier"] = (df["adr"] > adr_upper).astype("Int64")
    n_adr = df["flag_adr_outlier"].sum()
    log(report, f"  flag_adr_outlier  (ADR > {adr_upper:.2f}): {n_adr:,} rows")

    # Lead time outlier: > 365 days
    df["flag_long_lead_time"] = (df["lead_time"] > 365).astype("Int64")
    n_lt = df["flag_long_lead_time"].sum()
    log(report, f"  flag_long_lead_time (lead_time > 365)   : {n_lt:,} rows")

    # Zero ADR on non-complimentary bookings (possible data quality)
    mask_zero_adr = (df["adr"] == 0) & (~df["market_segment"].isin(["Complementary"]))
    df["flag_zero_adr"] = mask_zero_adr.astype("Int64")
    n_zero = df["flag_zero_adr"].sum()
    log(report, f"  flag_zero_adr (ADR=0, non-compl.)       : {n_zero:,} rows")

    return df


# ---------------------------------------------------------------------------
# STEP 8 — FEATURE ENGINEERING
# ---------------------------------------------------------------------------

MONTH_ORDER = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def engineer_features(df: pd.DataFrame, report: list) -> pd.DataFrame:
    """
    New columns added (all derivable from existing fields — no fabrication):

    total_nights          : stays_in_weekend_nights + stays_in_week_nights
    arrival_date          : combined arrival year/month/day as datetime
    arrival_month_num     : integer month (1–12) for sorting/modelling
    room_type_match       : 1 if reserved_room_type == assigned_room_type
    is_high_season        : 1 if arrival month is June, July, or August
    revenue_estimate      : adr × total_nights (proxy for booking value)
    total_guests          : adults + children + babies
    """
    log(report, section("STEP 8 — FEATURE ENGINEERING"))

    # total_nights
    df["total_nights"] = (
        df["stays_in_weekend_nights"].astype(float) +
        df["stays_in_week_nights"].astype(float)
    ).astype("Int64")
    log(report, "  total_nights = stays_in_weekend_nights + stays_in_week_nights")

    # arrival_date
    df["arrival_month_num"] = df["arrival_date_month"].map(MONTH_ORDER).astype("Int64")
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" +
        df["arrival_month_num"].astype(str).str.zfill(2) + "-" +
        df["arrival_date_day_of_month"].astype(str).str.zfill(2),
        errors="coerce",
    )
    invalid_dates = df["arrival_date"].isna().sum()
    log(report, f"  arrival_date constructed ({invalid_dates} parse failures)")

    # room_type_match
    df["room_type_match"] = (
        df["reserved_room_type"].astype(str) == df["assigned_room_type"].astype(str)
    ).astype("Int64")
    match_rate = df["room_type_match"].mean() * 100
    log(report, f"  room_type_match: {match_rate:.1f}% of bookings got their requested room type")

    # is_high_season (June, July, August)
    df["is_high_season"] = df["arrival_month_num"].isin([6, 7, 8]).astype("Int64")
    n_hs = df["is_high_season"].sum()
    log(report, f"  is_high_season (Jun–Aug): {n_hs:,} rows ({n_hs/len(df)*100:.1f}%)")

    # revenue_estimate
    df["revenue_estimate"] = (df["adr"] * df["total_nights"].astype(float)).round(2)
    log(report, f"  revenue_estimate = adr × total_nights  (mean: {df['revenue_estimate'].mean():.2f})")

    # total_guests
    df["total_guests"] = (
        df["adults"].astype(float) +
        df["children"].astype(float).fillna(0) +
        df["babies"].astype(float)
    ).astype("Int64")
    log(report, f"  total_guests = adults + children + babies  (mean: {df['total_guests'].mean():.2f})")

    return df


# ---------------------------------------------------------------------------
# STEP 9 — FINAL VALIDATION
# ---------------------------------------------------------------------------

def final_validation(df: pd.DataFrame, report: list) -> None:
    log(report, section("STEP 9 — FINAL VALIDATION"))

    log(report, f"  Final row count        : {len(df):,}")
    log(report, f"  Final column count     : {df.shape[1]}")

    nulls = df.isna().sum()
    remaining_nulls = nulls[nulls > 0]
    if len(remaining_nulls):
        log(report, f"  Remaining nulls:\n{remaining_nulls.to_string()}")
    else:
        log(report, "  No unexpected nulls remaining. ✓")

    log(report, f"  is_canceled values     : {dict(df['is_canceled'].value_counts().to_dict())}")
    log(report, f"  ADR range              : {df['adr'].min():.2f} – {df['adr'].max():.2f}")
    log(report, f"  Lead time range        : {df['lead_time'].min()} – {df['lead_time'].max()}")
    log(report, f"  total_nights range     : {df['total_nights'].min()} – {df['total_nights'].max()}")
    log(report, f"  Hotel types            : {dict(df['hotel'].value_counts().to_dict())}")
    log(report, f"  Date range             : {df['arrival_date'].min().date()} – {df['arrival_date'].max().date()}")

    # Confirm no negative values in key numeric columns
    for col in ["adr", "lead_time", "stays_in_weekend_nights", "stays_in_week_nights"]:
        n_neg = (df[col].astype(float) < 0).sum()
        if n_neg:
            log(report, f"  WARNING: {n_neg} negative values in {col}")
        else:
            log(report, f"  {col}: no negative values. ✓")


# ---------------------------------------------------------------------------
# STEP 10 — SAVE OUTPUT
# ---------------------------------------------------------------------------

def save_output(df: pd.DataFrame, path: str, report: list) -> None:
    log(report, section("STEP 10 — SAVE OUTPUT"))
    df.to_csv(path, index=False)
    size_mb = Path(path).stat().st_size / (1024 ** 2)
    log(report, f"  Saved: {path}  ({size_mb:.2f} MB, {len(df):,} rows, {df.shape[1]} columns)")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    report: list = []

    log(report, "Hotel Bookings — Phase 2 Data Cleaning Pipeline")
    log(report, f"pandas {pd.__version__}  |  numpy {np.__version__}")

    df = load_raw(INPUT_FILE, report)
    df = validate_schema(df, report)
    df = replace_null_strings(df, report)
    df = cast_dtypes(df, report)
    df = handle_missing(df, report)
    df = remove_invalid_rows(df, report)
    df = flag_outliers(df, report)
    df = engineer_features(df, report)
    final_validation(df, report)
    save_output(df, str(OUTPUT_FILE), report)

    # Write audit report
    report_text = "\n".join(report)
    REPORT_FILE.write_text(report_text, encoding="utf-8")
    print(f"\nAudit report saved: {REPORT_FILE}")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
