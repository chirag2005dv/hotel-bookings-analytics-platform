import pandas as pd

df = pd.read_csv("hotel_bookings_cleaned.csv")

print("=== SHAPE ===")
print(f"Rows: {len(df):,}  Cols: {df.shape[1]}")

print("\n=== DTYPES ===")
for c in df.columns:
    print(f"  {c}: {df[c].dtype}")

print("\n=== NULL CHECK ===")
nulls = df.isna().sum()
has_nulls = nulls[nulls > 0]
print(has_nulls if len(has_nulls) else "  No nulls found.")

print("\n=== KEY METRICS ===")
cancel_rate = df["is_canceled"].mean() * 100
print(f"Cancellation rate  : {cancel_rate:.2f}%")
print(f"ADR min            : {df['adr'].min():.2f}")
print(f"ADR max            : {df['adr'].max():.2f}")
print(f"ADR mean           : {df['adr'].mean():.2f}")
print(f"Lead time min/max  : {df['lead_time'].min()} / {df['lead_time'].max()}")
print(f"total_nights >= 1  : {(df['total_nights'] >= 1).all()}")
print(f"total_guests >= 1  : {(df['total_guests'] >= 1).all()}")
print(f"Negative ADR rows  : {(df['adr'] < 0).sum()}")
zero_nights = ((df['stays_in_weekend_nights'] == 0) & (df['stays_in_week_nights'] == 0)).sum()
print(f"Zero-night rows    : {zero_nights}")

print("\n=== NEW FEATURE SPOT CHECK ===")
new_cols = [
    "total_nights", "arrival_date", "arrival_month_num", "room_type_match",
    "is_high_season", "revenue_estimate", "total_guests",
    "flag_adr_outlier", "flag_long_lead_time", "flag_zero_adr",
]
for c in new_cols:
    print(f"  {c}: nulls={df[c].isna().sum()}  sample={df[c].iloc[0]}")

print("\n=== ROW AUDIT ===")
print(f"Raw rows     : 119,390")
print(f"Cleaned rows : {len(df):,}")
print(f"Removed      : {119390 - len(df):,}")

print("\n=== HOTEL SPLIT ===")
print(df["hotel"].value_counts().to_string())

print("\n=== FLAG SUMMARY ===")
print(f"flag_adr_outlier    : {df['flag_adr_outlier'].sum():,}")
print(f"flag_long_lead_time : {df['flag_long_lead_time'].sum():,}")
print(f"flag_zero_adr       : {df['flag_zero_adr'].sum():,}")

print("\nValidation complete.")
