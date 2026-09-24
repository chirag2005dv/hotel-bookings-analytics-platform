# Preprocessing Decisions — Hotel Bookings Dataset
## Phase 2: Data Cleaning & Preparation

**Pipeline script:** `data_cleaning.py`  
**Input:** `hotel_bookings.csv` (119,390 rows × 32 columns)  
**Output:** `hotel_bookings_cleaned.csv`  
**Audit log:** `cleaning_report.txt`

---

## 1. Schema Validation

- All **32 expected columns** were verified by name before any transformation.
- Column order was enforced to match the canonical schema defined in `EXPECTED_COLUMNS`.
- Any extra columns not in the expected schema are retained but appended at the end.

---

## 2. NULL String Replacement

| Column | Issue | Decision |
|--------|-------|----------|
| `agent` | 16,340 rows stored as literal `"NULL"` string | Replaced with `NaN` before type casting |
| `company` | 112,593 rows stored as literal `"NULL"` string | Replaced with `NaN` before type casting |

**Rationale:** The CSV uses `"NULL"` as a text placeholder rather than a true missing value marker. Without replacement, these values would be treated as valid category labels in downstream analysis.

Empty string (`""`) values across all columns are also replaced with `NaN` as a general safeguard.

---

## 3. Data Type Casting

All columns are loaded as `str` (via `dtype=str`) to prevent pandas from silently misinterpreting mixed-type fields. They are then cast explicitly:

| Target Type | Columns |
|-------------|---------|
| `Int64` (nullable int) | `is_canceled`, `lead_time`, `arrival_date_year`, `arrival_date_week_number`, `arrival_date_day_of_month`, `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `babies`, `is_repeated_guest`, `previous_cancellations`, `previous_bookings_not_canceled`, `booking_changes`, `days_in_waiting_list`, `required_car_parking_spaces`, `total_of_special_requests` |
| `float64` | `adr`, `children` |
| `category` | `hotel`, `arrival_date_month`, `meal`, `country`, `market_segment`, `distribution_channel`, `reserved_room_type`, `assigned_room_type`, `deposit_type`, `customer_type`, `reservation_status` |
| `datetime64` | `reservation_status_date` |
| `str` (kept as-is) | `agent`, `company` (ID/label columns) |

**Special case — `children`:** Cast to `float64` (not `Int64`) because:
- It contains `"NA"` strings (4 rows) that must become `NaN`
- The value `10` is retained as a numeric (flagged as suspicious but not removed — could theoretically be a group booking)
- pandas `Int64` nullable integers cannot represent `NaN` without a subsequent fill

---

## 4. Missing Value Handling

| Column | Null Count | Fill Strategy | Rationale |
|--------|-----------|---------------|-----------|
| `country` | 488 (0.41%) | `"Unknown"` | Too small to justify row removal; preserves full booking record |
| `agent` | 16,340 (13.7%) | `"0"` | Sentinel value meaning "no agent"; consistent with published dataset conventions (Nuno Antonio et al.) |
| `company` | 112,593 (94.3%) | `"0"` | 94% null — filling with `"0"` = "no company affiliation"; dropping would lose nearly all data |
| `children` | 4 (0.003%) | `0` | Modal value; only 4 rows affected |

**What was NOT imputed:**  
- No statistical imputation (mean/median/mode) was applied to `adr`, `lead_time`, or other numeric fields — these had no missing values after NULL string replacement.

---

## 5. Invalid Row Removal

Rows removed outright (not flagged) because they are logically inconsistent:

| Rule | Count Removed | Rationale |
|------|--------------|-----------|
| `stays_in_weekend_nights == 0 AND stays_in_week_nights == 0` | ~715 rows | A booking with zero stay duration is not a valid hotel stay |
| `adults == 0 AND children == 0 AND babies == 0` | ~403 rows (after zero-night removal) | No guests = not a valid booking |
| `adr < 0` | 1 row (ADR = –6.38) | Negative revenue is meaningless; single clear data entry error |
| `adr > 5,400` | 0 rows | Hard-cap safety net; no rows exceeded this threshold |

**Total rows removed: ~1,118** (all invalid by business logic, not statistical criteria).

---

## 6. Outlier Flagging (Non-Destructive)

Rather than removing potentially genuine high-value or long-range bookings, three binary flag columns were added:

| Flag Column | Condition | Rows Flagged |
|-------------|-----------|-------------|
| `flag_adr_outlier` | `adr > mean + 3σ` | Computed at runtime |
| `flag_long_lead_time` | `lead_time > 365 days` | Computed at runtime |
| `flag_zero_adr` | `adr == 0` AND `market_segment != 'Complementary'` | Computed at runtime |

**Rationale:** The ADR=5,400 row (Canceled, Non Refund, City Hotel) is an extreme outlier but is a real booking record. Flagging preserves it for inclusion/exclusion decisions in Phase 3 modelling without permanently removing it.

---

## 7. Feature Engineering

All new columns are **derived purely from existing fields** — no data is fabricated:

| New Column | Formula | Purpose |
|------------|---------|---------|
| `total_nights` | `stays_in_weekend_nights + stays_in_week_nights` | Core stay-length metric |
| `arrival_date` | Parsed from year + month name + day | Enables time-series analysis |
| `arrival_month_num` | Integer 1–12 mapped from month name | Sortable/numeric month for ML |
| `room_type_match` | `1` if `reserved_room_type == assigned_room_type` | Measures room upgrade/downgrade |
| `is_high_season` | `1` if arrival month ∈ {June, July, August} | Seasonality binary flag |
| `revenue_estimate` | `adr × total_nights` | Proxy for booking value |
| `total_guests` | `adults + children + babies` | Guest count per booking |

---

## 8. What Was Deliberately NOT Changed

| Item | Reason |
|------|--------|
| `meal = "Undefined"` (1,169 rows) | Not removed — "Undefined" is a valid observed category, not a missing value |
| `market_segment = "Undefined"` (2 rows) | Retained — negligible count |
| `children = 10` (1 row) | Retained as numeric — suspicious but not provably wrong |
| `deposit_type` encoding | Left as categorical strings — no re-encoding in cleaning phase |
| `is_canceled` vs `reservation_status` discrepancy | Both columns retained; the relationship is documented for Phase 3 modelling (No-Show rows have `is_canceled=1` but `reservation_status=No-Show`) |
| Duplicate detection | No exact-row duplicates were found in Phase 1 sampling; full deduplication deferred to Phase 3 if required |

---

## 9. Original Data Preservation

- **`hotel_bookings.csv`** is never modified — it is opened read-only.
- All transformations produce a new file: **`hotel_bookings_cleaned.csv`**.
- The audit log **`cleaning_report.txt`** records every transformation with counts.

---

## 10. Reproducibility

The pipeline is fully deterministic:
- No random sampling, shuffling, or seed-dependent operations are performed in this phase.
- Re-running `python data_cleaning.py` on the same input will always produce identical output.
- All thresholds (ADR cap, lead time flag, season months) are defined as named constants at the top of `data_cleaning.py`.
