-- =============================================================================
-- hotel_queries.sql
-- Phase 4 — SQL Analytics Library
-- Hotel Bookings Dataset
--
-- Engine  : SQLite 3.39+ (via Python sqlite3)
-- Table   : bookings  (loaded from hotel_bookings_cleaned.csv)
-- Runner  : sql_analytics.py
--
-- Sections
-- --------
--   Q01  Core KPI Snapshot
--   Q02  Cancellation Rate by Hotel Type
--   Q03  Cancellation Rate by Market Segment
--   Q04  Cancellation Rate by Deposit Type
--   Q05  Cancellation Rate by Lead Time Bucket
--   Q06  Cancellation Rate by Customer Type
--   Q07  Revenue & ADR by Market Segment
--   Q08  Monthly ADR Trend (Both Hotels)
--   Q09  Monthly Booking Volume & Cancellation Rate
--   Q10  Year-over-Year Revenue & ADR Trends
--   Q11  Top 15 Countries by Booking Volume & Cancellation Rate
--   Q12  Booking Channel (Distribution Channel) Performance
--   Q13  Length of Stay Analysis by Hotel & Segment
--   Q14  Repeat vs New Guest — Cancellation & ADR
--   Q15  Room Type Upgrade/Downgrade Rate by Hotel
--   Q16  Deposit Type — Revenue Impact
--   Q17  Lead Time vs Revenue Correlation Buckets
--   Q18  Special Requests Impact on Cancellation
--   Q19  Month-over-Month Booking Growth (Year × Month)
--   Q20  High-Value Booking Segment (Top 10% Revenue Bookings)
-- =============================================================================


-- ---------------------------------------------------------------------------
-- Q01  Core KPI Snapshot
-- ---------------------------------------------------------------------------
-- Purpose: Single-row summary of the most important business metrics
--          across the entire dataset (checked-out bookings only for revenue).
-- ---------------------------------------------------------------------------
-- @query_id: Q01
-- @title: Core KPI Snapshot
SELECT
    COUNT(*)                                                   AS total_bookings,
    SUM(is_canceled)                                           AS total_canceled,
    ROUND(AVG(CAST(is_canceled AS REAL)) * 100, 2)             AS cancellation_rate_pct,
    COUNT(CASE WHEN reservation_status = 'Check-Out' THEN 1 END) AS total_checked_out,
    COUNT(CASE WHEN reservation_status = 'No-Show'   THEN 1 END) AS total_no_show,
    ROUND(AVG(CASE WHEN is_canceled = 0 THEN adr END), 2)      AS avg_adr_stayed,
    ROUND(SUM(CASE WHEN is_canceled = 0 THEN revenue_estimate ELSE 0 END), 2)
                                                               AS total_revenue_estimate,
    ROUND(AVG(CAST(lead_time AS REAL)), 1)                     AS avg_lead_time_days,
    ROUND(AVG(CASE WHEN is_canceled = 0 THEN total_nights END), 2)
                                                               AS avg_length_of_stay,
    ROUND(AVG(CAST(is_repeated_guest AS REAL)) * 100, 2)       AS repeat_guest_rate_pct,
    ROUND(AVG(CAST(room_type_match AS REAL)) * 100, 2)         AS room_match_rate_pct
FROM bookings;


-- ---------------------------------------------------------------------------
-- Q02  Cancellation Rate by Hotel Type
-- ---------------------------------------------------------------------------
-- Purpose: Compare cancellation behaviour between City and Resort Hotels.
-- ---------------------------------------------------------------------------
-- @query_id: Q02
-- @title: Cancellation Rate by Hotel Type
SELECT
    hotel,
    COUNT(*)                                                   AS total_bookings,
    SUM(is_canceled)                                           AS canceled,
    COUNT(*) - SUM(is_canceled)                                AS not_canceled,
    ROUND(AVG(CAST(is_canceled AS REAL)) * 100, 2)             AS cancellation_rate_pct,
    ROUND(AVG(CASE WHEN is_canceled = 0 THEN adr END), 2)      AS avg_adr_stayed,
    ROUND(AVG(CASE WHEN is_canceled = 0 THEN total_nights END), 2)
                                                               AS avg_nights_stayed
FROM bookings
GROUP BY hotel
ORDER BY cancellation_rate_pct DESC;


-- ---------------------------------------------------------------------------
-- Q03  Cancellation Rate by Market Segment
-- ---------------------------------------------------------------------------
-- Purpose: Identify which booking channels carry the highest cancellation risk.
-- ---------------------------------------------------------------------------
-- @query_id: Q03
-- @title: Cancellation Rate by Market Segment
SELECT
    market_segment,
    COUNT(*)                                                   AS total_bookings,
    SUM(is_canceled)                                           AS canceled,
    ROUND(AVG(CAST(is_canceled AS REAL)) * 100, 2)             AS cancellation_rate_pct,
    ROUND(AVG(CASE WHEN is_canceled = 0 THEN adr END), 2)      AS avg_adr_stayed,
    ROUND(AVG(CAST(lead_time AS REAL)), 1)                     AS avg_lead_time_days
FROM bookings
WHERE market_segment NOT IN ('Undefined')
GROUP BY market_segment
ORDER BY cancellation_rate_pct DESC;


-- ---------------------------------------------------------------------------
-- Q04  Cancellation Rate by Deposit Type
-- ---------------------------------------------------------------------------
-- Purpose: Expose the paradox that Non Refund deposits correlate with
--          near-100% cancellation, while No Deposit is far lower.
-- ---------------------------------------------------------------------------
-- @query_id: Q04
-- @title: Cancellation Rate by Deposit Type
SELECT
    deposit_type,
    COUNT(*)                                                   AS total_bookings,
    SUM(is_canceled)                                           AS canceled,
    ROUND(AVG(CAST(is_canceled AS REAL)) * 100, 2)             AS cancellation_rate_pct,
    ROUND(AVG(CASE WHEN is_canceled = 0 THEN adr END), 2)      AS avg_adr_stayed,
    ROUND(AVG(CAST(lead_time AS REAL)), 1)                     AS avg_lead_time_days
FROM bookings
GROUP BY deposit_type
ORDER BY cancellation_rate_pct DESC;


-- ---------------------------------------------------------------------------
-- Q05  Cancellation Rate by Lead Time Bucket
-- ---------------------------------------------------------------------------
-- Purpose: Quantify how cancellation probability grows with booking horizon.
-- Lead time buckets: same-week, monthly, quarterly, half-year, annual, 1yr+
-- ---------------------------------------------------------------------------
-- @query_id: Q05
-- @title: Cancellation Rate by Lead Time Bucket
SELECT
    CASE
        WHEN lead_time BETWEEN 0   AND 7   THEN '01_0-7d'
        WHEN lead_time BETWEEN 8   AND 30  THEN '02_8-30d'
        WHEN lead_time BETWEEN 31  AND 90  THEN '03_31-90d'
        WHEN lead_time BETWEEN 91  AND 180 THEN '04_91-180d'
        WHEN lead_time BETWEEN 181 AND 365 THEN '05_181-365d'
        ELSE                                    '06_365d+'
    END                                                        AS lead_time_bucket,
    COUNT(*)                                                   AS total_bookings,
    SUM(is_canceled)                                           AS canceled,
    ROUND(AVG(CAST(is_canceled AS REAL)) * 100, 2)             AS cancellation_rate_pct,
    ROUND(AVG(CAST(lead_time AS REAL)), 1)                     AS avg_lead_time_days
FROM bookings
GROUP BY lead_time_bucket
ORDER BY lead_time_bucket;


-- ---------------------------------------------------------------------------
-- Q06  Cancellation Rate by Customer Type
-- ---------------------------------------------------------------------------
-- Purpose: Understand whether Transient, Group, or Contract customers
--          cancel at meaningfully different rates.
-- ---------------------------------------------------------------------------
-- @query_id: Q06
-- @title: Cancellation Rate by Customer Type
SELECT
    customer_type,
    COUNT(*)                                                   AS total_bookings,
    SUM(is_canceled)                                           AS canceled,
    ROUND(AVG(CAST(is_canceled AS REAL)) * 100, 2)             AS cancellation_rate_pct,
    ROUND(AVG(CASE WHEN is_canceled = 0 THEN adr END), 2)      AS avg_adr_stayed,
    ROUND(AVG(CAST(lead_time AS REAL)), 1)                     AS avg_lead_time_days,
    ROUND(AVG(CASE WHEN is_canceled = 0 THEN total_nights END), 2)
                                                               AS avg_nights_stayed
FROM bookings
GROUP BY customer_type
ORDER BY cancellation_rate_pct DESC;


-- ---------------------------------------------------------------------------
-- Q07  Revenue & ADR by Market Segment
-- ---------------------------------------------------------------------------
-- Purpose: Rank segments by actual revenue generated (checked-out only).
-- ---------------------------------------------------------------------------
-- @query_id: Q07
-- @title: Revenue and ADR by Market Segment (Checked-Out Bookings)
SELECT
    market_segment,
    COUNT(*)                                                   AS stayed_bookings,
    ROUND(SUM(revenue_estimate), 2)                            AS total_revenue,
    ROUND(AVG(revenue_estimate), 2)                            AS avg_revenue_per_booking,
    ROUND(AVG(adr), 2)                                         AS avg_adr,
    ROUND(AVG(CAST(total_nights AS REAL)), 2)                  AS avg_nights,
    ROUND(SUM(revenue_estimate) * 100.0
          / SUM(SUM(revenue_estimate)) OVER (), 2)             AS revenue_share_pct
FROM bookings
WHERE is_canceled = 0
  AND market_segment NOT IN ('Undefined')
GROUP BY market_segment
ORDER BY total_revenue DESC;


-- ---------------------------------------------------------------------------
-- Q08  Monthly ADR Trend by Hotel Type
-- ---------------------------------------------------------------------------
-- Purpose: Reveal seasonality in pricing — when do hotels command premium rates?
-- ---------------------------------------------------------------------------
-- @query_id: Q08
-- @title: Monthly ADR Trend by Hotel Type
SELECT
    arrival_date_month,
    arrival_month_num,
    ROUND(AVG(CASE WHEN hotel = 'City Hotel'   THEN adr END), 2) AS city_hotel_adr,
    ROUND(AVG(CASE WHEN hotel = 'Resort Hotel' THEN adr END), 2) AS resort_hotel_adr,
    ROUND(AVG(adr), 2)                                           AS overall_avg_adr,
    COUNT(*)                                                     AS bookings
FROM bookings
WHERE is_canceled = 0
GROUP BY arrival_date_month, arrival_month_num
ORDER BY arrival_month_num;


-- ---------------------------------------------------------------------------
-- Q09  Monthly Booking Volume & Cancellation Rate
-- ---------------------------------------------------------------------------
-- Purpose: Identify which months drive the most demand and which are
--          most cancellation-prone.
-- ---------------------------------------------------------------------------
-- @query_id: Q09
-- @title: Monthly Booking Volume and Cancellation Rate
SELECT
    arrival_month_num,
    arrival_date_month,
    COUNT(*)                                                    AS total_bookings,
    SUM(is_canceled)                                            AS canceled,
    COUNT(*) - SUM(is_canceled)                                 AS checked_out_or_active,
    ROUND(AVG(CAST(is_canceled AS REAL)) * 100, 2)              AS cancellation_rate_pct,
    ROUND(AVG(adr), 2)                                          AS avg_adr_all,
    ROUND(SUM(CASE WHEN is_canceled = 0 THEN revenue_estimate ELSE 0 END), 2)
                                                                AS monthly_revenue_est
FROM bookings
GROUP BY arrival_month_num, arrival_date_month
ORDER BY arrival_month_num;


-- ---------------------------------------------------------------------------
-- Q10  Year-over-Year Revenue & ADR Trends
-- ---------------------------------------------------------------------------
-- Purpose: Track growth in ADR and revenue across 2015, 2016, 2017.
--          Note: 2015 is partial (Jul–Dec) and 2017 is partial (Jan–Aug).
-- ---------------------------------------------------------------------------
-- @query_id: Q10
-- @title: Year-over-Year Revenue and ADR Trends
SELECT
    arrival_date_year,
    COUNT(*)                                                   AS total_bookings,
    SUM(is_canceled)                                           AS canceled,
    COUNT(*) - SUM(is_canceled)                                AS stayed,
    ROUND(AVG(CAST(is_canceled AS REAL)) * 100, 2)             AS cancellation_rate_pct,
    ROUND(AVG(CASE WHEN is_canceled = 0 THEN adr END), 2)      AS avg_adr_stayed,
    ROUND(SUM(CASE WHEN is_canceled = 0 THEN revenue_estimate ELSE 0 END), 2)
                                                               AS total_revenue,
    ROUND(AVG(CASE WHEN is_canceled = 0 THEN revenue_estimate END), 2)
                                                               AS avg_revenue_per_booking
FROM bookings
GROUP BY arrival_date_year
ORDER BY arrival_date_year;


-- ---------------------------------------------------------------------------
-- Q11  Top 15 Countries by Booking Volume & Cancellation Rate
-- ---------------------------------------------------------------------------
-- Purpose: Understand geographic demand distribution and cancel risk by origin.
-- ---------------------------------------------------------------------------
-- @query_id: Q11
-- @title: Top 15 Countries by Booking Volume and Cancellation Rate
SELECT
    country,
    COUNT(*)                                                   AS total_bookings,
    SUM(is_canceled)                                           AS canceled,
    ROUND(AVG(CAST(is_canceled AS REAL)) * 100, 2)             AS cancellation_rate_pct,
    ROUND(AVG(CASE WHEN is_canceled = 0 THEN adr END), 2)      AS avg_adr_stayed,
    ROUND(SUM(CASE WHEN is_canceled = 0 THEN revenue_estimate ELSE 0 END), 2)
                                                               AS total_revenue_est
FROM bookings
WHERE country != 'Unknown'
GROUP BY country
ORDER BY total_bookings DESC
LIMIT 15;


-- ---------------------------------------------------------------------------
-- Q12  Booking Channel (Distribution Channel) Performance
-- ---------------------------------------------------------------------------
-- Purpose: Compare TA/TO, Direct, Corporate, and GDS channels on
--          volume, cancellation, ADR, and revenue.
-- ---------------------------------------------------------------------------
-- @query_id: Q12
-- @title: Distribution Channel Performance
SELECT
    distribution_channel,
    COUNT(*)                                                   AS total_bookings,
    ROUND(AVG(CAST(is_canceled AS REAL)) * 100, 2)             AS cancellation_rate_pct,
    ROUND(AVG(CASE WHEN is_canceled = 0 THEN adr END), 2)      AS avg_adr_stayed,
    ROUND(SUM(CASE WHEN is_canceled = 0 THEN revenue_estimate ELSE 0 END), 2)
                                                               AS total_revenue_est,
    ROUND(AVG(CAST(lead_time AS REAL)), 1)                     AS avg_lead_time_days
FROM bookings
WHERE distribution_channel NOT IN ('Undefined')
GROUP BY distribution_channel
ORDER BY total_bookings DESC;


-- ---------------------------------------------------------------------------
-- Q13  Length of Stay Analysis by Hotel & Market Segment
-- ---------------------------------------------------------------------------
-- Purpose: Identify which guest types stay the longest and which segments
--          drive the most overnight demand.
-- ---------------------------------------------------------------------------
-- @query_id: Q13
-- @title: Length of Stay by Hotel and Market Segment
SELECT
    hotel,
    market_segment,
    COUNT(*)                                                        AS bookings,
    ROUND(AVG(CAST(total_nights AS REAL)), 2)                       AS avg_total_nights,
    ROUND(AVG(CAST(stays_in_weekend_nights AS REAL)), 2)            AS avg_weekend_nights,
    ROUND(AVG(CAST(stays_in_week_nights AS REAL)), 2)               AS avg_week_nights,
    ROUND(AVG(adr), 2)                                              AS avg_adr
FROM bookings
WHERE is_canceled = 0
  AND market_segment NOT IN ('Undefined')
GROUP BY hotel, market_segment
ORDER BY hotel, avg_total_nights DESC;


-- ---------------------------------------------------------------------------
-- Q14  Repeat vs New Guest — Cancellation, ADR, Revenue
-- ---------------------------------------------------------------------------
-- Purpose: Quantify the value and loyalty of repeat guests vs first-timers.
-- ---------------------------------------------------------------------------
-- @query_id: Q14
-- @title: Repeat vs New Guest Performance
SELECT
    CASE is_repeated_guest WHEN 1 THEN 'Repeat Guest' ELSE 'New Guest' END
                                                               AS guest_type,
    COUNT(*)                                                   AS total_bookings,
    ROUND(AVG(CAST(is_canceled AS REAL)) * 100, 2)             AS cancellation_rate_pct,
    ROUND(AVG(CASE WHEN is_canceled = 0 THEN adr END), 2)      AS avg_adr_stayed,
    ROUND(AVG(CASE WHEN is_canceled = 0 THEN total_nights END), 2)
                                                               AS avg_nights_stayed,
    ROUND(SUM(CASE WHEN is_canceled = 0 THEN revenue_estimate ELSE 0 END), 2)
                                                               AS total_revenue_est,
    ROUND(AVG(CAST(total_of_special_requests AS REAL)), 2)     AS avg_special_requests,
    ROUND(AVG(CAST(lead_time AS REAL)), 1)                     AS avg_lead_time_days
FROM bookings
GROUP BY is_repeated_guest
ORDER BY is_repeated_guest DESC;


-- ---------------------------------------------------------------------------
-- Q15  Room Type Upgrade/Downgrade Rate by Hotel
-- ---------------------------------------------------------------------------
-- Purpose: How often do guests receive a different room than they reserved?
--          Upgrades (higher letter) vs downgrades vs exact matches.
-- ---------------------------------------------------------------------------
-- @query_id: Q15
-- @title: Room Type Assignment vs Reservation
SELECT
    hotel,
    CASE
        WHEN reserved_room_type = assigned_room_type THEN 'Exact Match'
        WHEN assigned_room_type < reserved_room_type  THEN 'Upgrade (lower letter = better)'
        ELSE                                               'Downgrade'
    END                                                        AS assignment_type,
    COUNT(*)                                                   AS bookings,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY hotel), 2)
                                                               AS pct_of_hotel,
    ROUND(AVG(adr), 2)                                         AS avg_adr,
    ROUND(AVG(CAST(is_canceled AS REAL)) * 100, 2)             AS cancellation_rate_pct
FROM bookings
GROUP BY hotel, assignment_type
ORDER BY hotel, bookings DESC;


-- ---------------------------------------------------------------------------
-- Q16  Deposit Type — Revenue Impact on Stayed Bookings
-- ---------------------------------------------------------------------------
-- Purpose: Compare actual revenue realised per deposit type to understand
--          whether deposit policy influences booking value.
-- ---------------------------------------------------------------------------
-- @query_id: Q16
-- @title: Deposit Type Revenue Impact (Stayed Bookings Only)
SELECT
    deposit_type,
    COUNT(*)                                                   AS stayed_bookings,
    ROUND(AVG(adr), 2)                                         AS avg_adr,
    ROUND(AVG(CAST(total_nights AS REAL)), 2)                  AS avg_nights,
    ROUND(SUM(revenue_estimate), 2)                            AS total_revenue,
    ROUND(AVG(revenue_estimate), 2)                            AS avg_revenue_per_booking
FROM bookings
WHERE is_canceled = 0
GROUP BY deposit_type
ORDER BY total_revenue DESC;


-- ---------------------------------------------------------------------------
-- Q17  Lead Time Buckets — Revenue & ADR for Stayed Bookings
-- ---------------------------------------------------------------------------
-- Purpose: Do advance planners pay more or less per night than last-minute
--          bookers? Important for pricing strategy.
-- ---------------------------------------------------------------------------
-- @query_id: Q17
-- @title: Lead Time Buckets vs Revenue and ADR (Stayed Bookings)
SELECT
    CASE
        WHEN lead_time BETWEEN 0   AND 7   THEN '01_0-7d (same-week)'
        WHEN lead_time BETWEEN 8   AND 30  THEN '02_8-30d (1 month)'
        WHEN lead_time BETWEEN 31  AND 90  THEN '03_31-90d (quarter)'
        WHEN lead_time BETWEEN 91  AND 180 THEN '04_91-180d (half-year)'
        WHEN lead_time BETWEEN 181 AND 365 THEN '05_181-365d (annual)'
        ELSE                                    '06_365d+ (long-range)'
    END                                                        AS lead_time_bucket,
    COUNT(*)                                                   AS stayed_bookings,
    ROUND(AVG(adr), 2)                                         AS avg_adr,
    ROUND(AVG(CAST(total_nights AS REAL)), 2)                  AS avg_nights,
    ROUND(AVG(revenue_estimate), 2)                            AS avg_revenue_per_booking,
    ROUND(SUM(revenue_estimate), 2)                            AS total_revenue
FROM bookings
WHERE is_canceled = 0
GROUP BY lead_time_bucket
ORDER BY lead_time_bucket;


-- ---------------------------------------------------------------------------
-- Q18  Special Requests Impact on Cancellation & ADR
-- ---------------------------------------------------------------------------
-- Purpose: Test whether engaged guests (more special requests) are more
--          likely to show up and spend more.
-- ---------------------------------------------------------------------------
-- @query_id: Q18
-- @title: Special Requests vs Cancellation Rate and ADR
SELECT
    total_of_special_requests,
    COUNT(*)                                                   AS total_bookings,
    SUM(is_canceled)                                           AS canceled,
    ROUND(AVG(CAST(is_canceled AS REAL)) * 100, 2)             AS cancellation_rate_pct,
    ROUND(AVG(CASE WHEN is_canceled = 0 THEN adr END), 2)      AS avg_adr_stayed,
    ROUND(AVG(CASE WHEN is_canceled = 0 THEN total_nights END), 2)
                                                               AS avg_nights_stayed
FROM bookings
GROUP BY total_of_special_requests
ORDER BY total_of_special_requests;


-- ---------------------------------------------------------------------------
-- Q19  Month-over-Month Booking Cohort (Year × Month Grid)
-- ---------------------------------------------------------------------------
-- Purpose: Build a year × month grid of booking volumes and ADR
--          to spot seasonal patterns year-over-year.
-- ---------------------------------------------------------------------------
-- @query_id: Q19
-- @title: Year x Month Booking Volume and ADR Heatmap Grid
SELECT
    arrival_date_year,
    arrival_month_num,
    arrival_date_month,
    COUNT(*)                                                   AS total_bookings,
    SUM(is_canceled)                                           AS canceled,
    ROUND(AVG(CAST(is_canceled AS REAL)) * 100, 2)             AS cancellation_rate_pct,
    ROUND(AVG(CASE WHEN is_canceled = 0 THEN adr END), 2)      AS avg_adr_stayed,
    ROUND(SUM(CASE WHEN is_canceled = 0 THEN revenue_estimate ELSE 0 END), 2)
                                                               AS monthly_revenue_est
FROM bookings
GROUP BY arrival_date_year, arrival_month_num, arrival_date_month
ORDER BY arrival_date_year, arrival_month_num;


-- ---------------------------------------------------------------------------
-- Q20  High-Value Booking Segment (Top-Decile Revenue Bookings)
-- ---------------------------------------------------------------------------
-- Purpose: Profile the top 10% of bookings by revenue estimate — who are
--          the highest-value guests and how do they differ from average?
-- ---------------------------------------------------------------------------
-- @query_id: Q20
-- @title: High-Value Booking Profile (Top 10% by Revenue)
WITH revenue_threshold AS (
    SELECT revenue_estimate
    FROM bookings
    WHERE is_canceled = 0
    ORDER BY revenue_estimate DESC
    LIMIT (SELECT CAST(COUNT(*) * 0.10 AS INT) FROM bookings WHERE is_canceled = 0)
),
threshold_val AS (
    SELECT MIN(revenue_estimate) AS threshold FROM revenue_threshold
)
SELECT
    'Top 10% Revenue Bookings'                                 AS segment,
    COUNT(*)                                                   AS bookings,
    ROUND(AVG(adr), 2)                                         AS avg_adr,
    ROUND(AVG(CAST(total_nights AS REAL)), 2)                  AS avg_nights,
    ROUND(AVG(revenue_estimate), 2)                            AS avg_revenue,
    ROUND(SUM(revenue_estimate), 2)                            AS total_revenue,
    ROUND(AVG(CAST(lead_time AS REAL)), 1)                     AS avg_lead_time,
    ROUND(AVG(CAST(total_of_special_requests AS REAL)), 2)     AS avg_special_requests,
    (SELECT ROUND(threshold, 2) FROM threshold_val)            AS min_revenue_threshold
FROM bookings
WHERE is_canceled = 0
  AND revenue_estimate >= (SELECT threshold FROM threshold_val)

UNION ALL

SELECT
    'Bottom 90% Revenue Bookings'                              AS segment,
    COUNT(*)                                                   AS bookings,
    ROUND(AVG(adr), 2)                                         AS avg_adr,
    ROUND(AVG(CAST(total_nights AS REAL)), 2)                  AS avg_nights,
    ROUND(AVG(revenue_estimate), 2)                            AS avg_revenue,
    ROUND(SUM(revenue_estimate), 2)                            AS total_revenue,
    ROUND(AVG(CAST(lead_time AS REAL)), 1)                     AS avg_lead_time,
    ROUND(AVG(CAST(total_of_special_requests AS REAL)), 2)     AS avg_special_requests,
    NULL                                                       AS min_revenue_threshold
FROM bookings
WHERE is_canceled = 0
  AND revenue_estimate < (SELECT threshold FROM threshold_val);
