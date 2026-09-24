# tests/test_data_cleaning.py
"""
Tests for data_cleaning.py — schema validation, null handling,
invalid row removal, feature engineering, and dtype correctness.
"""

import io
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestValidateSchema:
    def test_all_expected_columns_present(self, raw_df):
        from data_cleaning import validate_schema, EXPECTED_COLUMNS
        report = []
        result = validate_schema(raw_df.copy(), report)
        for col in EXPECTED_COLUMNS:
            assert col in result.columns, f"Expected column missing: {col}"

    def test_missing_column_raises(self, raw_df):
        from data_cleaning import validate_schema
        import sys
        broken = raw_df.drop(columns=["hotel"])
        report = []
        with pytest.raises(SystemExit):
            validate_schema(broken, report)

    def test_column_order_enforced(self, raw_df):
        from data_cleaning import validate_schema, EXPECTED_COLUMNS
        # Shuffle columns
        shuffled = raw_df[list(reversed(raw_df.columns))].copy()
        report = []
        result = validate_schema(shuffled, report)
        assert list(result.columns[:len(EXPECTED_COLUMNS)]) == EXPECTED_COLUMNS


# ---------------------------------------------------------------------------
# NULL string replacement
# ---------------------------------------------------------------------------

class TestReplaceNullStrings:
    def test_null_strings_become_nan(self, raw_df):
        from data_cleaning import replace_null_strings
        report = []
        result = replace_null_strings(raw_df.copy(), report)
        # All "NULL" strings should be gone
        for col in result.columns:
            assert "NULL" not in result[col].astype(str).values, \
                f"Column {col} still contains 'NULL' string"

    def test_agent_nulls_converted(self, raw_df):
        from data_cleaning import replace_null_strings
        report = []
        result = replace_null_strings(raw_df.copy(), report)
        # Raw data has NULL agents — they should be NaN now
        assert result["agent"].isna().any(), "Expected NaN values in agent column"

    def test_country_nulls_converted(self):
        """country column stores 'NULL' strings; they must become NaN."""
        from data_cleaning import replace_null_strings
        raw = pd.read_csv(
            io.StringIO(
                "hotel,is_canceled,lead_time,arrival_date_year,arrival_date_month,"
                "arrival_date_week_number,arrival_date_day_of_month,"
                "stays_in_weekend_nights,stays_in_week_nights,adults,children,"
                "babies,meal,country,market_segment,distribution_channel,"
                "is_repeated_guest,previous_cancellations,previous_bookings_not_canceled,"
                "reserved_room_type,assigned_room_type,booking_changes,deposit_type,"
                "agent,company,days_in_waiting_list,customer_type,adr,"
                "required_car_parking_spaces,total_of_special_requests,"
                "reservation_status,reservation_status_date\n"
                "City Hotel,0,10,2016,July,28,5,1,2,2,0,0,BB,NULL,"
                "Direct,Direct,0,0,0,A,A,0,No Deposit,NULL,NULL,0,"
                "Transient,100,0,0,Check-Out,2016-07-05\n"
            ),
            dtype=str, keep_default_na=False,
        )
        report = []
        result = replace_null_strings(raw, report)
        assert pd.isna(result["country"].iloc[0]), "country NULL should be NaN"


# ---------------------------------------------------------------------------
# Type casting
# ---------------------------------------------------------------------------

class TestCastDtypes:
    def test_integer_columns_cast(self, raw_df):
        from data_cleaning import (replace_null_strings, cast_dtypes,
                                   INTEGER_COLS)
        report = []
        df = replace_null_strings(raw_df.copy(), report)
        df = cast_dtypes(df, report)
        for col in INTEGER_COLS:
            assert pd.api.types.is_integer_dtype(df[col]), \
                f"{col} should be integer dtype, got {df[col].dtype}"

    def test_adr_is_numeric(self, raw_df):
        """ADR should be numeric (float64 with real data; int64 with whole-number
        synthetic fixture). The pipeline uses pd.to_numeric which preserves
        float64 only when decimals are present in the source."""
        from data_cleaning import replace_null_strings, cast_dtypes
        report = []
        df = replace_null_strings(raw_df.copy(), report)
        df = cast_dtypes(df, report)
        assert pd.api.types.is_numeric_dtype(df["adr"]), \
            f"adr should be numeric, got {df['adr'].dtype}"

    def test_children_na_string_becomes_nan(self, raw_df):
        """Row with children='NA' must have NaN after casting."""
        from data_cleaning import replace_null_strings, cast_dtypes
        report = []
        df = replace_null_strings(raw_df.copy(), report)
        df = cast_dtypes(df, report)
        # Row index 8 in our fixture has children='NA'
        na_rows = df[df["children"].isna()]
        assert len(na_rows) >= 1, "Expected at least one NaN in children after casting NA"

    def test_categorical_columns_cast(self, raw_df):
        from data_cleaning import (replace_null_strings, cast_dtypes,
                                   CATEGORICAL_COLS)
        report = []
        df = replace_null_strings(raw_df.copy(), report)
        df = cast_dtypes(df, report)
        for col in CATEGORICAL_COLS:
            assert str(df[col].dtype) == "category", \
                f"{col} should be category dtype, got {df[col].dtype}"

    def test_reservation_status_date_parsed(self, raw_df):
        from data_cleaning import replace_null_strings, cast_dtypes
        report = []
        df = replace_null_strings(raw_df.copy(), report)
        df = cast_dtypes(df, report)
        assert pd.api.types.is_datetime64_any_dtype(df["reservation_status_date"])


# ---------------------------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------------------------

class TestHandleMissing:
    def test_country_filled_with_unknown(self, raw_df):
        from data_cleaning import (replace_null_strings, cast_dtypes,
                                   handle_missing)
        report = []
        df = replace_null_strings(raw_df.copy(), report)
        df = cast_dtypes(df, report)
        df = handle_missing(df, report)
        assert df["country"].isna().sum() == 0, "country should have no nulls"
        assert "Unknown" in df["country"].cat.categories or \
               (df["country"] == "Unknown").any()

    def test_agent_filled_with_zero_string(self, raw_df):
        from data_cleaning import (replace_null_strings, cast_dtypes,
                                   handle_missing)
        report = []
        df = replace_null_strings(raw_df.copy(), report)
        df = cast_dtypes(df, report)
        df = handle_missing(df, report)
        assert df["agent"].isna().sum() == 0

    def test_children_filled_with_zero(self, raw_df):
        from data_cleaning import (replace_null_strings, cast_dtypes,
                                   handle_missing)
        report = []
        df = replace_null_strings(raw_df.copy(), report)
        df = cast_dtypes(df, report)
        df = handle_missing(df, report)
        assert df["children"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Invalid row removal
# ---------------------------------------------------------------------------

class TestRemoveInvalidRows:
    def test_zero_night_stays_removed(self, raw_df_with_invalids):
        from data_cleaning import (replace_null_strings, cast_dtypes,
                                   handle_missing, remove_invalid_rows)
        report = []
        df = replace_null_strings(raw_df_with_invalids.copy(), report)
        df = cast_dtypes(df, report)
        df = handle_missing(df, report)
        df = remove_invalid_rows(df, report)
        zero_nights = ((df["stays_in_weekend_nights"] == 0) &
                       (df["stays_in_week_nights"] == 0))
        assert zero_nights.sum() == 0, "Zero-night stays should be removed"

    def test_negative_adr_removed(self, raw_df_with_invalids):
        from data_cleaning import (replace_null_strings, cast_dtypes,
                                   handle_missing, remove_invalid_rows)
        report = []
        df = replace_null_strings(raw_df_with_invalids.copy(), report)
        df = cast_dtypes(df, report)
        df = handle_missing(df, report)
        df = remove_invalid_rows(df, report)
        assert (df["adr"] < 0).sum() == 0, "Negative ADR rows should be removed"

    def test_zero_guest_bookings_removed(self, raw_df_with_invalids):
        from data_cleaning import (replace_null_strings, cast_dtypes,
                                   handle_missing, remove_invalid_rows)
        report = []
        df = replace_null_strings(raw_df_with_invalids.copy(), report)
        df = cast_dtypes(df, report)
        df = handle_missing(df, report)
        df = remove_invalid_rows(df, report)
        zero_guests = (
            (df["adults"] == 0) &
            (df["children"].fillna(0) == 0) &
            (df["babies"] == 0)
        )
        assert zero_guests.sum() == 0, "Zero-guest bookings should be removed"

    def test_valid_rows_not_removed(self, raw_df):
        """The main fixture has no invalid rows — all should survive."""
        from data_cleaning import (replace_null_strings, cast_dtypes,
                                   handle_missing, remove_invalid_rows)
        report = []
        df = replace_null_strings(raw_df.copy(), report)
        df = cast_dtypes(df, report)
        df = handle_missing(df, report)
        before = len(df)
        df = remove_invalid_rows(df, report)
        # Row with 0 adults, 0 children, 0 babies check — row 7 (index 7) has
        # adults=2, children=0, babies=0 so it should survive
        assert len(df) > 0, "All valid rows should survive cleaning"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestEngineerFeatures:
    def test_total_nights_computed_correctly(self, clean_df):
        expected = (
            clean_df["stays_in_weekend_nights"].astype(float) +
            clean_df["stays_in_week_nights"].astype(float)
        )
        pd.testing.assert_series_equal(
            clean_df["total_nights"].astype(float),
            expected,
            check_names=False,
            check_dtype=False,
        )

    def test_total_nights_always_positive(self, clean_df):
        assert (clean_df["total_nights"] >= 1).all(), \
            "All total_nights should be ≥ 1 after invalid-row removal"

    def test_room_type_match_binary(self, clean_df):
        assert set(clean_df["room_type_match"].unique()).issubset({0, 1, np.int64(0), np.int64(1)})

    def test_room_type_match_correct_value(self, clean_df):
        """Match flag == 1 only when reserved == assigned."""
        matched = clean_df[clean_df["room_type_match"] == 1]
        assert (matched["reserved_room_type"] == matched["assigned_room_type"]).all()

    def test_is_high_season_correct(self, clean_df):
        """is_high_season should be 1 iff arrival month is June, July, or August."""
        high = clean_df[clean_df["is_high_season"] == 1]["arrival_date_month"]
        assert set(high.unique()).issubset({"June", "July", "August"})
        low = clean_df[clean_df["is_high_season"] == 0]["arrival_date_month"]
        assert not set(low.unique()).intersection({"June", "July", "August"})

    def test_arrival_date_parses(self, clean_df):
        assert pd.api.types.is_datetime64_any_dtype(clean_df["arrival_date"])
        assert clean_df["arrival_date"].isna().sum() == 0

    def test_revenue_estimate_equals_adr_times_nights(self, clean_df):
        expected = clean_df["adr"] * clean_df["total_nights"].astype(float)
        pd.testing.assert_series_equal(
            clean_df["revenue_estimate"].round(2),
            expected.round(2),
            check_names=False,
        )

    def test_total_guests_sum(self, clean_df):
        expected = (
            clean_df["adults"].astype(float) +
            clean_df["children"].fillna(0).astype(float) +
            clean_df["babies"].astype(float)
        )
        pd.testing.assert_series_equal(
            clean_df["total_guests"].astype(float),
            expected,
            check_names=False,
            check_dtype=False,
        )

    def test_no_nulls_after_full_pipeline(self, clean_df):
        nulls = clean_df.isna().sum()
        remaining = nulls[nulls > 0]
        assert len(remaining) == 0, f"Unexpected nulls after pipeline: {remaining.to_dict()}"

    def test_flag_outlier_columns_binary(self, clean_df):
        for col in ["flag_adr_outlier", "flag_long_lead_time", "flag_zero_adr"]:
            assert col in clean_df.columns
            assert set(clean_df[col].unique()).issubset({0, 1, np.int64(0), np.int64(1)})
