# tests/conftest.py
"""
Shared fixtures for the Hotel Bookings test suite.

Fixtures
--------
raw_df          — minimal synthetic raw CSV loaded as a DataFrame (mirrors
                  the real schema; does NOT read hotel_bookings.csv).
clean_df        — cleaned version of raw_df produced by running the full
                  cleaning pipeline on the in-memory fixture.
sqlite_conn     — in-memory SQLite connection with the cleaned fixture loaded
                  as table "bookings".
"""

import io
import sqlite3

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Minimal synthetic raw data  (mimics hotel_bookings.csv schema exactly)
# ---------------------------------------------------------------------------

RAW_CSV = """\
hotel,is_canceled,lead_time,arrival_date_year,arrival_date_month,arrival_date_week_number,arrival_date_day_of_month,stays_in_weekend_nights,stays_in_week_nights,adults,children,babies,meal,country,market_segment,distribution_channel,is_repeated_guest,previous_cancellations,previous_bookings_not_canceled,reserved_room_type,assigned_room_type,booking_changes,deposit_type,agent,company,days_in_waiting_list,customer_type,adr,required_car_parking_spaces,total_of_special_requests,reservation_status,reservation_status_date
Resort Hotel,0,342,2015,July,27,1,0,2,2,0,0,BB,PRT,Direct,Direct,0,0,0,C,C,3,No Deposit,NULL,NULL,0,Transient,75,0,0,Check-Out,2015-07-01
City Hotel,1,100,2016,August,32,10,1,2,1,0,0,HB,GBR,Online TA,TA/TO,0,0,0,A,A,0,No Deposit,304,NULL,0,Transient,120,0,1,Canceled,2016-08-05
Resort Hotel,0,0,2015,July,27,3,0,1,1,0,0,BB,DEU,Corporate,Corporate,0,0,0,B,B,1,No Deposit,NULL,NULL,0,Transient,90,1,2,Check-Out,2015-07-03
City Hotel,0,7,2016,March,10,15,2,3,2,0,0,BB,FRA,Direct,Direct,1,0,1,C,C,0,No Deposit,NULL,NULL,0,Contract,150,0,3,Check-Out,2016-03-15
Resort Hotel,1,500,2017,May,20,5,1,1,1,0,0,SC,ESP,Groups,TA/TO,0,2,0,D,D,0,Non Refund,210,NULL,0,Transient,200,0,0,Canceled,2017-05-05
City Hotel,0,14,2016,June,24,20,0,4,2,1,0,BB,ITA,Offline TA/TO,TA/TO,0,0,0,A,B,2,No Deposit,NULL,NULL,5,Transient-Party,85,0,1,Check-Out,2016-06-20
Resort Hotel,0,30,2015,August,33,8,2,2,2,0,0,HB,PRT,Direct,Direct,0,0,0,C,C,0,No Deposit,NULL,NULL,0,Transient,180,1,2,Check-Out,2015-08-08
City Hotel,1,250,2017,January,3,1,0,0,2,0,0,BB,GBR,Online TA,TA/TO,0,1,0,A,A,0,No Deposit,NULL,NULL,0,Transient,95,0,0,Canceled,2017-01-01
Resort Hotel,0,5,2016,December,50,25,0,3,2,NA,0,BB,PRT,Complementary,Direct,0,0,0,C,C,0,No Deposit,NULL,NULL,0,Transient,0,0,1,Check-Out,2016-12-25
City Hotel,1,180,2017,August,32,12,1,2,2,0,0,FB,USA,Online TA,TA/TO,0,0,0,B,C,1,Non Refund,150,NULL,0,Transient,310,0,2,Canceled,2017-08-12
Resort Hotel,0,3,2016,July,28,4,1,3,1,0,0,BB,PRT,Direct,Direct,0,0,0,D,D,0,No Deposit,NULL,NULL,0,Transient,65,0,0,Check-Out,2016-07-04
City Hotel,0,0,2015,July,27,2,0,1,1,0,0,BB,PRT,Direct,Direct,0,0,0,A,A,0,No Deposit,NULL,NULL,0,Transient,110,1,3,No-Show,2015-07-02
"""

# Invalid rows that should be removed by the cleaning pipeline
RAW_CSV_WITH_INVALIDS = """\
hotel,is_canceled,lead_time,arrival_date_year,arrival_date_month,arrival_date_week_number,arrival_date_day_of_month,stays_in_weekend_nights,stays_in_week_nights,adults,children,babies,meal,country,market_segment,distribution_channel,is_repeated_guest,previous_cancellations,previous_bookings_not_canceled,reserved_room_type,assigned_room_type,booking_changes,deposit_type,agent,company,days_in_waiting_list,customer_type,adr,required_car_parking_spaces,total_of_special_requests,reservation_status,reservation_status_date
Resort Hotel,0,5,2015,July,27,1,0,0,0,0,0,BB,PRT,Direct,Direct,0,0,0,C,C,0,No Deposit,NULL,NULL,0,Transient,80,0,0,Check-Out,2015-07-01
City Hotel,0,10,2016,August,32,10,0,0,2,0,0,BB,GBR,Direct,Direct,0,0,0,A,A,0,No Deposit,NULL,NULL,0,Transient,100,0,0,Check-Out,2016-08-10
Resort Hotel,0,3,2015,July,27,3,1,2,1,0,0,BB,DEU,Direct,Direct,0,0,0,C,C,0,No Deposit,NULL,NULL,0,Transient,-5.0,0,0,Check-Out,2015-07-03
"""


@pytest.fixture
def raw_df():
    """Synthetic raw DataFrame that mirrors hotel_bookings.csv schema."""
    return pd.read_csv(io.StringIO(RAW_CSV), dtype=str, keep_default_na=False)


@pytest.fixture
def raw_df_with_invalids():
    """Raw DataFrame containing rows that must be removed during cleaning."""
    return pd.read_csv(
        io.StringIO(RAW_CSV_WITH_INVALIDS), dtype=str, keep_default_na=False
    )


@pytest.fixture
def clean_df(raw_df):
    """
    Cleaned DataFrame produced by running the full pipeline on raw_df.
    Uses the same steps as data_cleaning.py but operates in-memory.
    """
    import sys
    sys.path.insert(0, str(__file__).replace("tests/conftest.py", "").replace("tests\\conftest.py", ""))
    from data_cleaning import (
        validate_schema, replace_null_strings, cast_dtypes,
        handle_missing, remove_invalid_rows, flag_outliers, engineer_features,
    )
    report = []
    df = raw_df.copy()
    df = validate_schema(df, report)
    df = replace_null_strings(df, report)
    df = cast_dtypes(df, report)
    df = handle_missing(df, report)
    df = remove_invalid_rows(df, report)
    df = flag_outliers(df, report)
    df = engineer_features(df, report)
    return df


@pytest.fixture
def sqlite_conn(clean_df):
    """In-memory SQLite connection with clean_df loaded as 'bookings'."""
    conn = sqlite3.connect(":memory:")
    clean_df.to_sql("bookings", conn, if_exists="replace", index=False)
    yield conn
    conn.close()
