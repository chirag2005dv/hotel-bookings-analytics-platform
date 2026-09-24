# tests/test_sql_analytics.py
"""
Tests for sql_analytics.py — SQL file parsing, query execution,
cross-validation, and result shape correctness.
"""

import sqlite3
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# SQL file parsing
# ---------------------------------------------------------------------------

class TestParseSqlFile:
    def test_parses_all_20_queries(self):
        from sql_analytics import parse_sql_file
        queries = parse_sql_file("hotel_queries.sql")
        assert len(queries) == 20, f"Expected 20 queries, got {len(queries)}"

    def test_each_query_has_required_keys(self):
        from sql_analytics import parse_sql_file
        queries = parse_sql_file("hotel_queries.sql")
        for q in queries:
            assert "query_id" in q
            assert "title" in q
            assert "sql" in q
            assert q["query_id"].startswith("Q"), \
                f"query_id should start with Q: {q['query_id']}"
            assert len(q["sql"]) > 10, \
                f"SQL body too short for {q['query_id']}"

    def test_query_ids_unique(self):
        from sql_analytics import parse_sql_file
        queries = parse_sql_file("hotel_queries.sql")
        ids = [q["query_id"] for q in queries]
        assert len(ids) == len(set(ids)), "Duplicate query IDs found"

    def test_query_ids_sequential(self):
        from sql_analytics import parse_sql_file
        queries = parse_sql_file("hotel_queries.sql")
        ids = sorted([q["query_id"] for q in queries])
        assert ids[0] == "Q01"
        assert ids[-1] == "Q20"


# ---------------------------------------------------------------------------
# Query execution on the fixture DB
# ---------------------------------------------------------------------------

class TestRunQueries:
    def test_all_queries_execute_without_error(self, sqlite_conn, tmp_path, monkeypatch):
        """Run all queries and write CSVs to a temp dir (not sql_results/)."""
        from sql_analytics import parse_sql_file
        import pathlib

        # Redirect output to tmp_path so real sql_results/ is not overwritten
        monkeypatch.setattr("sql_analytics.RESULTS_DIR", tmp_path)

        from sql_analytics import run_queries
        queries = parse_sql_file("hotel_queries.sql")
        results = run_queries(sqlite_conn, queries)
        assert len(results) == 20
        for qid, df in results.items():
            assert isinstance(df, pd.DataFrame), \
                f"Query {qid} should return a DataFrame"

    def test_q01_returns_one_row(self, sqlite_conn):
        from sql_analytics import parse_sql_file, run_queries
        queries = {q["query_id"]: q for q in parse_sql_file("hotel_queries.sql")}
        result = pd.read_sql_query(queries["Q01"]["sql"], sqlite_conn)
        assert len(result) == 1, "Q01 (Core KPI Snapshot) should return exactly 1 row"

    def test_q01_total_bookings_matches_fixture(self, sqlite_conn, clean_df):
        """Q01 total_bookings must match the row count of the fixture."""
        from sql_analytics import parse_sql_file
        queries = {q["query_id"]: q for q in parse_sql_file("hotel_queries.sql")}
        result = pd.read_sql_query(queries["Q01"]["sql"], sqlite_conn)
        assert int(result["total_bookings"].iloc[0]) == len(clean_df), \
            "Q01 total_bookings should equal the fixture row count"

    def test_q01_cancellation_rate_in_valid_range(self, sqlite_conn):
        from sql_analytics import parse_sql_file
        queries = {q["query_id"]: q for q in parse_sql_file("hotel_queries.sql")}
        result = pd.read_sql_query(queries["Q01"]["sql"], sqlite_conn)
        rate = float(result["cancellation_rate_pct"].iloc[0])
        assert 0 <= rate <= 100, f"Cancellation rate {rate} out of range [0, 100]"

    def test_q02_returns_two_hotel_types(self, sqlite_conn):
        from sql_analytics import parse_sql_file
        queries = {q["query_id"]: q for q in parse_sql_file("hotel_queries.sql")}
        result = pd.read_sql_query(queries["Q02"]["sql"], sqlite_conn)
        assert len(result) == 2, "Q02 should return 2 hotel types"
        assert set(result["hotel"].tolist()) == {"City Hotel", "Resort Hotel"}

    def test_q05_lead_time_buckets_sum_to_total(self, sqlite_conn, clean_df):
        """The sum of bookings across all lead-time buckets must equal total bookings."""
        from sql_analytics import parse_sql_file
        queries = {q["query_id"]: q for q in parse_sql_file("hotel_queries.sql")}
        result = pd.read_sql_query(queries["Q05"]["sql"], sqlite_conn)
        assert result["total_bookings"].sum() == len(clean_df), \
            "Q05 lead-time bucket totals must sum to total row count"

    def test_q09_twelve_months(self, sqlite_conn):
        from sql_analytics import parse_sql_file
        queries = {q["query_id"]: q for q in parse_sql_file("hotel_queries.sql")}
        result = pd.read_sql_query(queries["Q09"]["sql"], sqlite_conn)
        assert len(result) <= 12, "Q09 should return at most 12 months"

    def test_q07_revenue_shares_sum_near_100(self, sqlite_conn):
        """Revenue share percentages from Q07 must sum to ~100%."""
        from sql_analytics import parse_sql_file
        queries = {q["query_id"]: q for q in parse_sql_file("hotel_queries.sql")}
        result = pd.read_sql_query(queries["Q07"]["sql"], sqlite_conn)
        if "revenue_share_pct" in result.columns and len(result) > 0:
            total_share = result["revenue_share_pct"].sum()
            assert abs(total_share - 100.0) < 1.0, \
                f"Revenue shares sum to {total_share:.2f}, expected ~100"

    def test_q20_two_segments_returned(self, sqlite_conn):
        from sql_analytics import parse_sql_file
        queries = {q["query_id"]: q for q in parse_sql_file("hotel_queries.sql")}
        result = pd.read_sql_query(queries["Q20"]["sql"], sqlite_conn)
        assert len(result) == 2, "Q20 should return 2 rows (Top 10% and Bottom 90%)"


# ---------------------------------------------------------------------------
# Cross-validation function
# ---------------------------------------------------------------------------

class TestCrossValidate:
    def test_all_checks_pass_on_real_data(self):
        """Cross-validate against actual Q01 result from sql_results/."""
        from sql_analytics import cross_validate, PHASE3_KPIS
        import pathlib
        q01_path = pathlib.Path("sql_results/Q01_Core_KPI_Snapshot.csv")
        if not q01_path.exists():
            pytest.skip("sql_results/Q01_Core_KPI_Snapshot.csv not found")
        q01_df = pd.read_csv(q01_path)
        results = {"Q01": q01_df}
        lines = cross_validate(results)
        # All checks should pass
        assert any("ALL CHECKS PASSED" in ln for ln in lines), \
            "Cross-validation failed against Phase 3 KPIs"

    def test_fail_detected_on_wrong_value(self, sqlite_conn):
        """If Q01 returns wrong values, cross_validate must report FAIL."""
        from sql_analytics import cross_validate
        # Inject a fabricated Q01 with wrong total
        bad_q01 = pd.DataFrame([{
            "total_bookings":         999999,   # wrong
            "total_canceled":         0,
            "cancellation_rate_pct":  0.0,
            "total_checked_out":      0,
            "total_no_show":          0,
            "avg_adr_stayed":         0.0,
            "total_revenue_estimate": 0.0,
            "avg_lead_time_days":     0.0,
            "repeat_guest_rate_pct":  0.0,
            "room_match_rate_pct":    0.0,
        }])
        lines = cross_validate({"Q01": bad_q01})
        assert any("FAIL" in ln for ln in lines), \
            "cross_validate should report FAIL for wrong total_bookings"
