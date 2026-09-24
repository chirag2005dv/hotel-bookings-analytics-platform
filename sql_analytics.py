"""
sql_analytics.py
================
Phase 4 — SQL Analytics Runner
Hotel Bookings Dataset

Usage:
    python sql_analytics.py

What it does:
  1. Loads hotel_bookings_cleaned.csv into an in-memory SQLite database
     (table name: bookings).
  2. Parses hotel_queries.sql and extracts every named query block.
  3. Executes each query and saves results to:
       - sql_results/<query_id>_<title>.csv   (machine-readable)
       - sql_results_report.txt               (human-readable report)
  4. Runs a cross-validation block comparing key SQL results against
     known Phase 3 KPI values.

Engine: Python sqlite3 (SQLite 3.39.4, built-in)
"""

import re
import sqlite3
import textwrap
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

INPUT_CSV    = "hotel_bookings_cleaned.csv"
SQL_FILE     = "hotel_queries.sql"
RESULTS_DIR  = Path("sql_results")
REPORT_FILE  = "sql_results_report.txt"

RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# STEP 1 — LOAD CSV INTO SQLITE
# ---------------------------------------------------------------------------

def load_to_sqlite(csv_path: str) -> sqlite3.Connection:
    print("Loading CSV into SQLite (in-memory)...")
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(":memory:")
    df.to_sql("bookings", conn, if_exists="replace", index=False)
    row_count = conn.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
    col_count = len(conn.execute("PRAGMA table_info(bookings)").fetchall())
    print(f"  Table 'bookings': {row_count:,} rows x {col_count} columns")
    return conn


# ---------------------------------------------------------------------------
# STEP 2 — PARSE SQL FILE INTO NAMED QUERY BLOCKS
# ---------------------------------------------------------------------------

def parse_sql_file(sql_path: str) -> list[dict]:
    """
    Extracts query blocks from hotel_queries.sql.
    Each block must have:
        -- @query_id: Q##
        -- @title: <Human readable title>
    followed by the SQL statement ending at the next block or EOF.
    Returns list of dicts: {query_id, title, sql}
    """
    text = Path(sql_path).read_text(encoding="utf-8")

    # Split on query_id markers
    pattern = re.compile(
        r"--\s*@query_id:\s*(\w+)\s*\n"   # @query_id line
        r"--\s*@title:\s*(.+?)\s*\n"       # @title line
        r"(.*?)"                            # SQL body
        r"(?=--\s*@query_id:|\Z)",          # up to next query or EOF
        re.DOTALL,
    )

    queries = []
    for m in pattern.finditer(text):
        qid   = m.group(1).strip()
        title = m.group(2).strip()
        sql   = m.group(3).strip()
        # Remove trailing separator comments
        sql = re.sub(r"\n*--\s*-{10,}.*$", "", sql, flags=re.DOTALL).strip()
        if sql:
            queries.append({"query_id": qid, "title": title, "sql": sql})

    print(f"  Parsed {len(queries)} queries from {sql_path}")
    return queries


# ---------------------------------------------------------------------------
# STEP 3 — EXECUTE QUERIES & SAVE RESULTS
# ---------------------------------------------------------------------------

def run_queries(conn: sqlite3.Connection, queries: list[dict]) -> dict[str, pd.DataFrame]:
    results = {}
    for q in queries:
        qid   = q["query_id"]
        title = q["title"]
        sql   = q["sql"]
        try:
            df = pd.read_sql_query(sql, conn)
            results[qid] = df
            # Save CSV
            safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")[:60]
            out_path = RESULTS_DIR / f"{qid}_{safe_title}.csv"
            df.to_csv(out_path, index=False)
            print(f"  [{qid}] {title}  -> {len(df)} rows  -> {out_path.name}")
        except Exception as e:
            print(f"  [{qid}] ERROR: {e}")
            results[qid] = pd.DataFrame()
    return results


# ---------------------------------------------------------------------------
# STEP 4 — GENERATE HUMAN-READABLE REPORT
# ---------------------------------------------------------------------------

def generate_report(queries: list[dict], results: dict[str, pd.DataFrame]) -> str:
    lines = []
    lines.append("Hotel Bookings — Phase 4 SQL Analytics Report")
    lines.append("=" * 70)
    lines.append(f"Engine : SQLite (Python built-in sqlite3)")
    lines.append(f"Source : {INPUT_CSV}")
    lines.append(f"Queries: {len(queries)}")
    lines.append("")

    for q in queries:
        qid   = q["query_id"]
        title = q["title"]
        df    = results.get(qid, pd.DataFrame())

        lines.append(f"{'─' * 70}")
        lines.append(f"[{qid}] {title}")
        lines.append(f"{'─' * 70}")

        if df.empty:
            lines.append("  (no results or error)")
        else:
            lines.append(df.to_string(index=False))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# STEP 5 — CROSS-VALIDATION vs PHASE 3 KPIs
# ---------------------------------------------------------------------------

PHASE3_KPIS = {
    "total_bookings":          118564,
    "total_canceled":          44176,
    "cancellation_rate_pct":   37.26,   # ± 0.05
    "avg_adr_stayed":          101.01,  # ± 0.10
    "total_revenue_estimate":  25986976.03,
    "avg_lead_time_days":      104.5,   # ± 0.1
    "repeat_guest_rate_pct":   2.95,    # ± 0.01
    "room_match_rate_pct":     87.81,   # ± 0.01
}

TOLERANCE = {
    "cancellation_rate_pct":   0.05,
    "avg_adr_stayed":          0.10,
    "avg_lead_time_days":      0.10,
    "repeat_guest_rate_pct":   0.02,
    "room_match_rate_pct":     0.02,
    "total_bookings":          0,
    "total_canceled":          0,
    "total_revenue_estimate":  1.0,
}


def cross_validate(results: dict[str, pd.DataFrame]) -> list[str]:
    lines = []
    lines.append("=" * 70)
    lines.append("CROSS-VALIDATION: SQL Results vs Phase 3 KPIs")
    lines.append("=" * 70)

    q01 = results.get("Q01", pd.DataFrame())
    if q01.empty:
        lines.append("  Q01 result missing — cannot validate.")
        return lines

    row = q01.iloc[0]

    checks = [
        ("total_bookings",         int(row["total_bookings"])),
        ("total_canceled",         int(row["total_canceled"])),
        ("cancellation_rate_pct",  float(row["cancellation_rate_pct"])),
        ("avg_adr_stayed",         float(row["avg_adr_stayed"])),
        ("total_revenue_estimate", float(row["total_revenue_estimate"])),
        ("avg_lead_time_days",     float(row["avg_lead_time_days"])),
        ("repeat_guest_rate_pct",  float(row["repeat_guest_rate_pct"])),
        ("room_match_rate_pct",    float(row["room_match_rate_pct"])),
    ]

    all_pass = True
    for key, sql_val in checks:
        expected = PHASE3_KPIS[key]
        tol      = TOLERANCE.get(key, 0.05)
        diff     = abs(sql_val - expected)
        status   = "PASS" if diff <= tol else "FAIL"
        if status == "FAIL":
            all_pass = False
        lines.append(
            f"  {status}  {key:35s}  SQL={sql_val}  Expected={expected}  |diff|={diff:.4f}"
        )

    lines.append("")
    lines.append("  Overall: " + ("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"))
    return lines


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    report_lines: list[str] = []

    print("\n" + "=" * 65)
    print("  PHASE 4 — SQL ANALYTICS RUNNER")
    print("=" * 65)

    # Step 1
    print("\n[1] Loading data into SQLite...")
    conn = load_to_sqlite(INPUT_CSV)

    # Step 2
    print("\n[2] Parsing SQL query library...")
    queries = parse_sql_file(SQL_FILE)

    # Step 3
    print("\n[3] Executing queries...")
    results = run_queries(conn, queries)

    # Step 4
    print("\n[4] Generating report...")
    report_text = generate_report(queries, results)
    report_lines.extend(report_text.splitlines())

    # Step 5
    print("\n[5] Cross-validating against Phase 3 KPIs...")
    val_lines = cross_validate(results)
    for ln in val_lines:
        print("  " + ln)
    report_lines.append("")
    report_lines.extend(val_lines)

    # Save report
    full_report = "\n".join(report_lines)
    Path(REPORT_FILE).write_text(full_report, encoding="utf-8")
    print(f"\n  Report saved: {REPORT_FILE}")

    # Summary
    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    print(f"  Result CSVs : {len(csv_files)} files in {RESULTS_DIR}/")
    print("\nPhase 4 SQL Analytics complete.")

    conn.close()


if __name__ == "__main__":
    main()
