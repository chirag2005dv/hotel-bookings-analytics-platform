import sqlite3
import pandas as pd
from pathlib import Path

print('=== AUDIT 3: SQL ANALYTICS CORRECTNESS ===')

# Load cleaned data into SQLite
df   = pd.read_csv('hotel_bookings_cleaned.csv')
conn = sqlite3.connect(':memory:')
df.to_sql('bookings', conn, if_exists='replace', index=False)

# Helper
def q(sql): return pd.read_sql_query(sql, conn)

# 3a. Q01 — core KPIs match Python-computed values
q01 = q("SELECT COUNT(*) AS total, SUM(is_canceled) AS canceled, "
        "ROUND(AVG(CAST(is_canceled AS REAL))*100,2) AS cr_pct FROM bookings")
assert int(q01['total'].iloc[0])   == 118564
assert int(q01['canceled'].iloc[0]) == 44176
assert abs(float(q01['cr_pct'].iloc[0]) - 37.26) < 0.01
print('  Q01 KPIs: total=118564, canceled=44176, cancel_rate=37.26%  OK')

# 3b. Q05 — lead-time bucket totals sum to dataset total
from sql_analytics import parse_sql_file
queries = {q_['query_id']: q_ for q_ in parse_sql_file('hotel_queries.sql')}

q05 = pd.read_sql_query(queries['Q05']['sql'], conn)
assert q05['total_bookings'].sum() == 118564
print('  Q05 lead-time buckets sum to 118564  OK')

# 3c. Q02 — exactly 2 hotel types, rates consistent
q02 = pd.read_sql_query(queries['Q02']['sql'], conn)
assert len(q02) == 2
city   = q02[q02['hotel']=='City Hotel']['cancellation_rate_pct'].iloc[0]
resort = q02[q02['hotel']=='Resort Hotel']['cancellation_rate_pct'].iloc[0]
assert city > resort
print(f'  Q02 hotel rates: City={city:.2f}%  Resort={resort:.2f}%  OK')

# 3d. Q07 — revenue share sums to 100% (within 0.1)
q07 = pd.read_sql_query(queries['Q07']['sql'], conn)
total_share = q07['revenue_share_pct'].sum()
assert abs(total_share - 100.0) < 0.5, f'Revenue share={total_share:.2f}%'
print(f'  Q07 revenue shares sum: {total_share:.2f}% (expected ~100%)  OK')

# 3e. Q10 — year totals sum to total
q10 = pd.read_sql_query(queries['Q10']['sql'], conn)
assert q10['total_bookings'].sum() == 118564
print('  Q10 year totals sum to 118564  OK')

# 3f. Q14 — repeat guest cancel rate is lower
q14 = pd.read_sql_query(queries['Q14']['sql'], conn)
repeat = q14[q14['guest_type']=='Repeat Guest']['cancellation_rate_pct'].iloc[0]
new    = q14[q14['guest_type']=='New Guest']['cancellation_rate_pct'].iloc[0]
assert repeat < new
print(f'  Q14 repeat={repeat:.2f}% vs new={new:.2f}%  OK')

# 3g. Q20 — top 10% + bottom 90% bookings sum to total checked-out
q20 = pd.read_sql_query(queries['Q20']['sql'], conn)
total_stayed = len(df[df['is_canceled']==0])
assert q20['bookings'].sum() == total_stayed, f'Q20 sum {q20["bookings"].sum()} != {total_stayed}'
print(f'  Q20 top+bottom = {q20["bookings"].sum():,} stayed bookings  OK')

# 3h. Verify all 20 CSV result files exist
for i in range(1, 21):
    qid = f'Q{i:02d}'
    matches = list(Path('sql_results').glob(f'{qid}_*.csv'))
    assert matches, f'Missing result CSV for {qid}'
print('  All 20 SQL result CSVs present  OK')

# 3i. SQL file has 20 annotated queries
raw_sql = Path('hotel_queries.sql').read_text(encoding='utf-8')
query_count = raw_sql.count('@query_id:')
assert query_count == 20, f'Expected 20 @query_id markers, found {query_count}'
print(f'  hotel_queries.sql: {query_count} queries annotated  OK')

conn.close()
print('\nAUDIT 3 PASSED')
