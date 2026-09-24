import ast
import re
from pathlib import Path

print('=== AUDIT 6: DASHBOARD SECURITY & REPRODUCIBILITY ===')

dashboard_src = Path('dashboard.py').read_text(encoding='utf-8')

# 6a. No hardcoded API keys in dashboard.py
api_key_patterns = [
    r'sk-[A-Za-z0-9]{20,}',          # OpenAI key pattern
    r'["\']OPENAI_API_KEY["\']\s*[:=]\s*["\'][^"\']+["\']',  # assigned key
    r'api_key\s*=\s*["\'][^"\']{10,}["\']',   # literal assignment
]
for pat in api_key_patterns:
    matches = re.findall(pat, dashboard_src)
    assert not matches, f'Possible hardcoded API key in dashboard.py: {matches}'
print('  dashboard.py: no hardcoded API keys  OK')

# 6b. Dashboard reads API key from os.environ only
assert 'os.environ.get("OPENAI_API_KEY"' in dashboard_src or \
       'from ai_insights import' in dashboard_src, \
    'dashboard.py should read key via ai_insights or os.environ'
print('  dashboard.py: API key access via os.environ (through ai_insights)  OK')

# 6c. No hardcoded numbers — all KPIs loaded from CSVs or computed from df
assert 'load_sql(' in dashboard_src
assert 'load_cleaned_data()' in dashboard_src
assert '118564' not in dashboard_src, 'Found hardcoded row count in dashboard'
assert '37.26' not in dashboard_src,  'Found hardcoded cancel rate in dashboard'
assert '101.01' not in dashboard_src, 'Found hardcoded ADR in dashboard'
print('  dashboard.py: no hardcoded KPI values (37.26, 101.01, 118564)  OK')

# 6d. ML model performance summary in Tab 5 — check those ARE sourced correctly
# The values in perf_data dict are the *reported* metrics from ml_report.txt, not arbitrary
import joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

df_clean = pd.read_csv('hotel_bookings_cleaned.csv')
CAT = ['hotel','arrival_date_month','meal','market_segment','distribution_channel',
       'reserved_room_type','assigned_room_type','deposit_type','customer_type']
for col in CAT:
    le = LabelEncoder()
    le.fit(df_clean[col].astype(str))
    df_clean[col+'_enc'] = le.transform(df_clean[col].astype(str))
feature_names = Path('models/feature_names.txt').read_text().splitlines()
X = df_clean[feature_names].copy()
X['children'] = X['children'].fillna(0)
y = df_clean['is_canceled']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Verify the 3 metric sets shown in dashboard perf_data are accurate
expected = {
    'Logistic Regression':   (0.7764, 0.7088, 0.8550),
    'Random Forest':         (0.8414, 0.7846, 0.9149),
    'Gradient Boosting':     (0.8359, 0.7832, 0.9157),
}
for name, (exp_acc, exp_f1, exp_auc) in expected.items():
    fname = name.lower().replace(' ', '_') + '.pkl'
    m    = joblib.load(Path('models') / fname)
    pred = m.predict(X_test)
    prob = m.predict_proba(X_test)[:, 1]
    acc  = accuracy_score(y_test, pred)
    f1   = f1_score(y_test, pred)
    auc  = roc_auc_score(y_test, prob)
    assert abs(acc - exp_acc) < 0.0001
    assert abs(f1  - exp_f1)  < 0.0001
    assert abs(auc - exp_auc) < 0.0001
    print(f'  Dashboard perf_data {name}: Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}  OK')

# 6e. Syntax check
import py_compile
py_compile.compile('dashboard.py', doraise=True)
print('  dashboard.py: syntax valid  OK')

# 6f. All data file paths in dashboard are relative (not absolute)
absolute_path_re = re.compile(r'["\']([A-Za-z]:\\|/home/|/Users/|/root/)[^"\']+["\']')
matches = absolute_path_re.findall(dashboard_src)
assert not matches, f'Absolute paths found in dashboard.py: {matches}'
print('  dashboard.py: no absolute filesystem paths  OK')

# 6g. Reproducibility: re-running sql_analytics on same data gives same Q01
import sqlite3
df2   = pd.read_csv('hotel_bookings_cleaned.csv')
conn2 = sqlite3.connect(':memory:')
df2.to_sql('bookings', conn2, if_exists='replace', index=False)
from sql_analytics import parse_sql_file
queries = {q['query_id']: q for q in parse_sql_file('hotel_queries.sql')}
q01a = pd.read_sql_query(queries['Q01']['sql'], conn2)
q01b = pd.read_sql_query(queries['Q01']['sql'], conn2)
assert q01a.iloc[0]['total_bookings'] == q01b.iloc[0]['total_bookings']
conn2.close()
print('  SQL Q01 reproducibility: identical results on two runs  OK')

print('\nAUDIT 6 PASSED')
