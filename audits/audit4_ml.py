import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

print('=== AUDIT 4: FEATURE ENGINEERING & ML EVALUATION ===')

df = pd.read_csv('hotel_bookings_cleaned.csv')

# 4a. Feature count and naming
feature_names = (Path('models') / 'feature_names.txt').read_text().splitlines()
assert len(feature_names) == 31, f'Expected 31 features, got {len(feature_names)}'
print(f'  Feature names: {len(feature_names)} features  OK')

# 4b. No leakage columns in feature set
LEAKAGE = ['reservation_status', 'reservation_status_date',
           'revenue_estimate', 'flag_adr_outlier', 'flag_long_lead_time',
           'flag_zero_adr', 'arrival_date']
for lc in LEAKAGE:
    assert lc not in feature_names, f'Leakage column in features: {lc}'
print(f'  Leakage check: none of {LEAKAGE} in feature list  OK')

# 4c. Rebuild feature matrix
CATEGORICAL_FEATURES = [
    'hotel','arrival_date_month','meal','market_segment','distribution_channel',
    'reserved_room_type','assigned_room_type','deposit_type','customer_type',
]
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    le.fit(df[col].astype(str))
    df[col + '_enc'] = le.transform(df[col].astype(str))

X = df[feature_names].copy()
X['children'] = X['children'].fillna(0)
y = df['is_canceled']

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

# 4d. Verify stratification: test set cancel rate matches overall
test_cr = y_test.mean() * 100
overall_cr = y.mean() * 100
assert abs(test_cr - overall_cr) < 0.01, f'Stratification failed: {test_cr:.4f} != {overall_cr:.4f}'
print(f'  Train/test stratification: test cancel rate={test_cr:.2f}% (overall={overall_cr:.2f}%)  OK')

# 4e. Re-evaluate all three saved models and verify reported metrics
models_expected = {
    'logistic_regression.pkl': {'roc_auc': 0.8550, 'f1': 0.7088, 'acc': 0.7764},
    'random_forest.pkl':       {'roc_auc': 0.9149, 'f1': 0.7846, 'acc': 0.8414},
    'gradient_boosting.pkl':   {'roc_auc': 0.9157, 'f1': 0.7832, 'acc': 0.8359},
}
for fname, exp in models_expected.items():
    m    = joblib.load(Path('models') / fname)
    pred = m.predict(X_test)
    prob = m.predict_proba(X_test)[:, 1]
    auc  = roc_auc_score(y_test, prob)
    f1   = f1_score(y_test, pred)
    acc  = accuracy_score(y_test, pred)
    assert abs(auc - exp['roc_auc']) < 0.0001, f'{fname} AUC {auc:.4f} != {exp["roc_auc"]}'
    assert abs(f1  - exp['f1'])      < 0.0001, f'{fname} F1 {f1:.4f} != {exp["f1"]}'
    assert abs(acc - exp['acc'])     < 0.0001, f'{fname} Acc {acc:.4f} != {exp["acc"]}'
    print(f'  {fname}: AUC={auc:.4f} F1={f1:.4f} Acc={acc:.4f}  OK')

# 4f. best_model.pkl == gradient_boosting.pkl (same object)
best  = joblib.load(Path('models') / 'best_model.pkl')
gb    = joblib.load(Path('models') / 'gradient_boosting.pkl')
best_auc = roc_auc_score(y_test, best.predict_proba(X_test)[:, 1])
gb_auc   = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])
assert abs(best_auc - gb_auc) < 1e-8, 'best_model.pkl differs from gradient_boosting.pkl'
print(f'  best_model.pkl == gradient_boosting.pkl (AUC={best_auc:.4f})  OK')

# 4g. Model determinism — same prediction on same input twice
sample = X_test.iloc[:10]
pred1 = best.predict_proba(sample)
pred2 = best.predict_proba(sample)
np.testing.assert_array_equal(pred1, pred2)
print('  Prediction determinism: identical results on two calls  OK')

# 4h. Feature importance CSV exists and top feature is deposit_type
fi = pd.read_csv(Path('models') / 'feature_importance.csv')
assert fi.iloc[0]['feature'] == 'deposit_type_enc', \
    f'Top MDI feature is {fi.iloc[0]["feature"]}, expected deposit_type_enc'
assert (fi['importance'] >= 0).all()
print(f'  MDI top feature: {fi.iloc[0]["feature"]} ({fi.iloc[0]["importance"]:.4f})  OK')

# 4i. ml_report.txt contains real metrics
ml_text = Path('ml_report.txt').read_text(encoding='utf-8')
assert '0.9157' in ml_text
assert 'Gradient Boosting' in ml_text
print('  ml_report.txt contains expected metrics  OK')

print('\nAUDIT 4 PASSED')
