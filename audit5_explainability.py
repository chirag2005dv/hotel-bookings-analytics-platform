import pandas as pd
from pathlib import Path

print('=== AUDIT 5: EXPLAINABILITY CODE INTEGRITY ===')

# 5a. All 8 explainability plots present and non-empty
EX_PLOTS = [
    'EX_shap_summary_bar.png',
    'EX_shap_summary_beeswarm.png',
    'EX_shap_dependence_top4.png',
    'EX_shap_waterfall_cancel.png',
    'EX_shap_waterfall_nocancel.png',
    'EX_permutation_importance.png',
    'EX_partial_dependence.png',
    'EX_method_comparison_heatmap.png',
]
for p in EX_PLOTS:
    path = Path('plots') / p
    assert path.exists(), f'Missing: {p}'
    assert path.stat().st_size > 20000, f'Suspiciously small: {p} ({path.stat().st_size} bytes)'
print(f'  Explainability plots: all 8 present, all >{20}KB  OK')

# 5b. SHAP CSV integrity
shap_df = pd.read_csv(Path('models') / 'shap_feature_importance.csv')
assert len(shap_df) == 31, f'Expected 31 SHAP rows, got {len(shap_df)}'
assert 'feature' in shap_df.columns and 'mean_abs_shap' in shap_df.columns
assert (shap_df['mean_abs_shap'] >= 0).all()
top_shap = shap_df.sort_values('mean_abs_shap', ascending=False).iloc[0]
assert 'deposit' in top_shap['feature'].lower(), f'Top SHAP feature: {top_shap["feature"]}'
print(f'  SHAP CSV: 31 features, top={top_shap["feature"]} ({top_shap["mean_abs_shap"]:.4f})  OK')

# 5c. Permutation importance CSV integrity
perm_df = pd.read_csv(Path('models') / 'permutation_importance.csv')
assert len(perm_df) == 31
top_perm = perm_df.sort_values('importance_mean', ascending=False).iloc[0]
assert 'deposit' in top_perm['label'].lower(), f'Top Permutation: {top_perm["label"]}'
assert (perm_df['importance_std'] >= 0).all()
print(f'  Permutation CSV: 31 features, top={top_perm["label"]} ({top_perm["importance_mean"]:.4f})  OK')

# 5d. Feature ranking comparison: 31 rows, 3 rank columns
rank_df = pd.read_csv(Path('models') / 'feature_ranking_comparison.csv')
assert len(rank_df) == 31
for col in ['mdi_rank', 'shap_rank', 'perm_rank']:
    assert col in rank_df.columns
    assert set(rank_df[col].tolist()) == set(range(1, 32)), f'{col} ranks not 1..31'
print('  Ranking comparison: 31 features, ranks 1–31 per method  OK')

# 5e. Consensus top-3 features (deposit, lead_time, special_requests)
rank_df['avg_rank'] = rank_df[['mdi_rank','shap_rank','perm_rank']].mean(axis=1)
top3 = rank_df.sort_values('avg_rank').head(3)['label'].tolist()
assert any('Deposit' in f or 'deposit' in f for f in top3), f'Deposit not in top-3: {top3}'
assert any('Lead' in f or 'lead' in f for f in top3), f'Lead Time not in top-3: {top3}'
assert any('Special' in f or 'special' in f for f in top3), f'Special Requests not in top-3: {top3}'
print(f'  Consensus top-3: {top3}  OK')

# 5f. Prediction vs causation disclaimer is documented
ex_text = Path('explainability_report.txt').read_text(encoding='utf-8')
for keyword in ['PREDICTION', 'CAUSATION', 'associations', 'do NOT']:
    assert keyword in ex_text, f'Disclaimer keyword missing: {keyword}'
print('  explainability_report.txt: prediction/causation disclaimer present  OK')

# 5g. Waterfall charts exist for both classes
for fname in ['EX_shap_waterfall_cancel.png', 'EX_shap_waterfall_nocancel.png']:
    assert (Path('plots') / fname).exists()
print('  Waterfall charts: both cancel and not-cancel examples present  OK')

print('\nAUDIT 5 PASSED')
