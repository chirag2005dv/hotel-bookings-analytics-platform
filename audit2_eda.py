import pandas as pd
import numpy as np
from pathlib import Path

print('=== AUDIT 2: EDA ACCURACY & KPI LOGIC ===')

df     = pd.read_csv('hotel_bookings_cleaned.csv')
stayed = df[df['is_canceled'] == 0]

cr  = df['is_canceled'].mean() * 100
adr = stayed['adr'].mean()
rev = stayed['revenue_estimate'].sum()
lt  = df['lead_time'].mean()

assert abs(cr  - 37.26)          < 0.01
assert abs(adr - 101.01)         < 0.01
assert abs(rev - 25986976.03)    < 1.0
assert abs(lt  - 104.5)          < 0.1
print(f'  Cancel rate   : {cr:.2f}%  OK')
print(f'  Avg ADR       : ${adr:.2f}  OK')
print(f'  Total revenue : ${rev:,.2f}  OK')
print(f'  Avg lead time : {lt:.1f} days  OK')

# Seasonality
monthly_adr = stayed[stayed['hotel']=='Resort Hotel'].groupby('arrival_date_month')['adr'].mean()
assert monthly_adr.idxmax() == 'August'
print(f'  Resort peak month: August (${monthly_adr.max():.2f})  OK')

# Hotel cancel rates
city_cr   = df[df['hotel']=='City Hotel']['is_canceled'].mean() * 100
resort_cr = df[df['hotel']=='Resort Hotel']['is_canceled'].mean() * 100
assert city_cr > resort_cr
assert abs(city_cr - 41.91) < 0.01
assert abs(resort_cr - 28.01) < 0.01
print(f'  Hotel cancel: City={city_cr:.2f}%  Resort={resort_cr:.2f}%  OK')

nr_cr     = df[df['deposit_type']=='Non Refund']['is_canceled'].mean() * 100
assert nr_cr > 99.0
print(f'  Non-Refund cancel: {nr_cr:.2f}%  OK')

repeat_cr = df[df['is_repeated_guest']==1]['is_canceled'].mean() * 100
new_cr    = df[df['is_repeated_guest']==0]['is_canceled'].mean() * 100
assert repeat_cr < new_cr
print(f'  Repeat vs new: {repeat_cr:.2f}% vs {new_cr:.2f}%  OK')

kpi_text = open('kpi_report.txt', encoding='utf-8').read()
assert '37.' in kpi_text and '101' in kpi_text and '118,564' in kpi_text
print('  kpi_report.txt: key values present  OK')

eda_plots = [
    'C1_cancellation_by_hotel.png','C2_cancellation_by_market_segment.png',
    'C3_cancellation_by_deposit_type.png','C4_cancellation_by_lead_time_bucket.png',
    'D1_monthly_adr_by_hotel.png','D2_adr_distribution.png',
    'D3_adr_by_market_segment.png','D4_monthly_revenue_estimate.png',
    'E1_volume_by_market_segment.png','E2_distribution_channel_mix.png',
    'E3_segment_cancel_vs_adr.png','F1_lead_time_distribution.png',
    'F2_cancellation_by_booking_changes.png','F3_cancellation_by_special_requests.png',
    'G1_monthly_bookings_and_cancel_rate.png','G2_yearly_trends.png',
    'G3_cancellation_heatmap_hotel_month.png','H1_top_countries.png',
    'H2_length_of_stay_distribution.png','H3_meal_plan_by_hotel.png',
    'H4_new_vs_repeat_guest_cancellation.png',
]
missing = [p for p in eda_plots if not (Path('plots') / p).exists()]
assert not missing, f'Missing: {missing}'
sizes = [(Path('plots') / p).stat().st_size for p in eda_plots]
assert min(sizes) > 10000, f'Small plot: {eda_plots[sizes.index(min(sizes))]}'
print(f'  EDA plots: all 21 present, min size {min(sizes)//1024}KB  OK')

print('\nAUDIT 2 PASSED')
