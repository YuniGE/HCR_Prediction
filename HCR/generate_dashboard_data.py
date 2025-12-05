import pandas as pd
import json
from datetime import datetime

# Load data
df = pd.read_excel('Threshold_Combined.xlsx')
df['Released_date'] = pd.to_datetime(df['Released date'].astype(str), format='%Y%m%d')
df['Release_year'] = df['Released_date'].dt.year
df['Paper_age'] = df['Release_year'] - df['Year']
df['Date_str'] = df['Released_date'].dt.strftime('%Y-%m-%d')

# 1. Time series data by field - threshold evolution over release dates
time_series_data = {}
for field in df['RESEARCH FIELDS'].unique():
    field_df = df[df['RESEARCH FIELDS'] == field]
    # For each publication year, track threshold over time
    field_data = {}
    for year in sorted(field_df['Year'].unique()):
        year_df = field_df[field_df['Year'] == year].sort_values('Released_date')
        field_data[int(year)] = {
            'dates': year_df['Date_str'].tolist(),
            'thresholds': year_df['Threshold-Highly Cited'].tolist()
        }
    time_series_data[field] = field_data

# 2. Latest thresholds by field for comparison
latest_date = df['Released date'].max()
latest_df = df[df['Released date'] == latest_date]
field_comparison = latest_df.groupby('RESEARCH FIELDS').apply(
    lambda x: dict(zip(x['Year'].tolist(), x['Threshold-Highly Cited'].tolist()))
).to_dict()

# 3. Threshold vs Paper Age data
age_analysis = df.groupby(['RESEARCH FIELDS', 'Paper_age'])['Threshold-Highly Cited'].mean().reset_index()
age_data = {}
for field in age_analysis['RESEARCH FIELDS'].unique():
    field_age = age_analysis[age_analysis['RESEARCH FIELDS'] == field]
    age_data[field] = {
        'ages': field_age['Paper_age'].tolist(),
        'thresholds': field_age['Threshold-Highly Cited'].tolist()
    }

# 4. Heatmap data - latest thresholds
heatmap_data = {
    'fields': sorted(df['RESEARCH FIELDS'].unique().tolist()),
    'years': sorted(latest_df['Year'].unique().tolist()),
    'values': []
}
for field in heatmap_data['fields']:
    row = []
    for year in heatmap_data['years']:
        val = latest_df[(latest_df['RESEARCH FIELDS'] == field) & (latest_df['Year'] == year)]['Threshold-Highly Cited']
        row.append(int(val.values[0]) if len(val) > 0 else None)
    heatmap_data['values'].append(row)

# 5. Average threshold by field
avg_by_field = df.groupby('RESEARCH FIELDS')['Threshold-Highly Cited'].mean().sort_values(ascending=True)
field_avg_data = {
    'fields': avg_by_field.index.tolist(),
    'averages': avg_by_field.values.tolist()
}

# 6. Growth rate analysis
growth_data = {}
for field in df['RESEARCH FIELDS'].unique():
    field_df = df[df['RESEARCH FIELDS'] == field]
    # Calculate average threshold by paper age
    age_avg = field_df.groupby('Paper_age')['Threshold-Highly Cited'].mean()
    growth_data[field] = {
        'ages': age_avg.index.tolist(),
        'thresholds': age_avg.values.tolist()
    }

# 7. Prediction data
pred_df = pd.read_excel('Predicted_Thresholds_Jan2026.xlsx')
prediction_data = {}
for field in pred_df['Research Field'].unique():
    field_pred = pred_df[pred_df['Research Field'] == field]
    prediction_data[field] = {
        'years': field_pred['Publication Year'].tolist(),
        'predicted': field_pred['Predicted Threshold'].tolist()
    }

# Combine all data
dashboard_data = {
    'time_series': time_series_data,
    'field_comparison': field_comparison,
    'age_analysis': age_data,
    'heatmap': heatmap_data,
    'field_avg': field_avg_data,
    'growth': growth_data,
    'predictions': prediction_data,
    'metadata': {
        'fields': sorted(df['RESEARCH FIELDS'].unique().tolist()),
        'latest_date': latest_df['Date_str'].iloc[0],
        'date_range': [df['Date_str'].min(), df['Date_str'].max()],
        'year_range': [int(df['Year'].min()), int(df['Year'].max())]
    }
}

# Save to JSON
with open('dashboard_data.json', 'w') as f:
    json.dump(dashboard_data, f)

print("Dashboard data exported to dashboard_data.json")
print(f"Fields: {len(dashboard_data['metadata']['fields'])}")
print(f"Date range: {dashboard_data['metadata']['date_range']}")

