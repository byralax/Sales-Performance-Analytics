# Sales-Performance-Analytics
Python-powered dashboard analyzing 2M+ sales records with predictive insights
✅ Project Structure
kotlin
Copy
Edit
sales-performance-analytics/
├── data/
│   └── sales_2024.csv           ✅ (You already have this)
├── notebooks/
│   ├── EDA.ipynb                🔍 Exploratory analysis
│   └── forecast_model.ipynb     📈 Forecast with ML
├── dashboard/
│   └── app.py                   📊 Interactive dashboard
├── models/
│   └── linear_model.pkl         🧠 Trained ML model
├── requirements.txt             📦 Dependencies
└── README.md                    📘 Project overview
📦 1. requirements.txt
txt
Copy
Edit
pandas
numpy
plotly
dash
scikit-learn
matplotlib
seaborn
📊 2. notebooks/EDA.ipynb (summary content)
I'll describe the content to paste in Jupyter:

python
Copy
Edit
# EDA.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/sales_2024.csv')
df['date'] = pd.to_datetime(df['date'])

# Monthly revenue trend
monthly = df.resample('M', on='date').sum(numeric_only=True)
monthly['revenue'].plot(title="Monthly Revenue Trend", figsize=(12, 5))

# Top 5 products
top_products = df.groupby('product')['revenue'].sum().sort_values(ascending=False)
top_products.plot(kind='bar', title="Top Performing Products")

# Region heatmap
pivot = pd.pivot_table(df, values='revenue', index='region', columns='product', aggfunc='sum')
sns.heatmap(pivot, annot=False, cmap='viridis')
🤖 3. notebooks/forecast_model.ipynb
python
Copy
Edit
# forecast_model.ipynb

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('../data/sales_2024.csv')
df['date'] = pd.to_datetime(df['date'])

# Aggregate monthly
monthly = df.resample('M', on='date').sum(numeric_only=True).reset_index()
monthly['month'] = monthly['date'].dt.month
monthly['year'] = monthly['date'].dt.year

# Features
X = monthly[['month', 'year']]
y = monthly['revenue']

model = LinearRegression()
model.fit(X, y)

# Save model
with open('../models/linear_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Predict next 12 months
import numpy as np
future = pd.DataFrame({
    'month': list(range(1, 13)),
    'year': [2025]*12
})
future['predicted_revenue'] = model.predict(future)
print(future)
📈 4. dashboard/app.py
python
Copy
Edit
# app.py

import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc
from datetime import datetime

df = pd.read_csv('../data/sales_2024.csv')
df['date'] = pd.to_datetime(df['date'])

app = Dash(__name__)

monthly = df.resample('M', on='date').sum(numeric_only=True).reset_index()
top_regions = df.groupby('region')['revenue'].sum().reset_index()

app.layout = html.Div([
    html.H1("📊 Sales Performance Dashboard", style={'textAlign': 'center'}),
    dcc.Graph(figure=px.line(monthly, x='date', y='revenue', title='Monthly Revenue')),
    dcc.Graph(figure=px.pie(top_regions, names='region', values='revenue', title='Revenue by Region'))
])

if __name__ == '__main__':
    app.run_server(debug=True)
📘 5. README.md
Already complete (you can copy from above).



Run locally with:

bash
Copy
Edit
pip install -r requirements.txt
cd dashboard
python app.py
