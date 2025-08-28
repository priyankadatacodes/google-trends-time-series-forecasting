# Save this as app.py and run with: streamlit run app.py

import streamlit as st
import pandas as pd
from pytrends.request import TrendReq
from datetime import datetime, timedelta
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

st.title("Google Trends Time Series Forecasting Dashboard")

# Set keywords list
keywords=['Artificial Intelligence','Machine Learning','Data Analytics']

# Sidebar inputs
selected_keyword = st.sidebar.selectbox("Select keyword", keywords)
days = st.sidebar.slider("Select number of days for historical data", 60, 365, 180)
forecast_days = st.sidebar.slider("Select number of days to forecast", 7, 60, 30)

# Fetch data function
@st.cache_data(show_spinner=True)
def fetch_data(keyword, days):
    pytrends = TrendReq(hl='en-US', tz=330)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    timeframe = start_date.strftime('%Y-%m-%d') + ' ' + end_date.strftime('%Y-%m-%d')
    pytrends.build_payload([keyword], timeframe=timeframe)
    df = pytrends.interest_over_time()
    if not df.empty and 'isPartial' in df.columns:
        df = df.drop(columns=['isPartial'])
    return df

# Prepare data for Prophet
def prepare_prophet_df(series):
    df_prophet = pd.DataFrame()
    df_prophet['ds'] = series.index
    df_prophet['y'] = series.values
    return df_prophet

# Forecast function using Prophet
@st.cache_resource(show_spinner=True)
def prophet_forecast(data, forecast_period):
    model = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=True)
    model.fit(data)
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)
    return model, forecast

# Main workflow
st.header(f"Keyword: {selected_keyword}")

data_df = fetch_data(selected_keyword, days)

if data_df.empty:
    st.error("No data found for the selected keyword and timeframe.")
else:
    st.subheader("Historical Data")
    st.line_chart(data_df[selected_keyword])

    prophet_df = prepare_prophet_df(data_df[selected_keyword])

    model, forecast = prophet_forecast(prophet_df, forecast_days)

    st.subheader("Forecast Plot")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Show forecast components
    st.subheader("Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # Download forecast data
    csv_buffer = StringIO()
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    st.download_button(label="Download Forecast Data as CSV", data=csv_data, file_name=f"{selected_keyword}_forecast.csv", mime='text/csv')

