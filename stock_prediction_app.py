# -*- coding: utf-8 -*-
"""Untitled17.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mDWwCOLeFW-kWhXvAcU2-9lEJzsorGRC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# Streamlit App Title
st.title("Stock Market Prediction using SARIMA")

# Upload dataset
uploaded_file = st.file_uploader("/content/AAPL.csv", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
    df = df.asfreq('B')  # Ensure business day frequency
    df = df[['Close']]
    df.loc[:, 'Close'] = df['Close'].ffill()  # Fill missing values

    # Display dataset
    st.subheader("Dataset")
    st.write(df.tail())

    # Plot the series
    st.subheader("Stock Price Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Close'], label='Close Price')
    ax.set_title("Stock Price Over Time")
    ax.legend()
    st.pyplot(fig)

    # Check for stationarity using ADF test
    st.subheader("Stationarity Check")
    adf_test = adfuller(df['Close'].dropna())
    st.write(f'ADF Statistic: {adf_test[0]}')
    st.write(f'p-value: {adf_test[1]}')

    # If data is not stationary, apply differencing
    if adf_test[1] > 0.05:
        df_diff = df.diff().dropna()
        st.write("Data is not stationary. Applying differencing.")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_diff, label='Differenced Data (1st Order)')
        ax.set_title("Differenced Data (1st Order)")
        ax.legend()
        st.pyplot(fig)
    else:
        df_diff = df.copy()
        st.write("Data is stationary.")

    # Plot ACF & PACF
    st.subheader("ACF and PACF Plots")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_acf(df_diff, lags=40, ax=ax1)
    plot_pacf(df_diff, lags=40, ax=ax2)
    st.pyplot(fig)

    # Split dataset into train and test
    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]

    # Fit SARIMA Model
    st.subheader("SARIMA Model Training")
    sarima_model = SARIMAX(train['Close'], order=(3,1,3), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False)
    sarima_result = sarima_model.fit(disp=False)
    st.write("SARIMA Model Fitted Successfully!")

    # Forecast on test data
    sarima_pred = sarima_result.forecast(steps=len(test))

    # Calculate Evaluation Metrics
    st.subheader("Model Evaluation")
    def evaluate_model(true, pred, model_name):
        mae = mean_absolute_error(true, pred)
        mse = mean_squared_error(true, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(true, pred)
        st.write(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2 Score: {r2:.4f}")
        return mae, rmse, r2

    evaluate_model(test['Close'], sarima_pred, "SARIMA Model")

    # Plot Test Predictions
    st.subheader("Test Predictions")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test['Close'], label='Actual')
    ax.plot(sarima_pred, label='Predicted', linestyle='dashed')
    ax.set_title("SARIMA Test Predictions")
    ax.legend()
    st.pyplot(fig)

    # Forecast Next 30 Days
    st.subheader("30-Day Future Predictions")
    future_forecast = sarima_result.get_forecast(steps=30)
    forecast_mean = future_forecast.predicted_mean
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')
    forecast_df = pd.DataFrame({'Predicted': forecast_mean.values}, index=future_dates)

    # Plot Future Predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Close'][-100:], label='Historical Data')
    ax.plot(forecast_df, label='Future Predictions', linestyle='dashed')
    ax.set_title("SARIMA 30-Day Future Predictions")
    ax.legend()
    st.pyplot(fig)

    # Display Forecasted Values
    st.write("Forecasted Values for the Next 30 Days:")
    st.write(forecast_df)

else:
    st.write("Please upload a CSV file to get started.")