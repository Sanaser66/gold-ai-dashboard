import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime

st.set_page_config(page_title="Gold AI Dashboard", layout="wide")
st.title("📈 Gold (XAU/USD) AI-Powered Trading Dashboard")

# User parameters
start_date = st.sidebar.date_input("Start Date", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Download data
data = yf.download("XAUUSD=X", start=start_date, end=end_date)
data.dropna(inplace=True)
st.subheader("Historical Gold Price (XAU/USD)")
st.line_chart(data['Close'])

# Feature Engineering
data['Return'] = data['Close'].pct_change()
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()
data.dropna(inplace=True)

# Prepare Data for ML
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Close']])

def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i+time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X, y = create_dataset(data_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, y, epochs=5, batch_size=32, verbose=0)

# Predict
train_predict = model.predict(X)
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
actual = scaler.inverse_transform(y.reshape(-1, 1))

# Plot predictions
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(actual, label='Actual')
ax.plot(train_predict, label='LSTM Predict')
ax.set_title('LSTM Prediction vs Actual')
ax.legend()
st.pyplot(fig)

# XGBoost for comparison
X_flat = X.reshape(X.shape[0], X.shape[1])
xgb_model = XGBRegressor(n_estimators=100)
xgb_model.fit(X_flat, y)
y_pred = xgb_model.predict(X_flat)
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

# Plot XGBoost
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(actual, label='Actual')
ax2.plot(y_pred, label='XGBoost Predict')
ax2.set_title('XGBoost Prediction vs Actual')
ax2.legend()
st.pyplot(fig2)

st.success("✅ Model training and predictions complete!")

# Optional Trade Suggestion
latest_close = data['Close'].iloc[-1]
next_pred = model.predict(X[-1].reshape(1, 60, 1))
next_pred = scaler.inverse_transform(next_pred)[0][0]

direction = "📈 BUY" if next_pred > latest_close else "📉 SELL"
st.metric(label="Suggested Trade", value=direction, delta=round(next_pred - latest_close, 2))
