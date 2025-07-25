import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

# Page config
st.set_page_config(page_title="Gold Price Predictor", layout="wide")

st.title("📈 Gold Price Forecasting Dashboard")

# Load Data
@st.cache_data
def load_data():
    data = yf.download('XAUUSD=X', start='2010-01-01')
    return data

data = load_data()
st.subheader("Gold Price History")
st.line_chart(data['Close'])

# Feature Engineering
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Close']].values)

look_back = 60

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, look_back)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train-Test Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
with st.spinner("Training model..."):
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

# Predictions
predictions = model.predict(X_test)
predictions_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test)

# Show prediction results
st.subheader("Predicted vs Actual")
results = pd.DataFrame({
    'Actual': y_test_rescaled.flatten(),
    'Predicted': predictions_rescaled.flatten()
})
st.line_chart(results)

# RMSE
rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
st.metric("RMSE", f"{rmse:.2f}")
