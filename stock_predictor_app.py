import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt

# --- App Configuration ---
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Prediction App")
st.markdown("This app uses a Long Short-Term Memory (LSTM) network to predict the closing price of a stock.")


# --- Data Loading and Caching ---
# Using cache_data for data loading to avoid re-downloading
@st.cache_data
def load_data(stock_ticker, start_date, end_date):
    """Loads historical stock data from Yahoo Finance."""
    try:
        data = yf.download(stock_ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for ticker '{stock_ticker}'. Please check the stock symbol.")
            return None
        return data['Close']
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


# --- Model Training and Caching ---
# Using cache_resource for the model and scaler to avoid retraining on every interaction
@st.cache_resource
def train_model(data):
    """
    Prepares data, trains the LSTM model, and returns the model and scaler.
    """
    if data is None or len(data) < 100:
        return None, None

    # 1. Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))

    # 2. Create training dataset
    x_train, y_train = [], []
    for i in range(100, len(scaled_data)):
        x_train.append(scaled_data[i - 100:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # 3. Build and train the LSTM model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(64, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model (using fewer epochs for a web app to keep it responsive)
    model.fit(x_train, y_train, batch_size=1, epochs=5)

    return model, scaler


# --- Sidebar for User Input ---
st.sidebar.header("User Input")
stock = st.sidebar.text_input("Enter Stock Ticker", "GOOG").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

if st.sidebar.button("Predict", type="primary"):
    # --- Main Logic ---
    # 1. Load Data
    data = load_data(stock, start_date, end_date)

    if data is not None:
        # 2. Train Model (or retrieve from cache)
        with st.spinner(f"Training model for {stock}... This may take a few minutes the first time."):
            model, scaler = train_model(data)

        if model is not None and scaler is not None:
            st.success(f"Model trained successfully for {stock}!")

            # --- Visualization ---
            st.subheader("Original Closing Price")
            fig1 = plt.figure(figsize=(12, 6))
            plt.plot(data)
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.title(f"{stock} Closing Price History")
            st.pyplot(fig1)

            # --- Prediction ---
            # 1. Prepare the last 100 days of data for prediction
            last_100_days = np.array(data[-100:]).reshape(-1, 1)
            last_100_days_scaled = scaler.transform(last_100_days)

            x_input = last_100_days_scaled.reshape(1, -1)
            x_input = np.reshape(x_input, (1, 100, 1))

            # 2. Get the prediction
            predicted_price_scaled = model.predict(x_input)
            predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]

            st.metric(f"Predicted Price for next trading day for {stock}", f"${predicted_price:.2f}")

            # --- Plotting Prediction vs Original ---
            st.subheader("Prediction vs Original Data")
            fig2 = plt.figure(figsize=(12, 6))

            # Prepare data for plotting predictions
            test_data = np.array(data).reshape(-1, 1)
            scaled_test_data = scaler.transform(test_data)

            x_test_plot = []
            for i in range(100, len(scaled_test_data)):
                x_test_plot.append(scaled_test_data[i - 100:i, 0])
            x_test_plot = np.array(x_test_plot)
            x_test_plot = np.reshape(x_test_plot, (x_test_plot.shape[0], x_test_plot.shape[1], 1))

            predictions = model.predict(x_test_plot)
            predictions_unscaled = scaler.inverse_transform(predictions)

            plt.plot(data.index, data.values, label="Original Price")
            # Create a date range for the predictions
            prediction_dates = data.index[100:]
            plt.plot(prediction_dates, predictions_unscaled, label="Predicted Price", color='red')
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.title(f"{stock} Price Prediction")
            plt.legend()
            st.pyplot(fig2)

        else:
            st.warning("Could not train the model. The dataset might be too small (less than 100 days).")

