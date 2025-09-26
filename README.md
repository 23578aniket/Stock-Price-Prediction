📈 Stock Price Prediction App
<img width="2502" height="1303" alt="image" src="https://github.com/user-attachments/assets/488cff2b-7fee-4713-9f98-04939b67885c" />


View the Live Deployed Application Here [https://stockpredictorapppy-3jhfdzsjddtfvmr8fvnzff.streamlit.app/]
1. Project Overview
This project is a web-based application that predicts future stock prices using a Long Short-Term Memory (LSTM) neural network. The app, built with Streamlit, allows users to select any stock ticker available on Yahoo Finance, fetch its historical data, and generate a prediction for the next trading day.

The entire application is self-contained and demonstrates a professional machine learning workflow. When a user selects a new stock, the app fetches the data, preprocesses it, and trains a new LSTM model on-the-fly. To ensure a fast and responsive user experience, both the trained model and its corresponding data scaler are intelligently cached, preventing a lengthy retraining process on every interaction.

2. Tech Stack & Libraries
Language: Python

Web Framework: Streamlit

Machine Learning: TensorFlow (Keras)

Data Handling: Pandas, NumPy

Data Scaling: Scikit-learn (MinMaxScaler)

Data Source: yfinance library for real-time stock data

Visualization: Matplotlib

3. Key Features
Live Data Fetching: Pulls historical stock data directly from Yahoo Finance.

Dynamic Model Training: Trains a new LSTM model specifically for the user-selected stock ticker.

Intelligent Caching: Caches both the trained model and the data scaler (@st.cache_resource) to provide instant predictions after the initial training.

Robust Data Handling: Correctly manages the model and scaler together to prevent data leakage and ensure accurate predictions.

Interactive Visualization: Displays the historical stock price and the model's predictions overlaid on the original data using Matplotlib charts.

Simple UI: A clean and intuitive sidebar allows users to easily input the stock ticker and date range.

4. How to Run Locally
To run this project on your local machine, please follow these steps:

Clone the Repository:

git clone [https://github.com/your-username/stock-prediction-app.git](https://github.com/your-username/stock-prediction-app.git)
cd stock-prediction-app

Install Dependencies:
Make sure you have Python 3.8+ installed. Then, install all the required libraries using the requirements.txt file.

pip install -r requirements.txt

Run the Streamlit App:
Execute the following command in your terminal:

streamlit run app.py

The application will open in a new tab in your web browser. The first time you predict a stock, the model training may take a few minutes.
