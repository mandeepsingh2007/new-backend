import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Ensure it works without GUI (important for Render)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# Reproducibility
np.random.seed(42)

# Output directory for Render (safe even on cloud)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_stock_data(user_input, period="1y", interval="1d"):
    try:
        if ".NS" not in user_input:
            ticker = f"{user_input}.NS"
        else:
            ticker = user_input

        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            print(f"⚠️ No data found for {ticker}")
            return None

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.to_csv(os.path.join(OUTPUT_DIR, f"{user_input}_data.csv"))
        return df

    except Exception as e:
        print(f"❌ Error fetching stock data: {e}")
        return None

def get_news_summary(stock_name, ticker_symbol):
    message = (
        f"To get latest news about {stock_name} ({ticker_symbol}), refer:\n"
        "- Yahoo Finance\n- Economic Times\n- Moneycontrol\n- Bloomberg\n"
        "Gemini API was removed due to compatibility issues."
    )
    print(message)
    return message

def preprocess_data(df, feature='Close', test_size=0.2):
    data = df[feature].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    train_len = int(np.ceil(len(scaled_data) * (1 - test_size)))
    train_data = scaled_data[:train_len]
    test_data = scaled_data[train_len - 60:]
    return train_data, test_data, scaler

def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    return X.reshape(X.shape[0], X.shape[1], 1), y

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stop], verbose=1)
    return model

def make_predictions(model, test_data, scaler, time_steps=60):
    if len(test_data) <= time_steps:
        print("⚠️ Not enough test data.")
        return None, None
    X_test = [test_data[i-time_steps:i, 0] for i in range(time_steps, len(test_data))]
    X_test = np.array(X_test).reshape(-1, time_steps, 1)
    predictions = scaler.inverse_transform(model.predict(X_test))
    y_test_actual = scaler.inverse_transform(test_data[time_steps:].reshape(-1, 1))
    return predictions, y_test_actual

def evaluate_model(predictions, actual):
    rmse = np.sqrt(np.mean((predictions - actual) ** 2))
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    return rmse, mape

def plot_results(df, predictions, actual, feature, ticker):
    train = df[:len(df)-len(predictions)]
    valid = df[len(df)-len(predictions):]
    valid_df = valid.copy()
    valid_df['Predictions'] = predictions
    plt.figure(figsize=(16, 8))
    plt.title(f'{ticker} {feature} Price Prediction')
    plt.plot(train[feature])
    plt.plot(valid_df[[feature, 'Predictions']])
    plt.legend(['Train', 'Actual', 'Predictions'])
    filepath = os.path.join(OUTPUT_DIR, f"{ticker}_prediction.png")
    plt.savefig(filepath)
    plt.close()
    return filepath

def forecast_future(model, last_sequence, scaler, days=30):
    future_preds = []
    seq = last_sequence.reshape((1, last_sequence.shape[0], 1))
    for _ in range(days):
        pred = model.predict(seq)[0][0]
        future_preds.append(pred)
        seq = np.append(seq[:, 1:, :], [[[pred]]], axis=1)
    return scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

def generate_future_dates(last_date, days):
    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date)
    future_dates = []
    current = last_date
    while len(future_dates) < days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            future_dates.append(current)
    return pd.DatetimeIndex(future_dates)

def analyze_trend(future_df, feature='Predicted_Close'):
    first = future_df[feature].iloc[0]
    last = future_df[feature].iloc[-1]
    change = last - first
    percent = (change / first) * 100
    trend = "UPWARD" if percent > 0 else "DOWNWARD"
    volatility = future_df[feature].std()
    return {
        "trend": trend,
        "percent_change": round(percent, 2),
        "volatility": round(volatility, 2)
    }

def run_stock_prediction(ticker, feature='Close', time_steps=60, test_size=0.2, epochs=50, forecast_days=30):
    df = get_stock_data(ticker)
    if df is None or df.empty:
        return None
    train_data, test_data, scaler = preprocess_data(df, feature, test_size)
    X_train, y_train = create_sequences(train_data, time_steps)
    model = build_lstm_model((X_train.shape[1], 1))
    model = train_model(model, X_train, y_train, epochs)
    predictions, y_test_actual = make_predictions(model, test_data, scaler, time_steps)
    if predictions is None:
        return None
    evaluate_model(predictions, y_test_actual)
    plot_results(df, predictions, y_test_actual, feature, ticker)
    last_sequence = test_data[-time_steps:]
    future_prices = forecast_future(model, last_sequence, scaler, forecast_days)
    future_dates = generate_future_dates(df.index[-1], forecast_days)
    future_df = pd.DataFrame({f"Predicted_{feature}": future_prices.flatten()}, index=future_dates)
    forecast_csv_path = os.path.join(OUTPUT_DIR, f"{ticker}_forecast.csv")
    future_df.to_csv(forecast_csv_path)
    return {
        "future_df": future_df,
        "trend_analysis": analyze_trend(future_df, f"Predicted_{feature}"),
        "csv_path": forecast_csv_path
    }

if __name__ == "__main__":
    user_input = 'RELIANCE'
    result = run_stock_prediction(user_input)
    if result:
        print(result["future_df"].head())
        print("Trend Analysis:", result["trend_analysis"])
