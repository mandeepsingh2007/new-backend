import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

np.random.seed(42)

# Output directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_stock_data(user_input, period="1y", interval="1d"):
    print("[INFO] Fetching stock data...")
    try:
        ticker = f"{user_input}.NS" if ".NS" not in user_input else user_input
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            raise Exception("Fetched DataFrame is empty.")

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"[SUCCESS] Data fetched for {ticker}: {df.shape[0]} rows")

        df.to_csv(os.path.join(OUTPUT_DIR, f"{user_input}_data.csv"), index=True)
        return df
    except Exception as e:
        print(f"[ERROR] Could not fetch stock data: {e}")
        return None

def get_news_summary(stock_name, ticker_symbol):
    print(f"[INFO] Getting news summary for {stock_name} ({ticker_symbol})")
    return f"No live news feed — check Yahoo Finance, Economic Times, etc. for {stock_name}"

def preprocess_data(df, feature='Close', test_size=0.2):
    print("[INFO] Preprocessing data...")
    data = df[feature].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    training_data_len = int(np.ceil(len(scaled_data) * (1 - test_size)))
    train_data = scaled_data[:training_data_len]
    test_data = scaled_data[training_data_len - 60:]
    print(f"[INFO] Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    return train_data, test_data, scaler

def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    print(f"[INFO] Sequence shape: X={X.shape}, y={y.shape}")
    return X, y

def build_lstm_model(input_shape):
    print("[INFO] Building LSTM model...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
    print("[INFO] Training model...")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=1
    )
    return model, history

def make_predictions(model, test_data, scaler, time_steps=60):
    print("[INFO] Making predictions...")
    X_test = []
    y_test = test_data[time_steps:, 0]
    for i in range(time_steps, len(test_data)):
        X_test.append(test_data[i-time_steps:i, 0])
    X_test = np.array(X_test).reshape(-1, time_steps, 1)
    predictions = scaler.inverse_transform(model.predict(X_test))
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    return predictions, y_actual

def evaluate_model(predictions, actual):
    print("[INFO] Evaluating model...")
    rmse = np.sqrt(np.mean((predictions - actual) ** 2))
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    print(f"[RESULT] RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    return rmse, mape

def plot_results(df, predictions, actual, feature='Close', ticker='Stock'):
    print("[INFO] Plotting prediction results...")
    plt.figure(figsize=(16, 8))
    train = df[:len(df) - len(predictions)]
    valid = df[len(df) - len(predictions):]
    valid_df = valid.copy()
    valid_df['Predictions'] = predictions
    plt.title(f'{ticker} {feature} Price Prediction')
    plt.plot(train[feature])
    plt.plot(valid_df[[feature, 'Predictions']])
    plt.legend(['Train', 'Actual', 'Predicted'])
    plt.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_prediction.png"))
    plt.close()

def forecast_future(model, last_sequence, scaler, days=30):
    print("[INFO] Forecasting future prices...")
    future_preds = []
    current_seq = last_sequence.reshape((1, last_sequence.shape[0], 1))
    for _ in range(days):
        pred = model.predict(current_seq)[0]
        future_preds.append(pred[0])
        current_seq = np.append(current_seq[:, 1:, :], [[pred]], axis=1)
    return scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

def generate_future_dates(last_date, days):
    print("[INFO] Generating future dates...")
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
    print("[INFO] Analyzing trend...")
    start = future_df[feature].iloc[0]
    end = future_df[feature].iloc[-1]
    percent = ((end - start) / start) * 100
    trend = "UPWARD" if percent > 0 else "DOWNWARD"
    print(f"[TREND] {trend} ({percent:.2f}%) from ₹{start:.2f} to ₹{end:.2f}")
    return {"trend": trend, "percent_change": percent}

def run_stock_prediction(ticker, feature='Close', time_steps=60, test_size=0.2, epochs=50, forecast_days=30):
    df = get_stock_data(ticker)
    if df is None:
        print("[ERROR] Data fetch failed. Exiting...")
        return None, None, None, None

    train, test, scaler = preprocess_data(df, feature, test_size)
    X_train, y_train = create_sequences(train, time_steps)
    model = build_lstm_model((X_train.shape[1], 1))
    model, _ = train_model(model, X_train, y_train, epochs)
    predictions, actual = make_predictions(model, test, scaler, time_steps)
    evaluate_model(predictions, actual)
    plot_results(df, predictions, actual, feature, ticker)

    last_sequence = test[-time_steps:]
    future_preds = forecast_future(model, last_sequence, scaler, forecast_days)
    future_dates = generate_future_dates(df.index[-1], forecast_days)
    future_df = pd.DataFrame({'Date': future_dates, f'Predicted_{feature}': future_preds.flatten()})
    future_df.set_index('Date', inplace=True)

    future_df.to_csv(os.path.join(OUTPUT_DIR, f"{ticker}_forecast.csv"))
    plt.figure(figsize=(16, 8))
    plt.title(f"{ticker} Future Forecast")
    plt.plot(df[feature][-30:], label='Recent History')
    plt.plot(future_df[f'Predicted_{feature}'], label='Forecast')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_forecast.png"))
    plt.close()

    print(f"[SUCCESS] Files saved to '{OUTPUT_DIR}/'")
    return model, scaler, df, future_df

if __name__ == "__main__":
    print("\n=== INDIAN STOCK PRICE PREDICTION SYSTEM ===")
    ticker = "RELIANCE.NS"
    get_news_summary("Reliance Industries", ticker)
    model, scaler, df, future_df = run_stock_prediction(ticker)
    if future_df is not None:
        analyze_trend(future_df, "Predicted_Close")
    print("=== PREDICTION COMPLETE ===\n")
