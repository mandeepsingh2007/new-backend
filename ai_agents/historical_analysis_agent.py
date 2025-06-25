import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def historical_stock_analysis(ticker: str) -> dict:
    """
    Analyze the historical stock data of the given ticker for the past 1 year (daily interval).
    Returns basic feature engineering, model accuracy, and latest feature values.
    """
    # Define analysis period
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)  # 1 year

    try:
        # Download 1 year of daily data
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            progress=False
        )
    except Exception as e:
        return {"error": f"Failed to fetch data: {str(e)}"}

    if df.empty or len(df) < 30:
        return {"error": "Insufficient data returned for analysis."}

    # Feature engineering
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df.dropna(inplace=True)

    # Target: Predict if next day's return is positive
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)

    features = ['Return', 'MA5', 'MA20']
    if df[features].isnull().values.any():
        return {"error": "NaNs in feature set after processing."}

    X = df[features]
    y = df['Target']

    if len(X) < 50:
        return {"error": "Not enough data to train model."}

    # Split (preserving time order)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    except Exception as e:
        return {"error": f"Model training failed: {str(e)}"}

    # Fix: Convert any non-string keys to strings
    latest_features = df.iloc[-1][features].round(4).to_dict()
    latest_features_str_keys = {str(k): v for k, v in latest_features.items()}

    return {
        "ticker": ticker,
        "period_days": 365,
        "features_used": features,
        "accuracy": round(accuracy, 3),
        "latest_data_point": latest_features_str_keys
    }
