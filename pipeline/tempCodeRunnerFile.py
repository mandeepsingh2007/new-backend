import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from typing import List, Dict
import google.generativeai as genai
import sqlite3
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

user_input = input("Enter the stock name (e.g., RELIANCE): ").strip().upper()
symbol = input("enter stock ticker (e.g., RELIANCE.NS): ").strip().upper()
def get_stock_data(user_input):
    ticker = f"{user_input}.NS"
    stock = yf.Ticker(ticker)

    # Get minute-level data for last 7 days
    df = stock.history(period="7d", interval="1m")

    # Process the DataFrame
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    print(df)
    print("\n\n")
    print(df.head())
    print("\n\n")
    print(df.tail())
    print("\n\n")
    print(df.columns)
    print("\n\n")
    print(df.index)

    df.to_csv(f"{user_input}.csv", index=True)

    return df
def get_news_using_gemini(stock_name, ticker_symbol):
# Replace with your actual Gemini API key
    API_KEY = "AIzaSyAG4zSrJ-tt06NVMO3LxyjhPGqzYUXs7-k"

    # Initialize Gemini API
    genai.configure(api_key=API_KEY)

    # Initialize the model (using Gemini Pro)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    # Construct the prompt
    prompt = (
        f"Find the financial and all stock related news about {stock_name} ({ticker_symbol}) from the past 30 days. "
        "Summarize the key developments, including any major financial reports, partnerships, "
        "management changes, regulatory updates, or market trends affecting the stock. "
        "Provide sources and publication dates if available."
        "every single article, new, social status, leadership that can effect ther stock price"
    )

    # Generate the response from Gemini
    response = model.generate_content(prompt)


    print("\n📈 Latest News Summary:")
    print(response.text)
    # Return the response text
    text = response.text
    # Save text to a file (example path)
    with open("generated_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return response.text


def setup_database():
    """Setup SQLite database for storing market and news data"""
    conn = sqlite3.connect('trading_assistant.db')
    c = conn.cursor()
    
    # Create market data table
    c.execute('''
    CREATE TABLE IF NOT EXISTS market_data (
        id INTEGER PRIMARY KEY,
        ticker TEXT,
        date TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        timestamp INTEGER
    )
    ''')
    
    # Create news data table
    c.execute('''
    CREATE TABLE IF NOT EXISTS news_data (
        id INTEGER PRIMARY KEY,
        ticker TEXT,
        title TEXT,
        description TEXT,
        date TEXT,
        source TEXT,
        timestamp INTEGER
    )
    ''')
    
    conn.commit()
    conn.close()
    
def store_market_data(df, ticker):
    """Store market data in database"""
    conn = sqlite3.connect('trading_assistant.db')
    df_copy = df.reset_index()
    df_copy['ticker'] = ticker
    df_copy['timestamp'] = int(datetime.now().timestamp())
    df_copy.to_sql('market_data', conn, if_exists='append', index=False)
    conn.close()

def clean_market_data(df):
    """Clean and preprocess market data"""
    # Handle missing values
    df = df.fillna(method='ffill')  # Forward fill
    
    # Handle outliers using IQR method
    for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Replace outliers with bounds
        df[column] = df[column].clip(lower_bound, upper_bound)
    
    return df

def preprocess_news_data(df):
    """Preprocess news data"""
    # Convert date to datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['title'])
    
    # Combine title and description for sentiment analysis
    df['content'] = df['title'] + " " + df['description'].fillna("")
    
    # Clean text
    df['content'] = df['content'].str.lower()
    df['content'] = df['content'].str.replace(r'[^\w\s]', '', regex=True)
    
    return df

 # Technical Analysis Library


def engineer_features(df):
    """Create technical indicators and features using pandas/numpy only"""
    # Basic returns
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close']/df['Close'].shift(1))
    
    # Volatility (20-day rolling standard deviation of returns)
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Moving Averages
    df['sma_10'] = df['Close'].rolling(window=10).mean()
    df['sma_30'] = df['Close'].rolling(window=30).mean()
    
    # Exponential Moving Average
    df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Calculate RSI
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD)
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Price levels
    df['distance_from_sma_10'] = (df['Close'] - df['sma_10']) / df['sma_10']
    df['distance_from_sma_30'] = (df['Close'] - df['sma_30']) / df['sma_30']
    
    # Time-based features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    df['bb_std'] = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Indicator crossovers (signals)
    df['sma_crossover'] = np.where(
        (df['sma_10'] > df['sma_30']) & (df['sma_10'].shift(1) <= df['sma_30'].shift(1)), 
        1, 
        np.where(
            (df['sma_10'] < df['sma_30']) & (df['sma_10'].shift(1) >= df['sma_30'].shift(1)), 
            -1, 0
        )
    )
    
    # Rate of Change
    df['price_roc'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Support and Resistance (Simple version)
    df['resistance'] = df['High'].rolling(window=20).max()
    df['support'] = df['Low'].rolling(window=20).min()
    df['distance_to_resistance'] = (df['resistance'] - df['Close']) / df['Close']
    df['distance_to_support'] = (df['Close'] - df['support']) / df['Close']
    
    # Momentum Indicators
    df['momentum'] = df['Close'] - df['Close'].shift(4)
    
    # Volume features
    df['volume_ma'] = df['Volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']
    
    # Clean up NaN values created by rolling windows
    df = df.dropna()
    
    return df


def normalize_data(df, feature_columns):
    """Normalize features to [0,1] range"""
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df_scaled, scaler

def standardize_data(df, feature_columns):
    """Standardize features to have zero mean and unit variance"""
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df_scaled, scaler

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def create_sequence_dataset(df, target_column='Close', sequence_length=10, forecast_horizon=5):
    """
    Create sequences for time series forecasting
    - sequence_length: number of time steps to look back
    - forecast_horizon: number of time steps to predict ahead
    """
    features = df.drop(columns=[target_column]).values
    target = df[target_column].values
    
    X, y = [], []
    for i in range(len(df) - sequence_length - forecast_horizon + 1):
        X.append(features[i:i+sequence_length])
        y.append(target[i+sequence_length:i+sequence_length+forecast_horizon])
    
    return np.array(X), np.array(y)

def train_val_test_split(X, y, val_size=0.15, test_size=0.15):
    """
    Split dataset into train, validation and test sets
    - Respects time series order (not random)
    """
    n = len(X)
    test_idx = int(n * (1 - test_size))
    val_idx = int(n * (1 - test_size - val_size))
    
    X_train, y_train = X[:val_idx], y[:val_idx]
    X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
    X_test, y_test = X[test_idx:], y[test_idx:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def add_price_patterns(df, window=5):
    """Identify price patterns in market data"""
    # Pattern: Higher highs and higher lows (Uptrend)
    df['higher_highs'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
    df['higher_lows'] = (df['Low'] > df['Low'].shift(1)) & (df['Low'].shift(1) > df['Low'].shift(2))
    df['uptrend'] = df['higher_highs'] & df['higher_lows']
    
    # Pattern: Lower highs and lower lows (Downtrend)
    df['lower_highs'] = (df['High'] < df['High'].shift(1)) & (df['High'].shift(1) < df['High'].shift(2))
    df['lower_lows'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
    df['downtrend'] = df['lower_highs'] & df['lower_lows']
    
    # Pattern: Price consolidation (Sideways)
    high_range = df['High'].rolling(window=window).max() - df['High'].rolling(window=window).min()
    low_range = df['Low'].rolling(window=window).max() - df['Low'].rolling(window=window).min()
    df['consolidation'] = (high_range < df['Close'] * 0.03) & (low_range < df['Close'] * 0.03)
    
    return df

def add_advanced_features(df):
    """Add more sophisticated features"""
    # Stochastic Oscillator
    n = 14  # lookback period
    df['lowest_low'] = df['Low'].rolling(window=n).min()
    df['highest_high'] = df['High'].rolling(window=n).max()
    df['stoch_k'] = 100 * ((df['Close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low']))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Price velocity and acceleration
    df['price_velocity'] = df['Close'].diff(1)
    df['price_acceleration'] = df['price_velocity'].diff(1)
    
    # Volatility ratio
    df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=30).mean()
    
    # Price distance from moving averages
    df['price_ma_ratio'] = df['Close'] / df['sma_30']
    
    # Efficiency Ratio
    df['direction'] = abs(df['Close'] - df['Close'].shift(10))
    df['volatility_sum'] = df['Close'].diff().abs().rolling(window=10).sum()
    df['efficiency_ratio'] = df['direction'] / df['volatility_sum']
    
    return df

def create_market_regime_features(df):
    """Identify market regimes based on volatility and trend"""
    # Volatility regimes
    vol_quantile = df['volatility'].rolling(window=60).quantile(0.75)
    df['high_volatility'] = df['volatility'] > vol_quantile
    
    # Trend strength
    df['adx'] = calculate_adx(df, period=14)
    df['strong_trend'] = df['adx'] > 25
    
    # Market regime classification
    df['bull_market'] = (df['Close'] > df['sma_30']) & (df['sma_10'] > df['sma_30'])
    df['bear_market'] = (df['Close'] < df['sma_30']) & (df['sma_10'] < df['sma_30'])
    df['neutral_market'] = ~(df['bull_market'] | df['bear_market'])
    
    # Volatility regime classification
    df['low_vol_regime'] = df['volatility'] < df['volatility'].rolling(window=60).quantile(0.25)
    df['medium_vol_regime'] = (df['volatility'] >= df['volatility'].rolling(window=60).quantile(0.25)) & \
                             (df['volatility'] <= df['volatility'].rolling(window=60).quantile(0.75))
    df['high_vol_regime'] = df['volatility'] > df['volatility'].rolling(window=60).quantile(0.75)
    
    return df

def calculate_adx(df, period=14):
    """Calculate Average Directional Index (ADX) without talib"""
    # True Range
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Directional Movement
    up_move = df['High'] - df['High'].shift()
    down_move = df['Low'].shift() - df['Low']
    
    # Positive Directional Movement (+DM)
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    pos_di = 100 * (pd.Series(pos_dm).rolling(window=period).mean() / atr)
    
    # Negative Directional Movement (-DM)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    neg_di = 100 * (pd.Series(neg_dm).rolling(window=period).mean() / atr)
    
    # Directional Index
    dx = 100 * np.abs((pos_di - neg_di) / (pos_di + neg_di))
    
    # Average Directional Index
    adx = dx.rolling(window=period).mean()
    
    return adx

class TradingDataPipeline:
    """Complete data pipeline for trading assistant without TALib dependency"""
    
    def __init__(self, ticker_symbols, start_date, end_date):
        self.ticker_symbols = ticker_symbols
        self.start_date = start_date
        self.end_date = end_date
        self.market_data = None
        self.news_data = None
        self.feature_scaler = None
        self.sequence_length = 10
        self.forecast_horizon = 5
        self.feature_engineering_level = 'basic'  # Options: 'basic', 'advanced', 'full'
        
    def run_pipeline(self, feature_level='basic'):
        """Execute the full data pipeline"""
        self.feature_engineering_level = feature_level
        
        # Setup database
        setup_database()
        
        # Fetch data
        # Fetch data
        self.market_data = get_stock_data(self.ticker_symbols)
        self.news_data = get_news_using_gemini(self.stock_name, self.ticker_symbols)
        
        # Store data
        for ticker in self.ticker_symbols:
            if isinstance(self.market_data, pd.DataFrame):
                ticker_data = self.market_data.copy()
            else:  # MultiIndex DataFrame
                ticker_data = self.market_data[ticker].copy()
            store_market_data(ticker_data, ticker)
        
        # Clean and preprocess
        self.market_data = clean_market_data(self.market_data)
        self.news_data = preprocess_news_data(self.news_data)
        
        # Feature engineering based on selected level
        self.market_data = engineer_features(self.market_data)
        
        if feature_level in ['advanced', 'full']:
            self.market_data = add_advanced_features(self.market_data)
            self.market_data = add_price_patterns(self.market_data)
            
        if feature_level == 'full':
            self.market_data = create_market_regime_features(self.market_data)
        
        # Handle missing values after feature engineering
        self.market_data = self.market_data.dropna()
        
        # Define feature columns based on engineering level
        if feature_level == 'basic':
            feature_columns = [
                'returns', 'volatility', 'sma_10', 'sma_30', 'ema_10',
                'rsi', 'macd', 'macd_signal', 'distance_from_sma_10', 
                'distance_from_sma_30', 'volume_ratio'
            ]
        elif feature_level == 'advanced':
            feature_columns = [
                'returns', 'volatility', 'sma_10', 'sma_30', 'ema_10',
                'rsi', 'macd', 'macd_signal', 'atr', 'distance_from_sma_10', 
                'distance_from_sma_30', 'volume_ratio', 'stoch_k', 'stoch_d',
                'price_velocity', 'price_acceleration', 'uptrend', 'downtrend'
            ]
        else:  # 'full'
            # Use all numeric columns except for date/time and target columns
            exclude_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            feature_columns = [col for col in self.market_data.columns 
                              if col not in exclude_columns 
                              and np.issubdtype(self.market_data[col].dtype, np.number)]
        
        # Normalize features
        self.market_data, self.feature_scaler = standardize_data(
            self.market_data, feature_columns)
        
        # Create sequences for ML
        X, y = create_sequence_dataset(
            self.market_data, 
            target_column='Close',
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon
        )
        
        # Split data
        return train_val_test_split(X, y)
    def prepare_latest_data(self):
        """Prepare the most recent data for prediction"""
        if self.market_data is None:
            raise ValueError("Pipeline must be run before preparing latest data")
            
        latest_sequence = self.market_data.iloc[-self.sequence_length:].copy()
        # Apply same preprocessing steps as in pipeline
        return latest_sequence
    
def schedule_data_updates():
    """Set up scheduled data updates"""
    import schedule
    import time
    
    def update_data():
        # Define what data needs updating
        tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL']
        end_date = datetime.now().strftime('%Y-%m-%d')
        # Get yesterday's date for incremental updates
        start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Initialize pipeline for just the new data
        pipeline = TradingDataPipeline(tickers, start_date, end_date)
        pipeline.run_pipeline()
        print(f"Data updated at {datetime.now()}")
    
    # Schedule updates for market data
    schedule.every().day.at("18:00").do(update_data)  # After market close
    
    # Run the scheduler in a separate thread
    import threading
    cease_continuous_run = threading.Event()
    
    class ScheduleThread(threading.Thread):
        @classmethod
        def run(cls):
            while not cease_continuous_run.is_set():
                schedule.run_pending()
                time.sleep(60)
    
    continuous_thread = ScheduleThread()
    continuous_thread.start()
    
    return cease_continuous_run  # Return this to stop the scheduler if needed

def main():
    # Define parameters
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL']
    start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Initialize and run the data pipeline
    pipeline = TradingDataPipeline(tickers, start_date, end_date)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = pipeline.run_pipeline()
    
    print(f"Training set shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
    print(f"Test set shape: {X_test.shape}, {y_test.shape}")
    
    # Schedule regular updates
    stop_updates = schedule_data_updates()
    
if __name__ == "__main__":
    main()