# ai_agents/yfinance_agent.py

import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, days: int = 7) -> pd.DataFrame:
    ticker = ticker.strip().upper() + ".NS"
    stock = yf.Ticker(ticker)

    df = stock.history(period=f"{days}d", interval="1m")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.to_csv(f"{ticker}.csv", index=True)

    return df
