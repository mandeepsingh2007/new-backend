# ai_agents/yfinance_agent.py

import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, days: int = 7) -> list:
    ticker = ticker.strip().upper() + ".NS"
    stock = yf.Ticker(ticker)

    df = stock.history(period=f"{days}d", interval="1d")  # use daily data for clarity
    if df.empty:
        return []

    df.reset_index(inplace=True)  # move the datetime index into a 'Date' column
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Optional: Save to CSV for debugging
    df.to_csv(f"{ticker}.csv", index=False)

    # Convert DataFrame to list of dictionaries (JSON-serializable)
    stock_data = []
    for _, row in df.iterrows():
        stock_data.append({
            "Date": row["Date"].strftime("%Y-%m-%d"),  # Ensure JSON-safe format
            "Open": row["Open"],
            "High": row["High"],
            "Low": row["Low"],
            "Close": row["Close"],
            "Volume": row["Volume"]
        })

    return stock_data
