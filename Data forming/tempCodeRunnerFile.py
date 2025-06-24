import yfinance as yf
import pandas as pd

# NSE stock with .NS suffix
user_input = input("Enter the stock ticker (e.g., RELIANCE): ").strip().upper()
# Append .NS to the ticker for NSE stocks
ticker = f"{user_input}.NS"
stock = yf.Ticker(ticker)

# Get minute-level data for last 7 days
df = stock.history(period="30d", interval="1m")

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
print("\n\n")
print("Data saved to CSV file.")