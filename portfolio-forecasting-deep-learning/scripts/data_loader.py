# scripts/data_loader.py

import yfinance as yf
import pandas as pd
import os

def download_data(tickers, start="2010-01-01", end="2020-12-31", save_path="data/assets.csv"):
    os.makedirs("data", exist_ok=True)
    df_all = []

    for ticker in tickers:
        print(f"Downloading: {ticker}")
        try:
            data = yf.Ticker(ticker).history(start=start, end=end)
            if data.empty:
                print(f"Skipped {ticker} (no data)")
                continue
            data = data.reset_index()  # Move Date from index to column
            data["Ticker"] = ticker
            df_all.append(data)
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

    final_df = pd.concat(df_all, ignore_index=True)
    final_df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    tickers = [
        "^GSPC",   # S&P 500
        "^FTSE",   # FTSE 100
        "^N225",   # Nikkei 225
        "EEM",     # Emerging Markets ETF
        "GLD",     # Gold ETF
        "^TNX"     # 10-Year US Treasury
    ]
    download_data(tickers)
