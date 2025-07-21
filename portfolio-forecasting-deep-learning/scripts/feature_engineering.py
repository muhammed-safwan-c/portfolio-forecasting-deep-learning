# scripts/feature_engineering.py

import pandas as pd
import os

def preprocess_features(input_path="data/assets.csv", output_path="data/features.csv"):
    os.makedirs("data", exist_ok=True)

    # Load the cleaned dataset
    df = pd.read_csv(input_path)
    
    # Keep only essential columns
    df = df[['Date', 'Ticker', 'Close']].dropna()

    # Ensure Date column is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by ticker and date
    df = df.sort_values(by=['Ticker', 'Date'])

    # Calculate daily returns
    df['Return'] = df.groupby('Ticker')['Close'].pct_change()

    # Rolling window features
    df['MA_7'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=7).mean())
    df['MA_30'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=30).mean())
    df['Volatility_30'] = df.groupby('Ticker')['Return'].transform(lambda x: x.rolling(window=30).std())

    # Drop rows with NaNs caused by rolling/shift
    df = df.dropna().reset_index(drop=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Feature data saved to {output_path}")

if __name__ == "__main__":
    preprocess_features()
