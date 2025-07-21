import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def evaluate_portfolio(df):
    df = df.copy()
    df["Portfolio_Value"] = (1 + df["Return"]).cumprod()

    # Annual return
    total_days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days
    annual_return = (df["Portfolio_Value"].iloc[-1]) ** (365 / total_days) - 1

    # Annualized volatility
    daily_vol = df["Return"].std()
    annual_volatility = daily_vol * np.sqrt(252)

    # Sharpe Ratio (assume risk-free rate = 0)
    sharpe_ratio = annual_return / annual_volatility

    # Max Drawdown
    rolling_max = df["Portfolio_Value"].cummax()
    drawdown = df["Portfolio_Value"] / rolling_max - 1
    max_drawdown = drawdown.min()

    # Print metrics
    print(f" Portfolio Evaluation Metrics")
    print(f"Annual Return:       {annual_return:.2%}")
    print(f"Annual Volatility:   {annual_volatility:.2%}")
    print(f"Sharpe Ratio:        {sharpe_ratio:.2f}")
    print(f"Max Drawdown:        {max_drawdown:.2%}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Portfolio_Value"], label="Portfolio Value")
    plt.title("Portfolio Cumulative Return")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run():
    df = pd.read_csv("data/lstm_predictions.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    # Use predicted return directly (1-day ahead return)
    df["Return"] = df["Return"].shift(-1)  # Shift to align with predicted day

    # Optional strategy: skip days with negative predicted return
    # df.loc[df["Predicted_Return"] < 0, "Return"] = 0

    df.dropna(inplace=True)
    evaluate_portfolio(df)


if __name__ == "__main__":
    run()
