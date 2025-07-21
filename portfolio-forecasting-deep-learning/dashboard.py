import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Sidebar model selection
model_choice = st.sidebar.selectbox("Choose Model", ["LSTM", "Transformer"])

# Load prediction file based on model choice
if model_choice == "LSTM":
    pred_df = pd.read_csv("data/lstm_predictions.csv")
    st.markdown("### Model: LSTM (Return Forecasting)")
elif model_choice == "Transformer":
    pred_df = pd.read_csv("data/transformer_predictions.csv")
    st.markdown("### Model: Transformer (Return Forecasting)")

# Parse date column
pred_df["Date"] = pd.to_datetime(pred_df["Date"])

# Evaluation function
def calculate_metrics(df):
    df = df.copy()
    df["Daily Return"] = df["Return"] * df["Predicted_Return"]
    df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod()

    ann_return = df["Daily Return"].mean() * 252
    ann_volatility = df["Daily Return"].std() * (252 ** 0.5)
    sharpe_ratio = ann_return / ann_volatility if ann_volatility != 0 else 0

    cumulative = df["Cumulative Return"]
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    return ann_return, ann_volatility, sharpe_ratio, max_drawdown, df

# Calculate metrics
ann_ret, ann_vol, sharpe_ratio, mdd, plot_df = calculate_metrics(pred_df)

# Streamlit UI
st.title("📊 Portfolio Performance Dashboard")

# Metric cards in a single horizontal row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Annual Return", f"{ann_ret * 100:.2f}%")
col2.metric("Annual Volatility", f"{ann_vol * 100:.2f}%")
col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
col4.metric("Max Drawdown", f"{mdd * 100:.2f}%")

# Line chart
st.markdown("### 📈 Cumulative Return Over Time")
fig, ax = plt.subplots(figsize=(12, 6))  # Wider figure
ax.plot(plot_df["Date"], plot_df["Cumulative Return"], color="blue", linewidth=1.5)

# Axis formatting
ax.set_title("Cumulative Portfolio Return", fontsize=14)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Cumulative Return", fontsize=12)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.tick_params(axis='x', rotation=45)

st.pyplot(fig)
