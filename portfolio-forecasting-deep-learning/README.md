# Deep Learning for Portfolio Management & Asset Allocation

This project demonstrates how deep learning models (LSTM and Transformer) can enhance portfolio return forecasting and dynamic asset allocation.

## Project Structure
├── data/                         # Input & output CSVs (features, predictions)
├── models/                       # Saved model weights (.pt files)
├── scripts/
│   ├── data_loader.py            # Downloads and stores historical financial data
│   ├── feature_engineering.py    # Creates technical indicators
│   ├── lstm_model.py             # Trains LSTM model and saves predictions
│   ├── transformer_model.py      # Trains Transformer model and saves predictions
│   ├── portfolio_allocation.py   # Evaluates performance metrics
├── dashboard.py                  # Streamlit dashboard for visualization
├── requirements.txt
├── README.md

## How to Run
### **Install dependencies:**
pip install -r requirements.txt

### **Run pipeline:**

python scripts/data_loader.py
python scripts/feature_engineering.py
python scripts/lstm_model.py
python scripts/transformer_model.py
python scripts/portfolio_allocation.py

### **Launch Dashboard:**

streamlit run dashboard.py

## 📊 Models Used
**LSTM:** Recurrent model capturing long-term trends in asset returns.

**Transformer:** Attention-based model capturing temporal dependencies and volatility patterns.

### 📈 Evaluation Metrics

Annualized Return
Annualized Volatility
Sharpe Ratio
Maximum Drawdown

## 📌 Note
**The focus is on demonstrating:**
Deep learning in time-series forecasting
Practical asset allocation using model predictions
Performance evaluation using standard financial KPIs