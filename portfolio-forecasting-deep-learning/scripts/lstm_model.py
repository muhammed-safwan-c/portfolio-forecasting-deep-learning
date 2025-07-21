import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import math
import os

# ----- Hyperparameters -----
SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
TARGET_TICKER = "EEM"

# ----- Load data -----
df = pd.read_csv("data/features.csv")
df = df[df["Ticker"] == TARGET_TICKER].copy()
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)

features = ["Close", "Return", "MA_7", "MA_30", "Volatility_30"]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

X, y = [], []
for i in range(SEQUENCE_LENGTH, len(scaled_data)):
    X.append(scaled_data[i-SEQUENCE_LENGTH:i])
    y.append(df["Return"].iloc[i])
X, y = np.array(X), np.array(y)

# ----- Split -----
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ----- DataLoader -----
train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
test_ds = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ----- LSTM Model -----
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out.squeeze()

model = LSTMModel(input_size=X.shape[2])
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# ----- Training -----
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_dl):.6f}")

# ----- Evaluation -----
model.eval()
preds_list, actual_list, dates = [], [], []
with torch.no_grad():
    for xb, yb in test_dl:
        preds = model(xb)
        preds_list.extend(preds.numpy())
        actual_list.extend(yb.numpy())

rmse = math.sqrt(mean_squared_error(actual_list, preds_list))
print(f"\nTest RMSE: {rmse:.6f}")

# ----- Save Model -----
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/lstm_model.pt")

# ----- Save Predictions -----
preds_df = df.iloc[SEQUENCE_LENGTH + split:].copy()
preds_df["Predicted_Return"] = preds_list
preds_df.to_csv("data/lstm_predictions.csv", index=False)
