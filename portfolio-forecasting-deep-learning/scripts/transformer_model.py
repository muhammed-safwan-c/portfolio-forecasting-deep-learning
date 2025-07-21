import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

# Hyperparameters
SEQ_LENGTH = 30
EPOCHS = 30
LR = 0.001
MODEL_SAVE_PATH = "models/transformer_model.pt"
PREDICTION_PATH = "data/transformer_predictions.csv"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_dim=64, n_heads=2, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x[:, -1])  # Use last time step

def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

def train_model():
    df = pd.read_csv("data/features.csv")
    df = df[df['Ticker'] == 'EEM'].copy()
    df = df.dropna().reset_index(drop=True)

    features = ['Close', 'Return', 'MA_7', 'MA_30', 'Volatility_30']
    X_raw = df[features].values
    y_raw = df['Return'].values
    dates = df['Date'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    X_seq, y_seq = create_sequences(X_scaled, y_raw, SEQ_LENGTH)
    date_seq = dates[SEQ_LENGTH:]

    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    test_dates = date_seq[split:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    model = TransformerModel(input_size=X_train.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = loss_fn(output, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
        rmse = np.sqrt(mean_squared_error(y_test.numpy(), predictions))
        print(f"\nTransformer Test RMSE: {rmse:.6f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Transformer model saved to {MODEL_SAVE_PATH}")

    # Save predictions to CSV
    os.makedirs("data", exist_ok=True)
    output_df = pd.DataFrame({
        "Date": test_dates,
        "Return": y_test.numpy().flatten(),
        "Predicted_Return": predictions.flatten()
    })
    output_df.to_csv(PREDICTION_PATH, index=False)
    print(f"Predictions saved to {PREDICTION_PATH}")

if __name__ == "__main__":
    train_model()
