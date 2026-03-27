import os
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import mysql.connector
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

cors_origins = os.getenv("CORS_ORIGINS", "*")
CORS(app, origins=cors_origins)

# config
MODEL_EPOCHS_DEFAULT = int(os.getenv("MODEL_EPOCHS", "20"))
MODEL_EPOCHS_MAX = 50
MODEL_REFRESH_HOURS = int(os.getenv("MODEL_REFRESH_HOURS", "24"))

# ✅ FIXED: Render-safe temp storage
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/tmp/models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASS", ""),
    "database": os.getenv("DB_NAME", "stocks")
}

# ✅ FIXED: Safe DB connection
def get_db():
    try:
        if not DB_CONFIG.get("password"):
            return None
        return mysql.connector.connect(**DB_CONFIG)
    except:
        return None


def model_file_path(symbol, tag):
    clean_symbol = "".join([c for c in symbol.upper() if c.isalnum() or c == "_"])
    return MODEL_DIR / f"{clean_symbol}_{tag}.pt"


def model_needs_refresh(path):
    if not path.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age > timedelta(hours=MODEL_REFRESH_HOURS)


# ─── MODELS ─────────────────────────────────────────────

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ─── HELPERS ─────────────────────────────────────────────

def clean_ohlcv(df):
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna()


def prepare_data(df, window):
    features = ['Open', 'High', 'Low', 'Close', 'Volume']

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i][3])

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).view(-1, 1),
        scaler
    )


def train_model(model, X, y, epochs=50):
    opt = optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()

    return model


def forecast_future(model, scaler, df, window, future_days=15):
    features = ['Open', 'High', 'Low', 'Close', 'Volume']

    last_seq = scaler.transform(df[features])[-window:]
    cur = torch.tensor(last_seq.reshape(1, window, 5), dtype=torch.float32)

    preds = []

    model.eval()
    with torch.no_grad():
        for _ in range(future_days):
            pred = model(cur).item()
            preds.append(pred)

            new_row = cur[:, -1, :].clone()
            new_row[0][3] = pred

            cur = torch.cat((cur[:, 1:, :], new_row.unsqueeze(0)), dim=1)

    preds = np.array(preds).reshape(-1, 1)

    dummy = np.zeros((len(preds), 5))
    dummy[:, 3] = preds[:, 0]

    return scaler.inverse_transform(dummy)[:, 3]


def build_future_df(preds, last_date):
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=len(preds), freq='B')

    rows = {"Open": [], "High": [], "Low": [], "Close": [], "Volume": []}

    prev = float(preds[0])

    for price in preds:
        price = float(price)

        opn = prev
        hi = max(opn, price) * 1.01
        lo = min(opn, price) * 0.99
        vol = 100000

        rows["Open"].append(round(opn, 2))
        rows["High"].append(round(hi, 2))
        rows["Low"].append(round(lo, 2))
        rows["Close"].append(round(price, 2))
        rows["Volume"].append(vol)

        prev = price

    df = pd.DataFrame(rows, index=future_dates)
    df.index = df.index.strftime("%Y-%m-%d")

    return df


def compute_metrics(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted[:len(actual)])

    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    return round(mae, 2), round(rmse, 2)


def df_to_records(df):
    records = []
    for date, row in df.iterrows():
        records.append({
            "date": str(date),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"])
        })
    return records


# ─── ROUTES ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models_cached": len(list(MODEL_DIR.glob("*.pt")))
    })


# ✅ FIXED DB ROUTES
@app.route("/api/stocks", methods=["GET"])
def get_stocks():
    conn = get_db()
    if conn is None:
        return jsonify([])
    cursor = conn.cursor()
    cursor.execute("SELECT symbol, name FROM companies")
    stocks = [{"symbol": s, "name": n} for s, n in cursor.fetchall()]
    conn.close()
    return jsonify(stocks)


@app.route("/api/stocks", methods=["POST"])
def add_stock():
    conn = get_db()
    if conn is None:
        return jsonify({"error": "Database not configured"}), 500

    data = request.json or {}
    symbol = data.get("symbol", "").upper().strip()
    name = data.get("name", "").strip()

    cursor = conn.cursor()
    cursor.execute("INSERT INTO companies (symbol,name) VALUES(%s,%s)", (symbol, name))
    conn.commit()
    conn.close()

    return jsonify({"success": True})


@app.route("/api/stocks/<symbol>", methods=["DELETE"])
def delete_stock(symbol):
    conn = get_db()
    if conn is None:
        return jsonify({"error": "Database not configured"}), 500

    cursor = conn.cursor()
    cursor.execute("DELETE FROM companies WHERE symbol=%s", (symbol.upper(),))
    conn.commit()
    conn.close()

    return jsonify({"success": True})


@app.route("/api/predict/<symbol>")
def predict_stock(symbol):
    try:
        df = clean_ohlcv(yf.download(symbol, period="4y", progress=False, threads=False))

        if df.empty:
            return jsonify({"error": "No data"}), 404

        window = 60
        last_date = df.index[-1]

        X, y, scaler = prepare_data(df, window)

        models = {
            "RNN": SimpleRNN(5, 64),
            "LSTM": SimpleLSTM(5, 64),
            "GRU": SimpleGRU(5, 64)
        }

        results = {}
        accuracy = {}

        for tag, net in models.items():
            model_path = model_file_path(symbol, tag)

            if model_path.exists():
                net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            else:
                net = train_model(net, X, y, epochs=20)
                torch.save(net.state_dict(), str(model_path))

            preds = forecast_future(net, scaler, df, window)

            df_fut = build_future_df(preds, last_date)
            results[tag] = df_to_records(df_fut)

            actual = df["Close"].tail(15).values
            mae, rmse = compute_metrics(actual, preds)

            accuracy[tag] = {"mae": mae, "rmse": rmse}

        return jsonify({
            "predicted": results,
            "accuracy": accuracy
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


# ─── RUN ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)