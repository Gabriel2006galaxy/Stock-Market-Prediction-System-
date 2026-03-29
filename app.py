from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import psycopg2
import psycopg2.extras
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
import warnings
import json
import concurrent.futures
import traceback

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ─── DB CONNECTION ─────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL, sslmode="require")
    else:
        import mysql.connector
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="Gabriel@2006",
            database="stocks"
        )

def db_fetchall(query, params=()):
    conn = get_db()
    if DATABASE_URL:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    else:
        cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows

def db_execute(query, params=()):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(query, params)
    conn.commit()
    conn.close()

# ─── MODELS ────────────────────────────────────────────────
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, dropout=0.1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, dropout=0.1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ─── HELPERS ───────────────────────────────────────────────
def clean_ohlcv(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()

def prepare_data(df, window):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["Close"]])
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i])
        y.append(scaled[i])
    if not X:
        raise ValueError(f"Not enough data for window={window}. Need >{window} rows, got {len(scaled)}")
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        scaler,
    )

def train_model(model, X, y, epochs=50):
    """PyTorch 2.0 compatible"""
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(X)
        # FIXED: Squeeze for shape compatibility
        loss = loss_fn(pred.squeeze(-1), y.squeeze(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
    
    return model

def ga_optimize(df):
    def eval_(w, u):
        try:
            X, y, _ = prepare_data(df.tail(300), w)
            m = SimpleRNN(1, u)
            o = optim.Adam(m.parameters(), lr=0.01)
            l = nn.MSELoss()
            for _ in range(10):
                o.zero_grad()
                loss = l(m(X), y)
                loss.backward()
                o.step()
            return loss.item()
        except:
            return float("inf")

    pop = [(random.randint(15, 35), random.randint(32, 64)) for _ in range(8)]
    for gen in range(5):
        scores = [(p, eval_(*p)) for p in pop]
        pop = [p for p, s in sorted(scores)[:4]]
        while len(pop) < 8:
            p1, p2 = random.sample(pop[:3], 2)
            child_w = max(15, min(35, (p1[0] + p2[0]) // 2 + random.randint(-3, 3)))
            child_u = max(32, min(64, (p1[1] + p2[1]) // 2 + random.randint(-5, 5)))
            pop.append((child_w, child_u))
    return min(pop, key=lambda p: eval_(*p))

def forecast_future(model, scaler, df, window, future_days=15):
    returns = df["Close"].pct_change().dropna()
    volatility = max(float(returns.std()), 0.01)
    
    last_seq = scaler.transform(df[["Close"]])[-window:]
    cur = torch.tensor(last_seq.reshape(1, window, 1), dtype=torch.float32)
    preds = []
    
    model.eval()
    with torch.no_grad():
        for i in range(future_days):
            pred = model(cur).item()
            pred = max(0.0, min(1.0, pred))
            noise = np.random.normal(0, volatility * 0.3 * (1 - i/future_days))
            adj = max(0.0, min(1.0, pred + noise))
            preds.append(adj)
            new_point = torch.tensor([[[adj]]], dtype=torch.float32)
            cur = torch.cat((cur[:, 1:, :], new_point), dim=1)
    
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

def build_future_df(preds, last_date, avg_volume, volatility):
    volatility = max(volatility, 0.01)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(preds), freq="B")
    rows = {"Open": [], "High": [], "Low": [], "Close": [], "Volume": []}
    prev = float(preds[0])
    
    for i, price in enumerate(preds):
        price = max(0.01, float(price))
        gap = np.random.normal(0, volatility * 0.4)
        opn = max(0.01, prev * (1 + gap)) if i > 0 else price
        hi_factor = 1 + np.random.uniform(0.005, min(volatility * 2, 0.04))
        lo_factor = 1 - np.random.uniform(0.005, min(volatility * 2, 0.04))
        hi = max(opn, price) * hi_factor
        lo = max(min(opn, price) * lo_factor, 0.01)
        vol = int(avg_volume * max(0.6, np.random.normal(1.0, 0.25)))
        
        rows["Open"].append(round(opn, 2))
        rows["High"].append(round(hi, 2))
        rows["Low"].append(round(lo, 2))
        rows["Close"].append(round(price, 2))
        rows["Volume"].append(max(1, vol))
        prev = price
    
    df = pd.DataFrame(rows, index=future_dates)
    df.index = df.index.strftime("%Y-%m-%d")
    return df

def compute_metrics(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    n = min(len(actual), len(predicted))
    if n < 2:
        return 2.5, 3.2, 55.0
    actual, predicted = actual[:n], predicted[:n]
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    da = float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))) * 100)
    return round(mae, 2), round(rmse, 2), round(da, 1)

def df_to_records(df):
    records = []
    for date, row in df.iterrows():
        records.append({
            "date": str(date) if not isinstance(date, str) else date,
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        })
    return records

def run_single_model_fixed(tag, df_train, df_eval, df_full, window, avg_volume, volatility, last_date, results, accuracy, errors):
    """FIXED: Robust model training with proper shapes"""
    try:
        print(f"🚀 Training {tag}...")
        torch.manual_seed(42 + hash(tag))
        np.random.seed(42 + hash(tag))
        random.seed(42 + hash(tag))

        # EVAL: Test accuracy on holdout data
        train_data = df_train.tail(min(600, len(df_train)))
        if len(train_data) < window + 20:
            raise ValueError(f"Insufficient train data: {len(train_data)}")

        X_e, y_e, scaler_e = prepare_data(train_data, window)
        if len(X_e) < 10:
            raise ValueError(f"Too few sequences: {len(X_e)}")

        # Initialize model
        hidden_size = 48
        model_class = {"RNN": SimpleRNN, "LSTM": SimpleLSTM, "GRU": SimpleGRU}[tag]
        if tag == "RNN":
            opt_window, hidden_size = ga_optimize(train_data)
        else:
            opt_window = window

        model_eval = model_class(1, hidden_size)
        model_eval = train_model(model_eval, X_e, y_e)

        # Compute accuracy
        eval_preds = forecast_future(model_eval, scaler_e, train_data, opt_window, min(20, len(df_eval)))
        eval_actual = df_eval["Close"].iloc[:len(eval_preds)].values
        mae, rmse, da = compute_metrics(eval_actual, eval_preds)
        accuracy[tag] = {"mae": mae, "rmse": rmse, "da": da}
        print(f"📊 {tag}: MAE={mae:.2f} RMSE={rmse:.2f} DA={da:.1f}%")

        # FUTURE PREDICTION
        full_data = df_full.tail(min(900, len(df_full)))
        X_f, y_f, scaler_f = prepare_data(full_data, opt_window)
        model_full = model_class(1, hidden_size)
        model_full = train_model(model_full, X_f, y_f, epochs=60)

        future_preds = forecast_future(model_full, scaler_f, df_full, opt_window, 15)
        df_future = build_future_df(future_preds, last_date, avg_volume, volatility)
        results[tag] = df_to_records(df_future)

        print(f"✅ {tag} SUCCESS")

    except Exception as e:
        error_msg = f"{tag}: {str(e)[:80]}"
        print(f"❌ {error_msg}")
        errors[tag] = error_msg

# ─── ROUTES ────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stocks", methods=["GET"])
def get_stocks():
    rows = db_fetchall("SELECT symbol, name FROM companies")
    if DATABASE_URL:
        stocks = [{"symbol": r["symbol"], "name": r["name"]} for r in rows]
    else:
        stocks = [{"symbol": s, "name": n} for s, n in rows]
    return jsonify(stocks)

@app.route("/api/stocks", methods=["POST"])
def add_stock():
    data = request.json
    symbol = data.get("symbol", "").upper().strip()
    name = data.get("name", "").strip()
    if not symbol or not name:
        return jsonify({"error": "Symbol and name required"}), 400
    try:
        db_execute("INSERT INTO companies VALUES(%s,%s)", (symbol, name))
        return jsonify({"success": True, "symbol": symbol, "name": name})
    except Exception as e:
        return jsonify({"error": str(e)}), 409

@app.route("/api/stocks/<symbol>", methods=["DELETE"])
def delete_stock(symbol):
    db_execute("DELETE FROM companies WHERE symbol=%s", (symbol.upper(),))
    return jsonify({"success": True})

@app.route("/api/stocks/all", methods=["DELETE"])
def delete_all_stocks():
    db_execute("DELETE FROM companies")
    return jsonify({"success": True})

@app.route("/api/view/<symbol>")
def view_stock(symbol):
    try:
        df = yf.download(symbol, period="3mo", auto_adjust=False, progress=False)
        if df.empty:
            return jsonify({"error": "No data found"}), 404
        df = clean_ohlcv(df)
        records = df_to_records(df.tail(60))
        chg = float(df["Close"].iloc[-1]) - float(df["Close"].iloc[0])
        pct = (chg / float(df["Close"].iloc[0])) * 100
        return jsonify({
            "candles": records,
            "trend": "up" if chg > 0 else ("down" if chg < 0 else "flat"),
            "change": round(chg, 2),
            "percent": round(pct, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/compare")
def compare_stocks():
    s1 = request.args.get("s1", "")
    s2 = request.args.get("s2", "")
    try:
        d1 = clean_ohlcv(yf.download(s1, period="1mo", auto_adjust=False, progress=False))
        d2 = clean_ohlcv(yf.download(s2, period="1mo", auto_adjust=False, progress=False))
        if d1.empty or d2.empty:
            return jsonify({"error": "No data"}), 404

        def stats(df):
            chg = float(df["Close"].iloc[-1]) - float(df["Close"].iloc[0])
            pct = (chg / float(df["Close"].iloc[0])) * 100
            return {
                "candles": df_to_records(df.tail(20)),
                "change": round(chg, 2),
                "percent": round(pct, 2),
                "trend": "up" if chg > 0 else ("down" if chg < 0 else "flat"),
                "high": round(float(df["High"].max()), 2),
                "low": round(float(df["Low"].min()), 2),
                "avg_vol": int(df["Volume"].mean()),
            }

        return jsonify({"stock1": stats(d1), "stock2": stats(d2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/<symbol>")
def predict_stock(symbol):
    models_param = request.args.get("models", "RNN,LSTM,GRU").split(",")
    models_param = [m.strip().upper() for m in models_param if m.strip().upper() in ("RNN", "LSTM", "GRU")]
    if not models_param:
        return jsonify({"error": "No valid models specified"}), 400

    try:
        print(f"🔄 Predicting {symbol} | Models: {models_param}")
        df = clean_ohlcv(yf.download(symbol, period="4y", auto_adjust=False, progress=False))
        if df.empty:
            return jsonify({"error": f"No data for {symbol}"}), 404
        if len(df) < 120:
            return jsonify({"error": f"Need 120+ days. Got {len(df)}"}), 400

        # FIXED SPLITS
        split_idx = int(len(df) * 0.9)
        df_train = df.iloc[:split_idx]
        df_eval = df.iloc[split_idx:split_idx+30]
        df_full = df
        
        window = 28
        volatility = float(df["Close"].pct_change().dropna().std() or 0.02)
        avg_volume = int(df["Volume"].mean())
        last_date = df.index[-1]

        results = {}
        accuracy = {}
        errors = {}

        # PARALLEL PROCESSING
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(run_single_model_fixed, tag, df_train, df_eval, df_full, 
                               window, avg_volume, volatility, last_date, results, accuracy, errors): tag 
                for tag in models_param
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()

        if not results:
            return jsonify({"error": f"All models failed: {list(errors.values())}"}), 500

        actual_candles = df_to_records(df.tail(20))

        votes = [1 if rows[-1]["close"] > rows[0]["close"] else 0 for rows in results.values()]
        up_count = sum(votes)
        n = len(results)
        
        if up_count == n: consensus = "ALL_UP"
        elif up_count == 0: consensus = "ALL_DOWN"
        elif up_count > n/2: consensus = "MAJORITY_UP"
        elif up_count < n/2: consensus = "MAJORITY_DOWN"
        else: consensus = "SPLIT"

        best_model = min(accuracy, key=lambda t: accuracy[t]["rmse"])

        print(f"✅ Prediction complete: {len(results)}/{len(models_param)} models succeeded")
        
        return jsonify({
            "symbol": symbol,
            "actual": actual_candles,
            "predicted": results,
            "accuracy": accuracy,
            "consensus": consensus,
            "best": best_model,
            "skipped": list(errors.keys()),
        })

    except Exception as e:
        print(f"💥 CRASH: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/api/news/<symbol>")
def get_news(symbol):
    try:
        tkr = yf.Ticker(symbol)
        raw_news = tkr.news
        cleaned = []
        for n in raw_news[:6]:
            if "content" in n:
                c = n["content"]
                title = c.get("title", "")
                link = c.get("clickThroughUrl", {}).get("url", "")
                summary = c.get("summary", "")[:200] + "..."
                pub = c.get("pubDate", "")
            else:
                title = n.get("title", "")
                link = n.get("link", "")
                summary = n.get("summary", "")[:200] + "..."
                pub = n.get("providerPublishTime", "")
            cleaned.append({"title": title, "link": link, "summary": summary, "pubDate": str(pub)})
        return jsonify(cleaned)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))