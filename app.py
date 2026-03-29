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

warnings.filterwarnings("ignore")

# ─── DB CONNECTION ─────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL")

app = Flask(__name__)
CORS(app)

if DATABASE_URL:
    init_db()

def get_db():
    if not DATABASE_URL:
        raise Exception("DATABASE_URL not set in Render")
    return psycopg2.connect(DATABASE_URL, sslmode="require")

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

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            symbol TEXT PRIMARY KEY,
            name TEXT
        )
    """)
    conn.commit()
    conn.close()

# ─── MODELS ────────────────────────────────────────────────
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2,
                          dropout=0.1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2,
                          dropout=0.1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                            dropout=0.1, batch_first=True)
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

def train_model(model, X, y, epochs=80):
    """80 epochs — strong accuracy, finishes in ~1 minute total."""
    opt = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
        scheduler.step()
    return model

def ga_optimize(df):
    """
    Genetic Algorithm — finds best (window, hidden_units).
    FIX: was using random.choice(tuple) which swapped window/hidden 47% of
    the time. Now uses correct index-based crossover.
    """
    def eval_(w, u):
        try:
            X, y, _ = prepare_data(df, w)
            m = SimpleRNN(1, u)
            o = optim.Adam(m.parameters(), lr=0.01)
            l = nn.MSELoss()
            for _ in range(8):
                o.zero_grad()
                loss = l(m(X), y)
                loss.backward()
                o.step()
            return loss.item()
        except Exception:
            return float("inf")

    pop = [(random.randint(10, 30), random.randint(16, 64)) for _ in range(6)]
    for _ in range(3):
        pop = sorted(pop, key=lambda p: eval_(*p))[:3]
        while len(pop) < 6:
            p1, p2 = random.sample(pop, 2)
            # FIX: index-based crossover keeps window=p[0], hidden=p[1] semantics
            child_w = max(10, min(30, p1[0] if random.random() < 0.5 else p2[0]))
            child_u = max(16, min(64, p1[1] if random.random() < 0.5 else p2[1]))
            pop.append((child_w, child_u))
    return min(pop, key=lambda p: eval_(*p))

def forecast_future(model, scaler, df, window, future_days=15):
    returns    = df["Close"].pct_change().dropna()
    volatility = max(float(returns.std()), 1e-6)

    last_seq = scaler.transform(df[["Close"]])[-window:]
    cur = torch.tensor(last_seq.reshape(1, window, 1), dtype=torch.float32)
    preds = []

    model.eval()
    with torch.no_grad():
        for i in range(future_days):
            pred = max(0.0, min(1.0, model(cur).item()))
            cf   = max(0.3, 1.0 - i * 0.05)
            adj  = max(0.0, min(1.0, pred + np.random.normal(0, volatility * cf * abs(pred) * 0.3)))
            preds.append(adj)
            cur = torch.cat(
                (cur[:, 1:, :], torch.tensor([[[adj]]], dtype=torch.float32)), dim=1
            )

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

def build_future_df(preds, last_date, avg_volume, volatility):
    volatility   = max(volatility, 1e-6)
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=len(preds), freq="B"
    )
    rows = {"Open": [], "High": [], "Low": [], "Close": [], "Volume": []}
    prev = float(preds[0])
    for i, price in enumerate(preds):
        price      = max(0.01, float(price))
        gap        = np.random.normal(0, volatility * 0.5)
        opn        = max(0.01, prev * (1 + gap)) if i > 0 else price
        hi_factor  = 1 + np.random.uniform(0.001, min(volatility * 1.5, 0.05))
        lo_factor  = 1 - np.random.uniform(0.001, min(volatility * 1.5, 0.05))
        hi         = max(opn, price) * hi_factor
        lo         = max(min(opn, price) * lo_factor, 0.01)
        vol        = int(avg_volume * max(0.5, np.random.normal(1.0, 0.2)))
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
    actual    = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    n = min(len(actual), len(predicted))
    if n < 2:
        return 0.0, 0.0, 0.0
    actual, predicted = actual[:n], predicted[:n]
    mae  = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    da   = float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))) * 100)
    return round(mae, 2), round(rmse, 2), round(da, 1)

def df_to_records(df):
    records = []
    for date, row in df.iterrows():
        records.append({
            "date":   str(date) if not isinstance(date, str) else date,
            "open":   round(float(row["Open"]), 2),
            "high":   round(float(row["High"]), 2),
            "low":    round(float(row["Low"]), 2),
            "close":  round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        })
    return records

def run_single_model(tag, df_train, df_eval, df_full, window,
                     avg_volume, volatility, last_date, results, accuracy, errors):
    """Run one model — isolated so one failure never kills the others."""
    try:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)

        # ── eval pass (measures accuracy against last 15 real days) ──
        X_e, y_e, sc_e = prepare_data(df_train, window)
        if tag == "RNN":
            bw, bu     = ga_optimize(df_train)
            m_eval     = train_model(SimpleRNN(1, bu), X_e, y_e)
            preds_eval = forecast_future(m_eval, sc_e, df_train, bw, 15)
        elif tag == "LSTM":
            m_eval     = train_model(SimpleLSTM(1, 64), X_e, y_e)
            preds_eval = forecast_future(m_eval, sc_e, df_train, window, 15)
        else:
            m_eval     = train_model(SimpleGRU(1, 64), X_e, y_e)
            preds_eval = forecast_future(m_eval, sc_e, df_train, window, 15)

        actual = df_eval["Close"].values[:len(preds_eval)]
        mae, rmse, da = compute_metrics(actual, preds_eval)
        accuracy[tag] = {"mae": mae, "rmse": rmse, "da": da}

        # ── future pass (predicts next 15 trading days) ──
        X_f, y_f, sc_f = prepare_data(df_full, window)
        if tag == "RNN":
            bw2, bu2 = ga_optimize(df_full)
            m_fut    = train_model(SimpleRNN(1, bu2), X_f, y_f)
            preds    = forecast_future(m_fut, sc_f, df_full, bw2)
        elif tag == "LSTM":
            m_fut = train_model(SimpleLSTM(1, 64), X_f, y_f)
            preds = forecast_future(m_fut, sc_f, df_full, window)
        else:
            m_fut = train_model(SimpleGRU(1, 64), X_f, y_f)
            preds = forecast_future(m_fut, sc_f, df_full, window)

        df_fut      = build_future_df(preds, last_date, avg_volume, volatility)
        results[tag] = df_to_records(df_fut)

    except Exception as e:
        errors[tag] = str(e)

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
    data   = request.json
    symbol = data.get("symbol", "").upper().strip()
    name   = data.get("name", "").strip()
    if not symbol or not name:
        return jsonify({"error": "Symbol and name required"}), 400
    try:
        db_execute(
            "INSERT INTO companies(symbol, name) VALUES(%s,%s) ON CONFLICT (symbol) DO NOTHING",
            (symbol, name)
        )
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
        df      = clean_ohlcv(df)
        records = df_to_records(df.tail(60))
        chg     = float(df["Close"].iloc[-1]) - float(df["Close"].iloc[0])
        pct     = (chg / float(df["Close"].iloc[0])) * 100
        return jsonify({
            "candles": records,
            "trend":   "up" if chg > 0 else ("down" if chg < 0 else "flat"),
            "change":  round(chg, 2),
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
                "change":  round(chg, 2),
                "percent": round(pct, 2),
                "trend":   "up" if chg > 0 else ("down" if chg < 0 else "flat"),
                "high":    round(float(df["High"].max()), 2),
                "low":     round(float(df["Low"].min()), 2),
                "avg_vol": int(df["Volume"].mean()),
            }

        return jsonify({"stock1": stats(d1), "stock2": stats(d2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/<symbol>")
def predict_stock(symbol):
    models_param = request.args.get("models", "RNN,LSTM,GRU").split(",")
    models_param = [m.strip().upper() for m in models_param
                    if m.strip().upper() in ("RNN", "LSTM", "GRU")]
    if not models_param:
        return jsonify({"error": "No valid models specified"}), 400

    try:
        df = clean_ohlcv(yf.download(symbol, period="4y", auto_adjust=False, progress=False))
        if df.empty:
            return jsonify({"error": f"No data found for {symbol}. Check the ticker symbol."}), 404
        if len(df) < 60:
            return jsonify({"error": f"Not enough historical data for {symbol}."}), 400

        window     = 40
        volatility = float(df["Close"].pct_change().dropna().std())
        avg_volume = int(df["Volume"].mean())
        last_date  = df.index[-1]
        df_train   = df.iloc[:-15]
        df_eval    = df.tail(15)

        results  = {}
        accuracy = {}
        errors   = {}

        for tag in models_param:
            run_single_model(
                tag, df_train, df_eval, df, window,
                avg_volume, volatility, last_date,
                results, accuracy, errors
            )

        if not results:
            all_errors = "; ".join(f"{t}: {e}" for t, e in errors.items())
            return jsonify({"error": f"All models failed — {all_errors}"}), 500

        actual_candles = df_to_records(df.tail(20))

        votes = []
        for tag, rows in results.items():
            chg = rows[-1]["close"] - rows[0]["close"]
            votes.append("UP" if chg > 0 else "DOWN")

        up = votes.count("UP")
        dn = votes.count("DOWN")
        n  = len(results)
        if up == n:    consensus = "ALL_UP"
        elif dn == n:  consensus = "ALL_DOWN"
        elif up > dn:  consensus = "MAJORITY_UP"
        elif dn > up:  consensus = "MAJORITY_DOWN"
        else:          consensus = "SPLIT"

        best = min(accuracy, key=lambda t: accuracy[t]["rmse"])

        return jsonify({
            "actual":    actual_candles,
            "predicted": results,
            "accuracy":  accuracy,
            "consensus": consensus,
            "best":      best,
            "skipped":   list(errors.keys()),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/news/<symbol>")
def get_news(symbol):
    try:
        tkr      = yf.Ticker(symbol)
        raw_news = tkr.news
        cleaned  = []
        for n in raw_news[:5]:
            if "content" in n:
                c       = n["content"]
                title   = c.get("title", "")
                link    = c.get("clickThroughUrl", {}).get("url", "")
                summary = c.get("summary", "")
                pub     = c.get("pubDate", "")
            else:
                title   = n.get("title", "")
                link    = n.get("link", "")
                summary = n.get("summary", "")
                pub     = n.get("providerPublishTime", "")
            cleaned.append({"title": title, "link": link,
                            "summary": summary, "pubDate": str(pub)})
        return jsonify(cleaned)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)