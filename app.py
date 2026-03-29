from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")
torch.set_num_threads(1)

app = Flask(__name__)
CORS(app)

DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    try:
        if DATABASE_URL:
            return psycopg2.connect(DATABASE_URL, sslmode='require')
        return psycopg2.connect(host="localhost", user="postgres", password="your_password", database="stocks")
    except Exception as e:
        print(f"DB Error: {e}")
        return None

def init_db():
    try:
        conn = get_db()
        if conn:
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS companies (symbol VARCHAR(20) PRIMARY KEY, name VARCHAR(255) NOT NULL)")
            conn.commit()
            conn.close()
    except Exception as e:
        print(f"Init DB error: {e}")

init_db()

# ── MODELS ────────────────────────────────────────────────────
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)
    def forward(self, x):
        o, _ = self.rnn(x)
        return self.fc(o[:, -1, :])

class SimpleGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)
    def forward(self, x):
        o, _ = self.gru(x)
        return self.fc(o[:, -1, :])

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1, :])

# ── HELPERS ───────────────────────────────────────────────────
def clean_ohlcv(df):
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

def prepare_data(df, win=20):
    sc = MinMaxScaler()
    sd = sc.fit_transform(df[['Close']].values)
    X, y = [], []
    for i in range(win, len(sd)):
        X.append(sd[i-win:i])
        y.append(sd[i])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y, sc

def train_model(model, X, y, epochs=50):
    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
    return model

def forecast_future(model, sc, df, win=20, days=15):
    vol = float(df['Close'].pct_change().dropna().std()) if len(df) > 2 else 0.02
    arr = sc.transform(df[['Close']].values)[-win:]
    cur = torch.tensor(arr.reshape(1, win, 1), dtype=torch.float32)
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(days):
            p   = model(cur).item()
            adj = p + np.random.normal(0, vol * max(0.1, 1 - i * 0.05) * abs(p) * 0.15)
            preds.append(adj)
            nxt = torch.tensor([[[adj]]], dtype=torch.float32)
            cur = torch.cat((cur[:, 1:, :], nxt), dim=1)
    return sc.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

def build_future_df(preds, last_date, avg_vol, vol):
    dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(preds), freq='B')
    rows  = {"Open": [], "High": [], "Low": [], "Close": [], "Volume": []}
    prev  = float(preds[0])
    for p in preds:
        p = float(p)
        rows["Open"].append(round(prev, 2))
        rows["Close"].append(round(p, 2))
        rows["High"].append(round(max(prev, p) * (1 + vol * 0.3), 2))
        rows["Low"].append(round(min(prev, p) * (1 - vol * 0.3), 2))
        rows["Volume"].append(int(avg_vol * max(0.5, np.random.normal(1.0, 0.15))))
        prev = p
    result = pd.DataFrame(rows, index=dates)
    result.index = result.index.strftime("%Y-%m-%d")
    return result

def df_to_records(df):
    out = []
    for d, r in df.iterrows():
        out.append({
            "date":   str(d),
            "open":   round(float(r["Open"]), 2),
            "high":   round(float(r["High"]), 2),
            "low":    round(float(r["Low"]), 2),
            "close":  round(float(r["Close"]), 2),
            "volume": int(r["Volume"])
        })
    return out

# ── ROUTES ────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stocks", methods=["GET"])
def get_stocks():
    conn = get_db()
    if not conn:
        return jsonify([])
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT symbol, name FROM companies ORDER BY symbol ASC")
        return jsonify(cur.fetchall())
    finally:
        conn.close()

@app.route("/api/stocks", methods=["POST"])
def add_stock():
    d = request.json
    s = d.get("symbol", "").upper().strip()
    n = d.get("name", "").strip()
    if not s or not n:
        return jsonify({"error": "Symbol and name required"}), 400
    conn = get_db()
    if not conn:
        return jsonify({"error": "DB error"}), 500
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO companies VALUES(%s,%s) ON CONFLICT (symbol) DO NOTHING", (s, n))
        conn.commit()
        return jsonify({"success": True})
    finally:
        conn.close()

@app.route("/api/stocks/<symbol>", methods=["DELETE"])
def delete_stock(symbol):
    conn = get_db()
    if not conn:
        return jsonify({"error": "DB error"}), 500
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM companies WHERE symbol=%s", (symbol.upper(),))
        conn.commit()
        return jsonify({"success": True})
    finally:
        conn.close()

@app.route("/api/stocks/all", methods=["DELETE"])
def delete_all_stocks():
    conn = get_db()
    if not conn:
        return jsonify({"error": "DB error"}), 500
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM companies")
        conn.commit()
        return jsonify({"success": True})
    finally:
        conn.close()

@app.route("/api/view/<symbol>")
def view_stock(symbol):
    try:
        df = clean_ohlcv(yf.download(symbol, period="6mo", auto_adjust=False, progress=False))
        if df.empty:
            return jsonify({"error": f"No data for {symbol}"}), 404
        chg = float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-2])
        pct = (chg / float(df["Close"].iloc[-2])) * 100
        return jsonify({
            "candles": df_to_records(df.tail(60)),
            "trend":   "up" if chg > 0 else "down",
            "change":  round(chg, 2),
            "percent": round(pct, 2)
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
            return jsonify({"error": "No data for one or both symbols"}), 404
        def st(df):
            chg = float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-2])
            return {
                "candles":  df_to_records(df.tail(20)),
                "change":   round(chg, 2),
                "percent":  round((chg / float(df["Close"].iloc[-2])) * 100, 2),
                "trend":    "up" if chg > 0 else "down",
                "high":     round(float(df["High"].max()), 2),
                "low":      round(float(df["Low"].min()), 2),
                "avg_vol":  int(df["Volume"].mean())
            }
        return jsonify({"stock1": st(d1), "stock2": st(d2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/<symbol>")
def predict_stock(symbol):
    try:
        # Download 1 year — fast and sufficient
        df = clean_ohlcv(yf.download(symbol, period="1y", auto_adjust=False, progress=False))
        if df.empty or len(df) < 60:
            return jsonify({"error": f"Not enough data for {symbol}. Try adding .NS for Indian stocks (e.g. INFY.NS)"}), 404

        mods     = request.args.get("models", "RNN,LSTM,GRU").split(",")
        WIN      = 20   # fixed window — fast
        EPOCHS   = 50   # enough for a trend, not slow
        vol      = float(df['Close'].pct_change().dropna().std())
        avg_vol  = int(df["Volume"].mean())
        last_date = df.index[-1]

        # Split: train on all-but-last-15, eval on last 15
        d_tr = df.iloc[:-15]
        d_ev = df.tail(15)

        res, acc = {}, {}

        for tag in mods:
            try:
                # Build model
                if tag == "RNN":
                    model_cls = SimpleRNN
                elif tag == "LSTM":
                    model_cls = SimpleLSTM
                else:
                    model_cls = SimpleGRU

                # --- Evaluation phase ---
                X_e, y_e, sc_e = prepare_data(d_tr, WIN)
                m_eval = train_model(model_cls(), X_e, y_e, EPOCHS)
                preds_eval = forecast_future(m_eval, sc_e, d_tr, WIN, 15)

                actual = d_ev["Close"].values
                min_len = min(len(actual), len(preds_eval))
                a, p = actual[:min_len], preds_eval[:min_len]
                mae  = round(float(np.mean(np.abs(a - p))), 2)
                rmse = round(float(np.sqrt(np.mean((a - p) ** 2))), 2)
                da   = round(float(np.mean(np.sign(np.diff(a)) == np.sign(np.diff(p))) * 100), 1) if min_len > 1 else 0.0
                acc[tag] = {"mae": mae, "rmse": rmse, "da": da}

                # --- Future prediction phase ---
                X_f, y_f, sc_f = prepare_data(df, WIN)
                m_fut = train_model(model_cls(), X_f, y_f, EPOCHS)
                future_preds = forecast_future(m_fut, sc_f, df, WIN, 15)
                res[tag] = df_to_records(build_future_df(future_preds, last_date, avg_vol, vol))

            except Exception as model_err:
                print(f"Model {tag} error: {model_err}")
                continue

        if not res:
            return jsonify({"error": "All models failed. Check server logs."}), 500

        best = min(acc, key=lambda k: acc[k]["rmse"])
        up   = sum(1 for t in res if res[t][-1]["close"] > res[t][0]["close"])
        dn   = len(res) - up

        if up == len(res):      consensus = "ALL_UP"
        elif dn == len(res):    consensus = "ALL_DOWN"
        elif up > dn:           consensus = "MAJORITY_UP"
        elif dn > up:           consensus = "MAJORITY_DOWN"
        else:                   consensus = "SPLIT"

        return jsonify({
            "actual":    df_to_records(df.tail(20)),
            "predicted": res,
            "accuracy":  acc,
            "consensus": consensus,
            "best":      best
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/news/<symbol>")
def get_news(symbol):
    try:
        raw = yf.Ticker(symbol).news
        cln = []
        for n in (raw or [])[:6]:
            t = n.get("title")
            l = n.get("link")
            s = n.get("summary", "")
            pub = n.get("providerPublishTime", "")
            if not t and "content" in n:
                c = n["content"]
                t   = c.get("title")
                l   = (c.get("clickThroughUrl") or {}).get("url") or (c.get("canonicalUrl") or {}).get("url")
                s   = c.get("summary", "")
                pub = c.get("pubDate", "")
            if t and l:
                cln.append({
                    "title":   t,
                    "link":    l,
                    "summary": s[:200] + "..." if len(s) > 200 else s,
                    "pubDate": str(pub)
                })
        return jsonify(cln)
    except Exception as e:
        print(f"News error for {symbol}: {e}")
        return jsonify([])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)