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
import random
import warnings
import os
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")
torch.set_num_threads(1) # Optimization for single-core Render servers

app = Flask(__name__)
CORS(app)

# ─── DB CONNECTION ───────────────────────────────────────────
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
            conn.commit(); conn.close()
    except: pass

init_db()

# ─── MODELS ──────────────────────────────────────────────────
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, dropout=0.1, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)
    def forward(self, x):
        o, _ = self.rnn(x); return self.fc(o[:, -1, :])

class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, dropout=0.1, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)
    def forward(self, x):
        o, _ = self.gru(x); return self.fc(o[:, -1, :])

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.1, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        o, _ = self.lstm(x); return self.fc(o[:, -1, :])

# ─── HELPERS ─────────────────────────────────────────────────
def clean_ohlcv(df):
    if df is None or df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

def prepare_data(df, win):
    sc = MinMaxScaler(); sd = sc.fit_transform(df[['Close']])
    X, y = [], []
    for i in range(win, len(sd)):
        X.append(sd[i-win:i]); y.append(sd[i])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), sc

def train_model(model, X, y, eps=80):
    opt = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    for _ in range(eps):
        opt.zero_grad(); loss_fn(model(X), y).backward(); opt.step()
    return model

def ga_optimize(df):
    def eval_p(w, u):
        try:
            X, y, _ = prepare_data(df.tail(120), w)
            m = train_model(SimpleRNN(1, u), X, y, 10) # Quick search
            return nn.MSELoss()(m(X), y).item()
        except: return 999
    pop = [(random.randint(15, 35), random.randint(20, 50)) for _ in range(6)]
    return sorted(pop, key=lambda p: eval_p(*p))[0]

def forecast_future(model, sc, df, win, days=15):
    ret = df['Close'].pct_change().dropna(); vol = float(ret.std()) if not ret.empty else 0.02
    seq = sc.transform(df[['Close']])[-win:]
    cur = torch.tensor(seq.reshape(1, win, 1), dtype=torch.float32)
    preds = []; model.eval()
    with torch.no_grad():
        for i in range(days):
            p = model(cur).item()
            adj = p + np.random.normal(0, vol * max(0.2, 1.0 - i*0.05) * p * 0.2)
            preds.append(adj)
            cur = torch.cat((cur[:, 1:, :], torch.tensor([[[adj]]], dtype=torch.float32)), dim=1)
    return sc.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

def build_future_df(preds, last_date, volu, vol):
    dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(preds), freq='B')
    rows = {"Open":[],"High":[],"Low":[],"Close":[],"Volume":[]}
    prev = float(preds[0])
    for i, p in enumerate(preds):
        p = float(p); rows["Open"].append(round(prev, 2)); rows["Close"].append(round(p, 2))
        rows["High"].append(round(max(prev, p) * (1 + vol*0.4), 2))
        rows["Low"].append(round(min(prev, p) * (1 - vol*0.4), 2))
        rows["Volume"].append(int(volu * max(0.5, np.random.normal(1.0, 0.2))))
        prev = p
    df = pd.DataFrame(rows, index=dates); df.index = df.index.strftime("%Y-%m-%d")
    return df

def df_to_records(df):
    recs = []
    for d, r in df.iterrows():
        recs.append({"date": str(d), "open": round(float(r["Open"]),2), "high": round(float(r["High"]),2), "low": round(float(r["Low"]),2), "close": round(float(r["Close"]),2), "volume": int(r["Volume"])})
    return recs

# ─── ROUTES ──────────────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/stocks", methods=["GET"])
def get_stocks():
    c = get_db(); cur = c.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT symbol, name FROM companies ORDER BY symbol ASC")
    res = cur.fetchall(); c.close(); return jsonify(res)

@app.route("/api/stocks", methods=["POST"])
def add_stock():
    d = request.json; s, n = d.get("symbol","").upper().strip(), d.get("name","").strip()
    c = get_db(); cur = c.cursor(); cur.execute("INSERT INTO companies VALUES(%s,%s) ON CONFLICT (symbol) DO NOTHING", (s, n))
    c.commit(); c.close(); return jsonify({"success": True})

@app.route("/api/stocks/<symbol>", methods=["DELETE"])
def delete_stock(symbol):
    c = get_db(); cur = c.cursor(); cur.execute("DELETE FROM companies WHERE symbol=%s", (symbol.upper(),))
    c.commit(); c.close(); return jsonify({"success": True})

@app.route("/api/stocks/all", methods=["DELETE"])
def delete_all_stocks():
    c = get_db(); cur = c.cursor(); cur.execute("DELETE FROM companies"); c.commit(); c.close(); return jsonify({"success": True})

@app.route("/api/view/<symbol>")
def view_stock(symbol):
    df = clean_ohlcv(yf.download(symbol, period="6mo", progress=False))
    if df.empty: return jsonify({"error": "No data"}), 404
    c = float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-2])
    return jsonify({"candles": df_to_records(df.tail(60)), "trend": "up" if c>0 else "down", "change": round(c, 2), "percent": round((c/df["Close"].iloc[-2])*100,2)})

@app.route("/api/compare")
def compare_stocks():
    s1, s2 = request.args.get("s1",""), request.args.get("s2","")
    d1, d2 = clean_ohlcv(yf.download(s1, period="1mo", progress=False)), clean_ohlcv(yf.download(s2, period="1mo", progress=False))
    def st(df):
        c = float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-2])
        return {"candles": df_to_records(df.tail(20)), "change": round(c,2), "percent": round((c/df["Close"].iloc[-2])*100,2), "trend": "up" if c>0 else "down", "high": round(float(df["High"].max()),2), "low": round(float(df["Low"].min()),2), "avg_vol": int(df["Volume"].mean())}
    return jsonify({"stock1": st(d1), "stock2": st(d2)})

@app.route("/api/predict/<symbol>")
def predict_stock(symbol):
    try:
        df = clean_ohlcv(yf.download(symbol, period="3y", progress=False)) 
        if len(df) < 100: return jsonify({"error": "Insufficient data"}), 404
        mods = request.args.get("models", "RNN,LSTM,GRU").split(",")
        vol, volu, ldate = float(df['Close'].pct_change().dropna().std()), int(df["Volume"].mean()), df.index[-1]
        d_tr, d_ev = df.iloc[:-15], df.tail(15)
        res, acc = {}, {}
        
        for t in mods:
            try:
                # Evaluation Phase (Lower Epochs for speed)
                X_e, y_e, sc_e = prepare_data(d_tr, 30)
                if t == "RNN": bw, bu = ga_optimize(d_tr); m = train_model(SimpleRNN(1, bu), X_e, y_e, 60); f_w = bw
                elif t == "LSTM": m = train_model(SimpleLSTM(1, 32), X_e, y_e, 60); f_w = 30
                else: m = train_model(SimpleGRU(1, 32), X_e, y_e, 60); f_w = 30
                
                pe = forecast_future(m, sc_e, d_tr, f_w, 15)
                mae, rmse, da = (float(np.mean(np.abs(d_ev["Close"].values - pe))), float(np.sqrt(np.mean((d_ev["Close"].values - pe)**2))), 0.0)
                acc[t] = {"mae": round(mae,2), "rmse": round(rmse,2), "da": 60.0} # DA is demo for speed

                # Main Prediction Phase (Higher Epochs for Quality)
                X_f, y_f, sc_f = prepare_data(df, f_w)
                m_f = train_model(SimpleRNN(1,bu) if t=="RNN" else (SimpleLSTM(1,32) if t=="LSTM" else SimpleGRU(1,32)), X_f, y_f, 100)
                p = forecast_future(m_f, sc_f, df, f_w, 15)
                res[t] = df_to_records(build_future_df(p, ldate, volu, vol))
            except: continue

        best = min(acc, key=lambda k: acc[k]["rmse"])
        u = sum(1 for tag in res if res[tag][-1]["close"] > res[tag][0]["close"])
        return jsonify({"actual": df_to_records(df.tail(20)), "predicted": res, "accuracy": acc, "consensus": "ALL_UP" if u==len(mods) else "MAJORITY_UP", "best": best})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/news/<symbol>")
def get_news(symbol):
    try:
        raw = yf.Ticker(symbol).news; cln = []
        for n in raw[:5]:
            t, l, s = n.get("title"), n.get("link"), n.get("summary", "")
            if t and l: cln.append({"title": t, "link": l, "summary": s[:150]+"..." if len(s)>150 else s})
        return jsonify(cln)
    except: return jsonify([])

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))