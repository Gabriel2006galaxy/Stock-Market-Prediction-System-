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
import json
import os
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ─── DB CONNECTION (POSTGRESQL) ────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    try:
        if DATABASE_URL:
            # Render PostgreSQL connection
            conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        else:
            # Local fallback (you can adjust this for local testing)
            conn = psycopg2.connect(
                host="localhost",
                user="postgres",
                password="your_password",
                database="stocks"
            )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def init_db():
    conn = get_db()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    symbol VARCHAR(20) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL
                )
            """)
            conn.commit()
            print("Database initialized successfully.")
        except Exception as e:
            print(f"Error initializing database: {e}")
        finally:
            conn.close()

init_db()

# ─── MODELS ──────────────────────────────────────────────────
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, dropout=0.1, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, dropout=0.1, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.1, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ─── HELPERS ─────────────────────────────────────────────────
def clean_ohlcv(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in ["Open","High","Low","Close","Adj Close","Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()

def prepare_data(df, window):
    if len(df) <= window:
        raise ValueError(f"Insufficient data for window size {window}")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])
    X, y   = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            scaler)

def train_model(model, X, y, epochs=100):
    opt     = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
        scheduler.step()
    return model

def ga_optimize(df, window=20):
    def eval_(w, u):
        try:
            X, y, _ = prepare_data(df, w)
            m = SimpleRNN(1, u)
            o = optim.Adam(m.parameters(), lr=0.01)
            l = nn.MSELoss()
            for _ in range(5): # Fewer epochs for GA speed
                o.zero_grad()
                loss = l(m(X), y)
                loss.backward()
                o.step()
            return loss.item()
        except:
            return float('inf')

    pop = [(random.randint(5, 20), random.randint(5, 50)) for _ in range(10)]
    for _ in range(3):
        pop = sorted(pop, key=lambda p: eval_(*p))[:5]
        while len(pop) < 10:
            if len(pop) < 2: break
            p1, p2 = random.sample(pop, 2)
            pop.append((random.choice(p1), random.choice(p2)))
    return min(pop, key=lambda p: eval_(*p))

def forecast_future(model, scaler, df, window, future_days=15):
    returns    = df['Close'].pct_change().dropna()
    volatility = returns.std() if not returns.empty else 0.02
  
    last_seq = scaler.transform(df[['Close']])[-window:]
    cur      = torch.tensor(last_seq.reshape(1, window, 1), dtype=torch.float32)
    preds    = []

    model.eval()
    with torch.no_grad():
        for i in range(future_days):
            pred = model(cur).item()
            cf   = max(0.3, 1.0 - i * 0.05)
            adj  = pred + np.random.normal(0, volatility * cf * pred * 0.3)
            preds.append(adj)
            cur  = torch.cat(
                (cur[:, 1:, :],
                 torch.tensor([[[adj]]], dtype=torch.float32)), dim=1)

    return scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

def build_future_df(preds, last_date, avg_volume, volatility):
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(preds), freq='B')
    rows = {"Open":[],"High":[],"Low":[],"Close":[],"Volume":[]}
    prev = float(preds[0])
    for i, price in enumerate(preds):
        price = float(price)
        gap   = np.random.normal(0, volatility * 0.5)
        opn   = prev * (1 + gap) if i > 0 else prev
        hi    = max(opn, price) * (1 + np.random.uniform(0.001, volatility * 1.5))
        lo    = min(opn, price) * (1 - np.random.uniform(0.001, volatility * 1.5))
        vol   = int(avg_volume * max(0.5, np.random.normal(1.0, 0.2)))
        rows["Open"].append(round(opn,2))
        rows["High"].append(round(hi,2))
        rows["Low"].append(round(lo,2))
        rows["Close"].append(round(price,2))
        rows["Volume"].append(vol)
        prev = price
    df = pd.DataFrame(rows, index=future_dates)
    df.index = df.index.strftime("%Y-%m-%d")
    return df

def compute_metrics(actual, predicted):
    actual    = np.array(actual)
    predicted = np.array(predicted)
    # Ensure same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    mae  = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted)**2)))
    
    # Directional accuracy
    if min_len > 1:
        da = float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))) * 100)
    else:
        da = 0.0
    return round(mae,2), round(rmse,2), round(da,1)

def df_to_records(df):
    records = []
    for date, row in df.iterrows():
        records.append({
            "date":   str(date) if not isinstance(date, str) else date,
            "open":   round(float(row["Open"]),2),
            "high":   round(float(row["High"]),2),
            "low":    round(float(row["Low"]),2),
            "close":  round(float(row["Close"]),2),
            "volume": int(row["Volume"])
        })
    return records

# ─── ROUTES ──────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stocks", methods=["GET"])
def get_stocks():
    conn   = get_db()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT symbol, name FROM companies ORDER BY symbol ASC")
        stocks = cursor.fetchall()
        return jsonify(stocks)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route("/api/stocks", methods=["POST"])
def add_stock():
    data   = request.json
    symbol = data.get("symbol","").upper().strip()
    name   = data.get("name","").strip()
    if not symbol or not name:
        return jsonify({"error": "Symbol and name required"}), 400
    
    conn = get_db()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO companies (symbol, name) VALUES(%s,%s) ON CONFLICT (symbol) DO UPDATE SET name = EXCLUDED.name", (symbol, name))
        conn.commit()
        return jsonify({"success": True, "symbol": symbol, "name": name})
    except Exception as e:
        return jsonify({"error": str(e)}), 409
    finally:
        conn.close()

@app.route("/api/stocks/<symbol>", methods=["DELETE"])
def delete_stock(symbol):
    conn = get_db()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM companies WHERE symbol=%s", (symbol.upper(),))
        conn.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route("/api/stocks/all", methods=["DELETE"])
def delete_all_stocks():
    conn = get_db()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM companies")
        conn.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route("/api/view/<symbol>")
def view_stock(symbol):
    try:
        df = yf.download(symbol, period="6mo", auto_adjust=False, progress=False)
        if df.empty:
            return jsonify({"error": f"No data found for {symbol}"}), 404
        df      = clean_ohlcv(df)
        records = df_to_records(df.tail(60))
        if len(df) < 2:
            return jsonify({"error": "Insufficient historical data"}), 400
            
        chg     = float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-2])
        pct     = (chg / float(df["Close"].iloc[-2])) * 100
        return jsonify({
            "candles": records,
            "trend":   "up" if chg > 0 else ("down" if chg < 0 else "flat"),
            "change":  round(chg, 2),
            "percent": round(pct, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/compare")
def compare_stocks():
    s1 = request.args.get("s1","")
    s2 = request.args.get("s2","")
    try:
        d1 = clean_ohlcv(yf.download(s1, period="1mo", auto_adjust=False, progress=False))
        d2 = clean_ohlcv(yf.download(s2, period="1mo", auto_adjust=False, progress=False))
        if d1.empty or d2.empty:
            return jsonify({"error": "No data found for one or both symbols"}), 404

        def stats(df):
            if len(df) < 2: return None
            chg = float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-2])
            pct = (chg / float(df["Close"].iloc[-2])) * 100
            return {
                "candles": df_to_records(df.tail(20)),
                "change":  round(chg,2),
                "percent": round(pct,2),
                "trend":   "up" if chg>0 else ("down" if chg<0 else "flat"),
                "high":    round(float(df["High"].max()),2),
                "low":     round(float(df["Low"].min()),2),
                "avg_vol": int(df["Volume"].mean())
            }
        
        st1, st2 = stats(d1), stats(d2)
        if not st1 or not st2:
             return jsonify({"error": "Insufficient data for comparison"}), 400
             
        return jsonify({"stock1": st1, "stock2": st2})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/<symbol>")
def predict_stock(symbol):
    models_param = request.args.get("models", "RNN,LSTM,GRU").split(",")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    try:
        df = clean_ohlcv(yf.download(symbol, period="5y", auto_adjust=False, progress=False))
        if df.empty or len(df) < 100:
            return jsonify({"error": "Insufficient historical data (min 100 days required)"}), 404

        window     = 40
        volatility = float(df['Close'].pct_change().dropna().std()) if len(df) > 1 else 0.02
        avg_volume = int(df["Volume"].mean())
        last_date  = df.index[-1]
        
        # Split for evaluation
        df_train   = df.iloc[:-15]
        df_eval    = df.tail(15)

        results    = {}
        accuracy   = {}

        for tag in models_param:
            try:
                # ── eval on last 15 days ──
                X_e, y_e, sc_e = prepare_data(df_train, window)
                if tag == "RNN":
                    bw, bu = ga_optimize(df_train)
                    m_eval = train_model(SimpleRNN(1, bu), X_e, y_e, epochs=50)
                    preds_eval = forecast_future(m_eval, sc_e, df_train, bw, 15)
                    final_window = bw
                elif tag == "LSTM":
                    m_eval = train_model(SimpleLSTM(1, 64), X_e, y_e, epochs=50)
                    preds_eval = forecast_future(m_eval, sc_e, df_train, window, 15)
                    final_window = window
                else:
                    m_eval = train_model(SimpleGRU(1, 64), X_e, y_e, epochs=50)
                    preds_eval = forecast_future(m_eval, sc_e, df_train, window, 15)
                    final_window = window

                actual = df_eval["Close"].values[:len(preds_eval)]
                mae, rmse, da = compute_metrics(actual, preds_eval)
                accuracy[tag] = {"mae": mae, "rmse": rmse, "da": da}

                # ── future prediction ──
                X_f, y_f, sc_f = prepare_data(df, final_window)
                if tag == "RNN":
                    m_fut  = train_model(SimpleRNN(1, bu), X_f, y_f, epochs=80)
                    preds  = forecast_future(m_fut, sc_f, df, final_window)
                elif tag == "LSTM":
                    m_fut = train_model(SimpleLSTM(1, 64), X_f, y_f, epochs=80)
                    preds = forecast_future(m_fut, sc_f, df, final_window)
                else:
                    m_fut = train_model(SimpleGRU(1, 64), X_f, y_f, epochs=80)
                    preds = forecast_future(m_fut, sc_f, df, final_window)

                df_fut = build_future_df(preds, last_date, avg_volume, volatility)
                results[tag] = df_to_records(df_fut)
            except Exception as model_err:
                print(f"Error training {tag}: {model_err}")
                continue

        if not results:
            return jsonify({"error": "All models failed to train"}), 500

        actual_candles = df_to_records(df.tail(20))

        # consensus
        votes = []
        for tag, rows in results.items():
            chg = rows[-1]["close"] - rows[0]["close"]
            votes.append("UP" if chg > 0 else "DOWN")

        up = votes.count("UP")
        dn = votes.count("DOWN")
        if up == len(votes):   consensus = "ALL_UP"
        elif dn == len(votes): consensus = "ALL_DOWN"
        elif up > dn:           consensus = "MAJORITY_UP"
        elif dn > up:           consensus = "MAJORITY_DOWN"
        else:                   consensus = "SPLIT"

        best = min(accuracy, key=lambda t: accuracy[t]["rmse"])

        return jsonify({
            "actual":    actual_candles,
            "predicted": results,
            "accuracy":  accuracy,
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
        tkr = yf.Ticker(symbol)
        raw_news = tkr.news
        cleaned = []
        if not raw_news:
             return jsonify([])
             
        for n in raw_news[:8]: # Show more news
            # Handle different yfinance response formats
            title = n.get("title")
            link = n.get("link")
            summary = n.get("summary", "")
            pub = n.get("providerPublishTime") or n.get("pubDate")
            
            # Nested content format
            if not title and "content" in n:
                c = n["content"]
                title = c.get("title")
                link = c.get("clickThroughUrl", {}).get("url") or c.get("canonicalUrl", {}).get("url")
                summary = c.get("summary", "")
                pub = c.get("pubDate")
            
            if title and link:
                cleaned.append({
                    "title": title, 
                    "link": link, 
                    "summary": summary[:200] + "..." if len(summary) > 200 else summary, 
                    "pubDate": str(pub) if pub else ""
                })
        return jsonify(cleaned)
    except Exception as e:
        print(f"News error for {symbol}: {e}")
        return jsonify([])

if __name__ == "__main__":
    # Use port from environment for Render
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)