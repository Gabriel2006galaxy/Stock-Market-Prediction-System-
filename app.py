from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import psycopg2
import psycopg2.extras
import os
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import json

warnings.filterwarnings("ignore")

# ─── DB CONNECTION ─────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL")

app = Flask(__name__)
CORS(app)

# init_db() will be called after definition in the __main__ block

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

init_db()

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
    try:
        # Prepare data
        df_train = df_train.copy()
        df_train["MA5"] = df_train["Close"].rolling(5).mean()
        df_train["MA10"] = df_train["Close"].rolling(10).mean()
        df_train["Return"] = df_train["Close"].pct_change()

        df_train = df_train.dropna()

        X = df_train[["Close", "MA5", "MA10", "Return"]].values
        y = df_train["Close"].values

        # Choose model
        if tag == "RNN":
            model = LinearRegression()
        elif tag == "LSTM":
            model = RandomForestRegressor(n_estimators=50)
        else:
            model = SVR()

        # Train
        model.fit(X, y)

        # Predict next 15 days (iterative rolling forecast)
        preds = []
        current_input = X[-1].copy()

        for _ in range(15):
            pred = model.predict([current_input])[0]
            preds.append(pred)

            # shift features forward
            prev_close_val = current_input[0]

            current_input[0] = pred
            current_input[1] = (current_input[1]*4 + pred)/5
            current_input[2] = (current_input[2]*9 + pred)/10

            if prev_close_val != 0:
                current_input[3] = (pred - prev_close_val) / prev_close_val
            else:
                current_input[3] = 0

        # Format output
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=15, freq="B")

        data = []
        prev_close = df_train["Close"].iloc[-1]

        for d, p in zip(future_dates, preds):
            open_price = prev_close
            close_price = p

            high_price = max(open_price, close_price) * (1 + np.random.uniform(0.001, 0.01))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0.001, 0.01))

            data.append({
                "date": d.strftime("%Y-%m-%d"),
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close_price),
                "volume": int(avg_volume)
            })

            prev_close = close_price

        results[tag] = data

        # Compute accuracy on eval set
        df_eval_copy = df_eval.copy()
        df_eval_copy["MA5"] = df_eval_copy["Close"].rolling(5).mean()
        df_eval_copy["MA10"] = df_eval_copy["Close"].rolling(10).mean()
        df_eval_copy["Return"] = df_eval_copy["Close"].pct_change()
        df_eval_copy = df_eval_copy.dropna()

        if len(df_eval_copy) > 0:
            X_eval = df_eval_copy[["Close", "MA5", "MA10", "Return"]].values
            y_true = df_eval_copy["Close"].values

            y_pred = model.predict(X_eval[:len(y_true)])

            mae = mean_absolute_error(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred) ** 0.5

            direction = np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))
            da = np.mean(direction) * 100 if len(direction) > 0 else 0

            accuracy[tag] = {
                "mae": float(round(mae, 2)),
                "rmse": float(round(rmse, 2)),
                "da": float(round(da, 2))
            }
        else:
            accuracy[tag] = {"mae": 0, "rmse": 0, "da": 0}

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
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)