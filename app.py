from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import psycopg2
import psycopg2.extras
import os
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

DATABASE_URL = os.environ.get("DATABASE_URL")

app = Flask(__name__)
CORS(app)

def get_db():
    if not DATABASE_URL:
        raise Exception("DATABASE_URL not set in Render")
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def db_fetchall(query, params=()):
    conn = get_db()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows

def db_execute(query, params=()):
    conn = get_db()
    cur  = conn.cursor()
    cur.execute(query, params)
    conn.commit()
    conn.close()

def init_db():
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            symbol TEXT PRIMARY KEY,
            name   TEXT
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
            "open":   round(float(row["Open"]),  2),
            "high":   round(float(row["High"]),  2),
            "low":    round(float(row["Low"]),   2),
            "close":  round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        })
    return records

# Features — all lagged so no data leakage
FEATURES = ["Lag1","Lag2","Lag3","MA5","MA10","MA20","Return1","Return5","Std5"]

def add_features(df):
    """
    Lagged features only — Target = NEXT day's close.
    No current Close in inputs, so no data leakage.
    """
    df = df.copy()
    df["Lag1"]    = df["Close"].shift(1)
    df["Lag2"]    = df["Close"].shift(2)
    df["Lag3"]    = df["Close"].shift(3)
    df["MA5"]     = df["Close"].rolling(5,  min_periods=1).mean().shift(1)
    df["MA10"]    = df["Close"].rolling(10, min_periods=1).mean().shift(1)
    df["MA20"]    = df["Close"].rolling(20, min_periods=1).mean().shift(1)
    df["Return1"] = df["Close"].pct_change(1).fillna(0).shift(1)
    df["Return5"] = df["Close"].pct_change(5).fillna(0).shift(1)
    df["Std5"]    = df["Close"].rolling(5,  min_periods=1).std().fillna(0).shift(1)
    df["Target"]  = df["Close"].shift(-1)
    return df.dropna()


def build_future_candles(preds, last_close, last_date, avg_volume, hist_volatility):
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=len(preds), freq="B"
    )
    swing      = max(min(hist_volatility, 0.03), 0.003)
    data       = []
    prev_close = last_close

    for d, close_price in zip(future_dates, preds):
        close_price = float(close_price)
        open_price  = float(prev_close)
        high_price  = max(open_price, close_price) * (1 + np.random.uniform(0.002, swing * 1.5))
        low_price   = min(open_price, close_price) * (1 - np.random.uniform(0.002, swing * 1.5))
        data.append({
            "date":   d.strftime("%Y-%m-%d"),
            "open":   round(open_price,  2),
            "high":   round(high_price,  2),
            "low":    round(low_price,   2),
            "close":  round(close_price, 2),
            "volume": int(avg_volume * max(0.6, np.random.normal(1.0, 0.15))),
        })
        prev_close = close_price
    return data


def run_single_model(tag, df_train, df_eval, df_full,
                     avg_volume, volatility, last_date,
                     results, accuracy, errors):
    try:
        # ── Build training features ──────────────────────────────────────
        train_feat = add_features(df_train)
        X_train    = train_feat[FEATURES].values
        y_train    = train_feat["Target"].values

        # Models:
        # RNN  → Ridge regression (regularised linear, no leakage, realistic errors)
        # LSTM → GradientBoosting (captures non-linear patterns)
        # GRU  → RandomForest    (ensemble, different predictions from above two)
        scaler = None
        if tag == "RNN":
            scaler = StandardScaler()
            model  = Ridge(alpha=1.0)
            model.fit(scaler.fit_transform(X_train), y_train)

        elif tag == "LSTM":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

        else:  # GRU
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

        # ── Evaluate on held-out last 15 days ───────────────────────────
        # Prepend 20 rows of training context so all rolling windows are warm
        context   = pd.concat([df_train.tail(20), df_eval])
        eval_feat = add_features(context).iloc[-len(df_eval):]

        X_eval = eval_feat[FEATURES].values
        y_true = eval_feat["Target"].values

        y_pred = model.predict(scaler.transform(X_eval) if scaler else X_eval)

        n = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:n], y_pred[:n]

        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
        da   = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100) if n >= 2 else 0.0

        accuracy[tag] = {
            "mae":  round(mae,  2),
            "rmse": round(rmse, 2),
            "da":   round(da,   1),
        }

        # ── Forecast next 15 trading days (autoregressive) ──────────────
        full_feat  = add_features(df_full)
        last_state = full_feat[FEATURES].values[-1].copy()

        # Track last 20 closes for accurate rolling feature updates
        hist = list(df_full["Close"].values[-20:])

        preds = []
        for _ in range(15):
            inp  = last_state.reshape(1, -1)
            pred = float(model.predict(scaler.transform(inp) if scaler else inp)[0])
            preds.append(pred)

            # Append prediction and recompute all lagged features
            hist.append(pred)
            arr = np.array(hist)

            last_state[0] = arr[-2]                           # Lag1
            last_state[1] = arr[-3]                           # Lag2
            last_state[2] = arr[-4]                           # Lag3
            last_state[3] = arr[-6:-1].mean()                 # MA5
            last_state[4] = arr[-11:-1].mean() if len(arr) >= 11 else arr[:-1].mean()  # MA10
            last_state[5] = arr[-21:-1].mean() if len(arr) >= 21 else arr[:-1].mean()  # MA20
            last_state[6] = (arr[-2] - arr[-3]) / arr[-3] if arr[-3] != 0 else 0       # Return1
            last_state[7] = (arr[-2] - arr[-7]) / arr[-7] if len(arr) >= 7 and arr[-7] != 0 else 0  # Return5
            last_state[8] = arr[-6:-1].std() if len(arr) >= 6 else 0                   # Std5

        results[tag] = build_future_candles(
            preds, float(df_full["Close"].iloc[-1]),
            last_date, avg_volume, volatility
        )

    except Exception as e:
        errors[tag] = str(e)


# ─── ROUTES ────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stocks", methods=["GET"])
def get_stocks():
    rows = db_fetchall("SELECT symbol, name FROM companies")
    return jsonify([{"symbol": r["symbol"], "name": r["name"]} for r in rows])

@app.route("/api/stocks", methods=["POST"])
def add_stock():
    data   = request.json
    symbol = data.get("symbol", "").upper().strip()
    name   = data.get("name",   "").strip()
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
                "low":     round(float(df["Low"].min()),  2),
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
            return jsonify({"error": f"No data found for {symbol}."}), 404
        if len(df) < 60:
            return jsonify({"error": f"Not enough historical data for {symbol}."}), 400

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
                tag, df_train, df_eval, df,
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

        best = max(accuracy, key=lambda t: accuracy[t]["da"])

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)