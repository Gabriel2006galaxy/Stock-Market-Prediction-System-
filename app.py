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
import difflib
import re


warnings.filterwarnings("ignore")


DATABASE_URL = os.environ.get("DATABASE_URL")


app = Flask(__name__)
CORS(app)


def get_db():
    if not DATABASE_URL:
        raise Exception("DATABASE_URL not set in environment.")
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


def analyze_sentiment(text):
    text = text.lower()

    pos_words = ["gain","profit","rise","growth","surge","strong","beat","record"]
    neg_words = ["loss","fall","drop","decline","weak","miss","crash","down"]

    pos = sum(word in text for word in pos_words)
    neg = sum(word in text for word in neg_words)

    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    else:
        return "neutral"


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


def find_suggestion(user_symbol, known_symbols):
    """
    Given user‑typed symbol and list of known symbols, returns (suggested_symbol, suggested_name)
    or None if no good match.
    """
    if not known_symbols:
        return None
    upper = user_symbol.upper().strip()
    if not upper:
        return None

    # 1) exact match
    for s in known_symbols:
        if upper == s["symbol"].upper():
            return s["symbol"], s.get("name", s["symbol"])
    # 2) fuzzy match on symbol
    # Build list of symbols that are NOT already in the watchlist
    watched_syms = {s["symbol"].upper() for s in known_symbols}
    candidates   = [s for s in known_symbols
            if s["symbol"].upper() not in watched_syms]

# 2) fuzzy match on symbol
    choices = [s["symbol"] for s in candidates]
    matches = difflib.get_close_matches(upper, choices, n=1, cutoff=0.3)
    if matches:

        sym = matches[0]
        s   = next((s for s in candidates if s["symbol"] == sym), None)
        if s:
            name = s.get("name", "Company")
            return sym, name
    return None

# Common fallback stocks if DB is empty
COMMON_TICKERS = [
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corp."},
    {"symbol": "GOOGL", "name": "Alphabet Inc."},
    {"symbol": "AMZN", "name": "Amazon.com Inc."},
    {"symbol": "TSLA", "name": "Tesla Inc."},
    {"symbol": "NVDA", "name": "NVIDIA Corp."},
]


# Features — all lagged so no data leakage
FEATURES = ["Lag1","Lag2","Lag3","MA5","MA10","MA20","Return1","Return5","Std5"]


def add_features(df):
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
        train_feat = add_features(df_train)
        X_train    = train_feat[FEATURES].values
        y_train    = train_feat["Target"].values

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

        full_feat  = add_features(df_full)
        last_state = full_feat[FEATURES].values[-1].copy()
        hist = list(df_full["Close"].values[-20:])

        preds = []
        for _ in range(15):
            inp  = last_state.reshape(1, -1)
            pred = float(model.predict(scaler.transform(inp) if scaler else inp)[0])
            preds.append(pred)

            hist.append(pred)
            arr = np.array(hist)

            last_state[0] = arr[-2]
            last_state[1] = arr[-3]
            last_state[2] = arr[-4]
            last_state[3] = arr[-6:-1].mean()
            last_state[4] = arr[-11:-1].mean() if len(arr) >= 11 else arr[:-1].mean()
            last_state[5] = arr[-21:-1].mean() if len(arr) >= 21 else arr[:-1].mean()
            last_state[6] = (arr[-2] - arr[-3]) / arr[-3] if arr[-3] != 0 else 0
            last_state[7] = (arr[-2] - arr[-7]) / arr[-7] if len(arr) >= 7 and arr[-7] != 0 else 0
            last_state[8] = arr[-6:-1].std() if len(arr) >= 6 else 0

        results[tag] = build_future_candles(
            preds, float(df_full["Close"].iloc[-1]),
            last_date, avg_volume, volatility
        )

    except Exception as e:
        errors[tag] = str(e)
# ┬────────────────────────────────────────────────────
# │ ROUTES
# ┴────────────────────────────────────────────────────
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
    name   = data.get("name", "").strip()

    if not symbol:
        return jsonify({"error": "Ticker symbol is required."}), 400
    if not name:
        return jsonify({"error": "Company name is required."}), 400
    if " " in symbol:
        return jsonify({"error": f"Ticker '{symbol}' contains spaces. Please remove them."}), 400

    # Optional: flag likely Indian stock without .NS / .BO
    KNOWN_US = ["AAPL","MSFT","GOOG","GOOGL","AMZN","META","TSLA","NVDA","NFLX","AMD","INTC","CRM","ORCL","IBM","UBER","LYFT","SNAP","SPOT"]
    looksIndian = bool(re.match(r"^[A-Z]{2,15}$", symbol)) and \
                  (not "." in symbol) and \
                  (not symbol in KNOWN_US)
    if looksIndian:
        return jsonify({"error": f"'{symbol}' looks like an Indian stock. Try {symbol}.NS or {symbol}.BO."}), 400

    # Check if symbol actually exists in yfinance
    try:
        df = yf.download(symbol, period="1d", progress=False)
        if df.empty:
            db_stocks = db_fetchall("SELECT symbol, name FROM companies")
            known = db_stocks if db_stocks else COMMON_TICKERS
            suggestion = find_suggestion(symbol, known)
            if suggestion:
                s_sym, s_name = suggestion
                return jsonify({
                    "error": f"Invalid stock symbol '{symbol}'",
                    "suggestion": f"{s_sym} ({s_name})"
                }), 404
            else:
                return jsonify({"error": f"No price data found for '{symbol}'."}), 404
    except Exception as e:
        return jsonify({"error": f"Could not verify ticker '{symbol}': {str(e)}"}), 500
    
    # 🔥 CHECK IF STOCK ALREADY EXISTS
    existing = db_fetchall(
    "SELECT * FROM companies WHERE symbol = %s",
    (symbol,))

    if existing:
        return jsonify({"error": "Stock already there"}), 400

    try:
        db_execute(
            "INSERT INTO companies(symbol, name) VALUES(%s,%s)",
            (symbol, name)
        )
        return jsonify({"success": True, "symbol": symbol, "name": name})
    except Exception as e:
        return jsonify({"error": f"Could not save stock: {str(e)}"}), 409


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
    symbol = symbol.upper().strip()
    if not symbol:
        return jsonify({"error": "Ticker symbol required."}), 400

    try:
        df = yf.download(symbol, period="3mo", auto_adjust=False, progress=False)
        df = clean_ohlcv(df)

        if df.empty:
            # Build from DB + fallback
            db_stocks = db_fetchall("SELECT symbol, name FROM companies")
            known = db_stocks if db_stocks else COMMON_TICKERS
            suggestion = find_suggestion(symbol, known)

            if suggestion:
                s_sym, s_name = suggestion
                return jsonify({
                    "error": f"Invalid stock symbol '{symbol}'",
                    "suggestion": f"{s_sym} ({s_name})"
                }), 404
            else:
                return jsonify({"error": f"No price data available for '{symbol}'."}), 404

        records = df_to_records(df.tail(60))
        chg = float(df["Close"].iloc[-1]) - float(df["Close"].iloc[0])
        pct = (chg / float(df["Close"].iloc[0])) * 100

        return jsonify({
            "candles": records,
            "trend":   "up" if chg > 0 else ("down" if chg < 0 else "flat"),
            "change":  round(chg, 2),
            "percent": round(pct, 2),
        })
    except Exception as e:
        return jsonify({"error": f"Failed to load data for '{symbol}': {str(e)}"}), 500


@app.route("/api/compare")
def compare_stocks():
    s1 = request.args.get("s1", "").upper()
    s2 = request.args.get("s2", "").upper()

    if not s1 or not s2:
        return jsonify({"error": "Two stock symbols are required for comparison."}), 400
    if s1 == s2:
        return jsonify({"error": "Please select two different stocks to compare."}), 400

    try:
        d1 = clean_ohlcv(yf.download(s1, period="1mo", auto_adjust=False, progress=False))
        d2 = clean_ohlcv(yf.download(s2, period="1mo", auto_adjust=False, progress=False))

        if d1.empty and d2.empty:
            return jsonify({"error": f"No data found for either '{s1}' or '{s2}'. Check both tickers."}), 404
        if d1.empty:
            return jsonify({"error": f"No data found for '{s1}'. Check the ticker symbol."}), 404
        if d2.empty:
            return jsonify({"error": f"No data found for '{s2}'. Check the ticker symbol."}), 404

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
        return jsonify({"error": f"Comparison failed: {str(e)}"}), 500


@app.route("/api/predict/<symbol>")
def predict_stock(symbol):
    symbol = symbol.upper().strip()
    if not symbol:
        return jsonify({"error": "Ticker symbol required."}), 400

    models_param = request.args.get("models", "RNN,LSTM,GRU").split(",")
    models_param = [m.strip().upper() for m in models_param
                    if m.strip().upper() in ("RNN", "LSTM", "GRU")]
    if not models_param:
        return jsonify({"error": "No valid models specified. Choose from RNN, LSTM, GRU."}), 400

    try:
        df = clean_ohlcv(yf.download(symbol, period="4y", auto_adjust=False, progress=False))

        if df.empty:
            # Build from DB + fallback
            db_stocks = db_fetchall("SELECT symbol, name FROM companies")
            known = db_stocks if db_stocks else COMMON_TICKERS
            suggestion = find_suggestion(symbol, known)

            if suggestion:
                s_sym, s_name = suggestion
                return jsonify({
                    "error": f"Invalid stock symbol '{symbol}'",
                    "suggestion": f"{s_sym} ({s_name})"
                }), 404
            else:
                return jsonify({"error": f"No price data available for '{symbol}'."}), 404

        if len(df) < 60:
            return jsonify({
                "error": (
                    f"Only {len(df)} days of data found for '{symbol}'. "
                    f"At least 60 days of history is needed to run predictions. "
                    f"This stock may be too recently listed."
                )
            }), 400

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
            return jsonify({"error": f"All models failed to train. Details: {all_errors}"}), 500

        actual_candles = df_to_records(df.tail(20))

        votes = []
        for tag, rows in results.items():
            chg = rows[-1]["close"] - rows[0]["close"]
            votes.append("UP" if chg > 0 else "DOWN")

        up = votes.count("UP")
        dn = votes.count("DOWN")
        n  = len(results)
        if up == n:
            consensus = "ALL_UP"
        elif dn == n:
            consensus = "ALL_DOWN"
        elif up > dn:
            consensus = "MAJORITY_UP"
        elif dn > up:
            consensus = "MAJORITY_DOWN"
        else:
            consensus = "SPLIT"

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
        return jsonify({"error": f"Prediction failed for '{symbol}': {str(e)}"}), 500


@app.route("/api/news/<symbol>")
def get_news(symbol):
    symbol = symbol.upper()
    try:
        tkr      = yf.Ticker(symbol)
        raw_news = tkr.news
        if not raw_news:
            return jsonify([])
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
            sentiment = analyze_sentiment(title + " " + summary)
            cleaned.append({"title": title, "link": link,
                            "summary": summary, "pubDate": str(pub),
                            "sentiment": sentiment})
        return jsonify(cleaned)
    except Exception as e:
        return jsonify({"error": f"Could not fetch news for '{symbol}': {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)