import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss
from datetime import datetime, timezone

st.set_page_config(page_title="BTC Forecast Lab", page_icon="₿", layout="wide")
st.title("₿ BTC Forecast Lab")
st.caption("Research dashboard • public market data • paper trading only • no accuracy claims before validation")

HORIZONS = {
    "15 minutes": {"timeframe": "1m", "bars": 15},
    "4 hours": {"timeframe": "5m", "bars": 48},
    "24 hours": {"timeframe": "15m", "bars": 96},
    "7 days": {"timeframe": "1h", "bars": 168},
}
TIMEFRAME_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}

with st.sidebar:
    st.header("Configuration")
    exchange_id = st.selectbox("Exchange", ["binance", "coinbase", "kraken"], index=0)
    symbol = st.text_input("Market symbol", "BTC/USDT" if exchange_id == "binance" else "BTC/USD")
    candle_limit = st.slider("Candles per horizon", 300, 1000, 1000, 100)
    fee_bps = st.number_input("Trading fee (basis points, each side)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    slippage_bps = st.number_input("Slippage estimate (basis points, each side)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
    st.caption("Costs are assumptions; replace them with your venue's actual fees and observed execution slippage.")
    refresh = st.button("Fetch market data & evaluate", type="primary")

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv(exchange_name, market_symbol, timeframe, limit):
    ex_class = getattr(ccxt, exchange_name)
    ex = ex_class({"enableRateLimit": True, "timeout": 20000})
    if not ex.has.get("fetchOHLCV"):
        raise RuntimeError(f"{exchange_name} does not support OHLCV through CCXT.")
    raw = ex.fetch_ohlcv(market_symbol, timeframe=timeframe, limit=limit)
    if not raw:
        raise RuntimeError("Exchange returned no candles.")
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    # Exclude the current candle because it may be incomplete.
    if len(df) > 1:
        df = df.iloc[:-1].copy()
    return df

def make_features(df):
    x = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    ret = close.pct_change()
    x["ret_1"] = ret
    x["ret_3"] = close.pct_change(3)
    x["ret_6"] = close.pct_change(6)
    x["ret_12"] = close.pct_change(12)
    x["volatility_12"] = ret.rolling(12).std()
    x["volatility_48"] = ret.rolling(48).std()
    x["volume_change"] = df["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    x["ema_gap"] = (ema_fast / ema_slow) - 1
    x["sma_gap"] = (close / close.rolling(20).mean()) - 1
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi_14"] = (100 - 100 / (1 + rs)) / 100
    return x.replace([np.inf, -np.inf], np.nan)

def evaluate_and_forecast(df, horizon_bars):
    features = make_features(df)
    future_return = df["close"].shift(-horizon_bars) / df["close"] - 1
    labels = (future_return > 0).astype(int)
    valid = features.notna().all(axis=1) & future_return.notna()
    X = features.loc[valid]
    y = labels.loc[valid]
    if len(X) < 180 or y.nunique() < 2:
        raise ValueError(f"Not enough usable history/class diversity ({len(X)} labelled rows). Fetch more data or use a shorter horizon.")
    split = int(len(X) * 0.8)
    if split < 100 or len(X) - split < 30:
        raise ValueError("Insufficient rows for a meaningful chronological holdout.")
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ])
    model.fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]
    pred_test = (p_test >= 0.5).astype(int)
    metrics = {
        "test_rows": len(X_test),
        "accuracy": accuracy_score(y_test, pred_test),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred_test),
        "brier": brier_score_loss(y_test, p_test),
        "base_rate": float(y_test.mean()),
        "test_start": str(df.loc[X_test.index[0], "timestamp"]),
        "test_end": str(df.loc[X_test.index[-1], "timestamp"]),
    }
    # Refit on all currently label-available rows for the latest forecast.
    model.fit(X, y)
    latest_features = features.iloc[[-1]]
    if latest_features.isna().any(axis=None):
        raise ValueError("Latest candle does not have enough indicator history.")
    probability_up = float(model.predict_proba(latest_features)[0, 1])
    last_close = float(df["close"].iloc[-1])
    return probability_up, last_close, metrics, X, y, model, features

def paper_signal(prob_up, fee_bps, slippage_bps):
    # A conservative deadband accounts for round-trip costs and forecast uncertainty.
    round_trip_cost_pct = 2 * (fee_bps + slippage_bps) / 10_000
    edge = abs(prob_up - 0.5)
    # This is a transparent signal threshold, not an expected-return model.
    threshold = max(0.08, round_trip_cost_pct * 2)
    if prob_up >= 0.5 + threshold:
        return "PAPER LONG", round_trip_cost_pct
    if prob_up <= 0.5 - threshold:
        return "PAPER SHORT / AVOID LONG", round_trip_cost_pct
    return "NO TRADE", round_trip_cost_pct

if "results" not in st.session_state:
    st.session_state.results = None

if refresh or st.session_state.results is None:
    results = {}
    errors = {}
    with st.spinner("Fetching candles and running chronological holdout evaluations…"):
        for name, cfg in HORIZONS.items():
            try:
                df = fetch_ohlcv(exchange_id, symbol, cfg["timeframe"], candle_limit)
                p_up, price, metrics, X, y, model, feats = evaluate_and_forecast(df, cfg["bars"])
                signal, cost = paper_signal(p_up, fee_bps, slippage_bps)
                results[name] = {
                    "df": df, "p_up": p_up, "price": price, "metrics": metrics,
                    "signal": signal, "cost": cost, "timeframe": cfg["timeframe"],
                    "bars": cfg["bars"]
                }
            except Exception as e:
                errors[name] = str(e)
    st.session_state.results = {"results": results, "errors": errors, "timestamp": datetime.now(timezone.utc).isoformat()}
data = st.session_state.results
results = data["results"]
errors = data["errors"]

if not results:
    st.error("No horizon produced a valid forecast. Check the exchange, symbol, network, and available history.")
else:
    latest = next(iter(results.values()))
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest reference price", f"{latest['price']:,.2f}")
    c2.metric("Exchange / market", f"{exchange_id} · {symbol}")
    c3.metric("Last refresh (UTC)", data["timestamp"][:19].replace("T", " "))
    st.info("Forecast probabilities are model outputs, not guarantees. Holdout metrics below are historical diagnostics, not evidence of future profitability.")
    cards = st.columns(4)
    for i, (name, r) in enumerate(results.items()):
        with cards[i]:
            st.subheader(name)
            st.metric("Probability of up move", f"{r['p_up']:.1%}")
            st.metric("Probability of down / flat", f"{1-r['p_up']:.1%}")
            st.write(f"**Paper signal:** {r['signal']}")
            st.caption(f"Training model: logistic regression • {r['timeframe']} candles • target {r['bars']} bars")
    st.subheader("Paper-trading ledger")
    st.caption("These are simulated positions only. No exchange orders are sent. Positions live in this browser session and reset when the session is lost.")
    if "paper_ledger" not in st.session_state:
        st.session_state.paper_ledger = []
    trade_cols = st.columns(4)
    for i, (name, r) in enumerate(results.items()):
        with trade_cols[i]:
            side = 1 if r["signal"] == "PAPER LONG" else (-1 if r["signal"] == "PAPER SHORT / AVOID LONG" else 0)
            label = "Simulate long" if side == 1 else ("Simulate short" if side == -1 else "No trade signal")
            if st.button(label, key=f"paper_{name}", disabled=(side == 0)):
                st.session_state.paper_ledger.append({
                    "opened_utc": datetime.now(timezone.utc).isoformat(),
                    "horizon": name,
                    "side": side,
                    "entry_price": float(r["price"]),
                    "fee_bps_each_side": float(fee_bps),
                    "slippage_bps_each_side": float(slippage_bps),
                })
    if st.session_state.paper_ledger:
        ledger_rows = []
        current_price = float(latest["price"])
        for pos in st.session_state.paper_ledger:
            side = int(pos["side"])
            gross = side * (current_price / pos["entry_price"] - 1)
            costs = 2 * (pos["fee_bps_each_side"] + pos["slippage_bps_each_side"]) / 10000
            net = gross - costs
            ledger_rows.append({
                "Opened (UTC)": pos["opened_utc"][:19].replace("T", " "),
                "Horizon": pos["horizon"],
                "Side": "Long" if side == 1 else "Short",
                "Entry": pos["entry_price"],
                "Latest reference": current_price,
                "Gross P&L %": f"{gross:.3%}",
                "Estimated net P&L %": f"{net:.3%}",
                "Cost assumption (round trip)": f"{costs:.3%}",
            })
        st.dataframe(pd.DataFrame(ledger_rows), use_container_width=True, hide_index=True)
        if st.button("Clear paper ledger"):
            st.session_state.paper_ledger = []
            st.rerun()
    else:
        st.write("No simulated positions yet. Choose a simulated long/short button when a signal is available.")

    st.subheader("Chronological holdout validation")
    rows = []
    for name, r in results.items():
        m = r["metrics"]
        rows.append({
            "Horizon": name,
            "Holdout rows": m["test_rows"],
            "Accuracy": f"{m['accuracy']:.1%}",
            "Balanced accuracy": f"{m['balanced_accuracy']:.1%}",
            "Brier score (lower better)": f"{m['brier']:.4f}",
            "Holdout up-rate": f"{m['base_rate']:.1%}",
            "Test start (UTC)": m["test_start"],
            "Test end (UTC)": m["test_end"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Holdout uses the latest 20% of labelled observations in a chronological split. Overlapping forward-return labels can make adjacent test rows dependent; treat these metrics as preliminary. A stronger evaluation should use purged/embargoed walk-forward folds and compare to baselines.")
    st.subheader("Price history")
    chosen = st.selectbox("Chart horizon / candle interval", list(results.keys()))
    df = results[chosen]["df"]
    fig = go.Figure(data=[go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="BTC")])
    fig.update_layout(height=420, xaxis_title="UTC time", yaxis_title="Price", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Paper-trading controls")
    a, b, c = st.columns(3)
    a.metric("Assumed round-trip cost", f"{2*(fee_bps+slippage_bps)/100:.3f}%")
    b.metric("Execution mode", "SIMULATION ONLY")
    c.metric("Live orders", "DISABLED")
    st.write("The initial build emits paper signals only; it does not submit or persist simulated fills. Do not interpret a probability threshold as a proven trading edge.")
if errors:
    with st.expander("Data / model warnings"):
        for name, err in errors.items():
            st.warning(f"{name}: {err}")

st.divider()
st.caption("Data: public exchange OHLCV via CCXT. Only closed candles are used. No exchange keys are requested. Not financial advice.")
