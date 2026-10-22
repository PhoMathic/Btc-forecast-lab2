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
st.caption("Walk-forward research • public market data • paper trading only • no live orders")

HORIZONS = {
    "15 minutes": {"timeframe": "1m", "bars": 15},
    "4 hours": {"timeframe": "5m", "bars": 48},
    "24 hours": {"timeframe": "15m", "bars": 96},
    "7 days": {"timeframe": "1h", "bars": 168},
}

with st.sidebar:
    st.header("Configuration")
    exchange_id = st.selectbox("Exchange", ["kraken", "coinbase", "binance"], index=0)
    default_symbol = "BTC/USD" if exchange_id in ("kraken", "coinbase") else "BTC/USDT"
    symbol = st.text_input("Market symbol", default_symbol)
    candle_limit = st.slider("Candles per horizon", 300, 1000, 1000, 100)
    fee_bps = st.number_input("Trading fee (basis points, each side)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    slippage_bps = st.number_input("Slippage estimate (basis points, each side)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
    min_train = st.slider("Minimum training rows", min_value=200, max_value=600, value=300, step=50)
    folds = st.slider("Walk-forward test folds", min_value=2, max_value=5, value=3)
    st.caption("Costs are assumptions. Set them to realistic venue fees and slippage.")
    refresh = st.button("Fetch market data & evaluate", type="primary")

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv(exchange_name, market_symbol, timeframe, limit):
    ex_class = getattr(ccxt, exchange_name)
    ex = ex_class({"enableRateLimit": True, "timeout": 20000})
    if not ex.has.get("fetchOHLCV"):
        raise RuntimeError(f"{exchange_name} does not support OHLCV through CCXT.")
    markets = ex.load_markets()
    if market_symbol not in markets:
        suggestions = [s for s in markets if s.startswith("BTC/")][:8]
        raise ValueError(f"Market '{market_symbol}' is not listed on {exchange_name}. Example BTC markets: {suggestions}")
    raw = ex.fetch_ohlcv(market_symbol, timeframe=timeframe, limit=limit)
    if not raw:
        raise RuntimeError("Exchange returned no candles.")
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    # Exclude the newest potentially incomplete candle.
    if len(df) > 1:
        df = df.iloc[:-1].copy()
    return df

def make_features(df):
    close = df["close"].astype(float)
    ret = close.pct_change()
    x = pd.DataFrame(index=df.index)
    x["ret_1"] = ret
    x["ret_3"] = close.pct_change(3)
    x["ret_6"] = close.pct_change(6)
    x["ret_12"] = close.pct_change(12)
    x["volatility_12"] = ret.rolling(12).std()
    x["volatility_48"] = ret.rolling(48).std()
    x["volume_change"] = df["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    x["ema_gap"] = ema_fast / ema_slow - 1
    x["sma_gap"] = close / close.rolling(20).mean() - 1
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi_14"] = (100 - 100 / (1 + rs)) / 100
    return x.replace([np.inf, -np.inf], np.nan)

def new_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ])

def max_drawdown(returns):
    if len(returns) == 0:
        return np.nan
    equity = np.cumprod(1 + np.asarray(returns, dtype=float))
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    dd = equity / peaks - 1
    return float(np.min(dd)) if len(dd) else 0.0

def walk_forward_validate(df, horizon, fee_bps, slippage_bps, min_train=300, n_folds=3):
    """Expanding-window folds. Purge horizon rows before each test fold."""
    feats = make_features(df)
    future_ret = df["close"].shift(-horizon) / df["close"] - 1
    labels = (future_ret > 0).astype(int)
    valid = feats.notna().all(axis=1) & future_ret.notna() & np.isfinite(future_ret)
    # Preserve original row numbers so purge is measured in candle rows.
    rows = np.flatnonzero(valid.to_numpy())
    X = feats.loc[valid].reset_index(drop=True)
    y = labels.loc[valid].reset_index(drop=True)
    fr = future_ret.loc[valid].reset_index(drop=True)
    if len(X) < min_train + horizon + 40:
        raise ValueError(f"Not enough usable rows for walk-forward testing: {len(X)} usable; need about {min_train + horizon + 40}. Increase candles or lower minimum training rows.")
    # Split the post-minimum sample into chronological test blocks.
    remaining = len(X) - min_train
    block = max(20, remaining // n_folds)
    preds = []
    fold_id = 0
    test_start = min_train
    cost = 2 * (fee_bps + slippage_bps) / 10000.0
    while test_start < len(X) and fold_id < n_folds:
        test_end = len(X) if fold_id == n_folds - 1 else min(len(X), test_start + block)
        # The test boundary is in valid-row coordinates; use original candle row indices for purge.
        first_test_original_row = rows[test_start]
        train_mask = rows < (first_test_original_row - horizon)
        train_positions = np.flatnonzero(train_mask)
        if len(train_positions) < 100 or y.iloc[train_positions].nunique() < 2:
            test_start = test_end
            fold_id += 1
            continue
        model = new_model()
        model.fit(X.iloc[train_positions], y.iloc[train_positions])
        p = model.predict_proba(X.iloc[test_start:test_end])[:, 1]
        for j, prob in enumerate(p, start=test_start):
            preds.append({
                "position": int(j),
                "row": int(rows[j]),
                "timestamp": df.loc[rows[j], "timestamp"],
                "prob_up": float(prob),
                "actual_up": int(y.iloc[j]),
                "future_return": float(fr.iloc[j]),
                "fold": fold_id + 1,
                "train_up_rate": float(y.iloc[train_positions].mean())
            })
        test_start = test_end
        fold_id += 1
    out = pd.DataFrame(preds)
    if out.empty or out["actual_up"].nunique() < 2:
        raise ValueError("Walk-forward folds produced too few predictions or only one class. Increase candle history.")
    out["model_pred"] = (out["prob_up"] >= 0.5).astype(int)
    out["majority_baseline_pred"] = (out["train_up_rate"] >= 0.5).astype(int)
    out["baseline_prob"] = out["train_up_rate"].clip(0, 1)
    out["model_brier_component"] = (out["prob_up"] - out["actual_up"]) ** 2
    out["baseline_brier_component"] = (out["baseline_prob"] - out["actual_up"]) ** 2
    # Signals use fixed thresholds, and every trade pays estimated round-trip costs.
    out["side"] = np.where(out["prob_up"] >= 0.58, 1, np.where(out["prob_up"] <= 0.42, -1, 0))
    out["net_return"] = np.where(out["side"] != 0, out["side"] * out["future_return"] - cost, 0.0)
    # Avoid overlapping trades: take a signal only after the prior horizon has elapsed.
    selected = []
    next_allowed_row = -1
    for ix, row in out.sort_values("row").iterrows():
        if int(row["side"]) != 0 and int(row["row"]) >= next_allowed_row:
            selected.append(ix)
            next_allowed_row = int(row["row"]) + horizon
    trades = out.loc[selected].copy()
    model_acc = accuracy_score(out["actual_up"], out["model_pred"])
    baseline_acc = accuracy_score(out["actual_up"], out["majority_baseline_pred"])
    model_bal = balanced_accuracy_score(out["actual_up"], out["model_pred"])
    baseline_bal = balanced_accuracy_score(out["actual_up"], out["majority_baseline_pred"])
    summary = {
        "predictions": int(len(out)),
        "folds_used": int(out["fold"].nunique()),
        "model_accuracy": float(model_acc),
        "majority_accuracy": float(baseline_acc),
        "model_balanced_accuracy": float(model_bal),
        "majority_balanced_accuracy": float(baseline_bal),
        "model_brier": float(brier_score_loss(out["actual_up"], out["prob_up"])),
        "historical_rate_brier": float(out["baseline_brier_component"].mean()),
        "test_up_rate": float(out["actual_up"].mean()),
        "trades": int(len(trades)),
        "win_rate": float((trades["net_return"] > 0).mean()) if len(trades) else np.nan,
        "net_compounded_return": float((1 + trades["net_return"]).prod() - 1) if len(trades) else 0.0,
        "net_mean_trade_return": float(trades["net_return"].mean()) if len(trades) else np.nan,
        "max_drawdown": max_drawdown(trades["net_return"]) if len(trades) else np.nan,
        "fee_bps_each_side": float(fee_bps),
        "slippage_bps_each_side": float(slippage_bps),
        "cost_round_trip_pct": cost * 100,
        "test_start": str(out["timestamp"].min()),
        "test_end": str(out["timestamp"].max()),
    }
    return summary, out, trades

def evaluate_and_forecast(df, horizon_bars, fee_bps, slippage_bps, min_train, folds):
    features = make_features(df)
    future_return = df["close"].shift(-horizon_bars) / df["close"] - 1
    labels = (future_return > 0).astype(int)
    valid = features.notna().all(axis=1) & future_return.notna() & np.isfinite(future_return)
    X = features.loc[valid]
    y = labels.loc[valid]
    if len(X) < max(180, min_train + horizon_bars + 40) or y.nunique() < 2:
        raise ValueError(f"Not enough usable labelled rows ({len(X)}). Increase candle history or lower minimum training rows.")
    latest_features = features.iloc[[-1]]
    if latest_features.isna().any(axis=None):
        raise ValueError("Latest candle lacks enough indicator history.")
    model = new_model()
    model.fit(X, y)
    p_up = float(model.predict_proba(latest_features)[0, 1])
    metrics, predictions, trades = walk_forward_validate(df, horizon_bars, fee_bps, slippage_bps, min_train, folds)
    return p_up, float(df["close"].iloc[-1]), metrics, predictions, trades

if "results" not in st.session_state:
    st.session_state.results = None

if refresh or st.session_state.results is None:
    results, errors = {}, {}
    with st.spinner("Fetching candles and running purged walk-forward tests…"):
        for name, cfg in HORIZONS.items():
            try:
                df = fetch_ohlcv(exchange_id, symbol, cfg["timeframe"], candle_limit)
                p_up, price, metrics, predictions, trades = evaluate_and_forecast(
                    df, cfg["bars"], fee_bps, slippage_bps, min_train, folds
                )
                cost = 2 * (fee_bps + slippage_bps) / 10000
                threshold = max(0.08, cost * 2)
                signal = "PAPER LONG" if p_up >= 0.5 + threshold else ("PAPER SHORT / AVOID LONG" if p_up <= 0.5 - threshold else "NO TRADE")
                results[name] = {
                    "df": df, "p_up": p_up, "price": price, "metrics": metrics,
                    "predictions": predictions, "trades": trades, "signal": signal,
                    "timeframe": cfg["timeframe"], "bars": cfg["bars"]
                }
            except Exception as e:
                errors[name] = f"{type(e).__name__}: {e}"
    st.session_state.results = {"results": results, "errors": errors, "timestamp": datetime.now(timezone.utc).isoformat()}
data = st.session_state.results
results, errors = data["results"], data["errors"]

if not results:
    st.error("No horizon produced a valid forecast. Check exchange, symbol, network, and history.")
else:
    first = next(iter(results.values()))
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest reference price", f"{first['price']:,.2f}")
    c2.metric("Exchange / market", f"{exchange_id} · {symbol}")
    c3.metric("Last refresh (UTC)", data["timestamp"][:19].replace("T", " "))
    st.info("Research only. Forecast probabilities are not guarantees. Walk-forward results are historical diagnostics, not proof of future profitability.")
    cards = st.columns(4)
    for i, (name, r) in enumerate(results.items()):
        with cards[i]:
            st.subheader(name)
            st.metric("Probability of up move", f"{r['p_up']:.1%}")
            st.metric("Probability of down / flat", f"{1-r['p_up']:.1%}")
            st.write(f"**Paper signal:** {r['signal']}")
            st.caption(f"Logistic regression • {r['timeframe']} candles • target {r['bars']} bars")

    st.subheader("Purged walk-forward validation vs baselines")
    rows = []
    for name, r in results.items():
        m = r["metrics"]
        rows.append({
            "Horizon": name, "Predictions": m["predictions"], "Folds": m["folds_used"],
            "Model accuracy": f"{m['model_accuracy']:.1%}",
            "Majority baseline accuracy": f"{m['majority_accuracy']:.1%}",
            "Model balanced accuracy": f"{m['model_balanced_accuracy']:.1%}",
            "Majority balanced accuracy": f"{m['majority_balanced_accuracy']:.1%}",
            "Model Brier ↓": f"{m['model_brier']:.4f}",
            "Historical-rate Brier ↓": f"{m['historical_rate_brier']:.4f}",
            "Test up-rate": f"{m['test_up_rate']:.1%}"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Each fold trains only on earlier rows. Training labels whose forward horizon overlaps the test start are purged. Baselines are calculated using information available from the training fold.")
    st.subheader("Trading-cost-aware paper strategy")
    cost_rows = []
    for name, r in results.items():
        m = r["metrics"]
        cost_rows.append({
            "Horizon": name, "Non-overlapping trades": m["trades"],
            "Win rate after costs": "—" if np.isnan(m["win_rate"]) else f"{m['win_rate']:.1%}",
            "Compounded net return": f"{m['net_compounded_return']:.2%}",
            "Mean net return / trade": "—" if np.isnan(m["net_mean_trade_return"]) else f"{m['net_mean_trade_return']:.3%}",
            "Max drawdown": "—" if np.isnan(m["max_drawdown"]) else f"{m['max_drawdown']:.2%}",
            "Round-trip cost": f"{m['cost_round_trip_pct']:.3f}%"
        })
    st.dataframe(pd.DataFrame(cost_rows), use_container_width=True, hide_index=True)
    st.warning("Backtest is a simplified research simulation using close-to-close forward returns, fixed thresholds and assumed costs. It excludes spread dynamics, funding, market impact, latency, partial fills and taxes. Do not use it as a profitability claim.")
    st.subheader("Fold predictions")
    chosen = st.selectbox("Show prediction rows for", list(results.keys()))
    st.dataframe(results[chosen]["predictions"].tail(100), use_container_width=True, hide_index=True)
    st.subheader("Price history")
    chosen_chart = st.selectbox("Chart horizon / candle interval", list(results.keys()))
    df = results[chosen_chart]["df"]
    fig = go.Figure(data=[go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="BTC")])
    fig.update_layout(height=420, xaxis_title="UTC time", yaxis_title="Price", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Paper-trading controls")
    a, b, c = st.columns(3)
    a.metric("Assumed round-trip cost", f"{2*(fee_bps+slippage_bps)/100:.3f}%")
    b.metric("Execution mode", "SIMULATION ONLY")
    c.metric("Live orders", "DISABLED")

if errors:
    with st.expander("Data / model warnings", expanded=True):
        for name, err in errors.items():
            st.warning(f"{name}: {err}")

st.divider()
st.caption("Data: public exchange OHLCV via CCXT. Closed candles only. No exchange keys or live orders. Research, not financial advice.")
