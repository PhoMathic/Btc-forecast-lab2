"""BTC Forecast Lab — walk-forward research dashboard.

Revision changes vs the original app.py:
- All modeling logic lives in btc_core.py (shared with validation.py).
- Two-model walk-forward comparison (logistic vs gradient boosting) with a
  transparent champion-selection rule per horizon.
- Cost-aware thresholds optimized inside each training window (never on test
  data); the displayed paper signal uses exactly the backtested rule.
- Per-fold diagnostics (incl. purge proof), ROC AUC, equity curve, trade log
  with entry/exit times, and long-only vs long/short comparison.
- Stale-config warning when sidebar settings change after a run.
- No automatic fetch on page load (press the button); up to 2,000 candles.
"""

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from datetime import datetime, timezone

from btc_core import MODEL_NAMES, evaluate_and_forecast

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
    candle_limit = st.slider("Candles per horizon", 300, 2000, 1000, 100)
    fee_bps = st.number_input("Trading fee (basis points, each side)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    slippage_bps = st.number_input("Slippage estimate (basis points, each side)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
    min_train = st.slider("Minimum training rows", min_value=200, max_value=600, value=300, step=50)
    folds = st.slider("Walk-forward test folds", min_value=2, max_value=5, value=3)
    model_options = st.multiselect("Models to compare", list(MODEL_NAMES), default=list(MODEL_NAMES))
    st.caption("Costs are assumptions. Set them to realistic venue fees and slippage.")
    refresh = st.button("Fetch market data & evaluate", type="primary")

config = {
    "exchange": exchange_id, "symbol": symbol, "candles": candle_limit,
    "fee_bps": fee_bps, "slippage_bps": slippage_bps,
    "min_train": min_train, "folds": folds, "models": sorted(model_options),
}


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


def champion_key(metrics):
    """Rank models: prefer enough trades (>=8) ranked by mean net return per
    trade; otherwise fall back to Brier score (lower is better)."""
    if metrics["trades"] >= 8 and not np.isnan(metrics.get("net_mean_trade_return", np.nan)):
        return (2, metrics["net_mean_trade_return"])
    return (1, -metrics["model_brier"])


if "run" not in st.session_state:
    st.session_state.run = None

if refresh:
    if not model_options:
        st.error("Select at least one model to compare.")
    else:
        results, errors = {}, {}
        with st.spinner("Fetching candles and running purged walk-forward tests…"):
            for name, cfg in HORIZONS.items():
                for model_name in model_options:
                    key = f"{name} · {model_name}"
                    try:
                        df = fetch_ohlcv(exchange_id, symbol, cfg["timeframe"], candle_limit)
                        p_up, price, metrics, predictions, trades, per_fold, signal = evaluate_and_forecast(
                            df, cfg["bars"], fee_bps, slippage_bps, min_train, folds, model_name
                        )
                        results[key] = {
                            "df": df, "p_up": p_up, "price": price, "metrics": metrics,
                            "predictions": predictions, "trades": trades, "per_fold": per_fold,
                            "signal": signal, "timeframe": cfg["timeframe"], "bars": cfg["bars"],
                            "horizon": name, "model": model_name,
                        }
                    except Exception as e:
                        errors[key] = f"{type(e).__name__}: {e}"
        st.session_state.run = {
            "results": results, "errors": errors,
            "timestamp": datetime.now(timezone.utc).isoformat(), "config": config,
        }

data = st.session_state.run
if data is None:
    st.info("Set the configuration in the sidebar, then press **Fetch market data & evaluate**.")
else:
    results, errors = data["results"], data["errors"]
    if data.get("config") != config:
        st.warning("Sidebar settings changed since the last run — the results below reflect the "
                   "previous configuration. Press **Fetch market data & evaluate** to rerun with current settings.")
    if not results:
        st.error("No horizon produced a valid forecast. Check exchange, symbol, network, and history.")
    else:
        champions = {}
        for k, r in results.items():
            h = r["horizon"]
            if h not in champions or champion_key(r["metrics"]) > champion_key(champions[h]["metrics"]):
                champions[h] = r

        first = next(iter(champions.values()))
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest reference price", f"{first['price']:,.2f}")
        c2.metric("Exchange / market", f"{exchange_id} · {symbol}")
        c3.metric("Last refresh (UTC)", data["timestamp"][:19].replace("T", " "))
        st.info("Research only. Forecast probabilities are not guarantees. Walk-forward results are historical diagnostics, not proof of future profitability.")

        cards = st.columns(4)
        for i, (name, r) in enumerate(champions.items()):
            with cards[i]:
                st.subheader(name)
                st.metric("Probability of up move", f"{r['p_up']:.1%}")
                m = r["metrics"]
                if m["long_threshold"] is None or pd.isna(m["long_threshold"]):
                    thr_txt = "no historical edge ≥ costs"
                else:
                    thr_txt = f"long ≥ {m['long_threshold']:.2f}, short ≤ {m['short_threshold']:.2f}"
                st.write(f"**Paper signal:** {r['signal']}")
                st.caption(f"{r['model']} • {r['timeframe']} candles • target {r['bars']} bars • {thr_txt}")

        st.subheader("Model comparison (purged walk-forward, net of costs)")
        cmp_rows = []
        for k, r in sorted(results.items()):
            m = r["metrics"]
            cmp_rows.append({
                "Horizon": r["horizon"], "Model": r["model"], "Trades": m["trades"],
                "Win rate after costs": "—" if np.isnan(m["win_rate"]) else f"{m['win_rate']:.1%}",
                "Compounded net return": f"{m['net_compounded_return']:.2%}",
                "Mean net / trade": "—" if np.isnan(m["net_mean_trade_return"]) else f"{m['net_mean_trade_return']:.3%}",
                "Max drawdown": "—" if np.isnan(m["max_drawdown"]) else f"{m['max_drawdown']:.2%}",
                "Balanced accuracy": f"{m['model_balanced_accuracy']:.1%}",
                "Brier ↓": f"{m['model_brier']:.4f}",
                "AUC": "—" if np.isnan(m["roc_auc"]) else f"{m['roc_auc']:.3f}",
            })
        st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
        st.caption("Champion per horizon (cards above) = highest mean net return per trade with ≥8 trades; otherwise best Brier score. "
                   "Thresholds are optimized inside each training window only — test data never influences them.")

        st.subheader("Purged walk-forward validation vs baselines (champion models)")
        rows = []
        for name, r in champions.items():
            m = r["metrics"]
            rows.append({
                "Horizon": name, "Predictions": m["predictions"], "Folds": m["folds_used"],
                "Model accuracy": f"{m['model_accuracy']:.1%}",
                "Majority baseline accuracy": f"{m['majority_accuracy']:.1%}",
                "Model balanced accuracy": f"{m['model_balanced_accuracy']:.1%}",
                "Majority balanced accuracy": f"{m['majority_balanced_accuracy']:.1%}",
                "Model Brier ↓": f"{m['model_brier']:.4f}",
                "Historical-rate Brier ↓": f"{m['historical_rate_brier']:.4f}",
                "AUC": "—" if np.isnan(m["roc_auc"]) else f"{m['roc_auc']:.3f}",
                "Test up-rate": f"{m['test_up_rate']:.1%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Each fold trains only on earlier rows. Training labels whose forward horizon overlaps the test start are purged. "
                   "Baselines are calculated using information available from the training fold.")

        st.subheader("Trading-cost-aware paper strategy (champion models)")
        cost_rows = []
        for name, r in champions.items():
            m = r["metrics"]
            cost_rows.append({
                "Horizon": name, "Non-overlapping trades": m["trades"],
                "Win rate after costs": "—" if np.isnan(m["win_rate"]) else f"{m['win_rate']:.1%}",
                "Compounded net return": f"{m['net_compounded_return']:.2%}",
                "Mean net return / trade": "—" if np.isnan(m["net_mean_trade_return"]) else f"{m['net_mean_trade_return']:.3%}",
                "Max drawdown": "—" if np.isnan(m["max_drawdown"]) else f"{m['max_drawdown']:.2%}",
                "Round-trip cost": f"{m['cost_round_trip_pct']:.3f}%",
            })
        st.dataframe(pd.DataFrame(cost_rows), use_container_width=True, hide_index=True)
        st.warning("Backtest is a simplified research simulation using close-to-close forward returns, training-derived thresholds and "
                   "assumed costs. It excludes spread dynamics, funding, market impact, latency, partial fills and taxes. "
                   "Do not use it as a profitability claim.")

        st.subheader("Per-fold diagnostics (champion model)")
        chosen_pf = st.selectbox("Horizon", list(champions.keys()), key="pf_h")
        st.dataframe(champions[chosen_pf]["per_fold"], use_container_width=True, hide_index=True)
        st.caption("Purge check: last_train_row < first_test_row − horizon for every fold. long_thr/short_thr are chosen from a "
                   "held-out slice of that fold's training window.")

        st.subheader("Equity curve & trade log (champion model)")
        chosen_eq = st.selectbox("Horizon", list(champions.keys()), key="eq_h")
        tr = champions[chosen_eq]["trades"]
        if len(tr):
            tr_disp = tr.copy()
            tr_disp["side"] = np.where(tr_disp["side"] > 0, "LONG", "SHORT")
            fig = go.Figure(go.Scatter(
                x=tr_disp["exit_time"], y=tr_disp["cumulative_net"],
                mode="lines+markers", name="Equity (net of costs)",
            ))
            fig.update_layout(height=360, xaxis_title="Exit time (UTC)", yaxis_title="Cumulative net return")
            st.plotly_chart(fig, use_container_width=True)
            show = tr_disp[["fold", "entry_time", "exit_time", "side", "prob_up",
                            "future_return", "net_return", "cumulative_net"]].rename(
                columns={"future_return": "gross_return"})
            st.dataframe(show, use_container_width=True, hide_index=True)
        else:
            st.info("No trades passed the cost-aware thresholds in the walk-forward test — that is an honest result, not an error.")

        st.subheader("Long-only vs long/short (champion models)")
        lo_rows = []
        for name, r in champions.items():
            m = r["metrics"]
            lo_rows.append({
                "Horizon": name,
                "Long/short trades": m["trades"],
                "Long/short compounded net": f"{m['net_compounded_return']:.2%}",
                "Long-only trades": m["long_only_trades"],
                "Long-only compounded net": f"{m['long_only_compounded_return']:.2%}",
                "Long-only max drawdown": "—" if np.isnan(m["long_only_max_drawdown"]) else f"{m['long_only_max_drawdown']:.2%}",
            })
        st.dataframe(pd.DataFrame(lo_rows), use_container_width=True, hide_index=True)

        with st.expander("Fold prediction rows"):
            chosen_pred = st.selectbox("Horizon", list(champions.keys()), key="pr_h")
            st.dataframe(champions[chosen_pred]["predictions"].tail(100), use_container_width=True, hide_index=True)

        st.subheader("Price history")
        chosen_chart = st.selectbox("Chart horizon / candle interval", list(champions.keys()), key="ch_h")
        df = champions[chosen_chart]["df"]
        fig = go.Figure(data=[go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"],
                                             low=df["low"], close=df["close"], name="BTC")])
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
