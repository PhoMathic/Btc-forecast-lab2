# BTC Forecast Lab

A dashboard-first Bitcoin forecasting research starter. It fetches public OHLCV candles through CCXT and evaluates a transparent logistic-regression baseline at four horizons:

- 15 minutes using 1-minute candles
- 4 hours using 5-minute candles
- 24 hours using 15-minute candles
- 7 days using 1-hour candles

**Paper/research only.** This project does not submit exchange orders and does not request API keys. No accuracy or profitability is claimed.

## Run locally

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Choose an exchange and symbol in the sidebar, then select **Fetch market data & evaluate**. Public market data availability and symbol conventions differ by exchange.

## What is implemented

- Public OHLCV ingestion via CCXT, with rate limiting enabled
- Exclusion of the newest potentially incomplete candle
- Features: returns, rolling volatility, EMA gap, SMA gap, RSI-like momentum, volume change
- Separate model for each horizon
- Chronological 80/20 holdout diagnostics: accuracy, balanced accuracy, Brier score, holdout up-rate
- Assumed trading fees and slippage shown transparently
- Rule-based paper signal / no-trade band; no order execution

## Important limitations

1. The dashboard's first evaluation is a chronological holdout, not a complete walk-forward study. The `validation.py` script adds expanding-window walk-forward evaluation and purges training observations whose forward-return labels overlap the test fold.
2. A 0.5 probability threshold is not automatically a trading edge. Probabilities are not yet calibrated, and the model is a baseline.
3. Holdout rows have overlapping forward-return labels; adjacent observations are not independent.
4. Simulated strategy returns in `validation.py` assume a simplified close-to-close fill, a fixed cost per round trip, and a conservative probability threshold. This does not model spread dynamics, latency, partial fills, funding, market impact, or stop-loss path.
5. Exchange candles may be incomplete, missing, revised, or rate-limited. Preserve raw downloads and record exchange/symbol/timeframe/timestamp for any serious study.
6. Historical performance does not guarantee future performance.

## Reproducible walk-forward validation

Export OHLCV data with columns `timestamp,open,high,low,close,volume` to CSV. Then run, for example:

```bash
python validation.py btc_1m.csv --horizon 15 --min-train 300 --test-size 100 --fee-bps 10 --slippage-bps 5 --out predictions.csv
```

`--horizon` is measured in rows, not clock time. Use a CSV whose candle interval matches the intended horizon. The script writes one row per out-of-sample forecast and prints summary metrics. Keep a copy of the source CSV, its date range, exchange, and command line to reproduce results.

## Validation protocol before trusting a signal

- Compare to majority-class and zero-return/no-trade baselines.
- Evaluate separate market regimes and multiple time periods.
- Use purged expanding-window walk-forward folds; consider an embargo at least as long as the forecast horizon.
- Report Brier score and reliability/calibration bins, not accuracy alone.
- Account for fees, bid/ask spread, slippage, latency and funding where applicable.
- Track every live-time forecast before the outcome is known.
- Run paper trading for a meaningful period before considering any real-money use.
- Do not tune repeatedly against the final holdout.

## Data and validation resources

- CCXT OHLCV manual: https://github.com/ccxt/ccxt/wiki/Manual
- Freqtrade lookahead-bias guidance: https://www.freqtrade.io/en/stable/lookahead-analysis/
- scikit-learn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
