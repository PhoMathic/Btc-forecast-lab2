# BTC Forecast Lab — Walk-forward upgrade

This version integrates purged expanding-window walk-forward validation, baseline comparisons, and a simplified cost-aware paper strategy into the Streamlit dashboard.

## Deploy on Streamlit Community Cloud

Replace the existing `app.py` in your GitHub repository with this version. Keep `requirements.txt` unchanged. `validation.py` is included as a separate reusable CSV validation script.

- Exchange default: Kraken
- Market default: BTC/USD
- Candle count default: 1,000 per horizon
- Fees/slippage are editable assumptions, not fetched from your account.
- No exchange credentials or live orders are used.

## What the dashboard adds

- Chronological expanding-window test blocks.
- Purging: training rows whose forward-return label overlaps the start of a test block are excluded.
- Model accuracy and balanced accuracy compared with a training-fold majority baseline.
- Brier score compared with a training-fold historical-rate probability baseline.
- Fixed probability thresholds (>= 0.58 long, <= 0.42 short) for a research-only paper strategy.
- Estimated round-trip fees and slippage.
- Non-overlapping trades by horizon, compounded net return, win rate, and maximum drawdown.
- Prediction rows for review.

## Limitations

This is a research prototype, not evidence of a profitable strategy. Backtest returns use close-to-close forward returns and simplified costs. It does not model spread variation, funding, borrow costs, market impact, latency, partial fills, taxes, or operational outages. Results can be unstable with small samples. Compare against baselines and validate across more data and market regimes before relying on signals.

## Android deployment

1. Download and extract the ZIP.
2. In GitHub, open `btc-forecast-lab` and open `app.py`.
3. Use the file edit menu if available, or choose Add file → Upload files to upload the replacement `app.py` from this bundle. If uploading with the same name, GitHub will prompt you to replace/commit the changed file.
4. Commit the change to `main`.
5. Streamlit Community Cloud should redeploy after the commit. If not, open Manage app and reboot/redeploy.
6. Open the app, leave Kraken / BTC/USD selected, then tap Fetch market data & evaluate.
7. Read the **Purged walk-forward validation vs baselines** and **Trading-cost-aware paper strategy** tables.

## Standalone CSV validator

`validation.py` expects a CSV with `timestamp,open,high,low,close,volume`. Example:
`python validation.py candles.csv --horizon 15 --min-train 300 --test-size 100 --fee-bps 10 --slippage-bps 5`

Note: `--horizon` is in candle rows, so use the matching candle interval for the chosen forecast horizon.
