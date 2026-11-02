"""Standalone purged walk-forward validator for BTC Forecast Lab.

Uses the shared btc_core module (same logic as the Streamlit dashboard).
CSV columns: timestamp, open, high, low, close, volume.

Example:
    python validation.py candles.csv --horizon 15 --min-train 300 --folds 3 \
        --model gradient_boosting --fee-bps 10 --slippage-bps 5

Note: --horizon is in candle rows, so use the matching candle interval for
the chosen forecast horizon.
"""
import argparse
import json
import pandas as pd

from btc_core import MODEL_NAMES, walk_forward_validate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--min-train", type=int, default=300)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--model", choices=MODEL_NAMES, default="logistic")
    parser.add_argument("--fee-bps", type=float, default=10)
    parser.add_argument("--slippage-bps", type=float, default=5)
    parser.add_argument("--out", default="walk_forward_predictions.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    metrics, predictions, trades, per_fold = walk_forward_validate(
        df, args.horizon, args.fee_bps, args.slippage_bps, args.min_train, args.folds, args.model
    )
    predictions.to_csv(args.out, index=False)
    per_fold.to_csv(args.out.replace(".csv", "_folds.csv"), index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Predictions saved to {args.out}")
    print(f"Per-fold summary saved to {args.out.replace('.csv', '_folds.csv')}")


if __name__ == "__main__":
    main()
