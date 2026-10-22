"""Reproducible chronological walk-forward validation on a local OHLCV CSV.

CSV columns required: timestamp, open, high, low, close, volume.
Timestamp should be UTC or parseable as UTC. This script never downloads data
and is deterministic given the input file and settings.
"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss

def features_for(df):
    c = df.close.astype(float)
    r = c.pct_change()
    x = pd.DataFrame(index=df.index)
    x["ret_1"] = r
    x["ret_3"] = c.pct_change(3)
    x["ret_6"] = c.pct_change(6)
    x["ret_12"] = c.pct_change(12)
    x["vol_12"] = r.rolling(12).std()
    x["vol_48"] = r.rolling(48).std()
    x["ema_gap"] = c.ewm(span=12, adjust=False).mean() / c.ewm(span=26, adjust=False).mean() - 1
    x["sma_gap"] = c / c.rolling(20).mean() - 1
    d = c.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    x["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))) / 100
    x["volume_change"] = df.volume.pct_change().replace([np.inf, -np.inf], np.nan)
    return x.replace([np.inf, -np.inf], np.nan)

def walk_forward(df, horizon, min_train=300, test_size=100, fee_bps=10, slippage_bps=5):
    x = features_for(df)
    future_ret = df.close.shift(-horizon) / df.close - 1
    y = (future_ret > 0).astype(int)
    ok = x.notna().all(axis=1) & future_ret.notna()
    x, y, future_ret = x.loc[ok], y.loc[ok], future_ret.loc[ok]
    predictions = []
    start = min_train
    while start < len(x):
        end = min(start + test_size, len(x))
        # Purge training rows whose future-return label would overlap the test start.
        train_end = start - horizon
        if train_end < 100:
            start = end
            continue
        X_train, y_train = x.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = x.iloc[start:end], y.iloc[start:end]
        if y_train.nunique() < 2 or len(X_test) == 0:
            start = end
            continue
        model = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))])
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        pred = (p >= 0.5).astype(int)
        # Simple directional strategy: long if P(up) >= 0.58, short if <= 0.42, otherwise flat.
        side = np.where(p >= 0.58, 1, np.where(p <= 0.42, -1, 0))
        cost = 2 * (fee_bps + slippage_bps) / 10000
        net_return = side * future_ret.iloc[start:end].to_numpy() - (side != 0) * cost
        for idx, pp, actual, pr, net in zip(x.index[start:end], p, y_test.to_numpy(), future_ret.iloc[start:end].to_numpy(), net_return):
            predictions.append({"row": int(idx), "prob_up": float(pp), "actual_up": int(actual), "future_return": float(pr), "strategy_net_return": float(net)})
        start = end
    out = pd.DataFrame(predictions)
    if out.empty:
        raise ValueError("Not enough data for requested horizon and fold settings.")
    summary = {
        "n_predictions": int(len(out)),
        "directional_accuracy_at_0_5": float(accuracy_score(out.actual_up, out.prob_up >= 0.5)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(out.actual_up, out.prob_up >= 0.5)),
        "brier_score": float(brier_score_loss(out.actual_up, out.prob_up)),
        "actual_up_rate": float(out.actual_up.mean()),
        "net_strategy_total_return_compounded": float((1 + out.strategy_net_return).prod() - 1),
        "net_strategy_mean_return_per_forecast": float(out.strategy_net_return.mean()),
        "fee_bps_each_side": fee_bps,
        "slippage_bps_each_side": slippage_bps,
        "horizon_bars": horizon,
        "warning": "Overlapping horizons and simplified fills can still bias strategy returns; this is research output, not a profitability claim."
    }
    return out, summary

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--min-train", type=int, default=300)
    p.add_argument("--test-size", type=int, default=100)
    p.add_argument("--fee-bps", type=float, default=10)
    p.add_argument("--slippage-bps", type=float, default=5)
    p.add_argument("--out", default="walk_forward_predictions.csv")
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    out, summary = walk_forward(df, args.horizon, args.min_train, args.test_size, args.fee_bps, args.slippage_bps)
    out.to_csv(args.out, index=False)
    print(json.dumps(summary, indent=2))
    print(f"Predictions saved to {args.out}")

if __name__ == "__main__":
    main()
