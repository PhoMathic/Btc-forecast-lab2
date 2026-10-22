"""Reusable purged walk-forward validation helpers for BTC Forecast Lab.

This module uses local OHLCV CSV data and is deterministic for fixed input/settings.
CSV columns: timestamp, open, high, low, close, volume.
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
    ok = x.notna().all(axis=1) & future_ret.notna() & np.isfinite(future_ret)
    original_rows = np.flatnonzero(ok.to_numpy())
    x = x.loc[ok].reset_index(drop=True)
    y = y.loc[ok].reset_index(drop=True)
    future_ret = future_ret.loc[ok].reset_index(drop=True)
    preds = []
    cost = 2 * (fee_bps + slippage_bps) / 10000
    start = min_train
    while start < len(x):
        end = min(start + test_size, len(x))
        first_test_row = original_rows[start]
        train_pos = np.flatnonzero(original_rows < first_test_row - horizon)
        if len(train_pos) < 100 or y.iloc[train_pos].nunique() < 2:
            start = end
            continue
        model = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))])
        model.fit(x.iloc[train_pos], y.iloc[train_pos])
        p = model.predict_proba(x.iloc[start:end])[:, 1]
        for j, prob in zip(range(start, end), p):
            preds.append({
                "row": int(original_rows[j]),
                "timestamp": df.loc[original_rows[j], "timestamp"],
                "prob_up": float(prob),
                "actual_up": int(y.iloc[j]),
                "future_return": float(future_ret.iloc[j]),
                "train_up_rate": float(y.iloc[train_pos].mean())
            })
        start = end
    out = pd.DataFrame(preds)
    if out.empty:
        raise ValueError("Not enough data for requested horizon and fold settings.")
    out["model_pred"] = (out.prob_up >= 0.5).astype(int)
    out["majority_pred"] = (out.train_up_rate >= 0.5).astype(int)
    out["baseline_prob"] = out.train_up_rate
    out["side"] = np.where(out.prob_up >= 0.58, 1, np.where(out.prob_up <= 0.42, -1, 0))
    out["net_return"] = np.where(out.side != 0, out.side * out.future_return - cost, 0.0)
    selected, next_allowed = [], -1
    for idx, row in out.sort_values("row").iterrows():
        if row.side != 0 and int(row.row) >= next_allowed:
            selected.append(idx)
            next_allowed = int(row.row) + horizon
    trades = out.loc[selected]
    equity = np.cumprod(1 + trades.net_return.to_numpy()) if len(trades) else np.array([])
    if len(equity):
        peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
        max_dd = float(np.min(equity / peak - 1))
    else:
        max_dd = None
    summary = {
        "n_predictions": int(len(out)),
        "folds_not_explicit": "Predictions are generated in chronological test blocks.",
        "model_accuracy": float(accuracy_score(out.actual_up, out.model_pred)),
        "majority_baseline_accuracy": float(accuracy_score(out.actual_up, out.majority_pred)),
        "model_balanced_accuracy": float(balanced_accuracy_score(out.actual_up, out.model_pred)),
        "model_brier": float(brier_score_loss(out.actual_up, out.prob_up)),
        "historical_rate_brier": float(np.mean((out.baseline_prob - out.actual_up) ** 2)),
        "actual_up_rate": float(out.actual_up.mean()),
        "trades": int(len(trades)),
        "win_rate_after_costs": float((trades.net_return > 0).mean()) if len(trades) else None,
        "compounded_net_return": float(np.prod(1 + trades.net_return) - 1) if len(trades) else 0.0,
        "max_drawdown": max_dd,
        "fee_bps_each_side": float(fee_bps),
        "slippage_bps_each_side": float(slippage_bps),
        "round_trip_cost_pct": float(cost * 100),
        "warning": "Simplified close-to-close simulation; excludes funding, spread dynamics, market impact, latency, partial fills and taxes."
    }
    return out, summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--min-train", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--fee-bps", type=float, default=10)
    parser.add_argument("--slippage-bps", type=float, default=5)
    parser.add_argument("--out", default="walk_forward_predictions.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True, errors="raise")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    predictions, summary = walk_forward(df, args.horizon, args.min_train, args.test_size, args.fee_bps, args.slippage_bps)
    predictions.to_csv(args.out, index=False)
    print(json.dumps(summary, indent=2))
    print(f"Predictions saved to {args.out}")

if __name__ == "__main__":
    main()
