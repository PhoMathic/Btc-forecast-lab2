"""Shared core for BTC Forecast Lab.

Feature engineering, model factory, and purged expanding-window walk-forward
validation with cost-aware thresholds optimized strictly inside each training
window. Imported by app.py (Streamlit dashboard) and validation.py (CLI) so
both entry points share one implementation and cannot drift apart.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

MODEL_NAMES = ("logistic", "gradient_boosting")


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
    x["volume_change"] = df["volume"].astype(float).pct_change().replace([np.inf, -np.inf], np.nan)
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


def new_model(name="logistic", random_state=42):
    if name == "logistic":
        return Pipeline([
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        ])
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    raise ValueError(f"Unknown model: {name}. Choose from {MODEL_NAMES}.")


def max_drawdown(returns):
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return np.nan
    equity = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return float(np.min(equity / peaks - 1))


def optimize_thresholds(y, future_ret, p, cost):
    """Grid-search long/short probability thresholds on TRAINING rows only.

    Returns (long_thr, short_thr) maximizing mean net return per signal after
    round-trip costs, or (None, None) when no threshold pair beats staying flat.
    """
    y = np.asarray(y)
    future_ret = np.asarray(future_ret, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(p) < 50 or len(np.unique(y)) < 2:
        return None, None
    long_grid = np.arange(52, 81, 2) / 100  # 0.52 .. 0.80 exactly
    short_grid = np.arange(20, 49, 2) / 100  # 0.20 .. 0.48 exactly
    best_mean, best_pair = 0.0, (None, None)
    for lt in long_grid:
        long_hit = p >= lt
        for st in short_grid:
            side = np.where(long_hit, 1, np.where(p <= st, -1, 0))
            if not np.any(side):
                continue
            net = side * future_ret - cost * (side != 0)
            m = float(net.mean())
            if m > best_mean:
                best_mean, best_pair = m, (float(lt), float(st))
    return best_pair


def _select_non_overlapping(out, side_col, horizon):
    selected, next_allowed = [], -1
    for ix, row in out.sort_values("row").iterrows():
        if int(row[side_col]) != 0 and int(row["row"]) >= next_allowed:
            selected.append(ix)
            next_allowed = int(row["row"]) + horizon
    return selected


def walk_forward_validate(df, horizon, fee_bps, slippage_bps, min_train=300, n_folds=3, model_name="logistic"):
    """Expanding-window folds. Training rows whose forward-return label overlaps
    the start of a test block are purged. Trade thresholds are optimized on a
    held-out slice of each training window; test data never influences them."""
    feats = make_features(df)
    close = df["close"].astype(float)
    future_ret = close.shift(-horizon) / close - 1
    labels = (future_ret > 0).astype(int)
    valid = feats.notna().all(axis=1) & future_ret.notna() & np.isfinite(future_ret)
    rows = np.flatnonzero(valid.to_numpy())
    X = feats.loc[valid].reset_index(drop=True)
    y = labels.loc[valid].reset_index(drop=True)
    fr = future_ret.loc[valid].reset_index(drop=True)
    if len(X) < min_train + horizon + 40:
        raise ValueError(
            f"Not enough usable rows for walk-forward testing: {len(X)} usable; "
            f"need about {min_train + horizon + 40}. Increase candles or lower minimum training rows."
        )
    remaining = len(X) - min_train
    block = max(20, remaining // n_folds)
    cost = 2 * (fee_bps + slippage_bps) / 10000.0
    preds = []
    fold_id, test_start = 0, min_train
    while test_start < len(X) and fold_id < n_folds:
        test_end = len(X) if fold_id == n_folds - 1 else min(len(X), test_start + block)
        first_test_original_row = rows[test_start]
        train_positions = np.flatnonzero(rows < (first_test_original_row - horizon))
        if len(train_positions) < 100 or y.iloc[train_positions].nunique() < 2:
            test_start = test_end
            fold_id += 1
            continue
        # Cost-aware thresholds from a held-out slice of the TRAINING window only.
        cut = int(len(train_positions) * 0.7)
        if len(train_positions) - cut >= 60:
            fit_pos, thr_pos = train_positions[:cut], train_positions[cut:]
        else:
            fit_pos, thr_pos = train_positions, train_positions
        thr_model = new_model(model_name)
        thr_model.fit(X.iloc[fit_pos], y.iloc[fit_pos])
        p_thr = thr_model.predict_proba(X.iloc[thr_pos])[:, 1]
        lt, st = optimize_thresholds(
            y.iloc[thr_pos].to_numpy(), fr.iloc[thr_pos].to_numpy(), p_thr, cost
        )
        model = new_model(model_name)
        model.fit(X.iloc[train_positions], y.iloc[train_positions])
        p = model.predict_proba(X.iloc[test_start:test_end])[:, 1]
        last_train_row = int(rows[train_positions[-1]])
        for j, prob in zip(range(test_start, test_end), p):
            preds.append({
                "position": int(j),
                "row": int(rows[j]),
                "timestamp": df.loc[rows[j], "timestamp"],
                "prob_up": float(prob),
                "actual_up": int(y.iloc[j]),
                "future_return": float(fr.iloc[j]),
                "fold": fold_id + 1,
                "train_up_rate": float(y.iloc[train_positions].mean()),
                "n_train": int(len(train_positions)),
                "last_train_row": last_train_row,
                "first_test_row": int(first_test_original_row),
                "long_thr": lt,
                "short_thr": st,
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

    def _side(r):
        lt, st = r["long_thr"], r["short_thr"]
        if lt is None or pd.isna(lt):
            return 0
        return 1 if r["prob_up"] >= lt else (-1 if r["prob_up"] <= st else 0)

    out["side"] = out.apply(_side, axis=1)
    out["side_long_only"] = out.apply(
        lambda r: 1 if (r["long_thr"] is not None and not pd.isna(r["long_thr"]) and r["prob_up"] >= r["long_thr"]) else 0,
        axis=1,
    )
    out["net_return"] = np.where(out["side"] != 0, out["side"] * out["future_return"] - cost, 0.0)
    out["net_return_long_only"] = np.where(
        out["side_long_only"] != 0, out["side_long_only"] * out["future_return"] - cost, 0.0
    )

    sel = _select_non_overlapping(out, "side", horizon)
    trades = out.loc[sel].copy().sort_values("row")
    sel_lo = _select_non_overlapping(out, "side_long_only", horizon)
    trades_lo = out.loc[sel_lo].copy().sort_values("row")

    if len(trades):
        exit_rows = np.minimum(trades["row"].to_numpy() + horizon, len(df) - 1)
        trades["entry_time"] = pd.to_datetime(df.loc[trades["row"].to_numpy(), "timestamp"].to_numpy())
        trades["exit_time"] = pd.to_datetime(df.loc[exit_rows, "timestamp"].to_numpy())
        trades["cumulative_net"] = (1 + trades["net_return"]).cumprod() - 1
    if len(trades_lo):
        trades_lo["cumulative_net"] = (1 + trades_lo["net_return_long_only"]).cumprod() - 1

    pf_rows = []
    for f, g in out.groupby("fold"):
        auc = roc_auc_score(g["actual_up"], g["prob_up"]) if g["actual_up"].nunique() == 2 else np.nan
        g_tr = trades[trades["fold"] == f] if len(trades) else trades
        pf_rows.append({
            "fold": int(f),
            "n_train": int(g["n_train"].iloc[0]),
            "n_test": int(len(g)),
            "last_train_row": int(g["last_train_row"].iloc[0]),
            "first_test_row": int(g["first_test_row"].iloc[0]),
            "accuracy": float(accuracy_score(g["actual_up"], g["model_pred"])),
            "balanced_accuracy": float(balanced_accuracy_score(g["actual_up"], g["model_pred"])),
            "brier": float(brier_score_loss(g["actual_up"], g["prob_up"])),
            "auc": auc,
            "test_up_rate": float(g["actual_up"].mean()),
            "long_thr": g["long_thr"].iloc[0],
            "short_thr": g["short_thr"].iloc[0],
            "trades": int(len(g_tr)),
            "fold_net_return": float((1 + g_tr["net_return"]).prod() - 1) if len(g_tr) else 0.0,
            "test_start": str(g["timestamp"].min()),
            "test_end": str(g["timestamp"].max()),
        })
    per_fold = pd.DataFrame(pf_rows)

    summary = {
        "model_name": model_name,
        "predictions": int(len(out)),
        "folds_used": int(out["fold"].nunique()),
        "model_accuracy": float(accuracy_score(out["actual_up"], out["model_pred"])),
        "majority_accuracy": float(accuracy_score(out["actual_up"], out["majority_baseline_pred"])),
        "model_balanced_accuracy": float(balanced_accuracy_score(out["actual_up"], out["model_pred"])),
        "majority_balanced_accuracy": float(balanced_accuracy_score(out["actual_up"], out["majority_baseline_pred"])),
        "model_brier": float(brier_score_loss(out["actual_up"], out["prob_up"])),
        "historical_rate_brier": float(out["baseline_brier_component"].mean()),
        "roc_auc": float(roc_auc_score(out["actual_up"], out["prob_up"])) if out["actual_up"].nunique() == 2 else np.nan,
        "test_up_rate": float(out["actual_up"].mean()),
        "long_threshold": per_fold["long_thr"].iloc[-1] if len(per_fold) else None,
        "short_threshold": per_fold["short_thr"].iloc[-1] if len(per_fold) else None,
        "trades": int(len(trades)),
        "win_rate": float((trades["net_return"] > 0).mean()) if len(trades) else np.nan,
        "net_compounded_return": float((1 + trades["net_return"]).prod() - 1) if len(trades) else 0.0,
        "net_mean_trade_return": float(trades["net_return"].mean()) if len(trades) else np.nan,
        "max_drawdown": max_drawdown(trades["net_return"]) if len(trades) else np.nan,
        "long_only_trades": int(len(trades_lo)),
        "long_only_compounded_return": float((1 + trades_lo["net_return_long_only"]).prod() - 1) if len(trades_lo) else 0.0,
        "long_only_max_drawdown": max_drawdown(trades_lo["net_return_long_only"]) if len(trades_lo) else np.nan,
        "fee_bps_each_side": float(fee_bps),
        "slippage_bps_each_side": float(slippage_bps),
        "cost_round_trip_pct": cost * 100,
        "threshold_rule": "Optimized on a held-out slice of each training window; test data never influences thresholds.",
        "test_start": str(out["timestamp"].min()),
        "test_end": str(out["timestamp"].max()),
    }
    return summary, out, trades, per_fold


def evaluate_and_forecast(df, horizon_bars, fee_bps, slippage_bps, min_train, folds, model_name="logistic"):
    features = make_features(df)
    close = df["close"].astype(float)
    future_return = close.shift(-horizon_bars) / close - 1
    labels = (future_return > 0).astype(int)
    valid = features.notna().all(axis=1) & future_return.notna() & np.isfinite(future_return)
    X = features.loc[valid].reset_index(drop=True)
    y = labels.loc[valid].reset_index(drop=True)
    if len(X) < max(180, min_train + horizon_bars + 40) or y.nunique() < 2:
        raise ValueError(
            f"Not enough usable labelled rows ({len(X)}). Increase candle history or lower minimum training rows."
        )
    latest_features = features.iloc[[-1]]
    if latest_features.isna().any(axis=None):
        raise ValueError("Latest candle lacks enough indicator history.")
    model = new_model(model_name)
    model.fit(X, y)
    p_up = float(model.predict_proba(latest_features)[0, 1])
    metrics, predictions, trades, per_fold = walk_forward_validate(
        df, horizon_bars, fee_bps, slippage_bps, min_train, folds, model_name
    )
    lt, st = metrics["long_threshold"], metrics["short_threshold"]
    # The displayed signal uses exactly the rule that was backtested.
    if lt is None:
        signal = "NO TRADE (historical edge < costs)"
    elif p_up >= lt:
        signal = "PAPER LONG"
    elif p_up <= st:
        signal = "PAPER SHORT / AVOID LONG"
    else:
        signal = "NO TRADE"
    return p_up, float(close.iloc[-1]), metrics, predictions, trades, per_fold, signal
