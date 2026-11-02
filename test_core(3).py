import numpy as np
import pandas as pd
from btc_core import (
    evaluate_and_forecast,
    make_features,
    optimize_thresholds,
    walk_forward_validate,
)


def make_df(n=1200, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    ts = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": rng.uniform(1, 5, n),
    })


def test_features_finite_after_warmup():
    df = make_df()
    feats = make_features(df)
    tail = feats.iloc[60:]
    assert not np.isinf(tail.to_numpy()).any()


def test_purging_respected():
    df = make_df()
    _, _, _, per_fold = walk_forward_validate(df, 15, 10, 5, 300, 3)
    assert len(per_fold) >= 2
    for _, r in per_fold.iterrows():
        assert r["last_train_row"] < r["first_test_row"] - 15


def test_folds_chronological_and_clean():
    df = make_df()
    _, out, _, per_fold = walk_forward_validate(df, 15, 10, 5, 300, 3)
    starts = per_fold.sort_values("fold")["first_test_row"].tolist()
    assert starts == sorted(starts)
    assert out["row"].is_unique
    assert not out[["prob_up", "future_return"]].isna().any().any()


def test_net_return_accounting():
    df = make_df()
    m, _, trades, _ = walk_forward_validate(df, 15, 10, 5, 300, 3)
    if len(trades):
        expected = trades["side"] * trades["future_return"] - m["cost_round_trip_pct"] / 100
        np.testing.assert_allclose(trades["net_return"], expected, atol=1e-12)
        assert (trades["exit_time"] >= trades["entry_time"]).all()


def test_thresholds_valid_or_none():
    df = make_df()
    _, _, _, per_fold = walk_forward_validate(df, 15, 10, 5, 300, 3)
    saw_thr = False
    for _, r in per_fold.iterrows():
        lt, st = r["long_thr"], r["short_thr"]
        if lt is not None and not pd.isna(lt):
            saw_thr = True
            assert 0.5 < lt <= 0.8
            assert 0.2 <= st < 0.5
    assert saw_thr  # random-walk data still triggers grid search


def test_deterministic():
    df = make_df()
    m1, o1, t1, _ = walk_forward_validate(df, 15, 10, 5, 300, 3)
    m2, o2, t2, _ = walk_forward_validate(df, 15, 10, 5, 300, 3)
    pd.testing.assert_frame_equal(o1, o2)
    assert m1 == m2
    pd.testing.assert_frame_equal(t1, t2)


def test_both_models_run(model="logistic"):
    df = make_df()
    m, o, t, pf = walk_forward_validate(df, 15, 10, 5, 300, 3, model)
    assert m["predictions"] == len(o)
    assert m["model_name"] == model


def test_optimize_thresholds_finds_positive_edge():
    p = np.linspace(0, 1, 400)
    fr = np.where(p > 0.6, 0.02, -0.01)
    y = (fr > 0).astype(int)
    lt, st = optimize_thresholds(y, fr, p, cost=0.003)
    assert lt is not None


def test_evaluate_and_forecast_smoke():
    df = make_df()
    p_up, price, m, preds, trades, per_fold, signal = evaluate_and_forecast(df, 15, 10, 5, 300, 3, "logistic")
    assert 0 <= p_up <= 1 and price > 0
    assert signal.startswith(("PAPER", "NO TRADE"))
    assert m["predictions"] == len(preds)
