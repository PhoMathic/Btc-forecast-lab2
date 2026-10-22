import pandas as pd
import numpy as np
from validation import features_for, walk_forward

def synthetic_data(n=1600, seed=42):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0001, 0.004, n)
    close = 50000 * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="min", tz="UTC"),
        "open": close * (1 + rng.normal(0, 0.0005, n)),
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": rng.uniform(1, 100, n)
    })

def test_features_have_expected_columns():
    df = synthetic_data()
    x = features_for(df)
    assert "ret_1" in x.columns
    assert "rsi" in x.columns

def test_walk_forward_returns_out_of_sample_predictions():
    df = synthetic_data()
    pred, summary = walk_forward(df, horizon=5, min_train=300, test_size=100)
    assert len(pred) > 0
    assert 0 <= summary["brier_score"] <= 1
    assert "net_strategy_total_return_compounded" in summary
