"""
Shared pytest fixtures for the Market Analysis test suite.
All tests use synthetic deterministic data — no network calls in fixtures.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def ohlcv_df():
    """
    500 business days of synthetic OHLCV data starting 2023-01-03.
    Deterministic via seed 42 — results are stable across runs.
    """
    dates = pd.bdate_range("2023-01-03", periods=500)
    rng = np.random.default_rng(42)
    close = 400 + np.cumsum(rng.normal(0, 2, 500))
    open_ = close * rng.uniform(0.998, 1.002, 500)
    high  = np.maximum(close, open_) * rng.uniform(1.000, 1.005, 500)
    low   = np.minimum(close, open_) * rng.uniform(0.995, 1.000, 500)
    return pd.DataFrame({
        "Open":   open_,
        "High":   high,
        "Low":    low,
        "Close":  close,
        "Volume": rng.integers(50_000_000, 150_000_000, 500).astype(float),
    }, index=dates)


@pytest.fixture
def options_dfs():
    """
    Minimal synthetic calls/puts DataFrames and a spot price for GEX tests.
    Strikes span ±10% around spot=400 with realistic IV and OI.
    """
    spot = 400.0
    rng = np.random.default_rng(7)
    strikes = np.arange(360, 445, 5, dtype=float)
    n = len(strikes)

    calls = pd.DataFrame({
        "strike":            strikes,
        "impliedVolatility": rng.uniform(0.10, 0.40, n),
        "openInterest":      rng.integers(100, 5000, n).astype(float),
        "gamma":             rng.uniform(0.001, 0.05, n),
        "expiration":        "2024-03-15",
    })
    puts = pd.DataFrame({
        "strike":            strikes,
        "impliedVolatility": rng.uniform(0.10, 0.45, n),
        "openInterest":      rng.integers(100, 5000, n).astype(float),
        "gamma":             rng.uniform(0.001, 0.05, n),
        "expiration":        "2024-03-15",
    })
    return calls, puts, spot
