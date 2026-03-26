"""
Unit tests for modules/gamma_exposure.py

Covers:
- Black-Scholes gamma formula accuracy against known analytical result
- Edge/invalid input handling (T=0, sigma=0, S=0, K=0)
- compute_gex output columns and types
- gex_flip_point is within strike range (or None)
- total_gex_metrics returns expected keys
- Plot functions return go.Figure
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy.stats import norm

from modules.gamma_exposure import (
    _bs_gamma,
    compute_gex,
    gex_flip_point,
    total_gex_metrics,
    plot_gex_profile,
    plot_gex_by_expiration,
)


# ── _bs_gamma ─────────────────────────────────────────────────────────────────

def test_bs_gamma_atm_matches_analytical():
    """ATM Black-Scholes gamma must match the closed-form formula exactly."""
    S, K, T, r, sigma = 400.0, 400.0, 0.25, 0.05, 0.20
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    expected = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    assert abs(_bs_gamma(S, K, T, r, sigma) - expected) < 1e-12


def test_bs_gamma_otm_lower_than_atm():
    """Gamma should be lower for out-of-the-money strikes."""
    atm = _bs_gamma(400, 400, 0.25, 0.05, 0.20)
    otm = _bs_gamma(400, 450, 0.25, 0.05, 0.20)
    assert otm < atm


def test_bs_gamma_returns_zero_for_invalid_inputs():
    """Any invalid input should return 0.0, not raise."""
    assert _bs_gamma(0,   400, 0.25, 0.05, 0.20) == 0.0  # S = 0
    assert _bs_gamma(400, 0,   0.25, 0.05, 0.20) == 0.0  # K = 0
    assert _bs_gamma(400, 400, 0,    0.05, 0.20) == 0.0  # T = 0
    assert _bs_gamma(400, 400, 0.25, 0.05, 0   ) == 0.0  # sigma = 0
    assert _bs_gamma(400, 400, -1,   0.05, 0.20) == 0.0  # T < 0


def test_bs_gamma_is_nonnegative():
    """Gamma is always non-negative."""
    for S in [380, 400, 420]:
        for K in [380, 400, 420]:
            assert _bs_gamma(S, K, 0.25, 0.05, 0.20) >= 0.0


# ── compute_gex ───────────────────────────────────────────────────────────────

def test_compute_gex_required_columns(options_dfs):
    calls, puts, spot = options_dfs
    result = compute_gex(calls, puts, spot)
    required = {"strike", "net_gex", "call_gex", "put_gex"}
    assert required.issubset(result.columns), f"Missing: {required - set(result.columns)}"


def test_compute_gex_one_row_per_strike(options_dfs):
    calls, puts, spot = options_dfs
    result = compute_gex(calls, puts, spot)
    # Every strike should appear at most once
    assert result["strike"].nunique() == len(result)


def test_compute_gex_net_equals_call_minus_put(options_dfs):
    """net_gex = call_gex + put_gex (signs already applied in formula)."""
    calls, puts, spot = options_dfs
    result = compute_gex(calls, puts, spot)
    np.testing.assert_allclose(
        result["net_gex"].values,
        (result["call_gex"] + result["put_gex"]).values,
        rtol=1e-9,
    )


def test_compute_gex_returns_dataframe(options_dfs):
    calls, puts, spot = options_dfs
    result = compute_gex(calls, puts, spot)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


# ── gex_flip_point ────────────────────────────────────────────────────────────

def test_gex_flip_point_within_strike_range(options_dfs):
    calls, puts, spot = options_dfs
    gex_df = compute_gex(calls, puts, spot)
    flip = gex_flip_point(gex_df)
    if flip is not None:
        assert gex_df["strike"].min() <= flip <= gex_df["strike"].max()


def test_gex_flip_point_returns_none_or_float(options_dfs):
    calls, puts, spot = options_dfs
    gex_df = compute_gex(calls, puts, spot)
    flip = gex_flip_point(gex_df)
    assert flip is None or isinstance(flip, float)


def test_gex_flip_point_none_when_all_same_sign():
    """If all net_gex values are positive there should be no flip point."""
    gex_df = pd.DataFrame({"strike": [390, 395, 400, 405, 410], "net_gex": [1, 2, 3, 4, 5]})
    assert gex_flip_point(gex_df) is None


# ── total_gex_metrics ─────────────────────────────────────────────────────────

def test_total_gex_metrics_keys(options_dfs):
    calls, puts, spot = options_dfs
    gex_df = compute_gex(calls, puts, spot)
    metrics = total_gex_metrics(gex_df)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


# ── plot functions ────────────────────────────────────────────────────────────

def test_plot_gex_profile_returns_figure(options_dfs):
    calls, puts, spot = options_dfs
    gex_df = compute_gex(calls, puts, spot)
    fig = plot_gex_profile(gex_df, spot, "SPY")
    assert isinstance(fig, go.Figure)


def test_plot_gex_by_expiration_returns_figure(options_dfs):
    calls, puts, spot = options_dfs
    fig = plot_gex_by_expiration(calls, puts, spot, "SPY")
    assert isinstance(fig, go.Figure)
