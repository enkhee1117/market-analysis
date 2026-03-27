"""
Unit tests for modules/gamma_exposure.py

Covers:
- Black-Scholes gamma formula accuracy against known analytical result
- Edge/invalid input handling (T=0, sigma=0, S=0, K=0)
- GEX sign convention: calls positive (stabilizing), puts negative (destabilizing)
- GEX formula: Gamma * OI * 100 * Spot^2 * 0.01
- Scaling to billions (divide by 1e9)
- Black-Scholes fallback when API gamma is missing/zero
- gex_flip_point linear interpolation accuracy
- total_gex_metrics aggregation correctness
- Multi-expiration GEX aggregation
- Plot functions return go.Figure
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy.stats import norm

from modules.gamma_exposure import (
    _bs_gamma,
    _add_computed_gamma,
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


def test_bs_gamma_peaks_near_atm():
    """Gamma should peak at or near the at-the-money strike."""
    spot = 400.0
    strikes = np.arange(350, 451, 5)
    gammas = [_bs_gamma(spot, k, 0.25, 0.05, 0.20) for k in strikes]
    peak_strike = strikes[np.argmax(gammas)]
    assert abs(peak_strike - spot) <= 10  # peak within $10 of ATM


def test_bs_gamma_decreases_with_time():
    """ATM gamma should decrease as time to expiry increases (more time = flatter curve)."""
    short_term = _bs_gamma(400, 400, 0.05, 0.05, 0.20)  # ~2.5 weeks
    long_term = _bs_gamma(400, 400, 1.0, 0.05, 0.20)     # 1 year
    assert short_term > long_term


# ── GEX Sign Convention ──────────────────────────────────────────────────────

def test_call_gex_is_positive():
    """Dealers short calls → call GEX should always be positive (stabilizing)."""
    calls = pd.DataFrame({
        "strike": [400.0],
        "gamma": [0.01],
        "openInterest": [1000.0],
        "expiration": ["2025-06-20"],
    })
    puts = pd.DataFrame(columns=["strike", "gamma", "openInterest", "expiration"])
    result = compute_gex(calls, puts, 400.0)
    assert (result["call_gex"] > 0).all(), "Call GEX must be positive"


def test_put_gex_is_negative():
    """Dealers long puts → put GEX should always be negative (destabilizing)."""
    calls = pd.DataFrame(columns=["strike", "gamma", "openInterest", "expiration"])
    puts = pd.DataFrame({
        "strike": [400.0],
        "gamma": [0.01],
        "openInterest": [1000.0],
        "expiration": ["2025-06-20"],
    })
    result = compute_gex(calls, puts, 400.0)
    assert (result["put_gex"] < 0).all(), "Put GEX must be negative"


# ── GEX Formula Accuracy ─────────────────────────────────────────────────────

def test_gex_formula_single_call():
    """Verify exact GEX calculation: Gamma * OI * 100 * Spot^2 * 0.01"""
    gamma, oi, spot = 0.02, 5000.0, 500.0
    expected_call_gex = gamma * oi * 100 * (spot ** 2) * 0.01

    calls = pd.DataFrame({
        "strike": [500.0],
        "gamma": [gamma],
        "openInterest": [oi],
        "expiration": ["2025-06-20"],
    })
    puts = pd.DataFrame(columns=["strike", "gamma", "openInterest", "expiration"])
    result = compute_gex(calls, puts, spot)

    assert len(result) == 1
    np.testing.assert_allclose(result["call_gex"].iloc[0], expected_call_gex, rtol=1e-9)


def test_gex_formula_single_put():
    """Verify exact put GEX: -Gamma * OI * 100 * Spot^2 * 0.01"""
    gamma, oi, spot = 0.015, 3000.0, 500.0
    expected_put_gex = -gamma * oi * 100 * (spot ** 2) * 0.01

    calls = pd.DataFrame(columns=["strike", "gamma", "openInterest", "expiration"])
    puts = pd.DataFrame({
        "strike": [500.0],
        "gamma": [gamma],
        "openInterest": [oi],
        "expiration": ["2025-06-20"],
    })
    result = compute_gex(calls, puts, spot)

    assert len(result) == 1
    np.testing.assert_allclose(result["put_gex"].iloc[0], expected_put_gex, rtol=1e-9)


def test_net_gex_at_same_strike():
    """When calls and puts share a strike, net GEX = call_gex + put_gex."""
    spot = 400.0
    calls = pd.DataFrame({
        "strike": [400.0], "gamma": [0.02], "openInterest": [2000.0],
        "expiration": ["2025-06-20"],
    })
    puts = pd.DataFrame({
        "strike": [400.0], "gamma": [0.03], "openInterest": [3000.0],
        "expiration": ["2025-06-20"],
    })
    result = compute_gex(calls, puts, spot)

    call_gex = 0.02 * 2000 * 100 * (400 ** 2) * 0.01
    put_gex = -0.03 * 3000 * 100 * (400 ** 2) * 0.01
    expected_net = call_gex + put_gex

    assert len(result) == 1  # same strike, aggregated
    np.testing.assert_allclose(result["net_gex"].iloc[0], expected_net, rtol=1e-9)


# ── Billions Scaling ─────────────────────────────────────────────────────────

def test_billions_scaling():
    """_b columns must equal raw columns divided by 1e9."""
    calls = pd.DataFrame({
        "strike": [400.0], "gamma": [0.02], "openInterest": [5000.0],
        "expiration": ["2025-06-20"],
    })
    puts = pd.DataFrame({
        "strike": [400.0], "gamma": [0.015], "openInterest": [3000.0],
        "expiration": ["2025-06-20"],
    })
    result = compute_gex(calls, puts, 400.0)

    np.testing.assert_allclose(result["call_gex_b"].values, result["call_gex"].values / 1e9, rtol=1e-12)
    np.testing.assert_allclose(result["put_gex_b"].values, result["put_gex"].values / 1e9, rtol=1e-12)
    np.testing.assert_allclose(result["net_gex_b"].values, result["net_gex"].values / 1e9, rtol=1e-12)


# ── Black-Scholes Fallback ───────────────────────────────────────────────────

def test_bs_fallback_when_gamma_all_zero():
    """If API returns gamma=0, compute_gex should recompute via Black-Scholes."""
    spot = 400.0
    calls = pd.DataFrame({
        "strike": [400.0],
        "gamma": [0.0],  # zero = trigger B-S fallback
        "openInterest": [1000.0],
        "impliedVolatility": [0.20],
        "expiration": ["2027-06-20"],  # must be future date for T > 0
    })
    puts = pd.DataFrame(columns=["strike", "gamma", "openInterest", "impliedVolatility", "expiration"])
    result = compute_gex(calls, puts, spot)

    # Should have computed a non-zero GEX via B-S
    assert not result.empty
    assert result["call_gex"].iloc[0] > 0


def test_bs_fallback_when_gamma_column_missing():
    """If gamma column is absent entirely, compute_gex should use B-S."""
    spot = 400.0
    calls = pd.DataFrame({
        "strike": [400.0],
        "openInterest": [1000.0],
        "impliedVolatility": [0.20],
        "expiration": ["2027-06-20"],  # must be future date for T > 0
    })
    puts = pd.DataFrame(columns=["strike", "openInterest", "impliedVolatility", "expiration"])
    result = compute_gex(calls, puts, spot)

    assert not result.empty
    assert result["call_gex"].iloc[0] > 0


def test_add_computed_gamma_uses_expiration_for_tte():
    """_add_computed_gamma should parse expiration to calculate time-to-expiry."""
    spot = 400.0
    df = pd.DataFrame({
        "strike": [400.0],
        "impliedVolatility": [0.20],
        "expiration": ["2030-01-01"],  # far future = long T
        "openInterest": [1000.0],
    })
    result = _add_computed_gamma(df, spot)
    assert "gamma" in result.columns
    assert result["gamma"].iloc[0] > 0


# ── Edge Cases ───────────────────────────────────────────────────────────────

def test_compute_gex_empty_with_zero_spot():
    """Should return empty DataFrame when spot is 0 or None."""
    calls = pd.DataFrame({"strike": [400.0], "gamma": [0.01], "openInterest": [100.0]})
    assert compute_gex(calls, calls, 0).empty
    assert compute_gex(calls, calls, None).empty


def test_compute_gex_empty_when_no_oi():
    """Contracts with zero OI should be filtered out."""
    calls = pd.DataFrame({
        "strike": [400.0], "gamma": [0.01], "openInterest": [0.0],
        "expiration": ["2025-06-20"],
    })
    puts = pd.DataFrame(columns=["strike", "gamma", "openInterest", "expiration"])
    result = compute_gex(calls, puts, 400.0)
    assert result.empty


def test_compute_gex_empty_inputs():
    """Empty DataFrames should return empty result."""
    result = compute_gex(pd.DataFrame(), pd.DataFrame(), 400.0)
    assert result.empty


def test_compute_gex_none_inputs():
    """None inputs should return empty result."""
    result = compute_gex(None, None, 400.0)
    assert result.empty


# ── compute_gex (original tests) ────────────────────────────────────────────

def test_compute_gex_required_columns(options_dfs):
    calls, puts, spot = options_dfs
    result = compute_gex(calls, puts, spot)
    required = {"strike", "net_gex", "call_gex", "put_gex",
                "call_gex_b", "put_gex_b", "net_gex_b"}
    assert required.issubset(result.columns), f"Missing: {required - set(result.columns)}"


def test_compute_gex_one_row_per_strike(options_dfs):
    calls, puts, spot = options_dfs
    result = compute_gex(calls, puts, spot)
    assert result["strike"].nunique() == len(result)


def test_compute_gex_net_equals_call_plus_put(options_dfs):
    """net_gex = call_gex + put_gex (signs already applied in formula)."""
    calls, puts, spot = options_dfs
    result = compute_gex(calls, puts, spot)
    np.testing.assert_allclose(
        result["net_gex"].values,
        (result["call_gex"] + result["put_gex"]).values,
        rtol=1e-9,
    )


def test_compute_gex_sorted_by_strike(options_dfs):
    """Output should be sorted by strike price ascending."""
    calls, puts, spot = options_dfs
    result = compute_gex(calls, puts, spot)
    assert (result["strike"].diff().dropna() >= 0).all()


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


def test_gex_flip_point_interpolation_accuracy():
    """Verify linear interpolation: flip between strikes 400 and 410 where GEX crosses zero."""
    gex_df = pd.DataFrame({
        "strike": [390.0, 400.0, 410.0, 420.0],
        "net_gex": [100.0, 50.0, -50.0, -100.0],
    })
    flip = gex_flip_point(gex_df)
    # Linear interpolation: 400 + 50 * (410-400) / (50+50) = 405.0
    assert flip is not None
    np.testing.assert_allclose(flip, 405.0, atol=0.01)


def test_gex_flip_point_nearest_to_spot():
    """When spot is provided, flip point should be the crossing nearest to spot."""
    gex_df = pd.DataFrame({
        "strike": [200.0, 210.0, 400.0, 410.0],
        "net_gex": [10.0, -5.0, 8.0, -3.0],
    })
    # Without spot: picks crossing with largest magnitude
    flip_no_spot = gex_flip_point(gex_df)
    assert flip_no_spot is not None

    # With spot near 400: should pick the 400-410 crossing, not the 200-210 one
    flip_with_spot = gex_flip_point(gex_df, spot=405.0)
    assert flip_with_spot is not None
    assert 400.0 <= flip_with_spot <= 410.0

def test_gex_flip_point_finds_meaningful_crossing():
    """If multiple crossings exist, should find largest-magnitude one without spot."""
    gex_df = pd.DataFrame({
        "strike": [390.0, 400.0, 410.0, 420.0, 430.0],
        "net_gex": [10.0, -5.0, 8.0, -3.0, 1.0],
    })
    flip = gex_flip_point(gex_df)
    assert flip is not None
    # Should still find a valid crossing
    assert 390.0 <= flip <= 430.0


def test_gex_flip_point_empty_df():
    assert gex_flip_point(pd.DataFrame()) is None


# ── total_gex_metrics ─────────────────────────────────────────────────────────

def test_total_gex_metrics_keys(options_dfs):
    calls, puts, spot = options_dfs
    gex_df = compute_gex(calls, puts, spot)
    metrics = total_gex_metrics(gex_df)
    expected_keys = {"total_net_gex_b", "total_call_gex_b", "total_put_gex_b",
                     "peak_call_strike", "peak_put_strike"}
    assert expected_keys == set(metrics.keys())


def test_total_gex_metrics_net_equals_sum(options_dfs):
    """total_net should equal total_call + total_put."""
    calls, puts, spot = options_dfs
    gex_df = compute_gex(calls, puts, spot)
    m = total_gex_metrics(gex_df)
    np.testing.assert_allclose(
        m["total_net_gex_b"],
        m["total_call_gex_b"] + m["total_put_gex_b"],
        atol=0.001,
    )


def test_total_gex_metrics_empty():
    assert total_gex_metrics(pd.DataFrame()) == {}


def test_total_gex_metrics_call_positive_put_negative(options_dfs):
    """Total call GEX should be positive, total put GEX should be negative."""
    calls, puts, spot = options_dfs
    gex_df = compute_gex(calls, puts, spot)
    m = total_gex_metrics(gex_df)
    assert m["total_call_gex_b"] > 0, "Total call GEX should be positive"
    assert m["total_put_gex_b"] < 0, "Total put GEX should be negative"


# ── Massive.com API data format ──────────────────────────────────────────────

def test_compute_gex_with_massive_format():
    """compute_gex should work with Massive.com API column names (camelCase openInterest)."""
    spot = 550.0
    calls = pd.DataFrame({
        "strike": [540.0, 550.0, 560.0],
        "gamma": [0.008, 0.015, 0.007],
        "openInterest": [2000, 5000, 3000],
        "impliedVolatility": [0.22, 0.20, 0.23],
        "expiration": ["2025-07-18"] * 3,
    })
    puts = pd.DataFrame({
        "strike": [540.0, 550.0, 560.0],
        "gamma": [0.009, 0.014, 0.006],
        "openInterest": [1500, 4000, 2500],
        "impliedVolatility": [0.25, 0.21, 0.24],
        "expiration": ["2025-07-18"] * 3,
    })
    result = compute_gex(calls, puts, spot)
    assert len(result) == 3
    assert (result["call_gex"] > 0).all()
    assert (result["put_gex"] < 0).all()


# ── Multi-expiration aggregation ─────────────────────────────────────────────

def test_multi_expiration_aggregation():
    """GEX by expiration should compute independent GEX for each date."""
    spot = 400.0
    calls = pd.DataFrame({
        "strike": [400.0, 400.0],
        "gamma": [0.02, 0.01],
        "openInterest": [1000.0, 2000.0],
        "expiration": ["2025-06-20", "2025-07-18"],
    })
    puts = pd.DataFrame({
        "strike": [400.0, 400.0],
        "gamma": [0.015, 0.012],
        "openInterest": [800.0, 1500.0],
        "expiration": ["2025-06-20", "2025-07-18"],
    })

    # Verify each expiration computes independently
    for exp in ["2025-06-20", "2025-07-18"]:
        c = calls[calls["expiration"] == exp]
        p = puts[puts["expiration"] == exp]
        g = compute_gex(c, p, spot)
        assert not g.empty
        assert (g["call_gex"] > 0).all()
        assert (g["put_gex"] < 0).all()


# ── plot functions ────────────────────────────────────────────────────────────

def test_plot_gex_profile_returns_figure(options_dfs):
    calls, puts, spot = options_dfs
    gex_df = compute_gex(calls, puts, spot)
    fig = plot_gex_profile(gex_df, spot, "SPY")
    assert isinstance(fig, go.Figure)


def test_plot_gex_profile_empty_data():
    """Should return a Figure even with empty data (with error title)."""
    fig = plot_gex_profile(pd.DataFrame(), 400.0, "SPY")
    assert isinstance(fig, go.Figure)


def test_plot_gex_by_expiration_returns_figure(options_dfs):
    calls, puts, spot = options_dfs
    fig = plot_gex_by_expiration(calls, puts, spot, "SPY")
    assert isinstance(fig, go.Figure)


# ── View mode: Net GEX bars ─────────────────────────────────────────────────

def test_plot_gex_profile_net_view_returns_figure(options_dfs):
    """Net GEX view mode should produce a single bar trace (no call/put bars)."""
    calls, puts, spot = options_dfs
    gex_df = compute_gex(calls, puts, spot)
    fig = plot_gex_profile(gex_df, spot, "SPY", view_mode="Net GEX")
    assert isinstance(fig, go.Figure)
    # Should have exactly 1 Bar trace (Net GEX) — no Scatter line
    bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
    scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
    assert len(bar_traces) == 1, "Net GEX view should have a single bar trace"
    assert len(scatter_traces) == 0, "Net GEX view should have no scatter line"


def test_plot_gex_profile_call_put_view_has_two_bar_traces(options_dfs):
    """Call/Put view should have 2 Bar traces (call + put), no net GEX line."""
    calls, puts, spot = options_dfs
    gex_df = compute_gex(calls, puts, spot)
    fig = plot_gex_profile(gex_df, spot, "SPY", view_mode="Call / Put")
    bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
    scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
    assert len(bar_traces) == 2, "Call/Put view should have 2 bar traces"
    assert len(scatter_traces) == 0, "Call/Put view should have no scatter lines"


# ── Gamma Index ──────────────────────────────────────────────────────────────

from modules.gamma_exposure import compute_gamma_index


def test_gamma_index_returns_expected_keys(options_dfs):
    calls, puts, spot = options_dfs
    gex_df = compute_gex(calls, puts, spot)
    gi = compute_gamma_index(gex_df, spot)
    expected = {
        "gamma_index", "gamma_condition", "call_wall", "put_wall",
        "gamma_flip", "gamma_tilt", "gamma_concentration", "top_strikes",
    }
    assert expected == set(gi.keys())


def test_gamma_index_empty_input():
    gi = compute_gamma_index(pd.DataFrame(), 400.0)
    assert gi == {}


def test_gamma_index_zero_spot(options_dfs):
    calls, puts, _ = options_dfs
    gex_df = compute_gex(calls, puts, 400.0)
    gi = compute_gamma_index(gex_df, 0)
    assert gi == {}


def test_gamma_index_condition_positive():
    """Positive total net GEX should give stabilizing condition."""
    gex_df = pd.DataFrame({
        "strike": [400.0, 410.0],
        "call_gex": [1e9, 2e9],
        "put_gex": [-0.5e9, -0.5e9],
        "net_gex": [0.5e9, 1.5e9],
        "call_gex_b": [1.0, 2.0],
        "put_gex_b": [-0.5, -0.5],
        "net_gex_b": [0.5, 1.5],
    })
    gi = compute_gamma_index(gex_df, 405.0)
    assert gi["gamma_index"] > 0
    assert "Stabilizing" in gi["gamma_condition"]


def test_gamma_index_condition_negative():
    """Negative total net GEX should give destabilizing condition."""
    gex_df = pd.DataFrame({
        "strike": [400.0, 410.0],
        "call_gex": [0.5e9, 0.5e9],
        "put_gex": [-2e9, -3e9],
        "net_gex": [-1.5e9, -2.5e9],
        "call_gex_b": [0.5, 0.5],
        "put_gex_b": [-2.0, -3.0],
        "net_gex_b": [-1.5, -2.5],
    })
    gi = compute_gamma_index(gex_df, 405.0)
    assert gi["gamma_index"] < 0
    assert "Destabilizing" in gi["gamma_condition"]


def test_gamma_index_call_wall_is_peak_call_gex():
    """Call wall should be the strike with the highest call GEX."""
    gex_df = pd.DataFrame({
        "strike": [390.0, 400.0, 410.0],
        "call_gex": [1e9, 5e9, 2e9],
        "put_gex": [-1e9, -1e9, -1e9],
        "net_gex": [0, 4e9, 1e9],
        "call_gex_b": [1.0, 5.0, 2.0],
        "put_gex_b": [-1.0, -1.0, -1.0],
        "net_gex_b": [0.0, 4.0, 1.0],
    })
    gi = compute_gamma_index(gex_df, 400.0)
    assert gi["call_wall"] == 400.0


def test_gamma_index_put_wall_is_most_negative_put_gex():
    """Put wall should be the strike with the most negative put GEX."""
    gex_df = pd.DataFrame({
        "strike": [390.0, 400.0, 410.0],
        "call_gex": [1e9, 1e9, 1e9],
        "put_gex": [-1e9, -5e9, -2e9],
        "net_gex": [0, -4e9, -1e9],
        "call_gex_b": [1.0, 1.0, 1.0],
        "put_gex_b": [-1.0, -5.0, -2.0],
        "net_gex_b": [0.0, -4.0, -1.0],
    })
    gi = compute_gamma_index(gex_df, 400.0)
    assert gi["put_wall"] == 400.0


def test_gamma_tilt_bounds(options_dfs):
    """Gamma tilt should be between 0 and 1."""
    calls, puts, spot = options_dfs
    gex_df = compute_gex(calls, puts, spot)
    gi = compute_gamma_index(gex_df, spot)
    assert 0.0 <= gi["gamma_tilt"] <= 1.0


def test_gamma_concentration_bounds(options_dfs):
    """Gamma concentration should be between 0 and 1."""
    calls, puts, spot = options_dfs
    gex_df = compute_gex(calls, puts, spot)
    gi = compute_gamma_index(gex_df, spot)
    assert 0.0 <= gi["gamma_concentration"] <= 1.0


def test_gamma_index_top_strikes_max_five(options_dfs):
    """Top strikes list should have at most 5 entries."""
    calls, puts, spot = options_dfs
    gex_df = compute_gex(calls, puts, spot)
    gi = compute_gamma_index(gex_df, spot)
    assert len(gi["top_strikes"]) <= 5
    for s in gi["top_strikes"]:
        assert "strike" in s and "net_gex_b" in s


# ── Gamma Index History & Timeline ──────────────────────────────────────────

from modules.gamma_exposure import (
    save_gamma_index_snapshot,
    load_gamma_index_history,
    plot_gamma_index_timeline,
    _HISTORY_FILE,
)
import json
import os


def test_save_and_load_gamma_index_history(tmp_path, monkeypatch):
    """Saving a snapshot should be retrievable via load_gamma_index_history."""
    hist_file = str(tmp_path / "gamma_index_history.json")
    monkeypatch.setattr("modules.gamma_exposure._HISTORY_FILE", hist_file)
    monkeypatch.setattr("modules.gamma_exposure._HISTORY_DIR", str(tmp_path))

    gi = {"gamma_index": 1.234, "gamma_condition": "Positive (Stabilizing)",
          "call_wall": 410.0, "put_wall": 390.0, "gamma_flip": 400.0,
          "gamma_tilt": 0.6, "gamma_concentration": 0.45, "top_strikes": []}

    save_gamma_index_snapshot("SPY", gi, 405.0)
    df = load_gamma_index_history("SPY")
    assert len(df) == 1
    assert df["gamma_index"].iloc[0] == 1.234
    assert df["spot"].iloc[0] == 405.0


def test_save_snapshot_overwrites_same_day(tmp_path, monkeypatch):
    """Saving twice on the same day should keep only one entry."""
    hist_file = str(tmp_path / "gamma_index_history.json")
    monkeypatch.setattr("modules.gamma_exposure._HISTORY_FILE", hist_file)
    monkeypatch.setattr("modules.gamma_exposure._HISTORY_DIR", str(tmp_path))

    gi1 = {"gamma_index": 1.0, "gamma_condition": "Positive (Stabilizing)",
           "call_wall": None, "put_wall": None, "gamma_flip": None,
           "gamma_tilt": 0.5, "gamma_concentration": 0.3, "top_strikes": []}
    gi2 = {"gamma_index": 2.0, "gamma_condition": "Positive (Stabilizing)",
           "call_wall": None, "put_wall": None, "gamma_flip": None,
           "gamma_tilt": 0.5, "gamma_concentration": 0.3, "top_strikes": []}

    save_gamma_index_snapshot("SPY", gi1, 400.0)
    save_gamma_index_snapshot("SPY", gi2, 401.0)
    df = load_gamma_index_history("SPY")
    assert len(df) == 1
    assert df["gamma_index"].iloc[0] == 2.0  # second value wins


def test_load_history_empty_for_unknown_ticker(tmp_path, monkeypatch):
    hist_file = str(tmp_path / "gamma_index_history.json")
    monkeypatch.setattr("modules.gamma_exposure._HISTORY_FILE", hist_file)
    df = load_gamma_index_history("AAPL")
    assert df.empty


def test_save_snapshot_skips_empty_gamma_idx(tmp_path, monkeypatch):
    """Passing an empty dict should not write anything."""
    hist_file = str(tmp_path / "gamma_index_history.json")
    monkeypatch.setattr("modules.gamma_exposure._HISTORY_FILE", hist_file)
    monkeypatch.setattr("modules.gamma_exposure._HISTORY_DIR", str(tmp_path))
    save_gamma_index_snapshot("SPY", {}, 400.0)
    assert not os.path.exists(hist_file)


def test_plot_gamma_index_timeline_empty():
    """Timeline chart with no history should still return a valid Figure."""
    fig = plot_gamma_index_timeline("AAPL")  # no history for AAPL
    assert isinstance(fig, go.Figure)


def test_plot_gamma_index_timeline_with_data(tmp_path, monkeypatch):
    """Timeline chart should render with historical data."""
    hist_file = str(tmp_path / "gamma_index_history.json")
    monkeypatch.setattr("modules.gamma_exposure._HISTORY_FILE", hist_file)
    # Write some fake history
    history = [
        {"date": "2026-03-25", "ticker": "SPY", "spot": 640, "gamma_index": 1.5,
         "gamma_condition": "Positive (Stabilizing)", "call_wall": 660,
         "put_wall": 620, "gamma_flip": 645, "gamma_tilt": 0.55,
         "gamma_concentration": 0.4},
        {"date": "2026-03-26", "ticker": "SPY", "spot": 645, "gamma_index": -0.5,
         "gamma_condition": "Negative (Destabilizing)", "call_wall": 660,
         "put_wall": 620, "gamma_flip": 640, "gamma_tilt": 0.45,
         "gamma_concentration": 0.35},
    ]
    with open(hist_file, "w") as f:
        json.dump(history, f)

    fig = plot_gamma_index_timeline("SPY")
    assert isinstance(fig, go.Figure)
    # Should have real traces: positive fill, negative fill, main line
    assert len(fig.data) >= 3


def test_plot_gamma_index_timeline_with_proxy(tmp_path, monkeypatch):
    """Timeline chart should render with proxy data."""
    hist_file = str(tmp_path / "gamma_index_history.json")
    monkeypatch.setattr("modules.gamma_exposure._HISTORY_FILE", hist_file)

    proxy = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=60, freq="B"),
        "gamma_proxy": np.sin(np.linspace(0, 4 * np.pi, 60)) * 0.5,
        "vix": np.random.uniform(15, 25, 60),
        "realized_vol": np.random.uniform(12, 22, 60),
        "proxy_condition": ["Positive (Stabilizing)"] * 30 + ["Negative (Destabilizing)"] * 30,
    })
    fig = plot_gamma_index_timeline("SPY", proxy_df=proxy)
    assert isinstance(fig, go.Figure)
    # Should have proxy traces: pos fill, neg fill, proxy line
    assert len(fig.data) >= 3


# ── Historical Gamma Proxy ─────────────────────────────────────────────────

from modules.gamma_exposure import compute_historical_gamma_proxy, calibrate_proxy_to_real


def test_proxy_returns_expected_columns():
    """Proxy should return date, gamma_proxy, vix, realized_vol, proxy_condition."""
    dates = pd.bdate_range("2025-01-01", periods=100)
    spy_df = pd.DataFrame({"Close": 500 + np.cumsum(np.random.randn(100))}, index=dates)
    vix_df = pd.DataFrame({"Close": 18 + np.random.randn(100) * 3}, index=dates)
    result = compute_historical_gamma_proxy(spy_df, vix_df)
    assert not result.empty
    assert set(result.columns) >= {"date", "gamma_proxy", "vix", "realized_vol", "proxy_condition"}


def test_proxy_positive_in_low_vix():
    """Proxy should tend positive when VIX is consistently low."""
    dates = pd.bdate_range("2025-01-01", periods=100)
    spy_df = pd.DataFrame({"Close": np.linspace(500, 520, 100)}, index=dates)
    # Very low, stable VIX
    vix_df = pd.DataFrame({"Close": np.full(100, 13.0)}, index=dates)
    result = compute_historical_gamma_proxy(spy_df, vix_df)
    # Most recent values should be near zero or positive (stable environment)
    if not result.empty:
        tail_mean = result["gamma_proxy"].tail(20).mean()
        # Low stable VIX → should not be strongly negative
        assert tail_mean > -1.0


def test_proxy_negative_in_vix_spike():
    """Proxy should go negative when VIX spikes."""
    dates = pd.bdate_range("2025-01-01", periods=100)
    spy_df = pd.DataFrame({"Close": 500 - np.linspace(0, 50, 100)}, index=dates)
    vix = np.full(100, 15.0)
    vix[70:] = 35.0  # VIX spike
    vix_df = pd.DataFrame({"Close": vix}, index=dates)
    result = compute_historical_gamma_proxy(spy_df, vix_df)
    if not result.empty:
        spike_mean = result["gamma_proxy"].tail(10).mean()
        assert spike_mean < 0, "Proxy should be negative during VIX spike"


def test_proxy_empty_inputs():
    """Proxy should return empty DataFrame for empty inputs."""
    assert compute_historical_gamma_proxy(pd.DataFrame(), pd.DataFrame()).empty
    assert compute_historical_gamma_proxy(None, None).empty


def test_proxy_short_data():
    """Proxy should handle data shorter than warmup period."""
    dates = pd.bdate_range("2025-01-01", periods=10)
    spy_df = pd.DataFrame({"Close": range(10)}, index=dates)
    vix_df = pd.DataFrame({"Close": range(10)}, index=dates)
    result = compute_historical_gamma_proxy(spy_df, vix_df)
    assert result.empty  # Too short for 20-day rolling


def test_calibrate_proxy_scales_to_real():
    """Calibration should scale proxy to match real data magnitude."""
    proxy = pd.DataFrame({
        "date": pd.date_range("2026-03-25", periods=3),
        "gamma_proxy": [10.0, -10.0, 5.0],
    })
    real = pd.DataFrame({
        "date": pd.to_datetime(["2026-03-25", "2026-03-26"]),
        "gamma_index": [1.0, -1.0],
    })
    result = calibrate_proxy_to_real(proxy, real)
    # Proxy was 10x the real data, so should be scaled down ~10x
    assert abs(result["gamma_proxy"].iloc[0]) < 3.0  # rough check


def test_calibrate_no_overlap():
    """Calibration without overlap should normalize to ~0.5B std."""
    proxy = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=100),
        "gamma_proxy": np.random.randn(100) * 5,
    })
    real = pd.DataFrame()  # no real data
    result = calibrate_proxy_to_real(proxy, real)
    assert abs(result["gamma_proxy"].std() - 0.5) < 0.1
