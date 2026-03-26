"""
Unit tests for modules/day_of_week.py

Covers:
- Return column computation (close-to-close, intraday, overnight)
- Mathematical correctness of each formula
- dow_summary stats for all three return types
- filter_by_timeframe range slicing
- All four plot functions return go.Figure without errors
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pytest

from modules.day_of_week import (
    compute_dow_returns,
    filter_by_timeframe,
    dow_summary,
    plot_dow_comparison,
    plot_win_rate_comparison,
    plot_dow_distribution,
    plot_cumulative_by_dow,
    RETURN_TYPES,
    TIMEFRAMES,
)


# ── compute_dow_returns ───────────────────────────────────────────────────────

def test_required_columns_present(ohlcv_df):
    result = compute_dow_returns(ohlcv_df)
    expected = {"Return", "Intraday", "Overnight", "DayName", "DayOfWeek", "Close", "Open"}
    assert expected.issubset(result.columns)


def test_close_to_close_formula(ohlcv_df):
    result = compute_dow_returns(ohlcv_df)
    expected = ohlcv_df["Close"].pct_change() * 100
    pd.testing.assert_series_equal(
        result["Return"], expected.loc[result.index],
        check_names=False, rtol=1e-9,
    )


def test_intraday_formula(ohlcv_df):
    result = compute_dow_returns(ohlcv_df)
    expected = (ohlcv_df["Close"] - ohlcv_df["Open"]) / ohlcv_df["Open"] * 100
    pd.testing.assert_series_equal(
        result["Intraday"], expected.loc[result.index],
        check_names=False, rtol=1e-9,
    )


def test_overnight_formula(ohlcv_df):
    result = compute_dow_returns(ohlcv_df)
    prev_close = ohlcv_df["Close"].shift(1)
    expected = (ohlcv_df["Open"] - prev_close) / prev_close * 100
    pd.testing.assert_series_equal(
        result["Overnight"], expected.loc[result.index],
        check_names=False, rtol=1e-9,
    )


def test_return_plus_overnight_equals_close_to_close(ohlcv_df):
    """Overnight + Intraday should approximate Close-to-Close (within float rounding)."""
    result = compute_dow_returns(ohlcv_df)
    # (1 + overnight/100) * (1 + intraday/100) - 1 ≈ return/100
    reconstructed = ((1 + result["Overnight"] / 100) * (1 + result["Intraday"] / 100) - 1) * 100
    np.testing.assert_allclose(reconstructed.values, result["Return"].values, rtol=1e-6)


def test_no_nan_in_return_after_dropna(ohlcv_df):
    result = compute_dow_returns(ohlcv_df)
    assert result["Return"].isna().sum() == 0


def test_day_names_are_valid(ohlcv_df):
    result = compute_dow_returns(ohlcv_df)
    valid = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Other"}
    assert set(result["DayName"].unique()).issubset(valid)


def test_row_count_is_one_less_than_input(ohlcv_df):
    """First row is dropped because pct_change() creates NaN on row 0."""
    result = compute_dow_returns(ohlcv_df)
    assert len(result) == len(ohlcv_df) - 1


# ── filter_by_timeframe ───────────────────────────────────────────────────────

def test_filter_shorter_has_fewer_rows(ohlcv_df):
    df = compute_dow_returns(ohlcv_df)
    short = filter_by_timeframe(df, "1 Month")
    long_ = filter_by_timeframe(df, "12 Months")
    assert len(short) < len(long_)


def test_filter_all_dates_within_window(ohlcv_df):
    from datetime import timedelta
    df = compute_dow_returns(ohlcv_df)
    for label, days in TIMEFRAMES.items():
        sliced = filter_by_timeframe(df, label)
        if sliced.empty:
            continue
        cutoff = df.index.max() - timedelta(days=days)
        assert sliced.index.min() >= cutoff, f"{label}: dates outside window"


# ── dow_summary ───────────────────────────────────────────────────────────────

def test_summary_columns(ohlcv_df):
    df = compute_dow_returns(ohlcv_df)
    summary = dow_summary(df, ["Monday", "Friday"])
    expected_cols = {"Day", "Mean %", "Median %", "Win Rate", "Std %", "Count", "Best %", "Worst %"}
    assert expected_cols.issubset(summary.columns)


def test_summary_win_rate_between_0_and_100(ohlcv_df):
    df = compute_dow_returns(ohlcv_df)
    summary = dow_summary(df, ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    assert (summary["Win Rate"] >= 0).all() and (summary["Win Rate"] <= 100).all()


@pytest.mark.parametrize("col", ["Return", "Intraday", "Overnight"])
def test_summary_all_return_types(ohlcv_df, col):
    df = compute_dow_returns(ohlcv_df)
    summary = dow_summary(df, ["Monday", "Friday"], return_col=col)
    assert not summary.empty
    assert "Mean %" in summary.columns


def test_summary_empty_for_unknown_day(ohlcv_df):
    df = compute_dow_returns(ohlcv_df)
    summary = dow_summary(df, ["Sunday"])  # no Sundays in business day index
    assert summary.empty


# ── plot functions ────────────────────────────────────────────────────────────

DAYS = ["Monday", "Friday"]
TFS  = ["3 Months", "12 Months"]


def test_plot_dow_comparison_returns_figure(ohlcv_df):
    df = compute_dow_returns(ohlcv_df)
    fig = plot_dow_comparison(df, DAYS, TFS)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_plot_win_rate_comparison_returns_figure(ohlcv_df):
    df = compute_dow_returns(ohlcv_df)
    fig = plot_win_rate_comparison(df, DAYS, TFS)
    assert isinstance(fig, go.Figure)


def test_plot_dow_distribution_returns_figure(ohlcv_df):
    df = compute_dow_returns(ohlcv_df)
    fig = plot_dow_distribution(df, DAYS, "12 Months")
    assert isinstance(fig, go.Figure)


def test_plot_cumulative_returns_figure(ohlcv_df):
    df = compute_dow_returns(ohlcv_df)
    fig = plot_cumulative_by_dow(df, DAYS, "12 Months")
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("return_col", list(RETURN_TYPES.values()))
def test_all_plots_work_with_all_return_types(ohlcv_df, return_col):
    df = compute_dow_returns(ohlcv_df)
    assert isinstance(plot_dow_comparison(df, DAYS, TFS, return_col=return_col), go.Figure)
    assert isinstance(plot_win_rate_comparison(df, DAYS, TFS, return_col=return_col), go.Figure)
    assert isinstance(plot_dow_distribution(df, DAYS, "12 Months", return_col=return_col), go.Figure)
    assert isinstance(plot_cumulative_by_dow(df, DAYS, "12 Months", return_col=return_col), go.Figure)


def test_plots_handle_empty_data_gracefully():
    """Passing an empty DataFrame should not raise — returns empty Figure."""
    empty = pd.DataFrame(columns=["Return", "Intraday", "Overnight", "DayName", "DayOfWeek"])
    fig = plot_dow_comparison(empty, ["Monday"], ["1 Month"])
    assert isinstance(fig, go.Figure)
