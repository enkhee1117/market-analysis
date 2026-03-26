"""
Unit tests for modules/seasonality.py

Covers:
- compute_returns column generation and correctness
- monthly_seasonality: 12 rows, correct columns, win rate bounds
- monthly_return_pivot: shape and month-name columns
- weekly_seasonality and intramonth_seasonality: basic shape
- Plot functions return go.Figure
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pytest

from modules.seasonality import (
    compute_returns,
    monthly_seasonality,
    monthly_return_pivot,
    weekly_seasonality,
    intramonth_seasonality,
    plot_monthly_bar,
    plot_monthly_heatmap,
    plot_weekly_bar,
    plot_intramonth_bar,
    plot_annual_return_bar,
    MONTH_NAMES,
)


# ── MONTH_NAMES ───────────────────────────────────────────────────────────────

def test_month_names_has_12_entries():
    assert len(MONTH_NAMES) == 12


def test_month_names_are_strings():
    assert all(isinstance(m, str) for m in MONTH_NAMES)


# ── compute_returns ───────────────────────────────────────────────────────────

def test_compute_returns_adds_required_columns(ohlcv_df):
    result = compute_returns(ohlcv_df)
    required = {"Return", "Year", "Month", "MonthName", "Week", "DayOfMonth", "DayOfWeek"}
    assert required.issubset(result.columns)


def test_compute_returns_no_nan_in_return(ohlcv_df):
    result = compute_returns(ohlcv_df)
    assert result["Return"].isna().sum() == 0


def test_compute_returns_month_range(ohlcv_df):
    result = compute_returns(ohlcv_df)
    assert result["Month"].between(1, 12).all()


def test_compute_returns_day_of_month_range(ohlcv_df):
    result = compute_returns(ohlcv_df)
    assert result["DayOfMonth"].between(1, 31).all()


def test_compute_returns_month_name_matches_month_number(ohlcv_df):
    result = compute_returns(ohlcv_df)
    for _, row in result.iterrows():
        assert row["MonthName"] == MONTH_NAMES[row["Month"] - 1]


def test_compute_returns_formula(ohlcv_df):
    result = compute_returns(ohlcv_df)
    expected = ohlcv_df["Close"].pct_change() * 100
    pd.testing.assert_series_equal(
        result["Return"], expected.loc[result.index],
        check_names=False, rtol=1e-9,
    )


# ── monthly_seasonality ───────────────────────────────────────────────────────

def test_monthly_seasonality_has_12_rows(ohlcv_df):
    df = compute_returns(ohlcv_df)
    result = monthly_seasonality(df)
    assert len(result) == 12


def test_monthly_seasonality_columns(ohlcv_df):
    df = compute_returns(ohlcv_df)
    result = monthly_seasonality(df)
    required = {"Month", "MonthName", "Mean %", "Win Rate %", "p-value", "Significant", "Count"}
    assert required.issubset(result.columns)


def test_monthly_seasonality_win_rate_bounds(ohlcv_df):
    df = compute_returns(ohlcv_df)
    result = monthly_seasonality(df)
    assert (result["Win Rate %"] >= 0).all()
    assert (result["Win Rate %"] <= 100).all()


def test_monthly_seasonality_pvalue_bounds(ohlcv_df):
    df = compute_returns(ohlcv_df)
    result = monthly_seasonality(df)
    assert (result["p-value"] >= 0).all()
    assert (result["p-value"] <= 1).all()


def test_monthly_seasonality_significant_matches_pvalue(ohlcv_df):
    df = compute_returns(ohlcv_df)
    result = monthly_seasonality(df)
    for _, row in result.iterrows():
        assert row["Significant"] == (row["p-value"] < 0.05)


# ── monthly_return_pivot ──────────────────────────────────────────────────────

def test_monthly_return_pivot_has_12_month_columns(ohlcv_df):
    df = compute_returns(ohlcv_df)
    pivot = monthly_return_pivot(df)
    assert pivot.shape[1] == 12


def test_monthly_return_pivot_columns_are_month_names(ohlcv_df):
    df = compute_returns(ohlcv_df)
    pivot = monthly_return_pivot(df)
    assert list(pivot.columns) == MONTH_NAMES


def test_monthly_return_pivot_index_is_years(ohlcv_df):
    df = compute_returns(ohlcv_df)
    pivot = monthly_return_pivot(df)
    assert pivot.index.name == "Year" or all(isinstance(y, (int, np.integer)) for y in pivot.index)


# ── weekly / intramonth seasonality ──────────────────────────────────────────

def test_weekly_seasonality_returns_dataframe(ohlcv_df):
    df = compute_returns(ohlcv_df)
    result = weekly_seasonality(df)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "Mean %" in result.columns or "Mean%" in result.columns


def test_intramonth_seasonality_returns_dataframe(ohlcv_df):
    df = compute_returns(ohlcv_df)
    result = intramonth_seasonality(df)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


# ── plot functions ────────────────────────────────────────────────────────────

def test_plot_monthly_bar_returns_figure(ohlcv_df):
    df = compute_returns(ohlcv_df)
    monthly = monthly_seasonality(df)
    fig = plot_monthly_bar(monthly)
    assert isinstance(fig, go.Figure)


def test_plot_monthly_heatmap_returns_figure(ohlcv_df):
    df = compute_returns(ohlcv_df)
    pivot = monthly_return_pivot(df)
    fig = plot_monthly_heatmap(pivot)
    assert isinstance(fig, go.Figure)


def test_plot_annual_return_bar_returns_figure(ohlcv_df):
    df = compute_returns(ohlcv_df)
    fig = plot_annual_return_bar(df)
    assert isinstance(fig, go.Figure)
