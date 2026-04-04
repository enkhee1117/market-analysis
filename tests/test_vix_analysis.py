"""
Unit tests for modules/vix_analysis.py

Covers:
- derived VIX metrics
- term-structure snapshot generation
- VIX Central-style term-structure curve rendering
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modules.vix_analysis import (
    compute_vix_metrics,
    compute_vix_term_structure_snapshot,
    plot_vix_term_structure_curve,
)


def test_compute_vix_metrics_adds_expected_columns():
    dates = pd.bdate_range("2026-01-01", periods=30)
    df = pd.DataFrame({
        "VIX": np.linspace(14, 22, 30),
        "VVIX": np.linspace(80, 110, 30),
    }, index=dates)
    result = compute_vix_metrics(df)
    assert {"VVIX_VIX_Ratio", "VIX_ZScore_20d", "VIX_Chg_1d", "VIX_Chg_5d"}.issubset(result.columns)


def test_compute_vix_term_structure_snapshot_detects_contango():
    dates = pd.bdate_range("2026-03-30", periods=2)
    df = pd.DataFrame({
        "VIX9D": [15.0, 14.0],
        "VIX": [16.0, 15.0],
        "VIX3M": [18.0, 17.5],
        "VIX6M": [19.0, 18.5],
        "VIX1Y": [20.0, 19.5],
    }, index=dates)
    snapshot, summary = compute_vix_term_structure_snapshot(df)
    assert not snapshot.empty
    assert summary["regime"] == "Contango"
    assert summary["slope_1m_3m"] > 0
    assert list(snapshot["Tenor"]) == ["9D", "1M", "3M", "6M", "1Y"]


def test_compute_vix_term_structure_snapshot_detects_backwardation():
    dates = pd.bdate_range("2026-03-30", periods=2)
    df = pd.DataFrame({
        "VIX9D": [24.0, 28.0],
        "VIX": [22.0, 25.0],
        "VIX3M": [20.0, 21.0],
        "VIX6M": [19.0, 20.0],
        "VIX1Y": [18.0, 19.0],
    }, index=dates)
    _, summary = compute_vix_term_structure_snapshot(df)
    assert summary["regime"] == "Backwardation"
    assert summary["inversion_count"] >= 1


def test_plot_vix_term_structure_curve_returns_figure():
    dates = pd.bdate_range("2026-03-30", periods=2)
    df = pd.DataFrame({
        "VIX9D": [15.0, 14.0],
        "VIX": [16.0, 15.0],
        "VIX3M": [18.0, 17.5],
        "VIX6M": [19.0, 18.5],
        "VIX1Y": [20.0, 19.5],
    }, index=dates)
    fig = plot_vix_term_structure_curve(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_plot_vix_term_structure_curve_empty_when_no_columns():
    fig = plot_vix_term_structure_curve(pd.DataFrame({"VIX": []}))
    assert isinstance(fig, go.Figure)
