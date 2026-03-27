"""
Smoke tests — verify every public name imported by app.py actually exists.

These are the HIGHEST PRIORITY tests. They catch "cannot import name X"
errors before a single line of app code runs, preventing deployment failures.
"""


def test_day_of_week_exports():
    from modules.day_of_week import (
        compute_dow_returns,
        filter_by_timeframe,
        dow_summary,
        plot_dow_comparison,
        plot_win_rate_comparison,
        plot_dow_distribution,
        plot_cumulative_by_dow,
        TIMEFRAMES,
        RETURN_TYPES,
        DAY_NAMES,
        DAY_COLORS,
    )
    assert isinstance(TIMEFRAMES, dict)
    assert isinstance(RETURN_TYPES, dict)
    assert len(RETURN_TYPES) == 3, "Expected Close-to-Close, Intraday, Overnight"


def test_data_fetcher_exports():
    from modules.data_fetcher import (
        fetch_price_history,
        fetch_multi_tickers,
        fetch_options_chain,
        TIMEFRAME_MAP,
    )
    assert isinstance(TIMEFRAME_MAP, dict)


def test_gamma_exposure_exports():
    from modules.gamma_exposure import (
        compute_gex,
        gex_flip_point,
        total_gex_metrics,
        compute_gamma_index,
        plot_gex_profile,
        plot_gex_by_expiration,
    )


def test_vix_analysis_exports():
    from modules.vix_analysis import (
        compute_vix_metrics,
        plot_vix_panel,
        plot_vvix_vix_ratio,
        plot_vix_zscore,
        vix_summary_stats,
        VIX_REGIMES,
        TICKERS,
    )
    assert isinstance(VIX_REGIMES, list)
    assert isinstance(TICKERS, dict)
    assert "VIX" in TICKERS and "VVIX" in TICKERS and "SVIX" in TICKERS


def test_seasonality_exports():
    from modules.seasonality import (
        compute_returns,
        monthly_seasonality,
        monthly_return_pivot,
        weekly_seasonality,
        intramonth_seasonality,
        plot_monthly_bar,
        plot_monthly_heatmap,
        MONTH_NAMES,
    )
    assert len(MONTH_NAMES) == 12


def test_return_types_values_match_dataframe_columns():
    """RETURN_TYPES values must match actual column names produced by compute_dow_returns."""
    from modules.day_of_week import RETURN_TYPES
    expected_cols = {"Return", "Intraday", "Overnight"}
    assert set(RETURN_TYPES.values()) == expected_cols
