"""
Smoke tests — verify every public name imported by app.py actually exists.

These are the HIGHEST PRIORITY tests. They catch "cannot import name X"
errors before a single line of app code runs, preventing deployment failures.

>>> RULE: When you add a new import to app.py, you MUST add it here too. <<<
"""


def test_data_fetcher_exports():
    """Must match: from modules.data_fetcher import ... in app.py"""
    from modules.data_fetcher import (
        fetch_price_history,
        fetch_multi_tickers,
        fetch_options_chain,
        load_historical_options_chain,
        clear_today_cache,
        get_refresh_bucket,
        TIMEFRAME_MAP,
    )
    assert isinstance(TIMEFRAME_MAP, dict)
    assert callable(clear_today_cache)
    assert callable(get_refresh_bucket)
    assert callable(load_historical_options_chain)


def test_supabase_cache_exports():
    """Must match: from modules.supabase_cache import ... in app.py"""
    from modules.supabase_cache import (
        get_data_freshness,
        data_staleness_info,
        is_market_open,
    )
    assert callable(get_data_freshness)
    assert callable(data_staleness_info)
    assert callable(is_market_open)


def test_day_of_week_exports():
    """Must match: from modules.day_of_week import ... in app.py"""
    from modules.day_of_week import (
        compute_dow_returns,
        filter_by_timeframe,
        dow_summary,
        compute_conditional_probabilities,
        compute_conditional_chain,
        plot_dow_comparison,
        plot_win_rate_comparison,
        plot_dow_distribution,
        plot_cumulative_by_dow,
        build_conditional_table,
        plot_conditional_distribution,
        TIMEFRAMES,
        RETURN_TYPES,
    )
    assert isinstance(TIMEFRAMES, dict)
    assert isinstance(RETURN_TYPES, dict)
    assert len(RETURN_TYPES) == 3, "Expected Close-to-Close, Intraday, Overnight"
    assert callable(compute_conditional_probabilities)
    assert callable(compute_conditional_chain)
    assert callable(build_conditional_table)
    assert callable(plot_conditional_distribution)


def test_gamma_exposure_exports():
    """Must match: from modules.gamma_exposure import ... in app.py"""
    from modules.gamma_exposure import (
        compute_gex,
        gex_flip_point,
        total_gex_metrics,
        compute_gamma_index,
        compute_dex,
        total_dex_metrics,
        compute_iv_skew,
        compute_atm_iv_term_structure,
        filter_options_chain,
        summarize_chain_quality,
        aggregate_gex_by_expiration,
        save_gamma_index_snapshot,
        load_gamma_index_history,
        plot_gex_profile,
        plot_gex_by_expiration,
        plot_gamma_index_timeline,
        plot_dex_profile,
        plot_dex_by_expiration,
        plot_iv_skew,
        plot_atm_iv_term_structure,
        plot_atm_iv_term_structure_comparison,
    )
    # Verify all are callable
    for fn in (compute_gex, gex_flip_point, total_gex_metrics, compute_gamma_index,
               compute_dex, total_dex_metrics,
               compute_iv_skew, compute_atm_iv_term_structure,
               filter_options_chain, summarize_chain_quality,
               aggregate_gex_by_expiration, save_gamma_index_snapshot, load_gamma_index_history,
               plot_gex_profile, plot_gex_by_expiration,
               plot_gamma_index_timeline, plot_dex_profile, plot_dex_by_expiration,
               plot_iv_skew, plot_atm_iv_term_structure, plot_atm_iv_term_structure_comparison):
        assert callable(fn), f"{fn.__name__} is not callable"


def test_vix_analysis_exports():
    """Must match: from modules.vix_analysis import ... in app.py"""
    from modules.vix_analysis import (
        compute_vix_metrics,
        compute_vix_term_structure_snapshot,
        compute_vix_forward_returns,
        compute_vix_beta,
        plot_vvix_vix_ratio,
        plot_vix_zscore,
        plot_vix_term_structure_curve,
        plot_vix_forward_returns_bar,
        plot_vix_forward_returns_box,
        plot_vix_forward_win_rates,
        plot_vix_beta,
        vix_summary_stats,
    )
    assert callable(compute_vix_metrics)
    assert callable(compute_vix_term_structure_snapshot)
    assert callable(compute_vix_forward_returns)
    assert callable(compute_vix_beta)
    assert callable(plot_vvix_vix_ratio)
    assert callable(plot_vix_zscore)
    assert callable(plot_vix_term_structure_curve)
    assert callable(plot_vix_forward_returns_bar)
    assert callable(plot_vix_forward_returns_box)
    assert callable(plot_vix_forward_win_rates)
    assert callable(plot_vix_beta)
    assert callable(vix_summary_stats)


def test_seasonality_exports():
    """Must match: from modules.seasonality import ... in app.py"""
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
    )
    assert callable(plot_weekly_bar)
    assert callable(plot_intramonth_bar)
    assert callable(plot_annual_return_bar)


def test_return_types_values_match_dataframe_columns():
    """RETURN_TYPES values must match actual column names produced by compute_dow_returns."""
    from modules.day_of_week import RETURN_TYPES
    expected_cols = {"Return", "Intraday", "Overnight"}
    assert set(RETURN_TYPES.values()) == expected_cols
