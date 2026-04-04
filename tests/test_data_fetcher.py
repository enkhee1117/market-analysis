"""
Unit tests for modules/data_fetcher.py

yfinance network calls are fully mocked — these tests run offline.
Covers:
- MultiIndex column flattening (new yfinance behaviour)
- Timezone stripping
- fetch_multi_tickers returns only Close columns
- Empty DataFrame passthrough
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timezone
from unittest.mock import patch

from modules.data_fetcher import (
    fetch_price_history,
    fetch_multi_tickers,
    get_refresh_bucket,
    clear_today_cache,
    TIMEFRAME_MAP,
)


# ── TIMEFRAME_MAP ─────────────────────────────────────────────────────────────

def test_timeframe_map_has_expected_keys():
    expected = {"1 Month", "3 Months", "6 Months", "12 Months", "3 Years", "5 Years", "10 Years"}
    assert set(TIMEFRAME_MAP.keys()) == expected


def test_timeframe_map_values_are_tuples_of_strings():
    for label, (period, interval) in TIMEFRAME_MAP.items():
        assert isinstance(period, str), f"{label}: period must be str"
        assert isinstance(interval, str), f"{label}: interval must be str"


def test_get_refresh_bucket_daily_when_market_closed():
    fixed = datetime(2026, 4, 3, 18, 42)
    with patch("modules.data_fetcher._is_market_open_now", return_value=False):
        bucket = get_refresh_bucket("options", now=fixed)
    assert bucket == "20260403"


def test_get_refresh_bucket_rounds_intraday_options_to_5m():
    fixed = datetime(2026, 4, 3, 14, 7, tzinfo=timezone.utc)
    with patch("modules.data_fetcher._is_market_open_now", return_value=True):
        bucket = get_refresh_bucket("options", now=fixed)
    assert bucket == "202604031405"


# ── fetch_price_history ───────────────────────────────────────────────────────

def test_returns_dataframe_with_ohlcv_columns(ohlcv_df):
    with patch("yfinance.download", return_value=ohlcv_df):
        result = fetch_price_history.__wrapped__("SPY", period="1y")
    assert not result.empty
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in result.columns


def test_flattens_multiindex_columns(ohlcv_df):
    """yfinance sometimes returns MultiIndex columns — must be flattened."""
    multi = ohlcv_df.copy()
    multi.columns = pd.MultiIndex.from_tuples([(c, "SPY") for c in ohlcv_df.columns])
    with patch("yfinance.download", return_value=multi):
        result = fetch_price_history.__wrapped__("SPY", period="1y")
    assert not isinstance(result.columns, pd.MultiIndex), "MultiIndex was not flattened"
    assert "Close" in result.columns


def test_drops_timezone_from_index(ohlcv_df):
    tz_df = ohlcv_df.copy()
    tz_df.index = tz_df.index.tz_localize("America/New_York")
    with patch("yfinance.download", return_value=tz_df):
        result = fetch_price_history.__wrapped__("SPY", period="1y")
    assert result.index.tz is None, "Timezone was not stripped from index"


def test_returns_empty_df_when_yfinance_returns_empty(ohlcv_df):
    empty = pd.DataFrame()
    with patch("yfinance.download", return_value=empty):
        result = fetch_price_history.__wrapped__("INVALID", period="1y")
    assert result.empty


def test_index_is_datetime(ohlcv_df):
    with patch("yfinance.download", return_value=ohlcv_df):
        result = fetch_price_history.__wrapped__("SPY", period="1y")
    assert isinstance(result.index, pd.DatetimeIndex)


# ── fetch_multi_tickers ───────────────────────────────────────────────────────

def test_multi_tickers_returns_only_close_columns(ohlcv_df):
    with patch("modules.data_fetcher.fetch_price_history", return_value=ohlcv_df):
        result = fetch_multi_tickers.__wrapped__(["SPY", "QQQ"])
    assert set(result.columns) == {"SPY", "QQQ"}, "Should contain only ticker names as columns"


def test_multi_tickers_returns_empty_when_all_fail():
    empty = pd.DataFrame()
    with patch("modules.data_fetcher.fetch_price_history", return_value=empty):
        result = fetch_multi_tickers.__wrapped__(["INVALID1", "INVALID2"])
    assert result.empty


def test_clear_today_cache_can_scope_by_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.data_fetcher.CACHE_DIR", str(tmp_path))
    today = pd.Timestamp.today().date().isoformat()
    keep = tmp_path / f"QQQ_price_1y_1d_{today}.parquet"
    delete = tmp_path / f"SPY_price_1y_1d_{today}.parquet"
    keep.write_text("keep")
    delete.write_text("delete")

    clear_today_cache(prefix="SPY_price_")

    assert keep.exists()
    assert not delete.exists()
