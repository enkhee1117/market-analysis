import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import glob
import time
from datetime import datetime, timedelta, date
import streamlit as st


TIMEFRAME_MAP = {
    "1 Month":   ("1mo",  "1d"),
    "3 Months":  ("3mo",  "1d"),
    "6 Months":  ("6mo",  "1d"),
    "12 Months": ("1y",   "1d"),
    "3 Years":   ("3y",   "1d"),
    "5 Years":   ("5y",   "1wk"),
    "10 Years":  ("10y",  "1wk"),
}

# ── Daily file cache ─────────────────────────────────────────────────────────
# Data is fetched from Yahoo Finance once per day and stored as parquet files.
# Subsequent requests read from disk, eliminating rate-limit issues.

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")


def _cache_path(key: str, ext: str = "parquet") -> str:
    """Return path like cache/SPY_price_10y_1d_2026-03-26.parquet"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}_{date.today().isoformat()}.{ext}")


def _read_cache(key: str) -> pd.DataFrame | None:
    """Read today's cached parquet file if it exists."""
    path = _cache_path(key)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


def _write_cache(key: str, df: pd.DataFrame) -> None:
    """Write DataFrame to today's cache file."""
    if df.empty:
        return
    try:
        path = _cache_path(key)
        df.to_parquet(path)
    except Exception:
        pass  # Cache write failure is non-fatal


def clear_today_cache():
    """Delete all of today's cache files (used by Force Refresh button)."""
    if not os.path.exists(CACHE_DIR):
        return
    today = date.today().isoformat()
    for f in glob.glob(os.path.join(CACHE_DIR, f"*_{today}.*")):
        try:
            os.remove(f)
        except Exception:
            pass


# ── Data fetching functions ──────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def fetch_price_history(ticker: str, period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV price history for a ticker. Cached to disk daily."""
    cache_key = f"{ticker}_price_{period}_{interval}"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    _write_cache(cache_key, df)
    return df


@st.cache_data(ttl=86400)
def fetch_multi_tickers(tickers: list, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch closing prices for multiple tickers. Cached to disk daily."""
    tickers_key = "_".join(sorted(tickers)).replace("^", "")
    cache_key = f"{tickers_key}_multi_{period}_{interval}"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    data = {}
    for t in tickers:
        df = fetch_price_history(t, period=period, interval=interval)
        if not df.empty and "Close" in df.columns:
            data[t] = df["Close"]
    if data:
        result = pd.DataFrame(data).dropna(how="all")
        _write_cache(cache_key, result)
        return result
    return pd.DataFrame()


@st.cache_data(ttl=86400)
def fetch_options_chain(ticker: str):
    """Fetch all options chains for nearest expirations (up to 8). Cached to disk daily."""
    # Check for cached data first
    calls_cache_key = f"{ticker}_options_calls"
    puts_cache_key = f"{ticker}_options_puts"
    spot_path = _cache_path(f"{ticker}_options_spot", ext="json")

    cached_calls = _read_cache(calls_cache_key)
    cached_puts = _read_cache(puts_cache_key)
    cached_spot = None
    if os.path.exists(spot_path):
        try:
            with open(spot_path) as f:
                cached_spot = json.load(f).get("spot")
        except Exception:
            pass

    if cached_calls is not None and cached_puts is not None and cached_spot is not None:
        return cached_calls, cached_puts, cached_spot

    # Fetch fresh from Yahoo Finance
    tk = yf.Ticker(ticker)
    try:
        expirations = tk.options
    except Exception:
        return None, None, None

    if not expirations:
        return None, None, None

    spot = None
    try:
        spot = tk.fast_info.get("lastPrice") or tk.fast_info.get("regularMarketPrice")
    except Exception:
        pass
    if spot is None or spot == 0:
        try:
            hist = tk.history(period="1d")
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else None
        except Exception:
            pass

    all_calls = []
    all_puts = []
    for exp in expirations[:8]:
        try:
            chain = tk.option_chain(exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls["expiration"] = exp
            puts["expiration"] = exp
            all_calls.append(calls)
            all_puts.append(puts)
            time.sleep(0.2)  # Rate-limit protection
        except Exception:
            continue

    calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
    puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

    # Write to cache
    _write_cache(calls_cache_key, calls_df)
    _write_cache(puts_cache_key, puts_df)
    if spot is not None:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(spot_path, "w") as f:
                json.dump({"spot": spot}, f)
        except Exception:
            pass

    return calls_df, puts_df, spot
