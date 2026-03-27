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


# ── Massive.com API ──────────────────────────────────────────────────────────

MASSIVE_BASE_URL = "https://api.massive.com/v3"


def _get_massive_api_key() -> str | None:
    """Retrieve Massive API key from Streamlit secrets or environment."""
    try:
        return st.secrets["MASSIVE_API_KEY"]
    except Exception:
        return os.environ.get("MASSIVE_API_KEY")


def _massive_get(endpoint: str, params: dict | None = None) -> dict | None:
    """Make an authenticated GET request to Massive.com API."""
    import requests

    api_key = _get_massive_api_key()
    if not api_key:
        return None

    url = f"{MASSIVE_BASE_URL}{endpoint}"
    params = params or {}
    params["apiKey"] = api_key

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=86400)
def fetch_options_chain_massive(ticker: str):
    """
    Fetch full options chain from Massive.com API.
    Returns (calls_df, puts_df, spot) matching the format expected by compute_gex().

    Paginates through all results (250 per page).
    """
    # Check file cache first
    calls_cache_key = f"{ticker}_massive_calls"
    puts_cache_key = f"{ticker}_massive_puts"
    spot_path = _cache_path(f"{ticker}_massive_spot", ext="json")

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

    # Fetch from Massive.com API with pagination
    all_results = []
    endpoint = f"/snapshot/options/{ticker}"
    params = {"limit": 250, "order": "asc", "sort": "ticker"}

    while endpoint:
        data = _massive_get(endpoint, params)
        if data is None or data.get("status") != "OK":
            break

        results = data.get("results", [])
        if not results:
            break
        all_results.extend(results)

        # Handle pagination
        next_url = data.get("next_url")
        if next_url:
            # next_url is a full URL; extract path and params
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(next_url)
            endpoint = parsed.path.replace("/v3", "", 1) if "/v3" in parsed.path else parsed.path
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            # Remove apiKey from params (we add it in _massive_get)
            params.pop("apiKey", None)
        else:
            break

    if not all_results:
        return None, None, None

    # Parse results into calls_df and puts_df
    calls_rows = []
    puts_rows = []
    spot = None

    for contract in all_results:
        details = contract.get("details", {})
        greeks = contract.get("greeks", {})
        underlying = contract.get("underlying_asset", {})

        if spot is None and underlying.get("price"):
            spot = underlying["price"]

        row = {
            "strike": details.get("strike_price"),
            "expiration": details.get("expiration_date"),
            "gamma": greeks.get("gamma", 0) or 0,
            "delta": greeks.get("delta", 0) or 0,
            "theta": greeks.get("theta", 0) or 0,
            "vega": greeks.get("vega", 0) or 0,
            "openInterest": contract.get("open_interest", 0) or 0,
            "impliedVolatility": contract.get("implied_volatility", 0) or 0,
            "volume": contract.get("day", {}).get("volume", 0) or 0,
            "contractSymbol": details.get("ticker", ""),
        }

        if row["strike"] is None:
            continue

        contract_type = details.get("contract_type", "")
        if contract_type == "call":
            calls_rows.append(row)
        elif contract_type == "put":
            puts_rows.append(row)

    calls_df = pd.DataFrame(calls_rows) if calls_rows else pd.DataFrame()
    puts_df = pd.DataFrame(puts_rows) if puts_rows else pd.DataFrame()

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


# ── Options chain (with Massive fallback to Yahoo) ───────────────────────────

@st.cache_data(ttl=86400)
def fetch_options_chain(ticker: str):
    """
    Fetch options chain. Uses Massive.com API if an API key is configured,
    otherwise falls back to Yahoo Finance. Cached to disk daily.
    """
    # Try Massive.com first if API key is available
    if _get_massive_api_key():
        result = fetch_options_chain_massive(ticker)
        if result and result[0] is not None and not result[0].empty:
            return result

    # Fallback: Yahoo Finance
    return _fetch_options_chain_yfinance(ticker)


def _fetch_options_chain_yfinance(ticker: str):
    """Fetch options chain from Yahoo Finance (fallback)."""
    calls_cache_key = f"{ticker}_yf_options_calls"
    puts_cache_key = f"{ticker}_yf_options_puts"
    spot_path = _cache_path(f"{ticker}_yf_options_spot", ext="json")

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
            time.sleep(0.2)
        except Exception:
            continue

    calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
    puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

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
