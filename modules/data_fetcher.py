from __future__ import annotations

import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import glob
import time
from datetime import datetime, timedelta, date, timezone
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

# ── Daily two-layer cache ────────────────────────────────────────────────────
# Layer 1: Local parquet files in cache/ (fast, ephemeral on Streamlit Cloud)
# Layer 2: Supabase PostgreSQL (durable, survives redeploys)
# On read: check file → Supabase → miss.  On write: file + Supabase.

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")


def _is_market_open_now() -> bool:
    """Best-effort market-hours check used for cache bucketing."""
    try:
        from modules.supabase_cache import is_market_open
        return is_market_open()
    except Exception:
        return False


def get_refresh_bucket(dataset: str = "price", interval: str = "1d",
                       now: datetime | None = None) -> str:
    """
    Return a coarse time bucket used to refresh caches more often intraday.

    Price history is refreshed every 15 minutes during market hours.
    Options chains are refreshed every 5 minutes during market hours.
    Outside market hours we reuse a daily bucket.
    """
    now = now or datetime.now(timezone.utc)
    if not _is_market_open_now():
        return now.strftime("%Y%m%d")

    if dataset == "options":
        bucket_minutes = 15
    elif interval.endswith(("m", "h")):
        bucket_minutes = 5
    else:
        bucket_minutes = 15

    epoch_minutes = int(now.timestamp() // 60)
    bucket_start = epoch_minutes - (epoch_minutes % bucket_minutes)
    bucket_dt = datetime.fromtimestamp(bucket_start * 60, tz=timezone.utc)
    return bucket_dt.strftime("%Y%m%d%H%M")


def _bucketed_key(key: str, refresh_bucket: str | None) -> str:
    """Append a freshness bucket to the logical cache key when provided."""
    return f"{key}_{refresh_bucket}" if refresh_bucket else key


def _cache_path(key: str, ext: str = "parquet") -> str:
    """Return path like cache/SPY_price_10y_1d_2026-03-26.parquet"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}_{date.today().isoformat()}.{ext}")


def _cache_path_for_date(key: str, cache_date: str, ext: str = "parquet") -> str:
    """Return a cache path for an explicit cache date."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}_{cache_date}.{ext}")


def _supabase_table_for_key(key: str) -> str:
    """Determine which Supabase table to use based on the cache key."""
    return "price_cache" if "_price_" in key or "_multi_" in key else "options_cache"


def _read_cache(key: str) -> pd.DataFrame | None:
    """Read today's cached data. Tries local file first, then Supabase."""
    # Layer 1: local file (fast)
    path = _cache_path(key)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass

    # Layer 2: Supabase (durable)
    try:
        from modules.supabase_cache import read_cache_remote
        today = date.today().isoformat()
        remote_key = f"{key}_{today}"
        df = read_cache_remote(_supabase_table_for_key(key), remote_key)
        if df is not None:
            # Backfill local file for speed on next access
            try:
                os.makedirs(CACHE_DIR, exist_ok=True)
                df.to_parquet(_cache_path(key))
            except Exception:
                pass
            return df
    except Exception:
        pass

    return None


def _read_cache_for_date(key_prefix: str, cache_date: str) -> pd.DataFrame | None:
    """Read the latest cached DataFrame for a specific date and key prefix."""
    local_pattern = os.path.join(CACHE_DIR, f"{key_prefix}*_{cache_date}.parquet")
    local_files = sorted(glob.glob(local_pattern), reverse=True)
    for path in local_files:
        try:
            return pd.read_parquet(path)
        except Exception:
            continue

    try:
        from modules.supabase_cache import read_cache_remote_latest_for_date
        return read_cache_remote_latest_for_date(
            _supabase_table_for_key(key_prefix),
            cache_date,
            key_prefix,
        )
    except Exception:
        return None


def _write_cache(key: str, df: pd.DataFrame) -> None:
    """Write DataFrame to local file + Supabase."""
    if df.empty:
        return
    # Layer 1: local file
    try:
        path = _cache_path(key)
        df.to_parquet(path)
    except Exception:
        pass

    # Layer 2: Supabase (durable)
    try:
        from modules.supabase_cache import write_cache_remote
        today = date.today().isoformat()
        remote_key = f"{key}_{today}"
        write_cache_remote(_supabase_table_for_key(key), remote_key, df, today)
    except Exception:
        pass


def _read_spot_cache(spot_key: str) -> float | None:
    """Read cached spot price from local JSON file, then Supabase."""
    spot_path = _cache_path(spot_key, ext="json")
    # Layer 1: local file
    if os.path.exists(spot_path):
        try:
            with open(spot_path) as f:
                return json.load(f).get("spot")
        except Exception:
            pass
    # Layer 2: Supabase
    try:
        from modules.supabase_cache import read_spot_remote
        today = date.today().isoformat()
        return read_spot_remote(f"{spot_key}_{today}")
    except Exception:
        return None


def _read_spot_cache_for_date(spot_key_prefix: str, cache_date: str) -> float | None:
    """Read the latest cached spot price for a specific date and key prefix."""
    local_pattern = os.path.join(CACHE_DIR, f"{spot_key_prefix}*_{cache_date}.json")
    local_files = sorted(glob.glob(local_pattern), reverse=True)
    for path in local_files:
        try:
            with open(path) as f:
                return json.load(f).get("spot")
        except Exception:
            continue

    try:
        from modules.supabase_cache import read_spot_remote_latest_for_date
        return read_spot_remote_latest_for_date(cache_date, spot_key_prefix)
    except Exception:
        return None


def _write_spot_cache(spot_key: str, spot: float) -> None:
    """Write spot price to local JSON + Supabase."""
    # Layer 1: local file
    try:
        spot_path = _cache_path(spot_key, ext="json")
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(spot_path, "w") as f:
            json.dump({"spot": spot}, f)
    except Exception:
        pass
    # Layer 2: Supabase
    try:
        from modules.supabase_cache import write_spot_remote
        today = date.today().isoformat()
        write_spot_remote(f"{spot_key}_{today}", spot, today)
    except Exception:
        pass


def clear_today_cache(prefix: str | None = None):
    """Delete today's cached data, optionally scoped to a cache-key prefix."""
    today = date.today().isoformat()
    prefix_glob = f"{prefix}*" if prefix else "*"

    # Layer 1: local files
    if os.path.exists(CACHE_DIR):
        pattern = os.path.join(CACHE_DIR, f"{prefix_glob}_{today}.*")
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except Exception:
                pass

    # Layer 2: Supabase — delete today's entries so fresh data is fetched
    try:
        from modules.supabase_cache import _get_client
        client = _get_client()
        if client:
            for table in ("options_cache", "price_cache"):
                query = client.table(table).delete().eq("cache_date", today)
                if prefix:
                    query = query.like("cache_key", f"{prefix}%")
                query.execute()
    except Exception:
        pass


# ── Data fetching functions ──────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def fetch_price_history(
    ticker: str,
    period: str = "10y",
    interval: str = "1d",
    refresh_bucket: str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV price history for a ticker with intraday-aware caching."""
    refresh_bucket = refresh_bucket or get_refresh_bucket("price", interval=interval)
    cache_key = _bucketed_key(f"{ticker}_price_{period}_{interval}", refresh_bucket)
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        st.cache_data.clear()  # don't cache empty results for 24h
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
def fetch_multi_tickers(
    tickers: list,
    period: str = "1y",
    interval: str = "1d",
    refresh_bucket: str | None = None,
) -> pd.DataFrame:
    """Fetch closing prices for multiple tickers with intraday-aware caching."""
    refresh_bucket = refresh_bucket or get_refresh_bucket("price", interval=interval)
    tickers_key = "_".join(sorted(tickers)).replace("^", "")
    cache_key = _bucketed_key(f"{tickers_key}_multi_{period}_{interval}", refresh_bucket)
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    data = {}
    for t in tickers:
        df = fetch_price_history(t, period=period, interval=interval, refresh_bucket=refresh_bucket)
        if not df.empty and "Close" in df.columns:
            data[t] = df["Close"]
    if data:
        result = pd.DataFrame(data).dropna(how="all")
        _write_cache(cache_key, result)
        return result
    return pd.DataFrame()


# ── Massive.com API ──────────────────────────────────────────────────────────

MASSIVE_BASE_URL = "https://api.massive.com/v3"

# Rate-limit / circuit-breaker state
_massive_request_count = 0          # requests made this session
_massive_cooldown_until = 0.0       # monotonic time when cooldown expires
_MASSIVE_PAGE_DELAY = 0.3           # seconds between paginated requests
_MASSIVE_MAX_PAGES = 40             # hard cap: 40 pages × 250 = 10,000 contracts
_MASSIVE_MAX_RETRIES = 2            # retries on 429 / 5xx
_MASSIVE_COOLDOWN_SECS = 300        # 5-min cooldown after repeated 429s


def _get_massive_api_key() -> str | None:
    """Retrieve Massive API key from Streamlit secrets or environment."""
    try:
        return st.secrets["MASSIVE_API_KEY"]
    except Exception:
        return os.environ.get("MASSIVE_API_KEY")


_last_massive_error = ""


def get_last_massive_error() -> str:
    """Return the last Massive.com API error message (for UI display)."""
    return _last_massive_error


def _massive_get(endpoint: str, params: dict | None = None) -> dict | None:
    """
    Make an authenticated GET request to Massive.com API.

    Handles:
    - 429 / 5xx with exponential backoff (up to _MASSIVE_MAX_RETRIES)
    - Rate-limit headers (X-RateLimit-Remaining, Retry-After)
    - Circuit-breaker cooldown after repeated failures
    """
    global _last_massive_error, _massive_request_count, _massive_cooldown_until
    import requests

    # ── Circuit breaker: skip if in cooldown ──
    now = time.monotonic()
    if now < _massive_cooldown_until:
        remaining = int(_massive_cooldown_until - now)
        _last_massive_error = f"Rate-limit cooldown active ({remaining}s remaining)"
        return None

    api_key = _get_massive_api_key()
    if not api_key:
        _last_massive_error = "No MASSIVE_API_KEY configured"
        return None

    url = f"{MASSIVE_BASE_URL}{endpoint}"
    params = params or {}
    params["apiKey"] = api_key

    for attempt in range(_MASSIVE_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            _massive_request_count += 1

            # ── Respect rate-limit headers ──
            rl_remaining = resp.headers.get("X-RateLimit-Remaining")
            if rl_remaining is not None:
                try:
                    if int(rl_remaining) <= 2:
                        # Near the limit — slow down proactively
                        time.sleep(2.0)
                except (ValueError, TypeError):
                    pass

            # ── Handle 429 Too Many Requests ──
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else (2 ** attempt * 2)
                wait = min(wait, 60)  # cap at 60s
                if attempt < _MASSIVE_MAX_RETRIES:
                    time.sleep(wait)
                    continue
                else:
                    # Activate circuit-breaker cooldown
                    _massive_cooldown_until = time.monotonic() + _MASSIVE_COOLDOWN_SECS
                    _last_massive_error = (
                        f"Rate limited (429). Cooling down for "
                        f"{_MASSIVE_COOLDOWN_SECS}s to protect API quota."
                    )
                    return None

            # ── Handle 5xx server errors with retry ──
            if resp.status_code >= 500:
                if attempt < _MASSIVE_MAX_RETRIES:
                    time.sleep(2 ** attempt * 1)
                    continue
                _last_massive_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                return None

            # ── Other non-200 errors ──
            if resp.status_code != 200:
                _last_massive_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                return None

            data = resp.json()
            if data.get("status") != "OK":
                _last_massive_error = (
                    f"API error: {data.get('error', data.get('status', 'unknown'))}"
                )
                return None
            return data

        except requests.exceptions.Timeout:
            if attempt < _MASSIVE_MAX_RETRIES:
                time.sleep(2 ** attempt * 1)
                continue
            _last_massive_error = "Request timed out after retries"
            return None
        except Exception as e:
            _last_massive_error = str(e)[:200]
            return None

    return None  # should not reach here


def fetch_options_chain_massive(ticker: str, refresh_bucket: str | None = None):
    """
    Fetch full options chain from Massive.com API.
    Returns (calls_df, puts_df, spot) matching the format expected by compute_gex().

    Rate-limit protections:
    - File cache checked first (zero API calls on cache hit)
    - 0.3 s delay between paginated requests
    - Hard cap at 40 pages (10,000 contracts)
    - Exponential backoff on 429 / 5xx
    - 5-min circuit-breaker cooldown after repeated 429s
    """
    # Check cache (file → Supabase)
    refresh_bucket = refresh_bucket or get_refresh_bucket("options")
    calls_cache_key = _bucketed_key(f"{ticker}_massive_calls", refresh_bucket)
    puts_cache_key = _bucketed_key(f"{ticker}_massive_puts", refresh_bucket)
    spot_cache_key = _bucketed_key(f"{ticker}_massive_spot", refresh_bucket)

    cached_calls = _read_cache(calls_cache_key)
    cached_puts = _read_cache(puts_cache_key)
    cached_spot = _read_spot_cache(spot_cache_key)

    if cached_calls is not None and cached_puts is not None and cached_spot is not None:
        return cached_calls, cached_puts, cached_spot

    # Fetch from Massive.com API with pagination
    from urllib.parse import urlparse, parse_qs

    all_results = []
    endpoint = f"/snapshot/options/{ticker}"
    params = {"limit": 250, "order": "asc", "sort": "ticker"}
    page = 0

    while endpoint and page < _MASSIVE_MAX_PAGES:
        # Throttle between pages (skip delay on first request)
        if page > 0:
            time.sleep(_MASSIVE_PAGE_DELAY)

        data = _massive_get(endpoint, params)
        if data is None or data.get("status") != "OK":
            break

        results = data.get("results", [])
        if not results:
            break
        all_results.extend(results)
        page += 1

        # Handle pagination
        next_url = data.get("next_url")
        if next_url:
            parsed = urlparse(next_url)
            endpoint = parsed.path.replace("/v3", "", 1) if "/v3" in parsed.path else parsed.path
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
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

    # Write to cache (file + Supabase)
    _write_cache(calls_cache_key, calls_df)
    _write_cache(puts_cache_key, puts_df)
    if spot is not None:
        _write_spot_cache(spot_cache_key, spot)

    return calls_df, puts_df, spot


# ── Options chain (with Massive fallback to Yahoo) ───────────────────────────

def fetch_options_chain(ticker: str, refresh_bucket: str | None = None):
    """
    Fetch options chain. Uses Massive.com API if an API key is configured,
    otherwise falls back to Yahoo Finance.
    Returns (calls_df, puts_df, spot, source_name).
    """
    refresh_bucket = refresh_bucket or get_refresh_bucket("options")
    # Try Massive.com first if API key is available
    if _get_massive_api_key():
        result = fetch_options_chain_massive(ticker, refresh_bucket=refresh_bucket)
        if result and result[0] is not None and not result[0].empty:
            return result[0], result[1], result[2], "Massive.com"

    # Fallback: Yahoo Finance
    result = _fetch_options_chain_yfinance(ticker, refresh_bucket=refresh_bucket)
    return result[0], result[1], result[2], "Yahoo Finance"


def load_historical_options_chain(ticker: str, target_date: date | str,
                                  preferred_source: str | None = None):
    """
    Load a previously cached options-chain snapshot for a specific date.

    Returns (calls_df, puts_df, spot, source_name). Only dates that were
    cached previously are available.
    """
    cache_date = target_date.isoformat() if isinstance(target_date, date) else str(target_date)
    sources = []
    if preferred_source == "Massive.com":
        sources = [("Massive.com", "massive"), ("Yahoo Finance", "yf_options")]
    elif preferred_source == "Yahoo Finance":
        sources = [("Yahoo Finance", "yf_options"), ("Massive.com", "massive")]
    else:
        sources = [("Massive.com", "massive"), ("Yahoo Finance", "yf_options")]

    for source_name, source_key in sources:
        calls = _read_cache_for_date(f"{ticker}_{source_key}_calls", cache_date)
        puts = _read_cache_for_date(f"{ticker}_{source_key}_puts", cache_date)
        spot = _read_spot_cache_for_date(f"{ticker}_{source_key}_spot", cache_date)
        if calls is not None and puts is not None and spot is not None:
            return calls, puts, spot, source_name

    return None, None, None, None


def _fetch_options_chain_yfinance(ticker: str, refresh_bucket: str | None = None):
    """Fetch options chain from Yahoo Finance (fallback)."""
    refresh_bucket = refresh_bucket or get_refresh_bucket("options")
    calls_cache_key = _bucketed_key(f"{ticker}_yf_options_calls", refresh_bucket)
    puts_cache_key = _bucketed_key(f"{ticker}_yf_options_puts", refresh_bucket)
    spot_cache_key = _bucketed_key(f"{ticker}_yf_options_spot", refresh_bucket)

    cached_calls = _read_cache(calls_cache_key)
    cached_puts = _read_cache(puts_cache_key)
    cached_spot = _read_spot_cache(spot_cache_key)

    if cached_calls is not None and cached_puts is not None and cached_spot is not None:
        return cached_calls, cached_puts, cached_spot

    tk = yf.Ticker(ticker)
    try:
        expirations = tk.options
    except Exception:
        return None, None, None

    if not expirations:
        return None, None, None

    # Limit to first 15 expirations (~6 months) — far-dated options have
    # negligible gamma and fetching 50+ expirations adds seconds of latency.
    expirations = expirations[:15]

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
    for exp in expirations:  # fetch all available expirations
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
        _write_spot_cache(spot_cache_key, spot)

    return calls_df, puts_df, spot
