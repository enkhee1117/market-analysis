import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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


@st.cache_data(ttl=300)
def fetch_price_history(ticker: str, period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV price history for a ticker."""
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=300)
def fetch_multi_tickers(tickers: list, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch closing prices for multiple tickers."""
    data = {}
    for t in tickers:
        df = fetch_price_history(t, period=period, interval=interval)
        if not df.empty and "Close" in df.columns:
            data[t] = df["Close"]
    if data:
        return pd.DataFrame(data).dropna(how="all")
    return pd.DataFrame()


@st.cache_data(ttl=600)
def fetch_options_chain(ticker: str):
    """Fetch all options chains for nearest expirations (up to 6)."""
    tk = yf.Ticker(ticker)
    expirations = tk.options
    if not expirations:
        return None, None, None

    spot = tk.fast_info.get("lastPrice") or tk.fast_info.get("regularMarketPrice")
    if spot is None or spot == 0:
        hist = tk.history(period="1d")
        spot = float(hist["Close"].iloc[-1]) if not hist.empty else None

    all_calls = []
    all_puts = []
    # Use nearest 8 expirations to capture meaningful OI
    for exp in expirations[:8]:
        try:
            chain = tk.option_chain(exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls["expiration"] = exp
            puts["expiration"] = exp
            all_calls.append(calls)
            all_puts.append(puts)
        except Exception:
            continue

    calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
    puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

    return calls_df, puts_df, spot
