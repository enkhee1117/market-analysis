"""
Supabase persistent cache layer.

Provides durable storage that survives Streamlit Cloud redeploys.
All functions are **silent on failure** — the app works identically
when Supabase is unreachable; file cache and live APIs remain functional.

Tables expected in Supabase:
  - gamma_index_history  (ticker, date → gamma metrics)
  - options_cache        (cache_key → parquet bytes)
  - price_cache          (cache_key → parquet bytes)
"""

import io
import base64
import logging
import pandas as pd
from datetime import date

logger = logging.getLogger(__name__)

# ── Lazy client singleton ────────────────────────────────────────────────────

_client = None
_init_attempted = False


def _get_client():
    """Return the Supabase client, or None if credentials are missing."""
    global _client, _init_attempted
    if _init_attempted:
        return _client
    _init_attempted = True
    try:
        import streamlit as st
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except Exception:
        import os
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        return None

    try:
        from supabase import create_client
        _client = create_client(url, key)
    except Exception as e:
        logger.warning("Supabase init failed: %s", e)
        _client = None
    return _client


def is_available() -> bool:
    """Return True if Supabase is configured and reachable."""
    return _get_client() is not None


# ── Gamma Index History ──────────────────────────────────────────────────────

def save_gamma_snapshot_remote(ticker: str, entry: dict) -> None:
    """
    Upsert one gamma-index snapshot to Supabase.

    Uses ON CONFLICT (ticker, date) to overwrite same-day entries.
    """
    client = _get_client()
    if client is None:
        return
    try:
        row = {
            "date": entry.get("date", date.today().isoformat()),
            "ticker": ticker,
            "spot": entry.get("spot"),
            "gamma_index": entry.get("gamma_index"),
            "gamma_condition": entry.get("gamma_condition"),
            "call_wall": entry.get("call_wall"),
            "put_wall": entry.get("put_wall"),
            "gamma_flip": entry.get("gamma_flip"),
            "gamma_tilt": entry.get("gamma_tilt"),
            "gamma_concentration": entry.get("gamma_concentration"),
        }
        client.table("gamma_index_history").upsert(
            row, on_conflict="ticker,date"
        ).execute()
    except Exception as e:
        logger.warning("Supabase save_gamma_snapshot failed: %s", e)


def load_gamma_history_remote(ticker: str) -> list[dict] | None:
    """
    Load all gamma-index snapshots for a ticker from Supabase.

    Returns list of dicts sorted by date, or None on failure.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        resp = (
            client.table("gamma_index_history")
            .select("*")
            .eq("ticker", ticker)
            .order("date")
            .execute()
        )
        return resp.data if resp.data else None
    except Exception as e:
        logger.warning("Supabase load_gamma_history failed: %s", e)
        return None


# ── DataFrame Cache (parquet bytes in bytea columns) ─────────────────────────

def _df_to_bytes(df: pd.DataFrame) -> str:
    """Serialize DataFrame to base64-encoded parquet string (for bytea)."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _bytes_to_df(b64_str: str) -> pd.DataFrame:
    """Deserialize base64 parquet string back to DataFrame."""
    raw = base64.b64decode(b64_str)
    return pd.read_parquet(io.BytesIO(raw))


def _decode_bytea(encoded) -> bytes:
    """
    Decode a bytea value returned by Supabase.

    Supabase returns bytea columns as hex-escaped strings (\\x...).
    Since we store base64 text as bytea, we get hex-of-base64-ascii.
    Decode chain: hex → ASCII string → base64 decode → raw bytes.
    """
    if isinstance(encoded, bytes):
        return encoded
    if isinstance(encoded, str) and encoded.startswith("\\x"):
        # Hex-encoded: decode hex → bytes (which are ASCII of our base64)
        ascii_bytes = bytes.fromhex(encoded[2:])
        b64_str = ascii_bytes.decode("ascii")
        return base64.b64decode(b64_str)
    if isinstance(encoded, str):
        # Plain base64 string
        return base64.b64decode(encoded)
    return b""


def read_cache_remote(table: str, cache_key: str) -> pd.DataFrame | None:
    """
    Read a cached DataFrame from Supabase.

    Args:
        table: "options_cache" or "price_cache"
        cache_key: e.g. "SPY_massive_calls_2026-03-27"

    Returns DataFrame or None on miss/failure.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        resp = (
            client.table(table)
            .select("data_parquet")
            .eq("cache_key", cache_key)
            .limit(1)
            .execute()
        )
        if resp.data and resp.data[0].get("data_parquet"):
            raw = _decode_bytea(resp.data[0]["data_parquet"])
            return pd.read_parquet(io.BytesIO(raw))
        return None
    except Exception as e:
        logger.warning("Supabase read_cache_remote(%s, %s) failed: %s", table, cache_key, e)
        return None


def write_cache_remote(
    table: str, cache_key: str, df: pd.DataFrame, cache_date: str
) -> None:
    """
    Write a DataFrame to Supabase as parquet bytes.

    Args:
        table: "options_cache" or "price_cache"
        cache_key: unique key for this data
        df: the DataFrame to persist
        cache_date: ISO date string for retention management
    """
    client = _get_client()
    if client is None or df.empty:
        return
    try:
        b64 = _df_to_bytes(df)
        row = {
            "cache_key": cache_key,
            "cache_date": cache_date,
            "data_parquet": b64,
        }
        client.table(table).upsert(row, on_conflict="cache_key").execute()
    except Exception as e:
        logger.warning("Supabase write_cache_remote(%s, %s) failed: %s", table, cache_key, e)


# ── Spot Price Helpers ───────────────────────────────────────────────────────

def read_spot_remote(cache_key: str) -> float | None:
    """Read a cached spot price from options_cache (stored as JSON blob)."""
    client = _get_client()
    if client is None:
        return None
    try:
        resp = (
            client.table("options_cache")
            .select("data_parquet")
            .eq("cache_key", cache_key)
            .limit(1)
            .execute()
        )
        if resp.data and resp.data[0].get("data_parquet"):
            import json
            raw = _decode_bytea(resp.data[0]["data_parquet"])
            return json.loads(raw.decode("utf-8")).get("spot")
        return None
    except Exception as e:
        logger.warning("Supabase read_spot_remote(%s) failed: %s", cache_key, e)
        return None


def write_spot_remote(cache_key: str, spot: float, cache_date: str) -> None:
    """Write a spot price to options_cache as a small JSON blob."""
    client = _get_client()
    if client is None:
        return
    try:
        import json
        blob = json.dumps({"spot": spot}).encode("utf-8")
        b64 = base64.b64encode(blob).decode("ascii")
        row = {
            "cache_key": cache_key,
            "cache_date": cache_date,
            "data_parquet": b64,
        }
        client.table("options_cache").upsert(row, on_conflict="cache_key").execute()
    except Exception as e:
        logger.warning("Supabase write_spot_remote(%s) failed: %s", cache_key, e)


# ── Cleanup ──────────────────────────────────────────────────────────────────

def cleanup_old_cache(days: int = 7) -> None:
    """Delete cache entries older than `days` (run periodically)."""
    client = _get_client()
    if client is None:
        return
    from datetime import timedelta
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    try:
        client.table("options_cache").delete().lt("cache_date", cutoff).execute()
        client.table("price_cache").delete().lt("cache_date", cutoff).execute()
    except Exception as e:
        logger.warning("Supabase cleanup failed: %s", e)
