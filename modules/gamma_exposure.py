from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, date
from scipy.stats import norm
import os
import json

# Force Streamlit Cloud redeploy — module version 2026-04-08

INDEX_LIKE_TICKERS = {"SPY", "SPX", "QQQ", "IWM", "DIA"}


def _normalize_options_df(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return a lowercase-column copy of an options DataFrame."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    return out


def _option_oi_col(df: pd.DataFrame) -> str | None:
    """Return the standardized open-interest column name when present."""
    for col in ("openinterest", "open_interest"):
        if col in df.columns:
            return col
    return None


def _expiration_dte(expiration: str) -> int | None:
    """Convert an expiration string into calendar DTE."""
    try:
        exp = datetime.strptime(str(expiration), "%Y-%m-%d").date()
        return max((exp - datetime.now().date()).days, 0)
    except Exception:
        return None


def filter_options_chain(
    calls_df: pd.DataFrame | None,
    puts_df: pd.DataFrame | None,
    spot: float,
    selected_expiration: str = "All",
    dte_bucket: str = "All",
    moneyness_pct: float = 0.15,
    min_open_interest: int = 0,
    min_volume: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Filter an options chain for the active analytic lens.

    Returns filtered calls/puts plus metadata describing coverage and quality.
    """
    raw_calls = _normalize_options_df(calls_df)
    raw_puts = _normalize_options_df(puts_df)

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()

        if selected_expiration != "All" and "expiration" in out.columns:
            out = out[out["expiration"] == selected_expiration]

        if dte_bucket != "All" and "expiration" in out.columns:
            out["_dte"] = out["expiration"].map(_expiration_dte)
            lo, hi = {
                "0-7 DTE": (0, 7),
                "8-30 DTE": (8, 30),
                "31-90 DTE": (31, 90),
                "90+ DTE": (91, None),
            }.get(dte_bucket, (None, None))
            if lo is not None:
                out = out[out["_dte"] >= lo]
            if hi is not None:
                out = out[out["_dte"] <= hi]
        elif "expiration" in out.columns:
            out["_dte"] = out["expiration"].map(_expiration_dte)

        if spot and spot > 0 and "strike" in out.columns:
            lo = spot * (1 - moneyness_pct)
            hi = spot * (1 + moneyness_pct)
            out = out[out["strike"].between(lo, hi)]

        oi_col = _option_oi_col(out)
        if oi_col and min_open_interest > 0:
            out = out[out[oi_col].fillna(0) >= min_open_interest]

        if "volume" in out.columns and min_volume > 0:
            out = out[out["volume"].fillna(0) >= min_volume]

        return out.drop(columns=["_dte"], errors="ignore")

    filtered_calls = _apply(raw_calls)
    filtered_puts = _apply(raw_puts)

    all_filtered = pd.concat([filtered_calls, filtered_puts], ignore_index=True)
    all_raw = pd.concat([raw_calls, raw_puts], ignore_index=True)
    kept_contracts = len(all_filtered)
    raw_contracts = len(all_raw)
    kept_ratio = kept_contracts / raw_contracts if raw_contracts else 0.0

    expirations = sorted(all_filtered["expiration"].dropna().unique().tolist()) if "expiration" in all_filtered.columns else []
    dtes = [d for d in (_expiration_dte(exp) for exp in expirations) if d is not None]

    oi_total = 0.0
    oi_col = _option_oi_col(all_filtered)
    if oi_col:
        oi_total = float(all_filtered[oi_col].fillna(0).sum())

    volume_total = float(all_filtered["volume"].fillna(0).sum()) if "volume" in all_filtered.columns else 0.0
    gamma_coverage = (
        float(all_filtered["gamma"].fillna(0).ne(0).mean())
        if kept_contracts and "gamma" in all_filtered.columns
        else 0.0
    )

    metadata = {
        "raw_contracts": raw_contracts,
        "kept_contracts": kept_contracts,
        "kept_ratio": round(kept_ratio, 3),
        "expiration_count": len(expirations),
        "expirations": expirations,
        "min_dte": min(dtes) if dtes else None,
        "max_dte": max(dtes) if dtes else None,
        "total_open_interest": round(oi_total, 0),
        "total_volume": round(volume_total, 0),
        "gamma_coverage": round(gamma_coverage, 3),
        "selected_expiration": selected_expiration,
        "dte_bucket": dte_bucket,
        "moneyness_pct": moneyness_pct,
        "min_open_interest": min_open_interest,
        "min_volume": min_volume,
    }
    return filtered_calls, filtered_puts, metadata


def summarize_chain_quality(
    ticker: str,
    data_source: str,
    filter_meta: dict,
) -> dict:
    """Estimate a user-facing confidence level for the current options slice."""
    score = 0.0
    reasons = []

    if data_source == "Massive.com":
        score += 0.25
        reasons.append("provider includes greeks")
    else:
        score += 0.15
        reasons.append("fallback provider")

    if ticker in INDEX_LIKE_TICKERS:
        score += 0.2
        reasons.append("index/ETF assumptions are more reliable")
    else:
        score += 0.08
        reasons.append("single-stock positioning is less observable")

    kept_contracts = filter_meta.get("kept_contracts", 0)
    kept_ratio = filter_meta.get("kept_ratio", 0.0)
    expiration_count = filter_meta.get("expiration_count", 0)
    gamma_coverage = filter_meta.get("gamma_coverage", 0.0)
    total_open_interest = filter_meta.get("total_open_interest", 0.0)

    if kept_contracts >= 150:
        score += 0.15
        reasons.append("ample contract sample")
    elif kept_contracts >= 50:
        score += 0.10
    elif kept_contracts > 0:
        score += 0.05

    if kept_ratio >= 0.2:
        score += 0.10
    elif kept_ratio >= 0.1:
        score += 0.05
    else:
        reasons.append("slice is highly filtered")

    if expiration_count >= 3:
        score += 0.10
    elif expiration_count == 1:
        reasons.append("single-expiry view")

    if gamma_coverage >= 0.8:
        score += 0.10
    elif gamma_coverage < 0.4:
        reasons.append("heavy greek backfill")

    if total_open_interest >= 100000:
        score += 0.10
    elif total_open_interest >= 25000:
        score += 0.05

    score = max(0.05, min(score, 0.95))
    if score >= 0.75:
        label = "High"
    elif score >= 0.5:
        label = "Medium"
    else:
        label = "Low"

    return {"score": round(score, 2), "label": label, "reasons": reasons[:4]}


# ── Black-Scholes Gamma ───────────────────────────────────────────────────────

def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma for a European option."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def _add_computed_gamma(df: pd.DataFrame, spot: float, r: float = 0.05) -> pd.DataFrame:
    """Compute Black-Scholes gamma from IV + expiration if gamma column missing."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Parse expiration → time to expiry in years (vectorized)
    today = datetime.now().date()
    if "expiration" in df.columns:
        exp_ts = pd.to_datetime(df["expiration"], errors="coerce")
        delta_days = (exp_ts - pd.Timestamp(today)).dt.days.fillna(0).astype(int)
        df["T"] = np.maximum(delta_days.values, 0) / 365.0
    else:
        df["T"] = 30 / 365.0  # fallback 30 days

    iv_col = "impliedvolatility" if "impliedvolatility" in df.columns else None
    if iv_col and "strike" in df.columns:
        # Vectorized Black-Scholes gamma (replaces slow row-by-row .apply())
        S = spot
        K = df["strike"].values
        T = df["T"].values
        sigma = df[iv_col].values

        valid = (T > 0) & (sigma > 0) & (K > 0)
        gamma = np.zeros(len(df))

        if valid.any():
            Kv, Tv, sv = K[valid], T[valid], sigma[valid]
            d1 = (np.log(S / Kv) + (r + 0.5 * sv ** 2) * Tv) / (sv * np.sqrt(Tv))
            gamma[valid] = norm.pdf(d1) / (S * sv * np.sqrt(Tv))

        df["gamma"] = gamma
    return df


# ── Black-Scholes Delta ──────────────────────────────────────────────────────

def _bs_delta(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call") -> float:
    """Black-Scholes delta for a European option."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return float(norm.cdf(d1))
    else:
        return float(norm.cdf(d1) - 1)


def _add_computed_delta(df: pd.DataFrame, spot: float, option_type: str = "call",
                        r: float = 0.05) -> pd.DataFrame:
    """Compute Black-Scholes delta from IV + expiration if delta column missing."""
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    # Determine TTE (vectorized)
    today = datetime.now().date()
    exp_col = cols_lower.get("expiration")
    if exp_col and exp_col in df.columns:
        exp_ts = pd.to_datetime(df[exp_col], errors="coerce")
        delta_days = (exp_ts - pd.Timestamp(today)).dt.days.fillna(0).astype(int)
        df["_T"] = np.maximum(delta_days.values, 0) / 365.0
    else:
        df["_T"] = 30 / 365.0

    iv_col = cols_lower.get("impliedvolatility")
    strike_col = cols_lower.get("strike")
    if iv_col and strike_col:
        # Vectorized Black-Scholes delta (replaces slow row-by-row .apply())
        S = spot
        K = df[strike_col].values
        T = df["_T"].values
        sigma = df[iv_col].values

        valid = (T > 0) & (sigma > 0) & (K > 0)
        delta = np.zeros(len(df))

        if valid.any():
            Kv, Tv, sv = K[valid], T[valid], sigma[valid]
            d1 = (np.log(S / Kv) + (r + 0.5 * sv ** 2) * Tv) / (sv * np.sqrt(Tv))
            if option_type == "call":
                delta[valid] = norm.cdf(d1)
            else:
                delta[valid] = norm.cdf(d1) - 1

        df["delta"] = delta
    df.drop(columns=["_T"], inplace=True, errors="ignore")
    return df


# ── Delta Exposure (DEX) ────────────────────────────────────────────────────

def compute_dex(calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                spot: float) -> pd.DataFrame:
    """
    Compute dealer Delta Exposure (DEX) per strike.

    Convention (same dealer short-call / long-put assumption as GEX):
      - Dealers SHORT calls → dealer delta = -call_delta × OI × 100
      - Dealers LONG puts   → dealer delta = -put_delta × OI × 100
        (put delta is already negative, so this becomes positive)

    Positive net DEX → dealers need to sell shares to hedge (bearish pressure)
    Negative net DEX → dealers need to buy shares to hedge (bullish pressure)

    Returns DataFrame with columns: strike, call_dex, put_dex, net_dex,
    call_dex_m, put_dex_m, net_dex_m (millions of shares equivalent).
    """
    if spot is None or spot == 0:
        return pd.DataFrame()

    def _prep(df, side):
        if df is None or df.empty:
            return pd.DataFrame()
        d = df.copy()
        d.columns = [c.lower() for c in d.columns]

        # Compute delta from B-S if not present
        if "delta" not in d.columns or d["delta"].isna().all() or (d["delta"] == 0).all():
            d = _add_computed_delta(d, spot, option_type=side)

        if "openinterest" not in d.columns:
            return pd.DataFrame()
        d = d.rename(columns={"openinterest": "openInterest"})

        needed = ["strike", "delta", "openInterest"]
        missing = [c for c in needed if c not in d.columns]
        if missing:
            return pd.DataFrame()

        d = d[needed].dropna()
        d = d[d["openInterest"] > 0]
        d["side"] = side
        return d

    calls = _prep(calls_df, "call")
    puts = _prep(puts_df, "put")

    # Vectorized DEX computation (replaces slow .iterrows() loops)
    dex_parts = []
    if not calls.empty:
        call_dex = pd.DataFrame({
            "strike": calls["strike"].values,
            "call_dex": -calls["delta"].values * calls["openInterest"].values * 100,
            "put_dex": 0.0,
        })
        dex_parts.append(call_dex)

    if not puts.empty:
        put_dex = pd.DataFrame({
            "strike": puts["strike"].values,
            "call_dex": 0.0,
            "put_dex": -puts["delta"].values * puts["openInterest"].values * 100,
        })
        dex_parts.append(put_dex)

    if not dex_parts:
        return pd.DataFrame()

    df = pd.concat(dex_parts, ignore_index=True)
    dex = df.groupby("strike")[["call_dex", "put_dex"]].sum().reset_index()
    dex["net_dex"] = dex["call_dex"] + dex["put_dex"]

    # Scale to millions of shares
    scale = 1e6
    dex["call_dex_m"] = dex["call_dex"] / scale
    dex["put_dex_m"] = dex["put_dex"] / scale
    dex["net_dex_m"] = dex["net_dex"] / scale

    return dex.sort_values("strike").reset_index(drop=True)


def total_dex_metrics(dex_df: pd.DataFrame) -> dict:
    """Summary metrics for DEX."""
    if dex_df.empty:
        return {}
    return {
        "total_net_dex_m": round(float(dex_df["net_dex_m"].sum()), 2),
        "total_call_dex_m": round(float(dex_df["call_dex_m"].sum()), 2),
        "total_put_dex_m": round(float(dex_df["put_dex_m"].sum()), 2),
    }


# ── IV Skew ─────────────────────────────────────────────────────────────────

def compute_iv_skew(calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                    spot: float, expiration: str | None = None) -> pd.DataFrame:
    """
    Compute IV skew data for calls and puts, organized by strike.

    Returns DataFrame with columns:
      strike, call_iv, put_iv, moneyness (strike/spot ratio),
      iv_skew (put_iv - call_iv at same strike).
    """
    if spot is None or spot == 0:
        return pd.DataFrame()

    def _prep_iv(df):
        if df is None or df.empty:
            return pd.DataFrame()
        d = df.copy()
        d.columns = [c.lower() for c in d.columns]
        iv_col = "impliedvolatility"
        if iv_col not in d.columns or "strike" not in d.columns:
            return pd.DataFrame()
        # Use an explicit expiration when selected, otherwise default to
        # the 2nd future expiry to avoid same-day/0DTE IV artifacts.
        if "expiration" in d.columns:
            if expiration:
                exp = expiration
            else:
                nearest_exp = sorted(d["expiration"].unique())
                today_str = datetime.now().date().isoformat()
                valid_exps = [e for e in nearest_exp if e > today_str]
                if len(valid_exps) >= 2:
                    exp = valid_exps[1]
                elif valid_exps:
                    exp = valid_exps[0]
                else:
                    return pd.DataFrame()
            d = d[d["expiration"] == exp]

        d = d[["strike", iv_col]].dropna()
        d = d[d[iv_col] > 0.001]  # filter out zero/garbage IV
        # Average if multiple rows per strike
        d = d.groupby("strike")[iv_col].mean().reset_index()
        return d

    call_iv = _prep_iv(calls_df)
    put_iv = _prep_iv(puts_df)

    if call_iv.empty and put_iv.empty:
        return pd.DataFrame()

    # Merge on strike
    if not call_iv.empty:
        call_iv = call_iv.rename(columns={"impliedvolatility": "call_iv"})
    if not put_iv.empty:
        put_iv = put_iv.rename(columns={"impliedvolatility": "put_iv"})

    if not call_iv.empty and not put_iv.empty:
        merged = pd.merge(call_iv, put_iv, on="strike", how="outer")
    elif not call_iv.empty:
        merged = call_iv.copy()
        merged["put_iv"] = np.nan
    else:
        merged = put_iv.copy()
        merged["call_iv"] = np.nan

    merged["moneyness"] = merged["strike"] / spot
    merged["iv_skew"] = merged.get("put_iv", 0) - merged.get("call_iv", 0)
    merged = merged.sort_values("strike").reset_index(drop=True)
    return merged


def compute_atm_iv_term_structure(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    spot: float,
    min_open_interest: int = 0,
    min_volume: int = 0,
) -> pd.DataFrame:
    """
    Build an ATM IV term structure across expirations.

    For each future expiration, select the strike nearest to spot and compute:
    - call_atm_iv
    - put_atm_iv
    - atm_iv (average of call and put when both exist)
    """
    if spot is None or spot <= 0:
        return pd.DataFrame()

    def _prep(df: pd.DataFrame | None, side: str) -> pd.DataFrame:
        d = _normalize_options_df(df)
        if d.empty or "expiration" not in d.columns or "strike" not in d.columns:
            return pd.DataFrame()
        if "impliedvolatility" not in d.columns:
            return pd.DataFrame()

        d = d.dropna(subset=["expiration", "strike", "impliedvolatility"]).copy()
        d = d[d["impliedvolatility"] > 0.001]
        d["_dte"] = d["expiration"].map(_expiration_dte)
        d = d[d["_dte"].notna()]
        d = d[d["_dte"] > 0]

        oi_col = _option_oi_col(d)
        if oi_col and min_open_interest > 0:
            d = d[d[oi_col].fillna(0) >= min_open_interest]
        if "volume" in d.columns and min_volume > 0:
            d = d[d["volume"].fillna(0) >= min_volume]

        if d.empty:
            return pd.DataFrame()

        d["_distance"] = (d["strike"] - spot).abs()
        rows = []
        for exp, sub in d.groupby("expiration"):
            nearest = sub["_distance"].min()
            atm = sub[sub["_distance"] == nearest].copy()
            if atm.empty:
                continue
            # If both upper/lower strikes tie, average them.
            iv = float(atm["impliedvolatility"].mean())
            strike = float(atm["strike"].mean())
            dte = int(atm["_dte"].iloc[0])
            rows.append({
                "expiration": exp,
                "dte": dte,
                f"{side}_atm_iv": iv,
                f"{side}_atm_strike": strike,
            })
        return pd.DataFrame(rows)

    call_term = _prep(calls_df, "call")
    put_term = _prep(puts_df, "put")

    if call_term.empty and put_term.empty:
        return pd.DataFrame()

    if not call_term.empty and not put_term.empty:
        merged = pd.merge(call_term, put_term, on=["expiration", "dte"], how="outer")
    elif not call_term.empty:
        merged = call_term.copy()
    else:
        merged = put_term.copy()

    merged["atm_iv"] = merged[[c for c in ("call_atm_iv", "put_atm_iv") if c in merged.columns]].mean(axis=1)
    strike_cols = [c for c in ("call_atm_strike", "put_atm_strike") if c in merged.columns]
    if strike_cols:
        merged["atm_strike"] = merged[strike_cols].mean(axis=1)
    merged["iv_slope_from_front"] = (merged["atm_iv"] - merged["atm_iv"].iloc[0]) * 100 if not merged.empty else np.nan
    return merged.sort_values(["dte", "expiration"]).reset_index(drop=True)


def aggregate_gex_by_expiration(calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                                spot: float) -> pd.DataFrame:
    """Return per-expiration GEX totals as a DataFrame.

    Computes gamma once on the full chain then aggregates by expiration,
    avoiding repeated Black-Scholes calls per expiration.
    """
    calls = _normalize_options_df(calls_df)
    puts = _normalize_options_df(puts_df)
    if calls.empty or "expiration" not in calls.columns:
        return pd.DataFrame()

    spot2 = spot ** 2 * 0.01
    scale = 1e9

    def _gex_col(df, sign):
        """Add per-row GEX to a normalised options DataFrame."""
        if df.empty:
            return pd.DataFrame()
        d = df.copy()
        if "gamma" not in d.columns or d["gamma"].isna().all() or (d["gamma"] == 0).all():
            d = _add_computed_gamma(d, spot)
        oi_col = "openinterest" if "openinterest" in d.columns else None
        if oi_col is None or "gamma" not in d.columns or "expiration" not in d.columns:
            return pd.DataFrame()
        d = d.dropna(subset=["gamma", oi_col])
        d = d[(d["gamma"] > 0) & (d[oi_col] > 0)]
        d["_gex_b"] = sign * d["gamma"] * d[oi_col] * 100 * spot2 / scale
        return d[["expiration", "_gex_b"]]

    call_gex = _gex_col(calls, 1.0)
    put_gex = _gex_col(puts, -1.0)

    rows = []
    if not call_gex.empty:
        agg = call_gex.groupby("expiration")["_gex_b"].sum().reset_index()
        agg.columns = ["Expiration", "Call GEX ($B)"]
        rows.append(agg)
    if not put_gex.empty:
        agg = put_gex.groupby("expiration")["_gex_b"].sum().reset_index()
        agg.columns = ["Expiration", "Put GEX ($B)"]
        rows.append(agg)

    if not rows:
        return pd.DataFrame()

    from functools import reduce
    merged = reduce(lambda a, b: pd.merge(a, b, on="Expiration", how="outer"), rows)
    merged = merged.fillna(0.0)
    if "Call GEX ($B)" not in merged.columns:
        merged["Call GEX ($B)"] = 0.0
    if "Put GEX ($B)" not in merged.columns:
        merged["Put GEX ($B)"] = 0.0
    merged["Net GEX ($B)"] = merged["Call GEX ($B)"] + merged["Put GEX ($B)"]
    return merged.sort_values("Expiration").reset_index(drop=True)


def aggregate_dex_by_expiration(calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                                spot: float) -> pd.DataFrame:
    """Return per-expiration DEX totals as a DataFrame.

    Computes delta once on the full chain then aggregates by expiration,
    avoiding repeated Black-Scholes calls per expiration.
    """
    calls = _normalize_options_df(calls_df)
    puts = _normalize_options_df(puts_df)
    if calls.empty or "expiration" not in calls.columns:
        return pd.DataFrame()

    scale = 1e6

    def _dex_col(df, side):
        """Add per-row DEX to a normalised options DataFrame."""
        if df.empty:
            return pd.DataFrame()
        d = df.copy()
        if "delta" not in d.columns or d["delta"].isna().all() or (d["delta"] == 0).all():
            d = _add_computed_delta(d, spot, option_type=side)
        oi_col = "openinterest" if "openinterest" in d.columns else None
        if oi_col is None or "delta" not in d.columns or "expiration" not in d.columns:
            return pd.DataFrame()
        d = d.dropna(subset=["delta", oi_col])
        d = d[d[oi_col] > 0]
        d["_dex_m"] = -d["delta"] * d[oi_col] * 100 / scale
        return d[["expiration", "_dex_m"]]

    call_dex = _dex_col(calls, "call")
    put_dex = _dex_col(puts, "put")

    rows = []
    if not call_dex.empty:
        agg = call_dex.groupby("expiration")["_dex_m"].sum().reset_index()
        agg.columns = ["Expiration", "Call DEX (M)"]
        rows.append(agg)
    if not put_dex.empty:
        agg = put_dex.groupby("expiration")["_dex_m"].sum().reset_index()
        agg.columns = ["Expiration", "Put DEX (M)"]
        rows.append(agg)

    if not rows:
        return pd.DataFrame()

    from functools import reduce
    merged = reduce(lambda a, b: pd.merge(a, b, on="Expiration", how="outer"), rows)
    merged = merged.fillna(0.0)
    if "Call DEX (M)" not in merged.columns:
        merged["Call DEX (M)"] = 0.0
    if "Put DEX (M)" not in merged.columns:
        merged["Put DEX (M)"] = 0.0
    merged["Net DEX (M)"] = merged["Call DEX (M)"] + merged["Put DEX (M)"]
    return merged.sort_values("Expiration").reset_index(drop=True)


# ── GEX Calculation ──────────────────────────────────────────────────────────

def compute_gex(calls_df: pd.DataFrame, puts_df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """
    Compute dealer Gamma Exposure (GEX) per strike.

    Convention (standard market-maker assumption):
      - Dealers are SHORT calls  → negative delta hedge pressure
      - Dealers are LONG puts    → positive delta hedge pressure
      Call GEX =  Gamma × OI × 100 × Spot² × 0.01   (positive = stabilizing)
      Put  GEX = -Gamma × OI × 100 × Spot² × 0.01   (negative = destabilizing)
      Net GEX per strike = Call GEX + Put GEX

    Gamma is computed from Black-Scholes if not provided by the data source.
    """
    if spot is None or spot == 0:
        return pd.DataFrame()

    def _clean(df, side):
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Compute gamma from B-S if not present
        if "gamma" not in df.columns or df["gamma"].isna().all() or (df["gamma"] == 0).all():
            df = _add_computed_gamma(df, spot)

        if "openinterest" not in df.columns:
            return pd.DataFrame()

        df = df.rename(columns={"openinterest": "openInterest"})
        needed = ["strike", "gamma", "openInterest"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            return pd.DataFrame()

        df = df[needed].dropna(subset=["gamma", "openInterest"])
        df = df[df["gamma"] > 0]
        df = df[df["openInterest"] > 0]
        df["side"] = side
        return df

    calls = _clean(calls_df, "call")
    puts  = _clean(puts_df,  "put")

    # Vectorized GEX computation (replaces slow .iterrows() loops)
    spot2 = spot ** 2 * 0.01
    gex_parts = []
    if not calls.empty:
        call_gex = pd.DataFrame({
            "strike": calls["strike"].values,
            "call_gex": calls["gamma"].values * calls["openInterest"].values * 100 * spot2,
            "put_gex": 0.0,
        })
        gex_parts.append(call_gex)

    if not puts.empty:
        put_gex = pd.DataFrame({
            "strike": puts["strike"].values,
            "call_gex": 0.0,
            "put_gex": -puts["gamma"].values * puts["openInterest"].values * 100 * spot2,
        })
        gex_parts.append(put_gex)

    if not gex_parts:
        return pd.DataFrame()

    df = pd.concat(gex_parts, ignore_index=True)
    gex = df.groupby("strike")[["call_gex", "put_gex"]].sum().reset_index()
    gex["net_gex"] = gex["call_gex"] + gex["put_gex"]

    # Scale to billions for readability
    scale = 1e9
    gex["call_gex_b"]  = gex["call_gex"] / scale
    gex["put_gex_b"]   = gex["put_gex"]  / scale
    gex["net_gex_b"]   = gex["net_gex"]  / scale

    return gex.sort_values("strike").reset_index(drop=True)


def gex_flip_point(gex_df: pd.DataFrame, spot: float | None = None) -> float | None:
    """
    Find the gamma flip point: the strike where net GEX crosses zero.

    When *spot* is provided the function returns the crossing **nearest
    to spot** (which is the one traders actually care about).  Without
    spot it falls back to the crossing with the largest absolute GEX
    magnitude on either side — i.e. the most "meaningful" one.

    Returns the interpolated strike price, or None if no crossing exists.
    """
    if gex_df.empty or "net_gex" not in gex_df.columns:
        return None

    df = gex_df.sort_values("strike")
    net = df["net_gex"].values
    strikes = df["strike"].values

    # Collect ALL zero crossings
    crossings = []
    for i in range(len(net) - 1):
        if net[i] * net[i + 1] < 0:
            s1, s2 = strikes[i], strikes[i + 1]
            g1, g2 = net[i], net[i + 1]
            flip = s1 - g1 * (s2 - s1) / (g2 - g1)
            # Magnitude = sum of absolute GEX on both sides of crossing
            magnitude = abs(g1) + abs(g2)
            crossings.append((round(flip, 2), magnitude))

    if not crossings:
        return None

    if spot is not None and spot > 0:
        # Return the crossing nearest to spot
        crossings.sort(key=lambda c: abs(c[0] - spot))
    else:
        # Return the crossing with the largest surrounding GEX magnitude
        crossings.sort(key=lambda c: -c[1])

    return crossings[0][0]


def total_gex_metrics(gex_df: pd.DataFrame) -> dict:
    """Return summary metrics for the GEX profile."""
    if gex_df.empty:
        return {}
    total_net  = gex_df["net_gex_b"].sum()
    total_call = gex_df["call_gex_b"].sum()
    total_put  = gex_df["put_gex_b"].sum()
    peak_call_strike = gex_df.loc[gex_df["call_gex_b"].idxmax(), "strike"] if not gex_df.empty else None
    peak_put_strike  = gex_df.loc[gex_df["put_gex_b"].idxmin(),  "strike"] if not gex_df.empty else None
    return {
        "total_net_gex_b":  round(total_net, 3),
        "total_call_gex_b": round(total_call, 3),
        "total_put_gex_b":  round(total_put, 3),
        "peak_call_strike": peak_call_strike,
        "peak_put_strike":  peak_put_strike,
    }


# ── Gamma Index (inspired by SpotGamma) ─────────────────────────────────────

def compute_gamma_index(gex_df: pd.DataFrame, spot: float) -> dict:
    """
    Compute a Gamma Index and derived key levels, inspired by SpotGamma's
    GEX Index methodology.

    Metrics returned
    ----------------
    gamma_index : float
        Total Net GEX in $ billions — the estimated dealer hedging flow
        triggered by a 1 % move in the underlying.  Positive → stabilising
        (dealers sell rallies / buy dips), negative → destabilising.
    gamma_condition : str
        Human-readable label: "Positive (Stabilizing)" or "Negative (Destabilizing)".
    call_wall : float
        Strike with the largest absolute call GEX — acts as resistance.
    put_wall : float
        Strike with the largest absolute put GEX — acts as support.
    gamma_flip : float | None
        Strike where net GEX crosses zero.
    gamma_tilt : float
        (Σ net_gex above spot) / (Σ |net_gex| above + below).
        > 0.5 → bullish tilt, < 0.5 → bearish tilt.
    gamma_concentration : float
        Fraction of total |GEX| within ± 2 % of spot — higher means
        gamma is clustered near the money (stronger pin risk).
    top_strikes : list[dict]
        Top 5 strikes by absolute net GEX with their values.
    """
    if gex_df.empty or spot is None or spot == 0:
        return {}

    df = gex_df.copy()

    # ── Core index value (total net GEX in $B) ──
    gamma_index = float(df["net_gex_b"].sum())

    # ── Condition label ──
    if gamma_index > 0:
        condition = "Positive (Stabilizing)"
    elif gamma_index < 0:
        condition = "Negative (Destabilizing)"
    else:
        condition = "Neutral"

    # ── Key levels: Call Wall / Put Wall ──
    call_wall = float(df.loc[df["call_gex_b"].idxmax(), "strike"]) if df["call_gex_b"].abs().sum() > 0 else None
    put_wall  = float(df.loc[df["put_gex_b"].idxmin(),  "strike"]) if df["put_gex_b"].abs().sum() > 0 else None

    # ── Gamma Flip ──
    flip = gex_flip_point(df, spot)

    # ── Gamma Tilt (above-spot fraction of net GEX) ──
    above = df[df["strike"] > spot]
    below = df[df["strike"] <= spot]
    total_abs = df["net_gex"].abs().sum()
    if total_abs > 0:
        above_net = above["net_gex"].sum()
        tilt = (above_net / total_abs + 1) / 2  # map from [-1,1] → [0,1]
        tilt = round(max(0.0, min(1.0, tilt)), 3)
    else:
        tilt = 0.5

    # ── Gamma Concentration (within ±2% of spot) ──
    lo = spot * 0.98
    hi = spot * 1.02
    near_spot = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
    if total_abs > 0:
        concentration = round(float(near_spot["net_gex"].abs().sum() / total_abs), 3)
    else:
        concentration = 0.0

    # ── Top 5 strikes by absolute net GEX ──
    df["abs_net"] = df["net_gex_b"].abs()
    top5 = df.nlargest(5, "abs_net")
    top_strikes = [
        {"strike": float(row["strike"]), "net_gex_b": round(float(row["net_gex_b"]), 4)}
        for _, row in top5.iterrows()
    ]

    return {
        "gamma_index":          round(gamma_index, 4),
        "gamma_condition":      condition,
        "call_wall":            call_wall,
        "put_wall":             put_wall,
        "gamma_flip":           flip,
        "gamma_tilt":           round(tilt, 3),
        "gamma_concentration":  concentration,
        "top_strikes":          top_strikes,
    }


# ── Gamma Index History ──────────────────────────────────────────────────────

_HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
_HISTORY_FILE = os.path.join(_HISTORY_DIR, "gamma_index_history.json")


def _load_gi_history() -> list[dict]:
    """Load the gamma-index history file (list of daily snapshots)."""
    if os.path.exists(_HISTORY_FILE):
        try:
            with open(_HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_gi_history(history: list[dict]) -> None:
    """Persist the gamma-index history list to disk."""
    try:
        os.makedirs(_HISTORY_DIR, exist_ok=True)
        with open(_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass


def save_gamma_index_snapshot(ticker: str, gamma_idx: dict, spot: float) -> None:
    """
    Append today's gamma-index snapshot to the history file + Supabase.

    One entry per (ticker, date) pair — overwrites if already present today.
    """
    if not gamma_idx:
        return
    today = date.today().isoformat()
    entry = {
        "date": today,
        "ticker": ticker,
        "spot": round(spot, 2),
        "gamma_index": gamma_idx.get("gamma_index"),
        "gamma_condition": gamma_idx.get("gamma_condition"),
        "call_wall": gamma_idx.get("call_wall"),
        "put_wall": gamma_idx.get("put_wall"),
        "gamma_flip": gamma_idx.get("gamma_flip"),
        "gamma_tilt": gamma_idx.get("gamma_tilt"),
        "gamma_concentration": gamma_idx.get("gamma_concentration"),
    }

    # Layer 1: local file
    history = _load_gi_history()
    history = [h for h in history if not (h.get("date") == today and h.get("ticker") == ticker)]
    history.append(entry)
    history.sort(key=lambda x: x.get("date", ""))
    history = history[-365:]
    _save_gi_history(history)

    # Layer 2: Supabase (durable)
    try:
        from modules.supabase_cache import save_gamma_snapshot_remote
        save_gamma_snapshot_remote(ticker, entry)
    except Exception:
        pass


def load_gamma_index_history(ticker: str) -> pd.DataFrame:
    """
    Return a DataFrame of historical gamma-index snapshots for a ticker.

    Tries local file first; falls back to Supabase if local is empty.
    """
    # Layer 1: local file
    history = _load_gi_history()
    rows = [h for h in history if h.get("ticker") == ticker]

    # Layer 2: Supabase fallback if local is empty
    if not rows:
        try:
            from modules.supabase_cache import load_gamma_history_remote
            remote = load_gamma_history_remote(ticker)
            if remote:
                rows = remote
                # Backfill local file with remote data
                merged = history + [r for r in remote
                                    if not any(h.get("date") == r.get("date")
                                               and h.get("ticker") == r.get("ticker")
                                               for h in history)]
                merged.sort(key=lambda x: x.get("date", ""))
                _save_gi_history(merged[-365:])
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def plot_gamma_index_timeline(ticker: str) -> go.Figure:
    """
    Line chart of the recorded Gamma Index over time.

    Shows the saved daily Gamma Index snapshots with green/red regime fill.
    """
    real_df = load_gamma_index_history(ticker)
    has_real = not real_df.empty and "gamma_index" in real_df.columns

    if not has_real:
        fig = go.Figure()
        fig.update_layout(
            title=f"Gamma Index Timeline — {ticker} (no history yet)",
            template="plotly_dark", height=400,
            annotations=[dict(
                text="History builds up one data-point per day.<br>Check back tomorrow!",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16, color="#8B9BBF"),
            )],
        )
        return fig

    fig = go.Figure()
    pos = real_df["gamma_index"].clip(lower=0)
    neg = real_df["gamma_index"].clip(upper=0)

    fig.add_trace(go.Scatter(
        x=real_df["date"], y=pos,
        fill="tozeroy", fillcolor="rgba(95,201,123,0.3)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=real_df["date"], y=neg,
        fill="tozeroy", fillcolor="rgba(232,92,92,0.3)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=real_df["date"], y=real_df["gamma_index"],
        mode="lines+markers",
        name="Gamma Index",
        line=dict(color="#F5E642", width=2.5),
        marker=dict(size=6),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>GI: %{y:+.3f}B<extra></extra>",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")

    fig.update_layout(
        title=f"{ticker} Gamma Index Timeline",
        xaxis_title="Date",
        yaxis_title="Gamma Index ($B)",
        template="plotly_dark",
        height=400,
        legend=dict(orientation="h", y=1.05),
        yaxis=dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.4)"),
    )

    return fig


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_price_with_gex_levels(
    price_df: pd.DataFrame,
    spot: float,
    ticker: str,
    call_wall: float | None = None,
    put_wall: float | None = None,
    gamma_flip: float | None = None,
    top_strikes: list | None = None,
) -> go.Figure:
    """
    Candlestick chart of recent price action with GEX key levels overlaid
    as horizontal lines: call wall (resistance), put wall (support),
    gamma flip (regime boundary).

    Y-axis is anchored to the price range — GEX levels that fall within
    the visible window are drawn as full lines; levels outside the window
    are shown as arrows/annotations at the chart edge.
    """
    fig = go.Figure()

    if price_df is None or price_df.empty:
        fig.update_layout(
            title=f"{ticker} — No price data available",
            template="plotly_dark", height=400,
        )
        return fig

    df = price_df.copy()
    df.index = pd.to_datetime(df.index)

    # Use last 20 trading days for more context
    df = df.tail(20)

    has_ohlc = all(c in df.columns for c in ["Open", "High", "Low", "Close"])

    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="Price",
            increasing_line_color="#5FC97B",
            decreasing_line_color="#E85C5C",
        ))
        price_min = float(df["Low"].min())
        price_max = float(df["High"].max())
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"],
            mode="lines", name="Close",
            line=dict(color="#F5E642", width=2),
        ))
        price_min = float(df["Close"].min())
        price_max = float(df["Close"].max())

    # ── Y-axis range: based on price with comfortable padding ──
    price_range = price_max - price_min
    pad = max(price_range * 0.25, spot * 0.01)  # at least 1% of spot
    y_min = price_min - pad
    y_max = price_max + pad

    # Gently extend if a GEX level is *just* outside (within 3% of spot)
    near_threshold = spot * 0.03
    for lvl in [call_wall, put_wall, gamma_flip]:
        if lvl is not None:
            if lvl < y_min and (y_min - lvl) < near_threshold:
                y_min = lvl - pad * 0.2
            elif lvl > y_max and (lvl - y_max) < near_threshold:
                y_max = lvl + pad * 0.2

    # ── Helper: draw a GEX level (line if visible, edge annotation if off-screen) ──
    def _add_level(value, label, color, dash="solid"):
        if value is None:
            return
        if y_min <= value <= y_max:
            # Level is visible — draw full horizontal line
            fig.add_hline(
                y=value, line_dash=dash, line_color=color, line_width=2,
                annotation_text=f"{label} ${value:,.0f}",
                annotation_position="right",
                annotation_font=dict(color=color, size=11),
            )
        else:
            # Level is off-screen — show arrow at chart edge
            arrow_y = y_max if value > y_max else y_min
            direction = "above" if value > y_max else "below"
            dist_pct = abs(value - spot) / spot * 100
            fig.add_annotation(
                x=df.index[-1], y=arrow_y,
                text=f"{'▲' if direction == 'above' else '▼'} {label} ${value:,.0f} ({dist_pct:.1f}% away)",
                showarrow=False,
                font=dict(color=color, size=10),
                xanchor="right",
                yanchor="bottom" if direction == "above" else "top",
            )

    _add_level(call_wall, "Call Wall", "#4C9BE8")
    _add_level(put_wall, "Put Wall", "#E85C5C")
    _add_level(gamma_flip, "Gamma Flip", "#F28C38", dash="dash")

    # Top strikes as subtle dotted lines (only if visible and not duplicating walls/flip)
    if top_strikes:
        existing = {call_wall, put_wall, gamma_flip}
        for ts in top_strikes[:3]:
            strike = ts.get("strike")
            if strike and strike not in existing and y_min <= strike <= y_max:
                fig.add_hline(
                    y=strike, line_dash="dot",
                    line_color="rgba(180,180,180,0.3)", line_width=1,
                )

    fig.update_layout(
        title=f"{ticker} Price + GEX Key Levels (20d)",
        template="plotly_dark",
        height=420,
        xaxis_rangeslider_visible=False,
        yaxis=dict(title="Price", range=[y_min, y_max]),
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def _trim_to_activity(df: pd.DataFrame, col: str, spot: float,
                      threshold_pct: float = 0.005) -> pd.DataFrame:
    """Trim a GEX/DEX DataFrame to the range of strikes with meaningful activity.

    Removes empty outer edges while keeping all strikes between the outermost
    ones that exceed `threshold_pct` of peak absolute value.  Always includes
    a small pad of 2 strikes on each side so bars don't sit on the edge.
    """
    peak = df[col].abs().max()
    if peak == 0:
        return df
    cutoff = peak * threshold_pct
    active = df[df[col].abs() >= cutoff]
    if active.empty:
        return df
    lo_strike = active["strike"].min()
    hi_strike = active["strike"].max()
    # Pad: include 2 strikes beyond the outermost active ones
    all_strikes = sorted(df["strike"].unique())
    lo_idx = max(0, all_strikes.index(lo_strike) - 2) if lo_strike in all_strikes else 0
    hi_idx = min(len(all_strikes) - 1, all_strikes.index(hi_strike) + 2) if hi_strike in all_strikes else len(all_strikes) - 1
    return df[(df["strike"] >= all_strikes[lo_idx]) & (df["strike"] <= all_strikes[hi_idx])].copy()


def plot_gex_profile(gex_df: pd.DataFrame, spot: float, ticker: str,
                     strike_range_pct: float = 0.10,
                     view_mode: str = "Call / Put",
                     call_wall: float | None = None,
                     put_wall: float | None = None) -> go.Figure:
    """
    Bar chart showing GEX per strike with a rangeslider for zoom control.

    Shows all strikes with meaningful activity (auto-trimmed), with the
    initial view focused on ±strike_range_pct around spot.  Users can
    drag the rangeslider to zoom in/out without losing data.

    view_mode:
      "Call / Put" — stacked call (blue) + put (red) bars.
      "Net GEX"    — single bar series coloured green (positive) / red (negative).
    """
    if gex_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No options data available for {ticker}", template="plotly_dark")
        return fig

    # ── Trim to strikes with meaningful activity (remove empty edges) ──
    sub = _trim_to_activity(gex_df, "net_gex_b", spot)

    flip = gex_flip_point(gex_df, spot)

    fig = go.Figure()

    if view_mode == "Net GEX":
        colors = ["#5FC97B" if v >= 0 else "#E85C5C" for v in sub["net_gex_b"]]
        fig.add_trace(go.Bar(
            x=sub["strike"],
            y=sub["net_gex_b"],
            name="Net GEX",
            marker_color=colors,
            opacity=0.85,
            hovertemplate="Strike: %{x}<br>Net GEX: %{y:.3f}B<extra></extra>",
        ))
    else:
        fig.add_trace(go.Bar(
            x=sub["strike"],
            y=sub["call_gex_b"],
            name="Call GEX",
            marker_color="#4C9BE8",
            opacity=0.8,
            hovertemplate="Strike: %{x}<br>Call GEX: %{y:.3f}B<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=sub["strike"],
            y=sub["put_gex_b"],
            name="Put GEX",
            marker_color="#E85C5C",
            opacity=0.8,
            hovertemplate="Strike: %{x}<br>Put GEX: %{y:.3f}B<extra></extra>",
        ))

    # Spot price line
    fig.add_vline(
        x=spot, line_dash="dash", line_color="white", line_width=2,
        annotation_text=f"Spot: {spot:.2f}",
        annotation_position="top right", annotation_font_color="white",
    )

    # Gamma flip line
    if flip is not None:
        color = "#5FC97B" if flip > spot else "#F28C38"
        fig.add_vline(
            x=flip, line_dash="dot", line_color=color, line_width=2,
            annotation_text=f"Flip: {flip:.0f}",
            annotation_position="top left", annotation_font_color=color,
        )

    # ── Initial zoom: ±range% of spot, expanded to include key levels ──
    view_lo = spot * (1 - strike_range_pct)
    view_hi = spot * (1 + strike_range_pct)
    pad = spot * 0.02
    if call_wall is not None and call_wall > view_hi:
        view_hi = call_wall + pad
    if put_wall is not None and put_wall < view_lo:
        view_lo = put_wall - pad

    view_label = "Net" if view_mode == "Net GEX" else "Call / Put"
    fig.update_layout(
        barmode="relative",
        title=f"{ticker} Dealer GEX Profile — {view_label}",
        xaxis_title="Strike Price",
        yaxis_title="GEX ($ Billions)",
        template="plotly_dark",
        height=550,
        legend=dict(orientation="h", y=1.05),
        yaxis=dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.4)"),
        xaxis=dict(
            range=[view_lo, view_hi],
            rangeslider=dict(visible=True, thickness=0.06),
        ),
    )
    return fig


def plot_gex_by_expiration(calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                           spot: float, ticker: str,
                           max_months: int = 6) -> go.Figure:
    """Bar chart of call GEX (up) and put GEX (down) per expiration date.

    Args:
        max_months: Only show expirations within this many months.
                    Set to 0 to show all.
    """
    df = aggregate_gex_by_expiration(calls_df, puts_df, spot)
    if df.empty:
        return go.Figure()

    # ── Trim to near-term expirations so bars are visible ──
    if max_months > 0:
        from datetime import timedelta
        cutoff = (date.today() + timedelta(days=max_months * 30)).isoformat()
        df = df[df["Expiration"] <= cutoff]
        if df.empty:
            df = aggregate_gex_by_expiration(calls_df, puts_df, spot).head(12)

    fig = go.Figure()

    # Call GEX bars (positive / upward)
    fig.add_trace(go.Bar(
        x=df["Expiration"], y=df["Call GEX ($B)"],
        name="Call GEX", marker_color="#4C9BE8", opacity=0.85,
        hovertemplate="%{x}<br>Call GEX: %{y:.2f}$B<extra></extra>",
    ))
    # Put GEX bars (negative / downward)
    fig.add_trace(go.Bar(
        x=df["Expiration"], y=df["Put GEX ($B)"],
        name="Put GEX", marker_color="#E85C5C", opacity=0.85,
        hovertemplate="%{x}<br>Put GEX: %{y:.2f}$B<extra></extra>",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

    # ── Smart y-axis range ──
    all_vals = pd.concat([df["Call GEX ($B)"], df["Put GEX ($B)"]]).dropna()
    if len(all_vals) > 2:
        q1, q3 = all_vals.quantile(0.05), all_vals.quantile(0.95)
        iqr = q3 - q1
        y_lo = min(all_vals.min(), q1 - 1.5 * iqr)
        y_hi = max(all_vals.max(), q3 + 1.5 * iqr)
        pad = (y_hi - y_lo) * 0.1
        y_range = [y_lo - pad, y_hi + pad]
    else:
        y_range = None

    fig.update_layout(
        barmode="relative",
        title=f"{ticker} GEX by Expiration (next {max_months}mo)",
        xaxis_title="Expiration",
        yaxis_title="GEX ($ Billions)",
        template="plotly_dark",
        height=400,
        legend=dict(orientation="h", y=1.05),
        yaxis=dict(
            zeroline=True, zerolinecolor="rgba(255,255,255,0.4)",
            range=y_range,
        ),
    )
    return fig


def plot_dex_by_expiration(calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                           spot: float, ticker: str,
                           max_months: int = 6) -> go.Figure:
    """Bar chart of call DEX (up) and put DEX (down) per expiration date.

    Args:
        max_months: Only show expirations within this many months to keep
                    the chart readable.  Set to 0 to show all.
    """
    df = aggregate_dex_by_expiration(calls_df, puts_df, spot)
    if df.empty:
        return go.Figure()

    # ── Trim to near-term expirations so bars are visible ──
    if max_months > 0:
        from datetime import timedelta
        cutoff = (date.today() + timedelta(days=max_months * 30)).isoformat()
        df = df[df["Expiration"] <= cutoff]
        if df.empty:
            df = aggregate_dex_by_expiration(calls_df, puts_df, spot).head(12)

    fig = go.Figure()

    # Call DEX bars (positive / upward)
    fig.add_trace(go.Bar(
        x=df["Expiration"], y=df["Call DEX (M)"],
        name="Call DEX", marker_color="#4C9BE8", opacity=0.85,
        hovertemplate="%{x}<br>Call DEX: %{y:.1f}M shares<extra></extra>",
    ))
    # Put DEX bars (negative / downward)
    fig.add_trace(go.Bar(
        x=df["Expiration"], y=df["Put DEX (M)"],
        name="Put DEX", marker_color="#E85C5C", opacity=0.85,
        hovertemplate="%{x}<br>Put DEX: %{y:.1f}M shares<extra></extra>",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

    # ── Smart y-axis range to avoid outlier squishing ──
    all_vals = pd.concat([df["Call DEX (M)"], df["Put DEX (M)"]]).dropna()
    if len(all_vals) > 2:
        q1, q3 = all_vals.quantile(0.05), all_vals.quantile(0.95)
        iqr = q3 - q1
        y_lo = min(all_vals.min(), q1 - 1.5 * iqr)
        y_hi = max(all_vals.max(), q3 + 1.5 * iqr)
        pad = (y_hi - y_lo) * 0.1
        y_range = [y_lo - pad, y_hi + pad]
    else:
        y_range = None

    fig.update_layout(
        barmode="relative",
        title=f"{ticker} DEX by Expiration (next {max_months}mo)",
        xaxis_title="Expiration",
        yaxis_title="DEX (Millions of Shares)",
        template="plotly_dark",
        height=400,
        legend=dict(orientation="h", y=1.05),
        yaxis=dict(
            zeroline=True, zerolinecolor="rgba(255,255,255,0.4)",
            range=y_range,
        ),
    )
    return fig


def plot_dex_profile(dex_df: pd.DataFrame, spot: float, ticker: str,
                     strike_range_pct: float = 0.10,
                     call_wall: float | None = None,
                     put_wall: float | None = None) -> go.Figure:
    """
    Bar chart of dealer Delta Exposure (DEX) per strike with rangeslider.

    Shows all strikes with meaningful activity (auto-trimmed), initial
    view focused on ±strike_range_pct around spot.

    Positive net DEX = dealers need to sell shares (bearish hedge pressure).
    Negative net DEX = dealers need to buy shares (bullish hedge pressure).
    """
    if dex_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No DEX data for {ticker}", template="plotly_dark")
        return fig

    # ── Trim to strikes with meaningful activity ──
    sub = _trim_to_activity(dex_df, "net_dex_m", spot)

    fig = go.Figure()

    # Call DEX bars
    fig.add_trace(go.Bar(
        x=sub["strike"], y=sub["call_dex_m"],
        name="Call DEX", marker_color="#4C9BE8", opacity=0.8,
        hovertemplate="Strike: %{x}<br>Call DEX: %{y:.2f}M shares<extra></extra>",
    ))
    # Put DEX bars
    fig.add_trace(go.Bar(
        x=sub["strike"], y=sub["put_dex_m"],
        name="Put DEX", marker_color="#E85C5C", opacity=0.8,
        hovertemplate="Strike: %{x}<br>Put DEX: %{y:.2f}M shares<extra></extra>",
    ))
    # Net DEX line
    fig.add_trace(go.Scatter(
        x=sub["strike"], y=sub["net_dex_m"],
        name="Net DEX", mode="lines",
        line=dict(color="#F5E642", width=2),
        hovertemplate="Strike: %{x}<br>Net DEX: %{y:.2f}M shares<extra></extra>",
    ))

    fig.add_vline(
        x=spot, line_dash="dash", line_color="white", line_width=2,
        annotation_text=f"Spot: {spot:.2f}",
        annotation_position="top right", annotation_font_color="white",
    )

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

    # ── Initial zoom: ±range% of spot, expanded to include key levels ──
    view_lo = spot * (1 - strike_range_pct)
    view_hi = spot * (1 + strike_range_pct)
    pad = spot * 0.02
    if call_wall is not None and call_wall > view_hi:
        view_hi = call_wall + pad
    if put_wall is not None and put_wall < view_lo:
        view_lo = put_wall - pad

    fig.update_layout(
        barmode="relative",
        title=f"{ticker} Dealer Delta Exposure (DEX)",
        xaxis_title="Strike Price",
        yaxis_title="DEX (Millions of Shares)",
        template="plotly_dark",
        height=450,
        legend=dict(orientation="h", y=1.05),
        yaxis=dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.4)"),
        xaxis=dict(
            range=[view_lo, view_hi],
            rangeslider=dict(visible=True, thickness=0.06),
        ),
    )
    return fig


def plot_iv_skew(iv_df: pd.DataFrame, spot: float, ticker: str,
                 strike_range_pct: float = 0.15) -> go.Figure:
    """
    IV smile/skew chart: implied volatility vs strike for calls and puts.

    Shows the classic volatility smile/skew with:
    - Call IV (blue line)
    - Put IV (red line)
    - IV skew (put − call) as shaded area
    - Spot price vertical line
    """
    if iv_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No IV data for {ticker}", template="plotly_dark")
        return fig

    lo = spot * (1 - strike_range_pct)
    hi = spot * (1 + strike_range_pct)
    sub = iv_df[(iv_df["strike"] >= lo) & (iv_df["strike"] <= hi)].copy()
    if sub.empty:
        sub = iv_df.copy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.06,
        subplot_titles=[f"{ticker} Implied Volatility Skew", "Put-Call IV Spread"],
    )

    # Call IV
    if "call_iv" in sub.columns:
        valid_calls = sub.dropna(subset=["call_iv"])
        fig.add_trace(go.Scatter(
            x=valid_calls["strike"], y=valid_calls["call_iv"] * 100,
            name="Call IV", mode="lines+markers",
            line=dict(color="#4C9BE8", width=2),
            marker=dict(size=3),
            hovertemplate="Strike: %{x}<br>Call IV: %{y:.1f}%<extra></extra>",
        ), row=1, col=1)

    # Put IV
    if "put_iv" in sub.columns:
        valid_puts = sub.dropna(subset=["put_iv"])
        fig.add_trace(go.Scatter(
            x=valid_puts["strike"], y=valid_puts["put_iv"] * 100,
            name="Put IV", mode="lines+markers",
            line=dict(color="#E85C5C", width=2),
            marker=dict(size=3),
            hovertemplate="Strike: %{x}<br>Put IV: %{y:.1f}%<extra></extra>",
        ), row=1, col=1)

    # IV Skew (put - call)
    if "iv_skew" in sub.columns:
        valid_skew = sub.dropna(subset=["iv_skew"])
        colors = ["#E85C5C" if v > 0 else "#4C9BE8" for v in valid_skew["iv_skew"]]
        fig.add_trace(go.Bar(
            x=valid_skew["strike"], y=valid_skew["iv_skew"] * 100,
            name="Put−Call Skew", marker_color=colors, opacity=0.7,
            hovertemplate="Strike: %{x}<br>Skew: %{y:+.1f}pp<extra></extra>",
        ), row=2, col=1)

    # Spot line on both subplots
    for row in [1, 2]:
        fig.add_vline(
            x=spot, line_dash="dash", line_color="white", line_width=1.5,
            row=row, col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", y=1.08),
        showlegend=True,
    )
    fig.update_yaxes(title_text="IV (%)", row=1, col=1)
    fig.update_yaxes(title_text="Skew (pp)", zeroline=True,
                     zerolinecolor="rgba(255,255,255,0.3)", row=2, col=1)
    fig.update_xaxes(title_text="Strike Price", row=2, col=1)

    return fig


def plot_atm_iv_term_structure(term_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Plot ATM IV across expirations to show the underlying's options IV term structure."""
    if term_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No {ticker} Options ATM IV Term Structure", template="plotly_dark")
        return fig

    fig = go.Figure()

    def _customdata(frame: pd.DataFrame, strike_col: str) -> np.ndarray:
        strike_vals = frame[strike_col] if strike_col in frame.columns else pd.Series(np.nan, index=frame.index)
        return np.column_stack([frame["expiration"].astype(str).values, strike_vals.values])

    if "call_atm_iv" in term_df.columns and term_df["call_atm_iv"].notna().any():
        calls = term_df.dropna(subset=["call_atm_iv"])
        fig.add_trace(go.Scatter(
            x=calls["dte"], y=calls["call_atm_iv"] * 100,
            mode="lines+markers",
            name="Call ATM IV",
            line=dict(color="#4C9BE8", width=2),
            marker=dict(size=7),
            customdata=_customdata(calls, "call_atm_strike"),
            hovertemplate="DTE: %{x}<br>Expiration: %{customdata[0]}<br>Call ATM IV: %{y:.1f}%<br>ATM Strike: %{customdata[1]:.1f}<extra></extra>",
        ))

    if "put_atm_iv" in term_df.columns and term_df["put_atm_iv"].notna().any():
        puts = term_df.dropna(subset=["put_atm_iv"])
        fig.add_trace(go.Scatter(
            x=puts["dte"], y=puts["put_atm_iv"] * 100,
            mode="lines+markers",
            name="Put ATM IV",
            line=dict(color="#E85C5C", width=2),
            marker=dict(size=7),
            customdata=_customdata(puts, "put_atm_strike"),
            hovertemplate="DTE: %{x}<br>Expiration: %{customdata[0]}<br>Put ATM IV: %{y:.1f}%<br>ATM Strike: %{customdata[1]:.1f}<extra></extra>",
        ))

    avg = term_df.dropna(subset=["atm_iv"])
    fig.add_trace(go.Scatter(
        x=avg["dte"], y=avg["atm_iv"] * 100,
        mode="lines+markers",
        name="Average ATM IV",
        line=dict(color="#F5E642", width=3),
        marker=dict(size=8, symbol="diamond"),
        customdata=_customdata(avg, "atm_strike"),
        hovertemplate="DTE: %{x}<br>Expiration: %{customdata[0]}<br>ATM IV: %{y:.1f}%<br>ATM Strike: %{customdata[1]:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title=f"{ticker} Options ATM IV Term Structure",
        xaxis_title="Days to Expiration",
        yaxis_title="Implied Volatility (%)",
        template="plotly_dark",
        height=430,
        legend=dict(orientation="h", y=1.06),
        hovermode="x unified",
    )
    fig.update_xaxes(tickmode="linear")
    return fig


def plot_atm_iv_term_structure_comparison(curves: dict[str, pd.DataFrame], ticker: str) -> go.Figure:
    """Plot current and historical ATM IV term-structure curves together."""
    valid = {label: df for label, df in curves.items() if df is not None and not df.empty}
    if not valid:
        fig = go.Figure()
        fig.update_layout(title=f"No {ticker} Options ATM IV Term Structure", template="plotly_dark")
        return fig

    fig = go.Figure()
    palette = {
        "Current": "#F5E642",
        "1D Ago": "#4C9BE8",
        "2D Ago": "#7BB6F0",
        "3D Ago": "#A6CFF7",
        "1W Ago": "#F28C38",
        "2W Ago": "#F5B267",
        "1M Ago": "#E85C5C",
    }

    for idx, (label, df) in enumerate(valid.items()):
        sub = df.dropna(subset=["atm_iv"]).copy()
        if sub.empty:
            continue
        color = palette.get(label, None)
        fig.add_trace(go.Scatter(
            x=sub["dte"],
            y=sub["atm_iv"] * 100,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=3 if label == "Current" else 2, dash="solid" if label == "Current" else "dot"),
            marker=dict(size=8 if label == "Current" else 6),
            customdata=np.column_stack([sub["expiration"].astype(str).values, sub.get("atm_strike", pd.Series(np.nan, index=sub.index)).values]),
            hovertemplate="Curve: %{fullData.name}<br>DTE: %{x}<br>Expiration: %{customdata[0]}<br>ATM IV: %{y:.1f}%<br>ATM Strike: %{customdata[1]:.1f}<extra></extra>",
        ))

    fig.update_layout(
        title=f"{ticker} Options ATM IV Term Structure Comparison",
        xaxis_title="Days to Expiration",
        yaxis_title="Implied Volatility (%)",
        template="plotly_dark",
        height=450,
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
    )
    fig.update_xaxes(tickmode="linear")
    return fig
