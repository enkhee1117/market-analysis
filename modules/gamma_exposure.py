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

    # Parse expiration → time to expiry in years
    today = datetime.now().date()
    if "expiration" in df.columns:
        def _tte(exp_str):
            try:
                exp = datetime.strptime(str(exp_str), "%Y-%m-%d").date()
                days = max((exp - today).days, 0)
                return days / 365.0
            except Exception:
                return 0.0
        df["T"] = df["expiration"].apply(_tte)
    else:
        df["T"] = 30 / 365.0  # fallback 30 days

    iv_col = "impliedvolatility" if "impliedvolatility" in df.columns else None
    if iv_col and "strike" in df.columns:
        df["gamma"] = df.apply(
            lambda row: _bs_gamma(spot, row["strike"], row["T"], r, row[iv_col])
            if row[iv_col] > 0 else 0.0,
            axis=1,
        )
    return df


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

    rows = []
    for _, row in calls.iterrows():
        gex = row["gamma"] * row["openInterest"] * 100 * (spot ** 2) * 0.01
        rows.append({"strike": row["strike"], "call_gex": gex, "put_gex": 0.0})

    for _, row in puts.iterrows():
        gex = -row["gamma"] * row["openInterest"] * 100 * (spot ** 2) * 0.01
        rows.append({"strike": row["strike"], "call_gex": 0.0, "put_gex": gex})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
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


def compute_historical_gamma_proxy(spy_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a historical Gamma Environment Proxy from VIX and SPY data.

    The proxy is based on established options market microstructure:
    - When VIX is low / declining → dealers are likely in positive gamma
      (net long gamma via sold puts that are far OTM).  Market is pinned,
      realized vol < implied vol.
    - When VIX is high / spiking → dealers are likely in negative gamma
      (short gamma from bought protection / unwound positions).  Moves
      are amplified, realized vol > implied vol.

    Proxy signal per day
    --------------------
    1. VIX z-score (20d rolling) — captures regime relative to recent history.
    2. Implied-vs-Realized spread — VIX minus 20d realized vol of SPY.
       Positive spread → implied > realized → positive-gamma pinning.
    3. Combine:  proxy = −vix_z × (1 + iv_rv_spread / 100)
       Flip sign because low-VIX = positive-gamma.
    4. Normalise to have the same scale as real gamma index where overlap exists.

    Returns a DataFrame with columns: date, gamma_proxy, proxy_condition.
    """
    if spy_df is None or spy_df.empty or vix_df is None or vix_df.empty:
        return pd.DataFrame()

    # Prepare SPY data
    spy = spy_df.copy()
    if "Close" not in spy.columns:
        return pd.DataFrame()
    spy = spy[["Close"]].dropna()
    spy.index = pd.to_datetime(spy.index)

    # Prepare VIX data
    vix = vix_df.copy()
    if isinstance(vix, pd.DataFrame):
        if "VIX" in vix.columns:
            vix = vix[["VIX"]].dropna()
            vix.columns = ["vix"]
        elif "Close" in vix.columns:
            vix = vix[["Close"]].dropna()
            vix.columns = ["vix"]
        else:
            return pd.DataFrame()
    vix.index = pd.to_datetime(vix.index)

    # Merge on date
    merged = spy.join(vix, how="inner").dropna()
    if len(merged) < 30:
        return pd.DataFrame()

    # 1. VIX z-score (20-day rolling)
    vix_roll = merged["vix"].rolling(20)
    merged["vix_z"] = (merged["vix"] - vix_roll.mean()) / vix_roll.std()

    # 2. Realized vol of SPY (20-day annualized)
    merged["spy_ret"] = merged["Close"].pct_change()
    merged["realized_vol"] = merged["spy_ret"].rolling(20).std() * np.sqrt(252) * 100

    # 3. IV - RV spread (VIX is annualized % already)
    merged["iv_rv_spread"] = merged["vix"] - merged["realized_vol"]

    # 4. Raw proxy: negative VIX z-score, amplified by IV-RV spread
    #    When VIX is below average (z < 0) → proxy is positive (positive gamma)
    #    When VIX is above average (z > 0) → proxy is negative (negative gamma)
    #    IV-RV spread > 0 amplifies positive gamma (dealers collecting premium)
    merged["raw_proxy"] = -merged["vix_z"] * (1 + merged["iv_rv_spread"].clip(-50, 50) / 100)

    # 5. Smooth with 5-day EMA for less noise
    merged["gamma_proxy"] = merged["raw_proxy"].ewm(span=5, adjust=False).mean()

    # Drop warmup period
    merged = merged.dropna(subset=["gamma_proxy"])

    # Build result
    result = pd.DataFrame({
        "date": merged.index,
        "gamma_proxy": merged["gamma_proxy"].values,
        "vix": merged["vix"].values,
        "realized_vol": merged["realized_vol"].values,
    }).reset_index(drop=True)

    result["proxy_condition"] = result["gamma_proxy"].apply(
        lambda x: "Positive (Stabilizing)" if x > 0 else "Negative (Destabilizing)"
    )

    return result


def calibrate_proxy_to_real(proxy_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale the proxy so its magnitude matches the real gamma index
    where we have overlapping data points.

    If no overlap exists, normalize proxy to have a reasonable scale
    (std ≈ 0.5B to match typical GEX magnitudes).
    """
    if proxy_df.empty:
        return proxy_df

    proxy = proxy_df.copy()

    if not real_df.empty and "gamma_index" in real_df.columns and len(real_df) >= 2:
        # Find overlapping dates
        real_dates = set(pd.to_datetime(real_df["date"]).dt.date)
        proxy["_date"] = pd.to_datetime(proxy["date"]).dt.date
        overlap = proxy[proxy["_date"].isin(real_dates)]

        if len(overlap) >= 1:
            # Scale factor: match the mean absolute magnitude
            real_vals = real_df.set_index(pd.to_datetime(real_df["date"]).dt.date)["gamma_index"]
            proxy_vals = overlap.set_index("_date")["gamma_proxy"]
            common = proxy_vals.index.intersection(real_vals.index)
            if len(common) >= 1:
                real_scale = real_vals.loc[common].abs().mean()
                proxy_scale = proxy_vals.loc[common].abs().mean()
                if proxy_scale > 0:
                    scale = real_scale / proxy_scale
                    proxy["gamma_proxy"] *= scale
                    proxy.drop(columns=["_date"], inplace=True)
                    return proxy

        proxy.drop(columns=["_date"], inplace=True)

    # No overlap: normalize to ~0.5B std (typical magnitude)
    proxy_std = proxy["gamma_proxy"].std()
    if proxy_std > 0:
        proxy["gamma_proxy"] = proxy["gamma_proxy"] / proxy_std * 0.5

    return proxy


def plot_gamma_index_timeline(ticker: str, proxy_df: pd.DataFrame | None = None) -> go.Figure:
    """
    Line chart of the Gamma Index over time.

    Shows:
    - Real Gamma Index (solid yellow line) — from daily snapshots.
    - Historical Proxy (dashed gray line) — from VIX/SPY-derived estimate.
    - Green/red fill for positive/negative gamma regimes.
    """
    real_df = load_gamma_index_history(ticker)
    has_real = not real_df.empty and "gamma_index" in real_df.columns
    has_proxy = proxy_df is not None and not proxy_df.empty

    if not has_real and not has_proxy:
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

    # ── Historical Proxy (background) ──
    if has_proxy:
        pdf = proxy_df.copy()
        pdf["date"] = pd.to_datetime(pdf["date"])

        # Calibrate proxy to real data if available
        if has_real:
            pdf = calibrate_proxy_to_real(pdf, real_df)

        # Positive/negative fill areas for proxy
        pos_proxy = pdf["gamma_proxy"].clip(lower=0)
        neg_proxy = pdf["gamma_proxy"].clip(upper=0)

        fig.add_trace(go.Scatter(
            x=pdf["date"], y=pos_proxy,
            fill="tozeroy", fillcolor="rgba(95,201,123,0.12)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=pdf["date"], y=neg_proxy,
            fill="tozeroy", fillcolor="rgba(232,92,92,0.12)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))

        # Proxy line
        fig.add_trace(go.Scatter(
            x=pdf["date"], y=pdf["gamma_proxy"],
            mode="lines",
            name="Gamma Proxy (VIX-derived)",
            line=dict(color="rgba(180,180,180,0.5)", width=1.5, dash="dot"),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Proxy: %{y:+.3f}<extra></extra>",
        ))

    # ── Real Gamma Index (foreground) ──
    if has_real:
        # Colour fill: green above zero, red below
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
        # Main real line
        fig.add_trace(go.Scatter(
            x=real_df["date"], y=real_df["gamma_index"],
            mode="lines+markers",
            name="Gamma Index (Real)",
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

    # Add annotation explaining proxy
    if has_proxy and not has_real:
        fig.add_annotation(
            text="Proxy estimate from VIX/SPY data · Real data accumulates daily",
            xref="paper", yref="paper", x=0.5, y=-0.12,
            showarrow=False, font=dict(size=11, color="#8B9BBF"),
        )
    elif has_proxy and has_real:
        fig.add_annotation(
            text="Dotted line = VIX-derived proxy · Solid line = real GEX data",
            xref="paper", yref="paper", x=0.5, y=-0.12,
            showarrow=False, font=dict(size=11, color="#8B9BBF"),
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


def plot_gex_profile(gex_df: pd.DataFrame, spot: float, ticker: str,
                     strike_range_pct: float = 0.10,
                     view_mode: str = "Call / Put") -> go.Figure:
    """
    Bar chart showing GEX per strike.

    view_mode:
      "Call / Put" — stacked call (blue) + put (red) bars with net GEX yellow line.
      "Net GEX"    — single bar series coloured green (positive) / red (negative).
    """
    if gex_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No options data available for {ticker}", template="plotly_dark")
        return fig

    # Filter to ±range% of spot for readability
    lo = spot * (1 - strike_range_pct)
    hi = spot * (1 + strike_range_pct)
    sub = gex_df[(gex_df["strike"] >= lo) & (gex_df["strike"] <= hi)].copy()

    if sub.empty:
        sub = gex_df.copy()

    flip = gex_flip_point(gex_df, spot)

    fig = go.Figure()

    if view_mode == "Net GEX":
        # Color each bar by sign: green positive, red negative
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
        # Call GEX bars (positive)
        fig.add_trace(go.Bar(
            x=sub["strike"],
            y=sub["call_gex_b"],
            name="Call GEX",
            marker_color="#4C9BE8",
            opacity=0.8,
            hovertemplate="Strike: %{x}<br>Call GEX: %{y:.3f}B<extra></extra>",
        ))

        # Put GEX bars (negative)
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
        x=spot,
        line_dash="dash",
        line_color="white",
        line_width=2,
        annotation_text=f"Spot: {spot:.2f}",
        annotation_position="top right",
        annotation_font_color="white",
    )

    # Gamma flip line
    if flip is not None:
        color = "#5FC97B" if flip > spot else "#F28C38"
        fig.add_vline(
            x=flip,
            line_dash="dot",
            line_color=color,
            line_width=2,
            annotation_text=f"Flip: {flip:.2f}",
            annotation_position="top left",
            annotation_font_color=color,
        )

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
    )
    return fig


def plot_gex_by_expiration(calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                           spot: float, ticker: str) -> go.Figure:
    """Bar chart of total net GEX per expiration date."""
    if calls_df is None or calls_df.empty:
        return go.Figure()

    rows = []
    for exp in calls_df["expiration"].unique():
        c = calls_df[calls_df["expiration"] == exp]
        p = puts_df[puts_df["expiration"] == exp] if puts_df is not None else pd.DataFrame()
        g = compute_gex(c, p, spot)
        if not g.empty:
            rows.append({
                "Expiration": exp,
                "Call GEX ($B)": g["call_gex_b"].sum(),
                "Put GEX ($B)":  g["put_gex_b"].sum(),
                "Net GEX ($B)":  g["net_gex_b"].sum(),
            })

    if not rows:
        return go.Figure()

    df = pd.DataFrame(rows).sort_values("Expiration")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Expiration"], y=df["Call GEX ($B)"],
        name="Call GEX", marker_color="#4C9BE8", opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        x=df["Expiration"], y=df["Put GEX ($B)"],
        name="Put GEX", marker_color="#E85C5C", opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=df["Expiration"], y=df["Net GEX ($B)"],
        name="Net GEX", mode="lines+markers",
        line=dict(color="#F5E642", width=2),
    ))

    fig.update_layout(
        barmode="relative",
        title=f"{ticker} GEX by Expiration",
        xaxis_title="Expiration",
        yaxis_title="GEX ($ Billions)",
        template="plotly_dark",
        height=400,
        legend=dict(orientation="h", y=1.05),
        yaxis=dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.4)"),
    )
    return fig
