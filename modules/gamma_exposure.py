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


def plot_gamma_index_timeline(ticker: str) -> go.Figure:
    """
    Line chart of the Gamma Index over time, coloured by positive (green)
    vs negative (red) regime, with a zero reference line.
    """
    df = load_gamma_index_history(ticker)

    if df.empty or "gamma_index" not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title=f"Gamma Index Timeline — {ticker} (no history yet)",
            template="plotly_dark", height=350,
            annotations=[dict(
                text="History builds up one data-point per day.<br>Check back tomorrow!",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16, color="#8B9BBF"),
            )],
        )
        return fig

    # Colour fill: green above zero, red below
    pos = df["gamma_index"].clip(lower=0)
    neg = df["gamma_index"].clip(upper=0)

    fig = go.Figure()

    # Positive area
    fig.add_trace(go.Scatter(
        x=df["date"], y=pos,
        fill="tozeroy", fillcolor="rgba(95,201,123,0.25)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    # Negative area
    fig.add_trace(go.Scatter(
        x=df["date"], y=neg,
        fill="tozeroy", fillcolor="rgba(232,92,92,0.25)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    # Main line
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["gamma_index"],
        mode="lines+markers",
        name="Gamma Index",
        line=dict(color="#F5E642", width=2),
        marker=dict(size=5),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>GI: %{y:+.3f}B<extra></extra>",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")

    fig.update_layout(
        title=f"{ticker} Gamma Index Timeline",
        xaxis_title="Date",
        yaxis_title="Gamma Index ($B)",
        template="plotly_dark",
        height=350,
        legend=dict(orientation="h", y=1.05),
        yaxis=dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.4)"),
    )
    return fig


# ── Charts ────────────────────────────────────────────────────────────────────

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
