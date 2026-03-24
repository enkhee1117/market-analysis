import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
from scipy.stats import norm


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


def gex_flip_point(gex_df: pd.DataFrame) -> float | None:
    """
    Find the gamma flip point: the strike where net GEX crosses zero.
    Returns the interpolated strike price, or None if no crossing exists.
    """
    if gex_df.empty or "net_gex" not in gex_df.columns:
        return None

    df = gex_df.sort_values("strike")
    net = df["net_gex"].values
    strikes = df["strike"].values

    for i in range(len(net) - 1):
        if net[i] * net[i + 1] < 0:
            # Linear interpolation
            s1, s2 = strikes[i], strikes[i + 1]
            g1, g2 = net[i], net[i + 1]
            flip = s1 - g1 * (s2 - s1) / (g2 - g1)
            return round(flip, 2)
    return None


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


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_gex_profile(gex_df: pd.DataFrame, spot: float, ticker: str,
                     strike_range_pct: float = 0.10) -> go.Figure:
    """
    Bar chart showing GEX per strike, color-coded by call (positive) vs put (negative).
    Vertical lines for spot and gamma flip point.
    """
    if gex_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No options data available for {ticker}", template="plotly_dark")
        return fig

    # Filter to ±10% of spot for readability
    lo = spot * (1 - strike_range_pct)
    hi = spot * (1 + strike_range_pct)
    sub = gex_df[(gex_df["strike"] >= lo) & (gex_df["strike"] <= hi)].copy()

    if sub.empty:
        sub = gex_df.copy()

    flip = gex_flip_point(gex_df)

    fig = go.Figure()

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

    # Net GEX line
    fig.add_trace(go.Scatter(
        x=sub["strike"],
        y=sub["net_gex_b"],
        name="Net GEX",
        mode="lines+markers",
        line=dict(color="#F5E642", width=2),
        marker=dict(size=4),
        hovertemplate="Strike: %{x}<br>Net GEX: %{y:.3f}B<extra></extra>",
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

    fig.update_layout(
        barmode="relative",
        title=f"{ticker} Dealer Gamma Exposure (GEX) Profile",
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
