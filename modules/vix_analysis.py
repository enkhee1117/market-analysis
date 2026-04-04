import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


VIX_REGIMES = [
    (0,  15,  "#5FC97B", "Low (<15)"),
    (15, 20,  "#F5E642", "Moderate (15-20)"),
    (20, 30,  "#F28C38", "Elevated (20-30)"),
    (30, 999, "#E85C5C", "High (>30)"),
]

TICKERS = {"VIX": "^VIX", "VVIX": "^VVIX", "SVIX": "SVIX"}
TERM_STRUCTURE_TICKERS = {
    "VIX9D": "^VIX9D",
    "VIX": "^VIX",
    "VIX3M": "^VIX3M",
    "VIX6M": "^VIX6M",
    "VIX1Y": "^VIX1Y",
}
TERM_STRUCTURE_LABELS = {
    "VIX9D": "9D",
    "VIX": "1M",
    "VIX3M": "3M",
    "VIX6M": "6M",
    "VIX1Y": "1Y",
}
TERM_STRUCTURE_ORDER = ["VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y"]


def compute_vix_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns: VVIX/VIX ratio, rolling z-scores, VIX pct change."""
    out = df.copy()
    if "VIX" in out.columns and "VVIX" in out.columns:
        out["VVIX_VIX_Ratio"] = out["VVIX"] / out["VIX"]
    if "VIX" in out.columns:
        roll = out["VIX"].rolling(20)
        out["VIX_ZScore_20d"] = (out["VIX"] - roll.mean()) / roll.std()
        out["VIX_Chg_1d"] = out["VIX"].pct_change() * 100
        out["VIX_Chg_5d"] = out["VIX"].pct_change(5) * 100
    return out


def _add_regime_shading(fig, vix_series: pd.Series, row: int = 1):
    """Add background color bands to a figure based on VIX regime."""
    for lo, hi, color, _ in VIX_REGIMES:
        mask = (vix_series >= lo) & (vix_series < hi)
        # Find contiguous segments
        in_regime = False
        start = None
        dates = vix_series.index
        vals  = mask.values
        for i, (dt, v) in enumerate(zip(dates, vals)):
            if v and not in_regime:
                in_regime = True
                start = dt
            elif not v and in_regime:
                in_regime = False
                fig.add_vrect(
                    x0=start, x1=dt,
                    fillcolor=color, opacity=0.08,
                    layer="below", line_width=0,
                    row=row, col=1,
                )
        if in_regime and start is not None:
            fig.add_vrect(
                x0=start, x1=dates[-1],
                fillcolor=color, opacity=0.08,
                layer="below", line_width=0,
                row=row, col=1,
            )


def plot_vix_panel(df: pd.DataFrame) -> go.Figure:
    """
    Multi-panel chart:
    Row 1: VIX with regime shading + SVIX on secondary axis
    Row 2: VVIX
    """
    has_vix  = "VIX"  in df.columns
    has_vvix = "VVIX" in df.columns
    has_svix = "SVIX" in df.columns

    rows = 2 if has_vvix else 1
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4] if rows == 2 else [1.0],
        specs=[[{"secondary_y": True}]] + ([[{"secondary_y": False}]] if rows == 2 else []),
        vertical_spacing=0.08,
    )

    # VIX
    if has_vix:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VIX"],
            name="VIX", mode="lines",
            line=dict(color="#E85C5C", width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>VIX: %{y:.2f}<extra></extra>",
        ), row=1, col=1, secondary_y=False)

        _add_regime_shading(fig, df["VIX"], row=1)

        # Regime level lines
        for lvl, color, label in [(15, "#F5E642", "15"), (20, "#F28C38", "20"), (30, "#E85C5C", "30")]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=color,
                          line_width=1, opacity=0.5, row=1, col=1)

    # SVIX on secondary y-axis
    if has_svix:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SVIX"],
            name="SVIX", mode="lines",
            line=dict(color="#A575E8", width=1.5, dash="dot"),
            opacity=0.8,
            hovertemplate="Date: %{x|%Y-%m-%d}<br>SVIX: %{y:.2f}<extra></extra>",
        ), row=1, col=1, secondary_y=True)

    # VVIX
    if has_vvix:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VVIX"],
            name="VVIX", mode="lines",
            line=dict(color="#4C9BE8", width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>VVIX: %{y:.2f}<extra></extra>",
        ), row=2, col=1)

    fig.update_layout(
        title="VIX / VVIX / SVIX Analysis",
        template="plotly_dark",
        height=600,
        legend=dict(orientation="h", y=1.04),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="VIX",  row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="SVIX", row=1, col=1, secondary_y=True)
    if has_vvix:
        fig.update_yaxes(title_text="VVIX", row=2, col=1)
    return fig


def plot_vvix_vix_ratio(df: pd.DataFrame) -> go.Figure:
    """VVIX/VIX ratio with 1-std bands and highlighted extremes."""
    if "VVIX_VIX_Ratio" not in df.columns:
        return go.Figure()

    ratio = df["VVIX_VIX_Ratio"].dropna()
    mean  = ratio.mean()
    std   = ratio.std()

    fig = go.Figure()

    # Bands
    x = ratio.index.tolist()
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=([mean + std] * len(x)) + ([mean - std] * len(x)),
        fill="toself",
        fillcolor="rgba(76,155,232,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1 Std",
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=ratio.index, y=ratio.values,
        name="VVIX/VIX Ratio",
        mode="lines",
        line=dict(color="#4C9BE8", width=2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Ratio: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=mean,        line_dash="dash",  line_color="white",    line_width=1,
                  annotation_text=f"Mean: {mean:.2f}", annotation_position="right")
    fig.add_hline(y=mean + std,  line_dash="dot",   line_color="#4C9BE8",  line_width=1)
    fig.add_hline(y=mean - std,  line_dash="dot",   line_color="#4C9BE8",  line_width=1)
    fig.add_hline(y=mean + 2*std, line_dash="dot",  line_color="#F28C38",  line_width=1,
                  annotation_text="+2σ", annotation_position="right")

    fig.update_layout(
        title="VVIX/VIX Ratio (Vol-of-Vol Stress Indicator)",
        xaxis_title="Date",
        yaxis_title="VVIX / VIX",
        template="plotly_dark",
        height=380,
    )
    return fig


def plot_vix_zscore(df: pd.DataFrame) -> go.Figure:
    """VIX 20-day rolling z-score."""
    if "VIX_ZScore_20d" not in df.columns:
        return go.Figure()

    z = df["VIX_ZScore_20d"].dropna()

    colors = ["#E85C5C" if v > 1.5 else "#5FC97B" if v < -1.5 else "#4C9BE8" for v in z]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=z.index, y=z.values,
        name="VIX Z-Score (20d)",
        marker_color=colors,
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Z-Score: %{y:.2f}<extra></extra>",
    ))

    for lvl, color, label in [(2, "#E85C5C", "+2σ"), (-2, "#5FC97B", "-2σ"),
                               (1, "#F28C38", "+1σ"), (-1, "#F5E642", "-1σ")]:
        fig.add_hline(y=lvl, line_dash="dot", line_color=color, line_width=1,
                      annotation_text=label, annotation_position="right")

    fig.update_layout(
        title="VIX 20-Day Rolling Z-Score (Overbought/Oversold Vol)",
        xaxis_title="Date",
        yaxis_title="Z-Score",
        template="plotly_dark",
        height=380,
    )
    return fig


def compute_vix_term_structure_snapshot(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Build a VIX term-structure snapshot similar to VIX Central.

    Uses CBOE spot/term indices instead of futures contracts:
    9D, 1M, 3M, 6M, 1Y.
    """
    available = [col for col in TERM_STRUCTURE_ORDER if col in df.columns]
    if not available:
        return pd.DataFrame(), {}

    term = df[available].dropna(how="all").copy()
    if term.empty:
        return pd.DataFrame(), {}

    latest_idx = term.dropna(how="all").index.max()
    current_row = term.loc[latest_idx].dropna()

    previous_idx = None
    prior_candidates = term.index[term.index < latest_idx]
    if len(prior_candidates) > 0:
        previous_idx = prior_candidates[-1]
        previous_row = term.loc[previous_idx].reindex(current_row.index)
    else:
        previous_row = pd.Series(index=current_row.index, dtype=float)

    snapshot = pd.DataFrame({
        "tenor_key": current_row.index,
        "Tenor": [TERM_STRUCTURE_LABELS[k] for k in current_row.index],
        "Current": current_row.values.astype(float),
        "Previous": previous_row.values.astype(float),
    })
    front_month = current_row.get("VIX", np.nan)
    if np.isfinite(front_month) and front_month != 0:
        snapshot["Premium vs 1M %"] = (snapshot["Current"] / front_month - 1) * 100
    else:
        snapshot["Premium vs 1M %"] = np.nan

    current_curve = snapshot["Current"].tolist()
    front = float(front_month)
    back = float(current_row.get("VIX3M", np.nan)) if "VIX3M" in current_row.index else np.nan
    slope = back - front if np.isfinite(front) and np.isfinite(back) else np.nan

    regime = "Mixed"
    if np.isfinite(slope):
        regime = "Contango" if slope > 0 else "Backwardation" if slope < 0 else "Flat"

    inversion_count = 0
    for a, b in zip(current_curve, current_curve[1:]):
        if np.isfinite(a) and np.isfinite(b) and b < a:
            inversion_count += 1

    summary = {
        "current_date": latest_idx,
        "previous_date": previous_idx,
        "front_month": round(front, 2) if np.isfinite(front) else None,
        "three_month": round(back, 2) if np.isfinite(back) else None,
        "slope_1m_3m": round(float(slope), 2) if np.isfinite(slope) else None,
        "regime": regime,
        "inversion_count": inversion_count,
    }
    return snapshot, summary


def plot_vix_term_structure_curve(df: pd.DataFrame) -> go.Figure:
    """Plot the current VIX term structure in a VIX Central-style curve view."""
    snapshot, summary = compute_vix_term_structure_snapshot(df)
    if snapshot.empty:
        return go.Figure()

    regime = summary.get("regime", "Mixed")
    regime_color = "#5FC97B" if regime == "Contango" else "#E85C5C" if regime == "Backwardation" else "#F5E642"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=snapshot["Tenor"],
        y=snapshot["Current"],
        mode="lines+markers+text",
        name="Current",
        line=dict(color=regime_color, width=3),
        marker=dict(size=10, color=regime_color),
        text=[f"{v:.1f}" for v in snapshot["Current"]],
        textposition="top center",
        hovertemplate="Tenor: %{x}<br>Current: %{y:.2f}<extra></extra>",
    ))

    if snapshot["Previous"].notna().any():
        prev_label = "Previous"
        prev_date = summary.get("previous_date")
        if prev_date is not None:
            prev_label = f"Previous ({pd.to_datetime(prev_date).strftime('%Y-%m-%d')})"
        fig.add_trace(go.Scatter(
            x=snapshot["Tenor"],
            y=snapshot["Previous"],
            mode="lines+markers",
            name=prev_label,
            line=dict(color="rgba(180,180,180,0.7)", width=2, dash="dot"),
            marker=dict(size=7, color="rgba(180,180,180,0.8)"),
            hovertemplate="Tenor: %{x}<br>Previous: %{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        title=f"VIX Term Structure ({regime})",
        xaxis_title="Tenor",
        yaxis_title="Volatility Index Level",
        template="plotly_dark",
        height=420,
        legend=dict(orientation="h", y=1.06),
        hovermode="x unified",
        annotations=[
            dict(
                text="Index-based proxy using CBOE 9D / 1M / 3M / 6M / 1Y volatility indices",
                xref="paper", yref="paper", x=0.5, y=-0.18,
                showarrow=False, font=dict(size=11, color="#8B9BBF"),
            )
        ],
    )
    return fig


def plot_correlation_matrix(df: pd.DataFrame, spy_df: pd.DataFrame) -> go.Figure:
    """Heatmap: correlation between VIX, VVIX, SVIX, and SPY daily returns."""
    combined = df.copy()
    if not spy_df.empty and "Close" in spy_df.columns:
        spy_ret = spy_df["Close"].squeeze().pct_change() * 100
        spy_ret.name = "SPY Return %"
        combined = combined.join(spy_ret, how="left")

    corr = combined.dropna().corr()
    if corr.empty:
        return go.Figure()

    labels = corr.columns.tolist()
    z = corr.values

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels, y=labels,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        colorbar=dict(title="Correlation"),
    ))
    fig.update_layout(
        title="Correlation Matrix: VIX / VVIX / SVIX / SPY Returns",
        template="plotly_dark",
        height=400,
    )
    return fig


def vix_summary_stats(df: pd.DataFrame) -> dict:
    """Return key summary stats for VIX."""
    if "VIX" not in df.columns:
        return {}
    vix = df["VIX"].dropna()
    return {
        "current":    round(float(vix.iloc[-1]), 2),
        "mean":       round(float(vix.mean()), 2),
        "median":     round(float(vix.median()), 2),
        "percentile": round(float((vix < vix.iloc[-1]).mean() * 100), 1),
        "52w_high":   round(float(vix.tail(252).max()), 2),
        "52w_low":    round(float(vix.tail(252).min()), 2),
        "1d_chg":     round(float(vix.pct_change().iloc[-1] * 100), 2) if len(vix) > 1 else None,
    }
