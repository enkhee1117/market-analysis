"""
Market Analysis Dashboard
=========================
4 tabs:
  1. Day of Week Performance
  2. Gamma Exposure (GEX)
  3. VIX / Volatility Analysis
  4. SPY Seasonality
"""

import streamlit as st
import pandas as pd
import numpy as np

from modules.data_fetcher import (
    fetch_price_history,
    fetch_multi_tickers,
    fetch_options_chain,
    TIMEFRAME_MAP,
)
from modules.day_of_week import (
    compute_dow_returns,
    filter_by_timeframe,
    dow_summary,
    plot_dow_comparison,
    plot_win_rate_comparison,
    plot_dow_distribution,
    plot_cumulative_by_dow,
    TIMEFRAMES,
    RETURN_TYPES,
)
from modules.gamma_exposure import (
    compute_gex,
    gex_flip_point,
    total_gex_metrics,
    plot_gex_profile,
    plot_gex_by_expiration,
)
from modules.vix_analysis import (
    compute_vix_metrics,
    plot_vix_panel,
    plot_vvix_vix_ratio,
    plot_vix_zscore,
    plot_vix_term_structure_proxy,
    plot_correlation_matrix,
    vix_summary_stats,
)
from modules.seasonality import (
    compute_returns,
    monthly_seasonality,
    monthly_return_pivot,
    weekly_seasonality,
    intramonth_seasonality,
    plot_monthly_bar,
    plot_monthly_heatmap,
    plot_weekly_bar,
    plot_intramonth_bar,
    plot_annual_return_bar,
)


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Market Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #1E2535;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
        border: 1px solid #2D3A50;
    }
    .metric-label { font-size: 12px; color: #8B9BBF; margin-bottom: 4px; }
    .metric-value { font-size: 24px; font-weight: 700; color: #FFFFFF; }
    .metric-sub   { font-size: 12px; color: #8B9BBF; margin-top: 2px; }
    .positive { color: #5FC97B !important; }
    .negative { color: #E85C5C !important; }
    .neutral  { color: #F5E642 !important; }
</style>
""", unsafe_allow_html=True)


def colored_metric(label, value, suffix="", positive_is_good=True, sub=""):
    try:
        num = float(str(value).replace("%", "").replace("$", "").replace("B", ""))
        cls = "positive" if num >= 0 else "negative"
        if not positive_is_good:
            cls = "negative" if num >= 0 else "positive"
    except (ValueError, TypeError):
        cls = "neutral"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cls}">{value}{suffix}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Market Analysis")
st.sidebar.markdown("---")

st.sidebar.subheader("Global Settings")
spy_ticker = st.sidebar.selectbox("Primary ETF", ["SPY", "QQQ", "IWM"], index=0)
vix_period  = st.sidebar.selectbox("VIX History Period", ["1y", "2y", "3y", "5y"], index=1)
gex_ticker  = st.sidebar.radio("GEX Ticker", ["SPY", "SPX"], index=0,
                                help="SPX options = 10x multiplier vs SPY")

st.sidebar.markdown("---")
st.sidebar.caption("Data via Yahoo Finance · Refreshes every 5 min")
st.sidebar.caption("GEX assumes MM short calls, long puts")


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📅  Day of Week",
    "⚡  Gamma Exposure",
    "📊  VIX / Volatility",
    "🌀  Seasonality",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Day of Week Performance
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.title("Day of Week Performance")
    st.markdown("Analyze how SPY (or selected ETF) performs on specific weekdays across different timeframes.")

    # Return type selector
    return_type_label = st.radio(
        "Return Type",
        list(RETURN_TYPES.keys()),
        index=0,
        horizontal=True,
        help="Close-to-Close = full daily return. Intraday = open-to-close. Overnight = previous close-to-open.",
    )
    return_col = RETURN_TYPES[return_type_label]

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        selected_days = st.multiselect(
            "Days to Analyze",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            default=["Monday", "Tuesday", "Friday"],
        )
    with c2:
        selected_timeframes = st.multiselect(
            "Timeframes",
            list(TIMEFRAMES.keys()),
            default=["1 Month", "3 Months", "6 Months", "12 Months", "3 Years", "10 Years"],
        )
    with c3:
        dist_timeframe = st.selectbox(
            "Distribution Chart Timeframe",
            list(TIMEFRAMES.keys()),
            index=3,  # 12 Months default
        )

    if not selected_days:
        st.warning("Select at least one day.")
        st.stop()
    if not selected_timeframes:
        st.warning("Select at least one timeframe.")
        st.stop()

    with st.spinner(f"Fetching {spy_ticker} 10-year daily data..."):
        raw = fetch_price_history(spy_ticker, period="10y", interval="1d")

    if raw.empty:
        st.error(f"Could not fetch data for {spy_ticker}.")
        st.stop()

    dow_data = compute_dow_returns(raw)

    # Summary table for selected distribution timeframe
    filtered = filter_by_timeframe(dow_data, dist_timeframe)
    summary  = dow_summary(filtered, selected_days, return_col=return_col)

    if not summary.empty:
        st.subheader(f"Summary Stats — {dist_timeframe}  |  {return_type_label}")
        cols = st.columns(len(summary))
        for col, (_, row) in zip(cols, summary.iterrows()):
            with col:
                colored_metric(row["Day"], f"{row['Mean %']:+.3f}", "%",
                               sub=f"Win Rate: {row['Win Rate']}%  |  n={row['Count']}")
        def _color_val(v):
            try:
                f = float(v)
                if f > 0:
                    return "color: #5FC97B"
                elif f < 0:
                    return "color: #E85C5C"
            except (TypeError, ValueError):
                pass
            return ""
        st.dataframe(
            summary.style.format({
                "Mean %": "{:+.3f}", "Median %": "{:+.3f}",
                "Win Rate": "{:.1f}%", "Std %": "{:.3f}",
                "Best %": "{:+.2f}", "Worst %": "{:+.2f}",
            }).map(_color_val, subset=["Mean %", "Median %", "Best %", "Worst %"]),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")

    # Main comparison chart
    st.subheader("Mean Return by Timeframe")
    fig_cmp = plot_dow_comparison(dow_data, selected_days, selected_timeframes, return_col=return_col)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Win rate chart
    st.subheader("Win Rate by Timeframe")
    fig_wr = plot_win_rate_comparison(dow_data, selected_days, selected_timeframes, return_col=return_col)
    st.plotly_chart(fig_wr, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Return Distribution")
        fig_dist = plot_dow_distribution(dow_data, selected_days, dist_timeframe, return_col=return_col)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_b:
        st.subheader("Cumulative Return (Day-Only Strategy)")
        fig_cum = plot_cumulative_by_dow(dow_data, selected_days, dist_timeframe, return_col=return_col)
        st.plotly_chart(fig_cum, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Gamma Exposure
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.title(f"Dealer Gamma Exposure (GEX) — {gex_ticker}")
    st.markdown("""
    **Interpretation:**
    - **Positive GEX** → Dealers are net long gamma → sell rallies / buy dips → **stabilizing**
    - **Negative GEX** → Dealers are net short gamma → buy rallies / sell dips → **destabilizing / trending**
    - **Gamma Flip** = strike where net GEX crosses zero; acts as key support/resistance
    """)

    strike_range = st.slider("Strike Range Around Spot (±%)", 5, 20, 10, step=1)

    with st.spinner(f"Fetching {gex_ticker} options chain..."):
        calls_df, puts_df, spot = fetch_options_chain(gex_ticker)

    if calls_df is None or calls_df.empty or spot is None:
        st.error(f"Could not fetch options data for {gex_ticker}. Yahoo Finance may not support {gex_ticker} options directly.")
        if gex_ticker == "SPX":
            st.info("Try switching to SPY in the sidebar — SPX (^GSPC) options data availability varies.")
        st.stop()

    with st.spinner("Computing GEX..."):
        gex_df  = compute_gex(calls_df, puts_df, spot)
        flip    = gex_flip_point(gex_df)
        metrics = total_gex_metrics(gex_df)

    if gex_df.empty:
        st.warning("GEX data empty — options gamma/OI fields may be missing in the data.")
        st.stop()

    # Metrics row
    st.subheader("GEX Summary")
    m_cols = st.columns(6)
    with m_cols[0]: colored_metric("Spot Price", f"${spot:.2f}")
    with m_cols[1]:
        flip_str = f"${flip:.2f}" if flip else "N/A"
        colored_metric("Gamma Flip", flip_str, positive_is_good=False)
    with m_cols[2]:
        net = metrics.get("total_net_gex_b", 0)
        colored_metric("Net GEX", f"${net:.2f}B")
    with m_cols[3]:
        call_g = metrics.get("total_call_gex_b", 0)
        colored_metric("Call GEX", f"${call_g:.2f}B")
    with m_cols[4]:
        put_g = metrics.get("total_put_gex_b", 0)
        colored_metric("Put GEX", f"${put_g:.2f}B", positive_is_good=False)
    with m_cols[5]:
        if flip:
            diff = ((spot - flip) / spot) * 100
            colored_metric("Spot vs Flip", f"{diff:+.2f}%",
                          sub="above" if diff > 0 else "below")
        else:
            colored_metric("Spot vs Flip", "N/A")

    st.markdown("---")

    # GEX Profile
    fig_gex = plot_gex_profile(gex_df, spot, gex_ticker, strike_range_pct=strike_range / 100)
    st.plotly_chart(fig_gex, use_container_width=True)

    # GEX by expiration
    st.subheader("GEX by Expiration")
    fig_exp = plot_gex_by_expiration(calls_df, puts_df, spot, gex_ticker)
    st.plotly_chart(fig_exp, use_container_width=True)

    # Raw GEX data table
    with st.expander("Raw GEX Data Table"):
        lo = spot * (1 - strike_range / 100)
        hi = spot * (1 + strike_range / 100)
        sub_gex = gex_df[(gex_df["strike"] >= lo) & (gex_df["strike"] <= hi)].copy()
        sub_gex_disp = sub_gex[["strike", "call_gex_b", "put_gex_b", "net_gex_b"]].rename(
            columns={
                "strike": "Strike",
                "call_gex_b": "Call GEX ($B)",
                "put_gex_b":  "Put GEX ($B)",
                "net_gex_b":  "Net GEX ($B)",
            }
        )
        def _gex_color(v):
            try:
                f = float(v)
                return "color: #5FC97B" if f > 0 else ("color: #E85C5C" if f < 0 else "")
            except (TypeError, ValueError):
                return ""
        st.dataframe(
            sub_gex_disp.style.format({
                "Call GEX ($B)": "{:.4f}",
                "Put GEX ($B)":  "{:.4f}",
                "Net GEX ($B)":  "{:.4f}",
            }).map(_gex_color, subset=["Net GEX ($B)"]),
            use_container_width=True, hide_index=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — VIX / Volatility Analysis
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.title("VIX / VVIX / SVIX Analysis")
    st.markdown("""
    - **VIX** — 30-day implied volatility of S&P 500 options ("fear gauge")
    - **VVIX** — Volatility of VIX itself; measures the uncertainty of volatility
    - **SVIX** — 1x Short VIX Futures ETF; inversely tracks VIX futures
    """)

    with st.spinner("Fetching VIX / VVIX / SVIX data..."):
        vix_raw = fetch_multi_tickers(["^VIX", "^VVIX", "SVIX"],
                                       period=vix_period, interval="1d")
        spy_hist = fetch_price_history(spy_ticker, period=vix_period, interval="1d")

    if vix_raw.empty:
        st.error("Could not fetch VIX data.")
        st.stop()

    # Rename columns
    rename_map = {"^VIX": "VIX", "^VVIX": "VVIX", "SVIX": "SVIX"}
    vix_raw = vix_raw.rename(columns={k: v for k, v in rename_map.items() if k in vix_raw.columns})

    vix_df = compute_vix_metrics(vix_raw)
    stats  = vix_summary_stats(vix_df)

    # VIX summary metrics
    if stats:
        st.subheader("VIX Current Snapshot")
        sc = st.columns(6)
        with sc[0]: colored_metric("VIX Current", stats["current"],  positive_is_good=False)
        with sc[1]: colored_metric("52W High",     stats["52w_high"], positive_is_good=False)
        with sc[2]: colored_metric("52W Low",      stats["52w_low"],  positive_is_good=True)
        with sc[3]: colored_metric("Historical Mean", stats["mean"],  positive_is_good=False)
        with sc[4]: colored_metric("Percentile",   f"{stats['percentile']}%", positive_is_good=False,
                                   sub="vs full history")
        with sc[5]:
            chg = stats.get("1d_chg")
            colored_metric("1D Change", f"{chg:+.2f}%" if chg is not None else "N/A",
                           positive_is_good=False)

    st.markdown("---")

    # VIX Panel
    fig_vix = plot_vix_panel(vix_df)
    st.plotly_chart(fig_vix, use_container_width=True)

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("VVIX / VIX Ratio")
        fig_ratio = plot_vvix_vix_ratio(vix_df)
        st.plotly_chart(fig_ratio, use_container_width=True)

    with col_v2:
        st.subheader("VIX Z-Score (20-Day Rolling)")
        fig_z = plot_vix_zscore(vix_df)
        st.plotly_chart(fig_z, use_container_width=True)

    # Term structure proxy
    st.subheader("VIX Momentum / Term Structure Proxy")
    fig_ts = plot_vix_term_structure_proxy(vix_df)
    st.plotly_chart(fig_ts, use_container_width=True)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    fig_corr = plot_correlation_matrix(vix_df, spy_hist)
    st.plotly_chart(fig_corr, use_container_width=True)

    # VIX regime table
    with st.expander("VIX Regime Distribution"):
        if "VIX" in vix_df.columns:
            vix_vals = vix_df["VIX"].dropna()
            regimes = {
                "Low (< 15)":       (vix_vals < 15).sum(),
                "Moderate (15-20)": ((vix_vals >= 15) & (vix_vals < 20)).sum(),
                "Elevated (20-30)": ((vix_vals >= 20) & (vix_vals < 30)).sum(),
                "High (≥ 30)":      (vix_vals >= 30).sum(),
            }
            total = sum(regimes.values())
            regime_df = pd.DataFrame([
                {"Regime": k, "Days": v, "% of Time": f"{v/total*100:.1f}%"}
                for k, v in regimes.items()
            ])
            st.dataframe(regime_df, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Seasonality
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.title(f"{spy_ticker} Seasonality Analysis")
    st.markdown("Historical patterns by month, week, and day-of-month.")

    with st.spinner(f"Fetching {spy_ticker} 20-year data for seasonality..."):
        seas_raw = fetch_price_history(spy_ticker, period="20y", interval="1d")

    if seas_raw.empty:
        st.error(f"Could not fetch data for {spy_ticker}.")
        st.stop()

    seas_df    = compute_returns(seas_raw)
    monthly_df = monthly_seasonality(seas_df)
    pivot_df   = monthly_return_pivot(seas_df)
    weekly_df  = weekly_seasonality(seas_df)
    intra_df   = intramonth_seasonality(seas_df)

    # ── Monthly Seasonality
    st.subheader("Monthly Seasonality")

    if not monthly_df.empty:
        # Best / worst months
        best  = monthly_df.loc[monthly_df["Mean %"].idxmax()]
        worst = monthly_df.loc[monthly_df["Mean %"].idxmin()]
        high_wr = monthly_df.loc[monthly_df["Win Rate %"].idxmax()]

        sc = st.columns(4)
        with sc[0]: colored_metric("Best Month (Mean)", best["MonthName"],
                                   sub=f"{best['Mean %']:+.3f}%")
        with sc[1]: colored_metric("Worst Month (Mean)", worst["MonthName"],
                                   sub=f"{worst['Mean %']:+.3f}%", positive_is_good=False)
        with sc[2]: colored_metric("Highest Win Rate", high_wr["MonthName"],
                                   sub=f"{high_wr['Win Rate %']:.1f}%")
        with sc[3]: colored_metric("Data Points", f"{len(seas_df):,}",
                                   sub=f"{seas_df['Year'].min()}–{seas_df['Year'].max()}")

    fig_mb = plot_monthly_bar(monthly_df)
    st.plotly_chart(fig_mb, use_container_width=True)

    # Monthly stats table
    with st.expander("Monthly Statistics Table"):
        def _seasonal_color(v):
            try:
                f = float(v)
                return "color: #5FC97B" if f > 0 else ("color: #E85C5C" if f < 0 else "")
            except (TypeError, ValueError):
                return ""
        st.dataframe(
            monthly_df.style.format({
                "Mean %": "{:+.3f}", "Median %": "{:+.3f}",
                "Win Rate %": "{:.1f}", "Std %": "{:.3f}",
                "t-stat": "{:.3f}", "p-value": "{:.4f}",
            }).map(_seasonal_color, subset=["Mean %", "Win Rate %"]),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")

    # ── Monthly Heatmap
    st.subheader("Monthly Return Heatmap (Year × Month)")
    fig_hm = plot_monthly_heatmap(pivot_df)
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("---")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        # ── Annual Returns
        st.subheader("Annual Return by Year")
        fig_ann = plot_annual_return_bar(seas_df)
        st.plotly_chart(fig_ann, use_container_width=True)

    with col_s2:
        # ── Intra-Month
        st.subheader("Intra-Month Pattern (Day of Month)")
        fig_intra = plot_intramonth_bar(intra_df)
        st.plotly_chart(fig_intra, use_container_width=True)

    # ── Weekly Seasonality
    st.subheader("Weekly Seasonality (Week of Year)")
    fig_wk = plot_weekly_bar(weekly_df)
    st.plotly_chart(fig_wk, use_container_width=True)
