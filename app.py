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
from datetime import date

from modules.data_fetcher import (
    fetch_price_history,
    fetch_multi_tickers,
    fetch_options_chain,
    clear_today_cache,
    get_refresh_bucket,
    TIMEFRAME_MAP,
)
from modules.supabase_cache import (
    get_data_freshness,
    data_staleness_info,
    is_market_open,
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
    compute_gamma_index,
    compute_dex,
    total_dex_metrics,
    compute_iv_skew,
    filter_options_chain,
    summarize_chain_quality,
    aggregate_gex_by_expiration,
    save_gamma_index_snapshot,
    load_gamma_index_history,
    plot_price_with_gex_levels,
    plot_gex_profile,
    plot_gex_by_expiration,
    plot_gamma_index_timeline,
    plot_dex_profile,
    plot_dex_by_expiration,
    plot_iv_skew,
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
_GEX_PRESETS = ["SPY", "SPX", "QQQ", "AAPL", "NVDA", "MU", "TSLA", "AMZN", "META", "MSFT", "AMD", "GOOGL"]
_gex_preset = st.sidebar.selectbox("GEX Ticker", _GEX_PRESETS, index=0,
                                    help="SPX options use 10× multiplier vs SPY")
_gex_custom = st.sidebar.text_input("Or enter custom ticker", value="",
                                     placeholder="e.g. COIN, SOFI, ARM")
gex_ticker = _gex_custom.strip().upper() if _gex_custom.strip() else _gex_preset

st.sidebar.markdown("---")

import time as _time

# Get freshness info (cache this briefly to avoid hammering Supabase on every rerun)
@st.cache_data(ttl=60)
def _get_freshness():
    return get_data_freshness()

_freshness = _get_freshness()
_staleness = data_staleness_info(_freshness)
_market_open = is_market_open()
_price_bucket = get_refresh_bucket("price")
_options_bucket = get_refresh_bucket("options")


def _refresh_selected_data():
    prefixes = {
        f"{spy_ticker}_price_",
        f"{gex_ticker}_massive_",
        f"{gex_ticker}_yf_options_",
        f"{'^GSPC' if gex_ticker == 'SPX' else gex_ticker}_price_",
        "SPY_price_",
        "^VIX_price_",
        "^VVIX_price_",
        "SVIX_price_",
        "_".join(sorted(["^VIX", "^VVIX", "SVIX"])).replace("^", "") + "_multi_",
    }
    for prefix in prefixes:
        clear_today_cache(prefix=prefix)
    st.cache_data.clear()


def _refresh_all_data():
    clear_today_cache()
    st.cache_data.clear()

# Compact status line — always visible
_status_icon = {"fresh": "🟢", "stale": "🟡"}.get(_staleness["status"], "⚪")
_market_icon = "🔔" if _market_open else "🔕"
st.sidebar.caption(f"{_status_icon} {_staleness['message']}  {_market_icon} {'Market Open' if _market_open else 'Market Closed'}")

# Rate-limit protection: track last refresh time in session state
if "last_force_refresh" not in st.session_state:
    st.session_state["last_force_refresh"] = 0.0

_now = _time.time()
_cooldown_secs = 120 if _market_open else 300
_cooldown_remaining = max(0, _cooldown_secs - (_now - st.session_state["last_force_refresh"]))

if _cooldown_remaining > 0:
    st.sidebar.button(f"Refresh Selected (wait {int(_cooldown_remaining)}s)", disabled=True)
    st.sidebar.button("Refresh All Data", disabled=True)
else:
    selected_label = "Refresh Selected (Stale!)" if _staleness.get("is_stale") else "Refresh Selected Data"
    if st.sidebar.button(selected_label, type="primary" if _staleness.get("is_stale") else "secondary"):
        st.session_state["last_force_refresh"] = _now
        _refresh_selected_data()
        st.rerun()
    if st.sidebar.button("Refresh All Data"):
        st.session_state["last_force_refresh"] = _now
        _refresh_all_data()
        st.rerun()

# Detailed status in collapsible section
with st.sidebar.expander("Data Details"):
    st.caption(f"Options: {_staleness['options_age_str']}")
    st.caption(f"Prices: {_staleness['price_age_str']}")
    st.caption(f"Supabase: {'connected' if _freshness.get('supabase_connected') else 'local cache only'}")
    st.caption(f"Price refresh bucket: {_price_bucket}")
    st.caption(f"Options refresh bucket: {_options_bucket}")
    st.caption("GEX uses a dealer short-call / long-put heuristic")


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

@st.fragment
def _render_dow_tab():
    st.title("Day of Week Performance")
    st.markdown("Analyze how SPY (or selected ETF) performs on specific weekdays across different timeframes.")

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
            index=3,
        )

    if not selected_days:
        st.warning("Select at least one day.")
        return
    if not selected_timeframes:
        st.warning("Select at least one timeframe.")
        return

    with st.spinner(f"Fetching {spy_ticker} 10-year daily data..."):
        raw = fetch_price_history(spy_ticker, period="10y", interval="1d", refresh_bucket=_price_bucket)

    if raw.empty:
        st.error(f"Could not fetch data for {spy_ticker}.")
        return

    dow_data = compute_dow_returns(raw)

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

    st.subheader("Mean Return by Timeframe")
    fig_cmp = plot_dow_comparison(dow_data, selected_days, selected_timeframes, return_col=return_col)
    st.plotly_chart(fig_cmp, use_container_width=True)

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

with tab1:
    _render_dow_tab()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Gamma Exposure
# ═════════════════════════════════════════════════════════════════════════════

@st.fragment
def _render_gex_tab():
    st.title(f"Dealer Gamma Exposure (GEX) — {gex_ticker}")

    gex_view = st.radio("Chart View", ["Call / Put", "Net GEX"], horizontal=True)

    with st.spinner(f"Fetching {gex_ticker} options chain..."):
        raw_calls_df, raw_puts_df, spot, data_source = fetch_options_chain(
            gex_ticker,
            refresh_bucket=_options_bucket,
        )

    if raw_calls_df is None or raw_calls_df.empty or spot is None:
        st.error(f"Could not fetch options data for {gex_ticker}.")
        from modules.data_fetcher import get_last_massive_error
        massive_err = get_last_massive_error()
        if massive_err:
            st.warning(f"Massive.com API: {massive_err}")
        if gex_ticker == "SPX":
            st.info("Try switching to SPY in the sidebar — SPX options data availability varies.")
        return

    available_expirations = sorted(
        set(raw_calls_df.get("expiration", pd.Series(dtype=str)).dropna().tolist()) |
        set(raw_puts_df.get("expiration", pd.Series(dtype=str)).dropna().tolist())
    )

    st.caption(
        f"Data source: {data_source} · Spot ${spot:.2f} · "
        f"Refresh cadence: {'5 min intraday' if _market_open else 'daily after close'}"
    )

    fc1, fc2, fc3, fc4, fc5 = st.columns([1.15, 1.2, 1, 1, 1])
    with fc1:
        dte_bucket = st.selectbox(
            "DTE Bucket",
            ["All", "0-7 DTE", "8-30 DTE", "31-90 DTE", "90+ DTE"],
            index=1,
            help="Use shorter-dated expiries for a more tactical read of dealer positioning.",
        )
    with fc2:
        selected_expiration = st.selectbox(
            "Specific Expiration",
            ["All"] + available_expirations,
            index=0,
            help="Overrides the broad expiry lens when you want one exact expiration.",
        )
    with fc3:
        strike_window_pct = st.slider("Strike Window", 5, 40, 15, 5, format="%d%%")
    with fc4:
        min_open_interest = st.slider("Min OI", 0, 5000, 100, 50)
    with fc5:
        min_volume = st.slider("Min Volume", 0, 2000, 0, 50)

    effective_dte_bucket = "All" if selected_expiration != "All" else dte_bucket
    moneyness_pct = strike_window_pct / 100.0

    calls_df, puts_df, filter_meta = filter_options_chain(
        raw_calls_df,
        raw_puts_df,
        spot,
        selected_expiration=selected_expiration,
        dte_bucket=effective_dte_bucket,
        moneyness_pct=moneyness_pct,
        min_open_interest=min_open_interest,
        min_volume=min_volume,
    )
    confidence = summarize_chain_quality(gex_ticker, data_source, filter_meta)

    if calls_df.empty and puts_df.empty:
        st.warning("No contracts matched the current filter lens. Loosen the expiry or liquidity filters.")
        return

    with st.spinner("Computing GEX..."):
        gex_df  = compute_gex(calls_df, puts_df, spot)
        flip    = gex_flip_point(gex_df, spot)
        metrics = total_gex_metrics(gex_df)
        gamma_idx = compute_gamma_index(gex_df, spot)
        save_gamma_index_snapshot(gex_ticker, gamma_idx, spot)
        gex_exp_df = aggregate_gex_by_expiration(calls_df, puts_df, spot)

    if gex_df.empty:
        st.warning("GEX data empty — options gamma/OI fields may be missing in the data.")
        return

    lens_cols = st.columns(5)
    dominant_exp = None
    if not gex_exp_df.empty:
        dominant_exp = gex_exp_df.loc[gex_exp_df["Net GEX ($B)"].abs().idxmax(), "Expiration"]

    with lens_cols[0]:
        colored_metric("Contracts Used", f"{filter_meta['kept_contracts']:,}",
                       sub=f"{filter_meta['kept_ratio']:.0%} of fetched chain")
    with lens_cols[1]:
        expiry_sub = (
            f"{filter_meta['min_dte']}-{filter_meta['max_dte']} DTE"
            if filter_meta.get("min_dte") is not None and filter_meta.get("max_dte") is not None
            else "No expiry metadata"
        )
        colored_metric("Expirations", filter_meta["expiration_count"], sub=expiry_sub)
    with lens_cols[2]:
        colored_metric("Dominant Expiry", dominant_exp or "N/A",
                       sub="largest |net GEX| slice")
    with lens_cols[3]:
        colored_metric("Confidence", f"{confidence['score']:.0%}",
                       sub=confidence["label"])
    with lens_cols[4]:
        colored_metric("Open Interest", f"{filter_meta['total_open_interest']:,.0f}",
                       sub="contracts in active lens")

    with st.expander("Methodology & Coverage", expanded=False):
        st.markdown(
            "\n".join([
                f"- Assumption model: dealer short calls / long puts heuristic.",
                f"- Active lens: `{selected_expiration}` expiration, `{effective_dte_bucket}` DTE bucket, ±`{strike_window_pct}%` around spot.",
                f"- Liquidity floor: OI >= `{min_open_interest}` and volume >= `{min_volume}`.",
                f"- Contracts retained: `{filter_meta['kept_contracts']:,}` of `{filter_meta['raw_contracts']:,}` fetched.",
                f"- Confidence: **{confidence['label']}** ({confidence['score']:.0%}). Factors: {', '.join(confidence['reasons']) or 'balanced coverage'}.",
                f"- Data caveat: gamma/DEX are estimates from current OI and reported greeks, not observed dealer books.",
            ])
        )

    # ── Dynamic interpretation banner ─────────────────────────────────────
    gi_val = gamma_idx.get("gamma_index", 0)
    gi_cond = gamma_idx.get("gamma_condition", "N/A")
    gi_tilt = gamma_idx.get("gamma_tilt", 0.5)
    gi_conc = gamma_idx.get("gamma_concentration", 0)
    cw = gamma_idx.get("call_wall")
    pw = gamma_idx.get("put_wall")

    _interp_parts = []
    if gi_val > 0:
        _interp_parts.append(
            f"**{gex_ticker}** screens as **positive gamma** ({gi_val:+.3f}B) in the current lens. "
            f"That usually means more mean-reversion and pinning pressure."
        )
    elif gi_val < 0:
        _interp_parts.append(
            f"**{gex_ticker}** screens as **negative gamma** ({gi_val:+.3f}B) in the current lens. "
            f"That usually means faster directional moves and less damping."
        )
    else:
        _interp_parts.append(f"**{gex_ticker}** gamma is **roughly neutral** in the active slice.")

    if flip is not None:
        _flip_dist = ((spot - flip) / spot) * 100
        _flip_dir = "above" if _flip_dist > 0 else "below"
        _interp_parts.append(
            f"Spot ${spot:.2f} is **{abs(_flip_dist):.1f}% {_flip_dir}** "
            f"the gamma flip at ${flip:.2f}."
        )

    if cw and pw:
        _interp_parts.append(
            f"Key range: **${pw:,.0f}** (put wall / support) to "
            f"**${cw:,.0f}** (call wall / resistance)."
        )

    if gi_conc > 0.5:
        _interp_parts.append(
            f"Gamma is **heavily concentrated** ({gi_conc:.0%}) near spot — strong pin risk."
        )

    if confidence["label"] == "Low":
        _interp_parts.append(
            "Treat this as a low-confidence heuristic read because the filtered slice is thin or highly assumption-driven."
        )

    _banner_type = "success" if gi_val > 0 else ("warning" if gi_val < 0 else "info")
    getattr(st, _banner_type)("  \n".join(_interp_parts))

    # ── Gamma Index metrics row ───────────────────────────────────────────
    st.subheader("Gamma Index")

    gi_cols = st.columns(6)
    with gi_cols[0]:
        colored_metric("Gamma Index", f"{gi_val:+.3f}B",
                       sub=gi_cond)
    with gi_cols[1]:
        colored_metric("Call Wall", f"${cw:,.0f}" if cw else "N/A",
                       sub="Resistance")
    with gi_cols[2]:
        colored_metric("Put Wall", f"${pw:,.0f}" if pw else "N/A",
                       sub="Support", positive_is_good=False)
    with gi_cols[3]:
        flip_str = f"${flip:.2f}" if flip else "N/A"
        colored_metric("Gamma Flip", flip_str, positive_is_good=False,
                       sub="Zero-Gamma Level")
    with gi_cols[4]:
        tilt_label = "Bullish" if gi_tilt > 0.55 else ("Bearish" if gi_tilt < 0.45 else "Balanced")
        colored_metric("Gamma Tilt", f"{gi_tilt:.1%}",
                       sub=tilt_label)
    with gi_cols[5]:
        colored_metric("Concentration", f"{gi_conc:.0%}",
                       sub="within ±2% of spot")

    top_strikes = gamma_idx.get("top_strikes", [])
    if top_strikes:
        with st.expander("Top 5 Strikes by Gamma"):
            ts_df = pd.DataFrame(top_strikes).rename(
                columns={"strike": "Strike", "net_gex_b": "Net GEX ($B)"}
            )
            def _ts_color(v):
                try:
                    f = float(v)
                    return "color: #5FC97B" if f > 0 else ("color: #E85C5C" if f < 0 else "")
                except (TypeError, ValueError):
                    return ""
            st.dataframe(
                ts_df.style.format({"Strike": "${:,.0f}", "Net GEX ($B)": "{:+.4f}"})
                .map(_ts_color, subset=["Net GEX ($B)"]),
                use_container_width=True, hide_index=True,
            )

    # ── Price chart with GEX levels ──────────────────────────────────────
    # Use the underlying ticker for price data (SPX → ^GSPC)
    _price_ticker = "^GSPC" if gex_ticker == "SPX" else gex_ticker
    _price_hist = fetch_price_history(_price_ticker, period="1mo", interval="1d", refresh_bucket=_price_bucket)
    fig_price = plot_price_with_gex_levels(
        _price_hist, spot, gex_ticker,
        call_wall=cw, put_wall=pw, gamma_flip=flip,
        top_strikes=top_strikes,
    )
    st.plotly_chart(fig_price, use_container_width=True)

    fig_gi_timeline = plot_gamma_index_timeline(gex_ticker)
    st.plotly_chart(fig_gi_timeline, use_container_width=True)

    st.markdown("---")

    fig_gex = plot_gex_profile(gex_df, spot, gex_ticker,
                               strike_range_pct=moneyness_pct,
                               view_mode=gex_view,
                               call_wall=cw, put_wall=pw)
    st.plotly_chart(fig_gex, use_container_width=True)

    exp_calls, exp_puts, _ = filter_options_chain(
        raw_calls_df,
        raw_puts_df,
        spot,
        selected_expiration="All",
        dte_bucket=effective_dte_bucket,
        moneyness_pct=moneyness_pct,
        min_open_interest=min_open_interest,
        min_volume=min_volume,
    )

    col_gex_exp, col_dex_exp = st.columns(2)
    with col_gex_exp:
        st.subheader("GEX by Expiration")
        fig_exp = plot_gex_by_expiration(exp_calls, exp_puts, spot, gex_ticker)
        st.plotly_chart(fig_exp, use_container_width=True)
    with col_dex_exp:
        st.subheader("DEX by Expiration")
        fig_dex_exp = plot_dex_by_expiration(exp_calls, exp_puts, spot, gex_ticker)
        st.plotly_chart(fig_dex_exp, use_container_width=True)

    st.markdown("---")

    # ── DEX + IV Skew side by side ────────────────────────────────────────
    col_dex, col_iv = st.columns(2)

    with col_dex:
        st.subheader("Delta Exposure (DEX)")
        with st.spinner("Computing DEX..."):
            dex_df = compute_dex(calls_df, puts_df, spot)
        if not dex_df.empty:
            dex_metrics = total_dex_metrics(dex_df)
            net_dex = dex_metrics.get("total_net_dex_m", 0)
            dex_label = (
                "Dealers sell shares into strength" if net_dex > 0
                else "Dealers buy shares into weakness" if net_dex < 0
                else "Balanced hedge pressure"
            )
            colored_metric("Net Delta Exposure", f"{net_dex:+.1f}M shares",
                           sub=dex_label)
            fig_dex = plot_dex_profile(dex_df, spot, gex_ticker,
                                       strike_range_pct=moneyness_pct,
                                       call_wall=cw, put_wall=pw)
            st.plotly_chart(fig_dex, use_container_width=True)
        else:
            st.info("DEX data unavailable — delta could not be computed.")

    with col_iv:
        st.subheader("IV Skew")
        with st.spinner("Computing IV skew..."):
            iv_df = compute_iv_skew(
                calls_df,
                puts_df,
                spot,
                expiration=None if selected_expiration == "All" else selected_expiration,
            )
        if not iv_df.empty:
            # Show ATM IV and skew summary
            atm_mask = (iv_df["moneyness"] >= 0.98) & (iv_df["moneyness"] <= 1.02)
            atm_data = iv_df[atm_mask]
            if not atm_data.empty:
                atm_cols = [c for c in ("call_iv", "put_iv") if c in atm_data.columns]
                if atm_cols:
                    atm_iv = atm_data[atm_cols].stack().mean() * 100
                    colored_metric("ATM Implied Vol", f"{atm_iv:.1f}%", sub="Calls + puts near spot")
            fig_iv = plot_iv_skew(iv_df, spot, gex_ticker)
            st.plotly_chart(fig_iv, use_container_width=True)
        else:
            st.info("IV skew data unavailable.")

    st.markdown("---")

    # ── Raw GEX data + export ────────────────────────────────────────────
    lo = spot * (1 - moneyness_pct)
    hi = spot * (1 + moneyness_pct)
    sub_gex = gex_df[(gex_df["strike"] >= lo) & (gex_df["strike"] <= hi)].copy()
    sub_gex_disp = sub_gex[["strike", "call_gex_b", "put_gex_b", "net_gex_b"]].rename(
        columns={
            "strike": "Strike",
            "call_gex_b": "Call GEX ($B)",
            "put_gex_b":  "Put GEX ($B)",
            "net_gex_b":  "Net GEX ($B)",
        }
    )

    with st.expander("Raw GEX Data Table"):
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

    # ── Export buttons ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Export Data")
    exp_cols = st.columns(3)
    with exp_cols[0]:
        _gex_csv = sub_gex_disp.to_csv(index=False)
        st.download_button(
            "Download Active GEX Slice (CSV)",
            data=_gex_csv,
            file_name=f"{gex_ticker}_gex_{date.today().isoformat()}.csv",
            mime="text/csv",
        )
    with exp_cols[1]:
        _gi_hist = load_gamma_index_history(gex_ticker)
        if not _gi_hist.empty:
            _gi_csv = _gi_hist.to_csv(index=False)
            st.download_button(
                "Download Gamma Index History (CSV)",
                data=_gi_csv,
                file_name=f"{gex_ticker}_gamma_index_history.csv",
                mime="text/csv",
            )
        else:
            st.button("Download Gamma Index History (CSV)", disabled=True,
                       help="No history data yet")
    with exp_cols[2]:
        # Full GEX data (not filtered by strike range)
        _full_gex_disp = gex_df[["strike", "call_gex_b", "put_gex_b", "net_gex_b"]].rename(
            columns={
                "strike": "Strike",
                "call_gex_b": "Call GEX ($B)",
                "put_gex_b":  "Put GEX ($B)",
                "net_gex_b":  "Net GEX ($B)",
            }
        )
        _full_csv = _full_gex_disp.to_csv(index=False)
        st.download_button(
            "Download Filtered GEX (All Visible Strikes)",
            data=_full_csv,
            file_name=f"{gex_ticker}_gex_full_{date.today().isoformat()}.csv",
            mime="text/csv",
        )

with tab2:
    _render_gex_tab()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — VIX / Volatility Analysis
# ═════════════════════════════════════════════════════════════════════════════

@st.fragment
def _render_vix_tab():
    st.title("VIX / VVIX / SVIX Analysis")
    st.markdown("""
    - **VIX** — 30-day implied volatility of S&P 500 options ("fear gauge")
    - **VVIX** — Volatility of VIX itself; measures the uncertainty of volatility
    - **SVIX** — 1x Short VIX Futures ETF; inversely tracks VIX futures
    """)

    with st.spinner("Fetching VIX / VVIX / SVIX data..."):
        vix_raw = fetch_multi_tickers(["^VIX", "^VVIX", "SVIX"],
                                       period=vix_period, interval="1d", refresh_bucket=_price_bucket)
        spy_hist = fetch_price_history(spy_ticker, period=vix_period, interval="1d", refresh_bucket=_price_bucket)

    if vix_raw.empty:
        st.error("Could not fetch VIX data.")
        return

    rename_map = {"^VIX": "VIX", "^VVIX": "VVIX", "SVIX": "SVIX"}
    vix_raw = vix_raw.rename(columns={k: v for k, v in rename_map.items() if k in vix_raw.columns})

    vix_df = compute_vix_metrics(vix_raw)
    stats  = vix_summary_stats(vix_df)

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

    st.subheader("VIX Momentum / Term Structure Proxy")
    fig_ts = plot_vix_term_structure_proxy(vix_df)
    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader("Correlation Matrix")
    fig_corr = plot_correlation_matrix(vix_df, spy_hist)
    st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("VIX Regime Distribution"):
        if "VIX" in vix_df.columns:
            vix_vals = vix_df["VIX"].dropna()
            regimes = {
                "Low (< 15)":       (vix_vals < 15).sum(),
                "Moderate (15-20)": ((vix_vals >= 15) & (vix_vals < 20)).sum(),
                "Elevated (20-30)": ((vix_vals >= 20) & (vix_vals < 30)).sum(),
                "High (>= 30)":     (vix_vals >= 30).sum(),
            }
            total = sum(regimes.values())
            regime_df = pd.DataFrame([
                {"Regime": k, "Days": v, "% of Time": f"{v/total*100:.1f}%"}
                for k, v in regimes.items()
            ])
            st.dataframe(regime_df, use_container_width=True, hide_index=True)

with tab3:
    _render_vix_tab()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Seasonality
# ═════════════════════════════════════════════════════════════════════════════

@st.fragment
def _render_seasonality_tab():
    st.title(f"{spy_ticker} Seasonality Analysis")
    st.markdown("Historical patterns by month, week, and day-of-month.")

    with st.spinner(f"Fetching {spy_ticker} 20-year data for seasonality..."):
        seas_raw = fetch_price_history(spy_ticker, period="20y", interval="1d", refresh_bucket=_price_bucket)

    if seas_raw.empty:
        st.error(f"Could not fetch data for {spy_ticker}.")
        return

    seas_df    = compute_returns(seas_raw)
    monthly_df = monthly_seasonality(seas_df)
    pivot_df   = monthly_return_pivot(seas_df)
    weekly_df  = weekly_seasonality(seas_df)
    intra_df   = intramonth_seasonality(seas_df)

    st.subheader("Monthly Seasonality")

    if not monthly_df.empty:
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

    st.subheader("Monthly Return Heatmap (Year x Month)")
    fig_hm = plot_monthly_heatmap(pivot_df)
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("---")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("Annual Return by Year")
        fig_ann = plot_annual_return_bar(seas_df)
        st.plotly_chart(fig_ann, use_container_width=True)

    with col_s2:
        st.subheader("Intra-Month Pattern (Day of Month)")
        fig_intra = plot_intramonth_bar(intra_df)
        st.plotly_chart(fig_intra, use_container_width=True)

    st.subheader("Weekly Seasonality (Week of Year)")
    fig_wk = plot_weekly_bar(weekly_df)
    st.plotly_chart(fig_wk, use_container_width=True)

with tab4:
    _render_seasonality_tab()
