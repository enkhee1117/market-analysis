from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


DAY_NAMES = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
DAY_COLORS = {
    "Monday":    "#4C9BE8",
    "Tuesday":   "#F28C38",
    "Wednesday": "#5FC97B",
    "Thursday":  "#E85C5C",
    "Friday":    "#A575E8",
}

TIMEFRAMES = {
    "1 Month":   30,
    "3 Months":  90,
    "6 Months":  180,
    "12 Months": 365,
    "3 Years":   1095,
    "5 Years":   1825,
    "10 Years":  3650,
}


RETURN_TYPES = {
    "Close-to-Close": "Return",
    "Open-to-Close (Intraday)": "Intraday",
    "Overnight (Close-to-Open)": "Overnight",
}


def compute_dow_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns (close-to-close, intraday, overnight) and annotate with day-of-week."""
    close = df["Close"].squeeze()
    open_ = df["Open"].squeeze()
    returns = close.pct_change() * 100  # close-to-close %
    intraday = (close - open_) / open_ * 100  # open-to-close %
    overnight = (open_ - close.shift(1)) / close.shift(1) * 100  # prev close to open %
    result = pd.DataFrame({
        "Close":     close,
        "Open":      open_,
        "Return":    returns,
        "Intraday":  intraday,
        "Overnight": overnight,
        "DayOfWeek": close.index.dayofweek,
        "DayName":   [DAY_NAMES.get(d, "Other") for d in close.index.dayofweek],
    }, index=close.index)
    return result.dropna(subset=["Return"])


def filter_by_timeframe(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Slice dataframe to the given timeframe label."""
    if df.empty:
        return df
    days = TIMEFRAMES.get(label, 365)
    cutoff = df.index.max() - timedelta(days=days)
    return df[df.index >= cutoff]


def dow_summary(df: pd.DataFrame, days: list, return_col: str = "Return") -> pd.DataFrame:
    """Return summary stats per day-of-week for the given day names."""
    rows = []
    for day in days:
        sub = df[df["DayName"] == day][return_col].dropna()
        if len(sub) == 0:
            continue
        rows.append({
            "Day":       day,
            "Mean %":    round(sub.mean(), 3),
            "Median %":  round(sub.median(), 3),
            "Win Rate":  round((sub > 0).mean() * 100, 1),
            "Std %":     round(sub.std(), 3),
            "Count":     len(sub),
            "Best %":    round(sub.max(), 2),
            "Worst %":   round(sub.min(), 2),
        })
    return pd.DataFrame(rows)


def plot_dow_comparison(all_data: pd.DataFrame, selected_days: list, timeframes: list,
                        return_col: str = "Return") -> go.Figure:
    """
    Grouped bar chart: X = timeframe, groups = day of week.
    Shows mean daily return for each day across timeframes.
    """
    rows = []
    for tf_label in timeframes:
        sub = filter_by_timeframe(all_data, tf_label)
        for day in selected_days:
            day_data = sub[sub["DayName"] == day][return_col].dropna()
            if len(day_data) == 0:
                continue
            rows.append({
                "Timeframe": tf_label,
                "Day": day,
                "Mean Return %": round(day_data.mean(), 3),
                "Win Rate %": round((day_data > 0).mean() * 100, 1),
                "Count": len(day_data),
            })

    if not rows:
        return go.Figure()

    summary = pd.DataFrame(rows)

    fig = go.Figure()
    for day in selected_days:
        sub = summary[summary["Day"] == day]
        color = DAY_COLORS.get(day, "#888888")
        fig.add_trace(go.Bar(
            name=day,
            x=sub["Timeframe"],
            y=sub["Mean Return %"],
            marker_color=color,
            text=[f"{v:+.3f}%" for v in sub["Mean Return %"]],
            textposition="outside",
            customdata=sub[["Win Rate %", "Count"]].values,
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Timeframe: %{x}<br>"
                "Mean Return: %{y:+.3f}%<br>"
                "Win Rate: %{customdata[0]:.1f}%<br>"
                "Sample Size: %{customdata[1]}<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="group",
        title="Mean Daily Return by Day of Week & Timeframe",
        xaxis_title="Timeframe",
        yaxis_title="Mean Return (%)",
        legend_title="Day of Week",
        template="plotly_dark",
        height=500,
        yaxis=dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.3)", zerolinewidth=1),
    )
    return fig


def plot_win_rate_comparison(all_data: pd.DataFrame, selected_days: list, timeframes: list,
                             return_col: str = "Return") -> go.Figure:
    """Bar chart of win rates by timeframe and day."""
    rows = []
    for tf_label in timeframes:
        sub = filter_by_timeframe(all_data, tf_label)
        for day in selected_days:
            day_data = sub[sub["DayName"] == day][return_col].dropna()
            if len(day_data) == 0:
                continue
            rows.append({
                "Timeframe": tf_label,
                "Day": day,
                "Win Rate %": round((day_data > 0).mean() * 100, 1),
                "Count": len(day_data),
            })

    if not rows:
        return go.Figure()

    summary = pd.DataFrame(rows)

    fig = go.Figure()
    for day in selected_days:
        sub = summary[summary["Day"] == day]
        color = DAY_COLORS.get(day, "#888888")
        fig.add_trace(go.Bar(
            name=day,
            x=sub["Timeframe"],
            y=sub["Win Rate %"],
            marker_color=color,
            text=[f"{v:.1f}%" for v in sub["Win Rate %"]],
            textposition="outside",
            customdata=sub["Count"].values,
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Timeframe: %{x}<br>"
                "Win Rate: %{y:.1f}%<br>"
                "Sample Size: %{customdata}<extra></extra>"
            ),
        ))

    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                  annotation_text="50% baseline", annotation_position="right")

    fig.update_layout(
        barmode="group",
        title="Win Rate by Day of Week & Timeframe",
        xaxis_title="Timeframe",
        yaxis_title="Win Rate (%)",
        legend_title="Day of Week",
        template="plotly_dark",
        height=500,
        yaxis=dict(range=[0, 100]),
    )
    return fig


def plot_dow_distribution(all_data: pd.DataFrame, selected_days: list, timeframe: str,
                          return_col: str = "Return") -> go.Figure:
    """Violin/box plot showing return distribution per selected day."""
    sub = filter_by_timeframe(all_data, timeframe)
    fig = go.Figure()
    for day in selected_days:
        day_data = sub[sub["DayName"] == day][return_col].dropna()
        if len(day_data) == 0:
            continue
        color = DAY_COLORS.get(day, "#888888")
        fig.add_trace(go.Violin(
            y=day_data,
            name=day,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            opacity=0.7,
            line_color=color,
            hoverinfo="y+name",
        ))

    fig.update_layout(
        title=f"Return Distribution by Day ({timeframe})",
        yaxis_title="Daily Return (%)",
        template="plotly_dark",
        height=450,
        showlegend=False,
        yaxis=dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.3)"),
    )
    return fig


def plot_cumulative_by_dow(all_data: pd.DataFrame, selected_days: list, timeframe: str,
                           return_col: str = "Return") -> go.Figure:
    """Cumulative return if you only invested on specific days."""
    sub = filter_by_timeframe(all_data, timeframe).copy()
    fig = go.Figure()
    for day in selected_days:
        day_data = sub[sub["DayName"] == day][return_col].dropna() / 100
        cumret = (1 + day_data).cumprod() - 1
        cumret *= 100
        color = DAY_COLORS.get(day, "#888888")
        fig.add_trace(go.Scatter(
            x=cumret.index,
            y=cumret.values,
            name=day,
            mode="lines",
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{day}</b><br>Date: %{{x|%Y-%m-%d}}<br>Cumulative: %{{y:.2f}}%<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig.update_layout(
        title=f"Cumulative Return Investing Only on Selected Days ({timeframe})",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        template="plotly_dark",
        height=400,
        legend_title="Day",
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Conditional Day-of-Week Analysis
# ═════════════════════════════════════════════════════════════════════════════

_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def compute_conditional_probabilities(df: pd.DataFrame,
                                       return_col: str = "Return") -> pd.DataFrame:
    """For each consecutive weekday pair, compute P(next green/red | today green/red).

    Returns a DataFrame with columns:
        Today, Today_Color, Tomorrow, P_Green, P_Red, Mean_Next, Median_Next, Count
    """
    df = df.copy()
    df["color"] = np.where(df[return_col] >= 0, "green", "red")
    df["next_return"] = df[return_col].shift(-1)
    df["next_color"] = df["color"].shift(-1)
    df["next_day"] = df["DayName"].shift(-1)
    # Only keep consecutive trading days (gap <= 4 calendar days handles weekends + holidays)
    df["gap"] = df.index.to_series().diff().shift(-1).dt.days
    df = df[(df["gap"] <= 4) & df["next_return"].notna()].copy()

    rows = []
    for today in _WEEKDAYS:
        for color in ["green", "red"]:
            sub = df[(df["DayName"] == today) & (df["color"] == color)]
            if len(sub) < 5:
                continue
            tomorrow = sub["next_day"].mode()
            if tomorrow.empty:
                continue
            p_green = (sub["next_color"] == "green").mean() * 100
            rows.append({
                "Today": today,
                "Today_Color": color,
                "Tomorrow": tomorrow.iloc[0],
                "P_Green": round(p_green, 1),
                "P_Red": round(100 - p_green, 1),
                "Mean_Next": round(sub["next_return"].mean(), 3),
                "Median_Next": round(sub["next_return"].median(), 3),
                "Count": len(sub),
            })
    return pd.DataFrame(rows)


def compute_conditional_chain(df: pd.DataFrame,
                              chain: list,
                              return_col: str = "Return") -> dict:
    """Given a chain like [("Monday", "red"), ("Tuesday", "red")], find what happens next.

    Args:
        chain: list of (day_name, "red"/"green") tuples representing consecutive days.

    Returns dict with keys:
        p_green, p_red, mean_return, median_return, count, next_day, returns (Series)
    """
    if not chain:
        return {"count": 0}

    df = df.copy()
    df["color"] = np.where(df[return_col] >= 0, "green", "red")
    df["gap_days"] = df.index.to_series().diff().dt.days

    chain_len = len(chain)
    # Build mask: find sequences matching the full chain
    matches = pd.Series(True, index=df.index)

    for offset, (day, color) in enumerate(chain):
        shifted_day = df["DayName"].shift(-offset)
        shifted_color = df["color"].shift(-offset)
        matches &= (shifted_day == day) & (shifted_color == color)
        # Check consecutive trading days (no gap > 4)
        if offset > 0:
            shifted_gap = df["gap_days"].shift(-offset)
            matches &= shifted_gap.fillna(99) <= 4

    # Get the row AFTER the chain
    next_return = df[return_col].shift(-(chain_len))
    next_color = df["color"].shift(-(chain_len))
    next_day_name = df["DayName"].shift(-(chain_len))
    next_gap = df["gap_days"].shift(-(chain_len))

    valid = matches & next_return.notna() & (next_gap.fillna(99) <= 4)
    result_returns = next_return[valid]

    if len(result_returns) == 0:
        return {"count": 0}

    p_green = (next_color[valid] == "green").mean() * 100
    return {
        "p_green": round(p_green, 1),
        "p_red": round(100 - p_green, 1),
        "mean_return": round(result_returns.mean(), 3),
        "median_return": round(result_returns.median(), 3),
        "std_return": round(result_returns.std(), 3),
        "best": round(result_returns.max(), 2),
        "worst": round(result_returns.min(), 2),
        "count": len(result_returns),
        "next_day": next_day_name[valid].mode().iloc[0] if not next_day_name[valid].mode().empty else "?",
        "returns": result_returns,
    }


def plot_conditional_heatmap(prob_df: pd.DataFrame) -> go.Figure:
    """Heatmap: rows = today's condition (e.g. 'Monday Green'), columns = P(next green)."""
    if prob_df.empty:
        return go.Figure()

    prob_df = prob_df.copy()
    prob_df["Label"] = prob_df["Today"] + " " + prob_df["Today_Color"].str.capitalize()

    # Pivot: rows = label, value = P_Green
    pivot = prob_df.set_index("Label")[["P_Green", "Count"]].copy()

    # Build annotation text
    annotations = [f"{p:.0f}%\n(n={c})" for p, c in zip(pivot["P_Green"], pivot["Count"])]

    # Color scale: red (low P_Green) to green (high P_Green)
    colors = ["#E85C5C" if p < 45 else "#F5E642" if p < 55 else "#5FC97B" for p in pivot["P_Green"]]

    fig = go.Figure(go.Bar(
        x=pivot.index,
        y=pivot["P_Green"],
        marker_color=colors,
        text=annotations,
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>P(Next Green): %{y:.1f}%<extra></extra>",
    ))

    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                  annotation_text="50% baseline", annotation_position="right")

    fig.update_layout(
        title="Next-Day P(Green) Given Today's Color",
        xaxis_title="",
        yaxis_title="Probability of Next Day Green (%)",
        template="plotly_dark",
        height=450,
        yaxis=dict(range=[0, 100]),
        xaxis=dict(tickangle=-45),
    )
    return fig


def plot_conditional_distribution(returns: pd.Series, chain_label: str) -> go.Figure:
    """Histogram of next-day returns given a condition chain."""
    if returns.empty:
        return go.Figure()

    win_rate = (returns >= 0).mean() * 100
    colors = ["#5FC97B" if r >= 0 else "#E85C5C" for r in returns]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=30,
        marker_color="#4C9BE8",
        opacity=0.8,
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="white", line_width=1)
    fig.add_vline(x=returns.mean(), line_dash="dot", line_color="#F5E642", line_width=2,
                  annotation_text=f"Mean: {returns.mean():+.3f}%",
                  annotation_position="top right", annotation_font_color="#F5E642")

    fig.update_layout(
        title=f"Next-Day Return Distribution | {chain_label}",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=400,
        annotations=[dict(
            text=f"Win Rate: {win_rate:.1f}%  |  n={len(returns)}",
            xref="paper", yref="paper", x=0.98, y=0.95,
            showarrow=False, font=dict(size=13, color="white"),
            bgcolor="rgba(0,0,0,0.5)", borderpad=4,
        )],
    )
    return fig
