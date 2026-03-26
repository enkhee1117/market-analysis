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
