import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add return % and date components to price DataFrame."""
    close = df["Close"].squeeze()
    out = pd.DataFrame({
        "Close":  close,
        "Return": close.pct_change() * 100,
    }, index=close.index)
    out["Year"]        = out.index.year
    out["Month"]       = out.index.month
    out["MonthName"]   = [MONTH_NAMES[m - 1] for m in out.index.month]
    out["Week"]        = out.index.isocalendar().week.astype(int)
    out["DayOfMonth"]  = out.index.day
    out["DayOfWeek"]   = out.index.dayofweek
    return out.dropna(subset=["Return"])


def monthly_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average return per calendar month, with count, std, win-rate, and t-test p-value.
    Returns sorted by month number.
    """
    rows = []
    for m in range(1, 13):
        sub = df[df["Month"] == m]["Return"].dropna()
        if len(sub) < 2:
            continue
        tstat, pval = stats.ttest_1samp(sub, 0)
        rows.append({
            "Month":        m,
            "MonthName":    MONTH_NAMES[m - 1],
            "Mean %":       round(sub.mean(), 3),
            "Median %":     round(sub.median(), 3),
            "Win Rate %":   round((sub > 0).mean() * 100, 1),
            "Std %":        round(sub.std(), 3),
            "Count":        len(sub),
            "t-stat":       round(tstat, 3),
            "p-value":      round(pval, 4),
            "Significant":  pval < 0.05,
        })
    return pd.DataFrame(rows)


def monthly_return_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot: rows = Year, columns = Month, values = sum of daily returns in that month.
    """
    df2 = df.copy()
    df2["MonthlyReturn"] = df2.groupby(["Year", "Month"])["Return"].transform("sum")
    pivot = df2.drop_duplicates(subset=["Year", "Month"]).pivot(
        index="Year", columns="Month", values="MonthlyReturn"
    )
    pivot.columns = [MONTH_NAMES[c - 1] for c in pivot.columns]
    return pivot


def weekly_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Average return per week-of-year (1-52)."""
    rows = []
    for w in range(1, 53):
        sub = df[df["Week"] == w]["Return"].dropna()
        if len(sub) < 2:
            continue
        rows.append({
            "Week":       w,
            "Mean %":     round(sub.mean(), 3),
            "Win Rate %": round((sub > 0).mean() * 100, 1),
            "Count":      len(sub),
        })
    return pd.DataFrame(rows)


def intramonth_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Average return per calendar day-of-month (1-31)."""
    rows = []
    for d in range(1, 32):
        sub = df[df["DayOfMonth"] == d]["Return"].dropna()
        if len(sub) < 5:
            continue
        rows.append({
            "Day":        d,
            "Mean %":     round(sub.mean(), 3),
            "Win Rate %": round((sub > 0).mean() * 100, 1),
            "Count":      len(sub),
        })
    return pd.DataFrame(rows)


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_monthly_bar(monthly_df: pd.DataFrame) -> go.Figure:
    """Bar chart: mean return per month with error bars and win rate overlay."""
    if monthly_df.empty:
        return go.Figure()

    colors = ["#5FC97B" if v >= 0 else "#E85C5C" for v in monthly_df["Mean %"]]
    error = monthly_df["Std %"] / np.sqrt(monthly_df["Count"])  # SE

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=monthly_df["MonthName"],
        y=monthly_df["Mean %"],
        name="Mean Daily Return %",
        marker_color=colors,
        error_y=dict(type="data", array=error.round(3), visible=True, color="rgba(255,255,255,0.5)"),
        customdata=monthly_df[["Win Rate %", "Count", "p-value"]].values,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Mean Return: %{y:+.3f}%<br>"
            "Win Rate: %{customdata[0]:.1f}%<br>"
            "Sample: %{customdata[1]}<br>"
            "p-value: %{customdata[2]:.4f}<extra></extra>"
        ),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=monthly_df["MonthName"],
        y=monthly_df["Win Rate %"],
        name="Win Rate %",
        mode="lines+markers",
        line=dict(color="#F5E642", width=2, dash="dot"),
        marker=dict(size=6),
        hovertemplate="Win Rate: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                  row=1, col=1)

    fig.update_layout(
        title="SPY Monthly Seasonality — Mean Daily Return per Month",
        template="plotly_dark",
        height=480,
        legend=dict(orientation="h", y=1.05),
    )
    fig.update_yaxes(title_text="Mean Return (%)", secondary_y=False)
    fig.update_yaxes(title_text="Win Rate (%)", secondary_y=True, range=[0, 100])
    return fig


def plot_monthly_heatmap(pivot_df: pd.DataFrame) -> go.Figure:
    """Year × Month heatmap of monthly returns."""
    if pivot_df.empty:
        return go.Figure()

    ordered_cols = [m for m in MONTH_NAMES if m in pivot_df.columns]
    z = pivot_df[ordered_cols].values
    years = [str(y) for y in pivot_df.index.tolist()]

    # Symmetric color scale
    abs_max = np.nanpercentile(np.abs(z[~np.isnan(z)]), 95) if np.any(~np.isnan(z)) else 5

    fig = go.Figure(go.Heatmap(
        z=z,
        x=ordered_cols,
        y=years,
        colorscale="RdYlGn",
        zmid=0,
        zmin=-abs_max,
        zmax=abs_max,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="Return %"),
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title="SPY Monthly Return Heatmap (Year × Month)",
        xaxis_title="Month",
        yaxis_title="Year",
        template="plotly_dark",
        height=max(400, len(years) * 22 + 100),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_weekly_bar(weekly_df: pd.DataFrame) -> go.Figure:
    """Bar chart: mean return per week-of-year."""
    if weekly_df.empty:
        return go.Figure()

    colors = ["#5FC97B" if v >= 0 else "#E85C5C" for v in weekly_df["Mean %"]]

    fig = go.Figure(go.Bar(
        x=weekly_df["Week"],
        y=weekly_df["Mean %"],
        name="Mean Return %",
        marker_color=colors,
        customdata=weekly_df[["Win Rate %", "Count"]].values,
        hovertemplate=(
            "Week %{x}<br>"
            "Mean Return: %{y:+.3f}%<br>"
            "Win Rate: %{customdata[0]:.1f}%<br>"
            "Sample: %{customdata[1]}<extra></extra>"
        ),
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

    fig.update_layout(
        title="SPY Seasonality by Week of Year",
        xaxis_title="Week of Year",
        yaxis_title="Mean Return (%)",
        template="plotly_dark",
        height=420,
    )
    return fig


def plot_intramonth_bar(intramonth_df: pd.DataFrame) -> go.Figure:
    """Bar chart: mean return by day-of-month."""
    if intramonth_df.empty:
        return go.Figure()

    colors = ["#5FC97B" if v >= 0 else "#E85C5C" for v in intramonth_df["Mean %"]]

    fig = go.Figure(go.Bar(
        x=intramonth_df["Day"],
        y=intramonth_df["Mean %"],
        marker_color=colors,
        customdata=intramonth_df[["Win Rate %", "Count"]].values,
        hovertemplate=(
            "Day %{x} of Month<br>"
            "Mean Return: %{y:+.3f}%<br>"
            "Win Rate: %{customdata[0]:.1f}%<br>"
            "Sample: %{customdata[1]}<extra></extra>"
        ),
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

    fig.update_layout(
        title="SPY Intra-Month Seasonality (Day of Month)",
        xaxis_title="Day of Month",
        yaxis_title="Mean Return (%)",
        template="plotly_dark",
        height=380,
        xaxis=dict(tickmode="linear", dtick=1),
    )
    return fig


def plot_annual_return_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart of annual SPY total return."""
    if "Year" not in df.columns or "Return" not in df.columns:
        return go.Figure()

    annual = df.groupby("Year")["Return"].sum().reset_index()
    annual.columns = ["Year", "Annual Return %"]
    colors = ["#5FC97B" if v >= 0 else "#E85C5C" for v in annual["Annual Return %"]]

    fig = go.Figure(go.Bar(
        x=annual["Year"].astype(str),
        y=annual["Annual Return %"],
        marker_color=colors,
        hovertemplate="Year: %{x}<br>Annual Return: %{y:.1f}%<extra></extra>",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

    fig.update_layout(
        title="SPY Annual Return by Year",
        xaxis_title="Year",
        yaxis_title="Return (%)",
        template="plotly_dark",
        height=380,
    )
    return fig
