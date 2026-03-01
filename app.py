"""
CRE Rate Monitor Dashboard
Pulls live Fed Funds Rate, SOFR, and 10Y Treasury data from FRED.

To run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from fredapi import Fred
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Your FRED API key — loaded from Streamlit secrets (cloud) or fallback (local)
# On Streamlit Cloud, this is set in the app's Secrets settings
# Locally, you can create a file at .streamlit/secrets.toml with: FRED_API_KEY = "your_key"
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except Exception:
    FRED_API_KEY = "YOUR_FRED_API_KEY_HERE"  # Will prompt user in sidebar if not set

# Define the data series we want to pull.
# You can add more series later just by adding rows here!
# Find series IDs at https://fred.stlouisfed.org
SERIES = {
    "DFF":       {"name": "Fed Funds Rate",   "color": "#00D4AA"},
    "SOFR":      {"name": "SOFR",             "color": "#3B82F6"},
    "DGS10":     {"name": "10Y Treasury",     "color": "#F59E0B"},
    "DFII5":     {"name": "5Y TIPS Yield",    "color": "#A78BFA"},
    "DFII10":    {"name": "10Y TIPS Yield",   "color": "#F472B6"},
    "DFII20":    {"name": "20Y TIPS Yield",   "color": "#FB923C"},
}

# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="CRE Rate Monitor",
    page_icon="📈",
    layout="wide",
)

# Custom CSS for a clean, dark look
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp { background-color: #0d0f12; }
    .metric-card {
        background: #14161b;
        border: 1px solid #1e2028;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 12px;
        color: #8b8f98;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .metric-change {
        font-size: 13px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .spread-box {
        background: #14161b;
        border: 1px solid #1e2028;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING (cached so it doesn't re-fetch every interaction)
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600)  # Cache for 1 hour, then re-fetch
def load_data(api_key, series_dict, start_date, end_date=None):
    """
    Pull each series from FRED and combine into one DataFrame.
    This function is cached — Streamlit will only call FRED once per hour.
    """
    fred = Fred(api_key=api_key)
    frames = {}

    for series_id, info in series_dict.items():
        try:
            data = fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
            )
            frames[series_id] = data
        except Exception as e:
            st.warning(f"Could not load {info['name']}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index.name = "Date"

    # Reindex to daily frequency and forward-fill gaps
    # (weekends, holidays, or series that publish less frequently)
    daily_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(daily_index)
    df = df.ffill()  # Forward-fill: carry the last known value into gaps
    df.index.name = "Date"

    # Drop rows where ALL values are NaN (only at the very start if series have different start dates)
    df = df.dropna(how="all")

    return df


# ─────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    api_key = FRED_API_KEY

    st.markdown("---")

    # Date range selector
    st.markdown("### Date Range")
    range_options = {
        "1 Month": 30,
        "1 Year": 365,
        "2 Years": 730,
        "3 Years": 1095,
        "5 Years": 1825,
        "All (since 2018)": None,
        "Custom": "custom",
    }
    selected_range = st.radio(
        "Select range",
        options=list(range_options.keys()),
        index=0,  # Default to "1 Month"
        label_visibility="collapsed",
    )

    # Custom date pickers (only shown when "Custom" is selected)
    if selected_range == "Custom":
        custom_start = st.date_input(
            "Start date",
            value=datetime(2018, 1, 1).date(),
            min_value=datetime(1954, 7, 1).date(),  # FRED has fed funds back to 1954
            max_value=datetime.now().date(),
        )
        custom_end = st.date_input(
            "End date",
            value=datetime.now().date(),
            min_value=datetime(1954, 7, 1).date(),
            max_value=datetime.now().date(),
        )

    st.markdown("---")

    # Series visibility toggles
    st.markdown("### Visible Series")
    visible = {}
    for series_id, info in SERIES.items():
        visible[series_id] = st.checkbox(
            info["name"],
            value=True,
            key=series_id,
        )

    st.markdown("---")

    # Chart type
    chart_mode = st.radio(
        "Chart Mode",
        ["Rates", "Spreads (bps)"],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#555'>Data: FRED / Federal Reserve<br>"
        "Updates hourly when app is open</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

# Header
st.markdown("# 📈 CRE Rate Monitor")
st.markdown("*Fed Funds · SOFR · 10Y Treasury — Live from FRED*")

if not api_key:
    st.info(
        "👈 **Enter your FRED API key in the sidebar to get started.**\n\n"
        "It's free — takes 2 minutes:\n"
        "1. Go to [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)\n"
        "2. Create an account (or log in)\n"
        "3. Request an API key\n"
        "4. Paste it in the sidebar"
    )
    st.stop()

# Calculate start and end dates
if selected_range == "Custom":
    start = custom_start.strftime("%Y-%m-%d")
    end = custom_end.strftime("%Y-%m-%d")
elif range_options[selected_range] is not None:
    start = (datetime.now() - timedelta(days=range_options[selected_range])).strftime("%Y-%m-%d")
    end = None
else:
    start = "2018-01-01"
    end = None

# Load data
with st.spinner("Fetching data from FRED..."):
    df = load_data(api_key, SERIES, start, end)

if df.empty:
    st.error("No data returned. Check your API key and try again.")
    st.stop()


# ─────────────────────────────────────────────
# METRIC CARDS (top row)
# ─────────────────────────────────────────────

cols = st.columns(len(SERIES))

for i, (series_id, info) in enumerate(SERIES.items()):
    with cols[i]:
        if series_id in df.columns:
            current = df[series_id].dropna().iloc[-1]
            previous = df[series_id].dropna().iloc[-2] if len(df[series_id].dropna()) > 1 else current
            change_bps = (current - previous) * 100
            period_min = df[series_id].min()
            period_max = df[series_id].max()
            period_avg = df[series_id].mean()

            arrow = "▲" if change_bps >= 0 else "▼"
            change_color = "#00D4AA" if change_bps >= 0 else "#EF4444"

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{info['name']}</div>
                <div class="metric-value" style="color:{info['color']}">{current:.2f}%</div>
                <div class="metric-change" style="color:{change_color}">
                    {arrow} {abs(change_bps):.0f} bps m/m
                </div>
                <div style="margin-top:8px; font-size:11px; color:#6b7080; font-family:monospace;">
                    Low {period_min:.2f} &nbsp;|&nbsp; High {period_max:.2f} &nbsp;|&nbsp; Avg {period_avg:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CHART
# ─────────────────────────────────────────────

fig = go.Figure()

if chart_mode == "Rates":
    for series_id, info in SERIES.items():
        if visible.get(series_id) and series_id in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[series_id],
                name=info["name"],
                line=dict(color=info["color"], width=2),
                hovertemplate="%{x|%b %d, %Y}: %{y:.2f}%<extra>" + info["name"] + "</extra>",
            ))

    fig.update_yaxes(title_text="Rate (%)", ticksuffix="%", rangemode="tozero", dtick=1)
    fig.update_layout(title="Interest Rates — Daily")

else:
    # Spread mode: calculate basis point differences
    if "DGS10" in df.columns and "DFF" in df.columns:
        spread_10y_ff = (df["DGS10"] - df["DFF"]) * 100
        fig.add_trace(go.Scatter(
            x=df.index, y=spread_10y_ff,
            name="10Y – Fed Funds",
            line=dict(color="#F59E0B", width=2),
            hovertemplate="%{x|%b %d, %Y}: %{y:.0f} bps<extra>10Y – Fed Funds</extra>",
        ))

    if "DGS10" in df.columns and "SOFR" in df.columns:
        spread_10y_sofr = (df["DGS10"] - df["SOFR"]) * 100
        fig.add_trace(go.Scatter(
            x=df.index, y=spread_10y_sofr,
            name="10Y – SOFR",
            line=dict(color="#A78BFA", width=2),
            hovertemplate="%{x|%b %d, %Y}: %{y:.0f} bps<extra>10Y – SOFR</extra>",
        ))

    if "SOFR" in df.columns and "DFF" in df.columns:
        spread_sofr_ff = (df["SOFR"] - df["DFF"]) * 100
        fig.add_trace(go.Scatter(
            x=df.index, y=spread_sofr_ff,
            name="SOFR – Fed Funds",
            line=dict(color="#34D399", width=2, dash="dash"),
            hovertemplate="%{x|%b %d, %Y}: %{y:.0f} bps<extra>SOFR – Fed Funds</extra>",
        ))

    if "DGS10" in df.columns and "DFII10" in df.columns:
        spread_breakeven = (df["DGS10"] - df["DFII10"]) * 100
        fig.add_trace(go.Scatter(
            x=df.index, y=spread_breakeven,
            name="10Y Breakeven Inflation",
            line=dict(color="#F472B6", width=2),
            hovertemplate="%{x|%b %d, %Y}: %{y:.0f} bps<extra>10Y Breakeven Inflation</extra>",
        ))

    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="#2d3139", line_width=1.5)
    fig.update_yaxes(title_text="Spread (bps)", ticksuffix=" bps")
    fig.update_layout(title="Rate Spreads — Monthly Averages")

# Shared layout styling
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#14161b",
    plot_bgcolor="#14161b",
    font=dict(family="JetBrains Mono, Fira Code, monospace", color="#c4c7cd"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        font=dict(size=12),
    ),
    margin=dict(l=60, r=20, t=60, b=40),
    height=450,
    xaxis=dict(
        gridcolor="#1e2028",
        tickfont=dict(size=10, color="#555960"),
    ),
    yaxis=dict(
        gridcolor="#1e2028",
        tickfont=dict(size=10, color="#555960"),
    ),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# CRE CONTEXT PANEL
# ─────────────────────────────────────────────

st.markdown("### Key Relationships for CRE Appraisal")

c1, c2, c3, c4 = st.columns(4)

if "DGS10" in df.columns and "DFF" in df.columns:
    spread_val = (df["DGS10"].dropna().iloc[-1] - df["DFF"].dropna().iloc[-1]) * 100
    with c1:
        color = "#00D4AA" if spread_val >= 0 else "#EF4444"
        st.markdown(f"""
        <div class="spread-box">
            <div class="metric-label">10Y – Fed Funds Spread</div>
            <div class="metric-value" style="color:{color}">{spread_val:.0f} bps</div>
            <div style="font-size:11px;color:#6b7080;margin-top:4px;">Negative = yield curve inversion</div>
        </div>
        """, unsafe_allow_html=True)

if "SOFR" in df.columns and "DFF" in df.columns:
    sofr_spread = (df["SOFR"].dropna().iloc[-1] - df["DFF"].dropna().iloc[-1]) * 100
    with c2:
        st.markdown(f"""
        <div class="spread-box">
            <div class="metric-label">SOFR – Fed Funds Spread</div>
            <div class="metric-value" style="color:#c4c7cd">{sofr_spread:.0f} bps</div>
            <div style="font-size:11px;color:#6b7080;margin-top:4px;">Usually tight — divergence flags stress</div>
        </div>
        """, unsafe_allow_html=True)

if "DGS10" in df.columns and "DFII10" in df.columns:
    breakeven = df["DGS10"].dropna().iloc[-1] - df["DFII10"].dropna().iloc[-1]
    with c3:
        st.markdown(f"""
        <div class="spread-box">
            <div class="metric-label">10Y Breakeven Inflation</div>
            <div class="metric-value" style="color:#F472B6">{breakeven:.2f}%</div>
            <div style="font-size:11px;color:#6b7080;margin-top:4px;">Market's expected avg inflation over 10Y</div>
        </div>
        """, unsafe_allow_html=True)

if "DGS10" in df.columns:
    t10_current = df["DGS10"].dropna().iloc[-1]
    cap_floor = t10_current + 1.50
    with c4:
        st.markdown(f"""
        <div class="spread-box">
            <div class="metric-label">Implied Cap Rate Floor</div>
            <div class="metric-value" style="color:#F59E0B">~{cap_floor:.1f}%</div>
            <div style="font-size:11px;color:#6b7080;margin-top:4px;">10Y + 150bps typical risk premium</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# RAW DATA TABLE (collapsible)
# ─────────────────────────────────────────────

with st.expander("📋 View Raw Data"):
    display_df = df.copy()
    display_df.index = display_df.index.strftime("%Y-%m-%d")
    # Rename columns to friendly names
    rename_map = {sid: info["name"] for sid, info in SERIES.items() if sid in display_df.columns}
    display_df = display_df.rename(columns=rename_map)
    display_df = display_df.round(2)
    st.dataframe(display_df.sort_index(ascending=False), use_container_width=True)

    # Download button
    csv = display_df.to_csv()
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv,
        file_name=f"rate_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
