"""
Equity & Market Indicators
Shiller CAPE ratio and other broad market valuation metrics.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from io import StringIO

# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Equity & Market Indicators",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)

CAPE_LONG_RUN_AVG = 17.0

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data(ttl=86400)
def load_cape_data():
    """
    Fetch Shiller CAPE history.
    Tries: 1) posix4e CSV, 2) posix4e JSON, 3) Shiller Excel.
    Returns a DataFrame with 'date' and 'cape' columns.
    """
    errors = []

    # Attempt 1: posix4e CSV
    try:
        resp = requests.get(
            "https://posix4e.github.io/shiller_wrapper_data/data/stock_market_data.csv",
            timeout=15,
        )
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df = df.rename(columns=str.lower)
        if "date_string" in df.columns:
            df = df.rename(columns={"date_string": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["cape"] = pd.to_numeric(df["cape"], errors="coerce")
        df = df[["date", "cape"]].dropna(subset=["cape"])
        df = df.sort_values("date").reset_index(drop=True)
        if len(df) > 0:
            return df
    except Exception as e:
        errors.append(f"CSV API: {e}")

    # Attempt 3: Shiller's Excel (requires xlrd)
    try:
        url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
        raw = pd.read_excel(url, sheet_name="Data", skiprows=7, header=1)
        date_col = raw.columns[0]
        cape_col = [c for c in raw.columns if "cape" in str(c).lower() or "p/e10" in str(c).lower()]
        if not cape_col:
            cape_col = [raw.columns[-1]]
        df = raw[[date_col, cape_col[0]]].copy()
        df.columns = ["date_frac", "cape"]
        df = df.dropna(subset=["cape"])
        df["date_frac"] = pd.to_numeric(df["date_frac"], errors="coerce")
        df = df.dropna(subset=["date_frac"])
        df["year"] = df["date_frac"].astype(int)
        df["month"] = ((df["date_frac"] % 1) * 12 + 1).round().astype(int).clip(1, 12)
        df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
        df = df[["date", "cape"]].sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        errors.append(f"Shiller Excel: {e}")

    st.error("Could not load CAPE data. Errors:\n" + "\n".join(errors))
    return pd.DataFrame(columns=["date", "cape"])


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

st.markdown("# 📊 Equity & Market Indicators")
st.markdown("*Shiller CAPE Ratio — S&P 500 Cyclically Adjusted P/E*")

cape_df = load_cape_data()

if cape_df.empty:
    st.error("No CAPE data available. Please try again later.")
    st.stop()

current_cape = cape_df["cape"].iloc[-1]
cape_date = cape_df["date"].iloc[-1]
percentile = (cape_df["cape"] < current_cape).mean() * 100
earnings_yield = (1 / current_cape) * 100

# ─────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────

c1, c2, c3 = st.columns(3)

with c1:
    color = "#EF4444" if current_cape > CAPE_LONG_RUN_AVG * 1.5 else (
        "#F59E0B" if current_cape > CAPE_LONG_RUN_AVG else "#00D4AA"
    )
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Shiller CAPE</div>
        <div class="metric-value" style="color:{color}">{current_cape:.1f}x</div>
        <div style="margin-top:6px; font-size:12px; color:#6b7080;">
            Long-run avg: {CAPE_LONG_RUN_AVG:.0f}x
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    pct_color = "#EF4444" if percentile > 80 else (
        "#F59E0B" if percentile > 50 else "#00D4AA"
    )
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Historical Percentile</div>
        <div class="metric-value" style="color:{pct_color}">{percentile:.0f}th</div>
        <div style="margin-top:6px; font-size:12px; color:#6b7080;">
            vs. all months since 1881
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">CAPE Earnings Yield</div>
        <div class="metric-value" style="color:#A78BFA">{earnings_yield:.2f}%</div>
        <div style="margin-top:6px; font-size:12px; color:#6b7080;">
            1 / CAPE — comparable to bond yields
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HISTORICAL CHART
# ─────────────────────────────────────────────

# Sidebar date range filter
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("### Date Range")
    range_options = {
        "10 Years": 120,
        "20 Years": 240,
        "50 Years": 600,
        "All (since 1881)": None,
    }
    selected_range = st.radio(
        "Select range",
        options=list(range_options.keys()),
        index=0,
        label_visibility="collapsed",
    )

months_back = range_options[selected_range]
if months_back is not None:
    chart_df = cape_df.tail(months_back)
else:
    chart_df = cape_df

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=chart_df["date"],
    y=chart_df["cape"],
    name="CAPE",
    line=dict(color="#A78BFA", width=2),
    hovertemplate="%{x|%b %Y}: %{y:.1f}x<extra>CAPE</extra>",
))

# Long-run average line
fig.add_hline(
    y=CAPE_LONG_RUN_AVG,
    line_dash="dash",
    line_color="#F59E0B",
    line_width=1,
    annotation_text=f"Long-run avg ({CAPE_LONG_RUN_AVG:.0f}x)",
    annotation_position="top left",
    annotation_font=dict(color="#F59E0B", size=11),
)

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
        title_text="CAPE Ratio",
        gridcolor="#1e2028",
        tickfont=dict(size=10, color="#555960"),
        ticksuffix="x",
    ),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# CAPE vs. SUBSEQUENT 10-YEAR RETURNS (Shiller reference)
# ─────────────────────────────────────────────

import os

st.markdown("### CAPE vs. Subsequent 10-Year Real Returns")

chart_path = os.path.join(os.path.dirname(__file__), "..", "assets", "shiller_cape_returns.png")
if os.path.exists(chart_path):
    st.image(chart_path, use_container_width=True)
    st.markdown(
        "<small style='color:#555'>Source: Robert Shiller, "
        "<em>Irrational Exuberance</em></small>",
        unsafe_allow_html=True,
    )
else:
    st.info("Reference chart not found. Place image at assets/shiller_cape_returns.png")

# ─────────────────────────────────────────────
# DATA TABLE
# ─────────────────────────────────────────────

with st.expander("📋 View Raw Data"):
    display_df = chart_df.copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m")
    display_df = display_df.rename(columns={"date": "Month", "cape": "CAPE"})
    display_df["CAPE"] = display_df["CAPE"].round(2)
    st.dataframe(display_df.sort_values("Month", ascending=False).reset_index(drop=True), use_container_width=True)

st.markdown(
    f"<small style='color:#555'>Data as of {cape_date.strftime('%B %Y')} · "
    f"Source: Robert Shiller / Yale · Updates monthly</small>",
    unsafe_allow_html=True,
)
