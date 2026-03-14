"""
Linneman Fund Flow Cap Rate Model
Streamlit page — Larriva & Linneman (2022), JPIF Vol. 40 No. 2, pp. 119-169.
"""

import os
import sys

# Ensure root-level modules are importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

from data_loader_linneman import (
    get_fred_api_key,
    load_fred_data,
    load_ncreif_data,
    merge_data,
    get_ncreif_last_updated,
    ncreif_needs_refresh,
    NCREIF_CSV_PATH,
)
from model_linneman import (
    run_full_model,
    run_office_deviation_model,
    TEXTBOOK_SENSITIVITY,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Linneman Cap Rate Model",
    page_icon="🏢",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d0f12; }
    .metric-card {
        background: #14161b;
        border: 1px solid #1e2028;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        height: 100%;
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
    .signal-expand   { color: #EF4444; font-size: 20px; font-weight: 700; }
    .signal-compress { color: #00D4AA; font-size: 20px; font-weight: 700; }
    .signal-neutral  { color: #8b8f98; font-size: 20px; font-weight: 700; }
    .spread-box {
        background: #14161b;
        border: 1px solid #1e2028;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .deviation-box {
        background: #14161b;
        border: 1px solid #1e2028;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
    }
    .vintage-tag {
        font-size: 11px;
        color: #555960;
        font-family: monospace;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Plotly base layout ────────────────────────────────────────────────────────
_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#14161b",
    plot_bgcolor="#14161b",
    font=dict(family="JetBrains Mono, Fira Code, monospace", color="#c4c7cd"),
    margin=dict(l=60, r=20, t=50, b=40),
    height=380,
    xaxis=dict(gridcolor="#1e2028", tickfont=dict(size=10, color="#555960")),
    yaxis=dict(gridcolor="#1e2028", tickfont=dict(size=10, color="#555960")),
    hovermode="x unified",
)

SECTOR_COLORS = {
    "multifamily": "#00D4AA",
    "industrial":  "#F59E0B",
    "office":      "#3B82F6",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fmt(val, decimals=2, suffix="%") -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    return f"{val:.{decimals}f}{suffix}"


def _r2_color(r2) -> str:
    if r2 is None or pd.isna(r2):
        return "#8b8f98"
    return "#00D4AA" if r2 >= 0.4 else "#F59E0B" if r2 >= 0.20 else "#EF4444"


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🏢 Linneman Fund Flow Cap Rate Model")
st.markdown(
    "*Larriva & Linneman (2022) · JPIF Vol. 40 No. 2 · VECM: cap rate ~ "
    "mortgage debt/GDP + unemployment*"
)

# ── NCREIF staleness warning ──────────────────────────────────────────────────
ncreif_updated = get_ncreif_last_updated()
if ncreif_needs_refresh():
    if ncreif_updated is None:
        st.warning(
            "**NCREIF data not found.**  "
            f"Save `data/ncreif_cap_rates.csv` to `{NCREIF_CSV_PATH}` to enable "
            "the VECM model.  See `README_linneman.md` for instructions.",
            icon="⚠️",
        )
    else:
        days_old = (datetime.now() - ncreif_updated).days
        st.warning(
            f"**NCREIF data is {days_old} days old** (last updated "
            f"{ncreif_updated.strftime('%Y-%m-%d')}).  "
            "Download the latest NPI cap rate report from ncreif.org and refresh "
            "`data/ncreif_cap_rates.csv`.",
            icon="⚠️",
        )

# ── Data loading ──────────────────────────────────────────────────────────────
api_key = get_fred_api_key()
if not api_key:
    st.error(
        "**FRED API key not found.**  "
        "Add `FRED_API_KEY` to `.streamlit/secrets.toml` or a `.env` file."
    )
    st.stop()

col_btn, col_vintage = st.columns([1, 4])
with col_btn:
    if st.button("🔄  Refresh FRED Data", use_container_width=True):
        load_fred_data.clear()
        st.rerun()

with st.spinner("Loading FRED data…"):
    fred_df, fred_err = load_fred_data(api_key)

if fred_err:
    st.error(f"FRED pull failed: {fred_err}")
    if fred_df.empty:
        st.stop()

ncreif_df = load_ncreif_data()
merged_df  = merge_data(fred_df, ncreif_df)

# ── Vintage ───────────────────────────────────────────────────────────────────
with col_vintage:
    parts = []
    if not fred_df.empty:
        parts.append(f"FRED: {fred_df.index[-1].strftime('%Y-Q') + str(fred_df.index[-1].quarter)}")
    if not ncreif_df.empty:
        latest_ncreif = ncreif_df.index[-1]
        parts.append(f"NCREIF: {latest_ncreif.strftime('%Y-Q') + str(latest_ncreif.quarter)}")
    if ncreif_updated:
        parts.append(f"CSV updated: {ncreif_updated.strftime('%Y-%m-%d')}")
    st.markdown(
        '<div class="vintage-tag">Data vintage — ' + " &nbsp;·&nbsp; ".join(parts) + "</div>",
        unsafe_allow_html=True,
    )

has_model_data = not merged_df.empty and len(merged_df) >= 20

# ── Run models ────────────────────────────────────────────────────────────────
mf_res = ind_res = None
office_dev = {}

if has_model_data:
    with st.spinner("Fitting VECM models…"):
        mf_res  = run_full_model(merged_df, "multifamily")
        ind_res = run_full_model(merged_df, "industrial")
        if "office_cap_rate" in merged_df.columns:
            office_dev = run_office_deviation_model(merged_df)
else:
    if ncreif_df.empty:
        st.info(
            "**Model requires NCREIF cap rate data.**  "
            "Add `data/ncreif_cap_rates.csv` to enable forecasting.  "
            "Macro charts below are still available.",
            icon="ℹ️",
        )
    else:
        st.warning(
            f"Only {len(merged_df)} overlapping quarters found — need 20+.",
            icon="⚠️",
        )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SIGNAL CARDS (Multifamily + Industrial)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### Signal Cards")
card_col1, card_col2 = st.columns(2)

for col, sector, res, label in [
    (card_col1, "multifamily", mf_res,  "Multifamily"),
    (card_col2, "industrial",  ind_res, "Industrial"),
]:
    with col:
        color = SECTOR_COLORS[sector]
        if res and "error" not in res:
            q1f  = res.get("q1_forecast")
            q4f  = res.get("q4_forecast")
            sig1 = res.get("signal_1q", "neutral")
            bps1 = res.get("signal_1q_bps", 0.0)
            sig4 = res.get("signal_4q", "neutral")
            bps4 = res.get("signal_4q_bps", 0.0)
            arrow1 = {"compress": "▼", "expand": "▲", "neutral": "→"}.get(sig1, "→")
            arrow4 = {"compress": "▼", "expand": "▲", "neutral": "→"}.get(sig4, "→")
            css1 = f"signal-{sig1}"
            css4 = f"signal-{sig4}"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color}">{_fmt(res['current'])}</div>
                <div style="margin-top:8px;font-size:11px;color:#8b8f98;font-family:monospace;">
                    CURRENT NCREIF CAP RATE
                </div>
                <hr style="border-color:#1e2028;margin:10px 0;">
                <div style="display:flex;justify-content:space-around;">
                    <div>
                        <div style="font-size:10px;color:#8b8f98;text-transform:uppercase;letter-spacing:1px;">
                            1Q Forecast
                        </div>
                        <div class="metric-value" style="font-size:22px;color:{color}">
                            {_fmt(q1f)}
                        </div>
                        <div class="{css1}" style="font-size:16px;">{arrow1} {bps1:+.0f} bps</div>
                    </div>
                    <div style="border-left:1px solid #1e2028;"></div>
                    <div>
                        <div style="font-size:10px;color:#8b8f98;text-transform:uppercase;letter-spacing:1px;">
                            4Q Forecast
                        </div>
                        <div class="metric-value" style="font-size:22px;color:{color}">
                            {_fmt(q4f)}
                        </div>
                        <div class="{css4}" style="font-size:16px;">{arrow4} {bps4:+.0f} bps</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif res and "error" in res:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div style="color:#EF4444;font-size:13px;margin-top:10px;">
                    Model error: {res['error']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div style="color:#555960;font-size:13px;margin-top:10px;">
                    Awaiting NCREIF data
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ACTUAL vs FITTED (Multifamily / Industrial)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### Chart 1 — Actual vs Model-Fitted Cap Rates")

if (mf_res and "error" not in mf_res) or (ind_res and "error" not in ind_res):
    c1_sector = st.radio(
        "Sector", ["Multifamily", "Industrial"], horizontal=True, key="chart1_sector"
    )
    res1   = mf_res if c1_sector == "Multifamily" else ind_res
    color1 = SECTOR_COLORS["multifamily" if c1_sector == "Multifamily" else "industrial"]

    if res1 and "error" not in res1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=res1["actual_series"].index,
            y=res1["actual_series"].values,
            name="Actual (NCREIF)",
            line=dict(color=color1, width=2),
            hovertemplate="%{x|%Y-Q%q}: %{y:.2f}%<extra>Actual</extra>",
        ))
        if not res1["fitted_series"].empty:
            fig1.add_trace(go.Scatter(
                x=res1["fitted_series"].index,
                y=res1["fitted_series"].values,
                name="VECM 1-step fit",
                line=dict(color="#F472B6", width=1.5, dash="dash"),
                hovertemplate="%{x|%Y-Q%q}: %{y:.2f}%<extra>Fitted</extra>",
            ))
        last_date = res1["last_date"]
        q1f = res1.get("q1_forecast")
        q4f = res1.get("q4_forecast")
        if q1f is not None:
            next_q = last_date + pd.DateOffset(months=3)
            fig1.add_trace(go.Scatter(
                x=[next_q], y=[q1f], mode="markers",
                marker=dict(symbol="star", size=12, color="#F472B6"),
                name="1Q Forecast",
                hovertemplate=f"1Q Forecast: {q1f:.2f}%<extra></extra>",
            ))
        if q4f is not None:
            four_q = last_date + pd.DateOffset(months=12)
            fig1.add_trace(go.Scatter(
                x=[four_q], y=[q4f], mode="markers",
                marker=dict(symbol="diamond", size=12, color="#A78BFA"),
                name="4Q Forecast",
                hovertemplate=f"4Q Forecast: {q4f:.2f}%<extra></extra>",
            ))
            fig1.add_trace(go.Scatter(
                x=[last_date, next_q if q1f else four_q, four_q],
                y=[res1["current"], q1f or res1["current"], q4f],
                mode="lines",
                line=dict(color="#555960", width=1, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))
        fig1.add_vline(x=last_date, line_dash="dot", line_color="#2d3139", line_width=1)
        fig1.update_layout(**{
            **_LAYOUT,
            "yaxis": {**_LAYOUT["yaxis"], "title_text": "Cap Rate (%)", "ticksuffix": "%"},
            "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        })
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Model not available for this sector.")
else:
    st.info("Cap rate chart requires NCREIF data and a fitted model.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — OFFICE STRUCTURAL DEVIATION MONITOR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### Office Structural Deviation Monitor")
st.markdown(
    "<small style='color:#555960'>"
    "VECM estimated on pre-2020 data. Gap = how much higher office cap rates are "
    "than the historical fund flow model predicts — a direct measure of the "
    "structural WFH demand discount priced in by the market."
    "</small>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

if office_dev and "error" not in office_dev:
    current_gap       = office_dev.get("current_gap")
    current_actual    = office_dev.get("current_actual")
    current_projected = office_dev.get("current_projected")

    dev_c1, dev_c2, dev_c3 = st.columns(3)
    with dev_c1:
        st.markdown(f"""<div class="deviation-box">
            <div class="metric-label">Actual Office Cap Rate</div>
            <div class="metric-value" style="color:#3B82F6">{_fmt(current_actual)}</div>
        </div>""", unsafe_allow_html=True)
    with dev_c2:
        st.markdown(f"""<div class="deviation-box">
            <div class="metric-label">Fund Flow Model Predicts</div>
            <div class="metric-value" style="color:#8b8f98">{_fmt(current_projected)}</div>
            <div style="font-size:10px;color:#555960;margin-top:4px;">pre-2020 VECM projection</div>
        </div>""", unsafe_allow_html=True)
    with dev_c3:
        gap_color = "#EF4444" if (current_gap or 0) > 0 else "#00D4AA"
        gap_label = "Structural WFH Premium" if (current_gap or 0) > 0 else "Below Model"
        st.markdown(f"""<div class="deviation-box">
            <div class="metric-label">{gap_label}</div>
            <div class="metric-value" style="color:{gap_color}">
                {f'+{current_gap:.2f}%' if current_gap and current_gap > 0 else _fmt(current_gap)}
            </div>
            <div style="font-size:10px;color:#555960;margin-top:4px;">
                {f'{round(current_gap*100):.0f} bps above model' if current_gap and current_gap > 0 else ''}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Deviation chart
    fig_dev = go.Figure()

    # Full historical actual (pre-2020 in muted blue, post-2020 in bright blue)
    actual_full = office_dev["actual_series_full"]
    pre2020 = actual_full[actual_full.index < office_dev["cutoff"]]
    post2020 = office_dev["actual_series_post"]
    proj     = office_dev["projected_series"]
    gap_s    = office_dev["gap_series"]

    fig_dev.add_trace(go.Scatter(
        x=pre2020.index, y=pre2020.values,
        name="Actual office (pre-2020)",
        line=dict(color="#3B82F6", width=2),
        hovertemplate="%{x|%Y-Q%q}: %{y:.2f}%<extra>Actual</extra>",
    ))
    fig_dev.add_trace(go.Scatter(
        x=post2020.index, y=post2020.values,
        name="Actual office (post-2020)",
        line=dict(color="#3B82F6", width=2.5),
        hovertemplate="%{x|%Y-Q%q}: %{y:.2f}%<extra>Actual</extra>",
    ))
    fig_dev.add_trace(go.Scatter(
        x=proj.index, y=proj.values,
        name="Fund flow model projection",
        line=dict(color="#8b8f98", width=1.5, dash="dash"),
        hovertemplate="%{x|%Y-Q%q}: %{y:.2f}%<extra>Model</extra>",
    ))

    # Shade the gap
    fig_dev.add_trace(go.Scatter(
        x=pd.concat([post2020, proj[::-1]]).index,
        y=pd.concat([post2020, proj[::-1]]).values,
        fill="toself",
        fillcolor="rgba(239,68,68,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig_dev.add_vline(
        x=office_dev["cutoff"], line_dash="dot",
        line_color="#EF4444", line_width=1,
        annotation_text="COVID / WFH break",
        annotation_font_color="#EF4444",
        annotation_font_size=10,
    )

    fig_dev.update_layout(**{
        **_LAYOUT,
        "height": 400,
        "yaxis": {**_LAYOUT["yaxis"], "title_text": "Cap Rate (%)", "ticksuffix": "%"},
        "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    })
    st.plotly_chart(fig_dev, use_container_width=True)

elif office_dev and "error" in office_dev:
    st.info(f"Office deviation monitor unavailable: {office_dev['error']}", icon="ℹ️")
else:
    st.info("Office deviation monitor requires `office_cap_rate` in the NCREIF CSV.", icon="ℹ️")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MACRO CHARTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### Chart 2 — Mortgage Debt Outstanding as % of GDP")

if not fred_df.empty and "mortgage_debt_pct_gdp" in fred_df.columns:
    mort = fred_df["mortgage_debt_pct_gdp"].dropna()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=mort.index, y=mort.values,
        name="Mortgage Debt / GDP",
        line=dict(color="#F59E0B", width=2),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.06)",
        hovertemplate="%{x|%Y-Q%q}: %{y:.1f}%<extra></extra>",
    ))
    peak_idx = mort.idxmax()
    peak_val = mort.max()
    fig2.add_annotation(
        x=peak_idx, y=peak_val,
        text=f"Peak<br>{peak_val:.1f}%",
        showarrow=True, arrowhead=2, arrowcolor="#EF4444",
        font=dict(color="#EF4444", size=11),
        bgcolor="#1a1c22", bordercolor="#EF4444", ax=40, ay=-40,
    )
    current_val  = mort.iloc[-1]
    current_date = mort.index[-1]
    fig2.add_annotation(
        x=current_date, y=current_val,
        text=f"Current<br>{current_val:.1f}%",
        showarrow=True, arrowhead=2, arrowcolor="#00D4AA",
        font=dict(color="#00D4AA", size=11),
        bgcolor="#1a1c22", bordercolor="#00D4AA", ax=-40, ay=-40,
    )
    fig2.update_layout(**{**_LAYOUT, "yaxis": {**_LAYOUT["yaxis"], "title_text": "% of GDP", "ticksuffix": "%"}})
    st.plotly_chart(fig2, use_container_width=True)

    c2a, c2b, c2c = st.columns(3)
    with c2a:
        st.markdown(f"""<div class="spread-box">
            <div class="metric-label">Current</div>
            <div class="metric-value" style="color:#F59E0B">{current_val:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c2b:
        st.markdown(f"""<div class="spread-box">
            <div class="metric-label">Peak</div>
            <div class="metric-value" style="color:#EF4444">{peak_val:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c2c:
        diff = current_val - peak_val
        color_diff = "#00D4AA" if diff < 0 else "#EF4444"
        st.markdown(f"""<div class="spread-box">
            <div class="metric-label">vs Peak</div>
            <div class="metric-value" style="color:{color_diff}">{diff:+.1f}%</div>
        </div>""", unsafe_allow_html=True)
else:
    st.info("FRED data unavailable — mortgage debt chart cannot be rendered.")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("### Chart 3 — Unemployment Rate (Quarterly Average)")

if not fred_df.empty and "unemployment_rate" in fred_df.columns:
    unemp = fred_df["unemployment_rate"].dropna()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=unemp.index, y=unemp.values,
        name="Unemployment Rate",
        line=dict(color="#A78BFA", width=2),
        hovertemplate="%{x|%Y-Q%q}: %{y:.1f}%<extra></extra>",
    ))
    fig3.update_layout(**{**_LAYOUT, "yaxis": {**_LAYOUT["yaxis"], "title_text": "Rate (%)", "ticksuffix": "%"}})
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("FRED data unavailable — unemployment chart cannot be rendered.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SENSITIVITY TABLE (Multifamily + Industrial)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### Sensitivity Table — Cap Rate Response to 100bp Shock")
st.markdown(
    "<small style='color:#555960'>Multifamily textbook benchmarks from Larriva & Linneman (2022) Figure 18. "
    "Industrial benchmarks are model-estimated only (not in original paper). "
    "16-quarter cumulative non-orthogonalised IRF.</small>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)


def _sens_cell(val, textbook_val=None) -> str:
    if val is None:
        return "N/A"
    if textbook_val is not None:
        color = "#00D4AA" if val * textbook_val > 0 else "#F59E0B"
    else:
        color = "#00D4AA" if val < 0 else "#EF4444"
    return f'<span style="color:{color};font-family:monospace;font-weight:700;">{val:+.0f} bps</span>'


tb_mf = TEXTBOOK_SENSITIVITY["multifamily"]
mf_sens  = mf_res.get("sensitivity", {})  if mf_res  else {}
ind_sens = ind_res.get("sensitivity", {}) if ind_res else {}

st.markdown(f"""
<table style="width:100%;border-collapse:collapse;font-size:13px;color:#c4c7cd;">
  <thead>
    <tr style="border-bottom:1px solid #1e2028;">
      <th style="text-align:left;padding:8px 12px;color:#8b8f98;">Shock</th>
      <th style="text-align:center;padding:8px 12px;color:#00D4AA;">Multifamily<br><small>Textbook</small></th>
      <th style="text-align:center;padding:8px 12px;color:#00D4AA;">Multifamily<br><small>Model Est.</small></th>
      <th style="text-align:center;padding:8px 12px;color:#F59E0B;">Industrial<br><small>Model Est.</small></th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom:1px solid #1e2028;">
      <td style="padding:8px 12px;">+100bp Mortgage Debt / GDP</td>
      <td style="text-align:center;padding:8px 12px;font-family:monospace;color:#8b8f98;">
          {tb_mf["mortgage_100bp_bps"]:+d} bps
      </td>
      <td style="text-align:center;padding:8px 12px;">
          {_sens_cell(mf_sens.get("mortgage_100bp_bps"), tb_mf["mortgage_100bp_bps"])}
      </td>
      <td style="text-align:center;padding:8px 12px;">
          {_sens_cell(ind_sens.get("mortgage_100bp_bps"))}
      </td>
    </tr>
    <tr>
      <td style="padding:8px 12px;">+100bp Unemployment Rate</td>
      <td style="text-align:center;padding:8px 12px;font-family:monospace;color:#8b8f98;">
          {tb_mf["unemployment_100bp_bps"]:+d} bps
      </td>
      <td style="text-align:center;padding:8px 12px;">
          {_sens_cell(mf_sens.get("unemployment_100bp_bps"), tb_mf["unemployment_100bp_bps"])}
      </td>
      <td style="text-align:center;padding:8px 12px;">
          {_sens_cell(ind_sens.get("unemployment_100bp_bps"))}
      </td>
    </tr>
  </tbody>
</table>
<p style="font-size:11px;color:#555960;margin-top:6px;">
  Textbook source: Larriva &amp; Linneman (2022) Figure 18 / Linneman REFAI Ch. 9 Supplement C.<br>
  Model sign convention: negative = cap rate falls (compression), positive = cap rate rises (expansion).
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MODEL DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🔬 Model Diagnostics"):
    for sector_label, res, color in [
        ("Multifamily", mf_res,  "#00D4AA"),
        ("Industrial",  ind_res, "#F59E0B"),
    ]:
        st.markdown(f"#### {sector_label}")
        if not res:
            st.markdown("*No results — NCREIF data required.*")
            continue
        if "error" in res:
            st.error(res["error"])
            continue

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            r2i = res.get("r2_insample")
            st.markdown(f"""<div class="spread-box">
                <div class="metric-label">In-Sample R²</div>
                <div class="metric-value" style="color:{_r2_color(r2i)};font-size:22px;">
                    {_fmt(r2i, 3, '')}
                </div>
                <div style="font-size:10px;color:#6b7080;margin-top:4px;">on Δcap rate</div>
            </div>""", unsafe_allow_html=True)
        with d2:
            r2o = res.get("r2_outsample")
            st.markdown(f"""<div class="spread-box">
                <div class="metric-label">OOS R²</div>
                <div class="metric-value" style="color:{_r2_color(r2o)};font-size:22px;">
                    {_fmt(r2o, 3, '')}
                </div>
                <div style="font-size:10px;color:#6b7080;margin-top:4px;">80/20 walk-fwd</div>
            </div>""", unsafe_allow_html=True)
        with d3:
            joh  = res.get("johansen", {})
            rank = joh.get("rank", "—")
            st.markdown(f"""<div class="spread-box">
                <div class="metric-label">Coint Rank</div>
                <div class="metric-value" style="color:{color};font-size:22px;">{rank}</div>
                <div style="font-size:10px;color:#6b7080;margin-top:4px;">Johansen trace (90%)</div>
            </div>""", unsafe_allow_html=True)
        with d4:
            st.markdown(f"""<div class="spread-box">
                <div class="metric-label">VECM Lags</div>
                <div class="metric-value" style="color:{color};font-size:22px;">
                    {res.get("k_ar_diff", "—")}
                </div>
                <div style="font-size:10px;color:#6b7080;margin-top:4px;">k_ar_diff (AIC)</div>
            </div>""", unsafe_allow_html=True)

        joh = res.get("johansen", {})
        if joh.get("trace_stats"):
            st.markdown("<br>**Johansen Trace Test** (rank selected at 90% significance)", unsafe_allow_html=True)
            rows = []
            cv90 = joh.get("crit_vals_90", joh.get("crit_vals_95", []))
            cv95 = joh.get("crit_vals_95", [])
            for i, (ts, cv90v, cv95v) in enumerate(
                zip(joh["trace_stats"], cv90, cv95)
            ):
                reject90 = "✅ Reject" if ts > cv90v else "❌ Fail"
                reject95 = "✅ Reject" if ts > cv95v else "❌ Fail"
                rows.append({
                    "H₀: rank ≤ r": i,
                    "Trace Stat": f"{ts:.2f}",
                    "90% CV": f"{cv90v:.2f}",
                    "Decision (90%)": reject90,
                    "95% CV": f"{cv95v:.2f}",
                    "Decision (95%)": reject95,
                })
            st.dataframe(pd.DataFrame(rows).set_index("H₀: rank ≤ r"), use_container_width=True)

        granger = res.get("granger", {})
        if granger:
            st.markdown("**Granger Causality (H₀: does NOT Granger-cause cap rate)**")
            gc_rows = []
            for key, label in [("mortgage", "Mortgage Debt / GDP"), ("unemployment", "Unemployment Rate")]:
                g = granger.get(key, {})
                p = g.get("p_value")
                sig = g.get("significant")
                gc_rows.append({
                    "Predictor": label,
                    "F-test p-value (lag 4)": f"{p:.4f}" if p is not None else "N/A",
                    "Significant at 10%?": "✅ Yes" if sig else "❌ No",
                })
            st.dataframe(pd.DataFrame(gc_rows).set_index("Predictor"), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="font-size:11px;color:#555960;line-height:1.6;">
<strong style="color:#8b8f98;">Forecasting sectors:</strong>
Multifamily and Industrial — strongest cointegrating relationships with fund flow variables
(Tsui &amp; Morgan 2025; adj. R² = 0.88 and 0.82 respectively).<br>
<strong style="color:#8b8f98;">Office deviation monitor:</strong>
Pre-2020 VECM projection vs. current actuals. Gap represents the structural WFH demand discount.
Office cap rates are not forecast directly due to the post-2020 structural break.<br><br>
<strong style="color:#8b8f98;">Data sources:</strong>
FRED ASTMA (total mortgages), GDP, UNRATE — pulls automatically, refreshes daily.<br>
NCREIF NPI appraisal-based cap rates — lags transactions by ~2–4 quarters.
Requires manual quarterly refresh from <a href="https://www.ncreif.org" style="color:#555960;">ncreif.org</a>.<br><br>
<strong style="color:#8b8f98;">Model:</strong>
Replicates Larriva &amp; Linneman (2022), <em>Journal of Property Investment &amp; Finance</em>, Vol. 40 No. 2, pp. 119–169.
VECM with Johansen cointegration (10% significance). Forecasts are statistical estimates, not investment advice.
</div>
""", unsafe_allow_html=True)
