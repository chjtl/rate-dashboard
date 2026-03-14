"""
Data loader for the Linneman Fund Flow Cap Rate Model.

Pulls from FRED:
  - ASTMA  – all sectors total mortgages, asset level ($M quarterly, Z.1)
  - GDP    – nominal GDP ($B, quarterly)
  - UNRATE – unemployment rate (%, monthly → quarterly avg)

Loads from CSV:
  - data/ncreif_cap_rates.csv  (date, multifamily_cap_rate, office_cap_rate)
"""

import os
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from fredapi import Fred

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
NCREIF_CSV_PATH = os.path.join(_HERE, "data", "ncreif_cap_rates.csv")
NCREIF_STALE_DAYS = 95


# ── API key ──────────────────────────────────────────────────────────────────

def get_fred_api_key() -> Optional[str]:
    """Return FRED API key from Streamlit secrets or .env / environment."""
    try:
        return st.secrets["FRED_API_KEY"]
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_HERE, ".env"))
    except ImportError:
        pass
    return os.getenv("FRED_API_KEY")


# ── FRED data ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def load_fred_data(api_key: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Pull quarterly FRED series needed for the Linneman model.

    Returns
    -------
    df : DataFrame
        Columns: mortgage_debt_pct_gdp, unemployment_rate
        Index  : quarterly DatetimeIndex (QS frequency)
    error : str or None
        Non-None when the pull fails; df will be empty in that case.
    """
    try:
        fred = Fred(api_key=api_key)

        astma  = fred.get_series("ASTMA")   # all sectors total mortgages, $M (Z.1)
        gdp    = fred.get_series("GDP")     # nominal GDP, $B
        unrate = fred.get_series("UNRATE")  # unemployment rate, monthly

        # Normalise all to QS (quarter-start) dates
        mort_q   = astma.resample("QS").last() / 1000  # convert $M → $B
        gdp_q    = gdp.resample("QS").last()
        unrate_q = unrate.resample("QS").mean()

        df = pd.DataFrame({
            "mortgage_debt":    mort_q,
            "gdp":              gdp_q,
            "unemployment_rate": unrate_q,
        }).dropna()

        df["mortgage_debt_pct_gdp"] = (df["mortgage_debt"] / df["gdp"]) * 100
        df = df[["mortgage_debt_pct_gdp", "unemployment_rate"]]
        df.index.name = "date"

        return df, None

    except Exception as exc:
        return pd.DataFrame(), str(exc)


# ── NCREIF CSV ───────────────────────────────────────────────────────────────

def load_ncreif_data() -> pd.DataFrame:
    """
    Load NCREIF NPI cap rate CSV.

    Required columns (decimal %, e.g. 4.50 means 4.50%):
        date                  – YYYY-MM-DD, first day of quarter
        multifamily_cap_rate
        industrial_cap_rate

    Optional column (used for deviation monitor):
        office_cap_rate

    Returns empty DataFrame if file is missing, malformed, or lacks required
    columns.
    """
    if not os.path.exists(NCREIF_CSV_PATH):
        return pd.DataFrame()

    try:
        df = pd.read_csv(NCREIF_CSV_PATH, parse_dates=["date"])
        df = df.set_index("date")
        # Normalise to QS in case dates aren't already quarter-start
        df.index = df.index.to_period("Q").to_timestamp()
        df.index.name = "date"
        df = df.sort_index()

        required = {"multifamily_cap_rate", "industrial_cap_rate"}
        if not required.issubset(df.columns):
            return pd.DataFrame()

        keep = sorted(required)
        if "office_cap_rate" in df.columns:
            keep.append("office_cap_rate")

        return df[keep].dropna(how="all")

    except Exception:
        return pd.DataFrame()


def merge_data(fred_df: pd.DataFrame, ncreif_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join FRED and NCREIF on quarterly date index."""
    if fred_df.empty or ncreif_df.empty:
        return pd.DataFrame()
    return fred_df.join(ncreif_df, how="inner").dropna()


# ── Vintage helpers ──────────────────────────────────────────────────────────

def get_ncreif_last_updated() -> Optional[datetime]:
    """Return file mtime of the NCREIF CSV, or None if not found."""
    if not os.path.exists(NCREIF_CSV_PATH):
        return None
    return datetime.fromtimestamp(os.path.getmtime(NCREIF_CSV_PATH))


def ncreif_needs_refresh() -> bool:
    """Return True if the NCREIF CSV is missing or older than NCREIF_STALE_DAYS."""
    ts = get_ncreif_last_updated()
    if ts is None:
        return True
    return (datetime.now() - ts).days > NCREIF_STALE_DAYS
