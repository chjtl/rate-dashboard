"""
Linneman Fund Flow Cap Rate Model — VECM implementation.  # v2

Reference: Larriva & Linneman (2022), JPIF Vol. 40 No. 2, pp. 119–169.
           Linneman REFAI Textbook, Chapter 9 Supplement C.

Variables per sector (column order in endog matrix):
    0 – cap_rate              (%, NCREIF NPI)
    1 – mortgage_debt_pct_gdp (%, FRED MDOAH+MDOAN / GDP)
    2 – unemployment_rate     (%, FRED UNRATE quarterly avg)

All modelled at quarterly frequency.
"""

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.vector_ar.vecm import (
        VECM,
        coint_johansen,
        select_coint_rank,
        select_order,
    )
    from statsmodels.tsa.stattools import grangercausalitytests
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False

# ── Constants ────────────────────────────────────────────────────────────────
CAP_IDX   = 0   # cap_rate column index in endog
MORT_IDX  = 1   # mortgage_debt_pct_gdp column index
UNEMP_IDX = 2   # unemployment_rate column index

MIN_OBS      = 20   # minimum quarters to attempt VECM
DEFAULT_LAGS = 2    # default k_ar_diff if lag selection fails
MAX_LAGS     = 6    # ceiling for lag search

# Textbook sensitivity benchmarks (Larriva & Linneman 2022, Figure 18)
TEXTBOOK_SENSITIVITY = {
    "multifamily": {"mortgage_100bp_bps": -22, "unemployment_100bp_bps": 1},
    "office":      {"mortgage_100bp_bps": -65, "unemployment_100bp_bps": 3},
}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _build_endog(data: pd.DataFrame, cap_col: str) -> pd.DataFrame:
    """Return 3-column DataFrame ordered [cap_rate, mortgage, unemployment]."""
    return data[[cap_col, "mortgage_debt_pct_gdp", "unemployment_rate"]].dropna()


def _select_lags(endog: pd.DataFrame) -> int:
    """Select k_ar_diff via AIC; fall back to DEFAULT_LAGS on any error."""
    try:
        maxlags = min(MAX_LAGS, max(1, len(endog) // 6))
        result = select_order(endog.values, maxlags=maxlags, deterministic="co")
        return max(1, result.selected_orders.get("aic", DEFAULT_LAGS))
    except Exception:
        return DEFAULT_LAGS


# ── Public model components ──────────────────────────────────────────────────

def run_johansen_test(endog: pd.DataFrame, k_ar_diff: int = 2) -> Dict:
    """
    Run Johansen trace cointegration test at 10% significance (det_order=0).

    Uses 10% critical values, consistent with Larriva & Linneman (2022) and
    standard practice in cointegration testing. k_ar_diff should match the
    VECM lag order.

    Returns dict with:
        rank          – selected cointegration rank (int)
        trace_stats   – list of trace statistics
        crit_vals_95  – list of 95% critical values (displayed for reference)
        crit_vals_90  – list of 90% critical values (used for rank selection)
        eigenvalues   – list of eigenvalues
        summary       – human-readable string
    """
    try:
        result = coint_johansen(endog.values, det_order=0, k_ar_diff=k_ar_diff)
        trace  = result.lr1.tolist()
        cv90   = result.cvt[:, 0].tolist()   # 90% critical values
        cv95   = result.cvt[:, 1].tolist()   # 95% critical values (display only)

        rank = 0
        for i, (stat, cv) in enumerate(zip(trace, cv90)):
            if stat > cv:
                rank = i + 1
            else:
                break

        return {
            "rank":          rank,
            "trace_stats":   trace,
            "crit_vals_95":  cv95,
            "crit_vals_90":  cv90,
            "eigenvalues":   result.eig.tolist(),
            "summary":       f"Johansen trace test: rank = {rank} (90% confidence)",
            "error":         None,
        }
    except Exception as exc:
        return {
            "rank": 1, "trace_stats": [], "crit_vals_95": [], "crit_vals_90": [],
            "eigenvalues": [], "summary": "Johansen test failed", "error": str(exc),
        }


def run_granger_tests(endog: pd.DataFrame, cap_col: str) -> Dict:
    """
    Granger causality tests:
      H₀: mortgage_debt_pct_gdp does NOT Granger-cause cap_rate
      H₀: unemployment_rate      does NOT Granger-cause cap_rate

    Returns p-values (F-test) at lag 4 for each predictor.
    """
    results = {}
    for predictor, key in [
        ("mortgage_debt_pct_gdp", "mortgage"),
        ("unemployment_rate",     "unemployment"),
    ]:
        try:
            pair = endog[[cap_col, predictor]].dropna()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc = grangercausalitytests(pair.values, maxlag=4, verbose=False)
            p_val = gc[4][0]["ssr_ftest"][1]
            results[key] = {
                "p_value":     round(float(p_val), 4),
                "significant": p_val < 0.10,
                "lag":         4,
                "error":       None,
            }
        except Exception as exc:
            results[key] = {
                "p_value": None, "significant": None, "lag": 4, "error": str(exc)
            }
    return results


def fit_vecm(endog: pd.DataFrame, coint_rank: int, k_ar_diff: int) -> Any:
    """Fit and return a VECMResults object."""
    coint_rank = max(1, min(coint_rank, endog.shape[1] - 1))
    model = VECM(
        endog.values,
        k_ar_diff=k_ar_diff,
        coint_rank=coint_rank,
        deterministic="co",   # restricted constant — Johansen Case 2
    )
    return model.fit()


def get_historical_fit(
    vecm_result: Any,
    endog: pd.DataFrame,
    cap_col: str,
) -> pd.Series:
    """
    Return in-sample fitted cap-rate levels from the VECM.

    In statsmodels 0.14+, fittedvalues returns 1-step-ahead level predictions
    for observations [k_ar : nobs_tot].  k_ar = k_ar_diff + 1.
    """
    k_ar  = vecm_result.k_ar          # k_ar = k_ar_diff + 1
    n_fit = len(vecm_result.fittedvalues)
    fitted_levels = vecm_result.fittedvalues[:, CAP_IDX]
    dates = endog.index[k_ar : k_ar + n_fit]
    return pd.Series(fitted_levels, index=dates, name="fitted")


def compute_r2_insample(
    vecm_result: Any,
    endog: pd.DataFrame,
    cap_col: str,
) -> Optional[float]:
    """
    R² on first-differences for the cap-rate equation.

    fittedvalues in statsmodels 0.14+ returns 1-step-ahead level predictions.
    We compute implied fitted differences and compare to actual differences.
    """
    try:
        k_ar  = vecm_result.k_ar      # k_ar = k_ar_diff + 1
        n_fit = len(vecm_result.fittedvalues)

        # Actual levels aligned to fittedvalues window
        actual_levels = endog[cap_col].values[k_ar : k_ar + n_fit]
        fitted_levels = vecm_result.fittedvalues[:, CAP_IDX]

        # R² on first differences to avoid spurious I(1) inflation
        d_actual = np.diff(actual_levels)
        d_fitted = np.diff(fitted_levels)
        n        = min(len(d_actual), len(d_fitted))
        d_actual, d_fitted = d_actual[:n], d_fitted[:n]

        ss_res = float(np.sum((d_actual - d_fitted) ** 2))
        ss_tot = float(np.sum((d_actual - d_actual.mean()) ** 2))
        return None if ss_tot == 0 else round(1 - ss_res / ss_tot, 4)
    except Exception:
        return None


def compute_r2_outsample(
    endog: pd.DataFrame,
    coint_rank: int,
    k_ar_diff: int,
    cap_col: str,
    train_frac: float = 0.80,
) -> Optional[float]:
    """
    Walk-forward out-of-sample R² in levels.
    Trains on the first train_frac of observations; tests on the rest.
    """
    try:
        n       = len(endog)
        cutoff  = int(n * train_frac)
        if cutoff < MIN_OBS:
            return None

        test_data = endog.iloc[cutoff:]
        actuals, preds = [], []

        for i in range(len(test_data)):
            window = endog.iloc[: cutoff + i]
            if len(window) < MIN_OBS:
                continue
            try:
                r    = fit_vecm(window, coint_rank, k_ar_diff)
                raw  = r.predict(steps=1)
                fc   = raw[0] if isinstance(raw, tuple) else raw
                preds.append(float(fc[0, CAP_IDX] if fc.ndim == 2 else fc[CAP_IDX]))
                actuals.append(float(test_data.iloc[i][cap_col]))
            except Exception:
                continue

        if len(actuals) < 3:
            return None

        a   = np.array(actuals)
        p   = np.array(preds)
        ss_res = float(np.sum((a - p) ** 2))
        ss_tot = float(np.sum((a - a.mean()) ** 2))
        return None if ss_tot == 0 else round(1 - ss_res / ss_tot, 4)
    except Exception:
        return None


def compute_sensitivity(vecm_result: Any) -> Dict:
    """
    Estimate cap-rate response (bps) to a 100bp unit shock in each regressor,
    using the 16-quarter cumulative non-orthogonalised IRF.

    A 100bp shock = 1 percentage-point, which equals 1 unit in all three
    series (each measured in %).
    """
    try:
        irf = vecm_result.irf(periods=16)
        cum = irf.cum_effects   # shape (17, k, k) — [horizon, response, impulse]
        return {
            "mortgage_100bp_bps":    round(float(cum[16, CAP_IDX, MORT_IDX])  * 100, 1),
            "unemployment_100bp_bps": round(float(cum[16, CAP_IDX, UNEMP_IDX]) * 100, 1),
            "error": None,
        }
    except Exception as exc:
        return {
            "mortgage_100bp_bps":    None,
            "unemployment_100bp_bps": None,
            "error": str(exc),
        }


# ── Office structural deviation monitor ──────────────────────────────────────

def run_office_deviation_model(data: pd.DataFrame) -> Dict:
    """
    Fit the fund flow VECM on pre-2020 office data, then project forward.

    The gap between the projection and actual post-2020 office cap rates
    quantifies the structural WFH premium — how much higher cap rates are
    than the historical fund flow relationship would predict.

    Returns dict with:
        actual_series_full   – full office cap rate history
        actual_series_post   – post-2020 actuals
        projected_series     – model projection from 2020 Q1 onward
        gap_series           – actual minus projected (positive = structural premium)
        current_gap          – latest gap in percentage points
        cutoff               – pd.Timestamp of break point
        error                – str if something failed, else None
    """
    cap_col = "office_cap_rate"
    if cap_col not in data.columns:
        return {"error": "office_cap_rate not in data — add it to ncreif_cap_rates.csv"}

    endog_full = _build_endog(data, cap_col)
    cutoff = pd.Timestamp("2020-01-01")
    endog_pre = endog_full[endog_full.index < cutoff]

    if len(endog_pre) < MIN_OBS:
        return {"error": f"Only {len(endog_pre)} pre-2020 quarters — need {MIN_OBS}+"}

    try:
        k_ar_diff  = _select_lags(endog_pre)
        johansen   = run_johansen_test(endog_pre, k_ar_diff=k_ar_diff)
        coint_rank = max(1, johansen["rank"])
        vecm_res   = fit_vecm(endog_pre, coint_rank, k_ar_diff)
    except Exception as exc:
        return {"error": f"Pre-2020 VECM failed: {exc}"}

    endog_post = endog_full[endog_full.index >= cutoff]
    n_post = len(endog_post)
    if n_post == 0:
        return {"error": "No post-2020 data available"}

    try:
        raw = vecm_res.predict(steps=n_post)
        fc  = raw[0] if isinstance(raw, tuple) else raw
        projected_vals = fc[:n_post, CAP_IDX]
        projected = pd.Series(projected_vals, index=endog_post.index, name="projected")
        actual_post = endog_post[cap_col]
        gap = actual_post - projected

        return {
            "actual_series_full":  endog_full[cap_col],
            "actual_series_post":  actual_post,
            "projected_series":    projected,
            "gap_series":          gap,
            "current_gap":         round(float(gap.iloc[-1]), 2) if len(gap) else None,
            "current_actual":      round(float(actual_post.iloc[-1]), 2) if len(actual_post) else None,
            "current_projected":   round(float(projected.iloc[-1]), 2) if len(projected) else None,
            "cutoff":              cutoff,
            "pre2020_coint_rank":  coint_rank,
            "error":               None,
        }
    except Exception as exc:
        return {"error": f"Projection failed: {exc}"}


# ── Master pipeline ──────────────────────────────────────────────────────────

def run_full_model(data: pd.DataFrame, sector: str) -> Dict:
    """
    Full VECM pipeline for one sector ('multifamily' or 'office').

    Parameters
    ----------
    data   : merged DataFrame from data_loader_linneman.merge_data()
    sector : 'multifamily' or 'office'

    Returns
    -------
    dict with keys:
        sector, cap_col, n_obs, current, last_date,
        q1_forecast, q4_forecast,
        signal_1q, signal_1q_bps, signal_4q, signal_4q_bps,
        actual_series, fitted_series,
        r2_insample, r2_outsample,
        sensitivity, johansen, granger, k_ar_diff,
        mort_series, unemp_series
        error (only present on failure)
    """
    if not STATSMODELS_OK:
        return {"error": "statsmodels not installed — run: pip install statsmodels"}

    cap_col = f"{sector}_cap_rate"
    if cap_col not in data.columns:
        return {"error": f"Column '{cap_col}' not found in merged data."}

    endog = _build_endog(data, cap_col)
    if len(endog) < MIN_OBS:
        return {"error": f"Only {len(endog)} quarters of data — need {MIN_OBS}+."}

    out: Dict = {
        "sector":  sector,
        "cap_col": cap_col,
        "n_obs":   len(endog),
        "mort_series":  data["mortgage_debt_pct_gdp"],
        "unemp_series": data["unemployment_rate"],
    }

    # ── Lag selection (must precede Johansen so both use same order) ──────────
    k_ar_diff = _select_lags(endog)

    # ── Johansen ──────────────────────────────────────────────────────────────
    johansen   = run_johansen_test(endog, k_ar_diff=k_ar_diff)
    out["johansen"] = johansen
    coint_rank = max(1, johansen["rank"])

    # ── Granger ───────────────────────────────────────────────────────────────
    out["granger"] = run_granger_tests(endog, cap_col)
    out["k_ar_diff"] = k_ar_diff
    try:
        vecm_res = fit_vecm(endog, coint_rank, k_ar_diff)
    except Exception as exc:
        return {**out, "error": f"VECM fitting failed: {exc}"}

    # ── Current level ─────────────────────────────────────────────────────────
    out["current"]   = float(endog[cap_col].iloc[-1])
    out["last_date"] = endog.index[-1]

    # ── Forecasts ─────────────────────────────────────────────────────────────
    try:
        raw = vecm_res.predict(steps=4)
        fc  = raw[0] if isinstance(raw, tuple) else raw   # shape (4, k) in levels
        out["q1_forecast"] = float(fc[0, CAP_IDX])
        out["q4_forecast"] = float(fc[3, CAP_IDX])
    except Exception as exc:
        out["q1_forecast"] = out["q4_forecast"] = None
        out["forecast_error"] = str(exc)

    # ── Directional signals ───────────────────────────────────────────────────
    def _signal(fcast_val):
        if fcast_val is None:
            return "neutral", 0.0
        diff_bps = (fcast_val - out["current"]) * 100
        if diff_bps < -10:
            label = "compress"
        elif diff_bps > 10:
            label = "expand"
        else:
            label = "neutral"
        return label, round(diff_bps, 1)

    out["signal_1q"], out["signal_1q_bps"] = _signal(out.get("q1_forecast"))
    out["signal_4q"], out["signal_4q_bps"] = _signal(out.get("q4_forecast"))

    # ── Historical fit ────────────────────────────────────────────────────────
    out["actual_series"] = endog[cap_col]
    try:
        out["fitted_series"] = get_historical_fit(vecm_res, endog, cap_col)
    except Exception:
        out["fitted_series"] = pd.Series(dtype=float)

    # ── Model diagnostics ─────────────────────────────────────────────────────
    out["r2_insample"]  = compute_r2_insample(vecm_res, endog, cap_col)
    out["r2_outsample"] = compute_r2_outsample(
        endog, coint_rank, k_ar_diff, cap_col
    )
    out["sensitivity"] = compute_sensitivity(vecm_res)

    return out
