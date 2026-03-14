# Linneman Fund Flow Cap Rate Model — Setup & Quarterly Refresh Guide

## Overview

This page replicates the cap rate fund flow model from:

> Larriva & Linneman (2022), *Journal of Property Investment & Finance*, Vol. 40 No. 2, pp. 119–169.
> Also documented in Linneman's *REFAI* textbook, Chapter 9 Supplement C.

**Model:** Vector Error Correction Model (VECM) forecasting cap rates one and four quarters ahead using:
- Lagged NCREIF NPI cap rates (by sector)
- Mortgage debt outstanding as % of GDP (fund flow proxy)
- Unemployment rate (risk/demand proxy)

---

## First-Time Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

New packages added: `statsmodels`, `python-dotenv`

### 2. FRED API key

The model pulls mortgage debt, GDP, and unemployment from FRED automatically.
Your key is shared with the main dashboard — no additional setup needed if
`FRED_API_KEY` is already in `.streamlit/secrets.toml`.

If using a `.env` file instead:
```
FRED_API_KEY=your_key_here
```

### 3. NCREIF cap rate data

NCREIF data requires a **manual quarterly download** (free NCREIF membership required).

**Steps:**
1. Log in at [ncreif.org](https://www.ncreif.org)
2. Navigate to: **Data → NPI → Performance → Cap Rate Trends**
3. Download the quarterly cap rate data for **Apartment (Multifamily)** and **Office**
4. Format the data as a CSV with these exact columns:

| Column | Format | Example |
|---|---|---|
| `date` | YYYY-MM-DD, first day of quarter | `2024-01-01` |
| `multifamily_cap_rate` | Decimal percent (not basis points) | `4.85` |
| `office_cap_rate` | Decimal percent | `6.10` |

5. Save as `data/ncreif_cap_rates.csv` (overwrite the placeholder)

**Minimum history:** Data back to at least 2000 is recommended (20+ quarters required to fit the VECM).

---

## Quarterly Refresh Workflow

The app will show a **warning banner** when the NCREIF CSV is older than 95 days.

**Each quarter (after NCREIF releases new data, typically 6–8 weeks after quarter end):**
1. Download latest NPI cap rate data from ncreif.org
2. Append the new row(s) to `data/ncreif_cap_rates.csv`
3. Click the **"🔄 Refresh FRED Data"** button in the app sidebar to re-pull macro data
4. The VECM will automatically re-fit with the updated dataset

---

## CSV Format Example

```csv
date,multifamily_cap_rate,office_cap_rate
2000-01-01,7.21,8.45
2000-04-01,7.18,8.41
2000-07-01,7.15,8.38
...
2024-01-01,5.20,6.85
2024-04-01,5.35,7.00
2024-07-01,5.42,7.10
2024-10-01,5.38,7.05
```

**Notes:**
- Dates must be the first day of each quarter: Jan 1, Apr 1, Jul 1, Oct 1
- Cap rates in decimal percent format (e.g., `4.85` = 4.85%, not `0.0485`)
- No gaps in the time series — fill any missing quarters with the prior value or leave blank for forward-fill
- Historical data back to at least 2000 is recommended

---

## Textbook Sensitivity Benchmarks

The sensitivity table on the page replicates Figure 18 from Larriva & Linneman (2022):

| Shock | Multifamily | Office |
|---|---|---|
| +100bp Mortgage Debt / GDP | −22 bps | −65 bps |
| +100bp Unemployment Rate | +1 bps | +3 bps |

Model-estimated values are computed from the 16-quarter cumulative IRF and will differ based on the data vintage used.

---

## Upgrade Path

NCREIF NPI data is **appraisal-based** and typically lags actual transaction prices by 2–4 quarters. For a more timely signal, this model can be upgraded to use:

- **Green Street Advisors** transactional cap rate indices (subscription required)
- **RCA/MSCI** transaction-based data

To upgrade: replace the NCREIF CSV with transactional data using the same column format.

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| "Insufficient data" error | Fewer than 20 overlapping quarters | Extend NCREIF history back to at least 2000 |
| "VECM fitting failed" | Collinear inputs or near-zero variance | Check for duplicate rows or constant columns in CSV |
| Sensitivity shows N/A | IRF computation error (usually edge case in statsmodels) | Typically resolves with more historical data |
| FRED pull fails | API key invalid or FRED is down | Check API key; FRED data will be served from cache for 24 hours |
