# CLAUDE.md — Project Guide for AI Agents

## Project Overview
Streamlit-based Market Analysis Dashboard deployed on **Streamlit Cloud**.
4 tabs: Day-of-Week, Gamma Exposure (GEX), VIX/Volatility, Seasonality.

## Tech Stack
- **Frontend:** Streamlit + Plotly (dark theme)
- **Data:** Yahoo Finance via `yfinance` (primary), Massive.com API (built but unused — no key)
- **Cache:** Two-layer — local parquet files (Layer 1) → Supabase PostgreSQL (Layer 2, durable)
- **Deployment:** Streamlit Cloud, Python 3.12, `requirements.txt`

## Critical Rules

### 1. `from __future__ import annotations` is MANDATORY
Every module file that uses union type hints (`pd.DataFrame | None`, `float | None`, etc.)
**must** have `from __future__ import annotations` as the first real import.
Without it, Streamlit Cloud crashes with `TypeError` because Python evaluates type hints
at runtime and `pd.DataFrame.__or__` may not be defined.

**Files that need it:** `gamma_exposure.py`, `data_fetcher.py`, `supabase_cache.py`

### 2. Import Smoke Tests Must Mirror app.py Exactly
`tests/test_imports.py` must import **every name** that `app.py` imports.
The CI smoke test in `.github/workflows/ci.yml` must also stay in sync.
When you add a new public function to a module and import it in `app.py`,
you **must** also add it to `test_imports.py` and `ci.yml`.

### 3. Streamlit Cloud Stale Deployments
Streamlit Cloud can serve cached/stale module versions even after a push.
If an import error persists despite the code being correct in git:
- Force a redeploy by making any trivial change (comment) and pushing again
- Use a diagnostic try/except wrapper in `app.py` to reveal the actual error,
  since Streamlit Cloud often redacts tracebacks

### 4. Never Commit Secrets
- `.streamlit/secrets.toml` is in `.gitignore`
- Template: `.streamlit/secrets.toml.example`
- Supabase ops are all silent-on-failure (app works without it)

## Architecture Decisions

### Options Data & Greeks
- Yahoo Finance provides: strike, OI, volume, IV, bid/ask, expiration (no Greeks)
- **Gamma and delta are computed via Black-Scholes** in `gamma_exposure.py`
- `_bs_gamma()` and `_bs_delta()` use IV + time-to-expiry to compute Greeks
- `_add_computed_gamma()` and `_add_computed_delta()` add Greeks to DataFrames

### GEX Sign Convention (Dealer Perspective)
- **Call GEX = positive** (dealers long gamma on calls)
- **Put GEX = negative** (dealers short gamma on puts — sign flipped)
- Formula: `GEX = Gamma × OI × 100 × Spot² × 0.01`, put GEX negated

### DEX Sign Convention (Dealer Perspective)
- **Call DEX = negative** (dealers short delta to hedge sold calls → must sell)
- **Put DEX = positive** (dealers long delta to hedge sold puts → must buy)
- Formula: `-delta × OI × 100` for calls, `-put_delta × OI × 100` for puts

### Chart Strike Range
- Default ±10% around spot, but **auto-expands** to include call wall and put wall
- Key levels outside the range get edge arrows with distance labels on the price chart
- By-expiration charts limited to next 6 months (far-dated options have negligible gamma)

### Historical Gamma Proxy
- Free historical options chains are unavailable
- Uses VIX/SPY-derived proxy: `proxy = -VIX_z_score × (1 + (VIX - realized_vol) / 100)`
- Calibrated against real GEX data when available via `calibrate_proxy_to_real()`

### Caching Strategy
- **Layer 1 (local file):** `cache/{ticker}_{type}_{date}.parquet` — survives app restarts
- **Layer 2 (Supabase):** Durable across redeploys — 3 tables: `gamma_index_history`, `options_cache`, `price_cache`
- `@st.cache_data(ttl=86400)` as in-memory fallback (24h TTL)

## File Map
| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit app — 4 tabs, sidebar, all UI |
| `modules/data_fetcher.py` | Yahoo Finance data fetching + file cache |
| `modules/gamma_exposure.py` | GEX, DEX, IV skew computation + all Plotly charts |
| `modules/supabase_cache.py` | Supabase persistent cache (silent on failure) |
| `modules/day_of_week.py` | Day-of-week return analysis |
| `modules/vix_analysis.py` | VIX metrics and charts |
| `modules/seasonality.py` | Monthly/weekly/intramonth seasonality |
| `tests/test_imports.py` | **HIGHEST PRIORITY** — mirrors app.py imports exactly |
| `tests/conftest.py` | Shared fixtures — synthetic OHLCV + options data |
| `.github/workflows/ci.yml` | CI pipeline — smoke test + full test suite |

## Testing
- **167+ tests**, all run offline (yfinance fully mocked)
- Pre-commit hook runs full suite before every commit
- `test_imports.py` = deployment safety net (catches missing exports)
- `test_requirements.py` = validates all PyPI packages exist
- Fixtures use deterministic seeds for reproducibility

## Common Gotchas
1. **Sparse GEX after ~6 months out** — Normal gamma physics (far-dated options have tiny gamma), not API throttling
2. **Yahoo provides 8,400+ contracts for SPY** across 28 expirations — data is comprehensive
3. **Massive.com code is dead code** — ~130 lines fully built but no API key configured; Yahoo suffices
4. **Options fixture uses past expiration (2024-03-15)** — TTE=0 means B-S delta returns 0; tests that need delta must supply explicit values
