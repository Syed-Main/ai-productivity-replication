"""Block 3 – Build the country × sector × year panel.

Key improvements over the original design
------------------------------------------
1. Capital stocks come from PWT 10.01 (``rnna``), extended beyond 2019
   with World Bank GFCF via the perpetual-inventory method.
2. Sectoral capital is allocated by **value-added share** (not employment
   share), so capital-labour ratios vary meaningfully across sectors.
3. A Bartik-style instrument is constructed for AI adoption:
       Z_ct = broadband_{c,2010} × mean(AI_{−c,t})
4. Entity identifiers and log-transformed variables are prepared for
   ``linearmodels`` panel and IV estimators.
"""
import numpy as np
import pandas as pd

import api_oecd
import api_worldbank as wb
import api_pwt
from config import (
    SECTORS, LABOUR_CODES, INVESTMENT_CODE, CONTROL_CODES,
    DELTA, G_INIT, ANALYSIS_START, PWT_END, IV_BASE_YEAR,
    STAGE2_CONTROLS, INCOME_CUTS, MAX_LAGS,
)


# ── helpers ─────────────────────────────────────────────────────────
def _perpetual_inventory(inv, k0=None):
    """K_t = (1 − δ) K_{t−1} + I_t.  If *k0* is given, start there."""
    inv = inv.sort_index().astype(float)
    inv = inv.reindex(range(inv.index.min(), inv.index.max() + 1)).interpolate(limit=2)
    inv = inv.dropna()
    if inv.empty:
        return pd.Series(dtype=float)
    start = k0 if k0 is not None else inv.iloc[0] / (DELTA + G_INIT)
    k = {inv.index[0]: start}
    for y in inv.index[1:]:
        k[y] = (1 - DELTA) * k[y - 1] + inv[y]
    return pd.Series(k)


# ── data-ingestion sub-routines ─────────────────────────────────────
def _output_and_shares(keep):
    va, sh = [], []
    for sec, (va_code, sh_code) in SECTORS.items():
        va.append(
            wb.first_available(va_code, keep)
              .rename(columns={"value": "Y"}).assign(sector=sec)
        )
        sh.append(
            wb.first_available(sh_code, keep)
              .rename(columns={"value": "share"}).assign(sector=sec)
        )
    return pd.concat(va, ignore_index=True), pd.concat(sh, ignore_index=True)


def _total_employment(keep):
    lf = wb.first_available(LABOUR_CODES["lf"], keep).rename(columns={"value": "lf"})
    un = wb.first_available(LABOUR_CODES["un"], keep).rename(columns={"value": "un"})
    e = lf.merge(un, on=["country", "year"])
    e["emp"] = e.lf * (1 - e.un / 100)
    return e[["country", "year", "emp"]]


def _wb_investment(keep):
    return wb.first_available(INVESTMENT_CODE, keep).rename(columns={"value": "I_wb"})


def _controls(keep):
    out = None
    for name, code in CONTROL_CODES.items():
        try:
            d = wb.first_available(code, keep).rename(columns={"value": name})
        except RuntimeError:
            print(f"  ↳ control '{name}' unavailable — skipped")
            continue
        out = d if out is None else out.merge(d, on=["country", "year"], how="outer")
    if out is not None and "ln_gdp_pc" in out.columns:
        out["ln_gdp_pc"] = np.log(out["ln_gdp_pc"].clip(lower=100))
    return out


# ── capital-stock construction ──────────────────────────────────────
def _build_capital(keep):
    """PWT capital through 2019, extended with WB GFCF beyond that."""
    pwt = api_pwt.load_pwt()
    inv = _wb_investment(keep)

    records = []
    for c in pwt.country.unique():
        pk = pwt.loc[pwt.country == c, ["year", "K_pwt"]].set_index("year")["K_pwt"]
        ik = inv.loc[inv.country == c, ["year", "I_wb"]].set_index("year")["I_wb"]
        if pk.empty:
            continue

        # Scale factor: WB investment is in constant 2015 US$,
        # PWT capital is in constant 2017 national prices.
        # We align them by matching the ratio at the last overlapping year.
        overlap = pk.index.intersection(ik.index)
        if len(overlap) < 3:
            # fall back to PWT-only
            for y, v in pk.items():
                records.append((c, int(y), v))
            continue

        last_ol = overlap.max()
        scale = pk[last_ol] / max(ik[last_ol], 1)

        # through PWT_END: use PWT directly
        for y, v in pk.items():
            records.append((c, int(y), v))

        # beyond PWT_END: extend with PI method
        ext_years = sorted(y for y in ik.index if y > pk.index.max())
        prev_k = pk.iloc[-1]
        for y in ext_years:
            prev_k = (1 - DELTA) * prev_k + ik[y] * scale
            records.append((c, int(y), prev_k))

    return pd.DataFrame(records, columns=["country", "year", "K_c"])


# ── Bartik instrument construction ──────────────────────────────────
def _bartik_instrument(p):
    """Z_ct = broadband_{c, IV_BASE_YEAR} × leave-one-out mean AI_{−c,t}."""
    if "broadband" not in p.columns or "ai" not in p.columns:
        return p

    # pre-period broadband (closest available year to IV_BASE_YEAR)
    bb = p.dropna(subset=["broadband"]).copy()
    bb["dist"] = (bb.year - IV_BASE_YEAR).abs()
    bb_base = (
        bb.sort_values("dist")
          .drop_duplicates(subset=["country"], keep="first")
          [["country", "broadband"]]
          .rename(columns={"broadband": "bb_base"})
    )

    # leave-one-out mean AI adoption by year
    ai_yr = p.dropna(subset=["ai"]).copy()
    ai_yr = ai_yr.drop_duplicates(subset=["country", "year"])
    ai_total = ai_yr.groupby("year")["ai"].agg(["sum", "count"])
    ai_yr = ai_yr.merge(ai_total, on="year")
    ai_yr["ai_loo"] = (ai_yr["sum"] - ai_yr["ai"]) / (ai_yr["count"] - 1).clip(lower=1)
    ai_loo = ai_yr[["country", "year", "ai_loo"]].copy()

    p = p.merge(bb_base, on="country", how="left")
    p = p.merge(ai_loo, on=["country", "year"], how="left")
    p["bartik"] = p["bb_base"] * p["ai_loo"]
    return p


# ── income classification ──────────────────────────────────────────
def _income_group(gdp_pc):
    if pd.isna(gdp_pc):
        return np.nan
    for label, thresh in sorted(INCOME_CUTS.items(), key=lambda x: x[1]):
        if gdp_pc <= thresh:
            return label
    return "High"


# ═══════════════════════════════════════════════════════════════════
#                         LIVE  PANEL
# ═══════════════════════════════════════════════════════════════════
def build_live_panel():
    """Download all sources and return the harmonised panel."""
    keep = wb.countries()
    print("Block 3: downloading World Bank series …")
    va, sh = _output_and_shares(keep)

    p = va.merge(sh, on=["country", "year", "sector"])
    p = p.merge(_total_employment(keep), on=["country", "year"])
    p["L"] = p.emp * p.share / 100                     # sectoral employment

    # ── capital: PWT-based, allocated by VA share ───────────────────
    print("Block 3: building capital stocks (PWT + WB GFCF extension) …")
    k = _build_capital(keep)
    p = p.merge(k, on=["country", "year"], how="left")

    # VA-share allocation (not employment share!)
    va_total = p.groupby(["country", "year"])["Y"].transform("sum")
    p["va_share"] = p["Y"] / va_total.clip(lower=1)
    p["K"] = p["K_c"] * p["va_share"]                  # sectoral capital

    # ── OECD AI adoption ────────────────────────────────────────────
    print("Block 3: downloading OECD AI adoption …")
    p = p.merge(api_oecd.ai_adoption(), on=["country", "year"], how="left")

    # ── controls ────────────────────────────────────────────────────
    ctrl = _controls(keep)
    if ctrl is not None:
        p = p.merge(ctrl, on=["country", "year"], how="left")

    # ── Bartik instrument ───────────────────────────────────────────
    p = _bartik_instrument(p)

    # ── income groups ───────────────────────────────────────────────
    if "ln_gdp_pc" in p.columns:
        p["income_group"] = np.exp(p["ln_gdp_pc"]).apply(_income_group)

    return _finalise(p)


# ═══════════════════════════════════════════════════════════════════
#                       SYNTHETIC  PANEL
# ═══════════════════════════════════════════════════════════════════
def build_synthetic_panel(seed=0, n_countries=60,
                          true_alpha=0.40, true_theta=0.10):
    """Simulated panel with known DGP for estimator validation.

    Key properties of the DGP:
    • Capital intensity differs across sectors (Industry > Services > Agri)
    • AI adoption correlates positively with pre-period broadband
    • TFP shocks are partly endogenous to lagged inputs (transmission bias)
    • Bartik instrument is valid (correlated with AI, excluded from TFP)
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2024)
    T = len(years)
    sectors = list(SECTORS.keys())
    sec_k_shift = {"Agriculture": -0.5, "Industry": 0.6, "Services": 0.0}
    rows = []

    for c in range(n_countries):
        # country-level characteristics
        dev   = rng.uniform(7.5, 11.0)        # ln(GDP/cap)
        bb    = max(0, rng.normal(20, 15))     # broadband penetration base
        fe_c  = rng.normal(0, 0.4)
        rd_c  = max(0.1, rng.normal(1.5, 0.7))
        tert  = max(5, rng.normal(40, 15))
        trade = max(10, rng.normal(60, 20))
        inet  = max(2, rng.normal(50, 20))
        gov   = rng.normal(0, 0.8)

        # AI adoption path: higher broadband → faster adoption
        ai_speed = 0.5 + 1.5 * (bb / 40)
        ai_path  = np.clip((years - 2014) * ai_speed + rng.normal(0, 2, T), 0, 100)

        for si, sec in enumerate(sectors):
            lnK = rng.normal(10, 0.8) + sec_k_shift[sec]
            lnL = rng.normal(8, 0.6)
            lnA = 0.0

            for i, y in enumerate(years):
                # inputs grow with partial endogeneity to productivity
                eps_k = rng.normal(0.04, 0.015)
                eps_l = rng.normal(0.01, 0.012)
                lnK += eps_k + 0.1 * lnA   # transmission bias
                lnL += eps_l + 0.05 * lnA  # transmission bias

                # TFP evolves with lagged AI effect
                ai_lag = ai_path[i - 1] if i > 0 else 0
                lnA += (0.01
                        + true_theta / 100 * ai_lag
                        + rng.normal(0, 0.018))

                lnY = (lnA
                       + true_alpha * lnK
                       + (1 - true_alpha) * lnL
                       + fe_c + 0.2 * si)

                rows.append((
                    f"C{c:02d}", sec, y,
                    np.exp(lnY), np.exp(lnK), np.exp(lnL),
                    ai_path[i] if y >= 2015 else np.nan,
                    dev + 0.02 * i + rng.normal(0, 0.05),  # ln_gdp_pc
                    tert + rng.normal(0, 3),
                    rd_c + rng.normal(0, 0.1),
                    trade + rng.normal(0, 3),
                    inet + 1.5 * i + rng.normal(0, 2),
                    bb + 0.8 * i + rng.normal(0, 1),        # broadband grows
                    gov + rng.normal(0, 0.05),
                ))

    cols = (["country", "sector", "year", "Y", "K", "L", "ai"]
            + list(CONTROL_CODES))
    p = pd.DataFrame(rows, columns=cols)

    # Build Bartik instrument in synthetic data
    p = _bartik_instrument(p)

    # Income group
    p["income_group"] = np.exp(p["ln_gdp_pc"]).apply(_income_group)

    return _finalise(p)


# ── shared finalisation ─────────────────────────────────────────────
def _finalise(p):
    """Add logs, entity id, growth rates, and lagged AI variables."""
    # drop rows with non-positive Y, K, L or before analysis start
    p = p[(p.Y > 0) & (p.K > 0) & (p.L > 0) & (p.year >= ANALYSIS_START)].copy()

    # log levels
    for v in ("Y", "K", "L"):
        p[f"ln{v}"] = np.log(p[v])

    # per-worker form for CRS restriction
    p["lnY_L"] = p.lnY - p.lnL
    p["lnK_L"] = p.lnK - p.lnL

    # entity identifier for panel estimators
    p["eid"] = p.country + "_" + p.sector

    # growth rates (log-differences × 100 → percentage points)
    p = p.sort_values(["country", "sector", "year"])
    for v in ("Y", "K", "L"):
        p[f"g_{v}"] = 100 * p.groupby("eid")[f"ln{v}"].diff()

    # lagged AI (multiple horizons for distributed-lag model)
    for lag in range(1, MAX_LAGS + 1):
        p[f"ai_lag{lag}"] = p.groupby("eid")["ai"].shift(lag)

    print(f"Panel finalised: {len(p):,} rows, "
          f"{p.country.nunique()} countries, "
          f"{p.sector.nunique()} sectors, "
          f"years {p.year.min()}–{p.year.max()}")

    return p
