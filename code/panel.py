"""Block 3 – Build the aggregate country × year and sectoral panels.

Key improvements:
-----------------
1. Aggregate Country-Year Panel (Primary):
   - Capital stocks from PWT 10.01 (``rnna``), extended beyond 2019 by applying
     the annual real growth rate of World Bank constant-price GFCF to the 2019 PWT
     real investment series (I_2019 = csh_i * rgdpna), accumulated via perpetual
     inventory (K_t = (1 - δ) K_{t-1} + I_t).
   - Törnqvist TFP growth calculated using time-varying labor shares (labsh).
2. Sectoral Country × Sector × Year Panel (Mechanism):
   - National AI adoption interacted with sectoral task exposure:
     AI_{cst} = AI_{ct} × Exposure_s (Services=1.0, Industry=0.6, Agriculture=0.15).
   - Accommodates country-by-year and sector-by-year fixed effects (Rajan-Zingales style).
3. Overidentified Bartik-style instruments:
   - Z1 = broadband_{c,2010} × mean(AI_{-c,t})
   - Z2 = internet_{c,2010} × mean(AI_{-c,t})
"""
import numpy as np
import pandas as pd

import api_oecd
import api_worldbank as wb
import api_pwt
from config import (
    SECTORS, LABOUR_CODES, INVESTMENT_CODE, CONTROL_CODES,
    DELTA, G_INIT, ANALYSIS_START, PWT_END, IV_BASE_YEAR,
    STAGE2_CONTROLS, INCOME_CUTS, MAX_LAGS, SECTOR_AI_EXPOSURE,
    WINSOR_PCTILES,
)
from utils import winsorise


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


def _wb_gdp(keep):
    return wb.first_available("NY.GDP.MKTP.KD", keep).rename(columns={"value": "Y_wb"})


def _controls(keep):
    out = None
    for name, code in CONTROL_CODES.items():
        try:
            d = wb.first_available(code, keep).rename(columns={"value": name})
        except RuntimeError:
            print(f"  -> control '{name}' unavailable — skipped")
            continue
        out = d if out is None else out.merge(d, on=["country", "year"], how="outer")
    if out is not None and "ln_gdp_pc" in out.columns:
        out["ln_gdp_pc"] = np.log(out["ln_gdp_pc"].clip(lower=100))
    return out


# ── capital-stock and macro extension ───────────────────────────────
def _build_capital_and_macro(keep):
    """PWT capital through 2019, extended with WB GFCF real growth rates.

    To extend PWT rnna past 2019:
      1. Baseline PWT real investment in 2019: I_{c,2019} = csh_i * rgdpna.
      2. For t > 2019, apply real GFCF growth: I_{c,t} = I_{c,t-1} * (I_{c,t}^{WB} / I_{c,t-1}^{WB}).
      3. Capital accumulation: K_{c,t} = (1 - δ_c) K_{c,t-1} + I_{c,t}.
      4. Similarly extend real GDP (rgdpna) and employment (emp) using WB real growth.
    """
    pwt = api_pwt.load_pwt()
    inv_wb = _wb_investment(keep)
    gdp_wb = _wb_gdp(keep)
    emp_wb = wb.first_available(LABOUR_CODES["lf"], keep).rename(columns={"value": "L_wb"})

    records = []
    countries = pwt.country.unique()

    for c in countries:
        sub_pwt = pwt[pwt.country == c].sort_values("year").set_index("year")
        sub_iwb = inv_wb[inv_wb.country == c].sort_values("year").set_index("year")["I_wb"]
        sub_ywb = gdp_wb[gdp_wb.country == c].sort_values("year").set_index("year")["Y_wb"]
        sub_lwb = emp_wb[emp_wb.country == c].sort_values("year").set_index("year")["L_wb"]

        if sub_pwt.empty or "K_pwt" not in sub_pwt.columns:
            continue

        pwt_max_yr = sub_pwt.index.max()
        d_c = sub_pwt["delta"].dropna().iloc[-1] if ("delta" in sub_pwt.columns and sub_pwt["delta"].notna().any()) else DELTA

        # 2019 real investment in PWT constant national prices
        if pwt_max_yr in sub_pwt.index and "csh_i" in sub_pwt.columns and "Y_pwt" in sub_pwt.columns:
            csh_i_val = sub_pwt.loc[pwt_max_yr, "csh_i"]
            y_val = sub_pwt.loc[pwt_max_yr, "Y_pwt"]
            i_pwt_prev = (csh_i_val * y_val) if (pd.notna(csh_i_val) and csh_i_val > 0) else (d_c * sub_pwt.loc[pwt_max_yr, "K_pwt"])
        else:
            i_pwt_prev = d_c * sub_pwt.loc[pwt_max_yr, "K_pwt"]

        k_pwt_prev = sub_pwt.loc[pwt_max_yr, "K_pwt"]
        y_pwt_prev = sub_pwt.loc[pwt_max_yr, "Y_pwt"] if "Y_pwt" in sub_pwt.columns else np.nan
        l_pwt_prev = sub_pwt.loc[pwt_max_yr, "L_pwt"] if "L_pwt" in sub_pwt.columns else np.nan
        labsh_prev = sub_pwt["labsh"].dropna().iloc[-1] if ("labsh" in sub_pwt.columns and sub_pwt["labsh"].notna().any()) else 0.60

        # PWT historical (through PWT_END)
        for y, row in sub_pwt.iterrows():
            records.append({
                "country": c,
                "year": int(y),
                "Y_c": row.get("Y_pwt", np.nan),
                "K_c": row.get("K_pwt", np.nan),
                "L_c": row.get("L_pwt", np.nan),
                "labsh_c": row.get("labsh", np.nan),
                "delta_c": row.get("delta", d_c),
                "source": "PWT"
            })

        # Forward extension beyond PWT_END (2020..2024)
        ext_years = sorted(y for y in sub_iwb.index if y > pwt_max_yr)
        for y in ext_years:
            if (y - 1) in sub_iwb.index and sub_iwb[y - 1] > 0 and y in sub_iwb.index:
                growth_i = sub_iwb[y] / sub_iwb[y - 1]
            else:
                growth_i = 1.02

            i_pwt_curr = i_pwt_prev * growth_i
            k_pwt_curr = (1 - d_c) * k_pwt_prev + i_pwt_curr

            if (y - 1) in sub_ywb.index and sub_ywb[y - 1] > 0 and y in sub_ywb.index:
                growth_y = sub_ywb[y] / sub_ywb[y - 1]
            else:
                growth_y = 1.02
            y_pwt_curr = y_pwt_prev * growth_y

            if (y - 1) in sub_lwb.index and sub_lwb[y - 1] > 0 and y in sub_lwb.index:
                growth_l = sub_lwb[y] / sub_lwb[y - 1]
            else:
                growth_l = 1.01
            l_pwt_curr = l_pwt_prev * growth_l

            records.append({
                "country": c,
                "year": int(y),
                "Y_c": y_pwt_curr,
                "K_c": k_pwt_curr,
                "L_c": l_pwt_curr,
                "labsh_c": labsh_prev,
                "delta_c": d_c,
                "source": "Extended"
            })

            i_pwt_prev = i_pwt_curr
            k_pwt_prev = k_pwt_curr
            y_pwt_prev = y_pwt_curr
            l_pwt_prev = l_pwt_curr

    return pd.DataFrame(records)


# ── Overidentified Bartik instruments ───────────────────────────────
def _bartik_instruments(p, id_col="country"):
    """Construct overidentified instruments Z1 and Z2:
       Z1_ct = broadband_{c, 2010} × leave-one-out mean AI_{-c,t}
       Z2_ct = internet_{c, 2010}   × leave-one-out mean AI_{-c,t}
    """
    if "ai" not in p.columns:
        return p

    # Pre-period broadband 2010 base
    if "broadband" in p.columns:
        bb = p.dropna(subset=["broadband"]).copy()
        bb["dist"] = (bb.year - IV_BASE_YEAR).abs()
        bb_base = (
            bb.sort_values("dist")
              .drop_duplicates(subset=["country"], keep="first")
              [["country", "broadband"]]
              .rename(columns={"broadband": "bb_base"})
        )
        p = p.merge(bb_base, on="country", how="left")
    else:
        p["bb_base"] = np.nan

    # Pre-period internet 2010 base
    if "internet" in p.columns:
        net = p.dropna(subset=["internet"]).copy()
        net["dist"] = (net.year - IV_BASE_YEAR).abs()
        net_base = (
            net.sort_values("dist")
               .drop_duplicates(subset=["country"], keep="first")
               [["country", "internet"]]
               .rename(columns={"internet": "net_base"})
        )
        p = p.merge(net_base, on="country", how="left")
    else:
        p["net_base"] = np.nan

    # Leave-one-out mean AI adoption by year
    ai_yr = p.dropna(subset=["ai"]).drop_duplicates(subset=["country", "year"]).copy()
    if not ai_yr.empty:
        ai_total = ai_yr.groupby("year")["ai"].agg(["sum", "count"])
        ai_yr = ai_yr.merge(ai_total, on="year")
        ai_yr["ai_loo"] = (ai_yr["sum"] - ai_yr["ai"]) / (ai_yr["count"] - 1).clip(lower=1)
        ai_loo = ai_yr[["country", "year", "ai_loo"]].copy()
        p = p.merge(ai_loo, on=["country", "year"], how="left")
    else:
        p["ai_loo"] = np.nan

    p["bartik1"] = p["bb_base"] * p["ai_loo"]
    p["bartik2"] = p["net_base"] * p["ai_loo"]

    # In sectoral panels, interact with sectoral exposure
    if "sector" in p.columns and "ai_exposure" in p.columns:
        p["bartik1_sec"] = p["bartik1"] * p["ai_exposure"]
        p["bartik2_sec"] = p["bartik2"] * p["ai_exposure"]

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
#                    PRIMARY: AGGREGATE PANEL
# ═══════════════════════════════════════════════════════════════════
def build_aggregate_panel():
    """Build the country × year panel for primary macro estimation.

    Directly uses PWT capital, GDP, employment, and labor shares,
    extended past 2019 via World Bank growth rates.
    Computes Solow-Jorgenson Törnqvist TFP residuals using time-varying
    labor shares.
    """
    keep = wb.countries()
    print("Block 3: building aggregate country-year panel (PWT + WB GFCF extension) …")
    macro = _build_capital_and_macro(keep)

    agg = macro.rename(columns={"Y_c": "Y", "K_c": "K", "L_c": "L", "labsh_c": "labsh"}).copy()
    agg = agg[(agg.year >= ANALYSIS_START) & (agg.Y > 0) & (agg.K > 0) & (agg.L > 0)].copy()

    # Controls
    ctrl = _controls(keep)
    if ctrl is not None:
        agg = agg.merge(ctrl, on=["country", "year"], how="left")

    # OECD AI adoption
    print("Block 3: merging OECD AI adoption …")
    agg = agg.merge(api_oecd.ai_adoption(), on=["country", "year"], how="left")

    # Solow-Jorgenson Törnqvist growth accounting:
    # time-varying capital elasticity: alpha_ct = 1 - labsh_ct
    agg = agg.sort_values(["country", "year"])
    agg["alpha"] = 1.0 - agg["labsh"].clip(0.10, 0.90)
    agg["alpha_bar"] = 0.5 * (agg["alpha"] + agg.groupby("country")["alpha"].shift(1))
    agg["beta_bar"] = 1.0 - agg["alpha_bar"]

    for v in ["Y", "K", "L"]:
        agg[f"ln{v}"] = np.log(agg[v])
        agg[f"g_{v}"] = 100 * agg.groupby("country")[f"ln{v}"].diff()

    agg["C_K"] = agg["alpha_bar"] * agg["g_K"]
    agg["C_L"] = agg["beta_bar"] * agg["g_L"]
    agg["C_TFP"] = agg["g_Y"] - agg["C_K"] - agg["C_L"]
    agg["dlnA"] = winsorise(agg["C_TFP"], WINSOR_PCTILES[0], WINSOR_PCTILES[1])

    # Per-worker forms
    agg["lnY_L"] = agg["lnY"] - agg["lnL"]
    agg["lnK_L"] = agg["lnK"] - agg["lnL"]

    # AI lags
    for lag in range(1, MAX_LAGS + 1):
        agg[f"ai_lag{lag}"] = agg.groupby("country")["ai"].shift(lag)

    # Overidentified Bartik instruments
    agg = _bartik_instruments(agg, id_col="country")

    # Income groups
    if "ln_gdp_pc" in agg.columns:
        agg["income_group"] = np.exp(agg["ln_gdp_pc"]).apply(_income_group)

    print(f"Aggregate panel finalised: {len(agg):,} rows, "
          f"{agg.country.nunique()} countries, "
          f"years {agg.year.min()}–{agg.year.max()}")
    return agg


# ═══════════════════════════════════════════════════════════════════
#                    SECONDARY: SECTORAL PANEL
# ═══════════════════════════════════════════════════════════════════
def build_live_panel():
    """Build the country × sector × year panel for mechanism analysis."""
    keep = wb.countries()
    print("Block 3: building sectoral panel (World Bank + PWT capital allocation) …")
    va, sh = _output_and_shares(keep)

    p = va.merge(sh, on=["country", "year", "sector"])
    p = p.merge(_total_employment(keep), on=["country", "year"])
    p["L"] = p.emp * p.share / 100                     # sectoral employment

    # Capital stocks (PWT-based, extended with WB real GFCF growth)
    macro = _build_capital_and_macro(keep)
    p = p.merge(macro[["country", "year", "K_c", "labsh_c"]], on=["country", "year"], how="left")

    # Value-added share allocation for sectoral capital
    va_total = p.groupby(["country", "year"])["Y"].transform("sum")
    p["va_share"] = p["Y"] / va_total.clip(lower=1)
    p["K"] = p["K_c"] * p["va_share"]

    # Sectoral AI exposure and interaction
    p["ai_exposure"] = p["sector"].map(SECTOR_AI_EXPOSURE)

    # OECD AI adoption
    p = p.merge(api_oecd.ai_adoption(), on=["country", "year"], how="left")
    p["ai_interact"] = p["ai"] * p["ai_exposure"]

    # Controls
    ctrl = _controls(keep)
    if ctrl is not None:
        p = p.merge(ctrl, on=["country", "year"], how="left")

    # Bartik instruments
    p = _bartik_instruments(p, id_col="country")

    # Income groups
    if "ln_gdp_pc" in p.columns:
        p["income_group"] = np.exp(p["ln_gdp_pc"]).apply(_income_group)

    # Finalise logs, growth rates, and identifiers
    p = p[(p.Y > 0) & (p.K > 0) & (p.L > 0) & (p.year >= ANALYSIS_START)].copy()

    for v in ("Y", "K", "L"):
        p[f"ln{v}"] = np.log(p[v])
    p["lnY_L"] = p.lnY - p.lnL
    p["lnK_L"] = p.lnK - p.lnL
    p["eid"] = p.country + "_" + p.sector
    p["cy"] = p.country + "_" + p.year.astype(str)
    p["sy"] = p.sector + "_" + p.year.astype(str)

    p = p.sort_values(["country", "sector", "year"])
    for v in ("Y", "K", "L"):
        p[f"g_{v}"] = 100 * p.groupby("eid")[f"ln{v}"].diff()

    # Sectoral TFP growth using time-varying labor shares
    p["alpha_c"] = 1.0 - p["labsh_c"].fillna(0.60).clip(0.10, 0.90)
    p["C_K"] = p["alpha_c"] * p["g_K"]
    p["C_L"] = (1.0 - p["alpha_c"]) * p["g_L"]
    p["C_TFP"] = p["g_Y"] - p["C_K"] - p["C_L"]
    p["dlnA"] = winsorise(p["C_TFP"], WINSOR_PCTILES[0], WINSOR_PCTILES[1])

    for lag in range(1, MAX_LAGS + 1):
        p[f"ai_lag{lag}"] = p.groupby("eid")["ai"].shift(lag)
        p[f"ai_int_lag{lag}"] = p.groupby("eid")["ai_interact"].shift(lag)

    print(f"Sectoral panel finalised: {len(p):,} rows, "
          f"{p.country.nunique()} countries, "
          f"{p.sector.nunique()} sectors, "
          f"years {p.year.min()}–{p.year.max()}")
    return p


# ═══════════════════════════════════════════════════════════════════
#                       SYNTHETIC  PANEL
# ═══════════════════════════════════════════════════════════════════
def build_synthetic_panel(seed=0, n_countries=60,
                          true_alpha=0.35, true_theta=0.08):
    """Simulated panel with known DGP for pipeline validation."""
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2024)
    T = len(years)
    sectors = list(SECTORS.keys())
    sec_k_shift = {"Agriculture": -0.5, "Industry": 0.6, "Services": 0.0}
    rows = []

    for c in range(n_countries):
        dev   = rng.uniform(7.5, 11.0)
        bb    = max(0, rng.normal(20, 15))
        inet  = max(2, rng.normal(50, 20))
        fe_c  = rng.normal(0, 0.4)
        rd_c  = max(0.1, rng.normal(1.5, 0.7))
        tert  = max(5, rng.normal(40, 15))
        trade = max(10, rng.normal(60, 20))

        ai_speed = 0.5 + 1.5 * (bb / 40)
        ai_path  = np.clip((years - 2014) * ai_speed + rng.normal(0, 2, T), 0, 100)

        for si, sec in enumerate(sectors):
            exposure = SECTOR_AI_EXPOSURE[sec]
            lnK = rng.normal(10, 0.8) + sec_k_shift[sec]
            lnL = rng.normal(8, 0.6)
            lnA = 0.0

            for i, y in enumerate(years):
                eps_k = rng.normal(0.04, 0.015)
                eps_l = rng.normal(0.01, 0.012)
                lnK += eps_k + 0.1 * lnA
                lnL += eps_l + 0.05 * lnA

                ai_lag = ai_path[i - 1] if i > 0 else 0
                lnA += (0.01
                        + true_theta / 100 * (ai_lag * exposure)
                        + rng.normal(0, 0.018))

                lnY = (lnA
                       + true_alpha * lnK
                       + (1 - true_alpha) * lnL
                       + fe_c + 0.2 * si)

                rows.append((
                    f"C{c:02d}", sec, y,
                    np.exp(lnY), np.exp(lnK), np.exp(lnL),
                    ai_path[i] if y >= 2015 else np.nan,
                    dev + 0.02 * i + rng.normal(0, 0.05),
                    tert + rng.normal(0, 3),
                    rd_c + rng.normal(0, 0.1),
                    trade + rng.normal(0, 3),
                    inet + 1.5 * i + rng.normal(0, 2),
                    bb + 0.8 * i + rng.normal(0, 1),
                    0.60,
                ))

    cols = ["country", "sector", "year", "Y", "K", "L", "ai",
            "ln_gdp_pc", "tertiary", "rd", "trade", "internet", "broadband", "labsh_c"]
    p = pd.DataFrame(rows, columns=cols)

    p["ai_exposure"] = p["sector"].map(SECTOR_AI_EXPOSURE)
    p["ai_interact"] = p["ai"] * p["ai_exposure"]
    p = _bartik_instruments(p, id_col="country")
    p["income_group"] = np.exp(p["ln_gdp_pc"]).apply(_income_group)

    p = p[(p.Y > 0) & (p.K > 0) & (p.L > 0) & (p.year >= ANALYSIS_START)].copy()
    for v in ("Y", "K", "L"):
        p[f"ln{v}"] = np.log(p[v])
    p["lnY_L"] = p.lnY - p.lnL
    p["lnK_L"] = p.lnK - p.lnL
    p["eid"] = p.country + "_" + p.sector
    p["cy"] = p.country + "_" + p.year.astype(str)
    p["sy"] = p.sector + "_" + p.year.astype(str)

    p = p.sort_values(["country", "sector", "year"])
    for v in ("Y", "K", "L"):
        p[f"g_{v}"] = 100 * p.groupby("eid")[f"ln{v}"].diff()

    p["alpha_c"] = 1.0 - p["labsh_c"]
    p["C_K"] = p["alpha_c"] * p["g_K"]
    p["C_L"] = (1.0 - p["alpha_c"]) * p["g_L"]
    p["C_TFP"] = p["g_Y"] - p["C_K"] - p["C_L"]
    p["dlnA"] = winsorise(p["C_TFP"], WINSOR_PCTILES[0], WINSOR_PCTILES[1])

    for lag in range(1, MAX_LAGS + 1):
        p[f"ai_lag{lag}"] = p.groupby("eid")["ai"].shift(lag)
        p[f"ai_int_lag{lag}"] = p.groupby("eid")["ai_interact"].shift(lag)

    return p
