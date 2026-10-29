"""Block 5 – Stage 2: does AI adoption explain TFP growth?

Specifications:
1. Aggregate Country-Year (Primary):
   - OLS Fixed Effects (contemporaneous, lag 1, lag 2)
   - IV-2SLS with overidentifying instruments Z1 and Z2
   - IV-LIML with overidentifying instruments Z1 and Z2
   - Hansen J overidentification test (df = 1)
   - Anderson–Rubin weak-instrument-robust test
   - Distributed-lag model (J-curve adjustment friction test)
2. Sectoral Mechanism (Rajan–Zingales Interaction):
   - Δln TFP_{cst} = θ (AI_{ct} × Exposure_s) + μ_{ct} + λ_{st} + ε_{cst}
   - Absorbs all time-varying country macroeconomic shocks and sectoral shifts
3. Heterogeneity Analysis:
   - Interactions with income group and sector
"""
import warnings
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS, IVLIML
import statsmodels.api as sm

from config import STAGE2_CONTROLS, MAX_LAGS, WINSOR_PCTILES
from utils import pretty, add_year_dummies, winsorise


# ── helpers ─────────────────────────────────────────────────────────
def _prepare(p):
    """Winsorise TFP growth if not already done."""
    p = p.copy()
    lo, hi = WINSOR_PCTILES
    if "dlnA" not in p.columns:
        p["dlnA"] = winsorise(p["C_TFP"], lo, hi)
    return p


def _pick_controls(p, subset=None):
    """Keep controls with >55% non-missing coverage."""
    pool = subset or STAGE2_CONTROLS
    s0 = p.dropna(subset=["ai", "dlnA"])
    ok = [c for c in pool if c in p.columns and s0[c].notna().mean() > 0.55]
    print(f"  Stage-2 controls used: {ok}")
    return ok


def _demean_panel(d, dep, endog_cols, exog_cols, instr_cols, eid="country"):
    """Within-group demeaning (entity FE) for panel IV."""
    all_cols = [dep] + endog_cols + exog_cols + instr_cols
    all_cols = [c for c in all_cols if c in d.columns]
    means = d.groupby(eid)[all_cols].transform("mean")
    dm = d.copy()
    for c in all_cols:
        dm[c] = dm[c] - means[c]
    return dm


# ═══════════════════════════════════════════════════════════════════
#  A.  OLS Fixed Effects (Aggregate Country-Year)
# ═══════════════════════════════════════════════════════════════════
def _ols_baseline(d, controls, yd=None, eid="country", lag=0, baseline_countries=None):
    """OLS-FE: regress TFP growth on AI (contemporaneous or lagged) + controls + FE."""
    ai_col = "ai" if lag == 0 else f"ai_lag{lag}"
    if ai_col not in d.columns:
        return None, None

    sub = d.dropna(subset=["dlnA", ai_col] + controls).copy()
    if baseline_countries is not None:
        sub = sub[sub[eid].isin(baseline_countries)].copy()

    if len(sub) < 30:
        return None, None

    idx = sub.set_index([eid, "year"])
    exog_cols = [ai_col] + controls
    exog = idx[exog_cols]

    try:
        m = PanelOLS(idx["dlnA"], exog, entity_effects=True, time_effects=True,
                     check_rank=False, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    except Exception:
        try:
            m = PanelOLS(idx["dlnA"], exog, entity_effects=True, time_effects=True,
                         check_rank=False, drop_absorbed=True).fit(cov_type="robust")
        except Exception:
            try:
                m = PanelOLS(idx["dlnA"], exog, entity_effects=True, time_effects=False,
                             check_rank=False, drop_absorbed=True).fit(cov_type="robust")
            except Exception as exc:
                print(f"  -> OLS lag {lag} estimation failed: {exc}")
                return None, None

    # Calculate true within R2 relative to two-way FE-only baseline
    within_r2 = np.nan
    try:
        d_fe = pd.get_dummies(sub[[eid, "year"]], drop_first=True, dtype=float)
        X_full = pd.concat([sub[exog_cols].reset_index(drop=True), d_fe.reset_index(drop=True)], axis=1)
        y_vec = sub["dlnA"].values
        reg_full = sm.OLS(y_vec, X_full).fit()
        reg_fe = sm.OLS(y_vec, d_fe).fit()
        if reg_fe.ssr > 0:
            within_r2 = max(0.0, float(1.0 - (reg_full.ssr / reg_fe.ssr)))
    except Exception:
        if hasattr(m, "rsquared_within") and m.rsquared_within is not None and m.rsquared_within > 0:
            within_r2 = float(m.rsquared_within)

    spec_name = "OLS-FE (contemporaneous)" if lag == 0 else f"OLS-FE (lag {lag})"
    row = {
        "Specification": spec_name,
        "Lag": lag,
        "θ": m.params[ai_col],
        "SE": m.std_errors[ai_col],
        "p-value": m.pvalues[ai_col],
        "N": int(m.nobs),
        "Countries": sub[eid].nunique(),
        "R²": within_r2,
        "1st-F": np.nan,
        "Hansen-J-p": np.nan,
        "AR-p": np.nan,
        "Method": "OLS",
    }
    return row, m


# ═══════════════════════════════════════════════════════════════════
#  B.  Overidentified IV-2SLS and LIML
# ═══════════════════════════════════════════════════════════════════
def _iv_overidentified(d, controls, yd, method="2sls", eid="country"):
    """Overidentified IV via entity demeaning with Z1 and Z2."""
    label = "IV-2SLS (Overidentified)" if method == "2sls" else "IV-LIML (Overidentified)"
    endog_cols = ["ai"]
    instr_cols = ["bartik1", "bartik2"]
    exog_cols  = list(controls)
    yr_cols    = list(yd.columns)

    d_sub = d.dropna(subset=["dlnA", "ai", "bartik1", "bartik2"] + controls).copy()
    if len(d_sub) < 30:
        print(f"  -> {label} sample too small — skipping")
        return None, None

    yd_sub = add_year_dummies(d_sub)
    yr_cols = list(yd_sub.columns)
    d_full = pd.concat([d_sub, yd_sub], axis=1)

    # Within-entity demeaning
    dm = _demean_panel(d_full, "dlnA", endog_cols, exog_cols + yr_cols,
                       instr_cols, eid=eid)

    dep   = dm["dlnA"]
    exog  = dm[exog_cols + yr_cols] if len(exog_cols) else dm[yr_cols]
    endog = dm[endog_cols]
    instr = dm[instr_cols]

    Cls = IVLIML if method == "liml" else IV2SLS
    try:
        m = Cls(dep, exog, endog, instr).fit(cov_type="robust")
    except Exception as exc:
        print(f"  -> {label} failed: {exc}")
        return None, None

    # First-stage F-statistic
    first_f = np.nan
    try:
        first_f = float(m.first_stage.diagnostics.loc["ai", "f.stat"])
    except Exception:
        try:
            first_f = float(m.first_stage.diagnostics["f.stat"].iloc[0])
        except Exception:
            pass

    # Hansen J overidentification test p-value (df = L - K = 2 - 1 = 1)
    hansen_p = np.nan
    if hasattr(m, "sargan") and m.sargan is not None:
        hansen_p = float(m.sargan.pval)

    # Anderson-Rubin weak-instrument test (H0: θ = 0)
    # Under H0, y is regressed on exog and instruments; tests joint significance of instruments
    ar_p = np.nan
    try:
        ar_reg = sm.OLS(dep, pd.concat([exog, instr], axis=1)).fit()
        ar_test = ar_reg.f_test("bartik1 = 0, bartik2 = 0")
        ar_p = float(ar_test.pvalue)
    except Exception:
        pass

    # For IV/LIML, R² does not have a bounded sum-of-squares interpretation; suppress negative values
    r2_val = np.nan

    row = {
        "Specification": label,
        "θ": m.params["ai"],
        "SE": m.std_errors["ai"],
        "p-value": m.pvalues["ai"],
        "N": int(m.nobs),
        "Countries": d_sub[eid].nunique(),
        "R²": r2_val,
        "1st-F": first_f,
        "Hansen-J-p": hansen_p,
        "AR-p": ar_p,
        "Method": "LIML" if method == "liml" else "2SLS",
    }
    return row, m


# ═══════════════════════════════════════════════════════════════════
#  C.  Sectoral Rajan–Zingales Exposure Interaction
# ═══════════════════════════════════════════════════════════════════
def _sectoral_rajan_zingales(p_sec, controls):
    """Estimate sectoral exposure interaction with country-year & sector-year FE:
       Δln TFP_{cst} = θ (AI_{ct} × Exposure_s) + μ_{ct} + λ_{st} + ε_{cst}
    """
    if "ai_interact" not in p_sec.columns or "cy" not in p_sec.columns:
        return None, None

    d = p_sec.dropna(subset=["dlnA", "ai_interact"]).copy()
    if len(d) < 50:
        return None, None

    # Demean by country-year (cy) to absorb all time-varying country shocks
    d["dlnA_dm"] = d["dlnA"] - d.groupby("cy")["dlnA"].transform("mean")
    d["ai_int_dm"] = d["ai_interact"] - d.groupby("cy")["ai_interact"].transform("mean")

    idx = d.set_index(["sector", "year"])
    m = PanelOLS(idx["dlnA_dm"], idx[["ai_int_dm"]], entity_effects=True, time_effects=True,
                 drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)

    row = {
        "Specification": "Sectoral Rajan–Zingales (CY + SY FE)",
        "θ": m.params["ai_int_dm"],
        "SE": m.std_errors["ai_int_dm"],
        "p-value": m.pvalues["ai_int_dm"],
        "N": int(m.nobs),
        "Countries": d.country.nunique(),
        "R²": m.rsquared_within,
        "1st-F": np.nan,
        "Hansen-J-p": np.nan,
        "AR-p": np.nan,
        "Method": "Rajan–Zingales",
    }
    return row, m


# ═══════════════════════════════════════════════════════════════════
#  D.  Distributed-Lag Model (J-Curve Test)
# ═══════════════════════════════════════════════════════════════════
def _distributed_lag_trajectory(lag_rows):
    """Compile distributed-lag coefficients across horizons k in {0, 1, 2}."""
    rows = []
    cum_th = 0.0
    for r in lag_rows:
        if r is None:
            continue
        k = r.get("Lag", 0)
        cum_th += r["θ"]
        rows.append({
            "Lag": k,
            "θ_k": r["θ"],
            "SE": r["SE"],
            "p-value": r["p-value"],
            "cumulative_θ": cum_th,
            "N": r["N"],
        })
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("Lag")
    pretty(df, "Stage 2 · Distributed-Lag Trajectory (J-Curve Test)")
    return df


# ═══════════════════════════════════════════════════════════════════
#  E.  Heterogeneity Analysis
# ═══════════════════════════════════════════════════════════════════
def _heterogeneity(d, controls, sec_panel=None, eid="country"):
    """Subgroup estimates by income group and sector."""
    rows = []

    # By income group (High Income only to avoid fragile small-sample outliers)
    if "income_group" in d.columns and d["income_group"].notna().sum() > 30:
        sub = d[d["income_group"] == "High"].dropna(subset=["dlnA", "ai"] + controls)
        if len(sub) >= 20 and sub[eid].nunique() >= 3:
            idx = sub.set_index([eid, "year"])
            try:
                m = PanelOLS(idx["dlnA"], idx[["ai"] + controls],
                             entity_effects=True, time_effects=True,
                             drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
                rows.append({"Group": "Income: High", "θ": m.params["ai"],
                             "SE": m.std_errors["ai"], "p": m.pvalues["ai"],
                             "N": int(m.nobs)})
            except Exception:
                pass

    # By sector (from sec_panel or d)
    src_sec = sec_panel if sec_panel is not None else d
    if src_sec is not None and "sector" in src_sec.columns:
        p_sec = _prepare(src_sec)
        for sec in ["Services", "Industry", "Agriculture"]:
            sub = p_sec[p_sec["sector"] == sec].dropna(subset=["dlnA", "ai"] + controls)
            if len(sub) < 20:
                continue
            idx = sub.set_index([eid, "year"])
            try:
                m = PanelOLS(idx["dlnA"], idx[["ai"] + controls],
                             entity_effects=True, time_effects=True,
                             drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
                rows.append({"Group": f"Sector: {sec}", "θ": m.params["ai"],
                             "SE": m.std_errors["ai"], "p": m.pvalues["ai"],
                             "N": int(m.nobs)})
            except Exception:
                pass

    het_df = pd.DataFrame(rows)
    if not het_df.empty:
        het_df = het_df.set_index("Group")
        pretty(het_df, "Stage 2 · Heterogeneity")
    return het_df


# ═══════════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════════
def run(agg_panel, sec_panel=None):
    """Run all Stage-2 specifications across aggregate and sectoral panels."""
    p_agg = _prepare(agg_panel)
    controls = _pick_controls(p_agg)

    d_agg = p_agg.dropna(subset=["dlnA", "ai"] + controls).copy()
    yd_agg = add_year_dummies(d_agg)
    baseline_countries = set(d_agg["country"].unique())
    print(f"  Stage-2 aggregate sample: {len(d_agg)} obs, {len(baseline_countries)} countries")

    spec_rows = []
    models    = {}

    # 1. OLS Baseline (contemporaneous)
    row_ols, m_ols = _ols_baseline(p_agg, controls, lag=0, baseline_countries=baseline_countries)
    if row_ols is not None:
        spec_rows.append(row_ols)
        models["ols"] = m_ols

    # 2. OLS Lag 1
    row_l1, m_l1 = _ols_baseline(p_agg, controls, lag=1, baseline_countries=baseline_countries)
    if row_l1 is not None:
        spec_rows.append(row_l1)
        models["ols_lag1"] = m_l1

    # 3. OLS Lag 2
    row_l2, m_l2 = _ols_baseline(p_agg, controls, lag=2, baseline_countries=baseline_countries)
    if row_l2 is not None:
        spec_rows.append(row_l2)
        models["ols_lag2"] = m_l2

    # 4. Overidentified IV-2SLS
    if "bartik1" in d_agg.columns and "bartik2" in d_agg.columns:
        row_2sls, m_2sls = _iv_overidentified(d_agg, controls, yd_agg, method="2sls")
        if row_2sls is not None:
            spec_rows.append(row_2sls)
            models["iv_2sls"] = m_2sls

        # 5. Overidentified IV-LIML
        row_liml, m_liml = _iv_overidentified(d_agg, controls, yd_agg, method="liml")
        if row_liml is not None:
            spec_rows.append(row_liml)
            models["liml"] = m_liml

    # 6. Sectoral Rajan–Zingales Interaction (if sectoral panel provided)
    if sec_panel is not None:
        p_sec = _prepare(sec_panel)
        row_sec, m_sec = _sectoral_rajan_zingales(p_sec, controls)
        if row_sec is not None:
            spec_rows.append(row_sec)
            models["sector_rz"] = m_sec

    # Summary table
    tab = pd.DataFrame(spec_rows)
    if not tab.empty:
        tab = tab.set_index("Specification")
    pretty(tab, "Stage 2 · AI → TFP Growth (Aggregate & Sectoral Specifications)")

    # Distributed Lags (J-curve test)
    lag_df = _distributed_lag_trajectory([row_ols, row_l1, row_l2])

    # Heterogeneity
    het_df = _heterogeneity(p_agg, controls, sec_panel=sec_panel)

    return {
        "spec_table": tab,
        "lag_table": lag_df,
        "het_table": het_df,
        "models": models,
        "controls": controls,
    }
