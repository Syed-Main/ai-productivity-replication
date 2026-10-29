"""Block 4 – Stage 1: production-function estimation and growth accounting.

Three estimators for the log-linear Cobb–Douglas production function:

1. **OLS Fixed Effects** (baseline, subject to transmission bias)
2. **IV in First Differences** (Anderson–Hsiao / Arellano–Bond style):
   removes entity FE via first-differencing, then instruments ΔlnK and
   ΔlnL with their t−2 lags to address input endogeneity.
3. **System GMM** (Blundell–Bond 1998, optional): combines moment
   conditions from the levels and differences equations.

Growth accounting then decomposes output growth into capital, labour
and TFP contributions using the IV-preferred elasticities.
"""
import warnings
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS

from config import ANALYSIS_START, GMM_MAX_LAGS
from utils import pretty, add_year_dummies, entity_demean


# ═══════════════════════════════════════════════════════════════════
#  Method 1:  OLS Fixed Effects (baseline)
# ═══════════════════════════════════════════════════════════════════
def _ols_fe(p):
    """Panel OLS with entity and time FE, entity-clustered SE."""
    idx = p.set_index(["eid", "year"])

    # ── unrestricted ────────────────────────────────────────────────
    dep  = idx["lnY"]
    exog = idx[["lnK", "lnL"]]
    free = PanelOLS(dep, exog, entity_effects=True, time_effects=True,
                    check_rank=False).fit(cov_type="clustered", cluster_entity=True)

    # ── CRS-restricted (per-worker form) ────────────────────────────
    dep_r  = idx["lnY_L"]
    exog_r = idx[["lnK_L"]]
    rest = PanelOLS(dep_r, exog_r, entity_effects=True, time_effects=True,
                    check_rank=False).fit(cov_type="clustered", cluster_entity=True)

    alpha_u, beta_u = free.params["lnK"], free.params["lnL"]
    se_au,   se_bu  = free.std_errors["lnK"], free.std_errors["lnL"]
    alpha_r          = rest.params["lnK_L"]
    se_ar            = rest.std_errors["lnK_L"]

    # CRS Wald test on unrestricted model
    crs_test = free.wald_test(formula="lnK + lnL = 1")

    summary = pd.DataFrame({
        "Estimate":   [alpha_u, beta_u, alpha_u + beta_u, alpha_r, 1 - alpha_r],
        "Std. error": [se_au,   se_bu,  np.nan,           se_ar,   se_ar],
    }, index=["α (OLS unrestricted)", "β (OLS unrestricted)", "α + β (OLS unrestricted)",
              "α (OLS CRS imposed)", "β = 1 − α (OLS CRS imposed)"])

    pretty(summary, "Stage 1 · OLS Fixed Effects")
    print(f"  CRS Wald test: stat = {crs_test.stat:.3f}, "
          f"p-value = {crs_test.pval:.4f}")
    print(f"  N = {int(free.nobs)}, entities = {p.eid.nunique()}, "
          f"R² (within) = {free.rsquared_within:.4f}")

    return {"alpha": alpha_r, "beta": 1 - alpha_r,
            "summary": summary, "crs_p": float(crs_test.pval),
            "nobs": int(free.nobs), "r2w": free.rsquared_within,
            "model": free, "method": "OLS-FE"}


# ═══════════════════════════════════════════════════════════════════
#  Method 2:  IV in First Differences (Anderson–Hsiao)
# ═══════════════════════════════════════════════════════════════════
def _iv_fd(p):
    """First-difference the model to remove entity FE, then instrument
    ΔlnK and ΔlnL with their t−2 lagged levels."""
    d = p.sort_values(["eid", "year"]).copy()

    # first differences
    for v in ("lnY", "lnK", "lnL"):
        d[f"d_{v}"] = d.groupby("eid")[v].diff()

    # lagged levels as instruments (t−2)
    d["lnK_lag2"] = d.groupby("eid")["lnK"].shift(2)
    d["lnL_lag2"] = d.groupby("eid")["lnL"].shift(2)

    # year dummies for time effects in the differenced equation
    d = d.dropna(subset=["d_lnY", "d_lnK", "d_lnL", "lnK_lag2", "lnL_lag2"])
    yr_dums = add_year_dummies(d)

    dep   = d["d_lnY"]
    exog  = yr_dums                                   # time effects only
    endog = d[["d_lnK", "d_lnL"]]                    # endogenous regressors
    instr = d[["lnK_lag2", "lnL_lag2"]]               # excluded instruments

    try:
        m = IV2SLS(dep, exog, endog, instr).fit(cov_type="robust")
    except Exception as exc:
        print(f"  ↳ IV-FD failed: {exc}")
        return None

    alpha_iv = m.params["d_lnK"]
    beta_iv  = m.params["d_lnL"]
    se_a     = m.std_errors["d_lnK"]
    se_b     = m.std_errors["d_lnL"]

    summary = pd.DataFrame({
        "Estimate":   [alpha_iv, beta_iv, alpha_iv + beta_iv],
        "Std. error": [se_a, se_b, np.nan],
    }, index=["α (IV-FD)", "β (IV-FD)", "α + β (IV-FD)"])

    pretty(summary, "Stage 1 · IV in First Differences")

    # first-stage diagnostics
    try:
        fs = m.first_stage
        print(f"  First-stage F (d_lnK): {fs.diagnostics['d_lnK']['f.stat']:.2f}")
        print(f"  First-stage F (d_lnL): {fs.diagnostics['d_lnL']['f.stat']:.2f}")
    except Exception:
        pass
    try:
        diag = m.summary
    except Exception:
        pass

    print(f"  N = {int(m.nobs)}, Sargan p = {m.sargan.pval:.4f}"
          if hasattr(m, "sargan") and m.sargan is not None else
          f"  N = {int(m.nobs)}")

    return {"alpha": alpha_iv, "beta": beta_iv,
            "summary": summary, "nobs": int(m.nobs),
            "model": m, "method": "IV-FD"}


# ═══════════════════════════════════════════════════════════════════
#  Method 3:  System GMM (Blundell–Bond) — optional
# ═══════════════════════════════════════════════════════════════════
def _system_gmm(p):
    """Blundell–Bond System GMM via pydynpd (optional dependency)."""
    try:
        from pydynpd import regression
    except ImportError:
        print("  ↳ pydynpd not installed — skipping System GMM")
        return None

    try:
        d = p[["eid", "year", "lnY", "lnK", "lnL"]].dropna().copy()
        d = d.rename(columns={"eid": "id"})

        # pydynpd requires dynamic specification with L1.lnY and colon lag syntax
        cmd = (f"lnY L1.lnY lnK lnL | "
               f"gmm(lnY, 2:{GMM_MAX_LAGS}) gmm(lnK, 2:{GMM_MAX_LAGS}) gmm(lnL, 2:{GMM_MAX_LAGS}) | "
               f"collapse onestep")

        m = regression.abond(cmd, d, ["id", "year"])
        if not hasattr(m, "models") or not m.models:
            print("  ↳ System GMM returned no models")
            return None

        model = m.models[0]
        rt = model.regression_table.set_index("variable")
        alpha_gmm = float(rt.loc["lnK", "coefficient"]) if "lnK" in rt.index else np.nan
        beta_gmm  = float(rt.loc["lnL", "coefficient"]) if "lnL" in rt.index else np.nan
        se_a      = float(rt.loc["lnK", "std_err"]) if "lnK" in rt.index else np.nan
        se_b      = float(rt.loc["lnL", "std_err"]) if "lnL" in rt.index else np.nan

        summary = pd.DataFrame({
            "Estimate":   [alpha_gmm, beta_gmm, alpha_gmm + beta_gmm],
            "Std. error": [se_a, se_b, np.nan],
        }, index=["α (Sys-GMM)", "β (Sys-GMM)", "α + β (Sys-GMM)"])

        pretty(summary, "Stage 1 · System GMM (Blundell–Bond)")

        try:
            if hasattr(model, "hansen") and model.hansen is not None:
                print(f"  Hansen J: stat = {model.hansen.test_value:.3f}, "
                      f"p = {model.hansen.p_value:.4f}")
            if hasattr(model, "AR_list") and model.AR_list:
                for ar in model.AR_list:
                    print(f"  AR({ar.order}): z = {ar.z_value:.3f}, "
                          f"p = {ar.P_value:.4f}")
        except Exception:
            pass

        return {"alpha": alpha_gmm, "beta": beta_gmm,
                "summary": summary, "model": model, "method": "Sys-GMM"}
    except (Exception, SystemExit, BaseException) as exc:
        print(f"  ↳ System GMM unavailable or failed: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════
#  Growth accounting decomposition
# ═══════════════════════════════════════════════════════════════════
def growth_accounting(p, alpha, beta):
    """Decompose output growth into K, L, and TFP contributions."""
    p = p.copy()
    p["C_K"]   = alpha * p["g_K"]
    p["C_L"]   = beta  * p["g_L"]
    p["C_TFP"] = p["g_Y"] - p["C_K"] - p["C_L"]
    return p.dropna(subset=["g_Y"])


# ═══════════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════════
def run(panel):
    """Estimate the production function via all methods, then decompose."""
    p = panel.copy()

    results = {}
    # 1. OLS FE (baseline)
    results["ols"] = _ols_fe(p)

    # 2. IV in First Differences (primary)
    iv = _iv_fd(p)
    if iv is not None:
        results["iv_fd"] = iv

    # 3. System GMM (robustness, optional)
    gmm = _system_gmm(p)
    if gmm is not None:
        results["sys_gmm"] = gmm

    # Select preferred estimates: IV-FD if available, else OLS
    preferred = results.get("iv_fd", results["ols"])
    alpha = preferred["alpha"]
    beta  = preferred["beta"]
    method = preferred["method"]
    print(f"\n  ★ Preferred Stage-1 method: {method}  "
          f"(α = {alpha:.4f}, β = {beta:.4f})")

    decomposed = growth_accounting(p, alpha, beta)

    # Summary table combining all methods
    combined = pd.concat([r["summary"] for r in results.values()])
    pretty(combined, "Stage 1 · Combined Estimates")

    return decomposed, results, combined
