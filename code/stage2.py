"""Block 5 – Stage 2: does AI adoption explain TFP growth?

Four estimator / specification families:

1. **OLS Fixed Effects** — baseline conditional association
2. **IV-2SLS with Bartik instrument** — causal identification
3. **LIML** — robustness to weak instruments
4. **Distributed-lag model** — J-curve hypothesis test
5. **Heterogeneity analysis** — interactions with income group and sector
"""
import warnings
import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS, IVLIML

from config import (STAGE2_CONTROLS, MAX_LAGS, WINSOR_PCTILES)
from utils import pretty, add_year_dummies, entity_demean, winsorise


# ── helpers ─────────────────────────────────────────────────────────
def _prepare(p):
    """Winsorise TFP growth, select usable controls, build entity id."""
    p = p.copy()
    lo, hi = WINSOR_PCTILES
    p["dlnA"] = winsorise(p["C_TFP"], lo, hi)
    return p


def _pick_controls(p, subset=None):
    """Keep controls with >55 % non-missing coverage."""
    pool = subset or STAGE2_CONTROLS
    s0 = p.dropna(subset=["ai", "dlnA"])
    ok = [c for c in pool if c in p.columns and s0[c].notna().mean() > 0.55]
    print(f"  Stage-2 controls used: {ok}")
    return ok


def _demean_panel(d, dep, endog_cols, exog_cols, instr_cols, eid="eid"):
    """Within-group demeaning (entity FE) for IV with panel structure."""
    all_cols = [dep] + endog_cols + exog_cols + instr_cols
    all_cols = [c for c in all_cols if c in d.columns]
    means = d.groupby(eid)[all_cols].transform("mean")
    dm = d.copy()
    for c in all_cols:
        dm[c] = dm[c] - means[c]
    return dm


def _run_iv(dep, exog, endog, instruments, cov, method="2sls"):
    """Unified IV wrapper returning a results dict."""
    Cls = IVLIML if method == "liml" else IV2SLS
    m = Cls(dep, exog, endog, instruments).fit(cov_type=cov)
    return m


def _diagnostics_str(m, endog_name="ai"):
    """Extract key IV diagnostics as a string."""
    parts = [f"N = {int(m.nobs)}"]
    # first stage F
    try:
        fs = m.first_stage
        f_stat = fs.diagnostics[endog_name]["f.stat"]
        parts.append(f"1st-stage F = {f_stat:.2f}")
    except Exception:
        pass
    # Sargan / Hansen
    try:
        if m.sargan is not None:
            parts.append(f"Sargan p = {m.sargan.pval:.4f}")
    except Exception:
        pass
    return ", ".join(parts)


# ═══════════════════════════════════════════════════════════════════
#  A.  OLS Fixed Effects (baseline)
# ═══════════════════════════════════════════════════════════════════
def _ols_baseline(d, controls, yd):
    """OLS-FE: regress TFP growth on AI + controls + entity + time FE."""
    from linearmodels.panel import PanelOLS

    idx = d.set_index(["eid", "year"])
    dep = idx["dlnA"]
    exog_cols = ["ai"] + controls
    exog = idx[exog_cols]

    m = PanelOLS(dep, exog, entity_effects=True, time_effects=True,
                 check_rank=False, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)


    row = {"Specification": "OLS-FE (current)",
           "θ": m.params["ai"], "SE": m.std_errors["ai"],
           "p-value": m.pvalues["ai"], "N": int(m.nobs),
           "Countries": d.country.nunique(), "R²": m.rsquared_within,
           "1st-F": np.nan, "Method": "OLS"}
    return row, m


# ═══════════════════════════════════════════════════════════════════
#  B.  IV-2SLS with Bartik instrument
# ═══════════════════════════════════════════════════════════════════
def _iv_bartik(d, controls, yd, method="2sls"):
    """FE-IV via entity demeaning + IV2SLS/LIML."""
    label = "IV-2SLS" if method == "2sls" else "LIML"

    endog_cols = ["ai"]
    instr_cols = ["bartik"]
    exog_cols  = list(controls)
    yr_cols    = list(yd.columns)

    d_full = pd.concat([d, yd], axis=1)

    # entity demeaning (manual FE removal)
    dm = _demean_panel(d_full, "dlnA", endog_cols, exog_cols + yr_cols,
                       instr_cols)

    dep   = dm["dlnA"]
    exog  = dm[exog_cols + yr_cols] if len(exog_cols) else dm[yr_cols]
    endog = dm[endog_cols]
    instr = dm[instr_cols]

    try:
        m = _run_iv(dep, exog, endog, instr, cov="robust", method=method)
    except Exception as exc:
        print(f"  ↳ {label} failed: {exc}")
        return None, None

    diag = _diagnostics_str(m)
    first_f = np.nan
    try:
        first_f = m.first_stage.diagnostics["ai"]["f.stat"]
    except Exception:
        pass

    row = {"Specification": f"{label} Bartik (current)",
           "θ": m.params["ai"], "SE": m.std_errors["ai"],
           "p-value": m.pvalues["ai"], "N": int(m.nobs),
           "Countries": d.country.nunique(), "R²": np.nan,
           "1st-F": first_f, "Method": label}
    return row, m


# ═══════════════════════════════════════════════════════════════════
#  C.  Distributed-lag model (J-curve test)
# ═══════════════════════════════════════════════════════════════════
def _distributed_lag(d, controls, yd):
    """OLS-FE with contemporaneous + lagged AI (0 to MAX_LAGS)."""
    from linearmodels.panel import PanelOLS

    ai_vars = ["ai"] + [f"ai_lag{k}" for k in range(1, MAX_LAGS + 1)
                        if f"ai_lag{k}" in d.columns]
    sub = d.dropna(subset=["dlnA"] + ai_vars + controls).copy()
    if len(sub) < 30:
        # If 3 lags is too restrictive on short panel, try with 1 or 2 lags
        ai_vars = ["ai"] + [f"ai_lag{k}" for k in range(1, 2) if f"ai_lag{k}" in d.columns]
        sub = d.dropna(subset=["dlnA"] + ai_vars + controls).copy()

    if len(sub) < 20:
        print("  ↳ distributed lag sample too small — skipping")
        return None, None

    idx = sub.set_index(["eid", "year"])
    dep = idx["dlnA"]
    exog = idx[ai_vars + controls]

    try:
        m = PanelOLS(dep, exog, entity_effects=True, time_effects=True,
                     check_rank=False, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    except Exception as exc:
        try:
            m = PanelOLS(dep, exog, entity_effects=True, time_effects=False,
                         check_rank=False, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
        except Exception as exc2:
            print(f"  ↳ distributed lag estimation failed: {exc2}")
            return None, None

    rows = []
    for v in ai_vars:
        if v in m.params:
            lag_k = 0 if v == "ai" else int(v.replace("ai_lag", ""))
            rows.append({
                "Lag": lag_k, "θ_k": m.params[v], "SE": m.std_errors[v],
                "p-value": m.pvalues[v],
            })
    if not rows:
        return None, None
    lag_df = pd.DataFrame(rows).set_index("Lag")
    lag_df["cumulative_θ"] = lag_df["θ_k"].cumsum()

    pretty(lag_df, "Stage 2 · Distributed-Lag (J-Curve Test)")
    print(f"  N = {int(m.nobs)}, Σθ = {lag_df['θ_k'].sum():.4f}")

    return lag_df, m


# ═══════════════════════════════════════════════════════════════════
#  D.  Heterogeneity analysis
# ═══════════════════════════════════════════════════════════════════
def _heterogeneity(d, controls, yd):
    """Interact AI with income group and sector dummies."""
    from linearmodels.panel import PanelOLS
    rows = []

    # ── by income group ─────────────────────────────────────────────
    if "income_group" in d.columns and d["income_group"].notna().sum() > 50:
        for grp in sorted(d["income_group"].dropna().unique()):
            sub = d[d["income_group"] == grp].dropna(
                subset=["dlnA", "ai"] + controls)
            if len(sub) < 30:
                continue
            idx = sub.set_index(["eid", "year"])
            try:
                m = PanelOLS(idx["dlnA"], idx[["ai"] + controls],
                             entity_effects=True, time_effects=True,
                             check_rank=False, drop_absorbed=True
                             ).fit(cov_type="clustered", cluster_entity=True)
                rows.append({"Group": f"Income: {grp}", "θ": m.params["ai"],
                             "SE": m.std_errors["ai"], "p": m.pvalues["ai"],
                             "N": int(m.nobs)})
            except Exception:
                pass

    # ── by sector ───────────────────────────────────────────────────
    for sec in sorted(d["sector"].unique()):
        sub = d[d["sector"] == sec].dropna(subset=["dlnA", "ai"] + controls)
        if len(sub) < 20:
            continue
        idx = sub.set_index(["eid", "year"])
        try:
            m = PanelOLS(idx["dlnA"], idx[["ai"] + controls],
                         entity_effects=True, time_effects=True,
                         check_rank=False, drop_absorbed=True
                         ).fit(cov_type="clustered", cluster_entity=True)
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
def run(panel):
    """Run all Stage-2 specifications; return tables and model objects."""
    p = _prepare(panel)
    controls = _pick_controls(p)

    # sample: need dlnA + ai + controls
    d = p.dropna(subset=["dlnA", "ai"] + controls).copy()
    yd = add_year_dummies(d)
    print(f"  Stage-2 estimation sample: {len(d)} obs, "
          f"{d.country.nunique()} countries")

    spec_rows = []
    models    = {}

    # A. OLS baseline
    row, m = _ols_baseline(d, controls, yd)
    spec_rows.append(row)
    models["ols"] = m

    # Also run OLS with lagged AI
    for lag_k in range(1, MAX_LAGS + 1):
        lag_var = f"ai_lag{lag_k}"
        if lag_var not in d.columns:
            continue
        d_lag = d.dropna(subset=[lag_var])
        if len(d_lag) < 50:
            continue
        from linearmodels.panel import PanelOLS
        idx = d_lag.set_index(["eid", "year"])
        try:
            ml = PanelOLS(idx["dlnA"], idx[[lag_var] + controls],
                          entity_effects=True, time_effects=True,
                          check_rank=False, drop_absorbed=True
                          ).fit(cov_type="clustered", cluster_entity=True)

            spec_rows.append({
                "Specification": f"OLS-FE (lag {lag_k})",
                "θ": ml.params[lag_var], "SE": ml.std_errors[lag_var],
                "p-value": ml.pvalues[lag_var], "N": int(ml.nobs),
                "Countries": d_lag.country.nunique(),
                "R²": ml.rsquared_within,
                "1st-F": np.nan, "Method": "OLS"})
        except Exception:
            pass

    # B. IV-2SLS Bartik
    if "bartik" in d.columns and d["bartik"].notna().sum() > 30:
        d_iv = d.dropna(subset=["bartik"]).copy()
        yd_iv = add_year_dummies(d_iv)
        row_iv, m_iv = _iv_bartik(d_iv, controls, yd_iv, method="2sls")
        if row_iv is not None:
            spec_rows.append(row_iv)
            models["iv_2sls"] = m_iv

        # C. LIML
        row_liml, m_liml = _iv_bartik(d_iv, controls, yd_iv, method="liml")
        if row_liml is not None:
            spec_rows.append(row_liml)
            models["liml"] = m_liml
    else:
        print("  ↳ Bartik instrument unavailable — skipping IV/LIML")

    # summary table
    tab = pd.DataFrame(spec_rows)
    if not tab.empty:
        tab = tab.set_index("Specification")
    pretty(tab, "Stage 2 · AI → TFP Growth (all specifications)")

    # D. Distributed lags
    d_dl = p.dropna(subset=["dlnA", "ai"] + controls)
    lag_df, m_dl = _distributed_lag(d_dl, controls, yd)

    # E. Heterogeneity
    het_df = _heterogeneity(d, controls, yd)

    # AI contribution (using preferred IV estimate if available)
    pref = models.get("iv_2sls", models.get("ols"))
    if pref is not None and "ai" in pref.params:
        theta = pref.params["ai"]
        delta_ai = d["ai"] - d.groupby("eid")["ai"].shift(1)
        c_ai = theta * delta_ai
        print(f"\n  Mean estimated AI contribution to TFP growth: "
              f"{c_ai.mean():.4f} pp/yr (θ = {theta:.4f})")

    return {
        "spec_table": tab,
        "lag_table": lag_df,
        "het_table": het_df,
        "models": models,
    }
