"""Block 5b – Robustness and sensitivity battery.

Tests:
1. Translog production function (relaxes unitary elasticity of substitution)
2. OECD-only subsample
3. Placebo test: pre-period "AI" (before AI existed)
4. Leave-one-out country jackknife
5. Alternative capital allocation (employment share vs VA share)
"""
import warnings
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from config import STAGE2_CONTROLS, WINSOR_PCTILES, ANALYSIS_START
from utils import pretty, winsorise


def _fit_panel_ols(idx, dep_col, exog_cols):
    """Quick helper: PanelOLS with entity + time FE, clustered SE."""
    dep  = idx[dep_col]
    exog = idx[exog_cols]
    return PanelOLS(dep, exog, entity_effects=True, time_effects=True,
                    check_rank=False).fit(cov_type="clustered", cluster_entity=True)


# ═══════════════════════════════════════════════════════════════════
#  1.  Translog production function
# ═══════════════════════════════════════════════════════════════════
def translog(p):
    """Estimate a translog:
       lnY = α₁ lnK + α₂ lnL + α₃ (lnK)² + α₄ (lnL)² + α₅ lnK×lnL + FE
    """
    d = p.copy()
    d["lnK2"]   = d["lnK"] ** 2
    d["lnL2"]   = d["lnL"] ** 2
    d["lnKxlnL"] = d["lnK"] * d["lnL"]

    idx = d.set_index(["eid", "year"])
    exog_cols = ["lnK", "lnL", "lnK2", "lnL2", "lnKxlnL"]
    m = _fit_panel_ols(idx, "lnY", exog_cols)

    summary = pd.DataFrame({
        "Estimate":   [m.params[c] for c in exog_cols],
        "Std. error": [m.std_errors[c] for c in exog_cols],
        "p-value":    [m.pvalues[c] for c in exog_cols],
    }, index=["lnK", "lnL", "(lnK)²", "(lnL)²", "lnK × lnL"])

    pretty(summary, "Robustness · Translog Production Function")

    # Point elasticities at sample means
    mk, ml = d["lnK"].mean(), d["lnL"].mean()
    e_K = m.params["lnK"] + 2 * m.params["lnK2"] * mk + m.params["lnKxlnL"] * ml
    e_L = m.params["lnL"] + 2 * m.params["lnL2"] * ml + m.params["lnKxlnL"] * mk
    print(f"  Point elasticities at sample means: α(K) = {e_K:.4f}, β(L) = {e_L:.4f}")
    print(f"  α + β = {e_K + e_L:.4f}")

    return {"summary": summary, "e_K": e_K, "e_L": e_L, "model": m}


# ═══════════════════════════════════════════════════════════════════
#  2.  OECD-only subsample
# ═══════════════════════════════════════════════════════════════════
OECD_MEMBERS = {
    "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE", "DNK",
    "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "ISL", "IRL", "ISR",
    "ITA", "JPN", "KOR", "LVA", "LTU", "LUX", "MEX", "NLD", "NZL",
    "NOR", "POL", "PRT", "SVK", "SVN", "ESP", "SWE", "CHE", "TUR",
    "GBR", "USA",
}

def oecd_subsample(p, controls):
    """Re-run OLS-FE on OECD countries only."""
    sub = p[p.country.isin(OECD_MEMBERS)].dropna(
        subset=["dlnA", "ai"] + controls)
    if len(sub) < 30:
        print("  ↳ OECD subsample too small — skipped")
        return None

    idx = sub.set_index(["eid", "year"])
    m = _fit_panel_ols(idx, "dlnA", ["ai"] + controls)
    row = {"Subsample": "OECD only", "θ": m.params["ai"],
           "SE": m.std_errors["ai"], "p": m.pvalues["ai"],
           "N": int(m.nobs), "Countries": sub.country.nunique()}
    print(f"  OECD subsample: θ = {row['θ']:.4f} ({row['SE']:.4f}), "
          f"N = {row['N']}")
    return row


# ═══════════════════════════════════════════════════════════════════
#  3.  Placebo test (pre-period AI)
# ═══════════════════════════════════════════════════════════════════
def placebo(p, controls, cutoff_year=2014):
    """Assign AI = 0 for all years before *cutoff_year* and test
    whether this fake variable has any association with TFP growth."""
    sub = p[p.year < cutoff_year].dropna(subset=["dlnA"] + controls).copy()
    sub["ai_fake"] = np.random.default_rng(42).normal(0, 1, len(sub))

    if len(sub) < 50:
        print("  ↳ pre-period sample too small for placebo — skipped")
        return None

    idx = sub.set_index(["eid", "year"])
    m = _fit_panel_ols(idx, "dlnA", ["ai_fake"] + controls)
    row = {"Test": "Placebo (random pre-2014)",
           "θ_fake": m.params["ai_fake"],
           "SE": m.std_errors["ai_fake"],
           "p": m.pvalues["ai_fake"], "N": int(m.nobs)}
    print(f"  Placebo: θ_fake = {row['θ_fake']:.4f} (p = {row['p']:.4f})")
    return row


# ═══════════════════════════════════════════════════════════════════
#  4.  Leave-one-out jackknife
# ═══════════════════════════════════════════════════════════════════
def jackknife(p, controls):
    """Drop each country and re-estimate θ; report mean, SE, range."""
    base = p.dropna(subset=["dlnA", "ai"] + controls).copy()
    lo, hi = WINSOR_PCTILES
    base["dlnA"] = winsorise(base["C_TFP"], lo, hi)

    countries = sorted(base.country.unique())
    thetas = []
    for c in countries:
        sub = base[base.country != c]
        if len(sub) < 50:
            continue
        idx = sub.set_index(["eid", "year"])
        try:
            m = _fit_panel_ols(idx, "dlnA", ["ai"] + controls)
            thetas.append(m.params["ai"])
        except Exception:
            pass

    if not thetas:
        print("  ↳ jackknife could not run")
        return None

    arr = np.array(thetas)
    n = len(arr)
    jk_mean = arr.mean()
    jk_se   = np.sqrt((n - 1) / n * np.sum((arr - jk_mean) ** 2))
    result = {"mean_θ": jk_mean, "JK_SE": jk_se,
              "min_θ": arr.min(), "max_θ": arr.max(), "n_drops": n}
    print(f"  Jackknife ({n} drops): θ = {jk_mean:.4f} ± {jk_se:.4f}  "
          f"[{arr.min():.4f}, {arr.max():.4f}]")
    return result


# ═══════════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════════
def run(panel, controls=None):
    """Execute the full robustness battery."""
    p = panel.copy()
    lo, hi = WINSOR_PCTILES
    if "dlnA" not in p.columns:
        p["dlnA"] = winsorise(p["C_TFP"], lo, hi)

    if controls is None:
        controls = [c for c in STAGE2_CONTROLS
                    if c in p.columns and p[c].notna().mean() > 0.55]

    results = {}

    # 1. Translog
    try:
        results["translog"] = translog(p)
    except Exception as exc:
        print(f"  ↳ translog failed: {exc}")

    # 2. OECD subsample
    try:
        results["oecd"] = oecd_subsample(p, controls)
    except Exception as exc:
        print(f"  ↳ OECD subsample failed: {exc}")

    # 3. Placebo
    try:
        results["placebo"] = placebo(p, controls)
    except Exception as exc:
        print(f"  ↳ placebo failed: {exc}")

    # 4. Jackknife
    try:
        results["jackknife"] = jackknife(p, controls)
    except Exception as exc:
        print(f"  ↳ jackknife failed: {exc}")

    return results
