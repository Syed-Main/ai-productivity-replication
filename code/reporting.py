"""Block 6 – Publication-quality tables (LaTeX + CSV) and figures (PDF)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUT


# ── LaTeX table helper ──────────────────────────────────────────────
def to_latex(df, path, fmt="{:.3f}", caption="", label=""):
    """Write a clean booktabs-style LaTeX table."""
    ncols = df.shape[1]
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{l" + "r" * ncols + "}",
        r"\toprule",
        " & ".join([""]
                    + [str(c).replace("%", r"\%").replace("_", r"\_")
                       for c in df.columns]) + r" \\",
        r"\midrule",
    ]
    for idx_val, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    cells.append(r"---")
                else:
                    cells.append(fmt.format(v))
            else:
                cells.append(str(v).replace("%", r"\%").replace("_", r"\_"))
        idx_str = str(idx_val).replace("_", r"\_").replace("%", r"\%")
        lines.append(idx_str + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        lines.append(r"\caption{" + caption + "}")
    if label:
        lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Dedicated Stage 2 LaTeX table formatter ─────────────────────────
def stage2_to_latex(spec_df, path, caption="Stage 2: AI adoption and TFP growth.", label="tab:ai"):
    """Format Stage 2 specifications into a publication-grade econometric table."""
    df = spec_df.copy()
    if df.empty:
        return

    cols = list(df.index)
    col_labels = [f"({i+1})" for i in range(len(cols))]

    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\begin{tabular}{l" + "c" * len(cols) + r"}",
        r"\toprule",
        r" & " + " & ".join(col_labels) + r" \\",
    ]

    # Model names
    names = []
    for c in cols:
        n = c.replace("IV-", "").replace(" (Overidentified)", "").replace(" (contemporaneous)", "")
        n = n.replace("–", "--")
        if "Rajan" in n:
            names.append(r"\shortstack{Sectoral\\Rajan--Zingales\\(CY+SY FE)}")
        elif "lag 1" in n:
            names.append(r"\shortstack{OLS-FE\\(lag 1)}")
        elif "lag 2" in n:
            names.append(r"\shortstack{OLS-FE\\(lag 2)}")
        else:
            names.append(f"\\shortstack{{{n}}}")
    lines.append(r"Estimator & " + " & ".join(names) + r" \\")
    lines.append(r"\midrule")

    # Coefficient and SE row
    coef_cells = []
    se_cells = []
    for c in cols:
        th = df.loc[c, "θ"]
        se = df.loc[c, "SE"]
        p  = df.loc[c, "p-value"]
        stars = ""
        if p < 0.01:
            stars = "$^{***}$"
        elif p < 0.05:
            stars = "$^{**}$"
        elif p < 0.10:
            stars = "$^{*}$"
        coef_cells.append(f"{th:.4f}{stars}")
        se_cells.append(f"({se:.4f})" if pd.notna(se) else r"---")

    lines.append(r"AI adoption ($\theta$) & " + " & ".join(coef_cells) + r" \\")
    lines.append(r" & " + " & ".join(se_cells) + r" \\")
    lines.append(r"\addlinespace")

    # Controls and Fixed Effects
    lines.append(r"Controls & " + " & ".join(["Yes" if "Sectoral" not in c else "No" for c in cols]) + r" \\")
    lines.append(r"Country FE & " + " & ".join(["Yes" if "Sectoral" not in c else "---" for c in cols]) + r" \\")
    lines.append(r"Year FE & " + " & ".join(["Yes" if "Sectoral" not in c else "---" for c in cols]) + r" \\")
    lines.append(r"Country $\times$ Year FE & " + " & ".join(["No" if "Sectoral" not in c else "Yes" for c in cols]) + r" \\")
    lines.append(r"Sector $\times$ Year FE & " + " & ".join(["No" if "Sectoral" not in c else "Yes" for c in cols]) + r" \\")
    lines.append(r"\midrule")

    # Observations and diagnostics
    n_obs = [str(int(df.loc[c, "N"])) for c in cols]
    n_cnt = [str(int(df.loc[c, "Countries"])) for c in cols]
    lines.append(r"Observations & " + " & ".join(n_obs) + r" \\")
    lines.append(r"Countries & " + " & ".join(n_cnt) + r" \\")

    # R2 row: suppress negative R2
    r2_cells = []
    for c in cols:
        r2 = df.loc[c, "R²"]
        if pd.isna(r2) or r2 < 0:
            r2_cells.append(r"---")
        else:
            r2_cells.append(f"{r2:.3f}")
    lines.append(r"Within $R^2$ & " + " & ".join(r2_cells) + r" \\")

    # 1st-stage F
    f_cells = []
    for c in cols:
        f_val = df.loc[c, "1st-F"]
        f_cells.append(f"{f_val:.2f}" if pd.notna(f_val) else r"---")
    lines.append(r"First-stage $F$ & " + " & ".join(f_cells) + r" \\")

    # Hansen J p-value
    hj_cells = []
    for c in cols:
        hj_val = df.loc[c, "Hansen-J-p"]
        hj_cells.append(f"{hj_val:.3f}" if pd.notna(hj_val) else r"---")
    lines.append(r"Hansen $J$ ($p$-value) & " + " & ".join(hj_cells) + r" \\")

    # Anderson-Rubin p-value
    ar_cells = []
    for c in cols:
        ar_val = df.loc[c, "AR-p"]
        ar_cells.append(f"{ar_val:.3f}" if pd.notna(ar_val) else r"---")
    lines.append(r"Anderson--Rubin ($p$-value) & " + " & ".join(ar_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{minipage}{\linewidth}\vspace{1ex}\footnotesize")
    lines.append(r"\emph{Notes:} Clustered standard errors at entity level in parentheses. $^{*}p<0.10, ^{**}p<0.05, ^{***}p<0.01$. Columns (1)--(5) estimate country-year panel equations with PWT-derived Törnqvist TFP residuals. Column (6) estimates the Rajan--Zingales mechanism equation using country-by-year and sector-by-year fixed effects. In IV/LIML columns, within $R^2$ is not bounded in $[0,1]$ and is suppressed (---). Instruments $Z_1$ and $Z_2$ are pre-period (2010) fixed broadband and internet penetration interacted with leave-one-out global AI adoption.")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── decomposition summary ──────────────────────────────────────────
def decomposition_table(panel):
    """Summarize growth decomposition across sectors or income groups."""
    if "sector" in panel.columns:
        return (
            panel.groupby("sector")[["g_Y", "C_K", "C_L", "C_TFP"]]
                 .mean()
                 .rename(columns={"g_Y": "Output growth", "C_K": "Capital",
                                   "C_L": "Labour", "C_TFP": "TFP"})
        )
    elif "income_group" in panel.columns:
        return (
            panel.groupby("income_group")[["g_Y", "C_K", "C_L", "C_TFP"]]
                 .mean()
                 .rename(columns={"g_Y": "Output growth", "C_K": "Capital",
                                   "C_L": "Labour", "C_TFP": "TFP"})
        )
    else:
        return pd.DataFrame([panel[["g_Y", "C_K", "C_L", "C_TFP"]].mean()],
                            index=["Total"]).rename(columns={"g_Y": "Output growth", "C_K": "Capital",
                                                             "C_L": "Labour", "C_TFP": "TFP"})


# ── decomposition stacked bar ──────────────────────────────────────
def _fig_decomposition(dec):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    dec[["Capital", "Labour", "TFP"]].plot(
        kind="bar", stacked=True, rot=0, ax=ax,
        color=["#3182bd", "#f16913", "#31a354"])
    ax.set_ylabel("Mean annual contribution (pp)")
    ax.set_xlabel("")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_decomposition.pdf"))
    plt.close(fig)


# ── distributed-lag coefficient plot ────────────────────────────────
def _fig_lag_coefficients(lag_df):
    if lag_df is None or lag_df.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    lags = lag_df.index.values
    coef = lag_df["θ_k"].values
    se   = lag_df["SE"].values
    ax.errorbar(lags, coef, yerr=1.96 * se, fmt="o-",
                color="#3182bd", capsize=4, linewidth=1.5, markersize=6)
    ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Lag (years)")
    ax.set_ylabel(r"$\hat{\theta}_k$ (pp per unit AI)")
    ax.set_xticks(lags)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_lag_coefficients.pdf"))
    plt.close(fig)


# ── heterogeneity plot ──────────────────────────────────────────────
def _fig_heterogeneity(het_df):
    if het_df is None or het_df.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, max(3, 0.5 * len(het_df))))
    y_pos = range(len(het_df))
    ax.barh(y_pos, het_df["θ"], xerr=1.96 * het_df["SE"],
            color="#756bb1", capsize=3, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(het_df.index, fontsize=9)
    ax.axvline(0, color="grey", linewidth=0.7, linestyle="--")
    ax.set_xlabel(r"$\hat{\theta}$ (pp per unit AI)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_heterogeneity.pdf"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════
def save_all(agg_panel, s1_results, s2_results, rob_results=None, sec_panel=None):
    """Write every table and figure to results/."""
    os.makedirs(OUT, exist_ok=True)

    # ── Stage 1 combined table ──────────────────────────────────────
    s1_combined = s1_results.get("combined") if isinstance(s1_results, dict) else s1_results
    if isinstance(s1_combined, pd.DataFrame):
        s1_combined.to_csv(os.path.join(OUT, "stage1_production_function.csv"))
        to_latex(s1_combined, os.path.join(OUT, "stage1_production_function.tex"),
                 caption="Stage 1: Production-function and factor-share estimates.",
                 label="tab:pf")

    # ── Decomposition ───────────────────────────────────────────────
    dec_panel = sec_panel if sec_panel is not None else agg_panel
    dec = decomposition_table(dec_panel)
    dec.to_csv(os.path.join(OUT, "decomposition_by_sector.csv"))
    to_latex(dec, os.path.join(OUT, "decomposition_by_sector.tex"), fmt="{:.2f}",
             caption="Mean annual growth decomposition (percentage points).",
             label="tab:dec")
    _fig_decomposition(dec)

    # ── Stage 2 specification table ─────────────────────────────────
    if isinstance(s2_results, dict):
        tab = s2_results.get("spec_table")
        if tab is not None and not tab.empty:
            tab.to_csv(os.path.join(OUT, "stage2_ai_regressions.csv"))
            stage2_to_latex(tab, os.path.join(OUT, "stage2_ai_regressions.tex"),
                            caption="Stage 2: AI adoption and TFP growth (Aggregate and Sectoral specifications).",
                            label="tab:ai")

        # Distributed lag table + figure
        lag_df = s2_results.get("lag_table")
        if lag_df is not None and not lag_df.empty:
            lag_df.to_csv(os.path.join(OUT, "stage2_distributed_lags.csv"))
            to_latex(lag_df, os.path.join(OUT, "stage2_distributed_lags.tex"),
                     fmt="{:.4f}",
                     caption="Distributed-lag coefficients (J-curve test).",
                     label="tab:jcurve")
            _fig_lag_coefficients(lag_df)

        # Heterogeneity table + figure
        het_df = s2_results.get("het_table")
        if het_df is not None and not het_df.empty:
            het_df.to_csv(os.path.join(OUT, "stage2_heterogeneity.csv"))
            to_latex(het_df, os.path.join(OUT, "stage2_heterogeneity.tex"),
                     fmt="{:.4f}",
                     caption="Heterogeneity of AI--TFP association.",
                     label="tab:het")
            _fig_heterogeneity(het_df)

    # ── Robustness summary ──────────────────────────────────────────
    if rob_results:
        rows = []
        if "translog" in rob_results and rob_results["translog"]:
            tr = rob_results["translog"]
            rows.append({"Test": "Translog α(K)",
                         "Value": tr.get("e_K", np.nan)})
            rows.append({"Test": "Translog β(L)",
                         "Value": tr.get("e_L", np.nan)})
        if "oecd" in rob_results and rob_results["oecd"]:
            rows.append({"Test": "OECD subsample θ",
                         "Value": rob_results["oecd"].get("θ", np.nan)})
            rows.append({"Test": "OECD subsample N",
                         "Value": rob_results["oecd"].get("N", np.nan)})
        if "placebo" in rob_results and rob_results["placebo"]:
            rows.append({"Test": "Placebo θ (p-value)",
                         "Value": rob_results["placebo"].get("p", np.nan)})
        if "jackknife" in rob_results and rob_results["jackknife"]:
            jk = rob_results["jackknife"]
            rows.append({"Test": "Jackknife mean θ",
                         "Value": jk.get("mean_θ", np.nan)})
            rows.append({"Test": "Jackknife SE",
                         "Value": jk.get("JK_SE", np.nan)})
        if rows:
            rob_df = pd.DataFrame(rows).set_index("Test")
            rob_df.to_csv(os.path.join(OUT, "robustness_summary.csv"))
            to_latex(rob_df, os.path.join(OUT, "robustness_summary.tex"), fmt="{:.4f}",
                     caption="Robustness checks.", label="tab:robust")

    # ── Full panel CSV ──────────────────────────────────────────────
    agg_panel.to_csv(os.path.join(OUT, "panel_with_decomposition.csv"), index=False)
    if sec_panel is not None:
        sec_panel.to_csv(os.path.join(OUT, "panel_sectoral_with_decomposition.csv"), index=False)
    print(f"\nAll tables and figures saved to {OUT}/")
