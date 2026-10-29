"""Block 6 – Publication-quality tables (LaTeX + CSV) and figures (PDF)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUT


# ── LaTeX table helper ──────────────────────────────────────────────
def to_latex(df, path, fmt="{:.3f}", caption="", label=""):
    """Write a booktabs-style LaTeX table."""
    ncols = df.shape[1]
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{l" + "r" * ncols + "}",
        r"\toprule",
        " & ".join([""]
                    + [c.replace("%", r"\%").replace("_", r"\_")
                       for c in df.columns]) + r" \\",
        r"\midrule",
    ]
    for idx_val, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    cells.append("")
                else:
                    cells.append(fmt.format(v))
            else:
                cells.append(str(v))
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


# ── decomposition summary ──────────────────────────────────────────
def decomposition_table(panel):
    return (
        panel.groupby("sector")[["g_Y", "C_K", "C_L", "C_TFP"]]
             .mean()
             .rename(columns={"g_Y": "Output growth", "C_K": "Capital",
                               "C_L": "Labour", "C_TFP": "TFP"})
    )


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
    fig.savefig(f"{OUT}/fig_decomposition.pdf")
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
    fig.savefig(f"{OUT}/fig_lag_coefficients.pdf")
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
    fig.savefig(f"{OUT}/fig_heterogeneity.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════
def save_all(panel, s1_results, s2_results, rob_results=None):
    """Write every table and figure to results/."""

    # ── Stage 1 combined table ──────────────────────────────────────
    s1_combined = s1_results.get("combined") if isinstance(s1_results, dict) else s1_results
    if isinstance(s1_combined, pd.DataFrame):
        s1_combined.to_csv(f"{OUT}/stage1_production_function.csv")
        to_latex(s1_combined, f"{OUT}/stage1_production_function.tex",
                 caption="Stage 1: production-function estimates.",
                 label="tab:pf")

    # ── Decomposition ───────────────────────────────────────────────
    dec = decomposition_table(panel)
    dec.to_csv(f"{OUT}/decomposition_by_sector.csv")
    to_latex(dec, f"{OUT}/decomposition_by_sector.tex", fmt="{:.2f}",
             caption="Mean annual growth decomposition (percentage points).",
             label="tab:dec")
    _fig_decomposition(dec)

    print(f"\nMean annual growth decomposition (percentage points):")
    print(dec.round(2))

    # ── Stage 2 specification table ─────────────────────────────────
    if isinstance(s2_results, dict):
        tab = s2_results.get("spec_table")
        if tab is not None and not tab.empty:
            tab.to_csv(f"{OUT}/stage2_ai_regressions.csv")
            to_latex(tab, f"{OUT}/stage2_ai_regressions.tex",
                     caption="Stage 2: AI adoption and TFP growth.",
                     label="tab:ai")

        # Distributed lag table + figure
        lag_df = s2_results.get("lag_table")
        if lag_df is not None and not lag_df.empty:
            lag_df.to_csv(f"{OUT}/stage2_distributed_lags.csv")
            to_latex(lag_df, f"{OUT}/stage2_distributed_lags.tex",
                     fmt="{:.4f}",
                     caption="Distributed-lag coefficients (J-curve test).",
                     label="tab:jcurve")
            _fig_lag_coefficients(lag_df)

        # Heterogeneity table + figure
        het_df = s2_results.get("het_table")
        if het_df is not None and not het_df.empty:
            het_df.to_csv(f"{OUT}/stage2_heterogeneity.csv")
            to_latex(het_df, f"{OUT}/stage2_heterogeneity.tex",
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
            rob_df.to_csv(f"{OUT}/robustness_summary.csv")
            to_latex(rob_df, f"{OUT}/robustness_summary.tex", fmt="{:.4f}",
                     caption="Robustness checks.", label="tab:robust")

    # ── Full panel CSV ──────────────────────────────────────────────
    panel.to_csv(f"{OUT}/panel_with_decomposition.csv", index=False)
    print(f"\nAll tables and figures saved to {OUT}/")
