"""Run the complete analysis pipeline.

Usage
-----
    python run_analysis.py               # live World Bank + OECD + PWT APIs
    python run_analysis.py --synthetic    # simulated data (pipeline test only)

Blocks
------
1  World Bank API        →  sectoral VA, employment, controls
2a OECD API              →  AI adoption
2b Penn World Table      →  capital stocks, human capital
3  Panel construction    →  harmonised country×sector×year panel
4  Stage 1               →  production function (OLS-FE, IV-FD, Sys-GMM)
5  Stage 2               →  AI → TFP (OLS-FE, IV-2SLS, LIML, dist. lag)
5b Robustness            →  translog, OECD subsample, placebo, jackknife
6  Reporting             →  tables (LaTeX/CSV), figures (PDF)
"""
import argparse
import sys
import warnings

import requests

import panel as panel_mod
import reporting
import robustness
import stage1
import stage2
from config import OUT
from utils import Tee

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="Use simulated data (pipeline validation only)")
    args = ap.parse_args()
    sys.stdout = Tee(f"{OUT}/run_log.txt")

    # ── data ────────────────────────────────────────────────────────
    if args.synthetic:
        print("=" * 60)
        print("  SYNTHETIC DATA — results validate the code only")
        print("=" * 60)
        panel = panel_mod.build_synthetic_panel()
    else:
        try:
            panel = panel_mod.build_live_panel()
        except requests.RequestException as exc:
            sys.exit(f"Could not reach the APIs: {exc}")

    print(f"\nPanel: {len(panel):,} rows, {panel.country.nunique()} countries, "
          f"{panel.sector.nunique()} sectors\n")

    # ── Stage 1: production function ────────────────────────────────
    decomposed, s1_results, s1_combined = stage1.run(panel)

    # ── Stage 2: AI → TFP ──────────────────────────────────────────
    s2_results = stage2.run(decomposed)

    # ── Robustness battery ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ROBUSTNESS BATTERY")
    print("=" * 60)
    rob_results = robustness.run(decomposed)

    # ── Reporting ───────────────────────────────────────────────────
    reporting.save_all(decomposed, {"combined": s1_combined},
                       s2_results, rob_results)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
