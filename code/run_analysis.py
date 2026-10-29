"""Run the complete analysis pipeline.

Usage
-----
    python run_analysis.py               # live World Bank + OECD + PWT APIs
    python run_analysis.py --synthetic    # simulated data (pipeline test only)

Blocks
------
1  World Bank API        ->  sectoral VA, employment, controls
2a OECD API              ->  AI adoption
2b Penn World Table      ->  capital stocks, human capital, labor shares
3  Panel construction    ->  aggregate country-year & sectoral panels
4  Stage 1               ->  Törnqvist growth accounting, one-step joint model, benchmarks
5  Stage 2               ->  AI -> TFP (OLS-FE, overidentified IV-2SLS & LIML, dist. lag, Rajan-Zingales)
5b Robustness            ->  OECD subsample, translog, placebo, jackknife
6  Reporting             ->  tables (LaTeX/CSV), figures (PDF)
"""
import argparse
import os
import shutil
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

    os.makedirs(OUT, exist_ok=True)
    sys.stdout = Tee(os.path.join(OUT, "run_log.txt"))

    # ── data ────────────────────────────────────────────────────────
    if args.synthetic:
        print("=" * 60)
        print("  SYNTHETIC DATA — results validate the code only")
        print("=" * 60)
        agg_panel = panel_mod.build_synthetic_panel()
        sec_panel = panel_mod.build_synthetic_panel()
    else:
        try:
            print("=" * 60)
            print("  CONSTRUCTING AGGREGATE AND SECTORAL PANELS")
            print("=" * 60)
            agg_panel = panel_mod.build_aggregate_panel()
            sec_panel = panel_mod.build_live_panel()
        except requests.RequestException as exc:
            sys.exit(f"Could not reach the APIs: {exc}")

    print(f"\nAggregate Panel: {len(agg_panel):,} rows, {agg_panel.country.nunique()} countries")
    print(f"Sectoral Panel:  {len(sec_panel):,} rows, {sec_panel.country.nunique()} countries, {sec_panel.sector.nunique()} sectors\n")

    # ── Stage 1: production function & growth accounting ────────────
    print("=" * 60)
    print("  STAGE 1: PRODUCTION FUNCTION & GROWTH ACCOUNTING")
    print("=" * 60)
    agg_decomposed, s1_results, s1_combined = stage1.run(agg_panel)

    # ── Stage 2: AI → TFP ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 2: AI ADOPTION -> TFP GROWTH")
    print("=" * 60)
    s2_results = stage2.run(agg_decomposed, sec_panel=sec_panel)

    # ── Robustness battery ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ROBUSTNESS BATTERY")
    print("=" * 60)
    controls = s2_results.get("controls")
    rob_results = robustness.run(agg_decomposed, controls=controls)

    # ── Reporting ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  REPORTING & EXPORT")
    print("=" * 60)
    reporting.save_all(agg_decomposed, {"combined": s1_combined},
                       s2_results, rob_results, sec_panel=sec_panel)

    # Synchronize to root results/
    root_results = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    if os.path.exists(root_results):
        for fname in os.listdir(OUT):
            src = os.path.join(OUT, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(root_results, fname))
        print(f"Synchronised results to {root_results}/")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
