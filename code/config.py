"""Block 0 – settings shared by every module.

Redesigned configuration supporting:
  - Penn World Table 10.01 capital stocks
  - Bartik-style instrumental variables
  - System GMM / IV-FD estimation
  - Distributed-lag J-curve specifications
"""
import os

# ── API end-points ──────────────────────────────────────────────────
WB       = "https://api.worldbank.org/v2"
OECD     = "https://sdmx.oecd.org/public/rest"
OECD_DATAFLOW    = "OECD.STI.DEP,DSD_ICT_B@DF_BUSINESSES"
OECD_MEASURE_RE  = r"artificial intelligence"
PWT_URL  = "https://raw.githubusercontent.com/ClaremontCollegesLibrary/stata-merge-tutorial/main/pwt1001.xlsx"   # PWT 10.01 .xlsx mirror

# ── Time windows ────────────────────────────────────────────────────
START, END     = 1990, 2024    # download window (early years feed K)
ANALYSIS_START = 2000          # first year used in Stage-1 regressions
PWT_END        = 2019          # last PWT vintage year
IV_BASE_YEAR   = 2010          # pre-period broadband for Bartik IV

# ── Capital-stock parameters ────────────────────────────────────────
DELTA   = 0.05   # depreciation rate for perpetual-inventory extension
G_INIT  = 0.03   # assumed steady-state growth for initial K

# ── Directories ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE_DIR, "cache")
OUT   = os.path.join(BASE_DIR, "results")

# ── World Bank indicator codes ──────────────────────────────────────
SECTORS = {
    "Agriculture": ("NV.AGR.TOTL.KD", "SL.AGR.EMPL.ZS"),
    "Industry":    ("NV.IND.TOTL.KD", "SL.IND.EMPL.ZS"),
    "Services":    (("NV.SRV.TOTL.KD", "NV.SRV.TETC.KD"), "SL.SRV.EMPL.ZS"),
}
LABOUR_CODES    = {"lf": "SL.TLF.TOTL.IN", "un": "SL.UEM.TOTL.ZS"}
INVESTMENT_CODE = "NE.GDI.FTOT.KD"   # gross fixed capital formation, const 2015 US$

CONTROL_CODES = {
    "ln_gdp_pc":  "NY.GDP.PCAP.KD",       # GDP per capita, constant (logged in panel.py)
    "tertiary":   "SE.TER.ENRR",           # gross tertiary enrolment ratio
    "rd":         "GB.XPD.RSDV.GD.ZS",     # R&D expenditure, % of GDP
    "trade":      "NE.TRD.GNFS.ZS",        # trade openness, % of GDP
    "internet":   "IT.NET.USER.ZS",        # individuals using the Internet, %
    "broadband":  "IT.NET.BBND.P2",        # fixed broadband subscriptions per 100
    "gov_eff":    "GE.EST",                # gov effectiveness estimate (WGI)
}

# Controls that actually enter the regressions (broadband and internet
# are reserved for the overidentified Bartik instruments Z1 and Z2 and
# excluded from Stage-2 controls to satisfy the exclusion restriction).
STAGE2_CONTROLS = ["ln_gdp_pc", "tertiary", "rd", "trade"]

# Sectoral AI exposure indices (Rajan-Zingales style interaction)
# Based on task exposure literature (Felten et al. AIOE / OECD ICT intensity)
SECTOR_AI_EXPOSURE = {
    "Services":    1.00,
    "Industry":    0.60,
    "Agriculture": 0.15,
}

# ── Econometric settings ────────────────────────────────────────────
MAX_LAGS         = 3            # distributed-lag horizons for J-curve
WINSOR_PCTILES   = (0.01, 0.99) # TFP-growth winsorisation bounds
GMM_MAX_LAGS     = 4            # max instrument lag depth for System GMM

# ── Income-group thresholds (World Bank 2024 GNI/cap, Atlas, US$) ──
INCOME_CUTS = {"Low": 1145, "Lower-middle": 4515, "Upper-middle": 14005}

for d in (CACHE, OUT):
    os.makedirs(d, exist_ok=True)
