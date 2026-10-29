# Separating AI from Capital: Identifying the Productivity Contribution of Artificial Intelligence across Countries and Sectors

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Status: Replicated](https://img.shields.io/badge/Replication-Verified-success.svg)](#replication-protocol)

This repository contains the complete empirical codebase, econometric estimation pipeline, and data extraction routines for the paper:

> **"Separating AI from Capital: Identifying the Productivity Contribution of Artificial Intelligence across Countries and Sectors"**  
> *Under Review / arXiv Preprint (2026)*

---

## 📌 Executive Summary of Findings

1. **First-Stage Growth Accounting**:
   - Implements nonparametric **Solow–Jorgenson Törnqvist** growth accounting anchored in empirical time-varying labor shares ($labsh_{ct}$) from Penn World Table 10.01 ($\bar{\alpha} = 0.489, \bar{\beta} = 0.511, \alpha+\beta = 1.000$).
   - One-step joint structural estimation directly incorporating AI adoption into the aggregate production function confirms no significant output shift beyond conventional inputs ($\hat{\theta}_{\text{AI}} = 0.000, p = 0.815, N = 102$).

2. **Capital Stock Splicing**:
   - Primary estimation at the aggregate country-year level where PWT 10.01 capital stocks (`rnna`) are directly measured.
   - Spliced post-2019 by applying the annual real growth rate of World Bank constant-price Gross Fixed Capital Formation (GFCF) to the 2019 PWT real investment series ($I_{2019} = csh\_i \times rgdpna$) and accumulating via perpetual inventory with country-specific depreciation rates ($\delta_c$).

3. **Stage 2 Causal Inference**:
   - **Contemporaneous OLS-FE**: $\hat{\theta} = +0.0048$ ($SE = 0.0516, p = 0.926, N = 102$).
   - **Overidentified Shift-Share IV (2SLS)**: Interacting 2010 broadband ($Z_1$) and internet ($Z_2$) penetration with global leave-one-out AI diffusion yields $\hat{\theta}_{\text{IV}} = +0.1531$ ($SE = 0.1878, p = 0.415$).
   - **Limited Information Maximum Likelihood (LIML)**: Median-unbiased estimation yields $\hat{\theta}_{\text{LIML}} = +0.2989$ ($SE = 0.4262, p = 0.483$).
   - **Specification Diagnostics**: Hansen $J$ overidentification test fails to reject instrument validity ($p = 0.071$ in 2SLS; $p = 0.086$ in LIML); weak-instrument-robust Anderson–Rubin test ($p = 0.136$).
   - **Rajan–Zingales Mechanism**: Sectoral task-exposure interaction with saturated country-by-year and sector-by-year fixed effects yields $\hat{\theta} = +0.0255$ ($SE = 0.0368, p = 0.487, N = 399$).

4. **Distributed Lags & Productivity J-Curve**:
   - Monotonic upward trajectory across horizon-specific regressions:
     - Horizon $k = 0$: $\hat{\theta}_0 = +0.0048$ ($SE = 0.0516, N = 102$)
     - Horizon $k = 1$: $\hat{\theta}_1 = +0.0541$ ($SE = 0.0940, N = 68$)
     - Horizon $k = 2$: $\hat{\theta}_2 = +0.0965$ ($SE = 0.0945, N = 65$)
     - Cumulative return: $\Theta_2 = +0.1555$.

---

## 🗂 Repository Structure

```
.
├── main.tex                       # Manuscript source file (LaTeX / arXiv)
├── requirements.txt               # Python package dependencies
├── LICENSE                        # MIT Open-Source License
├── README.md                      # Replication guide and documentation
│
├── code/                          # Complete Econometric Pipeline
│   ├── config.py                  # Global parameters, paths, and indicator mappings
│   ├── utils.py                   # Ingestion helpers, logging, caching, and demeaning
│   ├── api_worldbank.py           # Automated World Bank WDI REST API harvester
│   ├── api_pwt.py                 # Penn World Table (PWT 10.01) ingestion & mirrors
│   ├── api_oecd.py                # OECD SDMX enterprise AI adoption ingestion (G14_B)
│   ├── panel.py                   # Aggregate & sectoral panel construction & PI capital extension
│   ├── stage1.py                  # Factor shares, Törnqvist decomposition, joint production fn
│   ├── stage2.py                  # Aggregate OLS-FE, overidentified IV-2SLS, LIML, lags, RZ
│   ├── robustness.py              # Translog production, OECD subsample, placebo, jackknife
│   ├── reporting.py               # Automated generation of booktabs LaTeX tables and figures
│   └── run_analysis.py            # Master pipeline orchestrator
│
└── results/                       # Empirical Outputs (Synchronized)
    ├── stage1_production_function.tex
    ├── stage2_ai_regressions.tex
    ├── stage2_distributed_lags.tex
    ├── stage2_heterogeneity.tex
    ├── robustness_summary.tex
    ├── decomposition_by_sector.tex
    ├── fig_decomposition.pdf
    ├── fig_lag_coefficients.pdf
    └── fig_heterogeneity.pdf
```

---

## 🚀 Replication Protocol

### 1. Environment Setup

Clone this repository and create an isolated Python virtual environment:

```bash
git clone https://github.com/elyoon-research/ai-productivity-replication.git
cd ai-productivity-replication

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Executing the Master Pipeline

To execute the entire empirical workflow from start to finish, run:

```bash
cd code
python run_analysis.py
```

The script will automatically:
1. Verify raw data in local cache or download live series via the World Bank and OECD APIs.
2. Build the aggregate country-year panel ($N = 4,236$, 180 countries) and sectoral panel ($N = 11,606$, 166 countries).
3. Compute Törnqvist growth accounting factor shares and one-step joint structural elasticities.
4. Estimate Stage 2 regressions (contemporaneous OLS, distributed lags, overidentified 2SLS and LIML, Rajan–Zingales mechanism).
5. Run the robustness battery (translog production function, OECD subsample harmonization, placebo pre-period test, and leave-one-out country jackknife).
6. Export all publication-grade LaTeX tables, CSV summaries, and vector PDF figures directly into `results/` and `code/results/`.

---

## 📊 Empirical Data Sources

| Indicator | Repository | Code / Series | Description |
| :--- | :--- | :--- | :--- |
| **National Capital Stock** | PWT 10.01 | `rnna` | Real capital stock at constant 2017 national prices. |
| **Gross Fixed Capital Formation** | World Bank WDI | `NE.GDI.FTOT.KD` | Constant 2015 US\$ investment series for post-2019 PI extension. |
| **Labour Cost Share** | PWT 10.01 | `labsh` | Share of labour compensation in GDP at current national prices. |
| **Enterprise AI Adoption** | OECD SDMX | `DSD_ICT_B` (`G14_B`) | Percentage of enterprises using AI technologies (all enterprise sizes). |
| **Sectoral Value Added** | World Bank WDI | `NV.AGR/IND/SRV.TOTL.KD` | Constant 2015 US\$ output by sector (Agriculture, Industry, Services). |
| **Sectoral Employment** | World Bank WDI | `SL.AGR/IND/SRV.EMPL.ZS` | Sectoral employment shares multiplied by total national employment. |
| **2010 Fixed Broadband ($Z_1$)** | World Bank WDI | `IT.NET.BBND.P2` | Fixed broadband subscriptions per 100 people in base year 2010. |
| **2010 Internet Users ($Z_2$)** | World Bank WDI | `IT.NET.USER.ZS` | Percentage of individuals using the Internet in base year 2010. |
| **Macroeconomic Controls** | World Bank WDI | `NY.GDP.PCAP.KD`, `SE.TER.ENRR`, `GB.XPD.RSDV.GD.ZS`, `NE.TRD.GNFS.ZS` | Log GDP per capita, Tertiary enrolment, R&D intensity (% GDP), Trade openness. |

---

## 📜 Citation

If you use this codebase or replication package in your research, please cite:

```bibtex
@article{ai_productivity_2026,
  title={Separating AI from Capital: Identifying the Productivity Contribution of Artificial Intelligence across Countries and Sectors},
  author={First Author and Second Author},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/elyoon-research/ai-productivity-replication}
}
```

---

## 📄 License

This replication repository and codebase are distributed under the [MIT License](LICENSE).
