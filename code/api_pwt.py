"""Block 2b – Penn World Table 10.01 capital stocks and human capital.

PWT 10.01 provides econometrician-standard capital-stock estimates
(``rnna`` – capital stock at constant 2017 national prices, in millions)
and human-capital indices (``hc``).  We use PWT capital through its last
vintage year and extend forward with World-Bank GFCF via the perpetual-
inventory method.  This replaces the crude GFCF-only approach of the
original pipeline.

Source:  Feenstra, Inklaar and Timmer (2015), "The Next Generation of the
         Penn World Table", American Economic Review 105(10), 3150–3182.
         https://www.rug.nl/ggdc/productivity/pwt/
Data:    https://dataverse.nl/api/access/datafile/354098  (Stata .dta)
"""
import os
import pandas as pd
from config import CACHE, PWT_URL
from utils import cached_download


def load_pwt():
    """Download (once) and return selected PWT 10.01 variables.

    Returns
    -------
    DataFrame with columns:
        country  – ISO-3 alpha code
        year     – integer
        K_pwt    – capital stock at constant 2017 national prices (millions)
        hc       – human-capital index
        ctfp     – TFP level at current PPPs (USA = 1)
        rgdpna   – real GDP at constant 2017 national prices (millions)
    """
    xlsx_path = os.path.join(CACHE, "pwt1001.xlsx")
    dta_path = os.path.join(CACHE, "pwt1001.dta")

    if os.path.exists(xlsx_path):
        path = xlsx_path
    elif os.path.exists(dta_path):
        path = dta_path
    else:
        ext = "xlsx" if PWT_URL.endswith(".xlsx") else "dta"
        path = cached_download(PWT_URL, f"pwt1001.{ext}")

    print(f"  ↳ reading PWT 10.01 ({os.path.basename(path)}) ...")
    if path.endswith(".xlsx"):
        df = pd.read_excel(path, sheet_name="Data")
    else:
        df = pd.read_stata(path, convert_categoricals=False)

    keep = ["countrycode", "year", "rnna", "rgdpna", "emp", "labsh", "csh_i", "delta", "hc", "ctfp"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.rename(columns={"countrycode": "country", "rnna": "K_pwt", "emp": "L_pwt", "rgdpna": "Y_pwt"})
    df = df.dropna(subset=["K_pwt"])
    df["year"] = df["year"].astype(int)

    print(f"  ↳ PWT panel: {df.country.nunique()} countries, "
          f"{int(df.year.min())}–{int(df.year.max())}")
    return df


