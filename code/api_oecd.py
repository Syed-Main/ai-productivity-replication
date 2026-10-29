"""Block 2a – OECD SDMX REST API: AI adoption by businesses."""
import io
import numpy as np
import pandas as pd
from config import OECD, OECD_DATAFLOW, OECD_MEASURE_RE
from utils import cached_get


def parse_ai_adoption(csv_text):
    """Turn the OECD csvfilewithlabels download into a country–year AI series."""
    df = pd.read_csv(io.StringIO(csv_text), low_memory=False)
    mcols = [c for c in df.columns if "measure" in c.lower()]
    mask = np.zeros(len(df), dtype=bool)
    for c in mcols:
        mask |= (
            df[c].astype(str).str.contains(OECD_MEASURE_RE,
                                           case=False, regex=True, na=False)
        )
    df = df[mask]
    if df.empty:
        raise RuntimeError(
            "no AI indicator found — inspect the dataflow's measure labels "
            "and edit OECD_MEASURE_RE in config.py"
        )
    # keep whole-economy / all-size breakdowns where available
    for c in df.columns:
        if any(k in c.upper() for k in ("ACTIVITY", "SIZE")) and "_T" in set(df[c]):
            df = df[df[c] == "_T"]
    key = mcols[0]
    best = df.groupby(key)["OBS_VALUE"].count().idxmax()
    df = df[df[key] == best]
    print(f"  ↳ OECD AI measure used: {best}")
    out = (
        df.assign(year=pd.to_numeric(df["TIME_PERIOD"].astype(str).str[:4]))
          .groupby(["REF_AREA", "year"], as_index=False)["OBS_VALUE"]
          .mean()
          .rename(columns={"REF_AREA": "country", "OBS_VALUE": "ai"})
    )
    print(f"  ↳ OECD AI panel: {out.country.nunique()} countries, "
          f"years {out.year.min()}–{out.year.max()}, {len(out)} obs")
    return out


def ai_adoption():
    """Download and return the AI-adoption panel from the OECD."""
    txt = cached_get(
        f"{OECD}/data/{OECD_DATAFLOW}/all",
        {"startPeriod": "2015", "format": "csvfilewithlabels"},
        "oecd_ict_businesses.csv",
        as_json=False,
    )
    return parse_ai_adoption(txt)
